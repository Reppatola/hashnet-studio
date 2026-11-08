//! GPU-слой: все ядра в одном модуле (kernels.cu), данные живут на устройстве.
//! Методы только запускают ядра в одном потоке (без синхронизаций);
//! синхронизация происходит лишь в `download`.
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::sync::Arc;

pub type R = Result<(), Box<dyn std::error::Error>>;

const KERNELS: &str = include_str!("kernels.cu");

pub struct GpuBitwise {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    kernels: HashMap<&'static str, CudaFunction>,
}

/// Параметры SGEMM: C[m,n] = alpha·op(A)[m,k]·op(B)[k,n] + beta·C, строки-мажорный порядок.
/// Пакет: batch матриц, индекс z раскладывается в (z/nh, z%nh); смещения sa/sb/sc = (шаг по z/nh, шаг по z%nh).
#[derive(Clone, Copy, Debug)]
pub struct G {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub ta: bool,
    pub tb: bool,
    pub batch: usize,
    pub nh: usize,
    pub sa: (i64, i64),
    pub sb: (i64, i64),
    pub sc: (i64, i64),
    pub alpha: f32,
    pub beta: f32,
}

impl G {
    pub fn new(m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize) -> G {
        G { m, n, k, lda, ldb, ldc, ta: false, tb: false, batch: 1, nh: 1, sa: (0, 0), sb: (0, 0), sc: (0, 0), alpha: 1.0, beta: 0.0 }
    }
    pub fn ta(mut self) -> G {
        self.ta = true;
        self
    }
    pub fn tb(mut self) -> G {
        self.tb = true;
        self
    }
    pub fn alpha(mut self, a: f32) -> G {
        self.alpha = a;
        self
    }
    pub fn beta(mut self, b: f32) -> G {
        self.beta = b;
        self
    }
    pub fn batched(mut self, batch: usize, nh: usize, sa: (i64, i64), sb: (i64, i64), sc: (i64, i64)) -> G {
        self.batch = batch;
        self.nh = nh;
        self.sa = sa;
        self.sb = sb;
        self.sc = sc;
        self
    }
}

fn c1(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (((n + 255) / 256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}
fn crows(rows: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (rows.max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}
/// редукции: ограниченное число блоков (grid-stride)
fn cred(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (((n + 255) / 256).clamp(1, 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

macro_rules! launch {
    ($s:ident, $name:expr, $cfg:expr, $($arg:expr),+ $(,)?) => {{
        let f = $s.kernels.get($name).ok_or(concat!("нет ядра ", $name))?;
        let mut b = $s.stream.launch_builder(f);
        $( b.arg($arg); )+
        unsafe { b.launch($cfg)?; }
    }};
}

impl GpuBitwise {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        println!("✅ GPU: {}", ctx.name()?);
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNELS)?;
        let module = ctx.load_module(ptx)?;
        let names: [&'static str; 21] = [
            "sgemm", "embed_fwd", "embed_bwd", "ln_fwd", "ln_bwd", "colsum2", "aq", "copyk", "addk", "zerok",
            "gelu_fwd", "gelu_bwd", "softmax_fwd", "softmax_bwd", "loss_k", "absum", "ternarize", "sumsq", "adam", "ngram_fwd", "ngram_bwd",
        ];
        let mut kernels = HashMap::new();
        for n in names {
            kernels.insert(n, module.load_function(n)?);
        }
        Ok(Self { ctx, stream, kernels })
    }

    // ───────────── память ─────────────
    pub fn upload(&self, x: &[f32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if x.is_empty() {
            return Ok(self.stream.alloc_zeros::<f32>(1)?);
        }
        Ok(self.stream.clone_htod(x)?)
    }
    pub fn upload_u8(&self, x: &[u8]) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.stream.clone_htod(x)?)
    }
    pub fn zeros(&self, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Ok(self.stream.alloc_zeros::<f32>(n.max(1))?)
    }
    pub fn zeros_u8(&self, n: usize) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.stream.alloc_zeros::<u8>(n.max(1))?)
    }
    pub fn download(&self, x: &CudaSlice<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(x)?)
    }
    pub fn sync(&self) -> R {
        self.stream.synchronize()?;
        Ok(())
    }

    // ───────────── ядра ─────────────
    pub fn gemm(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>, g: G) -> R {
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((g.n + 63) / 64).max(1) as u32, ((g.m + 63) / 64).max(1) as u32, g.batch as u32),
            shared_mem_bytes: 0,
        };
        let (m, n, k) = (g.m as i32, g.n as i32, g.k as i32);
        let (lda, ldb, ldc) = (g.lda as i32, g.ldb as i32, g.ldc as i32);
        let (ta, tb, nh) = (g.ta as i32, g.tb as i32, g.nh as i32);
        let (sa1, sa2, sb1, sb2, sc1, sc2) = (g.sa.0, g.sa.1, g.sb.0, g.sb.1, g.sc.0, g.sc.1);
        let (alpha, beta) = (g.alpha, g.beta);
        launch!(self, "sgemm", cfg, a, b, c, &m, &n, &k, &lda, &ldb, &ldc, &ta, &tb, &nh, &sa1, &sa2, &sb1, &sb2, &sc1, &sc2, &alpha, &beta);
        Ok(())
    }

    pub fn embed_fwd(&self, emb: &CudaSlice<f32>, pos: &CudaSlice<f32>, tok: &CudaSlice<u8>, x: &mut CudaSlice<f32>, n: usize, t: usize, d: usize) -> R {
        let (t, d) = (t as i32, d as i32);
        launch!(self, "embed_fwd", crows(n), emb, pos, tok, x, &t, &d);
        Ok(())
    }
    pub fn embed_bwd(&self, dx: &CudaSlice<f32>, tok: &CudaSlice<u8>, g_emb: &mut CudaSlice<f32>, g_pos: &mut CudaSlice<f32>, n: usize, t: usize, d: usize) -> R {
        let (t, d) = (t as i32, d as i32);
        launch!(self, "embed_bwd", crows(n), dx, tok, g_emb, g_pos, &t, &d);
        Ok(())
    }
    pub fn ngram_fwd(&self, h3: &CudaSlice<f32>, h5: &CudaSlice<f32>, tok: &CudaSlice<u8>, x: &mut CudaSlice<f32>, n: usize, t: usize, d: usize, buckets: usize) -> R {
        let (t, d, bk) = (t as i32, d as i32, buckets as i32);
        launch!(self, "ngram_fwd", crows(n), h3, h5, tok, x, &t, &d, &bk);
        Ok(())
    }
    pub fn ngram_bwd(&self, dx: &CudaSlice<f32>, tok: &CudaSlice<u8>, g3: &mut CudaSlice<f32>, g5: &mut CudaSlice<f32>, n: usize, t: usize, d: usize, buckets: usize) -> R {
        let (t, d, bk) = (t as i32, d as i32, buckets as i32);
        launch!(self, "ngram_bwd", crows(n), dx, tok, g3, g5, &t, &d, &bk);
        Ok(())
    }
    pub fn ln_fwd(&self, x: &CudaSlice<f32>, g: &CudaSlice<f32>, b: &CudaSlice<f32>, out: &mut CudaSlice<f32>, xn: &mut CudaSlice<f32>, stds: &mut CudaSlice<f32>, rows: usize, d: usize) -> R {
        let d = d as i32;
        launch!(self, "ln_fwd", crows(rows), x, g, b, out, xn, stds, &d);
        Ok(())
    }
    /// dx = res + LN_bwd(dout); gamma/beta-градиенты накапливаются в dg/db
    pub fn ln_bwd(&self, dout: &CudaSlice<f32>, xn: &CudaSlice<f32>, stds: &CudaSlice<f32>, g: &CudaSlice<f32>, res: Option<&CudaSlice<f32>>, dx: &mut CudaSlice<f32>, dg: &mut CudaSlice<f32>, db: &mut CudaSlice<f32>, rows: usize, d: usize) -> R {
        let di = d as i32;
        let use_res = res.is_some() as i32;
        let r = res.unwrap_or(dout);
        launch!(self, "ln_bwd", crows(rows), dout, xn, stds, g, r, &use_res, dx, &di);
        let n = rows as i32;
        launch!(self, "colsum2", c1(d), dout, xn, dg, db, &n, &di);
        Ok(())
    }
    pub fn aq(&self, x: &CudaSlice<f32>, out: &mut CudaSlice<f32>, rows: usize, d: usize) -> R {
        let d = d as i32;
        launch!(self, "aq", crows(rows), x, out, &d);
        Ok(())
    }
    pub fn copy(&self, x: &CudaSlice<f32>, out: &mut CudaSlice<f32>, n: usize) -> R {
        let n = n as i32;
        launch!(self, "copyk", c1(n as usize), x, out, &n);
        Ok(())
    }
    pub fn add(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, out: &mut CudaSlice<f32>, n: usize) -> R {
        let n = n as i32;
        launch!(self, "addk", c1(n as usize), a, b, out, &n);
        Ok(())
    }
    pub fn zero(&self, x: &mut CudaSlice<f32>, n: usize) -> R {
        let n = n as i32;
        launch!(self, "zerok", c1(n as usize), x, &n);
        Ok(())
    }
    pub fn gelu_fwd(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize) -> R {
        let n = n as i32;
        launch!(self, "gelu_fwd", c1(n as usize), x, y, &n);
        Ok(())
    }
    pub fn gelu_bwd(&self, dy: &CudaSlice<f32>, x: &CudaSlice<f32>, dx: &mut CudaSlice<f32>, n: usize) -> R {
        let n = n as i32;
        launch!(self, "gelu_bwd", c1(n as usize), dy, x, dx, &n);
        Ok(())
    }
    pub fn softmax_fwd(&self, s: &mut CudaSlice<f32>, rows: usize, t: usize) -> R {
        let (r, t) = (rows as i32, t as i32);
        launch!(self, "softmax_fwd", c1(rows), s, &r, &t);
        Ok(())
    }
    pub fn softmax_bwd(&self, p: &CudaSlice<f32>, dp: &mut CudaSlice<f32>, rows: usize, t: usize) -> R {
        let (r, t) = (rows as i32, t as i32);
        launch!(self, "softmax_bwd", c1(rows), p, dp, &r, &t);
        Ok(())
    }
    /// суммарный NLL добавляется в *nll; если write_grad — пишет dlogits (уже умноженные на scale)
    pub fn loss(&self, logits: &CudaSlice<f32>, tgt: &CudaSlice<u8>, dl: &mut CudaSlice<f32>, nll: &mut CudaSlice<f32>, rows: usize, scale: f32, write_grad: bool) -> R {
        let wg = write_grad as i32;
        launch!(self, "loss_k", crows(rows), logits, tgt, dl, nll, &scale, &wg);
        Ok(())
    }
    pub fn absum(&self, w: &CudaSlice<f32>, len: usize, acc: &mut CudaSlice<f32>, idx: usize) -> R {
        let (l, i) = (len as i32, idx as i32);
        launch!(self, "absum", cred(len), w, &l, acc, &i);
        Ok(())
    }
    pub fn ternarize(&self, w: &CudaSlice<f32>, eff: &mut CudaSlice<f32>, len: usize, acc: &CudaSlice<f32>, idx: usize, zcnt: &mut CudaSlice<f32>) -> R {
        let (l, i) = (len as i32, idx as i32);
        launch!(self, "ternarize", cred(len), w, eff, &l, acc, &i, zcnt);
        Ok(())
    }
    pub fn sumsq(&self, g: &CudaSlice<f32>, len: usize, acc: &mut CudaSlice<f32>) -> R {
        let l = len as i32;
        launch!(self, "sumsq", cred(len), g, &l, acc);
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn adam(&self, w: &mut CudaSlice<f32>, g: &CudaSlice<f32>, m: &mut CudaSlice<f32>, v: &mut CudaSlice<f32>, len: usize, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, coef: f32, bc1: f32, bc2: f32) -> R {
        let l = len as i32;
        launch!(self, "adam", c1(len), w, g, m, v, &l, &lr, &b1, &b2, &eps, &wd, &coef, &bc1, &bc2);
        Ok(())
    }
}
