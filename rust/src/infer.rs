//! Быстрый инференс HashNet4D: веса — тернарные int8 {-1,0,+1} (+ один множитель на тензор),
//! активации — int8 absmax, внимание с KV-кэшем (каждый токен считается один раз, а не окно целиком).
//! Считает ту же функцию, что и обучение (QAT BitNet b1.58 + 8-бит активации).
use crate::gpu::GpuBitwise;
use crate::model::{Model, Res, BASE, TPL, VOCAB};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::sync::Arc;

const KERNELS: &str = include_str!("kernels_inf.cu");

macro_rules! launch {
    ($k:expr, $st:expr, $name:expr, $cfg:expr, $($arg:expr),+ $(,)?) => {{
        let f = $k.get($name).ok_or(concat!("нет ядра ", $name))?;
        let mut b = $st.launch_builder(f);
        $( b.arg($arg); )+
        unsafe { b.launch($cfg)?; }
    }};
}

struct Layer {
    w: [CudaSlice<i8>; 6], // wq wk wv wo w1 w2, каждая [out,in]
    gam: [f32; 6],
    ln1g: CudaSlice<f32>,
    ln1b: CudaSlice<f32>,
    ln2g: CudaSlice<f32>,
    ln2b: CudaSlice<f32>,
    kc: CudaSlice<f32>,
    vc: CudaSlice<f32>,
}

pub struct Infer {
    pub seq: usize,
    pub pos: usize,
    d: usize,
    dff: usize,
    nh: usize,
    dh: usize,
    stream: Arc<CudaStream>,
    kernels: HashMap<&'static str, CudaFunction>,
    emb: CudaSlice<f32>,
    posn: CudaSlice<f32>,
    ngram: usize,
    ng3: Option<CudaSlice<f32>>,
    ng5: Option<CudaSlice<f32>>,
    hist: Vec<u8>,
    wout: CudaSlice<f32>,
    lnfg: CudaSlice<f32>,
    lnfb: CudaSlice<f32>,
    layers: Vec<Layer>,
    x: CudaSlice<f32>,
    h: CudaSlice<f32>,
    qv: CudaSlice<f32>,
    ctx: CudaSlice<f32>,
    hid: CudaSlice<f32>,
    hff: CudaSlice<f32>,
    lg: CudaSlice<f32>,
    qa: CudaSlice<i8>,
    sc: CudaSlice<f32>,
    pub zero_frac: f32,
}

impl Infer {
    pub fn new(gpu: &GpuBitwise, model: &Model) -> Res<Infer> {
        let cfg = &model.cfg;
        let (d, dff, nh, dh, seq) = (cfg.d_model, cfg.d_ff(), cfg.n_heads, cfg.d_head(), cfg.seq_len);
        if d % 4 != 0 {
            return Err("инференс: d_model должен делиться на 4".into());
        }
        let ptx = compile_ptx(KERNELS)?;
        let module = gpu.ctx.load_module(ptx)?;
        let mut kernels = HashMap::new();
        for n in ["embed_tok", "add_rows2", "ln_f", "q8", "gelu_f", "tmatvec", "matvec_f32", "attn_dec"] {
            kernels.insert(n, module.load_function(n)?);
        }
        let mut zeros = 0u64;
        let mut total = 0u64;
        let mut layers = Vec::new();
        for l in 0..cfg.n_layers {
            let o = BASE + l * TPL;
            let mut ws: Vec<CudaSlice<i8>> = Vec::new();
            let mut gam = [0f32; 6];
            for j in 0..6 {
                let src = &model.t[o + j];
                let (r, c) = cfg.shape(o + j); // [in,out]
                let g = (src.iter().map(|x| x.abs()).sum::<f32>() / src.len() as f32).max(1e-8);
                let inv = 1.0 / g;
                let mut q = vec![0i8; r * c];
                for i in 0..r {
                    for k in 0..c {
                        let v = (src[i * c + k] * inv).round().clamp(-1.0, 1.0) as i8;
                        if v == 0 {
                            zeros += 1;
                        }
                        q[k * r + i] = v; // транспонируем в [out,in]
                    }
                }
                total += (r * c) as u64;
                gam[j] = g;
                ws.push(gpu.stream.clone_htod(&q)?);
            }
            let w: [CudaSlice<i8>; 6] = ws.try_into().map_err(|_| "внутренняя ошибка")?;
            layers.push(Layer {
                w,
                gam,
                ln1g: gpu.upload(&model.t[o + 6])?,
                ln1b: gpu.upload(&model.t[o + 7])?,
                ln2g: gpu.upload(&model.t[o + 8])?,
                ln2b: gpu.upload(&model.t[o + 9])?,
                kc: gpu.zeros(seq * d)?,
                vc: gpu.zeros(seq * d)?,
            });
        }
        Ok(Infer {
            seq,
            pos: 0,
            d,
            dff,
            nh,
            dh,
            stream: gpu.stream.clone(),
            kernels,
            emb: gpu.upload(&model.t[0])?,
            posn: gpu.upload(&model.t[1])?,
            ngram: cfg.ngram,
            ng3: if cfg.ngram > 0 { Some(gpu.upload(&model.t[cfg.ng_base()])?) } else { None },
            ng5: if cfg.ngram > 0 { Some(gpu.upload(&model.t[cfg.ng_base() + 1])?) } else { None },
            hist: Vec::new(),
            wout: gpu.upload(&model.t[2])?,
            lnfg: gpu.upload(&model.t[3])?,
            lnfb: gpu.upload(&model.t[4])?,
            layers,
            x: gpu.zeros(d)?,
            h: gpu.zeros(d)?,
            qv: gpu.zeros(d)?,
            ctx: gpu.zeros(d)?,
            hid: gpu.zeros(dff)?,
            hff: gpu.zeros(dff)?,
            lg: gpu.zeros(VOCAB)?,
            qa: gpu.stream.alloc_zeros::<i8>(d.max(dff))?,
            sc: gpu.zeros(1)?,
            zero_frac: zeros as f32 / total.max(1) as f32,
        })
    }

    /// сбросить контекст (кэш перезаписывается по мере продвижения pos)
    pub fn reset(&mut self) {
        self.pos = 0;
        self.hist.clear();
    }

    /// Подать один токен, вернуть логиты следующего байта.
    pub fn step(&mut self, token: u8) -> Res<Vec<f32>> {
        if self.pos >= self.seq {
            return Err("инференс: контекст заполнен (нужен reset)".into());
        }
        let Infer { d, dff, nh, dh, pos, stream, kernels, emb, posn, ngram, ng3, ng5, hist, wout, lnfg, lnfb, layers, x, h, qv, ctx, hid, hff, lg, qa, sc, .. } = self;
        let (d, dff, nh, dh) = (*d as i32, *dff as i32, *nh, *dh as i32);
        let p = *pos as i32;
        let tok = token as i32;
        let scale = 1.0f32 / (dh as f32).sqrt();
        let one = || LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let grid_mv = |outn: i32| LaunchConfig { grid_dim: ((((outn as u32) * 32) + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let grid_el = |n: i32| LaunchConfig { grid_dim: (((n as u32) + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (zero, one_i) = (0i32, 1i32);
        let off = p * d;

        launch!(kernels, stream, "embed_tok", one(), &*emb, &*posn, &mut *x, &tok, &p, &d);
        hist.push(token);
        if let (Some(h3), Some(h5)) = (ng3.as_ref(), ng5.as_ref()) {
            let pp = *pos;
            let i3 = crate::model::ng_hash(hist, pp, 3, *ngram) as i32;
            let i5 = crate::model::ng_hash(hist, pp, 5, *ngram) as i32;
            launch!(kernels, stream, "add_rows2", one(), h3, h5, &mut *x, &i3, &i5, &d);
        }
        for ly in layers.iter_mut() {
            let Layer { w, gam, ln1g, ln1b, ln2g, ln2b, kc, vc } = ly;
            launch!(kernels, stream, "ln_f", one(), &*x, &*ln1g, &*ln1b, &mut *h, &d);
            launch!(kernels, stream, "q8", one(), &*h, &mut *qa, &mut *sc, &d);
            launch!(kernels, stream, "tmatvec", grid_mv(d), &w[0], &*qa, &*sc, &gam[0], &mut *qv, &d, &d, &zero, &zero);
            launch!(kernels, stream, "tmatvec", grid_mv(d), &w[1], &*qa, &*sc, &gam[1], &mut *kc, &d, &d, &off, &zero);
            launch!(kernels, stream, "tmatvec", grid_mv(d), &w[2], &*qa, &*sc, &gam[2], &mut *vc, &d, &d, &off, &zero);
            let att_cfg = LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: ((p + 1) * 4) as u32 };
            launch!(kernels, stream, "attn_dec", att_cfg, &*qv, &*kc, &*vc, &mut *ctx, &p, &d, &dh, &scale);
            launch!(kernels, stream, "q8", one(), &*ctx, &mut *qa, &mut *sc, &d);
            launch!(kernels, stream, "tmatvec", grid_mv(d), &w[3], &*qa, &*sc, &gam[3], &mut *x, &d, &d, &zero, &one_i);
            launch!(kernels, stream, "ln_f", one(), &*x, &*ln2g, &*ln2b, &mut *h, &d);
            launch!(kernels, stream, "q8", one(), &*h, &mut *qa, &mut *sc, &d);
            launch!(kernels, stream, "tmatvec", grid_mv(dff), &w[4], &*qa, &*sc, &gam[4], &mut *hid, &d, &dff, &zero, &zero);
            launch!(kernels, stream, "gelu_f", grid_el(dff), &*hid, &mut *hff, &dff);
            launch!(kernels, stream, "q8", one(), &*hff, &mut *qa, &mut *sc, &dff);
            launch!(kernels, stream, "tmatvec", grid_mv(d), &w[5], &*qa, &*sc, &gam[5], &mut *x, &dff, &d, &zero, &one_i);
        }
        launch!(kernels, stream, "ln_f", one(), &*x, &*lnfg, &*lnfb, &mut *h, &d);
        let v = VOCAB as i32;
        launch!(kernels, stream, "matvec_f32", grid_el(v), &*h, &*wout, &mut *lg, &d, &v);
        *pos += 1;
        stream.synchronize()?;
        Ok(stream.clone_dtoh(&*lg)?)
    }
}
