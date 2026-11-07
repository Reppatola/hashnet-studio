//! HashNet4D — байтовая BitNet-подобная модель (тернарные веса {-1,0,+1}·scale).
//!
//! Версия 2: всё обучение на устройстве. Хост хранит только копию весов/Adam для чекпойнтов.
//!  * все тензоры (веса, градиенты, Adam, активации) живут в видеопамяти весь шаг;
//!  * один SGEMM-ядро с флагами транспонирования и пакетным режимом — внимание всех
//!    голов считается одним запуском (без gather/scatter и копирования через PCIe);
//!  * LayerNorm, GELU, fake-quant активаций, softmax, loss, квантизация весов и AdamW — ядра CUDA;
//!  * на хост за шаг возвращаются два числа: норма градиента и суммарный NLL.
use crate::gpu::{GpuBitwise, G, R};
use crate::util::Rng;
use cudarc::driver::CudaSlice;

pub type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub const VOCAB: usize = 256;
/// Тензоры 0..BASE: 0 w_emb, 1 w_pos, 2 w_out, 3 lnf_gamma, 4 lnf_beta
pub const BASE: usize = 5;
/// На слой: 0 wq, 1 wk, 2 wv, 3 wo, 4 w1, 5 w2, 6 g1, 7 b1, 8 g2, 9 b2
pub const TPL: usize = 10;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Emb,
    Pos,
    Out,
    Mat,
    Gain,
    Bias,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub seq_len: usize,
}

impl Config {
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }
    pub fn d_ff(&self) -> usize {
        4 * self.d_model
    }
    pub fn n_tensors(&self) -> usize {
        BASE + self.n_layers * TPL
    }
    pub fn validate(&self) -> Res<()> {
        if self.n_heads == 0 || self.d_model == 0 || self.d_model % self.n_heads != 0 {
            return Err("d_model должен делиться на heads без остатка".into());
        }
        if self.n_layers == 0 {
            return Err("layers должно быть >= 1".into());
        }
        if self.seq_len < 2 {
            return Err("seq должно быть >= 2".into());
        }
        Ok(())
    }
    pub fn kind(&self, i: usize) -> Kind {
        match i {
            0 => Kind::Emb,
            1 => Kind::Pos,
            2 => Kind::Out,
            3 => Kind::Gain,
            4 => Kind::Bias,
            _ => match (i - BASE) % TPL {
                0..=5 => Kind::Mat,
                6 | 8 => Kind::Gain,
                _ => Kind::Bias,
            },
        }
    }
    /// (строки, столбцы) — матрицы хранятся как [in, out] построчно
    pub fn shape(&self, i: usize) -> (usize, usize) {
        let d = self.d_model;
        let ff = self.d_ff();
        match i {
            0 => (VOCAB, d),
            1 => (self.seq_len, d),
            2 => (d, VOCAB),
            3 | 4 => (1, d),
            _ => match (i - BASE) % TPL {
                0..=3 => (d, d),
                4 => (d, ff),
                5 => (ff, d),
                _ => (1, d),
            },
        }
    }
    pub fn n_params(&self) -> u64 {
        (0..self.n_tensors()).map(|i| {
            let (r, c) = self.shape(i);
            (r * c) as u64
        }).sum()
    }
    pub fn n_mat_params(&self) -> u64 {
        (0..self.n_tensors()).filter(|&i| self.kind(i) == Kind::Mat).map(|i| {
            let (r, c) = self.shape(i);
            (r * c) as u64
        }).sum()
    }
}


pub struct Model {
    pub cfg: Config,
    pub t: Vec<Vec<f32>>,
}

impl Model {
    pub fn new(cfg: Config, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let res_scale = 1.0 / ((2 * cfg.n_layers) as f32).sqrt();
        let mut t = Vec::with_capacity(cfg.n_tensors());
        for i in 0..cfg.n_tensors() {
            let (r, c) = cfg.shape(i);
            let len = r * c;
            let v: Vec<f32> = match cfg.kind(i) {
                Kind::Emb | Kind::Pos | Kind::Out => (0..len).map(|_| rng.normal() * 0.02).collect(),
                Kind::Mat => {
                    let j = (i - BASE) % TPL;
                    let mut s = 1.0 / (r as f32).sqrt();
                    if j == 3 || j == 5 {
                        s *= res_scale;
                    }
                    (0..len).map(|_| rng.normal() * s).collect()
                }
                Kind::Gain => vec![1.0; len],
                Kind::Bias => vec![0.0; len],
            };
            t.push(v);
        }
        Model { cfg, t }
    }

    pub fn zero_grads(&self) -> Vec<Vec<f32>> {
        self.t.iter().map(|v| vec![0.0f32; v.len()]).collect()
    }
}

#[derive(Clone, Debug)]
pub struct Hyper {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub wd: f32,
    pub clip: f32,
}

/// Хост-копия состояния Adam (нужна только для чекпойнтов; шаг считается на GPU в `Trainer`)
pub struct Adam {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub step: u64,
}

impl Adam {
    pub fn new(model: &Model) -> Self {
        Adam { m: model.zero_grads(), v: model.zero_grads(), step: 0 }
    }
}

pub fn tensor_name(cfg: &Config, i: usize) -> String {
    match i {
        0 => "w_emb".into(),
        1 => "w_pos".into(),
        2 => "w_out".into(),
        3 => "lnf_gamma".into(),
        4 => "lnf_beta".into(),
        _ => {
            let l = (i - BASE) / TPL;
            let nm = ["wq", "wk", "wv", "wo", "w1", "w2", "ln1_g", "ln1_b", "ln2_g", "ln2_b"][(i - BASE) % TPL];
            let _ = cfg;
            format!("L{}.{}", l, nm)
        }
    }
}

// ───────────────────────────── состояние на GPU ─────────────────────────────

struct Layer {
    xn1: CudaSlice<f32>,
    std1: CudaSlice<f32>,
    a1: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    probs: CudaSlice<f32>,
    actx: CudaSlice<f32>,
    xn2: CudaSlice<f32>,
    std2: CudaSlice<f32>,
    a2: CudaSlice<f32>,
    hpre: CudaSlice<f32>,
    ah: CudaSlice<f32>,
}

/// Всё обучение на устройстве: веса, градиенты, Adam, активации. На хост за шаг
/// возвращаются только 2 числа (норма градиента и суммарный NLL).
pub struct Trainer {
    pub cfg: Config,
    /// false → «fp32-режим» без квантизации (контрольный эксперимент / проверка градиентов)
    pub quant: bool,
    pub cap_b: usize,
    pub adam_step: u64,
    p: Vec<CudaSlice<f32>>,
    g: Vec<CudaSlice<f32>>,
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    qw: Vec<Option<CudaSlice<f32>>>,
    qacc: CudaSlice<f32>,
    zc: CudaSlice<f32>,
    ssq: CudaSlice<f32>,
    nll: CudaSlice<f32>,
    tok: CudaSlice<u8>,
    tgt: CudaSlice<u8>,
    xs: Vec<CudaSlice<f32>>,
    layers: Vec<Layer>,
    h: CudaSlice<f32>,
    ctx: CudaSlice<f32>,
    att: CudaSlice<f32>,
    x2: CudaSlice<f32>,
    f: CudaSlice<f32>,
    hact: CudaSlice<f32>,
    hf: CudaSlice<f32>,
    xnf: CudaSlice<f32>,
    stdf: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    dl: CudaSlice<f32>,
    dx: CudaSlice<f32>,
    dx2: CudaSlice<f32>,
    dhf: CudaSlice<f32>,
    dah: CudaSlice<f32>,
    dhpre: CudaSlice<f32>,
    da2: CudaSlice<f32>,
    dctx: CudaSlice<f32>,
    dq: CudaSlice<f32>,
    dk: CudaSlice<f32>,
    dv: CudaSlice<f32>,
    da1: CudaSlice<f32>,
    dp: CudaSlice<f32>,
}

fn wq(qw: &[Option<CudaSlice<f32>>], i: usize) -> &CudaSlice<f32> {
    qw[i].as_ref().expect("квантованный тензор отсутствует")
}

fn two_mut(g: &mut [CudaSlice<f32>], i: usize, j: usize) -> (&mut CudaSlice<f32>, &mut CudaSlice<f32>) {
    assert!(i < j);
    let (a, b) = g.split_at_mut(j);
    (&mut a[i], &mut b[0])
}

fn aqx(gpu: &GpuBitwise, quant: bool, src: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, rows: usize, d: usize) -> R {
    if quant {
        gpu.aq(src, dst, rows, d)
    } else {
        gpu.copy(src, dst, rows * d)
    }
}

impl Trainer {
    pub fn new(gpu: &GpuBitwise, model: &Model, adam: &Adam, cap_b: usize, quant: bool) -> Res<Trainer> {
        let cfg = model.cfg.clone();
        let (d, dff, nh, tm) = (cfg.d_model, cfg.d_ff(), cfg.n_heads, cfg.seq_len);
        let nt = cfg.n_tensors();
        let nn = cap_b * tm;
        let z = |n: usize| gpu.zeros(n);
        let mut p = Vec::new();
        let mut g = Vec::new();
        let mut m = Vec::new();
        let mut v = Vec::new();
        let mut qw = Vec::new();
        for i in 0..nt {
            p.push(gpu.upload(&model.t[i])?);
            m.push(gpu.upload(&adam.m[i])?);
            v.push(gpu.upload(&adam.v[i])?);
            g.push(z(model.t[i].len())?);
            qw.push(if cfg.kind(i) == Kind::Mat { Some(z(model.t[i].len())?) } else { None });
        }
        let mut xs = Vec::new();
        for _ in 0..=cfg.n_layers {
            xs.push(z(nn * d)?);
        }
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
                xn1: z(nn * d)?,
                std1: z(nn)?,
                a1: z(nn * d)?,
                q: z(nn * d)?,
                k: z(nn * d)?,
                v: z(nn * d)?,
                probs: z(cap_b * nh * tm * tm)?,
                actx: z(nn * d)?,
                xn2: z(nn * d)?,
                std2: z(nn)?,
                a2: z(nn * d)?,
                hpre: z(nn * dff)?,
                ah: z(nn * dff)?,
            });
        }
        Ok(Trainer {
            quant,
            cap_b,
            adam_step: adam.step,
            p,
            g,
            m,
            v,
            qw,
            qacc: z(nt)?,
            zc: z(1)?,
            ssq: z(1)?,
            nll: z(1)?,
            tok: gpu.zeros_u8(nn)?,
            tgt: gpu.zeros_u8(nn)?,
            xs,
            layers,
            h: z(nn * d)?,
            ctx: z(nn * d)?,
            att: z(nn * d)?,
            x2: z(nn * d)?,
            f: z(nn * d)?,
            hact: z(nn * dff)?,
            hf: z(nn * d)?,
            xnf: z(nn * d)?,
            stdf: z(nn)?,
            logits: z(nn * VOCAB)?,
            dl: z(nn * VOCAB)?,
            dx: z(nn * d)?,
            dx2: z(nn * d)?,
            dhf: z(nn * d)?,
            dah: z(nn * dff)?,
            dhpre: z(nn * dff)?,
            da2: z(nn * d)?,
            dctx: z(nn * d)?,
            dq: z(nn * d)?,
            dk: z(nn * d)?,
            dv: z(nn * d)?,
            da1: z(nn * d)?,
            dp: z(cap_b * nh * tm * tm)?,
            cfg,
        })
    }

    /// Эффективные веса шага: absmean-тернарные (BitNet b1.58) или копия (fp32-режим)
    pub fn quantize(&mut self, gpu: &GpuBitwise) -> R {
        let nt = self.cfg.n_tensors();
        gpu.zero(&mut self.qacc, nt)?;
        gpu.zero(&mut self.zc, 1)?;
        for i in 0..nt {
            if self.cfg.kind(i) != Kind::Mat {
                continue;
            }
            let len = self.p[i].len();
            let dst = self.qw[i].as_mut().unwrap();
            if self.quant {
                gpu.absum(&self.p[i], len, &mut self.qacc, i)?;
                gpu.ternarize(&self.p[i], dst, len, &self.qacc, i, &mut self.zc)?;
            } else {
                gpu.copy(&self.p[i], dst, len)?;
            }
        }
        Ok(())
    }

    pub fn zero_frac(&self, gpu: &GpuBitwise) -> Res<f32> {
        let z = gpu.download(&self.zc)?[0];
        let tot = self.cfg.n_mat_params().max(1) as f32;
        Ok(z / tot)
    }

    pub fn zero_grads(&mut self, gpu: &GpuBitwise) -> R {
        for i in 0..self.g.len() {
            let n = self.g[i].len();
            gpu.zero(&mut self.g[i], n)?;
        }
        gpu.zero(&mut self.ssq, 1)?;
        gpu.zero(&mut self.nll, 1)?;
        Ok(())
    }

    fn forward(&mut self, gpu: &GpuBitwise, b: usize, t: usize) -> R {
        let Trainer { cfg, quant, p, qw, tok, xs, layers, h, ctx, att, x2, f, hact, hf, xnf, stdf, logits, .. } = self;
        let quant = *quant;
        let (d, dff, nh, dh) = (cfg.d_model, cfg.d_ff(), cfg.n_heads, cfg.d_head());
        let n = b * t;
        let scale = 1.0 / (dh as f32).sqrt();
        let sp = ((nh * t * t) as i64, (t * t) as i64); // шаги по (batch, head) для [b*nh, t, t]
        let sh = ((t * d) as i64, dh as i64); // шаги по (batch, head) для [b*t, d] с головами по столбцам

        gpu.embed_fwd(&p[0], &p[1], tok, &mut xs[0], n, t, d)?;
        for l in 0..cfg.n_layers {
            let o = BASE + l * TPL;
            let lc = &mut layers[l];
            let (xa, xb) = xs.split_at_mut(l + 1);
            let xin = &xa[l];
            let xout = &mut xb[0];

            gpu.ln_fwd(xin, &p[o + 6], &p[o + 7], h, &mut lc.xn1, &mut lc.std1, n, d)?;
            aqx(gpu, quant, h, &mut lc.a1, n, d)?;
            gpu.gemm(&lc.a1, wq(qw, o), &mut lc.q, G::new(n, d, d, d, d, d))?;
            gpu.gemm(&lc.a1, wq(qw, o + 1), &mut lc.k, G::new(n, d, d, d, d, d))?;
            gpu.gemm(&lc.a1, wq(qw, o + 2), &mut lc.v, G::new(n, d, d, d, d, d))?;

            // внимание: все b·nh голов одним запуском
            gpu.gemm(&lc.q, &lc.k, &mut lc.probs, G::new(t, t, dh, d, d, t).tb().alpha(scale).batched(b * nh, nh, sh, sh, sp))?;
            gpu.softmax_fwd(&mut lc.probs, b * nh * t, t)?;
            gpu.gemm(&lc.probs, &lc.v, ctx, G::new(t, dh, t, t, d, d).batched(b * nh, nh, sp, sh, sh))?;

            aqx(gpu, quant, ctx, &mut lc.actx, n, d)?;
            gpu.gemm(&lc.actx, wq(qw, o + 3), att, G::new(n, d, d, d, d, d))?;
            gpu.add(xin, att, x2, n * d)?;

            gpu.ln_fwd(x2, &p[o + 8], &p[o + 9], h, &mut lc.xn2, &mut lc.std2, n, d)?;
            aqx(gpu, quant, h, &mut lc.a2, n, d)?;
            gpu.gemm(&lc.a2, wq(qw, o + 4), &mut lc.hpre, G::new(n, dff, d, d, dff, dff))?;
            gpu.gelu_fwd(&lc.hpre, hact, n * dff)?;
            aqx(gpu, quant, hact, &mut lc.ah, n, dff)?;
            gpu.gemm(&lc.ah, wq(qw, o + 5), f, G::new(n, d, dff, dff, d, d))?;
            gpu.add(x2, f, xout, n * d)?;
        }
        gpu.ln_fwd(&xs[cfg.n_layers], &p[3], &p[4], hf, xnf, stdf, n, d)?;
        gpu.gemm(hf, &p[2], logits, G::new(n, VOCAB, d, d, VOCAB, VOCAB))?;
        Ok(())
    }

    fn backward(&mut self, gpu: &GpuBitwise, b: usize, t: usize) -> R {
        let Trainer { cfg, p, g, qw, tok, layers, hf, xnf, stdf, dl, dx, dx2, dhf, dah, dhpre, da2, dctx, dq, dk, dv, da1, dp, .. } = self;
        let (d, dff, nh, dh) = (cfg.d_model, cfg.d_ff(), cfg.n_heads, cfg.d_head());
        let n = b * t;
        let scale = 1.0 / (dh as f32).sqrt();
        let sp = ((nh * t * t) as i64, (t * t) as i64);
        let sh = ((t * d) as i64, dh as i64);

        // выходная проекция и финальный LN
        gpu.gemm(hf, dl, &mut g[2], G::new(d, VOCAB, n, d, VOCAB, VOCAB).ta().beta(1.0))?;
        gpu.gemm(dl, &p[2], dhf, G::new(n, d, VOCAB, VOCAB, VOCAB, d).tb())?;
        {
            let (gg, gb) = two_mut(g.as_mut_slice(), 3, 4);
            gpu.ln_bwd(dhf, xnf, stdf, &p[3], None, dx, gg, gb, n, d)?;
        }

        for l in (0..cfg.n_layers).rev() {
            let o = BASE + l * TPL;
            let lc = &layers[l];

            // ── FFN: out = x2 + ah·W2 ──
            gpu.gemm(&lc.ah, dx, &mut g[o + 5], G::new(dff, d, n, dff, d, d).ta().beta(1.0))?;
            gpu.gemm(dx, wq(qw, o + 5), dah, G::new(n, dff, d, d, d, dff).tb())?; // STE через aq
            gpu.gelu_bwd(dah, &lc.hpre, dhpre, n * dff)?;
            gpu.gemm(&lc.a2, dhpre, &mut g[o + 4], G::new(d, dff, n, d, dff, dff).ta().beta(1.0))?;
            gpu.gemm(dhpre, wq(qw, o + 4), da2, G::new(n, d, dff, dff, dff, d).tb())?;
            {
                let (gg, gb) = two_mut(g.as_mut_slice(), o + 8, o + 9);
                gpu.ln_bwd(da2, &lc.xn2, &lc.std2, &p[o + 8], Some(&*dx), dx2, gg, gb, n, d)?; // dx2 = dx + dLN
            }

            // ── внимание: x2 = x_in + actx·Wo ──
            gpu.gemm(&lc.actx, dx2, &mut g[o + 3], G::new(d, d, n, d, d, d).ta().beta(1.0))?;
            gpu.gemm(dx2, wq(qw, o + 3), dctx, G::new(n, d, d, d, d, d).tb())?;
            gpu.gemm(&lc.probs, dctx, dv, G::new(t, dh, t, t, d, d).ta().batched(b * nh, nh, sp, sh, sh))?; // dV = Pᵀ·dctx
            gpu.gemm(dctx, &lc.v, dp, G::new(t, t, dh, d, d, t).tb().batched(b * nh, nh, sh, sh, sp))?; // dP = dctx·Vᵀ
            gpu.softmax_bwd(&lc.probs, dp, b * nh * t, t)?; // dP → dS
            gpu.gemm(dp, &lc.k, dq, G::new(t, dh, t, t, d, d).alpha(scale).batched(b * nh, nh, sp, sh, sh))?;
            gpu.gemm(dp, &lc.q, dk, G::new(t, dh, t, t, d, d).ta().alpha(scale).batched(b * nh, nh, sp, sh, sh))?;

            gpu.gemm(&lc.a1, dq, &mut g[o], G::new(d, d, n, d, d, d).ta().beta(1.0))?;
            gpu.gemm(&lc.a1, dk, &mut g[o + 1], G::new(d, d, n, d, d, d).ta().beta(1.0))?;
            gpu.gemm(&lc.a1, dv, &mut g[o + 2], G::new(d, d, n, d, d, d).ta().beta(1.0))?;
            gpu.gemm(dq, wq(qw, o), da1, G::new(n, d, d, d, d, d).tb())?;
            gpu.gemm(dk, wq(qw, o + 1), da1, G::new(n, d, d, d, d, d).tb().beta(1.0))?;
            gpu.gemm(dv, wq(qw, o + 2), da1, G::new(n, d, d, d, d, d).tb().beta(1.0))?;
            {
                let (gg, gb) = two_mut(g.as_mut_slice(), o + 6, o + 7);
                gpu.ln_bwd(da1, &lc.xn1, &lc.std1, &p[o + 6], Some(&*dx2), dx, gg, gb, n, d)?; // dx = dx2 + dLN
            }
        }
        let (ge, gp) = two_mut(g.as_mut_slice(), 0, 1);
        gpu.embed_bwd(dx, tok, ge, gp, n, t, d)?;
        Ok(())
    }

    fn set_batch(&mut self, gpu: &GpuBitwise, toks: &[u8], tg: &[u8], b: usize, t: usize) -> R {
        if toks.len() != b * t || tg.len() != b * t {
            return Err("размер tokens/targets != b*t".into());
        }
        if b == 0 || b > self.cap_b || t == 0 || t > self.cfg.seq_len {
            return Err(format!("батч {}×{} не помещается в буферы ({}×{})", b, t, self.cap_b, self.cfg.seq_len).into());
        }
        self.tok = gpu.upload_u8(toks)?;
        self.tgt = gpu.upload_u8(tg)?;
        Ok(())
    }

    /// Один микро-батч: forward + loss + backward (градиенты накапливаются). Веса — после `quantize`.
    /// `scale` = 1/(число токенов в шаге).
    pub fn micro(&mut self, gpu: &GpuBitwise, toks: &[u8], tg: &[u8], b: usize, t: usize, scale: f32) -> R {
        self.set_batch(gpu, toks, tg, b, t)?;
        self.forward(gpu, b, t)?;
        gpu.loss(&self.logits, &self.tgt, &mut self.dl, &mut self.nll, b * t, scale, true)?;
        self.backward(gpu, b, t)
    }

    /// Клиппинг по глобальной норме + AdamW на GPU. Возвращает (норма, суммарный NLL шага, применён ли шаг).
    pub fn finish_step(&mut self, gpu: &GpuBitwise, hp: &Hyper) -> Res<(f64, f64, bool)> {
        for i in 0..self.g.len() {
            let n = self.g[i].len();
            gpu.sumsq(&self.g[i], n, &mut self.ssq)?;
        }
        let sq = gpu.download(&self.ssq)?[0] as f64;
        let nll = gpu.download(&self.nll)?[0] as f64;
        let norm = sq.sqrt();
        if !norm.is_finite() {
            return Ok((norm, nll, false));
        }
        let coef = if norm > hp.clip as f64 { (hp.clip as f64 / (norm + 1e-6)) as f32 } else { 1.0 };
        self.adam_step += 1;
        let tt = self.adam_step.min(i32::MAX as u64) as i32;
        let bc1 = 1.0 - hp.b1.powi(tt);
        let bc2 = 1.0 - hp.b2.powi(tt);
        for i in 0..self.g.len() {
            let decay = if self.cfg.kind(i) == Kind::Mat { hp.wd } else { 0.0 };
            let n = self.p[i].len();
            gpu.adam(&mut self.p[i], &self.g[i], &mut self.m[i], &mut self.v[i], n, hp.lr, hp.b1, hp.b2, hp.eps, decay, coef, bc1, bc2)?;
        }
        Ok((norm, nll, true))
    }

    /// средний NLL (нат/токен) на одном батче без градиентов; квантизация выполняется внутри
    pub fn loss_only(&mut self, gpu: &GpuBitwise, toks: &[u8], tg: &[u8], b: usize, t: usize) -> Res<f64> {
        self.quantize(gpu)?;
        self.set_batch(gpu, toks, tg, b, t)?;
        gpu.zero(&mut self.nll, 1)?;
        self.forward(gpu, b, t)?;
        gpu.loss(&self.logits, &self.tgt, &mut self.dl, &mut self.nll, b * t, 0.0, false)?;
        Ok(gpu.download(&self.nll)?[0] as f64 / (b * t) as f64)
    }

    /// средний NLL (нат/байт) по окнам длиной seq_len+1
    pub fn eval_nll(&mut self, gpu: &GpuBitwise, wins: &[Vec<u8>]) -> Res<f64> {
        let t = self.cfg.seq_len;
        self.quantize(gpu)?;
        gpu.zero(&mut self.nll, 1)?;
        let mut cnt = 0usize;
        for chunk in wins.chunks(self.cap_b) {
            let b = chunk.len();
            let mut toks = Vec::with_capacity(b * t);
            let mut tg = Vec::with_capacity(b * t);
            for w in chunk {
                toks.extend_from_slice(&w[..t]);
                tg.extend_from_slice(&w[1..t + 1]);
            }
            self.set_batch(gpu, &toks, &tg, b, t)?;
            self.forward(gpu, b, t)?;
            gpu.loss(&self.logits, &self.tgt, &mut self.dl, &mut self.nll, b * t, 0.0, false)?;
            cnt += b * t;
        }
        Ok(gpu.download(&self.nll)?[0] as f64 / cnt.max(1) as f64)
    }

    /// логиты последней позиции окна (b=1). Перед генерацией один раз вызвать `quantize`.
    pub fn logits_last(&mut self, gpu: &GpuBitwise, window: &[u8]) -> Res<Vec<f32>> {
        let t = window.len();
        self.set_batch(gpu, window, window, 1, t)?;
        self.forward(gpu, 1, t)?;
        let all = gpu.download(&self.logits)?;
        Ok(all[(t - 1) * VOCAB..t * VOCAB].to_vec())
    }

    // ───────────── обмен с хостом (чекпойнты, проверка градиентов) ─────────────
    pub fn download(&self, gpu: &GpuBitwise, model: &mut Model, adam: &mut Adam) -> R {
        for i in 0..self.p.len() {
            model.t[i] = gpu.download(&self.p[i])?;
            adam.m[i] = gpu.download(&self.m[i])?;
            adam.v[i] = gpu.download(&self.v[i])?;
        }
        adam.step = self.adam_step;
        Ok(())
    }

    pub fn download_grads(&self, gpu: &GpuBitwise) -> Res<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        for gi in &self.g {
            out.push(gpu.download(gi)?);
        }
        Ok(out)
    }

    pub fn set_param(&mut self, gpu: &GpuBitwise, i: usize, vals: &[f32]) -> R {
        self.p[i] = gpu.upload(vals)?;
        Ok(())
    }
}
