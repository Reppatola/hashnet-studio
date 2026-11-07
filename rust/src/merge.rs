//! Слияние чекпойнтов hashnet_proto (всё на процессоре, видеокарта не нужна).
//!
//!  avg    — взвешенное среднее весов                       (размер = размеру входов)
//!  slerp  — сферическая интерполяция двух моделей          (размер сохраняется)
//!  task   — арифметика задач: base + Σ α·(W − base)        (нужен --base)
//!  ties   — trim + elect sign + disjoint mean              (нужен --base)
//!  dare   — случайно выбросить долю разностей + task       (нужен --base)
//!  stack  — слои одной модели, затем слои другой           (размер растёт: 35М + 35М ≈ 70М)
//!
//! Для avg/slerp/task/ties/dare модели обязаны иметь одну архитектуру и общую точку старта.
//! Для stack совпадать должны d_model, heads, seq; число слоёв может быть разным.
use crate::ckpt::{self, Meta};
use crate::model::{Adam, Config, Kind, Model, Res, BASE, TPL};

pub struct Opts {
    pub method: String,
    pub models: Vec<String>,
    pub base: Option<String>,
    pub alphas: Vec<f32>,
    pub density: f32,
    pub lambda: f32,
    pub drop: f32,
    pub t: f32,
    pub take: Vec<(usize, usize)>,
    pub globals: String,
    pub out: String,
    pub seed: u64,
}

struct Rng64(u64);
impl Rng64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / 16777216.0
    }
}

// ───────────── операции над одним тензором ─────────────

pub fn avg_t(ts: &[&[f32]], w: &[f32]) -> Vec<f32> {
    let n = ts[0].len();
    let mut o = vec![0f32; n];
    for (m, t) in ts.iter().enumerate() {
        let a = w[m];
        for k in 0..n {
            o[k] += a * t[k];
        }
    }
    o
}

pub fn task_t(base: &[f32], ts: &[&[f32]], w: &[f32]) -> Vec<f32> {
    let mut o = base.to_vec();
    for (m, t) in ts.iter().enumerate() {
        for k in 0..o.len() {
            o[k] += w[m] * (t[k] - base[k]);
        }
    }
    o
}

pub fn dare_t(base: &[f32], ts: &[&[f32]], w: &[f32], p: f32, rng: &mut Rng64) -> Vec<f32> {
    let mut o = base.to_vec();
    let sc = 1.0 / (1.0 - p).max(1e-6);
    for (m, t) in ts.iter().enumerate() {
        for k in 0..o.len() {
            if p <= 0.0 || rng.f32() >= p {
                o[k] += w[m] * (t[k] - base[k]) * if p > 0.0 { sc } else { 1.0 };
            }
        }
    }
    o
}

pub fn ties_t(base: &[f32], ts: &[&[f32]], density: f32, lambda: f32) -> Vec<f32> {
    let n = base.len();
    let mut taus: Vec<Vec<f32>> = ts.iter().map(|t| (0..n).map(|k| t[k] - base[k]).collect()).collect();
    if density < 1.0 {
        let keep = ((n as f32) * density).ceil().max(1.0) as usize;
        if keep < n {
            for tau in taus.iter_mut() {
                let mut abs: Vec<f32> = tau.iter().map(|x| x.abs()).collect();
                let idx = n - keep;
                let _ = abs.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let thr = abs[idx];
                for x in tau.iter_mut() {
                    if x.abs() < thr {
                        *x = 0.0;
                    }
                }
            }
        }
    }
    let mut o = base.to_vec();
    for k in 0..n {
        let s: f32 = taus.iter().map(|t| t[k]).sum();
        if s == 0.0 {
            continue;
        }
        let sign = s.signum();
        let (mut sum, mut cnt) = (0f32, 0u32);
        for t in &taus {
            let v = t[k];
            if v != 0.0 && v.signum() == sign {
                sum += v;
                cnt += 1;
            }
        }
        if cnt > 0 {
            o[k] += lambda * sum / cnt as f32;
        }
    }
    o
}

pub fn slerp_t(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for k in 0..a.len() {
        dot += a[k] as f64 * b[k] as f64;
        na += (a[k] as f64).powi(2);
        nb += (b[k] as f64).powi(2);
    }
    let c = if na > 0.0 && nb > 0.0 { (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0) } else { 1.0 };
    let (wa, wb) = if c.abs() > 0.9995 {
        (1.0 - t as f64, t as f64)
    } else {
        let th = c.acos();
        (((1.0 - t as f64) * th).sin() / th.sin(), (t as f64 * th).sin() / th.sin())
    };
    (0..a.len()).map(|k| (wa * a[k] as f64 + wb * b[k] as f64) as f32).collect()
}

// ───────────── слияние моделей целиком ─────────────

fn same_cfg(a: &Config, b: &Config) -> bool {
    a == b
}

fn combine(o: &Opts, base: Option<&Model>, ms: &[&Model]) -> Res<Model> {
    let cfg = ms[0].cfg.clone();
    for m in ms {
        if !same_cfg(&m.cfg, &cfg) {
            return Err("у моделей разная архитектура (d_model/слои/головы/seq): для этого метода нужны одинаковые; для разных слоёв есть метод stack".into());
        }
    }
    if let Some(b) = base {
        if !same_cfg(&b.cfg, &cfg) {
            return Err("архитектура --base не совпадает с моделями".into());
        }
    }
    let n = ms.len();
    let need_base = matches!(o.method.as_str(), "task" | "ties" | "dare");
    if need_base && base.is_none() {
        return Err(format!("метод {} требует --base (общую исходную модель)", o.method).into());
    }
    if o.method == "slerp" && n != 2 {
        return Err("slerp работает ровно с двумя моделями".into());
    }
    let alphas: Vec<f32> = if o.alphas.is_empty() {
        match o.method.as_str() {
            "avg" => vec![1.0 / n as f32; n],
            _ => vec![0.5; n],
        }
    } else if o.alphas.len() == n {
        o.alphas.clone()
    } else {
        return Err(format!("--alpha: нужно {} чисел через запятую", n).into());
    };
    let wsum: f32 = alphas.iter().sum();
    let w_avg: Vec<f32> = if o.method == "avg" { alphas.iter().map(|a| a / wsum).collect() } else { alphas.clone() };
    let mut rng = Rng64(o.seed ^ 0xD1B54A32D192ED03);
    let mut t_out = Vec::with_capacity(cfg.n_tensors());
    for i in 0..cfg.n_tensors() {
        let ts: Vec<&[f32]> = ms.iter().map(|m| m.t[i].as_slice()).collect();
        let bt = base.map(|b| b.t[i].as_slice());
        let is_mat = matches!(cfg.kind(i), Kind::Mat);
        let v = match o.method.as_str() {
            "avg" => avg_t(&ts, &w_avg),
            "slerp" => slerp_t(ts[0], ts[1], o.t),
            "task" => task_t(bt.unwrap(), &ts, &w_avg),
            "dare" => dare_t(bt.unwrap(), &ts, &w_avg, if is_mat { o.drop } else { 0.0 }, &mut rng),
            "ties" => ties_t(bt.unwrap(), &ts, if is_mat { o.density } else { 1.0 }, o.lambda),
            m => return Err(format!("неизвестный метод '{}'", m).into()),
        };
        t_out.push(v);
    }
    Ok(Model { cfg, t: t_out })
}

fn stack(o: &Opts, ms: &[&Model]) -> Res<Model> {
    let c0 = &ms[0].cfg;
    for m in ms {
        if m.cfg.d_model != c0.d_model || m.cfg.n_heads != c0.n_heads || m.cfg.seq_len != c0.seq_len {
            return Err("для stack совпадать должны d_model, число голов и seq".into());
        }
    }
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for (i, m) in ms.iter().enumerate() {
        let (s, e) = o.take.get(i).cloned().unwrap_or((0, m.cfg.n_layers));
        if s >= e || e > m.cfg.n_layers {
            return Err(format!("--take для модели {}: диапазон {}-{} вне 0..{}", i + 1, s, e, m.cfg.n_layers).into());
        }
        spans.push((s, e));
    }
    let total: usize = spans.iter().map(|(s, e)| e - s).sum();
    let cfg = Config { d_model: c0.d_model, n_layers: total, n_heads: c0.n_heads, seq_len: c0.seq_len };
    let mut t: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_tensors());
    for i in 0..BASE {
        let v = match o.globals.as_str() {
            "last" => ms[ms.len() - 1].t[i].clone(),
            "avg" => {
                let ts: Vec<&[f32]> = ms.iter().map(|m| m.t[i].as_slice()).collect();
                avg_t(&ts, &vec![1.0 / ms.len() as f32; ms.len()])
            }
            _ => ms[0].t[i].clone(),
        };
        t.push(v);
    }
    for (mi, m) in ms.iter().enumerate() {
        let (s, e) = spans[mi];
        for l in s..e {
            for j in 0..TPL {
                t.push(m.t[BASE + l * TPL + j].clone());
            }
        }
    }
    Ok(Model { cfg, t })
}

fn flat_stats(base: &Model, ms: &[&Model]) {
    // косинус между векторами задач (W − base) по всем параметрам: близко к 0 — модели почти не мешают друг другу
    let taus: Vec<Vec<f64>> = ms
        .iter()
        .map(|m| m.t.iter().zip(&base.t).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (*x - *y) as f64)).collect())
        .collect();
    let norm = |v: &Vec<f64>| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let bn: f64 = base.t.iter().flat_map(|v| v.iter()).map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    for (i, t) in taus.iter().enumerate() {
        println!("  модель {}: |W−base| = {:.4}  ({:.2}% от |base|)", i + 1, norm(t), 100.0 * norm(t) / bn.max(1e-12));
    }
    for i in 0..taus.len() {
        for j in i + 1..taus.len() {
            let d: f64 = taus[i].iter().zip(&taus[j]).map(|(a, b)| a * b).sum();
            println!("  cos(τ{}, τ{}) = {:+.3}", i + 1, j + 1, d / (norm(&taus[i]) * norm(&taus[j])).max(1e-12));
        }
    }
}

pub fn run(o: &Opts) -> Res<()> {
    if o.models.is_empty() {
        return Err("нужен хотя бы один --model".into());
    }
    if !ckpt::list(&o.out).is_empty() {
        return Err(format!("в '{}' уже есть чекпойнты: укажи новую папку в --out", o.out).into());
    }
    let mut loaded: Vec<(Meta, Model)> = Vec::new();
    for p in &o.models {
        let path = ckpt::find_latest(p)?;
        let (meta, model, _adam) = ckpt::load(&path)?;
        println!("  загружен {}  (шаг {}, d={} слоёв={} голов={} seq={})", path.display(), meta.step, meta.cfg.d_model, meta.cfg.n_layers, meta.cfg.n_heads, meta.cfg.seq_len);
        loaded.push((meta, model));
    }
    let base_model: Option<Model> = match &o.base {
        Some(p) => {
            let path = ckpt::find_latest(p)?;
            let (_m, model, _a) = ckpt::load(&path)?;
            println!("  база     {}", path.display());
            Some(model)
        }
        None => None,
    };
    let refs: Vec<&Model> = loaded.iter().map(|x| &x.1).collect();
    if let Some(b) = &base_model {
        if refs.iter().all(|m| m.cfg == b.cfg) {
            flat_stats(b, &refs);
        }
    }
    let out_model = if o.method == "stack" { stack(o, &refs)? } else { combine(o, base_model.as_ref(), &refs)? };
    let params: usize = out_model.t.iter().map(|v| v.len()).sum();
    println!("  результат: метод {}, d={} слоёв={} голов={} seq={}, параметров {:.2} млн", o.method, out_model.cfg.d_model, out_model.cfg.n_layers, out_model.cfg.n_heads, out_model.cfg.seq_len, params as f64 / 1e6);
    if out_model.t.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
        return Err("в результате есть NaN/inf — слияние отменено".into());
    }
    let mut meta = loaded[0].0.clone();
    meta.cfg = out_model.cfg.clone();
    meta.step = loaded.iter().map(|x| x.0.step).max().unwrap_or(0);
    meta.phase_done = true;
    meta.phase_start = meta.step;
    let adam = Adam::new(&out_model);
    let p = ckpt::save(&o.out, &meta, &out_model, &adam, 3)?;
    println!("  сохранено: {}", p.display());
    println!("  Проверка:  hashnet_proto train --resume --ckpt-dir {} --steps 1 --lr 1e-12 --warmup 1   (напечатает val по всем сегментам)", o.out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(x: &[f32]) -> Vec<f32> { x.to_vec() }

    #[test]
    fn avg_and_task_half_equal() {
        let base = v(&[0.0, 1.0, -1.0, 2.0]);
        let a = v(&[1.0, 1.0, 0.0, 4.0]);
        let b = v(&[3.0, -1.0, 0.0, 0.0]);
        let ts: Vec<&[f32]> = vec![&a, &b];
        let avg = avg_t(&ts, &[0.5, 0.5]);
        let task = task_t(&base, &ts, &[0.5, 0.5]);
        for k in 0..4 { assert!((avg[k] - task[k]).abs() < 1e-6, "{} {}", avg[k], task[k]); }
        assert_eq!(avg, v(&[2.0, 0.0, 0.0, 2.0]));
    }

    #[test]
    fn dare_zero_drop_equals_task() {
        let base = v(&[0.0, 1.0, 2.0]);
        let a = v(&[1.0, 2.0, 3.0]);
        let ts: Vec<&[f32]> = vec![&a];
        let mut r = Rng64(1);
        assert_eq!(dare_t(&base, &ts, &[0.7], 0.0, &mut r), task_t(&base, &ts, &[0.7]));
    }

    #[test]
    fn dare_drop_keeps_expectation() {
        let n = 200000;
        let base = vec![0f32; n];
        let a = vec![1f32; n];
        let ts: Vec<&[f32]> = vec![&a];
        let mut r = Rng64(5);
        let o = dare_t(&base, &ts, &[1.0], 0.9, &mut r);
        let mean: f32 = o.iter().sum::<f32>() / n as f32;
        assert!((mean - 1.0).abs() < 0.05, "mean {}", mean);
    }

    #[test]
    fn ties_sign_election_and_trim() {
        let base = v(&[0.0, 0.0, 0.0, 0.0]);
        let a = v(&[4.0, 1.0, 0.1, -3.0]);
        let b = v(&[2.0, -3.0, 0.1, -1.0]);
        let ts: Vec<&[f32]> = vec![&a, &b];
        // без обрезки: [0]: оба +, среднее 3; [1]: сумма −2 → знак −, берётся только b=-3; [2]: +, среднее 0.1; [3]: оба −, среднее −2
        let o = ties_t(&base, &ts, 1.0, 1.0);
        assert_eq!(o, v(&[3.0, -3.0, 0.1, -2.0]));
        // с обрезкой оставляем 50% самых больших по модулю в каждой модели: a→[4,0,0,-3], b→[2,-3,0,0]
        // [0]: оба + → (4+2)/2 = 3; [1]: только b → -3; [2]: обрезано; [3]: только a → -3
        let o2 = ties_t(&base, &ts, 0.5, 1.0);
        assert_eq!(o2, v(&[3.0, -3.0, 0.0, -3.0]));
    }

    #[test]
    fn slerp_endpoints_and_norm() {
        let a = v(&[1.0, 0.0]);
        let b = v(&[0.0, 1.0]);
        let s0 = slerp_t(&a, &b, 0.0);
        let s1 = slerp_t(&a, &b, 1.0);
        assert!((s0[0] - 1.0).abs() < 1e-6 && s0[1].abs() < 1e-6);
        assert!((s1[1] - 1.0).abs() < 1e-6 && s1[0].abs() < 1e-6);
        let m = slerp_t(&a, &b, 0.5);
        assert!(((m[0] * m[0] + m[1] * m[1]) - 1.0).abs() < 1e-5);
        assert!((m[0] - 0.70710678).abs() < 1e-5);
    }

    fn opts(method: &str) -> Opts {
        Opts { method: method.into(), models: vec![], base: None, alphas: vec![], density: 1.0, lambda: 1.0, drop: 0.0, t: 0.5, take: vec![], globals: "first".into(), out: "x".into(), seed: 1 }
    }

    #[test]
    fn stack_layout_matches_config() {
        let cfg = Config { d_model: 16, n_layers: 2, n_heads: 2, seq_len: 8 };
        let a = Model::new(cfg.clone(), 1);
        let b = Model::new(cfg.clone(), 2);
        let o = stack(&opts("stack"), &[&a, &b]).unwrap();
        assert_eq!(o.cfg.n_layers, 4);
        assert_eq!(o.t.len(), o.cfg.n_tensors());
        for i in 0..o.t.len() {
            let (r, c) = o.cfg.shape(i);
            assert_eq!(o.t[i].len(), r * c, "тензор {}", i);
        }
        // слои: A0 A1 B0 B1; глобальные от A
        assert_eq!(o.t[0], a.t[0]);
        assert_eq!(o.t[BASE], a.t[BASE]);
        assert_eq!(o.t[BASE + 2 * TPL], b.t[BASE]);
        assert_eq!(o.t[BASE + 3 * TPL + 1], b.t[BASE + TPL + 1]);
        // срез слоёв: A[0..1] + B[1..2] → 2 слоя
        let mut op = opts("stack");
        op.take = vec![(0, 1), (1, 2)];
        let o2 = stack(&op, &[&a, &b]).unwrap();
        assert_eq!(o2.cfg.n_layers, 2);
        assert_eq!(o2.t[BASE + TPL], b.t[BASE + TPL]);
    }

    #[test]
    fn combine_avg_of_identical_is_identity_and_checks_cfg() {
        let cfg = Config { d_model: 16, n_layers: 2, n_heads: 2, seq_len: 8 };
        let a = Model::new(cfg.clone(), 1);
        let o = combine(&opts("avg"), None, &[&a, &a]).unwrap();
        for i in 0..a.t.len() { for k in 0..a.t[i].len() { assert!((o.t[i][k] - a.t[i][k]).abs() < 1e-6); } }
        let c2 = Config { d_model: 16, n_layers: 3, n_heads: 2, seq_len: 8 };
        let b = Model::new(c2, 2);
        assert!(combine(&opts("avg"), None, &[&a, &b]).is_err());
        assert!(combine(&opts("ties"), None, &[&a, &a]).is_err()); // нет base
    }

    #[test]
    fn end_to_end_ckpt_roundtrip() {
        let dir = std::env::temp_dir().join(format!("hn_merge_test_{}", std::process::id()));
        let d = |n: &str| dir.join(n).to_string_lossy().to_string();
        let cfg = Config { d_model: 16, n_layers: 2, n_heads: 2, seq_len: 8 };
        let mk = |seed: u64, step: u64| -> (Meta, Model) {
            let m = Model::new(cfg.clone(), seed);
            (Meta { cfg: cfg.clone(), step, phase: 1, rng: 7, tokens: 100, phase_done: false, phase_start: 0, phase_steps: 10, phase_lr: 1e-3, files: vec![], segs: vec![] }, m)
        };
        let (ma, a) = mk(1, 10);
        let (mb, b) = mk(2, 20);
        ckpt::save(&d("a"), &ma, &a, &Adam::new(&a), 3).unwrap();
        ckpt::save(&d("b"), &mb, &b, &Adam::new(&b), 3).unwrap();
        let mut o = opts("stack");
        o.models = vec![d("a"), d("b")];
        o.out = d("st");
        run(&o).unwrap();
        let (meta, m, ad) = ckpt::load(&ckpt::find_latest(&d("st")).unwrap()).unwrap();
        assert_eq!(meta.cfg.n_layers, 4);
        assert_eq!(meta.step, 20);
        assert!(meta.phase_done);
        assert_eq!(ad.step, 0);
        assert_eq!(m.t.len(), meta.cfg.n_tensors());
        assert_eq!(m.t[BASE + 2 * TPL], b.t[BASE]);
        let mut o2 = opts("avg");
        o2.models = vec![d("a"), d("b")];
        o2.out = d("av");
        run(&o2).unwrap();
        let (_m2, av, _) = ckpt::load(&ckpt::find_latest(&d("av")).unwrap()).unwrap();
        assert!((av.t[0][3] - 0.5 * (a.t[0][3] + b.t[0][3])).abs() < 1e-6);
        assert!(run(&o2).is_err()); // папка уже занята
        let _ = std::fs::remove_dir_all(&dir);
    }
}
