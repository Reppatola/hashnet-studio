mod ckpt;
mod merge;
mod data;
mod gpu;
mod infer;
mod model;
mod util;

use ckpt::Meta;
use data::{FileRec, Mixer, PlanOpts, Seg};
use model::{Adam, Config, Hyper, Model, Res, Trainer, VOCAB};
use std::collections::HashMap;
use std::str::FromStr;
use std::time::Instant;
use util::Rng;

// ───────────────────────────── аргументы ─────────────────────────────

struct Args {
    cmd: String,
    m: HashMap<String, Vec<String>>,
}

impl Args {
    fn parse() -> Args {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let cmd = argv.first().cloned().unwrap_or_else(|| "help".to_string());
        let mut m: HashMap<String, Vec<String>> = HashMap::new();
        let mut i = 1;
        while i < argv.len() {
            if let Some(k) = argv[i].strip_prefix("--") {
                let has_val = i + 1 < argv.len() && !argv[i + 1].starts_with("--");
                let v = if has_val { i += 1; argv[i].clone() } else { "true".to_string() };
                m.entry(k.to_string()).or_default().push(v);
            }
            i += 1;
        }
        Args { cmd, m }
    }
    fn s(&self, k: &str, def: &str) -> String {
        self.m.get(k).and_then(|v| v.last().cloned()).unwrap_or_else(|| def.to_string())
    }
    fn get<T: FromStr>(&self, k: &str, def: T) -> Res<T> {
        match self.m.get(k).and_then(|v| v.last()) {
            None => Ok(def),
            Some(s) => s.parse::<T>().map_err(|_| format!("--{}: не могу разобрать значение '{}'", k, s).into()),
        }
    }
    fn flag(&self, k: &str) -> bool {
        self.m.contains_key(k)
    }
    fn all(&self, k: &str) -> Vec<String> {
        self.m.get(k).cloned().unwrap_or_default()
    }
    fn has(&self, k: &str) -> bool {
        self.m.contains_key(k)
    }
}

// ───────────────────────────── вспомогательное ─────────────────────────────

fn bpb(nats: f64) -> f64 {
    nats / std::f64::consts::LN_2
}

fn lr_at(local: u64, total: u64, peak: f32, warmup: u64, min_frac: f32) -> f32 {
    let warm = warmup.min(total / 5).max(1);
    if local < warm {
        return peak * (local + 1) as f32 / warm as f32;
    }
    let prog = ((local - warm) as f32 / (total.saturating_sub(warm)).max(1) as f32).clamp(0.0, 1.0);
    let cos = 0.5 * (1.0 + (std::f32::consts::PI * prog).cos());
    peak * (min_frac + (1.0 - min_frac) * cos)
}

fn eval_all(tr: &mut Trainer, gpu: &gpu::GpuBitwise, vals: &[Vec<Vec<u8>>]) -> Res<Vec<Option<f64>>> {
    let mut out = Vec::new();
    for w in vals {
        if w.is_empty() {
            out.push(None);
        } else {
            out.push(Some(tr.eval_nll(gpu, w)?));
        }
    }
    Ok(out)
}

fn print_eval(title: &str, segs: &[Seg], res: &[Option<f64>], base: Option<&Vec<Option<f64>>>) {
    println!("  ── {} ──", title);
    for (i, s) in segs.iter().enumerate() {
        match res[i] {
            None => println!("     {:<44} (нет отложенной части)", s.label()),
            Some(l) => {
                let delta = match base.and_then(|b| b[i]) {
                    Some(b0) => format!("  Δ {:+.3} бит/байт к началу фазы", bpb(l) - bpb(b0)),
                    None => String::new(),
                };
                println!("     {:<44} val {:.4} нат = {:.3} бит/байт{}", s.label(), l, bpb(l), delta);
            }
        }
    }
}

fn sample_next(logits: &[f32], temp: f32, top_k: usize, rng: &mut Rng) -> u8 {
    if temp <= 1e-4 {
        let mut best = 0;
        for i in 1..VOCAB {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return best as u8;
    }
    let mut idx: Vec<usize> = (0..VOCAB).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    let k = if top_k == 0 { VOCAB } else { top_k.min(VOCAB) };
    let mx = logits[idx[0]];
    let probs: Vec<f32> = idx[..k].iter().map(|&i| ((logits[i] - mx) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let r = rng.f32() * sum;
    let mut c = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        c += p;
        if c >= r {
            return idx[j] as u8;
        }
    }
    idx[k - 1] as u8
}

fn generate(tr: &mut Trainer, gpu: &gpu::GpuBitwise, prompt: &[u8], len: usize, temp: f32, top_k: usize, rng: &mut Rng) -> Res<Vec<u8>> {
    tr.quantize(gpu)?;
    let t_max = tr.cfg.seq_len;
    let mut ctx: Vec<u8> = if prompt.is_empty() { vec![b'\n'] } else { prompt.to_vec() };
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let start = ctx.len().saturating_sub(t_max);
        let window = &ctx[start..];
        // каузальная модель: берём логиты ПОСЛЕДНЕЙ реальной позиции (без паддинга)
        let row = tr.logits_last(gpu, window)?;
        let b = sample_next(&row, temp, top_k, rng);
        out.push(b);
        ctx.push(b);
    }
    Ok(out)
}

fn print_model_info(cfg: &Config) {
    let total = cfg.n_params();
    let mat = cfg.n_mat_params();
    println!("  параметров всего: {}  (тернарных матриц: {})", util::count(total), util::count(mat));
    println!(
        "  память обучения (f32 вес + Adam m,v + градиенты ≈ 16 Б/параметр): {}",
        util::mb(total * 16)
    );
    println!(
        "  размер для инференса: матрицы ≈1.58 бит/вес = {}, остальное f32 = {}",
        util::mb((mat as f64 * 1.58 / 8.0) as u64),
        util::mb((total - mat) * 4)
    );
}

// ───────────────────────────── train ─────────────────────────────

fn cmd_train(a: &Args) -> Res<()> {
    util::install_ctrlc();
    let ckpt_dir = a.s("ckpt-dir", "ckpt");
    let resume = a.flag("resume");
    let keep: usize = a.get("keep", 3)?;
    let ckpt_every: u64 = a.get("ckpt-every", 1000)?;
    let eval_every: u64 = a.get("eval-every", 500)?;
    let log_every: u64 = a.get("log-every", 50)?;
    let batch: usize = a.get("batch", 16)?;
    let accum: usize = a.get("accum", 1)?;
    let warmup: u64 = a.get("warmup", 200)?;
    let min_lr_frac: f32 = a.get("min-lr-frac", 0.1)?;
    let replay_frac: f64 = a.get("replay-frac", 0.3)?;
    let seed: u64 = a.get("seed", 1234)?;
    let steps_arg: u64 = a.get("steps", 5000)?;
    let lr_arg: f32 = a.get("lr", 1e-3)?;
    if batch == 0 || accum == 0 {
        return Err("--batch и --accum должны быть >= 1".into());
    }

    let gpu = gpu::GpuBitwise::new()?;

    // ── модель: новая или из чекпойнта ──
    let (mut meta, mut model, mut adam, resumed) = if resume {
        let p = ckpt::find_latest(&ckpt_dir)?;
        let (m, mo, ad) = ckpt::load(&p)?;
        println!("↻ Продолжаю с чекпойнта {}  (шаг {}, фаза {}, токенов видел {})", p.display(), m.step, m.phase, util::count(m.tokens));
        for k in ["d-model", "layers", "heads", "seq"] {
            if a.has(k) {
                println!("  ⚠ --{} игнорируется: архитектура берётся из чекпойнта", k);
            }
        }
        (m, mo, ad, true)
    } else {
        if !ckpt::list(&ckpt_dir).is_empty() {
            return Err(format!(
                "В каталоге '{}' уже есть чекпойнты. Чтобы дообучать — добавь --resume. \
                 Чтобы начать с нуля — укажи другой --ckpt-dir.",
                ckpt_dir
            )
            .into());
        }
        let cfg = Config {
            d_model: a.get("d-model", 128)?,
            n_layers: a.get("layers", 4)?,
            n_heads: a.get("heads", 4)?,
            seq_len: a.get("seq", 64)?,
        };
        cfg.validate()?;
        let model = Model::new(cfg.clone(), seed);
        let adam = Adam::new(&model);
        let meta = Meta {
            cfg,
            step: 0,
            phase: 0,
            rng: seed ^ 0x5DEECE66D,
            tokens: 0,
            phase_done: true,
            phase_start: 0,
            phase_steps: 0,
            phase_lr: lr_arg,
            files: Vec::<FileRec>::new(),
            segs: Vec::<Seg>::new(),
        };
        (meta, model, adam, false)
    };
    let seq = model.cfg.seq_len;
    let win = seq + 1;
    print_model_info(&model.cfg);
    let quant = !a.flag("no-quant");
    if !quant {
        println!("  ⚠ --no-quant: fp32-режим (без тернаризации весов и квантизации активаций) — контрольный эксперимент");
    }
    // буфер активаций на GPU рассчитан на batch (но не меньше 8 — для валидации)
    let mut tr = Trainer::new(&gpu, &model, &adam, batch, quant)?;

    // ── план фазы обучения ──
    // --data путь  или  --data путь@доля  (доли относительные: файл.txt@0.4 другой.txt@0.2 ...)
    let mut data_paths: Vec<String> = Vec::new();
    let mut data_w: Vec<Option<f64>> = Vec::new();
    for d in a.all("data") {
        match d.rsplit_once('@') {
            Some((p, w)) if w.parse::<f64>().is_ok() => {
                data_paths.push(p.to_string());
                data_w.push(Some(w.parse::<f64>().unwrap()));
            }
            _ => {
                data_paths.push(d);
                data_w.push(None);
            }
        }
    }
    let continue_phase = resumed && !meta.phase_done && !a.flag("new-phase");
    if continue_phase {
        println!("  Прерванная фаза {} будет доведена до конца (шаги {}..{}).", meta.phase, meta.step, meta.phase_start + meta.phase_steps);
        if !data_paths.is_empty() {
            println!("  ⚠ --data игнорируется при продолжении прерванной фазы; чтобы начать новую фазу — добавь --new-phase");
        }
    } else {
        if data_paths.is_empty() && meta.segs.is_empty() {
            return Err("Укажи данные: --data файл.txt (можно несколько раз)".into());
        }
        if data_paths.is_empty() {
            println!("  --data не указан: продолжаю обучение на прежних данных.");
        }
        let opts = PlanOpts { win, replay_frac, val_cap: 4 << 20, weights: data_w };
        let (files, segs, rep) = data::plan(&data_paths, &meta.files, &meta.segs, meta.phase + 1, &opts)?;
        for l in &rep.lines {
            println!("  {}", l);
        }
        meta.files = files;
        meta.segs = segs;
        meta.phase += 1;
        meta.phase_start = meta.step;
        meta.phase_steps = steps_arg;
        meta.phase_lr = lr_arg;
        meta.phase_done = false;
    }

    let mixer = Mixer::new(&meta.segs, win)?;
    let vals: Vec<Vec<Vec<u8>>> = meta.segs.iter().map(|s| data::val_windows(s, win, 64)).collect::<Res<Vec<_>>>()?;

    println!("══════════════════════════════════════════════════════════════");
    println!(
        "  Фаза {}  |  d={} слоёв={} голов={} seq={}  |  batch={}×{}  lr={}  шагов={}",
        meta.phase, model.cfg.d_model, model.cfg.n_layers, model.cfg.n_heads, seq, batch, accum, meta.phase_lr, meta.phase_steps
    );
    println!("══════════════════════════════════════════════════════════════");

    let base = eval_all(&mut tr, &gpu, &vals)?;
    print_eval("валидация ДО фазы", &meta.segs, &base, None);

    let mut hp = Hyper { lr: meta.phase_lr, b1: 0.9, b2: 0.95, eps: 1e-8, wd: a.get("wd", 0.01)?, clip: a.get("clip", 1.0)? };
    let end_step = meta.phase_start + meta.phase_steps;
    let mut rng = Rng(meta.rng);
    let mut buf: Vec<u8> = Vec::new();
    let t0 = Instant::now();
    let mut ema: Option<f64> = None;
    let mut run_tokens = 0u64;
    let mut skipped = 0u64;
    let mut bad_streak = 0u64;
    let mut interrupted = false;
    let mut last_norm = 0.0f64;

    let mut step = meta.step;
    while step < end_step {
        if util::stop_requested() {
            interrupted = true;
            break;
        }
        let local = step - meta.phase_start;
        hp.lr = lr_at(local, meta.phase_steps, meta.phase_lr, warmup, min_lr_frac);
        tr.quantize(&gpu)?;
        tr.zero_grads(&gpu)?;
        let tokens_per_step = batch * seq * accum;
        let scale = 1.0 / tokens_per_step as f32;
        for _ in 0..accum {
            let mut toks = Vec::with_capacity(batch * seq);
            let mut tg = Vec::with_capacity(batch * seq);
            for _ in 0..batch {
                mixer.sample(&mut rng, win, &mut buf)?;
                toks.extend_from_slice(&buf[..seq]);
                tg.extend_from_slice(&buf[1..win]);
            }
            tr.micro(&gpu, &toks, &tg, batch, seq, scale)?;
        }
        let (norm, nll_sum, ok) = tr.finish_step(&gpu, &hp)?;
        let loss = nll_sum / tokens_per_step as f64;
        last_norm = norm;
        if !ok || !loss.is_finite() {
            skipped += 1;
            bad_streak += 1;
            if bad_streak > 50 {
                return Err("Слишком много NaN/Inf-шагов подряд — остановка (уменьши --lr).".into());
            }
        } else {
            bad_streak = 0;
            ema = Some(match ema {
                None => loss,
                Some(e) => 0.98 * e + 0.02 * loss,
            });
        }
        step += 1;
        meta.step = step;
        meta.tokens += tokens_per_step as u64;
        run_tokens += tokens_per_step as u64;

        if log_every > 0 && (step % log_every == 0 || step == end_step) {
            let el = t0.elapsed().as_secs_f64().max(1e-9);
            let run_steps = (run_tokens / (batch * seq * accum) as u64).max(1) as f64;
            let eta = (end_step - step) as f64 * el / run_steps;
            println!(
                "  [{:>7}/{}] loss {:.4} ({:.3} бит/байт) | lr {:.2e} | |g| {:.2} | {:.0} ток/с | ETA {}{}",
                step,
                end_step,
                ema.unwrap_or(f64::NAN),
                bpb(ema.unwrap_or(f64::NAN)),
                hp.lr,
                last_norm,
                run_tokens as f64 / el,
                util::dur(eta),
                if skipped > 0 { format!(" | пропущено {}", skipped) } else { String::new() }
            );
        }
        if eval_every > 0 && step % eval_every == 0 && step < end_step {
            let r = eval_all(&mut tr, &gpu, &vals)?;
            print_eval(&format!("валидация, шаг {}", step), &meta.segs, &r, Some(&base));
        }
        if ckpt_every > 0 && step % ckpt_every == 0 && step < end_step {
            meta.rng = rng.0;
            tr.download(&gpu, &mut model, &mut adam)?;
            let p = ckpt::save(&ckpt_dir, &meta, &model, &adam, keep)?;
            println!("  ✓ чекпойнт {}", p.display());
        }
    }

    meta.rng = rng.0;
    if interrupted {
        meta.phase_done = false;
        tr.download(&gpu, &mut model, &mut adam)?;
        let p = ckpt::save(&ckpt_dir, &meta, &model, &adam, keep)?;
        println!("\n⏸ Остановлено (Ctrl+C). Чекпойнт: {}", p.display());
        println!("  Продолжить: hashnet_proto train --resume --ckpt-dir {}", ckpt_dir);
        return Ok(());
    }
    meta.phase_done = true;
    tr.download(&gpu, &mut model, &mut adam)?;
    let p = ckpt::save(&ckpt_dir, &meta, &model, &adam, keep)?;
    println!("\n✓ Фаза {} завершена. Чекпойнт: {}", meta.phase, p.display());

    let fin = eval_all(&mut tr, &gpu, &vals)?;
    print_eval("валидация ПОСЛЕ фазы (Δ — изменение за фазу; рост на старых данных = забывание)", &meta.segs, &fin, Some(&base));
    tr.quantize(&gpu)?;
    println!("  доля нулей в тернарных весах: {:.1}%", tr.zero_frac(&gpu)? * 100.0);

    // пример генерации
    let prompt: Vec<u8> = vals.iter().find(|v| !v.is_empty()).map(|v| v[0][..seq.min(32)].to_vec()).unwrap_or_else(|| b"\n".to_vec());
    let mut grng = Rng::new(7);
    let g = generate(&mut tr, &gpu, &prompt, 200, 0.8, 40, &mut grng)?;
    println!("\n--- пример генерации ---\n{}{}\n", String::from_utf8_lossy(&prompt), String::from_utf8_lossy(&g));
    Ok(())
}


// ───────────────────────────── check: численная проверка градиентов ─────────────────────────────

/// Сравнивает аналитические градиенты (ядра backward) с конечными разностями на крошечной модели в fp32-режиме.
/// Запуск: hashnet_proto check
fn cmd_check(_a: &Args) -> Res<()> {
    let gpu = gpu::GpuBitwise::new()?;
    let cfg = Config { d_model: 32, n_layers: 2, n_heads: 2, seq_len: 12 };
    cfg.validate()?;
    let mut model = Model::new(cfg.clone(), 5);
    let mut rng = Rng::new(99);
    // чтобы градиенты LN не были вырожденными — немного шума в gamma/beta
    for i in 0..cfg.n_tensors() {
        match cfg.kind(i) {
            model::Kind::Gain => model.t[i].iter_mut().for_each(|x| *x = 1.0 + 0.2 * rng.normal()),
            model::Kind::Bias => model.t[i].iter_mut().for_each(|x| *x = 0.1 * rng.normal()),
            _ => {}
        }
    }
    let adam = Adam::new(&model);
    let (b, t) = (3usize, cfg.seq_len);
    let mut tr = Trainer::new(&gpu, &model, &adam, b, false)?;
    let mut toks = Vec::new();
    let mut tg = Vec::new();
    for _ in 0..b {
        let w: Vec<u8> = (0..=t).map(|_| (rng.below(12) as u8) + 97).collect();
        toks.extend_from_slice(&w[..t]);
        tg.extend_from_slice(&w[1..]);
    }
    tr.quantize(&gpu)?;
    tr.zero_grads(&gpu)?;
    tr.micro(&gpu, &toks, &tg, b, t, 1.0 / (b * t) as f32)?;
    let grads = tr.download_grads(&gpu)?;
    let base = tr.loss_only(&gpu, &toks, &tg, b, t)?;
    println!("fp32-режим, loss = {:.5} нат (ln 256 = {:.3})", base, (256f64).ln());
    let eps = 1e-2f32;
    let (mut bad, mut tot) = (0, 0);
    for i in 0..cfg.n_tensors() {
        // 3 элемента с наибольшим |g| — у них относительная погрешность конечных разностей минимальна
        let mut idx: Vec<usize> = (0..grads[i].len()).collect();
        idx.sort_by(|&x, &y| grads[i][y].abs().partial_cmp(&grads[i][x].abs()).unwrap_or(std::cmp::Ordering::Equal));
        let mut worst = 0.0f64;
        let mut shown = 0;
        for &j in idx.iter().take(3) {
            let an = grads[i][j] as f64;
            if an.abs() < 1e-5 {
                continue;
            }
            let orig = model.t[i][j];
            model.t[i][j] = orig + eps;
            tr.set_param(&gpu, i, &model.t[i])?;
            let lp = tr.loss_only(&gpu, &toks, &tg, b, t)?;
            model.t[i][j] = orig - eps;
            tr.set_param(&gpu, i, &model.t[i])?;
            let lm = tr.loss_only(&gpu, &toks, &tg, b, t)?;
            model.t[i][j] = orig;
            tr.set_param(&gpu, i, &model.t[i])?;
            let num = (lp - lm) / (2.0 * eps as f64);
            let rel = (num - an).abs() / (num.abs().max(an.abs()) + 1e-8);
            worst = worst.max(rel);
            shown += 1;
        }
        if shown == 0 {
            println!("  {:<12} (градиент ≈ 0, пропущен)", model::tensor_name(&cfg, i));
            continue;
        }
        tot += 1;
        let ok = worst < 0.03;
        if !ok {
            bad += 1;
        }
        println!("  {:<12} макс. отн. ошибка {:.2e}  {}", model::tensor_name(&cfg, i), worst, if ok { "OK" } else { "✗ РАСХОЖДЕНИЕ" });
    }
    if bad == 0 {
        println!("\n✓ Градиенты совпадают с численными ({} тензоров).", tot);
    } else {
        println!("\n✗ Расхождение в {} из {} тензоров — backward считает неверно.", bad, tot);
    }
    Ok(())
}

// ───────────────────────────── gen / info ─────────────────────────────

/// Быстрая генерация: тернарные веса int8 + KV-кэш. При заполнении контекста (seq) окно
/// сдвигается: кэш сбрасывается и пересчитывается по последней половине контекста.
fn generate_fast(inf: &mut infer::Infer, prompt: &[u8], len: usize, temp: f32, top_k: usize, no_repeat: usize, rng: &mut Rng) -> Res<Vec<u8>> {
    let seq = inf.seq;
    let mut ctx: Vec<u8> = if prompt.is_empty() { vec![b'\n'] } else { prompt.to_vec() };
    let refill = |inf: &mut infer::Infer, ctx: &[u8]| -> Res<Vec<f32>> {
        inf.reset();
        let start = if ctx.len() >= seq { ctx.len() - seq / 2 } else { 0 };
        let mut lg = Vec::new();
        for &c in &ctx[start..] {
            lg = inf.step(c)?;
        }
        Ok(lg)
    };
    let mut logits = refill(inf, &ctx)?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        if no_repeat >= 2 && ctx.len() >= no_repeat {
            // запрещаем байт, который продолжил бы уже встречавшийся дословно кусок из no_repeat байт
            let m = no_repeat - 1;
            let key = ctx[ctx.len() - m..].to_vec();
            let mut banned = [false; VOCAB];
            let mut nb = 0;
            for i in 0..ctx.len() - m {
                if ctx[i..i + m] == key[..] {
                    let nx = ctx[i + m] as usize;
                    if !banned[nx] { banned[nx] = true; nb += 1; }
                }
            }
            if nb > 0 && nb < VOCAB {
                for (i, &bn) in banned.iter().enumerate() {
                    if bn { logits[i] = -1e30; }
                }
            }
        }
        let b = sample_next(&logits, temp, top_k, rng);
        out.push(b);
        ctx.push(b);
        logits = if inf.pos < seq { inf.step(b)? } else { refill(inf, &ctx)? };
    }
    Ok(out)
}

fn cmd_gen(a: &Args) -> Res<()> {
    let p = ckpt::find_latest(&a.s("ckpt", "ckpt"))?;
    let (meta, model, adam) = ckpt::load(&p)?;
    let gpu = gpu::GpuBitwise::new()?;
    let prompt = a.s("prompt", "");
    let len: usize = a.get("len", 200)?;
    let temp: f32 = a.get("temp", 0.8)?;
    let top_k: usize = a.get("top-k", 40)?;
    let seed: u64 = a.get("seed", 42)?;
    println!("(чекпойнт {}, шаг {})", p.display(), meta.step);
    let mut rng = Rng::new(seed);

    if a.flag("no-quant") || a.flag("slow") {
        // медленный путь: полное окно через код обучения (нужен для fp32-моделей)
        let mut tr = Trainer::new(&gpu, &model, &adam, 1, !a.flag("no-quant"))?;
        let out = generate(&mut tr, &gpu, prompt.as_bytes(), len, temp, top_k, &mut rng)?;
        println!("{}{}", prompt, String::from_utf8_lossy(&out));
        return Ok(());
    }

    let mut inf = infer::Infer::new(&gpu, &model)?;
    println!("(инференс: тернарные веса int8, доля нулей {:.1}%, KV-кэш)", inf.zero_frac * 100.0);

    if a.flag("verify") {
        // сверка с кодом обучения: те же логиты, что считает fake-quant forward?
        let text = if prompt.is_empty() { "fn main() {\n    let x = 5;\n    println!(\"{}\", x);\n}\n// Привет, мир\n".to_string() } else { prompt.clone() };
        let bytes: Vec<u8> = text.bytes().take(model.cfg.seq_len).collect();
        let mut tr = Trainer::new(&gpu, &model, &adam, 1, true)?;
        tr.quantize(&gpu)?;
        let (mut worst, mut agree) = (0.0f32, 0usize);
        for i in 0..bytes.len() {
            let fast = inf.step(bytes[i])?;
            let slow = tr.logits_last(&gpu, &bytes[..=i])?;
            let d = fast.iter().zip(slow.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            worst = worst.max(d);
            let am = |v: &[f32]| v.iter().enumerate().fold(0, |b, (j, &x)| if x > v[b] { j } else { b });
            if am(&fast) == am(&slow) {
                agree += 1;
            }
        }
        println!("сверка на {} позициях: макс. |Δ логита| = {:.2e}, argmax совпал в {}/{}", bytes.len(), worst, agree, bytes.len());
        return Ok(());
    }

    let t0 = Instant::now();
    let no_repeat: usize = a.get("no-repeat", 0)?;
    let out = generate_fast(&mut inf, prompt.as_bytes(), len, temp, top_k, no_repeat, &mut rng)?;
    let el = t0.elapsed().as_secs_f64();
    println!("{}{}", prompt, String::from_utf8_lossy(&out));
    eprintln!("\n[{} байт за {:.2}с = {:.0} байт/с]", out.len(), el, out.len() as f64 / el.max(1e-9));
    Ok(())
}

fn cmd_info(a: &Args) -> Res<()> {
    let p = ckpt::find_latest(&a.s("ckpt", "ckpt"))?;
    let (meta, _model, adam) = ckpt::load(&p)?;
    println!("Чекпойнт: {}", p.display());
    println!("  шаг {} | фаза {} ({}) | токенов {} | Adam-шагов {}", meta.step, meta.phase, if meta.phase_done { "завершена" } else { "ПРЕРВАНА" }, util::count(meta.tokens), adam.step);
    println!("  d={} слоёв={} голов={} seq={}", meta.cfg.d_model, meta.cfg.n_layers, meta.cfg.n_heads, meta.cfg.seq_len);
    print_model_info(&meta.cfg);
    println!("  файлы:");
    for f in &meta.files {
        println!("    {}  ({} байт при последнем обучении)", f.path, f.len);
    }
    println!("  сегменты данных:");
    for s in &meta.segs {
        println!("    {}  train {} байт, val {} байт, вес {:.3}", s.label(), s.train_len(), s.val_len(), s.weight);
    }
    Ok(())
}

fn help() {
    println!(
        "HashNet4D — байтовая BitNet-модель с дообучением

КОМАНДЫ
  train  — обучение / дообучение
  gen    — генерация из чекпойнта
  info   — сведения о чекпойнте
  check  — проверка градиентов на крошечной модели (fp32, конечные разности)

ПЕРВОЕ ОБУЧЕНИЕ (микро-модель)
  hashnet_proto train --data data_full.txt --steps 5000

ДООБУЧЕНИЕ (новые данные + часть старых)
  hashnet_proto train --resume --data data_full.txt --data new.txt --steps 2000 --replay-frac 0.3
    · дописанные в конец файла байты распознаются как новые;
    · остальное идёт как «старые данные» (повтор), доля задаётся --replay-frac;
    · валидация по каждому сегменту показывает, нет ли забывания.
  hashnet_proto train --resume --steps 1000        (без --data: продолжить на прежних данных)
  Ctrl+C сохраняет чекпойнт; повторный --resume доводит фазу до конца.

ПАРАМЕТРЫ train
  --data ФАЙЛ[@доля] (повторяемый; доли относительные, напр. rust.txt@4 dialogs.txt@2)   --steps N  шагов в фазе (5000)
  --d-model 128 --layers 4 --heads 4 --seq 64  (только для новой модели)
  --batch 16         последовательностей на шаг   --accum 1   накопление градиента
  --lr 1e-3  --warmup 200  --min-lr-frac 0.1  --wd 0.01  --clip 1.0
  --replay-frac 0.3  доля старых данных при дообучении
  --ckpt-dir ckpt    --ckpt-every 1000  --keep 3   --eval-every 500  --log-every 50
  --new-phase        начать новую фазу, даже если прошлая была прервана
  --seed N
  --no-quant         fp32 без квантизации (контрольный эксперимент)

СЛИЯНИЕ МОДЕЛЕЙ (на процессоре, видеокарта не нужна)
  hashnet_proto merge --method avg|slerp|task|ties|dare|stack|grow --model A --model B [--model C ...] [--base B0] --out ПАПКА
  hashnet_proto merge --method grow --model A --add-layers N [--place end|spread] [--init ident|copy] [--eps 0] --out ПАПКА   (рост в глубину одной модели)
    avg    среднее (--alpha 0.5,0.5)      slerp  сфера между двумя (--t 0.5)
    task   base+Σα(W−base) (--alpha)      ties   (--density 0.2 --lambda 1.0)     dare  (--drop 0.3 --alpha)
    stack  слои A, затем слои B: размер растёт (--take 0-6,0-6  --globals first|last|avg)
  после stack модель нужно «подлечить» дообучением: train --resume --ckpt-dir ПАПКА --data ... --new-phase

ПАРАМЕТРЫ gen:  --ckpt ckpt --prompt \"текст\" --len 200 --temp 0.8 --top-k 40\n  --no-repeat N  не повторять дословно куски из N байт (напр. 24; 0 = выкл)   --verify  сверить быстрый инференс с кодом обучения   --slow  медленный путь (окно целиком)"
    );
}

// ───────────────────────────── merge ─────────────────────────────
fn cmd_merge(a: &Args) -> Res<()> {
    let parse_list = |k: &str| -> Res<Vec<f32>> {
        let s = a.s(k, "");
        if s.is_empty() { return Ok(vec![]); }
        s.split(',').map(|x| x.trim().parse::<f32>().map_err(|_| format!("--{}: не число '{}'", k, x).into())).collect()
    };
    let mut take = Vec::new();
    let ts = a.s("take", "");
    if !ts.is_empty() {
        for part in ts.split(',') {
            let part = part.trim();
            let (s0, e0) = match part.split_once('-') {
                Some((x, y)) => (x.parse::<usize>(), y.parse::<usize>()),
                None => (Ok(0usize), part.parse::<usize>()),
            };
            take.push((s0.map_err(|_| format!("--take: не понял '{}'", part))?, e0.map_err(|_| format!("--take: не понял '{}'", part))?));
        }
    }
    let o = merge::Opts {
        method: a.s("method", "avg"),
        models: a.all("model"),
        base: if a.has("base") { Some(a.s("base", "")) } else { None },
        alphas: parse_list("alpha")?,
        density: a.get("density", 0.2f32)?,
        lambda: a.get("lambda", 1.0f32)?,
        drop: a.get("drop", 0.3f32)?,
        t: a.get("t", 0.5f32)?,
        take,
        globals: a.s("globals", "first"),
        out: a.s("out", "ckpt_merged"),
        seed: a.get("seed", 1234u64)?,
        add_layers: a.get("add-layers", 0usize)?,
        place: a.s("place", "end"),
        init: a.s("init", "ident"),
        eps: a.get("eps", 0.0f32)?,
    };
    merge::run(&o)
}

fn main() {
    let a = Args::parse();
    let r = match a.cmd.as_str() {
        "train" => cmd_train(&a),
        "gen" => cmd_gen(&a),
        "info" => cmd_info(&a),
        "check" => cmd_check(&a),
        "merge" => cmd_merge(&a),
        _ => {
            help();
            Ok(())
        }
    };
    if let Err(e) = r {
        eprintln!("❌ {}", e);
        std::process::exit(1);
    }
}
