//! Потоковые данные: файлы НЕ загружаются целиком — окна читаются через pread.
//! Данные описываются «сегментами» (файл + диапазон байт). Это основа дообучения:
//!  * новый файл            → новый сегмент [0, размер);
//!  * дописанный в конец    → новый сегмент [старая_длина, новый_размер);
//!  * всё, что обучалось ранее, остаётся «старыми» сегментами для повтора (replay).
use crate::model::Res;
use crate::util::Rng;
use std::fs::{self, File};
use std::io;

#[derive(Clone, Debug)]
pub struct Seg {
    pub path: String,
    pub start: u64,
    pub end: u64,
    pub val_start: u64,
    pub val_end: u64,
    pub phase: u32,
    pub weight: f64,
}

impl Seg {
    pub fn train_len(&self) -> u64 {
        self.end - self.start
    }
    pub fn val_len(&self) -> u64 {
        self.val_end - self.val_start
    }
    pub fn label(&self) -> String {
        let name = std::path::Path::new(&self.path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| self.path.clone());
        format!("ф{}:{}[{}..{}]", self.phase, name, self.start, self.end)
    }
}

#[derive(Clone, Debug)]
pub struct FileRec {
    pub path: String,
    pub len: u64,
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let n = f.seek_read(&mut buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
        }
        done += n;
    }
    Ok(())
}

/// Взвешенный смеситель сегментов
pub struct Mixer {
    files: Vec<File>,
    segs: Vec<Seg>,
    cum: Vec<f64>,
}

impl Mixer {
    pub fn new(segs: &[Seg], win: usize) -> Res<Self> {
        let mut files = Vec::new();
        let mut cum = Vec::new();
        let mut acc = 0.0f64;
        for s in segs {
            if s.train_len() < win as u64 {
                return Err(format!("сегмент {} короче окна ({} байт)", s.label(), win).into());
            }
            files.push(File::open(&s.path).map_err(|e| format!("не открыть {}: {}", s.path, e))?);
            acc += s.weight.max(0.0);
            cum.push(acc);
        }
        if acc <= 0.0 {
            return Err("нет данных для обучения (сумма весов = 0)".into());
        }
        Ok(Mixer { files, segs: segs.to_vec(), cum })
    }

    /// Читает случайное окно из `win` байт. Возвращает индекс сегмента.
    pub fn sample(&self, rng: &mut Rng, win: usize, out: &mut Vec<u8>) -> io::Result<usize> {
        let total = *self.cum.last().unwrap();
        let r = rng.f64() * total;
        let mut idx = self.cum.partition_point(|&c| c <= r);
        if idx >= self.segs.len() {
            idx = self.segs.len() - 1;
        }
        let s = &self.segs[idx];
        let span = s.train_len() - win as u64 + 1;
        let off = s.start + rng.below(span);
        out.resize(win, 0);
        read_at(&self.files[idx], out, off)?;
        Ok(idx)
    }
}

/// Детерминированные окна валидации (равномерно по отложенной части сегмента)
pub fn val_windows(seg: &Seg, win: usize, max_n: usize) -> Res<Vec<Vec<u8>>> {
    let vl = seg.val_len();
    if vl < win as u64 {
        return Ok(Vec::new());
    }
    let n = ((vl / win as u64) as usize).min(max_n).max(1);
    let stride = vl / n as u64;
    let f = File::open(&seg.path)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut buf = vec![0u8; win];
        read_at(&f, &mut buf, seg.val_start + i as u64 * stride)?;
        out.push(buf);
    }
    Ok(out)
}

pub struct PlanOpts {
    pub win: usize,
    pub replay_frac: f64,
    pub val_cap: u64,
    /// относительные доли новых источников (параллельно `paths`); None — не задано
    pub weights: Vec<Option<f64>>,
}

pub struct PlanReport {
    pub lines: Vec<String>,
}

fn canon(p: &str) -> String {
    fs::canonicalize(p).map(|x| x.to_string_lossy().to_string()).unwrap_or_else(|_| p.to_string())
}

/// Планирование новой фазы обучения.
/// Возвращает (обновлённый список файлов, сегменты с весами, отчёт).
pub fn plan(
    paths: &[String],
    prev_files: &[FileRec],
    prev_segs: &[Seg],
    phase: u32,
    o: &PlanOpts,
) -> Res<(Vec<FileRec>, Vec<Seg>, PlanReport)> {
    let mut rep = PlanReport { lines: Vec::new() };
    let mut files: Vec<FileRec> = prev_files.to_vec();

    // старые сегменты: оставляем только те, чей файл ещё существует
    let mut old: Vec<Seg> = Vec::new();
    for s in prev_segs {
        match fs::metadata(&s.path) {
            Ok(m) if m.len() >= s.end => old.push(s.clone()),
            Ok(_) => rep.lines.push(format!("⚠ файл {} стал короче, чем было при обучении — сегмент {} пропущен", s.path, s.label())),
            Err(_) => rep.lines.push(format!("⚠ файл {} не найден — сегмент {} пропущен", s.path, s.label())),
        }
    }

    let mut new: Vec<Seg> = Vec::new();
    let mut hints: Vec<Option<f64>> = Vec::new();
    for (pi, p) in paths.iter().enumerate() {
        let cp = canon(p);
        let size = fs::metadata(&cp).map_err(|e| format!("не найден файл {}: {}", p, e))?.len();
        let known = files.iter().position(|f| f.path == cp);
        let from = match known {
            Some(i) => {
                let l = files[i].len;
                if size < l {
                    rep.lines.push(format!("⚠ {} стал короче ({} < {}) — новых данных нет", p, size, l));
                    size
                } else {
                    l
                }
            }
            None => 0,
        };
        if size > from {
            let len = size - from;
            let mut val = (len / 50).min(o.val_cap);
            if val < 2 * o.win as u64 {
                val = 0;
            }
            let train_end = size - val;
            if train_end - from < o.win as u64 {
                rep.lines.push(format!("⚠ {}: новых данных слишком мало ({} байт) — пропущено", p, len));
            } else {
                rep.lines.push(match known {
                    Some(_) => format!("＋ {}: дописано {} байт (с позиции {}) → новые данные", p, len, from),
                    None => format!("＋ {}: новый файл, {} байт", p, len),
                });
                new.push(Seg { path: cp.clone(), start: from, end: train_end, val_start: train_end, val_end: size, phase, weight: 0.0 });
                hints.push(o.weights.get(pi).copied().flatten());
            }
        } else {
            rep.lines.push(format!("＝ {}: без изменений ({} байт) — остаётся как старые данные", p, size));
        }
        match known {
            Some(i) => files[i].len = size,
            None => files.push(FileRec { path: cp, len: size }),
        }
    }

    let old_len: u64 = old.iter().map(|s| s.train_len()).sum();
    let new_len: u64 = new.iter().map(|s| s.train_len()).sum();
    if old_len + new_len == 0 {
        return Err("нет данных: ни старых, ни новых сегментов".into());
    }
    let (old_w, new_w) = if old_len == 0 {
        (0.0, 1.0)
    } else if new_len == 0 {
        (1.0, 0.0)
    } else {
        (o.replay_frac.clamp(0.0, 0.95), 1.0 - o.replay_frac.clamp(0.0, 0.95))
    };
    for s in old.iter_mut() {
        s.weight = old_w * s.train_len() as f64 / old_len.max(1) as f64;
    }
    if hints.iter().any(|h| h.is_some()) {
        // заданы доли: незаданные считаются 1.0; доли относительные
        let ws: Vec<f64> = hints.iter().map(|h| h.unwrap_or(1.0).max(0.0)).collect();
        let tot = ws.iter().sum::<f64>().max(1e-12);
        for (s, w) in new.iter_mut().zip(ws.iter()) {
            s.weight = new_w * w / tot;
        }
    } else {
        for s in new.iter_mut() {
            s.weight = new_w * s.train_len() as f64 / new_len.max(1) as f64;
        }
    }
    rep.lines.push(format!(
        "Смесь: старые {:.0}% ({} байт, {} сегм.) / новые {:.0}% ({} байт, {} сегм.)",
        old_w * 100.0, old_len, old.len(), new_w * 100.0, new_len, new.len()
    ));
    for s in &new {
        rep.lines.push(format!("  доля в смеси {:>5.1}%  {}", s.weight * 100.0, s.label()));
    }
    let mut segs = old;
    segs.extend(new);
    Ok((files, segs, rep))
}
