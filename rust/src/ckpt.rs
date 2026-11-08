//! Чекпойнты: веса + состояние Adam + манифест данных + позиция в фазе.
//! Запись атомарная (tmp → fsync → rename), хранится последние `keep` файлов.
use crate::data::{FileRec, Seg};
use crate::model::{Adam, Config, Model, Res};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"HN4DCKP1";

#[derive(Clone, Debug)]
pub struct Meta {
    pub cfg: Config,
    pub step: u64,
    pub phase: u32,
    pub rng: u64,
    pub tokens: u64,
    pub phase_done: bool,
    pub phase_start: u64,
    pub phase_steps: u64,
    pub phase_lr: f32,
    pub files: Vec<FileRec>,
    pub segs: Vec<Seg>,
}

fn wu64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn wu32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn wf64<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn wstr<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    wu64(w, s.len() as u64)?;
    w.write_all(s.as_bytes())
}
fn wf32s<W: Write>(w: &mut W, x: &[f32]) -> io::Result<()> {
    // little-endian (x86/ARM) — совпадает с форматом файла
    let bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
    w.write_all(bytes)
}

fn ru64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn ru32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn rf64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
fn rstr<R: Read>(r: &mut R) -> io::Result<String> {
    let n = ru64(r)? as usize;
    if n > (1 << 20) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "строка слишком длинная"));
    }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "не utf8"))
}
fn rf32s<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut v = vec![0.0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
    r.read_exact(bytes)?;
    Ok(v)
}

pub fn list(dir: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("ckpt_") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

/// путь к файлу ИЛИ к каталогу (берётся самый свежий чекпойнт)
pub fn find_latest(path: &str) -> Res<PathBuf> {
    let p = Path::new(path);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    list(path).pop().ok_or_else(|| format!("в {} нет чекпойнтов", path).into())
}

pub fn save(dir: &str, meta: &Meta, model: &Model, adam: &Adam, keep: usize) -> Res<PathBuf> {
    fs::create_dir_all(dir)?;
    let final_p = Path::new(dir).join(format!("ckpt_{:010}.bin", meta.step));
    let tmp = Path::new(dir).join(format!("ckpt_{:010}.bin.tmp", meta.step));
    {
        let f = File::create(&tmp)?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        w.write_all(MAGIC)?;
        wu32(&mut w, 1)?;
        for v in [meta.cfg.d_model, meta.cfg.n_layers, meta.cfg.n_heads, meta.cfg.seq_len] {
            wu64(&mut w, v as u64)?;
        }
        wu64(&mut w, meta.step)?;
        wu32(&mut w, meta.phase)?;
        wu64(&mut w, meta.rng)?;
        wu64(&mut w, meta.tokens)?;
        w.write_all(&[meta.phase_done as u8])?;
        wu64(&mut w, meta.phase_start)?;
        wu64(&mut w, meta.phase_steps)?;
        w.write_all(&meta.phase_lr.to_le_bytes())?;
        wu64(&mut w, meta.files.len() as u64)?;
        for f in &meta.files {
            wstr(&mut w, &f.path)?;
            wu64(&mut w, f.len)?;
        }
        wu64(&mut w, meta.segs.len() as u64)?;
        for s in &meta.segs {
            wstr(&mut w, &s.path)?;
            wu64(&mut w, s.start)?;
            wu64(&mut w, s.end)?;
            wu64(&mut w, s.val_start)?;
            wu64(&mut w, s.val_end)?;
            wu32(&mut w, s.phase)?;
            wf64(&mut w, s.weight)?;
        }
        wu64(&mut w, model.t.len() as u64)?;
        for t in &model.t {
            wu64(&mut w, t.len() as u64)?;
            wf32s(&mut w, t)?;
        }
        wu64(&mut w, adam.step)?;
        for t in &adam.m {
            wf32s(&mut w, t)?;
        }
        for t in &adam.v {
            wf32s(&mut w, t)?;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    if final_p.exists() {
        fs::remove_file(&final_p)?;
    }
    fs::rename(&tmp, &final_p)?;
    // ротация
    let all = list(dir);
    if keep > 0 && all.len() > keep {
        for p in &all[..all.len() - keep] {
            let _ = fs::remove_file(p);
        }
    }
    Ok(final_p)
}

pub fn load(path: &Path) -> Res<(Meta, Model, Adam)> {
    let f = File::open(path).map_err(|e| format!("не открыть {}: {}", path.display(), e))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err("это не чекпойнт HashNet4D (неверная сигнатура)".into());
    }
    let ver = ru32(&mut r)?;
    if ver != 1 {
        return Err(format!("неподдерживаемая версия чекпойнта {}", ver).into());
    }
    let cfg = Config {
        d_model: ru64(&mut r)? as usize,
        n_layers: ru64(&mut r)? as usize,
        n_heads: ru64(&mut r)? as usize,
        seq_len: ru64(&mut r)? as usize,
    };
    cfg.validate()?;
    let step = ru64(&mut r)?;
    let phase = ru32(&mut r)?;
    let rng = ru64(&mut r)?;
    let tokens = ru64(&mut r)?;
    let mut b1 = [0u8; 1];
    r.read_exact(&mut b1)?;
    let phase_done = b1[0] != 0;
    let phase_start = ru64(&mut r)?;
    let phase_steps = ru64(&mut r)?;
    let mut lrb = [0u8; 4];
    r.read_exact(&mut lrb)?;
    let phase_lr = f32::from_le_bytes(lrb);
    let nf = ru64(&mut r)? as usize;
    let mut files = Vec::new();
    for _ in 0..nf {
        files.push(FileRec { path: rstr(&mut r)?, len: ru64(&mut r)? });
    }
    let ns = ru64(&mut r)? as usize;
    let mut segs = Vec::new();
    for _ in 0..ns {
        segs.push(Seg {
            path: rstr(&mut r)?,
            start: ru64(&mut r)?,
            end: ru64(&mut r)?,
            val_start: ru64(&mut r)?,
            val_end: ru64(&mut r)?,
            phase: ru32(&mut r)?,
            weight: rf64(&mut r)?,
        });
    }
    let nt = ru64(&mut r)? as usize;
    if nt != cfg.n_tensors() {
        return Err("число тензоров не совпадает с конфигурацией".into());
    }
    let mut t = Vec::with_capacity(nt);
    for i in 0..nt {
        let len = ru64(&mut r)? as usize;
        let (rr, cc) = cfg.shape(i);
        if len != rr * cc {
            return Err(format!("тензор {}: размер {} != ожидаемого {}", i, len, rr * cc).into());
        }
        t.push(rf32s(&mut r, len)?);
    }
    let astep = ru64(&mut r)?;
    let mut m = Vec::with_capacity(nt);
    for i in 0..nt {
        m.push(rf32s(&mut r, t[i].len())?);
    }
    let mut v = Vec::with_capacity(nt);
    for i in 0..nt {
        v.push(rf32s(&mut r, t[i].len())?);
    }
    let meta = Meta { cfg: cfg.clone(), step, phase, rng, tokens, phase_done, phase_start, phase_steps, phase_lr, files, segs };
    Ok((meta, Model { cfg, t }, Adam { m, v, step: astep }))
}
