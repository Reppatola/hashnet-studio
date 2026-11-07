//! scan — разведка перед подготовкой датасета: сколько на дисках реального Rust-кода,
//! сколько из него дубликаты, сколько функций и какого они размера (чтобы выбрать seq),
//! сколько функций с doc-комментариями (готовые пары «описание → код») и тестов.
//! Только std, без внешних крейтов.
//!
//! Запуск:   cargo run --release --bin scan
//!           cargo run --release --bin scan -- --paths /путь/1 /путь/2 --out /куда/писать
//! Без --paths сам находит все смонтированные диски (Linux: /proc/mounts, Windows: буквы дисков).
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_ANALYZE: u64 = 4 << 20; // файлы больше 4 МБ считаем сгенерированными: учитываем объём, но не режем на функции
const BUCKETS: [usize; 7] = [256, 512, 1024, 2048, 4096, 8192, usize::MAX];
const CODE_EXT: [&str; 18] = ["py", "js", "ts", "tsx", "jsx", "c", "h", "cpp", "hpp", "cc", "go", "java", "kt", "cs", "rb", "php", "sh", "lua"];

// ───────────── хеши (два независимых 64-бит → 128 бит) ─────────────
fn hash128(b: &[u8]) -> u128 {
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x9E3779B97F4A7C15;
    for &x in b {
        h1 = (h1 ^ x as u64).wrapping_mul(0x100000001b3);
        h2 = (h2.rotate_left(5) ^ x as u64).wrapping_mul(0xff51afd7ed558ccd);
    }
    ((h1 as u128) << 64) | h2 as u128
}

struct Shards(Vec<Mutex<HashSet<u128>>>);
impl Shards {
    fn new() -> Self {
        Shards((0..64).map(|_| Mutex::new(HashSet::new())).collect())
    }
    /// true, если элемент новый
    fn insert(&self, h: u128) -> bool {
        self.0[(h as usize) & 63].lock().unwrap().insert(h)
    }
}

// ───────────── статистика ─────────────
#[derive(Default, Clone)]
struct Stats {
    files: u64,
    bytes: u64,
    uniq_files: u64,
    uniq_bytes: u64,
    skipped_big: u64,
    fns: u64,
    uniq_fns: u64,
    uniq_fn_bytes: u64,
    hist: [u64; 7],
    hist_bytes: [u64; 7],
    fn_doc: u64,
    fn_doc_fits: [u64; 7],
    fn_test: u64,
    fn_unsafe: u64,
    gen_files: u64,
    gen_bytes: u64,
    /// крейт → (файлов, байт, функций, unsafe-функций, файлов с пометкой генерации) — только уникальное
    krates: HashMap<String, [u64; 5]>,
}
impl Stats {
    fn merge(&mut self, o: &Stats) {
        self.files += o.files;
        self.bytes += o.bytes;
        self.uniq_files += o.uniq_files;
        self.uniq_bytes += o.uniq_bytes;
        self.skipped_big += o.skipped_big;
        self.fns += o.fns;
        self.uniq_fns += o.uniq_fns;
        self.uniq_fn_bytes += o.uniq_fn_bytes;
        self.fn_doc += o.fn_doc;
        self.fn_test += o.fn_test;
        self.fn_unsafe += o.fn_unsafe;
        self.gen_files += o.gen_files;
        self.gen_bytes += o.gen_bytes;
        for (k, v) in &o.krates {
            let e = self.krates.entry(k.clone()).or_insert([0; 5]);
            for i in 0..5 {
                e[i] += v[i];
            }
        }
        for i in 0..7 {
            self.hist[i] += o.hist[i];
            self.hist_bytes[i] += o.hist_bytes[i];
            self.fn_doc_fits[i] += o.fn_doc_fits[i];
        }
    }
}

// ───────────── лёгкий лексер Rust: комментарии / строки / символы ─────────────
fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80
}

/// Если в позиции i начинается комментарий, строка или символьный литерал — возвращает позицию после него.
fn skip_lit(b: &[u8], i: usize) -> Option<usize> {
    let n = b.len();
    let c = b[i];
    if c == b'/' && i + 1 < n {
        if b[i + 1] == b'/' {
            let mut j = i + 2;
            while j < n && b[j] != b'\n' {
                j += 1;
            }
            return Some(j);
        }
        if b[i + 1] == b'*' {
            let mut depth = 1;
            let mut j = i + 2;
            while j < n && depth > 0 {
                if b[j] == b'/' && j + 1 < n && b[j + 1] == b'*' {
                    depth += 1;
                    j += 2;
                } else if b[j] == b'*' && j + 1 < n && b[j + 1] == b'/' {
                    depth -= 1;
                    j += 2;
                } else {
                    j += 1;
                }
            }
            return Some(j);
        }
        return None;
    }
    if c == b'"' {
        let mut j = i + 1;
        while j < n {
            if b[j] == b'\\' {
                j += 2;
            } else if b[j] == b'"' {
                return Some(j + 1);
            } else {
                j += 1;
            }
        }
        return Some(n);
    }
    // сырые строки r"..", r#".."#, br".."
    if (c == b'r' || (c == b'b' && i + 1 < n && b[i + 1] == b'r')) && (i == 0 || !is_ident(b[i - 1])) {
        let mut j = if c == b'b' { i + 2 } else { i + 1 };
        let mut hashes = 0;
        while j < n && b[j] == b'#' {
            hashes += 1;
            j += 1;
        }
        if j < n && b[j] == b'"' {
            j += 1;
            while j < n {
                if b[j] == b'"' {
                    let mut k = 0;
                    while k < hashes && j + 1 + k < n && b[j + 1 + k] == b'#' {
                        k += 1;
                    }
                    if k == hashes {
                        return Some(j + 1 + hashes);
                    }
                }
                j += 1;
            }
            return Some(n);
        }
        return None;
    }
    if c == b'\'' {
        // символ или lifetime
        if i + 1 < n && b[i + 1] == b'\\' {
            let mut j = i + 2;
            while j < n && b[j] != b'\'' && b[j] != b'\n' {
                j += 1;
            }
            return Some((j + 1).min(n));
        }
        // 'x' (в т.ч. многобайтный UTF-8 символ)
        let mut j = i + 1;
        if j < n && b[j] >= 0x80 {
            while j < n && b[j] >= 0x80 && j < i + 6 {
                j += 1;
            }
        } else {
            j += 1;
        }
        if j < n && b[j] == b'\'' {
            return Some(j + 1);
        }
        return Some(i + 1); // lifetime
    }
    None
}

struct FnInfo {
    start: usize,
    end: usize,
    doc: bool,
    test: bool,
    is_unsafe: bool,
}

/// Находит функции с телом (методы в impl тоже; вложенные внутрь тела не заходим).
fn find_fns(b: &[u8]) -> Vec<FnInfo> {
    let n = b.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if let Some(j) = skip_lit(b, i) {
            i = j.max(i + 1);
            continue;
        }
        if b[i] == b'f' && i + 2 < n && b[i + 1] == b'n' && (b[i + 2] == b' ' || b[i + 2] == b'\t') && (i == 0 || !is_ident(b[i - 1])) {
            // сигнатура до '{' или ';' на нулевой глубине скобок
            let mut j = i + 2;
            let mut depth = 0i32;
            let mut body = None;
            while j < n {
                if let Some(k) = skip_lit(b, j) {
                    j = k.max(j + 1);
                    continue;
                }
                match b[j] {
                    b'(' | b'[' => depth += 1,
                    b')' | b']' => depth -= 1,
                    b';' if depth <= 0 => break,
                    b'{' if depth <= 0 => {
                        body = Some(j);
                        break;
                    }
                    _ => {}
                }
                j += 1;
            }
            if let Some(open) = body {
                let mut d = 1;
                let mut k = open + 1;
                while k < n && d > 0 {
                    if let Some(m) = skip_lit(b, k) {
                        k = m.max(k + 1);
                        continue;
                    }
                    match b[k] {
                        b'{' => d += 1,
                        b'}' => d -= 1,
                        _ => {}
                    }
                    k += 1;
                }
                if d == 0 {
                    // начало: строка с `fn` + идущие выше атрибуты и doc-комментарии
                    let mut s = i;
                    while s > 0 && b[s - 1] != b'\n' {
                        s -= 1;
                    }
                    let line_txt = &b[s..i];
                    let is_unsafe = find_sub(line_txt, b"unsafe");
                    let (mut doc, mut test) = (false, false);
                    loop {
                        if s == 0 {
                            break;
                        }
                        let pe = s - 1; // '\n' предыдущей строки
                        let mut ps = pe;
                        while ps > 0 && b[ps - 1] != b'\n' {
                            ps -= 1;
                        }
                        let t = trim(&b[ps..pe]);
                        if t.starts_with(b"///") || t.starts_with(b"/**") {
                            doc = true;
                        } else if t.starts_with(b"#[") {
                            if find_sub(t, b"test") {
                                test = true;
                            }
                        } else {
                            break;
                        }
                        s = ps;
                    }
                    out.push(FnInfo { start: s, end: k, doc, test, is_unsafe });
                    i = k;
                    continue;
                }
            }
            i = j.max(i + 1);
            continue;
        }
        i += 1;
    }
    out
}

fn find_sub(h: &[u8], n: &[u8]) -> bool {
    h.windows(n.len()).any(|w| w == n)
}
fn trim(b: &[u8]) -> &[u8] {
    let mut s = 0;
    while s < b.len() && (b[s] == b' ' || b[s] == b'\t' || b[s] == b'\r') {
        s += 1;
    }
    &b[s..]
}

fn bucket(sz: usize) -> usize {
    BUCKETS.iter().position(|&m| sz <= m).unwrap_or(6)
}

// ───────────── диски ─────────────
fn unescape_mount(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' && i + 3 < b.len() + 0 && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c)) {
            let v = (b[i + 1] - b'0') * 64 + (b[i + 2] - b'0') * 8 + (b[i + 3] - b'0');
            out.push(v);
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

#[cfg(unix)]
fn detect_roots() -> (Vec<PathBuf>, Vec<PathBuf>) {
    let ok_fs = ["ext2", "ext3", "ext4", "btrfs", "xfs", "ntfs", "ntfs3", "fuseblk", "vfat", "exfat", "f2fs", "zfs", "jfs", "reiserfs"];
    let bad_prefix = ["/boot", "/snap", "/var/snap", "/var/lib/docker", "/run", "/sys", "/proc", "/dev"];
    let mut roots = Vec::new();
    let txt = fs::read_to_string("/proc/mounts").unwrap_or_default();
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        let (_dev, mp, fsty) = (it.next(), it.next(), it.next());
        if let (Some(mp), Some(fsty)) = (mp, fsty) {
            let mp = unescape_mount(mp);
            let run_media = mp == "/run/media" || mp.starts_with("/run/media/");
            let bad = !run_media && bad_prefix.iter().any(|p| mp == *p || mp.starts_with(&format!("{}/", p)));
            if ok_fs.contains(&fsty) && !bad {
                roots.push(PathBuf::from(mp));
            }
        }
    }
    roots.sort();
    roots.dedup();
    let all = roots.clone();
    // что осталось за бортом — чтобы не гадать, почему диск не попал в скан
    let pseudo = ["proc", "sysfs", "tmpfs", "devtmpfs", "devpts", "cgroup", "cgroup2", "securityfs", "pstore", "bpf", "debugfs", "tracefs", "configfs", "fusectl", "mqueue", "hugetlbfs", "autofs", "binfmt_misc", "efivarfs", "overlay", "squashfs", "ramfs", "fuse.gvfsd-fuse", "fuse.portal", "nsfs"];
    let mut shown = false;
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        let (dev, mp, fsty) = (it.next(), it.next(), it.next());
        if let (Some(dev), Some(mp), Some(fsty)) = (dev, mp, fsty) {
            let mp = unescape_mount(mp);
            if pseudo.contains(&fsty) || roots.iter().any(|r| r.to_string_lossy() == mp) || !dev.starts_with("/dev/") {
                continue;
            }
            if !shown {
                println!("Смонтировано, но НЕ сканируется:");
                shown = true;
            }
            println!("  ✗ {} ({}) на {} — тип ФС не в списке или служебный путь", mp, fsty, dev);
        }
    }
    if shown {
        println!("  (добавь нужный путь вручную: --paths <путь>)\n");
    }
    (roots, all)
}

#[cfg(windows)]
fn detect_roots() -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut roots = Vec::new();
    for c in b'A'..=b'Z' {
        let p = PathBuf::from(format!("{}:\\", c as char));
        if p.exists() {
            roots.push(p);
        }
    }
    let all = roots.clone();
    (roots, all)
}

#[cfg(not(any(unix, windows)))]
fn detect_roots() -> (Vec<PathBuf>, Vec<PathBuf>) {
    (vec![PathBuf::from(".")], vec![PathBuf::from(".")])
}

// ───────────── обход ─────────────
struct Walk {
    other_mounts: HashSet<PathBuf>,
    ext_bytes: BTreeMap<String, (u64, u64)>,
    repos: u64,
    projects: u64,
    dirs: u64,
    other_files: u64,
    errors: u64,
    skipped_target: u64,
}

fn walk(root: &Path, w: &mut Walk, tx: &mpsc::Sender<PathBuf>, prog: &Progress) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        w.dirs += 1;
        let rd = match fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => {
                w.errors += 1;
                continue;
            }
        };
        let entries: Vec<_> = rd.filter_map(|e| e.ok()).collect();
        let has_cargo = entries.iter().any(|e| e.file_name() == "Cargo.toml");
        if has_cargo {
            w.projects += 1;
        }
        for e in entries {
            let ft = match e.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            let path = e.path();
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                let name = e.file_name();
                let name = name.to_string_lossy();
                if name == ".git" {
                    w.repos += 1;
                    continue;
                }
                if name == "node_modules" || (name == "target" && has_cargo) {
                    w.skipped_target += 1;
                    continue;
                }
                if path != root && w.other_mounts.contains(&path) {
                    continue; // другой диск — сканируется отдельно
                }
                stack.push(path);
            } else if ft.is_file() {
                prog.seen.fetch_add(1, Ordering::Relaxed);
                let ext = path.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
                if ext == "rs" {
                    let _ = tx.send(path);
                } else if CODE_EXT.contains(&ext.as_str()) || ext == "md" || ext == "toml" {
                    if let Ok(m) = e.metadata() {
                        let ent = w.ext_bytes.entry(ext).or_insert((0, 0));
                        ent.0 += 1;
                        ent.1 += m.len();
                    }
                } else {
                    w.other_files += 1;
                }
            }
        }
    }
}

struct Progress {
    seen: AtomicU64,
    rs: AtomicU64,
    rs_bytes: AtomicU64,
}

/// «имя крейта» по пути: для ~/.cargo/registry — имя без версии; иначе — папка перед `src`
fn crate_key(path: &Path) -> (String, bool) {
    let comps: Vec<String> = path.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
    if let Some(r) = comps.iter().position(|c| c == "registry") {
        if comps.get(r + 1).map(|s| s == "src").unwrap_or(false) {
            if let Some(nv) = comps.get(r + 3) {
                // name-1.2.3 → name
                let b = nv.as_bytes();
                let mut cut = nv.len();
                for i in (1..b.len()).rev() {
                    if b[i - 1] == b'-' && b[i].is_ascii_digit() {
                        cut = i - 1;
                        break;
                    }
                }
                return (nv[..cut].to_string(), true);
            }
        }
    }
    if let Some(s) = comps.iter().rposition(|c| c == "src") {
        if s >= 1 {
            return (format!("проект:{}", comps[s - 1]), false);
        }
    }
    (format!("проект:{}", comps.get(comps.len().saturating_sub(2)).cloned().unwrap_or_default()), false)
}

fn looks_generated(b: &[u8]) -> bool {
    let head = &b[..b.len().min(2048)];
    let l: Vec<u8> = head.iter().map(|c| c.to_ascii_lowercase()).collect();
    find_sub(&l, b"@generated") || find_sub(&l, b"automatically generated") || find_sub(&l, b"auto-generated") || find_sub(&l, b"bindgen") || find_sub(&l, b"do not edit")
}

// ───────────── обработка одного .rs ─────────────
fn process(path: &Path, st: &mut Stats, files_set: &Shards, fn_set: &Shards, manifest: &Mutex<Vec<(PathBuf, u64)>>) {
    let len = match fs::metadata(path) {
        Ok(m) => m.len(),
        Err(_) => return,
    };
    if len == 0 {
        return;
    }
    st.files += 1;
    st.bytes += len;
    let mut buf = Vec::with_capacity(len as usize);
    match File::open(path).and_then(|mut f| f.read_to_end(&mut buf)) {
        Ok(_) => {}
        Err(_) => return,
    }
    let h = hash128(&buf);
    if !files_set.insert(h) {
        return; // дубликат целого файла
    }
    st.uniq_files += 1;
    st.uniq_bytes += len;
    manifest.lock().unwrap().push((path.to_path_buf(), len));
    let (key, _) = crate_key(path);
    let generated = looks_generated(&buf);
    if generated {
        st.gen_files += 1;
        st.gen_bytes += len;
    }
    {
        let e = st.krates.entry(key.clone()).or_insert([0; 5]);
        e[0] += 1;
        e[1] += len;
        e[4] += generated as u64;
    }
    if len > MAX_ANALYZE {
        st.skipped_big += 1;
        return;
    }
    for f in find_fns(&buf) {
        st.fns += 1;
        let text = &buf[f.start..f.end];
        if !fn_set.insert(hash128(text)) {
            continue;
        }
        let sz = text.len();
        let bk = bucket(sz);
        st.uniq_fns += 1;
        st.uniq_fn_bytes += sz as u64;
        {
            let e = st.krates.get_mut(&key).unwrap();
            e[2] += 1;
            e[3] += f.is_unsafe as u64;
        }
        st.hist[bk] += 1;
        st.hist_bytes[bk] += sz as u64;
        if f.doc {
            st.fn_doc += 1;
            st.fn_doc_fits[bk] += 1;
        }
        if f.test {
            st.fn_test += 1;
        }
        if f.is_unsafe {
            st.fn_unsafe += 1;
        }
    }
}

fn gb(x: u64) -> String {
    let g = x as f64 / (1u64 << 30) as f64;
    if g >= 1.0 {
        format!("{:.2} ГБ", g)
    } else {
        format!("{:.1} МБ", x as f64 / (1u64 << 20) as f64)
    }
}
fn pct(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut out_dir = PathBuf::from(".");
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--paths" => {
                i += 1;
                while i < argv.len() && !argv[i].starts_with("--") {
                    paths.push(PathBuf::from(&argv[i]));
                    i += 1;
                }
                continue;
            }
            "--out" => {
                i += 1;
                if i < argv.len() {
                    out_dir = PathBuf::from(&argv[i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    let (auto_roots, all_mounts) = detect_roots();
    let roots = if paths.is_empty() { auto_roots } else { paths };
    if roots.is_empty() {
        eprintln!("❌ не нашёл ни одного диска — укажи пути: --paths /путь");
        std::process::exit(1);
    }
    println!("Сканирую:");
    for r in &roots {
        println!("  • {}", r.display());
    }
    println!("(пропускаю: target/ рядом с Cargo.toml, node_modules, содержимое .git — учитываю только число репозиториев)\n");

    let t0 = Instant::now();
    let prog = Arc::new(Progress { seen: AtomicU64::new(0), rs: AtomicU64::new(0), rs_bytes: AtomicU64::new(0) });
    let files_set = Arc::new(Shards::new());
    let fn_set = Arc::new(Shards::new());
    let manifest: Arc<Mutex<Vec<(PathBuf, u64)>>> = Arc::new(Mutex::new(Vec::new()));
    let (tx, rx) = mpsc::channel::<PathBuf>();
    let rx = Arc::new(Mutex::new(rx));
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(2, 16);
    let mut handles = Vec::new();
    for _ in 0..nthreads {
        let (rx, fs_, fn_, mf, pr) = (rx.clone(), files_set.clone(), fn_set.clone(), manifest.clone(), prog.clone());
        handles.push(std::thread::spawn(move || {
            let mut st = Stats::default();
            loop {
                let p = { rx.lock().unwrap().recv() };
                match p {
                    Ok(p) => {
                        let before = st.bytes;
                        process(&p, &mut st, &fs_, &fn_, &mf);
                        pr.rs.fetch_add(1, Ordering::Relaxed);
                        pr.rs_bytes.fetch_add(st.bytes - before, Ordering::Relaxed);
                    }
                    Err(_) => break,
                }
            }
            st
        }));
    }
    // индикатор прогресса
    let pr2 = prog.clone();
    let stop = Arc::new(AtomicU64::new(0));
    let stop2 = stop.clone();
    let reporter = std::thread::spawn(move || {
        while stop2.load(Ordering::Relaxed) == 0 {
            std::thread::sleep(Duration::from_secs(2));
            eprintln!("  … файлов просмотрено {}, .rs обработано {} ({})", pr2.seen.load(Ordering::Relaxed), pr2.rs.load(Ordering::Relaxed), gb(pr2.rs_bytes.load(Ordering::Relaxed)));
        }
    });

    let mut w = Walk { other_mounts: HashSet::new(), ext_bytes: BTreeMap::new(), repos: 0, projects: 0, dirs: 0, other_files: 0, errors: 0, skipped_target: 0 };
    for m in &all_mounts {
        if !roots.contains(m) || roots.len() > 1 {
            w.other_mounts.insert(m.clone());
        }
    }
    for r in &roots {
        walk(r, &mut w, &tx, &prog);
    }
    drop(tx);
    let mut st = Stats::default();
    for h in handles {
        st.merge(&h.join().unwrap());
    }
    stop.store(1, Ordering::Relaxed);
    let _ = reporter.join();
    let secs = t0.elapsed().as_secs_f64();

    // ───────── отчёт ─────────
    let mut rep = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(rep, "══════════ ОТЧЁТ СКАНИРОВАНИЯ ({:.0} с) ══════════", secs);
    let _ = writeln!(rep, "Каталогов: {}, ошибок доступа: {}, пропущено target/node_modules: {}", w.dirs, w.errors, w.skipped_target);
    let _ = writeln!(rep, "Git-репозиториев: {}   Cargo-проектов (Cargo.toml): {}", w.repos, w.projects);
    let _ = writeln!(rep, "\n── Rust (.rs) ──");
    let _ = writeln!(rep, "Файлов: {}   объём: {}", st.files, gb(st.bytes));
    let _ = writeln!(rep, "Уникальных (по содержимому): {}   объём: {}   дублей: {:.1}% по объёму", st.uniq_files, gb(st.uniq_bytes), 100.0 - pct(st.uniq_bytes, st.bytes));
    let _ = writeln!(rep, "Файлов >4 МБ (вероятно сгенерированные, не нарезаны): {}", st.skipped_big);
    let _ = writeln!(rep, "Оценка размера после zstd (÷4.5, грубо): {}", gb((st.uniq_bytes as f64 / 4.5) as u64));
    let _ = writeln!(rep, "\n── Функции (с телом, без дублей) ──");
    let _ = writeln!(rep, "Найдено всего: {}   уникальных: {}   суммарно {}", st.fns, st.uniq_fns, gb(st.uniq_fn_bytes));
    let _ = writeln!(rep, "С doc-комментарием (готовая пара «описание → код»): {} ({:.1}%)", st.fn_doc, pct(st.fn_doc, st.uniq_fns));
    let _ = writeln!(rep, "Тесты (#[test]): {}   unsafe: {}", st.fn_test, st.fn_unsafe);
    let _ = writeln!(rep, "\nРазмер функции (байт) → сколько помещается в окно seq:");
    let mut cum = 0u64;
    let names = ["≤256", "≤512", "≤1024", "≤2048", "≤4096", "≤8192", ">8192"];
    for k in 0..7 {
        cum += st.hist[k];
        let _ = writeln!(rep, "  {:>6}: {:>10} функций ({:>5.1}%)   накопительно {:>5.1}%   из них с doc: {}", names[k], st.hist[k], pct(st.hist[k], st.uniq_fns), pct(cum, st.uniq_fns), st.fn_doc_fits[k]);
    }
    let _ = writeln!(rep, "\n── Качество: автогенерация и версии крейтов ──");
    let _ = writeln!(rep, "Файлов с пометкой генерации (@generated / bindgen / auto-generated): {} ({:.1}% уникальных файлов), объём {} ({:.1}% по объёму)", st.gen_files, pct(st.gen_files, st.uniq_files), gb(st.gen_bytes), pct(st.gen_bytes, st.uniq_bytes));
    let reg: Vec<_> = st.krates.keys().filter(|k| !k.starts_with("проект:")).collect();
    let _ = writeln!(rep, "Крейтов из ~/.cargo/registry (разных имён): {}   проектов вне registry: {}", reg.len(), st.krates.len() - reg.len());
    let _ = writeln!(rep, "\nТоп-25 «крейтов» по объёму (файлы | объём | уник. функций | из них unsafe | файлов-генераций):");
    let mut kv: Vec<_> = st.krates.iter().collect();
    kv.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
    for (k, v) in kv.iter().take(25) {
        let _ = writeln!(rep, "  {:<34} {:>6} | {:>9} | {:>8} | {:>8} ({:>3.0}%) | {}", k, v[0], gb(v[1]), v[2], v[3], pct(v[3], v[2]), v[4]);
    }
    let _ = writeln!(rep, "\nТоп-15 по доле unsafe (среди крейтов с ≥2000 функций):");
    let mut ku: Vec<_> = st.krates.iter().filter(|(_, v)| v[2] >= 2000).collect();
    ku.sort_by(|a, b| pct(b.1[3], b.1[2]).partial_cmp(&pct(a.1[3], a.1[2])).unwrap());
    for (k, v) in ku.iter().take(15) {
        let _ = writeln!(rep, "  {:<34} функций {:>8}, unsafe {:>3.0}%", k, v[2], pct(v[3], v[2]));
    }
    let top5: u64 = kv.iter().take(5).map(|(_, v)| v[2]).sum();
    let _ = writeln!(rep, "\nПервые 5 крейтов по объёму дают {:.1}% всех уникальных функций.", pct(top5, st.uniq_fns));
    let _ = writeln!(rep, "\n── Другие файлы кода и документации (только объём, без дедупликации) ──");
    let mut ex: Vec<_> = w.ext_bytes.iter().collect();
    ex.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    for (e, (n, b)) in ex {
        let _ = writeln!(rep, "  .{:<5} {:>9} файлов  {}", e, n, gb(*b));
    }
    let _ = writeln!(rep, "  прочих файлов (не код): {}", w.other_files);
    print!("\n{}", rep);

    let _ = fs::create_dir_all(&out_dir);
    let rp = out_dir.join("scan_report.txt");
    let mp = out_dir.join("rs_files.txt");
    let _ = fs::write(&rp, &rep);
    if let Ok(f) = File::create(&mp) {
        let mut bw = BufWriter::new(f);
        let mut m = manifest.lock().unwrap();
        m.sort();
        for (p, l) in m.iter() {
            let _ = writeln!(bw, "{}\t{}", l, p.display());
        }
    }
    println!("\nСохранено: {}  и  {} (список уникальных .rs: размер<TAB>путь)", rp.display(), mp.display());
}
