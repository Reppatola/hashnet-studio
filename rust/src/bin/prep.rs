//! prep — подготовка датасета Rust: чистка, дедупликация, сырой корпус + пары «описание → код».
//! Только std. Запуск:
//!   cargo run --release --bin prep -- --paths "/media/tola/Новый том/rust_repos" --out "/media/tola/Новый том/rust_data"
//! (можно несколько путей, в т.ч. ~/.cargo/registry/src — оставляется последняя версия каждого крейта)
//!
//! Выход (в --out):
//!   rust_corpus.txt  — целые файлы с заголовком «// ===== путь =====», порядок перемешан
//!   rust_pairs.txt   — «User: <описание>\nAssistant:\n<код функции>» только для функций с doc-комментарием
//!   prep_report.txt  — что взято, что и почему отброшено
//! Дальше обучение читает эти файлы как обычные: train --data rust_corpus.txt (и/или rust_pairs.txt).
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

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


fn looks_generated(b: &[u8]) -> bool {
    let head = &b[..b.len().min(2048)];
    let l: Vec<u8> = head.iter().map(|c| c.to_ascii_lowercase()).collect();
    find_sub(&l, b"@generated") || find_sub(&l, b"automatically generated") || find_sub(&l, b"auto-generated") || find_sub(&l, b"bindgen") || find_sub(&l, b"do not edit")
}


struct Opts {
    max_file: usize,
    pair_max: usize,
    min_doc: usize,
    max_doc: usize,
    exclude: Vec<String>,
}

/// исключаемые по пути: автогенерированные FFI-обёртки, вендоренные копии, интринсики
fn path_excluded(comps: &[String], o: &Opts) -> Option<&'static str> {
    for c in comps {
        let l = c.to_lowercase();
        if l == "windows" || l.starts_with("windows-") || l.starts_with("windows_") || l.starts_with("winapi") || l == "linux-raw-sys" || l == "libc" || l.ends_with("-sys") || l.ends_with("_sys") || l == "core_arch" || l == "stdarch" {
            return Some("исключённый крейт (FFI/обёртки/интринсики)");
        }
        if l == "vendor" || l == "third_party" || l == "third-party" || l == "external" {
            return Some("вендоренная копия");
        }
        if o.exclude.iter().any(|e| &l == e) {
            return Some("исключено пользователем");
        }
    }
    None
}

fn parse_version(s: &str) -> Vec<u64> {
    s.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap_or(0)).collect()
}

/// registry/src/<index>/name-1.2.3/... → Some((name, version, индекс компоненты name-ver))
fn registry_info(comps: &[String]) -> Option<(String, Vec<u64>)> {
    let r = comps.iter().position(|c| c == "registry")?;
    if comps.get(r + 1).map(|s| s == "src").unwrap_or(false) {
        let nv = comps.get(r + 3)?;
        let b = nv.as_bytes();
        for i in (1..b.len()).rev() {
            if b[i - 1] == b'-' && b[i].is_ascii_digit() {
                return Some((nv[..i - 1].to_string(), parse_version(&nv[i..])));
            }
        }
        return Some((nv.clone(), vec![0]));
    }
    None
}

fn walk(root: &Path, out: &mut Vec<(PathBuf, u64)>, skipped: &mut BTreeMap<&'static str, u64>) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let entries: Vec<_> = rd.filter_map(|e| e.ok()).collect();
        let has_cargo = entries.iter().any(|e| e.file_name() == "Cargo.toml");
        for e in entries {
            let ft = match e.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ft.is_symlink() {
                continue;
            }
            let p = e.path();
            if ft.is_dir() {
                let name = e.file_name().to_string_lossy().to_string();
                if name == ".git" || name == "node_modules" || (name == "target" && has_cargo) {
                    continue;
                }
                stack.push(p);
            } else if ft.is_file() && p.extension().map(|x| x == "rs").unwrap_or(false) {
                match e.metadata() {
                    Ok(m) => out.push((p, m.len())),
                    Err(_) => *skipped.entry("нет доступа").or_insert(0) += 1,
                }
            }
        }
    }
}

/// нормализация: без BOM, LF, без хвостовых пробелов, с переводом строки в конце
fn clean_text(s: &str) -> String {
    let s = s.strip_prefix('\u{feff}').unwrap_or(s);
    let mut o = String::with_capacity(s.len() + 1);
    for line in s.replace("\r\n", "\n").replace('\r', "\n").split('\n') {
        o.push_str(line.trim_end());
        o.push('\n');
    }
    while o.ends_with("\n\n") {
        o.pop();
    }
    o
}

/// None — файл годен; Some(причина) — отброшен
fn bad_file(text: &str) -> Option<&'static str> {
    if text.len() < 40 {
        return Some("слишком короткий");
    }
    let lines = text.lines().count().max(1);
    let longest = text.lines().map(|l| l.len()).max().unwrap_or(0);
    if longest > 1000 || text.len() / lines > 120 {
        return Some("минифицирован / длинные строки");
    }
    let mut data = 0usize;
    let mut nonws = 0usize;
    for c in text.bytes() {
        if !c.is_ascii_whitespace() {
            nonws += 1;
            if c.is_ascii_digit() || c == b',' || c == b'x' || c == b'X' {
                data += 1;
            }
        }
    }
    if nonws > 0 && data * 100 / nonws > 45 {
        return Some("таблица данных");
    }
    None
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// очищенное описание из строк `///`; None — не годится
fn clean_doc(doc_lines: &[String], o: &Opts) -> Option<String> {
    let mut keep: Vec<&str> = Vec::new();
    for l in doc_lines {
        let t = l.trim();
        if t.starts_with("# ") || t.starts_with("```") || t.starts_with("## ") {
            break; // «# Examples», «# Safety», примеры — дальше не берём
        }
        keep.push(t);
    }
    let d = collapse(&keep.join(" "));
    if d.len() < o.min_doc || d.len() > o.max_doc {
        return None;
    }
    let words = d.split(' ').count();
    let chars = d.chars().count().max(1);
    let letters = d.chars().filter(|c| c.is_alphabetic() || *c == ' ').count(); // русские описания тоже подходят
    if words < 4 || letters * 100 / chars < 70 {
        return None;
    }
    let lower = d.to_lowercase();
    if lower.starts_with("todo") || lower.starts_with("fixme") || lower.starts_with("see ") || lower.starts_with("deprecated") {
        return None;
    }
    Some(d)
}

const PREFIXES: [&str; 4] = ["", "Write a Rust function. ", "Implement this in Rust: ", "Rust: "];

struct Rec {
    corpus: String,
    pairs: Vec<String>,
    pair_bytes: usize,
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut out_dir = PathBuf::from(".");
    let mut o = Opts { max_file: 512 << 10, pair_max: 1500, min_doc: 25, max_doc: 600, exclude: Vec::new() };
    let mut no_pairs = false;
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
                out_dir = PathBuf::from(&argv[i]);
            }
            "--max-file-kb" => {
                i += 1;
                o.max_file = argv[i].parse::<usize>().unwrap_or(512) << 10;
            }
            "--pair-max" => {
                i += 1;
                o.pair_max = argv[i].parse().unwrap_or(1500);
            }
            "--exclude" => {
                i += 1;
                o.exclude = argv[i].split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect();
            }
            "--no-pairs" => no_pairs = true,
            _ => {}
        }
        i += 1;
    }
    if paths.is_empty() {
        eprintln!("Использование: prep --paths <папка> [<папка>...] --out <папка> [--max-file-kb 512] [--pair-max 1500] [--exclude имя1,имя2] [--no-pairs]");
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let mut skipped: BTreeMap<&'static str, u64> = BTreeMap::new();

    // 1. список файлов
    let mut all: Vec<(PathBuf, u64)> = Vec::new();
    let mut root_of: HashMap<PathBuf, PathBuf> = HashMap::new();
    for r in &paths {
        let before = all.len();
        walk(r, &mut all, &mut skipped);
        for (p, _) in &all[before..] {
            root_of.insert(p.clone(), r.clone());
        }
    }
    println!("Найдено .rs: {}", all.len());

    // 2. фильтры по пути + последняя версия крейта из registry
    let mut latest: HashMap<String, Vec<u64>> = HashMap::new();
    let comps_of = |p: &Path| -> Vec<String> { p.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect() };
    for (p, _) in &all {
        if let Some((n, v)) = registry_info(&comps_of(p)) {
            let e = latest.entry(n).or_insert_with(Vec::new);
            if v > *e {
                *e = v;
            }
        }
    }
    let mut list: Vec<(PathBuf, u64)> = Vec::new();
    for (p, len) in all {
        let comps = comps_of(&p);
        if let Some(why) = path_excluded(&comps, &o) {
            *skipped.entry(why).or_insert(0) += 1;
            continue;
        }
        if let Some((n, v)) = registry_info(&comps) {
            if latest.get(&n).map(|l| *l != v).unwrap_or(false) {
                *skipped.entry("старая версия крейта").or_insert(0) += 1;
                continue;
            }
        }
        if len as usize > o.max_file {
            *skipped.entry("слишком большой файл").or_insert(0) += 1;
            continue;
        }
        list.push((p, len));
    }
    // детерминированное перемешивание по хешу пути (чтобы конец файла, из которого берётся валидация, был разнородным)
    list.sort_by_key(|(p, _)| hash128(p.to_string_lossy().as_bytes()));
    println!("После фильтров по пути: {}", list.len());

    // 3. обработка в потоках
    let list = Arc::new(list);
    let root_of = Arc::new(root_of);
    let next = Arc::new(AtomicUsize::new(0));
    let file_dups = Arc::new(Shards::new());
    let fn_dups = Arc::new(Shards::new());
    let doc_dups = Arc::new(Shards::new());
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(2, 16);
    let (tx, rx) = mpsc::sync_channel::<Result<Rec, &'static str>>(256);
    let mut handles = Vec::new();
    for _ in 0..nthreads {
        let (list, root_of, next, fd, fnd, dd, tx) = (list.clone(), root_of.clone(), next.clone(), file_dups.clone(), fn_dups.clone(), doc_dups.clone(), tx.clone());
        let pair_max = o.pair_max;
        let (min_doc, max_doc) = (o.min_doc, o.max_doc);
        let oo = Opts { max_file: o.max_file, pair_max, min_doc, max_doc, exclude: Vec::new() };
        let want_pairs = !no_pairs;
        handles.push(std::thread::spawn(move || loop {
            let idx = next.fetch_add(1, Ordering::Relaxed);
            if idx >= list.len() {
                break;
            }
            let (p, _) = &list[idx];
            let mut raw = Vec::new();
            if File::open(p).and_then(|mut f| f.read_to_end(&mut raw)).is_err() {
                let _ = tx.send(Err("не прочитан"));
                continue;
            }
            let s = match String::from_utf8(raw) {
                Ok(s) => s,
                Err(_) => {
                    let _ = tx.send(Err("не UTF-8"));
                    continue;
                }
            };
            let text = clean_text(&s);
            if looks_generated(text.as_bytes()) {
                let _ = tx.send(Err("сгенерирован (@generated/bindgen/…)"));
                continue;
            }
            if let Some(why) = bad_file(&text) {
                let _ = tx.send(Err(why));
                continue;
            }
            let norm: Vec<u8> = text.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
            if !fd.insert(hash128(&norm)) {
                let _ = tx.send(Err("дубликат файла (с точностью до пробелов)"));
                continue;
            }
            let root = root_of.get(p).cloned().unwrap_or_default();
            let rel = p.strip_prefix(&root).unwrap_or(p).to_string_lossy().to_string();
            let label = format!("{}/{}", root.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default(), rel);
            let mut corpus = format!("// ===== {} =====\n", label);
            corpus.push_str(&text);
            corpus.push('\n');
            let mut pairs = Vec::new();
            let mut pair_bytes = 0;
            if want_pairs {
                let b = text.as_bytes();
                for f in find_fns(b) {
                    if f.test {
                        continue;
                    }
                    if !f.doc {
                        continue;
                    }
                    let seg = &text[f.start..f.end];
                    let mut doc_lines: Vec<String> = Vec::new();
                    let mut code = String::new();
                    let mut in_head = true;
                    for line in seg.lines() {
                        let t = line.trim_start();
                        if in_head && t.starts_with("///") {
                            doc_lines.push(t.trim_start_matches('/').to_string());
                        } else {
                            in_head = false;
                            code.push_str(line);
                            code.push('\n');
                        }
                    }
                    let doc = match clean_doc(&doc_lines, &oo) {
                        Some(d) => d,
                        None => continue,
                    };
                    if doc.len() + code.len() > pair_max {
                        continue;
                    }
                    if !fnd.insert(hash128(code.as_bytes())) || !dd.insert(hash128(doc.as_bytes())) {
                        continue;
                    }
                    let pre = PREFIXES[(hash128(doc.as_bytes()) % 4) as usize];
                    let rec = format!("User: {}{}\nAssistant:\n{}\n", pre, doc, code);
                    pair_bytes += rec.len();
                    pairs.push(rec);
                }
            }
            let _ = tx.send(Ok(Rec { corpus, pairs, pair_bytes }));
        }));
    }
    drop(tx);

    // 4. запись
    fs::create_dir_all(&out_dir).expect("не создать --out");
    let mut wc = BufWriter::with_capacity(1 << 20, File::create(out_dir.join("rust_corpus.txt")).expect("corpus"));
    let mut wp = BufWriter::with_capacity(1 << 20, File::create(out_dir.join("rust_pairs.txt")).expect("pairs"));
    let (mut n_files, mut b_corpus, mut n_pairs, mut b_pairs) = (0u64, 0u64, 0u64, 0u64);
    let mut done = 0u64;
    for r in rx {
        done += 1;
        match r {
            Ok(rec) => {
                n_files += 1;
                b_corpus += rec.corpus.len() as u64;
                let _ = wc.write_all(rec.corpus.as_bytes());
                for p in &rec.pairs {
                    let _ = wp.write_all(p.as_bytes());
                }
                n_pairs += rec.pairs.len() as u64;
                b_pairs += rec.pair_bytes as u64;
            }
            Err(why) => *skipped.entry(why).or_insert(0) += 1,
        }
        if done % 20000 == 0 {
            eprintln!("  … обработано {} / {}", done, list.len());
        }
    }
    for h in handles {
        let _ = h.join();
    }
    let _ = wc.flush();
    let _ = wp.flush();

    let mut rep = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(rep, "══════════ ОТЧЁТ prep ({:.0} с) ══════════", t0.elapsed().as_secs_f64());
    let _ = writeln!(rep, "Корпус: {} файлов, {:.2} ГБ  → rust_corpus.txt", n_files, b_corpus as f64 / 1073741824.0);
    let _ = writeln!(rep, "Пары:   {} шт., {:.2} МБ  → rust_pairs.txt (средний размер {} байт)", n_pairs, b_pairs as f64 / 1048576.0, if n_pairs > 0 { b_pairs / n_pairs } else { 0 });
    let _ = writeln!(rep, "\nОтброшено:");
    for (k, v) in &skipped {
        let _ = writeln!(rep, "  {:>9}  {}", v, k);
    }
    print!("\n{}", rep);
    let _ = fs::write(out_dir.join("prep_report.txt"), &rep);
    println!("\nФайлы в {}", out_dir.display());
}
