#!/usr/bin/env python3
"""
split_corpus.py — нарезает большой смешанный текстовый файл на файлы по типам источников
и выбрасывает сгенерированный код (AWS SDK, svd2rust/PAC, bindgen/FFI-таблицы, prost, windows-crate).

Как работает: файл читается потоком, режется на блоки ~32 КБ по границам строк, каждый блок
классифицируется по признакам (доля кириллицы, маркеры Rust, User:/Assistant:, markdown, тире-диалоги,
плотность сгенерированных конструкций), результат слегка сглаживается соседними блоками
и дописывается в файл своего класса. Только стандартная библиотека Python.

Классы (файлы <out>/<класс>.txt):
  rust          — обычный (написанный людьми) Rust-код
  chat_code     — диалоги User/Assistant и ответы с блоками кода (``` ... ```)
  chat_ru       — русские диалоги/ответы ассистента без кода (наука, право, ...)
  chat_en       — то же по-английски
  dialog_ru     — русская художественная проза с репликами через тире
  prose_ru      — русский связный текст без диалогов и разметки
  prose_en      — английская проза/статьи
  code_other    — прочий код вне чатов (python, js, ...)
  rust_generated, junk — по умолчанию НЕ записываются (только считаются); --keep-generated чтобы сохранить

Запуск:
  python3 split_corpus.py combined_v3.txt --out split_out            # полная нарезка
  python3 split_corpus.py combined_v3.txt --dry-run --limit-gb 1     # быстрая проба: только отчёт
"""
import argparse, os, random, string, sys, time, re
import multiprocessing as mp

BLOCK = 32 * 1024
JOB = 32 * 1024 * 1024
LAT_BYTES = string.ascii_letters.encode()

RUST_TOKENS = [b"fn ", b"let mut ", b"if let ", b"while let ", b"let Some(", b"let (", b"let _ ", b"&self", b"&mut ", b"impl ",
               b"impl<", b"#[", b"#![", b"unwrap()", b".unwrap_or", b"Result<", b"Option<", b"Some(", b"None,", b"None)",
               b"None;", b"Ok(", b"Err(", b"use std", b"use crate", b"crate::", b"super::", b"self::", b"::new(", b"::from(",
               b"pub fn", b"pub struct", b"pub enum", b"pub(", b"pub mod", b"pub use", b"pub trait", b"-> Self", b" -> ",
               b" => ", b"&'a ", b"&'static", b"'static", b"Vec<", b"Box<", b"Arc<", b".iter()", b".collect", b".clone()",
               b".into()", b"?;", b"unsafe {", b"unsafe fn", b"println!", b"format!", b"assert", b"matches!", b"vec![",
               b"///", b"//!", b"#[derive", b"#[cfg", b"#[test]", b"mod tests", b"extern crate", b"::<", b"dyn ", b"where "]
# маркеры сгенерированного кода. Каждая группа со своим порогом (вхождений на КБ).
GEN_GROUPS = [
    (1.0, [b"aws_smithy", b"aws_sdk", b"SdkError", b"aws_types", b"std::option::Option", b"std::string::String",
           b"std::vec::Vec", b"std::result::Result", b"std::clone::Clone", b"std::cmp::PartialEq", b"fluent_builders",
           b"::std::option::Option", b"::std::string::String"]),                       # aws-sdk-rust и подобные
    (0.5, [b"::prost::", b"#[prost(", b"::prost::alloc::"]),   # prost/tonic
    (1.0, [b"::std::os::raw::", b"__bindgen", b"bindgen_", b"::core::ffi::c_", b"::libc::c_"]),  # bindgen
    (1.0, [b"windows_core::", b"::windows::core", b"windows_targets", b"windows_sys::"]),  # windows-rs
    (1.0, [b"__sym0", b"__action", b"__state", b"__Nonterminal", b"__lookahead", b"__ParseError", b"__reduce"]),  # lalrpop-парсеры
    (1.5, [b"from_glib", b"to_glib_none", b"to_glib_full", b'feature = "dox"', b"IsA<", b"glib::wrapper!", b"ffi::"]),  # gir/gtk-rs
]
PAC_DOC = b'#[doc = "'
PAC_MARK = [b"crate::Readable", b"crate::Writable", b"crate::Reg<", b"crate::FieldReader", b"crate::BitReader",
            b"crate::FieldWriter", b"crate::BitWriter", b"RegisterSpec", b"_SPEC"]
RE_CONST = re.compile(rb'^[ \t]*pub const [A-Z0-9_]+: [^=\n]+ = [^;\n]+;[ \t]*$', re.M)
RE_FENCE = re.compile(rb'^[ \t]*```', re.M)
RE_HEAD = re.compile(rb'^#{1,6} ', re.M)
RE_BULLB = re.compile(rb'^[ \t]*(?:[-*] \*\*|\d+\. \*\*)', re.M)
RE_UA = re.compile(('(?:^|\\n)(?:User|Assistant|Human|Пользователь|Ассистент): ?').encode(), re.M)
RE_DASH = re.compile('^— '.encode(), re.M)
SYMS = [b"{", b"}", b";", b"(", b")", b"<", b">", b"="]


class Th:
    rust = 1.5      # Rust-маркеров на КБ
    ends = 0.15     # доля строк, оканчивающихся на ; { } , (признак C-подобного кода)
    md = 1.0        # markdown-маркеров (** # ``` пункты) на КБ -> чат
    sym = 0.04      # плотность символов кода
    dash = 0.06     # доля строк с "— " -> диалог
    gen = 1.0       # множитель к порогам генерации (больше = мягче)


def features(b):
    n = len(b)
    kb = n / 1024.0
    lines = b.count(b"\n") + 1
    cyr = b.count(b"\xd0") + b.count(b"\xd1")
    lat = n - len(b.translate(None, LAT_BYTES))
    f = {"n": n, "lines": lines}
    f["cyr"] = cyr / (cyr + lat + 1.0)
    f["rust"] = (sum(b.count(t) for t in RUST_TOKENS) + 0.3 * b.count(b"::")) / kb
    f["ends"] = (b.count(b";\n") + b.count(b"{\n") + b.count(b"}\n") + b.count(b",\n")) / lines
    f["gen"] = max(sum(b.count(t) for t in toks) / kb / thr for thr, toks in GEN_GROUPS)
    pdoc = b.count(PAC_DOC) / kb
    f["pac"] = pdoc / 3.0 if (pdoc and any(t in b for t in PAC_MARK)) else 0.0
    f["consts"] = len(RE_CONST.findall(b)) / lines
    f["fence"] = len(RE_FENCE.findall(b))
    f["md"] = (b.count(b"**") // 2 + len(RE_HEAD.findall(b)) + f["fence"] + len(RE_BULLB.findall(b))) / kb
    f["ua"] = len(RE_UA.findall(b))
    f["sym"] = sum(b.count(s) for s in SYMS) / n
    f["dash"] = len(RE_DASH.findall(b)) / lines
    return f


def classify(b, th):
    """возвращает (класс, причина, признаки)"""
    if len(b) < 256 or not b.strip():
        return "junk", "short", None
    f = features(b)
    n = len(b)
    rusty = f["rust"] >= th.rust and f["ends"] >= th.ends
    if f["gen"] * th.gen >= 1.0:
        return "rust_generated", "gen:groups", f
    if f["pac"] >= 1.0:
        return "rust_generated", "gen:pac", f
    if f["consts"] >= 0.4 and f["lines"] >= 20:
        return "rust_generated", "gen:consts", f
    digits = n - len(b.translate(None, b"0123456789"))
    if digits / n >= 0.25 or b.count(b" ") / n < 0.04:
        return "junk", "numeric/base64", f
    if f["ua"] >= 2:
        why = "chat:ua>=2"
    elif f["ua"] >= 1 and not rusty:
        why = "chat:ua1"
    elif f["fence"] >= 2:
        why = "chat:fence>=2"
    elif f["md"] >= th.md and not rusty:
        why = "chat:md"
    else:
        why = None
    if why:
        if f["fence"] >= 1 or rusty:
            return "chat_code", why, f
        return ("chat_ru" if f["cyr"] >= 0.1 else "chat_en"), why, f
    if rusty and f["cyr"] < 0.6:
        return "rust", "rusty", f
    if f["sym"] >= th.sym and f["cyr"] < 0.3:
        return "code_other", "sym", f
    if f["cyr"] >= 0.5:
        return ("dialog_ru", "dash", f) if f["dash"] >= th.dash else ("prose_ru", "ru", f)
    return "prose_en", "else", f


def smooth(cls):
    out = list(cls)
    for i in range(1, len(cls) - 1):
        if cls[i - 1] == cls[i + 1] != cls[i]:
            out[i] = cls[i - 1]
    # длинные ответы чата без маркеров: связный текст сразу после чата (до 8 блоков) считаем продолжением чата
    run = 0
    for i, c in enumerate(out):
        if c.startswith("chat_"):
            run = 8 if i == 0 or True else run
            last = c
        elif run > 0 and c in ("prose_ru", "prose_en"):
            out[i] = "chat_ru" if c == "prose_ru" else "chat_en"
            run -= 1
        else:
            run = 0
    return out


def process(args):
    path, a, b, th = args
    with open(path, "rb") as f:
        f.seek(a)
        if a > 0:
            f.readline()
        pos = f.tell()
        if pos >= b:
            return {}, [], 0, {}
        data = f.read(b - pos)
        data += f.readline()
    data = data.replace(b"\x00", b"")  # редкие бинарные вкрапления
    # режем на блоки по границам строк
    blocks, i, n = [], 0, len(data)
    while i < n:
        j = min(i + BLOCK, n)
        if j < n:
            k = data.find(b"\n", j)
            j = n if k < 0 else k + 1
        blocks.append(data[i:j])
        i = j
    res3 = [classify(x, th) for x in blocks]
    cls = smooth([r[0] for r in res3])
    why_cnt = {}
    for c, r in zip(cls, res3):
        key = (c, r[1] if r[0] == c else "smooth")
        why_cnt[key] = why_cnt.get(key, 0) + 1
    out, order = {}, []
    for c, x in zip(cls, blocks):
        out.setdefault(c, []).append(x)
    res = {c: b"".join(v) for c, v in out.items()}
    # образцы: начало блока на смене класса
    samples, prev = [], None
    for c, x, r in zip(cls, blocks, res3):
        if c != prev:
            ft = ""
            if r[2]:
                ft = " [" + r[1] + " " + " ".join(f"{k}={r[2][k]:.2f}" for k in ("rust", "ends", "gen", "md", "cyr", "sym")) + f" ua={r[2]['ua']} fence={r[2]['fence']}]"
            samples.append((c, ft + " " + x[:200].decode("utf-8", "ignore").replace("\n", "\\n")))
        prev = c
    return res, samples, len(data), why_cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", default="split_out")
    ap.add_argument("--dry-run", action="store_true", help="ничего не писать, только отчёт")
    ap.add_argument("--limit-gb", type=float, default=0, help="обработать только первые N ГБ (проба)")
    ap.add_argument("--keep-generated", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 8))
    ap.add_argument("--rust-thr", type=float, default=Th.rust)
    ap.add_argument("--gen-mult", type=float, default=Th.gen, help="множитель чувствительности к сгенерированному коду")
    ap.add_argument("--md-thr", type=float, default=Th.md)
    a = ap.parse_args()
    th = Th()
    th.rust, th.gen, th.md = a.rust_thr, a.gen_mult, a.md_thr

    size = os.path.getsize(a.src)
    if a.limit_gb:
        size = min(size, int(a.limit_gb * 1e9))
    jobs = [(a.src, s, min(s + JOB, size), th) for s in range(0, size, JOB)]
    drop = set() if a.keep_generated else {"rust_generated", "junk"}
    handles = {}
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
    tot, samples_by, t0, done_b = {}, {}, time.time(), 0
    why_tot = {}
    rnd = random.Random(1)
    with mp.Pool(a.jobs) as pool:
        for res, samples, nb, wc in pool.imap(process, jobs, chunksize=1):
            done_b += nb
            for k, v in wc.items():
                why_tot[k] = why_tot.get(k, 0) + v
            for c, data in res.items():
                tot[c] = tot.get(c, 0) + len(data)
                if not a.dry_run and c not in drop:
                    if c not in handles:
                        handles[c] = open(os.path.join(a.out, c + ".txt"), "ab")
                    handles[c].write(data)
            for c, s in samples:
                samples_by.setdefault(c, []).append(s)
            el = time.time() - t0
            print(f"\r{done_b/1e9:6.2f}/{size/1e9:.2f} ГБ  {done_b/1e6/max(el,1e-9):5.1f} МБ/с", end="", file=sys.stderr, flush=True)
    for h in handles.values():
        h.close()
    print(file=sys.stderr)

    total = sum(tot.values()) or 1
    lines = [f"Источник: {a.src}  обработано {total/1e9:.2f} ГБ  ({'пробный прогон' if a.dry_run else 'записано в ' + a.out})", ""]
    for c, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        mark = "  (не записан)" if (c in drop or a.dry_run) else ""
        lines.append(f"{c:15} {v/1e9:7.3f} ГБ  {100*v/total:5.1f}%{mark}")
    lines.append("")
    lines.append("Почему блоки попали в класс (число блоков по 32 КБ):")
    for (c, w), v in sorted(why_tot.items(), key=lambda kv: (kv[0][0], -kv[1])):
        lines.append(f"   {c:15} {w:16} {v}")
    lines.append("")
    lines.append("Образцы (по 5 случайных начал сегментов каждого класса — проверь глазами!):")
    for c in sorted(samples_by):
        lines.append(f"--- {c}")
        for s in rnd.sample(samples_by[c], min(5, len(samples_by[c]))):
            lines.append("   " + s[:330])
    rep = "\n".join(lines)
    print(rep)
    if not a.dry_run:
        with open(os.path.join(a.out, "split_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep + "\n")
        keep = [c for c in tot if c not in drop]
        print("\nПример аргументов обучения (веса подбери сам):")
        print("  " + " ".join(f"--data {a.out}/{c}.txt@1" for c in sorted(keep)))


if __name__ == "__main__":
    main()
