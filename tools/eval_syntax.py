#!/usr/bin/env python3
"""Метрика полезности для кода: доля синтаксически корректных Rust-фрагментов.

Для каждого промпта (начало функции/структуры, заканчивающееся на «{») модель продолжает текст.
Считаем:
  закрыто  — фигурные скобки закрылись в пределах длины генерации (модель довела конструкцию до конца);
  разбирается — закрытый фрагмент принимает парсер rustfmt (только синтаксис, имена не проверяются).
Итог: «разбирается / N» — главная цифра. Бит/байт этого не показывает.

  python3 tools/eval_syntax.py --ckpt workspace/checkpoints/model12_14m --n 40 --len 400
  python3 tools/eval_syntax.py --self-test        # проверка самого скрипта (без видеокарты)
  python3 tools/eval_syntax.py --reference FILE   # доля разбирающихся функций в настоящем коде (калибровка)

Нужны: rustfmt (идёт с rustup), собранный hashnet_proto.
"""
import argparse, os, re, subprocess, sys, shutil

PROMPTS = [
 "fn add(a: i32, b: i32) -> i32 {", "fn is_prime(n: u64) -> bool {", "fn factorial(n: u64) -> u64 {",
 "fn max_of(v: &[i32]) -> Option<i32> {", "pub fn new(name: String) -> Self {", "fn main() {",
 "impl Default for Config {", "fn parse_line(line: &str) -> Result<(String, i32), String> {",
 "fn reverse(s: &str) -> String {", "pub fn len(&self) -> usize {", "fn fib(n: u32) -> u32 {",
 "impl fmt::Display for Point {", "fn read_file(path: &str) -> std::io::Result<String> {", "fn sum(v: &[f64]) -> f64 {",
 "pub struct Stack<T> {", "fn swap(a: &mut i32, b: &mut i32) {", "fn count_words(text: &str) -> usize {",
 "impl<T> Stack<T> {", "pub enum Token {", "fn gcd(a: u64, b: u64) -> u64 {", "fn contains(&self, key: &str) -> bool {",
 "fn clamp(x: f32, lo: f32, hi: f32) -> f32 {", "pub fn push(&mut self, item: T) {", "fn to_upper(s: &str) -> String {",
 "impl Iterator for Counter {", "fn binary_search(v: &[i32], target: i32) -> Option<usize> {", "fn abs_diff(a: i32, b: i32) -> i32 {",
 "pub trait Shape {", "fn is_palindrome(s: &str) -> bool {", "fn mean(v: &[f64]) -> Option<f64> {",
 "fn handle_request(req: Request) -> Response {", "fn bubble_sort(v: &mut Vec<i32>) {", "pub fn get(&self, index: usize) -> Option<&T> {",
 "fn square(x: f64) -> f64 {", "impl Drop for Guard {", "fn first_word(s: &str) -> &str {", "fn min_max(v: &[i32]) -> (i32, i32) {",
 "pub fn is_empty(&self) -> bool {", "fn to_string(&self) -> String {", "impl From<i32> for Wrapper {", "fn collatz(mut n: u64) -> u32 {",
]

def scan_depth(text):
    """возвращает индекс (включительно) конца первой верхнеуровневой конструкции {...} или None.
    Учитывает строки, сырые строки, символы, комментарии (в т.ч. вложенные)."""
    i, n, depth, opened = 0, len(text), 0, False
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i); i = n if j < 0 else j + 1; continue
        if text.startswith("/*", i):
            d = 1; i += 2
            while i < n and d:
                if text.startswith("/*", i): d += 1; i += 2
                elif text.startswith("*/", i): d -= 1; i += 2
                else: i += 1
            continue
        m = re.match(r'b?r(#*)"', text[i:i + 12])
        if m:
            end = '"' + m.group(1); j = text.find(end, i + m.end())
            i = n if j < 0 else j + len(end); continue
        if c == '"' or (c == 'b' and text.startswith('b"', i)):
            i += 2 if c == 'b' else 1
            while i < n and text[i] != '"': i += 2 if text[i] == "\\" else 1
            i += 1; continue
        if c == "'":
            m = re.match(r"'(\\.[^']*|[^\\'])'", text[i:i + 12])
            if m: i += m.end(); continue
        if c == "{": depth += 1; opened = True
        elif c == "}":
            depth -= 1
            if opened and depth == 0: return i
            if depth < 0: return None
        i += 1
    return None

def parses(src):
    p = subprocess.run(["rustfmt", "--edition", "2021", "--emit", "stdout"], input=src.encode("utf-8", "replace"),
                       capture_output=True, timeout=30)
    return p.returncode == 0

def extract(out):
    lines = out.splitlines()
    s = next((i for i, l in enumerate(lines) if "инференс" in l), -1)
    e = next((i for i, l in enumerate(lines) if i > s and re.match(r"^\[\d+\s*байт", l)), len(lines))
    return "\n".join(lines[s + 1:e])

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)

def generate(a, prompt, i=0):
    cmd = [a.bin, "gen", "--ckpt", a.ckpt, "--len", str(a.len), "--temp", str(a.temp), "--top-k", str(a.top_k),
           "--seed", str(a.seed + i), "--prompt", prompt]
    if a.no_repeat: cmd += ["--no-repeat", str(a.no_repeat)]
    if a.loop_n: cmd += ["--loop-n", str(a.loop_n), "--loop-k", str(a.loop_k)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0: raise RuntimeError(r.stderr[-400:] or r.stdout[-400:])
    g = extract(r.stdout)
    return g if g.startswith(prompt) else prompt + g

def evaluate(texts):
    closed = ok = 0; rows = []
    for t in texts:
        k = scan_depth(t)
        if k is None: rows.append((False, False, t)); continue
        closed += 1
        frag = t[:k + 1]
        good = parses(frag)
        ok += good; rows.append((True, good, frag))
    return closed, ok, rows

def self_test():
    assert scan_depth("fn a() { let s = \"}\"; // }\n /* { */ let c = '}'; let l: &'a str = x; }") is not None
    assert scan_depth("fn a() { if x {") is None
    assert scan_depth('fn a() { let r = r#"}"#; }') is not None
    t = "fn add(a: i32, b: i32) -> i32 { a + b } trailing"
    assert t[:scan_depth(t) + 1] == "fn add(a: i32, b: i32) -> i32 { a + b }"
    assert parses("fn add(a: i32, b: i32) -> i32 { a + b }")
    assert not parses("fn add(a: i32 b) -> i32 { a + }")
    assert extract("(инференс: x)\nabc\ndef\n[10 байт за 1 с]") == "abc\ndef"
    c, o, _ = evaluate(["fn a() { 1 }", "fn b() { 1 +", "fn c( { }"])
    assert (c, o) == (2, 1), (c, o)
    print("self-test: ok")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--len", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.5); ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--no-repeat", type=int, default=0, help="0 = выкл. Малые N (3-6) ЛОМАЮТ код: запрещают повторять любые 4 байта, а в коде имена повторяются постоянно")
    ap.add_argument("--loop-n", type=int, default=0, help="защита от петель: окно N байт (напр. 24); 0 = выкл")
    ap.add_argument("--loop-k", type=int, default=2, help="сколько повторов окна допускается до запрета")
    ap.add_argument("--seed", type=int, default=1000, help="у каждого образца свой seed (иначе повтор промпта даёт тот же текст)")
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rust", "target", "release", "hashnet_proto"))
    ap.add_argument("--prompts"); ap.add_argument("--show", type=int, default=3, help="сколько неудачных примеров показать")
    ap.add_argument("--self-test", action="store_true"); ap.add_argument("--reference")
    a = ap.parse_args()
    if not shutil.which("rustfmt"): sys.exit("нужен rustfmt (rustup component add rustfmt)")
    if a.self_test: return self_test()
    if a.reference:
        txt = open(a.reference, encoding="utf-8", errors="replace").read()
        items = [m.group(0) for m in re.finditer(r"(?m)^(?:pub )?fn [^\n{]*\{", txt)][: a.n]
        texts = []
        for it in items:
            k = txt.find(it); end = scan_depth(txt[k:k + 4000]); texts.append(txt[k:k + end + 1] if end is not None else txt[k:k + 400])
        c, o, _ = evaluate(texts)
        print(f"настоящий код: закрыто {c}/{len(texts)}, разбирается {o}/{len(texts)}"); return
    if not a.ckpt: sys.exit("нужен --ckpt (или --self-test / --reference)")
    prompts = [l.rstrip("\n").replace("\\n", "\n") for l in open(a.prompts, encoding="utf-8")] if a.prompts else PROMPTS
    prompts = (prompts * (a.n // len(prompts) + 1))[: a.n]
    texts = []
    for i, p in enumerate(prompts):
        try: texts.append(generate(a, p, i))
        except Exception as e: sys.exit(f"генерация не удалась: {e}")
        print(f"\rгенерация {i + 1}/{len(prompts)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    c, o, rows = evaluate(texts)
    n = len(texts)
    lc, hc = wilson(c, n); lo, ho = wilson(o, n)
    print(f"закрыто (скобки сошлись за {a.len} байт): {c}/{n} = {100 * c / n:.0f}%  (95% ИС {100 * lc:.0f}–{100 * hc:.0f}%)")
    print(f"РАЗБИРАЕТСЯ как Rust:                      {o}/{n} = {100 * o / n:.0f}%  (95% ИС {100 * lo:.0f}–{100 * ho:.0f}%)")
    print(f"из закрытых разбирается: {o}/{c}" + (f" = {100 * o / c:.0f}%" if c else ""))
    bad = [r for r in rows if not (r[0] and r[1])][: a.show]
    for _, _, t in bad:
        print("\n── пример неудачи ──\n" + t[:300])

if __name__ == "__main__":
    main()
