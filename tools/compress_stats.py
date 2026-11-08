#!/usr/bin/env python3
"""Сжимаемость кусков корпуса (zlib) — бесплатная оценка «сколько в куске новой информации».

  stats   ФАЙЛ  [--chunk 4096] [--sample 50000]   распределение сжимаемости + доля точных дубликатов (по выборке)
  filter  ФАЙЛ  --out НОВЫЙ [--min-ratio 0.15] [--max-ratio 1.0] [--dedup] [--keep-tail 4194304]
                                                   выбрасывает куски, которые слишком хорошо сжимаются (повторы, шаблоны)
ratio = сжатый размер / исходный: 0.1 — почти сплошной повтор, 0.35 — обычный код, ~1.0 — случайные/бинарные данные.
Хвост файла (по умолчанию 4 МиБ — это валидация программы) копируется без изменений: val остаётся тем же.
"""
import argparse, hashlib, os, random, sys, zlib


def ratio(b):
    return len(zlib.compress(b, 3)) / max(1, len(b))


def pct(xs, p):
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def cmd_stats(a):
    size = os.path.getsize(a.file)
    n = max(1, (size - a.keep_tail) // a.chunk)
    k = min(a.sample, n)
    rnd = random.Random(1)
    idx = sorted(rnd.sample(range(n), k))
    rs, seen, dup = [], set(), 0
    with open(a.file, "rb") as f:
        for i in idx:
            f.seek(i * a.chunk)
            b = f.read(a.chunk)
            if len(b) < a.chunk:
                continue
            rs.append(ratio(b))
            h = hashlib.blake2b(b, digest_size=8).digest()
            if h in seen:
                dup += 1
            seen.add(h)
    rs.sort()
    print(f"файл {size/1e9:.2f} ГБ, кусков по {a.chunk} Б: {n}, в выборке {len(rs)}")
    print("перцентили ratio (меньше = сжимается сильнее):")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  {p:>2}%: {pct(rs, p):.3f}")
    print("гистограмма:")
    edges = [0, .05, .10, .15, .20, .25, .30, .35, .40, .50, .70, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        c = sum(1 for r in rs if lo <= r < hi)
        print(f"  {lo:.2f}–{min(hi,1.0):.2f}: {c/len(rs)*100:5.1f}%  " + "█" * int(c / len(rs) * 60))
    print(f"точные дубликаты кусков внутри выборки: {dup/max(1,len(rs))*100:.2f}% (в полном файле их больше)")


def cmd_filter(a):
    size = os.path.getsize(a.file)
    body = size - a.keep_tail
    kept = dropped = dups = 0
    seen = set()
    with open(a.file, "rb") as f, open(a.out, "wb") as o:
        pos = 0
        while pos < body:
            b = f.read(min(a.chunk, body - pos))
            pos += len(b)
            if len(b) == a.chunk:
                r = ratio(b)
                bad = r < a.min_ratio or r > a.max_ratio
                if not bad and a.dedup:
                    h = hashlib.blake2b(b, digest_size=8).digest()
                    if h in seen:
                        bad = True
                        dups += 1
                    seen.add(h)
                if bad:
                    dropped += 1
                    continue
            o.write(b)
            kept += 1
        f.seek(body)
        o.write(f.read())  # хвост (валидация) без изменений
    tot = kept + dropped
    print(f"кусков: {tot}; оставлено {kept} ({kept/max(1,tot)*100:.1f}%), выброшено {dropped} (из них дубликатов {dups})")
    print(f"размер: {size/1e9:.3f} → {os.path.getsize(a.out)/1e9:.3f} ГБ; хвост {a.keep_tail} Б скопирован без изменений")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stats", "filter"):
        p = sp.add_parser(name)
        p.add_argument("file")
        p.add_argument("--chunk", type=int, default=4096)
        p.add_argument("--keep-tail", type=int, default=4194304)
        if name == "stats":
            p.add_argument("--sample", type=int, default=50000)
        else:
            p.add_argument("--out", required=True)
            p.add_argument("--min-ratio", type=float, default=0.15)
            p.add_argument("--max-ratio", type=float, default=1.0)
            p.add_argument("--dedup", action="store_true")
    a = ap.parse_args()
    if a.cmd == "filter" and os.path.abspath(a.out) == os.path.abspath(a.file):
        sys.exit("--out не должен совпадать с исходным файлом")
    {"stats": cmd_stats, "filter": cmd_filter}[a.cmd](a)


if __name__ == "__main__":
    main()
