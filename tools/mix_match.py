#!/usr/bin/env python3
"""Оценка «длинного контекста»: что даст смесь модели с match-моделью, которая видит весь предыдущий текст хвоста,
а не только окно 512 байт.

  hashnet_proto dumpnll --ckpt МОДЕЛЬ --out nll.bin --val-windows 1024
  python3 tools/mix_match.py nll.bin ФАЙЛ_С_ХВОСТОМ_VAL [--tail 4194304] [--minlen 8]

Окна val лежат в хвосте файла с шагом tail/n. Match-модель: ищет прошлое вхождение последних minlen байт (только ВНУТРИ хвоста,
обучающие данные не используются) и предсказывает байт, который шёл за ним. Смесь p=λ·p_модели+(1-λ)·p_match, λ — по длине совпадения.
Печатает общий выигрыш и выигрыш по участкам окна (начало окна — там у модели мало контекста).
"""
import argparse, struct, sys
import numpy as np


def read_dump(path):
    with open(path, "rb") as f:
        if f.read(4) != b"NLLD":
            sys.exit("это не файл dumpnll")
        n, t = struct.unpack("<II", f.read(8))
        wins = np.frombuffer(f.read(n * (t + 1)), dtype=np.uint8).reshape(n, t + 1)
        nll = np.frombuffer(f.read(n * t * 4), dtype="<f4").reshape(n, t)
    return wins, nll


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump"); ap.add_argument("file")
    ap.add_argument("--tail", type=int, default=4194304)
    ap.add_argument("--minlen", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=48)
    a = ap.parse_args()
    wins, nll = read_dump(a.dump)
    n, t = nll.shape
    with open(a.file, "rb") as f:
        f.seek(-a.tail, 2)
        tail = f.read(a.tail)
    stride = a.tail // n
    # проверка: окна действительно лежат в хвосте с этим шагом
    ok = sum(1 for i in range(0, n, max(1, n // 20)) if tail[i * stride:i * stride + t + 1] == bytes(wins[i]))
    if ok < len(range(0, n, max(1, n // 20))):
        sys.exit("окна из dump не совпадают с хвостом файла: проверьте --tail и файл (должен быть тем же, на котором обучалась модель)")
    want = {}  # индекс входного байта j -> (i,p)
    for i in range(n):
        for p in range(t):
            want[i * stride + p] = (i, p)
    pred = np.zeros((n, t), dtype=np.int32) - 1
    mlen = np.zeros((n, t), dtype=np.int32)
    last = {}
    m = a.minlen
    end = (n - 1) * stride + t
    for j in range(m - 1, end):
        key = tail[j - m + 1:j + 1]
        q = last.get(key)
        if j in want and q is not None:
            L = m
            while L < a.maxlen and q - L >= 0 and tail[q - L] == tail[j - L]:
                L += 1
            i, p = want[j]
            pred[i, p] = tail[q + 1]
            mlen[i, p] = L
        last[key] = j
    tgt = wins[:, 1:t + 1]
    pm = np.exp(-nll.astype(np.float64))
    has = pred >= 0
    hit = has & (pred == tgt)
    print(f"позиций с совпадением (≥{m} байт): {has.mean()*100:.1f}%; из них предсказал верно: {hit.sum()/max(1,has.sum())*100:.1f}%")
    bits = lambda pr: float(-np.log2(np.maximum(pr, 1e-12)).mean())
    # λ и уверенность match по корзинам длины
    edges = [0, m, 12, 16, 24, 32, 10 ** 9]
    cb = np.digitize(mlen, edges[1:-1]) * has + (~has) * 99
    grid = np.linspace(0.02, 0.99, 98)
    mix = pm.copy()
    for b in np.unique(cb):
        if b == 99:
            continue
        mk = cb == b
        acc = hit[mk].mean()
        conf = float(np.clip(acc, 0.01, 0.999))
        pmatch = np.where(hit[mk], conf, (1 - conf) / 255.0)
        sc = [float(-np.log2(np.maximum(l * pm[mk] + (1 - l) * pmatch, 1e-12)).sum()) for l in grid]
        l = grid[int(np.argmin(sc))]
        mix[mk] = l * pm[mk] + (1 - l) * pmatch
        print(f"  длина совпадения {edges[b]}–{edges[b+1] if edges[b+1] < 10**8 else '∞'}: {mk.sum()} позиций, верно {acc*100:.0f}%, λ={l:.2f}")
    print(f"\nмодель:               {bits(pm):.3f} бит/байт")
    print(f"смесь с match-моделью: {bits(mix):.3f} бит/байт  (выигрыш {bits(pm)-bits(mix):+.3f})")
    for lo, hi in ((0, 64), (64, 256), (256, t)):
        hi = min(hi, t)
        if lo >= hi:
            continue
        print(f"  позиции окна {lo}–{hi}: модель {bits(pm[:, lo:hi]):.3f} → смесь {bits(mix[:, lo:hi]):.3f}")
    print("\nКак читать: выигрыш > 0.02 — модели не хватает контекста 512 байт (длинный контекст/кэш повторов оправдан); ≈0 — хватает.")


if __name__ == "__main__":
    main()
