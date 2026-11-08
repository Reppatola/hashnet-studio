#!/usr/bin/env python3
"""Утечка val в обучение: какая доля «отпечатков» хвоста (val) встречается в остальной части файла.

Берутся все окна по W байт (по умолчанию 32). Из них оставляются только окна, чей хэш делится на 64
(выбор зависит от СОДЕРЖИМОГО, поэтому одинаковый текст попадает в выборку одинаково и в обучении, и в val).
Результат: доля таких окон val, которые есть в обучающей части. 0% — утечки нет, высокие значения —
в val много почти-дубликатов обучающих файлов (тогда val-цифры слишком оптимистичны).

  python3 tools/overlap_check.py ФАЙЛ [--tail 4194304] [--window 32] [--train-bytes N]
"""
import argparse, os, sys
import numpy as np

MUL = np.uint64(0x9E3779B97F4A7C15)
MOD = 64


def window_hashes(buf, w):
    """хэш каждого окна из w байт (полином, uint64 с переполнением) — векторно; длина n-w+1"""
    x = np.frombuffer(buf, dtype=np.uint8).astype(np.uint64)
    n = len(x) - w + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(w):
            h = h * MUL + (x[j:j + n] + np.uint64(1))
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return h


def sampled(buf, w):
    h = window_hashes(buf, w)
    return h[(h % np.uint64(MOD)) == 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--tail", type=int, default=4194304)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--train-bytes", type=int, default=0, help="сколько байт с начала считать обучением (по умолчанию всё до хвоста)")
    a = ap.parse_args()
    size = os.path.getsize(a.file)
    body = a.train_bytes or (size - a.tail)
    w = a.window
    parts = []
    block = 64 << 20
    with open(a.file, "rb") as f:
        pos = 0
        while pos < body:
            n = min(block, body - pos)
            f.seek(pos)
            buf = f.read(n + (w - 1 if pos + n < body else 0))
            parts.append(sampled(buf, w))
            pos += n
            print(f"\rобучение: {pos/1e9:.2f}/{body/1e9:.2f} ГБ", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        f.seek(size - a.tail)
        tail = f.read(a.tail)
    train = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)
    th = sampled(tail, w)
    if len(th) == 0:
        sys.exit("в хвосте нет окон в выборке (слишком короткий хвост)")
    idx = np.searchsorted(train, th)
    idx[idx >= len(train)] = len(train) - 1
    hit = train[idx] == th
    print(f"окон в выборке: обучение {len(train):,} уникальных, val {len(th):,}")
    print(f"доля окон val ({w} байт), найденных в обучении: {hit.mean()*100:.2f}%")
    # по блокам val: сколько «почти-дубликатов»
    k = max(1, len(th) // 16)
    print("по 16 частям хвоста:", " ".join(f"{hit[i*k:(i+1)*k].mean()*100:.0f}%" for i in range(16)))


if __name__ == "__main__":
    main()
