#!/usr/bin/env python3
"""Нарезка большого файла на N НЕПЕРЕСЕКАЮЩИХСЯ кусков (для обучения нескольких маленьких моделей на разных данных).

  python3 make_shards.py big.txt --out shards --n 6 [--max-gb-each 0.5] [--mode chunks|contiguous] [--chunk-mb 8]

mode=chunks (по умолчанию): файл режется на блоки по --chunk-mb МБ; куски берутся из блоков, РАВНОМЕРНО по всему файлу,
   блоки разных кусков не пересекаются, поэтому у всех кусков одинаковое «качество», но разное содержимое.
mode=contiguous: кусок k — подряд идущая область файла (специалисты по разным частям корпуса).
--max-gb-each ограничивает размер каждого куска (0 — без ограничения: весь файл поровну).
Границы выровнены по строкам. Создаёт shard_1.txt … shard_N.txt и печатает размеры.
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--max-gb-each", type=float, default=0.0)
ap.add_argument("--mode", choices=["chunks", "contiguous"], default="chunks")
ap.add_argument("--chunk-mb", type=float, default=8.0)
a = ap.parse_args()
if a.n < 2: sys.exit("--n должно быть >= 2")
if not os.path.isfile(a.src): sys.exit(f"нет файла: {a.src}")
size = os.path.getsize(a.src)
chunk = max(1 << 16, int(a.chunk_mb * 1e6))
os.makedirs(a.out, exist_ok=True)
paths = [os.path.join(a.out, f"shard_{k+1}.txt") for k in range(a.n)]
for p in paths:
    if os.path.exists(p): sys.exit(f"уже существует: {p} (удалите или выберите другую папку)")
cap = int(a.max_gb_each * 1e9)
f = open(a.src, "rb")
outs = [open(p, "wb") for p in paths]
written = [0] * a.n
t0 = time.time()

def read_block(start, end):
    """строки, НАЧИНАЮЩИЕСЯ в [start,end); последняя дочитывается до конца строки"""
    f.seek(start)
    if start > 0:
        f.seek(start - 1)
        f.readline()                       # пропуск строки, начавшейся в предыдущем блоке
        if f.tell() > end: return b""
    pos = f.tell()
    if pos >= size: return b""
    data = f.read(max(end - pos, 0)) + f.readline()
    return data if data.endswith(b"\n") else data + b"\n"

if a.mode == "chunks":
    B = max(size // chunk, 1)
    per = B // a.n
    if per < 1: sys.exit("файл слишком мал для такого числа кусков")
    M = per if not cap else min(per, max(cap // chunk, 1))
    for m in range(M):
        base = ((m * B) // M) // a.n * a.n
        for k in range(a.n):
            j = base + k
            data = read_block(j * chunk, (j + 1) * chunk)
            outs[k].write(data); written[k] += len(data)
        if m % 20 == 0:
            print(f"\r{m+1}/{M} блоков на кусок", end="", file=sys.stderr, flush=True)
else:
    S = size // a.n
    if cap: S = min(S, cap)
    for k in range(a.n):
        start = k * (size // a.n); end = start + S
        pos = start
        while pos < end:
            e = min(pos + chunk, end)
            data = read_block(pos, e)
            outs[k].write(data); written[k] += len(data)
            pos = e
        print(f"\rкусок {k+1}/{a.n}", end="", file=sys.stderr, flush=True)
for o in outs: o.close()
f.close()
print(f"\nготово за {time.time()-t0:.0f} с, режим {a.mode}")
for p, w in zip(paths, written):
    print(f"  {p}: {w/1e9:.3f} ГБ")
