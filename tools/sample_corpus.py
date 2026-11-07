#!/usr/bin/env python3
"""Выборка ровно ~N ГБ из большого текстового файла: равномерно по всему файлу кусками по --chunk-mb МБ
(границы выровнены по строкам). Пример:
  python3 sample_corpus.py split_out/rust.txt split_out/rust_2gb.txt --gb 2
"""
import argparse, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--gb", type=float, default=2.0)
ap.add_argument("--chunk-mb", type=float, default=1.0)
a = ap.parse_args()
size = os.path.getsize(a.src)
chunk = int(a.chunk_mb * 1e6)
total = max(size // chunk, 1)
want = min(total, max(int(a.gb * 1e9 // chunk), 1))
t0 = time.time(); written = 0; kept = 0
with open(a.src, "rb") as f, open(a.dst, "wb") as o:
    for k in range(total):
        # равномерный выбор want из total кусков
        if (k * want) // total == ((k - 1) * want) // total and k > 0:
            continue
        start = k * chunk
        f.seek(start)
        if start > 0:
            f.readline()                # выровнять начало по строке
        pos = f.tell()
        end = start + chunk
        if pos < end:
            data = f.read(end - pos) + f.readline()
        else:
            continue
        o.write(data); written += len(data); kept += 1
        if kept % 200 == 0:
            print(f"\r{written/1e9:5.2f} ГБ из {a.gb} ГБ", end="", file=sys.stderr, flush=True)
print(f"\nготово: {a.dst}  {written/1e9:.3f} ГБ, кусков {kept} из {total}, {time.time()-t0:.0f} с")
