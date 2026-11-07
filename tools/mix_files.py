#!/usr/bin/env python3
"""Соединение нескольких файлов в один обучающий файл (куски по 1 МБ перемешиваются равномерно).

  python3 mix_files.py --out mix.txt a.txt@3 b.txt@1 c.txt
  python3 mix_files.py --out mix.txt --target-gb 2 rust.txt@7 prose_ru.txt@2 chat_ru.txt@1

Вес (@N):
  • без --target-gb: сколько раз взять файл целиком (0.5 — половина, равномерно по файлу; 2 — дважды);
  • с --target-gb:   доля от итогового размера (веса 7:2:1 → 70% / 20% / 10%); если файл меньше доли — повторяется.
Один «готовый» файл — это один вызов; для нескольких маленьких моделей вызовите несколько раз с разными наборами.
"""
import argparse, os, sys, time

def parse(spec):
    p, _, w = spec.rpartition("@")
    if p and os.path.exists(p) and not os.path.exists(spec):
        try: return p, float(w)
        except ValueError: pass
    return spec, 1.0

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--target-gb", type=float, default=0)
ap.add_argument("--chunk-mb", type=float, default=1.0)
a = ap.parse_args()
chunk = int(a.chunk_mb * 1e6)
items = []
for s in a.files:
    p, w = parse(s)
    if not os.path.isfile(p): sys.exit(f"нет файла: {p}")
    if w <= 0: sys.exit(f"вес должен быть > 0: {s}")
    items.append((p, w, os.path.getsize(p)))
if os.path.exists(a.out): sys.exit(f"файл уже существует: {a.out} (удалите или выберите другое имя)")
if a.target_gb:
    tw = sum(w for _, w, _ in items)
    need = [max(1, round(a.target_gb * 1e9 * w / tw / chunk)) for _, w, _ in items]
else:
    need = [max(1, round(sz * w / chunk)) for _, w, sz in items]
plan = []   # для каждого файла: список индексов кусков
for (p, w, sz), n in zip(items, need):
    tot = max(sz // chunk, 1)
    plan.append([ (i * tot) // n % tot if n <= tot else i % tot for i in range(n)])
done = [0] * len(items)
total_need = sum(need); written = 0; t0 = time.time()
fs = [open(p, "rb") for p, _, _ in items]
try:
    with open(a.out, "wb") as o:
        for k in range(total_need):
            j = min((i for i in range(len(items)) if done[i] < need[i]), key=lambda i: done[i] / need[i])
            f = fs[j]; sz = items[j][2]
            start = plan[j][done[j]] * chunk; done[j] += 1
            f.seek(start)
            if start > 0: f.readline()
            pos = f.tell()
            if pos >= sz: continue
            data = f.read(max(start + chunk - pos, 0)) + f.readline()
            if data and not data.endswith(b"\n"): data += b"\n"
            o.write(data); written += len(data)
            if k % 100 == 0:
                print(f"\r{k+1}/{total_need} кусков, {written/1e9:.2f} ГБ", end="", file=sys.stderr, flush=True)
finally:
    for f in fs: f.close()
print(f"\nготово: {a.out}  {written/1e9:.3f} ГБ, из {len(items)} файл(ов), {time.time()-t0:.0f} с")
for (p, w, sz), n in zip(items, need):
    print(f"  {os.path.basename(p)}: вес {w:g}, кусков {n} (~{n*chunk/1e9:.2f} ГБ)")
