#!/usr/bin/env python3
"""Сравнение val (бит/байт) нескольких запусков по их логам (workspace/logs/*.log или вывод терминала).

  python3 tools/compare_runs.py workspace/logs/A.log workspace/logs/B.log [--name A --name B]

Печатает таблицу: val ДО фазы, на контрольных шагах, ПОСЛЕ фазы — по каждому логу в своём столбце.
Если в логе несколько сегментов данных, берётся среднее по ним.
"""
import argparse, os, re, sys

def parse(path):
    pts = {}          # метка -> список val по сегментам
    order = []
    label = None
    title = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if title is None and line.startswith("$ "):
            m = re.search(r"--ckpt-dir\s+'?([^'\s]+(?: [^'\s-][^'\s]*)*)'?", line)
            title = os.path.basename(m.group(1).rstrip("/")) if m else None
        if "валидация" in line and "──" in line:
            if "ДО фазы" in line: label = "до"
            elif "ПОСЛЕ фазы" in line: label = "после"
            else:
                m = re.search(r"шаг\s+(\d+)", line)
                label = int(m.group(1)) if m else None
            if label is not None and label not in pts:
                pts[label] = []; order.append(label)
            continue
        if label is not None:
            m = re.search(r"=\s*([\d.]+)\s*бит/байт", line)
            if m and "val" in line:
                pts[label].append(float(m.group(1)))
            elif line.strip() and not line.startswith(" ") and "val" not in line:
                label = None
    res = {k: sum(v) / len(v) for k, v in pts.items() if v}
    return title, res

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--name", action="append", default=[])
a = ap.parse_args()
runs = []
for i, p in enumerate(a.logs):
    if not os.path.isfile(p): sys.exit(f"нет файла: {p}")
    t, r = parse(p)
    runs.append((a.name[i] if i < len(a.name) else (t or os.path.basename(p)), r))
keys = set()
for _, r in runs: keys |= set(r)
steps = sorted(k for k in keys if isinstance(k, int))
rows = (["до"] if "до" in keys else []) + steps + (["после"] if "после" in keys else [])
w = max(12, *(len(n) for n, _ in runs)) + 2
print(f"{'val, бит/байт':<16}" + "".join(f"{n:>{w}}" for n, _ in runs))
for k in rows:
    lab = {"до": "до фазы", "после": "после фазы"}.get(k, f"шаг {k}")
    print(f"{lab:<16}" + "".join(f"{r[k]:>{w}.3f}" if k in r else f"{'—':>{w}}" for _, r in runs))
if any("до" in r and "после" in r for _, r in runs):
    print()
    for n, r in runs:
        if "до" in r and "после" in r:
            print(f"{n}: Δ за фазу = {r['после'] - r['до']:+.3f} бит/байт")
