#!/usr/bin/env python3
"""Сколько дало бы смешивание нашей модели с ОБЫЧНОЙ счётной n-граммной моделью (без обучения, только подсчёт).

1) hashnet_proto dumpnll --ckpt МОДЕЛЬ --out nll.bin --val-windows 1024   (выгрузка вероятностей модели на val-окнах)
2) python3 tools/mix_ngram.py nll.bin ОБУЧАЮЩИЙ_ФАЙЛ [--train-mb 150] [--max-order 6]

n-граммная модель: интерполяция Виттена—Белла порядков 0..K по первым --train-mb МБ файла (хвост-val в них не входит).
Смесь вероятностей p = λ·p_модели + (1-λ)·p_nграмм считается по вероятности истинного байта, поэтому хватает одного числа на позицию.
Печатает: бит/байт модели, n-граммной модели по порядкам, смеси с одним λ и смеси с λ, зависящей от самого длинного найденного контекста.
"""
import argparse, os, struct, sys
import numpy as np

M = np.uint64(0x9E3779B97F4A7C15)
LOG2E = 1.0 / np.log(2.0)


def read_dump(path):
    with open(path, "rb") as f:
        if f.read(4) != b"NLLD":
            sys.exit("это не файл dumpnll")
        n, t = struct.unpack("<II", f.read(8))
        wins = np.frombuffer(f.read(n * (t + 1)), dtype=np.uint8).reshape(n, t + 1)
        nll = np.frombuffer(f.read(n * t * 4), dtype="<f4").reshape(n, t)
    return wins, nll


def ctx_hash(x, k, order_tag):
    """хэш k байт, предшествующих каждой позиции i>=k (массив длины len(x)-k)"""
    n = len(x) - k
    h = np.full(n, order_tag, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * M + (x[j:j + n].astype(np.uint64) + np.uint64(1))
        h ^= h >> np.uint64(31)
    return h


def pair_hash(c, nxt):
    with np.errstate(over="ignore"):
        p = (c ^ ((nxt.astype(np.uint64) + np.uint64(1)) * np.uint64(0xBF58476D1CE4E5B9)))
        p ^= p >> np.uint64(29)
        p *= M
    return p


def build(x, k):
    """таблицы для порядка k: ctx -> (n, d); (ctx,next) -> count"""
    c = ctx_hash(x, k, 1000 + k)
    nxt = x[k:]
    pr = pair_hash(c, nxt)
    uc, nc = np.unique(c, return_counts=True)
    up, first, cnt = np.unique(pr, return_index=True, return_counts=True)
    cu, dist = np.unique(c[first], return_counts=True)  # cu == uc (все контексты), dist = число разных продолжений
    assert len(cu) == len(uc)
    return uc, nc.astype(np.float64), dist.astype(np.float64), up, cnt.astype(np.float64)


def lookup(keys, table_keys, vals):
    idx = np.searchsorted(table_keys, keys)
    idx[idx >= len(table_keys)] = len(table_keys) - 1
    hit = table_keys[idx] == keys
    return hit, np.where(hit, vals[idx], 0.0), idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump"); ap.add_argument("train")
    ap.add_argument("--train-mb", type=int, default=150)
    ap.add_argument("--max-order", type=int, default=6)
    a = ap.parse_args()
    wins, nll = read_dump(a.dump)
    n, t = nll.shape
    with open(a.train, "rb") as f:
        x = np.frombuffer(f.read(a.train_mb << 20), dtype=np.uint8)
    print(f"окон {n}, позиций {n*t}; обучение для счёта: {len(x)/1e6:.0f} МБ", file=sys.stderr)
    tgt = wins[:, 1:t + 1].astype(np.uint8)          # (n,t)
    pm = np.exp(-nll.astype(np.float64))               # вероятность истинного байта у модели
    pos = np.tile(np.arange(t), (n, 1))                # позиция внутри окна (контекст — только внутри окна)
    flat_w = wins.reshape(-1)

    # порядок 0
    cnt0 = np.bincount(x, minlength=256).astype(np.float64)
    n0, d0 = cnt0.sum(), float((cnt0 > 0).sum())
    lam0 = n0 / (n0 + d0)
    p = lam0 * cnt0[tgt] / n0 + (1 - lam0) / 256.0
    best = np.zeros((n, t), dtype=np.int32)            # самый длинный порядок, у которого контекст встречался
    conf = np.zeros((n, t))                            # n контекста этого порядка
    per_order = {0: p.copy()}
    for k in range(1, a.max_order + 1):
        uc, nc, dist, up, cnt = build(x, k)
        # контекст k байт перед позицией p окна: wins[i, p-k+1 .. p] включительно (вход), следующий = tgt
        ok = pos >= k - 1
        h = np.zeros((n, t), dtype=np.uint64)
        with np.errstate(over="ignore"):
            hh = np.full((n, t), 1000 + k, dtype=np.uint64)
            for j in range(k):
                # байт j-го в контексте: wins[i, p-k+1+j]
                idx = np.clip(pos - k + 1 + j, 0, t)
                hh = hh * M + (np.take_along_axis(wins, idx.astype(np.int64), axis=1)[:, :t].astype(np.uint64) + np.uint64(1))
            hh ^= hh >> np.uint64(31)
        hit_c, nn, ic = lookup(hh.reshape(-1), uc, nc)
        dd = np.where(hit_c, dist[ic], 0.0)
        pk = pair_hash(hh.reshape(-1), tgt.reshape(-1))
        _, cc, _ = lookup(pk, up, cnt)
        hit_c = hit_c.reshape(n, t) & ok
        nn = nn.reshape(n, t); dd = dd.reshape(n, t); cc = cc.reshape(n, t)
        lam = np.where(hit_c, nn / np.maximum(nn + dd, 1e-9), 0.0)
        p = np.where(hit_c, lam * cc / np.maximum(nn, 1e-9) + (1 - lam) * p, p)
        best = np.where(hit_c, k, best)
        conf = np.where(hit_c, nn, conf)
        per_order[k] = p.copy()
        print(f"  порядок {k} готов (контекстов {len(uc):,})", file=sys.stderr)
        del uc, nc, dist, up, cnt
    bits = lambda pr: float(-np.log2(np.maximum(pr, 1e-12)).mean())
    print(f"\nмодель:                      {bits(pm):.3f} бит/байт")
    for k, pp in per_order.items():
        print(f"n-граммы до порядка {k}:        {bits(pp):.3f} бит/байт")
    png = per_order[a.max_order]
    grid = np.linspace(0.05, 0.99, 95)
    sc = [bits(l * pm + (1 - l) * png) for l in grid]
    lbest = grid[int(np.argmin(sc))]
    print(f"\nсмесь, один λ={lbest:.2f}:      {min(sc):.3f} бит/байт  (выигрыш {bits(pm)-min(sc):+.3f})")
    # λ по самому длинному найденному порядку и уверенности
    cb = np.digitize(conf, [1, 2, 4, 8, 16, 64])
    key = best * 10 + cb
    tot = 0.0
    mix = np.zeros((n, t))
    for kv in np.unique(key):
        m = key == kv
        s2 = [float(-np.log2(np.maximum(l * pm[m] + (1 - l) * png[m], 1e-12)).sum()) for l in grid]
        l = grid[int(np.argmin(s2))]
        mix[m] = l * pm[m] + (1 - l) * png[m]
    print(f"смесь, λ по порядку/уверенности: {bits(mix):.3f} бит/байт  (выигрыш {bits(pm)-bits(mix):+.3f})")
    print("\nКак читать: выигрыш > 0.02 бит/байт — внешняя «память» из счётчиков даёт то, чего модель не выучила; ≈0 — модель уже знает всё, что знают счётчики.")


if __name__ == "__main__":
    main()
