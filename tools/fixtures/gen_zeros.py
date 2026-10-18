#!/usr/bin/env python3
"""Regenerate the zeros fixture: columns holding both -0.0 and +0.0, with REAL xgboost 3.1.2
QuantileDMatrix cuts at several max_bin. Which zero leads an equal run follows std::sort, so
these pin Sketch's mixed-zero path (xgboost-data-prep Unit 3).

Usage: python3 gen_zeros.py   (needs xgboost==3.1.2; writes ./zeros/)
"""
import json, os
import numpy as np
import xgboost as xgb

N = 50000
BINS = (256, 64, 16)


def signed_zeros(rng, k):
    z = np.zeros(k)
    z[rng.random(k) < 0.5] = -0.0
    return z


def columns(rng):
    n = N
    u = rng.uniform(-100, 100, n); m = rng.random(n) < 2 / 9; u[m] = signed_zeros(rng, m.sum()); yield u
    u = rng.uniform(-5, 5, n); m = rng.random(n) < 0.9; u[m] = signed_zeros(rng, m.sum()); yield u
    yield signed_zeros(rng, n)
    u = np.zeros(n); u[: n // 2] = -0.0; u[n // 2 :] = rng.uniform(-1, 1, n - n // 2); u[n // 2 : n // 2 + 1000] = 0.0; yield u
    u = np.zeros(n); u[n // 2 :] = -0.0; u[:1000] = rng.uniform(-1, 1, 1000); yield u
    u = rng.uniform(-100, 100, n); m = rng.random(n) < 0.001; u[m] = signed_zeros(rng, m.sum()); yield u
    u = rng.uniform(-3, 3, n); m = rng.random(n) < 0.3; u[m] = signed_zeros(rng, m.sum()); u[rng.random(n) < 0.1] = np.nan; yield u
    u = (rng.integers(0, 7, n) - 3).astype(np.float64); z = u == 0; u[z] = signed_zeros(rng, z.sum()); yield u
    for s in range(8):
        u = rng.normal(0, 1 + s, n); m = rng.random(n) < (0.05 + 0.1 * s); u[m] = signed_zeros(rng, m.sum()); yield u


def main():
    if xgb.__version__ != "3.1.2":
        raise SystemExit(f"need xgboost==3.1.2 (the pinned reference), got {xgb.__version__}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zeros")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1234)
    X = np.stack(list(columns(rng)), axis=1)
    np.save(os.path.join(out, "X.npy"), X)
    zero_cuts = {}
    for mb in BINS:
        q = xgb.QuantileDMatrix(X, missing=np.nan, max_bin=mb)
        cp, cv = q.get_quantile_cut()
        np.save(os.path.join(out, f"cut_ptrs_{mb}.npy"), cp.astype(np.int64))
        np.save(os.path.join(out, f"cut_values_{mb}.npy"), cv.astype(np.float64))
        zs = []
        for f in range(X.shape[1]):
            c = cv[cp[f] : cp[f + 1]].astype(np.float32)
            zs.append("".join("-" if np.signbit(v) else "+" for v in c if v == 0))
        zero_cuts[str(mb)] = zs
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"xgboost": xgb.__version__, "generator": "gen_zeros.py", "rows": N,
                   "features": X.shape[1], "max_bins": list(BINS), "zero_cut_signs": zero_cuts}, f, indent=1)
        f.write("\n")
    print(json.dumps(zero_cuts))


if __name__ == "__main__":
    main()
