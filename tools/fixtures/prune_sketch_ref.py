#!/usr/bin/env python3
"""Independent float32 reference for the pruned GK quantile sketch (plan U10.2.2).

This reimplements XGBoost 3.1.2's GPU device-sketch cut-point algorithm from the
source (src/common/quantile.{h,cu}, hist_util.cu) in numpy, INDEPENDENTLY of the
cajeta port, so the two can be cross-checked bit-for-bit without an NVIDIA box.

IMPORTANT: this is an ALGORITHM self-check, NOT the GPU parity reference. Bit-exact
agreement with the real XGBoost 3.1.2 GPU sketch is validated separately against the
`large_reg` golden fixture generated on NVIDIA. This file locks cajeta's `Sketch.cuts`
to the algorithm as understood from the source; the fixture confirms the understanding.

`--selfcheck` writes tools/fixtures/sketch_selfcheck/{X.npy,cut_ptrs.npy,cut_values.npy}
for a small fixed dataset that exercises: no-prune, Stage-C prune, and Stage-A prune.
"""
import argparse, json, os
import numpy as np

KFACTOR = 8


def ceil_pos(v):
    t = int(v)
    return t + 1 if float(t) < v else t


def required_sample_cuts(num_rows, max_bin):
    eps = 1.0 / (KFACTOR * max_bin)          # double
    nlevel = 1
    while True:
        ls = ceil_pos(nlevel / eps) + 1
        ls = min(num_rows, ls)
        limit_size = ls
        if (1 << nlevel) * limit_size >= num_rows:
            break
        nlevel += 1
    return limit_size


def set_prune(rm, rx, wm, val, to):
    """WQSummary::SetPrune (quantile.h:220) on parallel float32 arrays -> new arrays."""
    size = len(val)
    if size <= to:
        return rm.copy(), rx.copy(), wm.copy(), val.copy()
    f32 = np.float32
    begin = rx[0]
    rng = f32(rm[size - 1] - rx[0])
    nn = to - 1
    orm = [rm[0]]; orx = [rx[0]]; owm = [wm[0]]; ov = [val[0]]
    i = 1
    lastidx = 0
    for k in range(1, nn):
        dx2 = f32(f32(2.0) * (f32(f32(f32(k) * rng) / f32(nn)) + begin))
        while i < size - 1 and dx2 >= f32(rx[i + 1] + rm[i + 1]):
            i += 1
        if i == size - 1:
            break
        rminnext_i = f32(rm[i] + wm[i])
        rmaxprev_i1 = f32(rx[i + 1] - wm[i + 1])
        if dx2 < f32(rminnext_i + rmaxprev_i1):
            if i != lastidx:
                orm.append(rm[i]); orx.append(rx[i]); owm.append(wm[i]); ov.append(val[i]); lastidx = i
        else:
            if i + 1 != lastidx:
                orm.append(rm[i + 1]); orx.append(rx[i + 1]); owm.append(wm[i + 1]); ov.append(val[i + 1]); lastidx = i + 1
    if lastidx != size - 1:
        orm.append(rm[size - 1]); orx.append(rx[size - 1]); owm.append(wm[size - 1]); ov.append(val[size - 1])
    return (np.array(orm, np.float32), np.array(orx, np.float32),
            np.array(owm, np.float32), np.array(ov, np.float32))


def sentinel(v, sign):
    # (float)( (double)v +/- ( (double)fabsf(v) + 1e-5 ) )   -- 1e-5 is a double literal
    v = np.float32(v)
    return np.float32(np.float64(v) + sign * (np.float64(abs(v)) + 1e-5))


def cuts(X, max_bin):
    n, nf = X.shape
    cut_ptrs = [0]
    cut_values = []
    for f in range(nf):
        col = X[:, f].astype(np.float32)
        col = np.sort(col[~np.isnan(col)])
        if col.size == 0:
            cut_ptrs.append(len(cut_values))
            continue
        vals, counts = np.unique(col, return_counts=True)     # ascending distinct + counts
        cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.float32)
        rm = cum[:-1].copy(); rx = cum[1:].copy(); wm = counts.astype(np.float32); dv = vals.astype(np.float32)
        # Stage A
        ncpf = required_sample_cuts(n, max_bin)
        to_a = min(ncpf, col.size)
        rm, rx, wm, dv = set_prune(rm, rx, wm, dv, to_a)
        # Stage C
        rm, rx, wm, dv = set_prune(rm, rx, wm, dv, max_bin + 1)
        s = len(dv)
        k = min(s, max_bin)
        cut_values.append(sentinel(dv[0], -1.0))              # min sentinel
        for e in range(1, k):
            cut_values.append(np.float32(dv[e]))              # S[1..k-1]
        cut_values.append(sentinel(dv[s - 1], +1.0))          # max sentinel (back = global max)
        cut_ptrs.append(len(cut_values))
    return np.array(cut_ptrs, np.int64), np.array(cut_values, np.float32)


def build_selfcheck_dataset():
    """3 features exercising the three paths, at max_bin = 8:
       f0: 6 distinct  -> no prune (<= 8).
       f1: 40 distinct, 40 rows -> Stage C prune only (40 > 9, but 40 <= ncpf).
       f2: 300 rows, 300 distinct -> Stage A active (300 > requiredSampleCuts).
    Padded to 300 rows; short features repeat values (counts > 1) which the summary
    handles. A couple of NaNs check the missing path."""
    rng = np.random.default_rng(7)
    n = 300
    X = np.empty((n, 3), np.float64)
    X[:, 0] = rng.integers(0, 6, size=n).astype(np.float64) * 1.5 - 2.0     # 6 distinct
    X[:, 1] = (rng.integers(0, 40, size=n).astype(np.float64) - 20.0) * 0.25  # 40 distinct
    X[:, 2] = rng.standard_normal(n)                                        # ~300 distinct
    X[0, 0] = np.nan
    X[1, 2] = np.nan
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--max-bin", type=int, default=8)
    args = ap.parse_args()
    if not args.selfcheck:
        raise SystemExit("pass --selfcheck to write the self-check fixture")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "sketch_selfcheck")
    os.makedirs(out, exist_ok=True)
    X = build_selfcheck_dataset()
    cp, cv = cuts(X, args.max_bin)
    np.save(os.path.join(out, "X.npy"), X)
    np.save(os.path.join(out, "cut_ptrs.npy"), cp)
    # stored float64 (exact widening of the float32 result), matching the golden
    # fixture format so FixtureLoader.cutValues() (loadF64) reads it and cajeta
    # compares at float64 — bit-exact since each value is an exact float32.
    np.save(os.path.join(out, "cut_values.npy"), cv.astype(np.float64))
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump({
            "note": "ALGORITHM self-check for the pruned GK sketch (plan U10.2.2) — "
                    "NOT a GPU parity reference. Locks cajeta Sketch.cuts to the "
                    "float32 algorithm from xgboost-ref; GPU 3.1.2 parity is validated "
                    "against large_reg on NVIDIA.",
            "max_bin": args.max_bin,
            "rows": int(X.shape[0]), "features": int(X.shape[1]),
            "cuts_per_feature": np.diff(cp).tolist(),
        }, fh, indent=2)
    print(f"wrote self-check -> {out}")
    print(f"cuts/feature: {np.diff(cp).tolist()}  (total {cv.size})")


if __name__ == "__main__":
    main()
