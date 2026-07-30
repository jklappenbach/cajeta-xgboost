#!/usr/bin/env python3
"""Regenerate the sketch_selfcheck fixture from REAL xgboost 3.1.2.

gpu-numeric-fidelity U3 established that `QuantileDMatrix(numpy)` sketches on
the CPU HostSketchContainer during construction — that is where every golden
fixture's cut_values come from (pip 3.1.2 CPU reproduces them bit-for-bit) —
so the expected cuts here are the REAL `get_quantile_cut()` output, not a
reimplementation. cajeta's `Sketch.cuts` ports the CPU `WQuantileSketch` GK
queue/level pipeline (quantile.h) and is locked to this fixture by
`SketchPruneTest`, and to the NVIDIA-box golden `large_reg` fixture by
`LargeSketchTest` (both bit-exact).

Requires xgboost==3.1.2 (the pinned reference version):

    python prune_sketch_ref.py --selfcheck
"""
import argparse
import json
import os

import numpy as np
import xgboost as xgb


def build_selfcheck_dataset():
    """3 features exercising the paths at max_bin = 8:
       f0: 6 distinct  -> no prune (<= 8).
       f1: 40 distinct, 300 rows -> final prune only.
       f2: ~300 distinct -> queue/level machinery + both prunes.
    A couple of NaNs check the missing path."""
    rng = np.random.default_rng(7)
    n = 300
    X = np.empty((n, 3), np.float64)
    X[:, 0] = rng.integers(0, 6, size=n).astype(np.float64) * 1.5 - 2.0
    X[:, 1] = (rng.integers(0, 40, size=n).astype(np.float64) - 20.0) * 0.25
    X[:, 2] = rng.standard_normal(n)
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
    if xgb.__version__ != "3.1.2":
        raise SystemExit(f"need xgboost==3.1.2 (the pinned reference), got {xgb.__version__}")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "sketch_selfcheck")
    os.makedirs(out, exist_ok=True)
    X = build_selfcheck_dataset()
    d = xgb.QuantileDMatrix(X, missing=np.nan, max_bin=args.max_bin)
    cp, cv = d.get_quantile_cut()
    cp = np.asarray(cp, np.int64)
    cv = np.asarray(cv, np.float32)
    np.save(os.path.join(out, "X.npy"), X)
    np.save(os.path.join(out, "cut_ptrs.npy"), cp)
    np.save(os.path.join(out, "cut_values.npy"), cv.astype(np.float64))
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump({
            "note": "REAL xgboost 3.1.2 CPU QuantileDMatrix cuts (get_quantile_cut) — "
                    "the authoritative reference; QuantileDMatrix(numpy) sketches on "
                    "the CPU HostSketchContainer, which is what ALL fixture cuts come from.",
            "xgboost_version": xgb.__version__,
            "max_bin": args.max_bin,
            "rows": int(X.shape[0]), "features": int(X.shape[1]),
            "cuts_per_feature": np.diff(cp).tolist(),
        }, fh, indent=2)
    print(f"wrote self-check -> {out}")
    print(f"cuts/feature: {np.diff(cp).tolist()}  (total {cv.size})")


if __name__ == "__main__":
    main()
