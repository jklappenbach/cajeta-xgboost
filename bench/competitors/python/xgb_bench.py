#!/usr/bin/env python3
"""The competitor side of the dev.cajeta.xgboost benchmarks: upstream XGBoost,
emitting the same ids, variants and schema-v2 columns as the Cajeta harness.

    xgb_bench.py --data <dir> --out <python.csv> [--bench <name>] [--list]

  ml/xgb-hist-fit  QuantileDMatrix + train, tree_method hist, device cpu: the
                   end-to-end time xgbbench's bench.py measured, raw matrix to
                   model (Cajeta's GBDT.fit sketches and bins inside too)
  ml/xgb-gpu-fit   the same with device cuda (gpu_hist); skipped under
                   CAJETA_BENCH_NO_GPU=1 without any CUDA call
  ml/xgb-predict   inplace_predict margins of xgb/model.json over all of X

Checks are the Cajeta rows' checks with the same tolerances (TOL).
"""
import argparse
import os
import sys

import numpy as np
import xgboost as xgb

import benchcsv as bc
from gen_data import logloss, params

# Identical to bench/src/main/cajeta/bench/xgb/Tol.cajeta.
TOL = {
    "logloss": 0.01,      # |logloss / logloss_ref - 1|
    "margin": 1e-5,       # max |margin - margin_ref|
}
WT = {"xgb-hist-fit": (1, 3), "xgb-gpu-fit": (1, 3), "xgb-predict": (2, 10)}
NAMES = ["xgb-hist-fit", "xgb-gpu-fit", "xgb-predict"]
VARIANTS = {"xgb-hist-fit": "cpu", "xgb-gpu-fit": "device", "xgb-predict": "cpu"}


def meta(name):
    return dict(benchmark=name, area="ml", variant=VARIANTS[name], library="xgboost",
                library_version=xgb.__version__)


def run(name, d):
    w, t = bc.warmup_trials(*WT[name])
    x = np.load(os.path.join(d, "xgb/X.npy"))
    y = np.load(os.path.join(d, "xgb/y.npy"))
    rounds, depth, max_bin = (int(v) for v in np.load(os.path.join(d, "xgb/config.npy")))
    ds = f"xgb binary {x.shape[0]}x{x.shape[1]} 5% NaN seed 20261025"
    mdl = f"gbdt binary:logistic {rounds} rounds depth {depth} max_bin {max_bin}"
    m = meta(name)

    if name in ("xgb-hist-fit", "xgb-gpu-fit"):
        device = "cpu"
        if name == "xgb-gpu-fit":
            if bc.no_gpu():
                return bc.skipped_row(bc.GPU_HELD, backend="gpu",
                                      platform=f"xgboost {xgb.__version__} gpu_hist", model=mdl,
                                      dataset_name=ds, **m)
            device = "cuda"
        state = {}

        def fit():
            dm = xgb.QuantileDMatrix(x, label=y, missing=np.nan, max_bin=max_bin)
            state["bst"] = xgb.train(params(device=device), dm, num_boost_round=rounds)
            state["bst"].inplace_predict(x[:1])

        samples = bc.measure(fit, w, t)
        ref = float(np.load(os.path.join(d, "ref/hist-logloss.npy"))[0])
        got = logloss(y, state["bst"].inplace_predict(x))
        rel = abs(got / ref - 1.0)
        note = (f"QuantileDMatrix + train, {'gpu_hist' if device == 'cuda' else 'hist'} on "
                f"{device}; train logloss {got:.8g} vs upstream hist {ref:.8g}, relative |d| "
                f"{rel:.3g} <= {TOL['logloss']:g}; nthread default ({os.cpu_count()} cpus)")
        return bc.measured_row(samples, w, input_size=x.shape[0], unit="rows/s", check_ok=rel <= TOL["logloss"],
                               note=note, backend="cuda" if device == "cuda" else "cpu",
                               platform=f"xgboost {xgb.__version__} "
                                        f"{'gpu_hist' if device == 'cuda' else 'hist'}",
                               model=mdl, dataset_name=ds, **m)

    if name == "xgb-predict":
        bst = xgb.Booster()
        bst.load_model(os.path.join(d, "xgb/model.json"))
        bst.set_param({"device": "cpu"})
        state = {}

        def predict():
            state["m"] = bst.inplace_predict(x, predict_type="margin")

        samples = bc.measure(predict, w, t)
        ref = np.load(os.path.join(d, "ref/predict-margin.npy"))
        diff = float(np.max(np.abs(state["m"].astype(np.float32) - ref)))
        note = (f"inplace_predict margins of xgb/model.json; max |margin - ref| {diff:.3g} <= "
                f"{TOL['margin']:g}")
        return bc.measured_row(samples, w, input_size=x.shape[0], unit="rows/s",
                               check_ok=diff <= TOL["margin"], note=note,
                               platform=f"xgboost {xgb.__version__} cpu", model=mdl,
                               dataset_name=ds, **m)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="python.csv")
    ap.add_argument("--bench")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = [n for n in NAMES if a.bench in (None, n)]
    if a.list:
        for n in names:
            print(f"  ml/{n}  [{VARIANTS[n]}]")
        return 0
    if not a.data:
        ap.error("--data is required with a run")
    rows, bad = [], 0
    for n in names:
        try:
            r = run(n, a.data)
        except Exception as e:  # a failing benchmark never stops the run
            r = bc.failed_row(f"{type(e).__name__}: {e}", **meta(n))
        print(f"  {r['status']}  ml/{n} [python xgboost] variant='{r['variant']}'  "
              f"median_ns={r['median_ns']}  {r['notes']}", flush=True)
        bad += r["status"] in ("failed", "invalid")
        rows.append(r)
    bc.write(a.out, rows)
    print(f"xgb_bench.py: wrote {a.out} ({len(rows)} rows)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
