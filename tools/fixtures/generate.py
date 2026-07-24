#!/usr/bin/env python3
"""Golden-fixture generator for cajeta-xgboost parity (plan U2).

Trains reference XGBoost under a PINNED, DETERMINISTIC config and dumps the
ground truth cajeta must reproduce bit-for-bit:

  manifest.json  — xgboost version + the exact params + dataset spec
  X.npy, y.npy   — the training matrix (float64) and labels
  model.json     — xgboost's native model dump (every tree / split / weight)
  margins.npy    — per-round raw output margin on the holdout (shape [rounds, n])
  preds.npy      — final holdout prediction (post inverse-link)

The PARITY REFERENCE is the NVIDIA GPU algorithm: pass `--device cuda` on an
NVIDIA box. `--device cpu` (the default here) produces a format-development /
loader fixture ONLY — XGBoost's CPU `hist` does NOT bit-match its GPU `hist`, so
a cpu fixture is not the parity ground truth (it exercises the format + the
cajeta loader, nothing more). The committed reference fixtures are cuda.

Usage:
  python3 generate.py --out fixtures/tiny_reg --objective reg:squarederror \
      --rows 200 --features 5 --device cpu
"""
import argparse, json, os, sys
import numpy as np


def build_dataset(rows, features, seed, missing_frac, n_class):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, features)).astype(np.float64)
    if n_class >= 2:
        # a separable-ish signal for classification
        w = rng.standard_normal(features)
        logits = X @ w
        if n_class == 2:
            y = (logits > np.median(logits)).astype(np.float64)
        else:
            # bucket the score into n_class ordered classes
            q = np.quantile(logits, np.linspace(0, 1, n_class + 1)[1:-1])
            y = np.digitize(logits, q).astype(np.float64)
    else:
        w = rng.standard_normal(features)
        y = (X @ w + 0.1 * rng.standard_normal(rows)).astype(np.float64)
    if missing_frac > 0:
        mask = rng.random((rows, features)) < missing_frac
        X[mask] = np.nan
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", default="reg:squarederror")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--features", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--missing-frac", type=float, default=0.0)
    ap.add_argument("--num-class", type=int, default=0,
                    help="0=regression, 2=binary, >2=multiclass")
    ap.add_argument("--num-round", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--base-score", type=float, default=0.5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda = the parity REFERENCE (NVIDIA); cpu = format/loader dev only")
    args = ap.parse_args()

    import xgboost as xgb

    n_class = args.num_class
    X, y = build_dataset(args.rows, args.features, args.seed,
                         args.missing_frac, n_class)

    # The pinned DETERMINISTIC config (spec §6.4).
    params = {
        "objective": args.objective,
        "tree_method": "hist",
        "device": args.device,
        "seed": args.seed,
        "max_depth": args.max_depth,
        "max_bin": args.max_bin,
        "eta": args.eta,
        "reg_lambda": args.reg_lambda,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "base_score": args.base_score,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "nthread": 1,
    }
    if n_class > 2:
        params["num_class"] = n_class

    dtrain = xgb.DMatrix(X, label=y, missing=np.nan)
    booster = xgb.train(params, dtrain, num_boost_round=args.num_round)

    # Per-round margins (holdout == train here; parity is on the same rows).
    rounds = args.num_round
    margins = np.stack([
        booster.predict(dtrain, iteration_range=(0, r + 1), output_margin=True)
        for r in range(rounds)
    ]).astype(np.float64)
    preds = booster.predict(dtrain).astype(np.float64)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "X.npy"), X)
    np.save(os.path.join(args.out, "y.npy"), y)
    np.save(os.path.join(args.out, "margins.npy"), margins)
    np.save(os.path.join(args.out, "preds.npy"), preds)
    booster.save_model(os.path.join(args.out, "model.json"))

    manifest = {
        "xgboost_version": xgb.__version__,
        "is_reference": args.device == "cuda",
        "note": ("NVIDIA GPU reference" if args.device == "cuda"
                 else "CPU format/loader fixture — NOT the parity reference"),
        "params": params,
        "dataset": {
            "rows": args.rows, "features": args.features, "seed": args.seed,
            "missing_frac": args.missing_frac, "num_class": n_class,
        },
        "num_round": rounds,
        "artifacts": ["X.npy", "y.npy", "margins.npy", "preds.npy", "model.json"],
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote fixture -> {args.out}  (xgboost {xgb.__version__}, "
          f"device={args.device}, rounds={rounds})")
    # TODO (U3/U9): also dump XGBoost's internal bin edges + per-instance bin
    # indices + per-round grad/hess, for the binned-boundary (§7.1) histogram-level
    # parity. Requires QuantileDMatrix cut extraction / a custom objective hook.


if __name__ == "__main__":
    main()
