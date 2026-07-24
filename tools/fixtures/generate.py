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


def compute_grad_hess(objective, in_margins, y, n_class):
    """Per-round (grad, hess) reconstructed from the objective + the margin going
    INTO each round. NOTE: XGBoost carries grad/hess as float32 GradientPair;
    this reconstruction is float64 and its last-bit agreement with XGBoost's
    internal pairs is nailed (or captured via a custom objective) in U5. eps
    values follow XGBoost."""
    if objective == "reg:squarederror":
        grad = in_margins - y                      # [rounds, n]
        hess = np.ones_like(in_margins)
    elif objective == "binary:logistic":
        p = 1.0 / (1.0 + np.exp(-in_margins))
        grad = p - y
        hess = np.maximum(p * (1.0 - p), 1e-16)
    elif objective in ("multi:softprob", "multi:softmax"):
        m = in_margins - in_margins.max(axis=2, keepdims=True)   # [rounds, n, K]
        e = np.exp(m); p = e / e.sum(axis=2, keepdims=True)
        onehot = np.zeros_like(p)
        idx = y.astype(np.int64)
        for r in range(p.shape[0]):
            onehot[r, np.arange(p.shape[1]), idx] = 1.0
        grad = p - onehot
        hess = np.maximum(2.0 * p * (1.0 - p), 1e-16)
    else:
        raise SystemExit(f"grad/hess not implemented for objective {objective}")
    return grad.astype(np.float64), hess.astype(np.float64)


def bin_indices(X, cut_ptrs, cut_vals):
    """Per-instance bin index per feature, derived from XGBoost's cut edges.
    The cuts are the ground truth; this assignment rule (and NaN→missing) is
    validated/corrected against XGBoost's GHistIndex in U9. Missing → -1."""
    n, f = X.shape
    out = np.full((n, f), -1, dtype=np.int32)
    for j in range(f):
        cuts = cut_vals[cut_ptrs[j]:cut_ptrs[j + 1]]
        col = X[:, j]
        present = ~np.isnan(col)
        # bin = index of the first cut strictly greater than the value, minus the
        # leading -inf sentinel (upper_bound convention).
        b = np.searchsorted(cuts, col[present], side="right") - 1
        out[present, j] = np.clip(b, 0, len(cuts) - 1).astype(np.int32)
    return out


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

    # QuantileDMatrix is the hist/GPU-native matrix and exposes the cut points
    # (the exact bin edges the reference used) — the binned-boundary ground truth.
    dtrain = xgb.QuantileDMatrix(X, label=y, missing=np.nan, max_bin=args.max_bin)
    booster = xgb.train(params, dtrain, num_boost_round=args.num_round)

    rounds = args.num_round
    # The margin GOING INTO each round. Round 0's input is the base margin — got
    # EMPIRICALLY from a zero-round booster (objective-agnostic; correct for
    # multiclass per-class base_score too). `predict(iteration_range=(0,0))` is
    # NOT the base — end=0 means "all trees".
    base_booster = xgb.train(params, dtrain, num_boost_round=0)
    base_margin = base_booster.predict(dtrain, output_margin=True).astype(np.float64)
    cum = [booster.predict(dtrain, iteration_range=(0, r + 1), output_margin=True)
           for r in range(rounds)]
    margins = np.stack(cum).astype(np.float64)
    in_margins = np.stack([base_margin] + cum[:-1]).astype(np.float64)
    grad, hess = compute_grad_hess(args.objective, in_margins, y, n_class)
    preds = booster.predict(dtrain).astype(np.float64)

    # XGBoost's exact bin edges (CSR: per-feature slices of `values` via `ptrs`),
    # plus per-instance bin indices computed from those cuts (rule validated in U9).
    cut_ptrs, cut_vals = dtrain.get_quantile_cut()
    cut_ptrs = np.asarray(cut_ptrs, dtype=np.int64)
    cut_vals = np.asarray(cut_vals, dtype=np.float64)
    bins = bin_indices(X, cut_ptrs, cut_vals)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "X.npy"), X)
    np.save(os.path.join(args.out, "y.npy"), y)
    np.save(os.path.join(args.out, "margins.npy"), margins)
    np.save(os.path.join(args.out, "preds.npy"), preds)
    np.save(os.path.join(args.out, "grad.npy"), grad)
    np.save(os.path.join(args.out, "hess.npy"), hess)
    np.save(os.path.join(args.out, "cut_ptrs.npy"), cut_ptrs)
    np.save(os.path.join(args.out, "cut_values.npy"), cut_vals)
    np.save(os.path.join(args.out, "bins.npy"), bins)
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
        "artifacts": ["X.npy", "y.npy", "margins.npy", "preds.npy", "grad.npy",
                      "hess.npy", "cut_ptrs.npy", "cut_values.npy", "bins.npy",
                      "model.json"],
        "grad_hess_note": "reconstructed float64; float32-GradientPair parity in U5",
        "bins_note": "derived from cut_values; GHistIndex-rule parity in U9",
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
