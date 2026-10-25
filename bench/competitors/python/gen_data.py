#!/usr/bin/env python3
"""Writes every input the dev.cajeta.xgboost benchmarks read, once, from one
seed, plus upstream XGBoost's reference results on that same data.

    gen_data.py <out-dir>        (bench.sh: data/smoke or data/full)

The data is ~/code/ml/xgbbench's gen.py shape (normal features, 5% NaN, a
noisy linear score), here as a binary target: bench/ml/XGBoost.md checks
logloss. Sizes are the page's (1e6 x 50, 100 rounds, depth 6) unless
CAJETA_BENCH_SMOKE=1 (1e4 x 20, 10 rounds), recorded in manifest.tsv.

  xgb/X.npy, xgb/y.npy           the matrix both sides fit and predict
  xgb/config.npy                 [rounds, max_depth, max_bin]
  xgb/model.json                 upstream `hist` trained on X, y (CPU)
  xgb/model.cajeta               the same trees in dev.cajeta.xgboost's
                                 ModelIO layout, for ml/xgb-predict
  ref/hist-logloss.npy           upstream `hist`'s training logloss
  ref/predict-margin.npy         upstream's margins for X from model.json

The GPU is never touched: device="cpu" throughout.
"""
import hashlib
import json
import os
import sys

import numpy as np
import xgboost as xgb

import benchcsv

SEED = 20261025


def params(rounds_unused=None, device="cpu"):
    """xgbbench's parameters, binary:logistic, base_score pinned at 0.5 so the
    base margin is logit(0.5) = 0 on both sides."""
    return {"tree_method": "hist", "device": device, "seed": 0, "max_depth": 6,
            "max_bin": 256, "eta": 0.3, "reg_lambda": 1.0, "gamma": 0.0,
            "min_child_weight": 1.0, "base_score": 0.5, "objective": "binary:logistic"}


def logloss(y, p):
    p = np.clip(p.astype(np.float64), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def to_modelio(booster):
    """The booster's trees as ModelIO.serialize writes a Model: little-endian
    64-bit words — rounds, cap, f32 bits of baseMargin, deterministic,
    numClass, nodeCount[rounds], then nine [rounds*cap] arrays (splitIndices,
    splitConditions, defaultLeft, leftChildren, rightChildren, baseWeights,
    isLeaf, lossChanges, sumHessian), then the sfuCaptured trailer word."""
    j = json.loads(booster.save_raw("json"))
    trees = j["learner"]["gradient_booster"]["model"]["trees"]
    base_score = float(j["learner"]["learner_model_param"]["base_score"].strip("[]"))
    base_margin = np.float32(np.log(base_score / (1.0 - base_score)))
    r = len(trees)
    cap = max(len(t["left_children"]) for t in trees)
    a = r * cap
    si = np.zeros(a, np.int64)
    sc = np.zeros(a, np.float32)
    dl = np.zeros(a, np.int64)
    lc = np.full(a, -1, np.int64)
    rc = np.full(a, -1, np.int64)
    bw = np.zeros(a, np.float32)
    leaf = np.zeros(a, np.int64)
    lch = np.zeros(a, np.float32)
    sh = np.zeros(a, np.float64)
    counts = np.zeros(r, np.int64)
    for i, t in enumerate(trees):
        n = len(t["left_children"])
        o = i * cap
        counts[i] = n
        si[o:o + n] = t["split_indices"]
        sc[o:o + n] = np.asarray(t["split_conditions"], np.float32)
        dl[o:o + n] = t["default_left"]
        lc[o:o + n] = t["left_children"]
        rc[o:o + n] = t["right_children"]
        bw[o:o + n] = np.asarray(t["base_weights"], np.float32)
        leaf[o:o + n] = np.asarray(t["left_children"]) == -1
        lch[o:o + n] = np.asarray(t["loss_changes"], np.float32)
        sh[o:o + n] = t["sum_hessian"]
    words = [np.array([r, cap, base_margin.view(np.uint32), 1, 1], np.uint64), counts.view(np.uint64),
             si.view(np.uint64), sc.view(np.uint32).astype(np.uint64), dl.view(np.uint64),
             lc.view(np.uint64), rc.view(np.uint64), bw.view(np.uint32).astype(np.uint64),
             leaf.view(np.uint64), lch.view(np.uint32).astype(np.uint64), sh.view(np.uint64),
             np.array([0], np.uint64)]
    return np.concatenate(words).astype("<u8").tobytes()


def main(out):
    smoke = benchcsv.smoke()
    n, nf, rounds = (10_000, 20, 10) if smoke else (1_000_000, 50, 100)
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((n, nf))
    w = rng.standard_normal(nf)
    s = x @ w + 0.1 * rng.standard_normal(n)
    y = (s > 0).astype(np.float64)
    x[rng.random((n, nf)) < 0.05] = np.nan

    d = os.path.join(out, "xgb")
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(out, "ref"), exist_ok=True)
    np.save(os.path.join(d, "X.npy"), x)
    np.save(os.path.join(d, "y.npy"), y)
    np.save(os.path.join(d, "config.npy"), np.array([rounds, 6, 256], np.int64))

    dm = xgb.QuantileDMatrix(x, label=y, missing=np.nan, max_bin=256)
    bst = xgb.train(params(), dm, num_boost_round=rounds)
    bst.save_model(os.path.join(d, "model.json"))
    with open(os.path.join(d, "model.cajeta"), "wb") as f:
        f.write(to_modelio(bst))
    p = bst.inplace_predict(x)
    np.save(os.path.join(out, "ref", "hist-logloss.npy"), np.array([logloss(y, p)]))
    margin = bst.inplace_predict(x, predict_type="margin").astype(np.float32)
    np.save(os.path.join(out, "ref", "predict-margin.npy"), margin)

    lines = []
    for root, _, files in os.walk(out):
        for fn in sorted(files):
            if fn == "manifest.tsv":
                continue
            path = os.path.join(root, fn)
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            lines.append((os.path.relpath(path, out), h))
    with open(os.path.join(out, "manifest.tsv"), "w") as f:
        f.write(f"# dev.cajeta.xgboost bench data, seed {SEED}, scale {'smoke' if smoke else 'full'}\n")
        f.write(f"# n {n} nf {nf} rounds {rounds} depth 6 max_bin 256 binary:logistic, 5% NaN; "
                f"xgboost {xgb.__version__}\n")
        f.write("file\tsha256\n")
        for rel, h in sorted(lines):
            f.write(f"{rel}\t{h}\n")
    print(f"gen_data: {n} x {nf}, {rounds} rounds into {out} (seed {SEED}); "
          f"upstream hist train logloss {logloss(y, p):.6g}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: gen_data.py <out-dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
