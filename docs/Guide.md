# Guide

## What this is

`dev.cajeta.xgboost` trains and evaluates gradient-boosted decision trees
with XGBoost's **GPU algorithm** (`gpu_hist`) semantics: quantile-sketch
binning, an int64 fixed-point gradient histogram, and the same split-gain
arithmetic — reproduced in deterministic CPU code. The goal is not "about
the same model"; it is **the same bits** as the NVIDIA reference.

It is a library (`.cja`), resolved from Olla:

```jsonc
"dependencies": { "dev.cajeta.xgboost": "0.1.0" }
```

## Quickstart

```cajeta
import cajeta.math.Tensor;
import dev.cajeta.xgboost.api.GBDT;
import dev.cajeta.xgboost.api.Params;
import dev.cajeta.xgboost.booster.Model;

// x: [n, nf] row-major float64 Tensor; y: float64[n] targets.
Params p = heap Params();          // XGBoost's defaults, verbatim
p.rounds = (int64) 20;
p.maxDepth = (int64) 5;

Model m = GBDT.fit(x, y, n, nf, p);
float32[] preds = GBDT.predict(m, x, n, nf);
```

`fit` runs raw matrix → quantile cuts → bins → boosting loop, all
deterministic. Run the [tour](Tour.md) (`cajeta tour`) for the full surface
in action.

## `Params` — XGBoost's knobs

Every field maps 1:1 to an XGBoost parameter; the no-arg constructor **is**
XGBoost's default configuration.

| Field | XGBoost name | Default |
|---|---|---|
| `rounds` | `num_boost_round` | 10 |
| `maxDepth` | `max_depth` | 6 |
| `maxLeaves` | `max_leaves` | 0 (unbounded) |
| `lambda` | `reg_lambda` | 1.0 |
| `alpha` | `reg_alpha` | 0.0 |
| `gamma` | `min_split_loss` | 0.0 |
| `minChildWeight` | `min_child_weight` | 1.0 |
| `eta` | `learning_rate` | 0.3 |
| `baseScore` | `base_score` | 0.5 |
| `maxBin` | `max_bin` | 256 |
| `subsample` | `subsample` | 1.0 |
| `colsampleByTree` | `colsample_bytree` | 1.0 |

## `GBDT` — fit and predict

- `GBDT.fit(x, label, n, nf, p)` — deterministic, bit-exact training.
  **Throws** if `p` engages a stochastic-sampling knob (`subsample` or
  `colsampleByTree` < 1): you cannot forfeit reproducibility silently.
- `GBDT.fitNonDeterministic(...)` — the distinctly-named entry point for
  stochastic configs; the mode is recorded on the model. (The sampling
  engine itself — RNG parity with the reference — is a later unit; today
  this entry point refuses rather than mislabel.)
- `GBDT.predict(m, x, n, nf)` — per-row raw margin; under
  `reg:squarederror`'s identity link, that is the prediction.

## `Model` — an open data structure

Trees are flat arrays with a per-tree stride `cap`: node `k` of tree `r`
lives at index `r*cap + k`. Nothing is hidden — walk it directly (the tour's
sections 5–6 do):

| Field | Meaning |
|---|---|
| `rounds`, `cap`, `nodeCount[r]` | tree count, stride, per-tree node counts |
| `splitIndices` | split feature (internal nodes) |
| `splitConditions` | threshold (internal) / leaf value (leaf) |
| `defaultLeft` | where a missing value goes |
| `leftChildren` / `rightChildren` | -1 for leaves |
| `baseWeights`, `isLeaf` | raw CalcWeight / leaf flags |
| `deterministic` | the audit bit: was this trained bit-exact? |

## `ModelIO` — serialization

`ModelIO.serialize(m)` → `int8[]`; `ModelIO.deserialize(buf)` → `Model`.
The round-trip is bit-exact — deserialized models predict identically, and
the `deterministic` audit bit survives.

## Objectives

`reg:squarederror` is the public `fit` surface today. The
`binary:logistic` and `multi:softmax` objective kernels
(`objective/Logistic`, `objective/Softmax`) implement the reference
arithmetic — including the device-transcendental model for `expf` (see
[Determinism.md](Determinism.md)) — and are exercised by the parity suite;
their public `fit` wiring follows the remaining parity units.
