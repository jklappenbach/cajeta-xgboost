# The tour

```
./run-tour.sh          # builds the library + resolves dev.cajeta.ml, compiles, runs
cajeta tour            # the same, through the manifest task
```

The tour lives in `tour/src/dev/cajeta/xgboost/tour/`, one self-checking demo
per topic, registered in `Tour.cajeta`. It is **self-checking**: every claim it
prints is asserted, and it exits non-zero if any stops being true — so it
doubles as a public-surface smoke test. `scripts/check-library-tour-coverage.sh`
holds it to the whole surface (23/23 public types today) and runs in CI.

The dataset (`Data.cajeta`) is a house-price table with **named features** —
`sqft_living`, `bedrooms`, `age_years`, `lot_sqft` — derived arithmetically from
the row index, so there is no RNG, no file, and no clock anywhere in it and the
tour prints identical bits on every machine. `lot_sqft` is genuinely **missing**
on every 17th listing while still driving the price, which is what makes the
learned default direction observable rather than merely asserted.

What each demo demonstrates:

1. **`ParamsDemo`** — the full knob table (`rounds`, `maxDepth`, `maxLeaves`,
   `lambda`/`alpha`/`gamma`/`minChildWeight`, `eta`, `baseScore`, `maxBin`) with
   XGBoost's defaults, which are also the deterministic config set; `reg_lambda`
   visibly flattening the model; and the contract — `fit()` **refuses**
   `subsample < 1` with a message naming the escape hatch, `fitNonDeterministic`
   refuses too (RNG parity is a later unit) and stamps its mode on the model.
2. **`PrepDemo`** — the QuantileDMatrix-equivalent path run by hand:
   `Sketch.cuts` → the `cutPtrs` layout → `Binner.bin`/`searchBin`, with
   `maxBin` visibly buying resolution (20 / 57 / 197 cut points at maxBin
   4 / 16 / 64) and missing values binning to `-1`. Bin once, train many.
3. **`TrainDemo`** — `GBDT.fit`; two independent fits produce **bit-identical**
   predictions on all 400 rows, checked bit-by-bit; then the open `Model` —
   tree 0 walked node by node, the binary-tree invariant, the root split
   printed in the data's own vocabulary.
4. **`MissingDemo`** — missing values are **routed**, not ignored: blanking a
   recorded `lot_sqft` moves the prediction on 241 listings, and a hand walk
   that follows `defaultLeft` reproduces `TreeWalker.leaf` and
   `TreeWalker.marginSingle` exactly.
5. **`ImportanceDemo`** — `Importance.forModel(model, nf, weight, gain, cover)`
   reproducing `get_score` straight off a trained model, cross-checked against a
   hand walk for `weight`; plus `addTree`, the per-tree primitive underneath.
6. **`ObjectivesDemo`** — the three tasks the same table supports: the
   gradients themselves (`SquaredError`, `Logistic`, `Softmax`), then a real
   `binary:logistic` fit (probabilities strictly inside (0,1), ~99.8% training
   accuracy, and the link-free `margin > 0 ⇔ p > 0.5` decision) and a real
   `multi:softmax` fit (3 price tiers, 12 rounds × 3 classes = 36 trees
   round-major, rows summing to 1, a bit-identical refit). Set
   `Params.objective` (+ `numClass`) and `GBDT.fit` does the rest.
7. **`PersistDemo`** — `ModelIO.serialize` → a real file → read back →
   `deserialize` → bit-identical predictions, audit bit intact. Train once,
   score later, which is the deployment story.
8. **`InternalsDemo`** — one boosting round through the machinery:
   `SquaredError.gradHess` → `GradientQuantiser` (fixed-point scale) →
   `Histogram.buildAll` → the **subtraction trick** (`parent − left == right`,
   bin for bin, exactly — the property int64 accumulation buys) →
   `SplitFinder.findBest`/`calcGain`/`leafWeight` → `TreeBuilder.growTree` →
   `NodeStats`, closing with `Booster.train` and its per-round margin history
   showing the error fall round over round.
9. **`FastMathDemo`** — the device-fidelity layer: whether the probed SFU
   capture tables are loaded, and `expf` / `fdividef` / `fmaf` / the mantissa
   tables in action. This is why Logistic's gradients can match a GPU's.
10. **`EstimatorDemo`** — `XGBRegressor`, the `dev.cajeta.ml.Predictor`
    conformer: `fit`/`score`, then the payoff — the boosted model handed to
    `Split.crossValScore` like any other ecosystem estimator, with `model()`
    keeping `ModelIO` and the tree walk reachable.

## Objectives and importance (previously plan gaps D1/D2 — both closed)

- **D1 — objective-parameterized training.** `Params.objective` selects the
  loss and `Params.numClass` the group count; `Booster.train` grows K trees per
  round over K gradient slices (round-major, class of tree `j` is `j % K`), and
  `GBDT.predictProba` / `predictClass` apply the inverse link. The regression
  path is unchanged and still bit-exact — the parity fixtures pin it.
- **D2 — importance from a Model.** `TreeBuilder` records per-node
  `lossChanges`/`sumHessian` during growth (neither is recoverable from a
  finished tree), they ride through `Booster` → `Model` → `ModelIO`, and
  `Importance.forModel` sums `get_score`'s three totals.

What remains open is *parity*, not reachability: full bit-exact agreement with
XGBoost's multiclass output is still gated on the two U12 transcendental items
(`expf`, and a GPU split-selection near-tie), per plan unit 7.3.1.
