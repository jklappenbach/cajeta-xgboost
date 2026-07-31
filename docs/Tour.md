# The tour

```
cajeta tour
```

builds the library, compiles `tour/src/dev/cajeta/xgboost/tour/Tour.cajeta`
against it, and runs it. The tour is **self-checking**: every claim it
prints is asserted, and it exits non-zero if any stops being true — it
doubles as a public-surface smoke test.

What each section demonstrates:

1. **Params** — XGBoost's knobs with XGBoost's defaults; the no-arg
   constructor is the deterministic config set.
2. **`GBDT.fit`** — training from a raw `[n, nf]` matrix; two fits produce
   **bit-identical** predictions on all rows, checked bit-by-bit.
3. **The contract** — `fit()` *throws* on `subsample < 1` rather than
   silently trade away reproducibility; the labeled escape hatch is
   `fitNonDeterministic`, and the mode lands in `Model.deterministic`.
4. **Predict** — raw margins under the identity link; the tour's synthetic
   dataset has planted threshold structure and the check confirms the
   trees found it.
5. **The open Model** — walks tree 0 node by node (flat arrays, per-tree
   stride), verifies the binary-tree invariant, prints the root split in
   plain language.
6. **Importance** — split-weight per feature computed in ~15 lines
   directly off the Model arrays; the decoy feature earns zero splits.
7. **Serialization** — `ModelIO` round-trip, bit-identical predictions
   after, audit bit intact.
8. **FastMath** — the device-fidelity layer: whether the probed SFU
   capture tables are loaded, and the device-model `expf` in action.
9. **The estimator protocol** — `XGBRegressor`, the
   `dev.cajeta.ml.Predictor` conformer: `fit`/`score` over the same data,
   then the payoff — the boosted model handed to `Split.crossValScore`
   like any other ecosystem estimator, no xgboost-specific glue.

The dataset is synthesized arithmetically (no RNG, no files), so the tour
prints identical bits on every machine — which is the library's whole
thesis, demonstrated by the demonstration itself.
