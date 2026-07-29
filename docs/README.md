# cajeta-xgboost documentation

A bit-exact-to-NVIDIA GBDT (gradient-boosted decision tree) trainer and
predictor in pure Cajeta: XGBoost's `gpu_hist` algorithm reproduced so that
training on the CPU yields the same bits the reference produces on CUDA.

| Document | What it covers |
|---|---|
| [Guide.md](Guide.md) | Quickstart, the public API (`GBDT`, `Params`, `Model`, `ModelIO`), model layout |
| [Tour.md](Tour.md) | Narrative companion to the runnable tour (`cajeta tour`) |
| [Determinism.md](Determinism.md) | The bit-exactness contract: what is exact, how, and what is still gated |

Engineering workflow artifacts (specs, plans) live in the cajeta repo's
`specs/` + `agents/` per the project workflow — `docs/` here is user-facing
only.
