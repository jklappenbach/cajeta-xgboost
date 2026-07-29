# Determinism — the bit-exactness contract

The claim this library makes is unusual and worth stating precisely:

> Training with `GBDT.fit` on any CPU produces **bit-identical** models and
> predictions, run to run and machine to machine — and, for the covered
> pipeline, bit-identical to what XGBoost's `gpu_hist` produces on NVIDIA
> hardware.

## Why that is hard

Floating-point summation orders, hardware transcendentals, and GPU
parallel-reduction schedules all produce *valid but different* last bits.
Reproducing the reference exactly means reproducing its arithmetic, not its
math:

- **Quantile sketch + binning** — the GPU sketch's merge/prune arithmetic,
  reproduced exactly (including the pruned-sketch structure and endpoints).
- **Histogram** — the GPU accumulates gradients in **int64 fixed point**
  (a quantised grid), not float sums. So does this library. That single
  design choice removes summation-order sensitivity outright.
- **Split gain** — the reference divides with `__fdividef`, which is the
  SFU `MUFU.RCP` instruction: a silicon lookup table, *not* the
  correctly-rounded reciprocal. `FastMath.fdividef` reproduces it from a
  probed capture of that table.
- **Objectives** — device `expf` is not libm's. It reduces the argument
  into [0,1) and evaluates the SFU `MUFU.EX2` table (`FastMath.expf`
  models the exact libdevice sequence: saturate, round-down FMA exponent
  split, two-part log2(e) reduction, table lookup, 2^n multiply).
  `Logistic`/`Softmax` mirror the reference op order around it, down to
  the 88.7 clamp and the float32-then-widen accumulation.

## The capture tables

`MUFU.RCP` and `MUFU.EX2` have no closed form. Ground truth is a one-time
capture from the reference GPU (RTX 4090): 2^23-entry tables, ~32 MB each,
`.gitignore`d, loaded at runtime via `RCP_TABLE` / `EX2_TABLE` env paths
(`run-tests.sh` wires them when present).

Without the captures — e.g. in CI, which is CPU-only — `FastMath` falls
back to **correctly-rounded** math: same API, same plumbing, not the SFU
bits. Every capture-dependent test detects this and self-skips, so the
suite is honest in both environments. (The `expf` model is validated
bit-exact against a full 2^23-point device sweep on all 8,310,140
normal-output points; the residual is the denormal tail, which the
objectives cannot observe.)

## The API contract

- `GBDT.fit` **refuses** stochastic-sampling configs — you cannot lose
  reproducibility by accident.
- `Model.deterministic` records the mode — reproducibility is auditable
  after the fact.
- `ModelIO` round-trips are bit-exact.

## Current parity status (suite: 50 passed / 0 failed)

Bit-exact against the NVIDIA reference fixtures today: quantile cuts and
binning (tiny + large), round-0 trees for regression, the full per-round
regression boosting loop, margins/predictions, histogram quantisation,
serialization, and round-0 binary/multiclass grad/hess.

Still gated (tracked in the `gpu-numeric-fidelity` plan):

- **Split-selection near-ties** — a handful of nodes where the reference's
  gain comparison comes down to op order inside the GPU evaluate kernel;
  reproducing it needs a device probe (planned on the self-hosted NVIDIA
  runner).
- **Multi-round binary/multiclass grad parity** — blocked on fixtures: the
  current `grad.npy` is a numpy reconstruction, not a device dump, so
  there is no device ground truth to compare against yet (same probe
  program).
- **Pruned-sketch interior at scale** — structure and endpoints are exact;
  the interior is held to a ≤1-rank bound pending the device scan model.
