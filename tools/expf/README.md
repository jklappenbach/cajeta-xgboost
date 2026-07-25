# `expf` fidelity — device single-precision exp

XGBoost's `binary:logistic` (and multiclass softmax) compute gradients with
`common::Sigmoid(x) = 1 / (expf(min(-x, 88.7f)) + 1 + 1e-16f)` — device `expf`, and
the build uses **no `--use_fast_math`**, so it's the *accurate* `expf` (Cody-Waite
reduction + polynomial), not the one-op SFU `__expf`. cajeta computes the sigmoid with
its `Math.exp` intrinsic (libm `expf` on CPU). The U6 multi-round grad test diverges by
~1 ULP — but it compares against the fixture dump, which was made with **numpy**'s
`expf`. Device `expf`, libm `expf`, and numpy `expf` can all differ. The reference that
actually decides the trees is **device `expf`**.

So the first question is cheap: **does device `expf` == host libm `expf`?** If yes,
cajeta already matches the trees and there is no gap (the dump was just numpy-flavored).
If no, the probe data lets me model device `expf` (its `ex2.approx`-based reduction).

## Run on the NVIDIA box

```bash
cd tools/expf
nvcc -arch=native -O2 probe_expf.cu -o probe_expf && ./probe_expf
```

It prints how device `expf` relates to cheap models (`__expf`, `exp2f(x·log2e)`,
`ex2.approx`), and writes `expf_sweep.npy` — device `expf(x)` for `x` on a fixed grid
(`x_i = -90 + i·180/(2^23-1)`), which I regenerate identically on the host to diff
against libm `expf`.

Then `scp` it to me (it's `.gitignore`d — a build input, never committed):

```bash
scp expf_sweep.npy proton:/home/julian/code/cpp/cajeta-xgboost/tools/expf/
```

Paste the printed model-mismatch block too. From there I diff device-vs-libm `expf`,
and either (a) confirm no gap, or (b) model device `expf` on CPU (same probe→model
pattern as `__fdividef`; `ex2.approx.f32` is the SFU op to table if needed).

## Finding (2026-07-24) — structure confirmed; exact float32 coefficients remain

Against `expf_sweep.npy` (device `expf` over the FMA grid), device `expf` is the
**accurate** path (no `--use_fast_math`), and its structure reproduces cleanly:

1. Cody–Waite reduction: `j = rint(x·log2e)`, `f = x − j·ln2` (`f ∈ ~[−0.347, 0.347]`).
2. `e^f` via a **degree-6 minimax polynomial** — the fitted coefficients are *tuned*,
   not Taylor: `c5 ≈ 0.008360` (vs `1/5! = 0.008333`), `c6 ≈ 0.001384` (vs `1/6! =
   0.001389`). Fit residual ≈ 1.6e-7 (~1 ULP): the float64 fit sits at the noise floor,
   so the residual is the reduction/eval order, not the model shape.
3. Scale by `2^j` (`ldexp`).

So bit-exact device `expf` is a **coefficient-nail-down** problem: recover the exact
float32 minimax coefficients + the two-part `ln2` split + the FMA evaluation order, in
emulated float32. It is a `gpu-numeric-fidelity` item (see
`../../../cajeta/specs/gpu-numeric-fidelity-spec.md`), the same class as the
split-selection near-tie: "reproduce the device's exact float arithmetic bit-for-bit."
