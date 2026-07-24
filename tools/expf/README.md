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
