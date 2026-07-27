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

## ~~Finding (2026-07-24) — structure confirmed; exact float32 coefficients remain~~ (SUPERSEDED 2026-07-27)

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

## ~~Finding (2026-07-25) — both CPU models fall short; needs a finer capture~~ (SUPERSEDED 2026-07-27)

`gpu-numeric-fidelity` U1.2.1. Two CPU reproductions were tried against the full
`expf_sweep.npy` (harnesses committed here):

- `analyze_poly_model.py` — Cody–Waite reduce → degree-6 Horner poly → `ldexp`, with
  **coordinate descent** over each float32 coefficient (±8 ULP) and the `ln2` low part.
  Stalls at **~30% bit-mismatch**. Right shape (~1 ULP everywhere), wrong exact op
  sequence — a plain Horner poly is not the device algorithm.
- `analyze_ex2_model.py` — the `probe_expf.cu` model `ex2.approx(reduce(x))`, emulating
  `ex2.approx` from the uniform `ex2_table.npy` via `ldexp(table[round(frac·2²³)], floor)`.
  **~93% mismatch**, errors spread ±5+ ULP. The uniform 2²³ table cannot reproduce the
  SFU's internal interpolation (the "~15 ULP short" wall): `MUFU.EX2` uses a piecewise
  quadratic (Oberman–Siu), not a uniform table.

**Conclusion / blocker:** bit-exact device `expf` bottoms out in the exact `MUFU.EX2`
interpolation, which the current captures don't contain. Unblock needs one of:
(a) a probe that captures `ex2.approx` at the *exact* float32 inputs the reduction
produces (not a uniform grid) — then a direct table keyed by input bits is exact; or
(b) probe `ex2.approx` densely enough per binade to fit the Oberman–Siu piecewise-quad
coefficients; or (c) lift the accurate-`expf` SASS from the pinned toolchain's libdevice
and transcribe its exact op sequence + constants. All three need the NVIDIA box.

## ~~Refinement (2026-07-25) — it's njuffa accurate `expf`; needs the libdevice IR~~ (WRONG — see below)

XGBoost's device `expf` (no `--use_fast_math`) is the **accurate** path, and it is
Norbert Juffa's well-known algorithm (`analyze_njuffa_model.py`): magic-number `rint`
(`fma(x, log2e, 1.5·2²³) − 1.5·2²³`), two-part `ln2` reduction, degree-6 minimax poly,
`ldexp`. With his classic coefficients it reproduces `expf_sweep.npy` to **maxULP = 2**
(31% of points off by 1–2 ULP). Coordinate descent over every float32 coefficient +
the `ln2`/`log2e` constants (`analyze_coeff_descent.py`) only reaches ~30% — the
residual is **structural** (the exact op association of `1 + f + f²·P`, `ldexp` vs
bit-`scalbn`, `rintf` vs the magic number), not coefficient tuning.

So this is NOT the SFU `ex2.approx` after all (that was the *fast* `__expf`); it's a
pure-FMA polynomial, fully reproducible on CPU once the exact op sequence is known.
**Cheapest unblock — no hardware probe, just a file on the box:** disassemble
`__nv_expf` from the pinned toolchain's libdevice and transcribe it verbatim:

```bash
# on Phoenix (the CUDA toolkit that built the 3.1.2 device objects)
llvm-dis "$(find / -name 'libdevice*.bc' 2>/dev/null | head -1)" -o - \
  | sed -n '/define.*__nv_expf/,/^}/p'
```

That IR gives the exact FMA order + float32 constants → transcribe into `FastMath.expf`
and it is bit-exact. (This machine's only `libdevice*.bc` are empty clang test stubs.)

---

## RESOLVED (2026-07-27) — the libdevice IR; bit-exact on every normal result

The `__nv_expf` IR arrived from Phoenix and **refutes the njuffa-polynomial reading
above**. There is no minimax polynomial. The SFU `ex2.approx` *is* the core — applied
to an argument reduced into `[0,1)`, with the `2^n` reapplied as a separate multiply.
That is why coefficient descent stalled at maxULP=2: it was fitting a polynomial to a
hardware table. The earlier `ex2` attempt (`analyze_ex2_model.py`, 93% mismatch) had
the right instruction but the wrong reduction — it fed `ex2` the *unreduced* `x·log2e`
and rebuilt with `ldexp`, instead of the IR's saturate/`fma_rd` exponent split.

Transcribed (non-FTZ path; `__nvvm_reflect` selects FTZ, and XGBoost builds without it):

```
C  = L2E_HI / 252.0f                        // folded at compile time, round-to-nearest
j  = saturate(fmaf(a, C, 0.5f))             // -> [0,1]; NaN -> 0
i  = fma_rd(j, 252.0f, 8388609.0f)          // round-toward -inf; 8388609 = 2^23 + 1
n  = i - 8388735.0f                         // 8388735 = 2^23 + 127; n in [-126,126], exact
f  = fmaf(a, L2E_LO, fmaf(a, L2E_HI, -n))   // reduced arg, in [0,1)
s  = bitcast<float>(bitcast<int>(i) << 23)  // = 2^n
return ex2.approx.ftz(f) * s
```

Constants decoded from the IR's bitcasts: `1069066811 = 0x3FB8AA3B` = `L2E_HI`
(log2 e), `849703008 = 0x32A57060` = `L2E_LO` (1.9259630e-8, the log2 e tail),
`1262485504 = 0x4B400000` = `2^23`. The `<< 23` works because the low 9 bits of
`0x4B000000` are zero, so `bits(i) << 23` lands `(n + 127)` in the exponent field.

`ex2.approx` is indexed by **truncating** the fraction to 23 bits — `floor(f·2^23)`,
not `rint`. Round-to-nearest indexing gives 29.3% mismatch; truncation gives 0.23%.

**Validation** (`model_libdevice.c`, committed here — build: `cc -O2 -mfma
model_libdevice.c -o model_libdevice -lm`), against the full `expf_sweep.npy`:

| domain | points | mismatches |
|---|---|---|
| **normal results** (the whole XGBoost range) | 8,310,140 | **0 — bit-exact** |
| denormal tail (`x < -87.34`, saturate clamped) | 18,948 | 18,948 (≤2 ULP) |

The denormal tail is **not** a model defect: there the clamp pushes `f` outside `[0,1)`
and the reconstruction needs `ex2.approx` at negative arguments, which
`ex2_table.npy` (captured only over `[0,1)`) does not contain. 16,089 of those match
round-toward-zero on the denormal; the rest are unresolvable without a wider capture.
**It is also unobservable through XGBoost.** `Sigmoid(x) = 1/(expf(min(-x,88.7)) + 1
+ 1e-16)` is exactly `1.0f` in float32 once `expf` drops below ~3e-8 (ULP near 1.0 is
6e-8); the denormal tail starts at 1e-38, thirty orders of magnitude further down. So
every `expf` value the gradient path can actually distinguish is bit-exact.

### Shipping `ex2.approx`

No coefficient fit is needed — `FastMath.ex2Mantissa` follows the existing
`rcpMantissa` pattern: the capture stays `.gitignore`d and loads at runtime from an
env-var path (`EX2_TABLE`, as `RCP_TABLE` does), with an accurate-math placeholder
when absent so the plumbing stays testable off-GPU.

For the record, the table's structure was confirmed to be Oberman–Siu piecewise
quadratic (a 64-segment quadratic fits it to 0.62 float32 ULP, and adding segments
does not improve past ~0.57 — that floor is the table's own float32 rounding). A fit
is therefore viable if the table ever needs to be eliminated, but it cannot be made
bit-exact from float64 fitting alone.
