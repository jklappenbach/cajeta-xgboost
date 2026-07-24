# `__fdividef` fidelity — probing the SFU reciprocal

Device XGBoost scores split gains with `__fdividef` (`src/tree/split_evaluator.h:126`,
`Divide()`), the fast approximate FP32 division = `a * rcp.approx.f32(b)`. On
knife-edge near-tied splits this picks a different boundary than accurate division,
which can flip a whole feature's `min_child_weight` validity — the sole cause of the
`tiny_reg_mcw10` divergence (the tree is otherwise 23/23 nodes bit-identical).

`rcp.approx.f32` (`MUFU.RCP`) has **no closed form** — it is a silicon lookup table
(±2 ULP vs IEEE, deterministic). The only way to reproduce it bit-for-bit is to
probe the hardware. Reciprocal factors exactly by exponent
(`rcp(2^E·m) = ldexp(rcp(m), −E)`), so we only need the mantissa table over
`m ∈ [1, 2)` — `2^23` entries — not the full 32-bit input space.

## Run this on the NVIDIA box (same GPU family as the reference fixtures)

```bash
cd tools/fdividef
nvcc -arch=native -O2 probe_rcp.cu -o probe_rcp
./probe_rcp
```

The probe identifies which CPU-reproducible sequence equals `__fdividef` bit-for-bit
(a raw `a*rcp` does NOT — it's Newton-refined). Output:

```
over 268435456 samples:
  exponent-factoring       mismatches = 0   <-- MATCH
  M0 a*rcp                 mismatches = <big>
  M1 rcp+1 Newton          mismatches = 0   <-- MATCH    (example)
  M2 rcp+2 Newton          mismatches = ...
  M3 __fdiv_rn (correct)   mismatches = ...
  M4 a*__frcp_rn           mismatches = ...
=> factoring HOLDS ...; __fdividef model = the M* line reading MATCH above.
examples (hex bits): ...
GPU: <name>  sm_XX
wrote rcp_mantissa.npy (8388608 entries, 33.6 MB)
```

## Result (RTX 4090, sm_89)

- `exponent-factoring` = 0 → the 2^23 mantissa table fully captures `MUFU.RCP`.
- `MD div.approx.f32` = 0 → **`__fdividef` is exactly `div.approx.f32`**.
- `M0 a*rcp` disagrees only on huge `b` (exp 253–254, |b|≈2^127): `__fdividef`
  flushes to 0 there. **For every normal-range `b` (all XGBoost feeds it),
  `a * rcp.approx.f32(b)` == `__fdividef` bit-for-bit.**

So the model is settled: `fdividef(a,b) = a * rcp.approx.f32(b)`, with a flush-to-0
guard for near-`FLT_MAX` `b`. The one CPU-side ingredient is `rcp.approx.f32` — the
SFU reciprocal (NOT the correctly-rounded reciprocal), which is the captured table.

## Getting the table to me

`rcp_mantissa.npy` is the ground truth. `scp` it to `proton` (it's `.gitignore`d, so
it stays out of git):

```bash
scp rcp_mantissa.npy proton:/home/julian/code/cpp/cajeta-xgboost/tools/fdividef/
```

Then I: (1) validate it closes `mcw10` (node-5 feature-4 gain via `a*rcp` from the
table), (2) wire `FastMath.fdividef` into `SplitFinder`, and (3) build the compact
CPU-reproducible seed (reverse-engineered SFU interpolator) so CI regenerates the
table without a GPU.

## The 33 MB table is a build input, never committed (regenerate-in-CI)

CI is CPU-only, so `rcp_mantissa.npy` is **not** committed (it's `.gitignore`d). It is
a one-time ground-truth capture. From it I reverse-engineer the SFU's compact
interpolator — the per-interval quadratic coefficients `MUFU.RCP` actually uses
(Oberman–Siu) — and **verify the reconstruction reproduces all 2^23 entries
bit-for-bit**. Only that tiny seed (a few hundred bytes) is committed; `FastMath`
computes `rcp(m)` from it on CPU, so CI regenerates the behaviour with no GPU and no
large artifact.

Getting the capture to me (I work on a non-NVIDIA box):
- **preferred:** `scp tools/fdividef/rcp_mantissa.npy proton:<repo>/tools/fdividef/`
  (stays out of git history entirely), or
- a throwaway commit of the `.npy` that I pull and immediately `git rm` after
  extracting the seed.

Then ping me — I fit + verify the coefficients, wire `FastMath.fdividef` into
`SplitFinder`, and re-enable `ConfigParityTest::minChildWeightTreeBitIdentical`.

> Research risk: if the SFU interpolator can't be reproduced bit-exactly on CPU
> (fit doesn't verify against all 2^23), we fall back — git-lfs the delta-compressed
> table, or add a GPU CI runner. We decide that only if the clean path fails.
