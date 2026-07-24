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

Expected output:

```
verify: exponent-factoring mismatches = 0 / 268435456
verify: fdividef==a*rcp mismatches     = 0 / 268435456
GPU: <name>  sm_XX
wrote rcp_mantissa.npy (8388608 entries, 33.6 MB)
```

Both mismatch counts **must be 0** — that confirms (a) the mantissa table fully
captures `MUFU.RCP` and (b) `__fdividef` really is `a * rcp.approx.f32(b)` on this
GPU. If either is non-zero, stop and ping me (the model needs revisiting).

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
