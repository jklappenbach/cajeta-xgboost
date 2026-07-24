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

Then commit `rcp_mantissa.npy` here and ping me. I delta-compress it into the cajeta
`FastMath` table, wire `FastMath.fdividef` into `SplitFinder` gain scoring, and
re-enable `ConfigParityTest::minChildWeightTreeBitIdentical`.

> Note: the 33.6 MB raw table is a probe artifact; the committed cajeta table will
> be the delta-compressed form (the reciprocal is monotonic, so neighbours differ by
> ~1 ULP → a few MB). If you'd rather keep even that out of git, say so and we'll
> git-lfs it or regenerate-in-CI.
