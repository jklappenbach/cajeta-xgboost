# Golden fixtures — the parity ground truth

`generate.py` trains reference XGBoost under a pinned, deterministic config and
dumps what cajeta must reproduce bit-for-bit (plan U2, spec §1.6 / §7).

## The reference is the NVIDIA GPU algorithm

Parity is against **XGBoost's GPU `hist`** (`--device cuda`, deterministic), run
on an NVIDIA box. XGBoost's CPU `hist` does **not** bit-match its GPU `hist`, so:

- `--device cuda` → the **parity reference** (`manifest.is_reference == true`).
  These are the fixtures cajeta's bit-exact tests assert against.
- `--device cpu` → a **format / loader** fixture only (`is_reference == false`).
  Useful for developing the loader and the harness where no NVIDIA GPU is
  present; **never** a parity target.

## Format (one directory per config)

| File | Contents |
|------|----------|
| `manifest.json` | xgboost version, the exact params, dataset spec, `is_reference` |
| `X.npy` | training matrix, `float64`, `NaN` = missing |
| `y.npy` | labels, `float64` |
| `model.json` | XGBoost native dump — every tree / split / default-dir / leaf weight |
| `margins.npy` | per-round raw output margin, shape `[rounds, n]` |
| `preds.npy` | final prediction (post inverse-link) |

`.npy` is read on the cajeta side via `cajeta.math.npio.Npy`.

## Generate

```
# format/loader dev (any machine):
python3 generate.py --out tiny_reg_cpu --objective reg:squarederror --device cpu

# the reference (NVIDIA box):
python3 generate.py --out tiny_reg --objective reg:squarederror --device cuda
```

Requires `xgboost` + `numpy` (and CUDA + an NVIDIA GPU for `--device cuda`).

## Pending (U3 / U9)

The harness still needs to dump XGBoost's **internal bin edges + per-instance bin
indices + per-round grad/hess**, for the binned-boundary histogram-level parity
(§7.1). Tracked as a TODO in `generate.py`.

## Sketch tie-order fixtures

- `zeros/` (`gen_zeros.py`, needs xgboost==3.1.2): columns holding both `-0.0`
  and `+0.0`, with the real `QuantileDMatrix` cuts at max_bin 256, 64 and 16 in
  `cut_ptrs_<b>.npy` / `cut_values_<b>.npy`. Which zero leads an equal run
  follows `std::sort`, so these pin the sketch's mixed-zero path.
- `stdsort/` (`gen_stdsort.py`, builds `stdsort_ref.cpp` with g++): libstdc++'s
  `std::sort` and `std::partial_sort` permutations of tie-heavy queues,
  including McIlroy adversarial inputs that exhaust introsort's depth limit.
