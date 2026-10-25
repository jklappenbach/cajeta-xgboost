# cajeta-xgboost/bench

`dev.cajeta.xgboost` measured against upstream XGBoost, on the shared harness
`dev.cajeta.bench`. The ids, workloads and checks are defined by
`bench/ml/XGBoost.md` in the compiler repo (spec
`specs/benchmark-reference-spec.md` §7.2; the xgboost pages live under
`bench/ml/`, §12.8). This is `~/code/ml/xgbbench` moved in: its
`Bench.cajeta` `GBDT.fit` timing became `ml/xgb-hist-fit` and `ml/xgb-gpu-fit`,
its `bench.py` (QuantileDMatrix + train) became the upstream side of both, and
`gen.py`'s data shape (normal features, 5% NaN, a noisy linear score) is
`competitors/python/gen_data.py`'s. `ml/xgb-predict` is new. (`Phases.cajeta`
and `Prep.cajeta`, the per-phase timers, are not rows: `GBDT.lastPhases()`
already carries that split.)

## Run

```sh
cd bench
./bench.sh                                    # Cajeta rows only
./bench.sh --competitors                      # and the upstream XGBoost rows
./bench.sh --bench xgb-predict --competitors  # one id, both sides
CAJETA_BENCH_SMOKE=1 ./bench.sh --competitors # 1e4 x 20, 10 rounds, warmup 1, trials 3
./test/bench_smoke.sh                         # the smoke gate
```

`bench.sh` runs, each step timed: **data** (`gen_data.py` writes `X`, `y`, the
config, the upstream model and the reference results once, from seed
`20261025`, into `data/smoke/` or `data/full/`, gitignored), **build**
(`cajeta release` → `build/xgb-bench`; `NO_BUILD=1` reuses it), **cajeta**
(`build/xgb-bench --run` → `cajeta.csv`, `env.csv`), **python**
(`--competitors`: `competitors/python/xgb_bench.py` → `python.csv`) and
**merge** (`results.csv`, one schema-v2 header; competitor versions, load
average, GPU state and the data seed appended to `env.csv`), into
`results/<machine>/<run-id>/` (gitignored; the publish step copies non-smoke
runs into the compiler repo, spec §9.4).

It exports the SFU captures the library's `run-tests.sh` exports
(`RCP_TABLE=../tools/fdividef/rcp_mantissa.npy`,
`EX2_TABLE=../tools/expf/ex2_table.npy`, when present; each fit row's notes say
`SFU tables captured` or `fallback`) and `CUDA_PATH=/usr/local/cuda` for ptxas.
Defaults: `cajeta` on PATH else `../../../cpp/cajeta/build/src/cajeta`
(`CAJETA=`), the compiler repo's competitor venv for Python (`BENCH_PYTHON=`;
xgboost 3.4.1, numpy).

| Environment | |
|---|---|
| `CAJETA_BENCH_SMOKE=1` | 1e4 × 20 and 10 rounds instead of 1e6 × 50 and 100; warmup 1, trials 3; rows tagged `smoke` |
| `CAJETA_BENCH_NO_GPU=1` | `ml/xgb-gpu-fit` is `skipped` on both sides and the GPU is not touched (upstream gets no `device="cuda"` call) |
| `CAJETA_XPU_BACKEND` | which backend the Cajeta device fit runs on (`cuda`, `hip`, `vulkan`, `cpu`); also its row's `backend` |
| `CAJETA_XGB_BENCH_DATA` | the data directory (bench.sh sets it) |

## The rows

| id | variant | one iteration | check |
|---|---|---|---|
| `ml/xgb-hist-fit` | cpu | `GBDT.fit` from the raw matrix (sketch, bin, boost), binary:logistic, depth 6, max_bin 256, eta 0.3, base_score 0.5; upstream: `QuantileDMatrix` + `train`, `tree_method=hist`, `device=cpu` | training logloss within 1% (relative) of upstream `hist`'s |
| `ml/xgb-gpu-fit` | device | the same with `device "gpu"`; upstream `device=cuda` (gpu_hist) | the same |
| `ml/xgb-predict` | cpu | `GBDT.predict` margins for every row; upstream `inplace_predict(predict_type="margin")` | every margin within 1e-5 of upstream's |

`xgb-predict` walks **upstream's** trees on both sides: `gen_data.py` trains
the model with upstream `hist`, saves `model.json` for the upstream row, and
converts the same JSON trees into `ModelIO`'s layout (`model.cajeta`) for the
Cajeta row, which reads it with `ModelIO.deserialize`. So the check compares
two predictors on one forest, not two trainers.

## Tolerances, and why

| check | tolerance | why |
|---|---|---|
| fit: training logloss | 1% relative | `dev.cajeta.xgboost` reproduces NVIDIA `gpu_hist` bit for bit, its CPU path included; upstream's CPU `hist` sketches its quantile cuts differently, so the two forests are close but not equal. A real regression (a wrong split, a lost round) moves logloss far more than 1% |
| predict: margins | 1e-5 absolute | the same trees, both summing float32 leaf values onto the base margin in tree order; NaN follows each node's default direction on both sides |

## Notes

- The `release` flavor emits the library's gpu-release backends
  (`nvptx,amdgpu,vulkan,cpu`); a backend whose toolchain is absent is dropped
  by the compiler. The dependency itself comes from Olla (or `~/.olla`) at
  `0.7.0`, and the published archive's own kernels are what its README says
  (Vulkan and CPU); measuring CUDA kernels needs an archive built where
  ptxas is present.
- A `{ "path": ".." }` dependency resolves nothing on cajeta 0.46.3, so a
  working tree ahead of 0.7.0 is measured by `cajeta install` in `..` first.
- Every benchmark class declares a constructor; see cajeta-ml's
  `bench/README.md` for the measured reason.

## Measured (Phoenix, 2026-10-25, smoke)

The smoke run (run id `20261025T1915Z-0.46.3`; `results/` is gitignored here,
because a smoke run is never committed, spec §1.5.1 and §9.3): `test/bench_smoke.sh` PASS in 46 s — data 6 s,
Cajeta rows 17 s, upstream rows 20 s — at load average ~20 from other
sessions, so not numbers to quote. Checks: Cajeta's training logloss 0.272769
against upstream `hist`'s 0.273769 (0.37% apart, tolerance 1%); predict
margins on upstream's trees identical (max |Δ| 0). The release build took
5:09 (the library's kernels for every backend; amdgpu dropped here, no
ld.lld). At full size (1e6 × 50, 100 rounds) a Cajeta CPU fit is roughly 1000×
the smoke's work (3.3 s), so budget about an hour per fit.
