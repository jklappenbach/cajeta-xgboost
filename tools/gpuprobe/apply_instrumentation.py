#!/usr/bin/env python3
"""Apply the evaluate-agent instrumentation to xgboost v3.1.2 source.

EOL-agnostic replacement (git apply kept tripping over Git-for-Windows CRLF
materialization on the runner). Same edits as instrument_eval_3.1.2.patch:
print every tile winner reaching DeviceSplitCandidate::Update (accept/reject
via before/after loss_chg) and every post-reduce per-node winner.

    python apply_instrumentation.py <xgboost-src-root>
"""
import sys

TILE_OLD = """        GradientPairInt64 left = missing_left ? bin + missing : bin;
        GradientPairInt64 right = parent_sum - left;
        best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx, left, right,
                           false, param, rounding);
      }"""

TILE_NEW = """        GradientPairInt64 left = missing_left ? bin + missing : bin;
        GradientPairInt64 right = parent_sum - left;
        float probe_before = best_split->loss_chg;
        best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx, left, right,
                           false, param, rounding);
        printf("[tile] nidx=%d fidx=%d tile=%d gidx=%d gain=%.9g ml=%d lh=%lld rh=%lld before=%.9g after=%.9g\\n",
               nidx, fidx, scan_begin, split_gidx, static_cast<double>(gain),
               static_cast<int>(missing_left), static_cast<long long>(left.GetQuantisedHess()),
               static_cast<long long>(right.GetQuantisedHess()), static_cast<double>(probe_before),
               static_cast<double>(best_split->loss_chg));
      }"""

NODE_OLD = """  dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(
      temp.data().get(), temp_storage_bytes, feature_best_splits.data(), out_splits.data(),
      num_segments, reduce_offset, reduce_offset + 1, ctx->CUDACtx()->Stream()));
}

void GPUHistEvaluator::CopyToHost"""

NODE_NEW = """  dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(
      temp.data().get(), temp_storage_bytes, feature_best_splits.data(), out_splits.data(),
      num_segments, reduce_offset, reduce_offset + 1, ctx->CUDACtx()->Stream()));

  {  // [probe] print each reduced per-node winner
    auto d_out = out_splits;
    auto d_in = d_inputs;
    dh::LaunchN(out_splits.size(), ctx->CUDACtx()->Stream(), [=] __device__(size_t i) {
      auto const &s = d_out[i];
      printf("[node] nidx=%d findex=%d loss_chg=%.9g fvalue=%.9g dir=%d lh=%lld rh=%lld\\n",
             d_in[i].nidx, s.findex, static_cast<double>(s.loss_chg),
             static_cast<double>(s.fvalue), static_cast<int>(s.dir),
             static_cast<long long>(s.left_sum.GetQuantisedHess()),
             static_cast<long long>(s.right_sum.GetQuantisedHess()));
    });
  }
}

void GPUHistEvaluator::CopyToHost"""


def edit(path: str, pairs) -> None:
    src = open(path, newline="").read()
    eol = "\r\n" if "\r\n" in src else "\n"
    norm = src.replace("\r\n", "\n")
    for old, new, tag in pairs:
        if old not in norm:
            print(f"NEEDLE MISSING in {path}: {tag}", file=sys.stderr)
            sys.exit(1)
        norm = norm.replace(old, new, 1)
    open(path, "w", newline="").write(norm.replace("\n", eol) if eol == "\r\n" else norm)


PRUNE_OLD = """    d_out[idx] = BinarySearchQuery(it, it + in_column.size(), q);
  });"""

PRUNE_NEW = """    d_out[idx] = BinarySearchQuery(it, it + in_column.size(), q);
    if (column_id == 0 && (to <= 300 || idx < 8 || (idx & 511) == 0)) {
      auto e = d_out[idx];
      printf("[prune] to=%d n=%d idx=%d q=%.9g rmin=%.9g rmax=%.9g wmin=%.9g v=%.9g\\n",
             (int)to, (int)in_column.size(), (int)idx, (double)q,
             (double)e.rmin, (double)e.rmax, (double)e.wmin, (double)e.value);
    }
  });"""

PRUNE_HOST_OLD = """  auto d_columns_ptr_in = this->columns_ptr_.ConstDeviceSpan();"""

PRUNE_HOST_NEW = """  fprintf(stderr, "[prune-host] col0_in=%d to=%d\\n", (int)this->Column(0).size(), (int)to);
  auto d_columns_ptr_in = this->columns_ptr_.ConstDeviceSpan();"""


def main() -> int:
    root = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "tiles"
    if mode == "tiles":
        path = f"{root}/src/tree/gpu_hist/evaluate_splits.cu"
        edit(path, ((TILE_OLD, TILE_NEW, "tile"), (NODE_OLD, NODE_NEW, "node")))
        print(f"instrumented {path} (2 probes)")
    elif mode == "sketch":
        path = f"{root}/src/common/quantile.cu"
        edit(path, ((PRUNE_OLD, PRUNE_NEW, "prune"), (PRUNE_HOST_OLD, PRUNE_HOST_NEW, "prune-host")))
        print(f"instrumented {path} (sketch probes)")
    else:
        print(f"unknown mode {mode}", file=sys.stderr)
        return 1
    # FindOpenMP's CUDA component fails under the VS generator on the runner;
    # OpenMP is host-side only and the probe pins nthread=1, so let
    # -DUSE_OPENMP=OFF actually stick instead of being FORCEd back on.
    edit(
        f"{root}/CMakeLists.txt",
        (('set(USE_OPENMP ON CACHE BOOL "CUDA requires OpenMP" FORCE)',
          '# [probe] USE_OPENMP left as passed (host-side only here)',
          "openmp-force"),),
    )
    print("neutralized the CUDA-forces-OpenMP cache line")
    return 0


def tagcount(s: str) -> int:
    return s.count("[tile]") + s.count("[node]")


if __name__ == "__main__":
    sys.exit(main())
