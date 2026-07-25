#!/usr/bin/env python3
"""Fetch a real public benchmark and emit a golden parity fixture (plan U10.1.2).

Downloads the **covertype** dataset (7-class forest cover, 54 features) via
scikit-learn's `fetch_covtype` (which pulls from its upstream source — nothing
proprietary is redistributed in this repo), takes a fixed-seed subset, and trains
the PINNED reference XGBoost through the same `generate.py` code path as the
synthetic fixtures, so the fixture format is identical.

Run this ON THE NVIDIA BOX (the parity reference is the GPU algorithm):

    python fetch_benchmark.py                       # cuda, 50k-row subset
    python fetch_benchmark.py --rows 20000          # smaller
    python fetch_benchmark.py --device cpu          # driver dry-run (NOT a reference)

It writes `tools/fixtures/covertype/` (multi:softprob, num_class=7). The parity
TEST that consumes it is wired later — this script only produces the ground truth
so the fixture is ready the moment we enable large-multiclass parity (which is
also gated on the U12 `expf` link for rounds 1+; round-0 trees are bit-exact).

Storage: covertype is continuous+categorical with many distinct values, so a 50k
subset is a few MB per array — commit plainly alongside the synthetic large_*.
"""
import argparse, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generate.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000, help="subset size (fixed-seed sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="cuda = the parity REFERENCE (NVIDIA); cpu = driver dry-run only")
    ap.add_argument("--num-round", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    from sklearn.datasets import fetch_covtype
    print(">> fetching covertype (scikit-learn)")
    ds = fetch_covtype()
    X_all = np.asarray(ds.data, dtype=np.float64)
    y_all = np.asarray(ds.target, dtype=np.int64) - 1        # 1..7 -> 0..6

    rng = np.random.default_rng(args.seed)
    n = min(args.rows, X_all.shape[0])
    idx = rng.choice(X_all.shape[0], size=n, replace=False)
    X = X_all[idx]
    y = y_all[idx].astype(np.float64)
    print(f">> subset: {X.shape[0]} rows x {X.shape[1]} features, {int(y.max()) + 1} classes")

    out = os.path.join(HERE, "covertype" if args.device == "cuda" else "covertype_cpu")
    os.makedirs(out, exist_ok=True)
    xpath = os.path.join(out, "_X_input.npy")
    ypath = os.path.join(out, "_y_input.npy")
    np.save(xpath, X)
    np.save(ypath, y)

    cmd = [sys.executable, GEN, "--out", out, "--device", args.device,
           "--objective", "multi:softprob", "--num-class", "7",
           "--x-npy", xpath, "--y-npy", ypath,
           "--num-round", str(args.num_round), "--max-depth", str(args.max_depth)]
    subprocess.run(cmd, check=True)

    import extract_trees
    sys.path.insert(0, HERE)
    extract_trees.dump_trees(out)
    os.remove(xpath)
    os.remove(ypath)
    print(f"\nDone -> {out} (device={args.device}). "
          + ("REFERENCE: commit + push, then ping me." if args.device == "cuda"
             else "cpu dry-run — NOT a parity reference."))


if __name__ == "__main__":
    main()
