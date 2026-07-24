#!/usr/bin/env python3
"""Generate the full set of golden reference fixtures in one command.

Run this ON AN NVIDIA BOX (default --device cuda) — the outputs are the parity
ground truth cajeta must reproduce bit-for-bit. Whatever xgboost version is
installed becomes the PINNED reference (recorded in each manifest); parity is
version-specific, so keep it fixed once we lock in.

    python tools/fixtures/gen_reference.py                 # cuda (the reference)
    python tools/fixtures/gen_reference.py --device cpu    # local dry-run of the driver

Commits: the tiny/medium fixtures here are small and committed. Large-dataset
fixtures (U10) are intentionally NOT in this set — their storage (git-lfs vs.
regenerate-in-CI) is decided in U10.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generate.py")

# (name, extra generate.py args). Small enough to commit; covers the v1
# objectives + a missing-value case for the tree-builder units (U3–U7).
CONFIGS = [
    ("tiny_reg",         ["--objective", "reg:squarederror", "--rows", "200",  "--features", "5"]),
    ("tiny_reg_missing", ["--objective", "reg:squarederror", "--rows", "500",  "--features", "8", "--missing-frac", "0.15"]),
    ("tiny_binary",      ["--objective", "binary:logistic",  "--rows", "500",  "--features", "8", "--num-class", "2"]),
    ("tiny_multi",       ["--objective", "multi:softprob",   "--rows", "600",  "--features", "8", "--num-class", "4"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="cuda = the parity REFERENCE (run on NVIDIA); cpu = driver dry-run")
    args = ap.parse_args()

    for name, extra in CONFIGS:
        out = os.path.join(HERE, name if args.device == "cuda" else name + "_cpu")
        cmd = [sys.executable, GEN, "--out", out, "--device", args.device] + extra
        print(f">> {name}  ({args.device})")
        subprocess.run(cmd, check=True)
    print(f"\nDone. {len(CONFIGS)} fixtures under {HERE} (device={args.device}).")
    if args.device == "cuda":
        print("These are the REFERENCE. Commit + push them, then ping me to pull.")


if __name__ == "__main__":
    main()
