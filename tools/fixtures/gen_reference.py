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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generate.py")

# (name, extra generate.py args). Small enough to commit; covers the v1
# objectives + a missing-value case for the tree-builder units (U3–U7).
CONFIGS = [
    ("tiny_reg",         ["--objective", "reg:squarederror", "--rows", "200",  "--features", "5"]),
    ("tiny_reg_missing", ["--objective", "reg:squarederror", "--rows", "500",  "--features", "8", "--missing-frac", "0.15"]),
    ("tiny_binary",      ["--objective", "binary:logistic",  "--rows", "500",  "--features", "8", "--num-class", "2"]),
    ("tiny_multi",       ["--objective", "multi:softprob",   "--rows", "600",  "--features", "8", "--num-class", "4"]),
    # U4 config-variants of tiny_reg — same data, one regularization knob moved
    # each, so a single tree can be asserted bit-identical across depth/λ/γ/mcw
    # (plan 4.1.2 / 4.3.1). ConfigParityTest reads these; it stays @Disabled until
    # they are regenerated on NVIDIA and committed.
    ("tiny_reg_d2",      ["--objective", "reg:squarederror", "--rows", "200",  "--features", "5", "--max-depth", "2"]),
    ("tiny_reg_d6",      ["--objective", "reg:squarederror", "--rows", "200",  "--features", "5", "--max-depth", "6"]),
    ("tiny_reg_mcw10",   ["--objective", "reg:squarederror", "--rows", "200",  "--features", "5", "--min-child-weight", "10"]),
    ("tiny_reg_gamma1",  ["--objective", "reg:squarederror", "--rows", "200",  "--features", "5", "--gamma", "1.0"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="cuda = the parity REFERENCE (run on NVIDIA); cpu = driver dry-run")
    args = ap.parse_args()

    import extract_trees
    for name, extra in CONFIGS:
        out = os.path.join(HERE, name if args.device == "cuda" else name + "_cpu")
        cmd = [sys.executable, GEN, "--out", out, "--device", args.device] + extra
        print(f">> {name}  ({args.device})")
        subprocess.run(cmd, check=True)
        extract_trees.dump_trees(out)   # trees/ arrays for cajeta-side comparison
    print(f"\nDone. {len(CONFIGS)} fixtures under {HERE} (device={args.device}).")
    if args.device == "cuda":
        print("These are the REFERENCE. Commit + push them, then ping me to pull.")


if __name__ == "__main__":
    main()
