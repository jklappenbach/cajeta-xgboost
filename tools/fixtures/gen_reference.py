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

    # U10 large-dataset parity (spec §7.3). ≥100k rows, ≥50 features, controlled
    # missing fraction, depth 6 to exercise real tree depth + bin occupancy. These
    # are NOT committed by default — they are big (~60 MB each), so their storage
    # (git-lfs vs regenerate) is decided in U10. Generate on NVIDIA with e.g.
    #   python gen_reference.py --only large_reg,large_binary,large_multi
    # Every feature is continuous (>> max_bin distinct) so XGBoost prunes to 256
    # bins; the committed `bins`/`cut_*` are the §7.1 identical-bins ground truth.
    ("large_reg",    ["--objective", "reg:squarederror", "--rows", "100000", "--features", "50", "--missing-frac", "0.05", "--max-depth", "6"]),
    ("large_binary", ["--objective", "binary:logistic",  "--rows", "100000", "--features", "50", "--num-class", "2", "--missing-frac", "0.05", "--max-depth", "6"]),
    ("large_multi",  ["--objective", "multi:softprob",   "--rows", "100000", "--features", "50", "--num-class", "4", "--missing-frac", "0.05", "--max-depth", "6"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="cuda = the parity REFERENCE (run on NVIDIA); cpu = driver dry-run")
    ap.add_argument("--only", default="", metavar="NAMES",
                    help="comma-separated fixture names to (re)generate; default = all. "
                         "Use this to add new configs without re-touching the committed "
                         "reference fixtures, e.g. --only tiny_reg_d2,tiny_reg_d6,"
                         "tiny_reg_mcw10,tiny_reg_gamma1")
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    configs = [c for c in CONFIGS if not only or c[0] in only]
    if only:
        missing = only - {c[0] for c in CONFIGS}
        if missing:
            raise SystemExit(f"--only names not in CONFIGS: {sorted(missing)}")

    import extract_trees
    for name, extra in configs:
        out = os.path.join(HERE, name if args.device == "cuda" else name + "_cpu")
        cmd = [sys.executable, GEN, "--out", out, "--device", args.device] + extra
        print(f">> {name}  ({args.device})")
        subprocess.run(cmd, check=True)
        extract_trees.dump_trees(out)   # trees/ arrays for cajeta-side comparison
    print(f"\nDone. {len(configs)} fixtures under {HERE} (device={args.device}).")
    if args.device == "cuda":
        print("These are the REFERENCE. Commit + push them, then ping me to pull.")


if __name__ == "__main__":
    main()
