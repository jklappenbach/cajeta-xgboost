#!/usr/bin/env python3
"""Retrain a fixture on the real GPU and report selected split nodes.

gpu-numeric-fidelity 2.2.1 follow-up: from the committed node-14 dump, both
the port and a verbatim device replay of the evaluate agent score feature 1
(96.706) above the reference's recorded feature 4 (96.521), and CPU-hist 3.3.0
also picks feature 1. This retrains the fixture with the ACTUAL pip wheel on
the ACTUAL GPU to (a) confirm the recorded reference reproduces today on
3.1.2, and (b) see whether the pick changed by 3.3.0 (run once per version).

    python train_repro.py <fixture-dir> [nodes...]
"""
import json
import sys

import numpy as np
import xgboost as xgb


def main() -> int:
    fdir = sys.argv[1]
    nodes_of_interest = [int(a) for a in sys.argv[2:]] or [5, 13, 14]

    X = np.load(f"{fdir}/X.npy")
    y = np.load(f"{fdir}/y.npy")
    params = json.load(open(f"{fdir}/manifest.json"))["params"]
    d = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, d, num_boost_round=3)

    print(f"xgboost {xgb.__version__}  device={params.get('device')}")
    tree = json.loads(bst.get_dump(dump_format="json")[0])
    flat = {}

    def walk(node):
        flat[node["nodeid"]] = node
        for c in node.get("children", []):
            walk(c)

    walk(tree)
    for nid in nodes_of_interest:
        n = flat.get(nid)
        if n is None:
            print(f"node {nid}: MISSING")
        elif "split" in n:
            print(f"node {nid}: split={n['split']} cond={n['split_condition']!r}")
        else:
            print(f"node {nid}: leaf={n['leaf']!r}")

    # Full comparison against the recorded reference arrays, if present.
    try:
        ref_si = np.load(f"{fdir}/trees/t0_split_indices.npy")
        ref_lc = np.load(f"{fdir}/trees/t0_left_children.npy")
        mismatches = []
        for nid in range(len(ref_si)):
            n = flat.get(nid)
            if n is None:
                mismatches.append((nid, "missing"))
                continue
            if ref_lc[nid] != -1:  # reference split node
                got = int(n["split"].lstrip("f")) if "split" in n else -1
                if got != int(ref_si[nid]):
                    mismatches.append((nid, f"split {got} vs ref {int(ref_si[nid])}"))
        print(f"vs recorded reference: {len(mismatches)} split mismatches: {mismatches}")
    except FileNotFoundError:
        print("no recorded reference arrays found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
