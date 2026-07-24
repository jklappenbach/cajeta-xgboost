#!/usr/bin/env python3
"""Extract each reference tree's node arrays from a fixture's model.json into
`.npy`, so cajeta compares against them via `Npy` (no JSON parser needed on the
cajeta side). Runs on ANY machine — it only reads the committed model.json, no
GPU/xgboost required.

Per tree `t<idx>` under `<fixture>/trees/`:
  t<idx>_split_indices.npy    int32   feature per node (leaf: 0)
  t<idx>_split_conditions.npy float32 threshold per node (leaf: leaf value)
  t<idx>_default_left.npy      int32   missing-goes-left flag
  t<idx>_left_children.npy     int32   left child (leaf: -1)
  t<idx>_right_children.npy    int32   right child (leaf: -1)
  t<idx>_base_weights.npy      float32 node weight (leaf: the leaf weight)
  t<idx>_sum_hessian.npy       float64 hessian sum per node

XGBoost stores split_conditions / base_weights as float32 — the dtypes here
match, so cajeta compares bit-for-bit at the model's own precision.
"""
import json, os, sys
import numpy as np

FIELDS = {
    "split_indices": np.int32, "split_conditions": np.float32,
    "default_left": np.int32, "left_children": np.int32,
    "right_children": np.int32, "base_weights": np.float32,
    "sum_hessian": np.float64,
}


def dump_trees(fixture_dir):
    model = json.load(open(os.path.join(fixture_dir, "model.json")))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    out = os.path.join(fixture_dir, "trees")
    os.makedirs(out, exist_ok=True)
    for idx, t in enumerate(trees):
        for field, dt in FIELDS.items():
            arr = np.asarray(t[field], dtype=dt)
            np.save(os.path.join(out, f"t{idx}_{field}.npy"), arr)
    # a tiny count file so the loader knows how many trees
    np.save(os.path.join(out, "num_trees.npy"), np.asarray([len(trees)], dtype=np.int32))
    return len(trees)


def main():
    dirs = sys.argv[1:] or [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), d)
        for d in ("tiny_reg", "tiny_reg_missing", "tiny_binary", "tiny_multi")
    ]
    for d in dirs:
        if os.path.exists(os.path.join(d, "model.json")):
            n = dump_trees(d)
            print(f"  {d}: {n} trees -> trees/")
        else:
            print(f"  {d}: no model.json, skipped")


if __name__ == "__main__":
    main()
