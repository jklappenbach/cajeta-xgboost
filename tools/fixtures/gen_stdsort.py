#!/usr/bin/env python3
"""Regenerate the stdsort fixture: libstdc++'s std::sort and std::partial_sort permutations of
tie-heavy (value, tag) queues, the ground truth for Sketch.stdSort (xgboost-data-prep Unit 3).

Usage: python3 gen_stdsort.py [--cxx g++]   (writes ./stdsort/)
"""
import argparse, os, struct, subprocess, tempfile
import numpy as np

SIZES = [1, 2, 3, 4, 5, 15, 16, 17, 18, 31, 32, 33, 64, 100, 257, 1000, 4097]


def zeros(rng, k):
    z = np.zeros(k, dtype=np.float32)
    z[rng.random(k) < 0.5] = np.float32(-0.0)
    return z


def patterns(rng, n):
    a = rng.uniform(-3, 3, n).astype(np.float32)
    m = rng.random(n) < 0.4
    a[m] = zeros(rng, int(m.sum()))
    yield a
    b = (rng.integers(0, 5, n) - 2).astype(np.float32)
    z = b == 0
    b[z] = zeros(rng, int(z.sum()))
    yield b
    yield zeros(rng, n)
    c = np.sort(b, kind="stable")
    yield c
    yield c[::-1].copy()
    d = np.concatenate([np.arange(n // 2), np.arange(n - n // 2)[::-1]]).astype(np.float32) % 3 - 1
    zz = d == 0
    d[zz] = zeros(rng, int(zz.sum()))
    yield d


def run(cxx_bin, mode, cases):
    payload = bytearray()
    for v in cases:
        payload += struct.pack("<q", len(v))
        bits = v.view(np.int32)
        for i in range(len(v)):
            payload += struct.pack("<ii", int(bits[i]), i)
    args = [cxx_bin] + ([mode] if mode else [])
    out = subprocess.run(args, input=bytes(payload), capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int32).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "stdsort"))
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, "stdsort_ref")
        subprocess.run([args.cxx, "-O2", "-std=c++17", "-o", exe, os.path.join(here, "stdsort_ref.cpp")], check=True)
        rng = np.random.default_rng(20261017)
        cases = [p for n in SIZES for p in patterns(rng, n)]
        for n in (100, 1000, 4097):
            kv = np.frombuffer(subprocess.run([exe, "killer", str(n)], capture_output=True, check=True).stdout, dtype=np.int32)
            cases.append(kv.astype(np.float32))
            t = (kv // 3 - n // 6).astype(np.float32)
            zt = t == 0
            t[zt] = zeros(rng, int(zt.sum()))
            cases.append(t)
        os.makedirs(args.out, exist_ok=True)
        offs = np.zeros(len(cases) + 1, dtype=np.int64)
        offs[1:] = np.cumsum([len(c) for c in cases])
        np.save(os.path.join(args.out, "offsets.npy"), offs)
        np.save(os.path.join(args.out, "in_bits.npy"), np.concatenate(cases).view(np.int32))
        for mode, name in ((None, "sort"), ("partial", "partial")):
            r = run(exe, mode, cases)
            np.save(os.path.join(args.out, f"{name}_bits.npy"), np.ascontiguousarray(r[:, 0]))
            np.save(os.path.join(args.out, f"{name}_tags.npy"), np.ascontiguousarray(r[:, 1]))
        ver = subprocess.run([args.cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            f.write('{"generator": "gen_stdsort.py", "cxx": "%s", "cases": %d, "entries": %d}\n' % (ver, len(cases), int(offs[-1])))
        print(len(cases), "cases,", int(offs[-1]), "entries")


if __name__ == "__main__":
    main()
