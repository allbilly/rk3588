#!/usr/bin/env python3
"""Capture all uncaptured fenced shapes via KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1/2.

Saves dumps to /home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem<N>/dump_gemN.txt
(NOT /tmp). Builds matched .rknn models via gen_conv2d_models.py --custom when missing.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path("/home/orangepi/rk3588")
OPS_RKNN = Path("/home/orangepi/npu/ops_rknn")
DUMP_ROOT = OPS_RKNN / "dump"
MODELS_DIR = OPS_RKNN / "models"
GEN_TOOL = OPS_RKNN / "gen_conv2d_models.py"
GDB_SCRIPT = WORKTREE / "conv_expt/gdb/rknn_prefix_replay.gdb"
CONV2D_MULTI = OPS_RKNN / "conv2d_multi"
SWEEP = WORKTREE / "sweep_results/conv_py_217_sweep_20260603_121116_summary.txt"


def parse_shape(shape):
    s = shape.replace("conv2d_cc_b1_", "b1_")
    m = re.match(r"b1_c(\d+)_h(\d+)_w(\d+)_oc(\d+)_wic(\d+)_k(\d+)x(\d+)_g(\d+)(?:_s1_pvalid)?$", s)
    if not m:
        return None
    c, h, w, oc, wic, kh, kw, g = m.groups()
    return dict(batch=1, in_c=int(c), in_h=int(h), in_w=int(w), out_c=int(oc),
                weight_in_c=int(wic), kh=int(kh), kw=int(kw), groups=int(g))


def model_name(shape):
    return f"match_{shape}"


def slug_from_shape(shape):
    s = shape.replace("conv2d_cc_b1_", "cc_b1_")
    m = re.match(r"b1_c(\d+)_h(\d+)_w\d+_oc(\d+)_wic(\d+)_k(\d+)x(\d+)_g(\d+)(?:_s1_pvalid)?$", s)
    if m:
        c, h, oc, wic, kh, kw, g = m.groups()
        suf = "_s1pvalid" if "_s1_pvalid" in s else ""
        return f"c{c}_h{h}_oc{oc}{suf}"
    m = re.match(r"cc_b1_c(\d+)_h(\d+)_w\d+_oc(\d+)_wic(\d+)_k(\d+)x(\d+)_g(\d+)(?:_s1_pvalid)?$", s)
    if m:
        c, h, oc = m.group(1), m.group(2), m.group(3)
        return f"cc_c{c}_h{h}_oc{oc}"
    return None


def build_matched(shape, force=False):
    p = parse_shape(shape)
    if p is None:
        return None
    name = model_name(shape)
    rknn_path = MODELS_DIR / f"{name}.rknn"
    if rknn_path.exists() and not force:
        return rknn_path
    cmd = ["python3", str(GEN_TOOL), "--custom",
           "--batch", str(p["batch"]), "--in-ch", str(p["in_c"]),
           "--out-ch", str(p["out_c"]), "--height", str(p["in_h"]),
           "--width", str(p["in_w"]), "--k-h", str(p["kh"]), "--k-w", str(p["kw"]),
           "--groups", str(p["groups"]), "--name", name,
           "--out-dir", str(MODELS_DIR)]
    if force:
        cmd.append("--force")
    print(f"  [build] {' '.join(cmd[:5])}... --name {name}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(f"  [build] FAILED: {r.stderr[:300]}")
        return None
    return rknn_path if rknn_path.exists() else None


def capture(shape, gem_n, timeout=120):
    rknn_path = MODELS_DIR / f"{model_name(shape)}.rknn"
    if not rknn_path.exists():
        print(f"  [capture gem={gem_n}] no rknn at {rknn_path}, skipping")
        return False
    slug = slug_from_shape(shape)
    if slug is None:
        print(f"  [capture gem={gem_n}] cannot derive slug, skipping")
        return False
    dump_dir = DUMP_ROOT / f"prefix_{slug}_keep1_gem{gem_n}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_file = dump_dir / f"dump_gem{gem_n}.txt"
    if dump_file.exists() and dump_file.stat().st_size > 100:
        print(f"  [capture gem={gem_n}] {dump_file.name} exists ({dump_file.stat().st_size}B), skipping")
        return True
    p = parse_shape(shape)
    case_args = [model_name(shape), str(p["batch"]), str(p["in_c"]),
                 str(p["in_h"]), str(p["in_w"]), str(p["out_c"]),
                 str(p["kh"]), str(p["kw"]), str(p["groups"])]
    env = os.environ.copy()
    env["KEEP_TASKS"] = "1"
    env["PREFIX_MODE"] = "linear"
    env["DUMP_GEM"] = str(gem_n)
    env["DUMP_DIR"] = str(dump_dir)
    # gdb script uses cwd to find dump.py AND ./conv2d_multi binary; both live in OPS_RKNN
    cwd_dir = OPS_RKNN
    cmd = ["gdb", "-q", "-batch", "-x", str(GDB_SCRIPT),
           "--args", str(CONV2D_MULTI), "--case"] + case_args
    print(f"  [capture gem={gem_n}] slug={slug} timeout={timeout}s")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(cwd_dir))
    except subprocess.TimeoutExpired:
        print(f"  [capture gem={gem_n}] TIMEOUT after {timeout}s")
        return False
    dt = time.time() - t0
    if not dump_file.exists() or dump_file.stat().st_size < 100:
        print(f"  [capture gem={gem_n}] FAILED dt={dt:.1f}s")
        return False
    print(f"  [capture gem={gem_n}] OK dt={dt:.1f}s size={dump_file.stat().st_size}B")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gem", type=int, default=0, help="0 = both 1+2, else N")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--shape", help="only process this one shape (debug)")
    ap.add_argument("--offset", type=int, default=0, help="skip first N shapes")
    args = ap.parse_args()

    fenced = []
    for line in SWEEP.read_text().splitlines():
        if line.startswith("FENCED "):
            fenced.append(line.split(" ", 1)[1])
    if args.shape:
        fenced = [args.shape]
    if args.offset:
        fenced = fenced[args.offset:]
    if args.limit:
        fenced = fenced[:args.limit]
    print(f"Processing {len(fenced)} shapes (offset={args.offset})")

    success = 0
    fail = 0
    for i, shape in enumerate(fenced, 1):
        slug = slug_from_shape(shape)
        print(f"[{i}/{len(fenced)}] {shape} (slug={slug})")
        if slug is None:
            print(f"  SKIP: cannot derive slug")
            fail += 1
            continue
        if not args.skip_build:
            rknn = build_matched(shape)
            if rknn is None:
                print(f"  SKIP: build failed")
                fail += 1
                continue
        gems = [1, 2] if args.gem == 0 else [args.gem]
        for gem_n in gems:
            if args.skip_capture:
                continue
            ok = capture(shape, gem_n)
            if ok:
                success += 1
            else:
                fail += 1
        if i % 10 == 0:
            print(f"  [health] simple_add.py check")
            r = subprocess.run(["python3", str(WORKTREE / "examples/simple_add.py")],
                               capture_output=True, text=True, timeout=30)
            print(f"  [health] rc={r.returncode} {'PASS' if 'PASS' in r.stdout else 'FAIL'}")
    print(f"\nDone: success={success} fail={fail}")


if __name__ == "__main__":
    main()
