#!/usr/bin/env python3
"""Sweep harness that runs every literal test shape in conv_new.py through
``examples/conv.py`` and records per-shape status.

This script intentionally does not modify any NPU state itself; it shells out to
``python examples/conv.py <shape>`` and classifies the output. The status
categories match the ones used in conv_plan.md:

    PASS    - shape ran on the NPU and produced correct output
    FENCED  - shape was rejected before allocation with a "fenced" message
    FAIL    - shape ran on the NPU but the output was wrong
    ERROR   - Python exception other than the "fenced" ValueError
    TIMEOUT - shape ran past the per-shape wall-clock budget
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def collect_shapes() -> list[dict]:
    """Re-use the literal shape list from conv_new.py.

    conv_new.py keeps the list inside its __main__ block, so we extract the
    literal source slice and eval it in a sandbox that only permits dict
    literals and arithmetic.
    """
    src_path = REPO_ROOT / "examples" / "kernel_6_18" / "conv_new.py"
    text = src_path.read_text()
    start = text.find("shapes = [")
    if start < 0:
        raise RuntimeError("could not find `shapes = [` in conv_new.py")
    depth = 0
    end = None
    cursor = text.find("[", start)
    if cursor < 0:
        raise RuntimeError("`shapes = [` had no opening bracket")
    bracket_open = cursor
    while cursor < len(text):
        ch = text[cursor]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = cursor + 1
                break
        cursor += 1
    if end is None:
        raise RuntimeError("could not find end of `shapes` list")
    literal_src = text[bracket_open:end]
    safe_globals = {"__builtins__": {}, "dict": dict, "range": range, "True": True, "False": False}
    safe_locals: dict = {}
    shapes = eval(literal_src, safe_globals, safe_locals)
    return list(shapes)


def normalize_shape_name(shape: dict) -> str:
    """Map a conv_new.py shape dict to a name that examples/conv.py accepts.

    ``examples/conv.py`` uses an encoded ``b..c..h..w..oc..wic..k..g..``
    name. Some conv_new.py entries use a short descriptive name like
    ``conv2d_1x6_1x1_4x4``; for the sweep we still want to run them, so we
    derive an encoded alias and stash the original on the dict.
    """
    name = shape["name"]
    if name.startswith("b") or name.startswith("conv2d_b"):
        return name
    # conv1d_*_as_conv2d shapes and conv2d_<ic>x<oc>_<kh>x<kw>_<h>x<w> shapes
    # both need encoding.
    return f"b{shape['batch']}_c{shape['in_c']}_h{shape['in_h']}_w{shape['in_w']}_oc{shape['out_c']}_wic{shape['weight_in_c']}_k{shape['kh']}x{shape['kw']}_g{shape['groups']}"


def run_one(shape_name: str, timeout: int) -> tuple[str, str]:
    """Run conv.py on a single shape and return (status, line).

    Status is one of: PASS, FENCED, FAIL, ERROR, TIMEOUT.
    Line is the first interesting output line for debugging.
    """
    # If the original name is not in the encoded b..c..h.. format, use the
    # encoded alias that we synthesized in normalize_shape_name.
    if not (shape_name.startswith("b") or shape_name.startswith("conv2d_b")):
        for s in collect_shapes():
            if s["name"] == shape_name:
                shape_name = normalize_shape_name(s)
                break
    cmd = [sys.executable, "examples/conv.py", shape_name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"timeout after {timeout}s"

    out = proc.stdout + proc.stderr
    lines = [ln for ln in out.splitlines() if ln.strip()]
    joined = "\n".join(lines)

    if "PASS" in joined:
        return "PASS", "\n".join(lines[-3:]) if lines else ""
    if "fenced" in joined.lower() or "disabled" in joined.lower():
        return "FENCED", lines[-1] if lines else ""
    if "FAIL" in joined:
        return "FAIL", "\n".join(lines[-3:]) if lines else ""
    if proc.returncode != 0:
        return "ERROR", (lines[-1] if lines else f"rc={proc.returncode}")
    return "ERROR", f"no PASS/FAIL token; rc={proc.returncode}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N shapes (for smoke tests)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "sweep_results")
    parser.add_argument("--shapes-file", type=Path, help="optional list of shape names to run")
    parser.add_argument("--skip-health", action="store_true", help="skip the pre/post simple_add.py health check")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--filter", type=str, default="", help="regex to filter shape names (e.g. '^conv2d_\\\\d')")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"conv_py_217_sweep_{timestamp}_summary.txt"
    log_path = args.output_dir / f"conv_py_217_sweep_{timestamp}_detail.log"

    if args.shapes_file:
        wanted = {line.strip() for line in args.shapes_file.read_text().splitlines() if line.strip()}
        shapes = [s for s in collect_shapes() if s["name"] in wanted]
        print(f"running {len(shapes)} named shapes from {args.shapes_file}")
    else:
        shapes = collect_shapes()
    if args.limit:
        shapes = shapes[:args.limit]

    print(f"total_shapes={len(shapes)}")

    pre_health_rc = -1
    post_health_rc = -1
    if not args.skip_health:
        print("running pre_health examples/simple_add.py ...")
        pre = subprocess.run([sys.executable, "examples/simple_add.py"], cwd=REPO_ROOT, capture_output=True)
        pre_health_rc = pre.returncode
        print(f"pre_health_rc={pre_health_rc}")

    counts: Counter[str] = Counter()
    by_status: dict[str, list[str]] = {k: [] for k in ("PASS", "FENCED", "FAIL", "ERROR", "TIMEOUT")}
    detail_log: list[str] = []

    started = time.time()
    for idx, shape in enumerate(shapes, 1):
        name = shape["name"]
        if args.filter and not re.search(args.filter, name):
            continue
        batch = shape.get("batch", 1)
        in_c = shape["in_c"]
        in_h = shape["in_h"]
        in_w = shape["in_w"]
        out_c = shape["out_c"]
        kh, kw, groups = shape["kh"], shape["kw"], shape["groups"]
        elapsed = time.time() - started
        print(f"[{idx:3d}/{len(shapes)}] {name} (b={batch} c={in_c} h={in_h}x{in_w} oc={out_c} k={kh}x{kw} g={groups}) ...", flush=True)
        status, line = run_one(name, args.timeout)
        counts[status] += 1
        by_status[status].append(name)
        detail_log.append(f"{status:8s} {name} :: {line}")
        print(f"           {status} (elapsed {elapsed:6.1f}s)  {line[:200]}")
        if args.stop_on_error and status in ("ERROR", "TIMEOUT"):
            print("stop_on_error triggered; aborting sweep")
            break

    if not args.skip_health:
        print("running post_health examples/simple_add.py ...")
        post = subprocess.run([sys.executable, "examples/simple_add.py"], cwd=REPO_ROOT, capture_output=True)
        post_health_rc = post.returncode
        print(f"post_health_rc={post_health_rc}")

    total = sum(counts.values())
    counts_repr = dict(counts)
    print()
    print(f"summary={summary_path}")
    print(f"pre_health_rc={pre_health_rc} post_health_rc={post_health_rc} total={total} counts={counts_repr}")

    with summary_path.open("w") as f:
        f.write(f"timestamp={timestamp}\n")
        f.write(f"total={total} counts={counts_repr}\n")
        f.write(f"pre_health_rc={pre_health_rc} post_health_rc={post_health_rc}\n")
        for status, names in by_status.items():
            for name in names:
                f.write(f"{status} {name}\n")
    with log_path.open("w") as f:
        f.write("\n".join(detail_log) + "\n")
    print(f"detail_log={log_path}")


if __name__ == "__main__":
    main()
