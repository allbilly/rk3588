#!/usr/bin/env python3
"""Sweep the 217 conv_new.py shapes through conv_grok.conv.run_shape.

Usage (from repo root):
  python3 conv_grok/sweep_217.py --classify
  python3 conv_grok/sweep_217.py --sweep [--limit N] [--timeout SEC] [--start N] [--pattern REGEX]
  python3 conv_grok/conv.py --sweep ...   # thin redirect into this module
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sweep_217 import collect_shapes, normalize_shape_name  # noqa: E402

from conv_grok.conv import (  # noqa: E402
    _is_depthwise,
    plan_depthwise_rows,
    plan_local_serial_rows,
    run_shape,
    shape_from_name,
)

SLOW_DW_TASK_THRESHOLD = 500
DEFAULT_TIMEOUT = 180
HEALTH_SCRIPT = "examples/simple_add.py"


def shape_dict_for_run(raw: dict) -> dict:
    """Build a run_shape-ready dict from a conv_new.py shape entry."""
    name = normalize_shape_name(raw)
    s = shape_from_name(name) if (name.startswith("b") or name.startswith("conv2d_b")) else None
    if s is None:
        s = dict(
            name=name,
            batch=int(raw["batch"]),
            in_c=int(raw["in_c"]),
            in_h=int(raw["in_h"]),
            in_w=int(raw["in_w"]),
            out_c=int(raw["out_c"]),
            weight_in_c=int(raw["weight_in_c"]),
            kh=int(raw["kh"]),
            kw=int(raw["kw"]),
            groups=int(raw["groups"]),
            stride=int(raw.get("stride", 1)),
        )
    else:
        # Prefer encoded name; keep stride from raw if present.
        s["stride"] = int(raw.get("stride", s.get("stride", 1)))
        s["name"] = name
    return s


def estimate_depthwise_tasks(s: dict) -> int:
    """Serial depthwise (current): channels × Y windows."""
    base = dict(s, name=s["name"] + "_dw_est", batch=1, in_c=1, out_c=2, weight_in_c=1, groups=1)
    y_rows, _, _, _ = plan_local_serial_rows(base)
    return s["batch"] * s["out_c"] * max(1, len(y_rows))


def classify_one(s: dict) -> tuple[str, int, str]:
    """Return (path_or_split, est_tasks, note) without touching the NPU."""
    if _is_depthwise(s["in_c"], s["out_c"], s["groups"]):
        tasks = estimate_depthwise_tasks(s)
        note = f"SLOW_DW est_tasks={tasks}" if tasks > SLOW_DW_TASK_THRESHOLD else ""
        return "depthwise_serial", tasks, note
    if s["groups"] != 1:
        in_per = s["in_c"] // s["groups"]
        out_per = s["out_c"] // s["groups"]
        gshape = dict(s, name=s["name"] + "_g0", batch=1, in_c=in_per, out_c=out_per,
                      weight_in_c=in_per, groups=1)
        rows, _, _, _ = plan_local_serial_rows(gshape)
        tasks = s["batch"] * s["groups"] * len(rows)
        return "grouped_serial", tasks, ""
    rows, _, _, _ = plan_local_serial_rows(s)
    return rows[0]["split_method"], len(rows) * s["batch"], ""


def run_health() -> int:
    return subprocess.run(
        [sys.executable, HEALTH_SCRIPT], cwd=REPO_ROOT, capture_output=True
    ).returncode


def run_one_inprocess(s: dict, timeout: int) -> tuple[str, str]:
    """Import run_shape in-process. Soft wall-clock only (do not kill NPU jobs)."""
    path, est_tasks, note = classify_one(s)
    if path == "depthwise_serial" and est_tasks > SLOW_DW_TASK_THRESHOLD:
        # Still try, but use the generous timeout budget; annotate detail.
        pass
    t0 = time.time()
    try:
        ok, max_diff, tasks, kind = run_shape(s, dry_run=False)
        elapsed = time.time() - t0
        if elapsed > timeout:
            # Completed but over budget — still report real result, note overrun.
            detail = f"{kind} tasks={tasks} max_diff={max_diff:.4f} elapsed={elapsed:.1f}s>{timeout}s {note}"
            return ("PASS" if ok else "FAIL"), detail.strip()
        detail = f"{kind} tasks={tasks} max_diff={max_diff:.4f} elapsed={elapsed:.1f}s {note}".strip()
        return ("PASS" if ok else "FAIL"), detail
    except Exception as exc:
        elapsed = time.time() - t0
        err = f"{type(exc).__name__}: {exc} elapsed={elapsed:.1f}s {note}".strip()
        if elapsed > timeout:
            return "TIMEOUT", err
        return "ERROR", err


def filter_shapes(shapes: list[dict], start: int, limit: int, pattern: str) -> list[dict]:
    out = shapes[start:]
    if pattern:
        rx = re.compile(pattern)
        out = [s for s in out if rx.search(s["name"]) or rx.search(normalize_shape_name(s))]
    if limit:
        out = out[:limit]
    return out


def do_classify(shapes: list[dict]) -> int:
    hist: Counter[str] = Counter()
    slow = []
    for raw in shapes:
        s = shape_dict_for_run(raw)
        path, tasks, note = classify_one(s)
        hist[path] += 1
        if note:
            slow.append((s["name"], path, tasks, note))
        print(f"{path:18s} tasks~{tasks:5d}  {s['name']}" + (f"  ({note})" if note else ""))
    print()
    print(f"total={len(shapes)} split_histogram={dict(hist)}")
    if slow:
        print(f"slow_depthwise={len(slow)} (est_tasks>{SLOW_DW_TASK_THRESHOLD}):")
        for name, path, tasks, note in slow[:20]:
            print(f"  {note}  {name}")
        if len(slow) > 20:
            print(f"  ... {len(slow) - 20} more")
    return 0


def do_sweep(shapes: list[dict], timeout: int, skip_health: bool, stop_on_error: bool) -> int:
    out_dir = REPO_ROOT / "sweep_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"conv_grok_217_sweep_{timestamp}_summary.txt"
    log_path = out_dir / f"conv_grok_217_sweep_{timestamp}_detail.log"

    pre_health_rc = post_health_rc = -1
    if not skip_health:
        print(f"running pre_health {HEALTH_SCRIPT} ...")
        pre_health_rc = run_health()
        print(f"pre_health_rc={pre_health_rc}")

    counts: Counter[str] = Counter()
    by_status: dict[str, list[str]] = {k: [] for k in ("PASS", "FAIL", "ERROR", "TIMEOUT", "FENCED")}
    detail_log: list[str] = []
    started = time.time()

    for idx, raw in enumerate(shapes, 1):
        s = shape_dict_for_run(raw)
        path, est, note = classify_one(s)
        print(
            f"[{idx:3d}/{len(shapes)}] {s['name']} "
            f"(b={s['batch']} c={s['in_c']} h={s['in_h']}x{s['in_w']} oc={s['out_c']} "
            f"k={s['kh']}x{s['kw']} g={s['groups']} path={path} est~{est}"
            f"{' ' + note if note else ''}) ...",
            flush=True,
        )
        status, line = run_one_inprocess(s, timeout)
        counts[status] += 1
        by_status.setdefault(status, []).append(s["name"])
        detail_log.append(f"{status:8s} {s['name']} :: {line}")
        elapsed = time.time() - started
        print(
            f"           {status} running={dict(counts)} elapsed={elapsed:6.1f}s  {line[:200]}",
            flush=True,
        )
        if stop_on_error and status in ("ERROR", "TIMEOUT"):
            print("stop_on_error triggered; aborting sweep")
            break

    if not skip_health:
        print(f"running post_health {HEALTH_SCRIPT} ...")
        post_health_rc = run_health()
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
    return 0 if counts.get("FAIL", 0) == 0 and counts.get("ERROR", 0) == 0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="conv_grok 217-shape sweep / classify")
    parser.add_argument("--sweep", action="store_true", help="run shapes on NPU via run_shape")
    parser.add_argument("--classify", action="store_true", help="dry-run planner split histogram (no NPU)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-shape soft wall-clock (s)")
    parser.add_argument("--limit", type=int, default=0, help="only first N shapes after --start")
    parser.add_argument("--start", type=int, default=0, help="skip first N shapes")
    parser.add_argument("--pattern", type=str, default="", help="regex filter on shape name")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)

    if not args.sweep and not args.classify:
        parser.error("specify --sweep and/or --classify")

    shapes = filter_shapes(collect_shapes(), args.start, args.limit, args.pattern)
    print(f"total_shapes={len(shapes)} (from 217 pool, start={args.start} limit={args.limit or 'all'})")

    rc = 0
    if args.classify:
        rc = do_classify(shapes) or rc
    if args.sweep:
        rc = do_sweep(shapes, args.timeout, args.skip_health, args.stop_on_error) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
