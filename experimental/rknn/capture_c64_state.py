#!/usr/bin/env python3
"""Capture RKNN/raw C64/H56 state with explicit safety gates.

Default mode is dry-run: it prints the commands without submitting work.
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experimental" / "rknn"
RKNN_MODEL = Path("/home/orangepi/npu/ops_rknn/models/conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1.rknn")
RKNN_RUNNER = Path("/home/orangepi/npu/ops_rknn/run_generic")
C64_SHAPE = "b1_c64_h56_w56_oc128_wic64_k1x1_g1"


def cmd_text(cmd, env=None):
    prefix = ""
    if env:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
    return prefix + " ".join(shlex.quote(str(part)) for part in cmd)


def run(cmd, *, env=None, log=None, execute=False):
    print(f"$ {cmd_text(cmd, env)}")
    if not execute:
        return 0
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    if log is None:
        return subprocess.call(cmd, cwd=ROOT, env=proc_env)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"$ {cmd_text(cmd, env)}\n")
        fh.flush()
        return subprocess.call(cmd, cwd=ROOT, env=proc_env, stdout=fh, stderr=subprocess.STDOUT)


def tee_cmd(cmd, log):
    return ["bash", "-lc", f"{cmd_text(cmd)} 2>&1 | tee -a {shlex.quote(str(log))}"]


def snapshot_commands(label):
    return [
        [sys.executable, "experimental/rknn/snapshot_rknpu_sysfs.py", "--label", label],
        [sys.executable, "experimental/rknn/snapshot_rknpu_mmio_readonly.py", "--label", label],
    ]


def rknn_command(mmio, sysfs):
    env = {}
    if mmio:
        env["RKNPU_MMIO_SNAPSHOT"] = "1"
    if sysfs:
        env["RKNPU_SYSFS_SNAPSHOT"] = "1"
    return [
        "gdb", "-q", "-batch",
        "-x", "experimental/rknn/capture_rknpu_ioctl_readonly.gdb",
        "--args", str(RKNN_RUNNER), str(RKNN_MODEL),
    ], env


def raw_command():
    env = {
        "RK3588_CONV_DIRECT_SPATIAL": "1",
        "RK3588_CONV_DIRECT_SPATIAL_UNSAFE": "1",
        "RK3588_CONV_C64_H56_SPARSE_UNSAFE": "1",
        "RK3588_CONV_DIRECT_SPATIAL_TASKS": "rknpu_sparse_task_gem",
        "RK3588_CONV_RKNN_MEM_SYNC": "1",
        "RK3588_CONV_RKNN_SKIP_RESET": "1",
        "RK3588_CONV_RKNN_INIT_ACTIONS": "1",
    }
    return [sys.executable, "examples/conv_tiles.py", C64_SHAPE], env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rknn", "raw"), required=True)
    parser.add_argument("--execute", action="store_true", help="actually run commands; default prints only")
    parser.add_argument("--allow-raw-c64", action="store_true", help="required with --execute --mode raw")
    parser.add_argument("--mmio", action="store_true", help="request /dev/mem MMIO snapshots")
    parser.add_argument("--sysfs", action="store_true", help="request sysfs snapshots")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    if args.mode == "raw" and args.execute and not args.allow_raw_c64:
        raise SystemExit("--execute --mode raw requires --allow-raw-c64")

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    log = OUT_DIR / f"capture_c64_state_{args.mode}_{tag}.log"
    print(f"log={log}")
    print("dry_run=" + str(not args.execute))

    if args.mode == "rknn":
        cmd, env = rknn_command(args.mmio, args.sysfs)
        return run(cmd, env=env, log=log, execute=args.execute)

    rc = 0
    for cmd in snapshot_commands("before_raw_c64"):
        rc = run(cmd, log=log, execute=args.execute) or rc
    cmd, env = raw_command()
    rc = run(cmd, env=env, log=log, execute=args.execute) or rc
    for cmd in snapshot_commands("after_raw_c64"):
        rc = run(cmd, log=log, execute=args.execute) or rc
    print("# Optional next-job check, known to timeout while C64/H56 remains polluted:")
    print("# " + cmd_text([sys.executable, "examples/simple_add.py"]))
    for cmd in snapshot_commands("after_next_job_or_recovery"):
        print("# " + cmd_text(cmd))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
