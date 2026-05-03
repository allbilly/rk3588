import argparse
import os
import subprocess
import sys


def run_cmd(cmd):
    print()
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def maybe_allow_unsafe(cmd, args):
    if args.allow_unsafe_submit:
        cmd.append("--allow-unsafe-submit")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run the RK3588 NPU multicore experiment matrix.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--elementwise-n", type=int, default=4096)
    parser.add_argument("--gemm-size", type=int, default=32)
    parser.add_argument("--regcmd-mode", choices=("offset", "absolute"), default="offset")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-gemm", action="store_true")
    parser.add_argument("--skip-individual", action="store_true")
    parser.add_argument("--include-separate-submits", action="store_true", help="Run one submit per core. This may lock up affected kernels.")
    parser.add_argument("--include-nonzero-cores", action="store_true", help="Probe core 1 and core 2 individually. This may lock up affected kernels.")
    parser.add_argument("--include-risky-split3", action="store_true", help="Run the known-risky single-submit multi-range experiments.")
    parser.add_argument("--use-rk3588-tricore-tail-layout", action="store_true", help="For risky split3 probes, map core ranges to subcore_task[2..4] per allbilly/rknpu_driver.")
    parser.add_argument("--allow-unsafe-submit", action="store_true", help="Pass through to raw submit examples; required for crash-prone probes.")
    args = parser.parse_args()

    if not os.path.exists("/dev/dri/card1"):
        print("SKIP: /dev/dri/card1 not present")
        return 0

    failures = 0
    commands = []
    split3_layout = "rk3588-tricore-tail" if args.use_rk3588_tricore_tail_layout else "direct"

    if args.include_risky_split3:
        commands.append(maybe_allow_unsafe([
            args.python,
            "experimental/multicore_elementwise.py",
            "--n",
            str(args.elementwise_n),
            "--ops",
            "ADD,MUL,SUB",
            "--core-mask",
            "0x7",
            "--core-ranges",
            "split3",
            "--regcmd-mode",
            args.regcmd_mode,
            "--subcore-layout",
            split3_layout,
        ], args))

    if args.include_separate_submits:
        commands.append(maybe_allow_unsafe([
            args.python,
            "experimental/multicore_elementwise.py",
            "--n",
            str(args.elementwise_n),
            "--ops",
            "ADD,MUL,SUB",
            "--separate-submits",
            "--regcmd-mode",
            args.regcmd_mode,
            "--subcore-layout",
            split3_layout,
        ], args))

    if not args.skip_individual:
        commands.append([
            args.python,
            "experimental/multicore_elementwise.py",
            "--n",
            str(args.elementwise_n),
            "--ops",
            "ADD",
            "--core-mask",
            "0x1",
            "--core-ranges",
            "0:0:1",
            "--regcmd-mode",
            args.regcmd_mode,
        ])
        if args.include_nonzero_cores:
            commands.extend([
                maybe_allow_unsafe([
                    args.python,
                    "experimental/multicore_elementwise.py",
                    "--n",
                    str(args.elementwise_n),
                    "--ops",
                    "ADD",
                    "--core-mask",
                    "0x2",
                    "--core-ranges",
                    "1:0:1",
                    "--regcmd-mode",
                    args.regcmd_mode,
                ], args),
                maybe_allow_unsafe([
                    args.python,
                    "experimental/multicore_elementwise.py",
                    "--n",
                    str(args.elementwise_n),
                    "--ops",
                    "ADD",
                    "--core-mask",
                    "0x4",
                    "--core-ranges",
                    "2:0:1",
                    "--regcmd-mode",
                    args.regcmd_mode,
                ], args),
            ])

    if not commands:
        print("No hardware probes selected. Use --include-separate-submits, --include-nonzero-cores, or --include-risky-split3 to opt in.")
        return 0

    if args.include_risky_split3 and not args.skip_gemm:
        commands.append(maybe_allow_unsafe([
            args.python,
            "experimental/multicore_gemm.py",
            "--m",
            str(args.gemm_size),
            "--n",
            str(args.gemm_size),
            "--k",
            str(args.gemm_size),
            "--core-mask",
            "0x7",
            "--core-ranges",
            "split3",
            "--regcmd-mode",
            args.regcmd_mode,
        ], args))

    for cmd in commands:
        ret = run_cmd(cmd)
        if ret != 0:
            failures += 1
            print(f"FAIL: exit={ret}")
            if not args.keep_going:
                return ret

    if failures:
        print(f"{failures} probe command(s) failed")
        return 1
    print("ALL MULTICORE PROBES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
