import subprocess
import os
from pathlib import Path


OPS_RKNN_ADD = Path("/home/orangepi/npu/ops_rknn/add")
OPS_RKNN_DIR = OPS_RKNN_ADD.parent


def run_width(width: int) -> int:
    if not OPS_RKNN_ADD.exists():
        raise FileNotFoundError(f"missing reference binary: {OPS_RKNN_ADD}")
    env = os.environ.copy()
    env.setdefault("RKNN_CORE_MASK", "1")
    proc = subprocess.run([str(OPS_RKNN_ADD), str(width)], cwd=str(OPS_RKNN_DIR), env=env, check=False)
    return proc.returncode
