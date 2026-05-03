import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "add_rawbuf_pcchain.py"


def main() -> int:
    cmd = [
        sys.executable,
        str(PROBE),
        "--amount-style",
        "body",
        "--regcmd-mode",
        "absolute",
        "--descriptor-amount",
        "segment",
        "--segment-elements",
        "4096",
        *sys.argv[1:],
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
