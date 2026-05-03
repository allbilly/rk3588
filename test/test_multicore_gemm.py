from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multicore_gemm_defaults_are_safe():
    src = (ROOT / "experimental" / "multicore_gemm.py").read_text()
    assert 'default="core0"' in src
    assert "default=0x1" in src
    assert "--tile-n" in src
    assert "--tiles" in src
    assert "--execution" in src
    assert "sequential-core0" in src
    assert "unsafe-split3" in src
    assert "requires explicit --allow-unsafe-submit" in src
    assert "--allow-unsafe-submit" in src
    assert "--subcore-layout" in src
    assert "rk3588-tricore-tail" in src
    assert "Unsafe raw multicore GEMM submit refused" in src
