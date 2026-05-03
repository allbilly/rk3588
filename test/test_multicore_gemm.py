from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multicore_gemm_defaults_are_safe():
    src = (ROOT / "examples" / "multicore_gemm.py").read_text()
    assert 'default="core0"' in src
    assert "default=0x1" in src
    assert "--allow-unsafe-submit" in src
    assert "Unsafe raw multicore GEMM submit refused" in src
