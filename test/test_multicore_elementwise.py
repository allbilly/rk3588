from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multicore_elementwise_defaults_are_safe():
    src = (ROOT / "examples" / "multicore_elementwise.py").read_text()
    assert 'default="core0"' in src
    assert "default=0x1" in src
    assert "--separate-submits" in src
    assert "--allow-unsafe-submit" in src
    assert "Unsafe raw multicore submit refused" in src


def test_multicore_probe_requires_risky_opt_in():
    src = (ROOT / "examples" / "multicore_probe.py").read_text()
    assert "--include-risky-split3" in src
    assert "--include-separate-submits" in src
    assert "--include-nonzero-cores" in src
    assert "--allow-unsafe-submit" in src
    assert "if args.include_risky_split3:" in src
