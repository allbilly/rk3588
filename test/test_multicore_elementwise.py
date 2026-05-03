from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multicore_elementwise_defaults_are_safe():
    src = (ROOT / "experimental" / "multicore_elementwise.py").read_text()
    assert 'default="core0"' in src
    assert "default=0x1" in src
    assert 'default="ADD"' in src
    assert "--task-count" in src
    assert "--tile-flat" in src
    assert "--tiles" in src
    assert "--execution" in src
    assert "sequential-core0" in src
    assert "unsafe-split3" in src
    assert "requires explicit --allow-unsafe-submit" in src
    assert "--separate-submits" in src
    assert "Run each task as a sequential core-0 submit" in src
    assert "--allow-unsafe-submit" in src
    assert "--subcore-layout" in src
    assert "rk3588-tricore-tail" in src
    assert "--pc-chain" in src
    assert "--pc-chain-style" in src
    assert "--task-op-idx" in src
    assert "--submit-block" in src
    assert "Unsafe raw multicore submit refused" in src


def test_multicore_probe_requires_risky_opt_in():
    src = (ROOT / "experimental" / "multicore_probe.py").read_text()
    assert "--include-risky-split3" in src
    assert "--include-separate-submits" in src
    assert "--include-nonzero-cores" in src
    assert "--allow-unsafe-submit" in src
    assert "if args.include_risky_split3:" in src
