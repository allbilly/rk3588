from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multicore_probe_does_not_run_risky_paths_by_default():
    src = (ROOT / "examples" / "multicore_probe.py").read_text()
    assert "--include-separate-submits" in src
    assert "--include-nonzero-cores" in src
    assert "--include-risky-split3" in src
    assert "--allow-unsafe-submit" in src
    assert "if args.include_separate_submits:" in src
    assert "if args.include_nonzero_cores:" in src
    assert "if args.include_risky_split3:" in src
