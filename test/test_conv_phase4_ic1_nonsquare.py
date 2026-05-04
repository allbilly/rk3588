import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import conv


IC1_NONSQUARE_SHAPES = [
    (2, 1),
    (2, 3),
    (3, 1),
    (3, 5),
    (1, 3),
    (1, 5),
]


def test_ic1_nonsquare_uses_nhwc_input_pack():
    for kh, kw in IC1_NONSQUARE_SHAPES:
        params = conv.compute_conv2d_params(1, 6, kh, kw, (5, 7), groups=1)

        assert params["input_pack_c2"] == 2
        assert params["use_nhwc"]


def test_ic1_nonsquare_uses_kh_major_weights():
    for kh, kw in IC1_NONSQUARE_SHAPES:
        assert conv._is_kh_major(6, 1, kh, kw, groups=1)


def test_non_1x1_conv_uses_single_direct_submit(monkeypatch):
    calls = []

    def fake_submit(params, input_nchw, weight_ochw, is_1x1):
        calls.append((params["in_channels"], params["out_channels"], params["kernel_h"], params["kernel_w"], is_1x1))
        return np.zeros((1, params["out_channels"], params["out_h"], params["out_w"]), dtype=np.float16)

    monkeypatch.setattr(conv, "DRY_RUN", False)
    monkeypatch.setattr(conv, "_npu_submit", fake_submit)

    result, inp, wt = conv.run_conv2d(16, 6, 5, 2, (64, 64), groups=1)

    assert result.shape == (1, 6, 60, 63)
    assert inp.shape == (1, 16, 64, 64)
    assert wt.shape == (6, 16, 5, 2)
    assert calls == [(16, 6, 5, 2, False)]
