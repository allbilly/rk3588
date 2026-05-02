import sys
from pathlib import Path

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
