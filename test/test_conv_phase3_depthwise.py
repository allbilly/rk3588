import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import conv
import rockchip as rk


def _reg_value(regs, addr):
    for _, value, reg_addr in regs:
        if reg_addr == addr:
            return value
    raise AssertionError(f"register 0x{addr:04x} not emitted")


def test_depthwise_registers_use_aligned_kernel_but_single_weight_kernel():
    params = conv.compute_conv2d_params(32, 32, 1, 1, (32, 32), groups=32)
    regs = conv.build_conv2d_regs(params, 0, 0, 0)

    assert params["is_depthwise"]
    assert params["align_c"] == 32
    assert params["weight_kernels"] == 1
    assert params["weight_bytes_per_kernel"] == 1 * 1 * 32 * 2
    assert _reg_value(regs, rk.REG_CNA_CONV_CON1) & 0xF == 3
    assert _reg_value(regs, rk.REG_CNA_WEIGHT_SIZE1) == 64
    assert _reg_value(regs, rk.REG_CNA_WEIGHT_SIZE2) & 0x3FFF == 1


def test_depthwise_weight_expansion_uses_diagonal_kernel_slots():
    out_channels, ic_per_group, kh, kw, align_c = 3, 1, 3, 3, 8
    src = np.arange(out_channels * ic_per_group * kh * kw, dtype=np.float16)
    packed_weight_size = out_channels * kh * kw * align_c * 2

    packed = conv._pack_depthwise_expanded_weights_fp16(
        src, out_channels, ic_per_group, kh, kw, packed_weight_size)

    slot_sz = kh * kw
    for oc in range(out_channels):
        assert np.array_equal(
            packed[oc * slot_sz:(oc + 1) * slot_sz],
            src[oc * slot_sz:(oc + 1) * slot_sz],
        )
    assert np.count_nonzero(packed[out_channels * slot_sz:]) == 0


def test_depthwise_output_unpack_c2_uses_lane_width():
    params_32 = conv.compute_conv2d_params(32, 32, 1, 1, (32, 32), groups=32)
    params_3 = conv.compute_conv2d_params(3, 3, 3, 3, (11, 28), groups=3)

    assert conv._output_unpack_c2(params_32) == params_32["align_c"] == 32
    assert conv._output_unpack_c2(params_3) == params_3["align_c"] == 8


def test_depthwise_above_eight_channels_uses_sliced_submit(monkeypatch):
    calls = []

    def fake_submit(params, input_nchw, weight_ochw, is_1x1):
        calls.append((params["in_channels"], params["out_channels"], params["groups"]))
        return np.zeros((1, params["out_channels"], params["out_h"], params["out_w"]), dtype=np.float16)

    monkeypatch.setattr(conv, "DRY_RUN", False)
    monkeypatch.setattr(conv, "_npu_submit", fake_submit)

    result, inp, wt = conv.run_conv2d(32, 32, 1, 1, (32, 32), groups=32)

    assert result.shape == (1, 32, 32, 32)
    assert inp.shape == (1, 32, 32, 32)
    assert wt.shape == (32, 1, 1, 1)
    assert calls == [(8, 8, 8), (8, 8, 8), (8, 8, 8), (8, 8, 8)]
