#!/usr/bin/env python3
"""Standalone first-principles FP16 CONV for RK3588 NPU (gemm.py style, <1000 lines).

CBUF planner emits NONE/BY_Y/BY_K/BY_YK. BY_YK is an independent Y×K cartesian
product of local tiles (no exact11 / hex blobs / shape-name OVERRIDES).
Each tile: open → alloc → pack → 1-task submit → unpack → reset → close.
"""
import os, mmap, sys, ctypes, argparse
from fcntl import ioctl
import numpy as np

# Allow `python3 conv_grok/conv.py` and `python3 -m conv_grok.conv` alike.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 0 << 1
FP16_BYTES = 2
FP32_BYTES = 4
FP16_ATOM_ELEMENTS = 16
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_MAX_CONV_FLAT_STRIDE = 992
UNPACK_C2 = FP16_ATOM_ELEMENTS // FP16_BYTES
PC_CHAIN_TAIL_QWORDS = 4
MIN_HW_OC = 2
POINTWISE_WIDE_MIN_OC = 32

DEFAULT_SMOKE = (
    "conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1",
    "b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid",
)
# Extra hard shapes beyond the 217 set (formula stress: large RGB, tall pw, fat K).
EXTRA_HARD_SHAPES = (
    "b1_c3_h384_w384_oc32_wic3_k3x3_g1_s1_pvalid",
    "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid",
    "b1_c64_h112_w112_oc128_wic64_k1x1_g1_s1_pvalid",
    "b1_c256_h28_w28_oc512_wic256_k1x1_g1_s1_pvalid",
    "b1_c512_h14_w14_oc256_wic512_k1x1_g1_s1_pvalid",
    "b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid",
)
LIST_SHAPES = (
    ("smoke", "conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1"),
    ("smoke", "b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid"),
    ("try", "b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid"),
    ("try", "b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid"),
    ("try", "conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1"),
    ("try", "conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1"),
    ("extra", "b1_c3_h384_w384_oc32_wic3_k3x3_g1_s1_pvalid"),
    ("extra", "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid"),
    ("extra", "b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid"),
)


class reg:
    CNA = 0x0201; CORE = 0x0801; DPU = 0x1001; PC = 0x0081; PC_REG = 0x0101; VERSION = 0x0041
    OPERATION_ENABLE = 0x0008; PC_BASE_ADDRESS = 0x0010; PC_REGISTER_AMOUNTS = 0x0014
    S_POINTER = 0x4004; FEATURE_MODE_CFG = 0x400c; DATA_FORMAT = 0x4010
    DST_BASE_ADDR = 0x4020; DST_SURF_STRIDE = 0x4024
    DATA_CUBE_WIDTH = 0x4030; DATA_CUBE_HEIGHT = 0x4034; DATA_CUBE_NOTCH = 0x4038
    DATA_CUBE_CHANNEL = 0x403c; BS_CFG = 0x4040; BS_OW_CFG = 0x4050
    WDMA_SIZE_0 = 0x4058; WDMA_SIZE_1 = 0x405c; BN_CFG = 0x4060; EW_CFG = 0x4070
    EW_CVT_SCALE_VALUE = 0x4078; OUT_CVT_SCALE = 0x4084; SURFACE_ADD = 0x40c0
    CNA_CONV_CON1 = 0x100c; CNA_CONV_CON2 = 0x1010; CNA_CONV_CON3 = 0x1014
    CNA_DATA_SIZE0 = 0x1020; CNA_DATA_SIZE1 = 0x1024; CNA_DATA_SIZE2 = 0x1028
    CNA_DATA_SIZE3 = 0x102c; CNA_WEIGHT_SIZE0 = 0x1030; CNA_WEIGHT_SIZE1 = 0x1034
    CNA_WEIGHT_SIZE2 = 0x1038; CNA_CBUF_CON0 = 0x1040; CNA_CBUF_CON1 = 0x1044
    CNA_CVT_CON0 = 0x104c; CNA_CVT_CON1 = 0x1050; CNA_CVT_CON2 = 0x1054
    CNA_CVT_CON3 = 0x1058; CNA_CVT_CON4 = 0x105c; CNA_CVT_CON5 = 0x1180
    CNA_FEATURE_DATA_ADDR = 0x1070; CNA_DMA_CON0 = 0x1078; CNA_DMA_CON1 = 0x107c
    CNA_DMA_CON2 = 0x1080; CNA_FC_DATA_SIZE0 = 0x1084; CNA_FC_DATA_SIZE1 = 0x1088
    CNA_DCOMP_ADDR0 = 0x1110
    CORE_MISC_CFG = 0x3010; CORE_DATAOUT_SIZE_0 = 0x3014; CORE_DATAOUT_SIZE_1 = 0x3018
    CORE_RESERVED_3030 = 0x3030


class rknpu_mem_create(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("size", ctypes.c_uint64),
                ("obj_addr", ctypes.c_uint64), ("dma_addr", ctypes.c_uint64), ("sram_size", ctypes.c_uint64)]


class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64)]


class rknpu_mem_destroy(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("obj_addr", ctypes.c_uint64)]


class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]


class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]


class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32), ("timeout", ctypes.c_uint32), ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32), ("task_counter", ctypes.c_uint32), ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64), ("iommu_domain_id", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("task_base_addr", ctypes.c_uint64), ("hw_elapse_time", ctypes.c_int64), ("core_mask", ctypes.c_uint32),
        ("fence_fd", ctypes.c_int32), ("subcore_task", rknpu_subcore_task * 5),
    ]


class struct_rknpu_task(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("op_idx", ctypes.c_uint32), ("enable_mask", ctypes.c_uint32),
                ("int_mask", ctypes.c_uint32), ("int_clear", ctypes.c_uint32), ("int_status", ctypes.c_uint32),
                ("regcfg_amount", ctypes.c_uint32), ("regcfg_offset", ctypes.c_uint32), ("regcmd_addr", ctypes.c_uint64)]


def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)


DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_MEM_DESTROY = _IOWR('d', 0x44, ctypes.sizeof(rknpu_mem_destroy))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))


def _ceil_div(x, y):
    return (x + y - 1) // y


def _align_up(x, align):
    return _ceil_div(x, align) * align


def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr


def shape_from_name(name):
    core = name[7:] if name.startswith("conv2d_") else name
    fields = core.split("_")
    vals = {field[:3] if field.startswith("wic") else field[0]: field for field in fields}
    try:
        kh, kw = (int(x) for x in vals["k"][1:].split("x"))
        return dict(name=name, batch=int(vals["b"][1:]), in_c=int(vals["c"][1:]), in_h=int(vals["h"][1:]),
                    in_w=int(vals["w"][1:]), out_c=int(vals["o"][2:]), weight_in_c=int(vals["wic"][3:]),
                    kh=kh, kw=kw, groups=int(vals["g"][1:]), stride=int(vals.get("s", "s1")[1:]))
    except (KeyError, ValueError):
        raise ValueError("expected encoded shape like b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid")


def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c


def _is_pointwise_wide(s):
    # IC>=32: 1x1 uses 32-channel weight atoms (c40 fails with kh-major pack).
    return s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1 and s["in_c"] >= 32


def _conv_align_c(in_c, groups, out_c):
    if not _is_depthwise(in_c, out_c, groups) and (groups > 1 or in_c > 4):
        return 16
    return max(8, min(1 << (max(1, in_c) - 1).bit_length(), 32 if _is_depthwise(in_c, out_c, groups) else 16))


def _conv_input_pack_c2(in_c, groups, out_c, align_c):
    if in_c == 1:
        return 2
    if _is_depthwise(in_c, out_c, groups) or groups > 1 or in_c > 4:
        return 8
    return align_c


def _conv_params(s):
    in_c, in_h, in_w, out_c = s["in_c"], s["in_h"], s["in_w"], s["out_c"]
    kh, kw, groups, stride = s["kh"], s["kw"], s["groups"], s.get("stride", 1)
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    is_spatial = kh != 1 or kw != 1
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    align_c = _conv_align_c(in_c, groups, out_c)
    align_out_c = max(32 if _is_pointwise_wide(s) else 16, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_atoms = max(1, out_h * out_w)
    out_width_stride = out_atoms if not is_spatial else _align_up(out_atoms, 4)
    input_pack_c2 = _conv_input_pack_c2(in_c, groups, out_c, align_c)
    use_nhwc = not is_depthwise and not (groups > 1 and is_spatial) and in_c < input_pack_c2
    return dict(is_depthwise=is_depthwise, is_spatial=is_spatial, out_h=out_h, out_w=out_w, align_c=align_c,
                align_out_c=align_out_c, width_stride=width_stride, out_width_stride=out_width_stride,
                input_pack_c2=input_pack_c2, use_nhwc=use_nhwc, stride=stride)


def _dma_strides(in_h, width_stride, use_nhwc_pack):
    # Surf stride is width_stride*(in_h-4) even when in_h<4 (wraps as uint32, e.g. h3→0xfffffffd).
    # Clamping to 0 breaks tiny spatial maps (c128/h3 native FAIL ~155).
    if use_nhwc_pack:
        return width_stride, width_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, (width_stride * (in_h - 4)) & 0xffffffff


def _cbuf_entries(width_stride, align_c, in_h, is_depthwise):
    row_entries = max(1, _ceil_div(width_stride * align_c, 2 * FP16_ATOM_ELEMENTS))
    return row_entries if align_c >= 16 or is_depthwise else row_entries * in_h * 4


def _feature_grains(row_bytes, floor_grains, use_nhwc_pack=False, is_spatial=False, is_depthwise=False):
    if use_nhwc_pack and is_spatial:
        return floor_grains
    if is_depthwise and is_spatial:
        return min(13, floor_grains)
    even_rows_per_two_banks = (_ceil_div(2 * CBUF_BANK_SIZE, row_bytes) + 1) & ~1
    return min(floor_grains, even_rows_per_two_banks)


def _data_bank(width_stride, feature_grains, align_c, use_nhwc_pack=False, is_spatial=False, is_depthwise=False):
    if is_spatial and (use_nhwc_pack or is_depthwise):
        return RK_CBUF_BANKS - 1
    return int(np.clip(_ceil_div(width_stride * feature_grains * align_c * FP16_BYTES, CBUF_BANK_SIZE), 1, RK_CBUF_BANKS - 1))


def _pointwise_data_in_c(in_c):
    """Pointwise-wide DMA/weight channel align: 32-atom packing (matches conv_gemm)."""
    return max(32, _align_up(in_c, 32))


def _pack_pointwise_wide(weight, out_c, in_c):
    # Pad IC to 32-atoms; IC-block-major layout. Unpadded IC (e.g. 72→80 reg vs 72 pack) fails.
    aligned_in_c = _pointwise_data_in_c(in_c)
    aligned_out_c = max(32, _align_up(out_c, 16))
    padded = np.zeros((aligned_out_c, aligned_in_c), dtype=np.float16)
    padded[:out_c, :in_c] = weight[:out_c, :in_c, 0, 0]
    return np.concatenate([
        padded[oc:oc + 16].reshape(-1, aligned_in_c // 32, 32).transpose(1, 0, 2).ravel()
        for oc in range(0, aligned_out_c, 16)
    ])


def _pack_kh_major(weight, out_c, in_c, kh, kw, c2_out):
    aligned_in_c = c2_out * _ceil_div(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:, :in_c] = weight
    if kh != 1 or kw != 1:
        return np.concatenate([padded[oc:oc + 16, ic:ic + 32, y, x].ravel()
                               for oc in range(0, out_c, 16) for ic in range(0, aligned_in_c, 32)
                               for y in range(kh) for x in range(kw)])
    return np.concatenate([padded[oc:oc + 16].transpose(2, 3, 0, 1).ravel() for oc in range(0, out_c, 16)])


def pack_weights(weight_full, s, p):
    if _is_pointwise_wide(s):
        return _pack_pointwise_wide(weight_full, s["out_c"], s["in_c"])
    return _pack_kh_major(weight_full, s["out_c"], s["in_c"], s["kh"], s["kw"], p["align_c"])


def pack_input(input_nchw, p):
    in_c, in_h, in_w = input_nchw.shape
    if p["use_nhwc"]:
        out = np.zeros((in_h, p["width_stride"], in_c), dtype=np.float16)
        out[:, :in_w] = input_nchw.transpose(1, 2, 0)
        return out.ravel()
    c2 = p["input_pack_c2"]
    c1 = _ceil_div(in_c, c2)
    padded = np.zeros((c1 * c2, in_h, p["width_stride"]), dtype=np.float16)
    padded[:in_c, :, :in_w] = input_nchw
    return padded.reshape(c1, c2, in_h, p["width_stride"]).transpose(0, 2, 3, 1).ravel()


def unpack_output(out_raw, out_c, out_h, out_w, out_width_stride, c2):
    c1 = out_raw.size // (out_width_stride * c2)
    packed = out_raw.reshape(1, c1, 1, out_width_stride, c2)
    return packed[0, :, 0, :out_h * out_w, :].transpose(0, 2, 1).reshape(c1 * c2, out_h * out_w)[:out_c].reshape(out_c, out_h, out_w)


def make_regs(s, p, in_dma, wt_dma, out_dma, out_fp16, full_data_bank=False):
    in_c, in_h, in_w, out_c = s["in_c"], s["in_h"], s["in_w"], s["out_c"]
    kh, kw = s["kh"], s["kw"]
    align_c, align_out_c = p["align_c"], p["align_out_c"]
    out_h, out_w, is_spatial = p["out_h"], p["out_w"], p["is_spatial"]
    data_in_c = _pointwise_data_in_c(in_c) if _is_pointwise_wide(s) else _align_up(in_c, align_c)
    weight_bytes_per_kernel = kh * kw * data_in_c * FP16_BYTES
    if full_data_bank and not is_spatial:
        # Do not clamp grains below tile H (clamp caused c960 wrong output).
        feature_grains = in_h
        wt_banks = max(1, _ceil_div(weight_bytes_per_kernel * out_c, CBUF_BANK_SIZE))
        data_bank = max(1, min(RK_CBUF_BANKS - 1, RK_CBUF_BANKS - wt_banks))
    else:
        feature_grains = _feature_grains(p["width_stride"] * data_in_c * FP16_BYTES, in_h + kh,
                                         p["use_nhwc"], is_spatial, False)
        if full_data_bank:
            wt_banks = max(1, _ceil_div(weight_bytes_per_kernel * out_c, CBUF_BANK_SIZE))
            data_bank = max(1, min(RK_CBUF_BANKS - 1, RK_CBUF_BANKS - wt_banks))
        else:
            data_bank = _data_bank(
                p["width_stride"], feature_grains, data_in_c, p["use_nhwc"], is_spatial, False)
    out_precision = 2 if out_fp16 else 5
    size_e = 1 if out_fp16 else 3
    out_channel_field = align_out_c - 1
    cvt_con0 = 0x0b if is_spatial and not p["is_depthwise"] else 1
    cvt_con5 = ((1 << in_c) - 1) if p["use_nhwc"] else 0
    conv_con1 = (2 << 4) | (2 << 7)
    if p["use_nhwc"] and in_c <= 4:
        conv_con1 |= (1 << 30) | (1 << 29) | ((7 + in_c) << 12)
    line_stride, surf_stride = _dma_strides(in_h, p["width_stride"], p["use_nhwc"])
    return [
        E(reg.DPU, reg.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.CNA, reg.CNA_CONV_CON1, conv_con1),
        E(reg.CNA, reg.CNA_CONV_CON2, feature_grains << 4),
        E(reg.CNA, reg.CNA_CONV_CON3, (1 << 3) | 1),
        E(reg.CNA, reg.CNA_DATA_SIZE0, (p["width_stride"] << 16) | in_h),
        E(reg.CNA, reg.CNA_DATA_SIZE1, ((in_c - 1) << 16) | data_in_c),
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_w * out_h),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_per_kernel * out_c),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2, (kw << 24) | (kh << 16) | out_c),
        E(reg.CNA, reg.CNA_CBUF_CON0, ((RK_CBUF_BANKS - data_bank) << 4) | data_bank),
        E(reg.CNA, reg.CNA_CBUF_CON1, _cbuf_entries(p["width_stride"], data_in_c, in_h, False)),
        E(reg.CNA, reg.CNA_CVT_CON0, cvt_con0), E(reg.CNA, reg.CNA_CVT_CON1, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON2, 1 << 16), E(reg.CNA, reg.CNA_CVT_CON3, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON4, 1 << 16), E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        E(reg.CNA, reg.CNA_DMA_CON1, line_stride), E(reg.CNA, reg.CNA_DMA_CON2, surf_stride),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, (in_w << 16) | in_h),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, data_in_c), E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, cvt_con5),
        E(reg.CORE, reg.CORE_MISC_CFG, 2 << 8),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field), E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1)),
        E(reg.DPU, reg.DATA_FORMAT, (out_precision << 29) | (2 << 26) | 2),
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma), E(reg.DPU, reg.DST_SURF_STRIDE, p["out_width_stride"] << 4),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1), E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),
        E(reg.DPU, reg.DATA_CUBE_NOTCH, 0),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL, ((out_c - 1) << 16) | out_channel_field),
        E(reg.DPU, reg.BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.BS_OW_CFG, (size_e << 8) | (size_e << 5) | (size_e << 2) | (1 << 1)),
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field),
        E(reg.DPU, reg.WDMA_SIZE_1, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.DPU, reg.BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1),
        E(reg.DPU, reg.OUT_CVT_SCALE, ((1 << 16) | 1) if out_fp16 else 0),
        E(reg.DPU, reg.SURFACE_ADD, (p["out_width_stride"] * 2) << 4),
    ]


# --- compact CBUF planner (formula-only; BY_YK = Y×K cartesian local tiles) ---

def _mesa_entries_per_slice(input_width, input_channels):
    atomics_per_entry = CBUF_ENTRY_BYTES // 16
    total_c_atomics = _ceil_div(input_channels * FP16_BYTES, 16)
    last_c_atomics = total_c_atomics % atomics_per_entry
    int_c_entries = (total_c_atomics // atomics_per_entry) * input_width
    frac = input_width if last_c_atomics == 3 else _ceil_div(last_c_atomics * input_width, atomics_per_entry)
    return int_c_entries + frac


def _pointwise_oc_tile_c(in_c, weight_banks=1):
    # Cap OC so one K-tile weight fits in `weight_banks` CBUF banks (32-aligned IC).
    data_in = _pointwise_data_in_c(in_c) if in_c >= 32 else max(1, in_c)
    max_tile = (max(1, weight_banks) * CBUF_BANK_SIZE) // (data_in * FP16_BYTES)
    return 32 if max_tile >= 32 else 16 if max_tile >= 16 else 8 if max_tile >= 8 else 4


def _compute_k_step(in_c, out_c, kh, kw, groups, p):
    is_dw, is_spatial = p["is_depthwise"], p["is_spatial"]
    aligned_in = _pointwise_data_in_c(in_c) if (not is_spatial and groups == 1 and in_c >= 32) else _align_up(in_c, p["align_c"])
    weight_banks = _ceil_div(kh * kw * aligned_in * FP16_BYTES * (1 if is_dw else out_c), CBUF_BANK_SIZE)
    k_step = out_c
    if is_dw and is_spatial:
        k_step = min(32, out_c)
    elif is_spatial and groups == 1 and not is_dw:
        row_bytes = p["width_stride"] * aligned_in * FP16_BYTES
        feature_rows = _feature_grains(row_bytes, p["in_h"] + kh, False, True, False)
        if weight_banks > 3 or feature_rows < p["in_h"]:
            k_step = 32 if out_c >= 32 else out_c
    elif not is_spatial and groups == 1:
        # Default: 1 weight bank (Y-split handles oversized features).
        # Upgrade to 2 banks when full-H features still fit in remaining data banks
        # (cuts c1024/c1280 tile counts roughly in half).
        pw_oc = _pointwise_oc_tile_c(in_c, 1)
        row_bytes = p["width_stride"] * aligned_in * FP16_BYTES
        feat_banks = _ceil_div(row_bytes * p["in_h"], CBUF_BANK_SIZE)
        if feat_banks < (RK_CBUF_BANKS - 2):
            pw_oc = max(pw_oc, _pointwise_oc_tile_c(in_c, 2))
        if weight_banks > 3 or out_c > pw_oc:
            k_step = pw_oc
    return min(k_step, out_c)


def _compute_y_step(in_c, out_c, kh, kw, in_h, in_w, groups, stride, k_step, p):
    is_spatial, is_dw, out_h = p["is_spatial"], p["is_depthwise"], p["out_h"]
    if is_dw and is_spatial:
        aligned_in = _align_up(in_c, p["align_c"])
        weight_banks = _ceil_div(_ceil_div(kh * kw * aligned_in * FP16_BYTES, CBUF_ENTRY_BYTES),
                                 CBUF_ENTRIES_PER_BANK) + 1
        input_banks = RK_CBUF_BANKS - weight_banks if weight_banks + 1 < RK_CBUF_BANKS else 7
        eps = max(1, _mesa_entries_per_slice(in_w, in_c))
        slices = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // eps)
        tile_h = min(out_h, max(1, (slices - kh) // stride + 1))
        if in_c > 64:
            row_bytes = in_w * _conv_align_c(in_c, in_c, in_c) * FP16_BYTES
            tile_h = min(tile_h, _feature_grains(row_bytes, out_h + kh, False, True, True) + 1)
        tile_h = max(10, tile_h) if tile_h < out_h else tile_h
        tile_h = min(tile_h, min(15, out_h + kh - 1) - kh + 1)
        return tile_h if tile_h == out_h or tile_h % 2 == 0 else max(1, tile_h - 1)

    aligned_in = _pointwise_data_in_c(in_c) if (not is_spatial and groups == 1 and in_c >= 32) else _align_up(in_c, p["align_c"])
    tile_aligned = aligned_in if (not is_spatial and groups == 1 and in_c >= 32) else _align_up(k_step if is_dw else in_c, p["align_c"])
    row_bytes = p["width_stride"] * tile_aligned * FP16_BYTES
    y_step = out_h
    tile_wb = max(1, _ceil_div(kh * kw * aligned_in * FP16_BYTES * (k_step if not is_dw else 1), CBUF_BANK_SIZE))
    remaining = max(1, RK_CBUF_BANKS - tile_wb)
    fg = _feature_grains(row_bytes, in_h + kh, False, is_spatial, is_dw)
    data_banks_needed = _ceil_div(row_bytes * fg, CBUF_BANK_SIZE)
    if data_banks_needed > remaining:
        y_step = max(1, out_h * remaining // max(1, data_banks_needed))
    if not is_spatial:
        if in_c <= 4 and not is_dw and p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE:
            y_step = min(y_step, max(1, RK_MAX_CONV_FLAT_STRIDE // p["out_w"]))
        elif groups == 1 and in_c >= 128 and out_c <= 32 and out_h == 28:
            # pointwise-wide h28: CBUF cannot hold full height at once (matches proven 22+6)
            y_step = min(y_step, 22)
        elif out_h > 50:
            cap = 25 if in_c >= 128 and out_c >= 128 else (32 if p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE else 50)
            y_step = min(y_step, cap)
        # Strict bank fit: exact fill of remaining data banks fails (c768_h20 y=11).
        if groups == 1 and in_c >= 32:
            while y_step > 1 and _ceil_div(row_bytes * y_step, CBUF_BANK_SIZE) >= remaining:
                y_step -= 1
    if is_spatial and groups == 1 and in_c == 3 and out_c <= 32 and kh == 3 and kw == 3 and in_h >= 224:
        # first-layer RGB: proven 224→48; scale down for taller maps (320→32, 384→24)
        if in_h <= 224:
            y_step = min(y_step, 48)
        elif in_h <= 320:
            y_step = min(y_step, 32)
        else:
            y_step = min(y_step, 24)
    if not is_dw:
        eps = max(1, _mesa_entries_per_slice(p["width_stride"], aligned_in))
        input_banks = RK_CBUF_BANKS - tile_wb if tile_wb + 1 < RK_CBUF_BANKS else 7
        max_input_h = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // eps)
        # Pointwise: stay strictly below entry-capacity (boundary y fails).
        if not is_spatial and groups == 1 and in_c >= 32 and max_input_h < out_h:
            max_input_h = max(1, max_input_h - 1)
        y_step = min(y_step, max(1, (max_input_h - kh) // stride + 1))
    if is_spatial and p["use_nhwc"]:
        max_grains = ((RK_CBUF_BANKS - 1) * CBUF_BANK_SIZE) // max(1, row_bytes)
        y_step = min(y_step, max(1, max_grains - 2 * kh + 1))
    return max(1, y_step)


def _windows_from_step(total, step):
    wins, start = [], 0
    while start < total:
        n = min(step, total - start)
        wins.append((start, n))
        start += n
    return wins


def plan_local_serial_rows(s):
    """Emit independent local-serial tiles. BY_YK = Y×K cartesian product."""
    in_c, in_h, in_w = s["in_c"], s["in_h"], s["in_w"]
    out_c, kh, kw, groups = s["out_c"], s["kh"], s["kw"], s["groups"]
    stride = s.get("stride", 1)
    p = _conv_params(s)
    out_h = p["out_h"]
    # planner uses same geometry fields as _conv_params
    pp = dict(p, in_c=in_c, in_h=in_h, in_w=in_w, out_c=out_c, kh=kh, kw=kw, groups=groups, stride=stride)
    k_step = _compute_k_step(in_c, out_c, kh, kw, groups, pp)
    y_step = _compute_y_step(in_c, out_c, kh, kw, in_h, in_w, groups, stride, k_step, pp)
    if groups == 1 and kh == 1 and kw == 1 and in_c >= 16 and y_step < out_h:
        tail = out_h % y_step
        if 0 < tail < 6 and y_step > 6:
            y_step -= 6 - tail
    if k_step < out_c and y_step < out_h:
        split = "BY_YK"
    elif k_step < out_c:
        split = "BY_K"
    elif y_step < out_h:
        split = "BY_Y"
    else:
        split = "NONE"
    y_wins = _windows_from_step(out_h, y_step)
    k_wins = _windows_from_step(out_c, k_step)

    def row(family, y_start, output_h, k_start, oc_count):
        input_h = min((output_h - 1) * stride + kh, in_h - y_start * stride)
        return dict(split_method=split, family=family, y_start=y_start, input_h=input_h,
                    output_h=output_h, k_start=k_start, oc_count=oc_count)

    if split == "NONE":
        rows = [row("setup", 0, out_h, 0, out_c)]
    elif split == "BY_Y":
        rows = [row("y_tile", ys, oh, 0, out_c) for ys, oh in y_wins]
    elif split == "BY_K":
        rows = [row("k_tile", 0, out_h, ks, oc) for ks, oc in k_wins]
    else:
        rows = [row("yk_tile", ys, oh, ks, oc) for ys, oh in y_wins for ks, oc in k_wins]
    return rows, p, y_step, k_step


# --- DRM / submit ---

def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    return mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset), mem_create


def mem_destroy(fd, mem_create):
    return ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY,
                 rknpu_mem_destroy(handle=mem_create.handle, obj_addr=mem_create.obj_addr))


def close_allocations(fd, allocations):
    for mapped, mem_create in reversed(allocations):
        try:
            mapped.close()
        except BufferError:
            pass
        finally:
            mem_destroy(fd, mem_create)


def write_tasks(task_map, regcmd_map, regcmd_mem, task_regs):
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    offsets, offset = [], 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + PC_CHAIN_TAIL_QWORDS, 2)
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            next_addr = regcmd_mem.dma_addr + offsets[idx + 1] * 8
            tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xfffffff0),
                    E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(len(task_regs[idx + 1]), 2) + 1),
                    E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
        else:
            tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, 0), E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                    E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
        for i, qword in enumerate(tail):
            regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = 0xd
        tasks[idx].int_mask = 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = len(regs)
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * 8


def npu_submit(fd, task_obj_addr, task_count, core_mask=1):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))
    submit = rknpu_submit(flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK, timeout=6000, task_start=0, task_number=task_count,
                          task_counter=0, priority=0, task_obj_addr=task_obj_addr, iommu_domain_id=0,
                          reserved=0, task_base_addr=0, hw_elapse_time=0, core_mask=core_mask, fence_fd=-1)
    submit.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit)


def post_submit_reset(fd):
    for reset_flag in (RKNPU_ACT_RESET, 6, RKNPU_ACT_RESET):
        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=reset_flag, value=0))


def hw_oc_for(s, oc_count):
    if _is_pointwise_wide(s) and oc_count < POINTWISE_WIDE_MIN_OC:
        return POINTWISE_WIDE_MIN_OC
    return max(MIN_HW_OC, oc_count)


def hw_out_fp16_for(s, p):
    return p["is_spatial"] or s["out_c"] >= 128 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE or _is_pointwise_wide(s)


class TileSession:
    """Reuse one DRM fd + BO set across many local-serial submits (depthwise/BY_K)."""

    def __init__(self, in_bytes=4 * 1024 * 1024, wt_bytes=4 * 1024 * 1024, out_bytes=4 * 1024 * 1024):
        self.fd = os.open("/dev/dri/card1", os.O_RDWR)
        self.task_map, self.task_mem = mem_allocate(self.fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        self.regcmd_map, self.regcmd_mem = mem_allocate(self.fd, 8192, RKNPU_MEM_NON_CACHEABLE)
        self.input_map, self.input_mem = mem_allocate(self.fd, in_bytes, RKNPU_MEM_NON_CACHEABLE)
        self.weight_map, self.weight_mem = mem_allocate(self.fd, wt_bytes, RKNPU_MEM_NON_CACHEABLE)
        self.output_map, self.output_mem = mem_allocate(self.fd, out_bytes, RKNPU_MEM_NON_CACHEABLE)
        self._allocs = ((self.task_map, self.task_mem), (self.regcmd_map, self.regcmd_mem),
                        (self.input_map, self.input_mem), (self.weight_map, self.weight_mem),
                        (self.output_map, self.output_mem))

    def close(self):
        close_allocations(self.fd, self._allocs)
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _copy_u16(dst_map, src_u16):
    nbytes = src_u16.nbytes
    if nbytes > len(dst_map):
        raise ValueError(f"tile buffer too small: need {nbytes} have {len(dst_map)}")
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(dst_map)),
                   src_u16.ctypes.data, nbytes)


def run_hw_tile(tile_shape, tile_in, tile_wt, full_data_bank=False, session=None):
    """One local tile submit. Optional TileSession reuses fd/BOs across tiles."""
    p = _conv_params(tile_shape)
    out_fp16 = True if full_data_bank else hw_out_fp16_for(tile_shape, p)
    read_dtype = np.float16 if out_fp16 else np.float32
    c2 = UNPACK_C2 if out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
    input_flat = np.ascontiguousarray(pack_input(tile_in, p).view(np.uint16))
    weight_flat = np.ascontiguousarray(pack_weights(tile_wt, tile_shape, p).view(np.uint16))
    out_count = _ceil_div(p["align_out_c"], c2) * p["out_width_stride"] * c2
    output_bytes = out_count * np.dtype(read_dtype).itemsize
    own = session is None
    if own:
        session = TileSession(max(4096, input_flat.nbytes), max(4096, weight_flat.nbytes), max(4096, output_bytes))
    try:
        _copy_u16(session.input_map, input_flat)
        _copy_u16(session.weight_map, weight_flat)
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(session.output_map)), 0, min(len(session.output_map), output_bytes))
        write_tasks(session.task_map, session.regcmd_map, session.regcmd_mem,
                    [make_regs(tile_shape, p, session.input_mem.dma_addr, session.weight_mem.dma_addr,
                               session.output_mem.dma_addr, out_fp16, full_data_bank=full_data_bank)])
        if npu_submit(session.fd, session.task_mem.obj_addr, 1) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(session.output_map, dtype=read_dtype, count=out_count).copy()
        post_submit_reset(session.fd)
    finally:
        if own:
            session.close()
    return unpack_output(out_raw, tile_shape["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], c2)


def compute_expected(inp, wt, s):
    """Vectorized NCHW reference: batch, groups (incl. depthwise), stride, valid pad."""
    stride = s.get("stride", 1)
    out_h = (s["in_h"] - s["kh"]) // stride + 1
    out_w = (s["in_w"] - s["kw"]) // stride + 1
    expected = np.zeros((s["batch"], s["out_c"], out_h, out_w), dtype=np.float32)
    i32, w32 = inp.astype(np.float32), wt.astype(np.float32)
    in_per = s["in_c"] // s["groups"]
    out_per = s["out_c"] // s["groups"]
    for n in range(s["batch"]):
        for g in range(s["groups"]):
            oc0, oc1 = g * out_per, (g + 1) * out_per
            xin = i32[n, g * in_per:(g + 1) * in_per]
            w = w32[oc0:oc1]
            for kh in range(s["kh"]):
                for kw in range(s["kw"]):
                    expected[n, oc0:oc1] += np.einsum(
                        "ihw,oi->ohw",
                        xin[:, kh:kh + out_h * stride:stride, kw:kw + out_w * stride:stride],
                        w[:, :, kh, kw],
                    )
    return expected


def _padded_weight(wt, oc_start, oc_count, hw_oc, weight_in_c, kh, kw):
    if hw_oc == oc_count:
        return wt[oc_start:oc_start + oc_count]
    out = np.zeros((hw_oc, weight_in_c, kh, kw), dtype=np.float16)
    out[:oc_count] = wt[oc_start:oc_start + oc_count]
    return out


def _tile_full_data_bank(s, row, p):
    """When to force data_bank=11 (y_tile / feature-heavy semantics).

    - Pointwise (incl. NONE/setup): always — large-IC NONE fails without full banks.
    - Spatial Y windows: always (matches make_y_tile_regs).
    - Spatial K tiles: only if OC-tile weights fit in 1 bank (else hang).
    """
    if not p["is_spatial"]:
        return True
    if row["family"] in ("y_tile", "yk_tile") or row["input_h"] < s["in_h"]:
        return True
    if row["family"] != "k_tile":
        return False
    oc = hw_oc_for(s, row["oc_count"])
    aligned_in = _align_up(s["in_c"], p["align_c"])
    wt_bytes = s["kh"] * s["kw"] * aligned_in * FP16_BYTES * oc
    return wt_bytes <= CBUF_BANK_SIZE


def run_planned(s, inp, wt):
    rows, p, _, _ = plan_local_serial_rows(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    tasks = 0
    with TileSession() as session:
        for n in range(s["batch"]):
            for row in rows:
                oc_count = row["oc_count"]
                hw_oc = hw_oc_for(s, oc_count)
                tile_shape = dict(s, name=s["name"] + "_local", batch=1, groups=1,
                                  in_h=row["input_h"], out_c=hw_oc, weight_in_c=s["in_c"])
                y0 = row["y_start"] * s.get("stride", 1)
                tile_in = inp[n, :, y0:y0 + row["input_h"], :]
                tile_wt = _padded_weight(wt, row["k_start"], oc_count, hw_oc, s["in_c"], s["kh"], s["kw"])
                tile = run_hw_tile(tile_shape, tile_in, tile_wt,
                                   full_data_bank=_tile_full_data_bank(s, row, p), session=session)[:oc_count]
                oh = min(tile.shape[1], row["output_h"], p["out_h"] - row["y_start"])
                got[n, row["k_start"]:row["k_start"] + oc_count, row["y_start"]:row["y_start"] + oh] = tile[:, :oh]
                tasks += 1
    return got, tasks, rows[0]["split_method"]


def run_depthwise_serial(s, inp, wt):
    """Depthwise as per-channel group=1 spatial tiles; reuse one TileSession."""
    p = _conv_params(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    tasks = 0
    base = dict(s, name=s["name"] + "_dw", batch=1, in_c=1, out_c=MIN_HW_OC, weight_in_c=1, groups=1)
    y_rows, _, _, _ = plan_local_serial_rows(base)
    with TileSession() as session:
        for n in range(s["batch"]):
            for ch in range(s["out_c"]):
                for row in y_rows:
                    tile_shape = dict(base, name=s["name"] + f"_dw{ch}", in_h=row["input_h"])
                    tile_wt = np.zeros((MIN_HW_OC, 1, s["kh"], s["kw"]), dtype=np.float16)
                    tile_wt[0] = wt[ch]
                    y0 = row["y_start"] * s.get("stride", 1)
                    tile_in = inp[n, ch:ch + 1, y0:y0 + row["input_h"], :]
                    y_semantics = row["family"] in ("y_tile", "yk_tile") or row["input_h"] < s["in_h"]
                    tile = run_hw_tile(tile_shape, tile_in, tile_wt, full_data_bank=y_semantics, session=session)[0]
                    oh = min(tile.shape[0], row["output_h"], p["out_h"] - row["y_start"])
                    got[n, ch, row["y_start"]:row["y_start"] + oh] = tile[:oh]
                    tasks += 1
    return got, tasks, "depthwise_serial"


def run_grouped_serial(s, inp, wt):
    if s["in_c"] % s["groups"] or s["out_c"] % s["groups"]:
        raise ValueError("grouped path requires divisible input/output channels")
    p = _conv_params(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    in_per, out_per = s["in_c"] // s["groups"], s["out_c"] // s["groups"]
    tasks = 0
    for n in range(s["batch"]):
        for g in range(s["groups"]):
            ic0, oc0 = g * in_per, g * out_per
            gshape = dict(s, name=s["name"] + f"_g{g}", batch=1, in_c=in_per, out_c=out_per,
                          weight_in_c=in_per, groups=1)
            g_got, g_tasks, _ = run_planned(gshape, inp[n:n + 1, ic0:ic0 + in_per], wt[oc0:oc0 + out_per])
            got[n, oc0:oc0 + out_per] = g_got[0]
            tasks += g_tasks
    return got, tasks, "grouped_serial"


def _prefer_pointwise_gemm(s, split):
    """GEMM escape hook — currently unused (native pointwise covers 217).

    Kept for a future unsafe body; 32-align pack + bank-aware data_bank + k_step cap
    reclaimed c40/c72/c528/c576/c832/c1024/c1280 families.
    """
    return False


def _prefer_spatial_gemm(s, split):
    """No spatial GEMM escape by default — tiny maps fixed via DMA surf formula.

    Kept as a hook if a future native body is proven unsafe; currently always False.
    """
    return False


_gemm_npu_mod = None


def _load_gemm_npu():
    """Load sibling gemm_npu.py by path (works for script and package import)."""
    global _gemm_npu_mod
    if _gemm_npu_mod is not None:
        return _gemm_npu_mod
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_npu.py")
    spec = importlib.util.spec_from_file_location("conv_grok_gemm_npu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _gemm_npu_mod = mod
    return mod


def run_shape(s, dry_run=False):
    rows, pp, y_step, k_step = plan_local_serial_rows(s)
    split = rows[0]["split_method"]
    print(f"plan shape={s['name']} split={split} rows={len(rows)} y_step={y_step} k_step={k_step} "
          f"out={pp['out_h']}x{pp['out_w']} groups={s['groups']}")
    if dry_run:
        for i, row in enumerate(rows[:16]):
            print(f"  [{i}] {row['family']} y={row['y_start']}+{row['output_h']} "
                  f"in_h={row['input_h']} k={row['k_start']}+{row['oc_count']}")
        if len(rows) > 16:
            print(f"  ... {len(rows) - 16} more rows")
        return True, 0.0, 0, split

    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    expected = compute_expected(inp, wt, s)

    if _is_depthwise(s["in_c"], s["out_c"], s["groups"]):
        got, tasks, kind = run_depthwise_serial(s, inp, wt)
    elif s["groups"] != 1:
        got, tasks, kind = run_grouped_serial(s, inp, wt)
    elif _prefer_pointwise_gemm(s, split):
        got, tasks, kind = _load_gemm_npu().run_pointwise_gemm(s, inp, wt)
    elif _prefer_spatial_gemm(s, split):
        got, tasks, kind = _load_gemm_npu().run_spatial_gemm(s, inp, wt)
    else:
        got, tasks, kind = run_planned(s, inp, wt)

    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got.astype(np.float32), expected, atol=0.12, rtol=0.02))
    print(f"shape={s['name']} kind={kind} tasks={tasks} {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    return ok, max_diff, tasks, kind


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone first-principles FP16 CONV (local serial)")
    parser.add_argument("shape", nargs="?", help="encoded shape name")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print planner rows only, no submit")
    parser.add_argument("--sweep", action="store_true", help="delegate to conv_grok/sweep_217.py")
    parser.add_argument("--classify", action="store_true", help="delegate to sweep --classify (no NPU)")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--pattern", type=str, default="")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--extra-hard", action="store_true", help="run EXTRA_HARD_SHAPES stress set")
    args = parser.parse_args(argv)
    if args.sweep or args.classify:
        # Keep conv.py lean: forward to sibling sweep harness.
        sweep_argv = []
        if args.sweep:
            sweep_argv.append("--sweep")
        if args.classify:
            sweep_argv.append("--classify")
        sweep_argv += ["--timeout", str(args.timeout)]
        if args.limit:
            sweep_argv += ["--limit", str(args.limit)]
        if args.start:
            sweep_argv += ["--start", str(args.start)]
        if args.pattern:
            sweep_argv += ["--pattern", args.pattern]
        if args.skip_health:
            sweep_argv.append("--skip-health")
        if args.stop_on_error:
            sweep_argv.append("--stop-on-error")
        import importlib.util
        sweep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_217.py")
        spec = importlib.util.spec_from_file_location("conv_grok_sweep_217", sweep_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main(sweep_argv)
    if args.list:
        print("encoded: [conv2d_]bN_cN_hN_wN_ocN_wicN_kHxW_gN[_sN][_pvalid]")
        for status, name in LIST_SHAPES:
            print(f"{status:8s} {name}")
        print("extra-hard:")
        for name in EXTRA_HARD_SHAPES:
            print(f"{'extra':8s} {name}")
        return 0
    if args.extra_hard:
        names = EXTRA_HARD_SHAPES
    else:
        names = (args.shape,) if args.shape else DEFAULT_SMOKE
    failed = 0
    for name in names:
        try:
            ok, _, _, _ = run_shape(shape_from_name(name), dry_run=args.dry_run)
            if not args.dry_run and not ok:
                failed += 1
        except Exception as exc:
            print(f"shape={name} ERROR: {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
