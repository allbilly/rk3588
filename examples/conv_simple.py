import os, mmap, sys, ctypes, argparse
from fcntl import ioctl
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conv_expt import conv_tile_planner as planner  # noqa: E402


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


class reg:
    CNA = 0x0201
    CORE = 0x0801
    DPU = 0x1001
    PC = 0x0081
    PC_REG = 0x0101
    VERSION = 0x0041
    OPERATION_ENABLE = 0x0008
    PC_BASE_ADDRESS = 0x0010
    PC_REGISTER_AMOUNTS = 0x0014
    S_POINTER = 0x4004
    FEATURE_MODE_CFG = 0x400c
    DATA_FORMAT = 0x4010
    DST_BASE_ADDR = 0x4020
    DST_SURF_STRIDE = 0x4024
    DATA_CUBE_WIDTH = 0x4030
    DATA_CUBE_HEIGHT = 0x4034
    DATA_CUBE_NOTCH = 0x4038
    DATA_CUBE_CHANNEL = 0x403c
    BS_CFG = 0x4040
    BS_OW_CFG = 0x4050
    WDMA_SIZE_0 = 0x4058
    WDMA_SIZE_1 = 0x405c
    BN_CFG = 0x4060
    EW_CFG = 0x4070
    EW_CVT_SCALE_VALUE = 0x4078
    OUT_CVT_SCALE = 0x4084
    SURFACE_ADD = 0x40c0
    CNA_CONV_CON1 = 0x100c
    CNA_CONV_CON2 = 0x1010
    CNA_CONV_CON3 = 0x1014
    CNA_DATA_SIZE0 = 0x1020
    CNA_DATA_SIZE1 = 0x1024
    CNA_DATA_SIZE2 = 0x1028
    CNA_DATA_SIZE3 = 0x102c
    CNA_WEIGHT_SIZE0 = 0x1030
    CNA_WEIGHT_SIZE1 = 0x1034
    CNA_WEIGHT_SIZE2 = 0x1038
    CNA_CBUF_CON0 = 0x1040
    CNA_CBUF_CON1 = 0x1044
    CNA_CVT_CON0 = 0x104c
    CNA_CVT_CON1 = 0x1050
    CNA_CVT_CON2 = 0x1054
    CNA_CVT_CON3 = 0x1058
    CNA_CVT_CON4 = 0x105c
    CNA_CVT_CON5 = 0x1180
    CNA_FEATURE_DATA_ADDR = 0x1070
    CNA_DMA_CON0 = 0x1078
    CNA_DMA_CON1 = 0x107c
    CNA_DMA_CON2 = 0x1080
    CNA_FC_DATA_SIZE0 = 0x1084
    CNA_FC_DATA_SIZE1 = 0x1088
    CNA_DCOMP_ADDR0 = 0x1110
    CORE_MISC_CFG = 0x3010
    CORE_DATAOUT_SIZE_0 = 0x3014
    CORE_DATAOUT_SIZE_1 = 0x3018
    CORE_RESERVED_3030 = 0x3030


class rknpu_mem_create(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("size", ctypes.c_uint64),
                ("obj_addr", ctypes.c_uint64), ("dma_addr", ctypes.c_uint64), ("sram_size", ctypes.c_uint64)]


class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64)]


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
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))


LIST_SHAPES = (
    ("supported", "b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid"),
    ("supported", "conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1"),
    ("try-NONE", "b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid"),
    ("try-NONE", "b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid"),
    ("try-NONE", "conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1"),
    ("disabled", "conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1"),
    ("disabled", "conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1"),
    ("disabled", "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid"),
)


def _ceil_div(x, y):
    return (x + y - 1) // y


def _align_up(x, align):
    return _ceil_div(x, align) * align


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


def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr


def _pairs(target, items):
    return tuple(E(target, addr, value) for addr, value in items)


def _zero_range(start, end):
    return tuple((addr, 0) for addr in range(start, end + 1, 4))


def setup_full_reg_qwords():
    cna = (
        (0x1040, 0xb1), (0x1104, 0), (0x1100, 0), (0x100c, 0x120),
        (0x100c, 0x120), (0x1010, 0x110), (0x1014, 9), (0x1020, 0xe000e),
        (0x1024, 0x1f0020), (0x1028, 0xc), (0x102c, 0x90), (0x1030, 0x12000),
        (0x1034, 0x240), (0x1038, 0x3030080), (0x1040, 0xb1), (0x1044, 0xe),
        (0x104c, 0xb), (0x1050, 0x10000), (0x1054, 0x10000), (0x1058, 0x10000),
        (0x105c, 0x10000), (0x1060, 0), (0x1064, 0), (0x1068, 0), (0x1070, 0),
        (0x1074, 0), (0x1078, 0xf000f), (0x107c, 0x38), (0x1080, 0x8c),
        (0x1084, 0xe000e), (0x1088, 0x20), (0x1100, 0), (0x1104, 0), (0x1110, 0),
        *_zero_range(0x1140, 0x117c), (0x1180, 0), (0x1184, 0),
    )
    core = ((0x3010, 0x200), (0x3014, 0xb000b), (0x3018, 0x7f), (0x301c, 0), (0x3030, 0))
    dpu = (
        (0x400c, 0x1e4), (0x4010, 0x48000002), (0x4014, 0), (0x4020, 0),
        (0x4024, 0x900), (0x4030, 0xb), (0x4034, 0xb), (0x4038, 0),
        (0x403c, 0x7f007f), (0x4040, 0x53), (0x4044, 0), (0x4048, 0),
        (0x404c, 0), (0x4050, 0x126), (0x4054, 0), (0x4058, 0x7f),
        (0x405c, 0xb000b), (0x4060, 0x53), (0x4064, 0), (0x4068, 0),
        (0x406c, 0), (0x4070, 0x383), (0x4074, 0), (0x4078, 1), (0x407c, 0),
        (0x4080, 0), (0x4084, 0x10001), (0x4088, 0), *_zero_range(0x4090, 0x409c),
        *_zero_range(0x40a0, 0x40ac), (0x40c0, 0x1200), (0x40c4, 0), *_zero_range(0x4100, 0x412c),
    )
    return _pairs(reg.CNA, cna[:4]) + (E(reg.DPU, 0x4004, 0xe),) + _pairs(reg.CNA, cna[4:]) + _pairs(reg.CORE, core) + _pairs(reg.DPU, dpu)


def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c


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
    in_c, in_h, in_w, out_c, kh, kw, groups = s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["kh"], s["kw"], s["groups"]
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    is_spatial = kh != 1 or kw != 1
    out_h, out_w = in_h - kh + 1, in_w - kw + 1
    align_c = _conv_align_c(in_c, groups, out_c)
    align_out_c = max(16, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_atoms = max(1, out_h * out_w)
    out_width_stride = out_atoms if (not is_spatial and out_atoms < 4) else _align_up(out_atoms, 4)
    input_pack_c2 = _conv_input_pack_c2(in_c, groups, out_c, align_c)
    use_nhwc = not is_depthwise and not (groups > 1 and is_spatial) and input_pack_c2 // in_c == 2
    return dict(is_depthwise=is_depthwise, is_spatial=is_spatial, out_h=out_h, out_w=out_w, align_c=align_c,
                align_out_c=align_out_c, width_stride=width_stride, out_width_stride=out_width_stride,
                input_pack_c2=input_pack_c2, use_nhwc=use_nhwc)


def _dma_strides(in_h, width_stride, use_nhwc_pack):
    if use_nhwc_pack:
        return width_stride, width_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, width_stride * (in_h - 4) if in_h > 4 else 0


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


def _is_pointwise_wide(s):
    return s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1 and s["in_c"] >= 64


def _pack_pointwise_wide(weight, out_c, in_c):
    aligned_in_c = max(32, _align_up(in_c, 32))
    padded = np.zeros((out_c, aligned_in_c), dtype=np.float16)
    padded[:out_c, :in_c] = weight[:, :in_c, 0, 0]
    return np.concatenate([padded[oc:oc + 16].reshape(-1, aligned_in_c // 32, 32).transpose(1, 0, 2).ravel() for oc in range(0, out_c, 16)])


def _pack_kh_major(weight, out_c, in_c, kh, kw, c2_out):
    aligned_in_c = c2_out * _ceil_div(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:, :in_c] = weight
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


def make_regs(s, p, in_dma, wt_dma, out_dma, out_fp16):
    if s["name"] == "b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid":
        return patch_regs(setup_full_reg_qwords(), {
            (reg.CNA, reg.CNA_FEATURE_DATA_ADDR): in_dma,
            (reg.CNA, reg.CNA_DCOMP_ADDR0): wt_dma,
            (reg.DPU, reg.DST_BASE_ADDR): out_dma,
        })
    in_c, in_h, in_w, out_c, kh, kw, groups = s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["kh"], s["kw"], s["groups"]
    align_c, align_out_c = p["align_c"], p["align_out_c"]
    out_h, out_w, is_spatial = p["out_h"], p["out_w"], p["is_spatial"]
    data_in_channel_aligned = _align_up(in_c, align_c)
    weight_channel_aligned = _align_up(in_c, 32) if _is_pointwise_wide(s) else data_in_channel_aligned
    weight_bytes_per_kernel = kh * kw * weight_channel_aligned * FP16_BYTES
    feature_grains = _feature_grains(p["width_stride"] * data_in_channel_aligned * FP16_BYTES, in_h + kh, p["use_nhwc"], is_spatial, False)
    data_bank = _data_bank(p["width_stride"], feature_grains, data_in_channel_aligned, p["use_nhwc"], is_spatial, False)
    out_precision = 2 if out_fp16 else 5
    size_e = 1 if out_fp16 else 3
    out_channel_field = align_out_c - 1
    cvt_con0 = 0x0b if is_spatial and not p["is_depthwise"] else 1
    cvt_con5 = ((1 << in_c) if p["use_nhwc"] else p["input_pack_c2"]) - 1
    conv_con1 = (2 << 4) | (2 << 7) | (((1 << 30) | (1 << 29) | ((7 + in_c) << 12)) if p["use_nhwc"] and in_c <= 4 else 0)
    regs = [
        E(reg.DPU, reg.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.CNA, reg.CNA_CONV_CON1, conv_con1),
        E(reg.CNA, reg.CNA_CONV_CON2, feature_grains << 4),
        E(reg.CNA, reg.CNA_CONV_CON3, (1 << 3) | 1),
        E(reg.CNA, reg.CNA_DATA_SIZE0, (p["width_stride"] << 16) | in_h),
        E(reg.CNA, reg.CNA_DATA_SIZE1, ((in_c - 1) << 16) | data_in_channel_aligned),
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_w * out_h),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_per_kernel * out_c),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2, (kw << 24) | (kh << 16) | out_c),
        E(reg.CNA, reg.CNA_CBUF_CON0, ((RK_CBUF_BANKS - data_bank) << 4) | data_bank),
        E(reg.CNA, reg.CNA_CBUF_CON1, _cbuf_entries(p["width_stride"], data_in_channel_aligned, in_h, False)),
        E(reg.CNA, reg.CNA_CVT_CON0, cvt_con0), E(reg.CNA, reg.CNA_CVT_CON1, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON2, 1 << 16), E(reg.CNA, reg.CNA_CVT_CON3, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON4, 1 << 16), E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        E(reg.CNA, reg.CNA_DMA_CON1, _dma_strides(in_h, p["width_stride"], p["use_nhwc"])[0]),
        E(reg.CNA, reg.CNA_DMA_CON2, _dma_strides(in_h, p["width_stride"], p["use_nhwc"])[1]),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, (in_w << 16) | in_h),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, data_in_channel_aligned), E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, cvt_con5),
        E(reg.CORE, reg.CORE_MISC_CFG, (2 << 8) | is_spatial), E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field), E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1)), E(reg.DPU, reg.DATA_FORMAT, (out_precision << 29) | (2 << 26) | 2),
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma), E(reg.DPU, reg.DST_SURF_STRIDE, p["out_width_stride"] << 4),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1), E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1), E(reg.DPU, reg.DATA_CUBE_NOTCH, 0),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL, ((out_c - 1) << 16) | out_channel_field),
        E(reg.DPU, reg.BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.BS_OW_CFG, (size_e << 8) | (size_e << 5) | (size_e << 2) | (1 << 1)),
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field), E(reg.DPU, reg.WDMA_SIZE_1, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.DPU, reg.BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1), E(reg.DPU, reg.EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1), E(reg.DPU, reg.OUT_CVT_SCALE, ((1 << 16) | 1) if out_fp16 else 0),
        E(reg.DPU, reg.SURFACE_ADD, (p["out_width_stride"] * max(2, align_out_c // 16)) << 4),
    ]
    return regs


def make_y_tile_regs(s, p, row, in_dma, wt_dma, out_dma, input_off):
    tile_shape = dict(s, in_h=row["input_h"])
    tile_p = _conv_params(tile_shape)
    regs = make_regs(tile_shape, tile_p, in_dma + input_off, wt_dma, out_dma + row["y_start"] * p["out_w"] * 16, True)
    data_bank = RK_CBUF_BANKS - 1
    return patch_regs(regs, {
        (reg.CNA, reg.CNA_CBUF_CON0): (int(row["weight_reuse"]) << 13) | ((RK_CBUF_BANKS - data_bank) << 4) | data_bank,
        (reg.DPU, reg.DST_SURF_STRIDE): p["out_width_stride"] << 4,
        (reg.DPU, reg.SURFACE_ADD): (p["out_width_stride"] * max(2, p["align_out_c"] // 16)) << 4,
    })


def patch_regs(regs, values):
    patched = []
    for qword in regs:
        key = (qword >> 48, qword & 0xffff)
        patched.append(E(key[0], key[1], values[key]) if key in values else qword)
    return patched


def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    return mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset), mem_create


def write_tasks(task_map, regcmd_map, regcmd_mem, task_regs):
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + PC_CHAIN_TAIL_QWORDS, 2)
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            next_addr = regcmd_mem.dma_addr + offsets[idx + 1] * ctypes.sizeof(ctypes.c_uint64)
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
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)


def npu_submit(fd, task_obj_addr, task_count):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))
    submit = rknpu_submit(flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK, timeout=6000, task_start=0, task_number=task_count,
                          task_counter=0, priority=0, task_obj_addr=task_obj_addr, iommu_domain_id=0,
                          reserved=0, task_base_addr=0, hw_elapse_time=0, core_mask=1, fence_fd=-1)
    submit.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit)


def validate_phase_a_shape(s):
    rows = planner.descriptor_rows_for_shape(s)
    families = {row["family"] for row in rows}
    if len(rows) == 1 and families == {"setup"} and rows[0]["split_method"] == "NONE":
        if _is_pointwise_wide(s):
            raise ValueError("pointwise-wide NONE is disabled after c144 mismatch; no allocation or submit attempted")
        return rows
    if rows and rows[0]["split_method"] == "BY_K":
        raise ValueError("BY_K/k_tile is disabled after h14 timed out with body-parity and PC-core fixes; no allocation or submit attempted")
    if rows and rows[0]["split_method"] == "BY_Y":
        if len(families) == 1 and families == {"y_tile"} and s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1:
            return rows
        raise ValueError("BY_Y/y_tile is disabled before allocation except pointwise row tiling")
    if rows and rows[0]["split_method"] == "BY_YK":
        raise ValueError("BY_YK is disabled before allocation; mixed Y/K setup and k_half semantics are unresolved")
    else:
        raise ValueError(f"Phase A supports only one setup/NONE descriptor; got split={rows[0]['split_method'] if rows else 'none'} families={sorted(families)} rows={len(rows)}")


def compute_expected(inp, wt, s):
    out_h, out_w = s["in_h"] - s["kh"] + 1, s["in_w"] - s["kw"] + 1
    expected = np.zeros((s["batch"], s["out_c"], out_h, out_w))
    i64, w64 = inp.astype(np.float64), wt.astype(np.float64)
    for n in range(s["batch"]):
        for oc in range(s["out_c"]):
            for ic in range(s["in_c"]):
                for i in range(s["kh"]):
                    for j in range(s["kw"]):
                        expected[n, oc] += i64[n, ic, i:i + out_h, j:j + out_w] * w64[oc, ic, i, j]
    return expected


def run_shape(s):
    rows = validate_phase_a_shape(s)
    p = _conv_params(s)
    hw_out_fp16 = p["is_spatial"] or s["out_c"] >= 128 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    weight_flat = pack_weights(wt, s, p).view(np.uint16)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        if rows[0]["split_method"] == "BY_Y":
            input_base = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
            task_regs = []
            input_offset = 0
            for row in rows:
                tile_shape = dict(s, in_h=row["input_h"])
                tile_flat = pack_input(inp[0, :, row["y_start"]:row["y_start"] + row["input_h"], :], _conv_params(tile_shape)).view(np.uint16)
                ctypes.memmove(input_base + input_offset, tile_flat.ctypes.data, tile_flat.nbytes)
                task_regs.append(make_y_tile_regs(s, p, row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, input_offset))
                input_offset = _align_up(input_offset + tile_flat.nbytes, 16)
        else:
            input_flat = pack_input(inp[0], p).view(np.uint16)
            (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
            task_regs = [make_regs(s, p, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, hw_out_fp16)]
        write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        if npu_submit(fd, task_mem.obj_addr, len(task_regs)) < 0:
            raise RuntimeError("npu_submit failed")
        read_dtype = np.float16 if hw_out_fp16 else np.float32
        c2 = UNPACK_C2 if hw_out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
        out_count = _ceil_div(p["align_out_c"], c2) * p["out_width_stride"] * c2
        out_raw = np.frombuffer(output_map, dtype=read_dtype, count=out_count).copy()
    finally:
        os.close(fd)
    got = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], c2).reshape(1, s["out_c"], p["out_h"], p["out_w"])
    expected = compute_expected(inp, wt, s)
    atol = 0.12 if hw_out_fp16 else 0.1
    max_diff = float(np.max(np.abs(got.astype(np.float64) - expected)))
    ok = bool(np.allclose(got, expected, atol=atol))
    print(f"shape={s['name']} split={rows[0]['split_method']} family={rows[0]['family']} tasks={len(task_regs)} regs={';'.join(str(len(regs)) for regs in task_regs)} hw_out={'fp16' if hw_out_fp16 else 'fp32'} {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Planner-driven Phase A CONV submit path")
    parser.add_argument("shape", nargs="?", help="supported shape name")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print("encoded shape syntax: [conv2d_]bN_cN_hN_wN_ocN_wicN_kHxW_gN[_sN][_pvalid]")
        print("known examples:")
        print("\n".join(f"{status:9s} {name}" for status, name in LIST_SHAPES))
        return 0
    if not args.shape:
        print("error: shape is required unless --list is used", file=sys.stderr)
        return 1
    try:
        run_shape(shape_from_name(args.shape))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
