import os, sys, mmap, ctypes, numpy as np
from dataclasses import dataclass
from fcntl import ioctl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../experimental/kernel_6_18")))
import rocket_runtime as rt

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
FP16_BYTES = 2
FP16_ATOM_ELEMENTS = 16
WEIGHT_ATOMIC_ELEMENTS = 32
UNPACK_C2 = 8
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_MAX_CONV_FLAT_STRIDE = 992
PC_CHAIN_TAIL_QWORDS = 4
DPU_CLEAR_REGS = tuple(range(0x4100, 0x4130, 4))
DIRECT_SPATIAL_ENV = "RK3588_CONV_DIRECT_SPATIAL"
DIRECT_SPATIAL_UNSAFE_ENV = "RK3588_CONV_DIRECT_SPATIAL_UNSAFE"
DIRECT_SPATIAL_TASKS_ENV = "RK3588_CONV_DIRECT_SPATIAL_TASKS"
TASK_POLICY_REG_LISTS = "reg_lists"
TASK_POLICY_RAW_SPANS = "raw_spans"
DIRECT_SPATIAL_POLICY_SINGLE_STREAM = "single_stream"
DIRECT_SPATIAL_POLICY_SPARSE_SINGLE = "sparse_single"
DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS = "rocket_record_amounts"
BUFFER_SCOPE_TILE = "tile"
BUFFER_SCOPE_FULL = "full"
UNPACK_KIND_NC1HWC2 = "nc1hwc2"
UNPACK_KIND_FLAT_1X1 = "flat_1x1"
UNPACK_KIND_DIRECT_SPATIAL = "direct_spatial"
FAMILY_DIRECT_SPATIAL = "direct_spatial"
FAMILY_DIRECT_SPATIAL_GATED = "direct_spatial_gated"
FAMILY_GROUPED_SERIAL = "grouped_serial"
FAMILY_SPATIAL_IM2COL_FALLBACK = "spatial_im2col_fallback"
FAMILY_SPATIAL_OC_SERIAL = "spatial_oc_serial"
FAMILY_DEPTHWISE_SPATIAL_TILED = "depthwise_spatial_tiled"
FAMILY_POINTWISE_OC = "pointwise_oc"
FAMILY_GENERIC_YK = "generic_yk"
DIRECT_SPATIAL_POLICY_PC_ROOT6 = "pc_root6"
DIRECT_SPATIAL_PC_ROOT6_RECORD_AMOUNTS = (
    108, 108, 104, 104, 26, 104, 104, 26, 104, 104, 26, 104, 104, 26, 104, 104, 26,
)
DIRECT_SPATIAL_PC_ROOT6_LINK_RECORDS = (1, None, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16)
DIRECT_SPATIAL_PC_ROOT6_ROOT_RECORDS = (0, 2, 5, 8, 11, 14)
DIRECT_SPATIAL_PC_ROOT6_STREAM_QWORDS = 1280
DIRECT_SPATIAL_PC_ROOT6_H40_SCHEDULE = (
    ("setup", 0, 23, 21, 0, 320),
    ("setup", 21, 19, 17, 0, 320),
    ("k_half", 0, 23, 21, 0, 160),
    ("k_half", 21, 19, 17, 0, 160),
    ("k_half", 0, 23, 21, 160, 160),
    ("k_half", 21, 19, 17, 160, 160),
    ("k_tile", 0, 23, 21, 0, 112),
    ("k_tile", 21, 19, 17, 0, 112),
    ("k_tile", 0, 23, 21, 112, 112),
    ("k_tile", 21, 19, 17, 112, 112),
    ("k_tile", 0, 23, 21, 224, 96),
    ("k_tile", 21, 19, 17, 224, 96),
)
DIRECT_SPATIAL_OUTC64_H20_POST = {
    32: ("k_half_y_tile", None),
    48: ("y_mid_k_tile", (16, 16, 16)),
    64: ("k_half_y_tile", None),
    96: ("k_half_k_tile", (32, 32, 32)),
    112: ("y_mid_y_tile", None),
    128: ("k_half_y_tile", None),
    160: ("k_half_y_tile", None),
    192: ("k_half_k_tile", (64, 64, 64)),
    224: ("k_half_y_tile", None),
    256: ("k_half_y_tile", None),
    288: ("k_half_k_tile", (96, 96, 96)),
    320: ("k_half_k_tile", (112, 112, 96)),
    384: ("k_half_k_tile", (128, 128, 128)),
    512: ("k_half_k_tile", (176, 176, 160)),
}
DIRECT_SPATIAL_CHAN320_H20_IN_CHANNELS = frozenset((32, 40, 64, 72, 96, 128, 192, 256))
DIRECT_SPATIAL_MIX72_H20_K_TILE_COUNTS = (96, 96, 96)
DIRECT_SPATIAL_CMP32_H14_Y_WINDOWS = ((0, 6, 4), (4, 6, 4), (8, 6, 4))
DIRECT_SPATIAL_POINTWISE_SIMPLE_Y_WINDOWS = {
    14: ((0, 5), (5, 5), (10, 4)),
    20: ((0, 7), (7, 7), (14, 6)),
    28: ((0, 10), (10, 9), (19, 9)),
    40: ((0, 14), (14, 13), (27, 13)),
    56: ((0, 19), (19, 19), (38, 18)),
}
DIRECT_SPATIAL_POINTWISE_CROSSED = {
    (256, 28, 512): {
        "setup": ((0, 9), (9, 9)),
        "k_half": ((0, 9), (9, 9), (18, 9), (27, 1)),
        "y_tile": ((0, 5), (15, 5), (5, 5), (20, 4), (10, 5), (24, 4)),
    },
    (528, 20, 32): {
        "setup": ((0, 15), (15, 5)),
        "k_half": ((0, 15), (15, 5)),
        "y_tile": ((0, 7), (7, 7), (14, 6)),
    },
    (528, 40, 32): {
        "setup": ((0, 7), (7, 7)),
        "k_half": ((0, 7), (7, 7), (14, 7), (21, 7), (28, 7), (35, 5)),
        "y_tile": ((0, 7), (21, 7), (7, 7), (28, 6), (14, 7), (34, 6)),
    },
}

class TileDesc:
    __slots__ = (
        "family", "family_bits", "grain_bits",
        "input_y", "input_h", "output_y", "output_h", "output_w",
        "oc_start", "oc_count",
        "feature_off", "weight_off", "output_off",
        "cbuf0", "cbuf1", "weight_reuse", "data_reuse", "pc_core",
        "extra",
    )
    def __init__(self, **kw):
        for s in self.__slots__:
            if s == "extra":
                default = {}
            elif s in {"grain_bits", "cbuf0", "cbuf1"}:
                default = None
            else:
                default = 0
            setattr(self, s, kw.get(s, default))
    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

@dataclass(frozen=True, slots=True)
class TileRecord:
    family: str
    split_method: int
    y_start: int
    y_count: int
    k_start: int
    k_count: int
    input_h: int
    output_w: int
    layout_mode: str
    unpack_kind: str
    task_policy: str = TASK_POLICY_REG_LISTS
    input_y: int = 0
    output_y: int = 0
    output_h: int = 0
    input_w: int = 0
    tile_in_c: int = 0
    input_c_start: int = 0
    tile_groups: int = 1
    hw_oc: int = 0
    out_offset: int = 0
    full_data_bank: bool = False
    reason: str = ""

class RegList(list):
    pass

class reg:
    CNA  = 0x0201; CORE = 0x0801; DPU  = 0x1001; RDMA = 0x2001
    PPU  = 0x4001; PPU_RDMA = 0x8001
    PC   = 0x0081; PC_REG = 0x0101; VERSION = 0x0041
    OPERATION_ENABLE    = 0x0008
    PC_BASE_ADDRESS     = 0x0010
    PC_REGISTER_AMOUNTS = 0x0014
    S_POINTER           = 0x4004
    FEATURE_MODE_CFG    = 0x400c
    DATA_FORMAT         = 0x4010
    DST_BASE_ADDR       = 0x4020
    DST_SURF_STRIDE     = 0x4024
    DATA_CUBE_WIDTH     = 0x4030
    DATA_CUBE_HEIGHT    = 0x4034
    DATA_CUBE_NOTCH     = 0x4038
    DATA_CUBE_CHANNEL   = 0x403c
    BS_CFG              = 0x4040
    BS_OW_CFG           = 0x4050
    WDMA_SIZE_0         = 0x4058
    WDMA_SIZE_1         = 0x405c
    BN_CFG              = 0x4060
    EW_CFG              = 0x4070
    EW_CVT_SCALE_VALUE  = 0x4078
    OUT_CVT_SCALE       = 0x4084
    SURFACE_ADD         = 0x40c0
    DPU_SURFACE_ADD_EXTRA = 0x40c4
    RDMA_S_POINTER      = 0x5004
    RDMA_DATA_CUBE_WIDTH  = 0x500c
    RDMA_DATA_CUBE_HEIGHT = 0x5010
    RDMA_DATA_CUBE_CHANNEL= 0x5014
    RDMA_ERDMA_CFG        = 0x5034
    RDMA_SRC_BASE_ADDR    = 0x5018
    RDMA_EW_BASE_ADDR     = 0x5038
    RDMA_FEATURE_MODE_CFG = 0x5044
    PPU_S_POINTER         = 0x6004
    PPU_DATA_CUBE_IN_WIDTH = 0x600c
    PPU_DATA_CUBE_IN_HEIGHT = 0x6010
    PPU_DATA_CUBE_IN_CHANNEL = 0x6014
    PPU_DATA_CUBE_OUT_WIDTH = 0x6018
    PPU_DATA_CUBE_OUT_HEIGHT = 0x601c
    PPU_DATA_CUBE_OUT_CHANNEL = 0x6020
    PPU_OPERATION_MODE_CFG = 0x6024
    PPU_POOLING_KERNEL_CFG = 0x6034
    PPU_RECIP_KERNEL_WIDTH = 0x6038
    PPU_RECIP_KERNEL_HEIGHT = 0x603c
    PPU_POOLING_PADDING_CFG = 0x6040
    PPU_PADDING_VALUE_1_CFG = 0x6044
    PPU_PADDING_VALUE_2_CFG = 0x6048
    PPU_DST_BASE_ADDR     = 0x6070
    PPU_DST_SURF_STRIDE   = 0x607c
    PPU_DATA_FORMAT       = 0x6084
    PPU_MISC_CTRL         = 0x60dc
    PPU_RDMA_S_POINTER    = 0x7004
    PPU_RDMA_CUBE_IN_WIDTH = 0x700c
    PPU_RDMA_CUBE_IN_HEIGHT = 0x7010
    PPU_RDMA_CUBE_IN_CHANNEL = 0x7014
    PPU_RDMA_SRC_BASE_ADDR = 0x701c
    PPU_RDMA_SRC_LINE_STRIDE = 0x7024
    PPU_RDMA_SRC_SURF_STRIDE = 0x7028
    PPU_RDMA_DATA_FORMAT  = 0x7030
    CNA_CONV_CON1          = 0x100c
    CNA_CONV_CON2          = 0x1010
    CNA_CONV_CON3          = 0x1014
    CNA_DATA_SIZE0         = 0x1020
    CNA_DATA_SIZE1         = 0x1024
    CNA_DATA_SIZE2         = 0x1028
    CNA_DATA_SIZE3         = 0x102c
    CNA_WEIGHT_SIZE0       = 0x1030
    CNA_WEIGHT_SIZE1       = 0x1034
    CNA_WEIGHT_SIZE2       = 0x1038
    CNA_CBUF_CON0          = 0x1040
    CNA_CBUF_CON1          = 0x1044
    CNA_CVT_CON0           = 0x104c
    CNA_CVT_CON1           = 0x1050
    CNA_CVT_CON2           = 0x1054
    CNA_CVT_CON3           = 0x1058
    CNA_CVT_CON4           = 0x105c
    CNA_CVT_CON5           = 0x1180
    CNA_FC_CON0            = 0x1060
    CNA_FC_CON1            = 0x1064
    CNA_PAD_CON0           = 0x1068
    CNA_FC_CON2            = 0x1074
    CNA_PAD_CON1           = 0x1184
    CNA_FEATURE_DATA_ADDR  = 0x1070
    CNA_DMA_CON0           = 0x1078
    CNA_DMA_CON1           = 0x107c
    CNA_DMA_CON2           = 0x1080
    CNA_FC_DATA_SIZE0      = 0x1084
    CNA_FC_DATA_SIZE1      = 0x1088
    CNA_DCOMP_ADDR0        = 0x1110
    CNA_DCOMP_ADDR1        = 0x1114
    CNA_DCOMP_REGNUM       = 0x1104
    CNA_DCOMP_CTRL         = 0x1100
    CNA_DCOMP_AMOUNT0      = 0x1140
    CORE_MISC_CFG          = 0x3010
    CORE_DATAOUT_SIZE_0    = 0x3014
    CORE_DATAOUT_SIZE_1    = 0x3018
    CORE_CLIP_TRUNCATE     = 0x301c
    CORE_RESERVED_3030     = 0x3030

class struct_rknpu_task(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("op_idx", ctypes.c_uint32),
        ("enable_mask", ctypes.c_uint32),
        ("int_mask", ctypes.c_uint32),
        ("int_clear", ctypes.c_uint32),
        ("int_status", ctypes.c_uint32),
        ("regcfg_amount", ctypes.c_uint32),
        ("regcfg_offset", ctypes.c_uint32),
        ("regcmd_addr", ctypes.c_uint64),
    ]

def mem_allocate(fd, size, flags=0):
    return rt.mem_allocate(fd, size, flags)

def npu_reset(fd):
    return rt.reset_npu(fd)

def npu_submit(task_count=1):
    npu_reset(fd)
    for bo in (regcmd_mem_create, input_mem_create, weight_mem_create, output_mem_create):
        rt.fini_bo(fd, bo)
    ret = rt.submit(fd, npu_tasks, task_count,
        in_bos=[regcmd_mem_create, input_mem_create, weight_mem_create],
        out_bos=[output_mem_create])
    rt.prep_bo(fd, output_mem_create)
    return ret

fd = rt.open_rocket_device()
task_map, tasks_mem_create = mem_allocate(fd, 64*1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, 512*1024, RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, 4*1024*1024, RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, 8*1024*1024, RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, 8*1024*1024, RKNPU_MEM_NON_CACHEABLE)
npu_tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
npu_regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

def _mesa_entries_per_slice(input_width, input_channels):
    atomics_per_entry = CBUF_ENTRY_BYTES // FP16_ATOM_ELEMENTS
    total_c_atomics = _ceil_div(input_channels * FP16_BYTES, FP16_ATOM_ELEMENTS)
    last_c_atomics = total_c_atomics % atomics_per_entry
    int_c_entries = (total_c_atomics // atomics_per_entry) * input_width
    frac_c_entries = input_width if last_c_atomics == 3 else _ceil_div(last_c_atomics * input_width, atomics_per_entry)
    return int_c_entries + frac_c_entries

def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c

def _conv_align_c(in_c, groups, out_c):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    if not is_depthwise and (groups > 1 or in_c > 4):
        return 16
    min_align = 16 if is_depthwise else 8
    return max(min_align, min(1 << (max(1, in_c) - 1).bit_length(), 32 if is_depthwise else 16))

def _pointwise_weight_atom_groups(in_c):
    return _ceil_div(max(in_c, 16), 32)

def _uses_pointwise_weight_atom_layout(in_c, kh, kw, groups):
    return groups == 1 and kh == 1 and kw == 1 and _pointwise_weight_atom_groups(in_c) > 1

def conv_output_hw(in_h, in_w, kh, kw, stride):
    return (in_h - kh) // stride + 1, (in_w - kw) // stride + 1

def _cdma_dc_feature_input_pack_c2(in_c, groups, out_c):
    if in_c == 1:
        return 2
    if not _is_depthwise(in_c, out_c, groups) and groups == 1 and 1 < in_c <= 4:
        return 8
    return 8

def _dma_strides(in_h, width_stride, use_nhwc_pack):
    if use_nhwc_pack:
        return width_stride, width_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, width_stride * (in_h - 4) if in_h > 4 else 0

def _cbuf_entries(width_stride, align_c, in_h, is_depthwise):
    row_entries = max(1, _ceil_div(width_stride * align_c, 2 * 16))
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

def _conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride=1):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    is_spatial = (kh != 1 or kw != 1)
    out_h, out_w = conv_output_hw(in_h, in_w, kh, kw, stride)
    align_c = 32 if _uses_pointwise_weight_atom_layout(in_c, kh, kw, groups) else _conv_align_c(in_c, groups, out_c)
    align_out_c = max(32, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_width_stride = out_h * out_w if not is_spatial else _align_up(out_h * out_w, 4)
    input_pack_c2 = _cdma_dc_feature_input_pack_c2(in_c, groups, out_c)
    use_nhwc = (not is_depthwise) and (not (groups > 1 and is_spatial)) and in_c < input_pack_c2
    return {"in_c": in_c, "in_h": in_h, "in_w": in_w, "out_c": out_c, "kh": kh, "kw": kw,
            "groups": groups, "stride": stride, "is_depthwise": is_depthwise, "is_spatial": is_spatial,
            "out_h": out_h, "out_w": out_w, "align_c": align_c, "align_out_c": align_out_c,
            "width_stride": width_stride, "out_width_stride": out_width_stride,
            "input_pack_c2": input_pack_c2, "use_nhwc": use_nhwc}

def make_conv2d_regs(p, in_dma, wt_dma, out_dma, weight_reuse=False, full_data_bank=False):
    in_c, in_h, in_w, out_c = p["in_c"], p["in_h"], p["in_w"], p["out_c"]
    kh, kw, groups, stride = p["kh"], p["kw"], p["groups"], p["stride"]
    is_depthwise, is_spatial = p["is_depthwise"], p["is_spatial"]
    out_h, out_w = p["out_h"], p["out_w"]
    align_c, align_out_c = p["align_c"], p["align_out_c"]
    width_stride, out_width_stride = p["width_stride"], p["out_width_stride"]
    use_nhwc_pack, input_pack_c2 = p["use_nhwc"], p["input_pack_c2"]

    data_in_channel_aligned = _align_up(in_c, align_c)
    weight_bytes_per_kernel = kh * kw * data_in_channel_aligned * FP16_BYTES
    weight_bytes_total = weight_bytes_per_kernel if is_depthwise else weight_bytes_per_kernel * out_c
    input_row_bytes = width_stride * data_in_channel_aligned * FP16_BYTES
    feature_grains = _feature_grains(input_row_bytes, in_h + kh, use_nhwc_pack, is_spatial, is_depthwise)
    if not is_spatial:
        feature_grains = max(feature_grains, 52)
    line_stride, surf_stride = _dma_strides(in_h, width_stride, use_nhwc_pack)
    cbuf_entries = _cbuf_entries(width_stride, data_in_channel_aligned, in_h, is_depthwise)
    data_bank = _data_bank(width_stride, feature_grains, data_in_channel_aligned, use_nhwc_pack, is_spatial, is_depthwise)
    if full_data_bank:
        data_bank = RK_CBUF_BANKS - 1
    out_channel_field = align_out_c - 1 if not is_depthwise else _align_up(align_out_c, 32) - 1
    mesa_aligned_small = (not is_depthwise) and groups == 1 and 1 < in_c <= 4
    cvt_con5 = 65535 if (in_c == 1 and not is_depthwise) else (0 if (not is_depthwise and groups == 1) else ((1 << in_c if use_nhwc_pack else input_pack_c2) - 1))

    npu_regs = [
        E(reg.DPU, reg.S_POINTER, ((1 << 3) | (1 << 2) | (1 << 1))),
        E(reg.RDMA, reg.RDMA_S_POINTER, 0),
        E(reg.RDMA, reg.RDMA_ERDMA_CFG, 0),
        E(reg.RDMA, reg.RDMA_FEATURE_MODE_CFG, 0),
        E(reg.CNA, reg.CNA_CONV_CON1,
            ((2 << 4) | (2 << 7) |
             (((1 << 30) | (1 << 29) | ((7 + in_c) << 12)) if (use_nhwc_pack and in_c <= 4 and not is_depthwise) else 0) |
             (3 if is_depthwise else 0))),
        E(reg.CNA, reg.CNA_CONV_CON2, (feature_grains << 4)),
        E(reg.CNA, reg.CNA_CONV_CON3, ((stride << 3) | (stride << 0))),
        E(reg.CNA, reg.CNA_DATA_SIZE0, ((width_stride << 16) | in_h)),
        E(reg.CNA, reg.CNA_DATA_SIZE1, (((in_c - 1) << 16) | data_in_channel_aligned)),
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_w * out_h),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_total),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2, ((kw << 24) | (kh << 16) | (1 if is_depthwise else out_c))),
        E(reg.CNA, reg.CNA_CBUF_CON0, ((weight_reuse << 13) | (RK_CBUF_BANKS - data_bank << 4) | data_bank)),
        E(reg.CNA, reg.CNA_CBUF_CON1, cbuf_entries),
        E(reg.CNA, reg.CNA_CVT_CON0, ((use_nhwc_pack << 3) | (use_nhwc_pack << 1) | 1)),
        E(reg.CNA, reg.CNA_CVT_CON1, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON2, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON3, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON4, (1 << 16)),
        E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0, ((15 << 16) | 15)),
        E(reg.CNA, reg.CNA_DMA_CON1, line_stride),
        E(reg.CNA, reg.CNA_DMA_CON2, surf_stride),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, ((in_w << 16) | in_h)),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, data_in_channel_aligned),
        E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, cvt_con5),
        E(reg.CORE, reg.CORE_MISC_CFG, ((2 << 8) | (is_depthwise << 1) | is_spatial)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, (((out_h - 1) << 16) | (out_w - 1))),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field),
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, ((15 << 5) | ((3 * is_depthwise) << 3) | (2 << 1))),
        E(reg.DPU, reg.DATA_FORMAT, ((2 << 29) | (2 << 26) | 2)),
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma),
        E(reg.DPU, reg.DST_SURF_STRIDE, out_width_stride << 4),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1),
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),
        E(reg.DPU, reg.DATA_CUBE_NOTCH, 0),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL, (((out_c - 1) << 16) | out_channel_field)),
        E(reg.DPU, reg.BS_CFG, ((1 << 6) | (1 << 4) | (1 << 1) | 1)),
        E(reg.DPU, reg.BS_OW_CFG, (((3 if is_depthwise else 1) << 8) | ((3 if is_depthwise else 1) << 5) | ((3 if is_depthwise else 1) << 2) | (1 << 1))),
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field),
        E(reg.DPU, reg.WDMA_SIZE_1, (((out_h - 1) << 16) | (out_w - 1))),
        E(reg.DPU, reg.BN_CFG, ((1 << 6) | (1 << 4) | (1 << 1) | 1)),
        E(reg.DPU, reg.EW_CFG, ((1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1)),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1),
        E(reg.DPU, reg.OUT_CVT_SCALE, ((1 << 16) | 1)),
    ]
    effective_align_out = max(16, _align_up(_ceil_div(out_c, groups), 16)) if (groups > 1 and not is_depthwise) else out_channel_field + 1
    npu_regs.append(E(reg.DPU, reg.SURFACE_ADD, (out_width_stride * max(2, effective_align_out // 16)) << 4))
    return npu_regs

def make_conv2d_regs_from_desc(p, desc, in_dma, wt_dma, out_dma):
    tile_p = dict(p)
    tile_p.update({
        "in_h": desc.input_h,
        "out_h": desc.output_h,
        "out_w": desc.output_w,
        "out_c": desc.oc_count,
        "align_out_c": max(32, _align_up(desc.oc_count, 16)),
        "out_width_stride": p["out_width_stride"],
    })
    regs = make_conv2d_regs(tile_p,
        in_dma + desc.feature_off,
        wt_dma + desc.weight_off,
        out_dma + desc.output_off,
        weight_reuse=desc.weight_reuse,
        full_data_bank=desc.extra.get("full_data_bank", False))

    if desc.family_bits or desc.grain_bits is not None:
        current = _find_reg_value(regs, reg.CNA_CONV_CON2)
        grain_bits = current if desc.grain_bits is None else desc.grain_bits
        _patch_reg(regs, reg.CNA_CONV_CON2, desc.family_bits | grain_bits)
    if desc.cbuf0 is not None:
        _patch_reg(regs, reg.CNA_CBUF_CON0, desc.cbuf0)
    if desc.cbuf1 is not None:
        _patch_reg(regs, reg.CNA_CBUF_CON1, desc.cbuf1)
    if p["is_spatial"]:
        _patch_reg(regs, reg.SURFACE_ADD, (p["out_width_stride"] * 2) << 4)
    return regs

def make_rknn_full_conv2d_regs_from_desc(p, desc, in_dma, wt_dma, out_dma):
    regs = make_conv2d_regs_from_desc(p, desc, in_dma, wt_dma, out_dma)
    def reg_value(reg_addr):
        return _find_reg_value(regs, reg_addr)
    dcomp_amounts = [E(reg.CNA, reg.CNA_DCOMP_AMOUNT0 + i * 4, 0) for i in range(16)]
    compiler_data_size1 = ((min(p["in_c"], 32) - 1) << 16) | _align_up(p["in_c"], p["align_c"])
    compiler_dma_con2 = p["width_stride"] * (p["in_h"] - 4) if p["in_h"] > 4 else 0
    compiler_cvt_con0 = 0x0b if p["is_spatial"] and not p["is_depthwise"] else reg_value(reg.CNA_CVT_CON0)
    compiler_core_misc = (2 << 8) | (p["is_depthwise"] << 1)
    return [
        E(reg.CNA, reg.CNA_CBUF_CON0, reg_value(reg.CNA_CBUF_CON0)),
        E(reg.CNA, reg.CNA_DCOMP_REGNUM, 0),
        E(reg.CNA, reg.CNA_DCOMP_CTRL, 0),
        E(reg.CNA, reg.CNA_CONV_CON1, reg_value(reg.CNA_CONV_CON1)),
        E(reg.DPU, reg.S_POINTER, reg_value(reg.S_POINTER)),
        E(reg.CNA, reg.CNA_CONV_CON1, reg_value(reg.CNA_CONV_CON1)),
        E(reg.CNA, reg.CNA_CONV_CON2, reg_value(reg.CNA_CONV_CON2)),
        E(reg.CNA, reg.CNA_CONV_CON3, reg_value(reg.CNA_CONV_CON3)),
        E(reg.CNA, reg.CNA_DATA_SIZE0, reg_value(reg.CNA_DATA_SIZE0)),
        E(reg.CNA, reg.CNA_DATA_SIZE1, compiler_data_size1),
        E(reg.CNA, reg.CNA_DATA_SIZE2, reg_value(reg.CNA_DATA_SIZE2)),
        E(reg.CNA, reg.CNA_DATA_SIZE3, reg_value(reg.CNA_DATA_SIZE3)),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, reg_value(reg.CNA_WEIGHT_SIZE0)),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, reg_value(reg.CNA_WEIGHT_SIZE1)),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2, reg_value(reg.CNA_WEIGHT_SIZE2)),
        E(reg.CNA, reg.CNA_CBUF_CON0, reg_value(reg.CNA_CBUF_CON0)),
        E(reg.CNA, reg.CNA_CBUF_CON1, reg_value(reg.CNA_CBUF_CON1)),
        E(reg.CNA, reg.CNA_CVT_CON0, compiler_cvt_con0),
        E(reg.CNA, reg.CNA_CVT_CON1, reg_value(reg.CNA_CVT_CON1)),
        E(reg.CNA, reg.CNA_CVT_CON2, reg_value(reg.CNA_CVT_CON2)),
        E(reg.CNA, reg.CNA_CVT_CON3, reg_value(reg.CNA_CVT_CON3)),
        E(reg.CNA, reg.CNA_CVT_CON4, reg_value(reg.CNA_CVT_CON4)),
        E(reg.CNA, reg.CNA_FC_CON0, 0),
        E(reg.CNA, reg.CNA_FC_CON1, 0),
        E(reg.CNA, reg.CNA_PAD_CON0, 0),
        E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, reg_value(reg.CNA_FEATURE_DATA_ADDR)),
        E(reg.CNA, reg.CNA_FC_CON2, 0),
        E(reg.CNA, reg.CNA_DMA_CON0, reg_value(reg.CNA_DMA_CON0)),
        E(reg.CNA, reg.CNA_DMA_CON1, reg_value(reg.CNA_DMA_CON1)),
        E(reg.CNA, reg.CNA_DMA_CON2, compiler_dma_con2),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, reg_value(reg.CNA_FC_DATA_SIZE0)),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, reg_value(reg.CNA_FC_DATA_SIZE1)),
        E(reg.CNA, reg.CNA_DCOMP_CTRL, 0),
        E(reg.CNA, reg.CNA_DCOMP_REGNUM, 0),
        E(reg.CNA, reg.CNA_DCOMP_ADDR0, reg_value(reg.CNA_DCOMP_ADDR0)),
        *dcomp_amounts,
        E(reg.CNA, reg.CNA_CVT_CON5, reg_value(reg.CNA_CVT_CON5)),
        E(reg.CNA, reg.CNA_PAD_CON1, 0),
        E(reg.CORE, reg.CORE_MISC_CFG, compiler_core_misc),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, reg_value(reg.CORE_DATAOUT_SIZE_0)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, reg_value(reg.CORE_DATAOUT_SIZE_1)),
        E(reg.CORE, reg.CORE_CLIP_TRUNCATE, 0),
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, reg_value(reg.FEATURE_MODE_CFG)),
        E(reg.DPU, reg.DATA_FORMAT, reg_value(reg.DATA_FORMAT)),
        E(reg.DPU, 0x4014, 0),
        E(reg.DPU, reg.DST_BASE_ADDR, reg_value(reg.DST_BASE_ADDR)),
        E(reg.DPU, reg.DST_SURF_STRIDE, reg_value(reg.DST_SURF_STRIDE)),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, reg_value(reg.DATA_CUBE_WIDTH)),
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, reg_value(reg.DATA_CUBE_HEIGHT)),
        E(reg.DPU, reg.DATA_CUBE_NOTCH, reg_value(reg.DATA_CUBE_NOTCH)),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL, reg_value(reg.DATA_CUBE_CHANNEL)),
        E(reg.DPU, reg.BS_CFG, reg_value(reg.BS_CFG)),
        E(reg.DPU, 0x4044, 0),
        E(reg.DPU, 0x4048, 0),
        E(reg.DPU, 0x404c, 0),
        E(reg.DPU, reg.BS_OW_CFG, reg_value(reg.BS_OW_CFG)),
        E(reg.DPU, 0x4054, 0),
        E(reg.DPU, reg.WDMA_SIZE_0, reg_value(reg.WDMA_SIZE_0)),
        E(reg.DPU, reg.WDMA_SIZE_1, reg_value(reg.WDMA_SIZE_1)),
        E(reg.DPU, reg.BN_CFG, reg_value(reg.BN_CFG)),
        E(reg.DPU, 0x4064, 0),
        E(reg.DPU, 0x4068, 0),
        E(reg.DPU, 0x406c, 0),
        E(reg.DPU, reg.EW_CFG, reg_value(reg.EW_CFG)),
        E(reg.DPU, 0x4074, 0),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, reg_value(reg.EW_CVT_SCALE_VALUE)),
        E(reg.DPU, 0x407c, 0),
        E(reg.DPU, 0x4080, 0),
        E(reg.DPU, reg.OUT_CVT_SCALE, reg_value(reg.OUT_CVT_SCALE)),
        E(reg.DPU, 0x4088, 0),
        E(reg.DPU, 0x4090, 0),
        E(reg.DPU, 0x4094, 0),
        E(reg.DPU, 0x4098, 0),
        E(reg.DPU, 0x409c, 0),
        E(reg.DPU, 0x40a0, 0),
        E(reg.DPU, 0x40a4, 0),
        E(reg.DPU, 0x40a8, 0),
        E(reg.DPU, 0x40ac, 0),
        E(reg.DPU, reg.SURFACE_ADD, reg_value(reg.SURFACE_ADD)),
        E(reg.DPU, reg.DPU_SURFACE_ADD_EXTRA, 0),
        *(E(reg.DPU, clear_reg, 0) for clear_reg in DPU_CLEAR_REGS),
    ]

def _patch_reg(regs, target_reg, value):
    found = False
    for i, cmd in enumerate(regs):
        if (cmd & 0xFFFF) == target_reg:
            regs[i] = (cmd & 0xFFFF000000000000) | ((value & 0xFFFFFFFF) << 16) | target_reg
            found = True
    if not found:
        raise KeyError(f"register 0x{target_reg:x} not found")

def _find_reg_value(regs, target_reg):
    for cmd in regs:
        if (cmd & 0xFFFF) == target_reg:
            return (cmd >> 16) & 0xFFFFFFFF
    raise KeyError(f"register 0x{target_reg:x} not found")

def _mesa_weight_banks(weights_width, weights_height, input_channels, output_channels, depthwise):
    weight_bytes = weights_width * weights_height * input_channels * FP16_BYTES
    if not depthwise:
        weight_bytes *= output_channels
    return _ceil_div(_ceil_div(weight_bytes, CBUF_ENTRY_BYTES), CBUF_ENTRIES_PER_BANK) + 1

def _pointwise_oc_tile_c(in_c):
    max_tile = CBUF_BANK_SIZE // (max(1, in_c) * FP16_BYTES)
    return 32 if max_tile >= 32 else 16 if max_tile >= 16 else 8 if max_tile >= 8 else 4

def _pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile, stride=1):
    return _mesa_output_tile_h(out_w, out_h, in_c, oc_tile, 1, 1, stride, False)

def _mesa_output_tile_h(input_width, out_h, input_channels, output_channels, kh, kw, stride, depthwise, input_banks=None):
    if input_banks is None:
        weight_banks = _mesa_weight_banks(kw, kh, input_channels, output_channels, depthwise)
        input_banks = RK_CBUF_BANKS - weight_banks if weight_banks + 1 < RK_CBUF_BANKS else 7
    entries_per_slice = max(1, _mesa_entries_per_slice(input_width, input_channels))
    input_slices = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // entries_per_slice)
    output_rows = max(1, (input_slices - kh) // stride + 1)
    return min(out_h, output_rows)

def _depthwise_tile_h(total_channels, out_h, in_w, kh, kw, stride=1):
    tile_h = _mesa_output_tile_h(in_w, out_h, total_channels, total_channels, kh, kw, stride, True, input_banks=7)
    if total_channels > 64:
        align_c = _conv_align_c(total_channels, total_channels, total_channels)
        row_bytes = in_w * align_c * FP16_BYTES
        max_feature_rows = _feature_grains(row_bytes, out_h + kh, is_spatial=True, is_depthwise=True) + 1
        tile_h = min(tile_h, max_feature_rows)
    tile_h = max(10, tile_h) if tile_h < out_h else tile_h
    return tile_h if tile_h == out_h or tile_h % 2 == 0 else tile_h - 1

def _needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
    if groups != 1 or kh != 1 or kw != 1:
        return False
    out_h = in_h
    out_w = in_w
    oc_tile = _pointwise_oc_tile_c(in_c)
    return out_c > oc_tile or (in_c >= 16 and out_c % oc_tile != 0) or _pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile) < out_h

# RKNN split_method enum (from decomp fcn.005f38f0)
_SPLIT_NONE = 0
_SPLIT_BY_Y = 1
_SPLIT_BY_K = 2
_SPLIT_BY_YK = 3

def _compute_k_step(in_c, out_c, kh, kw, groups, p):
    is_depthwise, is_spatial = p["is_depthwise"], p["is_spatial"]
    data_in_channel_aligned = _align_up(in_c, p["align_c"])
    weight_kernel_bytes = kh * kw * data_in_channel_aligned * FP16_BYTES
    tweight = weight_kernel_bytes * (1 if is_depthwise else out_c)
    weight_banks = _ceil_div(tweight, CBUF_BANK_SIZE)

    k_step = out_c
    if is_depthwise and is_spatial:
        k_step = min(32, out_c)
    elif is_spatial and groups == 1 and not is_depthwise and weight_banks > 3:
        k_step = 32 if out_c >= 32 else out_c
    elif not is_spatial and groups == 1:
        pw_oc = _pointwise_oc_tile_c(in_c)
        if weight_banks > 3:
            k_step = max(pw_oc, 32)
        elif out_c > pw_oc:
            k_step = pw_oc
    return min(k_step, out_c)

def _compute_y_step(in_c, out_c, kh, kw, in_h, in_w, groups, stride, k_step, p):
    is_spatial, is_depthwise = p["is_spatial"], p["is_depthwise"]
    out_h = p["out_h"]

    if is_depthwise and is_spatial:
        data_in_channel_aligned = _align_up(in_c, p["align_c"])
        weight_kernel_bytes = kh * kw * data_in_channel_aligned * FP16_BYTES
        weight_banks = _ceil_div(_ceil_div(weight_kernel_bytes, CBUF_ENTRY_BYTES), CBUF_ENTRIES_PER_BANK) + 1
        input_banks = RK_CBUF_BANKS - weight_banks if weight_banks + 1 < RK_CBUF_BANKS else 7
        ae = CBUF_ENTRY_BYTES // 16
        tca = _ceil_div(in_c * FP16_BYTES, 16)
        lca = tca % ae
        ice = (tca // ae) * in_w
        fce = in_w if lca == 3 else _ceil_div(lca * in_w, ae)
        eps = max(1, ice + fce)
        s = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // eps)
        tile_h = min(out_h, max(1, (s - kh) // stride + 1))
        if in_c > 64:
            row_bytes = in_w * _conv_align_c(in_c, in_c, in_c) * FP16_BYTES
            max_rows = _feature_grains(row_bytes, out_h + kh, True, True) + 1
            tile_h = min(tile_h, max_rows)
        tile_h = max(10, tile_h) if tile_h < out_h else tile_h
        max_input_rows = min(15, out_h + kh - 1)
        tile_h = min(tile_h, max_input_rows - kh + 1)
        tile_h = tile_h if tile_h == out_h or tile_h % 2 == 0 else tile_h - 1
        return tile_h

    data_in_channel_aligned = _align_up(in_c, p["align_c"])
    y_step = out_h
    tile_wb = _ceil_div(kh * kw * data_in_channel_aligned * FP16_BYTES * (k_step if not is_depthwise else 1), CBUF_BANK_SIZE)
    remaining = max(1, RK_CBUF_BANKS - tile_wb)
    tile_in_c = k_step if is_depthwise else in_c
    tile_data_in_channel_aligned = _align_up(tile_in_c, p["align_c"])
    row_bytes = p["width_stride"] * tile_data_in_channel_aligned * FP16_BYTES
    fg = _feature_grains(row_bytes, in_h + kh, False, is_spatial, is_depthwise)
    data_banks_needed = _ceil_div(row_bytes * fg, CBUF_BANK_SIZE)
    if data_banks_needed > remaining:
        y_step = max(1, out_h * remaining // max(1, data_banks_needed))

    if not is_spatial:
        out_w = p["out_w"]
        small_channel = in_c <= 4 and not is_depthwise
        if small_channel and p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE:
            y_step = min(y_step, max(1, RK_MAX_CONV_FLAT_STRIDE // out_w))
        elif out_h > 50:
            if in_c >= 128 and out_c >= 128:
                y_step = min(y_step, 25)
            elif p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE:
                y_step = min(y_step, 32)
            else:
                y_step = min(y_step, 50)

    if not is_depthwise:
        eps = _mesa_entries_per_slice(p["width_stride"], data_in_channel_aligned)
        input_banks = RK_CBUF_BANKS - tile_wb if tile_wb + 1 < RK_CBUF_BANKS else 7
        max_input_h = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // eps)
        max_y = max(1, (max_input_h - kh) // stride + 1)
        y_step = min(y_step, max_y)

    if is_spatial and p["use_nhwc"]:
        nhwc_data_bank = RK_CBUF_BANKS - 1
        nhwc_row_bytes = p["width_stride"] * tile_data_in_channel_aligned * FP16_BYTES
        max_grains = (nhwc_data_bank * CBUF_BANK_SIZE) // nhwc_row_bytes
        max_y_for_cbuf = max(1, max_grains - 2 * kh + 1)
        y_step = min(y_step, max_y_for_cbuf)

    return y_step

def _plan_conv_tiles(in_c, out_c, kh, kw, in_h, in_w, groups, stride):
    p = _conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    out_h = p["out_h"]
    k_step = _compute_k_step(in_c, out_c, kh, kw, groups, p)
    y_step = _compute_y_step(in_c, out_c, kh, kw, in_h, in_w, groups, stride, k_step, p)

    split_method = _SPLIT_NONE
    if k_step < out_c and y_step < out_h:
        split_method = _SPLIT_BY_YK
    elif k_step < out_c:
        split_method = _SPLIT_BY_K
    elif y_step < out_h:
        split_method = _SPLIT_BY_Y

    y_boundary = [0]
    while y_boundary[-1] < out_h:
        y_boundary.append(int(min(y_boundary[-1] + y_step, out_h)))
    k_boundary = [0]
    while k_boundary[-1] < out_c:
        k_boundary.append(int(min(k_boundary[-1] + k_step, out_c)))

    tiles = []
    for yi in range(len(y_boundary) - 1):
        ys = y_boundary[yi]
        y_span = y_boundary[yi + 1] - ys
        for ki in range(len(k_boundary) - 1):
            ks = k_boundary[ki]
            k_span = k_boundary[ki + 1] - ks
            tiles.append({"y_start": ys, "y_step": y_span, "k_start": ks, "k_step": k_span})
    return p, split_method, tiles

def plan_conv_tiles(in_c, out_c, kh, kw, in_h, in_w, groups, stride):
    p, split_method, tiles = _plan_conv_tiles(in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    y_boundary = sorted({0, *[t["y_start"] for t in tiles], *[t["y_start"] + t["y_step"] for t in tiles]})
    k_boundary = sorted({0, *[t["k_start"] for t in tiles], *[t["k_start"] + t["k_step"] for t in tiles]})
    return p, split_method, y_boundary, k_boundary, "", None, None

def execute_conv_descriptors(descs):
    for d in descs:
        _execute_conv_desc(d)

def _load_desc_buffers(d):
    if d.get("buffer_scope") == BUFFER_SCOPE_FULL:
        _load_full_input(d["packed_input"])
        _load_full_weight(d["packed_weight"])
    else:
        _load_tile_input(d["packed_input"])
        _load_tile_weight(d["packed_weight"])

def _execute_conv_desc(d):
    _load_desc_buffers(d)
    if d.get("clear_output", True):
        _clear_output()
    if d.get("task_policy") == TASK_POLICY_RAW_SPANS:
        _submit_raw_task_spans(d["regs"])
    else:
        _submit_task_regs(d["regs"])
    _unpack_tile_result(d)

def _store_tile_output(u, out):
    real_channels = u["oc_end"] - u["oc_start"]
    u["result"][u["n"],
                u["oc_start"]:u["oc_end"],
                u["y_start"]:u["y_start"] + u["out_h"]] = out[:real_channels, :u["out_h"]]

def _read_nc1hwc2_output(count, channels, out_h, out_w, out_width_stride, offset=0):
    return _unpack_nc1hwc2_output(_read_tile_output(count, offset),
                                  channels, out_h, out_w, UNPACK_C2, out_width_stride)

def _read_flat_1x1_output(count, hw_channels, out_h, out_w, out_width_stride, offset=0):
    return _unpack_flat_1x1_output(_read_tile_output(count, offset),
                                   hw_channels, out_h, out_w, out_width_stride, UNPACK_C2)

def _read_unpack_output(u):
    if u["kind"] == UNPACK_KIND_NC1HWC2:
        return _read_nc1hwc2_output(u["count"], u["channels"], u["hw_h"], u["out_w"],
                                    u["stride"], u.get("offset", 0))
    if u["kind"] == UNPACK_KIND_FLAT_1X1:
        return _read_flat_1x1_output(u["count"], u["hw_channels"], u["hw_h"], u["out_w"],
                                     u["stride"], u.get("offset", 0))
    if u["kind"] == UNPACK_KIND_DIRECT_SPATIAL:
        return _read_nc1hwc2_output(u["count"], u["channels"], u["out_h"], u["out_w"], u["stride"])
    raise ValueError(f"unknown unpack kind: {u['kind']}")

def _unpack_tile_result(d):
    u = d["unpack"]
    out = _read_unpack_output(u)
    if u["kind"] == UNPACK_KIND_DIRECT_SPATIAL:
        u["result"][u["n"]] = out[:u["channels"]]
    else:
        _store_tile_output(u, out)

def make_tile_exec_desc(family, packed_input, packed_weight, regs, unpack, meta=None,
                        buffer_scope=BUFFER_SCOPE_TILE, clear_output=True,
                        task_policy=TASK_POLICY_REG_LISTS):
    return {"family": family,
            "packed_input": packed_input,
            "packed_weight": packed_weight,
            "regs": regs,
            "unpack": unpack,
            "meta": meta or {},
            "buffer_scope": buffer_scope,
            "clear_output": clear_output,
            "task_policy": task_policy}

def make_tile_unpack(kind, result, n, oc_start, oc_end, y_start, out_h, hw_h,
                     out_w, channels, out_width_stride, count, hw_channels=None, offset=0):
    unpack = {"kind": kind, "result": result, "n": n,
              "oc_start": oc_start, "oc_end": oc_end,
              "y_start": y_start, "out_h": out_h, "hw_h": hw_h,
              "out_w": out_w, "channels": channels,
              "stride": out_width_stride, "count": count,
              "offset": offset}
    if hw_channels is not None:
        unpack["hw_channels"] = hw_channels
    return unpack

def make_direct_spatial_unpack(result, n, p):
    return {"kind": UNPACK_KIND_DIRECT_SPATIAL,
            "result": result,
            "n": n,
            "count": conv_output_count(p),
            "channels": p["out_c"],
            "out_h": p["out_h"],
            "out_w": p["out_w"],
            "stride": p["out_width_stride"]}

def make_nc1hwc2_tile_desc(family, result, n, packed_input, packed_weight, regs, oc_start, oc_end,
                           y_start, out_h, hw_h, out_w, channels, out_width_stride, count,
                           offset=0, meta=None):
    return make_tile_exec_desc(family, packed_input, packed_weight, regs,
                               make_tile_unpack(UNPACK_KIND_NC1HWC2, result, n, oc_start, oc_end,
                                                y_start, out_h, hw_h, out_w, channels,
                                                out_width_stride, count, offset=offset),
                               meta=meta)

def make_flat_1x1_tile_desc(family, result, n, packed_input, packed_weight, regs, oc_start, oc_end,
                            y_start, out_h, hw_h, out_w, channels, hw_channels,
                            out_width_stride, count, meta=None):
    return make_tile_exec_desc(family, packed_input, packed_weight, regs,
                               make_tile_unpack(UNPACK_KIND_FLAT_1X1, result, n, oc_start, oc_end,
                                                y_start, out_h, hw_h, out_w, channels,
                                                out_width_stride, count, hw_channels=hw_channels),
                               meta=meta)

def conv_output_count(p):
    return _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2

def conv_output_bytes(p):
    return conv_output_count(p) * FP16_BYTES

def materialize_conv_tile(p, input_tile, weight_tile, out_c, in_c, kh, kw, groups,
                          out_dma=None, full_data_bank=False):
    out_dma = output_mem_create.dma_addr if out_dma is None else out_dma
    packed_input = _pack_cdma_dc_feature_input_fp16(input_tile, p)
    packed_weight = pack_weights(weight_tile, out_c, in_c, kh, kw, p["align_c"], groups)
    regs = [make_conv2d_regs(p, input_mem_create.dma_addr, weight_mem_create.dma_addr,
                             out_dma, full_data_bank=full_data_bank)]
    return packed_input, packed_weight, regs, conv_output_count(p)

def tile_meta(y_start, out_h, oc_start, oc_end, tile_p, reason, **extra):
    meta = {"y": (y_start, y_start + out_h), "oc": (oc_start, oc_end),
            "tile_p": tile_p, "reason": reason}
    meta.update(extra)
    return meta

def make_record_tile_desc(record, result, n, packed_input, packed_weight, regs, tp, out_count,
                          hw_h=None, channels=None, offset=None):
    oc_end = record.k_start + record.k_count
    hw_h = record.y_count if hw_h is None else hw_h
    channels = record.k_count if channels is None else channels
    offset = record.out_offset if offset is None else offset
    meta = tile_meta(record.y_start, record.y_count, record.k_start, oc_end, tp,
                     record.reason, tile_record=record)
    if record.unpack_kind == UNPACK_KIND_NC1HWC2:
        return make_nc1hwc2_tile_desc(record.family, result, n, packed_input, packed_weight, regs,
                                      record.k_start, oc_end, record.y_start, record.y_count,
                                      hw_h, record.output_w, channels, tp["out_width_stride"],
                                      out_count, offset=offset, meta=meta)
    if record.unpack_kind == UNPACK_KIND_FLAT_1X1:
        return make_flat_1x1_tile_desc(record.family, result, n, packed_input, packed_weight, regs,
                                       record.k_start, oc_end, record.y_start, record.y_count,
                                       hw_h, record.output_w, channels, record.hw_oc,
                                       tp["out_width_stride"], out_count, meta=meta)
    raise ValueError(f"unknown record unpack kind: {record.unpack_kind}")

def _rknn_family_bits(family):
    return {
        "setup": 0x00000000,
        "y_mid": 0x10000000,
        "y_tile": 0x20000000,
        "k_half": 0x40000000,
        "k_tile": 0x50000000,
    }[family]

def _rknn_spatial_y_windows(in_h):
    return {
        7: [(0, 7, 5)],
        14: [(0, 14, 12)],
        16: [(0, 16, 14)],
        20: [(0, 20, 18)],
        24: [(0, 24, 22)],
        28: [(0, 28, 26)],
        32: [(0, 28, 26), (26, 6, 4)],
        36: [(0, 25, 23), (23, 13, 11)],
        40: [(0, 23, 21), (21, 19, 17)],
    }.get(in_h)

def _rknn_spatial_full_y_tile_windows(in_h, out_h):
    if in_h == 20 and out_h == 18:
        return [(0, 8, 6), (6, 8, 6), (12, 8, 6)]
    return None

def _rknn_spatial_pc_core(family):
    if family == "setup":
        return 0
    if family == "k_half":
        return 1
    if family == "k_tile":
        return 2
    return 0

def _make_observed_spatial_tile_desc(family, input_y, input_h, output_h, output_w,
                                     oc_start, oc_count, feature_off, weight_off, output_off,
                                     grain_bits=None, cbuf0=None, cbuf1=None):
    return TileDesc(
        family=family,
        family_bits=_rknn_family_bits(family),
        grain_bits=grain_bits,
        input_y=input_y,
        input_h=input_h,
        output_y=input_y,
        output_h=output_h,
        output_w=output_w,
        oc_start=oc_start,
        oc_count=oc_count,
        feature_off=feature_off,
        weight_off=weight_off,
        output_off=output_off,
        cbuf0=cbuf0,
        cbuf1=cbuf1,
        weight_reuse=family != "setup",
        data_reuse=False,
        pc_core=0,
    )

def _observed_spatial_desc_offsets(p, in_c, kh, kw, in_w):
    return {
        "weight_per_oc": kh * kw * in_c * FP16_BYTES,
        "output_per_oc": p["out_width_stride"] * FP16_BYTES,
        "feature_row": in_w * UNPACK_C2 * FP16_BYTES,
        "output_row": p["out_w"] * UNPACK_C2 * FP16_BYTES,
    }

def _append_observed_spatial_desc(descs, p, offsets, family, input_y, input_h, output_h,
                                  oc_start, oc_count, grain_bits=None, cbuf0=None):
    descs.append(_make_observed_spatial_tile_desc(
        family, input_y, input_h, output_h, p["out_w"],
        oc_start, oc_count,
        input_y * offsets["feature_row"],
        oc_start * offsets["weight_per_oc"],
        oc_start * offsets["output_per_oc"] + input_y * offsets["output_row"],
        grain_bits=grain_bits,
        cbuf0=cbuf0,
    ))

def _append_outc64_h20_setup(descs, p, offsets, in_h, out_c):
    _append_observed_spatial_desc(descs, p, offsets, "setup", 0, in_h, p["out_h"], 0, out_c)

def _append_outc64_h20_k_half(descs, p, offsets, in_h, out_c):
    half = _align_up(out_c // 2, 16)
    for oc_start, oc_count in ((0, half), (half, out_c - half)):
        _append_observed_spatial_desc(descs, p, offsets, "k_half", 0, in_h, p["out_h"], oc_start, oc_count)

def _append_outc64_h20_y_mid(descs, p, offsets, out_c):
    for input_y in (0, 9):
        _append_observed_spatial_desc(descs, p, offsets, "y_mid", input_y, 11, 9, 0, out_c)

def _append_outc64_h20_y_tile(descs, p, offsets, in_h, out_c):
    for input_y, input_rows, output_rows in _rknn_spatial_full_y_tile_windows(in_h, p["out_h"]):
        _append_observed_spatial_desc(descs, p, offsets, "y_tile", input_y, input_rows, output_rows, 0, out_c)

def _append_outc64_h20_k_tile(descs, p, offsets, in_h, k_counts):
    oc_start = 0
    for oc_count in k_counts:
        _append_observed_spatial_desc(descs, p, offsets, "k_tile", 0, in_h, p["out_h"], oc_start, oc_count)
        oc_start += oc_count

def _plan_outc64_h20_descs(p, in_c, out_c, kh, kw, in_h, in_w):
    mode_k_counts = DIRECT_SPATIAL_OUTC64_H20_POST.get(out_c)
    if not (in_c == 64 and in_h == 20 and mode_k_counts is not None):
        return []
    mode, k_counts = mode_k_counts
    offsets = _observed_spatial_desc_offsets(p, in_c, kh, kw, in_w)
    descs = []
    _append_outc64_h20_setup(descs, p, offsets, in_h, out_c)
    if mode.startswith("k_half"):
        _append_outc64_h20_k_half(descs, p, offsets, in_h, out_c)
    elif mode.startswith("y_mid"):
        _append_outc64_h20_y_mid(descs, p, offsets, out_c)
    if mode.endswith("y_tile"):
        _append_outc64_h20_y_tile(descs, p, offsets, in_h, out_c)
    elif mode.endswith("k_tile"):
        _append_outc64_h20_k_tile(descs, p, offsets, in_h, k_counts)
    return descs

def _plan_chan320_h20_descs(p, in_c, out_c, kh, kw, in_h, in_w):
    if not (out_c == 320 and in_h == 20 and in_c in DIRECT_SPATIAL_CHAN320_H20_IN_CHANNELS):
        return []
    offsets = _observed_spatial_desc_offsets(p, in_c, kh, kw, in_w)
    descs = []
    _append_outc64_h20_setup(descs, p, offsets, in_h, out_c)
    _append_outc64_h20_k_half(descs, p, offsets, in_h, out_c)
    if in_c in {32, 40}:
        _append_outc64_h20_y_tile(descs, p, offsets, in_h, out_c)
    else:
        _append_outc64_h20_k_tile(descs, p, offsets, in_h, (112, 112, 96))
    return descs

def _plan_mix72_h20_descs(p, in_c, out_c, kh, kw, in_h, in_w):
    if not (in_c == 72 and out_c == 288 and in_h == 20):
        return []
    offsets = _observed_spatial_desc_offsets(p, in_c, kh, kw, in_w)
    descs = []
    _append_outc64_h20_setup(descs, p, offsets, in_h, out_c)
    _append_outc64_h20_k_half(descs, p, offsets, in_h, out_c)
    _append_outc64_h20_k_tile(descs, p, offsets, in_h, DIRECT_SPATIAL_MIX72_H20_K_TILE_COUNTS)
    return descs

def _plan_cmp32_h14_descs(p, in_c, out_c, kh, kw, in_h, in_w):
    if not (in_c == 32 and out_c == 128 and in_h == 14):
        return []
    offsets = _observed_spatial_desc_offsets(p, in_c, kh, kw, in_w)
    descs = []
    _append_observed_spatial_desc(descs, p, offsets, "setup", 0, in_h, p["out_h"], 0, out_c,
                                  grain_bits=0x110)
    _append_outc64_h20_k_half(descs, p, offsets, in_h, out_c)
    for desc in descs[1:]:
        desc.grain_bits = 0x110
    for input_y, input_h, output_h in DIRECT_SPATIAL_CMP32_H14_Y_WINDOWS:
        _append_observed_spatial_desc(descs, p, offsets, "y_tile", input_y, input_h, output_h, 0, out_c,
                                      grain_bits=0x90)
    return descs

def _append_observed_pointwise_y_list(descs, p, offsets, family, k_windows, y_windows):
    for oc_start, oc_count in k_windows:
        for input_y, output_rows in y_windows:
            _append_observed_spatial_desc(descs, p, offsets, family, input_y,
                                          output_rows, output_rows, oc_start, oc_count)

def _plan_pointwise_simple_descs(p, in_c, out_c, kh, kw, in_h, in_w):
    if not (
        kh == 1 and kw == 1 and in_h == in_w and in_h in DIRECT_SPATIAL_POINTWISE_SIMPLE_Y_WINDOWS
        and ((in_c, out_c) in {(40, 320), (64, 128)} or
             (in_c, out_c, in_h) in {(528, 32, 14), (256, 512, 14)})
    ):
        return []
    offsets = _observed_spatial_desc_offsets(p, in_c, kh, kw, in_w)
    descs = []
    _append_outc64_h20_setup(descs, p, offsets, in_h, out_c)
    _append_outc64_h20_k_half(descs, p, offsets, in_h, out_c)
    for input_y, output_rows in DIRECT_SPATIAL_POINTWISE_SIMPLE_Y_WINDOWS[in_h]:
        _append_observed_spatial_desc(descs, p, offsets, "y_tile", input_y,
                                      output_rows, output_rows, 0, out_c)
    return descs

def _plan_pointwise_crossed_descs(p, in_c, out_c, kh, kw, in_h, in_w):
    cfg = DIRECT_SPATIAL_POINTWISE_CROSSED.get((in_c, in_h, out_c))
    if not (kh == 1 and kw == 1 and in_h == in_w and cfg is not None):
        return []
    offsets = _observed_spatial_desc_offsets(p, in_c, kh, kw, in_w)
    half = _align_up(out_c // 2, 16)
    descs = []
    _append_observed_pointwise_y_list(descs, p, offsets, "setup", ((0, out_c),), cfg["setup"])
    _append_observed_pointwise_y_list(descs, p, offsets, "k_half",
                                      ((0, half), (half, out_c - half)), cfg["k_half"])
    _append_observed_pointwise_y_list(descs, p, offsets, "y_tile", ((0, out_c),), cfg["y_tile"])
    return descs

def plan_observed_spatial_tile_descs(p, in_c, out_c, kh, kw, in_h, in_w, groups, stride):
    if not (groups == 1 and stride == 1 and in_h == in_w):
        return []
    pointwise_simple_descs = _plan_pointwise_simple_descs(p, in_c, out_c, kh, kw, in_h, in_w)
    if pointwise_simple_descs:
        return pointwise_simple_descs
    pointwise_crossed_descs = _plan_pointwise_crossed_descs(p, in_c, out_c, kh, kw, in_h, in_w)
    if pointwise_crossed_descs:
        return pointwise_crossed_descs
    if not (kh == 3 and kw == 3):
        return []
    outc64_h20_descs = _plan_outc64_h20_descs(p, in_c, out_c, kh, kw, in_h, in_w)
    if outc64_h20_descs:
        return outc64_h20_descs
    chan320_h20_descs = _plan_chan320_h20_descs(p, in_c, out_c, kh, kw, in_h, in_w)
    if chan320_h20_descs:
        return chan320_h20_descs
    mix72_h20_descs = _plan_mix72_h20_descs(p, in_c, out_c, kh, kw, in_h, in_w)
    if mix72_h20_descs:
        return mix72_h20_descs
    cmp32_h14_descs = _plan_cmp32_h14_descs(p, in_c, out_c, kh, kw, in_h, in_w)
    if cmp32_h14_descs:
        return cmp32_h14_descs

    if not (in_c == 160 and out_c == 320):
        return []
    y_windows = _rknn_spatial_y_windows(in_h)
    if y_windows is None:
        return []
    offsets = _observed_spatial_desc_offsets(p, in_c, kh, kw, in_w)

    observed_grain = {7: 0xa0, 14: 0xf0}.get(in_h, 0xc0 if in_h >= 28 else 0xf0)
    observed_cbuf0 = 0x39 if in_h >= 32 else {7: 0xb1, 16: 0x93, 20: 0x84, 24: 0x66, 28: 0x48}.get(in_h, None)
    families = [
        ("setup", [(0, out_c)]),
        ("k_half", [(0, 160), (160, 160)]),
        ("k_tile", [(0, 112), (112, 112), (224, 96)]),
    ]
    descs = []
    for family, k_windows in families:
        for oc_start, oc_count in k_windows:
            for input_y, input_rows, output_rows in y_windows:
                descs.append(_make_observed_spatial_tile_desc(
                    family, input_y, input_rows, output_rows, p["out_w"],
                    oc_start, oc_count,
                    input_y * offsets["feature_row"],
                    oc_start * offsets["weight_per_oc"],
                    oc_start * offsets["output_per_oc"] + input_y * offsets["output_row"],
                    grain_bits=observed_grain,
                    cbuf0=observed_cbuf0,
                ).update(pc_core=_rknn_spatial_pc_core(family)))
    return descs

def attach_direct_spatial_regs(desc, p, idx):
    desc.extra = {"full_data_bank": True}
    in_dma = input_mem_create.dma_addr
    weight_dma = weight_mem_create.dma_addr
    output_dma = output_mem_create.dma_addr
    regs = RegList(make_conv2d_regs_from_desc(p, desc, in_dma, weight_dma, output_dma))
    regs.extend(E(reg.DPU, clear_reg, 0) for clear_reg in DPU_CLEAR_REGS)
    regs.pc_core = desc.pc_core
    full_regs = RegList(make_rknn_full_conv2d_regs_from_desc(p, desc, in_dma, weight_dma, output_dma))
    full_regs.pc_core = desc.pc_core
    desc.extra = {
        "full_data_bank": True,
        "regs": [regs],
        "full_regs": full_regs,
        "index": idx,
    }

def build_direct_spatial_descs(n, input_nchw, weight_nchw, p, in_c, out_c, kh, kw, in_h, in_w, groups, stride,
                               descs=None):
    if descs is None:
        descs = plan_observed_spatial_tile_descs(p, in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    if not descs:
        return [], None, None
    packed_input = _pack_cdma_dc_feature_input_fp16(input_nchw[n], p)
    packed_weight = pack_weights(weight_nchw.reshape(out_c, in_c, kh, kw),
                                 out_c, in_c, kh, kw, p["align_c"], groups)
    for idx, desc in enumerate(descs):
        attach_direct_spatial_regs(desc, p, idx)
    return descs, packed_input, packed_weight

def _submit_task_regs(task_regs):
    assert len(task_regs) <= tasks_mem_create.size // ctypes.sizeof(struct_rknpu_task), "task buffer too small"
    write_regs_to_npu_task(task_regs)
    if npu_submit(task_count=len(task_regs)) < 0:
        raise RuntimeError("npu_submit failed")

def _submit_raw_task_spans(task_spans):
    assert len(task_spans) <= tasks_mem_create.size // ctypes.sizeof(struct_rknpu_task), "task buffer too small"
    regcmd_qwords = regcmd_mem_create.size // ctypes.sizeof(ctypes.c_uint64)
    for start, amount in task_spans:
        assert start >= 0 and amount > 0 and start + amount <= regcmd_qwords, "raw task span outside regcmd buffer"
    write_raw_task_spans(task_spans)
    if npu_submit(task_count=len(task_spans)) < 0:
        raise RuntimeError("npu_submit failed")

def _bo_addr(bo_map):
    return ctypes.addressof(ctypes.c_char.from_buffer(bo_map))

def _clear_bo(bo_map, bo_create):
    ctypes.memset(_bo_addr(bo_map), 0, bo_create.size)

def _copy_packed_to_bo(packed, bo_map, bo_create, clear=False):
    packed_u16 = np.ascontiguousarray(packed.view(np.uint16))
    assert packed_u16.nbytes <= bo_create.size
    if clear:
        _clear_bo(bo_map, bo_create)
    ctypes.memmove(_bo_addr(bo_map), packed_u16.ctypes.data, packed_u16.nbytes)

def _load_tile_input(packed_input):
    _copy_packed_to_bo(packed_input, input_map, input_mem_create)

def _load_tile_weight(packed_weight):
    _clear_bo(weight_map, weight_mem_create)
    if packed_weight is not None:
        _copy_packed_to_bo(packed_weight, weight_map, weight_mem_create)

def _clear_output():
    _clear_bo(output_map, output_mem_create)

def _read_tile_output(count, offset=0):
    return np.frombuffer(output_map, dtype=np.uint16, count=count, offset=offset).copy().view(np.float16)

def _load_full_input(packed_input):
    _copy_packed_to_bo(packed_input, input_map, input_mem_create, clear=True)

def _load_full_weight(packed_weight):
    _copy_packed_to_bo(packed_weight, weight_map, weight_mem_create, clear=True)

def direct_spatial_default_supported(descs):
    return descs is not None and direct_spatial_pc_root6_supported(descs)

def direct_spatial_diagnostic_policy_requested():
    return os.environ.get(DIRECT_SPATIAL_TASKS_ENV, "") in {
        TASK_POLICY_REG_LISTS,
        DIRECT_SPATIAL_POLICY_PC_ROOT6,
        DIRECT_SPATIAL_POLICY_SINGLE_STREAM,
        DIRECT_SPATIAL_POLICY_SPARSE_SINGLE,
        DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS,
    }

def direct_spatial_task_policy_for_descs(descs):
    value = os.environ.get(DIRECT_SPATIAL_TASKS_ENV, "")
    if value == "":
        if direct_spatial_default_supported(descs):
            return DIRECT_SPATIAL_POLICY_SINGLE_STREAM
        return TASK_POLICY_REG_LISTS
    if value == TASK_POLICY_REG_LISTS:
        return TASK_POLICY_REG_LISTS
    if value == DIRECT_SPATIAL_POLICY_PC_ROOT6:
        return DIRECT_SPATIAL_POLICY_PC_ROOT6
    if value == DIRECT_SPATIAL_POLICY_SINGLE_STREAM:
        return DIRECT_SPATIAL_POLICY_SINGLE_STREAM
    if value == DIRECT_SPATIAL_POLICY_SPARSE_SINGLE:
        return DIRECT_SPATIAL_POLICY_SPARSE_SINGLE
    if value == DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS:
        return DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS
    raise ValueError(f"unknown {DIRECT_SPATIAL_TASKS_ENV} policy: {value}")

def direct_spatial_plan_allowed(descs):
    return direct_spatial_diagnostic_policy_requested()

def direct_spatial_exec_task_policy(policy):
    if policy in {DIRECT_SPATIAL_POLICY_PC_ROOT6, DIRECT_SPATIAL_POLICY_SINGLE_STREAM,
                  DIRECT_SPATIAL_POLICY_SPARSE_SINGLE,
                  DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS}:
        return TASK_POLICY_RAW_SPANS
    return TASK_POLICY_REG_LISTS

def make_direct_spatial_tile_desc(result, n, packed_input, packed_weight, descs, p):
    policy = direct_spatial_task_policy_for_descs(descs)
    regs = direct_spatial_task_regs(descs, policy)
    return make_tile_exec_desc(
        FAMILY_DIRECT_SPATIAL,
        packed_input,
        packed_weight,
        regs,
        make_direct_spatial_unpack(result, n, p),
        meta={"direct_spatial_schedule": direct_spatial_desc_schedule(descs),
              "task_policy": policy},
        buffer_scope=BUFFER_SCOPE_FULL,
        task_policy=direct_spatial_exec_task_policy(policy),
    )

def direct_spatial_task_regs(descs, policy):
    if policy == DIRECT_SPATIAL_POLICY_PC_ROOT6:
        return direct_spatial_pc_root6_task_spans(descs)
    if policy == DIRECT_SPATIAL_POLICY_SINGLE_STREAM:
        return direct_spatial_single_stream_task_span(descs)
    if policy == DIRECT_SPATIAL_POLICY_SPARSE_SINGLE:
        return direct_spatial_sparse_single_task_span(descs)
    if policy == DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS:
        return direct_spatial_rocket_record_amounts_task_span(descs)
    return [desc.extra["regs"][0] for desc in descs]

def direct_spatial_desc_schedule(descs):
    return tuple(
        (desc.family, desc.input_y, desc.input_h, desc.output_h, desc.oc_start, desc.oc_count)
        for desc in descs
    )

def direct_spatial_pc_root6_supported(descs):
    return direct_spatial_desc_schedule(descs) == DIRECT_SPATIAL_PC_ROOT6_H40_SCHEDULE

def validate_direct_spatial_pc_root6_descs(descs):
    if not direct_spatial_pc_root6_supported(descs):
        raise ValueError("pc_root6 direct-spatial policy only supports the observed 160x40 -> 320 k3x3 descriptor schedule")

def _direct_spatial_pc_tail(base_qword, amount, pc_core):
    return [
        E(reg.PC_REG, reg.PC_BASE_ADDRESS,
          (regcmd_mem_create.dma_addr + base_qword * ctypes.sizeof(ctypes.c_uint64)) & 0xFFFFFFF0),
        E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, (pc_core << 16) | amount),
    ]

def direct_spatial_separator_regs(ppu_src_dma=0, ppu_dst_dma=0):
    return [
        E(reg.PPU, reg.PPU_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.PPU_RDMA, reg.PPU_RDMA_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.PPU, reg.PPU_DATA_CUBE_IN_WIDTH, 0),
        E(reg.PPU, reg.PPU_DATA_CUBE_IN_HEIGHT, 0),
        E(reg.PPU, reg.PPU_DATA_CUBE_IN_CHANNEL, 31),
        E(reg.PPU, reg.PPU_DATA_CUBE_OUT_WIDTH, 0),
        E(reg.PPU, reg.PPU_DATA_CUBE_OUT_HEIGHT, 0),
        E(reg.PPU, reg.PPU_DATA_CUBE_OUT_CHANNEL, 31),
        E(reg.PPU, reg.PPU_OPERATION_MODE_CFG, (1 << 4) | 1),
        E(reg.PPU, reg.PPU_POOLING_KERNEL_CFG, 0),
        E(reg.PPU, reg.PPU_RECIP_KERNEL_WIDTH, 0),
        E(reg.PPU, reg.PPU_RECIP_KERNEL_HEIGHT, 0),
        E(reg.PPU, reg.PPU_POOLING_PADDING_CFG, 0),
        E(reg.PPU, reg.PPU_PADDING_VALUE_1_CFG, 0),
        E(reg.PPU, reg.PPU_PADDING_VALUE_2_CFG, 0),
        E(reg.PPU, reg.PPU_DST_BASE_ADDR, ppu_dst_dma),
        E(reg.PPU, reg.PPU_DST_SURF_STRIDE, 1 << 4),
        E(reg.PPU, reg.PPU_DATA_FORMAT, 1 << 4),
        E(reg.PPU, reg.PPU_MISC_CTRL, 3),
        E(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_WIDTH, 0),
        E(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_HEIGHT, 0),
        E(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_CHANNEL, 31),
        E(reg.PPU_RDMA, reg.PPU_RDMA_SRC_BASE_ADDR, ppu_src_dma),
        E(reg.PPU_RDMA, reg.PPU_RDMA_SRC_LINE_STRIDE, 1 << 4),
        E(reg.PPU_RDMA, reg.PPU_RDMA_SRC_SURF_STRIDE, 1 << 4),
        E(reg.PPU_RDMA, reg.PPU_RDMA_DATA_FORMAT, 1),
    ]

def direct_spatial_pc_root6_record_starts():
    starts = []
    cursor = 0
    for amount in DIRECT_SPATIAL_PC_ROOT6_RECORD_AMOUNTS:
        starts.append(cursor)
        cursor += _align_up(amount + PC_CHAIN_TAIL_QWORDS, 16)
    return starts

def direct_spatial_pc_root6_backing_qwords():
    starts = direct_spatial_pc_root6_record_starts()
    return starts[-1] + DIRECT_SPATIAL_PC_ROOT6_RECORD_AMOUNTS[-1]

def direct_spatial_pc_root6_tail(desc_idx, desc, record_starts):
    record_idx = DIRECT_SPATIAL_PC_ROOT6_LINK_RECORDS[desc_idx]
    if record_idx is None:
        return []
    amount = DIRECT_SPATIAL_PC_ROOT6_RECORD_AMOUNTS[record_idx]
    pc_amount = _ceil_div(amount, 2) + 1
    return _direct_spatial_pc_tail(record_starts[record_idx], pc_amount, desc.pc_core)

def direct_spatial_pc_root6_boundaries(record_starts, stream_len):
    boundaries = [record_starts[idx] for idx in DIRECT_SPATIAL_PC_ROOT6_ROOT_RECORDS]
    boundaries.append(stream_len)
    return boundaries

def direct_spatial_pc_root6_expected_spans(stream_len=DIRECT_SPATIAL_PC_ROOT6_STREAM_QWORDS):
    starts = direct_spatial_pc_root6_record_starts()
    boundaries = direct_spatial_pc_root6_boundaries(starts, stream_len)
    return [(start, end - start) for start, end in zip(boundaries, boundaries[1:])]

def direct_spatial_compiler_stitched_stream(descs):
    bodies = [desc.extra["full_regs"] for desc in descs]
    record_starts = direct_spatial_pc_root6_record_starts()
    stream = []
    for desc_idx, body in enumerate(bodies):
        tail = direct_spatial_pc_root6_tail(desc_idx, descs[desc_idx], record_starts)
        if desc_idx == 0:
            stream.extend(body)
            stream.extend(tail)
        elif desc_idx == 1:
            stream.append(E(reg.PC, reg.OPERATION_ENABLE, 0x0d))
            stream.extend(body)
        elif desc_idx == 2:
            stream.append(E(reg.PC, reg.OPERATION_ENABLE, 0x0d))
            stream.extend(body[4:])
            stream.extend(tail)
        else:
            stream.extend(body[4:])
            stream.extend(tail)
    assert len(stream) == DIRECT_SPATIAL_PC_ROOT6_STREAM_QWORDS, "unexpected direct-spatial compiler stream length"
    return stream

def direct_spatial_sparse_backing_stream(descs):
    validate_direct_spatial_pc_root6_descs(descs)
    record_starts = direct_spatial_pc_root6_record_starts()
    stream = [0] * direct_spatial_pc_root6_backing_qwords()
    desc_idx = 0

    for record_idx, amount in enumerate(DIRECT_SPATIAL_PC_ROOT6_RECORD_AMOUNTS):
        start = record_starts[record_idx]
        if amount == 26:
            record = direct_spatial_separator_regs()
        else:
            body = descs[desc_idx].extra["full_regs"]
            tail = direct_spatial_pc_root6_tail(desc_idx, descs[desc_idx], record_starts)
            if desc_idx == 0:
                record = list(body) + tail
            elif desc_idx == 1:
                record = [E(reg.PC, reg.OPERATION_ENABLE, 0x0d)] + list(body)
            elif desc_idx == 2:
                record = [E(reg.PC, reg.OPERATION_ENABLE, 0x0d)] + list(body[4:]) + tail
            else:
                record = list(body[4:]) + tail
            desc_idx += 1
        assert len(record) >= amount, "direct-spatial sparse record shorter than task amount"
        assert start + len(record) <= len(stream), "direct-spatial sparse record outside backing stream"
        stream[start:start + len(record)] = record

    assert desc_idx == len(descs), "direct-spatial sparse stream did not consume all descriptors"
    return stream

def _rocket_kernel_pc_amount_for_task_count(regcmd_count):
    return (regcmd_count + 1) // 2 - 1

def _direct_spatial_pc_base_for_record_start(record_start):
    return (regcmd_mem_create.dma_addr + record_start * ctypes.sizeof(ctypes.c_uint64)) & 0xFFFFFFF0

def direct_spatial_rocket_record_amount_patches(stream):
    record_starts = direct_spatial_pc_root6_record_starts()
    start_by_pc_base = {
        _direct_spatial_pc_base_for_record_start(start): idx
        for idx, start in enumerate(record_starts)
    }
    patches = []
    for idx, qword in enumerate(stream):
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) != (reg.PC_REG, reg.PC_BASE_ADDRESS):
            continue
        if idx + 1 >= len(stream):
            raise ValueError(f"PC_BASE at sparse[{idx}] has no following PC_REGISTER_AMOUNTS")
        record_idx = start_by_pc_base.get(value)
        if record_idx is None:
            raise ValueError(f"PC_BASE at sparse[{idx}] targets non-record address 0x{value:x}")
        amount_qword = stream[idx + 1]
        amount_target = (amount_qword >> 48) & 0xffff
        amount_reg = amount_qword & 0xffff
        if (amount_target, amount_reg) != (reg.PC_REG, reg.PC_REGISTER_AMOUNTS):
            raise ValueError(f"sparse[{idx + 1}] is not PC_REGISTER_AMOUNTS")
        amount = _rocket_kernel_pc_amount_for_task_count(DIRECT_SPATIAL_PC_ROOT6_RECORD_AMOUNTS[record_idx])
        patches.append((idx + 1, amount))
    return patches

def direct_spatial_rocket_record_amounts_stream(descs):
    stream = direct_spatial_sparse_backing_stream(descs)
    for amount_qword, amount in direct_spatial_rocket_record_amount_patches(stream):
        qword = stream[amount_qword]
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        stream[amount_qword] = E(target, reg_addr, (value & 0xffff0000) | amount)
    return stream

def direct_spatial_pc_root6_task_spans(descs):
    stream, spans = direct_spatial_pc_root6_stream_and_spans(descs)
    for idx, qword in enumerate(stream):
        npu_regcmd[idx] = qword
    return spans

def direct_spatial_single_stream_task_span(descs):
    validate_direct_spatial_pc_root6_descs(descs)
    stream = direct_spatial_compiler_stitched_stream(descs)
    for idx, qword in enumerate(stream):
        npu_regcmd[idx] = qword
    return [(0, len(stream))]

def direct_spatial_sparse_single_task_span(descs):
    stream = direct_spatial_sparse_backing_stream(descs)
    for idx, qword in enumerate(stream):
        npu_regcmd[idx] = qword
    return [(0, len(stream))]

def direct_spatial_rocket_record_amounts_task_span(descs):
    stream = direct_spatial_rocket_record_amounts_stream(descs)
    for idx, qword in enumerate(stream):
        npu_regcmd[idx] = qword
    return [(0, len(stream))]

def direct_spatial_pc_root6_stream_and_spans(descs):
    validate_direct_spatial_pc_root6_descs(descs)
    stream = direct_spatial_compiler_stitched_stream(descs)
    spans = direct_spatial_pc_root6_expected_spans(len(stream))
    return stream, spans

def write_regs_to_npu_task(task_regs):
    def enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len, pc_core=0):
        enable_npu_units = E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)
        if next_offset is None:
            return [
                E(0x0001, 0, 0),
                E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                E(reg.VERSION, 0, 0),
                enable_npu_units,
            ]
        next_addr = regcmd_mem_create.dma_addr + next_offset * ctypes.sizeof(ctypes.c_uint64)
        return [
            E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
            E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, (pc_core << 16) | (_ceil_div(next_task_regs_len, 2) + 1)),
            E(reg.VERSION, 0, 0),
            enable_npu_units,
        ]

    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + PC_CHAIN_TAIL_QWORDS, 2)
    assert offset <= regcmd_mem_create.size // ctypes.sizeof(ctypes.c_uint64), "regcmd buffer too small"

    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            npu_regcmd[base + i] = qword
        next_task_regs_len = len(task_regs[idx + 1]) if idx + 1 < len(task_regs) else 0
        next_offset = offsets[idx + 1] if idx + 1 < len(task_regs) else None
        pc_core = getattr(regs, "pc_core", 0)
        tails = enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len, pc_core)
        for i, qword in enumerate(tails):
            npu_regcmd[base + len(regs) + i] = qword
        npu_tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)
        npu_tasks[idx].regcfg_amount = len(regs) + len(tails)
        npu_tasks[idx].op_idx = 0
        npu_tasks[idx].enable_mask = 0xd
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9)
        npu_tasks[idx].int_clear = 0x1ffff

def write_raw_task_spans(task_spans):
    for idx, (start, amount) in enumerate(task_spans):
        npu_tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + start * ctypes.sizeof(ctypes.c_uint64)
        npu_tasks[idx].regcfg_amount = amount
        npu_tasks[idx].op_idx = 0
        npu_tasks[idx].enable_mask = 0xd
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9)
        npu_tasks[idx].int_clear = 0x1ffff

def _expand_grouped_weights(weight, in_c, out_c, kh, kw, groups):
    weight_in_c = in_c // groups
    if groups == 1:
        return weight.reshape(out_c, in_c, kh, kw)
    expanded = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
    out_per_group = out_c // groups
    for oc in range(out_c):
        g = oc // out_per_group
        expanded[oc, g * weight_in_c:g * weight_in_c + weight_in_c] = weight[oc]
    return expanded

def _pack_kh_major(weight, out_c, in_c, kh, kw, c2_out):
    aligned_in_c = c2_out * _ceil_div(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:, :in_c] = weight
    ic_group = 32 if aligned_in_c >= 32 and aligned_in_c % 32 == 0 else c2_out
    blocks = []
    for oc in range(0, out_c, 16):
        block = padded[oc:oc + 16]
        block_oc = block.shape[0]
        blocks.append(block.reshape(block_oc, aligned_in_c // ic_group, ic_group, kh, kw).transpose(1, 3, 4, 0, 2).ravel())
    return np.concatenate(blocks)

def _pack_default(weight, out_c, in_c, kh, kw, c2_out):
    aligned_in_c = c2_out * _ceil_div(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:, :in_c] = weight
    return padded.transpose(0, 2, 3, 1).ravel()

def _pack_pointwise_wide(weight, out_c, in_c):
    aligned_in_c = max(32, _align_up(in_c, 32))
    aligned_out_c = _align_up(out_c, 16) if out_c >= 48 else out_c
    padded = np.zeros((aligned_out_c, aligned_in_c), dtype=np.float16)
    padded[:out_c, :in_c] = weight[:, :in_c, 0, 0]
    return np.concatenate([
        padded[oc:oc + 16].reshape(-1, aligned_in_c // 32, 32).transpose(1, 0, 2).ravel()
        for oc in range(0, aligned_out_c, 16)
    ])

def _pack_dw_spatial_major(weight, out_c, in_c, kh, kw, c2_out):
    packed = np.zeros((kh, kw, c2_out), dtype=np.float16)
    packed[:, :, :out_c] = weight[range(out_c), range(out_c)].transpose(1, 2, 0)
    return packed.ravel()

def _is_kh_major(out_c, in_c, kh, kw, groups):
    return (kh != 1 or kw != 1) and not _is_depthwise(in_c, out_c, groups) and not (groups > 1 and in_c != groups)

def choose_weight_layout(out_c, in_c, kh, kw, align_c, groups):
    if _is_depthwise(in_c, out_c, groups) and out_c <= align_c and kh == kw:
        return "depthwise_spatial"
    if _uses_pointwise_weight_atom_layout(in_c, kh, kw, groups):
        return "pointwise_wide"
    if _is_kh_major(out_c, in_c, kh, kw, groups):
        return "kh_major"
    return "default"

def pack_weights(weight_full, out_c, in_c, kh, kw, align_c, groups):
    layout = choose_weight_layout(out_c, in_c, kh, kw, align_c, groups)
    if layout == "depthwise_spatial":
        return _pack_dw_spatial_major(weight_full, out_c, in_c, kh, kw, align_c)
    if layout == "pointwise_wide":
        return _pack_pointwise_wide(weight_full, out_c, in_c)
    if layout == "kh_major":
        return _pack_kh_major(weight_full, out_c, in_c, kh, kw, align_c)
    return _pack_default(weight_full, out_c, in_c, kh, kw, align_c)

def _pack_cdma_dc_feature_input_fp16(input_nchw, p):
    in_c, in_h, in_w = input_nchw.shape
    if p["use_nhwc"]:
        out = np.zeros((in_h, p["width_stride"], in_c), dtype=np.float16)
        out[:, :in_w] = input_nchw.transpose(1, 2, 0)
        return out.ravel()
    c2 = p["input_pack_c2"]
    c1 = _ceil_div(_align_up(in_c, p["align_c"]), c2)
    padded = np.zeros((c1 * c2, in_h, p["width_stride"]), dtype=np.float16)
    padded[:in_c, :, :in_w] = input_nchw
    return padded.reshape(c1, c2, in_h, p["width_stride"]).transpose(0, 2, 3, 1).ravel()

def _unpack_flat_1x1_output(out_raw, out_c, out_h, out_w, out_width_stride, c2):
    c1 = out_raw.size // (out_width_stride * c2)
    packed = out_raw.reshape(1, c1, 1, out_width_stride, c2)
    return packed[0, :, 0, :out_h * out_w, :].transpose(0, 2, 1).reshape(c1 * c2, out_h * out_w)[:out_c].reshape(out_c, out_h, out_w)

def _unpack_nc1hwc2_output(out_raw, out_c, out_h, out_w, c2, out_width_stride=None):
    out_width_stride = out_width_stride or out_h * out_w
    c1 = out_raw.size // (out_width_stride * c2)
    packed = out_raw.reshape(c1, out_width_stride, c2)
    packed = packed[:, :out_h * out_w, :]
    return packed.transpose(0, 2, 1).reshape(c1 * c2, out_h * out_w)[:out_c].reshape(out_c, out_h, out_w)

def make_conv_test_data(batch, in_c, out_c, kh, kw, input_hw, groups):
    in_h, in_w = input_hw
    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, in_c // groups, kh, kw)).astype(np.float16)
    return input_nchw, weight_nchw

def _needs_spatial_im2col_fallback(p, in_c, out_c, kh, kw, groups):
    if not (p["is_spatial"] and groups == 1 and not p["is_depthwise"]):
        return False
    spatial_weight_banks = _mesa_weight_banks(kw, kh, in_c, out_c, False)
    return spatial_weight_banks > RK_CBUF_BANKS // 3 or conv_output_bytes(p) > output_mem_create.size

def _needs_spatial_oc_serial(p, in_c, out_c, groups):
    return (p["is_spatial"] and groups == 1 and not p["is_depthwise"] and
            out_c > UNPACK_C2 and (in_c % 16 != 0 or in_c >= 16))

def _needs_grouped_spatial_serial(p, groups):
    return p["is_spatial"] and groups > 1 and not p["is_depthwise"]

def classify_conv_plan(p, in_c, out_c, kh, kw, in_h, in_w, groups, split_method, tiles):
    is_spatial, is_depthwise = p["is_spatial"], p["is_depthwise"]
    direct_spatial_descs = plan_observed_spatial_tile_descs(
        p, in_c, out_c, kh, kw, in_h, in_w, groups, p["stride"])
    if direct_spatial_descs and direct_spatial_hw_enabled() and direct_spatial_plan_allowed(direct_spatial_descs):
        # Special handle: guarded direct spatial descriptors use full input/weight BOs and common executor submit policy.
        family = FAMILY_DIRECT_SPATIAL
    elif direct_spatial_descs and _needs_spatial_im2col_fallback(p, in_c, out_c, kh, kw, groups):
        # Special handle: do not silently use Python im2col for a proven RKNN direct-spatial schedule.
        family = FAMILY_DIRECT_SPATIAL_GATED
    elif _needs_grouped_spatial_serial(p, groups):
        # Special handle: grouped spatial weights are serialized group-by-group until native grouped descriptors are known.
        family = FAMILY_GROUPED_SERIAL
    elif _needs_spatial_im2col_fallback(p, in_c, out_c, kh, kw, groups):
        # Special handle: dense spatial shapes with high weight/output pressure use Python im2col as a temporary fallback.
        family = FAMILY_SPATIAL_IM2COL_FALLBACK
    elif _needs_spatial_oc_serial(p, in_c, out_c, groups):
        # Special handle: dense spatial OC tiling needs full data banks; low-IC 160x160 cases depend on this.
        family = FAMILY_SPATIAL_OC_SERIAL
    elif is_depthwise and is_spatial:
        # Special handle: depthwise spatial uses channel/Y tiling and diagonal expanded weights.
        family = FAMILY_DEPTHWISE_SPATIAL_TILED
    elif not is_spatial and not is_depthwise and groups == 1 and _needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
        # Special handle: large pointwise OC or height tiling uses flat 1x1 unpack and padded OC tiles.
        family = FAMILY_POINTWISE_OC
    else:
        family = FAMILY_GENERIC_YK
    plan = {"family": family, "tiles": tiles, "split_method": split_method}
    if family == FAMILY_DIRECT_SPATIAL:
        plan["direct_spatial_descs"] = direct_spatial_descs
    return plan

def tile_range(start, tile_size, total):
    end = min(start + tile_size, total)
    return end, end - start

def tile_ranges(total, tile_size):
    for start in range(0, total, tile_size):
        end, count = tile_range(start, tile_size, total)
        yield start, end, count

def output_row_tiles(out_h, row_tile_h):
    for out_row_start, _, tile_out_h in tile_ranges(out_h, row_tile_h):
        yield out_row_start, tile_out_h

def split_method_for_tile_shape(split_y, split_k):
    if split_y and split_k:
        return _SPLIT_BY_YK
    if split_y:
        return _SPLIT_BY_Y
    if split_k:
        return _SPLIT_BY_K
    return _SPLIT_NONE

def input_h_for_output_tile(tile_out_h, kh, stride):
    return (tile_out_h - 1) * stride + kh

def hardware_tile_h(tile_out_h, minimum):
    return max(minimum, tile_out_h)

def grouped_serial_tile_records(p, in_c, out_c, in_h, in_w, groups, stride):
    out_h, out_w = p["out_h"], p["out_w"]
    input_per_group = in_c // groups
    out_per_group = out_c // groups
    records = []
    for g, (oc_start, oc_end, tile_out_c) in enumerate(tile_ranges(out_c, out_per_group)):
        records.append(TileRecord(
            family=FAMILY_GROUPED_SERIAL,
            split_method=_SPLIT_BY_K,
            y_start=0,
            y_count=out_h,
            k_start=oc_start,
            k_count=tile_out_c,
            input_y=0,
            input_h=in_h,
            input_w=in_w,
            output_y=0,
            output_h=out_h,
            output_w=out_w,
            tile_in_c=input_per_group,
            input_c_start=g * input_per_group,
            tile_groups=1,
            hw_oc=tile_out_c,
            layout_mode="grouped_serial",
            unpack_kind=UNPACK_KIND_NC1HWC2,
            reason="grouped serial group slice"))
    return records

def materialize_grouped_serial_record(record, n, input_nchw, weight_nchw, result, kh, kw, stride):
    input_tile = input_nchw[n, record.input_c_start:record.input_c_start + record.tile_in_c]
    group_weight = weight_nchw[record.k_start:record.k_start + record.k_count].reshape(
        record.k_count, record.tile_in_c, kh, kw)
    tp = _conv_params(record.tile_in_c, record.input_h, record.input_w, record.hw_oc,
                      kh, kw, 1, stride=stride)
    packed_input, packed_weight, regs, out_count = materialize_conv_tile(
        tp, input_tile, group_weight, record.hw_oc, record.tile_in_c, kh, kw, 1)
    return make_record_tile_desc(record, result, n, packed_input, packed_weight, regs,
                                 tp, out_count, channels=tp["align_out_c"])

def spatial_oc_serial_tile_records(p, in_c, out_c, kh, in_w, stride):
    out_h, out_w = p["out_h"], p["out_w"]
    oc_tile = UNPACK_C2
    row_tile_h = 32
    split_method = split_method_for_tile_shape(row_tile_h < out_h, oc_tile < out_c)
    records = []
    for out_row_start, tile_out_h in output_row_tiles(out_h, 32):
        tile_in_h = input_h_for_output_tile(tile_out_h, kh, stride)
        for oc_start, oc_end, tile_out_c in tile_ranges(out_c, oc_tile):
            records.append(TileRecord(
                family=FAMILY_SPATIAL_OC_SERIAL,
                split_method=split_method,
                y_start=out_row_start,
                y_count=tile_out_h,
                k_start=oc_start,
                k_count=tile_out_c,
                input_y=out_row_start * stride,
                input_h=tile_in_h,
                input_w=in_w,
                output_y=out_row_start,
                output_h=tile_out_h,
                output_w=out_w,
                tile_in_c=in_c,
                tile_groups=1,
                hw_oc=tile_out_c,
                full_data_bank=True,
                layout_mode="spatial_oc_serial",
                unpack_kind=UNPACK_KIND_NC1HWC2,
                reason="dense spatial output-channel serial tile"))
    return records

def materialize_spatial_oc_serial_record(record, n, input_nchw, weight_nchw, result, kh, kw, stride):
    input_tile = input_nchw[n, :, record.input_y:record.input_y + record.input_h, :]
    tile_weight = weight_nchw[record.k_start:record.k_start + record.k_count].reshape(
        record.k_count, record.tile_in_c, kh, kw)
    tp = _conv_params(record.tile_in_c, record.input_h, record.input_w, record.hw_oc, kh, kw, 1, stride)
    packed_input, packed_weight, regs, out_count = materialize_conv_tile(
        tp, input_tile, tile_weight, record.hw_oc, record.tile_in_c, kh, kw, 1,
        full_data_bank=record.full_data_bank)
    return make_record_tile_desc(record, result, n, packed_input, packed_weight, regs, tp, out_count)

def depthwise_spatial_tile_records(p, out_c, kh, kw, in_w, stride):
    out_h, out_w = p["out_h"], p["out_w"]
    channel_tile = min(32, out_c)
    row_tile_h = max(1, _depthwise_tile_h(out_c, out_h, in_w, kh, kw, stride))
    split_method = split_method_for_tile_shape(row_tile_h < out_h, channel_tile < out_c)
    records = []
    for out_row_start, tile_out_h in output_row_tiles(out_h, row_tile_h):
        hw_tile_out_h = hardware_tile_h(tile_out_h, 2)
        tile_in_h = input_h_for_output_tile(hw_tile_out_h, kh, stride)
        for ch_start, ch_end, tile_c in tile_ranges(out_c, channel_tile):
            records.append(TileRecord(
                family=FAMILY_DEPTHWISE_SPATIAL_TILED,
                split_method=split_method,
                y_start=out_row_start,
                y_count=tile_out_h,
                k_start=ch_start,
                k_count=tile_c,
                input_y=out_row_start * stride,
                input_h=tile_in_h,
                input_w=in_w,
                output_y=out_row_start,
                output_h=tile_out_h,
                output_w=out_w,
                tile_in_c=tile_c,
                tile_groups=tile_c,
                hw_oc=tile_c,
                layout_mode="depthwise_spatial",
                unpack_kind=UNPACK_KIND_NC1HWC2,
                reason="depthwise spatial channel and row tile"))
    return records

def materialize_depthwise_spatial_record(record, n, input_nchw, weight_nchw, result, kh, kw, in_h, stride):
    real_in_h = min(record.input_h, in_h - record.input_y)
    input_tile = np.zeros((record.tile_in_c, record.input_h, record.input_w), dtype=np.float16)
    input_tile[:, :real_in_h] = input_nchw[n, record.k_start:record.k_start + record.k_count,
                                           record.input_y:record.input_y + real_in_h, :]
    tile_weight = np.zeros((record.k_count, record.k_count, kh, kw), dtype=np.float16)
    for local_c in range(record.k_count):
        tile_weight[local_c, local_c] = weight_nchw[record.k_start + local_c, 0]
    tp = _conv_params(record.tile_in_c, record.input_h, record.input_w, record.hw_oc,
                      kh, kw, record.tile_groups, stride=stride)
    packed_input, packed_weight, regs, out_count = materialize_conv_tile(
        tp, input_tile, tile_weight, record.hw_oc, record.tile_in_c, kh, kw, record.tile_groups)
    return make_record_tile_desc(record, result, n, packed_input, packed_weight, regs,
                                 tp, out_count, hw_h=hardware_tile_h(record.y_count, 2))

def padded_oc_count(tile_out_c, oc_tile):
    return oc_tile if tile_out_c < oc_tile else tile_out_c

def pointwise_oc_tile_records(p, in_c, out_c, in_w, stride):
    out_h, out_w = p["out_h"], p["out_w"]
    oc_tile = _pointwise_oc_tile_c(in_c)
    row_tile_h = _pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile, stride)
    split_method = split_method_for_tile_shape(row_tile_h < out_h, oc_tile < out_c)
    records = []
    for out_row_start, tile_out_h in output_row_tiles(out_h, row_tile_h):
        hw_tile_h = hardware_tile_h(tile_out_h, 7)
        for oc_start, oc_end, tile_out_c in tile_ranges(out_c, oc_tile):
            hw_out_c = padded_oc_count(tile_out_c, oc_tile)
            records.append(TileRecord(
                family=FAMILY_POINTWISE_OC,
                split_method=split_method,
                y_start=out_row_start,
                y_count=tile_out_h,
                k_start=oc_start,
                k_count=tile_out_c,
                input_y=out_row_start * stride,
                input_h=hw_tile_h,
                input_w=in_w,
                output_y=out_row_start,
                output_h=tile_out_h,
                output_w=out_w,
                tile_in_c=in_c,
                tile_groups=p["groups"],
                hw_oc=hw_out_c,
                full_data_bank=True,
                layout_mode="pointwise_oc",
                unpack_kind=UNPACK_KIND_FLAT_1X1,
                reason="pointwise output-channel tile"))
    return records

def materialize_pointwise_oc_record(record, n, input_nchw, weight_nchw, result, kh, kw, groups, stride):
    input_tile = np.zeros((record.tile_in_c, record.input_h, record.input_w), dtype=np.float16)
    input_tile[:, :record.y_count] = input_nchw[n, :, record.input_y:record.input_y + record.y_count, :]
    tile_weight = weight_nchw.reshape(-1, record.tile_in_c // groups, kh, kw)[record.k_start:record.k_start + record.k_count]
    tile_weight = pad_output_channels(tile_weight, record.k_count, record.hw_oc)
    tp = _conv_params(record.tile_in_c, record.input_h, record.input_w, record.hw_oc, kh, kw, groups, stride)
    packed_input, packed_weight, regs, out_count = materialize_conv_tile(
        tp, input_tile, tile_weight, record.hw_oc, record.tile_in_c, kh, kw, groups,
        full_data_bank=record.full_data_bank)
    return make_record_tile_desc(record, result, n, packed_input, packed_weight, regs,
                                 tp, out_count, hw_h=record.input_h)

def pad_output_channels(weight_tile, real_out_c, hw_out_c):
    if hw_out_c <= real_out_c:
        return weight_tile
    padded = np.zeros((hw_out_c,) + weight_tile.shape[1:], dtype=np.float16)
    padded[:real_out_c] = weight_tile
    return padded

def spatial_im2col_fallback_tile_records(p, in_c, out_c, kh, kw, stride):
    out_h, out_w = p["out_h"], p["out_w"]
    flat_c = in_c * kh * kw
    oc_tile = _pointwise_oc_tile_c(flat_c)
    row_tile_h = _pointwise_oc_tile_h(flat_c, out_h, out_w, oc_tile)
    split_method = split_method_for_tile_shape(row_tile_h < out_h, oc_tile < out_c)
    records = []
    for out_row_start, tile_out_h in output_row_tiles(out_h, row_tile_h):
        hw_tile_h = hardware_tile_h(tile_out_h, 7)
        for oc_start, oc_end, tile_out_c in tile_ranges(out_c, oc_tile):
            hw_out_c = padded_oc_count(tile_out_c, oc_tile)
            records.append(TileRecord(
                family=FAMILY_SPATIAL_IM2COL_FALLBACK,
                split_method=split_method,
                y_start=out_row_start,
                y_count=tile_out_h,
                k_start=oc_start,
                k_count=tile_out_c,
                input_y=out_row_start * stride,
                input_h=hw_tile_h,
                input_w=out_w,
                output_y=out_row_start,
                output_h=tile_out_h,
                output_w=out_w,
                tile_in_c=flat_c,
                hw_oc=hw_out_c,
                full_data_bank=True,
                layout_mode="spatial_im2col_fallback",
                unpack_kind=UNPACK_KIND_FLAT_1X1,
                reason="temporary software im2col fallback"))
    return records

def materialize_spatial_im2col_fallback_record(record, n, input_nchw, weight_nchw, result,
                                               in_c, kh, kw, stride):
    im2col = np.zeros((record.tile_in_c, record.input_h, record.input_w), dtype=np.float16)
    flat = 0
    for ic in range(in_c):
        for ky in range(kh):
            src_y = record.input_y + ky
            for kx in range(kw):
                im2col[flat, :record.y_count] = input_nchw[n, ic,
                                                           src_y:src_y + record.y_count * stride:stride,
                                                           kx:kx + record.output_w * stride:stride]
                flat += 1

    tile_weight = np.zeros((record.hw_oc, record.tile_in_c, 1, 1), dtype=np.float16)
    for local_oc, oc in enumerate(range(record.k_start, record.k_start + record.k_count)):
        flat = 0
        for ic in range(in_c):
            for ky in range(kh):
                for kx in range(kw):
                    tile_weight[local_oc, flat, 0, 0] = weight_nchw[oc, ic, ky, kx]
                    flat += 1

    tp = _conv_params(record.tile_in_c, record.input_h, record.input_w, record.hw_oc, 1, 1, 1)
    packed_input, packed_weight, regs, out_count = materialize_conv_tile(
        tp, im2col, tile_weight, record.hw_oc, record.tile_in_c, 1, 1, 1,
        full_data_bank=record.full_data_bank)
    return make_record_tile_desc(record, result, n, packed_input, packed_weight, regs,
                                 tp, out_count, hw_h=record.input_h)

def generic_yk_tile_records(plan, p, in_c, kh, in_h, in_w, stride):
    records = []
    out_offset = 0
    for tile in plan["tiles"]:
        ys, y_span = tile["y_start"], tile["y_step"]
        ks, k_span = tile["k_start"], tile["k_step"]
        if p["is_depthwise"]:
            tile_ic, tile_g, hw_oc = k_span, k_span, k_span
            input_h = min((y_span - 1) * stride + kh, in_h - ys * stride)
        else:
            tile_ic, tile_g = in_c, p["groups"]
            hw_oc = padded_oc_count(k_span, _pointwise_oc_tile_c(in_c))
            t_in = input_h_for_output_tile(y_span, kh, stride)
            input_h = min(t_in, in_h - ys * stride)
        records.append(TileRecord(
            family=FAMILY_GENERIC_YK,
            split_method=plan["split_method"],
            y_start=ys,
            y_count=y_span,
            k_start=ks,
            k_count=k_span,
            input_y=ys * stride,
            input_h=input_h,
            input_w=in_w,
            output_y=ys,
            output_h=y_span,
            output_w=p["out_w"],
            tile_in_c=tile_ic,
            tile_groups=tile_g,
            hw_oc=hw_oc,
            out_offset=out_offset,
            full_data_bank=(plan["split_method"] != _SPLIT_NONE or
                            (p["is_spatial"] and p["groups"] == 1)),
            layout_mode=("depthwise" if p["is_depthwise"] else "dense"),
            unpack_kind=UNPACK_KIND_NC1HWC2,
            reason="generic RKNN-style Y/K tile planner"))
        tile_p = _conv_params(tile_ic, input_h, in_w, hw_oc, kh, p["kw"], tile_g, stride)
        out_offset += conv_output_bytes(tile_p)
    return records

def _get_generic_input_tile(inp, record, p, in_c, kh, in_h, in_w, stride):
    ys, y_span = record.y_start, record.y_count
    k_span = record.k_count
    if p["is_depthwise"]:
        ks = record.k_start
        tn = min((y_span - 1) * stride + kh, in_h - ys * stride)
        ti = np.zeros((k_span, tn, in_w), dtype=np.float16)
        rih = min(tn, in_h - ys * stride)
        ti[:, :rih] = inp[ks:ks + k_span, ys * stride:ys * stride + rih, :]
        return ti, tn
    t_in = input_h_for_output_tile(y_span, kh, stride)
    hw_th = max(t_in, hardware_tile_h(y_span, 7))
    tn = min(t_in, in_h - ys * stride)
    ti = np.zeros((in_c, hw_th, in_w), dtype=np.float16)
    rih = min(tn, in_h - ys * stride)
    ti[:, :rih] = inp[:, ys * stride:ys * stride + rih, :]
    return ti, tn

def _get_generic_weight_tile(wt, record, p, in_c, kh, kw):
    ks, k_span = record.k_start, record.k_count
    groups = p["groups"]
    if p["is_depthwise"]:
        tw = np.zeros((k_span, k_span, kh, kw), dtype=np.float16)
        for i in range(k_span):
            tw[i, i] = wt[ks + i, 0]
        return tw
    tw = wt[ks:ks + k_span].reshape(k_span, in_c // groups, kh, kw)
    if groups > 1:
        tw = _expand_grouped_weights(tw, in_c, k_span, kh, kw, groups)
    return tw

def materialize_generic_tile_record(record, n, input_nchw, weight_nchw, result, p,
                                    in_c, kh, kw, in_h, in_w, stride):
    ti, tn_actual = _get_generic_input_tile(input_nchw[n], record, p, in_c, kh, in_h, in_w, stride)
    tw = _get_generic_weight_tile(weight_nchw, record, p, in_c, kh, kw)
    tw = pad_output_channels(tw, record.k_count, record.hw_oc)

    tp = _conv_params(record.tile_in_c, tn_actual, in_w, record.hw_oc, kh, kw,
                      record.tile_groups, stride)
    out_dma = output_mem_create.dma_addr + record.out_offset
    packed_input, packed_weight, regs, out_count = materialize_conv_tile(
        tp, ti, tw, record.hw_oc, record.tile_in_c, kh, kw, record.tile_groups,
        out_dma=out_dma, full_data_bank=record.full_data_bank)
    return make_record_tile_desc(record, result, n, packed_input, packed_weight, regs, tp, out_count)

def _build_direct_spatial_from_plan(plan, ctx):
    descs, packed_input, packed_weight = build_direct_spatial_descs(
        ctx.n, ctx.input_nchw, ctx.weight_nchw, ctx.p,
        ctx.in_c, ctx.out_c, ctx.kh, ctx.kw, ctx.in_h, ctx.in_w, ctx.groups, ctx.stride,
        descs=plan.get("direct_spatial_descs"))
    if not descs:
        return []
    return [make_direct_spatial_tile_desc(ctx.result, ctx.n, packed_input, packed_weight, descs, ctx.p)]

def _build_gated_direct_spatial_from_plan(plan, ctx):
    del plan, ctx
    raise RuntimeError(
        f"direct-spatial RKNN descriptor schedule is available but hardware execution is gated; "
        f"set {DIRECT_SPATIAL_ENV}=1 and {DIRECT_SPATIAL_UNSAFE_ENV}=1 for the explicit probe")

@dataclass(frozen=True, slots=True)
class ConvBuildContext:
    n: int
    input_nchw: np.ndarray
    weight_nchw: np.ndarray
    result: np.ndarray
    p: dict
    in_c: int
    out_c: int
    kh: int
    kw: int
    in_h: int
    in_w: int
    groups: int
    stride: int

def _record_planner(plan, ctx):
    family = plan["family"]
    if family == FAMILY_GROUPED_SERIAL:
        return grouped_serial_tile_records(ctx.p, ctx.in_c, ctx.out_c, ctx.in_h, ctx.in_w, ctx.groups, ctx.stride)
    if family == FAMILY_SPATIAL_IM2COL_FALLBACK:
        return spatial_im2col_fallback_tile_records(ctx.p, ctx.in_c, ctx.out_c, ctx.kh, ctx.kw, ctx.stride)
    if family == FAMILY_SPATIAL_OC_SERIAL:
        return spatial_oc_serial_tile_records(ctx.p, ctx.in_c, ctx.out_c, ctx.kh, ctx.in_w, ctx.stride)
    if family == FAMILY_DEPTHWISE_SPATIAL_TILED:
        return depthwise_spatial_tile_records(ctx.p, ctx.out_c, ctx.kh, ctx.kw, ctx.in_w, ctx.stride)
    if family == FAMILY_POINTWISE_OC:
        return pointwise_oc_tile_records(ctx.p, ctx.in_c, ctx.out_c, ctx.in_w, ctx.stride)
    if family == FAMILY_GENERIC_YK:
        return generic_yk_tile_records(plan, ctx.p, ctx.in_c, ctx.kh, ctx.in_h, ctx.in_w, ctx.stride)
    raise ValueError(f"unknown record family: {family}")

def _materialize_record(record, ctx):
    if record.family == FAMILY_GROUPED_SERIAL:
        return materialize_grouped_serial_record(record, ctx.n, ctx.input_nchw, ctx.weight_nchw,
                                                 ctx.result, ctx.kh, ctx.kw, ctx.stride)
    if record.family == FAMILY_SPATIAL_IM2COL_FALLBACK:
        return materialize_spatial_im2col_fallback_record(record, ctx.n, ctx.input_nchw, ctx.weight_nchw,
                                                          ctx.result, ctx.in_c, ctx.kh, ctx.kw, ctx.stride)
    if record.family == FAMILY_SPATIAL_OC_SERIAL:
        return materialize_spatial_oc_serial_record(record, ctx.n, ctx.input_nchw, ctx.weight_nchw,
                                                    ctx.result, ctx.kh, ctx.kw, ctx.stride)
    if record.family == FAMILY_DEPTHWISE_SPATIAL_TILED:
        return materialize_depthwise_spatial_record(record, ctx.n, ctx.input_nchw, ctx.weight_nchw,
                                                    ctx.result, ctx.kh, ctx.kw, ctx.in_h, ctx.stride)
    if record.family == FAMILY_POINTWISE_OC:
        return materialize_pointwise_oc_record(record, ctx.n, ctx.input_nchw, ctx.weight_nchw,
                                               ctx.result, ctx.kh, ctx.kw, ctx.groups, ctx.stride)
    if record.family == FAMILY_GENERIC_YK:
        return materialize_generic_tile_record(record, ctx.n, ctx.input_nchw, ctx.weight_nchw,
                                               ctx.result, ctx.p, ctx.in_c, ctx.kh, ctx.kw,
                                               ctx.in_h, ctx.in_w, ctx.stride)
    raise ValueError(f"unknown record family: {record.family}")

def _build_record_descriptors(plan, ctx):
    return [_materialize_record(record, ctx) for record in _record_planner(plan, ctx)]

DESCRIPTOR_PRODUCERS = {
    FAMILY_DIRECT_SPATIAL: _build_direct_spatial_from_plan,
    FAMILY_DIRECT_SPATIAL_GATED: _build_gated_direct_spatial_from_plan,
    FAMILY_GROUPED_SERIAL: _build_record_descriptors,
    FAMILY_SPATIAL_IM2COL_FALLBACK: _build_record_descriptors,
    FAMILY_SPATIAL_OC_SERIAL: _build_record_descriptors,
    FAMILY_DEPTHWISE_SPATIAL_TILED: _build_record_descriptors,
    FAMILY_POINTWISE_OC: _build_record_descriptors,
    FAMILY_GENERIC_YK: _build_record_descriptors,
}

def plan_conv_descriptors(n, input_nchw, weight_nchw, result, in_c, out_c, kh, kw, input_hw, groups=1, stride=1):
    in_h, in_w = input_hw
    p, split_method, tiles = _plan_conv_tiles(in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    plan = classify_conv_plan(p, in_c, out_c, kh, kw, in_h, in_w, groups,
                              split_method, tiles)
    ctx = ConvBuildContext(n, input_nchw, weight_nchw, result, p,
                           in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    producer = DESCRIPTOR_PRODUCERS[plan["family"]]
    return producer(plan, ctx)

def direct_spatial_hw_enabled():
    return (os.environ.get(DIRECT_SPATIAL_ENV) == "1" and
            os.environ.get(DIRECT_SPATIAL_UNSAFE_ENV) == "1")

def run_conv(batch, in_c, out_c, kh, kw, input_hw, groups=1, stride=1):
    in_h, in_w = input_hw
    input_nchw, weight_nchw = make_conv_test_data(batch, in_c, out_c, kh, kw, input_hw, groups)

    p = _conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    out_h, out_w = p["out_h"], p["out_w"]
    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)

    for n in range(batch):
        descs = plan_conv_descriptors(n, input_nchw, weight_nchw, result,
                                      in_c, out_c, kh, kw, input_hw, groups, stride)
        execute_conv_descriptors(descs)

    return result, input_nchw, weight_nchw

def compute_expected_nchw(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups=1, stride=1):
    out_h, out_w = conv_output_hw(in_h, in_w, kh, kw, stride)
    i64, w64 = inp.astype(np.float64), wt.astype(np.float64)
    expected = np.zeros((batch, out_c, out_h, out_w))
    for n in range(batch):
        for g in range(groups):
            for oc in range(g * out_c // groups, (g + 1) * out_c // groups):
                for ic in range(g * in_c // groups, (g + 1) * in_c // groups):
                    for i in range(kh):
                        for j in range(kw):
                            expected[n, oc] += i64[n, ic, i:i+stride*out_h:stride, j:j+stride*out_w:stride] * w64[oc, ic - g * in_c // groups, i, j]
    return expected

def select_shapes(shapes, requested):
    requested = set(requested)
    if not requested:
        return shapes
    available = {s["name"] for s in shapes}
    missing = sorted(requested - available)
    if missing:
        raise SystemExit(f"unknown conv shape(s): {', '.join(missing)}")
    return [s for s in shapes if s["name"] in requested]

def shape_stride(s):
    return s.get("stride", 1)

def shape_input_text(s):
    return f"{s['in_c']}x{s['in_h']}x{s['in_w']}"

def shape_output_text(s):
    out_h, out_w = conv_output_hw(s["in_h"], s["in_w"], s["kh"], s["kw"], shape_stride(s))
    return f"{s['out_c']}x{out_h}x{out_w}"

def shape_direct_spatial_descs(s):
    stride = shape_stride(s)
    p = _conv_params(s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["kh"], s["kw"], s["groups"], stride)
    return plan_observed_spatial_tile_descs(
        p, s["in_c"], s["out_c"], s["kh"], s["kw"], s["in_h"], s["in_w"], s["groups"], stride)

def shape_requires_direct_spatial_gate(s):
    descs = shape_direct_spatial_descs(s)
    if not descs or direct_spatial_hw_enabled():
        return False
    stride = shape_stride(s)
    p = _conv_params(s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["kh"], s["kw"], s["groups"], stride)
    return _needs_spatial_im2col_fallback(p, s["in_c"], s["out_c"], s["kh"], s["kw"], s["groups"])

def print_shape_result(s, nw, iw, status, max_diff=None):
    diff_text = "" if max_diff is None else f"  (max_diff={max_diff:.4f})"
    print(f"  {s['name']:<{nw}s} {shape_input_text(s):<{iw}s} -> {shape_output_text(s)}  {status}{diff_text}")

def run_shape_check(s):
    if shape_requires_direct_spatial_gate(s):
        return "GATED", None
    stride = shape_stride(s)
    result, inp, wt = run_conv(s["batch"], s["in_c"], s["out_c"], s["kh"], s["kw"],
                               (s["in_h"], s["in_w"]), groups=s["groups"], stride=stride)
    expected = compute_expected_nchw(inp, wt, s["batch"], s["in_c"], s["in_h"], s["in_w"],
                                     s["out_c"], s["kh"], s["kw"], groups=s["groups"], stride=stride)
    max_diff = float(np.max(np.abs(result.astype(np.float64) - expected)))
    ok = np.allclose(result, expected, atol=0.2) and not np.any(np.isinf(result))
    return "PASS" if ok else "FAIL", max_diff

def run_shape_suite(shapes, requested):
    shapes = select_shapes(shapes, requested)
    name_width = max(len(s["name"]) for s in shapes)
    input_width = max(len(shape_input_text(s)) for s in shapes)
    for s in shapes:
        status, max_diff = run_shape_check(s)
        print_shape_result(s, name_width, input_width, status, max_diff)

def conv_shape(name, batch, in_c, in_h, in_w, out_c, kh, kw, groups, stride=1):
    s = dict(name=name, batch=batch, in_c=in_c, in_h=in_h, in_w=in_w, out_c=out_c, kh=kh, kw=kw, groups=groups)
    if stride != 1:
        s["stride"] = stride
    return s

def shape_specs(specs): return [conv_shape(*spec) for spec in specs]

def conv2d_b1_shape(in_c, in_h, in_w, out_c, kh, kw, groups=1):
    return conv_shape(f"conv2d_b1_c{in_c}_h{in_h}_w{in_w}_oc{out_c}_wic{in_c // groups}_k{kh}x{kw}_g{groups}", 1, in_c, in_h, in_w, out_c, kh, kw, groups)

def conv2d_b1_shapes(specs): return [conv2d_b1_shape(*spec) for spec in specs]

def conv1d_shape(name, batch, in_c, kw, groups=1): return conv_shape(name, batch, in_c, 1, 11, 6, 1, kw, groups)

def grouped_spatial_shape(in_c, out_c, groups): return conv2d_b1_shape(in_c, 5, 5, out_c, 3, 3, groups)

def test_ops_conv2d_shape(name, in_c, kh, kw, groups=1): return conv_shape(name, 1, in_c, 5, 7, 6, kh, kw, groups)

def square_shape(in_c, hw, out_c, kh, kw, groups=1):
    return conv_shape(f"b1_c{in_c}_h{hw}_w{hw}_oc{out_c}_wic{in_c // groups}_k{kh}x{kw}_g{groups}", 1, in_c, hw, hw, out_c, kh, kw, groups)

def square_shapes(specs): return [square_shape(*spec) for spec in specs]

def cc_shape(in_c, hw, out_c, kh, kw, groups=1):
    return conv_shape(f"conv2d_cc_b1_c{in_c}_h{hw}_w{hw}_oc{out_c}_wic{in_c // groups}_k{kh}x{kw}_g{groups}", 1, in_c, hw, hw, out_c, kh, kw, groups)

def cc_shapes(specs): return [cc_shape(*spec) for spec in specs]

def pvalid_shape(in_c, hw, out_c, kh, kw, groups=1):
    return conv_shape(f"b1_c{in_c}_h{hw}_w{hw}_oc{out_c}_wic{in_c // groups}_k{kh}x{kw}_g{groups}_s1_pvalid", 1, in_c, hw, hw, out_c, kh, kw, groups)

def pvalid_shapes(specs): return [pvalid_shape(*spec) for spec in specs]
def pvalid_pointwise_shapes(in_c, hw, out_channels): return [pvalid_shape(in_c, hw, out_c, 1, 1) for out_c in out_channels]
def pvalid_pointwise_clusters(specs): return [pvalid_shape(in_c, hw, out_c, 1, 1) for in_c, hw, out_channels in specs for out_c in out_channels]

def conv_shapes():
    return [
        # -- 1x1 kernels (fully supported via NHWC mode + channel slicing for ic>=5) --
        *shape_specs((
            ("conv2d_1x6_1x1_4x4", 1, 1, 4, 4, 6, 1, 1, 1),
            ("conv2d_3x3_1x1_4x4", 1, 3, 4, 4, 3, 1, 1, 1),
            ("conv2d_4x2_1x1_4x4", 1, 4, 4, 4, 2, 1, 1, 1),
        )),
        *conv2d_b1_shapes((
            (4, 9, 9, 4, 1, 1),
            (3, 52, 52, 6, 1, 1),
            (96, 56, 56, 24, 1, 1),
            (144, 56, 56, 24, 1, 1),
            (144, 28, 28, 32, 1, 1),
            (192, 28, 28, 32, 1, 1),
            (192, 28, 28, 16, 1, 1),
            (256, 28, 28, 32, 1, 1),
        )),
        conv_shape("conv2d_16x16_1x1_8x8", 1, 16, 8, 8, 16, 1, 1, 1),
        conv2d_b1_shape(16, 32, 32, 16, 1, 1),

        # -- Non-1x1 kernels (partial output -- known NPU hardware limitation) --
        *conv2d_b1_shapes((
            (4, 9, 9, 4, 3, 3),
            (16, 18, 18, 16, 3, 3),
            (1, 5, 7, 6, 3, 3),
        )),
        conv_shape("conv2d_b2_c4_h9_w9_oc4_wic4_k3x3_g1", 2, 4, 9, 9, 4, 3, 3, 1),

        # Depthwise
        conv2d_b1_shape(3, 11, 28, 3, 3, 3, 3),

        # Non-square kernels
        conv_shape("conv2d_3x6_1x3_5x5", 1, 3, 5, 5, 6, 1, 3, 1),

        # -- test_ops.py _test_conv2d(cin=3): (3,5,7) @ (6,kh,kw) --
        conv2d_b1_shape(3, 5, 7, 6, 3, 3, 3),
        *[test_ops_conv2d_shape(f"conv2d_b1_c3_h5_w7_oc6_wic3_k{kh}x{kw}_g1", 3, kh, kw)
          for kh in (2, 3) for kw in (1, 3, 5)],

        # -- test_ops.py _test_conv2d(cin=1): (1,5,7) @ (6,kh,kw) --
        *[test_ops_conv2d_shape(name, 1, kh, kw)
          for name, kh, kw in (
              ("conv2d_1x6_2x1_5x7", 2, 1),
              ("conv2d_1x6_2x3_5x7", 2, 3),
              ("conv2d_1x6_3x1_5x7_b", 3, 1),
              ("conv2d_1x6_3x5_5x7", 3, 5),
          )],

        # -- Grouped convs from test_ops.py --
        *shape_specs((
            ("conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2", 1, 4, 1, 1, 2, 1, 1, 2),
            ("conv2d_4x4_1x1_1x1_g2", 1, 4, 1, 1, 4, 1, 1, 2),
        )),
        conv2d_b1_shape(32, 32, 32, 32, 1, 1, 32),
        conv2d_b1_shape(15, 5, 5, 35, 3, 3, 5),

        # -- Batch >1 coverage --
        *shape_specs((
            ("conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3", 2, 3, 11, 28, 3, 3, 3, 3),
            ("conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5", 4, 15, 5, 5, 35, 3, 3, 5),
        )),

        # -- Grouped output-channel variants --
        *[grouped_spatial_shape(in_c, out_c, groups)
          for in_c, groups, out_channels in (
              (4, 2, (4, 8, 12)),
              (6, 3, (6, 12, 18)),
              (15, 5, (20, 25, 30, 40)),
          )
          for out_c in out_channels],
        *shape_specs((
            ("conv2d_2x2_1x1_4x4", 1, 2, 4, 4, 2, 1, 1, 1),
            ("conv2d_8x8_1x1_5x5", 1, 8, 5, 5, 8, 1, 1, 1),
            ("conv2d_10x20_3x3_9x9", 1, 10, 9, 9, 20, 3, 3, 1),
            ("conv2d_16x16_3x3_9x9", 1, 16, 9, 9, 16, 3, 3, 1),
            ("conv2d_2x4_3x3_6x6", 1, 2, 6, 6, 4, 3, 3, 1),
            ("conv2d_2x4_2x2_5x5", 1, 2, 5, 5, 4, 2, 2, 1),
            ("conv2d_1x32_5x5_10x10", 1, 1, 10, 10, 32, 5, 5, 1),
            ("conv2d_8x4_4x4_10x10", 1, 8, 10, 10, 4, 4, 4, 1),
        )),

        # MobileNet layers
        *cc_shapes((
            (3, 224, 32, 3, 3),          # first spatial conv (RGB->32ch)
            (32, 112, 32, 3, 3, 32),     #  depthwise conv (3x3 sep)
            (32, 112, 64, 1, 1),         # pointwise (1x1 projection)
            (64, 112, 64, 3, 3, 64),     # depthwise conv (64ch)
            (64, 56, 128, 1, 1),         # pointwise expansion
            (128, 56, 128, 3, 3, 128),   # depthwise conv (128ch)
            (128, 56, 128, 1, 1),        # pointwise (1x1 projection, same-channel)
            (128, 28, 256, 1, 1),        # Pointwise 128->256
            (256, 28, 256, 3, 3, 256),   # Depthwise 256x28
            (256, 28, 256, 1, 1),        # Pointwise 256->256
            (256, 14, 512, 1, 1),        # Pointwise 256->512
            (512, 14, 512, 3, 3, 512),   # Depthwise 512x14
            (512, 14, 512, 1, 1),        # Pointwise 512->512
            # 7x7 and classifier-head fixes
            (512, 7, 1024, 1, 1),
            (1024, 7, 1024, 3, 3, 1024),
            (1024, 7, 1024, 1, 1),
            (1024, 7, 1024, 7, 7, 1024),
            (1024, 1, 1001, 1, 1),
        )),

        # conv1d expressed as conv2d: input H=1, W=input_size; kernel H=1, W=kernel_size.
        *[conv1d_shape(name, batch, in_c, kw, groups)
          for name, batch, in_c, kw, groups in (
              ("conv1d_bs1_as_conv2d", 1, 1, 1, 1),
              ("conv1d_bs8_as_conv2d", 8, 1, 1, 1),
              ("conv1d_bs1_612_as_conv2d", 1, 1, 2, 1),
              ("conv1d_bs1_615_as_conv2d", 1, 1, 5, 1),
              ("conv1d_bs1_1311_631_as_conv2d", 1, 3, 1, 1),
              ("conv1d_bs1_1311_632_as_conv2d", 1, 3, 2, 1),
              ("conv1d_bs1_1311_635_as_conv2d", 1, 3, 5, 1),
              ("conv1d_bs1_1311_615_g3_as_conv2d", 1, 3, 5, 3),
              ("conv1d_bs8_8111_611_as_conv2d", 8, 1, 1, 1),
              ("conv1d_bs8_8111_612_a_as_conv2d", 8, 1, 2, 1),
              ("conv1d_bs8_8111_612_b_as_conv2d", 8, 1, 2, 1),
              ("conv1d_bs8_8111_615_as_conv2d", 8, 1, 5, 1),
              ("conv1d_bs8_8311_631_as_conv2d", 8, 3, 1, 1),
              ("conv1d_bs8_8311_632_as_conv2d", 8, 3, 2, 1),
              ("conv1d_bs8_8311_635_a_as_conv2d", 8, 3, 5, 1),
              ("conv1d_bs8_8311_635_g3_as_conv2d", 8, 3, 5, 3),
          )],

        # -- Large spatial 1x1 conv where IC=3, OC=6 (fixed; promoted above) --
        *[conv_shape(f"1x3_{hw}x{hw}_k1", 1, 3, hw, hw, 6, 1, 1, 1)
          for hw in range(54, 74, 2)],

        # -- 1x1 conv large input channel >> output channel (wrong numerical result) --
        # From mobilenetv2 depthwise expand/project layers
        # fixed/promoted above: b1_c96_h56_w56_oc24_wic96_k1x1_g1
        # fixed/promoted above: b1_c144_h56_w56_oc24_wic144_k1x1_g1
        # fixed/promoted above: b1_c144_h28_w28_oc32_wic144_k1x1_g1
        # fixed/promoted above: b1_c192_h28_w28_oc32_wic192_k1x1_g1
        # fixed/promoted above: b1_c192_h28_w28_oc16_wic192_k1x1_g1
        # fixed/promoted above: b1_c256_h28_w28_oc32_wic256_k1x1_g1
        *square_shapes((
            (256, 14, 512, 1, 1),
            (384, 14, 96, 1, 1),
            (480, 14, 96, 1, 1),
            (480, 14, 16, 1, 1),
            (512, 14, 112, 1, 1),
            (512, 14, 24, 1, 1),
            (512, 14, 32, 1, 1),
            (512, 14, 512, 1, 1),
            (512, 7, 1024, 1, 1),
            (528, 14, 256, 1, 1),
            (528, 14, 160, 1, 1),
            (528, 14, 32, 1, 1),
            (528, 14, 128, 1, 1),
            (576, 14, 96, 1, 1),
            (832, 7, 48, 1, 1),
            (1024, 7, 1024, 1, 1),
            (1024, 1, 1001, 1, 1),
            (1280, 10, 24, 1, 1),
            (1280, 10, 546, 1, 1),
            (32, 112, 32, 3, 3, 32),
            (64, 112, 64, 3, 3, 64),
            (128, 56, 128, 3, 3, 128),
            (256, 28, 256, 3, 3, 256),
            (512, 14, 512, 3, 3, 512),
            (1024, 7, 1024, 3, 3, 1024),
            (1024, 7, 1024, 7, 7, 1024),
            (32, 112, 16, 1, 1),
            (32, 112, 64, 1, 1),
            (64, 56, 128, 1, 1),
            (128, 56, 128, 1, 1),
            (128, 28, 256, 1, 1),
            (256, 28, 256, 1, 1),
            (3, 224, 32, 3, 3),
        )),

        *pvalid_shapes((
            (96, 112, 96, 3, 3, 96),
            (144, 56, 144, 3, 3, 144),
            (192, 28, 96, 1, 1),
            (32, 14, 64, 3, 3),
        )),
        *pvalid_pointwise_shapes(528, 14, (256, 160)),
        *pvalid_shapes((
            (160, 14, 320, 3, 3),
            (160, 40, 320, 3, 3),
        )),
        pvalid_shape(528, 14, 32, 1, 1),
        pvalid_shape(32, 14, 128, 3, 3),
        pvalid_shape(528, 14, 128, 1, 1),
        *pvalid_shapes((
            (160, 7, 320, 3, 3),
            (32, 7, 128, 3, 3),
            (192, 7, 384, 3, 3),
        )),
        pvalid_shape(832, 7, 48, 1, 1),
        pvalid_shape(32, 150, 32, 3, 3, 32),
        *pvalid_pointwise_clusters((
            (32, 150, (16,)),
            (16, 150, (96,)),
        )),
        pvalid_shape(96, 150, 96, 3, 3, 96),
        pvalid_shape(96, 75, 24, 1, 1),
        pvalid_shape(144, 75, 144, 3, 3, 144),
        *pvalid_pointwise_clusters((
            (144, 75, (24,)),
            (144, 38, (32,)),
        )),
        pvalid_shape(192, 38, 192, 3, 3, 192),
        pvalid_shape(192, 38, 32, 1, 1),
        pvalid_shape(384, 19, 384, 3, 3, 384),
        *pvalid_pointwise_shapes(384, 19, (64, 96)),
        pvalid_shape(576, 19, 576, 3, 3, 576),
        *pvalid_pointwise_shapes(576, 19, (96, 12, 273)),
        pvalid_shape(960, 10, 960, 3, 3, 960),
        *pvalid_pointwise_shapes(1280, 10, (24, 546)),
        *pvalid_shapes((
            (256, 10, 512, 3, 3),
            (128, 5, 256, 3, 3),
        )),
        *pvalid_pointwise_shapes(256, 3, (24, 546, 128)),
        *pvalid_shapes((
            (128, 3, 256, 3, 3),
        )),
        *pvalid_pointwise_shapes(256, 2, (24, 546, 64)),
        *pvalid_shapes((
            (128, 1, 24, 1, 1),
            (3, 320, 32, 3, 3),
            (32, 160, 8, 1, 1),
            (8, 160, 16, 3, 3),
            (16, 160, 128, 3, 3),
            (128, 80, 16, 1, 1),
            (16, 80, 64, 3, 3),
            (64, 80, 16, 1, 1),
            (16, 80, 128, 3, 3),
            (16, 80, 128, 5, 5),
            (128, 40, 40, 1, 1),
            (40, 40, 160, 3, 3),
            (160, 40, 40, 1, 1),
            (40, 40, 320, 1, 1),
            (320, 40, 320, 3, 3, 320),
            (320, 20, 72, 1, 1),
            (72, 20, 576, 1, 1),
            (576, 20, 576, 3, 3, 576),
            (576, 20, 72, 1, 1),
            (72, 20, 288, 3, 3),
            (288, 20, 72, 1, 1),
            (576, 20, 576, 5, 5, 576),
            (576, 20, 96, 1, 1),
            (768, 20, 768, 5, 5, 768),
            (768, 20, 96, 1, 1),
            (768, 20, 768, 3, 3, 768),
            (768, 10, 120, 1, 1),
            (960, 10, 120, 1, 1),
            (480, 10, 480, 5, 5, 480),
            (480, 10, 120, 1, 1),
            (960, 10, 960, 5, 5, 960),
            (256, 10, 256, 3, 3, 256),
            (128, 3, 256, 1, 1),
            (128, 3, 128, 3, 3, 128),
            (128, 2, 256, 1, 1),
            (64, 1, 128, 1, 1),
            (96, 20, 96, 3, 3, 96),
            (384, 10, 384, 3, 3, 384),
            (512, 5, 512, 3, 3, 512),
            (256, 3, 256, 3, 3, 256),
            (96, 20, 12, 1, 1),
            (96, 20, 273, 1, 1),
            (384, 10, 546, 1, 1),
        )),
    ]

def main(argv):
    try:
        run_shape_suite(conv_shapes(), argv)
    finally:
        os.close(fd)

if __name__ == "__main__":
    main(sys.argv[1:])
