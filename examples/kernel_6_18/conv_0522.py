import os, sys, mmap, ctypes, numpy as np
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
            setattr(self, s, kw.get(s, 0 if s != "extra" else {}))
    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

class reg:
    CNA  = 0x0201; CORE = 0x0801; DPU  = 0x1001; RDMA = 0x2001
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
    RDMA_S_POINTER      = 0x5004
    RDMA_DATA_CUBE_WIDTH  = 0x500c
    RDMA_DATA_CUBE_HEIGHT = 0x5010
    RDMA_DATA_CUBE_CHANNEL= 0x5014
    RDMA_ERDMA_CFG        = 0x5034
    RDMA_SRC_BASE_ADDR    = 0x5018
    RDMA_EW_BASE_ADDR     = 0x5038
    RDMA_FEATURE_MODE_CFG = 0x5044
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
    CNA_FEATURE_DATA_ADDR  = 0x1070
    CNA_DMA_CON0           = 0x1078
    CNA_DMA_CON1           = 0x107c
    CNA_DMA_CON2           = 0x1080
    CNA_FC_DATA_SIZE0      = 0x1084
    CNA_FC_DATA_SIZE1      = 0x1088
    CNA_DCOMP_ADDR0        = 0x1110
    CNA_DCOMP_ADDR1        = 0x1114
    CORE_MISC_CFG          = 0x3010
    CORE_DATAOUT_SIZE_0    = 0x3014
    CORE_DATAOUT_SIZE_1    = 0x3018
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

def _cdma_dc_feature_input_pack_c2(in_c, groups, out_c, align_c):
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
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    align_c = 32 if _uses_pointwise_weight_atom_layout(in_c, kh, kw, groups) else _conv_align_c(in_c, groups, out_c)
    align_out_c = max(32, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_width_stride = out_h * out_w if not is_spatial else _align_up(out_h * out_w, 4)
    input_pack_c2 = _cdma_dc_feature_input_pack_c2(in_c, groups, out_c, align_c)
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
    regs = make_conv2d_regs(p,
        in_dma + desc.feature_off,
        wt_dma + desc.weight_off,
        out_dma + desc.output_off,
        weight_reuse=desc.weight_reuse,
        full_data_bank=desc.extra.get("full_data_bank", False))

    if desc.family_bits or desc.grain_bits:
        _patch_reg(regs, reg.CNA_CONV_CON2, (desc.grain_bits << 4) | desc.family_bits)
    if desc.cbuf0 or desc.cbuf1:
        _patch_reg(regs, reg.CNA_CBUF_CON0, desc.cbuf0)
        _patch_reg(regs, reg.CNA_CBUF_CON1, desc.cbuf1)
    return regs

def _patch_reg(regs, target_reg, value):
    for i, cmd in enumerate(regs):
        if (cmd & 0xFFFF) == target_reg:
            regs[i] = (cmd & 0xFFFF0000FFFF0000) | ((value & 0xFFFFFFFF) << 16) | target_reg
            return

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
    tile_in_c = k_step if is_depthwise else in_c
    tile_data_in_channel_aligned = _align_up(tile_in_c, p["align_c"])
    row_bytes = p["width_stride"] * tile_data_in_channel_aligned * FP16_BYTES

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
    return p, split_method, tiles, y_step, k_step

def _execute_conv_tiles(descs):
    for d in descs:
        _load_tile_input(d["packed_input"])
        _load_tile_weight(d["packed_weight"])
        _clear_output()
        _submit_single_tile(d["regs"])
        d["unpack_fn"]()

def _tile_to_desc(tile, split_method, p, first_tile=False):
    return TileDesc(
        input_y=tile["y_start"],
        input_h=tile["y_step"],
        output_y=tile["y_start"],
        output_h=tile["y_step"],
        output_w=p["out_w"],
        oc_start=tile["k_start"],
        oc_count=tile["k_step"],
        feature_off=0,
        weight_off=0,
        output_off=0,
        weight_reuse=not first_tile,
        pc_core=0,
        extra={"full_data_bank": split_method != 0},
    )

def _submit_single_tile(task_regs):
    write_regs_to_npu_task(task_regs)
    if npu_submit(task_count=len(task_regs)) < 0:
        raise RuntimeError("npu_submit failed")

def _load_tile_input(packed_input):
    inp = np.ascontiguousarray(packed_input.view(np.uint16))
    assert inp.nbytes <= input_mem_create.size
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), inp.ctypes.data, inp.nbytes)

def _load_tile_weight(packed_weight):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), 0, weight_mem_create.size)
    if packed_weight is not None:
        wt_bytes = np.ascontiguousarray(packed_weight.view(np.uint16))
        assert wt_bytes.nbytes <= weight_mem_create.size
        ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), wt_bytes.ctypes.data, wt_bytes.nbytes)

def _clear_output():
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)

def _read_tile_output(count, offset=0):
    return np.frombuffer(output_map, dtype=np.uint16, count=count, offset=offset).copy().view(np.float16)

def write_regs_to_npu_task(task_regs):
    def enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len):
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
            E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(next_task_regs_len, 2) + 1),
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
        tails = enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len)
        for i, qword in enumerate(tails):
            npu_regcmd[base + len(regs) + i] = qword
        npu_tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)
        npu_tasks[idx].regcfg_amount = len(regs) + len(tails)
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

def run_conv(batch, in_c, out_c, kh, kw, input_hw, groups=1, stride=1):
    in_h, in_w = input_hw
    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, in_c // groups, kh, kw)).astype(np.float16)

    p, split_method, tiles, y_step, k_step = _plan_conv_tiles(in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    out_h, out_w = p["out_h"], p["out_w"]
    is_spatial, is_depthwise = p["is_spatial"], p["is_depthwise"]
    align_c = p["align_c"]
    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)

    def _get_input_tile(inp, tile):
        ys, y_span = tile["y_start"], tile["y_step"]
        ks, k_span = tile["k_start"], tile["k_step"]
        if is_depthwise:
            tn = min((y_span - 1) * stride + kh, in_h - ys * stride)
            ti = np.zeros((k_span, tn, in_w), dtype=np.float16)
            rih = min(tn, in_h - ys * stride)
            ti[:, :rih] = inp[ks:ks + k_span, ys * stride:ys * stride + rih, :]
            return ti, tn
        t_in = (y_span - 1) * stride + kh
        hw_th = max(t_in, y_span, 7)
        tn = min(t_in, in_h - ys * stride)
        ti = np.zeros((in_c, hw_th, in_w), dtype=np.float16)
        rih = min(tn, in_h - ys * stride)
        ti[:, :rih] = inp[:, ys * stride:ys * stride + rih, :]
        return ti, tn

    def _get_weight_tile(wt, tile):
        ks, k_span = tile["k_start"], tile["k_step"]
        if is_depthwise:
            tw = np.zeros((k_span, k_span, kh, kw), dtype=np.float16)
            for i in range(k_span):
                tw[i, i] = wt[ks + i, 0]
            return tw
        tw = wt[ks:ks + k_span].reshape(k_span, in_c // groups, kh, kw)
        if groups > 1:
            tw = _expand_grouped_weights(tw, in_c, k_span, kh, kw, groups)
        return tw

    use_pointwise_oc_schedule = (not is_spatial and not is_depthwise and groups == 1 and
                                  _needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups))

    grouped_serial = is_spatial and groups > 1 and not is_depthwise
    spatial_output_bytes = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2 * FP16_BYTES
    spatial_weight_banks = _mesa_weight_banks(kw, kh, in_c, out_c, False) if is_spatial else 0
    spatial_im2col = is_spatial and groups == 1 and not is_depthwise and in_c >= 40 and spatial_weight_banks > RK_CBUF_BANKS // 3
    spatial_oc_serial = is_spatial and groups == 1 and not is_depthwise and out_c > UNPACK_C2 and (in_c % 16 != 0 or in_c >= 16)
    depthwise_spatial_tiled = is_depthwise and is_spatial

    if grouped_serial:
        input_per_group = in_c // groups
        out_per_group = out_c // groups
        gp = _conv_params(input_per_group, in_h, in_w, out_per_group, kh, kw, 1, stride=stride)
        group_out_c1 = _ceil_div(gp["align_out_c"], UNPACK_C2)
        group_out_count = group_out_c1 * gp["out_h"] * gp["out_w"] * UNPACK_C2
        for n in range(batch):
            descs = []
            for g in range(groups):
                input_start = g * input_per_group
                input_end = input_start + input_per_group
                oc_start = g * out_per_group
                oc_end = oc_start + out_per_group
                input_tile = input_nchw[n, input_start:input_end]
                packed_input = _pack_cdma_dc_feature_input_fp16(input_tile, gp)
                group_weight = weight_nchw[oc_start:oc_end].reshape(out_per_group, input_per_group, kh, kw)
                packed_weight = pack_weights(group_weight, out_per_group, input_per_group, kh, kw, gp["align_c"], 1)
                task_regs = [make_conv2d_regs(gp, input_mem_create.dma_addr,
                                             weight_mem_create.dma_addr,
                                             output_mem_create.dma_addr)]
                _os, _oe = oc_start, oc_end
                descs.append({"packed_input": packed_input, "packed_weight": packed_weight,
                              "regs": task_regs,
                              "unpack_fn": lambda _os=_os, _oe=_oe, _gc=group_out_count: (
                                  result.__setitem__((n, slice(_os, _oe)),
                                      _unpack_nc1hwc2_output(_read_tile_output(_gc),
                                          _oe - _os, out_h, out_w, UNPACK_C2))
                              )})
            _execute_conv_tiles(descs)
        return result, input_nchw, weight_nchw

    if spatial_im2col:
        flat_c = in_c * kh * kw
        oc_tile = _pointwise_oc_tile_c(flat_c)
        row_tile_h = _pointwise_oc_tile_h(flat_c, out_h, out_w, oc_tile)
        for n in range(batch):
            descs = []
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                hw_tile_h = max(7, tile_out_h)
                im2col = np.zeros((flat_c, hw_tile_h, out_w), dtype=np.float16)
                flat = 0
                for ic in range(in_c):
                    for ky in range(kh):
                        src_y = out_row_start * stride + ky
                        for kx in range(kw):
                            im2col[flat, :tile_out_h] = input_nchw[n, ic, src_y:src_y + tile_out_h * stride:stride, kx:kx + out_w * stride:stride]
                            flat += 1
                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    hw_out_c = oc_tile if tile_out_c < oc_tile else tile_out_c
                    tile_p = _conv_params(flat_c, hw_tile_h, out_w, hw_out_c, 1, 1, 1)
                    packed_input = _pack_cdma_dc_feature_input_fp16(im2col, tile_p)

                    tile_weight = np.zeros((hw_out_c, flat_c, 1, 1), dtype=np.float16)
                    for local_oc, oc in enumerate(range(oc_start, oc_end)):
                        flat = 0
                        for ic in range(in_c):
                            for ky in range(kh):
                                for kx in range(kw):
                                    tile_weight[local_oc, flat, 0, 0] = weight_nchw[oc, ic, ky, kx]
                                    flat += 1
                    packed_weight = pack_weights(tile_weight, hw_out_c, flat_c, 1, 1, tile_p["align_c"], 1)
                    task_regs = [make_conv2d_regs(tile_p, input_mem_create.dma_addr,
                                                  weight_mem_create.dma_addr,
                                                  output_mem_create.dma_addr,
                                                  full_data_bank=True)]
                    out_c1 = _ceil_div(tile_p["align_out_c"], UNPACK_C2)
                    out_count = out_c1 * tile_p["out_width_stride"] * UNPACK_C2
                    _os, _oe, _tc = oc_start, oc_end, tile_out_c
                    _ys, _yh, _hwh = out_row_start, tile_out_h, hw_tile_h
                    descs.append({"packed_input": packed_input, "packed_weight": packed_weight,
                                  "regs": task_regs,
                                  "unpack_fn": lambda _os=_os, _oe=_oe, _tc=_tc, _ys=_ys, _yh=_yh, _hwh=_hwh, _oc=out_count, _tp=tile_p: (
                                      result.__setitem__((n, slice(_os, _oe), slice(_ys, _ys + _yh)),
                                          _unpack_flat_1x1_output(_read_tile_output(_oc),
                                              _hwh if _tc < _oe - _os else _tc, _hwh, out_w, _tp["out_width_stride"], UNPACK_C2)[:_tc, :_yh])
                                  )})
            _execute_conv_tiles(descs)
        return result, input_nchw, weight_nchw

    if spatial_oc_serial:
        oc_tile = UNPACK_C2
        for n in range(batch):
            descs = []
            for out_row_start in range(0, out_h, 32):
                tile_out_h = min(32, out_h - out_row_start)
                tile_in_h = (tile_out_h - 1) * stride + kh
                input_tile = input_nchw[n, :, out_row_start * stride:out_row_start * stride + tile_in_h, :]
                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    tp = _conv_params(in_c, tile_in_h, in_w, tile_out_c, kh, kw, 1, stride)
                    packed_input = _pack_cdma_dc_feature_input_fp16(input_tile, tp)
                    tile_weight = np.zeros((tile_out_c, in_c, kh, kw), dtype=np.float16)
                    tile_weight[:] = weight_nchw[oc_start:oc_end].reshape(tile_out_c, in_c, kh, kw)
                    packed_weight = pack_weights(tile_weight, tile_out_c, in_c, kh, kw, tp["align_c"], 1)
                    regs = [make_conv2d_regs(tp, input_mem_create.dma_addr, weight_mem_create.dma_addr,
                                              output_mem_create.dma_addr)]
                    _rs, _re = oc_start, oc_end
                    _ys, _yh = out_row_start, tile_out_h
                    out_c1 = _ceil_div(tp["align_out_c"], 8)
                    out_count = out_c1 * tp["out_width_stride"] * 8
                    descs.append({"packed_input": packed_input, "packed_weight": packed_weight,
                                  "regs": regs,
                                  "unpack_fn": lambda _rs=_rs, _re=_re, _ys=_ys, _yh=_yh, _oc=out_count, _tp=tp: (
                                      result.__setitem__((n, slice(_rs, _re), slice(_ys, _ys + _yh)),
                                          _unpack_nc1hwc2_output(_read_tile_output(_oc),
                                              _re - _rs, _yh, out_w, 8, _tp["out_width_stride"]))
                                  )})
            _execute_conv_tiles(descs)
        return result, input_nchw, weight_nchw

    if depthwise_spatial_tiled:
        channel_tile = min(32, out_c)
        row_tile_h = max(1, _depthwise_tile_h(out_c, out_h, in_w, kh, kw, stride)) if is_spatial else out_h
        for n in range(batch):
            descs = []
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                hw_tile_out_h = max(2, tile_out_h)
                tile_in_h = (hw_tile_out_h - 1) * stride + kh
                for ch_start in range(0, out_c, channel_tile):
                    ch_end = min(ch_start + channel_tile, out_c)
                    tile_c = ch_end - ch_start
                    tile_p = _conv_params(tile_c, tile_in_h, in_w, tile_c, kh, kw, tile_c, stride=stride)
                    input_tile = np.zeros((tile_c, tile_in_h, in_w), dtype=np.float16)
                    real_in_h = min(tile_in_h, in_h - out_row_start * stride)
                    input_tile[:, :real_in_h] = input_nchw[n, ch_start:ch_end, out_row_start * stride:out_row_start * stride + real_in_h, :]
                    packed_input = _pack_cdma_dc_feature_input_fp16(input_tile, tile_p)
                    tile_weight = np.zeros((tile_c, tile_c, kh, kw), dtype=np.float16)
                    for local_c in range(tile_c):
                        tile_weight[local_c, local_c] = weight_nchw[ch_start + local_c, 0]
                    packed_weight = pack_weights(tile_weight, tile_c, tile_c, kh, kw, tile_p["align_c"], tile_c)
                    regs = [make_conv2d_regs(tile_p, input_mem_create.dma_addr, weight_mem_create.dma_addr,
                                             output_mem_create.dma_addr)]
                    out_c1 = _ceil_div(tile_p["align_out_c"], 8)
                    out_count = out_c1 * tile_p["out_width_stride"] * 8
                    _cs, _ce = ch_start, ch_end
                    _ys, _yh, _hwh = out_row_start, tile_out_h, hw_tile_out_h
                    descs.append({"packed_input": packed_input, "packed_weight": packed_weight,
                                  "regs": regs,
                                  "unpack_fn": lambda _cs=_cs, _ce=_ce, _ys=_ys, _yh=_yh, _hwh=_hwh, _oc=out_count, _tp=tile_p: (
                                      result.__setitem__((n, slice(_cs, _ce), slice(_ys, _ys + _yh)),
                                          _unpack_nc1hwc2_output(_read_tile_output(_oc),
                                              _ce - _cs, _hwh, out_w, 8, _tp["out_width_stride"])[:, :_yh])
                                  )})
            _execute_conv_tiles(descs)
        return result, input_nchw, weight_nchw

    for n in range(batch):
        descs = []
        out_offset = 0
        if use_pointwise_oc_schedule:
            oc_tile = _pointwise_oc_tile_c(in_c)
            row_tile_h = _pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile, stride)
            tw = weight_nchw.reshape(out_c, in_c // groups, kh, kw)
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                hw_tile_h = max(7, tile_out_h)
                ti = np.zeros((in_c, hw_tile_h, in_w), dtype=np.float16)
                ti[:, :tile_out_h] = input_nchw[n, :, out_row_start * stride:out_row_start * stride + tile_out_h, :]
                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    hw_out_c = oc_tile if tile_out_c < oc_tile else tile_out_c
                    tp = _conv_params(in_c, hw_tile_h, in_w, hw_out_c, kh, kw, groups, stride)
                    tile_wt = tw[oc_start:oc_end]
                    if hw_out_c > tile_out_c:
                        padded = np.zeros((hw_out_c,) + tile_wt.shape[1:], dtype=np.float16)
                        padded[:tile_out_c] = tile_wt
                        tile_wt = padded
                    packed_weight = pack_weights(tile_wt, hw_out_c, in_c, kh, kw, tp["align_c"], groups)
                    regs = [make_conv2d_regs(tp, input_mem_create.dma_addr, weight_mem_create.dma_addr,
                                             output_mem_create.dma_addr, full_data_bank=True)]
                    packed_input = _pack_cdma_dc_feature_input_fp16(ti, tp)
                    out_c1 = _ceil_div(tp["align_out_c"], 8)
                    out_count = out_c1 * tp["out_width_stride"] * 8
                    _os, _oe, _tc = oc_start, oc_end, tile_out_c
                    _ys, _yh, _hwh = out_row_start, tile_out_h, hw_tile_h
                    descs.append({"packed_input": packed_input, "packed_weight": packed_weight,
                                  "regs": regs,
                                  "unpack_fn": lambda _os=_os, _oe=_oe, _tc=_tc, _ys=_ys, _yh=_yh, _hwh=_hwh, _oc=out_count, _tp=tp: (
                                      result.__setitem__((n, slice(_os, _oe), slice(_ys, _ys + _yh)),
                                          _unpack_flat_1x1_output(_read_tile_output(_oc),
                                              _hwh if _tc < _oe - _os else _tc, _hwh, out_w, _tp["out_width_stride"], 8)[:_tc, :_yh])
                                  )})
            _execute_conv_tiles(descs)
            continue
        for tile in tiles:
            ks, k_span = tile["k_start"], tile["k_step"]
            ti, tn_actual = _get_input_tile(input_nchw[n], tile)
            tw = _get_weight_tile(weight_nchw, tile)

            tile_out_c = k_span
            if is_depthwise:
                tile_ic, tile_g = k_span, k_span
                hw_oc = tile_out_c
            else:
                tile_ic, tile_g = in_c, groups
                oc_tile_ = _pointwise_oc_tile_c(in_c)
                hw_oc = oc_tile_ if tile_out_c < oc_tile_ else tile_out_c
                if split_method == _SPLIT_BY_K and not is_spatial and hw_oc < tile_out_c:
                    hw_oc = _align_up(tile_out_c, 16)

            if hw_oc > tile_out_c:
                padded = np.zeros((hw_oc,) + tw.shape[1:], dtype=np.float16)
                padded[:tile_out_c] = tw
                tw = padded

            tp = _conv_params(tile_ic, tn_actual, in_w, hw_oc, kh, kw, tile_g, stride)
            packed_weight = pack_weights(tw, hw_oc, tile_ic, kh, kw, tp["align_c"], tile_g)

            fdb = split_method != _SPLIT_NONE or (is_spatial and groups == 1)
            out_dma = output_mem_create.dma_addr + out_offset
            regs = [make_conv2d_regs(tp, input_mem_create.dma_addr, weight_mem_create.dma_addr,
                                     out_dma, full_data_bank=fdb)]

            packed_input = _pack_cdma_dc_feature_input_fp16(ti, tp)
            out_c1 = _ceil_div(tp["align_out_c"], 8)
            out_count = out_c1 * tp["out_width_stride"] * 8
            _ks, _toc = ks, tile_out_c
            _ys, _yh = tile["y_start"], tile["y_step"]
            _off = out_offset
            descs.append({"packed_input": packed_input, "packed_weight": packed_weight,
                          "regs": regs,
                          "unpack_fn": lambda _ks=_ks, _toc=_toc, _ys=_ys, _yh=_yh, _off=_off, _oc=out_count, _tp=tp: (
                              result.__setitem__((n, slice(_ks, _ks + _toc), slice(_ys, _ys + _yh)),
                                  _unpack_nc1hwc2_output(
                                      np.frombuffer(output_map, dtype=np.uint16, count=_oc, offset=_off).copy().view(np.float16),
                                      _toc, _yh, out_w, 8, _tp["out_width_stride"]))
                          )})
            out_offset += tp["align_out_c"] * tp["out_width_stride"] * FP16_BYTES
        _execute_conv_tiles(descs)

    return result, input_nchw, weight_nchw

def compute_expected_nchw(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups=1, stride=1):
    out_h, out_w = (in_h - kh) // stride + 1, (in_w - kw) // stride + 1
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

if __name__ == "__main__":
    shapes = [
        # -- 1x1 kernels (fully supported via NHWC mode + channel slicing for ic>=5) --
        dict(name="conv2d_1x6_1x1_4x4",                batch=1, in_c=1,  in_h=4,  in_w=4,  out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
        dict(name="conv2d_3x3_1x1_4x4",                batch=1, in_c=3,  in_h=4,  in_w=4,  out_c=3, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="conv2d_4x2_1x1_4x4",                batch=1, in_c=4,  in_h=4,  in_w=4,  out_c=2, weight_in_c=4, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c4_h9_w9_oc4_wic4_k1x1_g1",  batch=1, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c3_h52_w52_oc6_wic3_k1x1_g1", batch=1, in_c=3,  in_h=52, in_w=52, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1", batch=1, in_c=96, in_h=56, in_w=56, out_c=24, weight_in_c=96, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1", batch=1, in_c=144, in_h=56, in_w=56, out_c=24, weight_in_c=144, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1", batch=1, in_c=144, in_h=28, in_w=28, out_c=32, weight_in_c=144, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1", batch=1, in_c=192, in_h=28, in_w=28, out_c=32, weight_in_c=192, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1", batch=1, in_c=192, in_h=28, in_w=28, out_c=16, weight_in_c=192, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1", batch=1, in_c=256, in_h=28, in_w=28, out_c=32, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="conv2d_16x16_1x1_8x8",              batch=1, in_c=16, in_h=8,  in_w=8,  out_c=16, weight_in_c=16, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c16_h32_w32_oc16_wic16_k1x1_g1", batch=1, in_c=16, in_h=32, in_w=32, out_c=16, weight_in_c=16, kh=1, kw=1, groups=1),

        # -- Non-1x1 kernels (partial output -- known NPU hardware limitation) --
        dict(name="conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1",  batch=1, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=3, kw=3, groups=1),
        dict(name="conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1", batch=1, in_c=16, in_h=18, in_w=18, out_c=16, weight_in_c=16, kh=3, kw=3, groups=1),
        dict(name="conv2d_b2_c4_h9_w9_oc4_wic4_k3x3_g1",  batch=2, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=3, kw=3, groups=1),
        dict(name="conv2d_b1_c1_h5_w7_oc6_wic1_k3x3_g1",  batch=1, in_c=1,  in_h=5,  in_w=7,  out_c=6, weight_in_c=1, kh=3, kw=3, groups=1),

        # Depthwise
        dict(name="conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3", batch=1, in_c=3, in_h=11, in_w=28, out_c=3, weight_in_c=1, kh=3, kw=3, groups=3),

        # Non-square kernels
        dict(name="conv2d_3x6_1x3_5x5", batch=1, in_c=3, in_h=5, in_w=5, out_c=6, weight_in_c=3, kh=1, kw=3, groups=1),

        # -- test_ops.py _test_conv2d(cin=3): (3,5,7) @ (6,kh,kw) --
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=3, groups=3),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x1_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=1, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x3_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=3, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x5_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=5, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x1_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=1, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x3_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=3, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x5_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=5, groups=1),

        # -- test_ops.py _test_conv2d(cin=1): (1,5,7) @ (6,kh,kw) --
        dict(name="conv2d_1x6_2x1_5x7", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=2, kw=1, groups=1),
        dict(name="conv2d_1x6_2x3_5x7", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=2, kw=3, groups=1),
        dict(name="conv2d_1x6_3x1_5x7_b", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=1, groups=1),
        dict(name="conv2d_1x6_3x5_5x7",   batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=5, groups=1),

        # -- Grouped convs from test_ops.py --
        dict(name="conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2",     batch=1,  in_c=4,  in_h=1,  in_w=1,  out_c=2,  weight_in_c=2,  kh=1, kw=1, groups=2),
        dict(name="conv2d_4x4_1x1_1x1_g2",                   batch=1,  in_c=4,  in_h=1,  in_w=1,  out_c=4,  weight_in_c=2,  kh=1, kw=1, groups=2),
        dict(name="conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32", batch=1, in_c=32, in_h=32, in_w=32, out_c=32, weight_in_c=1,  kh=1, kw=1, groups=32),
        dict(name="conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5",   batch=1,  in_c=15, in_h=5,  in_w=5,  out_c=35, weight_in_c=3,  kh=3, kw=3, groups=5),

        # -- Batch >1 coverage --
        dict(name="conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3", batch=2, in_c=3,  in_h=11, in_w=28, out_c=3,  weight_in_c=1, kh=3, kw=3, groups=3),
        dict(name="conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5", batch=4, in_c=15, in_h=5,  in_w=5,  out_c=35, weight_in_c=3, kh=3, kw=3, groups=5),

        # -- Grouped output-channel variants --
        dict(name="conv2d_b1_c4_h5_w5_oc4_wic2_k3x3_g2",   batch=1, in_c=4,  in_h=5, in_w=5, out_c=4,  weight_in_c=2, kh=3, kw=3, groups=2),
        dict(name="conv2d_b1_c4_h5_w5_oc8_wic2_k3x3_g2",   batch=1, in_c=4,  in_h=5, in_w=5, out_c=8,  weight_in_c=2, kh=3, kw=3, groups=2),
        dict(name="conv2d_b1_c4_h5_w5_oc12_wic2_k3x3_g2",  batch=1, in_c=4,  in_h=5, in_w=5, out_c=12, weight_in_c=2, kh=3, kw=3, groups=2),
        dict(name="conv2d_b1_c6_h5_w5_oc6_wic2_k3x3_g3",   batch=1, in_c=6,  in_h=5, in_w=5, out_c=6,  weight_in_c=2, kh=3, kw=3, groups=3),
        dict(name="conv2d_b1_c6_h5_w5_oc12_wic2_k3x3_g3",  batch=1, in_c=6,  in_h=5, in_w=5, out_c=12, weight_in_c=2, kh=3, kw=3, groups=3),
        dict(name="conv2d_b1_c6_h5_w5_oc18_wic2_k3x3_g3",  batch=1, in_c=6,  in_h=5, in_w=5, out_c=18, weight_in_c=2, kh=3, kw=3, groups=3),
        dict(name="conv2d_b1_c15_h5_w5_oc20_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=20, weight_in_c=3, kh=3, kw=3, groups=5),
        dict(name="conv2d_b1_c15_h5_w5_oc25_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=25, weight_in_c=3, kh=3, kw=3, groups=5),
        dict(name="conv2d_b1_c15_h5_w5_oc30_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=30, weight_in_c=3, kh=3, kw=3, groups=5),
        dict(name="conv2d_b1_c15_h5_w5_oc40_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=40, weight_in_c=3, kh=3, kw=3, groups=5),
        dict(name="conv2d_2x2_1x1_4x4",  batch=1, in_c=2, in_h=4, in_w=4, out_c=2, weight_in_c=2, kh=1, kw=1, groups=1),
        dict(name="conv2d_8x8_1x1_5x5",       batch=1, in_c=8,  in_h=5,  in_w=5,  out_c=8,  weight_in_c=8,  kh=1, kw=1, groups=1),
        dict(name="conv2d_10x20_3x3_9x9",     batch=1, in_c=10, in_h=9,  in_w=9,  out_c=20, weight_in_c=10, kh=3, kw=3, groups=1),
        dict(name="conv2d_16x16_3x3_9x9",     batch=1, in_c=16, in_h=9,  in_w=9,  out_c=16, weight_in_c=16, kh=3, kw=3, groups=1),
        dict(name="conv2d_2x4_3x3_6x6",       batch=1, in_c=2,  in_h=6,  in_w=6,  out_c=4,  weight_in_c=2,  kh=3, kw=3, groups=1),
        dict(name="conv2d_2x4_2x2_5x5",       batch=1, in_c=2,  in_h=5,  in_w=5,  out_c=4,  weight_in_c=2,  kh=2, kw=2, groups=1),
        dict(name="conv2d_1x32_5x5_10x10",    batch=1, in_c=1,  in_h=10, in_w=10, out_c=32, weight_in_c=1,  kh=5, kw=5, groups=1),
        dict(name="conv2d_8x4_4x4_10x10",     batch=1, in_c=8,  in_h=10, in_w=10, out_c=4,  weight_in_c=8,  kh=4, kw=4, groups=1),

        # MobileNet layers
        # first spatial conv (RGB->32ch)
        dict(name="conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1", batch=1, in_c=3, in_h=224, in_w=224, out_c=32, weight_in_c=3, kh=3, kw=3, groups=1),
        #  depthwise conv (3x3 sep)
        dict(name="conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32", batch=1, in_c=32, in_h=112, in_w=112, out_c=32, weight_in_c=1, kh=3, kw=3, groups=32),
        # pointwise (1x1 projection)
        dict(name="conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1", batch=1, in_c=32, in_h=112, in_w=112, out_c=64, weight_in_c=32, kh=1, kw=1, groups=1),
        # depthwise conv (64ch)
        dict(name="conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64", batch=1, in_c=64, in_h=112, in_w=112, out_c=64, weight_in_c=1, kh=3, kw=3, groups=64),
        # pointwise expansion
        dict(name="conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1", batch=1, in_c=64, in_h=56, in_w=56, out_c=128, weight_in_c=64, kh=1, kw=1, groups=1),
        # depthwise conv (128ch)
        dict(name="conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128", batch=1, in_c=128, in_h=56, in_w=56, out_c=128, weight_in_c=1, kh=3, kw=3, groups=128),
        # pointwise (1x1 projection, same-channel)
        dict(name="conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1", batch=1, in_c=128, in_h=56, in_w=56, out_c=128, weight_in_c=128, kh=1, kw=1, groups=1),
        # Pointwise 128->256
        dict(name="conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1", batch=1, in_c=128, in_h=28, in_w=28, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
        # Depthwise 256x28
        dict(name="conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256", batch=1, in_c=256, in_h=28, in_w=28, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
        # Pointwise 256->256
        dict(name="conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1", batch=1, in_c=256, in_h=28, in_w=28, out_c=256, weight_in_c=256, kh=1, kw=1, groups=1),
        # Pointwise 256->512
        dict(name="conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1", batch=1, in_c=256, in_h=14, in_w=14, out_c=512, weight_in_c=256, kh=1, kw=1, groups=1),
        # Depthwise 512x14
        dict(name="conv2d_cc_b1_c512_h14_w14_oc512_wic1_k3x3_g512", batch=1, in_c=512, in_h=14, in_w=14, out_c=512, weight_in_c=1, kh=3, kw=3, groups=512),
        # Pointwise 512->512
        dict(name="conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1", batch=1, in_c=512, in_h=14, in_w=14, out_c=512, weight_in_c=512, kh=1, kw=1, groups=1),
        # 7x7 and classifier-head fixes
        dict(name="conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1", batch=1, in_c=512, in_h=7, in_w=7, out_c=1024, weight_in_c=512, kh=1, kw=1, groups=1),
        dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1, kh=3, kw=3, groups=1024),
        dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1024, kh=1, kw=1, groups=1),
        dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1, kh=7, kw=7, groups=1024),
        dict(name="conv2d_cc_b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=1, in_w=1, out_c=1001, weight_in_c=1024, kh=1, kw=1, groups=1),

        # conv1d expressed as conv2d: input H=1, W=input_size; kernel H=1, W=kernel_size.
        dict(name="conv1d_bs1_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
        dict(name="conv1d_bs8_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
        dict(name="conv1d_bs1_612_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
        dict(name="conv1d_bs1_615_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=1),
        dict(name="conv1d_bs1_1311_631_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="conv1d_bs1_1311_632_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=2, groups=1),
        dict(name="conv1d_bs1_1311_635_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=5, groups=1),
        dict(name="conv1d_bs1_1311_615_g3_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=3),
        dict(name="conv1d_bs8_8111_611_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
        dict(name="conv1d_bs8_8111_612_a_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
        dict(name="conv1d_bs8_8111_612_b_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
        dict(name="conv1d_bs8_8111_615_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=1),
        dict(name="conv1d_bs8_8311_631_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="conv1d_bs8_8311_632_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=2, groups=1),
        dict(name="conv1d_bs8_8311_635_a_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=5, groups=1),
        dict(name="conv1d_bs8_8311_635_g3_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=3),

        # -- Large spatial 1x1 conv where IC=3, OC=6 (fixed; promoted above) --
        dict(name="1x3_54x54_k1",  batch=1, in_c=3, in_h=54, in_w=54, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_56x56_k1",  batch=1, in_c=3, in_h=56, in_w=56, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_58x58_k1",  batch=1, in_c=3, in_h=58, in_w=58, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_60x60_k1",  batch=1, in_c=3, in_h=60, in_w=60, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_62x62_k1",  batch=1, in_c=3, in_h=62, in_w=62, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_64x64_k1",  batch=1, in_c=3, in_h=64, in_w=64, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_66x66_k1",  batch=1, in_c=3, in_h=66, in_w=66, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_68x68_k1",  batch=1, in_c=3, in_h=68, in_w=68, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_70x70_k1",  batch=1, in_c=3, in_h=70, in_w=70, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="1x3_72x72_k1",  batch=1, in_c=3, in_h=72, in_w=72, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),

        # -- 1x1 conv large input channel >> output channel (wrong numerical result) --
        # From mobilenetv2 depthwise expand/project layers
        # fixed/promoted above: b1_c96_h56_w56_oc24_wic96_k1x1_g1
        # fixed/promoted above: b1_c144_h56_w56_oc24_wic144_k1x1_g1
        # fixed/promoted above: b1_c144_h28_w28_oc32_wic144_k1x1_g1
        # fixed/promoted above: b1_c192_h28_w28_oc32_wic192_k1x1_g1
        # fixed/promoted above: b1_c192_h28_w28_oc16_wic192_k1x1_g1
        # fixed/promoted above: b1_c256_h28_w28_oc32_wic256_k1x1_g1
        dict(name="b1_c256_h14_w14_oc512_wic256_k1x1_g1",  batch=1, in_c=256, in_h=14,  in_w=14,  out_c=512, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="b1_c384_h14_w14_oc96_wic384_k1x1_g1",   batch=1, in_c=384, in_h=14,  in_w=14,  out_c=96,  weight_in_c=384, kh=1, kw=1, groups=1),
        dict(name="b1_c480_h14_w14_oc96_wic480_k1x1_g1",   batch=1, in_c=480, in_h=14,  in_w=14,  out_c=96,  weight_in_c=480, kh=1, kw=1, groups=1),
        dict(name="b1_c480_h14_w14_oc16_wic480_k1x1_g1",   batch=1, in_c=480, in_h=14,  in_w=14,  out_c=16,  weight_in_c=480, kh=1, kw=1, groups=1),
        dict(name="b1_c512_h14_w14_oc112_wic512_k1x1_g1",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=112, weight_in_c=512, kh=1, kw=1, groups=1),
        dict(name="b1_c512_h14_w14_oc24_wic512_k1x1_g1",   batch=1, in_c=512, in_h=14,  in_w=14,  out_c=24,  weight_in_c=512, kh=1, kw=1, groups=1),
        dict(name="b1_c512_h14_w14_oc32_wic512_k1x1_g1",   batch=1, in_c=512, in_h=14,  in_w=14,  out_c=32,  weight_in_c=512, kh=1, kw=1, groups=1),
        dict(name="b1_c512_h14_w14_oc512_wic512_k1x1_g1",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=512, weight_in_c=512, kh=1, kw=1, groups=1),
        dict(name="b1_c512_h7_w7_oc1024_wic512_k1x1_g1",   batch=1, in_c=512, in_h=7,   in_w=7,   out_c=1024, weight_in_c=512, kh=1, kw=1, groups=1),
        dict(name="b1_c528_h14_w14_oc256_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=256, weight_in_c=528, kh=1, kw=1, groups=1),
        dict(name="b1_c528_h14_w14_oc160_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=160, weight_in_c=528, kh=1, kw=1, groups=1),
        dict(name="b1_c528_h14_w14_oc32_wic528_k1x1_g1",   batch=1, in_c=528, in_h=14,  in_w=14,  out_c=32,  weight_in_c=528, kh=1, kw=1, groups=1),
        dict(name="b1_c528_h14_w14_oc128_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=128, weight_in_c=528, kh=1, kw=1, groups=1),
        dict(name="b1_c576_h14_w14_oc96_wic576_k1x1_g1",   batch=1, in_c=576, in_h=14,  in_w=14,  out_c=96,  weight_in_c=576, kh=1, kw=1, groups=1),
        dict(name="b1_c832_h7_w7_oc48_wic832_k1x1_g1",     batch=1, in_c=832, in_h=7,   in_w=7,   out_c=48,  weight_in_c=832, kh=1, kw=1, groups=1),
        dict(name="b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1024, kh=1, kw=1, groups=1),
        dict(name="b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=1,  in_w=1,   out_c=1001, weight_in_c=1024, kh=1, kw=1, groups=1),
        dict(name="b1_c1280_h10_w10_oc24_wic1280_k1x1_g1", batch=1, in_c=1280, in_h=10, in_w=10,  out_c=24,  weight_in_c=1280, kh=1, kw=1, groups=1),
        dict(name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1", batch=1, in_c=1280, in_h=10, in_w=10, out_c=546, weight_in_c=1280, kh=1, kw=1, groups=1),

        dict(name="b1_c32_h112_w112_oc32_wic1_k3x3_g32",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=32,  weight_in_c=1,  kh=3, kw=3, groups=32),
        dict(name="b1_c64_h112_w112_oc64_wic1_k3x3_g64",   batch=1, in_c=64,  in_h=112, in_w=112, out_c=64,  weight_in_c=1,  kh=3, kw=3, groups=64),
        dict(name="b1_c128_h56_w56_oc128_wic1_k3x3_g128",  batch=1, in_c=128, in_h=56,  in_w=56,  out_c=128, weight_in_c=1,  kh=3, kw=3, groups=128),
        dict(name="b1_c256_h28_w28_oc256_wic1_k3x3_g256",  batch=1, in_c=256, in_h=28,  in_w=28,  out_c=256, weight_in_c=1,  kh=3, kw=3, groups=256),
        dict(name="b1_c512_h14_w14_oc512_wic1_k3x3_g512",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=512, weight_in_c=1,  kh=3, kw=3, groups=512),
        dict(name="b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1,  kh=3, kw=3, groups=1024),
        dict(name="b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1,  kh=7, kw=7, groups=1024),

        dict(name="b1_c32_h112_w112_oc16_wic32_k1x1_g1",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=16,  weight_in_c=32,  kh=1, kw=1, groups=1),
        dict(name="b1_c32_h112_w112_oc64_wic32_k1x1_g1",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=64,  weight_in_c=32,  kh=1, kw=1, groups=1),
        dict(name="b1_c64_h56_w56_oc128_wic64_k1x1_g1",    batch=1, in_c=64,  in_h=56,  in_w=56,  out_c=128, weight_in_c=64,  kh=1, kw=1, groups=1),
        dict(name="b1_c128_h56_w56_oc128_wic128_k1x1_g1",  batch=1, in_c=128, in_h=56,  in_w=56,  out_c=128, weight_in_c=128, kh=1, kw=1, groups=1),
        dict(name="b1_c128_h28_w28_oc256_wic128_k1x1_g1",  batch=1, in_c=128, in_h=28,  in_w=28,  out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
        dict(name="b1_c256_h28_w28_oc256_wic256_k1x1_g1",  batch=1, in_c=256, in_h=28,  in_w=28,  out_c=256, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="b1_c3_h224_w224_oc32_wic3_k3x3_g1",     batch=1, in_c=3,   in_h=224, in_w=224, out_c=32,  weight_in_c=3,   kh=3, kw=3, groups=1),

        dict(name="b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=112, in_w=112, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
        dict(name="b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid", batch=1, in_c=144, in_h=56, in_w=56, out_c=144, weight_in_c=1, kh=3, kw=3, groups=144),
        dict(name="b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid", batch=1, in_c=192, in_h=28, in_w=28, out_c=96, weight_in_c=192, kh=1, kw=1, groups=1),
        dict(name="b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=14, in_w=14, out_c=64, weight_in_c=32, kh=3, kw=3, groups=1),
        dict(name="b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=256, weight_in_c=528, kh=1, kw=1, groups=1),
        dict(name="b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=160, weight_in_c=528, kh=1, kw=1, groups=1),
        dict(name="b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid", batch=1, in_c=160, in_h=14, in_w=14, out_c=320, weight_in_c=160, kh=3, kw=3, groups=1),
        dict(name="b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=32, weight_in_c=528, kh=1, kw=1, groups=1),
        dict(name="b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=14, in_w=14, out_c=128, weight_in_c=32, kh=3, kw=3, groups=1),
        dict(name="b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=128, weight_in_c=528, kh=1, kw=1, groups=1),
        dict(name="b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid", batch=1, in_c=160, in_h=7, in_w=7, out_c=320, weight_in_c=160, kh=3, kw=3, groups=1),
        dict(name="b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=7, in_w=7, out_c=128, weight_in_c=32, kh=3, kw=3, groups=1),
        dict(name="b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid", batch=1, in_c=192, in_h=7, in_w=7, out_c=384, weight_in_c=192, kh=3, kw=3, groups=1),
        dict(name="b1_c832_h7_w7_oc48_wic832_k1x1_g1_s1_pvalid", batch=1, in_c=832, in_h=7, in_w=7, out_c=48, weight_in_c=832, kh=1, kw=1, groups=1),
        dict(name="b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid", batch=1, in_c=32, in_h=150, in_w=150, out_c=32, weight_in_c=1, kh=3, kw=3, groups=32),
        dict(name="b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid", batch=1, in_c=32, in_h=150, in_w=150, out_c=16, weight_in_c=32, kh=1, kw=1, groups=1),
        dict(name="b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid", batch=1, in_c=16, in_h=150, in_w=150, out_c=96, weight_in_c=16, kh=1, kw=1, groups=1),
        dict(name="b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=150, in_w=150, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
        dict(name="b1_c96_h75_w75_oc24_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=75, in_w=75, out_c=24, weight_in_c=96, kh=1, kw=1, groups=1),
        dict(name="b1_c144_h75_w75_oc144_wic1_k3x3_g144_s1_pvalid", batch=1, in_c=144, in_h=75, in_w=75, out_c=144, weight_in_c=1, kh=3, kw=3, groups=144),
        dict(name="b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid", batch=1, in_c=144, in_h=75, in_w=75, out_c=24, weight_in_c=144, kh=1, kw=1, groups=1),
        dict(name="b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid", batch=1, in_c=144, in_h=38, in_w=38, out_c=32, weight_in_c=144, kh=1, kw=1, groups=1),
        dict(name="b1_c192_h38_w38_oc192_wic1_k3x3_g192_s1_pvalid", batch=1, in_c=192, in_h=38, in_w=38, out_c=192, weight_in_c=1, kh=3, kw=3, groups=192),
        dict(name="b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid", batch=1, in_c=192, in_h=38, in_w=38, out_c=32, weight_in_c=192, kh=1, kw=1, groups=1),
        dict(name="b1_c384_h19_w19_oc384_wic1_k3x3_g384_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=384, weight_in_c=1, kh=3, kw=3, groups=384),
        dict(name="b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=64, weight_in_c=384, kh=1, kw=1, groups=1),
        dict(name="b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=96, weight_in_c=384, kh=1, kw=1, groups=1),
        dict(name="b1_c576_h19_w19_oc576_wic1_k3x3_g576_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=576, weight_in_c=1, kh=3, kw=3, groups=576),
        dict(name="b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=96, weight_in_c=576, kh=1, kw=1, groups=1),
        dict(name="b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=12, weight_in_c=576, kh=1, kw=1, groups=1),
        dict(name="b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=273, weight_in_c=576, kh=1, kw=1, groups=1),
        dict(name="b1_c960_h10_w10_oc960_wic1_k3x3_g960_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=960, weight_in_c=1, kh=3, kw=3, groups=960),
        dict(name="b1_c1280_h10_w10_oc24_wic1280_k1x1_g1_s1_pvalid", batch=1, in_c=1280, in_h=10, in_w=10, out_c=24, weight_in_c=1280, kh=1, kw=1, groups=1),
        dict(name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid", batch=1, in_c=1280, in_h=10, in_w=10, out_c=546, weight_in_c=1280, kh=1, kw=1, groups=1),
        dict(name="b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid", batch=1, in_c=256, in_h=10, in_w=10, out_c=512, weight_in_c=256, kh=3, kw=3, groups=1),
        dict(name="b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid", batch=1, in_c=128, in_h=5, in_w=5, out_c=256, weight_in_c=128, kh=3, kw=3, groups=1),
        dict(name="b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=24, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="b1_c256_h3_w3_oc546_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=546, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=128, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=256, weight_in_c=128, kh=3, kw=3, groups=1),
        dict(name="b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=24, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=546, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=64, weight_in_c=256, kh=1, kw=1, groups=1),
        dict(name="b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=1, in_w=1, out_c=24, weight_in_c=128, kh=1, kw=1, groups=1),
        dict(name="b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid", batch=1, in_c=3, in_h=320, in_w=320, out_c=32, weight_in_c=3, kh=3, kw=3, groups=1),
        dict(name="b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid", batch=1, in_c=32, in_h=160, in_w=160, out_c=8, weight_in_c=32, kh=1, kw=1, groups=1),
        dict(name="b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid", batch=1, in_c=8, in_h=160, in_w=160, out_c=16, weight_in_c=8, kh=3, kw=3, groups=1),
        dict(name="b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=160, in_w=160, out_c=128, weight_in_c=16, kh=3, kw=3, groups=1),
        dict(name="b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=80, in_w=80, out_c=16, weight_in_c=128, kh=1, kw=1, groups=1),
        dict(name="b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=64, weight_in_c=16, kh=3, kw=3, groups=1),
        dict(name="b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid", batch=1, in_c=64, in_h=80, in_w=80, out_c=16, weight_in_c=64, kh=1, kw=1, groups=1),
        dict(name="b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=128, weight_in_c=16, kh=3, kw=3, groups=1),
        dict(name="b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=128, weight_in_c=16, kh=5, kw=5, groups=1),
        dict(name="b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=40, in_w=40, out_c=40, weight_in_c=128, kh=1, kw=1, groups=1),
        dict(name="b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid", batch=1, in_c=40, in_h=40, in_w=40, out_c=160, weight_in_c=40, kh=3, kw=3, groups=1),
        dict(name="b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid", batch=1, in_c=160, in_h=40, in_w=40, out_c=40, weight_in_c=160, kh=1, kw=1, groups=1),
        dict(name="b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid", batch=1, in_c=40, in_h=40, in_w=40, out_c=320, weight_in_c=40, kh=1, kw=1, groups=1),
        dict(name="b1_c320_h40_w40_oc320_wic1_k3x3_g320_s1_pvalid", batch=1, in_c=320, in_h=40, in_w=40, out_c=320, weight_in_c=1, kh=3, kw=3, groups=320),
        dict(name="b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid", batch=1, in_c=320, in_h=20, in_w=20, out_c=72, weight_in_c=320, kh=1, kw=1, groups=1),
        dict(name="b1_c72_h20_w20_oc576_wic72_k1x1_g1_s1_pvalid", batch=1, in_c=72, in_h=20, in_w=20, out_c=576, weight_in_c=72, kh=1, kw=1, groups=1),
        dict(name="b1_c576_h20_w20_oc576_wic1_k3x3_g576_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=576, weight_in_c=1, kh=3, kw=3, groups=576),
        dict(name="b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=72, weight_in_c=576, kh=1, kw=1, groups=1),
        dict(name="b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid", batch=1, in_c=72, in_h=20, in_w=20, out_c=288, weight_in_c=72, kh=3, kw=3, groups=1),
        dict(name="b1_c288_h20_w20_oc72_wic288_k1x1_g1_s1_pvalid", batch=1, in_c=288, in_h=20, in_w=20, out_c=72, weight_in_c=288, kh=1, kw=1, groups=1),
        dict(name="b1_c576_h20_w20_oc576_wic1_k5x5_g576_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=576, weight_in_c=1, kh=5, kw=5, groups=576),
        dict(name="b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=96, weight_in_c=576, kh=1, kw=1, groups=1),
        dict(name="b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=768, weight_in_c=1, kh=5, kw=5, groups=768),
        dict(name="b1_c768_h20_w20_oc96_wic768_k1x1_g1_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=96, weight_in_c=768, kh=1, kw=1, groups=1),
        dict(name="b1_c768_h20_w20_oc768_wic1_k3x3_g768_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=768, weight_in_c=1, kh=3, kw=3, groups=768),
        dict(name="b1_c768_h10_w10_oc120_wic768_k1x1_g1_s1_pvalid", batch=1, in_c=768, in_h=10, in_w=10, out_c=120, weight_in_c=768, kh=1, kw=1, groups=1),
        dict(name="b1_c960_h10_w10_oc120_wic960_k1x1_g1_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=120, weight_in_c=960, kh=1, kw=1, groups=1),
        dict(name="b1_c480_h10_w10_oc480_wic1_k5x5_g480_s1_pvalid", batch=1, in_c=480, in_h=10, in_w=10, out_c=480, weight_in_c=1, kh=5, kw=5, groups=480),
        dict(name="b1_c480_h10_w10_oc120_wic480_k1x1_g1_s1_pvalid", batch=1, in_c=480, in_h=10, in_w=10, out_c=120, weight_in_c=480, kh=1, kw=1, groups=1),
        dict(name="b1_c960_h10_w10_oc960_wic1_k5x5_g960_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=960, weight_in_c=1, kh=5, kw=5, groups=960),
        dict(name="b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid", batch=1, in_c=256, in_h=10, in_w=10, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
        dict(name="b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
        dict(name="b1_c128_h3_w3_oc128_wic1_k3x3_g128_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=128, weight_in_c=1, kh=3, kw=3, groups=128),
        dict(name="b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=2, in_w=2, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
        dict(name="b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", batch=1, in_c=64, in_h=1, in_w=1, out_c=128, weight_in_c=64, kh=1, kw=1, groups=1),
        dict(name="b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
        dict(name="b1_c384_h10_w10_oc384_wic1_k3x3_g384_s1_pvalid", batch=1, in_c=384, in_h=10, in_w=10, out_c=384, weight_in_c=1, kh=3, kw=3, groups=384),
        dict(name="b1_c512_h5_w5_oc512_wic1_k3x3_g512_s1_pvalid", batch=1, in_c=512, in_h=5, in_w=5, out_c=512, weight_in_c=1, kh=3, kw=3, groups=512),
        dict(name="b1_c256_h3_w3_oc256_wic1_k3x3_g256_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
        dict(name="b1_c96_h20_w20_oc12_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=12, weight_in_c=96, kh=1, kw=1, groups=1),
        dict(name="b1_c96_h20_w20_oc273_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=273, weight_in_c=96, kh=1, kw=1, groups=1),
        dict(name="b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=10, in_w=10, out_c=546, weight_in_c=384, kh=1, kw=1, groups=1),
    ]
    def shape_stride(s):
        return s.get("stride", 1)
    nw = max(len(s["name"]) for s in shapes)
    iw = max(len(f"{s['in_c']}x{s['in_h']}x{s['in_w']}") for s in shapes)
    for s in shapes:
        stride = shape_stride(s)
        oh = (s["in_h"] - s["kh"]) // stride + 1
        ow = (s["in_w"] - s["kw"]) // stride + 1
        pass  # all shapes run, no skips
        r, inp, wt = run_conv(s["batch"], s["in_c"], s["out_c"], s["kh"], s["kw"],
                              (s["in_h"], s["in_w"]), groups=s["groups"], stride=stride)
        e = compute_expected_nchw(inp, wt, s["batch"], s["in_c"], s["in_h"], s["in_w"],
                                  s["out_c"], s["kh"], s["kw"], groups=s["groups"], stride=stride)
        md = float(np.max(np.abs(r.astype(np.float64) - e)))
        ok = np.allclose(r, e, atol=0.2) and not np.any(np.isinf(r))
        in_str = f"{s['in_c']}x{s['in_h']}x{s['in_w']}"
        out_str = f"{s['out_c']}x{oh}x{ow}"
        print(f"  {s['name']:<{nw}s} {in_str:<{iw}s} -> {out_str}  {'PASS' if ok else 'FAIL'}  (max_diff={md:.4f})")
    os.close(fd)
