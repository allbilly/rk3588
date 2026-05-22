import os, sys, mmap, ctypes, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../experimental/kernel_6_18")))
import rocket_runtime as rt


class struct_rknpu_task(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32), ("op_idx", ctypes.c_uint32),
        ("enable_mask", ctypes.c_uint32), ("int_mask", ctypes.c_uint32),
        ("int_clear", ctypes.c_uint32), ("int_status", ctypes.c_uint32),
        ("regcfg_amount", ctypes.c_uint32), ("regcfg_offset", ctypes.c_uint32),
        ("regcmd_addr", ctypes.c_uint64),
    ]

FP16_BYTES = 2
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
MIN_CHANNEL_TILE = 32
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
FP16_ATOM_ELEMENTS = 16
WEIGHT_ATOMIC_ELEMENTS = 32
PC_CHAIN_TAIL_QWORDS = 4
PAD_MODE_SAME = 1
PAD_MODE_VALID = 2

class reg:
    CNA  = 0x0201; CORE = 0x0801; DPU = 0x1001; PC = 0x0081; PC_REG = 0x0101; VERSION = 0x0041
    PPU = 0x4001; PPU_RDMA = 0x8001
    S_POINTER = 0x4004; FEATURE_MODE_CFG = 0x400c; DATA_FORMAT = 0x4010
    OFFSET_PEND = 0x4014; DST_BASE_ADDR = 0x4020; DST_SURF_STRIDE = 0x4024
    DATA_CUBE_WIDTH = 0x4030; DATA_CUBE_HEIGHT = 0x4034; DATA_CUBE_NOTCH = 0x4038
    DATA_CUBE_CHANNEL = 0x403c; BS_CFG = 0x4040; BS_ALU_CFG = 0x4044
    BS_MUL_CFG = 0x4048; BS_RELUX_CMP_VALUE = 0x404c; BS_OW_CFG = 0x4050
    BS_OW_OP = 0x4054; WDMA_SIZE_0 = 0x4058; WDMA_SIZE_1 = 0x405c
    BN_CFG = 0x4060; BN_ALU_CFG = 0x4064; BN_MUL_CFG = 0x4068
    BN_RELUX_CMP_VALUE = 0x406c; EW_CFG = 0x4070
    EW_CVT_OFFSET_VALUE = 0x4074; EW_CVT_SCALE_VALUE = 0x4078
    EW_RELUX_CMP_VALUE = 0x407c; OUT_CVT_OFFSET = 0x4080; OUT_CVT_SCALE = 0x4084
    OUT_CVT_SHIFT = 0x4088
    EW_OP_VALUE_0 = 0x4090; SURFACE_ADD = 0x40c0
    LUT_ACCESS_CFG = 0x4100; LUT_ACCESS_DATA = 0x4104; LUT_CFG = 0x4108
    LUT_INFO = 0x410c; LUT_LE_START = 0x4110; LUT_LE_END = 0x4114
    LUT_LO_START = 0x4118; LUT_LO_END = 0x411c
    LUT_LE_SLOPE_SCALE = 0x4120; LUT_LE_SLOPE_SHIFT = 0x4124
    LUT_LO_SLOPE_SCALE = 0x4128; LUT_LO_SLOPE_SHIFT = 0x412c
    CNA_CONV_CON1 = 0x100c; CNA_CONV_CON2 = 0x1010; CNA_CONV_CON3 = 0x1014
    CNA_DATA_SIZE0 = 0x1020; CNA_DATA_SIZE1 = 0x1024; CNA_DATA_SIZE2 = 0x1028
    CNA_DATA_SIZE3 = 0x102c; CNA_WEIGHT_SIZE0 = 0x1030; CNA_WEIGHT_SIZE1 = 0x1034
    CNA_WEIGHT_SIZE2 = 0x1038; CNA_CBUF_CON0 = 0x1040; CNA_CBUF_CON1 = 0x1044
    CNA_CVT_CON0 = 0x104c; CNA_CVT_CON1 = 0x1050; CNA_CVT_CON2 = 0x1054
    CNA_CVT_CON3 = 0x1058; CNA_CVT_CON4 = 0x105c; CNA_FC_CON0 = 0x1060
    CNA_FC_CON1 = 0x1064; CNA_PAD_CON0 = 0x1068
    CNA_FEATURE_DATA_ADDR = 0x1070; CNA_FC_CON2 = 0x1074
    CNA_DMA_CON0 = 0x1078; CNA_DMA_CON1 = 0x107c; CNA_DMA_CON2 = 0x1080
    CNA_FC_DATA_SIZE0 = 0x1084; CNA_FC_DATA_SIZE1 = 0x1088
    CNA_DCOMP_CTRL = 0x1100; CNA_DCOMP_REGNUM = 0x1104
    CNA_DCOMP_ADDR0 = 0x1110; CNA_DCOMP_AMOUNT0 = 0x1140
    CNA_CVT_CON5 = 0x1180; CNA_PAD_CON1 = 0x1184
    RDMA = 0x2001
    RDMA_S_POINTER = 0x5004; RDMA_ERDMA_CFG = 0x5034; RDMA_FEATURE_MODE_CFG = 0x5044
    CORE_MISC_CFG = 0x3010; CORE_DATAOUT_SIZE_0 = 0x3014
    CORE_DATAOUT_SIZE_1 = 0x3018; CORE_CLIP_TRUNCATE = 0x301c
    OPERATION_ENABLE = 0x0008; PC_BASE_ADDRESS = 0x0010; PC_REGISTER_AMOUNTS = 0x0014


def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr


def _ceil_div(x, y):
    return (x + y - 1) // y


def _align_up(x, align):
    return ((x + align - 1) // align) * align


def _is_depthwise(in_c, out_c, groups):
    return groups == in_c and groups == out_c


def _is_kh_major(out_c, in_c, kh, kw, groups):
    is_spatial = (kh != 1 or kw != 1)
    multi_input_group = groups > 1 and in_c != groups
    return is_spatial and not _is_depthwise(in_c, out_c, groups) and not multi_input_group


def _uses_pointwise_weight_atom_layout(in_c, kh, kw, groups):
    return groups == 1 and kh == 1 and kw == 1 and _ceil_div(in_c, WEIGHT_ATOMIC_ELEMENTS) > 1


def _pack_crs_order(weight, out_c, in_c, kh, kw, c2_out, layout):
    aligned_in_c = c2_out * _ceil_div(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:, :in_c] = weight
    if layout == "kh_major":
        return padded.transpose(2, 3, 0, 1).ravel()
    if layout == "default":
        return padded.transpose(0, 2, 3, 1).ravel()
    raise ValueError(layout)


def _pack_pointwise_wide(weight, out_c, in_c):
    aligned_in_c = max(WEIGHT_ATOMIC_ELEMENTS, _align_up(in_c, WEIGHT_ATOMIC_ELEMENTS))
    aligned_out_c = _align_up(out_c, 16) if out_c >= 48 else out_c
    padded = np.zeros((aligned_out_c, aligned_in_c), dtype=np.float16)
    padded[:out_c, :in_c] = weight[:, :in_c, 0, 0]
    return np.concatenate([
        padded[oc:oc + 16].reshape(-1, aligned_in_c // WEIGHT_ATOMIC_ELEMENTS, WEIGHT_ATOMIC_ELEMENTS).transpose(1, 0, 2).ravel()
        for oc in range(0, aligned_out_c, 16)
    ])


def _pack_dw_spatial_major(weight, out_c, in_c, kh, kw, c2_out):
    packed = np.zeros((kh, kw, c2_out), dtype=np.float16)
    packed[:, :, :out_c] = weight[range(out_c), range(out_c)].transpose(1, 2, 0)
    return packed.ravel()


def choose_weight_layout(out_c, in_c, kh, kw, align_c, groups):
    if _is_depthwise(in_c, out_c, groups) and out_c <= align_c and kh == kw:
        return "depthwise_spatial"
    if _uses_pointwise_weight_atom_layout(in_c, kh, kw, groups):
        return "pointwise_wide"
    if _is_kh_major(out_c, in_c, kh, kw, groups):
        return "kh_major"
    return "default"


def pack_conv_weights(weight, out_c, in_c, kh, kw, align_c, groups):
    layout = choose_weight_layout(out_c, in_c, kh, kw, align_c, groups)
    if layout == "depthwise_spatial":
        return _pack_dw_spatial_major(weight, out_c, in_c, kh, kw, align_c)
    if layout == "pointwise_wide":
        return _pack_pointwise_wide(weight, out_c, in_c)
    return _pack_crs_order(weight, out_c, in_c, kh, kw, align_c, layout)


def _pack_cdma_dc_feature_input(in_nchw, align_c, c2, width_stride, use_nhwc):
    """Pack NCHW into CDMA_DC format for conv."""
    n, c, h, w = in_nchw.shape
    if use_nhwc:
        out = np.zeros((h, width_stride, c), dtype=np.float16)
        out[:, :w] = in_nchw[0].transpose(1, 2, 0)
        return out.ravel()
    c1 = _ceil_div(max(c, align_c), c2)
    padded = np.zeros((c1 * c2, h, width_stride), dtype=np.float16)
    padded[:c, :, :w] = in_nchw[0]
    return padded.reshape(c1, c2, h, width_stride).transpose(0, 2, 3, 1).ravel()


def _conv_align_c(in_c, groups, out_c):
    if not _is_depthwise(in_c, out_c, groups) and groups == 1 and in_c > 1: return 16
    if not _is_depthwise(in_c, out_c, groups) and in_c > 4: return 16
    return np.clip(1 << (max(1, in_c) - 1).bit_length(), 8, 32)


def _input_pack_c2(in_c, groups, out_c, align_c):
    if in_c == 1: return 2
    if not _is_depthwise(in_c, out_c, groups) and groups == 1 and 1 < in_c <= 4: return 8
    return 8


def _should_use_nhwc(in_c, c2):
    return in_c > 0 and c2 // in_c == 2


def _unpack_nc1hwc2(out_raw, out_c, out_h, out_w, c2, width_stride):
    c1 = out_raw.size // (width_stride * c2)
    result = np.zeros((out_c, out_h, out_w), dtype=np.float32)
    for plane in range(c1):
        ps = width_stride * c2
        plane_raw = out_raw[plane * ps:plane * ps + out_h * out_w * c2]
        ch = min(c2, out_c - plane * c2)
        if ch <= 0: break
        result[plane * c2:plane * c2 + ch] = plane_raw.reshape(out_h, out_w, c2).transpose(2, 0, 1)[:ch]
    return result[np.newaxis]


def make_conv_tile_regs(tile_in_dma, tile_wt_dma, tile_out_dma,
                         in_h, in_w, in_c, oc_tile, kh, kw, out_h, out_w,
                         width_stride, line_stride, surf_stride,
                         align_c, ic_aligned, data_banks, weight_bank,
                         data_entries, dst_surf_stride, surface_add,
                         feature_grains, cvt_con5,
                         stride_h=1, stride_w=1, precision=2, is_spatial=False, is_depthwise=False):
    oc_align = _align_up(oc_tile, MIN_CHANNEL_TILE)
    data_in_channel_aligned = _align_up(in_c, align_c)
    wt_per_kernel = data_in_channel_aligned * kh * kw * FP16_BYTES
    wt_total = wt_per_kernel * oc_align
    out_atomics = out_w * out_h
    out_channel_field = oc_align - 1 if not is_depthwise else _align_up(oc_align, 32) - 1

    # cvt_con5: per-channel CVT mask 
    cvt_con5 = 65535 if (in_c == 1 and not is_depthwise) else 0

    use_nhwc_pack = False

    npu_regs = [
        E(reg.DPU, reg.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.RDMA, reg.RDMA_S_POINTER, 0),
        E(reg.RDMA, reg.RDMA_ERDMA_CFG, 0),
        E(reg.RDMA, reg.RDMA_FEATURE_MODE_CFG, 0),
        E(reg.CNA, reg.CNA_CONV_CON1,
            (2 << 4) | (2 << 7) | (3 if is_depthwise else 0)),
        E(reg.CNA, reg.CNA_CONV_CON2, (feature_grains << 4)),
        E(reg.CNA, reg.CNA_CONV_CON3, (stride_h << 3) | stride_w),
        E(reg.CNA, reg.CNA_DATA_SIZE0, (width_stride << 16) | in_h),
        E(reg.CNA, reg.CNA_DATA_SIZE1, ((in_c - 1) << 16) | ic_aligned),
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_atomics),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, wt_total),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, wt_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2, (kw << 24) | (kh << 16) | oc_tile),
        E(reg.CNA, reg.CNA_CBUF_CON0, ((RK_CBUF_BANKS - data_banks) << 4) | data_banks),
        E(reg.CNA, reg.CNA_CBUF_CON1, data_entries),
        E(reg.CNA, reg.CNA_CVT_CON0, 1),
        E(reg.CNA, reg.CNA_CVT_CON1, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON2, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON3, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON4, (1 << 16)),
        E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, tile_in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        E(reg.CNA, reg.CNA_DMA_CON1, line_stride),
        E(reg.CNA, reg.CNA_DMA_CON2, surf_stride),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, (in_w << 16) | in_h),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, ic_aligned),
        E(reg.CNA, reg.CNA_DCOMP_ADDR0, tile_wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, cvt_con5),
        E(reg.CORE, reg.CORE_MISC_CFG, (2 << 8) | (is_depthwise << 1) | is_spatial),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field),
        E(reg.CORE, 0x3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, (15 << 5) | ((3 * is_depthwise) << 3) | (2 << 1)),
        E(reg.DPU, reg.DATA_FORMAT, (2 << 29) | (2 << 26) | 2),
        E(reg.DPU, reg.DST_BASE_ADDR, tile_out_dma),
        E(reg.DPU, reg.DST_SURF_STRIDE, dst_surf_stride),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1),
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),
        E(reg.DPU, reg.DATA_CUBE_NOTCH, 0),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL, ((oc_align - 1) << 16) | out_channel_field),
        E(reg.DPU, reg.BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.BS_OW_CFG, (1 << 8) | (1 << 5) | (1 << 2) | (1 << 1)),
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field),
        E(reg.DPU, reg.WDMA_SIZE_1, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.DPU, reg.BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1),
        E(reg.DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
        E(reg.DPU, reg.SURFACE_ADD, surface_add),
    ]
    return npu_regs


def _feature_grains(row_bytes, floor_grains, use_nhwc_pack=False, is_spatial=False, is_depthwise=False):
    if use_nhwc_pack and is_spatial:
        return floor_grains
    if is_depthwise and is_spatial:
        return min(13, floor_grains)
    even_rows_per_two_banks = (_ceil_div(2 * CBUF_BANK_SIZE, row_bytes) + 1) & ~1
    return min(floor_grains, even_rows_per_two_banks)


def compute_expected(out_nchw, weight, kh, kw, stride, pad):
    n, ic, ih, iw = out_nchw.shape
    oh = (ih + 2 * pad - kh) // stride + 1
    ow = (iw + 2 * pad - kw) // stride + 1
    oc = weight.shape[0]
    out = np.zeros((n, oc, oh, ow), dtype=np.float32)
    padded = np.pad(out_nchw, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    for o in range(oc):
        for c in range(ic):
            for y in range(oh):
                for x in range(ow):
                    out[0, o, y, x] += np.sum(
                        padded[0, c, y * stride:y * stride + kh, x * stride:x * stride + kw] *
                        weight[o, c].astype(np.float32)
                    )
    return out


def _unpack_nc1hwc2_output(raw, out_h, out_w, oc):
    oc_align = _align_up(oc, MIN_CHANNEL_TILE)
    channels_per_block = FP16_ATOM_ELEMENTS // FP16_BYTES
    c2 = channels_per_block
    nc1 = oc_align // c2
    raw_fp32 = raw.view(np.float32).reshape(1, nc1, out_h, out_w, c2)
    out = np.zeros((1, oc, out_h, out_w), dtype=np.float32)
    for o in range(oc):
        ci = o // c2
        cj = o % c2
        out[0, o] = raw_fp32[0, ci, :, :, cj]
    return out


def run_conv(fd, in_nchw, weight_nchw, kh, kw, stride=1, groups=1, pad_mode=PAD_MODE_SAME):
    n, ic, ih, iw = in_nchw.shape
    oc = weight_nchw.shape[0]
    pad = kh // 2 if pad_mode == PAD_MODE_SAME else 0
    oh = (ih + 2 * pad - kh) // stride + 1
    ow = (iw + 2 * pad - kw) // stride + 1
    is_depthwise = _is_depthwise(ic, oc, groups)

    align_c = 32 if _uses_pointwise_weight_atom_layout(ic, kh, kw, groups) else _conv_align_c(ic, groups, oc)
    oc_align = max(32, _align_up(oc, 16))
    width_stride = _align_up(iw, max(1, _ceil_div(16, align_c)))
    out_atoms = max(1, oh * ow)
    is_spatial = (kh != 1 or kw != 1)
    out_width_stride = out_atoms if not is_spatial else _align_up(out_atoms, 4)
    c2 = _input_pack_c2(ic, groups, oc, align_c)
    use_nhwc = _should_use_nhwc(ic, c2)
    out_c2 = 8

    data_in_channel_aligned = _align_up(ic, align_c)
    if use_nhwc:
        row_bytes = width_stride * ic * FP16_BYTES
        line_stride = width_stride
        surf_stride = line_stride * (ih - 1) if ih > 1 else 0
        data_entries = _ceil_div(width_stride * ic, 2 * FP16_ATOM_ELEMENTS)
    else:
        row_bytes = width_stride * data_in_channel_aligned * FP16_BYTES
        line_stride = width_stride * 4
        surf_stride = width_stride * (ih - 4) if ih > 4 else 0
        data_entries = _ceil_div(width_stride * data_in_channel_aligned, 2 * FP16_ATOM_ELEMENTS)

    feature_grains = _feature_grains(row_bytes, ih + kh, use_nhwc, is_spatial, is_depthwise)
    if not is_spatial:
        feature_grains = max(feature_grains, 52)

    data_banks = RK_CBUF_BANKS - 1 if (is_spatial and (use_nhwc or is_depthwise)) else \
        int(np.clip(_ceil_div(width_stride * feature_grains * data_in_channel_aligned * FP16_BYTES, CBUF_BANK_SIZE), 1, RK_CBUF_BANKS - 1))
    weight_bank = RK_CBUF_BANKS - data_banks
    dest_surf_stride = out_width_stride << 4
    effective_align_out = max(16, _align_up(_ceil_div(oc, groups), 16)) if (groups > 1 and not is_depthwise) else oc_align
    surface_add = (out_width_stride * max(2, effective_align_out // 16)) << 4

    input_packed = _pack_cdma_dc_feature_input(in_nchw, align_c, c2, width_stride, use_nhwc).view(np.uint16).tolist()
    weight_packed = pack_conv_weights(weight_nchw, oc, ic, kh, kw, align_c, groups).view(np.uint16).tolist()

    input_bytes = len(input_packed) * FP16_BYTES
    weight_bytes = len(weight_packed) * FP16_BYTES
    output_bytes = out_width_stride * oc_align * 4
    max_reg_qwords = 4096
    max_tasks = _ceil_div(oc_align, MIN_CHANNEL_TILE) + 2

    in_buf, in_bo = rt.mem_allocate(fd, input_bytes)
    wt_buf, wt_bo = rt.mem_allocate(fd, weight_bytes)
    out_buf, out_bo = rt.mem_allocate(fd, output_bytes)
    reg_buf, reg_bo = rt.mem_allocate(fd, max_reg_qwords * 8)
    task_buf, task_bo = rt.mem_allocate(fd, max_tasks * ctypes.sizeof(struct_rknpu_task))

    task_ptrs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_buf)),
                            ctypes.POINTER(struct_rknpu_task))

    ct_inputs = (ctypes.c_uint16 * len(input_packed)).from_buffer(in_buf)
    ct_weights = (ctypes.c_uint16 * len(weight_packed)).from_buffer(wt_buf)
    ct_inputs[:] = input_packed
    ct_weights[:] = weight_packed

    task_regs = []
    wt_per_kernel = ic * kh * kw * FP16_BYTES
    print(f"  in_dma=0x{in_bo.dma_addr:x} wt_dma=0x{wt_bo.dma_addr:x} out_dma=0x{out_bo.dma_addr:x} reg_dma=0x{reg_bo.dma_addr:x}", flush=True)
    print(f"  width_stride={width_stride} line_stride={line_stride} surf_stride={surf_stride}", flush=True)
    print(f"  data_banks={data_banks} weight_bank={weight_bank} data_entries={data_entries}", flush=True)
    cvt_con5 = 65535 if (ic == 1 and not is_depthwise) else (0 if (not is_depthwise and groups == 1) else 0)
    regs = make_conv_tile_regs(in_bo.dma_addr, wt_bo.dma_addr, out_bo.dma_addr,
                               ih, iw, ic, oc_align, kh, kw,
                               oh, ow, width_stride, line_stride, surf_stride,
                               align_c, data_in_channel_aligned, data_banks, weight_bank,
                               data_entries, dest_surf_stride, surface_add,
                               feature_grains, cvt_con5,
                               stride, stride, 2)
    task_regs.append(regs)
    print(f"  tile_regs={len(regs)} qwords regcmd_addr=0x{task_ptrs[0].regcmd_addr:x}", flush=True)

    reg_arr = np.frombuffer(reg_buf, dtype=np.uint64)
    print(f"  first reg: 0x{reg_arr[0]:016x}", flush=True)
    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += len(regs) + PC_CHAIN_TAIL_QWORDS

    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        reg_arr[base:base + len(regs)] = regs
        next_len = len(task_regs[idx + 1]) if idx + 1 < len(task_regs) else 0
        next_off = offsets[idx + 1] if idx + 1 < len(task_regs) else None
        enable = E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)
        if next_off is None:
            tails = [E(0x0001, 0, 0), E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                     E(reg.VERSION, 0, 0), enable]
        else:
            na = reg_bo.dma_addr + next_off * 8
            tails = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, na & 0xFFFFFFF0),
                     E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(next_len, 2) + 1),
                     E(reg.VERSION, 0, 0), enable]
        reg_arr[base + len(regs):base + len(regs) + 4] = tails
        task_ptrs[idx].regcmd_addr = reg_bo.dma_addr + base * 8
        task_ptrs[idx].regcfg_amount = len(regs)
        task_ptrs[idx].enable_mask = 0xd
        task_ptrs[idx].int_mask = (1 << 8) | (1 << 9)
        task_ptrs[idx].int_clear = 0x1ffff

    rt.reset_npu(fd)
    rt.fini_bo(fd, in_bo)
    rt.fini_bo(fd, wt_bo)
    ret = rt.submit(fd, task_ptrs, len(task_regs),
                    in_bos=[reg_bo, in_bo, wt_bo], out_bos=[out_bo])
    print(f"  submit ret={ret}", flush=True)
    if ret != 0:
        raise RuntimeError(f"submit failed: {ret}")
    rt.prep_bo(fd, out_bo, timeout_ns=10000000000)

    raw = np.frombuffer(out_buf, dtype=np.float32, count=output_bytes // 4).copy()
    out = _unpack_nc1hwc2(raw, oc, oh, ow, out_c2, out_width_stride)
    return out


if __name__ == "__main__":
    np.random.seed(42)
    fd = rt.open_rocket_device()
    shapes = [
        (1, 8, 10, 16, 32, 1, 1, 1),
        (1, 3, 32, 32, 16, 1, 1, 1),
        (1, 8, 10, 16, 32, 3, 3, 1),
    ]
    for n, ic, ih, iw, oc, kh, kw, stride in shapes:
        in_nchw = (np.random.rand(n, ic, ih, iw) * 2 - 1).astype(np.float16)
        weight = (np.random.rand(oc, ic, kh, kw) * 0.1).astype(np.float16)
        r = run_conv(fd, in_nchw, weight, kh, kw, stride)
        expected = compute_expected(in_nchw, weight, kh, kw, stride, kh // 2)
        ok = np.allclose(r, expected, atol=0.2)
        md = np.max(np.abs(r.astype(np.float64) - expected.astype(np.float64)))
        print(f"{n}x{ic}x{ih}x{iw} -> oc={oc} k={kh}x{kw} s={stride}: "
              f"{'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")
        if not ok:
            print(f"  Input range: [{in_nchw.min():.4f}, {in_nchw.max():.4f}]")
            print(f"  Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
            print(f"  NPU range: [{r.min():.4f}, {r.max():.4f}]")
            print(f"  Expected range: [{expected.min():.4f}, {expected.max():.4f}]")
    os.close(fd)
