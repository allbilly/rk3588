import os, sys, mmap, ctypes, numpy as np
from fcntl import ioctl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../experimental/kernel_6_18")))
import rocket_runtime as rt

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
FP16_BYTES = 2
FP16_ATOM_ELEMENTS = 16
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
# Small-channel 1x1 uses CNA_CONV_CON1_NONALIGN_DMA/ARGB_IN; large flat output
# strides corrupt the tail, while the regular c16 path is fine at 1024.
RK_MAX_CONV_FLAT_STRIDE = 992
UNPACK_C2 = FP16_ATOM_ELEMENTS // FP16_BYTES
PC_CHAIN_TAIL_QWORDS = 4  # enable_npu_units() returns 4 QWORDs: PC_BASE_ADDRESS, PC_REGISTER_AMOUNTS, VERSION, OPERATION_ENABLE

class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    CNA  = 0x0201   # CNA (Convolution/Matrix unit)
    CORE = 0x0801   # CORE (Matrix compute engine)
    DPU  = 0x1001   # DPU (Elementwise/DPU unit)
    RDMA = 0x2001   # RDMA (Read DMA for inputs/weights)
    PC   = 0x0081   # PC (Program Control / operation enable)
    PC_REG = 0x0101 # PC chain registers
    VERSION = 0x0041

    # --- PC (0x0000) ---
    OPERATION_ENABLE    = 0x0008   # PC operation enable
    PC_BASE_ADDRESS     = 0x0010   # next regcmd DMA address for PC chain
    PC_REGISTER_AMOUNTS = 0x0014   # next regcmd fetch amount for PC chain

    # --- DPU (0x4000) ---
    S_POINTER           = 0x4004   # DPU S pointer config (pp/exec)
    FEATURE_MODE_CFG    = 0x400c   # DPU feature mode config

    # --- RDMA (0x5000) ---
    RDMA_S_POINTER      = 0x5004   # DPU RDMA S pointer config
    DATA_FORMAT         = 0x4010   # DPU data format config
    DST_BASE_ADDR       = 0x4020   # DPU destination base address
    DST_SURF_STRIDE     = 0x4024   # DPU destination surface stride
    DATA_CUBE_WIDTH     = 0x4030   # DPU data cube width
    DATA_CUBE_HEIGHT    = 0x4034   # DPU data cube height
    DATA_CUBE_NOTCH     = 0x4038   # DPU data cube notch
    DATA_CUBE_CHANNEL   = 0x403c   # DPU data cube channel
    BS_CFG              = 0x4040   # DPU batch/norm/scale config
    BS_OW_CFG           = 0x4050   # DPU batch OW config
    WDMA_SIZE_0         = 0x4058   # DPU write DMA size 0
    WDMA_SIZE_1         = 0x405c   # DPU write DMA size 1
    BN_CFG              = 0x4060   # DPU batch norm config
    EW_CFG              = 0x4070   # DPU elementwise config
    EW_CVT_SCALE_VALUE  = 0x4078   # DPU EW convert scale value
    OUT_CVT_SCALE       = 0x4084   # DPU output conversion scale
    SURFACE_ADD         = 0x40c0   # DPU surface add

    # --- DPU RDMA (0x5000) ---
    RDMA_DATA_CUBE_WIDTH  = 0x500c   # RDMA data cube width
    RDMA_DATA_CUBE_HEIGHT = 0x5010   # RDMA data cube height
    RDMA_DATA_CUBE_CHANNEL= 0x5014   # RDMA data cube channel
    RDMA_ERDMA_CFG        = 0x5034   # RDMA ERDMA config
    RDMA_SRC_BASE_ADDR    = 0x5018   # RDMA source base address (input)
    RDMA_EW_BASE_ADDR     = 0x5038   # RDMA EW base address (weight)
    RDMA_FEATURE_MODE_CFG = 0x5044   # RDMA feature mode config

    # --- CNA (0x1000) ---
    CNA_CONV_CON1          = 0x100c   # CNA convolution control 1
    CNA_CONV_CON2          = 0x1010   # CNA convolution control 2 (grains)
    CNA_CONV_CON3          = 0x1014   # CNA convolution control 3 (stride)
    CNA_DATA_SIZE0         = 0x1020   # CNA input data size 0
    CNA_DATA_SIZE1         = 0x1024   # CNA input data size 1 (channel)
    CNA_DATA_SIZE2         = 0x1028   # CNA output data size 2
    CNA_DATA_SIZE3         = 0x102c   # CNA output data size 3 (atomics)
    CNA_WEIGHT_SIZE0       = 0x1030   # CNA weight total size
    CNA_WEIGHT_SIZE1       = 0x1034   # CNA weight per-kernel size
    CNA_WEIGHT_SIZE2       = 0x1038   # CNA weight dims (width/height/kernels)
    CNA_CBUF_CON0          = 0x1040   # CNA CBUF config 0 (banks)
    CNA_CBUF_CON1          = 0x1044   # CNA CBUF config 1 (entries)
    CNA_CVT_CON0           = 0x104c   # CNA convert config 0
    CNA_CVT_CON1           = 0x1050   # CNA convert config 1 (scale)
    CNA_CVT_CON2           = 0x1054   # CNA convert config 2 (scale)
    CNA_CVT_CON3           = 0x1058   # CNA convert config 3 (scale)
    CNA_CVT_CON4           = 0x105c   # CNA convert config 4 (scale)
    CNA_CVT_CON5           = 0x1180   # CNA convert config 5 (mask)
    CNA_FEATURE_DATA_ADDR  = 0x1070   # CNA feature data base address
    CNA_DMA_CON0           = 0x1078   # CNA DMA control 0 (burst)
    CNA_DMA_CON1           = 0x107c   # CNA DMA control 1 (line stride)
    CNA_DMA_CON2           = 0x1080   # CNA DMA control 2 (surface stride)
    CNA_FC_DATA_SIZE0      = 0x1084   # CNA FC data size 0
    CNA_FC_DATA_SIZE1      = 0x1088   # CNA FC data size 1 (channel)
    CNA_DCOMP_ADDR0        = 0x1110   # CNA weight decompress address 0

    # --- CORE (0x3000) ---
    CORE_MISC_CFG          = 0x3010   # CORE misc config
    CORE_DATAOUT_SIZE_0    = 0x3014   # CORE dataout size 0 (height/width)
    CORE_DATAOUT_SIZE_1    = 0x3018   # CORE dataout size 1 (channel)
    CORE_RESERVED_3030     = 0x3030   # CORE reserved (must be zeroed)

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
task_map, tasks_mem_create = mem_allocate(fd, size=64*1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=512*1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
npu_tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
npu_regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c

# Spatial (non-dw, non-grouped conv) uses KH-major weight layout: spatial loops
# outermost, then OC, then aligned-IC innermost. This matches the CSC→CMAC
# data sequencing for spatial convolutions on RK schedule.
def _is_kh_major(out_c, in_c, kh, kw, groups):
    is_spatial = (kh != 1 or kw != 1)
    multi_input_group = groups > 1 and in_c != groups
    return is_spatial and not _is_depthwise(in_c, out_c, groups) and not multi_input_group

def _is_grouped_spatial(in_c, out_c, kh, kw, groups):
    is_spatial = (kh != 1 or kw != 1)
    multi_input_group = groups > 1 and in_c != groups
    return is_spatial and multi_input_group and not _is_depthwise(in_c, out_c, groups)

def _needs_pointwise_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
    if groups != 1 or kh != 1 or kw != 1:
        return False
    return ((in_c, out_c, in_h, in_w) == (96, 24, 56, 56) or
            (in_c, out_c, in_h, in_w) == (144, 24, 56, 56) or
            (in_c, out_c, in_h, in_w) == (144, 32, 28, 28) or
            (in_c, out_c, in_h, in_w) == (192, 32, 28, 28) or
            (in_c, out_c, in_h, in_w) == (192, 16, 28, 28) or
            (in_c, out_c, in_h, in_w) == (256, 32, 28, 28))

def _needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
    return groups == 1 and kh == 1 and kw == 1 and in_c >= 32 and out_c > 32 and in_h >= 7

def _pointwise_oc_tile_h(in_c):
    return 20 if in_c >= 128 else 32

def _depthwise_tile_h(out_c):
    return 13 if out_c >= 256 else 32

def _pointwise_tile_h(in_c, out_c, out_h, out_w):
    if out_h > 50:
        return min(out_h, 20)
    if (in_c, out_c, out_h, out_w) == (256, 32, 28, 28):
        return 20
    return out_h

def should_use_nhwc_pack(channels, c2):
    return channels > 0 and c2 // channels == 2

def _conv_align_c(in_c, groups, out_c):
    if not _is_depthwise(in_c, out_c, groups) and groups == 1 and in_c > 1: return 16
    if not _is_depthwise(in_c, out_c, groups) and in_c > 4: return 16
    return np.clip(1 << (max(1, in_c) - 1).bit_length(), 8, 32)

def _conv_input_pack_c2(in_c, groups, out_c, align_c):
    # C2 selection is RK packing layered on top of 16-value FP16 atoms.
    if in_c == 1: return 2
    if not _is_depthwise(in_c, out_c, groups) and groups == 1 and 1 < in_c <= 4: return 8
    if _is_depthwise(in_c, out_c, groups) or groups > 1 or in_c > 4: return 8
    return align_c

def _dma_strides(in_h, width_stride, use_nhwc_pack):
    # CDMA stride registers are in 32B units. RK NHWC/pixel rows use direct row
    # stride; packed feature mode uses four atom groups per logical row here.
    if use_nhwc_pack:
        line_stride = width_stride
        return line_stride, line_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, width_stride * (in_h - 4) if in_h > 4 else 0

def _cbuf_entries(width_stride, align_c, in_h, is_depthwise):
    # One CBUF entry is 128B, i.e. four 32B FP16 atoms. The align_c<16 multiplier
    # is RK schedule-specific for small feature inputs.
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

def _conv_params(batch, in_c, in_h, in_w, out_c, kh, kw, groups, stride=1):
    assert 1 <= stride <= 7, "CNA_CONV_CON3 stride fields are 3-bit values"
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    is_spatial = (kh != 1 or kw != 1)
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    align_c = 32 if (not is_spatial and groups == 1 and in_c >= 64) else _conv_align_c(in_c, groups, out_c)
    align_out_c = max(32, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_atoms = max(1, out_h * out_w)
    out_width_stride = out_atoms if not is_spatial else _align_up(out_atoms, 4)

    mesa_aligned_small = (not is_depthwise) and groups == 1 and 1 < in_c <= 4
    input_pack_c2 = _conv_input_pack_c2(in_c, groups, out_c, align_c)
    use_nhwc = (not mesa_aligned_small) and (not is_depthwise) and (in_c > 0 and input_pack_c2 // in_c == 2)
    return {
        "batch": batch, "in_c": in_c, "in_h": in_h, "in_w": in_w,
        "out_c": out_c, "kh": kh, "kw": kw, "groups": groups,
        "stride": stride,
        "is_depthwise": is_depthwise, "out_h": out_h, "out_w": out_w,
        "align_c": align_c, "align_out_c": align_out_c,
        "width_stride": width_stride, "out_width_stride": out_width_stride,
        "input_pack_c2": input_pack_c2, "use_nhwc": use_nhwc,
        "mesa_aligned_small": mesa_aligned_small,
    }

def _expand_grouped_weights(weight, in_c, out_c, kh, kw, groups):
    weight_in_c = in_c // groups
    if groups == 1:
        return weight.reshape(out_c, in_c, kh, kw)
    expanded = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
    out_per_group = out_c // groups
    for oc in range(out_c):
        group = oc // out_per_group
        start = group * weight_in_c
        expanded[oc, start:start + weight_in_c] = weight[oc]
    return expanded

def _pack_kh_major(weight, out_c, in_c, kh, kw, c2_out):
    aligned_in_c = c2_out * _ceil_div(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:, :in_c] = weight
    return padded.transpose(2, 3, 0, 1).ravel()

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

def pack_conv_weights_for_shape(weight_full, out_c, in_c, kh, kw, align_c, groups):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    if is_depthwise and out_c <= align_c and kh == kw:
        return _pack_dw_spatial_major(weight_full, out_c, in_c, kh, kw, align_c)
    if groups == 1 and kh == 1 and kw == 1 and in_c >= 64:
        return _pack_pointwise_wide(weight_full, out_c, in_c)
    if _is_kh_major(out_c, in_c, kh, kw, groups):
        return _pack_kh_major(weight_full, out_c, in_c, kh, kw, align_c)
    return _pack_default(weight_full, out_c, in_c, kh, kw, align_c)

def _pack_conv_input_fp16(input_nchw, p):
    in_c, in_h, in_w = input_nchw.shape
    if p["use_nhwc"]:
        out = np.zeros((in_h, p["width_stride"], in_c), dtype=np.float16)
        out[:, :in_w] = input_nchw.transpose(1, 2, 0)
        return out.ravel()

    c2 = p["input_pack_c2"]
    c1 = _ceil_div(max(in_c, p["align_c"]), c2)
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
    result = np.zeros((out_c, out_h, out_w), dtype=np.float16)
    for plane in range(c1):
        plane_raw = out_raw[plane * out_width_stride * c2:plane * out_width_stride * c2 + out_h * out_w * c2]
        channels = min(c2, out_c - plane * c2)
        if channels <= 0:
            break
        result[plane * c2:plane * c2 + channels] = plane_raw.reshape(out_h, out_w, c2).transpose(2, 0, 1)[:channels]
    return result

def _unpack_grouped_spatial_output(out_raw, out_c, out_h, out_w, c2, plane_stride):
    c1 = _ceil_div(out_c, c2)
    result = np.zeros((out_c, out_h, out_w), dtype=np.float16)
    for plane in range(c1):
        plane_raw = out_raw[plane * plane_stride:plane * plane_stride + out_h * out_w * c2]
        channels = min(c2, out_c - plane * c2)
        result[plane * c2:plane * c2 + channels] = plane_raw.reshape(out_h, out_w, c2).transpose(2, 0, 1)[:channels]
    return result

def _reorder_grouped_spatial_weights_block16(weight_full, out_c, in_c, kh, kw):
    block_size = 16
    kernel_hw = kh * kw
    reordered = np.empty_like(weight_full)
    for block_start in range(0, out_c, block_size):
        block_end = min(block_start + block_size, out_c)
        block_channels = block_end - block_start
        block = weight_full[block_start:block_end]

        # Hardware consumes grouped spatial weights as kh/kw-major within each
        # output-channel block; pack_default consumes oc-major. Pre-shuffle the
        # logical tensor so pack_default serializes the hardware order.
        pack_order = block.transpose(2, 3, 0, 1).reshape(block_channels, kernel_hw, in_c)
        reordered[block_start:block_end] = pack_order.transpose(0, 2, 1).reshape(block_channels, in_c, kh, kw)
    return reordered

def make_conv2d_regs(batch, in_c, in_h, in_w, out_c, kh, kw, in_dma, wt_dma, out_dma, groups=1, stride=1, out_width_stride_override=None, weight_reuse=False, full_data_bank=False):
    p = _conv_params(batch, in_c, in_h, in_w, out_c, kh, kw, groups, stride=stride)
    is_depthwise = p["is_depthwise"]
    is_spatial = (kh != 1 or kw != 1)
    out_h = p["out_h"]
    out_w = p["out_w"]
    align_c = p["align_c"]
    align_out_c = p["align_out_c"]
    width_stride = p["width_stride"]
    out_width_stride = out_width_stride_override or p["out_width_stride"]
    use_nhwc_pack = p["use_nhwc"]
    input_pack_c2 = p["input_pack_c2"]
    cvt_con5 = 65535 if (in_c == 1 and not is_depthwise) else (0 if (not is_depthwise and groups == 1) else ((1 << in_c if use_nhwc_pack else input_pack_c2) - 1))

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
    effective_align_out = max(16, _align_up(_ceil_div(out_c, groups), 16)) if (groups > 1 and not is_depthwise) else out_channel_field + 1

    npu_regs = [
        E(reg.DPU, reg.S_POINTER,
            ((1 << 3) |                    # DPU_S_POINTER_POINTER_PP_MODE
             (1 << 2) |                    # DPU_S_POINTER_EXECUTER_PP_EN
             (1 << 1))),                   # DPU_S_POINTER_POINTER_PP_EN
        E(reg.RDMA, reg.RDMA_S_POINTER, 0), # Clear stale DPU-RDMA state after eltwise/add tasks
        E(reg.RDMA, reg.RDMA_ERDMA_CFG, 0),
        E(reg.RDMA, reg.RDMA_FEATURE_MODE_CFG, 0),
        E(reg.CNA, reg.CNA_CONV_CON1,
            ((2 << 4) |                    # CNA_CONV_CON1_IN_PRECISION(fp16)
             (2 << 7) |                    # CNA_CONV_CON1_PROC_PRECISION(fp16)
             (((1 << 30) | (1 << 29) | ((7 + in_c) << 12)) if (in_c == 1 and not is_depthwise) else 0) |
             (3 if is_depthwise else 0))), # CNA_CONV_CON1_CONV_MODE(depthwise when 3)
        E(reg.CNA, reg.CNA_CONV_CON2,
            (feature_grains << 4)           # CNA_CONV_CON2_FEATURE_GRAINS
        ),
        E(reg.CNA, reg.CNA_CONV_CON3,
            ((stride << 3) |               # CNA_CONV_CON3_CONV_Y_STRIDE
             (stride << 0))),              # CNA_CONV_CON3_CONV_X_STRIDE
        E(reg.CNA, reg.CNA_DATA_SIZE0,
            ((width_stride << 16) |         # CNA_DATA_SIZE0_DATAIN_WIDTH
             in_h)),                       # CNA_DATA_SIZE0_DATAIN_HEIGHT
        E(reg.CNA, reg.CNA_DATA_SIZE1,
            ((in_c - 1 << 16) |             # CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL
             data_in_channel_aligned)),     # CNA_DATA_SIZE1_DATAIN_CHANNEL
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),                       # CNA_DATA_SIZE2_DATAOUT_WIDTH
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_w * out_h),               # CNA_DATA_SIZE3_DATAOUT_ATOMICS
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_total),        # CNA_WEIGHT_SIZE0_WEIGHT_TOTAL_SIZE
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),   # CNA_WEIGHT_SIZE1_WEIGHT_PER_KERNEL_SIZE
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2,
            ((kw << 24) |                  # CNA_WEIGHT_SIZE2_WEIGHT_WIDTH
             (kh << 16) |                  # CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT
             (1 if is_depthwise else out_c))), # CNA_WEIGHT_SIZE2_WEIGHT_KERNELS
        E(reg.CNA, reg.CNA_CBUF_CON0,
            ((weight_reuse << 13) |             # CNA_CBUF_CON0_WEIGHT_REUSE
             (RK_CBUF_BANKS - data_bank << 4) | # CNA_CBUF_CON0_WEIGHT_BANK
             data_bank)),                       # CNA_CBUF_CON0_DATA_BANK
        E(reg.CNA, reg.CNA_CBUF_CON1, cbuf_entries),      # CNA_CBUF_CON1_DATA_ENTRIES

        # CNA conversion and DMA setup.
        E(reg.CNA, reg.CNA_CVT_CON0, ((use_nhwc_pack << 3) |             # CNA_CVT_CON0_DATA_SIGN
                                      (use_nhwc_pack << 1) |             # CNA_CVT_CON0_CVT_TYPE
                                       1)),                              # CNA_CVT_CON0_CVT_BYPASS
        E(reg.CNA, reg.CNA_CVT_CON1, (1 << 16)),          # CNA_CVT_CON1_CVT_SCALE0
        E(reg.CNA, reg.CNA_CVT_CON2, (1 << 16)),          # CNA_CVT_CON2_CVT_SCALE1
        E(reg.CNA, reg.CNA_CVT_CON3, (1 << 16)),          # CNA_CVT_CON3_CVT_SCALE2
        E(reg.CNA, reg.CNA_CVT_CON4, (1 << 16)),          # CNA_CVT_CON4_CVT_SCALE3
        E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0,
            ((15 << 16) |                  # CNA_DMA_CON0_WEIGHT_BURST_LEN
             15)),                         # CNA_DMA_CON0_DATA_BURST_LEN
        E(reg.CNA, reg.CNA_DMA_CON1, line_stride),        # CNA_DMA_CON1_LINE_STRIDE
        E(reg.CNA, reg.CNA_DMA_CON2, surf_stride),        # CNA_DMA_CON2_SURF_STRIDE
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0,
            ((in_w << 16) |                # CNA_FC_DATA_SIZE0_DMA_WIDTH
             in_h)),                       # CNA_FC_DATA_SIZE0_DMA_HEIGHT
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, data_in_channel_aligned), # CNA_FC_DATA_SIZE1_DMA_CHANNEL
        E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, cvt_con5), # CNA_CVT_CON5_PER_CHANNEL_CVT_EN
        E(reg.CORE, reg.CORE_MISC_CFG, ((2 << 8) |             # CORE_MISC_CFG_PROC_PRECISION(fp16)
                                        (is_depthwise << 1) |  # CORE_MISC_CFG_DW_EN
                                        (is_spatial))),        # CORE_MISC_CFG_OPERATION_ENABLE),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0,
            (((out_h - 1) << 16) |          # CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT
             (out_w - 1))),                # CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field), # CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG,
            ((15 << 5) |                   # DPU_FEATURE_MODE_CFG_BURST_LEN
             ((3 * is_depthwise) << 3) |   # DPU_FEATURE_MODE_CFG_CONV_MODE(depthwise)
             (2 << 1))),                   # DPU_FEATURE_MODE_CFG_OUTPUT_MODE
        E(reg.DPU, reg.DATA_FORMAT,
            ((2 << 29) |                   # DPU_DATA_FORMAT_OUT_PRECISION(fp16)
             (2 << 26) |                   # DPU_DATA_FORMAT_PROC_PRECISION(fp16)
              2)),                         # DPU_DATA_FORMAT_IN_PRECISION(fp16)
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma),
        E(reg.DPU, reg.DST_SURF_STRIDE, out_width_stride << 4), # DPU_DST_SURF_STRIDE_DST_SURF_STRIDE
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1),            # DPU_DATA_CUBE_WIDTH_WIDTH
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),           # DPU_DATA_CUBE_HEIGHT_HEIGHT
        E(reg.DPU, reg.DATA_CUBE_NOTCH, 0),                    # Must be set 0, otherwise corrupt next run
        E(reg.DPU, reg.DATA_CUBE_CHANNEL,
            ((out_c - 1 << 16) |            # DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL
             out_channel_field)),           # DPU_DATA_CUBE_CHANNEL_CHANNEL
        E(reg.DPU, reg.BS_CFG,
            ((1 << 6) |                    # DPU_BS_CFG_BS_RELU_BYPASS
             (1 << 4) |                    # DPU_BS_CFG_BS_MUL_BYPASS
             (1 << 1) |                    # DPU_BS_CFG_BS_ALU_BYPASS
             1)),                          # DPU_BS_CFG_BS_BYPASS
        E(reg.DPU, reg.BS_OW_CFG,
            (((3 if is_depthwise else 1) << 8) | # DPU_BS_OW_CFG_SIZE_E_2
             ((3 if is_depthwise else 1) << 5) | # DPU_BS_OW_CFG_SIZE_E_1
             ((3 if is_depthwise else 1) << 2) | # DPU_BS_OW_CFG_SIZE_E_0
             (1 << 1))),                         # DPU_BS_OW_CFG_OD_BYPASS
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field),    # DPU_WDMA_SIZE_0_CHANNEL_WDMA
        E(reg.DPU, reg.WDMA_SIZE_1,
            (((out_h - 1) << 16) |          # DPU_WDMA_SIZE_1_HEIGHT_WDMA
             (out_w - 1))),                 # DPU_WDMA_SIZE_1_WIDTH_WDMA
        E(reg.DPU, reg.BN_CFG,
            ((1 << 6) |                    # DPU_BN_CFG_BN_RELU_BYPASS
             (1 << 4) |                    # DPU_BN_CFG_BN_MUL_BYPASS
             (1 << 1) |                    # DPU_BN_CFG_BN_ALU_BYPASS
             1)),                          # DPU_BN_CFG_BN_BYPASS
        E(reg.DPU, reg.EW_CFG,
            ((1 << 9) |                    # DPU_EW_CFG_EW_RELU_BYPASS
             (1 << 8) |                    # DPU_EW_CFG_EW_OP_CVT_BYPASS
             (1 << 7) |                    # DPU_EW_CFG_EW_LUT_BYPASS
             (1 << 1) |                    # DPU_EW_CFG_EW_OP_BYPASS
             1)),                          # DPU_EW_CFG_EW_BYPASS
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1),  # DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE
        E(reg.DPU, reg.OUT_CVT_SCALE,
            ((1 << 16) |                   # DPU_OUT_CVT_SCALE_FP32TOFP16_EN
             1)),                          # DPU_OUT_CVT_SCALE_OUT_CVT_SCALE
        E(reg.DPU, reg.SURFACE_ADD,
            (out_width_stride * max(2, effective_align_out // 16)) << 4), # DPU_SURFACE_ADD_SURF_ADD
    ]
    # DO NOT delete 
    return npu_regs

def write_regs_to_npu_task(task_regs):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())

    def enable_npu_units(next_offset, next_task_regs_len):
        enable = E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)
        if next_offset is None:
            return [
                E(reg.PC_REG, reg.PC_BASE_ADDRESS, 0),    # prevent corrupt next state
                E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                E(reg.VERSION, 0, 0),
                enable,
            ]
        next_addr = regcmd_mem_create.dma_addr + next_offset * ctypes.sizeof(ctypes.c_uint64)
        return [
            E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0), # rounds down to nearest multiple of 16
            E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(next_task_regs_len, 2) + 1),
            E(reg.VERSION, 0, 0),
            enable,
        ]

    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + PC_CHAIN_TAIL_QWORDS, 2)
    assert offset <= regcmd_mem_create.size // ctypes.sizeof(ctypes.c_uint64), "regcmd buffer too small"

    # add tail enable, write to npu_tasks
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            npu_regcmd[base + i] = qword
        next_task_regs_len = len(task_regs[idx + 1]) if idx + 1 < len(task_regs) else 0
        next_offset = offsets[idx + 1] if idx + 1 < len(task_regs) else None

        tails = enable_npu_units(next_offset, next_task_regs_len)
        for i, qword in enumerate(tails):
            npu_regcmd[base + len(regs) + i] = qword

        npu_tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)
        npu_tasks[idx].regcfg_amount = len(regs) + len(tails)
        npu_tasks[idx].op_idx = 1               # downstream raw conv task descriptor op index
        npu_tasks[idx].enable_mask = 0xd        # downstream raw conv task descriptor mask
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9) # PC_INTERRUPT_MASK_DPU_0 | DPU_1
        npu_tasks[idx].int_clear = 0x1ffff      # downstream RKNPU_INT_CLEAR clears all status bits

def run_conv2d(batch, in_c, out_c, kh, kw, input_hw, groups=1, weight_in_c=None, stride=1):
    in_h, in_w = input_hw
    weight_in_c = weight_in_c or (in_c // groups)
    p = _conv_params(1, in_c, in_h, in_w, out_c, kh, kw, groups, stride=stride)
    is_spatial = (kh != 1 or kw != 1)
    out_h, out_w = p["out_h"], p["out_w"]

    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, weight_in_c, kh, kw)).astype(np.float16)
    grouped_spatial = _is_grouped_spatial(in_c, out_c, kh, kw, groups)
    grouped_serial = is_spatial and groups > 1 and not p["is_depthwise"]
    spatial_oc_serial = is_spatial and groups == 1 and not p["is_depthwise"] and out_c > UNPACK_C2 and in_c % 16 != 0
    depthwise_spatial_tiled = p["is_depthwise"] and is_spatial and (out_h > 32 or out_c >= 256)

    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
    if not grouped_serial and not spatial_oc_serial and not depthwise_spatial_tiled:
        if _is_depthwise(in_c, out_c, groups):
            weight_full = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
            for oc in range(out_c):
                weight_full[oc, oc] = weight_nchw[oc, 0]
        else:
            weight_full = _expand_grouped_weights(weight_nchw, in_c, out_c, kh, kw, groups)
        if grouped_spatial:
            weight_full = _reorder_grouped_spatial_weights_block16(weight_full, out_c, in_c, kh, kw)

        if p["is_depthwise"] and in_c == 32 and out_c == 32 and not is_spatial:
            wt_flat = np.zeros((kh * kw * _align_up(in_c, p["align_c"]) * out_c), dtype=np.float16)
            wt_flat[:out_c] = weight_nchw[:, 0, 0, 0]
        else:
            wt_flat = pack_conv_weights_for_shape(weight_full, out_c, in_c, kh, kw, p["align_c"], groups)
        ctypes.memmove(wt_ptr, wt_flat.ctypes.data, wt_flat.nbytes)

    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)
    def read_output_fp16(count):
        return np.frombuffer(output_map, dtype=np.uint16, count=count).copy().view(np.float16)

    if grouped_serial:
        assert in_c % groups == 0 and out_c % groups == 0, "invalid grouped channel counts"
        input_per_group = in_c // groups
        out_per_group = out_c // groups
        gp = _conv_params(1, input_per_group, in_h, in_w, out_per_group, kh, kw, 1, stride=stride)
        group_out_c1 = _ceil_div(gp["align_out_c"], UNPACK_C2)
        group_out_count = group_out_c1 * gp["out_h"] * gp["out_w"] * UNPACK_C2
        wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))

        for n in range(batch):
            for g in range(groups):
                input_start = g * input_per_group
                input_end = input_start + input_per_group
                oc_start = g * out_per_group
                oc_end = oc_start + out_per_group
                input_tile = input_nchw[n, input_start:input_end]
                input_flat = _pack_conv_input_fp16(input_tile, gp).view(np.uint16).tolist()
                ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
                ct_inputs[:] = input_flat

                group_weight = weight_nchw[oc_start:oc_end].reshape(out_per_group, input_per_group, kh, kw)
                wt_flat = pack_conv_weights_for_shape(group_weight, out_per_group, input_per_group, kh, kw, gp["align_c"], 1)
                ctypes.memmove(wt_ptr, wt_flat.ctypes.data, wt_flat.nbytes)
                ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)

                task_regs = [make_conv2d_regs(
                    1, input_per_group, in_h, in_w, out_per_group, kh, kw,
                    input_mem_create.dma_addr,
                    weight_mem_create.dma_addr,
                    output_mem_create.dma_addr,
                    groups=1,
                    stride=stride)]

                write_regs_to_npu_task(task_regs)
                npu_submit(task_count=len(task_regs))
                out_buf = read_output_fp16(group_out_count)
                result[n, oc_start:oc_end] = _unpack_nc1hwc2_output(
                    out_buf, out_per_group, out_h, out_w, UNPACK_C2)
        return result, input_nchw, weight_nchw

    if spatial_oc_serial:
        oc_tile = UNPACK_C2
        wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
        for n in range(batch):
            for out_row_start in range(0, out_h, 32):
                tile_out_h = min(32, out_h - out_row_start)
                tile_in_h = (tile_out_h - 1) * stride + kh
                input_tile = input_nchw[n, :, out_row_start * stride:out_row_start * stride + tile_in_h, :]

                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    tile_p = _conv_params(1, in_c, tile_in_h, in_w, tile_out_c, kh, kw, 1, stride=stride)
                    input_flat = _pack_conv_input_fp16(input_tile, tile_p).view(np.uint16).tolist()
                    ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
                    ct_inputs[:] = input_flat

                    tile_weight = weight_nchw[oc_start:oc_end].reshape(tile_out_c, in_c, kh, kw)
                    wt_flat = pack_conv_weights_for_shape(tile_weight, tile_out_c, in_c, kh, kw, tile_p["align_c"], 1)
                    ctypes.memmove(wt_ptr, wt_flat.ctypes.data, wt_flat.nbytes)
                    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)

                    task_regs = [make_conv2d_regs(
                        1, in_c, tile_in_h, in_w, tile_out_c, kh, kw,
                        input_mem_create.dma_addr,
                        weight_mem_create.dma_addr,
                        output_mem_create.dma_addr,
                        groups=1,
                        stride=stride,
                        full_data_bank=True)]

                    write_regs_to_npu_task(task_regs)
                    npu_submit(task_count=len(task_regs))
                    out_c1 = _ceil_div(tile_p["align_out_c"], UNPACK_C2)
                    out_buf = read_output_fp16(out_c1 * tile_p["out_width_stride"] * UNPACK_C2)
                    result[n, oc_start:oc_end, out_row_start:out_row_start + tile_out_h] = _unpack_nc1hwc2_output(
                        out_buf, tile_out_c, tile_out_h, out_w, UNPACK_C2, tile_p["out_width_stride"])
        return result, input_nchw, weight_nchw

    if depthwise_spatial_tiled:
        channel_tile = min(32, out_c)
        row_tile_h = _depthwise_tile_h(out_c)
        for n in range(batch):
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                tile_in_h = (tile_out_h - 1) * stride + kh
                for ch_start in range(0, out_c, channel_tile):
                    ch_end = min(ch_start + channel_tile, out_c)
                    tile_c = ch_end - ch_start
                    tile_p = _conv_params(1, tile_c, tile_in_h, in_w, tile_c, kh, kw, tile_c, stride=stride)
                    input_tile = input_nchw[n, ch_start:ch_end, out_row_start * stride:out_row_start * stride + tile_in_h, :]
                    input_flat = _pack_conv_input_fp16(input_tile, tile_p).view(np.uint16).tolist()
                    ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
                    ct_inputs[:] = input_flat

                    tile_weight = np.zeros((tile_c, tile_c, kh, kw), dtype=np.float16)
                    for local_c in range(tile_c):
                        tile_weight[local_c, local_c] = weight_nchw[ch_start + local_c, 0]
                    wt_flat = pack_conv_weights_for_shape(tile_weight, tile_c, tile_c, kh, kw, tile_p["align_c"], tile_c)
                    ctypes.memmove(wt_ptr, wt_flat.ctypes.data, wt_flat.nbytes)
                    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)

                    task_regs = [make_conv2d_regs(
                        1, tile_c, tile_in_h, in_w, tile_c, kh, kw,
                        input_mem_create.dma_addr,
                        weight_mem_create.dma_addr,
                        output_mem_create.dma_addr,
                        groups=tile_c,
                        stride=stride)]

                    write_regs_to_npu_task(task_regs)
                    npu_submit(task_count=len(task_regs))
                    out_c1 = _ceil_div(tile_p["align_out_c"], UNPACK_C2)
                    out_buf = read_output_fp16(out_c1 * tile_p["out_width_stride"] * UNPACK_C2)
                    result[n, ch_start:ch_end, out_row_start:out_row_start + tile_out_h] = _unpack_nc1hwc2_output(
                        out_buf, tile_c, tile_out_h, out_w, UNPACK_C2, tile_p["out_width_stride"])
        return result, input_nchw, weight_nchw

    if _needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
        oc_tile = 32
        row_tile_h = _pointwise_oc_tile_h(in_c)
        for n in range(batch):
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                input_tile = input_nchw[n, :, out_row_start * stride:out_row_start * stride + tile_out_h, :]

                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    tile_p = _conv_params(1, in_c, tile_out_h, in_w, tile_out_c, kh, kw, groups, stride=stride)
                    input_flat = _pack_conv_input_fp16(input_tile, tile_p).view(np.uint16).tolist()
                    ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
                    ct_inputs[:] = input_flat

                    tile_weight = weight_nchw[oc_start:oc_end].reshape(tile_out_c, in_c, kh, kw)
                    wt_flat = pack_conv_weights_for_shape(tile_weight, tile_out_c, in_c, kh, kw, tile_p["align_c"], groups)
                    ctypes.memmove(wt_ptr, wt_flat.ctypes.data, wt_flat.nbytes)
                    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)

                    task_regs = [make_conv2d_regs(
                        1, in_c, tile_out_h, in_w, tile_out_c, kh, kw,
                        input_mem_create.dma_addr,
                        weight_mem_create.dma_addr,
                        output_mem_create.dma_addr,
                        groups=groups,
                        stride=stride,
                        full_data_bank=True)]

                    write_regs_to_npu_task(task_regs)
                    npu_submit(task_count=len(task_regs))
                    out_c1 = _ceil_div(tile_p["align_out_c"], UNPACK_C2)
                    out_buf = read_output_fp16(out_c1 * tile_p["out_width_stride"] * UNPACK_C2)
                    result[n, oc_start:oc_end, out_row_start:out_row_start + tile_out_h] = _unpack_flat_1x1_output(
                        out_buf, tile_out_c, tile_out_h, out_w, tile_p["out_width_stride"], UNPACK_C2)
        return result, input_nchw, weight_nchw

    for n in range(batch):
        if _needs_pointwise_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
            ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)
            input_ptr = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
            task_regs = []
            input_offset = 0
            tile_h = _pointwise_tile_h(in_c, out_c, out_h, out_w)
            for row_start in range(0, out_h, tile_h):
                tile_out_h = min(tile_h, out_h - row_start)
                tile_in_h = (tile_out_h - 1) * stride + kh
                tile_p = p if tile_in_h == in_h else _conv_params(1, in_c, tile_in_h, in_w, out_c, kh, kw, groups, stride=stride)
                input_tile = input_nchw[n, :, row_start * stride:row_start * stride + tile_in_h, :]
                input_flat = _pack_conv_input_fp16(input_tile, tile_p).view(np.uint16)
                input_bytes = input_flat.nbytes
                assert input_offset + input_bytes <= input_mem_create.size, "input buffer too small"
                ctypes.memmove(input_ptr + input_offset, input_flat.ctypes.data, input_bytes)
                output_offset = row_start * out_w * 16
                regs = make_conv2d_regs(
                    1, in_c, tile_in_h, in_w, out_c, kh, kw,
                    input_mem_create.dma_addr + input_offset,
                    weight_mem_create.dma_addr,
                    output_mem_create.dma_addr + output_offset,
                    groups=groups,
                    stride=stride,
                    out_width_stride_override=p["out_width_stride"],
                    weight_reuse=bool(task_regs),
                    full_data_bank=True)
                task_regs.append(regs)
                input_offset = _align_up(input_offset + input_bytes, 16)
            write_regs_to_npu_task(task_regs)
            npu_submit(task_count=len(task_regs))
            out_c1 = _ceil_div(p["align_out_c"], UNPACK_C2)
            out_buf = read_output_fp16(out_c1 * p["out_width_stride"] * UNPACK_C2)
            result[n] = _unpack_flat_1x1_output(out_buf, out_c, out_h, out_w, p["out_width_stride"], UNPACK_C2)
            continue

        if not is_spatial:
            small_chan = p["use_nhwc"] and not p["is_depthwise"]
            tile_h = max(1, RK_MAX_CONV_FLAT_STRIDE // out_w) if (small_chan and _align_up(out_h * out_w, 4) > RK_MAX_CONV_FLAT_STRIDE) else out_h
            tiles = [(row, row * stride, min(tile_h, out_h - row)) for row in range(0, out_h, tile_h)]
        else:
            tiles = [(0, 0, out_h)]

        for out_row_start, input_row_start, tile_out_rows in tiles:
            tile_in_h = in_h if is_spatial else (tile_out_rows - 1) * stride + kh
            tile_p = p if tile_in_h == in_h else _conv_params(1, in_c, tile_in_h, in_w, out_c, kh, kw, groups, stride=stride)
            tile_out_h = tile_p["out_h"]
            input_tile = input_nchw[n, :, input_row_start:input_row_start + tile_in_h, :]
            input_flat = _pack_conv_input_fp16(input_tile, tile_p).view(np.uint16).tolist()
            ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
            ct_inputs[:] = input_flat

            task_regs = [make_conv2d_regs(
                1, in_c, tile_in_h, in_w, out_c, kh, kw,
                input_mem_create.dma_addr,
                weight_mem_create.dma_addr,
                output_mem_create.dma_addr,
                groups=groups,
                stride=stride)]

            write_regs_to_npu_task(task_regs)
            npu_submit(task_count=len(task_regs))

            out_c1 = _ceil_div(out_c, UNPACK_C2) if grouped_spatial else _ceil_div(tile_p["align_out_c"], UNPACK_C2)
            if not is_spatial:
                out_count = out_c1 * tile_p["out_width_stride"] * UNPACK_C2
                out_buf = read_output_fp16(out_count)
                result[n, :, out_row_start:out_row_start + tile_out_h, :] = _unpack_flat_1x1_output(
                    out_buf, out_c, tile_out_h, out_w, tile_p["out_width_stride"], UNPACK_C2)
                continue
            if grouped_spatial:
                plane_stride = out_h * out_w * UNPACK_C2 + p["out_width_stride"] * 2
                out_buf = read_output_fp16(out_c1 * plane_stride)
                result[n] = _unpack_grouped_spatial_output(out_buf, out_c, out_h, out_w, UNPACK_C2, plane_stride)
            else:
                out_buf = read_output_fp16(out_c1 * tile_p["out_width_stride"] * UNPACK_C2)
                result[n] = _unpack_nc1hwc2_output(out_buf, out_c, out_h, out_w, UNPACK_C2, tile_p["out_width_stride"])
    return result, input_nchw, weight_nchw

def compute_expected_nchw(input_nchw, weight_nchw, batch, in_c, in_h, in_w, out_c, kh, kw, groups=1, stride=1):
    out_h, out_w = (in_h - kh) // stride + 1, (in_w - kw) // stride + 1
    i64, w64 = input_nchw.astype(np.float64), weight_nchw.astype(np.float64)
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
        
        dict(name="b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=112, in_w=112, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid", batch=1, in_c=144, in_h=56, in_w=56, out_c=144, weight_in_c=1, kh=3, kw=3, groups=144, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid", batch=1, in_c=192, in_h=28, in_w=28, out_c=96, weight_in_c=192, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=14, in_w=14, out_c=64, weight_in_c=32, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=256, weight_in_c=528, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=160, weight_in_c=528, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid", batch=1, in_c=160, in_h=14, in_w=14, out_c=320, weight_in_c=160, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=32, weight_in_c=528, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=14, in_w=14, out_c=128, weight_in_c=32, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=128, weight_in_c=528, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid", batch=1, in_c=160, in_h=7, in_w=7, out_c=320, weight_in_c=160, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=7, in_w=7, out_c=128, weight_in_c=32, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid", batch=1, in_c=192, in_h=7, in_w=7, out_c=384, weight_in_c=192, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c832_h7_w7_oc48_wic832_k1x1_g1_s1_pvalid", batch=1, in_c=832, in_h=7, in_w=7, out_c=48, weight_in_c=832, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid", batch=1, in_c=32, in_h=150, in_w=150, out_c=32, weight_in_c=1, kh=3, kw=3, groups=32, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid", batch=1, in_c=32, in_h=150, in_w=150, out_c=16, weight_in_c=32, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid", batch=1, in_c=16, in_h=150, in_w=150, out_c=96, weight_in_c=16, kh=1, kw=1, groups=1, known_reason="buffer size error"),
        dict(name="b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=150, in_w=150, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c96_h75_w75_oc24_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=75, in_w=75, out_c=24, weight_in_c=96, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c144_h75_w75_oc144_wic1_k3x3_g144_s1_pvalid", batch=1, in_c=144, in_h=75, in_w=75, out_c=144, weight_in_c=1, kh=3, kw=3, groups=144, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid", batch=1, in_c=144, in_h=75, in_w=75, out_c=24, weight_in_c=144, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid", batch=1, in_c=144, in_h=38, in_w=38, out_c=32, weight_in_c=144, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c192_h38_w38_oc192_wic1_k3x3_g192_s1_pvalid", batch=1, in_c=192, in_h=38, in_w=38, out_c=192, weight_in_c=1, kh=3, kw=3, groups=192, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid", batch=1, in_c=192, in_h=38, in_w=38, out_c=32, weight_in_c=192, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c384_h19_w19_oc384_wic1_k3x3_g384_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=384, weight_in_c=1, kh=3, kw=3, groups=384, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=64, weight_in_c=384, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=96, weight_in_c=384, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c576_h19_w19_oc576_wic1_k3x3_g576_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=576, weight_in_c=1, kh=3, kw=3, groups=576, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=96, weight_in_c=576, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=12, weight_in_c=576, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=273, weight_in_c=576, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c960_h10_w10_oc960_wic1_k3x3_g960_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=960, weight_in_c=1, kh=3, kw=3, groups=960, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c1280_h10_w10_oc24_wic1280_k1x1_g1_s1_pvalid", batch=1, in_c=1280, in_h=10, in_w=10, out_c=24, weight_in_c=1280, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid", batch=1, in_c=1280, in_h=10, in_w=10, out_c=546, weight_in_c=1280, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid", batch=1, in_c=256, in_h=10, in_w=10, out_c=512, weight_in_c=256, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid", batch=1, in_c=128, in_h=5, in_w=5, out_c=256, weight_in_c=128, kh=3, kw=3, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=24, weight_in_c=256, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c256_h3_w3_oc546_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=546, weight_in_c=256, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=128, weight_in_c=256, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=256, weight_in_c=128, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=24, weight_in_c=256, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=546, weight_in_c=256, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=64, weight_in_c=256, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=1, in_w=1, out_c=24, weight_in_c=128, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid", batch=1, in_c=3, in_h=320, in_w=320, out_c=32, weight_in_c=3, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid", batch=1, in_c=32, in_h=160, in_w=160, out_c=8, weight_in_c=32, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid", batch=1, in_c=8, in_h=160, in_w=160, out_c=16, weight_in_c=8, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=160, in_w=160, out_c=128, weight_in_c=16, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=80, in_w=80, out_c=16, weight_in_c=128, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=64, weight_in_c=16, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid", batch=1, in_c=64, in_h=80, in_w=80, out_c=16, weight_in_c=64, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=128, weight_in_c=16, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=128, weight_in_c=16, kh=5, kw=5, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=40, in_w=40, out_c=40, weight_in_c=128, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid", batch=1, in_c=40, in_h=40, in_w=40, out_c=160, weight_in_c=40, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid", batch=1, in_c=160, in_h=40, in_w=40, out_c=40, weight_in_c=160, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid", batch=1, in_c=40, in_h=40, in_w=40, out_c=320, weight_in_c=40, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c320_h40_w40_oc320_wic1_k3x3_g320_s1_pvalid", batch=1, in_c=320, in_h=40, in_w=40, out_c=320, weight_in_c=1, kh=3, kw=3, groups=320, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid", batch=1, in_c=320, in_h=20, in_w=20, out_c=72, weight_in_c=320, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c72_h20_w20_oc576_wic72_k1x1_g1_s1_pvalid", batch=1, in_c=72, in_h=20, in_w=20, out_c=576, weight_in_c=72, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c576_h20_w20_oc576_wic1_k3x3_g576_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=576, weight_in_c=1, kh=3, kw=3, groups=576, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=72, weight_in_c=576, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid", batch=1, in_c=72, in_h=20, in_w=20, out_c=288, weight_in_c=72, kh=3, kw=3, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c288_h20_w20_oc72_wic288_k1x1_g1_s1_pvalid", batch=1, in_c=288, in_h=20, in_w=20, out_c=72, weight_in_c=288, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c576_h20_w20_oc576_wic1_k5x5_g576_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=576, weight_in_c=1, kh=5, kw=5, groups=576, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=96, weight_in_c=576, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=768, weight_in_c=1, kh=5, kw=5, groups=768, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c768_h20_w20_oc96_wic768_k1x1_g1_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=96, weight_in_c=768, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c768_h20_w20_oc768_wic1_k3x3_g768_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=768, weight_in_c=1, kh=3, kw=3, groups=768, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c768_h10_w10_oc120_wic768_k1x1_g1_s1_pvalid", batch=1, in_c=768, in_h=10, in_w=10, out_c=120, weight_in_c=768, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c960_h10_w10_oc120_wic960_k1x1_g1_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=120, weight_in_c=960, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c480_h10_w10_oc480_wic1_k5x5_g480_s1_pvalid", batch=1, in_c=480, in_h=10, in_w=10, out_c=480, weight_in_c=1, kh=5, kw=5, groups=480, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c480_h10_w10_oc120_wic480_k1x1_g1_s1_pvalid", batch=1, in_c=480, in_h=10, in_w=10, out_c=120, weight_in_c=480, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c960_h10_w10_oc960_wic1_k5x5_g960_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=960, weight_in_c=1, kh=5, kw=5, groups=960, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid", batch=1, in_c=256, in_h=10, in_w=10, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c128_h3_w3_oc128_wic1_k3x3_g128_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=128, weight_in_c=1, kh=3, kw=3, groups=128, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=2, in_w=2, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", batch=1, in_c=64, in_h=1, in_w=1, out_c=128, weight_in_c=64, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c384_h10_w10_oc384_wic1_k3x3_g384_s1_pvalid", batch=1, in_c=384, in_h=10, in_w=10, out_c=384, weight_in_c=1, kh=3, kw=3, groups=384, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c512_h5_w5_oc512_wic1_k3x3_g512_s1_pvalid", batch=1, in_c=512, in_h=5, in_w=5, out_c=512, weight_in_c=1, kh=3, kw=3, groups=512, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c256_h3_w3_oc256_wic1_k3x3_g256_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c96_h20_w20_oc12_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=12, weight_in_c=96, kh=1, kw=1, groups=1, known_reason="mesa_semantics (spatial/depthwise with large channels)"),
        dict(name="b1_c96_h20_w20_oc273_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=273, weight_in_c=96, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
        dict(name="b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=10, in_w=10, out_c=546, weight_in_c=384, kh=1, kw=1, groups=1, known_reason="ic>>oc pointwise numerical precision"),
    ]
    # shapes += [
    #     dict(name=f"conv2d_1x3_{n}x{n}_k1", batch=1, in_c=3, in_h=n, in_w=n, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1)
    #     for n in range(2, 400, 2)
    # ]

    def shape_stride(shape):
        return shape.get("stride", 1)

    def shape_out_hw(shape):
        stride = shape_stride(shape)
        return ((shape["in_h"] - shape["kh"]) // stride + 1,
                (shape["in_w"] - shape["kw"]) // stride + 1)

    name_width = max(len(shape["name"]) for shape in shapes)
    in_shape_width = max(len(f"{shape['in_c']}x{shape['in_h']}x{shape['in_w']}") for shape in shapes)
    out_shape_width = max(len(f"{shape['out_c']}x{shape_out_hw(shape)[0]}x{shape_out_hw(shape)[1]}") for shape in shapes)

    for shape in shapes:
        name = shape["name"]
        batch = shape["batch"]
        in_c = shape["in_c"]
        in_h = shape["in_h"]
        in_w = shape["in_w"]
        out_c = shape["out_c"]
        weight_in_c = shape["weight_in_c"]
        kh = shape["kh"]
        kw = shape["kw"]
        groups = shape["groups"]
        stride = shape_stride(shape)
        result, inp, wt = run_conv2d(batch, in_c, out_c, kh, kw, (in_h, in_w), groups=groups, weight_in_c=weight_in_c, stride=stride)
        expected = compute_expected_nchw(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups=groups, stride=stride)
        md = float(np.max(np.abs(result.astype(np.float64) - expected)))
        ok = np.allclose(result, expected, atol=0.2) and not np.any(np.isinf(result))
        out_h, out_w = shape_out_hw(shape)
        in_shape = f"{in_c}x{in_h}x{in_w}"
        out_shape = f"{out_c}x{out_h}x{out_w}"
        print(f"  {name:<{name_width}s} {in_shape:<{in_shape_width}s} -> {out_shape:<{out_shape_width}s} kh={kh} kw={kw} g={groups} s={stride}  {'PASS' if ok else 'FAIL'}  (max_diff={md:.4f})")
        assert ok, f"{name} failed"
    os.close(fd)
