import os, mmap, ctypes, numpy as np
from fcntl import ioctl

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 1 << 2
FP16_BYTES = 2
FP32_BYTES = 4
FP16_ATOM_ELEMENTS = 16
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_MAX_CONV_FLAT_STRIDE = 992
UNPACK_C2 = FP16_ATOM_ELEMENTS // FP16_BYTES
MIN_CHANNEL_TILE = 32
RK_LINE_STRIDE_GROUP_CAP = 13
RK_MIN_WIDE_FEATURE_GRAINS = 80

class reg:
    CNA  = 0x0201
    CORE = 0x0801
    DPU  = 0x1001
    RDMA = 0x2001
    PC   = 0x0081
    PC_REG = 0x0101
    VERSION = 0x0041

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

    CORE_MISC_CFG          = 0x3010
    CORE_DATAOUT_SIZE_0    = 0x3014
    CORE_DATAOUT_SIZE_1    = 0x3018
    CORE_RESERVED_3030     = 0x3030

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("obj_addr", ctypes.c_uint64),
        ("dma_addr", ctypes.c_uint64),
        ("sram_size", ctypes.c_uint64),
    ]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]

class rknpu_action(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("value", ctypes.c_uint32),
    ]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
    ]

class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("timeout", ctypes.c_uint32),
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
        ("task_counter", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64),
        ("regcfg_obj_addr", ctypes.c_uint64),
        ("task_base_addr", ctypes.c_uint64),
        ("user_data", ctypes.c_uint64),
        ("core_mask", ctypes.c_uint32),
        ("fence_fd", ctypes.c_int32),
        ("subcore_task", rknpu_subcore_task * 5),
    ]

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

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)
DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))

def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    return buf, mem_create

def npu_reset(fd):
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

def npu_submit(fd, task_obj_addr, task_count=1, flags=0x1):
    npu_reset(fd)
    submit_struct = rknpu_submit(
        flags=flags, timeout=6000, task_start=0, task_number=task_count,
        task_counter=0, priority=0, task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0, task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

fd = os.open(f"/dev/dri/card1", os.O_RDWR)
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

# --- helper: GEMM treats a matmul as a 1x1 conv (H=M, W=1, C=K, N kernels) ---

def _gemm_as_conv_params(m, n, k):
    aligned_k = max(MIN_CHANNEL_TILE, _align_up(k, MIN_CHANNEL_TILE))
    align_out = max(MIN_CHANNEL_TILE, _align_up(n, MIN_CHANNEL_TILE))
    return {
        "m": m, "n": n, "k": k,
        "align_in": aligned_k,
        "align_out": align_out,
        "eff_k": aligned_k if aligned_k != aligned_k else k,
        "input_row_bytes": aligned_k * FP16_BYTES,
    }

# --- conv helpers ---

def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c

def _is_kh_major(out_c, in_c, kh, kw, groups):
    is_spatial = (kh != 1 or kw != 1)
    multi_input_group = groups > 1 and in_c != groups
    return is_spatial and not _is_depthwise(in_c, out_c, groups) and not multi_input_group

def _is_grouped_spatial(in_c, out_c, kh, kw, groups):
    is_spatial = (kh != 1 or kw != 1)
    multi_input_group = groups > 1 and in_c != groups
    return is_spatial and multi_input_group and not _is_depthwise(in_c, out_c, groups)

def should_use_nhwc_pack(channels, c2):
    return channels > 0 and c2 // channels == 2

def _conv_align_c(in_c, groups, out_c):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    if groups > 1 and not is_depthwise: return 16
    return max(8, min(1 << (max(1, in_c) - 1).bit_length(), 32 if is_depthwise else 16))

def _conv_input_pack_c2(in_c, groups, out_c, align_c):
    if in_c == 1: return 2
    if _is_depthwise(in_c, out_c, groups) or groups > 1 or in_c == 16: return 8
    return align_c

def _dma_strides(in_h, width_stride, use_nhwc_pack):
    if use_nhwc_pack:
        line_stride = width_stride
        return line_stride, line_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, width_stride * (in_h - 4) if in_h > 4 else 0

def _cbuf_entries(width_stride, align_c, in_h, is_depthwise):
    row_entries = max(1, _ceil_div(width_stride * align_c, 2 * FP16_ATOM_ELEMENTS))
    return row_entries if align_c >= 16 or is_depthwise else row_entries * in_h * 4

def _conv_params(batch, in_c, in_h, in_w, out_c, kh, kw, groups):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    is_spatial = (kh != 1 or kw != 1)
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    align_c = _conv_align_c(in_c, groups, out_c)
    align_out_c = max(16, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_atoms = max(1, out_h * out_w)
    out_width_stride = out_atoms if (not is_spatial and out_atoms < 4) else _align_up(out_atoms, 4)

    input_pack_c2 = _conv_input_pack_c2(in_c, groups, out_c, align_c)
    use_nhwc = (not is_depthwise and not (groups > 1 and is_spatial)
                and should_use_nhwc_pack(in_c, input_pack_c2))
    return {
        "batch": batch, "in_c": in_c, "in_h": in_h, "in_w": in_w,
        "out_c": out_c, "kh": kh, "kw": kw, "groups": groups,
        "is_depthwise": is_depthwise, "out_h": out_h, "out_w": out_w,
        "align_c": align_c, "align_out_c": align_out_c,
        "width_stride": width_stride, "out_width_stride": out_width_stride,
        "input_pack_c2": input_pack_c2, "use_nhwc": use_nhwc,
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

def _pack_dw_spatial_major(weight, out_c, in_c, kh, kw, c2_out):
    packed = np.zeros((out_c, kh, kw, c2_out), dtype=np.float16)
    packed[0, :, :, :out_c] = weight[range(out_c), range(out_c)].transpose(1, 2, 0)
    return packed.ravel()

def pack_conv_weights_for_shape(weight_full, out_c, in_c, kh, kw, align_c, groups):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    if is_depthwise and out_c <= align_c and kh == 3 and kw == 3:
        return _pack_dw_spatial_major(weight_full, out_c, in_c, kh, kw, align_c)
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
    c1 = _ceil_div(in_c, c2)
    padded = np.zeros((c1 * c2, in_h, p["width_stride"]), dtype=np.float16)
    padded[:in_c, :, :in_w] = input_nchw
    return padded.reshape(c1, c2, in_h, p["width_stride"]).transpose(0, 2, 3, 1).ravel()

def _unpack_flat_1x1_output(out_raw, out_c, out_h, out_w, out_width_stride, c2):
    c1 = _ceil_div(out_c, c2)
    packed = out_raw.reshape(1, c1, 1, out_width_stride, c2)
    return packed[0, :, 0, :out_h * out_w, :].transpose(0, 2, 1).reshape(c1 * c2, out_h * out_w)[:out_c].reshape(out_c, out_h, out_w)

def _unpack_nc1hwc2_output(out_raw, out_c, out_h, out_w, c2):
    c1 = _ceil_div(out_c, c2)
    packed = out_raw.reshape(1, c1, out_h, out_w, c2)
    return packed[0].transpose(0, 3, 1, 2).reshape(c1 * c2, out_h, out_w)[:out_c]

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
        pack_order = block.transpose(2, 3, 0, 1).reshape(block_channels, kernel_hw, in_c)
        reordered[block_start:block_end] = pack_order.transpose(0, 2, 1).reshape(block_channels, in_c, kh, kw)
    return reordered

# --- register builders ---

def make_conv2d_regs(batch, in_c, in_h, in_w, out_c, kh, kw, in_dma, wt_dma, out_dma, groups=1):
    p = _conv_params(batch, in_c, in_h, in_w, out_c, kh, kw, groups)
    is_depthwise = p["is_depthwise"]
    is_spatial = (kh != 1 or kw != 1)
    out_h = p["out_h"]
    out_w = p["out_w"]
    align_c = p["align_c"]
    align_out_c = p["align_out_c"]
    width_stride = p["width_stride"]
    out_width_stride = p["out_width_stride"]
    use_nhwc_pack = p["use_nhwc"]
    input_pack_c2 = p["input_pack_c2"]

    data_in_channel_aligned = _align_up(in_c, align_c)
    weight_bytes_per_kernel = kh * kw * data_in_channel_aligned * FP16_BYTES
    weight_bytes_total = weight_bytes_per_kernel * out_c

    input_row_bytes = width_stride * align_c * FP16_BYTES
    even_rows_per_two_banks = (_ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1
    feature_grains = max(in_h + kh, even_rows_per_two_banks)

    line_stride, surf_stride = _dma_strides(in_h, width_stride, use_nhwc_pack)
    cbuf_entries = _cbuf_entries(width_stride, align_c, in_h, is_depthwise)
    data_bank = int(np.clip(_ceil_div(width_stride * feature_grains * align_c * FP16_BYTES, CBUF_BANK_SIZE), 1, RK_CBUF_BANKS - 1))
    out_channel_field = align_out_c - 1 if not is_depthwise else _align_up(align_out_c, 32) - 1
    effective_align_out = max(16, _align_up(_ceil_div(out_c, groups), 16)) if (groups > 1 and not is_depthwise) else out_channel_field + 1
    cvt_data_flag = 0 if use_nhwc_pack else 1

    npu_regs = [
        E(reg.DPU, reg.S_POINTER,
            ((1 << 3) |
             (1 << 2) |
             (1 << 1))),
        E(reg.CNA, reg.CNA_CONV_CON1,
            ((2 << 4) |
             (2 << 7) |
             (((1 << 30) | (1 << 29) | ((7 + in_c) << 12)) if (use_nhwc_pack and in_c <= 4 and not is_depthwise) else 0) |
             (3 if is_depthwise else 0))),
        E(reg.CNA, reg.CNA_CONV_CON2,
            (feature_grains << 4)),
        E(reg.CNA, reg.CNA_CONV_CON3,
            ((1 << 3) |
             (1 << 0))),
        E(reg.CNA, reg.CNA_DATA_SIZE0,
            ((width_stride << 16) |
             in_h)),
        E(reg.CNA, reg.CNA_DATA_SIZE1,
            ((in_c - 1 << 16) |
             data_in_channel_aligned)),
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_w * out_h),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_total),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2,
            ((kw << 24) |
             (kh << 16) |
             (1 if is_depthwise else out_c))),
        E(reg.CNA, reg.CNA_CBUF_CON0,
            ((RK_CBUF_BANKS - data_bank << 4) |
             data_bank)),
        E(reg.CNA, reg.CNA_CBUF_CON1, cbuf_entries),

        E(reg.CNA, reg.CNA_CVT_CON0, ((cvt_data_flag << 3) |
                                       (cvt_data_flag << 1) |
                                        1)),
        E(reg.CNA, reg.CNA_CVT_CON1, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON2, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON3, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON4, (1 << 16)),
        E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0,
            ((15 << 16) |
             15)),
        E(reg.CNA, reg.CNA_DMA_CON1, line_stride),
        E(reg.CNA, reg.CNA_DMA_CON2, surf_stride),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0,
            ((in_w << 16) |
             in_h)),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, align_c),
        E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, (1 << in_c if use_nhwc_pack else input_pack_c2) - 1),
        E(reg.CORE, reg.CORE_MISC_CFG, ((2 << 8) |
                                        (is_depthwise << 1) |
                                        is_spatial)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0,
            (((out_h - 1) << 16) |
             (out_w - 1))),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field),
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG,
            ((15 << 5) |
             ((3 * is_depthwise) << 3) |
             (2 << 1))),
        E(reg.DPU, reg.DATA_FORMAT,
            ((2 << 29) |
             (2 << 26) |
              2)),
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma),
        E(reg.DPU, reg.DST_SURF_STRIDE, out_width_stride << 4),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1),
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),
        E(reg.DPU, reg.DATA_CUBE_NOTCH, 0),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL,
            ((out_c - 1 << 16) |
             out_channel_field)),
        E(reg.DPU, reg.BS_CFG,
            ((1 << 6) |
             (1 << 4) |
             (1 << 1) |
             1)),
        E(reg.DPU, reg.BS_OW_CFG,
            (((3 if is_depthwise else 1) << 8) |
             ((3 if is_depthwise else 1) << 5) |
             ((3 if is_depthwise else 1) << 2) |
             (1 << 1))),
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field),
        E(reg.DPU, reg.WDMA_SIZE_1,
            (((out_h - 1) << 16) |
             (out_w - 1))),
        E(reg.DPU, reg.BN_CFG,
            ((1 << 6) |
             (1 << 4) |
             (1 << 1) |
             1)),
        E(reg.DPU, reg.EW_CFG,
            ((1 << 9) |
             (1 << 8) |
             (1 << 7) |
             (1 << 1) |
             1)),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1),
        E(reg.DPU, reg.OUT_CVT_SCALE,
            ((1 << 16) |
             1)),
        E(reg.DPU, reg.SURFACE_ADD,
            (out_width_stride * (effective_align_out // 8)) << 4),
    ]
    return npu_regs

def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma):
    align_in = max(MIN_CHANNEL_TILE, _align_up(k, MIN_CHANNEL_TILE))
    align_out = max(MIN_CHANNEL_TILE, _align_up(n, MIN_CHANNEL_TILE))
    input_row_bytes = align_in * FP16_BYTES
    eff_k = align_in

    even_rows_per_two_banks = (_ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1
    feature_grains = max(RK_MIN_WIDE_FEATURE_GRAINS, even_rows_per_two_banks)

    data_banks = np.clip(_ceil_div(m * input_row_bytes, CBUF_BANK_SIZE), 1, RK_CBUF_BANKS - 1)
    line_stride = 4 * min(_ceil_div(eff_k, MIN_CHANNEL_TILE), RK_LINE_STRIDE_GROUP_CAP)
    notch_val = 8 * min(align_out // MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP) - 1

    npu_regs = [
        E(reg.DPU,  reg.S_POINTER,
                ((1 << 3) |
                (1 << 2)  |
                (1 << 1))
        ),
        E(reg.CNA,  reg.CNA_CONV_CON1,
                ((2 << 4) |
                (2 << 7)  |
                (1 << 29) )
        ),
        E(reg.CNA,  reg.CNA_CONV_CON2,
                ((feature_grains << 4))
        ),
        E(reg.CNA,  reg.CNA_CONV_CON3,
                ((1 << 3) |
                1)
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE0,
                ((1 << 16) |
                m)
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE1,
                (((align_in - 1) << 16) |
                align_in)
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE2,   1),
        E(reg.CNA,  reg.CNA_DATA_SIZE3,   m),
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE0, input_row_bytes * align_out),
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE1, input_row_bytes),
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE2,
                ((1 << 24) |
                (1 << 16) |
                align_out)
        ),
        E(reg.CNA,  reg.CNA_CBUF_CON0,
                (((RK_CBUF_BANKS - data_banks) << 4) |
                data_banks)
        ),
        E(reg.CNA,  reg.CNA_CBUF_CON1, _ceil_div(align_in, MIN_CHANNEL_TILE)),
        E(reg.CNA,  reg.CNA_CVT_CON0,
                ((1 << 3) |
                (1 << 1) |
                1)
        ),
        E(reg.CNA,  reg.CNA_CVT_CON1,
                (1 << 16)
        ),
        E(reg.CNA,  reg.CNA_CVT_CON2,
                (1 << 16)
        ),
        E(reg.CNA,  reg.CNA_CVT_CON3,
                (1 << 16)
        ),
        E(reg.CNA,  reg.CNA_CVT_CON4,
                (1 << 16)
        ),
        E(reg.CNA,  reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA,  reg.CNA_DMA_CON0,
                ((15 << 16) |
                15)
        ),
        E(reg.CNA,  reg.CNA_DMA_CON1, line_stride),
        E(reg.CNA,  reg.CNA_DMA_CON2, 0),
        E(reg.CNA,  reg.CNA_FC_DATA_SIZE0,
                ((1 << 16) |
                m)
        ),
        E(reg.CNA,  reg.CNA_FC_DATA_SIZE1, align_in),
        E(reg.CNA,  reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CORE, reg.CORE_MISC_CFG,
                ((2 << 8) |
                1)
        ),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0,
                (((m - 1) << 16) |
                 0)
        ),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, align_out - 1),
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU,  reg.FEATURE_MODE_CFG,
                ((15 << 5) |
                (2 << 1))
        ),
        E(reg.DPU,  reg.DATA_FORMAT,
                ((5 << 29) |
                (2 << 26) |
                2)
        ),
        E(reg.DPU,  reg.DST_BASE_ADDR, out_dma),
        E(reg.DPU,  reg.DST_SURF_STRIDE,
                (1 << 4)
        ),
        E(reg.DPU,  reg.DATA_CUBE_WIDTH, 0),
        E(reg.DPU,  reg.DATA_CUBE_HEIGHT, m - 1),
        E(reg.DPU,  reg.DATA_CUBE_NOTCH,
                ((notch_val << 16) |
                notch_val)
        ),
        E(reg.DPU,  reg.DATA_CUBE_CHANNEL,
                (((align_out - 1) << 16) |
                (align_out - 1))
        ),
        E(reg.DPU,  reg.BS_CFG,
                ((1 << 6) |
                (1 << 4)  |
                (1 << 1)  |
                1)
        ),
        E(reg.DPU,  reg.BS_OW_CFG,
                ((3 << 8) |
                (3 << 5)  |
                (3 << 2)  |
                (1 << 1))
        ),
        E(reg.DPU,  reg.WDMA_SIZE_0, align_out - 1),
        E(reg.DPU,  reg.WDMA_SIZE_1,
                (((m - 1) << 16) |
                 0)
        ),
        E(reg.DPU,  reg.BN_CFG,
                ((1 << 6) |
                (1 << 4)  |
                (1 << 1)  |
                1)
        ),
        E(reg.DPU,  reg.EW_CFG,
                ((1 << 9) |
                (1 << 8)  |
                (1 << 7)  |
                (1 << 1)  |
                1)
        ),
        E(reg.DPU,  reg.SURFACE_ADD,
                ((1 * 4) << 4)
        ),
    ]
    return npu_regs

def write_regs_to_npu_task(task_regs, mode="conv"):
    def enable_npu_units(next_offset, next_task_regs_len):
        enable = E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)
        if next_offset is None:
            return [enable]
        next_addr = regcmd_mem_create.dma_addr + next_offset * ctypes.sizeof(ctypes.c_uint64)
        return [
            E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
            E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(next_task_regs_len, 2) + 1),
            E(reg.VERSION, 0, 0),
            enable,
        ]

    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + 4, 2)
    assert offset <= regcmd_mem_create.size // ctypes.sizeof(ctypes.c_uint64), "regcmd buffer too small"

    regcmd_map[:offset * ctypes.sizeof(ctypes.c_uint64)] = b'\x00' * (offset * ctypes.sizeof(ctypes.c_uint64))

    if mode == "gemm":
        op_idx_val = 4
        enable_mask_val = 0x18
    else:
        op_idx_val = 1
        enable_mask_val = 0xd

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
        npu_tasks[idx].regcfg_amount = len(regs)
        npu_tasks[idx].op_idx = op_idx_val
        npu_tasks[idx].enable_mask = enable_mask_val
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9)
        npu_tasks[idx].int_clear = 0x1ffff

# --- conv runner ---

def run_conv2d(batch, in_c, out_c, kh, kw, input_hw, groups=1, weight_in_c=None):
    in_h, in_w = input_hw
    weight_in_c = weight_in_c or (in_c // groups)
    p = _conv_params(1, in_c, in_h, in_w, out_c, kh, kw, groups)
    is_spatial = (kh != 1 or kw != 1)
    out_h, out_w = p["out_h"], p["out_w"]

    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, weight_in_c, kh, kw)).astype(np.float16)

    if _is_depthwise(in_c, out_c, groups):
        weight_full = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
        for oc in range(out_c):
            weight_full[oc, oc] = weight_nchw[oc, 0]
    else:
        weight_full = _expand_grouped_weights(weight_nchw, in_c, out_c, kh, kw, groups)
    grouped_spatial = _is_grouped_spatial(in_c, out_c, kh, kw, groups)
    if grouped_spatial:
        weight_full = _reorder_grouped_spatial_weights_block16(weight_full, out_c, in_c, kh, kw)

    if p["is_depthwise"] and in_c == 32 and out_c == 32 and not is_spatial:
        wt_flat = np.zeros((kh * kw * _align_up(in_c, p["align_c"]) * out_c), dtype=np.float16)
        wt_flat[:out_c] = weight_nchw[:, 0, 0, 0]
    else:
        wt_flat = pack_conv_weights_for_shape(weight_full, out_c, in_c, kh, kw, p["align_c"], groups)
    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
    ctypes.memmove(wt_ptr, wt_flat.ctypes.data, wt_flat.nbytes)

    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)
    def read_output_fp16(count):
        return np.frombuffer(output_map, dtype=np.uint16, count=count).copy().view(np.float16)

    for n in range(batch):
        if not is_spatial:
            small_chan = in_c <= 4 and not p["is_depthwise"]
            tile_h = max(1, RK_MAX_CONV_FLAT_STRIDE // out_w) if (small_chan and _align_up(out_h * out_w, 4) > RK_MAX_CONV_FLAT_STRIDE) else out_h
            tiles = [(row, min(tile_h, out_h - row)) for row in range(0, out_h, tile_h)]
        else:
            tiles = [(0, in_h)]

        for row_start, tile_in_h in tiles:
            tile_p = p if tile_in_h == in_h else _conv_params(1, in_c, tile_in_h, in_w, out_c, kh, kw, groups)
            tile_out_h = tile_p["out_h"]
            input_tile = input_nchw[n, :, row_start:row_start + tile_in_h, :]
            input_flat = _pack_conv_input_fp16(input_tile, tile_p).view(np.uint16).tolist()
            ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
            ct_inputs[:] = input_flat

            task_regs = [make_conv2d_regs(
                1, in_c, tile_in_h, in_w, out_c, kh, kw,
                input_mem_create.dma_addr,
                weight_mem_create.dma_addr,
                output_mem_create.dma_addr,
                groups=groups)]

            write_regs_to_npu_task(task_regs, mode="conv")
            npu_submit(fd, tasks_mem_create.obj_addr, task_count=len(task_regs),
                       flags=(RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG))

            out_c1 = _ceil_div(out_c, UNPACK_C2)
            if not is_spatial:
                out_count = out_c1 * tile_p["out_width_stride"] * UNPACK_C2
                out_buf = read_output_fp16(out_count)
                result[n, :, row_start:row_start + tile_out_h, :] = _unpack_flat_1x1_output(
                    out_buf, out_c, tile_out_h, out_w, tile_p["out_width_stride"], UNPACK_C2)
                continue
            if grouped_spatial:
                plane_stride = out_h * out_w * UNPACK_C2 + p["out_width_stride"] * 2
                out_buf = read_output_fp16(out_c1 * plane_stride)
                result[n] = _unpack_grouped_spatial_output(out_buf, out_c, out_h, out_w, UNPACK_C2, plane_stride)
            else:
                out_buf = read_output_fp16(out_c1 * out_h * out_w * UNPACK_C2)
                result[n] = _unpack_nc1hwc2_output(out_buf, out_c, out_h, out_w, UNPACK_C2)
    return result, input_nchw, weight_nchw

def compute_expected_nchw(input_nchw, weight_nchw, batch, in_c, in_h, in_w, out_c, kh, kw, groups=1):
    out_h, out_w = in_h - kh + 1, in_w - kw + 1
    i64, w64 = input_nchw.astype(np.float64), weight_nchw.astype(np.float64)
    expected = np.zeros((batch, out_c, out_h, out_w))
    for n in range(batch):
        for g in range(groups):
            for oc in range(g * out_c // groups, (g + 1) * out_c // groups):
                for ic in range(g * in_c // groups, (g + 1) * in_c // groups):
                    for i in range(kh):
                        for j in range(kw):
                            expected[n, oc] += i64[n, ic, i:i+out_h, j:j+out_w] * w64[oc, ic - g * in_c // groups, i, j]
    return expected

# --- gemm runner ---

def _pack_gemm_input_fp16(a_matrix, m, k, align_in):
    packed = np.zeros(align_in * m, dtype=np.float16)
    packed.reshape(m, align_in)[:, :k] = a_matrix[:, :k]
    return packed.view(np.uint16).tolist()

def _pack_gemm_weights_fp16(b_matrix, n, k, align_in, align_out):
    weight = np.zeros((align_out, align_in), dtype=np.float16)
    weight_packed = np.zeros(align_out * align_in, dtype=np.float16)
    weight[:n, :k] = b_matrix.T[:n, :k]
    weight_packed[:] = weight.reshape(align_out // 16, 16, align_in // 32, 32).transpose(0, 2, 1, 3).ravel()
    return weight_packed.view(np.uint16).tolist()

def _unpack_gemm_output_fp32(output_raw, m, n, align_out):
    row_start = np.arange(m) * align_out
    return output_raw[row_start[:, None] + np.arange(n)]

def run_gemm(m, n, k, a_matrix, b_matrix):
    align_in = max(MIN_CHANNEL_TILE, _align_up(k, MIN_CHANNEL_TILE))
    align_out = max(MIN_CHANNEL_TILE, _align_up(n, MIN_CHANNEL_TILE))
    input_row_bytes = align_in * FP16_BYTES
    output_row_bytes = align_out * FP32_BYTES

    input_packed = _pack_gemm_input_fp16(a_matrix, m, k, align_in)
    weight_packed = _pack_gemm_weights_fp16(b_matrix, n, k, align_in, align_out)

    ct_inputs = (ctypes.c_uint16 * len(input_packed)).from_buffer(input_map)
    ct_weights = (ctypes.c_uint16 * len(weight_packed)).from_buffer(weight_map)
    ct_inputs[:] = input_packed
    ct_weights[:] = weight_packed

    task_regs = []
    m_tile = 10 * CBUF_BANK_SIZE // input_row_bytes if align_in <= 12 * 32 else 1
    for start in range(0, m, m_tile):
        tile_m = min(m_tile, m - start)
        tiled_input_dma = input_mem_create.dma_addr + start * input_row_bytes
        tiled_output_dma = output_mem_create.dma_addr + start * output_row_bytes
        task_regs.append(make_gemm_regs(tile_m, n, k, tiled_input_dma, weight_mem_create.dma_addr, tiled_output_dma))
    assert len(task_regs) <= tasks_mem_create.size // ctypes.sizeof(struct_rknpu_task), "task buffer too small"

    write_regs_to_npu_task(task_regs, mode="gemm")
    npu_submit(fd, tasks_mem_create.obj_addr, task_count=len(task_regs),
           flags=(RKNPU_JOB_PC | RKNPU_JOB_PINGPONG))

    out_nbytes = max(256, ((m - 1) * align_out + n) * FP32_BYTES)
    output = np.frombuffer(output_map, dtype=np.float32, count=out_nbytes // 4).copy()

    return _unpack_gemm_output_fp32(output, m, n, align_out)

if __name__ == "__main__":
    import sys
    run_conv = len(sys.argv) <= 1 or "conv" in sys.argv[1:]
    run_gemm_tests = len(sys.argv) <= 1 or "gemm" in sys.argv[1:]

    # --- GEMM tests ---
    if run_gemm_tests:
        print("=== GEMM tests ===")
        test_cases = [
            (2, 2, 1,
             np.array([[1], [3]], dtype=np.float16),
             np.array([[5, 6]], dtype=np.float16)),
        ]
        np.random.seed(42)
        for size in range(2, 520, 2):
            m = n = k = size
            a = np.random.randn(m, k).astype(np.float16)
            b = np.random.randn(k, n).astype(np.float16)
            test_cases.append((m, n, k, a, b))

        for m, n, k, a, b in test_cases:
            print(f"\n{m}x{n}x{k}:")
            r = run_gemm(m, n, k, a, b)
            if r is None:
                continue
            expected = a @ b
            ok = np.allclose(r, expected, atol=0.1)
            md = np.max(np.abs(r - expected))
            print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")
            assert ok, f"gemm shape {m}x{n}x{k} failed"

    # --- CONV tests ---
    if run_conv:
        print("\n=== CONV tests ===")
        shapes = [
            ("conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1", 1, 16, 18, 18, 16, 16, 3, 3, 1),
            ("conv2d_b1_c4_h9_w9_oc4_wic4_k1x1_g1", 1, 4, 9, 9, 4, 4, 1, 1, 1),
            ("conv2d_b1_c16_h32_w32_oc16_wic16_k1x1_g1", 1, 16, 32, 32, 16, 16, 1, 1, 1),
            ("conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1", 1, 4, 9, 9, 4, 4, 3, 3, 1),
            ("conv2d_b2_c4_h9_w9_oc4_wic4_k3x3_g1", 2, 4, 9, 9, 4, 4, 3, 3, 1),
            ("conv2d_b1_c1_h5_w7_oc6_wic1_k3x3_g1", 1, 1, 5, 7, 6, 1, 3, 3, 1),
            ("conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2", 1, 4, 1, 1, 2, 2, 1, 1, 2),
            ("conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32", 1, 32, 32, 32, 32, 1, 1, 1, 32),
            ("conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5", 1, 15, 5, 5, 35, 3, 3, 3, 5),
            ("conv2d_b1_c4_h5_w5_oc4_wic2_k3x3_g2", 1, 4, 5, 5, 4, 2, 3, 3, 2),
            ("conv2d_b1_c4_h5_w5_oc8_wic2_k3x3_g2", 1, 4, 5, 5, 8, 2, 3, 3, 2),
            ("conv2d_b1_c3_h5_w7_oc6_wic3_k3x3_g1", 1, 3, 5, 7, 6, 3, 3, 3, 1),
            ("conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3", 1, 3, 5, 7, 6, 1, 3, 3, 3),
            ("conv2d_b1_c3_h5_w7_oc6_wic3_k2x1_g1", 1, 3, 5, 7, 6, 3, 2, 1, 1),
            ("conv2d_b1_c3_h5_w7_oc6_wic3_k3x5_g1", 1, 3, 5, 7, 6, 3, 3, 5, 1),
        ]
        shapes += [(f"conv2d_1x3_{n}x{n}_k1", 1, 3, n, n, 6, 3, 1, 1, 1) for n in range(2, 80, 2)]

        name_width = max(len(shape[0]) for shape in shapes)
        for name, batch, in_c, in_h, in_w, out_c, weight_in_c, kh, kw, groups in shapes:
            result, inp, wt = run_conv2d(batch, in_c, out_c, kh, kw, (in_h, in_w), groups=groups, weight_in_c=weight_in_c)
            expected = compute_expected_nchw(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups=groups)
            md = float(np.max(np.abs(result.astype(np.float64) - expected)))
            ok = np.allclose(result, expected, atol=0.2) and not np.any(np.isinf(result))
            out_h = in_h - kh + 1
            out_w = in_w - kw + 1
            in_shape = f"{in_c}x{in_h}x{in_w}"
            out_shape = f"{out_c}x{out_h}x{out_w}"
            print(f"  {name:<{name_width}s} {in_shape:<12s} -> {out_shape:<12s} kh={kh} kw={kw} g={groups}  {'PASS' if ok else 'FAIL'}  (max_diff={md:.4f})")
            assert ok, f"{name} failed"
    os.close(fd)
