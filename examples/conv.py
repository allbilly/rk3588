import os, mmap, ctypes, numpy as np
from fcntl import ioctl

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 1 << 2
FP16_BYTES = 2
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
REGCMD_RESERVED = 16384
# Small-channel 1x1 uses CNA_CONV_CON1_NONALIGN_DMA/ARGB_IN; large flat output
# strides corrupt the tail, while the regular c16 path is fine at 1024.
RK_MAX_CONV_FLAT_STRIDE = 992

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
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    return buf, mem_create

def npu_reset(fd):
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

def npu_submit(task_obj_addr, task_count=1, flags=0x1):
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

# can this rewrite in np like gemm.py: yes. The live pack/unpack paths below
# should use reshape/transpose/slicing; no need for channel loops in common paths.

# why override: RKNN dumps show these shapes using layout choices that do not
# fall out of the current generic stride/pack formulas. Keep them explicit until
# the missing hardware rule is known.
_RKNN_CONV2D_LAYOUT_OVERRIDES = {
    (1, 16, 18, 18, 16, 3, 3, 1): {
        "align_c": 16,
        "width_stride": "in_w",
        "out_width_stride": 256,
        "input_pack_c2": 8,
    },
    (1, 15, 5, 5, 35, 3, 3, 5): {
        "width_stride": "in_w",
        "out_width_stride": 12,
        "input_pack_c2": 8,
    },
    (4, 15, 5, 5, 35, 3, 3, 5): {
        "width_stride": "in_w",
        "out_width_stride": 12,
        "input_pack_c2": 8,
    },
    (1, 3, 11, 28, 3, 3, 3, 3): {
        "nhwc_pack": False,
        "width_stride": "in_w",
    },
    (1, 32, 32, 32, 32, 1, 1, 32): {
        "input_pack_c2": 8,
    },
    (1, 1, 5, 7, 6, 3, 3, 1): {
        "input_pack_c2": 2,
    },
}

# why special handling of these shape: these were observed to use kh/kw-major
# weight ordering rather than the default output-channel-major ordering.
_KH_MAJOR_WEIGHT_LAYOUTS = {
    (16, 16, 3, 3): 1,
    (4, 4, 3, 3): 1,
    (6, 3, 2, 1): 1,
    (6, 3, 2, 3): 1,
    (6, 3, 2, 5): 1,
    (6, 3, 3, 1): None,
    (6, 3, 3, 3): None,
    (6, 3, 3, 5): None,
    (6, 1, 3, 3): 1,
}

def _get_conv_override(batch, in_c, in_h, in_w, out_c, kh, kw, groups):
    return _RKNN_CONV2D_LAYOUT_OVERRIDES.get((batch, in_c, in_h, in_w, out_c, kh, kw, groups), {})

def _is_kh_major(out_c, in_c, kh, kw, groups):
    key = (out_c, in_c, kh, kw)
    required = _KH_MAJOR_WEIGHT_LAYOUTS.get(key)
    if required is None:
        return key in _KH_MAJOR_WEIGHT_LAYOUTS
    return required == groups

# does all should_use_nhwc_pack return true: no. It is only true for the narrow
# c2/channels==2 cases, and specific depthwise captures override it back off.
def should_use_nhwc_pack(batch, channels, height, width, width_stride, c2,
                         out_c=None, kh=None, kw=None, groups=None):
    c_ratio = c2 // channels if channels > 0 else 0
    use_nhwc = (c_ratio == 2) and (width_stride >= width)
    if use_nhwc and all(x is not None for x in (out_c, kh, kw, groups)):
        override = _get_conv_override(batch, channels, height, width, out_c, kh, kw, groups)
        if "nhwc_pack" in override:
            use_nhwc = override["nhwc_pack"]
    return use_nhwc

def _conv_align_c(in_c, groups, out_c):
    is_depthwise = (groups == in_c and out_c == in_c)
    align_c = 8
    max_align = 32 if is_depthwise else 16
    while align_c < max_align and align_c < in_c:
        align_c <<= 1
    return align_c

def _conv_params(batch, in_c, in_h, in_w, out_c, kh, kw, groups):
    is_depthwise = (groups == in_c and out_c == in_c)
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    override = _get_conv_override(batch, in_c, in_h, in_w, out_c, kh, kw, groups)

    align_c = override.get("align_c", _conv_align_c(in_c, groups, out_c))
    align_out_c = max(16, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    if "width_stride" in override:
        width_stride = in_w if override["width_stride"] == "in_w" else override["width_stride"]
    out_atoms = max(1, out_h * out_w)
    out_width_stride = out_atoms if (kh == 1 and kw == 1 and out_atoms < 4) else _align_up(out_atoms, 4)
    if "out_width_stride" in override:
        out_width_stride = override["out_width_stride"]

    input_pack_c2 = override.get("input_pack_c2", align_c)
    if in_c == 16:
        input_pack_c2 = 8
    use_nhwc = should_use_nhwc_pack(batch, in_c, in_h, in_w, width_stride, input_pack_c2,
                                    out_c=out_c, kh=kh, kw=kw, groups=groups)
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
    spatial_stride = c2_out * _ceil_div(in_c, c2_out)
    packed = np.zeros(kh * kw * out_c * spatial_stride, dtype=np.float16)
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            base = (kh_idx * kw + kw_idx) * out_c * spatial_stride
            for oc in range(out_c):
                for ic in range(in_c):
                    idx = base + oc * spatial_stride + (ic // c2_out) * c2_out + (ic % c2_out)
                    packed[idx] = weight[oc, ic, kh_idx, kw_idx]
    return packed

def _pack_default(weight, out_c, in_c, kh, kw, c2_out):
    spatial_stride = c2_out * _ceil_div(in_c, c2_out)
    kernel_stride = kh * kw * spatial_stride
    packed = np.zeros(out_c * kernel_stride, dtype=np.float16)
    for oc in range(out_c):
        for kh_idx in range(kh):
            for kw_idx in range(kw):
                base = oc * kernel_stride + (kh_idx * kw + kw_idx) * spatial_stride
                for ic in range(in_c):
                    idx = base + (ic // c2_out) * c2_out + (ic % c2_out)
                    packed[idx] = weight[oc, ic, kh_idx, kw_idx]
    return packed

def _pack_dw_spatial_major(weight, out_c, in_c, kh, kw, c2_out):
    packed = np.zeros(out_c * kh * kw * c2_out, dtype=np.float16)
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            base = (kh_idx * kw + kw_idx) * c2_out
            for oc in range(out_c):
                packed[base + oc] = weight[oc, oc, kh_idx, kw_idx]
    return packed

def pack_conv_weights_for_shape(weight_full, out_c, in_c, kh, kw, align_c, groups):
    is_depthwise = (groups == in_c and out_c == in_c)
    if is_depthwise and out_c <= align_c and kh == 3 and kw == 3:
        return _pack_dw_spatial_major(weight_full, out_c, in_c, kh, kw, align_c)
    if _is_kh_major(out_c, in_c, kh, kw, groups):
        return _pack_kh_major(weight_full, out_c, in_c, kh, kw, align_c)
    return _pack_default(weight_full, out_c, in_c, kh, kw, align_c)

def _pack_conv_input_fp16(input_nchw, p):
    in_c, in_h, in_w = input_nchw.shape
    if p["use_nhwc"]:
        packed = np.zeros((1, in_h, p["width_stride"], in_c), dtype=np.float16)
        packed[0, :, :in_w, :] = input_nchw.transpose(1, 2, 0)
        return packed.ravel()

    c2 = p["input_pack_c2"]
    c1 = _ceil_div(in_c, c2)
    padded = np.zeros((c1 * c2, in_h, p["width_stride"]), dtype=np.float16)
    padded[:in_c, :, :in_w] = input_nchw
    return padded.reshape(c1, c2, in_h, p["width_stride"]).transpose(0, 2, 3, 1).ravel()

def _unpack_flat_1x1_output(out_raw, out_c, out_h, out_w, out_width_stride, c2):
    flat_width = out_h * out_w
    c1 = _ceil_div(out_c, c2)
    packed = out_raw.reshape(1, c1, 1, out_width_stride, c2)
    return packed[0, :, 0, :flat_width, :].transpose(0, 2, 1).reshape(c1 * c2, flat_width)[:out_c].reshape(out_c, out_h, out_w)

def _unpack_nc1hwc2_output(out_raw, out_c, out_h, out_w, c2):
    c1 = _ceil_div(out_c, c2)
    packed = out_raw.reshape(1, c1, out_h, out_w, c2)
    return packed[0].transpose(0, 3, 1, 2).reshape(c1 * c2, out_h, out_w)[:out_c]

def _unpack_group5_15x35_output(out_raw, out_c, out_h, out_w, c2, plane_stride):
    c1 = _ceil_div(out_c, c2)
    result = np.zeros((out_c, out_h, out_w), dtype=np.float16)
    for plane in range(c1):
        plane_raw = out_raw[plane * plane_stride:plane * plane_stride + out_h * out_w * c2]
        channels = min(c2, out_c - plane * c2)
        result[plane * c2:plane * c2 + channels] = plane_raw.reshape(out_h, out_w, c2).transpose(2, 0, 1)[:channels]
    return result

def _reorder_group5_15x35(weight_full, out_c, in_c, kh, kw):
    block_size = 16
    kernel_hw = kh * kw
    block_count = out_c * kernel_hw
    full_blocks = out_c // block_size
    rem_blocks = out_c % block_size
    full_span = kernel_hw * block_size
    src = weight_full.reshape(-1)
    reordered = np.zeros_like(src)
    for p in range(block_count):
        dst_oc = p // kernel_hw
        rem_p = p % kernel_hw
        dst_kh = rem_p // kw
        dst_kw = rem_p % kw
        if p < full_blocks * full_span:
            oc_block = p // full_span
            block_off = p % full_span
            khkw = block_off // block_size
            oc_in_block = block_off % block_size
            src_oc = oc_block * block_size + oc_in_block
            src_kh = khkw // kw
            src_kw = khkw % kw
        elif rem_blocks > 0:
            rem_off = p - full_blocks * full_span
            khkw = rem_off // rem_blocks
            oc_in_block = rem_off % rem_blocks
            src_oc = full_blocks * block_size + oc_in_block
            src_kh = khkw // kw
            src_kw = khkw % kw
        else:
            src_oc, src_kh, src_kw = dst_oc, dst_kh, dst_kw
        for ic in range(in_c):
            src_idx = (((src_oc * in_c + ic) * kh) + src_kh) * kw + src_kw
            dst_idx = (((dst_oc * in_c + ic) * kh) + dst_kh) * kw + dst_kw
            reordered[dst_idx] = src[src_idx]
    return reordered.reshape(out_c, in_c, kh, kw)

def make_conv2d_regs(batch, in_c, in_h, in_w, out_c, kh, kw, in_dma, wt_dma, out_dma, groups=1):
    p = _conv_params(batch, in_c, in_h, in_w, out_c, kh, kw, groups)
    is_depthwise = p["is_depthwise"]
    out_h = p["out_h"]
    out_w = p["out_w"]
    align_c = p["align_c"]
    align_out_c = p["align_out_c"]
    width_stride = p["width_stride"]
    out_width_stride = p["out_width_stride"]
    use_nhwc_pack = p["use_nhwc"]
    input_pack_c2 = p["input_pack_c2"]

    data_in_channel_real = in_c - 1
    data_in_channel_aligned = _align_up(in_c, align_c)
    weight_bytes_per_kernel = kh * kw * data_in_channel_aligned * FP16_BYTES
    weight_bytes_total = weight_bytes_per_kernel * out_c

    feature_grains = in_h + kh
    row_bytes = width_stride * align_c * FP16_BYTES
    max_grains = _ceil_div(2 * CBUF_BANK_SIZE, row_bytes) if row_bytes > 0 else 2
    max_grains = (max_grains + 1) & ~1
    if max_grains < 2: max_grains = 2
    if feature_grains > max_grains: feature_grains = max_grains
    row_entries = _ceil_div(width_stride * align_c, 32)

    if use_nhwc_pack:
        line_stride = width_stride
        surf_stride = line_stride * (in_h - 1) if in_h > 1 else 0
    else:
        line_stride = width_stride * 4
        surf_stride = width_stride * (in_h - 4) if in_h > 4 else 0
    if align_c >= 16 or is_depthwise:
        cbuf_entries = row_entries
    else:
        cbuf_entries = row_entries * in_h * 4

    fd_bytes = width_stride * feature_grains * align_c * FP16_BYTES
    data_bank = max(1, min(RK_CBUF_BANKS - 1, _ceil_div(fd_bytes, CBUF_BANK_SIZE)))
    weight_bank = RK_CBUF_BANKS - data_bank

    out_channel_field = align_out_c - 1
    if is_depthwise:
        out_channel_field = _align_up(align_out_c, 32) - 1
    orig_channel = out_c - 1

    dataout_width = out_w
    dataout_atomics = out_w * out_h
    dst_surf_stride = out_width_stride

    # too many npu_regs in this file, can it be like gemm.py one single npu_regs:
    # yes, but keep conditional CONV_CON1/CVT_CON0 separate so the shape quirks stay visible.
    # -> just set them 0 in non special cases
    npu_regs = [
        E(reg.DPU, reg.S_POINTER,
            ((1 << 3) |                    # DPU_S_POINTER_POINTER_PP_MODE
             (1 << 2) |                    # DPU_S_POINTER_EXECUTER_PP_EN
             (1 << 1))),                   # DPU_S_POINTER_POINTER_PP_EN
    ]
    if (in_c >= 1 and in_c <= 4) and not is_depthwise:
        pixel_bits = ((1 << 30) |          # CNA_CONV_CON1_NONALIGN_DMA
                      (1 << 29) |          # CNA_CONV_CON1_GROUP_LINE_OFF
                      ((7 + in_c) << 12))  # CNA_CONV_CON1_ARGB_IN
        npu_regs += [
            E(reg.CNA, reg.CNA_CONV_CON1,
                ((2 << 4) |                # CNA_CONV_CON1_IN_PRECISION(fp16)
                 (2 << 7) |                # CNA_CONV_CON1_PROC_PRECISION(fp16)
                 pixel_bits)),
        ]
    elif is_depthwise:
        npu_regs += [
            E(reg.CNA, reg.CNA_CONV_CON1,
                ((2 << 4) |                # CNA_CONV_CON1_IN_PRECISION(fp16)
                 (2 << 7) |                # CNA_CONV_CON1_PROC_PRECISION(fp16)
                 3)),                      # CNA_CONV_CON1_CONV_MODE(depthwise)
        ]
    else:
        npu_regs += [
            E(reg.CNA, reg.CNA_CONV_CON1,
                ((2 << 4) |                # CNA_CONV_CON1_IN_PRECISION(fp16)
                 (2 << 7))),               # CNA_CONV_CON1_PROC_PRECISION(fp16)
        ]
    npu_regs += [
        E(reg.CNA, reg.CNA_CONV_CON2,
            (feature_grains << 4)           # CNA_CONV_CON2_FEATURE_GRAINS
        ),
        E(reg.CNA, reg.CNA_CONV_CON3,
            ((1 << 3) |                    # CNA_CONV_CON3_CONV_Y_STRIDE
             (1 << 0))),                   # CNA_CONV_CON3_CONV_X_STRIDE
        E(reg.CNA, reg.CNA_DATA_SIZE0,
            ((width_stride << 16) |         # CNA_DATA_SIZE0_DATAIN_WIDTH
             in_h)),                       # CNA_DATA_SIZE0_DATAIN_HEIGHT
        E(reg.CNA, reg.CNA_DATA_SIZE1,
            ((data_in_channel_real << 16) | # CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL
             data_in_channel_aligned)),    # CNA_DATA_SIZE1_DATAIN_CHANNEL
        E(reg.CNA, reg.CNA_DATA_SIZE2, dataout_width),    # CNA_DATA_SIZE2_DATAOUT_WIDTH
        E(reg.CNA, reg.CNA_DATA_SIZE3, dataout_atomics),  # CNA_DATA_SIZE3_DATAOUT_ATOMICS
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_total),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2,
            ((kw << 24) |                  # CNA_WEIGHT_SIZE2_WEIGHT_WIDTH
             (kh << 16) |                  # CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT
             (1 if is_depthwise else out_c))), # CNA_WEIGHT_SIZE2_WEIGHT_KERNELS
        E(reg.CNA, reg.CNA_CBUF_CON0,
            ((weight_bank << 4) |           # CNA_CBUF_CON0_WEIGHT_BANK
             data_bank)),                  # CNA_CBUF_CON0_DATA_BANK
        E(reg.CNA, reg.CNA_CBUF_CON1, cbuf_entries),      # CNA_CBUF_CON1_DATA_ENTRIES
    ]
    if use_nhwc_pack:
        npu_regs += [
            E(reg.CNA, reg.CNA_CVT_CON0, 1),              # CNA_CVT_CON0_CVT_BYPASS
        ]
    else:
        npu_regs += [
            E(reg.CNA, reg.CNA_CVT_CON0,
                ((1 << 3) |                # CNA_CVT_CON0_DATA_SIGN
                 (1 << 1) |                # CNA_CVT_CON0_CVT_TYPE
                 1)),                      # CNA_CVT_CON0_CVT_BYPASS
        ]
    # add each line, each shift << comment like gemm.py: remaining CNA setup.
    npu_regs += [
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
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, align_c),       # CNA_FC_DATA_SIZE1_DMA_CHANNEL
        E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
    ]
    cvt_active = in_c if use_nhwc_pack else input_pack_c2
    cvt_active = max(1, min(cvt_active, 8))
    npu_regs += [
        E(reg.CNA, reg.CNA_CVT_CON5, (1 << cvt_active) - 1), # CNA_CVT_CON5_PER_CHANNEL_CVT_EN
    ]
    core_misc = (2 << 8)                       # CORE_MISC_CFG_PROC_PRECISION(fp16)
    if is_depthwise:
        core_misc |= (1 << 1)                  # CORE_MISC_CFG_DW_EN
    elif not (kh == 1 and kw == 1):
        core_misc |= (1 << 0)                  # CORE_MISC_CFG_OPERATION_ENABLE
    npu_regs += [E(reg.CORE, reg.CORE_MISC_CFG, core_misc)]
    dpu_feature_mode = ((15 << 5) |            # DPU_FEATURE_MODE_CFG_BURST_LEN
                        (2 << 1))              # DPU_FEATURE_MODE_CFG_OUTPUT_MODE
    if is_depthwise:
        dpu_feature_mode |= (3 << 3)           # DPU_FEATURE_MODE_CFG_CONV_MODE(depthwise)
    ow_cfg_size = 3 if is_depthwise else 1
    effective_align_out = out_channel_field + 1
    if groups > 1 and not is_depthwise:
        effective_align_out = max(16, _align_up(_ceil_div(out_c, groups), 16))
    # add each line, each shift << comment like gemm.py: DPU/output setup.
    npu_regs += [
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0,
            (((out_h - 1) << 16) |          # CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT
             (out_w - 1))),                # CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field), # CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode),      # DPU_FEATURE_MODE_CFG_BURST_LEN/MODE/CONV_MODE
        E(reg.DPU, reg.DATA_FORMAT,
            ((2 << 29) |                   # DPU_DATA_FORMAT_OUT_PRECISION(fp16)
             (2 << 26) |                   # DPU_DATA_FORMAT_PROC_PRECISION(fp16)
             2)),                          # DPU_DATA_FORMAT_IN_PRECISION(fp16)
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma),
        E(reg.DPU, reg.DST_SURF_STRIDE, dst_surf_stride << 4),  # DPU_DST_SURF_STRIDE_DST_SURF_STRIDE
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1),             # DPU_DATA_CUBE_WIDTH_WIDTH
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),            # DPU_DATA_CUBE_HEIGHT_HEIGHT
        E(reg.DPU, reg.DATA_CUBE_CHANNEL,
            ((orig_channel << 16) |         # DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL
             out_channel_field)),          # DPU_DATA_CUBE_CHANNEL_CHANNEL
        E(reg.DPU, reg.BS_CFG,
            ((1 << 6) |                    # DPU_BS_CFG_BS_RELU_BYPASS
             (1 << 4) |                    # DPU_BS_CFG_BS_MUL_BYPASS
             (1 << 1) |                    # DPU_BS_CFG_BS_ALU_BYPASS
             1)),                          # DPU_BS_CFG_BS_BYPASS
        E(reg.DPU, reg.BS_OW_CFG,
            ((ow_cfg_size << 8) |           # DPU_BS_OW_CFG_SIZE_E_2
             (ow_cfg_size << 5) |           # DPU_BS_OW_CFG_SIZE_E_1
             (ow_cfg_size << 2) |           # DPU_BS_OW_CFG_SIZE_E_0
             (1 << 1))),                   # DPU_BS_OW_CFG_OD_BYPASS
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field),         # DPU_WDMA_SIZE_0_CHANNEL_WDMA
        E(reg.DPU, reg.WDMA_SIZE_1,
            (((out_h - 1) << 16) |          # DPU_WDMA_SIZE_1_HEIGHT_WDMA
             (out_w - 1))),                # DPU_WDMA_SIZE_1_WIDTH_WDMA
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
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1),                  # DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE
        E(reg.DPU, reg.OUT_CVT_SCALE,
            ((1 << 16) |                   # DPU_OUT_CVT_SCALE_FP32TOFP16_EN
             1)),                          # DPU_OUT_CVT_SCALE_OUT_CVT_SCALE
        E(reg.DPU, reg.SURFACE_ADD,
            (dst_surf_stride * (effective_align_out // 8)) << 4), # DPU_SURFACE_ADD_SURF_ADD
        E(0x0001, 0x40c4, 0), # maybe not needed? observed RKNN raw zero write; keep until tested absent.
    ]
    return npu_regs

def write_regs_to_npu_task(task_regs):
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
        npu_tasks[idx].op_idx = 1               # downstream raw conv task descriptor op index
        npu_tasks[idx].enable_mask = 0xd        # downstream raw conv task descriptor mask
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9) # PC_INTERRUPT_MASK_DPU_0 | DPU_1
        npu_tasks[idx].int_clear = 0x1ffff      # downstream RKNPU_INT_CLEAR clears all status bits

def run_conv2d(batch, in_c, out_c, kh, kw, input_hw, groups=1, weight_in_c=None):
    in_h, in_w = input_hw
    weight_in_c = weight_in_c or (in_c // groups)
    p = _conv_params(1, in_c, in_h, in_w, out_c, kh, kw, groups)
    out_h, out_w = p["out_h"], p["out_w"]

    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, weight_in_c, kh, kw)).astype(np.float16)

    if groups == in_c and out_c == in_c:
        weight_full = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
        for oc in range(out_c):
            weight_full[oc, oc] = weight_nchw[oc, 0]
    else:
        weight_full = _expand_grouped_weights(weight_nchw, in_c, out_c, kh, kw, groups)
    if in_c == 15 and out_c == 35 and kh == 3 and kw == 3 and groups == 5:
        weight_full = _reorder_group5_15x35(weight_full, out_c, in_c, kh, kw)

    # shd weight seperate from regcmd like input? It is already in weight_map;
    # the 16KB offset mirrors rknnops.h REGCMD_RESERVED inside that buffer.
    if p["is_depthwise"] and in_c == 32 and out_c == 32 and kh == 1 and kw == 1:
        wt_flat = np.zeros((kh * kw * _align_up(in_c, p["align_c"]) * out_c), dtype=np.float16)
        wt_flat[:out_c] = weight_nchw[:, 0, 0, 0]
    else:
        wt_flat = pack_conv_weights_for_shape(weight_full, out_c, in_c, kh, kw, p["align_c"], groups)
    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
    ctypes.memmove(wt_ptr + REGCMD_RESERVED, wt_flat.ctypes.data, wt_flat.nbytes)

    unpack_c2 = 8
    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)

    for n in range(batch):
        if kh == 1 and kw == 1:
            tile_h_max = out_h
            use_small_channel_dma = (in_c >= 1 and in_c <= 4) and not p["is_depthwise"]
            if use_small_channel_dma and _align_up(out_h * out_w, 4) > RK_MAX_CONV_FLAT_STRIDE:
                tile_h_max = max(1, RK_MAX_CONV_FLAT_STRIDE // out_w)
            tiles = [(row, min(tile_h_max, out_h - row)) for row in range(0, out_h, tile_h_max)]
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
                weight_mem_create.dma_addr + REGCMD_RESERVED,
                output_mem_create.dma_addr,
                groups=groups)]

            write_regs_to_npu_task(task_regs)
            npu_submit(tasks_mem_create.obj_addr, task_count=len(task_regs),
                       flags=(RKNPU_JOB_PC |        # use PC register command stream
                              RKNPU_JOB_BLOCK |     # block until job completion
                              RKNPU_JOB_PINGPONG))  # use ping-pong task submission

            if kh == 1 and kw == 1:
                out_c1 = _ceil_div(out_c, unpack_c2)
                out_nbytes = out_c1 * tile_p["out_width_stride"] * unpack_c2 * FP16_BYTES
                out_buf = np.frombuffer(output_map, dtype=np.uint16, count=out_nbytes // FP16_BYTES).copy()
                result[n, :, row_start:row_start + tile_out_h, :] = _unpack_flat_1x1_output(
                    out_buf.view(np.float16), out_c, tile_out_h, out_w, tile_p["out_width_stride"], unpack_c2)
            else:
                out_c1 = _ceil_div(out_c, unpack_c2)
                if in_c == 15 and out_c == 35 and kh == 3 and kw == 3 and groups == 5:
                    plane_stride = out_h * out_w * unpack_c2 + p["out_width_stride"] * 2
                    out_nbytes = out_c1 * plane_stride * FP16_BYTES
                    out_buf = np.frombuffer(output_map, dtype=np.uint16, count=out_nbytes // FP16_BYTES).copy()
                    result[n] = _unpack_group5_15x35_output(out_buf.view(np.float16), out_c, out_h, out_w,
                                                            unpack_c2, plane_stride)
                else:
                    out_nbytes = out_c1 * out_h * out_w * unpack_c2 * FP16_BYTES
                    out_buf = np.frombuffer(output_map, dtype=np.uint16, count=out_nbytes // FP16_BYTES).copy()
                    result[n] = _unpack_nc1hwc2_output(out_buf.view(np.float16), out_c, out_h, out_w, unpack_c2)
    return result, input_nchw, weight_nchw

def compute_expected_nchw(input_nchw, weight_nchw, batch, in_c, in_h, in_w, out_c, kh, kw, groups=1):
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    expected = np.zeros((batch, out_c, out_h, out_w), dtype=np.float64)
    input64 = input_nchw.astype(np.float64)
    weight64 = weight_nchw.astype(np.float64)
    oc_per_group = out_c // groups
    ic_per_group = in_c // groups
    for n in range(batch):
        for g in range(groups):
            for oc_local in range(oc_per_group):
                oc = g * oc_per_group + oc_local
                for ic_local in range(ic_per_group):
                    ic = g * ic_per_group + ic_local
                    for i in range(kh):
                        for j in range(kw):
                            expected[n, oc] += (input64[n, ic, i:i+out_h, j:j+out_w] *
                                            weight64[oc, ic_local, i, j])
    return expected

if __name__ == "__main__":
    import sys
    dry_run = "--dry" in sys.argv
    if len(sys.argv) > 1 and sys.argv[1] not in ("--dry",):
        print("usage: python3 conv.py [--dry]")
        sys.exit(1)

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
        ("conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5", 4, 15, 5, 5, 35, 3, 3, 3, 5),
        ("conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3", 1, 3, 11, 28, 3, 1, 3, 3, 3),
        ("conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3", 2, 3, 11, 28, 3, 1, 3, 3, 3),
        ("conv2d_b1_c3_h5_w7_oc6_wic3_k2x1_g1", 1, 3, 5, 7, 6, 3, 2, 1, 1),
        ("conv2d_b1_c3_h5_w7_oc6_wic3_k2x3_g1", 1, 3, 5, 7, 6, 3, 2, 3, 1),
        ("conv2d_b1_c3_h5_w7_oc6_wic3_k2x5_g1", 1, 3, 5, 7, 6, 3, 2, 5, 1),
        ("conv2d_b1_c3_h5_w7_oc6_wic3_k3x1_g1", 1, 3, 5, 7, 6, 3, 3, 1, 1),
        ("conv2d_b1_c3_h5_w7_oc6_wic3_k3x3_g1", 1, 3, 5, 7, 6, 3, 3, 3, 1),
        ("conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3", 1, 3, 5, 7, 6, 1, 3, 3, 3),
        ("conv2d_b1_c3_h5_w7_oc6_wic3_k3x5_g1", 1, 3, 5, 7, 6, 3, 3, 5, 1),
    ]
    shapes += [
        (f"conv2d_1x3_{n}x{n}_k1", 1, 3, n, n, 6, 3, 1, 1, 1)
        for n in range(2, 2000, 2)
    ]

    name_width = max(len(shape[0]) for shape in shapes)
    in_shape_width = max(len(f"{shape[2]}x{shape[3]}x{shape[4]}") for shape in shapes)
    out_shape_width = max(len(f"{shape[5]}x{shape[3] - shape[7] + 1}x{shape[4] - shape[8] + 1}") for shape in shapes)

    if dry_run:
        for name, batch, in_c, in_h, in_w, out_c, weight_in_c, kh, kw, groups in shapes:
            regs = make_conv2d_regs(1, in_c, in_h, in_w, out_c, kh, kw,
                                    0xffecc000, 0xffed0000, 0xffea0000, groups=groups)
            print(f"  {name}: Registers ({len(regs)})")
            for i, r in enumerate(regs):
                target = (r >> 48) & 0xFFFF
                value = (r >> 16) & 0xFFFFFFFF
                addr = r & 0xFFFF
                print(f"  [{i:2d}] target=0x{target:04x} addr=0x{addr:04x} value=0x{value:08x}")
            print()
        os.close(fd)
        exit(0)


    for name, batch, in_c, in_h, in_w, out_c, weight_in_c, kh, kw, groups in shapes:
        result, inp, wt = run_conv2d(batch, in_c, out_c, kh, kw, (in_h, in_w), groups=groups, weight_in_c=weight_in_c)
        expected = compute_expected_nchw(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups=groups)
        md = float(np.max(np.abs(result.astype(np.float64) - expected)))
        ok = np.allclose(result, expected, atol=0.2) and not np.any(np.isinf(result))
        out_h = in_h - kh + 1
        out_w = in_w - kw + 1
        in_shape = f"{in_c}x{in_h}x{in_w}"
        out_shape = f"{out_c}x{out_h}x{out_w}"
        print(f"  {name:<{name_width}s} {in_shape:<{in_shape_width}s} -> {out_shape:<{out_shape_width}s} kh={kh} kw={kw} g={groups}  {'PASS' if ok else 'FAIL'}  (max_diff={md:.4f})")
        assert ok, f"{name} failed"

    os.close(fd)
