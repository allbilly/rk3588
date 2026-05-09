import os, mmap, ctypes, numpy as np
from fcntl import ioctl

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 1 << 2
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

def should_use_nhwc_pack(channels, c2):
    return channels > 0 and (c2 // channels == 2 or (channels == 2 and c2 // channels == 4))

def _conv_align_c(in_c, groups, out_c):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    if not is_depthwise and (groups > 1 or in_c > 4): return 16
    return max(8, min(1 << (max(1, in_c) - 1).bit_length(), 32 if is_depthwise else 16))

def _conv_input_pack_c2(in_c, groups, out_c, align_c):
    # C2 selection is RK packing layered on top of 16-value FP16 atoms.
    if in_c == 1: return 2
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
    return np.concatenate([
        padded[oc:oc + 16].transpose(2, 3, 0, 1).ravel()
        for oc in range(0, out_c, 16)
    ])

def _pack_default(weight, out_c, in_c, kh, kw, c2_out):
    aligned_in_c = c2_out * _ceil_div(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:, :in_c] = weight
    return padded.transpose(0, 2, 3, 1).ravel()

def _pack_pointwise_wide(weight, out_c, in_c):
    aligned_in_c = max(32, _align_up(in_c, 32))
    padded = np.zeros((out_c, aligned_in_c), dtype=np.float16)
    padded[:out_c, :in_c] = weight[:, :in_c, 0, 0]
    return np.concatenate([
        padded[oc:oc + 16].reshape(-1, aligned_in_c // 32, 32).transpose(1, 0, 2).ravel()
        for oc in range(0, out_c, 16)
    ])

def _pack_dw_spatial_major(weight, out_c, in_c, kh, kw, c2_out):
    blocks = _ceil_div(out_c, c2_out)
    packed = np.zeros((blocks, kh, kw, c2_out), dtype=np.float16)
    for block in range(blocks):
        start = block * c2_out
        end = min(start + c2_out, out_c)
        channels = np.arange(start, end)
        packed[block, :, :, :end - start] = weight[channels, channels].transpose(1, 2, 0)
    return packed.ravel()

def pack_conv_weights_for_shape(weight_full, out_c, in_c, kh, kw, align_c, groups):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    if is_depthwise and (kh != 1 or kw != 1):
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
    c1 = _ceil_div(in_c, c2)
    padded = np.zeros((c1 * c2, in_h, p["width_stride"]), dtype=np.float16)
    padded[:in_c, :, :in_w] = input_nchw
    return padded.reshape(c1, c2, in_h, p["width_stride"]).transpose(0, 2, 3, 1).ravel()

def _unpack_flat_1x1_output(out_raw, out_c, out_h, out_w, out_width_stride, c2):
    c1 = out_raw.size // (out_width_stride * c2)
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

        # Hardware consumes grouped spatial weights as kh/kw-major within each
        # output-channel block; pack_default consumes oc-major. Pre-shuffle the
        # logical tensor so pack_default serializes the hardware order.
        pack_order = block.transpose(2, 3, 0, 1).reshape(block_channels, kernel_hw, in_c)
        reordered[block_start:block_end] = pack_order.transpose(0, 2, 1).reshape(block_channels, in_c, kh, kw)
    return reordered

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

def _tile_data_bank(p, tile_in_h):
    if p["out_c"] % 16:
        return 1
    if p["out_h"] > 50:
        return RK_CBUF_BANKS - 1
    data_bytes = p["width_stride"] * tile_in_h * _align_up(p["in_c"], p["align_c"]) * FP16_BYTES
    input_channel_floor = _ceil_div(p["in_c"], 64)
    return int(np.clip(max(_ceil_div(data_bytes, CBUF_BANK_SIZE), input_channel_floor), 1, 8))

def _conv_tiles(p, is_spatial, grouped_spatial):
    if not is_spatial:
        small_channel = p["in_c"] <= 4 and not p["is_depthwise"]
        channel_tiled = p["out_c"] == 64 or p["out_c"] >= 128
        if small_channel:
            tile_h = max(1, RK_MAX_CONV_FLAT_STRIDE // p["out_w"]) if p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE else p["out_h"]
        elif p["out_h"] > 50:
            tile_h = 25 if (p["in_c"] >= 128 and p["out_c"] >= 128) else 50
        elif channel_tiled:
            row_bytes = p["width_stride"] * _align_up(p["in_c"], p["align_c"]) * FP16_BYTES
            tile_h = min(p["out_h"], max(1, (8 * CBUF_BANK_SIZE) // max(1, row_bytes)))
        else:
            tile_h = p["out_h"]
        return (not small_channel and (p["out_h"] > 50 or channel_tiled),
                [(row, min(tile_h, p["out_h"] - row)) for row in range(0, p["out_h"], tile_h)])

    if not grouped_spatial and p["out_h"] > 48:
        tile_out_h = 48
        return True, [(row, min(tile_out_h + p["kh"] - 1, p["in_h"] - row)) for row in range(0, p["out_h"], tile_out_h)]

    if p["is_depthwise"] and p["out_c"] > 32:
        if p["out_h"] <= 13:
            return True, [(0, p["in_h"])]
        tile_out_h = 13
        return True, [(row, min(tile_out_h + p["kh"] - 1, p["in_h"] - row)) for row in range(0, p["out_h"], tile_out_h)]

    return False, [(0, p["in_h"])]

def _direct_submit_repeat(p, batch, is_spatial, tile_count):
    if (not is_spatial and p["in_c"] <= 4 and not p["is_depthwise"] and tile_count > 1):
        return 2
    if p["is_depthwise"] and is_spatial:
        return 2
    single_atom_1d_nhwc = (p["use_nhwc"] and p["input_pack_c2"] < 8
                           and p["in_h"] == 1 and p["kh"] == 1)
    return 2 if batch > 1 and single_atom_1d_nhwc else 1

def _with_reg_value(regs, target, reg_addr, value):
    replacement = E(target, reg_addr, value)
    return [replacement if (q & 0xFFFF) == reg_addr and (q >> 48) == target else q for q in regs]

def _without_reg(regs, target, reg_addr):
    return [q for q in regs if not ((q & 0xFFFF) == reg_addr and (q >> 48) == target)]

def _with_cbuf_data_bank(regs, data_bank):
    return _with_reg_value(regs, reg.CNA, reg.CNA_CBUF_CON0, ((RK_CBUF_BANKS - data_bank) << 4) | data_bank)

def make_conv2d_regs(batch, in_c, in_h, in_w, out_c, kh, kw, in_dma, wt_dma, out_dma, groups=1, out_width_stride_override=None, weight_reuse=False, full_data_bank=False):
    p = _conv_params(batch, in_c, in_h, in_w, out_c, kh, kw, groups)
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
    cvt_channel_mask = (1 << in_c if use_nhwc_pack else input_pack_c2) - 1

    data_in_channel_aligned = _align_up(in_c, align_c)
    weight_bytes_per_kernel = kh * kw * data_in_channel_aligned * FP16_BYTES
    weight_total_bytes = weight_bytes_per_kernel if is_depthwise else weight_bytes_per_kernel * out_c
    feature_grains = _feature_grains(width_stride * data_in_channel_aligned * FP16_BYTES, in_h + kh, use_nhwc_pack, is_spatial, is_depthwise)
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
        E(reg.CNA, reg.CNA_CONV_CON1,
            ((2 << 4) |                    # CNA_CONV_CON1_IN_PRECISION(fp16)
             (2 << 7) |                    # CNA_CONV_CON1_PROC_PRECISION(fp16)
             (((1 << 30) | (1 << 29) | ((7 + in_c) << 12)) if (use_nhwc_pack and in_c <= 4 and not is_depthwise) else 0) |
             (3 if is_depthwise else 0))), # CNA_CONV_CON1_CONV_MODE(depthwise when 3)
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
            ((in_c - 1 << 16) |             # CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL
             data_in_channel_aligned)),     # CNA_DATA_SIZE1_DATAIN_CHANNEL
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),                       # CNA_DATA_SIZE2_DATAOUT_WIDTH
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_w * out_h),               # CNA_DATA_SIZE3_DATAOUT_ATOMICS
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_total_bytes),      # CNA_WEIGHT_SIZE0_WEIGHT_TOTAL_SIZE
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel), # CNA_WEIGHT_SIZE1_WEIGHT_PER_KERNEL_SIZE
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2,
            ((kw << 24) |                  # CNA_WEIGHT_SIZE2_WEIGHT_WIDTH
             (kh << 16) |                  # CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT
             (1 if is_depthwise else out_c))), # CNA_WEIGHT_SIZE2_WEIGHT_KERNELS
        E(reg.CNA, reg.CNA_CBUF_CON0,
            ((weight_reuse << 13) |             # CNA_CBUF_CON0_WEIGHT_REUSE
             (RK_CBUF_BANKS - data_bank << 4) | # CNA_CBUF_CON0_WEIGHT_BANK
              data_bank)),                       # CNA_CBUF_CON0_DATA_BANK
        E(reg.CNA, reg.CNA_CBUF_CON1, _cbuf_entries(width_stride, data_in_channel_aligned, in_h, is_depthwise)), # CNA_CBUF_CON1_DATA_ENTRIES
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
        E(reg.CNA, reg.CNA_DMA_CON1, _dma_strides(in_h, width_stride, use_nhwc_pack)[0]),        # CNA_DMA_CON1_LINE_STRIDE
        E(reg.CNA, reg.CNA_DMA_CON2, _dma_strides(in_h, width_stride, use_nhwc_pack)[1]),        # CNA_DMA_CON2_SURF_STRIDE
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0,
            ((in_w << 16) |                # CNA_FC_DATA_SIZE0_DMA_WIDTH
             in_h)),                       # CNA_FC_DATA_SIZE0_DMA_HEIGHT
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, data_in_channel_aligned), # CNA_FC_DATA_SIZE1_DMA_CHANNEL
        E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, cvt_channel_mask), # CNA_CVT_CON5_PER_CHANNEL_CVT_EN
        E(reg.CORE, reg.CORE_MISC_CFG, ((2 << 8) |             # CORE_MISC_CFG_PROC_PRECISION(fp16)
                                        (is_depthwise << 1) |  # CORE_MISC_CFG_DW_EN
                                        is_spatial)),          # CORE_MISC_CFG_OPERATION_ENABLE
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0,
            (((out_h - 1) << 16) |          # CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT
             (out_w - 1))),                # CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field), # CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),  # Must be set 0, otherwise corrupt next run
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
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1),             # DPU_DATA_CUBE_WIDTH_WIDTH
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),            # DPU_DATA_CUBE_HEIGHT_HEIGHT
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
        offset += _align_up(len(regs) + 4, 2)
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
        npu_tasks[idx].regcfg_amount = len(regs)
        npu_tasks[idx].op_idx = 1               # downstream raw conv task descriptor op index
        npu_tasks[idx].enable_mask = 0xd        # downstream raw conv task descriptor mask
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9) # PC_INTERRUPT_MASK_DPU_0 | DPU_1
        npu_tasks[idx].int_clear = 0x1ffff      # downstream RKNPU_INT_CLEAR clears all status bits

def submit_conv_tasks(task_regs, repeat=1):
    for _ in range(repeat):
        write_regs_to_npu_task(task_regs)
        npu_submit(tasks_mem_create.obj_addr, task_count=len(task_regs),
                   flags=(RKNPU_JOB_PC |
                          RKNPU_JOB_BLOCK))

def run_conv2d(batch, in_c, out_c, kh, kw, input_hw, groups=1, weight_in_c=None):
    in_h, in_w = input_hw
    weight_in_c = weight_in_c or (in_c // groups)
    p = _conv_params(1, in_c, in_h, in_w, out_c, kh, kw, groups)
    is_spatial = (kh != 1 or kw != 1)
    out_h, out_w = p["out_h"], p["out_w"]

    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, weight_in_c, kh, kw)).astype(np.float16)
    if groups == 1 and kh == 1 and kw == 1 and in_c >= 64:
        # DO NOT CHEAT CPU OFFLOAD
        pass
    if _is_depthwise(in_c, out_c, groups) and (kh != 1 or kw != 1) and in_c >= 32:
        # DO NOT CHEAT CPU OFFLOAD
        pass
 

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
        pc_chain_tiles, tiles = _conv_tiles(p, is_spatial, grouped_spatial)
        depthwise_channel_tiles = False
        compact_tail = (not is_spatial and out_c % 16 and out_h == 1 and out_w == 1)

        if pc_chain_tiles:
            input_ptr = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
            task_regs = []
            input_offset = 0
            output_channel_tile_size = 512 if compact_tail else 16
            output_channel_tiles = (not is_spatial and (out_c == 64 or out_c >= 128))
            depthwise_channel_tiles = p["is_depthwise"] and out_c >= 32
            if compact_tail:
                warmup_regs = make_conv2d_regs(
                    1, 1, 4, 4, 6, 1, 1,
                    input_mem_create.dma_addr,
                    weight_mem_create.dma_addr,
                    output_mem_create.dma_addr,
                    groups=1)
                submit_conv_tasks([warmup_regs], repeat=2)
            ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)

        for row_start, tile_in_h in tiles:
            tile_p = p if tile_in_h == in_h else _conv_params(1, in_c, tile_in_h, in_w, out_c, kh, kw, groups)
            tile_out_h = tile_p["out_h"]
            input_tile = input_nchw[n, :, row_start:row_start + tile_in_h, :]
            input_flat = _pack_conv_input_fp16(input_tile, tile_p).view(np.uint16)

            if pc_chain_tiles:
                input_bytes = input_flat.nbytes
                assert input_offset + input_bytes <= input_mem_create.size, "input buffer too small"
                if not depthwise_channel_tiles:
                    ctypes.memmove(input_ptr + input_offset, input_flat.ctypes.data, input_bytes)
                output_offset = row_start * out_w * 16
                if depthwise_channel_tiles:
                    depthwise_block_c = 32
                    for ch_start in range(0, out_c, depthwise_block_c):
                        ch_tile = min(depthwise_block_c, out_c - ch_start)
                        block_p = _conv_params(1, ch_tile, tile_in_h, in_w, ch_tile, kh, kw, ch_tile)
                        block_input_flat = _pack_conv_input_fp16(input_tile[ch_start:ch_start + ch_tile], block_p).view(np.uint16)
                        block_input_flat = block_input_flat.tolist()
                        ct_inputs = (ctypes.c_uint16 * len(block_input_flat)).from_buffer(input_map)
                        ct_inputs[:] = block_input_flat
                        block_weight_full = np.zeros((ch_tile, ch_tile, kh, kw), dtype=np.float16)
                        for local_c in range(ch_tile):
                            block_weight_full[local_c, local_c] = weight_nchw[ch_start + local_c, 0]
                        block_weight_flat = pack_conv_weights_for_shape(block_weight_full, ch_tile, ch_tile, kh, kw, block_p["align_c"], ch_tile)
                        ctypes.memmove(wt_ptr, block_weight_flat.ctypes.data, block_weight_flat.nbytes)
                        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)
                        regs = make_conv2d_regs(
                            1, ch_tile, tile_in_h, in_w, ch_tile, kh, kw,
                            input_mem_create.dma_addr,
                            weight_mem_create.dma_addr,
                            output_mem_create.dma_addr,
                            groups=ch_tile,
                            out_width_stride_override=p["out_width_stride"],
                            full_data_bank=True)
                        submit_conv_tasks([regs], repeat=2)
                        out_c1 = _ceil_div(max(16, _align_up(ch_tile, 16)), UNPACK_C2)
                        out_buf = read_output_fp16(out_c1 * p["out_width_stride"] * UNPACK_C2)
                        result[n, ch_start:ch_start + ch_tile, row_start:row_start + tile_out_h, :] = _unpack_flat_1x1_output(
                            out_buf, ch_tile, tile_out_h, out_w, p["out_width_stride"], UNPACK_C2)
                elif output_channel_tiles:
                    aligned_in_c = _align_up(in_c, p["align_c"])
                    for oc_start in range(0, out_c, output_channel_tile_size):
                        oc_tile = min(output_channel_tile_size, out_c - oc_start)
                        weight_offset = oc_start * kh * kw * aligned_in_c * FP16_BYTES
                        surface_offset = (oc_start // 16) * p["out_width_stride"] * 16 * FP16_BYTES
                        regs = make_conv2d_regs(
                            1, in_c, tile_in_h, in_w, oc_tile, kh, kw,
                            input_mem_create.dma_addr + input_offset,
                            weight_mem_create.dma_addr + weight_offset,
                            output_mem_create.dma_addr + output_offset + surface_offset,
                            groups=groups,
                            out_width_stride_override=p["out_width_stride"],
                            full_data_bank=True)
                        regs = _with_cbuf_data_bank(regs, _tile_data_bank(p, tile_in_h))
                        if compact_tail:
                            regs = _with_cbuf_data_bank(regs, 1)
                            regs = _with_reg_value(regs, reg.CNA, reg.CNA_DMA_CON2, (tile_p["width_stride"] * (tile_in_h - 4)) & 0x0fffffff)
                            regs = _without_reg(regs, reg.CNA, reg.CNA_CVT_CON5)
                            regs = _with_reg_value(regs, reg.DPU, reg.SURFACE_ADD, 2 << 4)
                        task_regs.append(regs)
                else:
                    task_regs.append(make_conv2d_regs(
                        1, in_c, tile_in_h, in_w, out_c, kh, kw,
                        input_mem_create.dma_addr + input_offset,
                        weight_mem_create.dma_addr,
                        output_mem_create.dma_addr + output_offset,
                        groups=groups,
                        out_width_stride_override=p["out_width_stride"],
                        weight_reuse=bool(task_regs),
                        full_data_bank=True))
                input_offset = _align_up(input_offset + input_bytes, 16)
                continue

            input_flat = input_flat.tolist()
            ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
            ct_inputs[:] = input_flat

            task_regs = [ make_conv2d_regs(
                            1, in_c, tile_in_h, in_w, out_c, kh, kw,
                            input_mem_create.dma_addr,
                            weight_mem_create.dma_addr,
                            output_mem_create.dma_addr,
                            groups=groups )]

            submit_conv_tasks(task_regs, repeat=_direct_submit_repeat(p, batch, is_spatial, len(tiles)))

            out_c1 = _ceil_div(out_c, UNPACK_C2) if grouped_spatial else _ceil_div(p["align_out_c"], UNPACK_C2)
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
                out_buf = read_output_fp16(out_c1 * tile_p["out_width_stride"] * UNPACK_C2)
                result[n] = _unpack_flat_1x1_output(out_buf, out_c, out_h, out_w, tile_p["out_width_stride"], UNPACK_C2)
        if pc_chain_tiles and task_regs:
            submit_repeat = 2 if output_channel_tiles else 1
            submit_conv_tasks(task_regs, repeat=submit_repeat)
        if pc_chain_tiles and not depthwise_channel_tiles:
            out_c1 = _ceil_div(p["align_out_c"], UNPACK_C2)
            out_buf = read_output_fp16(out_c1 * p["out_width_stride"] * UNPACK_C2)
            result[n] = _unpack_flat_1x1_output(out_buf, out_c, out_h, out_w, p["out_width_stride"], UNPACK_C2)
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

if __name__ == "__main__":
    shapes = [
        # ── 1x1 kernels (fully supported via NHWC mode + channel slicing for ic>=5) ──
        dict(name="conv2d_1x6_1x1_4x4",                batch=1, in_c=1,  in_h=4,  in_w=4,  out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
        dict(name="conv2d_3x3_1x1_4x4",                batch=1, in_c=3,  in_h=4,  in_w=4,  out_c=3, weight_in_c=3, kh=1, kw=1, groups=1),
        dict(name="conv2d_4x2_1x1_4x4",                batch=1, in_c=4,  in_h=4,  in_w=4,  out_c=2, weight_in_c=4, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c4_h9_w9_oc4_wic4_k1x1_g1",  batch=1, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=1, kw=1, groups=1),
        dict(name="conv2d_16x16_1x1_8x8",              batch=1, in_c=16, in_h=8,  in_w=8,  out_c=16, weight_in_c=16, kh=1, kw=1, groups=1),
        dict(name="conv2d_b1_c16_h32_w32_oc16_wic16_k1x1_g1", batch=1, in_c=16, in_h=32, in_w=32, out_c=16, weight_in_c=16, kh=1, kw=1, groups=1),

        # ── Non-1x1 kernels (partial output — known NPU hardware limitation) ──
        dict(name="conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1",  batch=1, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=3, kw=3, groups=1),
        dict(name="conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1", batch=1, in_c=16, in_h=18, in_w=18, out_c=16, weight_in_c=16, kh=3, kw=3, groups=1),
        dict(name="conv2d_b2_c4_h9_w9_oc4_wic4_k3x3_g1",  batch=2, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=3, kw=3, groups=1),
        dict(name="conv2d_b1_c1_h5_w7_oc6_wic1_k3x3_g1",  batch=1, in_c=1,  in_h=5,  in_w=7,  out_c=6, weight_in_c=1, kh=3, kw=3, groups=1),

        # Depthwise
        dict(name="conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3", batch=1, in_c=3, in_h=11, in_w=28, out_c=3, weight_in_c=1, kh=3, kw=3, groups=3),

        # Non-square kernels
        dict(name="conv2d_3x6_1x3_5x5", batch=1, in_c=3, in_h=5, in_w=5, out_c=6, weight_in_c=3, kh=1, kw=3, groups=1),

        # ── test_ops.py _test_conv2d(cin=3): (3,5,7) @ (6,kh,kw) ──
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=3, groups=3),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x1_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=1, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x3_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=3, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x5_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=5, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x1_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=1, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x3_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=3, groups=1),
        dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x5_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=5, groups=1),

        # ── test_ops.py _test_conv2d(cin=1): (1,5,7) @ (6,kh,kw) ──
        dict(name="conv2d_1x6_2x1_5x7", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=2, kw=1, groups=1),
        dict(name="conv2d_1x6_2x3_5x7", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=2, kw=3, groups=1),
        dict(name="conv2d_1x6_3x1_5x7_b", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=1, groups=1),
        dict(name="conv2d_1x6_3x5_5x7",   batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=5, groups=1),

        # ── Grouped convs from test_ops.py ──
        dict(name="conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2",     batch=1,  in_c=4,  in_h=1,  in_w=1,  out_c=2,  weight_in_c=2,  kh=1, kw=1, groups=2),
        dict(name="conv2d_4x4_1x1_1x1_g2",                   batch=1,  in_c=4,  in_h=1,  in_w=1,  out_c=4,  weight_in_c=2,  kh=1, kw=1, groups=2),
        dict(name="conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32", batch=1, in_c=32, in_h=32, in_w=32, out_c=32, weight_in_c=1,  kh=1, kw=1, groups=32),
        dict(name="conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5",   batch=1,  in_c=15, in_h=5,  in_w=5,  out_c=35, weight_in_c=3,  kh=3, kw=3, groups=5),

        # ── Batch >1 coverage ──
        dict(name="conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3", batch=2, in_c=3,  in_h=11, in_w=28, out_c=3,  weight_in_c=1, kh=3, kw=3, groups=3),
        dict(name="conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5", batch=4, in_c=15, in_h=5,  in_w=5,  out_c=35, weight_in_c=3, kh=3, kw=3, groups=5),

        # ── Grouped output-channel variants ──
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
        # first spatial conv (RGB→32ch)
        dict(name="conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1", batch=1, in_c=3, in_h=224, in_w=224, out_c=32, weight_in_c=3, kh=3, kw=3, groups=1),
        #  depthwise conv (3×3 sep)
        dict(name="conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32", batch=1, in_c=32, in_h=112, in_w=112, out_c=32, weight_in_c=1, kh=3, kw=3, groups=32),
        # pointwise (1×1 projection)
        dict(name="conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1", batch=1, in_c=32, in_h=112, in_w=112, out_c=64, weight_in_c=32, kh=1, kw=1, groups=1),
        # depthwise conv (64ch)
        dict(name="conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64", batch=1, in_c=64, in_h=112, in_w=112, out_c=64, weight_in_c=1, kh=3, kw=3, groups=64),
        # pointwise expansion
        dict(name="conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1", batch=1, in_c=64, in_h=56, in_w=56, out_c=128, weight_in_c=64, kh=1, kw=1, groups=1),
        # depthwise conv (128ch) 
        dict(name="conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128", batch=1, in_c=128, in_h=56, in_w=56, out_c=128, weight_in_c=1, kh=3, kw=3, groups=128),
        # pointwise (1×1 projection, same-channel)
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
    ]
    shapes_sweep = [dict(name=f"conv2d_1x3_{n}x{n}_k1", batch=1, in_c=3, in_h=n, in_w=n, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1) for n in range(2, 400, 2)]
    shapes += shapes_sweep
    # ── Known-issue / reference shapes (non-blocking, report-only) ──
    known_issue_shapes = [
    ]
    shapes += known_issue_shapes
    
    name_width = max(len(s["name"]) for s in shapes)
    in_shape_width = max(len(f"{s['in_c']}x{s['in_h']}x{s['in_w']}") for s in shapes)
    out_shape_width = max(len(f"{s['out_c']}x{s['in_h'] - s['kh'] + 1}x{s['in_w'] - s['kw'] + 1}") for s in shapes)

    for s in shapes:
        name, batch, in_c, in_h, in_w, out_c, weight_in_c, kh, kw, groups = \
            s["name"], s["batch"], s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["weight_in_c"], s["kh"], s["kw"], s["groups"]
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
