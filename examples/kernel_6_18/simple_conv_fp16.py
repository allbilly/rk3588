"""
hw_out_fp16 = out_fp16 or is_spatial or p["is_depthwise"] or
out_c >= 128 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE

The FP32 conv fallback path uses FP16 NPU writeback plus dtype cast for
some shapes. That is not CPU convolution offload, but it is also not true FP32 hardware writeback for those cases.
"""
import glob
import ast
import os, mmap, sys, time, ctypes, numpy as np
from fcntl import ioctl

FP16_BYTES = 2
FP32_BYTES = 4
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
    RDMA_S_POINTER        = 0x5004   # DPU RDMA S pointer config
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

DRM_COMMAND_BASE = 0x40

class drm_rocket_create_bo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("handle", ctypes.c_uint32),
        ("dma_address", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
    ]

class drm_rocket_prep_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("timeout_ns", ctypes.c_int64),
    ]

class drm_rocket_fini_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]

class drm_rocket_task(ctypes.Structure):
    _fields_ = [
        ("regcmd", ctypes.c_uint32),
        ("regcmd_count", ctypes.c_uint32),
    ]

class drm_rocket_job(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_uint64),
        ("in_bo_handles", ctypes.c_uint64),
        ("out_bo_handles", ctypes.c_uint64),
        ("task_count", ctypes.c_uint32),
        ("task_struct_size", ctypes.c_uint32),
        ("in_bo_handle_count", ctypes.c_uint32),
        ("out_bo_handle_count", ctypes.c_uint32),
    ]

class drm_rocket_submit(ctypes.Structure):
    _fields_ = [
        ("jobs", ctypes.c_uint64),
        ("job_count", ctypes.c_uint32),
        ("job_struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]

def _IOW(type_, nr, size):
    return (1 << 30) | (ord(type_) << 8) | nr | (size << 16)

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)

DRM_IOCTL_ROCKET_CREATE_BO = _IOWR('d', DRM_COMMAND_BASE + 0x00, ctypes.sizeof(drm_rocket_create_bo))
DRM_IOCTL_ROCKET_SUBMIT = _IOW('d', DRM_COMMAND_BASE + 0x01, ctypes.sizeof(drm_rocket_submit))
DRM_IOCTL_ROCKET_PREP_BO = _IOW('d', DRM_COMMAND_BASE + 0x02, ctypes.sizeof(drm_rocket_prep_bo))
DRM_IOCTL_ROCKET_FINI_BO = _IOW('d', DRM_COMMAND_BASE + 0x03, ctypes.sizeof(drm_rocket_fini_bo))

class RocketBO:
    __slots__ = ("handle", "size", "dma_address", "offset")
    def __init__(self, handle, size, dma_address, offset):
        self.handle = int(handle)
        self.size = int(size)
        self.dma_address = int(dma_address)
        self.offset = int(offset)

    @property
    def dma_addr(self):
        return self.dma_address

    @property
    def obj_addr(self):
        return self.handle

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

def _rocket_mem_allocate(fd, size):
    bo = drm_rocket_create_bo(size=size)
    ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, bo)
    buf = mmap.mmap(fd, bo.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return buf, RocketBO(bo.handle, bo.size, bo.dma_address, bo.offset)

def _rocket_prep_bo(fd, bo, timeout_ns=6_000_000_000):
    if timeout_ns > 0:
        timeout_ns = time.monotonic_ns() + timeout_ns
    ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, drm_rocket_prep_bo(handle=bo.handle, timeout_ns=timeout_ns))

def _rocket_fini_bo(fd, bo):
    ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))

def _rocket_submit(fd, vendor_tasks, task_count=1, in_bos=(), out_bos=()):
    rocket_tasks = (drm_rocket_task * task_count)()
    for i in range(task_count):
        rocket_tasks[i].regcmd = int(vendor_tasks[i].regcmd_addr) & 0xFFFFFFFF
        rocket_tasks[i].regcmd_count = int(vendor_tasks[i].regcfg_amount)
    in_handles = (ctypes.c_uint32 * len(in_bos))(*(bo.handle for bo in in_bos)) if in_bos else None
    out_handles = (ctypes.c_uint32 * len(out_bos))(*(bo.handle for bo in out_bos)) if out_bos else None
    job = drm_rocket_job(
        tasks=ctypes.addressof(rocket_tasks),
        in_bo_handles=ctypes.addressof(in_handles) if in_handles is not None else 0,
        out_bo_handles=ctypes.addressof(out_handles) if out_handles is not None else 0,
        task_count=task_count,
        task_struct_size=ctypes.sizeof(drm_rocket_task),
        in_bo_handle_count=len(in_bos),
        out_bo_handle_count=len(out_bos),
    )
    jobs = (drm_rocket_job * 1)(job)
    submit_struct = drm_rocket_submit(
        jobs=ctypes.addressof(jobs),
        job_count=1,
        job_struct_size=ctypes.sizeof(drm_rocket_job),
    )
    return ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, submit_struct)

def _open_rocket_device():
    path = os.environ.get("ROCKET_DEVICE")
    if path:
        return os.open(path, os.O_RDWR)
    candidates = (sorted(glob.glob("/dev/accel/accel*")) + sorted(glob.glob("/dev/dri/renderD*")) + sorted(glob.glob("/dev/dri/card*")))
    for c in candidates:
        try: return os.open(c, os.O_RDWR)
        except OSError: pass
    raise FileNotFoundError("No Rocket device found")

def npu_submit(task_count=1):
    for bo in (regcmd_mem_create, input_mem_create, weight_mem_create, output_mem_create):
        _rocket_fini_bo(fd, bo)
    ret = _rocket_submit(fd, npu_tasks, task_count,
        in_bos=[regcmd_mem_create, input_mem_create, weight_mem_create],
        out_bos=[output_mem_create])
    _rocket_prep_bo(fd, output_mem_create)
    return ret

fd = _open_rocket_device()
task_map, tasks_mem_create = _rocket_mem_allocate(fd, 64*1024)
regcmd_map, regcmd_mem_create = _rocket_mem_allocate(fd, 512*1024)
input_map, input_mem_create = _rocket_mem_allocate(fd, 16*1024*1024)
weight_map, weight_mem_create = _rocket_mem_allocate(fd, 4*1024*1024)
output_map, output_mem_create = _rocket_mem_allocate(fd, 16*1024*1024)
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
# outermost, then OC, then aligned-IC innermost. This matches the CSC->CMAC
# data sequencing for spatial convolutions on RK schedule.
def _is_kh_major(out_c, in_c, kh, kw, groups):
    is_spatial = (kh != 1 or kw != 1)
    multi_input_group = groups > 1 and in_c != groups
    return is_spatial and not _is_depthwise(in_c, out_c, groups) and not multi_input_group

def _is_grouped_spatial(in_c, out_c, kh, kw, groups):
    is_spatial = (kh != 1 or kw != 1)
    multi_input_group = groups > 1 and in_c != groups
    return is_spatial and multi_input_group and not _is_depthwise(in_c, out_c, groups)

def _needs_c96_oc24_pointwise_schedule(in_c, out_c, kh, kw, groups):
    return groups == 1 and kh == 1 and kw == 1 and in_c == 96 and out_c == 24

def _pointwise_split_add_chunks(in_c, out_c, in_h, in_w, kh, kw, groups):
    if groups != 1 or kh != 1 or kw != 1:
        return None
    if (in_h < 4 or in_w < 4) and 64 <= in_c <= 256:
        return [(0, in_c)]
    if in_c == 144 and out_c == 24 and in_h == 56 and in_w == 56:
        return [(0, 64), (64, 128), (128, 144)]
    if in_c == 144 and out_c == 32 and in_h == 28 and in_w == 28:
        return [(0, 64), (64, 128), (128, 144)]
    if in_c == 192 and out_c in (16, 32) and in_h == 28 and in_w == 28:
        return [(0, 64), (64, 128), (128, 192)]
    if in_c == 256 and out_c == 32 and in_h == 28 and in_w == 28:
        return [(0, 64), (64, 128), (128, 192), (192, 256)]
    if in_c >= 128 and out_c <= 128:
        chunk_c = 128 if out_c > 32 else 64
        return [(start, min(start + chunk_c, in_c)) for start in range(0, in_c, chunk_c)]
    return None

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
    align_c = 32 if (not is_spatial and groups == 1 and in_c >= 64) else _conv_align_c(in_c, groups, out_c)
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
    c1 = _ceil_div(_align_up(in_c, p["align_c"]), c2)
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
    result = np.zeros((out_c, out_h, out_w), dtype=out_raw.dtype)
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
        channel_tiled = p["out_c"] >= 48
        if _needs_c96_oc24_pointwise_schedule(p["in_c"], p["out_c"], p["kh"], p["kw"], p["groups"]):
            tile_h = min(p["out_h"], 20)
        elif small_channel:
            tile_h = max(1, RK_MAX_CONV_FLAT_STRIDE // p["out_w"]) if p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE else p["out_h"]
        elif p["out_h"] > 50:
            tile_h = 25 if (p["in_c"] >= 128 and p["out_c"] >= 128) else (32 if p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE else 50)
        elif channel_tiled:
            row_bytes = p["width_stride"] * _align_up(p["in_c"], p["align_c"]) * FP16_BYTES
            tile_h = min(p["out_h"], max(1, (8 * CBUF_BANK_SIZE) // max(1, row_bytes)))
        else:
            tile_h = p["out_h"]
        return (_needs_c96_oc24_pointwise_schedule(p["in_c"], p["out_c"], p["kh"], p["kw"], p["groups"])
                or (not small_channel and (p["out_h"] > 50 or channel_tiled)),
                [(row, min(tile_h, p["out_h"] - row)) for row in range(0, p["out_h"], tile_h)])

    if p["is_depthwise"] and p["out_c"] >= 32:
        if p["out_h"] <= 13:
            return True, [(0, p["in_h"])]
        tile_out_h = 32 if (p["out_c"] == 32 and p["kh"] == 3 and p["kw"] == 3) else (12 if max(p["kh"], p["kw"]) >= 5 else 13)
        return True, [(row, min(tile_out_h + p["kh"] - 1, p["in_h"] - row)) for row in range(0, p["out_h"], tile_out_h)]

    if not grouped_spatial and (p["out_h"] > 48 or p["out_c"] >= 48):
        tile_out_h = 32
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

def make_conv2d_regs(batch, in_c, in_h, in_w, out_c, kh, kw, in_dma, wt_dma, out_dma, groups=1, out_width_stride_override=None, weight_reuse=False, full_data_bank=False, out_fp16=False):
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
    out_precision = 2 if out_fp16 else 5
    size_e = 1 if out_fp16 else 3
    bs_size_e = 3 if is_depthwise else size_e

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
            ((out_precision << 29) |       # DPU_DATA_FORMAT_OUT_PRECISION
             (2 << 26) |                   # DPU_DATA_FORMAT_PROC_PRECISION(fp16)
              2)),                         # DPU_DATA_FORMAT_IN_PRECISION(fp16)
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma),
        E(reg.DPU, reg.DST_SURF_STRIDE, out_width_stride << 4), # DPU_DST_SURF_STRIDE_DST_SURF_STRIDE
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1),             # DPU_DATA_CUBE_WIDTH_WIDTH
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),            # DPU_DATA_CUBE_HEIGHT_HEIGHT
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
            ((bs_size_e << 8) |           # DPU_BS_OW_CFG_SIZE_E_2
             (bs_size_e << 5) |           # DPU_BS_OW_CFG_SIZE_E_1
             (bs_size_e << 2) |           # DPU_BS_OW_CFG_SIZE_E_0
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
            ((1 << 16) | 1) if out_fp16 else 0), # DPU_OUT_CVT_SCALE_FP32TOFP16_EN | OUT_CVT_SCALE
        E(reg.DPU, reg.SURFACE_ADD,
            (out_width_stride * max(2, effective_align_out // 16)) << 4) # DPU_SURFACE_ADD_SURF_ADD
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
        # rocket: HW fetches exactly regcfg_amount regs, so tail must be counted
        npu_tasks[idx].regcfg_amount = len(regs) + len(tails)
        npu_tasks[idx].op_idx = 1               # downstream raw conv task descriptor op index
        npu_tasks[idx].enable_mask = 0xd        # downstream raw conv task descriptor mask
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9) # PC_INTERRUPT_MASK_DPU_0 | DPU_1
        npu_tasks[idx].int_clear = 0x1ffff      # downstream RKNPU_INT_CLEAR clears all status bits

def submit_conv_tasks(task_regs, repeat=1):
    for _ in range(repeat):
        write_regs_to_npu_task(task_regs)
        npu_submit(task_count=len(task_regs))

def write_raw_npu_task(regs, op_idx, enable_mask, int_mask=(1 << 8) | (1 << 9)):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())
    assert len(regs) <= regcmd_mem_create.size // ctypes.sizeof(ctypes.c_uint64), "regcmd buffer too small"
    for i, qword in enumerate(regs):
        npu_regcmd[i] = qword
    npu_tasks[0].regcmd_addr = regcmd_mem_create.dma_addr
    npu_tasks[0].regcfg_amount = len(regs)
    npu_tasks[0].op_idx = op_idx
    npu_tasks[0].enable_mask = enable_mask
    npu_tasks[0].int_mask = int_mask
    npu_tasks[0].int_clear = 0x1ffff

def write_eltwise_regs_to_npu_task(task_regs):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())

    def make_tail(next_offset, next_body_len):
        enable = E(reg.PC, reg.OPERATION_ENABLE, 0x18)
        if next_offset is None:
            return [
                E(reg.PC_REG, reg.PC_BASE_ADDRESS, 0),
                E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                E(reg.VERSION, 0, 0),
                enable,
            ]
        next_addr = regcmd_mem_create.dma_addr + next_offset * ctypes.sizeof(ctypes.c_uint64)
        return [
            E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
            E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, next_body_len),
            E(reg.VERSION, 0, 0),
            enable,
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
        next_body_len = len(task_regs[idx + 1]) if idx + 1 < len(task_regs) else 0
        next_offset = offsets[idx + 1] if idx + 1 < len(task_regs) else None
        tails = make_tail(next_offset, next_body_len)
        for i, qword in enumerate(tails):
            npu_regcmd[base + len(regs) + i] = qword

        npu_tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)
        npu_tasks[idx].regcfg_amount = len(regs) + len(tails)
        npu_tasks[idx].op_idx = 4
        npu_tasks[idx].enable_mask = 0x18
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9)
        npu_tasks[idx].int_clear = 0x1ffff

# why add, is add really the best approach? too many lines added
def submit_fp16_add(a_dma, b_dma, out_dma, out_width_stride, align_out_c, repeat=2):
    total_elems = out_width_stride * align_out_c
    max_elems = 8000 * 8
    task_regs = []
    for start in range(0, total_elems, max_elems):
        tile_n = min(max_elems, total_elems - start)
        dataout_width = _ceil_div(tile_n, 8) - 1
        byte_offset = start * FP16_BYTES
        task_regs.append([
            E(reg.DPU,  reg.S_POINTER, 0x0000000E),
            E(reg.DPU,  reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1) | 1),
            E(reg.DPU,  reg.DATA_FORMAT, (2 << 29) | (2 << 26) | 2),
            E(reg.DPU,  reg.DATA_CUBE_WIDTH, dataout_width),
            E(reg.DPU,  reg.DATA_CUBE_HEIGHT, 0),
            E(reg.DPU,  reg.DATA_CUBE_NOTCH, 0),
            E(reg.DPU,  reg.DATA_CUBE_CHANNEL, (7 << 16) | 7),
            E(reg.DPU,  reg.EW_CFG, 0x108202c0),
            E(reg.DPU,  reg.OUT_CVT_SCALE, (1 << 16) | 1),
            E(reg.RDMA, reg.RDMA_S_POINTER, 0x0000000E),
            E(reg.RDMA, reg.RDMA_DATA_CUBE_WIDTH, dataout_width),
            E(reg.RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            E(reg.RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
            E(reg.RDMA, reg.RDMA_ERDMA_CFG, (1 << 30) | (2 << 2)),
            E(reg.DPU,  reg.DST_BASE_ADDR, out_dma + byte_offset),
            E(reg.RDMA, reg.RDMA_SRC_BASE_ADDR, a_dma + byte_offset),
            E(reg.RDMA, reg.RDMA_EW_BASE_ADDR, b_dma + byte_offset),
            E(reg.RDMA, reg.RDMA_FEATURE_MODE_CFG, (2 << 15) | (15 << 11) | (2 << 5) | (1 << 3) | 1),
        ])
    for _ in range(repeat):
        write_eltwise_regs_to_npu_task(task_regs)
        npu_submit(task_count=len(task_regs))

def submit_pointwise_to_output(input_nchw, weight_nchw, out_c):
    _, in_c, in_h, in_w = input_nchw.shape
    sched_out_c = _align_up(out_c, 16) if (out_c % 16 and (out_c > 32 or (in_h == 1 and in_w == 1))) else out_c
    p = _conv_params(1, in_c, in_h, in_w, sched_out_c, 1, 1, 1)
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)
    weight_full = np.zeros((sched_out_c, in_c, 1, 1), dtype=np.float16)
    weight_full[:out_c] = weight_nchw.reshape(out_c, in_c, 1, 1)
    wt_flat = pack_conv_weights_for_shape(weight_full, sched_out_c, in_c, 1, 1, p["align_c"], 1)
    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
    ctypes.memmove(wt_ptr, wt_flat.ctypes.data, wt_flat.nbytes)

    task_regs = []
    input_ptr = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
    input_offset = 0
    _, tiles = _conv_tiles(p, False, False)
    for row_start, tile_in_h in tiles:
        tile_p = p if tile_in_h == in_h else _conv_params(1, in_c, tile_in_h, in_w, out_c, 1, 1, 1)
        input_tile = input_nchw[0, :, row_start:row_start + tile_in_h, :]
        input_flat = _pack_conv_input_fp16(input_tile, tile_p).view(np.uint16)
        input_bytes = input_flat.nbytes
        assert input_offset + input_bytes <= input_mem_create.size, "input buffer too small"
        ctypes.memmove(input_ptr + input_offset, input_flat.ctypes.data, input_bytes)
        output_offset = row_start * in_w * 16
        if sched_out_c >= 48:
            aligned_in_c = max(32, _align_up(in_c, 32)) if in_c >= 64 else _align_up(in_c, tile_p["align_c"])
            hw_out_c = _align_up(sched_out_c, 16)
            for oc_start in range(0, hw_out_c, 16):
                oc_tile = min(16, hw_out_c - oc_start)
                weight_offset = oc_start * aligned_in_c * FP16_BYTES
                surface_offset = (oc_start // 16) * p["out_width_stride"] * 16 * FP16_BYTES
                regs = make_conv2d_regs(
                    1, in_c, tile_in_h, in_w, oc_tile, 1, 1,
                    input_mem_create.dma_addr + input_offset,
                    weight_mem_create.dma_addr + weight_offset,
                    output_mem_create.dma_addr + output_offset + surface_offset,
                    groups=1,
                    out_width_stride_override=p["out_width_stride"],
                    full_data_bank=True,
                    out_fp16=True)
                regs = _with_cbuf_data_bank(regs, _tile_data_bank(p, tile_in_h))
                task_regs.append(regs)
        else:
            regs = make_conv2d_regs(
                1, in_c, tile_in_h, in_w, sched_out_c, 1, 1,
                input_mem_create.dma_addr + input_offset,
                weight_mem_create.dma_addr,
                output_mem_create.dma_addr + output_offset,
                groups=1,
                out_width_stride_override=p["out_width_stride"],
                weight_reuse=bool(task_regs),
                full_data_bank=True,
                out_fp16=True)
            task_regs.append(regs)
        input_offset = _align_up(input_offset + input_bytes, 16)
    submit_conv_tasks(task_regs, repeat=2 if sched_out_c >= 48 else 1)

def run_conv2d(batch, in_c, out_c, kh, kw, input_hw, groups=1, weight_in_c=None, out_fp16=False):
    in_h, in_w = input_hw
    weight_in_c = weight_in_c or (in_c // groups)
    logical_out_c = out_c
    hw_sched_out_c = _align_up(out_c, 16) if (groups == 1 and kh == 1 and kw == 1 and in_c >= 64 and out_c >= 48) else out_c
    p = _conv_params(1, in_c, in_h, in_w, hw_sched_out_c, kh, kw, groups)
    is_spatial = (kh != 1 or kw != 1)
    out_h, out_w = p["out_h"], p["out_w"]
    grouped_spatial = _is_grouped_spatial(in_c, out_c, kh, kw, groups)
    hw_out_fp16 = (out_fp16 or is_spatial or p["is_depthwise"] or hw_sched_out_c >= 128
                   or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE
                   or _needs_c96_oc24_pointwise_schedule(in_c, out_c, kh, kw, groups)
                   or _pointwise_split_add_chunks(in_c, out_c, in_h, in_w, kh, kw, groups) is not None)

    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (logical_out_c, weight_in_c, kh, kw)).astype(np.float16)
    return_input_nchw = input_nchw
    return_weight_nchw = weight_nchw
    logical_out_h, logical_out_w = out_h, out_w
    if is_spatial and (groups == 1 or p["is_depthwise"]) and out_h == 1 and out_w == 1 and in_c >= 64:
        padded_input = np.zeros((batch, in_c, in_h + 1, in_w + 1), dtype=np.float16)
        padded_input[:, :, :in_h, :in_w] = input_nchw
        input_nchw = padded_input
        in_h, in_w = in_h + 1, in_w + 1
        p = _conv_params(1, in_c, in_h, in_w, hw_sched_out_c, kh, kw, groups)
        out_h, out_w = p["out_h"], p["out_w"]
    if is_spatial and groups == 1 and in_c > 32 and in_c % 32:
        padded_in_c = _align_up(in_c, 32)
        padded_input = np.zeros((batch, padded_in_c, in_h, in_w), dtype=np.float16)
        padded_input[:, :in_c] = input_nchw
        padded_weight = np.zeros((logical_out_c, padded_in_c, kh, kw), dtype=np.float16)
        padded_weight[:, :in_c] = weight_nchw
        input_nchw = padded_input
        weight_nchw = padded_weight
        in_c = padded_in_c
        weight_in_c = padded_in_c
        p = _conv_params(1, in_c, in_h, in_w, hw_sched_out_c, kh, kw, groups)
    if not is_spatial and groups == 1 and in_c > 32 and in_c % 32:
        padded_in_c = _align_up(in_c, 32)
        padded_input = np.zeros((batch, padded_in_c, in_h, in_w), dtype=np.float16)
        padded_input[:, :in_c] = input_nchw
        padded_weight = np.zeros((logical_out_c, padded_in_c, kh, kw), dtype=np.float16)
        padded_weight[:, :in_c] = weight_nchw
        input_nchw = padded_input
        weight_nchw = padded_weight
        in_c = padded_in_c
        weight_in_c = padded_in_c
        p = _conv_params(1, in_c, in_h, in_w, hw_sched_out_c, kh, kw, groups)

    out_dtype = np.float16 if out_fp16 else np.float32
    read_dtype = np.float16 if hw_out_fp16 else np.float32
    read_bytes = FP16_BYTES if hw_out_fp16 else FP32_BYTES
    unpack_c2 = UNPACK_C2 if hw_out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
    def read_output(count):
        return np.frombuffer(output_map, dtype=read_dtype, count=count).copy()

    split_add_chunks = _pointwise_split_add_chunks(in_c, out_c, in_h, in_w, kh, kw, groups)
    if split_add_chunks is not None:
        pad_small_spatial = in_h < 4 or in_w < 4
        acc_p = _conv_params(1, in_c, max(4, in_h), max(4, in_w), out_c, 1, 1, 1) if pad_small_spatial else p
        acc_out_h, acc_out_w = acc_p["out_h"], acc_p["out_w"]
        raw_elements = _ceil_div(acc_p["align_out_c"], UNPACK_C2) * acc_p["out_width_stride"] * UNPACK_C2
        raw_bytes = raw_elements * FP16_BYTES
        stage_a_offset = 8 * 1024 * 1024
        stage_b_offset = 3 * 1024 * 1024
        assert stage_a_offset + raw_bytes <= input_mem_create.size, "input staging buffer too small"
        assert stage_b_offset + raw_bytes <= weight_mem_create.size, "weight staging buffer too small"
        out_ptr = ctypes.addressof(ctypes.c_char.from_buffer(output_map))
        stage_a_ptr = ctypes.addressof(ctypes.c_char.from_buffer(input_map)) + stage_a_offset
        stage_b_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map)) + stage_b_offset
        stage_a_dma = input_mem_create.dma_addr + stage_a_offset
        stage_b_dma = weight_mem_create.dma_addr + stage_b_offset
        warmup_regs = make_conv2d_regs(
            1, 1, 4, 4, 6, 1, 1,
            input_mem_create.dma_addr,
            weight_mem_create.dma_addr,
            output_mem_create.dma_addr,
            groups=1,
            out_fp16=True)
        submit_conv_tasks([warmup_regs], repeat=2)
        partials = []
        for start_c, end_c in split_add_chunks:
            chunk_input = input_nchw[:, start_c:end_c]
            if pad_small_spatial:
                padded_chunk = np.zeros((batch, end_c - start_c, acc_p["out_h"], acc_p["out_w"]), dtype=np.float16)
                padded_chunk[:, :, :in_h, :in_w] = chunk_input
                chunk_input = padded_chunk
            submit_pointwise_to_output(chunk_input, weight_nchw[:, start_c:end_c], out_c)
            partials.append(np.frombuffer(output_map, dtype=np.float16, count=raw_elements).copy())

        if len(partials) == 1:
            out_buf = partials[0]
            result = _unpack_flat_1x1_output(out_buf, out_c, acc_out_h, acc_out_w, acc_p["out_width_stride"], UNPACK_C2).reshape(batch, out_c, acc_out_h, acc_out_w)
            if pad_small_spatial:
                result = result[:, :, :out_h, :out_w]
            return result.astype(out_dtype), input_nchw, weight_nchw

        zero_raw = np.zeros(raw_elements, dtype=np.float16)
        ctypes.memmove(stage_a_ptr, zero_raw.ctypes.data, raw_bytes)
        ctypes.memmove(stage_b_ptr, zero_raw.ctypes.data, raw_bytes)
        submit_fp16_add(stage_a_dma, stage_b_dma, output_mem_create.dma_addr, acc_p["out_width_stride"], acc_p["align_out_c"], repeat=1)

        ctypes.memmove(stage_a_ptr, partials[0].ctypes.data, raw_bytes)
        for chunk_idx, partial in enumerate(partials[1:], start=1):
            ctypes.memmove(stage_b_ptr, partial.ctypes.data, raw_bytes)
            submit_fp16_add(stage_a_dma, stage_b_dma, output_mem_create.dma_addr, acc_p["out_width_stride"], acc_p["align_out_c"])
            if chunk_idx + 1 < len(partials):
                stage_raw = np.frombuffer(output_map, dtype=np.float16, count=raw_elements).copy()
                ctypes.memmove(stage_a_ptr, stage_raw.ctypes.data, raw_bytes)

        out_buf = read_output(raw_elements)
        result = _unpack_flat_1x1_output(out_buf, out_c, acc_out_h, acc_out_w, acc_p["out_width_stride"], UNPACK_C2).reshape(batch, out_c, acc_out_h, acc_out_w)
        if pad_small_spatial:
            result = result[:, :, :out_h, :out_w]
        return result.astype(out_dtype), return_input_nchw, return_weight_nchw

    if _is_depthwise(in_c, out_c, groups):
        weight_full = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
        for oc in range(out_c):
            weight_full[oc, oc] = weight_nchw[oc, 0]
    else:
        weight_full = _expand_grouped_weights(weight_nchw, in_c, logical_out_c, kh, kw, groups)
    if grouped_spatial:
        weight_full = _reorder_grouped_spatial_weights_block16(weight_full, out_c, in_c, kh, kw)

    if p["is_depthwise"] and in_c == 32 and out_c == 32 and not is_spatial:
        wt_flat = np.zeros((kh * kw * _align_up(in_c, p["align_c"]) * out_c), dtype=np.float16)
        wt_flat[:out_c] = weight_nchw[:, 0, 0, 0]
    else:
        wt_flat = pack_conv_weights_for_shape(weight_full, logical_out_c, in_c, kh, kw, p["align_c"], groups)
    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
    ctypes.memmove(wt_ptr, wt_flat.ctypes.data, wt_flat.nbytes)

    result = np.zeros((batch, hw_sched_out_c, out_h, out_w), dtype=out_dtype)

    for n in range(batch):
        pc_chain_tiles, tiles = _conv_tiles(p, is_spatial, grouped_spatial)
        depthwise_channel_tiles = False
        compact_tail = (not is_spatial and out_c % 16 and out_h == 1 and out_w == 1)

        if pc_chain_tiles:
            input_ptr = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
            task_regs = []
            input_offset = 0
            output_channel_tile_size = 512 if compact_tail else 16
            output_channel_tiles = (not p["is_depthwise"] and not grouped_spatial and hw_sched_out_c >= 48)
            depthwise_channel_tiles = p["is_depthwise"] and out_c >= 32
            if compact_tail:
                warmup_regs = make_conv2d_regs(
                    1, 1, 4, 4, 6, 1, 1,
                    input_mem_create.dma_addr,
                    weight_mem_create.dma_addr,
                    output_mem_create.dma_addr,
                    groups=1,
                    out_fp16=hw_out_fp16)
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
                        hw_ch_tile = max(depthwise_block_c, ch_tile)
                        block_p = _conv_params(1, hw_ch_tile, tile_in_h, in_w, hw_ch_tile, kh, kw, hw_ch_tile)
                        block_input = np.zeros((hw_ch_tile, tile_in_h, in_w), dtype=np.float16)
                        block_input[:ch_tile] = input_tile[ch_start:ch_start + ch_tile]
                        block_input_flat = _pack_conv_input_fp16(block_input, block_p).view(np.uint16)
                        block_input_flat = block_input_flat.tolist()
                        ct_inputs = (ctypes.c_uint16 * len(block_input_flat)).from_buffer(input_map)
                        ct_inputs[:] = block_input_flat
                        block_weight_full = np.zeros((hw_ch_tile, hw_ch_tile, kh, kw), dtype=np.float16)
                        for local_c in range(ch_tile):
                            block_weight_full[local_c, local_c] = weight_nchw[ch_start + local_c, 0]
                        block_weight_flat = pack_conv_weights_for_shape(block_weight_full, hw_ch_tile, hw_ch_tile, kh, kw, block_p["align_c"], hw_ch_tile)
                        ctypes.memmove(wt_ptr, block_weight_flat.ctypes.data, block_weight_flat.nbytes)
                        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_mem_create.size)
                        regs = make_conv2d_regs(
                            1, hw_ch_tile, tile_in_h, in_w, hw_ch_tile, kh, kw,
                            input_mem_create.dma_addr,
                            weight_mem_create.dma_addr,
                            output_mem_create.dma_addr,
                            groups=hw_ch_tile,
                            out_width_stride_override=p["out_width_stride"],
                            full_data_bank=True,
                            out_fp16=hw_out_fp16)
                        submit_conv_tasks([regs], repeat=2)
                        out_c1 = _ceil_div(max(16, _align_up(hw_ch_tile, 16)), unpack_c2)
                        out_buf = read_output(out_c1 * p["out_width_stride"] * unpack_c2)
                        result[n, ch_start:ch_start + ch_tile, row_start:row_start + tile_out_h, :] = _unpack_flat_1x1_output(
                            out_buf, hw_ch_tile, tile_out_h, out_w, p["out_width_stride"], unpack_c2)[:ch_tile]
                elif output_channel_tiles:
                    aligned_in_c = max(32, _align_up(in_c, 32)) if (kh == 1 and kw == 1 and groups == 1 and in_c >= 64) else _align_up(in_c, p["align_c"])
                    for oc_start in range(0, hw_sched_out_c, output_channel_tile_size):
                        oc_tile = min(output_channel_tile_size, hw_sched_out_c - oc_start)
                        hw_oc_tile = oc_tile
                        weight_offset = oc_start * kh * kw * aligned_in_c * FP16_BYTES
                        surface_offset = (oc_start // 16) * p["out_width_stride"] * 16 * read_bytes
                        regs = make_conv2d_regs(
                            1, in_c, tile_in_h, in_w, hw_oc_tile, kh, kw,
                            input_mem_create.dma_addr + input_offset,
                            weight_mem_create.dma_addr + weight_offset,
                            output_mem_create.dma_addr + output_offset + surface_offset,
                            groups=groups,
                            out_width_stride_override=p["out_width_stride"],
                            full_data_bank=True,
                            out_fp16=hw_out_fp16)
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
                        full_data_bank=True,
                        out_fp16=hw_out_fp16))
                input_offset = _align_up(input_offset + input_bytes, 16)
                continue

            input_flat = input_flat.tolist()
            ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
            ct_inputs[:] = input_flat

            task_regs = [make_conv2d_regs(
                            1, in_c, tile_in_h, in_w, out_c, kh, kw,
                            input_mem_create.dma_addr,
                            weight_mem_create.dma_addr,
                            output_mem_create.dma_addr,
                            groups=groups,
                            out_fp16=hw_out_fp16)]

            submit_conv_tasks(task_regs, repeat=_direct_submit_repeat(p, batch, is_spatial, len(tiles)))

            out_c1 = _ceil_div(out_c, unpack_c2) if grouped_spatial else _ceil_div(p["align_out_c"], unpack_c2)
            if not is_spatial:
                out_count = out_c1 * tile_p["out_width_stride"] * unpack_c2
                out_buf = read_output(out_count)
                result[n, :, row_start:row_start + tile_out_h, :] = _unpack_flat_1x1_output(
                    out_buf, out_c, tile_out_h, out_w, tile_p["out_width_stride"], unpack_c2)
                continue
            if grouped_spatial:
                plane_stride = out_h * out_w * unpack_c2 + p["out_width_stride"] * 2
                out_buf = read_output(out_c1 * plane_stride)
                result[n] = _unpack_grouped_spatial_output(out_buf, out_c, out_h, out_w, unpack_c2, plane_stride)
            else:
                out_buf = read_output(out_c1 * tile_p["out_width_stride"] * unpack_c2)
                result[n] = _unpack_flat_1x1_output(out_buf, out_c, out_h, out_w, tile_p["out_width_stride"], unpack_c2)
        if pc_chain_tiles and task_regs:
            submit_repeat = 2 if output_channel_tiles else 1
            submit_conv_tasks(task_regs, repeat=submit_repeat)
        if pc_chain_tiles and not depthwise_channel_tiles:
            out_c1 = _ceil_div(p["align_out_c"], unpack_c2)
            out_buf = read_output(out_c1 * p["out_width_stride"] * unpack_c2)
            result[n] = _unpack_flat_1x1_output(out_buf, hw_sched_out_c, out_h, out_w, p["out_width_stride"], unpack_c2)
    return result[:, :logical_out_c, :logical_out_h, :logical_out_w], return_input_nchw, return_weight_nchw

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

def load_conv_legacy_shapes():
    def literal_dict_call(node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "dict":
            return None
        if node.args:
            return None
        out = {}
        for kw in node.keywords:
            if kw.arg is None:
                return None
            out[kw.arg] = ast.literal_eval(kw.value)
        return out

    conv_path = os.path.join(os.path.dirname(__file__), "conv_legacy.py")
    with open(conv_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=conv_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "shapes" in names and isinstance(node.value, ast.List):
                shapes = []
                for elt in node.value.elts:
                    shape = literal_dict_call(elt)
                    if shape is not None:
                        shapes.append(shape)
                return shapes
    raise RuntimeError("Could not find literal shapes list in conv_legacy.py")

if __name__ == "__main__":
    out_fp16 = "--out" in sys.argv and sys.argv[sys.argv.index("--out") + 1] == "fp16"
    shapes = load_conv_legacy_shapes()
    name_width = max(len(s["name"]) for s in shapes)
    in_shape_width = max(len(f"{s['in_c']}x{s['in_h']}x{s['in_w']}") for s in shapes)
    out_shape_width = max(len(f"{s['out_c']}x{s['in_h'] - s['kh'] + 1}x{s['in_w'] - s['kw'] + 1}") for s in shapes)

    failed = 0
    for s in shapes:
        name, batch, in_c, in_h, in_w, out_c, weight_in_c, kh, kw, groups = \
            s["name"], s["batch"], s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["weight_in_c"], s["kh"], s["kw"], s["groups"]
        result, inp, wt = run_conv2d(batch, in_c, out_c, kh, kw, (in_h, in_w), groups=groups, weight_in_c=weight_in_c, out_fp16=out_fp16)
        expected = compute_expected_nchw(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups=groups)
        if out_fp16:
            expected = expected.astype(np.float16)
        md = float(np.max(np.abs(result.astype(np.float64) - expected)))
        ok = np.allclose(result, expected, atol=0.2) and not np.any(np.isinf(result))
        assert ok, f"{s["name"]} failed"
        out_h = in_h - kh + 1
        out_w = in_w - kw + 1
        in_shape = f"{in_c}x{in_h}x{in_w}"
        out_shape = f"{out_c}x{out_h}x{out_w}"
        print(f"  {name:<{name_width}s} {in_shape:<{in_shape_width}s} -> {out_shape:<{out_shape_width}s} kh={kh} kw={kw} g={groups} out={'fp16' if out_fp16 else 'fp32'}  {'PASS' if ok else 'FAIL'}  (max_diff={md:.4f})")
        if not ok:
            failed += 1
    print(f"\nsimple_conv_fp16 matrix: {'ALL PASS' if failed == 0 else f'{failed} FAILURES'}")
    os.close(fd)
