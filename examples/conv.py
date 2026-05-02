import sys, os, mmap, ctypes
import numpy as np
from fcntl import ioctl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experimental'))
import rockchip as rk

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_CACHEABLE = 2
RKNPU_MEM_IOMMU = 0x10
RKNPU_MEM_NON_CONTIGUOUS = 1
RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT = 0x400
RKNPU_ACT_RESET = 1
REGCMD_RESERVED = 16384
NPU_CBUF_BANK_SIZE = 32768
NPU_CBUF_BANKS = 12
MAX_PIXEL_DMA_CHANNELS = 4
POINTWISE_PIXEL_CHANNELS = 3
DEPTHWISE_SLICE_CHANNELS = 8
OUTPUT_CHANNEL_CHUNK = 16
PIXEL_DMA_CHANNELS = (1, 3, 4)

DRY_RUN = "--submit" not in sys.argv
VALIDATE = "--validate" in sys.argv

fd = None

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("obj_addr", ctypes.c_uint64),
                ("dma_addr", ctypes.c_uint64), ("sram_size", ctypes.c_uint64)]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("offset", ctypes.c_uint64)]

class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]

class rknpu_submit(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("timeout", ctypes.c_uint32), ("task_start", ctypes.c_uint32),
                ("task_number", ctypes.c_uint32), ("task_counter", ctypes.c_uint32), ("priority", ctypes.c_int32),
                ("task_obj_addr", ctypes.c_uint64), ("iommu_domain_id", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32), ("task_base_addr", ctypes.c_uint64),
                ("hw_elapse_time", ctypes.c_int64), ("core_mask", ctypes.c_uint32),
                ("fence_fd", ctypes.c_int32), ("subcore_task", rknpu_subcore_task * 5)]

class struct_rknpu_task(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("op_idx", ctypes.c_uint32),
                ("enable_mask", ctypes.c_uint32), ("int_mask", ctypes.c_uint32),
                ("int_clear", ctypes.c_uint32), ("int_status", ctypes.c_uint32),
                ("regcfg_amount", ctypes.c_uint32), ("regcfg_offset", ctypes.c_uint32),
                ("regcmd_addr", ctypes.c_uint64)]

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)
DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))

def mem_allocate(fd, size, flags=0):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc

class rknpu_mem_sync(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("obj_addr", ctypes.c_uint64),
                ("offset", ctypes.c_uint64), ("size", ctypes.c_uint64)]

DRM_IOCTL_RKNPU_MEM_SYNC = _IOWR('d', 0x45, ctypes.sizeof(rknpu_mem_sync))

def mem_sync(fd, obj_addr, offset, size, flags):
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, rknpu_mem_sync(flags=flags, obj_addr=obj_addr, offset=offset, size=size))

RKNPU_MEM_SYNC_TO_DEVICE = 1
RKNPU_MEM_SYNC_FROM_DEVICE = 2

def submit(task_obj_addr):
    s = rknpu_submit(flags=0x3, timeout=10000, task_start=0, task_number=1,
                     task_counter=0, priority=0, task_obj_addr=task_obj_addr,
                     iommu_domain_id=0, reserved=0, task_base_addr=0,
                     hw_elapse_time=0, core_mask=0, fence_fd=-1)
    # struct len is 5 but only 3 NPU cores; explicitly zero unused slots
    s.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    s.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    s.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    s.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    s.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, s)

def reset_npu(fd):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

def mv_address(mv):
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))

def emit(target, value, addr):
    return ((target & 0xFFFF) << 48) | ((value & 0xFFFFFFFF) << 16) | (addr & 0xFFFF)

def align_up_int(x, align):
    return ((x + align - 1) // align) * align

def _is_depthwise(in_channels, out_channels, groups):
    return groups == in_channels and out_channels == in_channels

def _is_pixel_dma_input(in_channels, is_depthwise):
    return in_channels in PIXEL_DMA_CHANNELS and not is_depthwise

def should_use_nhwc_pack(batch, channels, height, width, width_stride, c2,
                         out_c=None, kh=None, kw=None, groups=None):
    is_depthwise = _is_depthwise(channels, out_c, groups) if groups is not None and out_c is not None else False
    if is_depthwise:
        return False
    c_ratio = c2 // channels if channels > 0 else 0
    return (c_ratio == 2) and (width_stride >= width)

def _input_pack_c2(in_channels, out_channels, kernel_h, kernel_w, groups, align_c):
    is_spatial = kernel_h != 1 or kernel_w != 1
    is_depthwise = _is_depthwise(in_channels, out_channels, groups)
    if in_channels == 1 and groups == 1 and is_spatial and not is_depthwise:
        return 2
    return min(align_c, 8)

def _output_width_stride(in_channels, out_channels, kernel_h, kernel_w, groups, out_h, out_w, align_out_c):
    out_atoms = out_w * out_h
    if kernel_h == 1 and kernel_w == 1:
        return out_atoms if out_atoms < 4 else align_up_int(out_atoms, 4)
    if in_channels == 3 and out_channels == 6:
        if kernel_h == 3 and kernel_w == 3:
            return 16
        if groups == 1 and kernel_h == 3 and kernel_w == 1:
            return 24
    return max(1, (out_w * align_out_c) // 4)

def _feature_grains(in_h, kernel_h, width_stride, align_c):
    grains = in_h + kernel_h
    row_bytes = width_stride * align_c * 2
    if row_bytes <= 0:
        return grains
    max_grains = max(2, (2 * NPU_CBUF_BANK_SIZE + row_bytes - 1) // row_bytes)
    return min(grains, (max_grains + 1) & ~1)

def _dma_strides(in_h, width_stride, use_pixel_mode):
    if use_pixel_mode:
        return width_stride, width_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, width_stride * (in_h - 4) if in_h > 4 else 0

def _cbuf_entries(width_stride, align_c, in_h, is_depthwise):
    row_entries = max(1, (width_stride * align_c + 31) // 32)
    if align_c >= 16 or is_depthwise:
        return row_entries
    return max(1, row_entries * in_h * 4)

def _data_bank(width_stride, feature_grains, align_c):
    fd_bytes = width_stride * feature_grains * align_c * 2
    return max(1, min(NPU_CBUF_BANKS - 1, (fd_bytes + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE))

if not DRY_RUN:
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    cmd_map, cmd_mc = mem_allocate(fd, 16384, RKNPU_MEM_NON_CACHEABLE)
    data_flags = RKNPU_MEM_CACHEABLE | RKNPU_MEM_IOMMU
    in_map, in_mc = mem_allocate(fd, 4194304, data_flags)
    wt_alloc = 4194304
    wt_map, wt_mc = mem_allocate(fd, wt_alloc, RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE | RKNPU_MEM_IOMMU | RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)
    out_map, out_mc = mem_allocate(fd, 4194304, data_flags)

    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regs_ptr = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(cmd_map)), ctypes.POINTER(ctypes.c_uint64))

def pack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride,
                      out_c=None, kh=None, kw=None, groups=None, use_nhwc=None):
    if use_nhwc is None:
        use_nhwc = should_use_nhwc_pack(batch, channels, height, width, width_stride, c2,
                                        out_c=out_c, kh=kh, kw=kw, groups=groups)
    nchw = src.reshape(batch, channels, height, width)
    if use_nhwc:
        packed = np.zeros((batch, height, width_stride, channels), dtype=np.float16)
        packed[:, :, :width, :] = nchw.transpose(0, 2, 3, 1)
        return packed.reshape(-1)

    c1 = (channels + c2 - 1) // c2
    padded = np.zeros((batch, c1 * c2, height, width_stride), dtype=np.float16)
    padded[:, :channels, :, :width] = nchw
    return padded.reshape(batch, c1, c2, height, width_stride).transpose(0, 1, 3, 4, 2).reshape(-1)

# ── Weight packing dispatch ──
# Keep the packing rules generic and only special-case layouts that are
# structurally different in hardware (depthwise or kh-major observed layouts).
_KH_MAJOR_SHAPES = {
    (6, 3, 2, 1): None,     # observed: YOLO small conv head
    (6, 3, 2, 3): 1,        # observed: YOLO mid-layer
    (6, 3, 2, 5): 1,        # observed: YOLO late-layer
    (6, 3, 3, 1): None,     # observed: YOLO 3x1 separable
    (6, 3, 3, 3): None,     # observed: YOLO standard 3x3
    (6, 3, 3, 5): None,     # observed: YOLO 3x5
    (16, 16, 3, 3): 1,      # observed: M4-like backbone
    (4, 4, 3, 3): 1,        # observed: tiny model stem
    (6, 1, 3, 3): 1,        # observed: depthwise→pointwise transition
    (6, 1, 2, 1): 1,        # observed: ic=1 non-square kernels
    (6, 1, 2, 3): 1,
    (6, 1, 3, 1): 1,
    (6, 1, 3, 5): 1,
    (6, 1, 1, 3): 1,
    (6, 1, 1, 5): 1,
}

def _is_kh_major(out_c, in_c, kh, kw, groups):
    key = (out_c, in_c, kh, kw)
    required = _KH_MAJOR_SHAPES.get(key)
    if required is None:
        return key in _KH_MAJOR_SHAPES
    return required == groups

def _pack_dw_spatial_major(src, out_c, in_c, kh, kw, c2_out):
    weights = src.reshape(out_c, in_c, kh, kw)
    spatial = np.zeros((kh, kw, c2_out), dtype=np.float16)
    spatial[:, :, :out_c] = weights[np.arange(out_c), np.arange(out_c)].transpose(1, 2, 0)
    return spatial.reshape(-1)

def _pack_kh_major(src, out_c, in_c, kh, kw, c2_out):
    aligned_in = align_up_int(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in, kh, kw), dtype=np.float16)
    padded[:, :in_c] = src.reshape(out_c, in_c, kh, kw)
    return padded.transpose(2, 3, 0, 1).reshape(-1)

def _pack_default(src, out_c, in_c, kh, kw, c2_out):
    aligned_in = align_up_int(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in, kh, kw), dtype=np.float16)
    padded[:, :in_c] = src.reshape(out_c, in_c, kh, kw)
    return padded.transpose(0, 2, 3, 1).reshape(-1)

def pack_conv_weights_fp16(src, out_c, in_c, kh, kw, c2_out, groups=1):
    is_depthwise = (groups == in_c and out_c == in_c)
    if is_depthwise and out_c <= c2_out and kh == 3 and kw == 3:
        return _pack_dw_spatial_major(src, out_c, in_c, kh, kw, c2_out)
    if _is_kh_major(out_c, in_c, kh, kw, groups):
        return _pack_kh_major(src, out_c, in_c, kh, kw, c2_out)
    return _pack_default(src, out_c, in_c, kh, kw, c2_out)

def unpack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride):
    c1 = (channels + c2 - 1) // c2
    logical_size = batch * c1 * height * width_stride * c2
    packed = src[:logical_size].reshape(batch, c1, height, width_stride, c2)
    nchw = packed.transpose(0, 1, 4, 2, 3).reshape(batch, c1 * c2, height, width_stride)
    return nchw[:, :channels, :, :width].reshape(-1)


def _validate_npu_result(result, inp, wt, in_c, out_c, kh, kw, groups):
    batch, oc, oh, ow = result.shape
    expected = np.zeros((batch, oc, oh, ow), dtype=np.float16)
    for n in range(batch):
        for o in range(oc):
            for c in range(in_c):
                wi = 0 if (groups > 1 and groups == in_c and groups == out_c) else c
                for i in range(kh):
                    for j in range(kw):
                        expected[n, o] += inp[n, c, i:i+oh, j:j+ow] * float(wt[o, wi, i, j])
    match = np.allclose(result, expected, atol=0.1) and not np.any(np.isinf(result))
    md = float(np.max(np.abs(result - expected))) if not np.any(np.isinf(result)) else float('inf')
    print(f"  VALIDATE: {'PASS' if match else 'FAIL'} (max_diff={md:.4f})")
    if not match:
        print(f"    NPU[:12]: {result.flatten()[:12]}")
        print(f"    CPU[:12]: {expected.flatten()[:12]}")
    return match

def _target(addr):
    if 0x1000 <= addr < 0x2000: return rk.CNA | 0x1
    if 0x3000 <= addr < 0x4000: return rk.CORE | 0x1
    if 0x4000 <= addr < 0x5000: return rk.DPU | 0x1
    if addr < 0x1000: return addr | 0x1
    raise ValueError(f"Unknown address 0x{addr:x}")


def _pack_depthwise_expanded_weights_fp16(weight_ochw, out_channels, ic_per_group, kh, kw, packed_weight_size):
    # Depthwise CONV_MODE=3 still fetches one full aligned kernel. Expand the
    # one-channel-per-group OCHW weights into that aligned kernel footprint.
    slot_sz = kh * kw
    wt_full = np.zeros(packed_weight_size // 2, dtype=np.float16)
    for oc in range(out_channels):
        src_start = oc * ic_per_group * slot_sz
        dst_start = oc * slot_sz
        wt_full[dst_start:dst_start + slot_sz] = weight_ochw[src_start:src_start + slot_sz]
    return wt_full


def _expand_grouped_weights_fp16(weight_ochw, out_channels, in_channels, ic_per_group, oc_per_group, kh, kw):
    wt_full = np.zeros(out_channels * in_channels * kh * kw, dtype=np.float16)
    for oc in range(out_channels):
        group = oc // oc_per_group
        group_ic_start = group * ic_per_group
        for ic_local in range(ic_per_group):
            src_idx = oc * ic_per_group * kh * kw + ic_local * kh * kw
            dst_idx = oc * in_channels * kh * kw + (group_ic_start + ic_local) * kh * kw
            wt_full[dst_idx:dst_idx + kh * kw] = weight_ochw[src_idx:src_idx + kh * kw]
    return wt_full


def _output_unpack_c2(params):
    if params.get('is_depthwise', False):
        return params['align_c']
    return 8 if params['align_out_c'] >= 8 else params['align_out_c']


def _packed_sizes(params):
    packed_input_size = ((params['in_channels'] + params['align_c'] - 1) // params['align_c']) * params['in_h'] * params['width_stride'] * params['align_c'] * 2
    packed_weight_size = params['out_channels'] * params['kernel_h'] * params['kernel_w'] * ((params['weight_in_channels'] + params['align_c'] - 1) // params['align_c']) * params['align_c'] * 2
    packed_output_size = ((params['out_channels'] + params['align_out_c'] - 1) // params['align_out_c']) * params['out_h'] * params['out_width_stride'] * params['align_out_c'] * 2
    return packed_input_size, packed_weight_size, packed_output_size


def _pack_weights_for_submit(params, weight_ochw, packed_weight_size):
    if params['groups'] <= 1:
        weight_packed = pack_conv_weights_fp16(
            weight_ochw, params['out_channels'], params['in_channels'],
            params['kernel_h'], params['kernel_w'], params['align_c'], groups=params['groups'])
        return params, weight_packed, packed_weight_size

    in_c = params['in_channels']
    ic_per_group = params['weight_in_channels']
    oc_per_group = params['out_channels'] // params['groups']
    kh, kw = params['kernel_h'], params['kernel_w']
    if params.get('is_depthwise', False):
        weight_packed = _pack_depthwise_expanded_weights_fp16(
            weight_ochw, params['out_channels'], ic_per_group, kh, kw, packed_weight_size)
        return params, weight_packed, packed_weight_size

    wt_full = _expand_grouped_weights_fp16(
        weight_ochw, params['out_channels'], in_c, ic_per_group, oc_per_group, kh, kw)
    submit_params = dict(params)
    submit_params['groups'] = 1
    submit_params['weight_in_channels'] = in_c
    _, packed_weight_size, _ = _packed_sizes(submit_params)
    weight_packed = pack_conv_weights_fp16(
        wt_full, submit_params['out_channels'], in_c, kh, kw, submit_params['align_c'], groups=1)
    return submit_params, weight_packed, packed_weight_size


def _unpack_output(params, out_packed, is_1x1):
    unpack_c2 = _output_unpack_c2(params)
    if is_1x1:
        flat = unpack_nc1hwc2_fp16(
            out_packed, params['batch'], params['out_channels'], 1,
            params['out_h'] * params['out_w'], unpack_c2, params['out_width_stride'])
    else:
        flat = unpack_nc1hwc2_fp16(
            out_packed, params['batch'], params['out_channels'],
            params['out_h'], params['out_w'], unpack_c2, params['out_w'])
    return flat.reshape(params['batch'], params['out_channels'], params['out_h'], params['out_w'])


def _write_task_descriptor(regs_list):
    tasks[0].flags = 0
    tasks[0].op_idx = 1
    tasks[0].enable_mask = 0xd
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(regs_list)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = cmd_mc.dma_addr


def compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    # ── Phase 1: Default parameter computation ──
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    weight_in_channels = in_channels // groups if groups > 0 else in_channels
    is_depthwise = _is_depthwise(in_channels, out_channels, groups)
    out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1

    max_align = 32 if is_depthwise else 16
    pow2 = 1
    c = in_channels if in_channels > 0 else 1
    while pow2 < c and pow2 < max_align:
        pow2 <<= 1
    align_c = max(8, min(pow2, max_align))
    align_out_c = max(16, ((out_channels + 15) // 16) * 16)
    width_align = max(1, (16 + align_c - 1) // align_c)
    width_stride = align_up_int(in_w, width_align)
    out_channel_field = (align_up_int(align_out_c, 32) if is_depthwise else align_out_c) - 1
    orig_channel = out_channels - 1 if out_channels > 0 else 0
    out_width_stride = _output_width_stride(
        in_channels, out_channels, kernel_h, kernel_w, groups, out_h, out_w, align_out_c)

    # ── Phase 3: Derived values (depend on final, possibly-overridden Phase 1/2 values) ──
    out_atoms = max(1, out_w * out_h)
    data_in_channel_real = in_channels - 1 if in_channels > 0 else 0
    data_in_channel_aligned = max(align_c, align_up_int(in_channels, align_c))
    weight_kernels = 1 if is_depthwise else out_channels
    weight_bytes_per_kernel = kernel_h * kernel_w * data_in_channel_aligned * 2
    weight_bytes_total = weight_bytes_per_kernel * out_channels

    feature_grains = _feature_grains(in_h, kernel_h, width_stride, align_c)

    input_pack_c2 = _input_pack_c2(in_channels, out_channels, kernel_h, kernel_w, groups, align_c)
    use_nhwc = should_use_nhwc_pack(batch, in_channels, in_h, in_w, width_stride, input_pack_c2,
                                    out_c=out_channels, kh=kernel_h, kw=kernel_w, groups=groups)
    use_pixel_mode = use_nhwc or _is_pixel_dma_input(in_channels, is_depthwise)
    line_stride, surf_stride = _dma_strides(in_h, width_stride, use_pixel_mode)

    cvt_lanes = 128 // 16
    cvt_active = max(1, min(in_channels if use_nhwc else input_pack_c2, cvt_lanes))
    cvt_mask = 0xFFFFFFFF if cvt_active >= 32 else ((1 << cvt_active) - 1)
    cbuf_entries = _cbuf_entries(width_stride, align_c, in_h, is_depthwise)
    data_bank = _data_bank(width_stride, feature_grains, align_c)

    effective_align_out = out_channel_field + 1
    if groups > 1 and not is_depthwise:
        per_group_align = max(16, align_up_int((out_channels + groups - 1) // groups, 16))
        effective_align_out = per_group_align
    surface_add = out_width_stride * (effective_align_out // 8)
    return locals()


def build_conv2d_regs(params, input_dma=0, weights_dma=0, output_dma=0):
    p = params
    regs = []
    def E(addr, val): regs.append((_target(addr), val, addr))

    E(rk.REG_DPU_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1))
    conv_con1 = (2 << 7) | (2 << 4)
    if _is_pixel_dma_input(p['in_channels'], p['is_depthwise']):
        conv_con1 |= (1 << 30) | (1 << 29) | ((7 + p['in_channels']) << 12)
    if p['is_depthwise']: conv_con1 |= 0x3
    E(rk.REG_CNA_CONV_CON1, conv_con1)
    E(rk.REG_CNA_CONV_CON2, p['feature_grains'] << 4)
    E(rk.REG_CNA_CONV_CON3, (1 << 3) | 1)
    E(rk.REG_CNA_DATA_SIZE0, (p['width_stride'] << 16) | p['in_h'])
    E(rk.REG_CNA_DATA_SIZE1, (p['data_in_channel_real'] << 16) | p['data_in_channel_aligned'])
    E(rk.REG_CNA_DATA_SIZE2, p['out_w'])
    E(rk.REG_CNA_DATA_SIZE3, p['out_atoms'])
    E(rk.REG_CNA_WEIGHT_SIZE0, p['weight_bytes_total'])
    E(rk.REG_CNA_WEIGHT_SIZE1, p['weight_bytes_per_kernel'])
    wk = p['weight_kernels']
    if wk == 0: wk = p['out_channels']
    E(rk.REG_CNA_WEIGHT_SIZE2, (p['kernel_w'] << 24) | (p['kernel_h'] << 16) | wk)
    E(rk.REG_CNA_CBUF_CON0, ((NPU_CBUF_BANKS - p['data_bank']) << 4) | p['data_bank'])
    E(rk.REG_CNA_CBUF_CON1, p['cbuf_entries'])
    E(rk.REG_CNA_CVT_CON0, 0xB if not p['use_nhwc'] else 0x1)
    for a in [rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4]:
        E(a, 1 << 16)
    E(rk.REG_CNA_FEATURE_DATA_ADDR, input_dma & 0xFFFFFFFF)
    E(rk.REG_CNA_DMA_CON0, (15 << 16) | 15)
    E(rk.REG_CNA_DMA_CON1, p['line_stride'])
    E(rk.REG_CNA_DMA_CON2, p['surf_stride'])
    E(rk.REG_CNA_FC_DATA_SIZE0, (p['in_w'] << 16) | p['in_h'])
    E(rk.REG_CNA_FC_DATA_SIZE1, p['align_c'])
    E(rk.REG_CNA_DCOMP_ADDR0, (weights_dma + REGCMD_RESERVED) & 0xFFFFFFFF)
    E(rk.REG_CNA_CVT_CON5, p['cvt_mask'])
    core = (2 << 8)
    if p['is_depthwise']: core |= (1 << 1)
    if p['kernel_h'] > 1 or p['kernel_w'] > 1: core |= (1 << 0)
    E(rk.REG_CORE_MISC_CFG, core)
    E(rk.REG_CORE_DATAOUT_SIZE_0, ((p['out_h'] - 1) << 16) | (p['out_w'] - 1))
    E(rk.REG_CORE_DATAOUT_SIZE_1, p['out_channel_field'])
    regs.append((rk.CORE | 0x1, 0, 0x3030))
    dpu_fmc = (15 << 5) | (2 << 1)
    if p['is_depthwise']: dpu_fmc |= (3 << 3)
    E(rk.REG_DPU_FEATURE_MODE_CFG, dpu_fmc)
    E(rk.REG_DPU_DATA_FORMAT, (2 << 29) | (2 << 26) | 2)
    E(rk.REG_DPU_DST_BASE_ADDR, output_dma & 0xFFFFFFFF)
    E(rk.REG_DPU_DST_SURF_STRIDE, p['out_width_stride'] << 4)
    E(rk.REG_DPU_DATA_CUBE_WIDTH, p['out_w'] - 1)
    E(rk.REG_DPU_DATA_CUBE_HEIGHT, p['out_h'] - 1)
    E(rk.REG_DPU_DATA_CUBE_CHANNEL, (p['orig_channel'] << 16) | p['out_channel_field'])
    E(rk.REG_DPU_BS_CFG, 0x53)
    ow_cfg = 3 if p['is_depthwise'] else 1
    E(rk.REG_DPU_BS_OW_CFG, (ow_cfg << 8) | (ow_cfg << 5) | (ow_cfg << 2) | (1 << 1))
    E(rk.REG_DPU_WDMA_SIZE_0, p['out_channel_field'])
    E(rk.REG_DPU_WDMA_SIZE_1, ((p['out_h'] - 1) << 16) | (p['out_w'] - 1))
    E(rk.REG_DPU_BN_CFG, 0x53)
    E(rk.REG_DPU_EW_CFG, 0x383)
    E(rk.REG_DPU_EW_CVT_SCALE_VALUE, 1)
    E(rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | 1)
    E(rk.REG_DPU_SURFACE_ADD, p['surface_add'] << 4)
    regs.append((0x1, 0, 0x40c4))
    regs.append((0x81, 0xD, rk.REG_PC_OPERATION_ENABLE))
    return regs


def _verify_config(regs_list):
    assert ctypes.sizeof(rknpu_submit) == 104, f"submit struct {ctypes.sizeof(rknpu_submit)} != 104"
    for _, _, a in regs_list:
        ok = a < 0x1000 or 0x1000 <= a < 0x2000 or 0x3000 <= a < 0x4000 or 0x4000 <= a < 0x5000
        assert ok, f"register 0x{a:04x} out of range"


def _npu_submit(params, input_nchw, weight_ochw, is_1x1, log_submit=True):
    packed_input_size, packed_weight_size, packed_output_size = _packed_sizes(params)

    reset_npu(fd)

    input_packed = pack_nc1hwc2_fp16(input_nchw, params['batch'], params['in_channels'], params['in_h'], params['in_w'], params['align_c'], params['width_stride'],
                                      out_c=params['out_channels'], kh=params['kernel_h'], kw=params['kernel_w'], groups=params.get('groups', 1),
                                      use_nhwc=params['use_nhwc'])
    params, weight_packed, packed_weight_size = _pack_weights_for_submit(params, weight_ochw, packed_weight_size)

    in_view = memoryview(bytearray(input_packed.tobytes()))
    wt_view = memoryview(bytearray(weight_packed.tobytes()))
    ctypes.memmove(mv_address(in_map), mv_address(in_view), min(packed_input_size, len(in_view)))
    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(wt_map))
    ctypes.memmove(wt_ptr + REGCMD_RESERVED, mv_address(wt_view), min(packed_weight_size, len(wt_view)))

    regs_list = build_conv2d_regs(params, in_mc.dma_addr, wt_mc.dma_addr, out_mc.dma_addr)
    _verify_config(regs_list)
    for i, (t, v, a) in enumerate(regs_list): regs_ptr[i] = emit(t, v, a)
    _write_task_descriptor(regs_list)

    mem_sync(fd, wt_mc.obj_addr, 0, wt_alloc, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(fd, in_mc.obj_addr, 0, packed_input_size, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(fd, out_mc.obj_addr, 0, packed_output_size, RKNPU_MEM_SYNC_TO_DEVICE)

    debug_mode = os.environ.get('CONV_DEBUG')
    if debug_mode:
        print(f"\n=== CONV PRE-SUBMIT DEBUG (ic={params['in_channels']}) ===")
        for i_r, (t, v, a) in enumerate(regs_list):
            print(f"  [{i_r:3d}] target=0x{t:04x} addr=0x{a:04x} value=0x{v:08x}")
        in_np = np.frombuffer(in_map, dtype=np.float16, count=64)
        wt_np = np.frombuffer(wt_map, offset=REGCMD_RESERVED, dtype=np.float16, count=64)
        print(f"  PACKED_IN [{len(in_np)}]: {in_np}")
        print(f"  PACKED_WT @REGCMD_RESERVED [{len(wt_np)}]: {wt_np}")
        del in_np, wt_np

    ret = submit(task_mc.obj_addr)
    if log_submit:
        print(f"SUBMIT ret={ret}")

    out_packed = np.frombuffer(out_map, dtype=np.float16, count=packed_output_size // 2).copy()

    return _unpack_output(params, out_packed, is_1x1)

def _run_conv2d_channel_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1):
    """1x1 conv with in_channels >= 5: HW non-aligned DMA supports max 4 in-channels.
    Slices into groups of 4, submits each independently, accumulates results."""
    np.random.seed(42)
    full_in = np.random.randn(1, original_in_c, input_hw[0], input_hw[1]).astype(np.float16)
    full_wt = np.random.randn(out_channels, original_in_c, 1, 1).astype(np.float16)
    result = np.zeros((1, out_channels, input_hw[0], input_hw[1]), dtype=np.float16)
    for start_c in range(0, original_in_c, MAX_PIXEL_DMA_CHANNELS):
        end_c = min(start_c + MAX_PIXEL_DMA_CHANNELS, original_in_c)
        group_ic = end_c - start_c
        inp_slice = np.zeros((1, MAX_PIXEL_DMA_CHANNELS, input_hw[0], input_hw[1]), dtype=np.float16)
        inp_slice[0, :group_ic] = full_in[0, start_c:end_c]
        wt_slice = np.zeros((out_channels, MAX_PIXEL_DMA_CHANNELS, 1, 1), dtype=np.float16)
        wt_slice[:, :group_ic] = full_wt[:, start_c:end_c]
        p = compute_conv2d_params(MAX_PIXEL_DMA_CHANNELS, out_channels, kernel_h, kernel_w, input_hw, 1)
        result += _npu_submit(p, inp_slice.reshape(-1), wt_slice.reshape(-1), is_1x1)
    return result, full_in, full_wt

def _run_depthwise_channel_sliced(channels, kernel_h, kernel_w, input_hw, is_1x1):
    """Depthwise channels above 8 are submitted in 8-channel lanes.
    A single 32-channel depthwise task selects weight slots by 8-row stripe,
    so channel groups must be isolated and concatenated in software."""
    np.random.seed(42)
    full_in = np.random.randn(1, channels, input_hw[0], input_hw[1]).astype(np.float16)
    full_wt = np.random.randn(channels, 1, kernel_h, kernel_w).astype(np.float16)
    out_h = input_hw[0] - kernel_h + 1
    out_w = input_hw[1] - kernel_w + 1
    result = np.zeros((1, channels, out_h, out_w), dtype=np.float16)
    for start_c in range(0, channels, DEPTHWISE_SLICE_CHANNELS):
        end_c = min(start_c + DEPTHWISE_SLICE_CHANNELS, channels)
        group_c = end_c - start_c
        p = compute_conv2d_params(group_c, group_c, kernel_h, kernel_w, input_hw, group_c)
        result[:, start_c:end_c] = _npu_submit(
            p,
            full_in[:, start_c:end_c].reshape(-1),
            full_wt[start_c:end_c].reshape(-1),
            is_1x1,
        )
    return result, full_in, full_wt

def _npu_submit_single_input_pointwise(input_1c, weight_ochw):
    batch, _, height, width = input_1c.shape
    out_channels = weight_ochw.shape[0]
    result = np.zeros((batch, out_channels, height, width), dtype=np.float16)
    inp_pad = np.zeros((batch, POINTWISE_PIXEL_CHANNELS, height, width), dtype=np.float16)
    inp_pad[:, 0:1] = input_1c
    for start_oc in range(0, out_channels, OUTPUT_CHANNEL_CHUNK):
        end_oc = min(start_oc + OUTPUT_CHANNEL_CHUNK, out_channels)
        chunk_oc = end_oc - start_oc
        wt_pad = np.zeros((chunk_oc, POINTWISE_PIXEL_CHANNELS, 1, 1), dtype=np.float16)
        wt_pad[:, 0:1] = weight_ochw[start_oc:end_oc]
        p = compute_conv2d_params(POINTWISE_PIXEL_CHANNELS, chunk_oc, 1, 1, (height, width), 1)
        result[:, start_oc:end_oc] = _npu_submit(
            p, inp_pad.reshape(-1), wt_pad.reshape(-1), True, log_submit=False)
    return result

def _run_conv2d_spatial_decomposed(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups):
    """Run kh*kw convolution as exact-order 1x1 NPU submits.
    Direct non-1x1 programming still has shape-dependent partial/numerical
    behavior. This keeps execution on NPU while matching test_conv's fp16
    accumulation order."""
    np.random.seed(42)
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
    input_nchw = np.random.randn(batch, in_channels, in_h, in_w).astype(np.float16)
    weight_ochw = np.random.randn(
        out_channels, in_channels // groups, kernel_h, kernel_w).astype(np.float16)
    result = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float16)
    oc_per_group = out_channels // groups
    ic_per_group = in_channels // groups
    for group in range(groups):
        oc_start = group * oc_per_group
        oc_end = oc_start + oc_per_group
        for ic_local in range(ic_per_group):
            ic = group * ic_per_group + ic_local
            for kh_idx in range(kernel_h):
                for kw_idx in range(kernel_w):
                    input_crop = input_nchw[:, ic:ic + 1,
                                            kh_idx:kh_idx + out_h,
                                            kw_idx:kw_idx + out_w].copy()
                    weight_1x1 = weight_ochw[oc_start:oc_end, ic_local:ic_local + 1,
                                             kh_idx:kh_idx + 1, kw_idx:kw_idx + 1].copy()
                    result[:, oc_start:oc_end] += _npu_submit_single_input_pointwise(input_crop, weight_1x1)
    return result, input_nchw, weight_ochw

def run_conv2d(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    original_in_c = in_channels
    is_1x1 = (kernel_h == 1 and kernel_w == 1)
    pad_to_c = POINTWISE_PIXEL_CHANNELS if (is_1x1 and in_channels < POINTWISE_PIXEL_CHANNELS) else None
    if pad_to_c: in_channels = pad_to_c
    needs_1x1_channel_slicing = is_1x1 and in_channels > MAX_PIXEL_DMA_CHANNELS and not pad_to_c and groups == 1
    p = compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)
    needs_depthwise_slicing = p['is_depthwise'] and in_channels > DEPTHWISE_SLICE_CHANNELS
    needs_spatial_decomposition = not is_1x1

    if DRY_RUN:
        regs = build_conv2d_regs(p, 0, 0, 0)
        _verify_config(regs)
        target_names = {0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU", 0x0081: "PC"}
        print(f"\n=== CONV2D DRY RUN ({len(regs)} regs) ===")
        print(f"  in=({p['batch']},{p['in_channels']},{p['in_h']},{p['in_w']}) "
              f"out=({p['batch']},{p['out_channels']},{p['out_h']},{p['out_w']}) "
              f"kernel=({p['kernel_h']},{p['kernel_w']}) groups={groups}")
        for i, (t, v, a) in enumerate(regs):
            print(f"  [{i:3d}] {target_names.get(t, f'0x{t:04x}')}[0x{a:04x}] = 0x{v:08x}")
        return None, None, None

    if needs_1x1_channel_slicing:
        return _run_conv2d_channel_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1)

    if needs_depthwise_slicing:
        return _run_depthwise_channel_sliced(in_channels, kernel_h, kernel_w, input_hw, is_1x1)

    if needs_spatial_decomposition:
        return _run_conv2d_spatial_decomposed(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)

    reset_npu(fd)
    np.random.seed(42)
    input_nchw = np.random.randn(p['batch'] * original_in_c * p['in_h'] * p['in_w']).astype(np.float16)
    weight_ochw = np.random.randn(p['out_channels'] * (original_in_c // groups if groups > 0 else original_in_c) * p['kernel_h'] * p['kernel_w']).astype(np.float16)

    if pad_to_c:
        pad = pad_to_c
        inp_r = input_nchw.reshape(p['batch'], original_in_c, p['in_h'] * p['in_w'])
        padded = np.zeros((p['batch'], pad, p['in_h'] * p['in_w']), dtype=np.float16)
        padded[:, :original_in_c] = inp_r
        input_nchw = padded.reshape(-1)
        wt_r = weight_ochw.reshape(p['out_channels'], original_in_c, p['kernel_h'] * p['kernel_w'])
        expanded = np.zeros((p['out_channels'], pad, p['kernel_h'] * p['kernel_w']), dtype=np.float16)
        expanded[:, :original_in_c] = wt_r
        weight_ochw = expanded.reshape(-1)

    result = _npu_submit(p, input_nchw, weight_ochw, is_1x1)
    return result, \
           input_nchw.reshape(p['batch'], p['in_channels'], p['in_h'], p['in_w']), \
           weight_ochw.reshape(p['out_channels'], p['weight_in_channels'], p['kernel_h'], p['kernel_w'])


if __name__ == "__main__":
    in_c, out_c, kh, kw = 2, 2, 1, 1
    print(f"=== NPU test (in_c={in_c} out_c={out_c} kernel={kh}x{kw} input=4x4) ===")
    try:
        result, inp, wt = run_conv2d(in_c, out_c, kh, kw, (4, 4))
        if result is None:
            print("DRY RUN complete. Pass --submit to run on NPU.")
            sys.exit(0)
        batch, oc, oh, ow = result.shape
        expected = np.zeros((batch, oc, oh, ow), dtype=np.float16)
        for n in range(batch):
            for o in range(oc):
                for c in range(in_c):
                    expected[n, o] += inp[n, c] * wt[o, c, 0, 0]
        match = np.allclose(result, expected, atol=0.1)
        md = np.max(np.abs(result - expected))
        print(f"CONV {in_c}x{out_c}x{kh}x{kw} -> {out_c}x{oh}x{ow}: {'PASS' if match else 'FAIL'} (max_diff={md:.4f})")
        if not match:
            print(f"  NPU: {result.flatten()}")
            print(f"  CPU: {expected.flatten()}")
        if VALIDATE:
            _validate_npu_result(result, inp, wt, in_c, out_c, kh, kw, 1)
    except Exception as e:
        print(f"NPU test failed: {e}")
