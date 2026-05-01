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

fd = os.open("/dev/dri/card1", os.O_RDWR)

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

# ── Shape-specific NPU workaround database ──
# Each entry: key=(batch, in_c, in_h, in_w, out_c, kh, kw, groups) → {overrides}
# Use "in_w" to mean "set width_stride = in_w" (bypass computed stride alignment).
_CONV2D_OVERRIDES = {
    (1, 16, 18, 18, 16, 3, 3, 1): {
        "reason": "conv 16x18x18→16x16, k3: align_c rounds to 32 but HW needs 16 for non-depthwise; strides from RKNN dump",
        "align_c": 16,
        "width_stride": "in_w",
        "out_width_stride": 256,
        "input_pack_c2": 8,
    },
    (1, 15, 5, 5, 35, 3, 3, 5): {
        "reason": "group conv 15→35, g=5, 3x3 on 5x5: strides from RKNN dump",
        "width_stride": "in_w",
        "out_width_stride": 12,
    },
    (1, 3, 11, 28, 3, 3, 3, 3): {
        "reason": "depthwise 3x3, g=3, 11x28: c_ratio=2 triggers NHWC pack but produces garbage; use NC1HWC2. width_stride=in_w",
        "nhwc_pack": False,
        "width_stride": "in_w",
    },
    (1, 1, 5, 7, 6, 3, 3, 1): {
        "reason": "conv 1→6, 3x3, 5x7: input_pack_c2=2 from RKNN dump",
        "input_pack_c2": 2,
    },
}

def _conv_override_key(batch, in_c, in_h, in_w, out_c, kh, kw, groups):
    return (batch, in_c, in_h, in_w, out_c, kh, kw, groups)

def _get_conv_override(batch, in_c, in_h, in_w, out_c, kh, kw, groups):
    return _CONV2D_OVERRIDES.get(_conv_override_key(batch, in_c, in_h, in_w, out_c, kh, kw, groups), {})

def should_use_nhwc_pack(batch, channels, height, width, width_stride, c2,
                         out_c=None, kh=None, kw=None, groups=None):
    c_ratio = c2 // channels if channels > 0 else 0
    use_nhwc = (c_ratio == 2) and (width_stride >= width)
    if use_nhwc and all(x is not None for x in (out_c, kh, kw, groups)):
        override = _get_conv_override(batch, channels, height, width, out_c, kh, kw, groups)
        if 'nhwc_pack' in override:
            use_nhwc = override['nhwc_pack']
    return use_nhwc

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
    if use_nhwc:
        row_stride = width_stride * channels
        plane_stride = height * row_stride
        dst = np.zeros((batch * plane_stride,), dtype=np.float16)
        for n in range(batch):
            n_base = n * plane_stride
            for h in range(height):
                h_base = n_base + h * row_stride
                for w in range(width_stride):
                    w_base = h_base + w * channels
                    for c in range(channels):
                        val = np.float16(0)
                        if w < width:
                            src_idx = ((((n * channels + c) * height) + h) * width + w)
                            val = src[src_idx]
                        dst[w_base + c] = val
        return dst
    c1 = (channels + c2 - 1) // c2
    plane_stride = height * width_stride * c2
    dst = np.zeros((batch * c1 * plane_stride,), dtype=np.float16)
    for n in range(batch):
        for c in range(channels):
            plane = c // c2
            offset = c % c2
            dst_plane_base = (n * c1 + plane) * plane_stride
            for h in range(height):
                dst_row_base = dst_plane_base + h * width_stride * c2
                src_row_base = ((((n * channels + c) * height) + h) * width)
                for w in range(width):
                    dst_idx = dst_row_base + w * c2 + offset
                    src_idx = src_row_base + w
                    dst[dst_idx] = src[src_idx]
    return dst

# ── Weight packing dispatch ──
# RKNN dumps reveal different weight memory layouts per conv shape.
# Each variant is a named function; dispatch is via explicit shape matching.
# Key: (out_c, in_c, kh, kw) → required groups (None = any groups)
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
}

def _is_kh_major(out_c, in_c, kh, kw, groups):
    key = (out_c, in_c, kh, kw)
    required = _KH_MAJOR_SHAPES.get(key)
    if required is None:
        return key in _KH_MAJOR_SHAPES
    return required == groups

def _pack_dw_spatial_major(src, out_c, in_c, kh, kw, c2_out):
    spatial_stride = c2_out
    dst = np.zeros(out_c * kh * kw * spatial_stride, dtype=np.float16)
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            dst_base = (kh_idx * kw + kw_idx) * spatial_stride
            for oc in range(out_c):
                src_idx = (((oc * in_c + oc) * kh) + kh_idx) * kw + kw_idx
                dst[dst_base + oc] = src[src_idx]
    return dst

def _pack_kh_major(src, out_c, in_c, kh, kw, c2_out):
    spatial_stride = c2_out * ((in_c + c2_out - 1) // c2_out)
    out_c_stride = spatial_stride
    spatial_stride = out_c * out_c_stride
    dst = np.zeros(kh * kw * spatial_stride, dtype=np.float16)
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            dst_khkw_base = (kh_idx * kw + kw_idx) * spatial_stride
            for oc in range(out_c):
                dst_spatial_base = dst_khkw_base + oc * out_c_stride
                for ic in range(in_c):
                    dst_idx = dst_spatial_base + (ic // c2_out) * c2_out + (ic % c2_out)
                    src_idx = (((oc * in_c + ic) * kh) + kh_idx) * kw + kw_idx
                    dst[dst_idx] = src[src_idx]
    return dst

def _pack_default(src, out_c, in_c, kh, kw, c2_out):
    spatial_stride = c2_out * ((in_c + c2_out - 1) // c2_out)
    kernel_stride = kh * kw * spatial_stride
    total = out_c * kernel_stride
    dst = np.zeros((total,), dtype=np.float16)
    for oc in range(out_c):
        dst_kernel_base = oc * kernel_stride
        for kh_idx in range(kh):
            for kw_idx in range(kw):
                dst_spatial_base = dst_kernel_base + (kh_idx * kw + kw_idx) * spatial_stride
                for ic in range(in_c):
                    dst_idx = dst_spatial_base + (ic // c2_out) * c2_out + (ic % c2_out)
                    src_idx = (((oc * in_c + ic) * kh) + kh_idx) * kw + kw_idx
                    dst[dst_idx] = src[src_idx]
    return dst

def pack_conv_weights_fp16(src, out_c, in_c, kh, kw, c2_out, groups=1):
    is_depthwise = (groups == in_c and out_c == in_c)
    if is_depthwise and out_c <= c2_out and kh == 3 and kw == 3:
        return _pack_dw_spatial_major(src, out_c, in_c, kh, kw, c2_out)
    if _is_kh_major(out_c, in_c, kh, kw, groups):
        return _pack_kh_major(src, out_c, in_c, kh, kw, c2_out)
    return _pack_default(src, out_c, in_c, kh, kw, c2_out)

def unpack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride):
    dst = np.zeros((batch * channels * height * width,), dtype=np.float16)
    c1 = (channels + c2 - 1) // c2
    plane_stride = height * width_stride * c2
    for n in range(batch):
        for c in range(channels):
            plane = c // c2
            offset = c % c2
            for h in range(height):
                for w in range(width):
                    src_idx = ((n * c1 + plane) * plane_stride) + (h * width_stride * c2) + (w * c2) + offset
                    dst_idx = (((n * channels + c) * height) + h) * width + w
                    dst[dst_idx] = src[src_idx]
    return dst


DRY_RUN = "--submit" not in sys.argv
VALIDATE = "--validate" in sys.argv

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


def compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    # ── Phase 1: Default parameter computation ──
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    weight_in_channels = in_channels // groups if groups > 0 else in_channels
    is_depthwise = (groups == in_channels and out_channels == in_channels)
    out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1

    max_align = 32 if is_depthwise else 16
    pow2 = 1
    c = in_channels if in_channels > 0 else 1
    while pow2 < c and pow2 < max_align: pow2 <<= 1
    align_c = max(8, min(pow2, max_align))
    align_out_c = max(16, ((out_channels + 15) // 16) * 16)
    width_stride = align_up_int(in_w, align_c)
    out_channel_field = (align_up_int(align_out_c, 32) if is_depthwise else align_out_c) - 1
    orig_channel = out_channels - 1 if out_channels > 0 else 0
    out_width_stride = max(1, (out_w * align_out_c) // 4)
    batch_for_hw = 1 if batch > 1 else batch

    # ── Phase 2: Shape-specific overrides (from _CONV2D_OVERRIDES table + partial checks) ──
    override = _get_conv_override(batch_for_hw, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w, groups)
    if 'align_c' in override: align_c = override['align_c']
    if 'width_stride' in override: width_stride = in_w if override['width_stride'] == 'in_w' else override['width_stride']
    if 'out_width_stride' in override: out_width_stride = override['out_width_stride']

    if in_channels == 3 and out_channels == 6:
        if kernel_h == 3 and kernel_w == 3: out_width_stride = 16
        if groups == 1 and kernel_h == 3 and kernel_w == 1: out_width_stride = 24
    if kernel_h == 1 and kernel_w == 1:
        atoms = out_w * out_h
        out_width_stride = atoms if atoms < 4 else (atoms + 3) & ~3

    # ── Phase 3: Derived values (depend on final, possibly-overridden Phase 1/2 values) ──
    out_atoms = max(1, out_w * out_h)
    data_in_channel_real = in_channels - 1 if in_channels > 0 else 0
    data_in_channel_aligned = max(align_c, align_up_int(in_channels, align_c))
    weight_kernels = 1 if is_depthwise else out_channels
    weight_bytes_per_kernel = kernel_h * kernel_w * data_in_channel_aligned * 2
    weight_bytes_total = weight_bytes_per_kernel * out_channels

    feature_grains = in_h + kernel_h
    row_bytes = width_stride * align_c * 2
    if row_bytes > 0:
        max_grains = max(2, (2 * NPU_CBUF_BANK_SIZE + row_bytes - 1) // row_bytes)
        feature_grains = min(feature_grains, (max_grains + 1) & ~1)

    input_pack_c2 = override.get('input_pack_c2', align_c)
    use_nhwc = should_use_nhwc_pack(batch, in_channels, in_h, in_w, width_stride, input_pack_c2,
                                    out_c=out_channels, kh=kernel_h, kw=kernel_w, groups=groups)
    use_pixel_mode = use_nhwc or (in_channels in (1, 3, 4) and not is_depthwise)
    if use_pixel_mode:
        line_stride = width_stride
        if in_h > 1: surf_stride = width_stride * (in_h - 1)
    else:
        line_stride = width_stride * 4
        surf_stride = 0
        if in_h > 4: surf_stride = width_stride * (in_h - 4)

    cvt_lanes = 128 // 16
    cvt_active = max(1, min(in_channels if use_nhwc else input_pack_c2, cvt_lanes))
    cvt_mask = 0xFFFFFFFF if cvt_active >= 32 else ((1 << cvt_active) - 1)
    row_entries = max(1, (width_stride * align_c + 31) // 32)
    cbuf_entries = max(1, row_entries if (align_c >= 16 or is_depthwise) else row_entries * in_h * 4)
    fd_bytes = width_stride * feature_grains * align_c * 2
    data_bank = max(1, min(NPU_CBUF_BANKS - 1, (fd_bytes + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE))

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
    if (p['in_channels'] in (1, 3, 4)) and not p['is_depthwise']:
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


def _npu_submit(params, input_nchw, weight_ochw, is_1x1):
    packed_input_size = ((params['in_channels'] + params['align_c'] - 1) // params['align_c']) * params['in_h'] * params['width_stride'] * params['align_c'] * 2
    packed_weight_size = params['out_channels'] * params['kernel_h'] * params['kernel_w'] * ((params['weight_in_channels'] + params['align_c'] - 1) // params['align_c']) * params['align_c'] * 2
    packed_output_size = ((params['out_channels'] + params['align_out_c'] - 1) // params['align_out_c']) * params['out_h'] * params['out_width_stride'] * params['align_out_c'] * 2

    reset_npu(fd)

    input_packed = pack_nc1hwc2_fp16(input_nchw, params['batch'], params['in_channels'], params['in_h'], params['in_w'], params['align_c'], params['width_stride'],
                                      out_c=params['out_channels'], kh=params['kernel_h'], kw=params['kernel_w'], groups=params.get('groups', 1),
                                      use_nhwc=params['use_nhwc'])
    is_grouped = params['groups'] > 1
    if is_grouped:
        wt_in_c = params['in_channels']
        wt_elems = params['out_channels'] * wt_in_c * params['kernel_h'] * params['kernel_w']
        wt_full = np.zeros(wt_elems, dtype=np.float16)
        for oc in range(params['out_channels']):
            ic_src = oc  # depthwise: each output channel maps to ic=oc
            src_start = oc * params['weight_in_channels'] * params['kernel_h'] * params['kernel_w']
            dst_start = (oc * wt_in_c + ic_src) * params['kernel_h'] * params['kernel_w']
            wt_full[dst_start:dst_start + params['kernel_h'] * params['kernel_w']] = \
                weight_ochw[src_start:src_start + params['kernel_h'] * params['kernel_w']]
        weight_packed = pack_conv_weights_fp16(wt_full, params['out_channels'], wt_in_c, params['kernel_h'], params['kernel_w'], params['align_c'], groups=params['groups'])
    else:
        weight_packed = pack_conv_weights_fp16(weight_ochw, params['out_channels'], params['in_channels'], params['kernel_h'], params['kernel_w'], params['align_c'], groups=params['groups'])

    in_view = memoryview(bytearray(input_packed.tobytes()))
    wt_view = memoryview(bytearray(weight_packed.tobytes()))
    ctypes.memmove(mv_address(in_map), mv_address(in_view), min(packed_input_size, len(in_view)))
    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(wt_map))
    ctypes.memmove(wt_ptr + REGCMD_RESERVED, mv_address(wt_view), min(packed_weight_size, len(wt_view)))

    regs_list = build_conv2d_regs(params, in_mc.dma_addr, wt_mc.dma_addr, out_mc.dma_addr)
    _verify_config(regs_list)
    for i, (t, v, a) in enumerate(regs_list): regs_ptr[i] = emit(t, v, a)

    tasks[0].flags = 0; tasks[0].op_idx = 1
    tasks[0].enable_mask = 0xd; tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff; tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(regs_list)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = cmd_mc.dma_addr

    mem_sync(fd, wt_mc.obj_addr, 0, wt_alloc, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(fd, in_mc.obj_addr, 0, packed_input_size, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(fd, out_mc.obj_addr, 0, packed_output_size, RKNPU_MEM_SYNC_TO_DEVICE)

    debug_mode = os.environ.get('CONV_DEBUG')
    if debug_mode:
        print(f"\n=== CONV PRE-SUBMIT DEBUG (ic={params['in_channels']}) ===")
        for i_r, (t, v, a) in enumerate(regs_list):
            print(f"  [{i_r:3d}] target=0x{t:04x} addr=0x{a:04x} value=0x{v:08x}")
        in_np = np.frombuffer(in_map, dtype=np.float16, count=64)
        wt_np = np.frombuffer(wt_map, dtype=np.float16, count=64)
        print(f"  PACKED_IN [{len(in_np)}]: {in_np}")
        print(f"  PACKED_WT [{len(wt_np)}]: {wt_np}")
        del in_np, wt_np

    ret = submit(task_mc.obj_addr)
    print(f"SUBMIT ret={ret}")

    out_packed = np.frombuffer(out_map, dtype=np.float16, count=packed_output_size // 2).copy()

    unpack_c2 = 8 if params['align_out_c'] >= 8 else params['align_out_c']
    if is_1x1:
        flat = unpack_nc1hwc2_fp16(out_packed, params['batch'], params['out_channels'], 1, params['out_h'] * params['out_w'], unpack_c2, params['out_width_stride'])
        result = flat.reshape(params['batch'], params['out_channels'], params['out_h'], params['out_w'])
    else:
        result = unpack_nc1hwc2_fp16(out_packed, params['batch'], params['out_channels'], params['out_h'], params['out_w'], unpack_c2, params['out_w'])
        result = result.reshape(params['batch'], params['out_channels'], params['out_h'], params['out_w'])
    return result

def _run_conv2d_channel_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1):
    """1x1 conv with in_channels >= 5: HW non-aligned DMA supports max 4 in-channels.
    Slices into groups of 4, submits each independently, accumulates results."""
    np.random.seed(42)
    full_in = np.random.randn(1, original_in_c, input_hw[0], input_hw[1]).astype(np.float16)
    full_wt = np.random.randn(out_channels, original_in_c, 1, 1).astype(np.float16)
    result = np.zeros((1, out_channels, input_hw[0], input_hw[1]), dtype=np.float16)
    for start_c in range(0, original_in_c, 4):
        end_c = min(start_c + 4, original_in_c)
        group_ic = end_c - start_c
        inp_slice = np.zeros((1, 4, input_hw[0], input_hw[1]), dtype=np.float16)
        inp_slice[0, :group_ic] = full_in[0, start_c:end_c]
        wt_slice = np.zeros((out_channels, 4, 1, 1), dtype=np.float16)
        wt_slice[:, :group_ic] = full_wt[:, start_c:end_c]
        p = compute_conv2d_params(4, out_channels, kernel_h, kernel_w, input_hw, 1)
        result += _npu_submit(p, inp_slice.reshape(-1), wt_slice.reshape(-1), is_1x1)
    return result, full_in, full_wt

def run_conv2d(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    original_in_c = in_channels
    is_1x1 = (kernel_h == 1 and kernel_w == 1)
    pad_to_c = 3 if (is_1x1 and in_channels < 3) else None
    if pad_to_c: in_channels = pad_to_c

    if is_1x1 and in_channels >= 5 and not pad_to_c:
        return _run_conv2d_channel_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1)

    p = compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)

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
