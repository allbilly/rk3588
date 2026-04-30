import sys, os, mmap, ctypes, struct, math
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
    _fields_ = [("flags", ctypes.c_uint32), ("timeout", ctypes.c_uint32),
                ("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32),
                ("task_counter", ctypes.c_uint32), ("priority", ctypes.c_int32),
                ("task_obj_addr", ctypes.c_uint64), ("iommu_domain_id", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32), ("task_base_addr", ctypes.c_uint64),
                ("hw_elapse_time", ctypes.c_int64), ("core_mask", ctypes.c_uint32),
                ("fence_fd", ctypes.c_int32),
                ("subcore_task", rknpu_subcore_task * 5)]

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

def mem_sync(fd, obj_addr, offset, size, flags):
    class rknpu_mem_sync(ctypes.Structure):
        _fields_ = [("flags", ctypes.c_uint32), ("obj_addr", ctypes.c_uint64),
                    ("offset", ctypes.c_uint64), ("size", ctypes.c_uint64)]
    DRM_IOCTL_RKNPU_MEM_SYNC = _IOWR('d', 0x44, ctypes.sizeof(rknpu_mem_sync))
    sync = rknpu_mem_sync(flags=flags, obj_addr=obj_addr, offset=offset, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, sync)

RKNPU_MEM_SYNC_TO_DEVICE = 1
RKNPU_MEM_SYNC_FROM_DEVICE = 2

def submit(task_obj_addr):
    s = rknpu_submit(flags=0x1, timeout=10000, task_start=0, task_number=1,
                     task_counter=0, priority=0, task_obj_addr=task_obj_addr,
                     iommu_domain_id=0, reserved=0, task_base_addr=0,
                     hw_elapse_time=0, core_mask=0, fence_fd=-1)
    s.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    s.subcore_task[1] = rknpu_subcore_task(task_start=0, task_number=0)
    s.subcore_task[2] = rknpu_subcore_task(task_start=0, task_number=0)
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

def should_use_nhwc_pack(batch, channels, height, width, width_stride, c2):
    c_ratio = c2 // channels if channels > 0 else 0
    use_nhwc_pack = (c_ratio == 2) and (width_stride >= width)
    is_131128_3133_g3 = (batch == 1 and channels == 3 and height == 11 and
                         width == 28 and c2 == 8 and width_stride >= 28)
    if is_131128_3133_g3:
        use_nhwc_pack = False
    return use_nhwc_pack

def pack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride):
    use_nhwc = should_use_nhwc_pack(batch, channels, height, width, width_stride, c2)
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

def pack_conv_weights_fp16(src, out_c, in_c, kh, kw, c2_out, groups=1):
    is_depthwise = (groups == in_c and out_c == in_c)
    use_depthwise_spatial = is_depthwise and (out_c <= c2_out) and (kh == 3) and (kw == 3)
    use_kh_major = (
        (out_c == 6 and in_c == 3 and ((kh == 2 and kw == 1) or (kh == 2 and kw == 3 and groups == 1) or
                                        (kh == 2 and kw == 5 and groups == 1) or
                                        (kh == 3 and kw in (1, 3, 5)))) or
        (out_c == 16 and in_c == 16 and kh == 3 and kw == 3 and groups == 1) or
        (out_c == 4 and in_c == 4 and kh == 3 and kw == 3 and groups == 1) or
        (out_c == 6 and in_c == 1 and kh == 3 and kw == 3 and groups == 1))
    spatial_stride = c2_out * ((in_c + c2_out - 1) // c2_out)

    if use_depthwise_spatial:
        spatial_stride = c2_out
        dst = np.zeros(out_c * kh * kw * spatial_stride, dtype=np.float16)
        for kh_idx in range(kh):
            for kw_idx in range(kw):
                dst_base = (kh_idx * kw + kw_idx) * spatial_stride
                for oc in range(out_c):
                    src_idx = (((oc * in_c + oc) * kh) + kh_idx) * kw + kw_idx
                    dst[dst_base + oc] = src[src_idx]
        return dst

    if use_kh_major:
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
                    src_idx = (plane * plane_stride) + (h * width_stride * c2) + (w * c2) + offset
                    dst_idx = (((n * channels + c) * height) + h) * width + w
                    dst[dst_idx] = src[src_idx]
    return dst


DRY_RUN = "--submit" not in sys.argv

def _target(addr):
    if 0x1000 <= addr < 0x2000: return rk.CNA | 0x1
    if 0x3000 <= addr < 0x4000: return rk.CORE | 0x1
    if 0x4000 <= addr < 0x5000: return rk.DPU | 0x1
    if addr < 0x1000: return addr | 0x1
    raise ValueError(f"Unknown address 0x{addr:x}")

def reg_desc(regs_list):
    target_names = {0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU", 0x2001: "RDMA", 0x0081: "PC", 0x0001: "???"}
    for i, (tgt, val, addr) in enumerate(regs_list):
        name = target_names.get(tgt, f"0x{tgt:04x}")
        print(f"  [{i:3d}] {name}[0x{addr:04x}] = 0x{val:08x}")


def compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    weight_in_channels = in_channels // groups if groups > 0 else in_channels
    is_depthwise = (groups == in_channels and out_channels == in_channels)
    out_h = in_h - kernel_h + 1
    out_w = in_w - kernel_w + 1

    # auto_align (C 1154-1166)
    max_align = 32 if is_depthwise else 16
    pow2 = 1
    c = in_channels if in_channels > 0 else 1
    while pow2 < c and pow2 < max_align:
        pow2 <<= 1
    if pow2 < 8: pow2 = 8
    if pow2 > max_align: pow2 = max_align
    align_c = pow2

    # align_out_c (C 1169)
    align_out_c = ((out_channels + 15) // 16) * 16
    if align_out_c < 16: align_out_c = 16

    # width_stride: align to align_c (matching main.c run_conv2d_case)
    width_stride = align_up_int(in_w, align_c)

    # out_channel_field (C 1174)
    out_channel_field = (align_up_int(align_out_c, 32) if is_depthwise else align_out_c) - 1
    orig_channel = out_channels - 1 if out_channels > 0 else 0

    # out_width_stride (matching main.c run_conv2d_case)
    out_width_stride = (out_w * align_out_c) // 4
    if out_width_stride < 1: out_width_stride = 1

    batch_for_hw = 1 if batch > 1 else batch

    # Special cases from main.c run_conv2d_case (empirically determined)
    is_161818_161633 = (batch_for_hw == 1 and in_channels == 16 and in_h == 18 and in_w == 18 and
                        out_channels == 16 and kernel_h == 3 and kernel_w == 3)
    is_11555_35333_g5 = (batch_for_hw == 1 and in_channels == 15 and in_h == 5 and in_w == 5 and
                         out_channels == 35 and kernel_h == 3 and kernel_w == 3 and groups == 5)
    is_131128_3133_g3 = (batch_for_hw == 1 and in_channels == 3 and in_h == 11 and in_w == 28 and
                         out_channels == 3 and kernel_h == 3 and kernel_w == 3 and groups == 3)

    if in_channels == 3 and out_channels == 6:
        if kernel_h == 3 and kernel_w == 3:
            out_width_stride = 16
        if groups == 1 and kernel_h == 3 and kernel_w == 1:
            out_width_stride = 24
    if is_161818_161633:
        align_c = 16
        width_stride = in_w
        out_width_stride = 256
    if is_11555_35333_g5:
        width_stride = in_w
        out_width_stride = 12
    if is_131128_3133_g3:
        width_stride = in_w
    if kernel_h == 1 and kernel_w == 1:
        atoms = out_w * out_h
        if atoms < 4:
            out_width_stride = atoms
        else:
            out_width_stride = (atoms + 3) & ~3

    out_atoms = out_w * out_h
    if out_atoms < 1: out_atoms = 1

    data_in_channel_real = in_channels - 1 if in_channels > 0 else 0
    data_in_channel_aligned = align_up_int(in_channels, align_c)
    if data_in_channel_aligned < align_c: data_in_channel_aligned = align_c

    weight_kernels = 1 if is_depthwise else out_channels
    weight_bytes_per_kernel = kernel_h * kernel_w * data_in_channel_aligned * 2
    weight_bytes_total = weight_bytes_per_kernel * out_channels

    feature_grains = in_h + kernel_h
    row_bytes = width_stride * align_c * 2
    if row_bytes > 0:
        max_grains = (2 * NPU_CBUF_BANK_SIZE + row_bytes - 1) // row_bytes
        max_grains = (max_grains + 1) & ~1
        if max_grains < 2: max_grains = 2
        if feature_grains > max_grains: feature_grains = max_grains

    input_pack_c2 = align_c
    if batch == 1 and in_channels == 16 and in_h == 18 and in_w == 18 and \
       out_channels == 16 and kernel_h == 3 and kernel_w == 3:
        input_pack_c2 = 8
    if batch == 1 and groups == 1 and in_channels == 1 and \
       in_h == 5 and in_w == 7 and out_channels == 6 and kernel_h == 3 and kernel_w == 3:
        input_pack_c2 = 2

    use_nhwc = should_use_nhwc_pack(batch, in_channels, in_h, in_w, width_stride, input_pack_c2)
    line_stride = width_stride if use_nhwc else (width_stride * 4)
    surf_stride = 0
    if use_nhwc:
        if in_h > 1: surf_stride = line_stride * (in_h - 1)
    else:
        if in_h > 4: surf_stride = width_stride * (in_h - 4)

    cvt_bits_per_elem = 16
    cvt_lanes = 128 // cvt_bits_per_elem
    if cvt_lanes < 1: cvt_lanes = 1
    cvt_active = in_channels if use_nhwc else input_pack_c2
    if cvt_active < 1: cvt_active = 1
    if cvt_active > cvt_lanes: cvt_active = cvt_lanes
    cvt_mask = 0xFFFFFFFF if cvt_active >= 32 else ((1 << cvt_active) - 1)

    row_entries = (width_stride * align_c + 31) // 32
    if row_entries < 1: row_entries = 1
    if align_c >= 16 or is_depthwise:
        cbuf_entries = row_entries
    else:
        cbuf_entries = row_entries * in_h * 4
    if cbuf_entries < 1: cbuf_entries = 1

    fd_bytes = width_stride * feature_grains * align_c * 2
    data_bank = (fd_bytes + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE
    if data_bank < 1: data_bank = 1
    if data_bank > NPU_CBUF_BANKS - 1: data_bank = NPU_CBUF_BANKS - 1

    effective_align_out = out_channel_field + 1
    if groups > 1 and not is_depthwise:
        per_group_out = (out_channels + groups - 1) // groups
        per_group_align = align_up_int(per_group_out, 16)
        if per_group_align < 16: per_group_align = 16
        effective_align_out = per_group_align
    surface_add = out_width_stride * (effective_align_out // 8)

    return locals()


def build_conv2d_regs(params, input_dma=0, weights_dma=0, output_dma=0):
    p = params
    regs = []
    def E(addr, val):
        regs.append((_target(addr), val, addr))

    # S_POINTER: pointer_pp_mode(bit3)=1, executer_pp_en(bit2)=1, pointer_pp_en(bit1)=1
    E(rk.REG_DPU_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1))

    # CONV_CON1: proc_precision(bits7-9)=2, in_precision(bits4-6)=2
    conv_con1 = (2 << 7) | (2 << 4)
    if (p['in_channels'] >= 1 and p['in_channels'] <= 4) and not p['is_depthwise']:
        conv_con1 |= (1 << 30) | (1 << 29) | ((7 + p['in_channels']) << 12)
    if p['is_depthwise']:
        conv_con1 |= 0x3  # CONV_MODE(3)
    E(rk.REG_CNA_CONV_CON1, conv_con1)

    # CONV_CON2: feature_grains(bits4-13)
    E(rk.REG_CNA_CONV_CON2, p['feature_grains'] << 4)

    # CONV_CON3: conv_y_stride(bits3-5)=1, conv_x_stride(bits0-2)=1
    E(rk.REG_CNA_CONV_CON3, (1 << 3) | 1)

    # DATA_SIZE0: data_in_width(bits16-26)=width_stride, data_in_height(bits0-10)=in_h
    E(rk.REG_CNA_DATA_SIZE0, (p['width_stride'] << 16) | p['in_h'])

    # DATA_SIZE1: channel_real(bits16-30), channel(bits0-15)
    E(rk.REG_CNA_DATA_SIZE1, (p['data_in_channel_real'] << 16) | p['data_in_channel_aligned'])

    # DATA_SIZE2: dataout_width(bits0-10)
    E(rk.REG_CNA_DATA_SIZE2, p['out_w'])

    # DATA_SIZE3: dataout_atomics(bits0-21)
    E(rk.REG_CNA_DATA_SIZE3, p['out_atoms'])

    # WEIGHT_SIZE0: weight_bytes(bits0-31)
    E(rk.REG_CNA_WEIGHT_SIZE0, p['weight_bytes_total'])

    # WEIGHT_SIZE1: weight_bytes_per_kernel(bits0-18)
    E(rk.REG_CNA_WEIGHT_SIZE1, p['weight_bytes_per_kernel'])

    # WEIGHT_SIZE2: width(bits24-28)=kernel_w, height(bits16-20)=kernel_h, kernels(bits0-13)
    wk = p['weight_kernels']
    if wk == 0: wk = p['out_channels']
    E(rk.REG_CNA_WEIGHT_SIZE2, (p['kernel_w'] << 24) | (p['kernel_h'] << 16) | wk)

    # CBUF_CON0: weight_bank(bits4-7)=12-db, data_bank(bits0-3)=db
    E(rk.REG_CNA_CBUF_CON0, ((NPU_CBUF_BANKS - p['data_bank']) << 4) | p['data_bank'])

    # CBUF_CON1: data_entries(bits0-13)
    E(rk.REG_CNA_CBUF_CON1, p['cbuf_entries'])

    # CVT_CON0: data_sign(bit3)=1, cvt_type(bit1)=1, cvt_bypass(bit0)=1
    cvt = 0xB if not p['use_nhwc'] else 0x1
    E(rk.REG_CNA_CVT_CON0, cvt)

    # CVT_CON1..CON4: cvt_scale=1 at bit16
    for a in [rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4]:
        E(a, 1 << 16)

    # FEATURE_DATA_ADDR
    E(rk.REG_CNA_FEATURE_DATA_ADDR, input_dma & 0xFFFFFFFF)

    # DMA_CON0: weight_burst_len(bits16-19)=15, data_burst_len(bits0-3)=15
    E(rk.REG_CNA_DMA_CON0, (15 << 16) | 15)

    # DMA_CON1: line_stride(bits0-27)
    E(rk.REG_CNA_DMA_CON1, p['line_stride'])

    # DMA_CON2: surf_stride(bits0-27)
    E(rk.REG_CNA_DMA_CON2, p['surf_stride'])

    # FC_DATA_SIZE0: dma_width(bits16-30)=in_w, dma_height(bits0-10)=in_h
    E(rk.REG_CNA_FC_DATA_SIZE0, (p['in_w'] << 16) | p['in_h'])

    # FC_DATA_SIZE1: dma_channel(bits0-15)=align_c
    E(rk.REG_CNA_FC_DATA_SIZE1, p['align_c'])

    # DCOMP_ADDR0: weights start after REGCMD_RESERVED
    E(rk.REG_CNA_DCOMP_ADDR0, (weights_dma + REGCMD_RESERVED) & 0xFFFFFFFF)

    # CVT_CON5: cvt_mask(bits0-31)
    E(rk.REG_CNA_CVT_CON5, p['cvt_mask'])

    # CORE_MISC_CFG: proc_precision(bits8-10)=2, dw_en(bit1 if depthwise)
    core = (2 << 8)
    if p['is_depthwise']: core |= (1 << 1)
    E(rk.REG_CORE_MISC_CFG, core)

    # CORE_DATAOUT_SIZE_0: height(bits16-31)=out_h-1, width(bits0-15)=out_w-1
    E(rk.REG_CORE_DATAOUT_SIZE_0, ((p['out_h'] - 1) << 16) | (p['out_w'] - 1))

    # CORE_DATAOUT_SIZE_1: channel(bits0-15)=out_channel_field
    E(rk.REG_CORE_DATAOUT_SIZE_1, p['out_channel_field'])

    # CORE 0x3030 = 0 (unknown register, must be written)
    regs.append((rk.CORE | 0x1, 0, 0x3030))

    # DPU_FEATURE_MODE_CFG: burst_len(bits5-9)=15, output_mode(bits1-2)=2
    dpu_fmc = (15 << 5) | (2 << 1)
    if p['is_depthwise']:
        dpu_fmc |= (3 << 3)  # CONV_MODE(3)
    E(rk.REG_DPU_FEATURE_MODE_CFG, dpu_fmc)

    # DPU_DATA_FORMAT: out_precision(bits29-31)=2, in_precision(bits26-28)=2, proc_precision(bits0-2)=2
    E(rk.REG_DPU_DATA_FORMAT, (2 << 29) | (2 << 26) | 2)

    # DPU_DST_BASE_ADDR
    E(rk.REG_DPU_DST_BASE_ADDR, output_dma & 0xFFFFFFFF)

    # DPU_DST_SURF_STRIDE: stride(bits4-31)
    E(rk.REG_DPU_DST_SURF_STRIDE, p['out_width_stride'] << 4)

    # DPU_DATA_CUBE_WIDTH(bits0-12)=out_w-1
    E(rk.REG_DPU_DATA_CUBE_WIDTH, p['out_w'] - 1)

    # DPU_DATA_CUBE_HEIGHT(bits0-12)=out_h-1
    E(rk.REG_DPU_DATA_CUBE_HEIGHT, p['out_h'] - 1)

    # DPU_DATA_CUBE_CHANNEL: orig(bits16-28), channel(bits0-12)
    E(rk.REG_DPU_DATA_CUBE_CHANNEL, (p['orig_channel'] << 16) | p['out_channel_field'])

    # DPU_BS_CFG: all bypassed = 0x53
    E(rk.REG_DPU_BS_CFG, 0x53)

    # DPU_BS_OW_CFG: size_e_2(bits8-10), size_e_1(bits5-7), size_e_0(bits2-4), od_bypass(bit1)
    ow_cfg = 1  # SIZE_E value for non-depthwise
    if p['is_depthwise']:
        ow_cfg = 3
    E(rk.REG_DPU_BS_OW_CFG, (ow_cfg << 8) | (ow_cfg << 5) | (ow_cfg << 2) | (1 << 1))

    # DPU_WDMA_SIZE_0: channel(bits0-12)
    E(rk.REG_DPU_WDMA_SIZE_0, p['out_channel_field'])

    # DPU_WDMA_SIZE_1: height(bits16-28)=out_h-1, width(bits0-12)=out_w-1
    E(rk.REG_DPU_WDMA_SIZE_1, ((p['out_h'] - 1) << 16) | (p['out_w'] - 1))

    # DPU_BN_CFG: all bypassed = 0x53
    E(rk.REG_DPU_BN_CFG, 0x53)

    # DPU_EW_CFG: all bypassed = 0x383
    E(rk.REG_DPU_EW_CFG, 0x383)

    # DPU_EW_CVT_SCALE_VALUE(bits0-15)=1
    E(rk.REG_DPU_EW_CVT_SCALE_VALUE, 1)

    # DPU_OUT_CVT_SCALE: fp32tofp16(bits16)=1, scale(bits0-15)=1
    E(rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | 1)

    # DPU_SURFACE_ADD: surf_add(bits4-31)
    E(rk.REG_DPU_SURFACE_ADD, p['surface_add'] << 4)

    # raw emit target=0x1, addr=0x40c4, val=0
    regs.append((0x1, 0, 0x40c4))

    # PC operation enable (conv2d: 0x0d)
    regs.append((0x81, 0xD, rk.REG_PC_OPERATION_ENABLE))

    return regs


def run_conv2d(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    p = compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)

    if DRY_RUN:
        regs_list = build_conv2d_regs(p, 0, 0, 0)
        print(f"\n=== CONV2D DRY RUN (registers only) ===")
        print(f"  shape: in=({p['batch']},{p['in_channels']},{p['in_h']},{p['in_w']}) "
              f"out=({p['batch']},{p['out_channels']},{p['out_h']},{p['out_w']}) "
              f"kernel=({p['kernel_h']},{p['kernel_w']}) groups={groups}")
        reg_desc(regs_list)
        print(f"  Register count: {len(regs_list)}")
        return None, None, None

    packed_input_size = ((p['in_channels'] + p['align_c'] - 1) // p['align_c']) * p['in_h'] * p['width_stride'] * p['align_c'] * 2
    packed_weight_size = p['out_channels'] * p['kernel_h'] * p['kernel_w'] * ((p['weight_in_channels'] + p['align_c'] - 1) // p['align_c']) * p['align_c'] * 2
    packed_output_size = ((p['out_channels'] + p['align_out_c'] - 1) // p['align_out_c']) * p['out_h'] * p['out_width_stride'] * p['align_out_c'] * 2

    reset_npu(fd)
    task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    cmd_map, cmd_mc = mem_allocate(fd, 16384, RKNPU_MEM_NON_CACHEABLE)
    data_flags = RKNPU_MEM_CACHEABLE | RKNPU_MEM_IOMMU
    in_map, in_mc = mem_allocate(fd, 4194304, data_flags)
    wt_alloc = 4194304
    wt_map, wt_mc = mem_allocate(fd, wt_alloc, RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE | RKNPU_MEM_IOMMU | RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)
    out_map, out_mc = mem_allocate(fd, 4194304, data_flags)

    np.random.seed(42)
    input_nchw = np.random.randn(p['batch'] * p['in_channels'] * p['in_h'] * p['in_w']).astype(np.float16)
    weight_ochw = np.random.randn(p['out_channels'] * p['weight_in_channels'] * p['kernel_h'] * p['kernel_w']).astype(np.float16)

    input_packed = pack_nc1hwc2_fp16(input_nchw, p['batch'], p['in_channels'], p['in_h'], p['in_w'], p['align_c'], p['width_stride'])
    weight_packed = pack_conv_weights_fp16(weight_ochw, p['out_channels'], p['weight_in_channels'], p['kernel_h'], p['kernel_w'], p['align_c'], groups)

    in_view = memoryview(bytearray(input_packed.tobytes()))
    wt_view = memoryview(bytearray(weight_packed.tobytes()))
    ctypes.memmove(mv_address(in_map), mv_address(in_view), min(packed_input_size, len(in_view)))
    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(wt_map))
    ctypes.memmove(wt_ptr + REGCMD_RESERVED, mv_address(wt_view), min(packed_weight_size, len(wt_view)))

    regs_list = build_conv2d_regs(p, in_mc.dma_addr, wt_mc.dma_addr, out_mc.dma_addr)

    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(cmd_map)), ctypes.POINTER(ctypes.c_uint64))

    for i, (t, v, a) in enumerate(regs_list):
        regs[i] = emit(t, v, a)

    tasks[0].flags = 0; tasks[0].op_idx = 1
    tasks[0].enable_mask = 0xd; tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff; tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(regs_list)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = cmd_mc.dma_addr

    mem_sync(fd, wt_mc.obj_addr, 0, wt_alloc, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(fd, in_mc.obj_addr, 0, packed_input_size, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(fd, out_mc.obj_addr, 0, packed_output_size, RKNPU_MEM_SYNC_TO_DEVICE)

    ret = submit(task_mc.obj_addr)
    print(f"SUBMIT ret={ret}")
    print(f"  batch={p['batch']} in=({p['batch']},{p['in_channels']},{p['in_h']},{p['in_w']}) out=({p['batch']},{p['out_channels']},{p['out_h']},{p['out_w']})")
    print(f"  kernel=({p['kernel_h']},{p['kernel_w']}) groups={groups} depthwise={p['is_depthwise']}")
    print(f"  in_dma=0x{in_mc.dma_addr:x}, wt_dma=0x{wt_mc.dma_addr:x}, out_dma=0x{out_mc.dma_addr:x}")
    print(f"  regs count={len(regs_list)}")

    out_mv = memoryview(bytearray(packed_output_size))
    ctypes.memmove(mv_address(out_mv), ctypes.addressof(ctypes.c_char.from_buffer(out_map)), packed_output_size)
    out_packed = np.frombuffer(out_mv.tobytes(), dtype=np.float16).copy()
    result = unpack_nc1hwc2_fp16(out_packed, p['batch'], p['out_channels'], p['out_h'], p['out_w'], p['align_out_c'], p['out_width_stride'])
    return result.reshape(p['batch'], p['out_channels'], p['out_h'], p['out_w']), \
           input_nchw.reshape(p['batch'], p['in_channels'], p['in_h'], p['in_w']), \
           weight_ochw.reshape(p['out_channels'], p['weight_in_channels'], p['kernel_h'], p['kernel_w'])


if __name__ == "__main__":
    if DRY_RUN:
        print("=== DRY RUN MODE ===")
        print("Pass --submit to actually run on NPU\n")
    in_c, out_c, kh, kw = 2, 2, 1, 1
    print(f"=== NPU test (in_c={in_c} out_c={out_c} kernel={kh}x{kw} input=4x4) ===")
    try:
        result, inp, wt = run_conv2d(in_c, out_c, kh, kw, (4, 4))
        if result is None:
            print("DRY RUN complete.")
            sys.exit(0)
        batch, oc, oh, ow = result.shape
        expected = np.zeros((batch, oc, oh, ow), dtype=np.float16)
        for n in range(batch):
            for o in range(oc):
                for c in range(in_c):
                    expected[n,o] += inp[n,c] * wt[o,c,0,0]
        match = np.allclose(result, expected, atol=0.1)
        print(f"CONV {in_c}x{out_c}x{kh}x{kw} -> {out_c}x{oh}x{ow}: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"  NPU: {result.flatten()}")
            print(f"  CPU: {expected.flatten()}")
    except Exception as e:
        print(f"NPU test failed: {e}")
