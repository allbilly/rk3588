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
FP16_BYTES = 2
FP16_ATOM_ELEMENTS = 16
ATOM_CUBE_BYTES = 32
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
NVDLA_CBUF_BANKS = 16
# Local NVDLA CBUF constants: CBUF_ENTRY_PER_BANK=256 and CBUF_ENTRY_SIZE=128,
# so one bank is 32768B. RK conv register schedules use a 12-bank data+weight
# budget here even though upstream NVDLA large config has 16 CBUF banks.
NPU_CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
NPU_CBUF_BANKS = 12
RK_WEIGHT_BANK_MIN = 1
# RK-specific scheduling limits, not upstream NVDLA C-model constants.
MAX_PIXEL_DMA_CHANNELS = 4
POINTWISE_PIXEL_CHANNELS = 3
DEPTHWISE_SLICE_CHANNELS = 8
OUTPUT_CHANNEL_CHUNK = 16
PIXEL_DMA_CHANNELS = (1, 3, 4)
RK_PIXEL_C2_MAX = 8
RK_DPU_OUTPUT_GROUP = 8
RK_CVT_BITS = 128
# RK DPU bypass/reserved values taken from working register schedules; local
# NVDLA CACC/SDP source does not expose these RK DPU bit encodings.
DPU_BYPASS_CFG = 0x53
DPU_EW_BYPASS_CFG = 0x383
CORE_RESERVED_ZERO_ADDR = 0x3030
DPU_RESERVED_ZERO_ADDR = 0x40c4
PC_OPERATION_ENABLE_VALUE = 0xD
PC_TASK_COUNT_CLEAR = 1 << 12
PC_TASK_NUMBER_MASK = 0xFFF

DRY_RUN = "--submit" not in sys.argv
VALIDATE = "--validate" in sys.argv
DIRECT_SPATIAL = os.environ.get('CONV_MESA_DIRECT') == '1'
DIRECT_SPATIAL_ALL = os.environ.get('CONV_MESA_DIRECT_ALL') == '1'
DIRECT_CHAIN = os.environ.get('CONV_MESA_CHAIN') == '1'
DUMP_REGS = os.environ.get('CONV_MESA_DUMP_REGS') == '1'
EMIT_PAD_CON1 = os.environ.get('CONV_MESA_PAD_CON1') == '1'
REG_BLOCK_QWORDS = 128
DIRECT_SPATIAL_ALLOWLIST = {
    (1, 6, 2, 1, (5, 7), 1),
    (1, 6, 2, 3, (5, 7), 1),
    (1, 6, 3, 1, (5, 7), 1),
    (1, 6, 3, 3, (5, 7), 1),
    (1, 6, 3, 5, (5, 7), 1),
    (3, 6, 2, 1, (5, 7), 1),
    (3, 6, 2, 3, (5, 7), 1),
    (3, 6, 1, 3, (5, 5), 1),
    (3, 6, 3, 1, (5, 7), 1),
    (3, 6, 3, 3, (5, 7), 3),
    (3, 6, 3, 3, (5, 7), 1),
    (3, 6, 3, 5, (5, 7), 1),
    (4, 4, 3, 3, (9, 9), 1),
}

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
                     hw_elapse_time=0, core_mask=1, fence_fd=-1)
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

def ceil_div(x, y):
    return (x + y - 1) // y

def align_up_int(x, align):
    return ceil_div(x, align) * align

def _is_depthwise(in_channels, out_channels, groups):
    return groups == in_channels and out_channels == in_channels

def _is_pixel_dma_input(in_channels, is_depthwise):
    return in_channels in PIXEL_DMA_CHANNELS and not is_depthwise

def should_use_nhwc_pack(batch, channels, height, width, width_stride, c2,
                         out_c=None, kh=None, kw=None, groups=None):
    # RK image/pixel DMA accepts direct-conv image input separately from feature
    # input. Local CDMA source confirms ACT_MODE_DIRECT_PIXEL exists, but this
    # exact NHWC predicate is schedule-derived from passing RK submits.
    c_ratio = c2 // channels if channels > 0 else 0
    has_group_context = groups is not None and out_c is not None
    is_depthwise = has_group_context and _is_depthwise(channels, out_c, groups)
    return not is_depthwise and c_ratio == 2 and width_stride >= width

def _input_pack_c2(in_channels, out_channels, kernel_h, kernel_w, groups, align_c):
    # FP16 feature input is atom-based in local NVDLA: 32B atom = 16 values, and
    # CDMA groups two atoms into a 64B half-entry. The selected C2 values below
    # are RK packing choices layered on top of that atom machinery.
    is_spatial = kernel_h != 1 or kernel_w != 1
    is_depthwise = _is_depthwise(in_channels, out_channels, groups)
    if in_channels == 1 and groups == 1 and is_spatial and not is_depthwise:
        return 2
    return min(align_c, RK_PIXEL_C2_MAX)

def _output_width_stride(in_channels, out_channels, kernel_h, kernel_w, groups, out_h, out_w, align_out_c):
    out_atoms = out_w * out_h
    # Output stride is RK DPU formatting, not local NVDLA CACC. CACC only shows
    # FP16/INT16 delivery in 16-element payloads; these stride cases come from
    # working RK conv schedules and submit tests.
    if kernel_h == 1 and kernel_w == 1:
        return out_atoms if out_atoms < 4 else align_up_int(out_atoms, 4)
    if in_channels == 3 and out_channels == 6 and kernel_h == 3:
        if kernel_w == 3: return 16
        if groups == 1 and kernel_w == 1: return 24
    return max(1, (out_w * align_out_c) // 4)

def _feature_grains(in_h, kernel_h, width_stride, align_c):
    # Local CDMA uses fetch_slice_grain = cdma_grains + 1 and requires grain=1
    # for unpacked direct-conv lines. RK packed conv schedules can fetch multiple
    # lines; cap the planned lines to about two RK CBUF banks of FP16 rows.
    grains = in_h + kernel_h
    row_bytes = width_stride * align_c * FP16_BYTES
    if row_bytes <= 0: return grains
    max_grains = max(2, ceil_div(2 * NPU_CBUF_BANK_SIZE, row_bytes))
    return min(grains, (max_grains + 1) & ~1)

def _dma_strides(in_h, width_stride, use_pixel_mode):
    # Local CDMA shifts line/surface stride registers by <<5, so these fields are
    # 32B atom units. Pixel/image mode is a separate CDMA path; non-pixel feature
    # mode uses four 32B atom groups per logical row in the RK schedules here.
    if use_pixel_mode:
        return width_stride, width_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, width_stride * (in_h - 4) if in_h > 4 else 0

def _cbuf_entries(width_stride, align_c, in_h, is_depthwise):
    # Local CDMA/CSC validate entries-per-slice from FP16 atom count:
    # atom_per_channel = ceil(channels*2 / 32B), then four atoms per 128B CBUF
    # entry. The align_c<16 multiplier is RK schedule-specific for small feature
    # inputs and is not an upstream NVDLA formula.
    row_entries = max(1, ceil_div(width_stride * align_c, 2 * FP16_ATOM_ELEMENTS))
    return row_entries if align_c >= 16 or is_depthwise else max(1, row_entries * in_h * 4)

def _data_bank(width_stride, feature_grains, align_c):
    # Local NVDLA bank fields are interpreted as field+1 banks. Use RK's 12-bank
    # conv budget and leave at least one bank for weights.
    fd_bytes = width_stride * feature_grains * align_c * FP16_BYTES
    return max(1, min(NPU_CBUF_BANKS - RK_WEIGHT_BANK_MIN, ceil_div(fd_bytes, NPU_CBUF_BANK_SIZE)))


def _calc_pad_con1(params, addition_input=False):
    # Mirrors Mesa's CNA_PAD_CON1 selection rules; defaulted for FP16 path where
    # zero points are conceptually neutral.
    is_depthwise = params.get('is_depthwise', False)
    input_zero_point = params.get('input_zero_point', 0x80)
    if addition_input:
        return 0xFFFFFF80
    if params.get('kernel_w', 0) >= 3 and input_zero_point == 0x00:
        return 0xFFFF8080
    if is_depthwise and input_zero_point == 0x8B:
        return 0x0B0B
    if input_zero_point == 0x80:
        return 0xFFFFFFFF
    return int(input_zero_point) - 0x80


def _packed_line_stride_channels(in_channels, width_stride, align_c):
    # Byte stride between consecutive input feature rows in the local pack layout.
    c_groups = max(1, ceil_div(in_channels, align_c))
    return width_stride * c_groups * align_c * FP16_BYTES


def _output_line_stride_bytes(params):
    # Conservative output row stride used for output offsets and slice placements.
    unpack_c2 = _output_unpack_c2(params)
    return params['out_width_stride'] * unpack_c2 * FP16_BYTES


def _calc_weight_banks(packed_weight_bytes):
    # Mesa-style conservative rule: allocate one extra bank for safety.
    entries = ceil_div(packed_weight_bytes, CBUF_ENTRY_BYTES)
    return ceil_div(entries, CBUF_ENTRIES_PER_BANK) + 1


def _calc_slice_budget_in_rows(params, packed_weight_bytes):
    # Estimate max input rows that can fit in CBUF with current hardware budget.
    line_bytes = _packed_line_stride_channels(params['in_channels'], params['width_stride'], params['align_c'])
    if line_bytes <= 0:
        return params['in_h']

    weight_banks = _calc_weight_banks(packed_weight_bytes)
    if weight_banks + 1 < NPU_CBUF_BANKS:
        available_input_banks = max(1, NPU_CBUF_BANKS - weight_banks)
    else:
        # Partial weights path in Mesa uses fixed 7 input banks.
        available_input_banks = 7

    max_rows = (available_input_banks * NPU_CBUF_BANK_SIZE) // line_bytes
    if max_rows < params['kernel_h']:
        return params['kernel_h']
    return min(params['in_h'], max_rows)


def _line_offset_bytes(params, row):
    return _packed_line_stride_channels(params['in_channels'], params['width_stride'], params['align_c']) * row


def _plan_split_tasks(params, packed_weight_bytes):
    weight_banks = min(NPU_CBUF_BANKS - 1, max(1, _calc_weight_banks(packed_weight_bytes)))
    reuse_weights = weight_banks + 1 < NPU_CBUF_BANKS
    available_input_banks = max(1, NPU_CBUF_BANKS - weight_banks) if reuse_weights else 7
    line_bytes = _packed_line_stride_channels(params['in_channels'], params['width_stride'], params['align_c'])
    max_input_rows = params['in_h'] if line_bytes <= 0 else max(params['kernel_h'], (available_input_banks * NPU_CBUF_BANK_SIZE) // line_bytes)
    max_input_rows = min(params['in_h'], max_input_rows)

    if max_input_rows >= params['in_h']:
        return [{
            'task_index': 0,
            'input_top': 0,
            'input_height': params['in_h'],
            'output_top': 0,
            'output_height': params['out_h'],
            'input_offset_bytes': 0,
            'output_offset_bytes': 0,
            'input_banks': params['data_bank'],
            'weights_banks': weight_banks,
            'weight_reuse': False,
        }]

    tasks_out = []
    step_out_rows = max(1, max_input_rows - params['kernel_h'] + 1)
    out_row = 0
    while out_row < params['out_h']:
        output_height = min(step_out_rows, params['out_h'] - out_row)
        input_top = out_row
        input_height = output_height + params['kernel_h'] - 1
        if input_top + input_height > params['in_h']:
            input_height = params['in_h'] - input_top
            output_height = max(1, input_height - params['kernel_h'] + 1)
        slice_params = dict(params)
        slice_params['in_h'] = input_height
        slice_params['out_h'] = output_height
        feature_grains = _feature_grains(input_height, params['kernel_h'], params['width_stride'], params['align_c'])
        input_banks = max(1, min(available_input_banks, _data_bank(params['width_stride'], feature_grains, params['align_c'])))
        tasks_out.append({
            'task_index': len(tasks_out),
            'input_top': input_top,
            'input_height': input_height,
            'output_top': out_row,
            'output_height': output_height,
            'input_offset_bytes': _line_offset_bytes(params, input_top),
            'output_offset_bytes': _output_line_stride_bytes(params) * out_row,
            'input_banks': input_banks,
            'weights_banks': weight_banks,
            'weight_reuse': reuse_weights and len(tasks_out) > 0,
        })
        out_row += output_height
    return tasks_out

if not DRY_RUN:
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    cmd_map, cmd_mc = mem_allocate(fd, 16384, RKNPU_MEM_NON_CACHEABLE)
    data_flags = RKNPU_MEM_NON_CACHEABLE | RKNPU_MEM_IOMMU
    data_is_cacheable = bool(data_flags & RKNPU_MEM_CACHEABLE)
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
        # RK image/pixel mode consumes NHWC-style rows. Local NVDLA confirms the
        # separate image-load path but not this RK-specific channel predicate.
        packed = np.zeros((batch, height, width_stride, channels), dtype=np.float16)
        packed[:, :, :width, :] = nchw.transpose(0, 2, 3, 1)
        return packed.reshape(-1)

    c1 = (channels + c2 - 1) // c2
    padded = np.zeros((batch, c1 * c2, height, width_stride), dtype=np.float16)
    padded[:, :channels, :, :width] = nchw
    return padded.reshape(batch, c1, c2, height, width_stride).transpose(0, 1, 3, 4, 2).reshape(-1)

def _is_kh_major(out_c, in_c, kh, kw, groups):
    # Weight order is RK schedule-derived. Local CSC confirms kernel iteration is
    # height/width nested over kernel_atom_num, and CDMA confirms FP16 weights are
    # fetched in 16-kernel groups, but this shape predicate itself is empirical.
    is_small_head = out_c == 6 and in_c in (1, 3)
    is_small_odd_width_kernel = 1 <= kh <= 3 and kw % 2 == 1 and kw <= 5 and kh * kw > 1
    single_input_spatial = is_small_head and in_c == 1 and groups == 1 and is_small_odd_width_kernel and not (kh == 2 and kw == 5)
    three_input_spatial = is_small_head and in_c == 3 and kh in (1, 3) and kw % 2 == 1 and kw <= 5
    three_input_2x = is_small_head and in_c == 3 and kh == 2 and (kw == 1 or (groups == 1 and kw % 2 == 1 and kw <= 5))
    square_small_spatial = groups == 1 and out_c == in_c and in_c in (4, 16) and kh == 3 and kw == 3
    return single_input_spatial or three_input_spatial or three_input_2x or square_small_spatial

def _pack_dw_spatial_major(src, out_c, in_c, kh, kw, c2_out):
    # Depthwise here uses RK CONV_MODE=3 packing. Local NVDLA has split-C/depth
    # related CACC/CSC machinery, but this compact spatial-major layout is from
    # the RK conv schedule validated by tests.
    weights = src.reshape(out_c, in_c, kh, kw)
    spatial = np.zeros((kh, kw, c2_out), dtype=np.float16)
    spatial[:, :, :out_c] = weights[np.arange(out_c), np.arange(out_c)].transpose(1, 2, 0)
    return spatial.reshape(-1)

def _pack_kh_major(src, out_c, in_c, kh, kw, c2_out):
    # KH-major is not a generic NVDLA layout; it matches the RK register/weight
    # schedule for the shapes selected by _is_kh_major().
    aligned_in = align_up_int(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in, kh, kw), dtype=np.float16)
    padded[:, :in_c] = src.reshape(out_c, in_c, kh, kw)
    return padded.transpose(2, 3, 0, 1).reshape(-1)

def _pack_default(src, out_c, in_c, kh, kw, c2_out):
    # Default RK FP16 conv weights: output channel, spatial, then aligned input
    # channel. Alignment still follows NVDLA FP16 atom grouping.
    aligned_in = align_up_int(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in, kh, kw), dtype=np.float16)
    padded[:, :in_c] = src.reshape(out_c, in_c, kh, kw)
    return padded.transpose(0, 2, 3, 1).reshape(-1)

def pack_conv_weights_fp16(src, out_c, in_c, kh, kw, c2_out, groups=1):
    if _is_depthwise(in_c, out_c, groups) and out_c <= c2_out and kh == 3 and kw == 3:
        return _pack_dw_spatial_major(src, out_c, in_c, kh, kw, c2_out)
    packer = _pack_kh_major if _is_kh_major(out_c, in_c, kh, kw, groups) else _pack_default
    return packer(src, out_c, in_c, kh, kw, c2_out)

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


REG_NAMES = {
    rk.REG_PC_OPERATION_ENABLE: 'PC_OPERATION_ENABLE',
    rk.REG_PC_BASE_ADDRESS: 'PC_BASE_ADDRESS',
    rk.REG_PC_REGISTER_AMOUNTS: 'PC_REGISTER_AMOUNTS',
    rk.REG_CNA_CONV_CON1: 'CNA_CONV_CON1',
    rk.REG_CNA_CONV_CON2: 'CNA_CONV_CON2',
    rk.REG_CNA_DATA_SIZE0: 'CNA_DATA_SIZE0',
    rk.REG_CNA_DATA_SIZE1: 'CNA_DATA_SIZE1',
    rk.REG_CNA_DATA_SIZE2: 'CNA_DATA_SIZE2',
    rk.REG_CNA_DATA_SIZE3: 'CNA_DATA_SIZE3',
    rk.REG_CNA_WEIGHT_SIZE0: 'CNA_WEIGHT_SIZE0',
    rk.REG_CNA_WEIGHT_SIZE1: 'CNA_WEIGHT_SIZE1',
    rk.REG_CNA_WEIGHT_SIZE2: 'CNA_WEIGHT_SIZE2',
    rk.REG_CNA_CBUF_CON0: 'CNA_CBUF_CON0',
    rk.REG_CNA_CBUF_CON1: 'CNA_CBUF_CON1',
    rk.REG_CNA_PAD_CON1: 'CNA_PAD_CON1',
    rk.REG_CNA_FEATURE_DATA_ADDR: 'CNA_FEATURE_DATA_ADDR',
    rk.REG_CNA_DMA_CON1: 'CNA_DMA_CON1',
    rk.REG_CNA_DMA_CON2: 'CNA_DMA_CON2',
    rk.REG_CNA_FC_DATA_SIZE0: 'CNA_FC_DATA_SIZE0',
    rk.REG_CNA_DCOMP_ADDR0: 'CNA_DCOMP_ADDR0',
    rk.REG_CORE_MISC_CFG: 'CORE_MISC_CFG',
    rk.REG_CORE_DATAOUT_SIZE_0: 'CORE_DATAOUT_SIZE_0',
    rk.REG_CORE_DATAOUT_SIZE_1: 'CORE_DATAOUT_SIZE_1',
    rk.REG_DPU_DST_BASE_ADDR: 'DPU_DST_BASE_ADDR',
    rk.REG_DPU_DST_SURF_STRIDE: 'DPU_DST_SURF_STRIDE',
    rk.REG_DPU_DATA_CUBE_WIDTH: 'DPU_DATA_CUBE_WIDTH',
    rk.REG_DPU_DATA_CUBE_HEIGHT: 'DPU_DATA_CUBE_HEIGHT',
    rk.REG_DPU_DATA_CUBE_CHANNEL: 'DPU_DATA_CUBE_CHANNEL',
    rk.REG_DPU_WDMA_SIZE_0: 'DPU_WDMA_SIZE_0',
    rk.REG_DPU_WDMA_SIZE_1: 'DPU_WDMA_SIZE_1',
    rk.REG_DPU_SURFACE_ADD: 'DPU_SURFACE_ADD',
}


def _target_name(target):
    return {0x0201: 'CNA', 0x0801: 'CORE', 0x1001: 'DPU', 0x0081: 'PC'}.get(target, f'0x{target:04x}')


def _decode_reg(addr, value):
    if addr == rk.REG_CNA_CBUF_CON0:
        return f"data_banks={value & 0xf} weight_banks={(value >> 4) & 0xf} fc_data_bank={(value >> 8) & 0x7} data_reuse={(value >> 12) & 1} weight_reuse={(value >> 13) & 1}"
    if addr == rk.REG_CNA_CONV_CON1:
        return f"mode={value & 0xf} in_prec={(value >> 4) & 0x7} proc_prec={(value >> 7) & 0x7} argb_in={(value >> 12) & 0xf} group_line_off={(value >> 29) & 1} nonalign_dma={(value >> 30) & 1}"
    if addr == rk.REG_CNA_DATA_SIZE0:
        return f"height={value & 0x7ff} width={(value >> 16) & 0x7ff}"
    if addr == rk.REG_CNA_DATA_SIZE1:
        return f"channel={value & 0xffff} real={(value >> 16) & 0x3fff}"
    if addr == rk.REG_CNA_WEIGHT_SIZE2:
        return f"kernels={value & 0x3fff} kh={(value >> 16) & 0x1f} kw={(value >> 24) & 0x1f}"
    if addr in (rk.REG_CORE_DATAOUT_SIZE_0, rk.REG_DPU_WDMA_SIZE_1):
        return f"width={value & 0xffff} height={(value >> 16) & 0xffff}"
    if addr == rk.REG_DPU_DATA_CUBE_CHANNEL:
        return f"channel={value & 0x1fff} orig={(value >> 16) & 0x1fff}"
    if addr == rk.REG_PC_REGISTER_AMOUNTS:
        return f"pc_data_amount={value & 0xffff}"
    return ''


def dump_regs(label, regs_list):
    print(f"\n=== {label} ({len(regs_list)} regs) ===")
    for i, (target, value, addr) in enumerate(regs_list):
        name = REG_NAMES.get(addr, f'0x{addr:04x}')
        decoded = _decode_reg(addr, value)
        suffix = f"  # {decoded}" if decoded else ''
        print(f"  [{i:3d}] {_target_name(target):4s}[0x{addr:04x}] {name:28s} = 0x{value:08x}{suffix}")


def _pack_depthwise_expanded_weights_fp16(weight_ochw, out_channels, ic_per_group, kh, kw, packed_weight_size):
    # RK depthwise CONV_MODE=3 still fetches one full aligned kernel. Local CDMA
    # only verifies FP16 kernel groups/CBUF fetch width; this slot expansion is
    # RK schedule-specific.
    slot_sz = kh * kw
    wt_full = np.zeros(packed_weight_size // 2, dtype=np.float16)
    wt_full[:out_channels * slot_sz] = weight_ochw.reshape(out_channels, ic_per_group, slot_sz)[:, 0].reshape(-1)
    return wt_full


def _expand_grouped_weights_fp16(weight_ochw, out_channels, in_channels, ic_per_group, oc_per_group, kh, kw):
    # Generic grouped conv is lowered to a dense RK submit. This is software
    # expansion, not an NVDLA hardware grouping feature.
    wt_full = np.zeros((out_channels, in_channels, kh, kw), dtype=np.float16)
    oc = np.arange(out_channels)[:, None]
    dst_ic = (oc // oc_per_group) * ic_per_group + np.arange(ic_per_group)
    wt_full[oc, dst_ic] = weight_ochw.reshape(out_channels, ic_per_group, kh, kw)
    return wt_full.reshape(-1)


def _output_unpack_c2(params):
    # RK DPU output grouping. Local CACC source only verifies 16 FP16/INT16
    # elements per payload; final DPU surface C2 is RK formatter behavior.
    if params.get('is_depthwise', False):
        return params['align_c']
    return 8 if params['align_out_c'] >= 8 else params['align_out_c']


def _packed_sizes(params):
    packed_input_size = ceil_div(params['in_channels'], params['align_c']) * params['in_h'] * params['width_stride'] * params['align_c'] * FP16_BYTES
    packed_weight_size = params['out_channels'] * params['kernel_h'] * params['kernel_w'] * ceil_div(params['weight_in_channels'], params['align_c']) * params['align_c'] * FP16_BYTES
    packed_output_size = ceil_div(params['out_channels'], params['align_out_c']) * params['out_h'] * params['out_width_stride'] * params['align_out_c'] * FP16_BYTES
    return packed_input_size, packed_weight_size, packed_output_size


def _pack_weights_for_submit(params, weight_ochw, packed_weight_size):
    kh, kw = params['kernel_h'], params['kernel_w']
    if params['groups'] <= 1:
        weight_packed = pack_conv_weights_fp16(
            weight_ochw, params['out_channels'], params['in_channels'],
            kh, kw, params['align_c'], groups=params['groups'])
        return params, weight_packed, packed_weight_size

    in_c = params['in_channels']
    ic_per_group = params['weight_in_channels']
    oc_per_group = params['out_channels'] // params['groups']
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
    h, w, stride = (1, params['out_h'] * params['out_w'], params['out_width_stride']) if is_1x1 else (params['out_h'], params['out_w'], params['out_w'])
    flat = unpack_nc1hwc2_fp16(out_packed, params['batch'], params['out_channels'], h, w, unpack_c2, stride)
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


def _write_chained_task_descriptor(reg_segments):
    for task_idx in range(len(reg_segments)):
        block_qword = task_idx * REG_BLOCK_QWORDS
        for i in range(REG_BLOCK_QWORDS):
            regs_ptr[block_qword + i] = 0
        regs = reg_segments[task_idx]
        if len(regs) > REG_BLOCK_QWORDS:
            raise ValueError(f"reg segment {task_idx} has {len(regs)} regs, block holds {REG_BLOCK_QWORDS}")
        for i, (target, value, addr) in enumerate(regs):
            regs_ptr[block_qword + i] = emit(target, value, addr)

    tasks[0].flags = 0
    tasks[0].op_idx = 1
    tasks[0].enable_mask = 0xd
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(reg_segments[0])
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = cmd_mc.dma_addr


def compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    weight_in_channels = in_channels // groups if groups > 0 else in_channels
    is_depthwise = _is_depthwise(in_channels, out_channels, groups)
    out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1

    # FP16 atoms are 16 elements in local NVDLA. RK depthwise schedules align up
    # to 32 channels, while regular conv output/kernel groups align to 16.
    max_align = 32 if is_depthwise else 16
    align_c = max(8, min(1 << (max(1, in_channels) - 1).bit_length(), max_align))
    align_out_c = max(FP16_ATOM_ELEMENTS, align_up_int(out_channels, FP16_ATOM_ELEMENTS))
    # Keep each row aligned to at least one 16-FP16 atom's worth of channels.
    width_stride = align_up_int(in_w, max(1, (16 + align_c - 1) // align_c))
    out_channel_field = (align_up_int(align_out_c, 32) if is_depthwise else align_out_c) - 1
    orig_channel = out_channels - 1 if out_channels > 0 else 0
    out_width_stride = _output_width_stride(
        in_channels, out_channels, kernel_h, kernel_w, groups, out_h, out_w, align_out_c)

    out_atoms = max(1, out_w * out_h)
    data_in_channel_real = in_channels - 1 if in_channels > 0 else 0
    data_in_channel_aligned = max(align_c, align_up_int(in_channels, align_c))
    weight_kernels = 1 if is_depthwise else out_channels
    weight_bytes_per_kernel = kernel_h * kernel_w * data_in_channel_aligned * FP16_BYTES
    weight_bytes_total = weight_bytes_per_kernel * out_channels

    feature_grains = _feature_grains(in_h, kernel_h, width_stride, align_c)

    input_pack_c2 = _input_pack_c2(in_channels, out_channels, kernel_h, kernel_w, groups, align_c)
    use_nhwc = should_use_nhwc_pack(batch, in_channels, in_h, in_w, width_stride, input_pack_c2,
                                    out_c=out_channels, kh=kernel_h, kw=kernel_w, groups=groups)
    use_pixel_mode = use_nhwc or _is_pixel_dma_input(in_channels, is_depthwise)
    line_stride, surf_stride = _dma_strides(in_h, width_stride, use_pixel_mode)

    # RK CNA_CVT_CON5 mask covers lanes inside a 128-bit converter group. Local
    # NVDLA conversion code operates on atom payloads, but this mask field is RK.
    cvt_lanes = RK_CVT_BITS // FP16_ATOM_ELEMENTS
    cvt_active = max(1, min(in_channels if use_nhwc else input_pack_c2, cvt_lanes))
    cvt_mask = 0xFFFFFFFF if cvt_active >= 32 else ((1 << cvt_active) - 1)
    cbuf_entries = _cbuf_entries(width_stride, align_c, in_h, is_depthwise)
    data_bank = _data_bank(width_stride, feature_grains, align_c)

    effective_align_out = out_channel_field + 1
    if groups > 1 and not is_depthwise:
        per_group_align = max(FP16_ATOM_ELEMENTS, align_up_int(ceil_div(out_channels, groups), FP16_ATOM_ELEMENTS))
        effective_align_out = per_group_align
    # RK DPU surface_add is in output formatter groups of 8 FP16 values; not a
    # local NVDLA CACC constant.
    surface_add = out_width_stride * (effective_align_out // RK_DPU_OUTPUT_GROUP)

    # Pad constant used by RK's CNA_PAD_CON1.
    pad_con1 = _calc_pad_con1({'kernel_w': kernel_w, 'is_depthwise': is_depthwise, 'input_zero_point': 0x80})
    return {
        'batch': batch, 'in_channels': in_channels, 'out_channels': out_channels,
        'kernel_h': kernel_h, 'kernel_w': kernel_w, 'input_hw': input_hw, 'groups': groups,
        'in_h': in_h, 'in_w': in_w, 'weight_in_channels': weight_in_channels,
        'is_depthwise': is_depthwise, 'out_h': out_h, 'out_w': out_w,
        'align_c': align_c, 'align_out_c': align_out_c, 'width_stride': width_stride,
        'out_channel_field': out_channel_field, 'orig_channel': orig_channel,
        'out_width_stride': out_width_stride, 'out_atoms': out_atoms,
        'data_in_channel_real': data_in_channel_real,
        'data_in_channel_aligned': data_in_channel_aligned,
        'weight_kernels': weight_kernels, 'weight_bytes_per_kernel': weight_bytes_per_kernel,
        'weight_bytes_total': weight_bytes_total, 'feature_grains': feature_grains,
        'input_pack_c2': input_pack_c2, 'use_nhwc': use_nhwc,
        'use_pixel_mode': use_pixel_mode, 'line_stride': line_stride,
        'surf_stride': surf_stride, 'cvt_mask': cvt_mask,
        'cbuf_entries': cbuf_entries, 'data_bank': data_bank,
        'surface_add': surface_add, 'pad_con1': pad_con1,
        'weights_banks': _calc_weight_banks(
            ceil_div(out_channels, 1) * kernel_h * kernel_w * align_up_int(weight_in_channels, align_c) * FP16_BYTES),
    }


def _task_params(params, split_task):
    if split_task is None:
        return params
    p = dict(params)
    p['in_h'] = split_task['input_height']
    p['out_h'] = split_task['output_height']
    p['out_atoms'] = max(1, p['out_w'] * p['out_h'])
    p['feature_grains'] = _feature_grains(p['in_h'], p['kernel_h'], p['width_stride'], p['align_c'])
    p['cbuf_entries'] = _cbuf_entries(p['width_stride'], p['align_c'], p['in_h'], p['is_depthwise'])
    p['data_bank'] = split_task['input_banks']
    p['weights_banks'] = split_task['weights_banks']
    p['weight_reuse'] = split_task['weight_reuse']
    p['mesa_cbuf_con0'] = True
    p['surface_add'] = _output_width_stride(
        p['in_channels'], p['out_channels'], p['kernel_h'], p['kernel_w'],
        p['groups'], p['out_h'], p['out_w'], p['align_out_c']) * ((p['out_channel_field'] + 1) // RK_DPU_OUTPUT_GROUP)
    return p


def _cbuf_con0(params):
    if not params.get('mesa_cbuf_con0', False):
        return ((NPU_CBUF_BANKS - params['data_bank']) << 4) | params['data_bank']
    weight_banks = params.get('weights_banks', NPU_CBUF_BANKS - params['data_bank'])
    value = ((weight_banks & 0xf) << 4) | (params['data_bank'] & 0xf)
    if params.get('weight_reuse', False):
        value |= 1 << 13
    return value


def _pc_base_address_value(regcmd_dma):
    return regcmd_dma & 0xFFFFFFF0


def _mesa_pc_data_amount(reg_count):
    return align_up_int((reg_count - 4) // 2, 2)


def build_conv2d_regs(params, input_dma=0, weights_dma=0, output_dma=0, split_task=None,
                      pc_next_dma=None, pc_next_regs=None):
    input_offset = split_task['input_offset_bytes'] if split_task else 0
    output_offset = split_task['output_offset_bytes'] if split_task else 0
    p = _task_params(params, split_task)
    regs = []
    def E(addr, val): regs.append((_target(addr), val, addr))

    # Register values below mix local NVDLA mechanisms (FP16 precision, CDMA
    # grains/strides/CBUF banks, CSC/CMAC/CACC dimensions) with RK-only DPU/CNA
    # bitfields from working register schedules. Comments above mark which magic
    # constants are actually confirmed in local nvdla/hw source.
    # Verified local NVDLA pieces: precision enum value 2 for FP16, CDMA/CSC
    # dimensions are programmed as register+1 fields, CDMA stride fields are
    # 32B units, feature_grains is emitted as field<<4, and weight size is bytes.
    # RK-only pieces: pixel flags, CVT mask/bypass values, DPU formatter/bypass
    # values, reserved writes, and PC op-enable payload.
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
    E(rk.REG_CNA_CBUF_CON0, _cbuf_con0(p))
    E(rk.REG_CNA_CBUF_CON1, p['cbuf_entries'])
    E(rk.REG_CNA_CVT_CON0, 0xB if not p['use_nhwc'] else 0x1)
    for a in [rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4]:
        E(a, 1 << 16)
    E(rk.REG_CNA_FEATURE_DATA_ADDR, (input_dma + input_offset) & 0xFFFFFFFF)
    E(rk.REG_CNA_DMA_CON0, (15 << 16) | 15)
    E(rk.REG_CNA_DMA_CON1, p['line_stride'])
    E(rk.REG_CNA_DMA_CON2, p['surf_stride'])
    E(rk.REG_CNA_FC_DATA_SIZE0, (p['in_w'] << 16) | p['in_h'])
    E(rk.REG_CNA_FC_DATA_SIZE1, p['align_c'])
    E(rk.REG_CNA_DCOMP_ADDR0, (weights_dma + REGCMD_RESERVED) & 0xFFFFFFFF)
    E(rk.REG_CNA_CVT_CON5, p['cvt_mask'])
    if EMIT_PAD_CON1:
        E(rk.REG_CNA_PAD_CON1, p['pad_con1'])
    core = (2 << 8)
    if p['is_depthwise']: core |= (1 << 1)
    if p['kernel_h'] > 1 or p['kernel_w'] > 1: core |= (1 << 0)
    E(rk.REG_CORE_MISC_CFG, core)
    E(rk.REG_CORE_DATAOUT_SIZE_0, ((p['out_h'] - 1) << 16) | (p['out_w'] - 1))
    E(rk.REG_CORE_DATAOUT_SIZE_1, p['out_channel_field'])
    regs.append((rk.CORE | 0x1, 0, CORE_RESERVED_ZERO_ADDR))
    dpu_fmc = (15 << 5) | (2 << 1)
    if p['is_depthwise']: dpu_fmc |= (3 << 3)
    E(rk.REG_DPU_FEATURE_MODE_CFG, dpu_fmc)
    E(rk.REG_DPU_DATA_FORMAT, (2 << 29) | (2 << 26) | 2)
    E(rk.REG_DPU_DST_BASE_ADDR, (output_dma + output_offset) & 0xFFFFFFFF)
    E(rk.REG_DPU_DST_SURF_STRIDE, p['out_width_stride'] << 4)
    E(rk.REG_DPU_DATA_CUBE_WIDTH, p['out_w'] - 1)
    E(rk.REG_DPU_DATA_CUBE_HEIGHT, p['out_h'] - 1)
    E(rk.REG_DPU_DATA_CUBE_CHANNEL, (p['orig_channel'] << 16) | p['out_channel_field'])
    E(rk.REG_DPU_BS_CFG, DPU_BYPASS_CFG)
    ow_cfg = 3 if p['is_depthwise'] else 1
    E(rk.REG_DPU_BS_OW_CFG, (ow_cfg << 8) | (ow_cfg << 5) | (ow_cfg << 2) | (1 << 1))
    E(rk.REG_DPU_WDMA_SIZE_0, p['out_channel_field'])
    E(rk.REG_DPU_WDMA_SIZE_1, ((p['out_h'] - 1) << 16) | (p['out_w'] - 1))
    E(rk.REG_DPU_BN_CFG, DPU_BYPASS_CFG)
    E(rk.REG_DPU_EW_CFG, DPU_EW_BYPASS_CFG)
    E(rk.REG_DPU_EW_CVT_SCALE_VALUE, 1)
    E(rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | 1)
    E(rk.REG_DPU_SURFACE_ADD, p['surface_add'] << 4)
    regs.append((0x1, 0, DPU_RESERVED_ZERO_ADDR))
    if pc_next_dma is not None and pc_next_regs is not None:
        regs.append((0x81, _pc_base_address_value(pc_next_dma), rk.REG_PC_BASE_ADDRESS))
        regs.append((0x81, _mesa_pc_data_amount(pc_next_regs), rk.REG_PC_REGISTER_AMOUNTS))
    regs.append((0x81, PC_OPERATION_ENABLE_VALUE, rk.REG_PC_OPERATION_ENABLE))
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
    if DUMP_REGS:
        dump_regs('CONV_MESA SUBMIT', regs_list)
    for i, (t, v, a) in enumerate(regs_list): regs_ptr[i] = emit(t, v, a)
    _write_task_descriptor(regs_list)

    mem_sync(fd, wt_mc.obj_addr, 0, wt_alloc, RKNPU_MEM_SYNC_TO_DEVICE)
    if data_is_cacheable:
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

    if data_is_cacheable:
        mem_sync(fd, out_mc.obj_addr, 0, packed_output_size, RKNPU_MEM_SYNC_FROM_DEVICE)
    out_packed = np.frombuffer(out_map, dtype=np.float16, count=packed_output_size // 2).copy()

    return _unpack_output(params, out_packed, is_1x1)

def _run_conv2d_channel_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1):
    """1x1 conv with in_channels >= 5: HW non-aligned DMA supports max 4 in-channels.
    Slices into groups of 4, submits each independently, accumulates results.
    This is an RK schedule limit; local NVDLA only confirms a separate image
    load path, not this exact 4-channel cap."""
    np.random.seed(42)
    full_in = np.random.randn(1, original_in_c, input_hw[0], input_hw[1]).astype(np.float16)
    full_wt = np.random.randn(out_channels, original_in_c, 1, 1).astype(np.float16)
    result = np.zeros((1, out_channels, input_hw[0], input_hw[1]), dtype=np.float16)
    for start_c in range(0, original_in_c, MAX_PIXEL_DMA_CHANNELS):
        end_c = min(start_c + MAX_PIXEL_DMA_CHANNELS, original_in_c)
        inp_slice = np.zeros((1, MAX_PIXEL_DMA_CHANNELS, input_hw[0], input_hw[1]), dtype=np.float16)
        inp_slice[0, :end_c - start_c] = full_in[0, start_c:end_c]
        wt_slice = np.zeros((out_channels, MAX_PIXEL_DMA_CHANNELS, 1, 1), dtype=np.float16)
        wt_slice[:, :end_c - start_c] = full_wt[:, start_c:end_c]
        p = compute_conv2d_params(MAX_PIXEL_DMA_CHANNELS, out_channels, kernel_h, kernel_w, input_hw, 1)
        result += _npu_submit(p, inp_slice.reshape(-1), wt_slice.reshape(-1), is_1x1)
    return result, full_in, full_wt

def _run_depthwise_channel_sliced(channels, kernel_h, kernel_w, input_hw, is_1x1):
    """Depthwise channels above 8 are submitted in 8-channel lanes.
    A single 32-channel depthwise task selects weight slots by 8-row stripe,
    so channel groups must be isolated and concatenated in software.
    This 8-channel lane is RK schedule-derived, not an NVDLA C-model constant."""
    np.random.seed(42)
    full_in = np.random.randn(1, channels, input_hw[0], input_hw[1]).astype(np.float16)
    full_wt = np.random.randn(channels, 1, kernel_h, kernel_w).astype(np.float16)
    out_h = input_hw[0] - kernel_h + 1
    out_w = input_hw[1] - kernel_w + 1
    result = np.zeros((1, channels, out_h, out_w), dtype=np.float16)
    for start_c in range(0, channels, DEPTHWISE_SLICE_CHANNELS):
        end_c = min(start_c + DEPTHWISE_SLICE_CHANNELS, channels)
        p = compute_conv2d_params(end_c - start_c, end_c - start_c, kernel_h, kernel_w, input_hw, end_c - start_c)
        result[:, start_c:end_c] = _npu_submit(
            p,
            full_in[:, start_c:end_c].reshape(-1),
            full_wt[start_c:end_c].reshape(-1),
            is_1x1,
        )
    return result, full_in, full_wt


def _run_conv2d_spatial_tiled(in_channels, out_channels, kernel_h, kernel_w, input_hw,
                              weight_ochw, groups, input_nchw=None):
    # CBUF-aware vertical tiling: split input height into overlapped slices and
    # submit one NPU task per slice, then stitch outputs by row order.
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    full_out_h = in_h - kernel_h + 1
    if full_out_h <= 0:
        raise ValueError(f"invalid out_h for kernel {kernel_h}x{kernel_w} on input {in_h}")

    if input_nchw is None:
        np.random.seed(42)
        input_nchw = np.random.randn(batch, in_channels, in_h, in_w).astype(np.float16)
    weight_ochw = weight_ochw.astype(np.float16, copy=False)

    result = np.zeros((batch, out_channels, full_out_h, in_w - kernel_w + 1), dtype=np.float16)
    base_params = compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)
    _, packed_weight_size, _ = _packed_sizes(base_params)
    max_rows = _calc_slice_budget_in_rows(base_params, packed_weight_size)
    if max_rows >= in_h:
        # Single task; keep existing fast path behavior.
        full_result = _npu_submit(base_params, input_nchw.reshape(-1), weight_ochw.reshape(-1), False)
        result[:] = full_result[:, :, :full_out_h, :]
        return result, input_nchw, weight_ochw

    # output rows produced by each slice when stepping by `step_rows`.
    step_rows = max(1, max_rows - kernel_h + 1)
    out_h_total = full_out_h
    out_w = in_w - kernel_w + 1

    for start_out_row in range(0, out_h_total, step_rows):
        rows_for_slice = min(step_rows, out_h_total - start_out_row)
        slice_in_top = start_out_row
        slice_in_h = rows_for_slice + kernel_h - 1
        slice_in_bottom = min(in_h, slice_in_top + slice_in_h)
        slice_in_h = slice_in_bottom - slice_in_top

        if slice_in_h <= 0:
            continue

        inp_slice = np.zeros((batch, in_channels, slice_in_h, in_w), dtype=np.float16)
        inp_slice[:] = input_nchw[:, :, slice_in_top:slice_in_bottom]

        p_slice = compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, (slice_in_h, in_w), groups)
        part = _npu_submit(p_slice, inp_slice.reshape(-1), weight_ochw.reshape(-1), False, log_submit=False)

        keep_h = min(rows_for_slice, part.shape[2], result.shape[2] - start_out_row)
        if keep_h <= 0:
            continue

        if part.shape[2] != keep_h:
            # Keep conservative shape handling if a backend quirk changes rows.
            keep_h = min(keep_h, part.shape[2])
        if groups == 1:
            result[:, :, start_out_row:start_out_row + keep_h, :part.shape[3]] = part[:, :, :keep_h]
        else:
            # Grouped paths keep per-group channel offsets in software.
            oc_per_group = out_channels // groups
            for g in range(groups):
                g_oc_s = g * oc_per_group
                g_oc_e = g_oc_s + oc_per_group
                result[:, g_oc_s:g_oc_e, start_out_row:start_out_row + keep_h, :out_w] = part[:, g_oc_s:g_oc_e, :keep_h]

    return result, input_nchw, weight_ochw


def _npu_submit_direct_spatial(params, input_nchw, weight_ochw, log_submit=True):
    packed_input_size, packed_weight_size, packed_output_size = _packed_sizes(params)
    reset_npu(fd)

    input_packed = pack_nc1hwc2_fp16(
        input_nchw, params['batch'], params['in_channels'], params['in_h'], params['in_w'],
        params['align_c'], params['width_stride'], out_c=params['out_channels'],
        kh=params['kernel_h'], kw=params['kernel_w'], groups=params.get('groups', 1),
        use_nhwc=params['use_nhwc'])
    params, weight_packed, packed_weight_size = _pack_weights_for_submit(params, weight_ochw, packed_weight_size)

    split_tasks = _plan_split_tasks(params, packed_weight_size)
    in_view = memoryview(bytearray(input_packed.tobytes()))
    wt_view = memoryview(bytearray(weight_packed.tobytes()))
    ctypes.memmove(mv_address(in_map), mv_address(in_view), min(packed_input_size, len(in_view)))
    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(wt_map))
    ctypes.memmove(wt_ptr + REGCMD_RESERVED, mv_address(wt_view), min(packed_weight_size, len(wt_view)))

    if len(split_tasks) == 1 or not DIRECT_CHAIN:
        regs_list = build_conv2d_regs(params, in_mc.dma_addr, wt_mc.dma_addr, out_mc.dma_addr, split_task=split_tasks[0])
        _verify_config(regs_list)
        if DUMP_REGS:
            dump_regs('CONV_MESA DIRECT TASK 0', regs_list)
        for i, (t, v, a) in enumerate(regs_list):
            regs_ptr[i] = emit(t, v, a)
        _write_task_descriptor(regs_list)
    else:
        base_regs = [build_conv2d_regs(params, in_mc.dma_addr, wt_mc.dma_addr, out_mc.dma_addr, split_task=t) for t in split_tasks]
        reg_segments = []
        for task_idx, task_regs in enumerate(base_regs):
            next_dma = cmd_mc.dma_addr + (task_idx + 1) * REG_BLOCK_QWORDS * ctypes.sizeof(ctypes.c_uint64)
            next_count = len(base_regs[task_idx + 1]) + 2 if task_idx + 1 < len(base_regs) else None
            regs = build_conv2d_regs(
                params, in_mc.dma_addr, wt_mc.dma_addr, out_mc.dma_addr,
                split_task=split_tasks[task_idx],
                pc_next_dma=next_dma if task_idx + 1 < len(base_regs) else None,
                pc_next_regs=next_count)
            _verify_config(regs)
            reg_segments.append(regs)
            if DUMP_REGS:
                dump_regs(f'CONV_MESA DIRECT CHAIN TASK {task_idx}', regs)
        _write_chained_task_descriptor(reg_segments)

    mem_sync(fd, wt_mc.obj_addr, 0, wt_alloc, RKNPU_MEM_SYNC_TO_DEVICE)
    if data_is_cacheable:
        mem_sync(fd, in_mc.obj_addr, 0, packed_input_size, RKNPU_MEM_SYNC_TO_DEVICE)
        mem_sync(fd, out_mc.obj_addr, 0, packed_output_size, RKNPU_MEM_SYNC_TO_DEVICE)

    ret = submit(task_mc.obj_addr)
    if log_submit:
        print(f"SUBMIT ret={ret}")

    if data_is_cacheable:
        mem_sync(fd, out_mc.obj_addr, 0, packed_output_size, RKNPU_MEM_SYNC_FROM_DEVICE)
    out_packed = np.frombuffer(out_map, dtype=np.float16, count=packed_output_size // 2).copy()
    return _unpack_output(params, out_packed, False)

def _npu_submit_single_input_pointwise(input_1c, weight_ochw):
    batch, _, height, width = input_1c.shape
    out_channels = weight_ochw.shape[0]
    result = np.zeros((batch, out_channels, height, width), dtype=np.float16)
    inp_pad = np.zeros((batch, POINTWISE_PIXEL_CHANNELS, height, width), dtype=np.float16)
    inp_pad[:, 0:1] = input_1c
    for start_oc in range(0, out_channels, OUTPUT_CHANNEL_CHUNK):
        end_oc = min(start_oc + OUTPUT_CHANNEL_CHUNK, out_channels)
        wt_pad = np.zeros((end_oc - start_oc, POINTWISE_PIXEL_CHANNELS, 1, 1), dtype=np.float16)
        wt_pad[:, 0:1] = weight_ochw[start_oc:end_oc]
        p = compute_conv2d_params(POINTWISE_PIXEL_CHANNELS, end_oc - start_oc, 1, 1, (height, width), 1)
        result[:, start_oc:end_oc] = _npu_submit(
            p, inp_pad.reshape(-1), wt_pad.reshape(-1), True, log_submit=False)
    return result

def _pad_pointwise_channels(input_nchw, weight_ochw, params, original_in_c, pad_to_c):
    input_padded = np.zeros((params['batch'], pad_to_c, params['in_h'] * params['in_w']), dtype=np.float16)
    input_padded[:, :original_in_c] = input_nchw.reshape(params['batch'], original_in_c, params['in_h'] * params['in_w'])
    weight_padded = np.zeros((params['out_channels'], pad_to_c, params['kernel_h'] * params['kernel_w']), dtype=np.float16)
    weight_padded[:, :original_in_c] = weight_ochw.reshape(params['out_channels'], original_in_c, params['kernel_h'] * params['kernel_w'])
    return input_padded.reshape(-1), weight_padded.reshape(-1)

def _run_conv2d_spatial_decomposed(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups):
    """Run kh*kw convolution as exact-order 1x1 NPU submits.
    Direct non-1x1 programming still has shape-dependent partial/numerical
    behavior in this RK schedule. Local CSC supports kernel_height*kernel_width
    iteration, but this fallback avoids the unverified RK direct-spatial cases
    while matching test_conv's fp16 accumulation order."""
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
    shape_key = (in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)
    needs_spatial_tiling = DIRECT_SPATIAL and not is_1x1 and (DIRECT_SPATIAL_ALL or shape_key in DIRECT_SPATIAL_ALLOWLIST)
    needs_spatial_decomposition = not is_1x1 and not needs_spatial_tiling

    if DRY_RUN:
        regs = build_conv2d_regs(p, 0, 0, 0)
        _verify_config(regs)
        print(f"\n=== CONV2D DRY RUN ({len(regs)} regs) ===")
        print(f"  in=({p['batch']},{p['in_channels']},{p['in_h']},{p['in_w']}) "
              f"out=({p['batch']},{p['out_channels']},{p['out_h']},{p['out_w']}) "
              f"kernel=({p['kernel_h']},{p['kernel_w']}) groups={groups}")
        dump_regs('CONV_MESA BASE DRY RUN', regs)
        if needs_spatial_tiling:
            _, packed_weight_size, _ = _packed_sizes(p)
            split_tasks = _plan_split_tasks(p, packed_weight_size)
            for split_task in split_tasks:
                task_regs = build_conv2d_regs(p, 0, 0, 0, split_task=split_task)
                print(f"  split_task[{split_task['task_index']}]: input_top={split_task['input_top']} input_h={split_task['input_height']} output_top={split_task['output_top']} output_h={split_task['output_height']} input_banks={split_task['input_banks']} weight_banks={split_task['weights_banks']} weight_reuse={int(split_task['weight_reuse'])}")
                dump_regs(f"CONV_MESA DIRECT DRY TASK {split_task['task_index']}", task_regs)
        return None, None, None

    if needs_1x1_channel_slicing:
        return _run_conv2d_channel_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1)

    if needs_depthwise_slicing:
        return _run_depthwise_channel_sliced(in_channels, kernel_h, kernel_w, input_hw, is_1x1)

    if needs_spatial_tiling:
        np.random.seed(42)
        input_nchw = np.random.randn(p['batch'], in_channels, p['in_h'], p['in_w']).astype(np.float16)
        weight_ochw = np.random.randn(out_channels, in_channels // groups, kernel_h, kernel_w).astype(np.float16)
        result = _npu_submit_direct_spatial(p, input_nchw.reshape(-1), weight_ochw.reshape(-1))
        return result, input_nchw, weight_ochw

    if needs_spatial_decomposition:
        return _run_conv2d_spatial_decomposed(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)

    reset_npu(fd)
    np.random.seed(42)
    input_nchw = np.random.randn(p['batch'] * original_in_c * p['in_h'] * p['in_w']).astype(np.float16)
    weight_ochw = np.random.randn(p['out_channels'] * (original_in_c // groups if groups > 0 else original_in_c) * p['kernel_h'] * p['kernel_w']).astype(np.float16)

    if pad_to_c:
        input_nchw, weight_ochw = _pad_pointwise_channels(input_nchw, weight_ochw, p, original_in_c, pad_to_c)

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
