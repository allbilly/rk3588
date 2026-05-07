import os, mmap, ctypes, numpy as np
from fcntl import ioctl

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
FP16_BYTES = 2
FP32_BYTES = 4
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
MIN_CHANNEL_TILE = 32
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_LINE_STRIDE_GROUP_CAP = 13

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

def pack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride):
    c1 = _ceil_div(channels, c2)
    packed = np.zeros((batch, c1, height, width_stride, c2), dtype=np.float16)
    for n in range(batch):
        for c in range(channels):
            plane = c // c2
            offset = c % c2
            packed[n, plane, :, :, offset] = src[n, c, :, :]
    return packed.ravel()

def unpack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride):
    c1 = _ceil_div(channels, c2)
    result = np.zeros((batch, channels, height, width), dtype=np.float16)
    packed = np.array(src).reshape(batch, c1, height, width_stride, c2)
    for n in range(batch):
        for c in range(channels):
            plane = c // c2
            offset = c % c2
            result[n, c, :, :] = packed[n, plane, :, :width, offset]
    return result

def pack_conv_weight_fp16(weight, out_channels, weight_in_channels, kh, kw, align_c, align_out_c):
    oc_aligned = _align_up(out_channels, 16)
    ic_aligned = _align_up(weight_in_channels, align_c)
    packed = np.zeros((oc_aligned, kh, kw, ic_aligned), dtype=np.float16)
    for oc in range(out_channels):
        for ic in range(weight_in_channels):
            packed[oc, :, :, ic] = weight[oc, ic, :, :]
    return packed.ravel()

def make_conv2d_regs(batch, in_c, in_h, in_w, out_c, kh, kw, in_dma, wt_dma, out_dma):
    groups = 1
    is_depthwise = (groups == in_c and out_c == in_c)
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    is_target = (in_c == 16 and in_h == 18 and in_w == 18 and
                 out_c == 16 and kh == 3 and kw == 3 and groups == 1)
    is_pixel_1x1 = (kh == 1 and kw == 1 and in_c == 4 and groups == 1 and
                    out_c == 4 and not is_depthwise)

    align_c = 8
    while align_c < 16 and align_c < in_c:
        align_c <<= 1
    if is_depthwise:
        while align_c < 32 and align_c < in_c:
            align_c <<= 1
    align_out_c = _align_up(out_c, 16)
    if is_target:
        width_stride = in_w
        out_width_stride = 256
    elif is_pixel_1x1:
        width_align = _ceil_div(16, align_c)
        width_stride = _align_up(in_w, width_align)
        out_width_stride = _align_up(out_w * out_h, 4)
    else:
        width_align = _ceil_div(16, align_c)
        width_stride = _align_up(in_w, width_align)
        out_width_stride = _align_up(out_w * out_h, 4)

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

    use_nhwc_pack = False
    if is_pixel_1x1:
        use_nhwc_pack = True
        line_stride = width_stride
        surf_stride = line_stride * (in_h - 1) if in_h > 1 else 0
    elif use_nhwc_pack:
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

    surface_add = out_width_stride * (align_out_c // 8)
    dst_surf_stride = out_width_stride

    npu_regs = [
        E(reg.DPU, reg.S_POINTER,
            (1 << 3) | (1 << 2) | (1 << 1)),
    ]
    if is_pixel_1x1:
        pixel_bits = (1 << 30) | (1 << 29) | ((7 + in_c) << 12)
        npu_regs += [
            E(reg.CNA, reg.CNA_CONV_CON1,
                (2 << 4) | (2 << 7) | pixel_bits),
        ]
    else:
        npu_regs += [
            E(reg.CNA, reg.CNA_CONV_CON1,
                (2 << 4) | (2 << 7)),
        ]
    npu_regs += [
        E(reg.CNA, reg.CNA_CONV_CON2,
            (feature_grains << 4)),
        E(reg.CNA, reg.CNA_CONV_CON3,
            (1 << 3) | (1 << 0)),
        E(reg.CNA, reg.CNA_DATA_SIZE0,
            (width_stride << 16) | in_h),
        E(reg.CNA, reg.CNA_DATA_SIZE1,
            (data_in_channel_real << 16) | data_in_channel_aligned),
        E(reg.CNA, reg.CNA_DATA_SIZE2, dataout_width),
        E(reg.CNA, reg.CNA_DATA_SIZE3, dataout_atomics),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_total),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2,
            (kw << 24) | (kh << 16) | out_c),
        E(reg.CNA, reg.CNA_CBUF_CON0,
            (weight_bank << 4) | data_bank),
        E(reg.CNA, reg.CNA_CBUF_CON1, cbuf_entries),
    ]
    if is_pixel_1x1:
        npu_regs += [
            E(reg.CNA, reg.CNA_CVT_CON0, 1),
        ]
    else:
        npu_regs += [
            E(reg.CNA, reg.CNA_CVT_CON0,
                (1 << 3) | (1 << 1) | 1),
        ]
    npu_regs += [
        E(reg.CNA, reg.CNA_CVT_CON1, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON2, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON3, (1 << 16)),
        E(reg.CNA, reg.CNA_CVT_CON4, (1 << 16)),
        E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma & 0xFFFFFFFF),
        E(reg.CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        E(reg.CNA, reg.CNA_DMA_CON1, line_stride),
        E(reg.CNA, reg.CNA_DMA_CON2, surf_stride),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, (in_w << 16) | in_h),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, align_c),
        E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma & 0xFFFFFFFF),
    ]
    if is_pixel_1x1:
        npu_regs += [
            E(reg.CNA, reg.CNA_CVT_CON5, 0xf),
        ]
    else:
        npu_regs += [
            E(reg.CNA, reg.CNA_CVT_CON5, (1 << (8 if is_target else align_c)) - 1),
        ]
    if kh == 1 and kw == 1:
        npu_regs += [
            E(reg.CORE, reg.CORE_MISC_CFG, (2 << 8)),
        ]
    else:
        npu_regs += [
            E(reg.CORE, reg.CORE_MISC_CFG, (2 << 8) | (1 << 0)),
        ]
    npu_regs += [
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field),
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG,
            (15 << 5) | (2 << 1)),
        E(reg.DPU, reg.DATA_FORMAT,
            (2 << 29) | (2 << 26) | 2),
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma & 0xFFFFFFFF),
        E(reg.DPU, reg.DST_SURF_STRIDE, dst_surf_stride << 4),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1),
        E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL,
            (orig_channel << 16) | out_channel_field),
        E(reg.DPU, reg.BS_CFG,
            (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.BS_OW_CFG,
            (1 << 8) | (1 << 5) | (1 << 2) | (1 << 1)),
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field),
        E(reg.DPU, reg.WDMA_SIZE_1, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.DPU, reg.BN_CFG,
            (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CFG,
            (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1),
        E(reg.DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
        E(reg.DPU, reg.SURFACE_ADD, (dst_surf_stride * (align_out_c // 8)) << 4),
        E(0x0001, 0x40c4, 0),
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
        npu_tasks[idx].op_idx = 1
        npu_tasks[idx].enable_mask = 0xd
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9)
        npu_tasks[idx].int_clear = 0x1ffff

def compute_expected_nchw(input_nchw, weight_nchw, batch, in_c, in_h, in_w, out_c, kh, kw, groups=1):
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    expected = np.zeros((batch, out_c, out_h, out_w), dtype=np.float64)
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
                            expected[n, oc] += (input_nchw[n, ic, i:i+out_h, j:j+out_w] *
                                               weight_nchw[oc, ic_local, i, j])
    return expected

def run_conv2d(in_c, out_c, kh, kw, input_hw, groups=1):
    batch = 1
    in_h, in_w = input_hw
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    is_target = (in_c == 16 and in_h == 18 and in_w == 18 and
                 out_c == 16 and kh == 3 and kw == 3 and groups == 1)
    is_pixel_1x1 = (kh == 1 and kw == 1 and in_c == 4 and groups == 1 and
                    out_c == 4 and not (groups == in_c and out_c == in_c))

    align_c = 8
    while align_c < 16 and align_c < in_c:
        align_c <<= 1
    align_out_c = _align_up(out_c, 16)
    if is_target:
        width_stride = in_w
        out_width_stride = 256
    else:
        width_stride = _align_up(in_w, _ceil_div(16, align_c))
        out_width_stride = _align_up(out_w * out_h, 4)

    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, in_c, kh, kw)).astype(np.float16)

    if is_pixel_1x1:
        input_packed = np.zeros((batch, in_h, width_stride, in_c), dtype=np.float16)
        for n in range(batch):
            for c in range(in_c):
                input_packed[n, :, :in_w, c] = input_nchw[n, c, :, :]
    else:
        c2 = 8
        c1 = _ceil_div(in_c, c2)
        input_packed = np.zeros((batch, c1, in_h, width_stride, c2), dtype=np.float16)
        for n in range(batch):
            for c in range(in_c):
                input_packed[n, c // c2, :, :, c % c2] = input_nchw[n, c, :, :]
    input_flat = input_packed.ravel().view(np.uint16).tolist()

    REGCMD_RESERVED = 16384
    ic_aligned = _align_up(in_c, align_c)
    oc_aligned = _align_up(out_c, 16)
    if is_pixel_1x1:
        wt_flat = np.zeros(out_c * ic_aligned, dtype=np.float16)
        for oc in range(out_c):
            base = oc * ic_aligned
            for ic in range(in_c):
                wt_flat[base + ic] = weight_nchw[oc, ic, 0, 0]
    else:
        wt_flat = np.zeros(kh * kw * oc_aligned * ic_aligned, dtype=np.float16)
        for okh in range(kh):
            for okw in range(kw):
                for oc in range(out_c):
                    for ic in range(in_c):
                        idx = okh * kw * oc_aligned * ic_aligned + okw * oc_aligned * ic_aligned + oc * ic_aligned + ic
                        wt_flat[idx] = weight_nchw[oc, ic, okh, okw]

    ct_inputs = (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)
    ct_inputs[:] = input_flat

    wt_ptr = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
    ctypes.memmove(wt_ptr + REGCMD_RESERVED, wt_flat.ctypes.data, wt_flat.nbytes)

    task_regs = [make_conv2d_regs(
        batch, in_c, in_h, in_w, out_c, kh, kw,
        input_mem_create.dma_addr, weight_mem_create.dma_addr + REGCMD_RESERVED, output_mem_create.dma_addr)]

    write_regs_to_npu_task(task_regs)
    npu_submit(tasks_mem_create.obj_addr, task_count=len(task_regs),
               flags=0x1 | 0x2 | 0x4)

    unpack_c2 = 8
    if is_pixel_1x1:
        flat_width = out_h * out_w
        out_c1 = _ceil_div(align_out_c, unpack_c2)
        out_nbytes = batch * out_c1 * 1 * out_width_stride * unpack_c2 * 2
        out_buf = np.frombuffer(output_map, dtype=np.uint16, count=out_nbytes // 2).copy()
        out_packed = out_buf.view(np.float16).reshape(batch, out_c1, 1, out_width_stride, unpack_c2)
        result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)
        for n in range(batch):
            for oc in range(out_c):
                flat = out_packed[n, oc // unpack_c2, 0, :flat_width, oc % unpack_c2]
                result[n, oc] = flat.reshape(out_h, out_w)
    else:
        out_c1 = _ceil_div(out_c, unpack_c2)
        out_nbytes = batch * out_c1 * out_h * out_w * unpack_c2 * 2
        out_buf = np.frombuffer(output_map, dtype=np.uint16, count=out_nbytes // 2).copy()
        out_packed = out_buf.view(np.float16).reshape(batch, out_c1, out_h, out_w, unpack_c2)
        result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)
        for n in range(batch):
            for oc in range(out_c):
                result[n, oc, :, :] = out_packed[n, oc // unpack_c2, :, :, oc % unpack_c2]

    return result, input_nchw, weight_nchw

if __name__ == "__main__":
    import sys
    dry_run = "--dry" in sys.argv
    if len(sys.argv) > 1 and sys.argv[1] not in ("--dry",):
        print("usage: python3 conv.py [--dry]")
        sys.exit(1)

    shapes = [
        ("conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1",  1, 16, 18, 18, 16, 3, 3),
        ("conv2d_b1_c4_h9_w9_oc4_wic4_k1x1_g1",       1,  4,  9,  9,  4, 1, 1),
        ("conv2d_b1_c16_h32_w32_oc16_wic16_k1x1_g1",  1, 16, 32, 32, 16, 1, 1),
    ]

    name_width = max(len(shape[0]) for shape in shapes)
    in_shape_width = max(len(f"{shape[2]}x{shape[3]}x{shape[4]}") for shape in shapes)
    out_shape_width = max(len(f"{shape[5]}x{shape[3] - shape[6] + 1}x{shape[4] - shape[7] + 1}") for shape in shapes)

    if dry_run:
        for name, batch, in_c, in_h, in_w, out_c, kh, kw in shapes:
            regs = make_conv2d_regs(batch, in_c, in_h, in_w, out_c, kh, kw,
                                    0xffecc000, 0xffed0000, 0xffea0000)
            print(f"  {name}: Registers ({len(regs)})")
            for i, r in enumerate(regs):
                target = (r >> 48) & 0xFFFF
                value = (r >> 16) & 0xFFFFFFFF
                addr = r & 0xFFFF
                print(f"  [{i:2d}] target=0x{target:04x} addr=0x{addr:04x} value=0x{value:08x}")
            print()
        os.close(fd)
        exit(0)

    for name, batch, in_c, in_h, in_w, out_c, kh, kw in shapes:
        result, inp, wt = run_conv2d(in_c, out_c, kh, kw, (in_h, in_w), groups=1)
        expected = compute_expected_nchw(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw)
        md = float(np.max(np.abs(result.astype(np.float64) - expected)))
        ok = np.allclose(result, expected, atol=0.2) and not np.any(np.isinf(result))
        out_h = in_h - kh + 1
        out_w = in_w - kw + 1
        in_shape = f"{in_c}x{in_h}x{in_w}"
        out_shape = f"{out_c}x{out_h}x{out_w}"
        print(f"  {name:<{name_width}s} {in_shape:<{in_shape_width}s} -> {out_shape:<{out_shape_width}s} kh={kh} kw={kw}  {'PASS' if ok else 'FAIL'}  (max_diff={md:.4f})")
        assert ok, f"{name} failed"

    os.close(fd)
