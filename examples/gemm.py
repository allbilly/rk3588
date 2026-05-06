from fcntl import ioctl
import os, mmap, sys
import ctypes
import numpy as np

class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    TARGET_CNA  = 0x0201   # CNA (Convolution/Matrix unit)
    TARGET_CORE = 0x0801   # CORE (Matrix compute engine)
    TARGET_DPU  = 0x1001   # DPU (Elementwise/DPU unit)
    TARGET_RDMA = 0x2001   # RDMA (Read DMA for inputs/weights)
    TARGET_PC   = 0x0081   # PC (Program Control / operation enable)

    # --- PC (0x0000) ---
    OPERATION_ENABLE    = 0x0008   # PC operation enable

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

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1

fd = os.open(f"/dev/dri/card1", os.O_RDWR)

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

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)
DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))

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
    mem_create = rknpu_mem_create(
        flags=flags, #0x10 | 0x2,  # KERNEL_MAPPING | NON_CACHEABLE
        size=size
    )
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    print(f"ret={ret}, handle={mem_create.handle}, obj_addr={mem_create.obj_addr:#x}, dma_addr={mem_create.dma_addr:#x}")

    # Map memory to access from userspace
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    print(f"Memory mapped at offset={mem_map.offset:#x}")
    return buf, mem_create

def submit(task_obj_addr):
    submit_struct = rknpu_submit(
        flags=0x1,  # PC | BLOCK (NOT NONBLOCK: ioctl would return before NPU finishes)
        timeout=6000,
        task_start=0,
        task_number=1,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=1,
        fence_fd=-1,
    )
    # struct len is 5 but only 3 NPU core
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)

    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

def reset_npu(fd):
    action = rknpu_action(flags=RKNPU_ACT_RESET, value=0)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, action)
    print(f"reset_npu ret={ret}")
    return ret

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def write_regs(npu_regs, clear_count):
    for i in range(clear_count):
        regcmd[i] = 0
    for i in range(len(npu_regs)):
        regcmd[i] = npu_regs[i]

def setup_task(op_idx, enable_mask, reg_count):
    tasks[0].flags  = 0;
    tasks[0].op_idx = op_idx;
    tasks[0].enable_mask = enable_mask;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = reg_count
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

def pack_input_row_major(m, n, k, a_matrix, in_pack, align_in):
    in_pack.reshape(m, align_in)[:, :k] = a_matrix[:, :k]

def pack_input_c2_8(m, n, k, a_matrix, in_pack, align_in):
    in_pack[:] = a_matrix[:, :k].reshape(m, -1, 8).transpose(1, 0, 2).ravel()

def pack_weight_tile_16x32(m, n, k, b_matrix, wt_pack, align_in, align_out):
    wt = np.zeros((align_out, align_in), dtype=np.float16)
    wt[:n, :k] = b_matrix.T[:n, :k]
    wt_pack[:] = wt.reshape(align_out // 16, 16, align_in // 32, 32).transpose(0, 2, 1, 3).ravel()

def decode_output_linear(m, n, k, raw, align_out):
    row_start = np.arange(m) * align_out
    return raw[row_start[:, None] + np.arange(n)]

def decode_output_c2_4(m, n, k, raw, align_out):
    return raw[:n // 4 * m * 4].reshape(n // 4, m, 4).transpose(1, 0, 2).reshape(m, n).copy()

FP16_BYTES = 2
FP32_BYTES = 4
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
MIN_CHANNEL_TILE = 32
C2_INPUT = 8
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_MAX_DATA_BANKS = RK_CBUF_BANKS - 1
RK_LINE_STRIDE_GROUP_CAP = 13
RK_MIN_WIDE_FEATURE_GRAINS = 80
RK_VERY_WIDE_K = 7872
RK_KN_LINE_STRIDE_START = 512

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

def _uses_c2_input(m, n, k):
    square = m == n == k
    cbuf_aligned = k % 64 == 0
    cbuf_friendly = 64 <= k <= 256
    return square and cbuf_aligned and cbuf_friendly

def _gemm_layout(m, n, k):
    aligned_k = max(MIN_CHANNEL_TILE, _align_up(k, MIN_CHANNEL_TILE))
    align_out = max(MIN_CHANNEL_TILE, _align_up(n, MIN_CHANNEL_TILE))
    align_in = max(aligned_k, align_out)
    pad_k = align_in != aligned_k
    eff_k = align_in if pad_k else k
    return align_in, align_out, eff_k, pad_k

def _rk_data_bank_count(input_bytes):
    return max(1, min(RK_MAX_DATA_BANKS, _ceil_div(input_bytes, CBUF_BANK_SIZE)))

def _rk_cbuf_data_entries(align_in):
    return _ceil_div(align_in, MIN_CHANNEL_TILE)

def _rk_line_stride(m, n, k, eff_k):
    if eff_k <= MIN_CHANNEL_TILE or eff_k >= RK_KN_LINE_STRIDE_START or _uses_c2_input(m, n, k):
        return 4
    return min(RK_LINE_STRIDE_GROUP_CAP, _ceil_div(eff_k, MIN_CHANNEL_TILE)) * 4

def _rk_no_group_line_off(m, n, k):
    if _uses_c2_input(m, n, k):
        return True
    is_kn_64 = k == 64 and n == 64
    is_kn_256 = k == 256 and n == 256
    is_kn_512 = k == 512 and n == 512
    is_kn_lg_512 = k > 512 and n > 512
    is_matmul_768 = m == 1 and k == 768 and n == 768
    is_matmul_768_2048 = m == 1 and k == 768 and n == 2048
    is_matmul_2048 = m == 1 and k == 2048 and n == 2048
    return (
        is_kn_64
        or is_kn_256
        or is_kn_512
        or is_kn_lg_512
        or is_matmul_768
        or is_matmul_768_2048
        or is_matmul_2048
    )

def _rk_feature_grains(m, align_in, eff_k):
    if eff_k > RK_VERY_WIDE_K:
        return 2
    if m <= RK_MIN_WIDE_FEATURE_GRAINS:
        return m + 1
    denom = align_in * FP16_BYTES
    grains = _ceil_div(2 * CBUF_BANK_SIZE, denom)
    grains = (grains + 1) & ~1
    return max(RK_MIN_WIDE_FEATURE_GRAINS, grains)

def _rk_notch_value(m, k, n, align_out, eff_k):
    notch_blocks = min(RK_LINE_STRIDE_GROUP_CAP, align_out // MIN_CHANNEL_TILE)
    notch_val = C2_INPUT * notch_blocks - 1
    if _rk_no_group_line_off(m, n, k) or eff_k > RK_VERY_WIDE_K:
        return 0
    return notch_val

def get_input_packer(m, n, k, align_in):
    if _uses_c2_input(m, n, k): return pack_input_c2_8
    return pack_input_row_major

def get_weight_packer(m, n, k, align_in):
    return pack_weight_tile_16x32

def get_output_decoder(m, n, k, align_out):
    return decode_output_c2_4 if _uses_c2_input(m, n, k) else decode_output_linear

def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma):
    align_in, align_out, eff_k, _ = _gemm_layout(m, n, k)
    no_group_line_off = _rk_no_group_line_off(m, n, k)
    line_stride = _rk_line_stride(m, n, k, eff_k)

    surf_groups = m // 4
    surf_stride = 0
    if align_in >= 64 and _uses_c2_input(m, n, k):
        surf_stride = line_stride * (surf_groups - 1) + int(surf_groups == 0)

    input_bytes = m * align_in * FP16_BYTES
    data_bank = _rk_data_bank_count(input_bytes)
    data_entries = _rk_cbuf_data_entries(align_in)

    dst_surf_stride = align_out if no_group_line_off else 1
    feature_grains = _rk_feature_grains(m, align_in, eff_k)
    notch_val = _rk_notch_value(m, k, n, align_out, eff_k)

    def E(target, reg_addr, value):
        return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

    conv_con1 = (2 << 7) | (2 << 4)
    if not no_group_line_off:
        conv_con1 |= 1 << 29

    npu_regs = [
        E(reg.TARGET_DPU,  reg.S_POINTER,       (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON1,    conv_con1),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON2,    feature_grains << 4),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON3,    (1 << 3) | 1),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE0,   (1 << 16) | m),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE1,   ((align_in - 1) << 16) | align_in),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE2,   1),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE3,   m),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE0, align_in * FP16_BYTES * align_out),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE1, align_in * FP16_BYTES),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | align_out),
        E(reg.TARGET_CNA,  reg.CNA_CBUF_CON0,    ((RK_CBUF_BANKS - data_bank) << 4) | data_bank),
        E(reg.TARGET_CNA,  reg.CNA_CBUF_CON1,    data_entries),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON0,     (1 << 3) | (1 << 1) | 1),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON1,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON2,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON3,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON4,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_FEATURE_DATA_ADDR, in_dma & 0xFFFFFFFF),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON0,     (15 << 16) | 15),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON1,     line_stride),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON2,     surf_stride),
        E(reg.TARGET_CNA,  reg.CNA_FC_DATA_SIZE0, (1 << 16) | m),
        E(reg.TARGET_CNA,  reg.CNA_FC_DATA_SIZE1, align_in),
        E(reg.TARGET_CNA,  reg.CNA_DCOMP_ADDR0,  wt_dma & 0xFFFFFFFF),
        E(reg.TARGET_CORE, reg.CORE_MISC_CFG,       (2 << 8) | 1),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_0, ((m - 1) << 16) | 0),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_1, align_out - 1),
        E(reg.TARGET_CORE, reg.CORE_RESERVED_3030,  0),
        E(reg.TARGET_DPU,  reg.FEATURE_MODE_CFG,  (15 << 5) | (2 << 1)),
        E(reg.TARGET_DPU,  reg.DATA_FORMAT,       (5 << 29) | (2 << 26) | 2),
        E(reg.TARGET_DPU,  reg.DST_BASE_ADDR,     out_dma & 0xFFFFFFFF),
        E(reg.TARGET_DPU,  reg.DST_SURF_STRIDE,   dst_surf_stride << 4),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_WIDTH,   0),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_HEIGHT,  m - 1),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_NOTCH,   (notch_val << 16) | notch_val),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_CHANNEL, ((align_out - 1) << 16) | (align_out - 1)),
        E(reg.TARGET_DPU,  reg.BS_CFG,            0x00000053),
        E(reg.TARGET_DPU,  reg.BS_OW_CFG,         0x0000036e),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_0,       align_out - 1),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_1,       ((m - 1) << 16) | 0),
        E(reg.TARGET_DPU,  reg.BN_CFG,            0x00000053),
        E(reg.TARGET_DPU,  reg.EW_CFG,            0x00000383),
        E(reg.TARGET_DPU,  reg.SURFACE_ADD,       (dst_surf_stride * 4) << 4),
        E(reg.TARGET_PC,   reg.OPERATION_ENABLE,  0x0000000d),
    ]
    return npu_regs

def run_gemm(m, n, k, a_matrix, b_matrix):
    align_in, align_out, _, pad_k = _gemm_layout(m, n, k)

    in_pack = np.zeros(align_in * m, dtype=np.float16)
    wt_pack = np.zeros(align_in * align_out, dtype=np.float16)

    pack_input = get_input_packer(m, n, k, align_in)
    pack_weight = get_weight_packer(m, n, k, align_in)

    if pad_k:
        pack_input = pack_input_row_major

    pack_input(m, n, k, a_matrix, in_pack, align_in)
    pack_weight(m, n, k, b_matrix, wt_pack, align_in, align_out)

    ct_inputs = (ctypes.c_uint16 * len(in_pack)).from_buffer(input_map)
    ct_weights = (ctypes.c_uint16 * len(wt_pack)).from_buffer(weight_map)
    ct_inputs[:] = in_pack.view(np.uint16).tolist()
    ct_weights[:] = wt_pack.view(np.uint16).tolist()

    out_nbytes = max(256, ((m - 1) * align_out + n) * FP32_BYTES)
    npu_regs = make_gemm_regs(m, n, k,
        input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)

    if "--dry" in sys.argv:
        print(f"\n=== GEMM {m}x{n}x{k} DRY RUN ===")
        target_names = {0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU", 0x2001: "RDMA", 0x0081: "PC"}
        for i, v in enumerate(npu_regs):
            tgt = (v >> 48) & 0xFFFF
            ra = v & 0xFFFF
            val = (v >> 16) & 0xFFFFFFFF
            print(f"  [{i:3d}] {target_names.get(tgt, f'0x{tgt:04x}')}[0x{ra:04x}] = 0x{val:08x}")
        return None

    write_regs(npu_regs, 64)
    setup_task(4, 0x18, len(npu_regs))

    reset_npu(fd)
    submit(tasks_mem_create.obj_addr)

    raw = np.frombuffer(output_map, dtype=np.float32, count=out_nbytes // 4).copy()

    decode_output = get_output_decoder(m, n, k, align_out)
    out = decode_output(m, n, k, raw, align_out)
    return out

if __name__ == "__main__":
    test_cases = [
        (2, 2, 1,
        np.array([[1], [3]], dtype=np.float16),
        np.array([[5, 6]], dtype=np.float16)),
    ]
    np.random.seed(42)
    square_sizes = (8, 9) + tuple(1 << exp for exp in (6, 8))
    for size in square_sizes:
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

    os.close(fd)
