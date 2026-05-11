import os, mmap, sys, ctypes, numpy as np
from fcntl import ioctl

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
# check where is source, is the source wrong?
RKNPU_JOB_BLOCK = 0 << 1
RKNPU_JOB_NONBLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 1 << 2
FP16_BYTES = 2
FP32_BYTES = 4
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
MIN_CHANNEL_TILE = 32
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_LINE_STRIDE_GROUP_CAP = 13
# minimum pipeline depth needed to keep the CSC→CMAC→CACC path fed, <80 result is wrong
RK_MIN_WIDE_FEATURE_GRAINS = 80 
RK_KN_LINE_STRIDE_START = 512
PC_CHAIN_TAIL_QWORDS = 4  # enable_npu_units_and_set_next_pc_addr() returns up to 4 QWORDs
OUTPUT_FP16 = False  # set by --out fp16
GEMM_INPUT_BANKS = RK_CBUF_BANKS - 2  # reserve 2 banks for the current GEMM layout
GEMM_MAX_ALIGN_IN = RK_CBUF_BANKS * MIN_CHANNEL_TILE  # 12 groups of 32 channels

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
        ("iommu_domain_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("task_base_addr", ctypes.c_uint64),
        ("hw_elapse_time", ctypes.c_int64),
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

def npu_reset(fd):
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

def npu_submit(task_obj_addr, task_count=1, flags=0x1):
    npu_reset(fd)
    submit_struct = rknpu_submit(
        flags=flags,
        timeout=6000,
        task_start=0,
        task_number=task_count,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        iommu_domain_id=0,
        reserved=0,
        task_base_addr=0,
        hw_elapse_time=0,
        core_mask=1,
        fence_fd=-1,
    )
    # 3 cores
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)
    if ret < 0:
        print(f"npu_submit failed: ret={ret}")
    return ret

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

def _gemm_layout(m, n, k):
    aligned_k = max(MIN_CHANNEL_TILE, _align_up(k, MIN_CHANNEL_TILE))
    align_out = max(MIN_CHANNEL_TILE, _align_up(n, MIN_CHANNEL_TILE))
    align_in = max(aligned_k, align_out)
    eff_k = align_in if align_in != aligned_k else k
    return align_in, align_out, eff_k

def _gemm_output_indices(m, n, align_out, out_fp16):
    row_stride = align_out * 2 if out_fp16 else align_out
    row_start = np.arange(m, dtype=np.int64) * row_stride
    if out_fp16:
        col_idx = (np.arange(n, dtype=np.int64) // 16) * 32 + (np.arange(n, dtype=np.int64) % 16)
    else:
        col_idx = np.arange(n, dtype=np.int64)
    return row_start[:, None] + col_idx[None, :]

def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma, out_fp16=False):
    align_in, align_out, eff_k = _gemm_layout(m, n, k)
    input_row_bytes = align_in * FP16_BYTES
    out_precision = 2 if out_fp16 else 5
    size_e = 1 if out_fp16 else 3

    # Pre-loads two banks worth of rows, even becoz NVDLA CSC does paired-atomics
    even_rows_per_two_banks = (_ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1   
    feature_grains = max(RK_MIN_WIDE_FEATURE_GRAINS, even_rows_per_two_banks)

    # max 11/12 for data, 1/12 for weight
    data_banks = np.clip(_ceil_div(m * input_row_bytes, CBUF_BANK_SIZE), 1, RK_CBUF_BANKS-1)
    # why 4 times ?
    line_stride = 4 * min(_ceil_div(eff_k, MIN_CHANNEL_TILE), RK_LINE_STRIDE_GROUP_CAP) 
    # why 8 times ?
    notch_val = 8 * min(align_out // MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP) - 1
    
    npu_regs = [
        E(reg.DPU,  reg.S_POINTER,
                ((1 << 3) |                           # DPU_S_POINTER_POINTER_PP_MODE
                (1 << 2)  |                           # DPU_S_POINTER_EXECUTER_PP_EN
                (1 << 1))                             # DPU_S_POINTER_POINTER_PP_EN
        ),
        E(reg.CNA,  reg.CNA_CONV_CON1, 
                ((2 << 4) |                           # CNA_CONV_CON1_IN_PRECISION(2)=fp16
                (2 << 7)  |                           # CNA_CONV_CON1_PROC_PRECISION(2)=fp16
                (1 << 29) )                           # CNA_CONV_CON1_GROUP_LINE_OFF
        ),
        E(reg.CNA,  reg.CNA_CONV_CON2,
                ((feature_grains << 4))              # CNA_CONV_CON2_FEATURE_GRAINS
        ),
        E(reg.CNA,  reg.CNA_CONV_CON3,
                ((1 << 3) |                           # CNA_CONV_CON3_OPERATION_ENABLE
                1)                                    # CNA_CONV_CON3_WEIGHT_REUSE
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE0,
                ((1 << 16) |                          # CNA_DATA_SIZE0_WIDTH
                m)                                    # CNA_DATA_SIZE0_HEIGHT
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE1,
                (((align_in - 1) << 16) |             # CNA_DATA_SIZE1_CHANNEL
                align_in)                             # CNA_DATA_SIZE1_ATOMICS
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE2,   1),
        E(reg.CNA,  reg.CNA_DATA_SIZE3,   m),
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE0, input_row_bytes * align_out),
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE1, input_row_bytes),
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE2,
                ((1 << 24) |                          # CNA_WEIGHT_SIZE2_KERNEL_WIDTH
                (1 << 16) |                           # CNA_WEIGHT_SIZE2_KERNEL_HEIGHT
                align_out)                            # CNA_WEIGHT_SIZE2_KERNELS
        ),
        E(reg.CNA,  reg.CNA_CBUF_CON0,
                (((RK_CBUF_BANKS - data_banks) << 4) | # CNA_CBUF_CON0_WEIGHT_BANK
                data_banks)                            # CNA_CBUF_CON0_DATA_BANK
        ),
        E(reg.CNA,  reg.CNA_CBUF_CON1, _ceil_div(align_in, MIN_CHANNEL_TILE)),
        E(reg.CNA,  reg.CNA_CVT_CON0,
                ((1 << 3) |                           # CNA_CVT_CON0_TRUNCATE
                (1 << 1) |                            # CNA_CVT_CON0_SCALE
                1)                                    # CNA_CVT_CON0_ENABLE
        ),
        E(reg.CNA,  reg.CNA_CVT_CON1,
                (1 << 16)                             # CNA_CVT_CON1_SCALE
        ),
        E(reg.CNA,  reg.CNA_CVT_CON2,
                (1 << 16)                             # CNA_CVT_CON2_SCALE
        ),
        E(reg.CNA,  reg.CNA_CVT_CON3,
                (1 << 16)                             # CNA_CVT_CON3_SCALE
        ),
        E(reg.CNA,  reg.CNA_CVT_CON4,
                (1 << 16)                             # CNA_CVT_CON4_SCALE
        ),
        E(reg.CNA,  reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA,  reg.CNA_DMA_CON0,
                ((15 << 16) |                         # CNA_DMA_CON0_READ_BURST
                15)                                   # CNA_DMA_CON0_WEIGHT_BURST
        ),
        E(reg.CNA,  reg.CNA_DMA_CON1, line_stride),
        E(reg.CNA,  reg.CNA_DMA_CON2, 0),  # surf_stride = 0 , cannot skip
        E(reg.CNA,  reg.CNA_FC_DATA_SIZE0,
                ((1 << 16) |                          # CNA_FC_DATA_SIZE0_WIDTH
                m)                                    # CNA_FC_DATA_SIZE0_HEIGHT
        ),
        E(reg.CNA,  reg.CNA_FC_DATA_SIZE1, align_in),
        E(reg.CNA,  reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CORE, reg.CORE_MISC_CFG,
                ((2 << 8) |                           # CORE_MISC_CFG_PROC_PRECISION
                1)                                    # CORE_MISC_CFG_OPERATION_ENABLE
        ),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0,
                (((m - 1) << 16) |                    # CORE_DATAOUT_SIZE_0_HEIGHT
                0)                                    # CORE_DATAOUT_SIZE_0_WIDTH
        ),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, align_out - 1),
        E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU,  reg.FEATURE_MODE_CFG,
                ((15 << 5) |                          # DPU_FEATURE_MODE_CFG_BYPASS
                (2 << 1))                             # DPU_FEATURE_MODE_CFG_MODE
        ),
        E(reg.DPU,  reg.DATA_FORMAT,
                ((out_precision << 29) |              # DPU_DATA_FORMAT_OUT_PRECISION
                (2 << 26) |                           # DPU_DATA_FORMAT_PROC_PRECISION
                2)                                    # DPU_DATA_FORMAT_IN_PRECISION
        ),
        E(reg.DPU,  reg.DST_BASE_ADDR, out_dma),
        E(reg.DPU,  reg.DST_SURF_STRIDE,
                (1 << 4)                              # DPU_DST_SURF_STRIDE
        ),
        E(reg.DPU,  reg.DATA_CUBE_WIDTH, 0),
        E(reg.DPU,  reg.DATA_CUBE_HEIGHT, m - 1),
        E(reg.DPU,  reg.DATA_CUBE_NOTCH,
                ((notch_val << 16) |                  # DPU_DATA_CUBE_NOTCH_SURF
                notch_val)                            # DPU_DATA_CUBE_NOTCH_LINE
        ),
        E(reg.DPU,  reg.DATA_CUBE_CHANNEL,
                (((align_out - 1) << 16) |            # DPU_DATA_CUBE_CHANNEL_CUBE
                (align_out - 1))                      # DPU_DATA_CUBE_CHANNEL_ATOMICS
        ),
        E(reg.DPU,  reg.BS_CFG,
                ((1 << 6) |                           # DPU_BS_CFG_BS_RELU_BYPASS
                (1 << 4)  |                           # DPU_BS_CFG_BS_MUL_BYPASS
                (1 << 1)  |                           # DPU_BS_CFG_BS_ALU_BYPASS
                1)                                    # DPU_BS_CFG_BS_BYPASS
        ),
        E(reg.DPU,  reg.BS_OW_CFG,
                ((size_e << 8) |                      # DPU_BS_OW_CFG_SIZE_E_2
                (size_e << 5)  |                      # DPU_BS_OW_CFG_SIZE_E_1
                (size_e << 2)  |                      # DPU_BS_OW_CFG_SIZE_E_0
                (1 << 1))                             # DPU_BS_OW_CFG_OD_BYPASS
        ),
        E(reg.DPU,  reg.WDMA_SIZE_0, align_out - 1),
        E(reg.DPU,  reg.WDMA_SIZE_1,
                (((m - 1) << 16) |                    # DPU_WDMA_SIZE_1_HEIGHT
                0)                                    # DPU_WDMA_SIZE_1_WIDTH
        ),
        E(reg.DPU,  reg.BN_CFG,
                ((1 << 6) |                           # DPU_BN_CFG_BN_RELU_BYPASS
                (1 << 4)  |                           # DPU_BN_CFG_BN_MUL_BYPASS
                (1 << 1)  |                           # DPU_BN_CFG_BN_ALU_BYPASS
                1)                                    # DPU_BN_CFG_BN_BYPASS
        ),
        # emit with default bypass?
        E(reg.DPU,  reg.EW_CFG,
                ((1 << 9) |                           # DPU_EW_CFG_EW_RELU_BYPASS
                (1 << 8)  |                           # DPU_EW_CFG_EW_OP_CVT_BYPASS
                (1 << 7)  |                           # DPU_EW_CFG_EW_LUT_BYPASS
                (1 << 1)  |                           # DPU_EW_CFG_EW_OP_BYPASS
                1)                                    # DPU_EW_CFG_EW_BYPASS
        ),
        E(reg.DPU, reg.OUT_CVT_SCALE,
                ((1 << 16) | 1) if out_fp16 else 0     # DPU_OUT_CVT_SCALE (FP32TOFP16_EN)
        ),
        E(reg.DPU,  reg.SURFACE_ADD,
                ((1 * 4) << 4)                        # DPU_SURFACE_ADD
        ),
    ]
    return npu_regs

def write_regs_to_npu_task(task_regs):
    def enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len):
        enable_npu_units = E(reg.PC, reg.OPERATION_ENABLE,
                            (6 << 1) |                           # PC_OPERATION_ENABLE_RESERVED_0 = units to enable(DPU/CNA/PPU), info not in TRM
                            1)                                   # PC_OPERATION_ENABLE_OP_EN

        if next_offset is None:
            # PC chain tail: NOP + reg_amounts=0 + VERSION + enable (matching mtx512 format)
            return [
                E(0x0001, 0, 0),                                  # NOP separator
                E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),       # PC_REGISTER_AMOUNTS = 0 (end chain)
                E(reg.VERSION, 0, 0),                            # VERSION (OP_40 equivalent)
                enable_npu_units,
            ]
        next_addr = regcmd_mem_create.dma_addr + next_offset* ctypes.sizeof(ctypes.c_uint64)
        return [
            E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
            E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(next_task_regs_len, 2) + 1),
            E(reg.VERSION, 0, 0),
            enable_npu_units
        ]

    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + PC_CHAIN_TAIL_QWORDS, 2)
    assert offset <= regcmd_mem_create.size // ctypes.sizeof(ctypes.c_uint64), "regcmd buffer too small"

    # chain mutilple tasks in one npu_submit
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            npu_regcmd[base + i] = qword
        next_task_regs_len = len(task_regs[idx + 1]) if idx + 1 < len(task_regs) else 0
        next_offset = offsets[idx + 1]               if idx + 1 < len(task_regs) else None
        
        tails = enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len)
        for i, qword in enumerate(tails):
            npu_regcmd[base + len(regs) + i] = qword
        # masks done-signals, rknpu_fuzz_status()
        # https://github.com/allbilly/rknpu_driver/blob/0e23a914e5322d6b7cdaf3de6e91fdf76b2a9055/rknpu_job.c#L615
        # 0-1	0x03	CNA_FEATURE / CNA_WEIGHT
        # 2-3	0x0c	CNA_CSC / CORE
        # 4-5	0x30	(CNA variants?)
        # 6-7	0xc0	(CNA variants?)
        # 8-9	0x300	DPU_0 / DPU_1
        # 10-11	0xc00	PPU_0 / PPU_1
        # write flag for npu_tasks
        npu_tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)
        npu_tasks[idx].regcfg_amount = len(regs)
        npu_tasks[idx].op_idx = 0
        npu_tasks[idx].enable_mask = 0xd                   # PC | CNA | DPU enable
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9)      # PC_INTERRUPT_MASK_DPU_0 | DPU_1.
        npu_tasks[idx].int_clear = 0x1ffff                 # Downstream RKNPU_INT_CLEAR clears all driver status bits.
        
def run_gemm(m, n, k, a_matrix, b_matrix, out_fp16=False):
    align_in, align_out, _ = _gemm_layout(m, n, k)
    input_row_bytes  = align_in  * FP16_BYTES
    out_bytes = FP16_BYTES if out_fp16 else FP32_BYTES
    out_dtype = np.float16 if out_fp16 else np.float32
    row_stride_bytes = align_out * 4
    row_stride_elems = align_out * 2 if out_fp16 else align_out

    input_packed = np.zeros(align_in * m, dtype=np.float16)
    input_packed.reshape(m, align_in)[:, :k] = a_matrix[:, :k]
    input_packed = input_packed.view(np.uint16).tolist()
    
    # pack_weight_tile_16x32
    weight = np.zeros((align_out, align_in), dtype=np.float16)
    weight_packed = np.zeros(align_out * align_in, dtype=np.float16)
    weight[:n, :k] = b_matrix.T[:n, :k]
    weight_packed[:] = weight.reshape(align_out // 16, 16, align_in // 32, 32).transpose(0, 2, 1, 3).ravel()
    weight_packed = weight_packed.view(np.uint16).tolist()
    
    # write np array to C array
    ct_inputs = (ctypes.c_uint16 * len(input_packed)).from_buffer(input_map)
    ct_weights = (ctypes.c_uint16 * len(weight_packed)).from_buffer(weight_map)
    ct_inputs[:] = input_packed
    ct_weights[:] = weight_packed

    # Prepare task_regs with split M
    task_regs = []
    # Keep the current 10-bank input budget, but derive it from the 12-bank CBUF geometry.
    m_tile = GEMM_INPUT_BANKS * CBUF_BANK_SIZE // input_row_bytes if align_in <= GEMM_MAX_ALIGN_IN else 1
    for start in range(0, m, m_tile):
        tile_m = min(m_tile, m - start)
        tiled_input_dma = input_mem_create.dma_addr + start * input_row_bytes
        tiled_output_dma = output_mem_create.dma_addr + start * row_stride_bytes
        task_regs.append(make_gemm_regs(tile_m, n, k, tiled_input_dma, weight_mem_create.dma_addr, tiled_output_dma, out_fp16=out_fp16))
    assert len(task_regs) <= tasks_mem_create.size // ctypes.sizeof(struct_rknpu_task), "task buffer too small"

    write_regs_to_npu_task(task_regs)
    out_nbytes = max(256, m * row_stride_bytes)
    output = np.frombuffer(output_map, dtype=out_dtype, count=out_nbytes // out_bytes)
    if out_fp16:
        output[:] = np.nan
    submit_flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG
    if npu_submit(tasks_mem_create.obj_addr, task_count=len(task_regs), flags=submit_flags) < 0:
        raise RuntimeError("npu_submit failed")

    if out_fp16:
        expected = _gemm_output_indices(m, n, align_out, True)
        result = output[expected].copy().reshape(m, n)
        if not np.isfinite(result).all():
            raise RuntimeError("fp16 output incomplete after blocking submit")
        return result

    output = output.copy()
    row_start = np.arange(m) * row_stride_elems
    return output[row_start[:, None] + np.arange(n)]

if __name__ == "__main__":
    out_fp16 = "--out" in sys.argv and sys.argv[sys.argv.index("--out") + 1] == "fp16"
    test_cases = [
        (2, 2, 1,
        np.array([[1], [3]], dtype=np.float16),
        np.array([[5, 6]], dtype=np.float16)),
    ]
    np.random.seed(42)
    for size in range(2,520,2):
        m = n = k = size
        a = np.random.randn(m, k).astype(np.float16)
        b = np.random.randn(k, n).astype(np.float16)
        test_cases.append((m, n, k, a, b))

    for m, n, k, a, b in test_cases:
        print(f"\n{m}x{n}x{k} ({'fp16' if out_fp16 else 'fp32'}):")
        r = run_gemm(m, n, k, a, b, out_fp16=out_fp16)
        if r is None:
            continue
        expected = (a @ b).astype(np.float16) if out_fp16 else a @ b
        ok = np.allclose(r, expected, atol=0.1)
        md = np.max(np.abs(r.astype(np.float64) - expected.astype(np.float64)))
        print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")
        assert ok , f"test shape {m, n, k} failed"
    os.close(fd)
