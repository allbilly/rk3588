from fcntl import ioctl
import os, mmap
import ctypes
import numpy as np

class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    TARGET_CNA  = 0x0201   # CNA (Convolution/Matrix unit)
    TARGET_CORE = 0x0801   # CORE (Matrix compute engine)
    TARGET_DPU  = 0x1001   # DPU (Elementwise/DPU unit)
    TARGET_RDMA = 0x2001   # RDMA (Read DMA for inputs/weights)
    TARGET_PC   = 0x0081   # PC (Program Control / operation enable)
    TARGET_PC_REG = 0x0101 # PC chain registers
    TARGET_VERSION = 0x0041

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

def submit(task_obj_addr, task_count=1, flags=0x1):
    submit_struct = rknpu_submit(
        flags=flags,
        timeout=6000,
        task_start=0,
        task_number=task_count,
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
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)

    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

def reset_npu(fd):
    ret = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))
    print(f"reset_npu ret={ret}")
    return ret

task_map, tasks_mem_create = mem_allocate(fd, size=64*1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=512*1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def pack_input_row_major(m, n, k, a_matrix, in_pack, align_in):
    in_pack.reshape(m, align_in)[:, :k] = a_matrix[:, :k]

def pack_weight_tile_16x32(m, n, k, b_matrix, wt_pack, align_in, align_out):
    wt = np.zeros((align_out, align_in), dtype=np.float16)
    wt[:n, :k] = b_matrix.T[:n, :k]
    wt_pack[:] = wt.reshape(align_out // 16, 16, align_in // 32, 32).transpose(0, 2, 1, 3).ravel()

def unpack_row_major(m, n, k, raw, align_out):
    row_start = np.arange(m) * align_out
    return raw[row_start[:, None] + np.arange(n)]

FP16_BYTES = 2
FP32_BYTES = 4
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
MIN_CHANNEL_TILE = 32
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_LINE_STRIDE_GROUP_CAP = 13
RK_MIN_WIDE_FEATURE_GRAINS = 80
RK_KN_LINE_STRIDE_START = 512

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

def _rk_feature_grains(m, align_in):
    input_row_bytes = align_in * FP16_BYTES
    two_bank_rows = _ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes)
    even_two_bank_rows = (two_bank_rows + 1) & ~1

    return max(RK_MIN_WIDE_FEATURE_GRAINS, even_two_bank_rows)

def _max_m_tile(k):
    align_in, _, _ = _gemm_layout(1, 1, k)
    # 448 is the first 32-aligned channel count above the 13-group line/notch cap.
    # WIP
    # if align_in >= 448: return 1

    # todo total 12 bank, why 10 for input and 2 for weight, which shd hv prioty
    input_cbuf_budget = 10 * CBUF_BANK_SIZE
    input_row_bytes = align_in * FP16_BYTES
    return input_cbuf_budget // input_row_bytes

def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma):
    align_in, align_out, eff_k = _gemm_layout(m, n, k)
    surf_stride = 0

    input_bytes = m * align_in * FP16_BYTES
    # 1 <= input_bytes//CBUF_BANK_SIZE <= RK_CBUF_BANKS-1 (leave 1 bank for weight)
    data_bank = max(1, min(RK_CBUF_BANKS-1, _ceil_div(input_bytes, CBUF_BANK_SIZE))) 
    line_stride = 4 if eff_k <= MIN_CHANNEL_TILE or eff_k >= RK_KN_LINE_STRIDE_START else min(RK_LINE_STRIDE_GROUP_CAP, _ceil_div(eff_k, MIN_CHANNEL_TILE)) * 4
    dst_surf_stride = 1
    notch_val = 8 * min(RK_LINE_STRIDE_GROUP_CAP, align_out // MIN_CHANNEL_TILE) - 1

    npu_regs = [
        E(reg.TARGET_DPU,  reg.S_POINTER,
                ((1 << 3) |                           # DPU_S_POINTER_POINTER_PP_MODE
                (1 << 2)  |                           # DPU_S_POINTER_EXECUTER_PP_EN
                (1 << 1))                             # DPU_S_POINTER_POINTER_PP_EN
        ),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON1, 
                ((2 << 4) |                           # CNA_CONV_CON1_IN_PRECISION(2)=fp16
                (2 << 7)  |                           # CNA_CONV_CON1_PROC_PRECISION(2)=fp16
                (1 << 29) )                           # CNA_CONV_CON1_GROUP_LINE_OFF
        ),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON2,
                (_rk_feature_grains(m, align_in) << 4) # CNA_CONV_CON2_FEATURE_GRAINS
        ),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON3,
                ((1 << 3) |                           # CNA_CONV_CON3_OPERATION_ENABLE
                1)                                    # CNA_CONV_CON3_WEIGHT_REUSE
        ),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE0,
                ((1 << 16) |                          # CNA_DATA_SIZE0_WIDTH
                m)                                    # CNA_DATA_SIZE0_HEIGHT
        ),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE1,
                (((align_in - 1) << 16) |             # CNA_DATA_SIZE1_CHANNEL
                align_in)                             # CNA_DATA_SIZE1_ATOMICS
        ),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE2,   1),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE3,   m),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE0, align_in * FP16_BYTES * align_out),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE1, align_in * FP16_BYTES),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE2,
                ((1 << 24) |                          # CNA_WEIGHT_SIZE2_KERNEL_WIDTH
                (1 << 16) |                           # CNA_WEIGHT_SIZE2_KERNEL_HEIGHT
                align_out)                            # CNA_WEIGHT_SIZE2_KERNELS
        ),
        E(reg.TARGET_CNA,  reg.CNA_CBUF_CON0,
                (((RK_CBUF_BANKS - data_bank) << 4) | # CNA_CBUF_CON0_WEIGHT_BANK
                data_bank)                            # CNA_CBUF_CON0_DATA_BANK
        ),
        E(reg.TARGET_CNA,  reg.CNA_CBUF_CON1, _ceil_div(align_in, MIN_CHANNEL_TILE)),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON0,
                ((1 << 3) |                           # CNA_CVT_CON0_TRUNCATE
                (1 << 1) |                            # CNA_CVT_CON0_SCALE
                1)                                    # CNA_CVT_CON0_ENABLE
        ),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON1,
                (1 << 16)                             # CNA_CVT_CON1_SCALE
        ),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON2,
                (1 << 16)                             # CNA_CVT_CON2_SCALE
        ),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON3,
                (1 << 16)                             # CNA_CVT_CON3_SCALE
        ),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON4,
                (1 << 16)                             # CNA_CVT_CON4_SCALE
        ),
        E(reg.TARGET_CNA,  reg.CNA_FEATURE_DATA_ADDR, in_dma & 0xFFFFFFFF),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON0,
                ((15 << 16) |                         # CNA_DMA_CON0_READ_BURST
                15)                                   # CNA_DMA_CON0_WEIGHT_BURST
        ),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON1,     line_stride),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON2,     surf_stride),
        E(reg.TARGET_CNA,  reg.CNA_FC_DATA_SIZE0,
                ((1 << 16) |                          # CNA_FC_DATA_SIZE0_WIDTH
                m)                                    # CNA_FC_DATA_SIZE0_HEIGHT
        ),
        E(reg.TARGET_CNA,  reg.CNA_FC_DATA_SIZE1, align_in),
        E(reg.TARGET_CNA,  reg.CNA_DCOMP_ADDR0,  wt_dma & 0xFFFFFFFF),
        E(reg.TARGET_CORE, reg.CORE_MISC_CFG,
                ((2 << 8) |                           # CORE_MISC_CFG_PROC_PRECISION
                1)                                    # CORE_MISC_CFG_OPERATION_ENABLE
        ),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_0,
                (((m - 1) << 16) |                    # CORE_DATAOUT_SIZE_0_HEIGHT
                0)                                    # CORE_DATAOUT_SIZE_0_WIDTH
        ),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_1, align_out - 1),
        E(reg.TARGET_CORE, reg.CORE_RESERVED_3030,  0),
        E(reg.TARGET_DPU,  reg.FEATURE_MODE_CFG,
                ((15 << 5) |                          # DPU_FEATURE_MODE_CFG_BYPASS
                (2 << 1))                             # DPU_FEATURE_MODE_CFG_MODE
        ),
        E(reg.TARGET_DPU,  reg.DATA_FORMAT,
                ((5 << 29) |                          # DPU_DATA_FORMAT_OUT_PRECISION
                (2 << 26) |                           # DPU_DATA_FORMAT_PROC_PRECISION
                2)                                    # DPU_DATA_FORMAT_IN_PRECISION
        ),
        E(reg.TARGET_DPU,  reg.DST_BASE_ADDR,     out_dma & 0xFFFFFFFF),
        E(reg.TARGET_DPU,  reg.DST_SURF_STRIDE,
                (dst_surf_stride << 4)                # DPU_DST_SURF_STRIDE
        ),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_WIDTH,   0),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_HEIGHT,  m - 1),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_NOTCH,
                ((notch_val << 16) |                  # DPU_DATA_CUBE_NOTCH_SURF
                notch_val)                            # DPU_DATA_CUBE_NOTCH_LINE
        ),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_CHANNEL,
                (((align_out - 1) << 16) |            # DPU_DATA_CUBE_CHANNEL_CUBE
                (align_out - 1))                      # DPU_DATA_CUBE_CHANNEL_ATOMICS
        ),
        E(reg.TARGET_DPU,  reg.BS_CFG,
                ((1 << 6) |                           # DPU_BS_CFG_BS_RELU_BYPASS
                (1 << 4)  |                           # DPU_BS_CFG_BS_MUL_BYPASS
                (1 << 1)  |                           # DPU_BS_CFG_BS_ALU_BYPASS
                1)                                    # DPU_BS_CFG_BS_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.BS_OW_CFG,
                ((3 << 8) |                           # DPU_BS_OW_CFG_SIZE_E_2
                (3 << 5)  |                           # DPU_BS_OW_CFG_SIZE_E_1
                (3 << 2)  |                           # DPU_BS_OW_CFG_SIZE_E_0
                (1 << 1))                             # DPU_BS_OW_CFG_OD_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_0,       align_out - 1),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_1,
                (((m - 1) << 16) |                    # DPU_WDMA_SIZE_1_HEIGHT
                0)                                    # DPU_WDMA_SIZE_1_WIDTH
        ),
        E(reg.TARGET_DPU,  reg.BN_CFG,
                ((1 << 6) |                           # DPU_BN_CFG_BN_RELU_BYPASS
                (1 << 4)  |                           # DPU_BN_CFG_BN_MUL_BYPASS
                (1 << 1)  |                           # DPU_BN_CFG_BN_ALU_BYPASS
                1)                                    # DPU_BN_CFG_BN_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.EW_CFG,
                ((1 << 9) |                           # DPU_EW_CFG_EW_RELU_BYPASS
                (1 << 8)  |                           # DPU_EW_CFG_EW_OP_CVT_BYPASS
                (1 << 7)  |                           # DPU_EW_CFG_EW_LUT_BYPASS
                (1 << 1)  |                           # DPU_EW_CFG_EW_OP_BYPASS
                1)                                    # DPU_EW_CFG_EW_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.SURFACE_ADD,
                ((dst_surf_stride * 4) << 4)          # DPU_SURFACE_ADD
        ),
    ]
    return npu_regs

def _pc_tail(next_offset_qwords, next_body_reg_count):
    def _pc_next_amount(body_reg_count):
        return _ceil_div(body_reg_count, 2) + 1

    if next_offset_qwords is None:
        return [
            0,
            E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            E(reg.TARGET_VERSION, 0, 0),
            E(reg.TARGET_PC, reg.OPERATION_ENABLE,
                    ((6 << 1) |                       # PC_OPERATION_ENABLE_RESERVED_0 captured GEMM payload
                    1)                                # PC_OPERATION_ENABLE_OP_EN
            ),
        ]
    next_addr = regcmd_mem_create.dma_addr + next_offset_qwords * ctypes.sizeof(ctypes.c_uint64)
    return [
        E(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
        E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, _pc_next_amount(next_body_reg_count)),
        E(reg.TARGET_VERSION, 0, 0),
        E(reg.TARGET_PC, reg.OPERATION_ENABLE,
                ((6 << 1) |                           # PC_OPERATION_ENABLE_RESERVED_0 captured GEMM payload
                1)                                    # PC_OPERATION_ENABLE_OP_EN
        ),
    ]

def _write_chained_tasks(task_regs):
    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + 4, 2)

    assert offset <= regcmd_mem_create.size // ctypes.sizeof(ctypes.c_uint64), "regcmd buffer too small"
    assert len(task_regs) <= tasks_mem_create.size // ctypes.sizeof(struct_rknpu_task), "task buffer too small"

    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        next_offset = offsets[idx + 1] if idx + 1 < len(task_regs) else None
        next_len = len(task_regs[idx + 1]) if idx + 1 < len(task_regs) else 0
        for i, qword in enumerate(_pc_tail(next_offset, next_len)):
            regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 4
        tasks[idx].enable_mask = 0x18                  # Downstream raw GEMM task descriptor mask.
        tasks[idx].int_mask = (1 << 8) | (1 << 9)      # PC_INTERRUPT_MASK_DPU_0 | DPU_1.
        tasks[idx].int_clear = 0x1ffff                 # Downstream RKNPU_INT_CLEAR clears all driver status bits.
        tasks[idx].regcfg_amount = len(regs)
        tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)

def run_gemm(m, n, k, a_matrix, b_matrix):
    align_in, align_out, _ = _gemm_layout(m, n, k)
    max_m = _max_m_tile(k)

    in_pack = np.zeros(align_in * m, dtype=np.float16)
    wt_pack = np.zeros(align_in * align_out, dtype=np.float16)

    pack_input_row_major(m, n, k, a_matrix, in_pack, align_in)
    pack_weight_tile_16x32(m, n, k, b_matrix, wt_pack, align_in, align_out)

    ct_inputs = (ctypes.c_uint16 * len(in_pack)).from_buffer(input_map)
    ct_weights = (ctypes.c_uint16 * len(wt_pack)).from_buffer(weight_map)
    ct_inputs[:] = in_pack.view(np.uint16).tolist()
    ct_weights[:] = wt_pack.view(np.uint16).tolist()

    task_regs = []
    for start in range(0, m, max_m):
        tile_m = min(max_m, m - start)
        in_dma = input_mem_create.dma_addr + start * align_in * FP16_BYTES
        out_dma = output_mem_create.dma_addr + start * align_out * FP32_BYTES
        regs = make_gemm_regs(tile_m, n, k, in_dma, weight_mem_create.dma_addr, out_dma)
        task_regs.append(regs)

    _write_chained_tasks(task_regs)
    reset_npu(fd)
    submit(tasks_mem_create.obj_addr, task_count=len(task_regs),
           flags=((1 << 0) |                          # RKNPU_JOB_PC
                  (0 << 1) |                          # RKNPU_JOB_BLOCK (absence of NONBLOCK)
                  (1 << 2)))                          # RKNPU_JOB_PINGPONG

    out_nbytes = max(256, ((m - 1) * align_out + n) * FP32_BYTES)
    raw = np.frombuffer(output_map, dtype=np.float32, count=out_nbytes // 4).copy()
    return unpack_row_major(m, n, k, raw, align_out)

if __name__ == "__main__":
    test_cases = [
        (2, 2, 1,
        np.array([[1], [3]], dtype=np.float16),
        np.array([[5, 6]], dtype=np.float16)),
    ]
    np.random.seed(42)
    for size in range(400,520,2):
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
        assert ok , f"test shape {m, n, k} failed"
    os.close(fd)
