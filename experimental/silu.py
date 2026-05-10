#!/usr/bin/env python3
from fcntl import ioctl
import os, mmap, sys, math
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
    BN_ALU_CFG          = 0x4064   # DPU batch norm ALU operand
    BN_MUL_CFG          = 0x4068   # DPU batch norm MUL operand
    EW_CFG              = 0x4070   # DPU elementwise config
    EW_CVT_SCALE_VALUE  = 0x4078   # DPU elementwise conversion scale
    OUT_CVT_SCALE       = 0x4084   # DPU output conversion scale
    SURFACE_ADD         = 0x40c0   # DPU surface add
    LUT_ACCESS_CFG      = 0x4100   # DPU LUT access config
    LUT_ACCESS_DATA     = 0x4104   # DPU LUT access data
    LUT_CFG             = 0x4108   # DPU LUT config
    LUT_INFO            = 0x410c   # DPU LUT table info
    LUT_LE_START        = 0x4110   # DPU LUT linear-exp start
    LUT_LO_END          = 0x411c   # DPU LUT linear-only end
    LUT_LO_SLOPE_SCALE  = 0x4128   # DPU LUT overflow slope scale
    LUT_LO_SLOPE_SHIFT  = 0x412c   # DPU LUT overflow slope shift

    # --- DPU RDMA (0x5000) ---
    RDMA_S_POINTER        = 0x5004   # RDMA S pointer config (pp/exec)
    RDMA_DATA_CUBE_WIDTH  = 0x500c   # RDMA data cube width
    RDMA_DATA_CUBE_HEIGHT = 0x5010   # RDMA data cube height
    RDMA_DATA_CUBE_CHANNEL= 0x5014   # RDMA data cube channel
    RDMA_ERDMA_CFG        = 0x5034   # RDMA ERDMA config
    RDMA_SRC_BASE_ADDR    = 0x5018   # RDMA source base address (input)
    RDMA_EW_BASE_ADDR     = 0x5038   # RDMA EW base address (weight)
    RDMA_FEATURE_MODE_CFG = 0x5044   # RDMA feature mode config
    RDMA_WEIGHT           = 0x5068   # RDMA LUT weight config

LUT_SIZE = 513
RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 0 << 1
RKNPU_JOB_PINGPONG = 1 << 2
FP16_BYTES = 2
PC_CHAIN_TAIL_QWORDS = 4
SILU_TILE_ELEMENTS = 8000 * 8

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

def submit(task_obj_addr, task_count=1):
    reset_npu(fd)
    submit_struct = rknpu_submit(
        flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
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

def submit_direct(task_obj_addr):
    reset_npu(fd)
    submit_struct = rknpu_submit(
        flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
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

task_map, tasks_mem_create = mem_allocate(fd, size=64*1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=2*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)

npu_tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
npu_regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))


def E(target, addr, value):
    return (target << 48) | ((int(value) & 0xffffffff) << 16) | (addr & 0xffff)

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

def write_f16(buf, vals):
    arr = np.asarray(vals, dtype=np.float16).view(np.uint16)
    (ctypes.c_uint16 * len(arr)).from_buffer(buf)[:] = arr.tolist()

def read_f16(buf, n):
    return np.frombuffer(buf, dtype=np.float16, count=n).copy()

def make_silu_lut():
    # Mirrors experimental/rknnops.h's alu_case_silu LUT construction.
    lut = [0] * (LUT_SIZE * 2)
    index_scale = 2824.0
    step = 32.0 / index_scale
    output_scale = 5664.8

    for i in range(LUT_SIZE):
        x = (LUT_SIZE - 1 - i) * step
        y = -x / (1.0 + math.exp(x))
        lut[i] = int(np.clip(round(y * output_scale), -32768, 32767))
    for i in range(LUT_SIZE):
        x = i * step
        y = x / (1.0 + math.exp(-x))
        lut[LUT_SIZE + i] = int(np.clip(round(y * output_scale), -32768, 32767))

    return lut, 0x6984, output_scale

def fill_lut_regs(lut):
    npu_regs = []
    for table_id, base in ((0, 0), (1, LUT_SIZE)):
        npu_regs.append(E(reg.TARGET_DPU, reg.LUT_ACCESS_CFG,
                ((2 << 16) |                          # DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE
                (table_id << 16))                     # DPU_LUT_ACCESS_CFG_LUT_TABLE_ID
        ))
        for i in range(LUT_SIZE):
            v = int(lut[base + i])
            data = v & 0xffff
            if v < 0:
                data |= 0xffff0000
            npu_regs.append(E(reg.TARGET_DPU, reg.LUT_ACCESS_DATA, data))
    return npu_regs

def silu_regs(n, input_dma, output_dma, include_lut=True, include_pc_enable=False, chain_rearm=False):
    lut, bn_mul_operand, output_scale = make_silu_lut()
    width = _ceil_div(n, 8) - 1
    if chain_rearm:
        s_pointer = ((1 << 3) | (1 << 2) |  (1 << 1)) # Re-arm both DPU ping-pong pointers for PC-chain segments.
    else:
        s_pointer = ((1 << 5) | (1 << 4))              # Clear executor/pointer PP for the standalone LUT+first tile submit.
    npu_regs = [
        *(fill_lut_regs(lut) if include_lut else ()),
        E(reg.TARGET_DPU,  reg.S_POINTER, s_pointer),
        E(reg.TARGET_RDMA, reg.RDMA_S_POINTER, s_pointer),
        E(reg.TARGET_DPU,  reg.FEATURE_MODE_CFG,
                ((15 << 5) |                          # DPU_FEATURE_MODE_CFG_BURST_LEN
                (2 << 1)  |                           # DPU_FEATURE_MODE_CFG_OUTPUT_MODE
                1)                                    # DPU_FEATURE_MODE_CFG_FLYING_MODE
        ),
        E(reg.TARGET_DPU,  reg.DATA_FORMAT,
                ((2 << 29) |                          # DPU_DATA_FORMAT_OUT_PRECISION
                (2 << 26) |                           # DPU_DATA_FORMAT_IN_PRECISION
                2)                                    # DPU_DATA_FORMAT_PROC_PRECISION
        ),
        E(reg.TARGET_DPU,  reg.DST_BASE_ADDR, output_dma & 0xffffffff),
        E(reg.TARGET_DPU,  reg.DST_SURF_STRIDE,
                (16 << 4)                             # DPU_DST_SURF_STRIDE
        ),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_WIDTH, width),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_CHANNEL,
                ((7 << 16) |                          # DPU_DATA_CUBE_CHANNEL_CUBE
                7)                                    # DPU_DATA_CUBE_CHANNEL_ATOMICS
        ),
        E(reg.TARGET_DPU,  reg.BS_CFG,
                ((1 << 6) |                           # DPU_BS_CFG_BS_RELU_BYPASS
                (1 << 4)  |                           # DPU_BS_CFG_BS_MUL_BYPASS
                (1 << 1)  |                           # DPU_BS_CFG_BS_ALU_BYPASS
                1)                                    # DPU_BS_CFG_BS_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.BS_OW_CFG,
                (1 << 1)                              # DPU_BS_OW_CFG_OD_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_0,
                7                                      # DPU_WDMA_SIZE_0_CHANNEL_WDMA
        ),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_1, width),
        E(reg.TARGET_DPU,  reg.BN_CFG,
                ((2 << 16) |                          # DPU_BN_CFG_BN_ALU_ALGO
                (1 << 6))                             # DPU_BN_CFG_BN_RELU_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.BN_ALU_CFG, 0x80000000),
        E(reg.TARGET_DPU,  reg.BN_MUL_CFG,
                (bn_mul_operand << 16)                # DPU_BN_MUL_CFG_BN_MUL_OPERAND
        ),
        E(reg.TARGET_DPU,  reg.EW_CFG,
                ((1 << 9) |                           # DPU_EW_CFG_EW_RELU_BYPASS
                (1 << 8)  |                           # DPU_EW_CFG_EW_OP_CVT_BYPASS
                (1 << 1))                             # DPU_EW_CFG_EW_OP_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.EW_CVT_SCALE_VALUE,
                1                                      # DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE
        ),
        E(reg.TARGET_DPU,  reg.OUT_CVT_SCALE,
                ((1 << 16) |                          # DPU_OUT_CVT_SCALE_FP32TOFP16_EN
                1)                                    # DPU_OUT_CVT_SCALE_OUT_CVT_SCALE
        ),
        E(reg.TARGET_DPU,  reg.SURFACE_ADD,
                (32 << 4)                             # DPU_SURFACE_ADD_SURF_ADD
        ),
        E(reg.TARGET_DPU,  0x40c4, 0),
        E(reg.TARGET_DPU,  reg.LUT_CFG,
                ((1 << 6) |                           # DPU_LUT_CFG_LUT_HYBRID_PRIORITY
                (1 << 5)  |                           # DPU_LUT_CFG_LUT_OFLOW_PRIORITY
                (2 << 2))                             # DPU_LUT_CFG_LUT_LO_LE_MUX
        ),
        E(reg.TARGET_DPU,  reg.LUT_INFO,
                ((5 << 16) |                          # DPU_LUT_INFO_LUT_LO_INDEX_SELECT
                (5 << 8))                             # DPU_LUT_INFO_LUT_LE_INDEX_SELECT
        ),
        E(reg.TARGET_DPU,  reg.LUT_LE_START, 0xffffc000),
        E(reg.TARGET_DPU,  reg.LUT_LO_END, 0x00004000),
        E(reg.TARGET_DPU,  reg.LUT_LO_SLOPE_SCALE,
                (16434 << 16)                         # DPU_LUT_LO_SLOPE_SCALE_LUT_SLOPE_OFLOW_SCALE
        ),
        E(reg.TARGET_DPU,  reg.LUT_LO_SLOPE_SHIFT,
                (13 << 5)                             # DPU_LUT_LO_SLOPE_SHIFT_LUT_SLOPE_OFLOW_SHIFT
        ),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, width),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL,
                7                                      # DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL
        ),
        E(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma & 0xffffffff),
        E(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG,
                1                                      # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE
        ),
        E(reg.TARGET_RDMA, reg.RDMA_FEATURE_MODE_CFG,
                ((2 << 15) |                          # DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION
                (15 << 11) |                          # DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN
                (2 << 5)  |                           # DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION
                (1 << 3)  |                           # DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN
                1)                                    # DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE
        ),
        E(reg.TARGET_RDMA, reg.RDMA_WEIGHT,
                ((1 << 24) |                          # DPU_RDMA_RDMA_WEIGHT_E_WEIGHT
                (1 << 16) |                           # DPU_RDMA_RDMA_WEIGHT_N_WEIGHT
                (1 << 8)  |                           # DPU_RDMA_RDMA_WEIGHT_B_WEIGHT
                1)                                    # DPU_RDMA_RDMA_WEIGHT_M_WEIGHT
        ),
    ] + ([E(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x00000018)] if include_pc_enable else [])
    return npu_regs, output_scale

def write_single_regs_to_npu_task(regs):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())
    for i, qword in enumerate(regs):
        npu_regcmd[i] = qword

    npu_tasks[0].flags = 0
    npu_tasks[0].op_idx = 4
    npu_tasks[0].enable_mask = 0x18
    npu_tasks[0].int_mask = 0x300
    npu_tasks[0].int_clear = 0x1ffff
    npu_tasks[0].int_status = 0
    npu_tasks[0].regcfg_amount = len(regs)
    npu_tasks[0].regcfg_offset = 0
    npu_tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

def write_regs_to_npu_task(task_regs):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())

    def make_tail(next_offset, next_body_len):
        enable = E(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18)
        if next_offset is None:
            return [
                0,                                     # terminal separator
                E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0), # PC_REGISTER_AMOUNTS = 0 (end chain)
                E(reg.TARGET_VERSION, 0, 0),          # VERSION (OP_40 equivalent)
                enable,
            ]
        next_addr = regcmd_mem_create.dma_addr + next_offset * ctypes.sizeof(ctypes.c_uint64)
        return [
            E(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xfffffff0),
            E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, next_body_len),
            E(reg.TARGET_VERSION, 0, 0),
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

        npu_tasks[idx].flags = 0
        npu_tasks[idx].op_idx = 4
        npu_tasks[idx].enable_mask = 0x18
        npu_tasks[idx].int_mask = 0x300
        npu_tasks[idx].int_clear = 0x1ffff
        npu_tasks[idx].int_status = 0
        npu_tasks[idx].regcfg_amount = len(regs) + PC_CHAIN_TAIL_QWORDS
        npu_tasks[idx].regcfg_offset = 0
        npu_tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)

def run_silu(a_vals):
    n = len(a_vals)
    output_scale = None
    write_f16(input_map, a_vals)
    output_map[: n * 2] = b"\x00" * (n * 2)

    first_tile = min(SILU_TILE_ELEMENTS, n)
    regs, output_scale = silu_regs(first_tile,
                                   input_mem_create.dma_addr,
                                   output_mem_create.dma_addr,
                                   include_lut=True,
                                   include_pc_enable=True)
    write_single_regs_to_npu_task(regs)
    ret = submit_direct(tasks_mem_create.obj_addr)
    print(f"SUBMIT ret={ret}")

    task_regs = []
    for start in range(first_tile, n, SILU_TILE_ELEMENTS):
        tile_n = min(SILU_TILE_ELEMENTS, n - start)
        regs, output_scale = silu_regs(tile_n,
                                       input_mem_create.dma_addr + start * FP16_BYTES,
                                       output_mem_create.dma_addr + start * FP16_BYTES,
                                       include_lut=False,
                                       include_pc_enable=False,
                                       chain_rearm=True)
        task_regs.append(regs)

    if task_regs:
        write_regs_to_npu_task(task_regs)
        ret = submit(tasks_mem_create.obj_addr, task_count=len(task_regs))
        print(f"SUBMIT ret={ret}")

    raw = read_f16(output_map, n)
    return raw, (raw.astype(np.float32) / output_scale).astype(np.float16)


def check_silu(n):
    a = np.linspace(-3.0, 3.0, n, dtype=np.float16)
    expected = (a.astype(np.float32) / (1.0 + np.exp(-a.astype(np.float32)))).astype(np.float16)

    raw, r_arr = run_silu(a)
    match = np.allclose(r_arr, expected, atol=0.01)
    print(f"shape=({n},) tasks={_ceil_div(n, SILU_TILE_ELEMENTS)}")
    print(f"input={a[:min(8, n)]}")
    print(f"NPU raw={raw[:min(8, n)]}")
    print(f"NPU decoded={r_arr[:min(8, n)]}")
    print(f"expected={expected[:min(8, n)]}")
    print(f"SILU {n} {'PASS' if match else 'FAIL'}")
    return match

if __name__ == "__main__":
    ok = True
    for n in (8, 128, 129, 1024, 8192, 65536, 130000):
        ok = check_silu(n) and ok

    os.close(fd)
    sys.exit(0 if ok else 1)
