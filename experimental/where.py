# known issue, still need 2 seperate submit, werid NPU state corrupte in same submit for large shape
# where math explannation in experimental/math.csv 
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
    BS_ALU_CFG          = 0x4044   # DPU batch ALU operand
    BS_MUL_CFG          = 0x4048   # DPU batch MUL operand
    BS_OW_CFG           = 0x4050   # DPU batch OW config
    WDMA_SIZE_0         = 0x4058   # DPU write DMA size 0
    WDMA_SIZE_1         = 0x405c   # DPU write DMA size 1
    BN_CFG              = 0x4060   # DPU batch norm config
    BN_MUL_CFG          = 0x4068   # DPU batch norm MUL operand
    BN_RELUX_CMP_VALUE  = 0x406c   # DPU batch norm relux compare value
    EW_CFG              = 0x4070   # DPU elementwise config
    OUT_CVT_SCALE       = 0x4084   # DPU output conversion scale
    OUT_CVT_SHIFT       = 0x4088   # DPU output conversion shift
    SURFACE_ADD         = 0x40c0   # DPU surface add

    # --- DPU RDMA (0x5000) ---
    RDMA_S_POINTER        = 0x5004   # RDMA S pointer config (pp/exec)
    RDMA_DATA_CUBE_WIDTH  = 0x500c   # RDMA data cube width
    RDMA_DATA_CUBE_HEIGHT = 0x5010   # RDMA data cube height
    RDMA_DATA_CUBE_CHANNEL= 0x5014   # RDMA data cube channel
    RDMA_ERDMA_CFG        = 0x5034   # RDMA ERDMA config
    RDMA_SRC_BASE_ADDR    = 0x5018   # RDMA source base address (input)
    RDMA_EW_BASE_ADDR     = 0x5038   # RDMA EW base address (weight)
    RDMA_FEATURE_MODE_CFG = 0x5044   # RDMA feature mode config

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 0 << 1
RKNPU_JOB_PINGPONG = 1 << 2
FP16_BYTES = 2
PC_CHAIN_TAIL_QWORDS = 4
WHERE_TILE_ELEMENTS = 8
PCCHAIN_TILE_ELEMENTS = 8000 * 8

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
    if task_count == 1:
        submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
        submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    else:
        submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
        submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)

    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

def reset_npu(fd):
    action = rknpu_action(flags=RKNPU_ACT_RESET, value=0)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, action)
    print(f"reset_npu ret={ret}")
    return ret

# EW_CFG values for each op
# Base config: data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1
_EW_BASE = (
    (1 << 28) |                                      # DPU_EW_CFG_EW_DATA_MODE
    (2 << 22) |                                      # DPU_EW_CFG_EW_DATA_SIZE
    (1 << 9)  |                                      # DPU_EW_CFG_EW_OP_SRC
    (1 << 7)  |                                      # DPU_EW_CFG_EW_RELU_BYPASS
    (1 << 6)                                         # DPU_EW_CFG_EW_LUT_BYPASS
)
EW_CFG_ADD  = _EW_BASE | (2 << 16)
EW_CFG_MUL  = _EW_BASE | (1 << 2) | (1 << 8)
EW_CFG_SUB  = _EW_BASE | (4 << 16)
EW_CFG_CMP  = _EW_BASE | 1

task_map, tasks_mem_create = mem_allocate(fd, size=64*1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=512*1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch0_map, scratch0_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch1_map, scratch1_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch2_map, scratch2_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch3_map, scratch3_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch4_map, scratch4_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)

npu_tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
npu_regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def E(target, reg_addr, value):
    return (target << 48) | ((int(value) & 0xffffffff) << 16) | reg_addr

def low32(value):
    return int(value) & ((1 << 32) - 1)

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align


def write_f16(buf, vals):
    arr = np.array(vals, dtype=np.float16).view(np.uint16)
    ct_vals = (ctypes.c_uint16 * len(arr)).from_buffer(buf)
    ct_vals[:] = arr.tolist()

def read_f16(buf, n):
    return np.frombuffer(buf, dtype=np.float16, count=n).copy()

def ew_regs(ew_cfg_val, n, input_mem, weight_mem, output_mem, pre_regs=(), chain_rearm=False):
    dataout_width = (n + 7) // 8 - 1
    # Re-arm both DPU ping-pong pointers for elementwise stages.
    s_pointer = (
        (1 << 3) |                                  # DPU_S_POINTER_POINTER_PP_MODE
        (1 << 2) |                                  # DPU_S_POINTER_EXECUTER_PP_EN
        (1 << 1)                                    # DPU_S_POINTER_POINTER_PP_EN
    )

    npu_regs = [
        *([E(reg.TARGET_DPU,  reg.S_POINTER, s_pointer)] if chain_rearm else []),
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
        E(reg.TARGET_DPU,  reg.DATA_CUBE_WIDTH, dataout_width),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_HEIGHT,
                0                                      # DPU_DATA_CUBE_HEIGHT_HEIGHT
        ),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_NOTCH,
                0                                      # DPU_DATA_CUBE_NOTCH
        ),
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
        E(reg.TARGET_DPU,  reg.BN_CFG,
                ((1 << 6) |                           # DPU_BN_CFG_BN_RELU_BYPASS
                (1 << 4)  |                           # DPU_BN_CFG_BN_MUL_BYPASS
                (1 << 1)  |                           # DPU_BN_CFG_BN_ALU_BYPASS
                1)                                    # DPU_BN_CFG_BN_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.BS_ALU_CFG,
                0                                      # DPU_BS_ALU_CFG_BS_ALU_OPERAND
        ),
        E(reg.TARGET_DPU,  reg.BS_MUL_CFG,
                0                                      # DPU_BS_MUL_CFG_BS_MUL_OPERAND
        ),
        E(reg.TARGET_DPU,  reg.BS_OW_CFG,
                (1 << 1)                              # DPU_BS_OW_CFG_OD_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_0,
                7                                      # DPU_WDMA_SIZE_0_CHANNEL_WDMA
        ),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_1,
                dataout_width                           # DPU_WDMA_SIZE_1_WIDTH_WDMA
        ),
        E(reg.TARGET_DPU,  reg.BN_MUL_CFG,
                0                                      # DPU_BN_MUL_CFG_BN_MUL_OPERAND
        ),
        E(reg.TARGET_DPU,  reg.BN_RELUX_CMP_VALUE,
                0                                      # DPU_BN_RELUX_CMP_VALUE
        ),
        *pre_regs,
        E(reg.TARGET_DPU,  reg.EW_CFG,
                ew_cfg_val                              # DPU_EW_CFG
        ),
        E(reg.TARGET_DPU,  reg.OUT_CVT_SCALE,
                ((1 << 16) |                          # DPU_OUT_CVT_SCALE_FP32TOFP16_EN
                1)                                    # DPU_OUT_CVT_SCALE_OUT_CVT_SCALE
        ),
        E(reg.TARGET_DPU,  reg.OUT_CVT_SHIFT,
                0                                      # DPU_OUT_CVT_SHIFT
        ),
        E(reg.TARGET_DPU,  reg.SURFACE_ADD,
                ((1 * 4) << 4)                        # DPU_SURFACE_ADD_SURF_ADD
        ),
        *([E(reg.TARGET_RDMA, reg.RDMA_S_POINTER, s_pointer)] if chain_rearm else []),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH,
                dataout_width                           # DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH
        ),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT,
                0                                      # DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT
        ),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL,
                7                                      # DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL
        ),
        E(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG,
                ((1 << 30) |                          # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE
                (2 << 2))                             # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE
        ),
        E(reg.TARGET_DPU,  reg.DST_BASE_ADDR,
                low32(output_mem.dma_addr)             # DPU_DST_BASE_ADDR_LOW
        ),
        E(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR,
                low32(input_mem.dma_addr)              # DPU_RDMA_RDMA_SRC_BASE_ADDR_LOW
        ),
        E(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR,
                low32(weight_mem.dma_addr)             # DPU_RDMA_RDMA_EW_BASE_ADDR_LOW
        ),
        E(reg.TARGET_RDMA, reg.RDMA_FEATURE_MODE_CFG,
                ((2 << 15) |                          # DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION
                (15 << 11) |                          # DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN
                (2 << 5)  |                           # DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION
                (1 << 3)  |                           # DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN
                1)                                    # DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE
        ),
    ]
    return npu_regs

def cmplt_regs(n, input_mem, output_mem, chain_rearm=False):
    # positive diff -> 1, else 0
    pre_regs = [
        E(reg.TARGET_DPU, reg.BS_CFG,
                ((1 << 18) |                          # DPU_BS_CFG_BS_ALU_SRC
                (1 << 6))                             # DPU_BS_CFG_BS_RELU_BYPASS
        ),
        E(reg.TARGET_DPU, reg.BS_ALU_CFG,
                0x33800000                            # DPU_BS_ALU_CFG_BS_ALU_OPERAND(0.25 fp32)
        ),
        E(reg.TARGET_DPU, reg.BS_MUL_CFG,
                0x40000000                            # DPU_BS_MUL_CFG_BS_MUL_OPERAND(2.0 fp32)
        ),
        E(reg.TARGET_DPU, reg.BN_CFG,
                ((1 << 18) |                          # DPU_BN_CFG_BN_ALU_SRC
                (2 << 6)  |                           # DPU_BN_CFG_BN_RELU_ALGO
                (1 << 1))                             # DPU_BN_CFG_BN_ALU_BYPASS
        ),
        E(reg.TARGET_DPU, reg.BN_MUL_CFG,
                0x7c000000                            # DPU_BN_MUL_CFG_BN_MUL_OPERAND(+inf fp16 lane)
        ),
        E(reg.TARGET_DPU, reg.BN_RELUX_CMP_VALUE,
                0x3f800000                            # DPU_BN_RELUX_CMP_VALUE(1.0 fp32)
        ),
    ]
    return ew_regs(EW_CFG_CMP, n, input_mem, input_mem, output_mem, pre_regs, chain_rearm=chain_rearm)

def write_regs_to_npu_task(task_regs):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())

    def make_tail(next_offset, next_body_len):
        enable = E(reg.TARGET_PC, reg.OPERATION_ENABLE,
                ((1 << 4) |                           # PC_OPERATION_ENABLE_DPU
                (1 << 3))                             # PC_OPERATION_ENABLE_DPU_RDMA
        )
        if next_offset is None:
            return [
                E(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS,
                        0                              # PC_BASE_ADDRESS = 0 (end chain)
                ),
                E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS,
                        0                              # PC_REGISTER_AMOUNTS = 0 (end chain)
                ),
                E(reg.TARGET_VERSION, 0,
                        0                              # VERSION (OP_40 equivalent)
                ),
                enable,
            ]
        next_addr = regcmd_mem_create.dma_addr + next_offset * ctypes.sizeof(ctypes.c_uint64)
        return [
            E(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS,
                    next_addr & ~((1 << 4) - 1)        # PC_BASE_ADDRESS_NEXT_ALIGNED
            ),
            E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS,
                    next_body_len                      # PC_REGISTER_AMOUNTS_NEXT_BODY_QWORDS
            ),
            E(reg.TARGET_VERSION, 0,
                    0                                  # VERSION (OP_40 equivalent)
            ),
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

def mem_at(mem_create, start):
    addr = mem_create.dma_addr + start * FP16_BYTES
    view = rknpu_mem_create()
    view.dma_addr = addr
    return view

def tiled_regs(reg_builder, n):
    regs = []
    for start in range(0, n, WHERE_TILE_ELEMENTS):
        tile_n = min(WHERE_TILE_ELEMENTS, n - start)
        regs.append(reg_builder(start, tile_n))
    return regs

def run_regs(task_regs):
    if len(task_regs) == 1:
        write_single_regs_to_npu_task(task_regs[0] + [
            E(reg.TARGET_PC, reg.OPERATION_ENABLE,
                    ((1 << 4) |                       # PC_OPERATION_ENABLE_DPU
                    (1 << 3))                         # PC_OPERATION_ENABLE_DPU_RDMA
            )
        ])
        ret = submit(tasks_mem_create.obj_addr, task_count=1)
    else:
        write_regs_to_npu_task(task_regs)
        ret = submit(tasks_mem_create.obj_addr, task_count=len(task_regs))
    print(f"SUBMIT ret={ret}")
    if ret != 0:
        raise RuntimeError(f"RKNPU submit failed: {ret}")

def where_tiles(n):
    return [(start, min(WHERE_TILE_ELEMENTS, n - start)) for start in range(0, n, WHERE_TILE_ELEMENTS)]

def run_tiled_stage(n, reg_builder):
    tiles = where_tiles(n)
    task_regs = [reg_builder(start, tile_n, chain_rearm=len(tiles) > 1) for start, tile_n in tiles]
    run_regs(task_regs)

def run_where_tile_chained(start, tile_n):
    # Full 8-stage WHERE as one PC chain times out. The longest stable split is
    # SUB->CMP, then the six post-compare EW stages in a second PC chain.
    run_regs([
        ew_regs(EW_CFG_SUB, tile_n, mem_at(input_mem_create, start), mem_at(weight_mem_create, start), mem_at(output_mem_create, start), chain_rearm=True),
        cmplt_regs(tile_n, mem_at(output_mem_create, start), mem_at(scratch3_mem_create, start), chain_rearm=True),
    ])
    cleanup_dpu_state()
    run_regs([
        ew_regs(EW_CFG_MUL, tile_n, mem_at(scratch0_mem_create, start), mem_at(scratch3_mem_create, start), mem_at(scratch4_mem_create, start), chain_rearm=True),
        ew_regs(EW_CFG_MUL, tile_n, mem_at(scratch0_mem_create, start), mem_at(scratch3_mem_create, start), mem_at(output_mem_create, start), chain_rearm=True),
        ew_regs(EW_CFG_SUB, tile_n, mem_at(scratch2_mem_create, start), mem_at(scratch3_mem_create, start), mem_at(scratch4_mem_create, start), chain_rearm=True),
        ew_regs(EW_CFG_MUL, tile_n, mem_at(scratch1_mem_create, start), mem_at(scratch4_mem_create, start), mem_at(scratch3_mem_create, start), chain_rearm=True),
        ew_regs(EW_CFG_MUL, tile_n, mem_at(scratch1_mem_create, start), mem_at(scratch4_mem_create, start), mem_at(scratch2_mem_create, start), chain_rearm=True),
        ew_regs(EW_CFG_ADD, tile_n, mem_at(output_mem_create, start), mem_at(scratch2_mem_create, start), mem_at(scratch3_mem_create, start), chain_rearm=True),
    ])

def run_where_tile_direct(start, tile_n):
    # Safe legacy sequence. The custom compare can leave state that makes later
    # PC-chain submits order-sensitive, so keep this path for default tests.
    run_regs([ew_regs(EW_CFG_SUB, tile_n, mem_at(input_mem_create, start), mem_at(weight_mem_create, start), mem_at(output_mem_create, start))])
    run_regs([cmplt_regs(tile_n, mem_at(output_mem_create, start), mem_at(scratch3_mem_create, start))])
    run_regs([ew_regs(EW_CFG_MUL, tile_n, mem_at(scratch0_mem_create, start), mem_at(scratch3_mem_create, start), mem_at(scratch4_mem_create, start))])
    run_regs([ew_regs(EW_CFG_MUL, tile_n, mem_at(scratch0_mem_create, start), mem_at(scratch3_mem_create, start), mem_at(output_mem_create, start))])
    run_regs([ew_regs(EW_CFG_SUB, tile_n, mem_at(scratch2_mem_create, start), mem_at(scratch3_mem_create, start), mem_at(scratch4_mem_create, start))])
    run_regs([ew_regs(EW_CFG_MUL, tile_n, mem_at(scratch1_mem_create, start), mem_at(scratch4_mem_create, start), mem_at(scratch3_mem_create, start))])
    run_regs([ew_regs(EW_CFG_MUL, tile_n, mem_at(scratch1_mem_create, start), mem_at(scratch4_mem_create, start), mem_at(scratch2_mem_create, start))])
    run_regs([ew_regs(EW_CFG_ADD, tile_n, mem_at(output_mem_create, start), mem_at(scratch2_mem_create, start), mem_at(scratch3_mem_create, start))])


def check_where(n, chain_phases=False):
    x = np.linspace(-1.0, 2.0, n, dtype=np.float16)
    if n >= 8:
        x[:8] = np.array([0.0, 0.25, 0.5, 0.75, 1.0, -1.0, 2.0, 0.49], dtype=np.float16)
    a = ((np.arange(n, dtype=np.float32) % 1024.0) + 10.0).astype(np.float16)
    b = np.ones(n, dtype=np.float16)
    expected = np.where(x > np.float16(0.5), a, b).astype(np.float16)

    write_f16(input_map, x)
    write_f16(weight_map, np.full(n, 0.5, dtype=np.float16))
    write_f16(scratch0_map, a)
    write_f16(scratch1_map, b)
    write_f16(scratch2_map, np.ones(n, dtype=np.float16))

    for start in range(0, n, WHERE_TILE_ELEMENTS):
        tile_n = min(WHERE_TILE_ELEMENTS, n - start)
        if chain_phases:
            run_where_tile_chained(start, tile_n)
        else:
            run_where_tile_direct(start, tile_n)

    got = read_f16(scratch3_map, n)
    ok = np.allclose(got, expected, atol=0.1)
    print(f"shape=({n},) tasks={_ceil_div(n, WHERE_TILE_ELEMENTS)}")
    print(f"x={x[:min(8, n)]}")
    print(f"a={a[:min(8, n)]}")
    print(f"b={b[:min(8, n)]}")
    print(f"NPU={got[:min(8, n)]}")
    print(f"expected={expected[:min(8, n)]}")
    print(f"WHERE {n} {'PASS' if ok else 'FAIL'}")
    return ok

def check_pcchain_sub(n):
    x = np.linspace(-1.0, 2.0, n, dtype=np.float16)
    y = np.full(n, 0.5, dtype=np.float16)
    expected = (x - y).astype(np.float16)

    write_f16(input_map, x)
    write_f16(weight_map, y)
    output_map[: n * FP16_BYTES] = b"\x00" * (n * FP16_BYTES)

    task_regs = []
    for start in range(0, n, PCCHAIN_TILE_ELEMENTS):
        tile_n = min(PCCHAIN_TILE_ELEMENTS, n - start)
        task_regs.append(ew_regs(EW_CFG_SUB, tile_n,
                                 mem_at(input_mem_create, start),
                                 mem_at(weight_mem_create, start),
                                 mem_at(output_mem_create, start),
                                 chain_rearm=True))
    run_regs(task_regs)

    got = read_f16(output_map, n)
    ok = np.allclose(got, expected, atol=0.01)
    print(f"pcchain_sub_shape=({n},) tasks={len(task_regs)}")
    print(f"PCCHAIN SUB NPU={got[:min(8, n)]}")
    print(f"PCCHAIN SUB expected={expected[:min(8, n)]}")
    print(f"PCCHAIN SUB {n} {'PASS' if ok else 'FAIL'}")
    return ok

def cleanup_dpu_state():
    # The custom compare path leaves enough BS/BN and ping-pong state that a
    # minimal later DPU op that does not program those registers can observe it.
    # Run one fully-specified EW ADD, like conv/gemm's explicit state writes, so
    # the next standalone op starts from a clean DPU/RDMA state.
    n = 8
    (ctypes.c_uint16 * n).from_buffer(input_map)[:] = [3] * n
    (ctypes.c_uint16 * n).from_buffer(weight_map)[:] = [5] * n
    regs = [
        E(reg.TARGET_DPU,  reg.S_POINTER,
                ((1 << 3) |                           # DPU_S_POINTER_POINTER_PP_MODE
                (1 << 2)  |                           # DPU_S_POINTER_EXECUTER_PP_EN
                (1 << 1))                             # DPU_S_POINTER_POINTER_PP_EN
        ),
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
        E(reg.TARGET_DPU,  reg.DATA_CUBE_WIDTH,
                0                                      # DPU_DATA_CUBE_WIDTH_WIDTH
        ),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_HEIGHT,
                0                                      # DPU_DATA_CUBE_HEIGHT_HEIGHT
        ),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_NOTCH,
                0                                      # DPU_DATA_CUBE_NOTCH
        ),
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
        E(reg.TARGET_DPU,  reg.BN_CFG,
                ((1 << 6) |                           # DPU_BN_CFG_BN_RELU_BYPASS
                (1 << 4)  |                           # DPU_BN_CFG_BN_MUL_BYPASS
                (1 << 1)  |                           # DPU_BN_CFG_BN_ALU_BYPASS
                1)                                    # DPU_BN_CFG_BN_BYPASS
        ),
        E(reg.TARGET_DPU,  reg.EW_CFG,
                EW_CFG_ADD                            # DPU_EW_CFG
        ),
        E(reg.TARGET_DPU,  reg.OUT_CVT_SCALE,
                ((1 << 16) |                          # DPU_OUT_CVT_SCALE_FP32TOFP16_EN
                1)                                    # DPU_OUT_CVT_SCALE_OUT_CVT_SCALE
        ),
        E(reg.TARGET_RDMA, reg.RDMA_S_POINTER,
                ((1 << 3) |                           # DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE
                (1 << 2)  |                           # DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN
                (1 << 1))                             # DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN
        ),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH,
                0                                      # DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH
        ),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT,
                0                                      # DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT
        ),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL,
                7                                      # DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL
        ),
        E(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG,
                ((1 << 30) |                          # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE
                (2 << 2))                             # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE
        ),
        E(reg.TARGET_DPU,  reg.DST_BASE_ADDR,
                low32(output_mem_create.dma_addr)      # DPU_DST_BASE_ADDR_LOW
        ),
        E(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR,
                low32(input_mem_create.dma_addr)       # DPU_RDMA_RDMA_SRC_BASE_ADDR_LOW
        ),
        E(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR,
                low32(weight_mem_create.dma_addr)      # DPU_RDMA_RDMA_EW_BASE_ADDR_LOW
        ),
        E(reg.TARGET_RDMA, reg.RDMA_FEATURE_MODE_CFG,
                ((2 << 15) |                          # DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION
                (15 << 11) |                          # DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN
                (2 << 5)  |                           # DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION
                (1 << 3)  |                           # DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN
                1)                                    # DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE
        ),
    ]
    run_regs([regs])
    run_regs([regs])

if __name__ == "__main__":
    ok = True
    if "--where-only" in sys.argv:
        ok = check_where(8)
        cleanup_dpu_state()
        os.close(fd)
        raise SystemExit(0 if ok else 1)
    if "--chain-where-only" in sys.argv:
        ok = check_where(8, chain_phases=True)
        cleanup_dpu_state()
        os.close(fd)
        raise SystemExit(0 if ok else 1)
    if "--pcchain-only" in sys.argv:
        for n in (65536, 130000):
            ok = check_pcchain_sub(n) and ok
        os.close(fd)
        raise SystemExit(0 if ok else 1)

    ok = check_where(8, chain_phases=True) and ok
    cleanup_dpu_state()
    os.close(fd)
    raise SystemExit(0 if ok else 1)
