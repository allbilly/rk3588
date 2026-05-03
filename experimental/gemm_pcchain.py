from fcntl import ioctl
import argparse
import ctypes
import mmap
import os

import numpy as np


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_OFFICIAL_TASK = 0x40B
RKNPU_MEM_OFFICIAL_TENSOR = 0x403
RKNPU_JOB_PC = 0x1
RKNPU_JOB_PINGPONG = 0x4

M = N = K = 394
ALIGN_IN = ALIGN_OUT = 416
ROWS_PER_TASK = 39
REGCMD_BYTES = 3072
TASKS = 11

TASK_OFFSETS = [0x000, 0x380, 0x700, 0x7C0, 0x840, 0x8C0, 0x940, 0x9C0, 0xA40, 0xAC0, 0xB40]
TASK_AMOUNTS = [108, 108, 13, 12, 12, 12, 12, 12, 12, 12, 17]
PC_NEXT_AMOUNTS = [55, 8, 7, 7, 7, 7, 7, 7, 7, 10, 0]
DRY_INPUT_BASE = 0xFFF0C000
DRY_WEIGHT_BASE = 0xFFF5D000
DRY_OUTPUT_BASE = 0xFFE16000
DRY_REGCMD_BASE = 0xFFFB3000


class R:
    CNA = 0x0201
    CORE = 0x0801
    DPU = 0x1001
    PC = 0x0081
    PC_REG = 0x0101
    VERSION = 0x0041

    OPERATION_ENABLE = 0x0008
    PC_BASE_ADDRESS = 0x0010
    PC_REGISTER_AMOUNTS = 0x0014

    CNA_CONV_CON1 = 0x100C
    CNA_CONV_CON2 = 0x1010
    CNA_CONV_CON3 = 0x1014
    CNA_DATA_SIZE0 = 0x1020
    CNA_DATA_SIZE1 = 0x1024
    CNA_DATA_SIZE2 = 0x1028
    CNA_DATA_SIZE3 = 0x102C
    CNA_WEIGHT_SIZE0 = 0x1030
    CNA_WEIGHT_SIZE1 = 0x1034
    CNA_WEIGHT_SIZE2 = 0x1038
    CNA_CBUF_CON0 = 0x1040
    CNA_CBUF_CON1 = 0x1044
    CNA_CVT_CON0 = 0x104C
    CNA_CVT_CON1 = 0x1050
    CNA_CVT_CON2 = 0x1054
    CNA_CVT_CON3 = 0x1058
    CNA_CVT_CON4 = 0x105C
    CNA_CVT_CON5 = 0x1060
    CNA_CVT_CON6 = 0x1064
    CNA_CVT_CON7 = 0x1068
    CNA_FEATURE_DATA_ADDR = 0x1070
    CNA_FEATURE_DATA_ADDR_HIGH = 0x1074
    CNA_DMA_CON0 = 0x1078
    CNA_DMA_CON1 = 0x107C
    CNA_DMA_CON2 = 0x1080
    CNA_FC_DATA_SIZE0 = 0x1084
    CNA_FC_DATA_SIZE1 = 0x1088
    CNA_DCOMP_CTRL = 0x1100
    CNA_RESERVED_1104 = 0x1104
    CNA_DCOMP_ADDR0 = 0x1110
    CNA_DCOMP_ZERO = 0x1140

    CORE_MISC_CFG = 0x3010
    CORE_DATAOUT_SIZE_0 = 0x3014
    CORE_DATAOUT_SIZE_1 = 0x3018
    CORE_RESERVED_301C = 0x301C
    CORE_RESERVED_3030 = 0x3030

    DPU_S_POINTER = 0x4004
    DPU_FEATURE_MODE_CFG = 0x400C
    DPU_DATA_FORMAT = 0x4010
    DPU_RESERVED_4014 = 0x4014
    DPU_DST_BASE_ADDR = 0x4020
    DPU_DST_SURF_STRIDE = 0x4024
    DPU_DATA_CUBE_WIDTH = 0x4030
    DPU_DATA_CUBE_HEIGHT = 0x4034
    DPU_DATA_CUBE_NOTCH = 0x4038
    DPU_DATA_CUBE_CHANNEL = 0x403C
    DPU_BS_CFG = 0x4040
    DPU_BS_ALU_SRC_VALUE = 0x4044
    DPU_BS_MUL_SRC_VALUE = 0x4048
    DPU_BS_RELU_CFG = 0x404C
    DPU_BS_OW_CFG = 0x4050
    DPU_BS_OW_OP = 0x4054
    DPU_WDMA_SIZE_0 = 0x4058
    DPU_WDMA_SIZE_1 = 0x405C
    DPU_BN_CFG = 0x4060
    DPU_BN_ALU_SRC_VALUE = 0x4064
    DPU_BN_MUL_SRC_VALUE = 0x4068
    DPU_BN_RELU_CFG = 0x406C
    DPU_EW_CFG = 0x4070
    DPU_EW_ALU_SRC_VALUE = 0x4074
    DPU_EW_MUL_SRC_VALUE = 0x4078
    DPU_EW_TRUNCATE_VALUE = 0x407C
    DPU_OUT_CVT_OFFSET = 0x4080
    DPU_OUT_CVT_SCALE = 0x4084
    DPU_OUT_CVT_SHIFT = 0x4088
    DPU_DST_DMA_CFG = 0x4090
    DPU_DST_COMPRESSION_EN = 0x4094
    DPU_D_FEATURE_MODE_CFG = 0x4098
    DPU_D_DST_DMA_CFG = 0x409C
    DPU_D_DATA_FORMAT = 0x40A0
    DPU_D_DST_BASE_ADDR = 0x40A4
    DPU_D_DST_SURF_STRIDE = 0x40A8
    DPU_D_DATA_CUBE_CHANNEL = 0x40AC
    DPU_SURFACE_ADD = 0x40C0
    DPU_SURF_ADD_HIGH = 0x40C4
    DPU_LUT_ACCESS_CFG = 0x4100
    DPU_LUT_ACCESS_DATA = 0x4104
    DPU_LUT_CFG = 0x4108
    DPU_LUT_INFO = 0x410C
    DPU_LUT_LE_START = 0x4110
    DPU_LUT_LE_END = 0x4114
    DPU_LUT_LO_START = 0x4118
    DPU_LUT_LO_END = 0x411C
    DPU_LUT_LE_SLOPE_SCALE = 0x4120
    DPU_LUT_LE_SLOPE_SHIFT = 0x4124
    DPU_LUT_LO_SLOPE_SCALE = 0x4128
    DPU_LUT_LO_SLOPE_SHIFT = 0x412C


REG_NAMES = {value: name for name, value in vars(R).items() if name.isupper() and isinstance(value, int)}
TARGET_NAMES = {R.CNA: "CNA", R.CORE: "CORE", R.DPU: "DPU", R.PC: "PC", R.PC_REG: "PC_REG", R.VERSION: "VERSION"}


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
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64)]


class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]


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


class rknpu_task(ctypes.Structure):
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


def _iowr(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)


DRM_IOCTL_RKNPU_MEM_CREATE = _iowr("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _iowr("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _iowr("d", 0x41, ctypes.sizeof(rknpu_submit))


def align_up(value, align):
    return ((value + align - 1) // align) * align


def E(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def reg_value(qword):
    return (qword >> 48) & 0xFFFF, qword & 0xFFFF, (qword >> 16) & 0xFFFFFFFF


def mem_allocate(fd, size, flags):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc


def pack_input_row_major(a):
    padded = np.zeros((M, ALIGN_IN), dtype=np.float16)
    padded[:, :K] = a
    return padded.ravel()


def pack_weight_tile_16x32(b):
    wt = np.zeros((ALIGN_OUT, ALIGN_IN), dtype=np.float16)
    wt[:N, :K] = b.T
    return wt.reshape(ALIGN_OUT // 16, 16, ALIGN_IN // 32, 32).transpose(0, 2, 1, 3).ravel()


def full_setup_regs(task_idx, input_dma, weight_dma, output_dma):
    row0 = task_idx * ROWS_PER_TASK
    input_addr = input_dma + row0 * ALIGN_IN * 2
    output_addr = output_dma + row0 * ALIGN_OUT * 4
    cbuf_con0 = 0xB1 if task_idx == 0 else 0x20B1
    return [
        E(R.CNA, R.CNA_CBUF_CON0, cbuf_con0),
        E(R.CNA, R.CNA_RESERVED_1104, 0),
        E(R.CNA, R.CNA_DCOMP_CTRL, 0),
        E(R.CNA, R.CNA_CONV_CON1, 0x20000120),
        E(R.DPU, R.DPU_S_POINTER, 0x0000000E),
        E(R.CNA, R.CNA_CONV_CON1, 0x20000120),
        E(R.CNA, R.CNA_CONV_CON2, 0x00000280),
        E(R.CNA, R.CNA_CONV_CON3, 0x00000009),
        E(R.CNA, R.CNA_DATA_SIZE0, 0x00010027),
        E(R.CNA, R.CNA_DATA_SIZE1, 0x019F01A0),
        E(R.CNA, R.CNA_DATA_SIZE2, 0x00000001),
        E(R.CNA, R.CNA_DATA_SIZE3, 0x00000027),
        E(R.CNA, R.CNA_WEIGHT_SIZE0, 0x00054800),
        E(R.CNA, R.CNA_WEIGHT_SIZE1, 0x00000340),
        E(R.CNA, R.CNA_WEIGHT_SIZE2, 0x010101A0),
        E(R.CNA, R.CNA_CBUF_CON0, cbuf_con0),
        E(R.CNA, R.CNA_CBUF_CON1, 0x0000000D),
        E(R.CNA, R.CNA_CVT_CON0, 0x0000000B),
        E(R.CNA, R.CNA_CVT_CON1, 0x00010000),
        E(R.CNA, R.CNA_CVT_CON2, 0x00010000),
        E(R.CNA, R.CNA_CVT_CON3, 0x00010000),
        E(R.CNA, R.CNA_CVT_CON4, 0x00010000),
        E(R.CNA, R.CNA_CVT_CON5, 0),
        E(R.CNA, R.CNA_CVT_CON6, 0),
        E(R.CNA, R.CNA_CVT_CON7, 0),
        E(R.CNA, R.CNA_FEATURE_DATA_ADDR, input_addr),
        E(R.CNA, R.CNA_FEATURE_DATA_ADDR_HIGH, 0),
        E(R.CNA, R.CNA_DMA_CON0, 0x000F000F),
        E(R.CNA, R.CNA_DMA_CON1, 0x00000034),
        E(R.CNA, R.CNA_DMA_CON2, 0),
        E(R.CNA, R.CNA_FC_DATA_SIZE0, 0x00010027),
        E(R.CNA, R.CNA_FC_DATA_SIZE1, 0x000001A0),
        E(R.CNA, R.CNA_DCOMP_CTRL, 0),
        E(R.CNA, R.CNA_RESERVED_1104, 0),
        E(R.CNA, R.CNA_DCOMP_ADDR0, weight_dma),
        E(R.CNA, R.CNA_DCOMP_ZERO, 0),
        *[E(R.CNA, 0x1144 + i * 4, 0) for i in range(17)],
        E(R.CORE, R.CORE_MISC_CFG, 0x00000201),
        E(R.CORE, R.CORE_DATAOUT_SIZE_0, 0x00260000),
        E(R.CORE, R.CORE_DATAOUT_SIZE_1, 0x0000019F),
        E(R.CORE, R.CORE_RESERVED_301C, 0),
        E(R.CORE, R.CORE_RESERVED_3030, 0),
        E(R.DPU, R.DPU_FEATURE_MODE_CFG, 0x000001E4),
        E(R.DPU, R.DPU_DATA_FORMAT, 0xA8000002),
        E(R.DPU, R.DPU_RESERVED_4014, 0),
        E(R.DPU, R.DPU_DST_BASE_ADDR, output_addr),
        E(R.DPU, R.DPU_DST_SURF_STRIDE, 0x00000010),
        E(R.DPU, R.DPU_DATA_CUBE_WIDTH, 0),
        E(R.DPU, R.DPU_DATA_CUBE_HEIGHT, 0x00000026),
        E(R.DPU, R.DPU_DATA_CUBE_NOTCH, 0x00670067),
        E(R.DPU, R.DPU_DATA_CUBE_CHANNEL, 0x019F019F),
        E(R.DPU, R.DPU_BS_CFG, 0x00000053),
        E(R.DPU, R.DPU_BS_ALU_SRC_VALUE, 0),
        E(R.DPU, R.DPU_BS_MUL_SRC_VALUE, 0),
        E(R.DPU, R.DPU_BS_RELU_CFG, 0),
        E(R.DPU, R.DPU_BS_OW_CFG, 0x0000036E),
        E(R.DPU, R.DPU_BS_OW_OP, 0),
        E(R.DPU, R.DPU_WDMA_SIZE_0, 0x0000019F),
        E(R.DPU, R.DPU_WDMA_SIZE_1, 0x00260000),
        E(R.DPU, R.DPU_BN_CFG, 0x00000053),
        E(R.DPU, R.DPU_BN_ALU_SRC_VALUE, 0),
        E(R.DPU, R.DPU_BN_MUL_SRC_VALUE, 0),
        E(R.DPU, R.DPU_BN_RELU_CFG, 0),
        E(R.DPU, R.DPU_EW_CFG, 0x00000383),
        E(R.DPU, R.DPU_EW_ALU_SRC_VALUE, 0),
        E(R.DPU, R.DPU_EW_MUL_SRC_VALUE, 1),
        E(R.DPU, R.DPU_EW_TRUNCATE_VALUE, 0),
        E(R.DPU, R.DPU_OUT_CVT_OFFSET, 0),
        E(R.DPU, R.DPU_OUT_CVT_SCALE, 1),
        E(R.DPU, R.DPU_OUT_CVT_SHIFT, 0),
        E(R.DPU, R.DPU_DST_DMA_CFG, 0),
        E(R.DPU, R.DPU_DST_COMPRESSION_EN, 0),
        E(R.DPU, R.DPU_D_FEATURE_MODE_CFG, 0),
        E(R.DPU, R.DPU_D_DST_DMA_CFG, 0),
        E(R.DPU, R.DPU_D_DATA_FORMAT, 0),
        E(R.DPU, R.DPU_D_DST_BASE_ADDR, 0),
        E(R.DPU, R.DPU_D_DST_SURF_STRIDE, 0),
        E(R.DPU, R.DPU_D_DATA_CUBE_CHANNEL, 0),
        E(R.DPU, R.DPU_SURFACE_ADD, 0x00000040),
        E(R.DPU, R.DPU_SURF_ADD_HIGH, 0),
        E(R.DPU, R.DPU_LUT_ACCESS_CFG, 0),
        E(R.DPU, R.DPU_LUT_ACCESS_DATA, 0),
        E(R.DPU, R.DPU_LUT_CFG, 0),
        E(R.DPU, R.DPU_LUT_INFO, 0),
        E(R.DPU, R.DPU_LUT_LE_START, 0),
        E(R.DPU, R.DPU_LUT_LE_END, 0),
        E(R.DPU, R.DPU_LUT_LO_START, 0),
        E(R.DPU, R.DPU_LUT_LO_END, 0),
        E(R.DPU, R.DPU_LUT_LE_SLOPE_SCALE, 0),
        E(R.DPU, R.DPU_LUT_LE_SLOPE_SHIFT, 0),
        E(R.DPU, R.DPU_LUT_LO_SLOPE_SCALE, 0),
        E(R.DPU, R.DPU_LUT_LO_SLOPE_SHIFT, 0),
    ]


def slice_update_regs(task_idx, input_dma, weight_dma, output_dma):
    row0 = task_idx * ROWS_PER_TASK
    rows = M - row0 if task_idx == TASKS - 1 else ROWS_PER_TASK
    input_addr = input_dma + row0 * ALIGN_IN * 2
    output_addr = output_dma + row0 * ALIGN_OUT * 4
    header = [
        E(R.CNA, R.CNA_CBUF_CON0, 0x000020B1),
        E(R.CNA, R.CNA_RESERVED_1104, 0),
        E(R.CNA, R.CNA_DCOMP_CTRL, 0),
        E(R.CNA, R.CNA_CONV_CON1, 0x20000120),
        E(R.DPU, R.DPU_S_POINTER, 0x0000000E),
    ]
    if task_idx == 2:
        header.append(E(R.CNA, R.CNA_CBUF_CON0, 0x000020B1))
    if rows == ROWS_PER_TASK:
        return header + [
            E(R.CNA, R.CNA_FEATURE_DATA_ADDR, input_addr),
            E(R.CNA, R.CNA_FC_DATA_SIZE0, 0x00010027),
            E(R.CNA, R.CNA_FC_DATA_SIZE1, 0x000001A0),
            E(R.CNA, R.CNA_DCOMP_ADDR0, weight_dma),
            E(R.DPU, R.DPU_DST_BASE_ADDR, output_addr),
            E(R.DPU, R.DPU_WDMA_SIZE_0, 0x0000019F),
            E(R.DPU, R.DPU_WDMA_SIZE_1, 0x00260000),
        ]
    return header + [
        E(R.CNA, R.CNA_CONV_CON2, 0x00000050),
        E(R.CNA, R.CNA_DATA_SIZE0, 0x00010004),
        E(R.CNA, R.CNA_DATA_SIZE3, 0x00000004),
        E(R.CNA, R.CNA_FEATURE_DATA_ADDR, input_addr),
        E(R.CNA, R.CNA_FC_DATA_SIZE0, 0x00010004),
        E(R.CNA, R.CNA_FC_DATA_SIZE1, 0x000001A0),
        E(R.CNA, R.CNA_DCOMP_ADDR0, weight_dma),
        E(R.CORE, R.CORE_DATAOUT_SIZE_0, 0x00030000),
        E(R.DPU, R.DPU_DST_BASE_ADDR, output_addr),
        E(R.DPU, R.DPU_DATA_CUBE_HEIGHT, 0x00000003),
        E(R.DPU, R.DPU_WDMA_SIZE_0, 0x0000019F),
        E(R.DPU, R.DPU_WDMA_SIZE_1, 0x00030000),
    ]


def build_task_regs(input_dma, weight_dma, output_dma):
    regs = []
    for task_idx in range(TASKS):
        if task_idx < 2:
            task_regs = full_setup_regs(task_idx, input_dma, weight_dma, output_dma)
        else:
            task_regs = slice_update_regs(task_idx, input_dma, weight_dma, output_dma)
        if len(task_regs) != TASK_AMOUNTS[task_idx]:
            raise RuntimeError(f"task {task_idx} has {len(task_regs)} regs, expected {TASK_AMOUNTS[task_idx]}")
        regs.append(task_regs)
    return regs


def pc_tail(task_idx, regcmd_dma):
    if task_idx + 1 < TASKS:
        next_addr = regcmd_dma + TASK_OFFSETS[task_idx + 1]
        return [
            E(R.PC_REG, R.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
            E(R.PC_REG, R.PC_REGISTER_AMOUNTS, PC_NEXT_AMOUNTS[task_idx]),
            E(R.VERSION, 0, 0),
            E(R.PC, R.OPERATION_ENABLE, 0x0D),
        ]
    return [
        0,
        E(R.PC_REG, R.PC_REGISTER_AMOUNTS, 0),
        E(R.VERSION, 0, 0),
        E(R.PC, R.OPERATION_ENABLE, 0x0D),
    ]


def write_regcmd(regcmd_map, regcmd_dma, task_regs):
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    for i in range(REGCMD_BYTES // ctypes.sizeof(ctypes.c_uint64)):
        regcmd[i] = 0
    for task_idx, regs in enumerate(task_regs):
        base = TASK_OFFSETS[task_idx] // ctypes.sizeof(ctypes.c_uint64)
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        for i, qword in enumerate(pc_tail(task_idx, regcmd_dma)):
            regcmd[base + TASK_AMOUNTS[task_idx] + i] = qword


def fill_tasks(task_map, task_obj_addr, regcmd_dma):
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
    for idx, (offset, amount) in enumerate(zip(TASK_OFFSETS, TASK_AMOUNTS)):
        tasks[idx].flags = 0
        tasks[idx].op_idx = 0
        tasks[idx].enable_mask = 0x0D
        tasks[idx].int_mask = 0x300
        tasks[idx].int_clear = 0x1FFFF
        tasks[idx].int_status = 0
        tasks[idx].regcfg_amount = amount
        tasks[idx].regcfg_offset = 0
        tasks[idx].regcmd_addr = regcmd_dma + offset
    return task_obj_addr


def submit(fd, task_obj_addr, mode, timeout):
    official = mode == "official"
    submit_struct = rknpu_submit(
        flags=RKNPU_JOB_PC | RKNPU_JOB_PINGPONG,
        timeout=timeout,
        task_start=0,
        task_number=TASKS * 3 if official else TASKS,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=0 if official else 1,
        fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(0, TASKS)
    submit_struct.subcore_task[1] = rknpu_subcore_task(0, TASKS if official else 0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(0, TASKS if official else 0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(0, 0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(0, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def print_dry_run(task_regs, regcmd_dma):
    for task_idx, regs in enumerate(task_regs):
        print(f"task[{task_idx}] offset=0x{TASK_OFFSETS[task_idx]:03x} amount={TASK_AMOUNTS[task_idx]}")
        for i, qword in enumerate(regs + pc_tail(task_idx, regcmd_dma)):
            target, addr, value = reg_value(qword)
            target_name = TARGET_NAMES.get(target, f"0x{target:04x}")
            reg_name = REG_NAMES.get(addr, f"REG_0x{addr:04x}")
            print(f"  [{i:3d}] {target_name}.{reg_name} = 0x{value:08x}")


def main():
    parser = argparse.ArgumentParser(description="Decoded RKNN 394x394x394 GEMM PC-chain reproducer.")
    parser.add_argument("--mode", choices=("core0", "official"), default="core0")
    parser.add_argument("--alloc-mode", choices=("raw", "official"), default="raw")
    parser.add_argument("--constant-data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    if args.constant_data:
        a = np.ones((M, K), dtype=np.float16)
        b = np.ones((K, N), dtype=np.float16)
        expected = np.full((M, N), K, dtype=np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        a = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
        b = rng.standard_normal((K, N), dtype=np.float32).astype(np.float16)
        expected = a @ b

    if args.dry:
        task_regs = build_task_regs(DRY_INPUT_BASE, DRY_WEIGHT_BASE, DRY_OUTPUT_BASE)
        print_dry_run(task_regs, DRY_REGCMD_BASE)
        return 0

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        task_flags = RKNPU_MEM_OFFICIAL_TASK if args.alloc_mode == "official" else RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE
        tensor_flags = RKNPU_MEM_OFFICIAL_TENSOR if args.alloc_mode == "official" else RKNPU_MEM_NON_CACHEABLE
        task_map, task_mc = mem_allocate(fd, TASKS * ctypes.sizeof(rknpu_task), task_flags)
        regcmd_map, regcmd_mc = mem_allocate(fd, REGCMD_BYTES, tensor_flags)
        input_map, input_mc = mem_allocate(fd, align_up(M * ALIGN_IN * 2, 4096), tensor_flags)
        weight_map, weight_mc = mem_allocate(fd, align_up(ALIGN_IN * ALIGN_OUT * 2, 4096), tensor_flags)
        output_words = M * ALIGN_OUT
        output_map, output_mc = mem_allocate(fd, align_up(output_words * 4, 4096), tensor_flags)

        np.frombuffer(input_map, dtype=np.float16, count=M * ALIGN_IN)[:] = pack_input_row_major(a)
        np.frombuffer(weight_map, dtype=np.float16, count=ALIGN_IN * ALIGN_OUT)[:] = pack_weight_tile_16x32(b)
        np.frombuffer(output_map, dtype=np.float32)[:] = np.nan

        task_regs = build_task_regs(input_mc.dma_addr, weight_mc.dma_addr, output_mc.dma_addr)
        write_regcmd(regcmd_map, regcmd_mc.dma_addr, task_regs)
        fill_tasks(task_map, task_mc.obj_addr, regcmd_mc.dma_addr)

        print(f"regcmd=0x{regcmd_mc.dma_addr:x} tasks={TASKS} mode={args.mode}")
        ret = submit(fd, task_mc.obj_addr, args.mode, args.timeout)
        print(f"submit ret={ret}")
        raw = np.frombuffer(output_map, dtype=np.float32, count=output_words).copy()
        got = raw[np.arange(M)[:, None] * ALIGN_OUT + np.arange(N)]
        ok = ret == 0 and np.allclose(got, expected, atol=args.atol, equal_nan=False)
        max_diff = float(np.nanmax(np.abs(got - expected)))
        print(f"394x394x394 pcchain {'PASS' if ok else 'FAIL'} max_diff={max_diff:.6f}")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
