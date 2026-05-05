#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import mmap
import os
from dataclasses import dataclass
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock, ioctl

import numpy as np


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_OFFICIAL_TASK = 0x40B
RKNPU_MEM_OFFICIAL_TENSOR = 0x403
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0x2
RKNPU_JOB_PINGPONG = 0x4
RKNPU_ACT_RESET = 1
LOCK_PATH = "/tmp/rk3588_npu_submit.lock"

TASK_ENABLE = 0x0D
TASK_INT_MASK = 0x300


class R:
    ZERO = 0x0000
    VERSION = 0x0041
    PC = 0x0081
    PC_REG = 0x0101
    CNA = 0x0201
    CORE = 0x0801
    DPU = 0x1001

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
    CNA_DCOMP_PAD_0 = 0x1144
    CNA_DCOMP_PAD_1 = 0x1148
    CNA_DCOMP_PAD_2 = 0x114C
    CNA_DCOMP_PAD_3 = 0x1150
    CNA_DCOMP_PAD_4 = 0x1154
    CNA_DCOMP_PAD_5 = 0x1158
    CNA_DCOMP_PAD_6 = 0x115C
    CNA_DCOMP_PAD_7 = 0x1160
    CNA_DCOMP_PAD_8 = 0x1164
    CNA_DCOMP_PAD_9 = 0x1168
    CNA_DCOMP_PAD_10 = 0x116C
    CNA_DCOMP_PAD_11 = 0x1170
    CNA_DCOMP_PAD_12 = 0x1174
    CNA_DCOMP_PAD_13 = 0x1178
    CNA_DCOMP_PAD_14 = 0x117C
    CNA_DCOMP_PAD_15 = 0x1180
    CNA_DCOMP_PAD_16 = 0x1184

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
TARGET_NAMES = {
    R.ZERO: "ZERO",
    R.VERSION: "VERSION",
    R.PC: "PC",
    R.PC_REG: "PC_REG",
    R.CNA: "CNA",
    R.CORE: "CORE",
    R.DPU: "DPU",
}


@dataclass(frozen=True)
class ConvTask:
    idx: int
    conv_con2: int
    data_size0: int
    data_size1: int
    data_size2: int
    data_size3: int
    weight_size0: int
    weight_size1: int
    weight_size2: int
    cbuf_con1: int
    input_offset: int
    fc_data_size0: int
    fc_data_size1: int
    weight_offset: int
    core_dataout_size0: int
    core_dataout_size1: int
    output_offset: int
    dst_surf_stride: int
    cube_width: int
    cube_height: int
    cube_channel: int
    wdma_size0: int
    wdma_size1: int
    surface_add: int


@dataclass(frozen=True)
class CaseSpec:
    name: str
    task_offsets: tuple[int, ...]
    input_size: int
    weight_size: int
    internal_size: int
    output_size: int
    check_offset: int
    check_size: int
    check_ranges: tuple[tuple[int, int], ...]
    cna_dma_con1: int
    cna_dma_con2: int
    dpu_bs_ow_cfg: int
    conv_tasks: tuple[ConvTask, ...]


CASES = {
    1: CaseSpec(
        name="small 1x3 RKNN conv pcchain",
        task_offsets=(0x940,),
        input_size=0x50,
        weight_size=0x2500,
        internal_size=0x380,
        output_size=0x300,
        check_offset=0,
        check_size=0x2A0,
        check_ranges=((0x000, 0x150), (0x180, 0x150)),
        cna_dma_con1=0x08,
        cna_dma_con2=0x20,
        dpu_bs_ow_cfg=0x126,
        conv_tasks=(
            ConvTask(0, 0x00000080, 0x00080005, 8, 7, 0x15, 0x120, 0x30, 0x01030006, 0x28, 0, 0x00070005, 8, 0, 0x00020006, 0x0F, 0, 0x180, 6, 2, 0x0005000F, 0x0F, 0x00020006, 0x300),
        ),
    ),
    4: CaseSpec(
        name="large 5x5 RKNN conv pcchain",
        task_offsets=(0x3A00, 0x3D80, 0x4200, 0x4680, 0x4B00, 0x4F80),
        input_size=0x140,
        weight_size=0x55C0,
        internal_size=0xA40,
        output_size=0x900,
        check_offset=0,
        check_size=0x900,
        check_ranges=((0x000, 0x900),),
        cna_dma_con1=0x10,
        cna_dma_con2=0x90,
        dpu_bs_ow_cfg=0x126,
        conv_tasks=(
            ConvTask(0, 0x000000F0, 0x0010000A, 8, 6, 0x24, 0x3200, 0x190, 0x05050020, 0xA0, 0, 0x000A000A, 8, 0, 0x00050005, 0x1F, 0, 0x240, 5, 5, 0x001F001F, 0x1F, 0x00050005, 0x480),
            ConvTask(1, 0x400000F0, 0x0010000A, 8, 6, 0x24, 0x1900, 0x190, 0x05050010, 0xA0, 0, 0x000A000A, 8, 0, 0x00050005, 0x0F, 0, 0x240, 5, 5, 0x000F000F, 0x0F, 0x00050005, 0x480),
            ConvTask(3, 0x400000F0, 0x0010000A, 8, 6, 0x24, 0x1900, 0x190, 0x05050010, 0xA0, 0, 0x000A000A, 8, 0x1900, 0x00050005, 0x0F, 0x480, 0x240, 5, 5, 0x000F000F, 0x0F, 0x00050005, 0x480),
            ConvTask(5, 0x200000B0, 0x00100006, 8, 6, 0x0C, 0x3200, 0x190, 0x05050020, 0x60, 0, 0x000A0006, 8, 0, 0x00010005, 0x1F, 0, 0x240, 5, 1, 0x001F001F, 0x1F, 0x00010005, 0x480),
            ConvTask(7, 0x200000B0, 0x00100006, 8, 6, 0x0C, 0x3200, 0x190, 0x05050020, 0x60, 0x40, 0x000A0006, 8, 0, 0x00010005, 0x1F, 0xC0, 0x240, 5, 1, 0x001F001F, 0x1F, 0x00010005, 0x480),
            ConvTask(9, 0x200000B0, 0x00100006, 8, 6, 0x0C, 0x3200, 0x190, 0x05050020, 0x60, 0x80, 0x000A0006, 8, 0, 0x00010005, 0x1F, 0x180, 0x240, 5, 1, 0x001F001F, 0x1F, 0x00010005, 0x480),
        ),
    ),
}


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


class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]


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


DRM_IOCTL_RKNPU_ACTION = _iowr("d", 0x40, ctypes.sizeof(rknpu_action))
DRM_IOCTL_RKNPU_MEM_CREATE = _iowr("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _iowr("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _iowr("d", 0x41, ctypes.sizeof(rknpu_submit))


def align_up(value, align):
    return ((value + align - 1) // align) * align


def task_amount(task: ConvTask):
    return 108 if task.idx == 0 else 104


def task_amounts(spec: CaseSpec):
    return [task_amount(task) for task in spec.conv_tasks]


def E(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | (addr & 0xFFFF)


def reg_value(qword):
    return (qword >> 48) & 0xFFFF, qword & 0xFFFF, (qword >> 16) & 0xFFFFFFFF


def mem_allocate(fd, size, flags):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc


def conv_body(spec: CaseSpec, task: ConvTask, input_dma, weight_dma, output_dma):
    regs = [
        E(R.DPU, R.DPU_S_POINTER, 0x0000000E),
        E(R.CNA, R.CNA_CONV_CON1, 0x60008120),
        E(R.CNA, R.CNA_CONV_CON2, task.conv_con2),
        E(R.CNA, R.CNA_CONV_CON3, 0x00000009),
        E(R.CNA, R.CNA_DATA_SIZE0, task.data_size0),
        E(R.CNA, R.CNA_DATA_SIZE1, task.data_size1),
        E(R.CNA, R.CNA_DATA_SIZE2, task.data_size2),
        E(R.CNA, R.CNA_DATA_SIZE3, task.data_size3),
        E(R.CNA, R.CNA_WEIGHT_SIZE0, task.weight_size0),
        E(R.CNA, R.CNA_WEIGHT_SIZE1, task.weight_size1),
        E(R.CNA, R.CNA_WEIGHT_SIZE2, task.weight_size2),
        E(R.CNA, R.CNA_CBUF_CON0, 0x000000B1),
        E(R.CNA, R.CNA_CBUF_CON1, task.cbuf_con1),
        E(R.CNA, R.CNA_CVT_CON0, 0x00000001),
        E(R.CNA, R.CNA_CVT_CON1, 0x00010000),
        E(R.CNA, R.CNA_CVT_CON2, 0x00010000),
        E(R.CNA, R.CNA_CVT_CON3, 0x00010000),
        E(R.CNA, R.CNA_CVT_CON4, 0x00010000),
        E(R.CNA, R.CNA_CVT_CON5, 0),
        E(R.CNA, R.CNA_CVT_CON6, 0),
        E(R.CNA, R.CNA_CVT_CON7, 0),
        E(R.CNA, R.CNA_FEATURE_DATA_ADDR, input_dma + task.input_offset),
        E(R.CNA, R.CNA_FEATURE_DATA_ADDR_HIGH, 0),
        E(R.CNA, R.CNA_DMA_CON0, 0x000F000F),
        E(R.CNA, R.CNA_DMA_CON1, spec.cna_dma_con1),
        E(R.CNA, R.CNA_DMA_CON2, spec.cna_dma_con2),
        E(R.CNA, R.CNA_FC_DATA_SIZE0, task.fc_data_size0),
        E(R.CNA, R.CNA_FC_DATA_SIZE1, task.fc_data_size1),
        E(R.CNA, R.CNA_DCOMP_CTRL, 0),
        E(R.CNA, R.CNA_RESERVED_1104, 0),
        E(R.CNA, R.CNA_DCOMP_ADDR0, weight_dma + task.weight_offset),
        E(R.CNA, R.CNA_DCOMP_ZERO, 0),
    ]
    regs.extend(E(R.CNA, R.CNA_DCOMP_PAD_0 + i * 4, 0) for i in range(15))
    regs.extend([
        E(R.CNA, R.CNA_DCOMP_PAD_15, 0x0000FFFF),
        E(R.CNA, R.CNA_DCOMP_PAD_16, 0),
        E(R.CORE, R.CORE_MISC_CFG, 0x00000200),
        E(R.CORE, R.CORE_DATAOUT_SIZE_0, task.core_dataout_size0),
        E(R.CORE, R.CORE_DATAOUT_SIZE_1, task.core_dataout_size1),
        E(R.CORE, R.CORE_RESERVED_301C, 0),
        E(R.CORE, R.CORE_RESERVED_3030, 0),
        E(R.DPU, R.DPU_FEATURE_MODE_CFG, 0x000001E4),
        E(R.DPU, R.DPU_DATA_FORMAT, 0x48000002),
        E(R.DPU, R.DPU_RESERVED_4014, 0),
        E(R.DPU, R.DPU_DST_BASE_ADDR, output_dma + task.output_offset),
        E(R.DPU, R.DPU_DST_SURF_STRIDE, task.dst_surf_stride),
        E(R.DPU, R.DPU_DATA_CUBE_WIDTH, task.cube_width),
        E(R.DPU, R.DPU_DATA_CUBE_HEIGHT, task.cube_height),
        E(R.DPU, R.DPU_DATA_CUBE_NOTCH, 0),
        E(R.DPU, R.DPU_DATA_CUBE_CHANNEL, task.cube_channel),
        E(R.DPU, R.DPU_BS_CFG, 0x00000053),
        E(R.DPU, R.DPU_BS_ALU_SRC_VALUE, 0),
        E(R.DPU, R.DPU_BS_MUL_SRC_VALUE, 0),
        E(R.DPU, R.DPU_BS_RELU_CFG, 0),
        E(R.DPU, R.DPU_BS_OW_CFG, spec.dpu_bs_ow_cfg),
        E(R.DPU, R.DPU_BS_OW_OP, 0),
        E(R.DPU, R.DPU_WDMA_SIZE_0, task.wdma_size0),
        E(R.DPU, R.DPU_WDMA_SIZE_1, task.wdma_size1),
        E(R.DPU, R.DPU_BN_CFG, 0x00000053),
        E(R.DPU, R.DPU_BN_ALU_SRC_VALUE, 0),
        E(R.DPU, R.DPU_BN_MUL_SRC_VALUE, 0),
        E(R.DPU, R.DPU_BN_RELU_CFG, 0),
        E(R.DPU, R.DPU_EW_CFG, 0x00000383),
        E(R.DPU, R.DPU_EW_ALU_SRC_VALUE, 0),
        E(R.DPU, R.DPU_EW_MUL_SRC_VALUE, 1),
        E(R.DPU, R.DPU_EW_TRUNCATE_VALUE, 0),
        E(R.DPU, R.DPU_OUT_CVT_OFFSET, 0),
        E(R.DPU, R.DPU_OUT_CVT_SCALE, 0x00010001),
        E(R.DPU, R.DPU_OUT_CVT_SHIFT, 0),
        E(R.DPU, R.DPU_DST_DMA_CFG, 0),
        E(R.DPU, R.DPU_DST_COMPRESSION_EN, 0),
        E(R.DPU, R.DPU_D_FEATURE_MODE_CFG, 0),
        E(R.DPU, R.DPU_D_DST_DMA_CFG, 0),
        E(R.DPU, R.DPU_D_DATA_FORMAT, 0),
        E(R.DPU, R.DPU_D_DST_BASE_ADDR, 0),
        E(R.DPU, R.DPU_D_DST_SURF_STRIDE, 0),
        E(R.DPU, R.DPU_D_DATA_CUBE_CHANNEL, 0),
        E(R.DPU, R.DPU_SURFACE_ADD, task.surface_add),
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
    ])
    if task.idx == 0:
        regs = [
            E(R.CNA, R.CNA_CBUF_CON0, 0x000000B1),
            E(R.CNA, R.CNA_RESERVED_1104, 0),
            E(R.CNA, R.CNA_DCOMP_CTRL, 0),
            E(R.CNA, R.CNA_CONV_CON1, 0x60008120),
            E(R.DPU, R.DPU_S_POINTER, 0x0000000E),
        ] + regs[1:]
    return regs


def pc_tail(task_idx, spec: CaseSpec, regcmd_dma):
    amounts = task_amounts(spec)
    if task_idx + 1 < len(amounts):
        return [
            E(R.PC_REG, R.PC_BASE_ADDRESS, (regcmd_dma + spec.task_offsets[task_idx + 1]) & 0xFFFFFFF0),
            E(R.PC_REG, R.PC_REGISTER_AMOUNTS, (amounts[task_idx + 1] + 2) // 2),
            E(R.VERSION, R.ZERO, 0),
            E(R.PC, R.OPERATION_ENABLE, TASK_ENABLE),
        ]
    return [
        0,
        E(R.PC_REG, R.PC_REGISTER_AMOUNTS, 0),
        E(R.VERSION, R.ZERO, 0),
        E(R.PC, R.OPERATION_ENABLE, TASK_ENABLE),
    ]


def build_task_regs(spec: CaseSpec, input_dma, weight_dma, output_dma):
    regs = []
    for idx, task in enumerate(spec.conv_tasks):
        body = conv_body(spec, task, input_dma, weight_dma, output_dma)
        amount = task_amount(task)
        if len(body) != amount:
            raise RuntimeError(f"task {idx} has {len(body)} regs, expected {amount}")
        regs.append(body)
    return regs


def regcmd_bytes(spec: CaseSpec):
    amounts = task_amounts(spec)
    return align_up(max(offset + (amounts[idx] + 4) * 8 for idx, offset in enumerate(spec.task_offsets)), 4096)


def write_regcmd(regcmd_map, spec: CaseSpec, regcmd_dma, task_regs):
    words = np.frombuffer(regcmd_map, dtype=np.uint64)
    words[:] = 0
    for task_idx, regs in enumerate(task_regs):
        base = spec.task_offsets[task_idx] // 8
        words[base:base + len(regs)] = regs
        words[base + len(regs):base + len(regs) + 4] = pc_tail(task_idx, spec, regcmd_dma)


def checked_output(output_map, spec: CaseSpec):
    parts = [
        np.frombuffer(output_map, dtype=np.float16, count=size // 2, offset=offset).copy()
        for offset, size in spec.check_ranges
    ]
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def fill_tasks(task_map, spec: CaseSpec, regcmd_dma):
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
    amounts = task_amounts(spec)
    for idx, offset in enumerate(spec.task_offsets):
        tasks[idx].flags = 0
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = TASK_ENABLE
        tasks[idx].int_mask = TASK_INT_MASK
        tasks[idx].int_clear = 0x1FFFF
        tasks[idx].int_status = 0
        tasks[idx].regcfg_amount = amounts[idx]
        tasks[idx].regcfg_offset = 0
        tasks[idx].regcmd_addr = regcmd_dma + offset


def submit(fd, task_obj_addr, task_count, mode, timeout, flags):
    official = mode == "official"
    req = rknpu_submit(
        flags=flags,
        timeout=timeout,
        task_start=0,
        task_number=task_count * 3 if official else task_count,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=0 if official else 1,
        fence_fd=-1,
    )
    req.subcore_task[0] = rknpu_subcore_task(0, task_count)
    req.subcore_task[1] = rknpu_subcore_task(0, task_count if official else 0)
    req.subcore_task[2] = rknpu_subcore_task(0, task_count if official else 0)
    req.subcore_task[3] = rknpu_subcore_task(0, 0)
    req.subcore_task[4] = rknpu_subcore_task(0, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, req)


def reset_npu(fd):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))


def print_dry_run(spec: CaseSpec, task_regs, regcmd_dma):
    amounts = task_amounts(spec)
    for task_idx, regs in enumerate(task_regs):
        print(f"task[{task_idx}] offset=0x{spec.task_offsets[task_idx]:04x} amount={amounts[task_idx]} enable=0x{TASK_ENABLE:x}")
        for i, qword in enumerate(regs + pc_tail(task_idx, spec, regcmd_dma)):
            target, addr, value = reg_value(qword)
            target_name = TARGET_NAMES.get(target, f"0x{target:04x}")
            reg_name = REG_NAMES.get(addr, f"REG_0x{addr:04x}")
            print(f"  [{i:3d}] {target_name}.{reg_name} = 0x{value:08x}")


def run_case(case_id, args):
    spec = CASES[case_id]
    task_count = len(spec.conv_tasks)
    dry_input, dry_weight, dry_output, dry_regcmd = 0xFFF10000, 0xFFF20000, 0xFFF30000, 0xFFF40000
    if args.dry:
        task_regs = build_task_regs(spec, dry_input, dry_weight, dry_output)
        print(f"case={case_id} {spec.name} regcmd_bytes=0x{regcmd_bytes(spec):x}")
        print_dry_run(spec, task_regs, dry_regcmd)
        return 0

    lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        try:
            flock(lock_fd, LOCK_EX | LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another NPU experiment holds {LOCK_PATH}; refusing parallel conv pcchain submit") from exc

        fd = os.open("/dev/dri/card1", os.O_RDWR)
        try:
            task_flags = RKNPU_MEM_OFFICIAL_TASK if args.alloc_mode == "official" else RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE
            tensor_flags = RKNPU_MEM_OFFICIAL_TENSOR if args.alloc_mode == "official" else RKNPU_MEM_NON_CACHEABLE
            task_map, task_mc = mem_allocate(fd, task_count * ctypes.sizeof(rknpu_task), task_flags)
            regcmd_map, regcmd_mc = mem_allocate(fd, regcmd_bytes(spec), tensor_flags)
            input_map, input_mc = mem_allocate(fd, align_up(spec.input_size, 4096), tensor_flags)
            weight_map, weight_mc = mem_allocate(fd, align_up(spec.weight_size, 4096), tensor_flags)
            internal_map, _internal_mc = mem_allocate(fd, align_up(spec.internal_size, 4096), tensor_flags)
            output_map, output_mc = mem_allocate(fd, align_up(spec.output_size, 4096), tensor_flags)

            for buf in (input_map, weight_map, internal_map):
                np.frombuffer(buf, dtype=np.uint8)[:] = 0
            np.frombuffer(output_map, dtype=np.float16)[:] = np.float16(7.0)

            task_regs = build_task_regs(spec, input_mc.dma_addr, weight_mc.dma_addr, output_mc.dma_addr)
            write_regcmd(regcmd_map, spec, regcmd_mc.dma_addr, task_regs)
            fill_tasks(task_map, spec, regcmd_mc.dma_addr)

            reset_npu(fd)
            ret = submit(fd, task_mc.obj_addr, task_count, args.mode, args.timeout, args.flags)
            out = checked_output(output_map, spec)
            max_abs = float(np.max(np.abs(out)))
            sentinel_left = int(np.count_nonzero(out == np.float16(7.0)))
            ok = ret == 0 and np.all(np.isfinite(out)) and max_abs == 0.0 and sentinel_left == 0
            print(f"case={case_id} mode={args.mode} tasks={task_count} submit ret={ret}")
            checked_size = sum(size for _offset, size in spec.check_ranges)
            print(f"output checked_size=0x{checked_size:x} max_abs={max_abs:.6f} sentinel_left={sentinel_left}")
            print(f"conv pcchain decoded case {case_id} {'PASS' if ok else 'FAIL'}")
            return 0 if ok else 1
        finally:
            os.close(fd)
    finally:
        flock(lock_fd, LOCK_UN)
        os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description="Standalone decoded RKNN conv PC-chain replay. No rawbuf import or runtime dump files.")
    parser.add_argument("--case", type=int, choices=sorted(CASES), default=None, help="run one case; omit to run all cases")
    parser.add_argument("--mode", choices=("core0", "official"), default="core0")
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--alloc-mode", choices=("raw", "official"), default="raw")
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    case_ids = [args.case] if args.case is not None else sorted(CASES)
    rc = 0
    for idx, case_id in enumerate(case_ids):
        if idx:
            print()
        rc |= run_case(case_id, args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
