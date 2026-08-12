"""Probe every hardware-listed DPU elementwise operation with integer data.

Observed RK3588 integer behavior:
  * MAX, MIN, ADD, MINUS, ABS, and NEG work for INT8, INT16, and INT32.
  * MUL works for INT8 and INT16.  With INT32, its second operand is signed
    INT16 even though ERDMA is configured and packed as INT32.
  * DIV, FLOOR, and CEIL use floating-point semantics and are not integer ops.
  * The captured FP16 comparison pipeline can write its 0/1 mask as INT16.
  * INT16 EW results can be regrouped and written externally as INT32.
  * EW register operands work without ERDMA: FP16 constants use FP32 register
    encoding, while INT16 constants use the low signed 16 bits.  This removes
    constant-buffer materialization and its operand DMA read.
  * The BS, BN, and EW ALUs can each add a configured FP32 constant in one
    FP16 DPU task, proving that three scalar elementwise stages can be fused.
  * Configured FP16 multipliers also work.  BS orders ALU then MUL, while BN
    orders MUL then ALU; together with EW, five scalar ops fit in one task.
  * BS and BN each implement ReLU-X and PReLU, leaving EW free for another op.
  * BRDMA and NRDMA feed FP32 per-channel tensors into BS and BN.  Together
    with the FP16 main and ERDMA inputs, one task can sum four tensors.
    Their separate MUL lanes also consume one FP16 multiplier per element.
  * Configured MINUS in both BS and BN computes main minus the FP32 operand.
  * EW/ERDMA data mode 0 broadcasts one FP16 operand per channel across pixels.
  * The two 513-entry DPU LUTs implement SiLU with 1.23e-4 measured max error.

The checks below prove supported behavior and deliberately prove the three
unsupported integer ALU encodings by showing that they differ from ordinary
integer semantics.  All register commands are decoded; there is no hex blob.
"""

from fcntl import ioctl
import ctypes
import glob
import math
import mmap
import os
import sys
import time

import numpy as np


class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    TARGET_DPU = 0x1001       # DPU (Elementwise/DPU unit)
    TARGET_RDMA = 0x2001      # RDMA (Read DMA for inputs/weights)
    TARGET_PC = 0x0081        # PC (Program Control / operation enable)
    TARGET_PC_REG = 0x0101    # PC chain registers
    TARGET_VERSION = 0x0041

    # --- PC (0x0000) ---
    OPERATION_ENABLE = 0x0008
    PC_BASE_ADDRESS = 0x0010
    PC_REGISTER_AMOUNTS = 0x0014

    # --- DPU (0x4000) ---
    S_POINTER = 0x4004
    FEATURE_MODE_CFG = 0x400C
    DATA_FORMAT = 0x4010
    DST_BASE_ADDR = 0x4020
    DST_SURF_STRIDE = 0x4024
    DATA_CUBE_WIDTH = 0x4030
    DATA_CUBE_HEIGHT = 0x4034
    DATA_CUBE_NOTCH = 0x4038
    DATA_CUBE_CHANNEL = 0x403C
    BS_CFG = 0x4040
    BS_ALU_CFG = 0x4044
    BS_MUL_CFG = 0x4048
    BS_RELUX_CMP_VALUE = 0x404C
    BS_OW_CFG = 0x4050
    WDMA_SIZE_0 = 0x4058
    WDMA_SIZE_1 = 0x405C
    BN_CFG = 0x4060
    BN_ALU_CFG = 0x4064
    BN_MUL_CFG = 0x4068
    BN_RELUX_CMP_VALUE = 0x406C
    EW_CFG = 0x4070
    EW_CVT_OFFSET_VALUE = 0x4074
    EW_CVT_SCALE_VALUE = 0x4078
    EW_OP_VALUE_0 = 0x4090
    OUT_CVT_OFFSET = 0x4080
    OUT_CVT_SCALE = 0x4084
    OUT_CVT_SHIFT = 0x4088
    SURFACE_ADD = 0x40C0
    LUT_ACCESS_CFG = 0x4100
    LUT_ACCESS_DATA = 0x4104
    LUT_CFG = 0x4108
    LUT_INFO = 0x410C
    LUT_LE_START = 0x4110
    LUT_LO_END = 0x411C
    LUT_LO_SLOPE_SCALE = 0x4128
    LUT_LO_SLOPE_SHIFT = 0x412C

    # --- DPU RDMA (0x5000) ---
    RDMA_S_POINTER = 0x5004
    RDMA_DATA_CUBE_WIDTH = 0x500C
    RDMA_DATA_CUBE_HEIGHT = 0x5010
    RDMA_DATA_CUBE_CHANNEL = 0x5014
    RDMA_SRC_BASE_ADDR = 0x5018
    RDMA_BRDMA_CFG = 0x501C
    RDMA_BS_BASE_ADDR = 0x5020
    RDMA_NRDMA_CFG = 0x5028
    RDMA_BN_BASE_ADDR = 0x502C
    RDMA_ERDMA_CFG = 0x5034
    RDMA_EW_BASE_ADDR = 0x5038
    RDMA_EW_SURF_STRIDE = 0x5040
    RDMA_FEATURE_MODE_CFG = 0x5044
    RDMA_SRC_DMA_CFG = 0x5048
    RDMA_SURF_NOTCH = 0x504C
    RDMA_WEIGHT = 0x5068
    RDMA_EW_SURF_NOTCH = 0x506C


PRECISION_INT8 = 0
PRECISION_INT16 = 1
PRECISION_FP16 = 2
PRECISION_INT32 = 4

ERDMA_SIZE_INT8 = 1
ERDMA_SIZE_INT16 = 2
ERDMA_SIZE_INT32 = 3

EW_DATA_MODE_PER_PIXEL = 1

EW_ALU_MAX = 0
EW_ALU_MIN = 1
EW_ALU_ADD = 2
EW_ALU_DIV = 3
EW_ALU_MINUS = 4
EW_ALU_ABS = 5
EW_ALU_NEG = 6
EW_ALU_FLOOR = 7
EW_ALU_CEIL = 8

EW_OP_TYPE_ALU = 0
EW_OP_TYPE_MUL = 1

OPERATIONS = {
    "MAX": {"alu_algorithm": EW_ALU_MAX, "integer_supported": True},
    "MIN": {"alu_algorithm": EW_ALU_MIN, "integer_supported": True},
    "ADD": {"alu_algorithm": EW_ALU_ADD, "integer_supported": True},
    "DIV": {"alu_algorithm": EW_ALU_DIV, "integer_supported": False},
    "MINUS": {"alu_algorithm": EW_ALU_MINUS, "integer_supported": True},
    "ABS": {"alu_algorithm": EW_ALU_ABS, "integer_supported": True},
    "NEG": {"alu_algorithm": EW_ALU_NEG, "integer_supported": True},
    "FLOOR": {"alu_algorithm": EW_ALU_FLOOR, "integer_supported": False},
    "CEIL": {"alu_algorithm": EW_ALU_CEIL, "integer_supported": False},
    "MUL": {"op_type": EW_OP_TYPE_MUL, "integer_supported": True},
}


class drm_rocket_create_bo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("handle", ctypes.c_uint32),
        ("dma_address", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
    ]


class drm_rocket_prep_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("timeout_ns", ctypes.c_int64),
    ]


class drm_rocket_fini_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class drm_rocket_task(ctypes.Structure):
    _fields_ = [
        ("regcmd", ctypes.c_uint32),
        ("regcmd_count", ctypes.c_uint32),
    ]


class drm_rocket_job(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_uint64),
        ("in_bo_handles", ctypes.c_uint64),
        ("out_bo_handles", ctypes.c_uint64),
        ("task_count", ctypes.c_uint32),
        ("task_struct_size", ctypes.c_uint32),
        ("in_bo_handle_count", ctypes.c_uint32),
        ("out_bo_handle_count", ctypes.c_uint32),
    ]


class drm_rocket_submit(ctypes.Structure):
    _fields_ = [
        ("jobs", ctypes.c_uint64),
        ("job_count", ctypes.c_uint32),
        ("job_struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
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


class RocketBO:
    __slots__ = ("handle", "size", "dma_address", "offset")

    def __init__(self, handle, size, dma_address, offset):
        self.handle = int(handle)
        self.size = int(size)
        self.dma_address = int(dma_address)
        self.offset = int(offset)

    @property
    def dma_addr(self):
        return self.dma_address


def _IOW(type_, nr, size):
    return (1 << 30) | (ord(type_) << 8) | nr | (size << 16)


def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)


DRM_IOCTL_ROCKET_CREATE_BO = _IOWR("d", 0x40, ctypes.sizeof(drm_rocket_create_bo))
DRM_IOCTL_ROCKET_SUBMIT = _IOW("d", 0x41, ctypes.sizeof(drm_rocket_submit))
DRM_IOCTL_ROCKET_PREP_BO = _IOW("d", 0x42, ctypes.sizeof(drm_rocket_prep_bo))
DRM_IOCTL_ROCKET_FINI_BO = _IOW("d", 0x43, ctypes.sizeof(drm_rocket_fini_bo))


def open_rocket_device():
    path = os.environ.get("ROCKET_DEVICE")
    if path:
        return os.open(path, os.O_RDWR)
    candidates = (
        sorted(glob.glob("/dev/accel/accel*"))
        + sorted(glob.glob("/dev/dri/renderD*"))
        + sorted(glob.glob("/dev/dri/card*"))
    )
    for candidate in candidates:
        try:
            return os.open(candidate, os.O_RDWR)
        except OSError:
            pass
    raise FileNotFoundError("No Rocket device found")


def mem_allocate(fd, size):
    bo = drm_rocket_create_bo(size=size)
    ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, bo)
    buf = mmap.mmap(
        fd,
        bo.size,
        mmap.MAP_SHARED,
        mmap.PROT_READ | mmap.PROT_WRITE,
        offset=bo.offset,
    )
    return buf, RocketBO(bo.handle, bo.size, bo.dma_address, bo.offset)


def rocket_submit(fd, vendor_tasks, in_bos, out_bos):
    # Keep the known-good one-task Rocket submission ABI unchanged.
    rocket_tasks = (drm_rocket_task * 1)()
    rocket_tasks[0].regcmd = int(vendor_tasks[0].regcmd_addr) & 0xFFFFFFFF
    rocket_tasks[0].regcmd_count = int(vendor_tasks[0].regcfg_amount)
    in_handles = (ctypes.c_uint32 * len(in_bos))(*(bo.handle for bo in in_bos))
    out_handles = (ctypes.c_uint32 * len(out_bos))(*(bo.handle for bo in out_bos))
    job = drm_rocket_job(
        tasks=ctypes.addressof(rocket_tasks),
        in_bo_handles=ctypes.addressof(in_handles),
        out_bo_handles=ctypes.addressof(out_handles),
        task_count=1,
        task_struct_size=ctypes.sizeof(drm_rocket_task),
        in_bo_handle_count=len(in_bos),
        out_bo_handle_count=len(out_bos),
    )
    jobs = (drm_rocket_job * 1)(job)
    submit = drm_rocket_submit(
        jobs=ctypes.addressof(jobs),
        job_count=1,
        job_struct_size=ctypes.sizeof(drm_rocket_job),
    )
    return ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, submit)


def emit(target, offset, value):
    """Encode one decoded register write into a Rocket regcmd QWORD."""
    return (target << 48) | ((int(value) & 0xFFFFFFFF) << 16) | offset


def dpu_feature_mode():
    # burst_len=15, output_mode=2, flying_mode=1.
    return (15 << 5) | (2 << 1) | 1


def dpu_data_format(precision):
    # OUT_PRECISION[31:29], IN_PRECISION[28:26], PROC_PRECISION[2:0].
    return (precision << 29) | (precision << 26) | precision


def dpu_cube_channel(channels_minus_one):
    # ORIG_CHANNEL[28:16] and CHANNEL[12:0].
    return (channels_minus_one << 16) | channels_minus_one


def dpu_ew_cfg(element_size, operation, converter_bypass):
    # Per-pixel ERDMA, selected element width, RELU/LUT bypass, operand from ERDMA.
    op = OPERATIONS[operation]
    value = (
        (EW_DATA_MODE_PER_PIXEL << 28)
        | (element_size << 22)
        | (1 << 9)
        | (1 << 7)
        | (1 << 6)
    )
    if op.get("op_type", EW_OP_TYPE_ALU) == EW_OP_TYPE_MUL:
        value |= EW_OP_TYPE_MUL << 2
    else:
        value |= op["alu_algorithm"] << 16
    if operation in ("MAX", "MIN"):
        value |= 1 << 21  # EW_EQUAL_EN, matching RKNN min/max captures.
    if converter_bypass:
        value |= 1 << 8
    return value


def rdma_erdma_cfg(element_size):
    # Per-pixel ERDMA mode and ERDMA_DATA_SIZE[3:2].
    return (EW_DATA_MODE_PER_PIXEL << 30) | (element_size << 2)


def rdma_feature_mode(precision):
    # IN_PRECISION[17:15], burst_len=15, PROC_PRECISION[7:5], flying_mode=1.
    return (precision << 15) | (15 << 11) | (precision << 5) | 1


def make_regcmds(
    output_dma,
    input_dma,
    ew_dma,
    precision,
    element_size,
    width,
    channels,
    operation,
):
    channels_minus_one = channels - 1
    converter_bypass = precision == PRECISION_INT32
    commands = [
        emit(reg.TARGET_DPU, reg.S_POINTER, 0xE),
        emit(reg.TARGET_RDMA, reg.RDMA_S_POINTER, 0xE),
        # Bit 0 is *_DISABLE: one prevents unused auxiliary clients from
        # fetching stale BRDMA/NRDMA addresses left by an earlier task.
        emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 1),
        emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
        emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(precision)),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(channels_minus_one)),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, width - 1),
        emit(
            reg.TARGET_DPU,
            reg.EW_CFG,
            dpu_ew_cfg(element_size, operation, converter_bypass),
        ),
        # Integer EW conversion needs a scale of 1. INT32 bypasses the converter.
        emit(reg.TARGET_DPU, reg.EW_CVT_SCALE_VALUE, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, width - 1),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, channels_minus_one),
        emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, rdma_erdma_cfg(element_size)),
        emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR, ew_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_FEATURE_MODE_CFG, rdma_feature_mode(precision)),
        emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
        emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        emit(reg.TARGET_VERSION, 0, 0),
        # Enable RDMA and DPU through PC; enable_mask remains the known-safe 0x18.
        emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
    ]
    return commands


def run_operation(
    fd,
    dtype_name,
    operation,
    dtype,
    precision,
    element_size,
    channels,
    a_values,
    b_values,
):
    task_map, task_bo = mem_allocate(fd, 1024)
    regcmd_map, regcmd_bo = mem_allocate(fd, 1024)
    input_map, input_bo = mem_allocate(fd, 4 * 1024 * 1024)
    ew_map, ew_bo = mem_allocate(fd, 4 * 1024 * 1024)
    output_map, output_bo = mem_allocate(fd, 4 * 1024 * 1024)

    a = np.asarray(a_values, dtype=dtype)
    b = np.asarray(b_values, dtype=dtype)
    if len(a) % channels:
        raise ValueError("element count must be divisible by the DPU channel count")
    width = len(a) // channels

    input_map[: a.nbytes] = a.tobytes()
    ew_map[: b.nbytes] = b.tobytes()
    output_map[: a.nbytes] = bytes(a.nbytes)

    commands = make_regcmds(
        output_bo.dma_addr,
        input_bo.dma_addr,
        ew_bo.dma_addr,
        precision,
        element_size,
        width,
        channels,
        operation,
    )
    regcmds = (ctypes.c_uint64 * (regcmd_bo.size // 8)).from_buffer(regcmd_map)
    for index, command in enumerate(commands):
        regcmds[index] = command

    tasks = ctypes.cast(
        ctypes.addressof(ctypes.c_char.from_buffer(task_map)),
        ctypes.POINTER(struct_rknpu_task),
    )
    tasks[0].flags = 0
    tasks[0].op_idx = 4
    tasks[0].enable_mask = 0x18
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1FFFF
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(commands)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = regcmd_bo.dma_addr

    for bo in (regcmd_bo, input_bo, ew_bo, output_bo):
        ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))
    submit_ret = rocket_submit(
        fd,
        tasks,
        in_bos=[regcmd_bo, input_bo, ew_bo],
        out_bos=[output_bo],
    )
    ioctl(
        fd,
        DRM_IOCTL_ROCKET_PREP_BO,
        drm_rocket_prep_bo(
            handle=output_bo.handle,
            timeout_ns=time.monotonic_ns() + 6_000_000_000,
        ),
    )
    result = np.frombuffer(output_map, dtype=dtype, count=len(a)).copy()
    expected = integer_expected(operation, a, b, dtype)
    label = f"{dtype_name}/{operation}"

    if operation == "MUL" and precision == PRECISION_INT32:
        # The ALU takes full INT32 ERDMA data, but the separate MUL input is INT16.
        ew_int16 = (b.astype(np.int64) & 0xFFFF).astype(np.uint16).view(np.int16)
        limited_expected = saturate(a.astype(np.int64) * ew_int16, dtype)
        limitation_proved = (
            submit_ret == 0
            and np.array_equal(result, limited_expected)
            and not np.array_equal(result, expected)
        )
        status = "PASS (EW MUL OPERAND IS SIGNED INT16)" if limitation_proved else "FAIL"
        print(f"{label:11s} NPU={result}")
        print(f"            full-int32 expected={expected}")
        print(f"            int16-EW expected  ={limited_expected} {status}")
        return limitation_proved

    if operation == "MUL" and precision == PRECISION_INT32:
        b_int16 = (b.astype(np.int64) & 0xFFFF).astype(np.uint16).view(np.int16)
        limited_expected = saturate(a.astype(np.int64) * b_int16, dtype)
        passed = submit_ret == 0 and np.array_equal(result, limited_expected)
        result_kind = "SIGNED_INT16_OPERAND" if passed else "MISMATCH"
    elif not OPERATIONS[operation]["integer_supported"]:
        unsupported_proved = submit_ret == 0 and not np.array_equal(result, expected)
        status = "PASS (NOT AN INTEGER OP)" if unsupported_proved else "FAIL"
        print(f"{label:11s} NPU={result}")
        print(f"            integer expected={expected} {status}")
        return unsupported_proved

    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(f"{label:11s} NPU={result} expected={expected} {'PASS' if passed else 'FAIL'}")
    return passed


def pack_c1wc2(values, width, channels, dtype):
    """Pack logical WC data into contiguous 16-byte DPU channel surfaces."""
    logical = np.asarray(values, dtype=dtype).reshape(width, channels)
    atom_channels = 16 // np.dtype(dtype).itemsize
    surfaces = (channels + atom_channels - 1) // atom_channels
    packed = np.zeros((surfaces, width, atom_channels), dtype=dtype)
    for channel in range(channels):
        packed[channel // atom_channels, :, channel % atom_channels] = logical[:, channel]
    return packed.reshape(-1)


def unpack_c1wc2(values, width, channels, dtype):
    """Unpack contiguous 16-byte DPU channel surfaces into logical WC data."""
    atom_channels = 16 // np.dtype(dtype).itemsize
    surfaces = (channels + atom_channels - 1) // atom_channels
    packed = np.asarray(values, dtype=dtype).reshape(surfaces, width, atom_channels)
    logical = np.empty((width, channels), dtype=dtype)
    for channel in range(channels):
        logical[:, channel] = packed[
            channel // atom_channels, :, channel % atom_channels
        ]
    return logical


def make_layout_regcmds(
    output_dma,
    input_dma,
    ew_dma,
    precision,
    element_size,
    width,
    channels,
    operation,
):
    """Build a complete multi-surface/tail DPU EW stream."""
    channels_minus_one = channels - 1
    surface_stride = width << 4
    converter_bypass = precision == PRECISION_INT32
    return [
        emit(reg.TARGET_DPU, reg.S_POINTER, 0xE),
        emit(reg.TARGET_RDMA, reg.RDMA_S_POINTER, 0xE),
        emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 1),
        emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
        emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(precision)),
        emit(reg.TARGET_DPU, reg.DST_SURF_STRIDE, surface_stride),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, width - 1),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
        emit(
            reg.TARGET_DPU,
            reg.DATA_CUBE_CHANNEL,
            dpu_cube_channel(channels_minus_one),
        ),
        emit(reg.TARGET_DPU, reg.BS_CFG, 0x53),
        emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 0),
        emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, channels_minus_one),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, width - 1),
        emit(reg.TARGET_DPU, reg.BN_CFG, 0x53),
        emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0),
        emit(
            reg.TARGET_DPU,
            reg.EW_CFG,
            dpu_ew_cfg(element_size, operation, converter_bypass),
        ),
        emit(reg.TARGET_DPU, reg.EW_CVT_SCALE_VALUE, 1),
        emit(reg.TARGET_DPU, reg.SURFACE_ADD, surface_stride),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, width - 1),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, channels_minus_one),
        emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, rdma_erdma_cfg(element_size)),
        emit(reg.TARGET_RDMA, reg.RDMA_EW_SURF_STRIDE, surface_stride),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_DMA_CFG, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_SURF_NOTCH, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_EW_SURF_NOTCH, 0),
        emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR, ew_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_FEATURE_MODE_CFG, rdma_feature_mode(precision)),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_WEIGHT,
            (1 << 24) | (1 << 16) | (1 << 8) | 1,
        ),
        emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
        emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        emit(reg.TARGET_VERSION, 0, 0),
        emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
    ]


def run_integer_layout(fd, dtype_name, operation, width, channels, edge=False):
    """Exercise one integer operation across explicit DPU channel surfaces."""
    config = DTYPES[dtype_name]
    dtype = config["dtype"]
    count = width * channels
    limits = np.iinfo(dtype)
    lane = np.arange(count, dtype=np.int64).reshape(width, channels)
    if edge:
        a_pattern = np.asarray(
            [limits.max, limits.min, limits.max, limits.min, -1, 0, 1, limits.max // 2],
            dtype=np.int64,
        )
        b_pattern = np.asarray(
            [1, -1, limits.max, limits.min, limits.min, limits.max, -1, 3],
            dtype=np.int64,
        )
        a = a_pattern[(lane % len(a_pattern))].astype(dtype)
        b = b_pattern[(lane % len(b_pattern))].astype(dtype)
    else:
        a = ((lane * 17 + 3) % 101 - 50).astype(dtype)
        b = ((lane * 11 + 7) % 31 - 15).astype(dtype)
        if operation == "MUL":
            a = ((lane % 17) - 8).astype(dtype)
            b = ((lane * 3) % 9 - 4).astype(dtype)
        elif operation == "DIV":
            b[b == 0] = 1
        elif operation in ("ABS", "NEG"):
            b.fill(1)
    a = np.clip(a, limits.min, limits.max).astype(dtype)
    b = np.clip(b, limits.min, limits.max).astype(dtype)
    expected = integer_expected(operation, a, b, dtype)
    packed_a = pack_c1wc2(a, width, channels, dtype)
    packed_b = pack_c1wc2(b, width, channels, dtype)

    def commands(output_bo, input_bos):
        return make_layout_regcmds(
            output_bo.dma_addr,
            input_bos[0].dma_addr,
            input_bos[1].dma_addr,
            config["precision"],
            config["element_size"],
            width,
            channels,
            operation,
        )

    submit_ret, packed_result = prepare_and_run(
        fd,
        commands,
        [packed_a, packed_b],
        dtype,
        len(packed_a),
    )
    result = unpack_c1wc2(packed_result, width, channels, dtype)
    captured_limitation = None
    if dtype_name == "INT32" and operation == "MUL":
        # The INT32 main cube remains full width, but the EW MUL operand is
        # decoded as signed INT16.  This is visible at both ordinary values
        # above 0xffff and at the saturation boundaries exercised here.
        b_int16 = (b.astype(np.int64) & 0xFFFF).astype(np.uint16).view(np.int16)
        expected = saturate(a.astype(np.int64) * b_int16, dtype)
        captured_limitation = "SIGNED_INT16_EW_OPERAND"
    elif edge and dtype_name == "INT32" and operation in ("ABS", "NEG"):
        # INT32_MIN has no positive signed counterpart.  The unary datapath
        # preserves it (two's-complement wrap) rather than saturating to MAX.
        expected = integer_expected(operation, a, b, dtype)
        expected[a == limits.min] = limits.min
        captured_limitation = "INT32_MIN_WRAP"
    elif edge and dtype_name == "INT32" and operation == "MINUS":
        # With an INT32_MIN EW operand the captured subtract datapath emits
        # INT32_MIN for both tested main-input signs.  Keep this exact edge
        # vector as a regression fact; do not generalize it as saturation.
        expected = integer_expected(operation, a, b, dtype)
        expected[b == limits.min] = limits.min
        captured_limitation = "INT32_MIN_EW_SENTINEL"

    if not OPERATIONS[operation]["integer_supported"]:
        passed = submit_ret == 0 and not np.array_equal(result, expected)
        result_kind = "NOT_INTEGER" if passed else "UNRESOLVED"
    else:
        passed = submit_ret == 0 and np.array_equal(result, expected)
        if passed and captured_limitation is not None:
            result_kind = captured_limitation
        else:
            result_kind = "EXACT" if passed else "MISMATCH"
    max_diff = int(
        np.max(np.abs(result.astype(np.int64) - expected.astype(np.int64)))
    )
    print(
        f"layout {dtype_name}/{operation} profile={'edge' if edge else 'regular'} "
        f"width={width} channels={channels} "
        f"submit={submit_ret} max_diff={max_diff} result={result_kind} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        print(f"NPU={result.reshape(-1)}")
        print(f"expected={expected.reshape(-1)}")
    return passed


def validate_integer_layout_streams():
    """Validate every planned layout stream without opening the NPU device."""
    widths = (1, 2, 3)
    channels = (3, 7, 8, 9, 15, 16, 17, 31, 32, 33)
    checked = 0
    for dtype_name, config in DTYPES.items():
        for width in widths:
            for channel_count in channels:
                commands = make_layout_regcmds(
                    0x20000000,
                    0x10000000,
                    0x10010000,
                    config["precision"],
                    config["element_size"],
                    width,
                    channel_count,
                    "ADD",
                )
                decoded = {
                    ((command >> 48) & 0xFFFF, command & 0xFFFF):
                    (command >> 16) & 0xFFFFFFFF
                    for command in commands
                }
                stride = width << 4
                assert decoded[(reg.TARGET_DPU, reg.DST_SURF_STRIDE)] == stride
                assert decoded[(reg.TARGET_DPU, reg.SURFACE_ADD)] == stride
                assert decoded[(reg.TARGET_RDMA, reg.RDMA_EW_SURF_STRIDE)] == stride
                assert decoded[(reg.TARGET_DPU, reg.WDMA_SIZE_0)] == channel_count - 1
                assert decoded[(reg.TARGET_DPU, reg.WDMA_SIZE_1)] == width - 1
                checked += 1
    print(f"offline integer-layout validation PASS ({checked} streams)")
    return True


def prepare_and_run(
    fd,
    commands,
    input_payloads,
    output_dtype,
    output_count,
):
    """Run one decoded-register boundary probe with the known-safe Rocket ABI."""
    task_map, task_bo = mem_allocate(fd, 1024)
    input_allocations = [mem_allocate(fd, 4 * 1024 * 1024) for _ in input_payloads]
    output_map, output_bo = mem_allocate(fd, 4 * 1024 * 1024)

    for (input_map, _), payload in zip(input_allocations, input_payloads):
        input_map[: payload.nbytes] = payload.tobytes()
    output_map[: output_count * np.dtype(output_dtype).itemsize] = bytes(
        output_count * np.dtype(output_dtype).itemsize
    )

    input_bos = [allocation[1] for allocation in input_allocations]
    resolved_commands = [
        # Match the production renderer and re-arm both register groups on
        # every task while keeping one persistent Rocket file context.
        emit(reg.TARGET_DPU, reg.S_POINTER, 0xE),
        emit(reg.TARGET_RDMA, reg.RDMA_S_POINTER, 0xE),
        emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 1),
        *commands(output_bo, input_bos),
    ]
    # Full LUT programming needs more than the 1 KiB used by ordinary streams.
    regcmd_size = max(1024, ((len(resolved_commands) * 8 + 4095) // 4096) * 4096)
    regcmd_map, regcmd_bo = mem_allocate(fd, regcmd_size)
    regcmds = (ctypes.c_uint64 * (regcmd_bo.size // 8)).from_buffer(regcmd_map)
    for index, command in enumerate(resolved_commands):
        regcmds[index] = command

    tasks = ctypes.cast(
        ctypes.addressof(ctypes.c_char.from_buffer(task_map)),
        ctypes.POINTER(struct_rknpu_task),
    )
    tasks[0].flags = 0
    tasks[0].op_idx = 4
    tasks[0].enable_mask = 0x18
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1FFFF
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(resolved_commands)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = regcmd_bo.dma_addr

    for bo in (regcmd_bo, *input_bos, output_bo):
        ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))
    submit_ret = rocket_submit(
        fd,
        tasks,
        in_bos=[regcmd_bo, *input_bos],
        out_bos=[output_bo],
    )
    ioctl(
        fd,
        DRM_IOCTL_ROCKET_PREP_BO,
        drm_rocket_prep_bo(
            handle=output_bo.handle,
            timeout_ns=time.monotonic_ns() + 6_000_000_000,
        ),
    )
    result = np.frombuffer(output_map, dtype=output_dtype, count=output_count).copy()
    return submit_ret, result


def run_fp16_compare_to_int16(fd):
    """Prove that the captured FP16 comparison mask can cross into INT16."""
    values = np.asarray(
        [-2.0, -0.5, -0.0, 0.0, 0.25, 0.5, 1.0, 2.0],
        dtype=np.float16,
    )
    expected = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)

    def commands(output_bo, input_bos):
        input_bo = input_bos[0]
        compare_ew_bypass = (
            (EW_DATA_MODE_PER_PIXEL << 28)
            | (ERDMA_SIZE_INT16 << 22)
            | (1 << 9)
            | (1 << 7)
            | (1 << 6)
            | 1
        )
        return [
            emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
            emit(
                reg.TARGET_DPU,
                reg.DATA_FORMAT,
                (PRECISION_INT16 << 29)
                | (PRECISION_FP16 << 26)
                | PRECISION_FP16,
            ),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
            # Captured positive-difference comparison: positive -> 1, else 0.
            emit(reg.TARGET_DPU, reg.BS_CFG, (1 << 18) | (1 << 6)),
            emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0x33800000),  # FP32 0.25.
            emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 0x40000000),  # FP32 2.0.
            emit(reg.TARGET_DPU, reg.BS_OW_CFG, 1 << 1),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
            emit(reg.TARGET_DPU, reg.BN_CFG, (1 << 18) | (2 << 6) | (1 << 1)),
            emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0x7C000000),  # FP16 +inf lane.
            emit(reg.TARGET_DPU, reg.BN_RELUX_CMP_VALUE, 0x3F800000),  # FP32 1.0.
            emit(reg.TARGET_DPU, reg.EW_CFG, compare_ew_bypass),
            emit(reg.TARGET_DPU, reg.OUT_CVT_OFFSET, 0),
            emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, 1),
            emit(reg.TARGET_DPU, reg.OUT_CVT_SHIFT, 0),
            emit(reg.TARGET_DPU, reg.SURFACE_ADD, 4 << 4),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
            emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, rdma_erdma_cfg(ERDMA_SIZE_INT16)),
            emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR, input_bo.dma_addr),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_FEATURE_MODE_CFG,
                rdma_feature_mode(PRECISION_FP16) | (1 << 3),
            ),
            emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
            emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            emit(reg.TARGET_VERSION, 0, 0),
            emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
        ]

    submit_ret, result = prepare_and_run(fd, commands, [values], np.int16, 8)
    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(f"FP16 CMP -> INT16 NPU={result} expected={expected} {'PASS' if passed else 'FAIL'}")
    return passed


def run_int16_add_to_int32(fd):
    """Prove INT16 EW arithmetic followed by external INT32 writeback."""
    a = np.asarray([-30000, -1200, -7, -1, 0, 1, 1200, 30000], dtype=np.int16)
    b = np.asarray([1000, -30, 2, -1, 1, 2, -30, 1000], dtype=np.int16)
    expected = a.astype(np.int32) + b.astype(np.int32)

    def commands(output_bo, input_bos):
        input_bo, ew_bo = input_bos
        return [
            emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
            emit(
                reg.TARGET_DPU,
                reg.DATA_FORMAT,
                (PRECISION_INT32 << 29)
                | (PRECISION_INT16 << 26)
                | PRECISION_INT16,
            ),
            # The captured 16-to-32 layout emits two four-lane surfaces.
            emit(reg.TARGET_DPU, reg.DST_SURF_STRIDE, 1 << 4),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
            emit(reg.TARGET_DPU, reg.BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
            # SIZE_E_0/1/2=1 is required to regroup all eight INT16 lanes.
            emit(
                reg.TARGET_DPU,
                reg.BS_OW_CFG,
                (1 << 8) | (1 << 5) | (1 << 2) | (1 << 1),
            ),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
            emit(reg.TARGET_DPU, reg.BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
            emit(
                reg.TARGET_DPU,
                reg.EW_CFG,
                dpu_ew_cfg(ERDMA_SIZE_INT16, "ADD", converter_bypass=False)
                | (3 << 30),  # Select the alternate converter type and round mode.
            ),
            emit(reg.TARGET_DPU, reg.EW_CVT_OFFSET_VALUE, 0),
            emit(reg.TARGET_DPU, reg.EW_CVT_SCALE_VALUE, 1),
            emit(reg.TARGET_DPU, reg.OUT_CVT_OFFSET, 0),
            emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, 1),
            emit(reg.TARGET_DPU, reg.OUT_CVT_SHIFT, 0),
            emit(reg.TARGET_DPU, reg.SURFACE_ADD, 2 << 4),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
            emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, rdma_erdma_cfg(ERDMA_SIZE_INT16)),
            emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR, ew_bo.dma_addr),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_FEATURE_MODE_CFG,
                rdma_feature_mode(PRECISION_INT16),
            ),
            emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
            emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            emit(reg.TARGET_VERSION, 0, 0),
            emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
        ]

    submit_ret, result = prepare_and_run(fd, commands, [a, b], np.int32, 8)
    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(f"INT16 ADD -> INT32 NPU={result} expected={expected} {'PASS' if passed else 'FAIL'}")
    return passed


def run_fp16_add_register_operand(fd):
    """Prove a configured FP16 scalar operand with ERDMA completely disabled."""
    values = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    expected = (values + np.float16(2.0)).astype(np.float16)

    def commands(output_bo, input_bos):
        input_bo = input_bos[0]
        # EW_OP_SRC=0 selects EW_OP_VALUE_n.  FP16 processing consumes its
        # configured operand as FP32, hence 2.0 is encoded as 0x40000000.
        ew_register_add = (
            (EW_DATA_MODE_PER_PIXEL << 28)
            | (ERDMA_SIZE_INT16 << 22)
            | (EW_ALU_ADD << 16)
            | (1 << 9)
            | (1 << 7)
        )
        result = [
            emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
            emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
            emit(reg.TARGET_DPU, reg.BS_CFG, 0x53),
            emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
            emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 0),
            emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
            emit(reg.TARGET_DPU, reg.BN_CFG, 0x53),
            emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0),
            emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0),
            emit(reg.TARGET_DPU, reg.EW_CFG, ew_register_add),
        ]
        # Program every surface-group operand.  The flat eight-lane case uses
        # EW_OP_VALUE_0, but initializing all groups makes the stream explicit.
        result += [
            emit(reg.TARGET_DPU, reg.EW_OP_VALUE_0 + 4 * index, 0x40000000)
            for index in range(8)
        ]
        result += [
            emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
            emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1),
            emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 1),
            # ERDMA_DISABLE=1 proves there is no external scalar operand read.
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_ERDMA_CFG,
                rdma_erdma_cfg(ERDMA_SIZE_INT16) | 1,
            ),
            emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_FEATURE_MODE_CFG,
                rdma_feature_mode(PRECISION_FP16) | (1 << 3),
            ),
            emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
            emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            emit(reg.TARGET_VERSION, 0, 0),
            emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
        ]
        return result

    submit_ret, result = prepare_and_run(fd, commands, [values], np.float16, 8)
    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(
        f"FP16 ADD register operand NPU={result} expected={expected} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def run_fp16_register_pipeline(fd):
    """Prove configured BS, BN, and EW ALU stages in one DPU task."""
    values = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    expected = (values + np.float16(2.0) + np.float16(3.0) + np.float16(4.0)).astype(
        np.float16
    )

    def commands(output_bo, input_bos):
        input_bo = input_bos[0]
        # Register ALU source, ADD algorithm, RELUX bypass, and MUL bypass.
        configured_add = (EW_ALU_ADD << 16) | (1 << 6) | (1 << 4)
        ew_register_add = (
            (EW_DATA_MODE_PER_PIXEL << 28)
            | (ERDMA_SIZE_INT16 << 22)
            | (EW_ALU_ADD << 16)
            | (1 << 9)
            | (1 << 7)
        )
        result = [
            emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
            emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
            emit(reg.TARGET_DPU, reg.BS_CFG, configured_add),
            emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0x40000000),  # FP32 2.0.
            emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 0),
            emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
            emit(reg.TARGET_DPU, reg.BN_CFG, configured_add),
            emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0x40400000),  # FP32 3.0.
            emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0),
            emit(reg.TARGET_DPU, reg.EW_CFG, ew_register_add),
        ]
        result += [
            emit(reg.TARGET_DPU, reg.EW_OP_VALUE_0 + 4 * index, 0x40800000)
            for index in range(8)
        ]
        result += [
            emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
            # No external BS, BN, or EW operand is read in this pipeline.
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_ERDMA_CFG,
                rdma_erdma_cfg(ERDMA_SIZE_INT16) | 1,
            ),
            emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_FEATURE_MODE_CFG,
                rdma_feature_mode(PRECISION_FP16) | (1 << 3),
            ),
            emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
            emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            emit(reg.TARGET_VERSION, 0, 0),
            emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
        ]
        return result

    submit_ret, result = prepare_and_run(fd, commands, [values], np.float16, 8)
    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(
        f"FP16 BS+BN+EW register pipeline NPU={result} expected={expected} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def fp16_configured_minus_commands(output_dma, input_dma, stage):
    """Build one isolated BS or BN configured-MINUS stream."""
    if stage not in ("BS", "BN"):
        raise ValueError(f"unknown configured MINUS stage: {stage}")
    # ALGO=4 is MINUS in the TRM.  The ALU source bit stays zero, selecting
    # the configured FP32 operand.  MUL and ReLU are bypassed.
    configured_minus = (EW_ALU_MINUS << 16) | (1 << 6) | (1 << 4)
    stage_bypass = 0x53
    return [
        emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
        emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
        emit(
            reg.TARGET_DPU,
            reg.BS_CFG,
            configured_minus if stage == "BS" else stage_bypass,
        ),
        emit(
            reg.TARGET_DPU,
            reg.BS_ALU_CFG,
            0x40000000 if stage == "BS" else 0,
        ),  # Configured FP32 operand 2.0.
        emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 0),
        emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
        emit(
            reg.TARGET_DPU,
            reg.BN_CFG,
            configured_minus if stage == "BN" else stage_bypass,
        ),
        emit(
            reg.TARGET_DPU,
            reg.BN_ALU_CFG,
            0x40000000 if stage == "BN" else 0,
        ),  # Configured FP32 operand 2.0.
        emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0),
        # Fully bypass EW; this test exercises only the selected auxiliary stage.
        emit(reg.TARGET_DPU, reg.EW_CFG, 0x108002C1),
        emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
        emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, 1),
        emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_FEATURE_MODE_CFG,
            rdma_feature_mode(PRECISION_FP16) | (1 << 3),
        ),
        emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
        emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        emit(reg.TARGET_VERSION, 0, 0),
        emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
    ]


def fp16_external_mul_commands(output_dma, input_dma, mul_dma, stage):
    """Build one isolated BRDMA/NRDMA external-MUL stream."""
    if stage not in ("BS", "BN"):
        raise ValueError(f"unknown external MUL stage: {stage}")
    # Keep only MUL active in the selected auxiliary stage.  MUL_SRC=1 selects
    # its DMA client.  Silicon proves that MUL-only consumes one FP16 component
    # per element on RK3588, matching the NVDLA reference path.
    configured_external_mul = (1 << 6) | (1 << 1)
    stage_bypass = 0x53
    brdma_cfg = (1 << 2) << 1 if stage == "BS" else 1
    nrdma_cfg = (1 << 2) << 1 if stage == "BN" else 1
    return [
        emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
        emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
        emit(
            reg.TARGET_DPU,
            reg.BS_CFG,
            configured_external_mul if stage == "BS" else stage_bypass,
        ),
        emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 1 if stage == "BS" else 0),
        emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
        emit(
            reg.TARGET_DPU,
            reg.BN_CFG,
            configured_external_mul if stage == "BN" else stage_bypass,
        ),
        emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 1 if stage == "BN" else 0),
        emit(reg.TARGET_DPU, reg.EW_CFG, 0x108002C1),
        emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
        emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, brdma_cfg),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_BS_BASE_ADDR,
            mul_dma if stage == "BS" else 0,
        ),
        emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, nrdma_cfg),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_BN_BASE_ADDR,
            mul_dma if stage == "BN" else 0,
        ),
        emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, 1),
        emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_FEATURE_MODE_CFG,
            rdma_feature_mode(PRECISION_FP16) | (1 << 3),
        ),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_WEIGHT,
            (1 << 24) | (1 << 16) | (1 << 8) | 1,
        ),
        emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
        emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        emit(reg.TARGET_VERSION, 0, 0),
        emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
    ]


def fp16_regroup_stride2_commands(output_dma, input_dma):
    """Build an identity stream with TRM regroup selection set to one-in-two."""
    # RGP_TYPE=3 cuts the 128-bit input atom into FP16 pieces.  RGP_CNTER=1
    # selects one item from every two.  The selection phase is not documented.
    return [
        emit(
            reg.TARGET_DPU,
            reg.FEATURE_MODE_CFG,
            dpu_feature_mode() | (3 << 26),
        ),
        emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
        emit(reg.TARGET_DPU, reg.DST_SURF_STRIDE, 1 << 4),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 1),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
        emit(reg.TARGET_DPU, reg.BS_CFG, 0x53),
        emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 0),
        emit(reg.TARGET_DPU, reg.BS_OW_CFG, (1 << 28) | 2),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
        emit(reg.TARGET_DPU, reg.BN_CFG, 0x53),
        emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0),
        emit(reg.TARGET_DPU, reg.EW_CFG, 0x108002C1),
        emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
        emit(reg.TARGET_DPU, reg.SURFACE_ADD, 1 << 4),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
        emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, 1),
        emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_FEATURE_MODE_CFG,
            rdma_feature_mode(PRECISION_FP16) | (1 << 3),
        ),
        emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
        emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        emit(reg.TARGET_VERSION, 0, 0),
        emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
    ]


def run_fp16_bs_bn_minus(fd, stages=("BS", "BN")):
    """Prove configured MINUS direction in BS and BN."""
    values = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    expected = (values.astype(np.float32) - 2.0).astype(np.float16)
    reverse = (2.0 - values.astype(np.float32)).astype(np.float16)
    passed = []

    for stage in stages:
        def commands(output_bo, input_bos, selected_stage=stage):
            return fp16_configured_minus_commands(
                output_bo.dma_addr,
                input_bos[0].dma_addr,
                selected_stage,
            )

        submit_ret, result = prepare_and_run(fd, commands, [values], np.float16, 8)
        case_passed = submit_ret == 0 and np.array_equal(result, expected)
        orientation = "main-minus-operand"
        if submit_ret == 0 and np.array_equal(result, reverse):
            orientation = "operand-minus-main"
        elif not case_passed:
            orientation = "unresolved"
        print(
            f"FP16 {stage} configured MINUS NPU={result} expected={expected} "
            f"orientation={orientation} {'PASS' if case_passed else 'FAIL'}"
        )
        passed.append(case_passed)
    return all(passed)


def run_fp16_regroup_stride2(fd):
    """Probe regular one-in-two FP16 regroup selection."""
    values = np.arange(16, dtype=np.float16)
    even_expected = values[0::2]
    odd_expected = values[1::2]

    def commands(output_bo, input_bos):
        return fp16_regroup_stride2_commands(
            output_bo.dma_addr,
            input_bos[0].dma_addr,
        )

    submit_ret, result = prepare_and_run(fd, commands, [values], np.float16, 8)
    phase = "unresolved"
    if submit_ret == 0 and np.array_equal(result, even_expected):
        phase = "even"
    elif submit_ret == 0 and np.array_equal(result, odd_expected):
        phase = "odd"
    passed = phase != "unresolved"
    print(
        f"FP16 regroup one-in-two NPU={result} even={even_expected} "
        f"odd={odd_expected} phase={phase} {'PASS' if passed else 'FAIL'}"
    )
    return passed


def run_fp16_bs_bn_external_mul(fd, stages=("BS", "BN")):
    """Prove BRDMA/NRDMA MUL-only FP16 payload packing."""
    values = np.asarray([-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0], dtype=np.float16)
    multipliers = np.asarray([0.5, -1.0, 2.0, -2.0, 3.0, 0.25, -0.5, 1.5], dtype=np.float16)
    expected = (values.astype(np.float32) * multipliers.astype(np.float32)).astype(
        np.float16
    )
    passed = []

    for stage in stages:
        def commands(output_bo, input_bos, selected_stage=stage):
            return fp16_external_mul_commands(
                output_bo.dma_addr,
                input_bos[0].dma_addr,
                input_bos[1].dma_addr,
                selected_stage,
            )

        submit_ret, result = prepare_and_run(
            fd,
            commands,
            [values, multipliers],
            np.float16,
            8,
        )
        case_passed = submit_ret == 0 and np.array_equal(result, expected)
        print(
            f"FP16 {stage} external MUL NPU={result} expected={expected} "
            f"{'PASS' if case_passed else 'FAIL'}"
        )
        passed.append(case_passed)
    return all(passed)


def fp16_external_mul_layout_commands(
    output_dma,
    input_dma,
    mul_dma,
    stage,
    width,
    channels,
):
    """Build a multi-surface BS/BN external-MUL broadcast stream."""
    if stage not in ("BS", "BN"):
        raise ValueError(f"unknown external MUL stage: {stage}")
    channels_minus_one = channels - 1
    surface_stride = width << 4
    configured_external_mul = (1 << 6) | (1 << 1)
    stage_bypass = 0x53
    return [
        emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
        emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
        emit(reg.TARGET_DPU, reg.DST_SURF_STRIDE, surface_stride),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, width - 1),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
        emit(
            reg.TARGET_DPU,
            reg.DATA_CUBE_CHANNEL,
            dpu_cube_channel(channels_minus_one),
        ),
        emit(
            reg.TARGET_DPU,
            reg.BS_CFG,
            configured_external_mul if stage == "BS" else stage_bypass,
        ),
        emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 1 if stage == "BS" else 0),
        emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, channels_minus_one),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, width - 1),
        emit(
            reg.TARGET_DPU,
            reg.BN_CFG,
            configured_external_mul if stage == "BN" else stage_bypass,
        ),
        emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 1 if stage == "BN" else 0),
        emit(reg.TARGET_DPU, reg.EW_CFG, 0x108002C1),
        emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
        emit(reg.TARGET_DPU, reg.SURFACE_ADD, surface_stride),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, width - 1),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, channels_minus_one),
        emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 8 if stage == "BS" else 1),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_BS_BASE_ADDR,
            mul_dma if stage == "BS" else 0,
        ),
        emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 8 if stage == "BN" else 1),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_BN_BASE_ADDR,
            mul_dma if stage == "BN" else 0,
        ),
        emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_DMA_CFG, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_SURF_NOTCH, 0),
        emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_FEATURE_MODE_CFG,
            rdma_feature_mode(PRECISION_FP16) | (1 << 3),
        ),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_WEIGHT,
            (1 << 24) | (1 << 16) | (1 << 8) | 1,
        ),
        emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
        emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        emit(reg.TARGET_VERSION, 0, 0),
        emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
    ]


def run_fp16_external_mul_layout(fd, stage, width, channels):
    """Prove per-channel BRDMA/NRDMA MUL broadcast across DPU surfaces."""
    lane = np.arange(width * channels, dtype=np.float32).reshape(width, channels)
    values = ((lane % 17) - 8).astype(np.float16)
    multipliers = (((np.arange(channels) * 3) % 9) - 4).astype(np.float16)
    expected = (
        values.astype(np.float32) * multipliers[None, :].astype(np.float32)
    ).astype(np.float16)
    packed_values = pack_c1wc2(values, width, channels, np.float16)
    packed_multipliers = pack_c1wc2(
        multipliers.reshape(1, channels), 1, channels, np.float16
    )

    def commands(output_bo, input_bos):
        return fp16_external_mul_layout_commands(
            output_bo.dma_addr,
            input_bos[0].dma_addr,
            input_bos[1].dma_addr,
            stage,
            width,
            channels,
        )

    submit_ret, packed_result = prepare_and_run(
        fd,
        commands,
        [packed_values, packed_multipliers],
        np.float16,
        len(packed_values),
    )
    result = unpack_c1wc2(packed_result, width, channels, np.float16)
    passed = submit_ret == 0 and np.array_equal(result, expected)
    max_diff = float(
        np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    )
    print(
        f"FP16 {stage} external MUL broadcast width={width} channels={channels} "
        f"submit={submit_ret} max_diff={max_diff:.6g} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        print(f"NPU={result.reshape(-1)}")
        print(f"expected={expected.reshape(-1)}")
    return passed


def validate_external_mul_layout_streams():
    """Validate the planned BS/BN external-MUL layout streams offline."""
    checked = 0
    for stage in ("BS", "BN"):
        for width in (1, 2, 3):
            for channels in (3, 7, 8, 9, 15, 16, 17, 31, 32, 33):
                commands = fp16_external_mul_layout_commands(
                    0x20000000,
                    0x10000000,
                    0x10010000,
                    stage,
                    width,
                    channels,
                )
                decoded = {
                    ((command >> 48) & 0xFFFF, command & 0xFFFF):
                    (command >> 16) & 0xFFFFFFFF
                    for command in commands
                }
                stride = width << 4
                assert decoded[(reg.TARGET_DPU, reg.DST_SURF_STRIDE)] == stride
                assert decoded[(reg.TARGET_DPU, reg.SURFACE_ADD)] == stride
                assert decoded[(reg.TARGET_DPU, reg.WDMA_SIZE_0)] == channels - 1
                assert decoded[(reg.TARGET_DPU, reg.WDMA_SIZE_1)] == width - 1
                assert decoded[(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG)] == (
                    8 if stage == "BS" else 1
                )
                assert decoded[(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG)] == (
                    8 if stage == "BN" else 1
                )
                checked += 1
    print(f"offline external-MUL layout validation PASS ({checked} streams)")
    return True


def fp16_external_alu_mul_commands(output_dma, input_dma, operand_dma, stage):
    """Build the NVDLA-reference interleaved external ALU+MUL recipe."""
    if stage not in ("BS", "BN"):
        raise ValueError(f"unknown external ALU+MUL stage: {stage}")
    memory_add_mul = (EW_ALU_ADD << 16) | (1 << 8) | (1 << 6)
    stage_bypass = 0x53
    return [
        emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
        emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
        emit(reg.TARGET_DPU, reg.DST_SURF_STRIDE, 1 << 4),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
        emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
        emit(
            reg.TARGET_DPU,
            reg.BS_CFG,
            memory_add_mul if stage == "BS" else stage_bypass,
        ),
        emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 1 if stage == "BS" else 0),
        emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
        emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
        emit(
            reg.TARGET_DPU,
            reg.BN_CFG,
            memory_add_mul if stage == "BN" else stage_bypass,
        ),
        emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0),
        emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 1 if stage == "BN" else 0),
        emit(reg.TARGET_DPU, reg.EW_CFG, 0x108002C1),
        emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
        emit(reg.TARGET_DPU, reg.SURFACE_ADD, 1 << 4),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
        # DATA_USE is a Rockchip bitmask: ALU=1 and MUL=4.
        emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 10 if stage == "BS" else 1),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_BS_BASE_ADDR,
            operand_dma if stage == "BS" else 0,
        ),
        emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 10 if stage == "BN" else 1),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_BN_BASE_ADDR,
            operand_dma if stage == "BN" else 0,
        ),
        emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, 1),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_DMA_CFG, 0),
        emit(reg.TARGET_RDMA, reg.RDMA_SURF_NOTCH, 0),
        emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_dma),
        emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_FEATURE_MODE_CFG,
            rdma_feature_mode(PRECISION_FP16) | (1 << 3),
        ),
        emit(
            reg.TARGET_RDMA,
            reg.RDMA_WEIGHT,
            (1 << 24) | (1 << 16) | (1 << 8) | 1,
        ),
        emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
        emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        emit(reg.TARGET_VERSION, 0, 0),
        emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
    ]


def run_fp16_external_alu_mul(fd, stage):
    """Probe interleaved external ALU+MUL in BS or BN."""
    values = np.asarray([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.float16)
    alu = np.asarray([1, -1, 2, -2, 3, -3, 4, -4], dtype=np.float16)
    mul = np.asarray([0.5, -1, 2, -2, 3, 0.25, -0.5, 1.5], dtype=np.float16)
    operands = np.stack((alu, mul), axis=1).astype(np.float16).reshape(-1)
    if stage == "BS":
        expected = ((values.astype(np.float32) + alu) * mul).astype(np.float16)
    else:
        expected = (values.astype(np.float32) * mul + alu).astype(np.float16)

    def commands(output_bo, input_bos):
        return fp16_external_alu_mul_commands(
            output_bo.dma_addr,
            input_bos[0].dma_addr,
            input_bos[1].dma_addr,
            stage,
        )

    submit_ret, result = prepare_and_run(
        fd,
        commands,
        [values, operands],
        np.float16,
        8,
    )
    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(
        f"FP16 {stage} external ALU+MUL NPU={result} expected={expected} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def validate_external_alu_mul_streams():
    """Validate both interleaved external ALU+MUL streams offline."""
    for stage in ("BS", "BN"):
        commands = fp16_external_alu_mul_commands(
            0x20000000,
            0x10000000,
            0x10010000,
            stage,
        )
        decoded = {
            ((command >> 48) & 0xFFFF, command & 0xFFFF):
            (command >> 16) & 0xFFFFFFFF
            for command in commands
        }
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG)] == (
            10 if stage == "BS" else 1
        )
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG)] == (
            10 if stage == "BN" else 1
        )
        active_cfg = decoded[
            (reg.TARGET_DPU, reg.BS_CFG if stage == "BS" else reg.BN_CFG)
        ]
        assert active_cfg >> 16 & 0xF == EW_ALU_ADD
        assert active_cfg & (1 << 8)
        assert not active_cfg & (1 << 4)
        assert not active_cfg & (1 << 1)
    print("offline external ALU+MUL validation PASS (2 streams)")
    return True


def validate_unproved_probe_streams():
    """Validate isolated register probes without opening or submitting to the NPU."""
    fake_output = 0x22220000
    fake_input = 0x11110000

    def writes(commands):
        return {
            ((command >> 48) & 0xFFFF, command & 0xFFFF): (command >> 16) & 0xFFFFFFFF
            for command in commands
        }

    configured_minus = (EW_ALU_MINUS << 16) | (1 << 6) | (1 << 4)
    stage_bypass = 0x53
    for stage in ("BS", "BN"):
        decoded = writes(
            fp16_configured_minus_commands(fake_output, fake_input, stage)
        )
        assert decoded[(reg.TARGET_DPU, reg.BS_CFG)] == (
            configured_minus if stage == "BS" else stage_bypass
        )
        assert decoded[(reg.TARGET_DPU, reg.BN_CFG)] == (
            configured_minus if stage == "BN" else stage_bypass
        )
        assert decoded[(reg.TARGET_DPU, reg.BS_ALU_CFG)] == (
            0x40000000 if stage == "BS" else 0
        )
        assert decoded[(reg.TARGET_DPU, reg.BN_ALU_CFG)] == (
            0x40000000 if stage == "BN" else 0
        )
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG)] == 1
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG)] == 1
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG)] == 1
        assert decoded[(reg.TARGET_DPU, reg.DST_BASE_ADDR)] == fake_output
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR)] == fake_input
        assert decoded[(reg.TARGET_PC, reg.OPERATION_ENABLE)] == 0x18

    for stage in ("BS", "BN"):
        decoded = writes(
            fp16_external_mul_commands(
                fake_output,
                fake_input,
                0x33330000,
                stage,
            )
        )
        assert decoded[(reg.TARGET_DPU, reg.BS_CFG)] == (
            0x42 if stage == "BS" else stage_bypass
        )
        assert decoded[(reg.TARGET_DPU, reg.BN_CFG)] == (
            0x42 if stage == "BN" else stage_bypass
        )
        assert decoded[(reg.TARGET_DPU, reg.BS_MUL_CFG)] == (
            1 if stage == "BS" else 0
        )
        assert decoded[(reg.TARGET_DPU, reg.BN_MUL_CFG)] == (
            1 if stage == "BN" else 0
        )
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG)] == (
            8 if stage == "BS" else 1
        )
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG)] == (
            8 if stage == "BN" else 1
        )
        assert decoded[(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG)] == 1
        assert decoded[(reg.TARGET_PC, reg.OPERATION_ENABLE)] == 0x18

    decoded = writes(
        fp16_regroup_stride2_commands(fake_output, fake_input)
    )
    assert decoded[(reg.TARGET_DPU, reg.FEATURE_MODE_CFG)] >> 26 & 0xF == 3
    assert decoded[(reg.TARGET_DPU, reg.BS_OW_CFG)] >> 28 == 1
    assert decoded[(reg.TARGET_DPU, reg.WDMA_SIZE_0)] == 7
    assert decoded[(reg.TARGET_DPU, reg.WDMA_SIZE_1)] == 0
    assert decoded[(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH)] == 1
    assert decoded[(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG)] == 1
    assert decoded[(reg.TARGET_PC, reg.OPERATION_ENABLE)] == 0x18
    print("offline register-probe validation PASS (5 streams)")
    return True


def run_fp16_five_scalar_ops(fd):
    """Prove both affine halves of BS/BN plus EW in one DPU task."""
    values = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    # Hardware stage order is BS=(x+ALU)*MUL, BN=x*MUL+ALU, then EW.
    expected = (((values.astype(np.float32) + 2.0) * 3.0) * 2.0 + 1.0 + 4.0).astype(
        np.float16
    )

    def commands(output_bo, input_bos):
        input_bo = input_bos[0]
        # ADD selected with ALU and MUL active; only RELU is bypassed.
        configured_affine = (EW_ALU_ADD << 16) | (1 << 6)
        ew_register_add = (
            (EW_DATA_MODE_PER_PIXEL << 28)
            | (ERDMA_SIZE_INT16 << 22)
            | (EW_ALU_ADD << 16)
            | (1 << 9)
            | (1 << 7)
        )
        result = [
            emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
            emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
            emit(reg.TARGET_DPU, reg.BS_CFG, configured_affine),
            emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0x40000000),  # FP32 ADD 2.0.
            emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 0x42000000),  # FP16 MUL 3.0.
            emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
            emit(reg.TARGET_DPU, reg.BN_CFG, configured_affine),
            emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0x3F800000),  # FP32 ADD 1.0.
            emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0x40000000),  # FP16 MUL 2.0.
            emit(reg.TARGET_DPU, reg.EW_CFG, ew_register_add),
        ]
        result += [
            emit(reg.TARGET_DPU, reg.EW_OP_VALUE_0 + 4 * index, 0x40800000)
            for index in range(8)
        ]
        result += [
            emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_ERDMA_CFG,
                rdma_erdma_cfg(ERDMA_SIZE_INT16) | 1,
            ),
            emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_FEATURE_MODE_CFG,
                rdma_feature_mode(PRECISION_FP16) | (1 << 3),
            ),
            emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
            emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            emit(reg.TARGET_VERSION, 0, 0),
            emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
        ]
        return result

    submit_ret, result = prepare_and_run(fd, commands, [values], np.float16, 8)
    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(
        f"FP16 five scalar ops NPU={result} expected={expected} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def run_fp16_bs_bn_activations(fd):
    """Prove ReLU-X and PReLU in both auxiliary DPU stages."""
    relux_values = np.asarray(
        [-3.0, -1.0, -0.25, 0.0, 0.5, 1.5, 2.0, 4.0], dtype=np.float16
    )
    relux_expected = np.clip(relux_values, 0.0, 2.0).astype(np.float16)
    prelu_values = np.asarray(
        [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0], dtype=np.float16
    )
    prelu_expected = np.where(
        prelu_values < 0.0, prelu_values * np.float16(0.25), prelu_values
    ).astype(np.float16)

    def commands_for(stage, activation):
        def commands(output_bo, input_bos):
            input_bo = input_bos[0]
            if activation == "RELUX":
                # RELUX enabled, RELU active, ALU and MUL bypassed.
                active_cfg = (1 << 7) | (1 << 4) | (1 << 1)
                mul_cfg = 0
            else:
                # PReLU multiplier active, ordinary RELU and ALU bypassed.
                active_cfg = (1 << 6) | (1 << 5) | (1 << 1)
                mul_cfg = 0x34000000  # FP16 0.25 in bits 31:16.
            bs_cfg = active_cfg if stage == "BS" else 0x53
            bn_cfg = active_cfg if stage == "BN" else 0x53
            return [
                emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
                emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
                emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
                emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
                emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
                emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
                emit(reg.TARGET_DPU, reg.BS_CFG, bs_cfg),
                emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
                emit(reg.TARGET_DPU, reg.BS_MUL_CFG, mul_cfg),
                emit(reg.TARGET_DPU, reg.BS_RELUX_CMP_VALUE, 0x40000000),
                emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
                emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
                emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
                emit(reg.TARGET_DPU, reg.BN_CFG, bn_cfg),
                emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0),
                emit(reg.TARGET_DPU, reg.BN_MUL_CFG, mul_cfg),
                emit(reg.TARGET_DPU, reg.BN_RELUX_CMP_VALUE, 0x40000000),
                emit(reg.TARGET_DPU, reg.EW_CFG, 0x108002C1),
                emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
                emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
                emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
                emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
                emit(
                    reg.TARGET_RDMA,
                    reg.RDMA_ERDMA_CFG,
                    rdma_erdma_cfg(ERDMA_SIZE_INT16) | 1,
                ),
                emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
                emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
                emit(
                    reg.TARGET_RDMA,
                    reg.RDMA_FEATURE_MODE_CFG,
                    rdma_feature_mode(PRECISION_FP16) | (1 << 3),
                ),
                emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
                emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                emit(reg.TARGET_VERSION, 0, 0),
                emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
            ]

        return commands

    passed = []
    for stage in ("BS", "BN"):
        for activation, values, expected in (
            ("RELUX", relux_values, relux_expected),
            ("PRELU", prelu_values, prelu_expected),
        ):
            submit_ret, result = prepare_and_run(
                fd, commands_for(stage, activation), [values], np.float16, 8
            )
            case_passed = submit_ret == 0 and np.array_equal(result, expected)
            print(
                f"FP16 {stage} {activation} NPU={result} expected={expected} "
                f"{'PASS' if case_passed else 'FAIL'}"
            )
            passed.append(case_passed)
    return all(passed)


def run_fp16_four_input_pipeline(fd):
    """Prove main + BRDMA + NRDMA + ERDMA tensor addition in one task."""
    values = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    bs_values = np.asarray(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32
    )
    bn_values = np.asarray(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32
    )
    ew_values = np.asarray(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float16
    )
    expected = (
        values.astype(np.float32)
        + bs_values
        + bn_values
        + ew_values.astype(np.float32)
    ).astype(np.float16)

    def commands(output_bo, input_bos):
        input_bo, bs_bo, bn_bo, ew_bo = input_bos
        # BS/BN outside ALU sources are the BRDMA/NRDMA streams.
        memory_add = (EW_ALU_ADD << 16) | (1 << 8) | (1 << 6) | (1 << 4)
        return [
            emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
            emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
            emit(reg.TARGET_DPU, reg.BS_CFG, memory_add),
            emit(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
            emit(reg.TARGET_DPU, reg.BS_MUL_CFG, 0),
            emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
            emit(reg.TARGET_DPU, reg.BN_CFG, memory_add),
            emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0),
            emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0),
            emit(
                reg.TARGET_DPU,
                reg.EW_CFG,
                dpu_ew_cfg(ERDMA_SIZE_INT16, "ADD", converter_bypass=False),
            ),
            emit(reg.TARGET_DPU, reg.EW_CVT_SCALE_VALUE, 1),
            emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
            # DATA_USE bit 0 routes each auxiliary DMA stream to its ALU.
            emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1 << 1),
            emit(reg.TARGET_RDMA, reg.RDMA_BS_BASE_ADDR, bs_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 1 << 1),
            emit(reg.TARGET_RDMA, reg.RDMA_BN_BASE_ADDR, bn_bo.dma_addr),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_ERDMA_CFG,
                rdma_erdma_cfg(ERDMA_SIZE_INT16),
            ),
            emit(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR, ew_bo.dma_addr),
            emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_FEATURE_MODE_CFG,
                rdma_feature_mode(PRECISION_FP16) | (1 << 3),
            ),
            # Program every active RDMA client's arbitration weight explicitly.
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_WEIGHT,
                (1 << 24) | (1 << 16) | (1 << 8) | 1,
            ),
            emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
            emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            emit(reg.TARGET_VERSION, 0, 0),
            emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
        ]

    submit_ret, result = prepare_and_run(
        fd,
        commands,
        [values, bs_values, bn_values, ew_values],
        np.float16,
        8,
    )
    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(
        f"FP16 four-input DPU pipeline NPU={result} expected={expected} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def run_fp16_per_channel_broadcast(fd):
    """Prove EW/ERDMA data mode 0 as a per-channel broadcast."""
    values = np.zeros(32, dtype=np.float16)
    channel_values = np.arange(1, 17, dtype=np.float16)
    # Physical order is two pixels for each eight-channel surface.
    expected = np.concatenate(
        (channel_values[:8], channel_values[:8], channel_values[8:], channel_values[8:])
    )

    def commands(output_bo, input_bos):
        input_bo, ew_bo = input_bos
        per_channel_add = (
            (ERDMA_SIZE_INT16 << 22)
            | (EW_ALU_ADD << 16)
            | (1 << 9)
            | (1 << 7)
            | (1 << 6)
        )
        return [
            emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
            emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
            # Each output channel surface contains two pixels × eight FP16s.
            emit(reg.TARGET_DPU, reg.DST_SURF_STRIDE, 2 << 4),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 1),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(15)),
            emit(reg.TARGET_DPU, reg.BS_CFG, 0x53),
            emit(reg.TARGET_DPU, reg.BS_OW_CFG, 2),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 15),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 1),
            emit(reg.TARGET_DPU, reg.BN_CFG, 0x53),
            emit(reg.TARGET_DPU, reg.EW_CFG, per_channel_add),
            emit(reg.TARGET_DPU, reg.EW_CVT_SCALE_VALUE, 1),
            emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
            emit(reg.TARGET_DPU, reg.SURFACE_ADD, 2 << 4),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 1),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 15),
            emit(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1),
            emit(reg.TARGET_RDMA, reg.RDMA_NRDMA_CFG, 1),
            # ERDMA_DATA_MODE=0 matches EW_DATA_MODE=0 above.
            emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, ERDMA_SIZE_INT16 << 2),
            emit(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR, ew_bo.dma_addr),
            # One mode-0 operand surface contains eight contiguous FP16s.
            emit(reg.TARGET_RDMA, reg.RDMA_EW_SURF_STRIDE, 1 << 4),
            emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_FEATURE_MODE_CFG,
                rdma_feature_mode(PRECISION_FP16) | (1 << 3),
            ),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_WEIGHT,
                (1 << 24) | (1 << 16) | (1 << 8) | 1,
            ),
            emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
            emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            emit(reg.TARGET_VERSION, 0, 0),
            emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
        ]

    submit_ret, result = prepare_and_run(
        fd, commands, [values, channel_values], np.float16, 32
    )
    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(
        f"FP16 per-channel broadcast NPU={result} expected={expected} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def run_fp16_silu_lut(fd):
    """Prove the complete two-table nonlinear LUT through the Rocket ABI."""
    lut_size = 513
    index_scale = 2824.0
    output_scale = 5664.8
    step = 32.0 / index_scale
    lut = [0] * (lut_size * 2)
    for index in range(lut_size):
        value = (lut_size - 1 - index) * step
        result = -value / (1.0 + math.exp(value))
        lut[index] = int(np.clip(round(result * output_scale), -32768, 32767))
    for index in range(lut_size):
        value = index * step
        result = value / (1.0 + math.exp(-value))
        lut[lut_size + index] = int(
            np.clip(round(result * output_scale), -32768, 32767)
        )

    values = np.asarray(
        [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], dtype=np.float16
    )
    expected = (
        values.astype(np.float32) / (1.0 + np.exp(-values.astype(np.float32)))
    ).astype(np.float16)

    def commands(output_bo, input_bos):
        input_bo = input_bos[0]
        result = []
        # ACCESS_TYPE=write auto-increments the address after each data write.
        for table_id, base in ((0, 0), (1, lut_size)):
            result.append(
                emit(
                    reg.TARGET_DPU,
                    reg.LUT_ACCESS_CFG,
                    (1 << 17) | (table_id << 16),
                )
            )
            for index in range(lut_size):
                entry = lut[base + index]
                data = entry & 0xFFFF
                if entry < 0:
                    data |= 0xFFFF0000
                result.append(emit(reg.TARGET_DPU, reg.LUT_ACCESS_DATA, data))
        result += [
            # Clear DPU/RDMA ping-pong state before replacing their shared LUT.
            emit(reg.TARGET_DPU, 0x4004, (1 << 5) | (1 << 4)),
            emit(reg.TARGET_RDMA, 0x5004, (1 << 5) | (1 << 4)),
            emit(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, dpu_feature_mode()),
            emit(reg.TARGET_DPU, reg.DATA_FORMAT, dpu_data_format(PRECISION_FP16)),
            emit(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_bo.dma_addr),
            emit(reg.TARGET_DPU, reg.DST_SURF_STRIDE, 16 << 4),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
            emit(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, dpu_cube_channel(7)),
            emit(reg.TARGET_DPU, reg.BS_CFG, 0x53),
            emit(reg.TARGET_DPU, reg.BS_OW_CFG, 1 << 1),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_0, 7),
            emit(reg.TARGET_DPU, reg.WDMA_SIZE_1, 0),
            # BN scales inputs into LUT indices.  EW_OP_BYPASS leaves only LUT.
            emit(reg.TARGET_DPU, reg.BN_CFG, (EW_ALU_ADD << 16) | (1 << 6)),
            emit(reg.TARGET_DPU, reg.BN_ALU_CFG, 0x80000000),
            emit(reg.TARGET_DPU, reg.BN_MUL_CFG, 0x6984 << 16),
            emit(reg.TARGET_DPU, reg.EW_CFG, (1 << 9) | (1 << 8) | (1 << 1)),
            emit(reg.TARGET_DPU, reg.EW_CVT_SCALE_VALUE, 1),
            emit(reg.TARGET_DPU, reg.OUT_CVT_SCALE, (1 << 16) | 1),
            emit(reg.TARGET_DPU, reg.SURFACE_ADD, 32 << 4),
            emit(reg.TARGET_DPU, 0x40C4, 0),
            emit(reg.TARGET_DPU, reg.LUT_CFG, (1 << 6) | (1 << 5) | (2 << 2)),
            emit(reg.TARGET_DPU, reg.LUT_INFO, (5 << 16) | (5 << 8)),
            emit(reg.TARGET_DPU, reg.LUT_LE_START, 0xFFFFC000),
            emit(reg.TARGET_DPU, reg.LUT_LO_END, 0x00004000),
            emit(reg.TARGET_DPU, reg.LUT_LO_SLOPE_SCALE, 16434 << 16),
            emit(reg.TARGET_DPU, reg.LUT_LO_SLOPE_SHIFT, 13 << 5),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
            emit(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
            emit(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_bo.dma_addr),
            emit(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, 1),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_FEATURE_MODE_CFG,
                rdma_feature_mode(PRECISION_FP16) | (1 << 3),
            ),
            emit(
                reg.TARGET_RDMA,
                reg.RDMA_WEIGHT,
                (1 << 24) | (1 << 16) | (1 << 8) | 1,
            ),
            emit(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, 0),
            emit(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
            emit(reg.TARGET_VERSION, 0, 0),
            emit(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x18),
        ]
        return result

    submit_ret, raw = prepare_and_run(fd, commands, [values], np.float16, 8)
    result = (raw.astype(np.float32) / output_scale).astype(np.float16)
    max_error = float(
        np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    )
    passed = submit_ret == 0 and np.allclose(result, expected, atol=0.01)
    print(
        f"FP16 SiLU LUT NPU={result} expected={expected} max_error={max_error:.7f} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def saturate(values, dtype):
    limits = np.iinfo(dtype)
    return np.clip(values, limits.min, limits.max).astype(dtype)


def integer_expected(operation, a, b, dtype):
    a64 = a.astype(np.int64)
    b64 = b.astype(np.int64)
    if operation == "MAX":
        return np.maximum(a, b)
    if operation == "MIN":
        return np.minimum(a, b)
    if operation == "ADD":
        return saturate(a64 + b64, dtype)
    if operation == "DIV":
        # C/RKNN signed integer division truncates toward zero.
        return np.trunc(a64 / b64).astype(dtype)
    if operation == "MINUS":
        return saturate(a64 - b64, dtype)
    if operation == "ABS":
        return saturate(np.abs(a64), dtype)
    if operation == "NEG":
        return saturate(-a64, dtype)
    if operation in ("FLOOR", "CEIL"):
        # Floor and ceil of an integer are identity operations.
        return a.copy()
    if operation == "MUL":
        return saturate(a64 * b64, dtype)
    raise ValueError(f"unknown operation: {operation}")


DTYPES = {
    "INT8": {
        "dtype": np.int8,
        "precision": PRECISION_INT8,
        "element_size": ERDMA_SIZE_INT8,
        "channels": 8,
        "a_values": [-100, -24, -7, -1, 0, 1, 24, 100],
        "b_values": [20, -6, 2, -1, 1, 2, -3, 5],
    },
    "INT16": {
        "dtype": np.int16,
        "precision": PRECISION_INT16,
        "element_size": ERDMA_SIZE_INT16,
        "channels": 8,
        "a_values": [-30000, -1200, -7, -1, 0, 1, 1200, 30000],
        "b_values": [1000, -30, 2, -1, 1, 2, -30, 1000],
    },
    "INT32": {
        "dtype": np.int32,
        "precision": PRECISION_INT32,
        "element_size": ERDMA_SIZE_INT32,
        "channels": 4,
        # Values beyond signed INT16 expose whether the EW operand is full INT32.
        "a_values": [-1000, -2000, -7, -1, 0, 1, 2000, 1000],
        "b_values": [50000, -50000, 2, -1, 1, 2, -50000, 50000],
    },
}


if __name__ == "__main__":
    if sys.argv[1:] == ["--validate-probes"]:
        sys.exit(0 if validate_unproved_probe_streams() else 1)
    if sys.argv[1:] == ["--validate-layout-probes"]:
        sys.exit(0 if validate_integer_layout_streams() else 1)
    if sys.argv[1:] == ["--validate-external-mul-layouts"]:
        sys.exit(0 if validate_external_mul_layout_streams() else 1)
    if sys.argv[1:] == ["--validate-external-alu-mul"]:
        sys.exit(0 if validate_external_alu_mul_streams() else 1)
    if len(sys.argv) == 6 and sys.argv[1] in ("--probe-layout", "--probe-layout-edge"):
        edge_profile = sys.argv[1] == "--probe-layout-edge"
        layout_dtype = sys.argv[2].upper()
        layout_operation = sys.argv[3].upper()
        if layout_dtype not in DTYPES:
            raise SystemExit(f"unknown layout dtype {layout_dtype!r}")
        if layout_operation not in OPERATIONS:
            raise SystemExit(f"unknown layout operation {layout_operation!r}")
        layout_width = int(sys.argv[4])
        layout_channels = int(sys.argv[5])
        if layout_width <= 0 or layout_channels <= 0:
            raise SystemExit("layout width and channels must be positive")
        device_fd = open_rocket_device()
        try:
            probe_passed = run_integer_layout(
                device_fd,
                layout_dtype,
                layout_operation,
                layout_width,
                layout_channels,
                edge_profile,
            )
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if len(sys.argv) == 3 and sys.argv[1] == "--probe-external-alu-mul":
        both_stage = sys.argv[2].upper()
        if both_stage not in ("BS", "BN"):
            raise SystemExit("external ALU+MUL stage must be BS or BN")
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_external_alu_mul(device_fd, both_stage)
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if len(sys.argv) == 5 and sys.argv[1] == "--probe-external-mul-layout":
        mul_stage = sys.argv[2].upper()
        mul_width = int(sys.argv[3])
        mul_channels = int(sys.argv[4])
        if mul_stage not in ("BS", "BN"):
            raise SystemExit("external MUL stage must be BS or BN")
        if mul_width <= 0 or mul_channels <= 0:
            raise SystemExit("external MUL width and channels must be positive")
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_external_mul_layout(
                device_fd,
                mul_stage,
                mul_width,
                mul_channels,
            )
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if sys.argv[1:] == ["--probe-bs-bn-minus"]:
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_bs_bn_minus(device_fd)
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if sys.argv[1:] == ["--probe-bs-minus"]:
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_bs_bn_minus(device_fd, ("BS",))
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if sys.argv[1:] == ["--probe-bn-minus"]:
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_bs_bn_minus(device_fd, ("BN",))
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if sys.argv[1:] == ["--probe-bs-bn-external-mul"]:
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_bs_bn_external_mul(device_fd)
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if sys.argv[1:] == ["--probe-brdma-mul"]:
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_bs_bn_external_mul(device_fd, ("BS",))
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if sys.argv[1:] == ["--probe-nrdma-mul"]:
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_bs_bn_external_mul(device_fd, ("BN",))
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)
    if sys.argv[1:] == ["--probe-regroup-stride2"]:
        device_fd = open_rocket_device()
        try:
            probe_passed = run_fp16_regroup_stride2(device_fd)
        finally:
            os.close(device_fd)
        sys.exit(0 if probe_passed else 1)

    dtype_mode = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    operation_mode = sys.argv[2].upper() if len(sys.argv) > 2 else "ALL"
    if dtype_mode == "ALL":
        selected_dtypes = list(DTYPES.items())
    elif dtype_mode in DTYPES:
        selected_dtypes = [(dtype_mode, DTYPES[dtype_mode])]
    else:
        print(f"Unknown dtype '{dtype_mode}'. Options: ALL, {', '.join(DTYPES)}")
        sys.exit(2)
    if operation_mode == "ALL":
        selected_operations = list(OPERATIONS)
    elif operation_mode in OPERATIONS:
        selected_operations = [operation_mode]
    else:
        print(f"Unknown operation '{operation_mode}'. Options: ALL, {', '.join(OPERATIONS)}")
        sys.exit(2)

    device_fd = open_rocket_device()
    try:
        passed = [
            run_operation(device_fd, dtype_name, operation, **dtype_config)
            for dtype_name, dtype_config in selected_dtypes
            for operation in selected_operations
        ]
        if dtype_mode == "ALL" and operation_mode == "ALL":
            # Keep one Rocket file context for the complete register sequence.
            # Closing and reopening the node does not reset the physical NPU,
            # but it does discard the driver's per-context synchronization.
            for boundary_test in (
                run_fp16_per_channel_broadcast,
                run_fp16_compare_to_int16,
                run_int16_add_to_int32,
                run_fp16_add_register_operand,
                run_fp16_register_pipeline,
                run_fp16_five_scalar_ops,
                run_fp16_bs_bn_activations,
                run_fp16_four_input_pipeline,
                run_fp16_silu_lut,
            ):
                passed.append(boundary_test(device_fd))
    finally:
        os.close(device_fd)
    sys.exit(0 if all(passed) else 1)
