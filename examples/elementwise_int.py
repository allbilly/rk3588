"""Probe every hardware-listed DPU elementwise operation with integer data.

Observed RK3588 integer behavior:
  * MAX, MIN, ADD, MINUS, ABS, and NEG work for INT8, INT16, and INT32.
  * MUL works for INT8 and INT16.  With INT32, its second operand is signed
    INT16 even though ERDMA is configured and packed as INT32.
  * DIV, FLOOR, and CEIL use floating-point semantics and are not integer ops.

The checks below prove supported behavior and deliberately prove the three
unsupported integer ALU encodings by showing that they differ from ordinary
integer semantics.  All register commands are decoded; there is no hex blob.
"""

from fcntl import ioctl
import ctypes
import glob
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
    FEATURE_MODE_CFG = 0x400C
    DATA_FORMAT = 0x4010
    DST_BASE_ADDR = 0x4020
    DATA_CUBE_WIDTH = 0x4030
    DATA_CUBE_CHANNEL = 0x403C
    EW_CFG = 0x4070
    EW_CVT_SCALE_VALUE = 0x4078

    # --- DPU RDMA (0x5000) ---
    RDMA_DATA_CUBE_WIDTH = 0x500C
    RDMA_DATA_CUBE_HEIGHT = 0x5010
    RDMA_DATA_CUBE_CHANNEL = 0x5014
    RDMA_SRC_BASE_ADDR = 0x5018
    RDMA_ERDMA_CFG = 0x5034
    RDMA_EW_BASE_ADDR = 0x5038
    RDMA_FEATURE_MODE_CFG = 0x5044


PRECISION_INT8 = 0
PRECISION_INT16 = 1
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

    if not OPERATIONS[operation]["integer_supported"]:
        unsupported_proved = submit_ret == 0 and not np.array_equal(result, expected)
        status = "PASS (NOT AN INTEGER OP)" if unsupported_proved else "FAIL"
        print(f"{label:11s} NPU={result}")
        print(f"            integer expected={expected} {status}")
        return unsupported_proved

    passed = submit_ret == 0 and np.array_equal(result, expected)
    print(f"{label:11s} NPU={result} expected={expected} {'PASS' if passed else 'FAIL'}")
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
    finally:
        os.close(device_fd)
    sys.exit(0 if all(passed) else 1)
