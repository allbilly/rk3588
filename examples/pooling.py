"""Exercise the RK3588 PPU pooling surface with decoded register writes.

Hardware results on RK3588:
  * Average, maximum, and minimum pooling are native PPU operations.
  * Global average pooling needs dimension-specific FP17 reciprocals; it does
    not need a host-side divide.
  * INDEX_EN writes local kernel coordinates as (row << 3) | column.
  * FP16, INT8, and INT16 work.  Tested INT32 and FP32 configurations submit
    successfully but write zero, disproving those formats on this PPU path.
  * Kernel/stride 16, padding 7, and non-multiple channel counts work in the
    isolated probes below.  Pixels are still stored in 16-byte atoms.

The PPU operation selector only contains average, maximum, and minimum.
INDEX_EN is pool-window position output, not a general ArgMax.  Sort, WHERE,
gather/scatter, IEEE helpers, and integer bit operations have no PPU selector.
Unpool/upsample belongs to DPU_RDMA, not PPU, and is therefore not submitted by
this PPU-only example.  All commands below name their register and field; there
is no captured hex command blob.
"""

import ctypes
from fcntl import ioctl
import glob
import math
import mmap
import os
import subprocess
import sys
import time

import numpy as np


class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    PPU = 0x4001
    PPU_RDMA = 0x8001
    PC = 0x0081

    # --- PC ---
    PC_OPERATION_ENABLE = 0x0008

    # --- PPU ---
    PPU_S_POINTER = 0x6004
    PPU_DATA_CUBE_IN_WIDTH = 0x600C
    PPU_DATA_CUBE_IN_HEIGHT = 0x6010
    PPU_DATA_CUBE_IN_CHANNEL = 0x6014
    PPU_DATA_CUBE_OUT_WIDTH = 0x6018
    PPU_DATA_CUBE_OUT_HEIGHT = 0x601C
    PPU_DATA_CUBE_OUT_CHANNEL = 0x6020
    PPU_OPERATION_MODE_CFG = 0x6024
    PPU_POOLING_KERNEL_CFG = 0x6034
    PPU_RECIP_KERNEL_WIDTH = 0x6038
    PPU_RECIP_KERNEL_HEIGHT = 0x603C
    PPU_POOLING_PADDING_CFG = 0x6040
    PPU_PADDING_VALUE_1_CFG = 0x6044
    PPU_PADDING_VALUE_2_CFG = 0x6048
    PPU_DST_BASE_ADDR = 0x6070
    PPU_DST_SURF_STRIDE = 0x607C
    PPU_DATA_FORMAT = 0x6084
    PPU_MISC_CTRL = 0x60DC

    # --- PPU read DMA ---
    PPU_RDMA_S_POINTER = 0x7004
    PPU_RDMA_OPERATION_ENABLE = 0x7008
    PPU_RDMA_CUBE_IN_WIDTH = 0x700C
    PPU_RDMA_CUBE_IN_HEIGHT = 0x7010
    PPU_RDMA_CUBE_IN_CHANNEL = 0x7014
    PPU_RDMA_SRC_BASE_ADDR = 0x701C
    PPU_RDMA_SRC_LINE_STRIDE = 0x7024
    PPU_RDMA_SRC_SURF_STRIDE = 0x7028
    PPU_RDMA_DATA_FORMAT = 0x7030


POOL_AVERAGE = 0
POOL_MAXIMUM = 1
POOL_MINIMUM = 2

PRECISION_INT8 = 0
PRECISION_INT16 = 1
PRECISION_FP16 = 2
PRECISION_INT32 = 4
PRECISION_FP32 = 5

RDMA_WIDTH_INT8 = 1
RDMA_WIDTH_INT16 = 2
RDMA_WIDTH_INT32 = 3

POOL_ENABLE_MASK = 0x60
POOL_INT_MASK = 0xC00
POOL_TASK_OP_IDX = 1


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
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]


class drm_rocket_task(ctypes.Structure):
    _fields_ = [("regcmd", ctypes.c_uint32), ("regcmd_count", ctypes.c_uint32)]


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


def _IOW(type_, number, size):
    return (1 << 30) | (ord(type_) << 8) | number | (size << 16)


def _IOWR(type_, number, size):
    return (3 << 30) | (ord(type_) << 8) | number | (size << 16)


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
    create = drm_rocket_create_bo(size=size)
    ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, create)
    buf = mmap.mmap(
        fd,
        create.size,
        mmap.MAP_SHARED,
        mmap.PROT_READ | mmap.PROT_WRITE,
        offset=create.offset,
    )
    return buf, RocketBO(create.handle, create.size, create.dma_address, create.offset)


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


def fp17(value):
    """Encode a finite positive value in the PPU HLS FP17 format."""
    significand, exponent = math.frexp(float(value))
    unbiased_exponent = exponent - 1
    fraction = round((significand * 2.0 - 1.0) * 1024.0)
    if fraction == 1024:
        fraction = 0
        unbiased_exponent += 1
    encoded_exponent = unbiased_exponent + 31
    if not 0 < encoded_exponent < 0x3F:
        raise ValueError(f"FP17 value is out of normal range: {value}")
    return (encoded_exponent << 10) | fraction


def make_pool_regcmds(
    input_dma,
    output_dma,
    in_shape,
    out_shape,
    channels,
    method,
    precision,
    rdma_width,
    kernel,
    stride,
    padding=(0, 0, 0, 0),
    index_en=False,
):
    in_h, in_w = in_shape
    out_h, out_w = out_shape
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_top, pad_right, pad_bottom, pad_left = padding
    out_area = out_h * out_w
    index_add = out_area * 2 if index_en else out_area

    # POINTER_PP_MODE, EXECUTER_PP_EN, and POINTER_PP_EN.
    ppu_pointer = (1 << 3) | (1 << 2) | (1 << 1)
    operation_mode = (int(index_en) << 30) | (1 << 4) | method
    kernel_cfg = (
        ((stride_h - 1) << 20)
        | ((stride_w - 1) << 16)
        | ((kernel_h - 1) << 8)
        | (kernel_w - 1)
    )
    padding_cfg = (
        (pad_bottom << 12) | (pad_right << 8) | (pad_top << 4) | pad_left
    )

    commands = [
        emit(reg.PPU, reg.PPU_S_POINTER, ppu_pointer),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_S_POINTER, ppu_pointer),
        emit(reg.PPU, reg.PPU_DATA_CUBE_IN_WIDTH, in_w - 1),
        emit(reg.PPU, reg.PPU_DATA_CUBE_IN_HEIGHT, in_h - 1),
        emit(reg.PPU, reg.PPU_DATA_CUBE_IN_CHANNEL, channels - 1),
        emit(reg.PPU, reg.PPU_DATA_CUBE_OUT_WIDTH, out_w - 1),
        emit(reg.PPU, reg.PPU_DATA_CUBE_OUT_HEIGHT, out_h - 1),
        emit(reg.PPU, reg.PPU_DATA_CUBE_OUT_CHANNEL, channels - 1),
        emit(reg.PPU, reg.PPU_OPERATION_MODE_CFG, operation_mode),
        emit(reg.PPU, reg.PPU_POOLING_KERNEL_CFG, kernel_cfg),
    ]
    if method == POOL_AVERAGE:
        commands += [
            emit(reg.PPU, reg.PPU_RECIP_KERNEL_WIDTH, fp17(1.0 / kernel_w)),
            emit(reg.PPU, reg.PPU_RECIP_KERNEL_HEIGHT, fp17(1.0 / kernel_h)),
        ]
    commands += [
        emit(reg.PPU, reg.PPU_POOLING_PADDING_CFG, padding_cfg),
        emit(reg.PPU, reg.PPU_PADDING_VALUE_1_CFG, 0),
        emit(reg.PPU, reg.PPU_PADDING_VALUE_2_CFG, 0),
        emit(reg.PPU, reg.PPU_DST_BASE_ADDR, output_dma),
        emit(reg.PPU, reg.PPU_DST_SURF_STRIDE, out_area << 4),
        emit(reg.PPU, reg.PPU_DATA_FORMAT, (index_add << 4) | precision),
        emit(reg.PPU, reg.PPU_MISC_CTRL, 3),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_WIDTH, in_w - 1),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_HEIGHT, in_h - 1),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_CHANNEL, channels - 1),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_SRC_BASE_ADDR, input_dma),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_SRC_LINE_STRIDE, in_w << 4),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_SRC_SURF_STRIDE, in_h * in_w << 4),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_DATA_FORMAT, rdma_width),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_OPERATION_ENABLE, 1),
        # PC enables the known PPU + PPU_RDMA pair.
        emit(reg.PC, reg.PC_OPERATION_ENABLE, POOL_ENABLE_MASK),
    ]
    return commands


def run_pool_case(case):
    fd = open_rocket_device()
    task_map, task_bo = mem_allocate(fd, 4096)
    regcmd_map, regcmd_bo = mem_allocate(fd, 4096)
    input_map, input_bo = mem_allocate(fd, 4 * 1024 * 1024)
    output_map, output_bo = mem_allocate(fd, 4 * 1024 * 1024)

    dtype = np.dtype(case["dtype"])
    atom_channels = 16 // dtype.itemsize
    in_h, in_w = case["in_shape"]
    out_h, out_w = case["out_shape"]
    channels = case["channels"]
    source = np.asarray(case["source"], dtype=dtype).reshape(in_h, in_w, channels)
    packed_input = np.ndarray(
        (in_h, in_w, atom_channels), dtype=dtype, buffer=input_map
    )
    packed_input.fill(case.get("tail_value", 0))
    packed_input[:, :, :channels] = source
    output_map[:] = bytes(len(output_map))

    commands = make_pool_regcmds(
        input_bo.dma_addr,
        output_bo.dma_addr,
        case["in_shape"],
        case["out_shape"],
        channels,
        case["method"],
        case["precision"],
        case["rdma_width"],
        case["kernel"],
        case["stride"],
        case.get("padding", (0, 0, 0, 0)),
        case.get("index_en", False),
    )
    regcmds = (ctypes.c_uint64 * (regcmd_bo.size // 8)).from_buffer(regcmd_map)
    for index, command in enumerate(commands):
        regcmds[index] = command

    tasks = ctypes.cast(
        ctypes.addressof(ctypes.c_char.from_buffer(task_map)),
        ctypes.POINTER(struct_rknpu_task),
    )
    tasks[0].flags = 0
    tasks[0].op_idx = POOL_TASK_OP_IDX
    tasks[0].enable_mask = POOL_ENABLE_MASK
    tasks[0].int_mask = POOL_INT_MASK
    tasks[0].int_clear = 0x1FFFF
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(commands)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = regcmd_bo.dma_addr

    for bo in (regcmd_bo, input_bo, output_bo):
        ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))
    submit_ret = rocket_submit(
        fd,
        tasks,
        in_bos=[regcmd_bo, input_bo],
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
    packed_output = np.frombuffer(
        output_map, dtype=dtype, count=out_h * out_w * atom_channels
    ).reshape(out_h, out_w, atom_channels).copy()
    got = packed_output[:, :, :channels]
    expected = np.asarray(case["expected"], dtype=dtype).reshape(got.shape)

    if case.get("expect_unsupported", False):
        ok = submit_ret == 0 and np.all(got == 0) and np.any(expected != 0)
        detail = "writes zero: format unsupported" if ok else "unexpected result"
    else:
        tolerance = case.get("atol", 0.0)
        ok = submit_ret == 0 and np.allclose(got, expected, atol=tolerance, rtol=0)
        detail = f"max_abs_diff={np.max(np.abs(got.astype(np.float64) - expected)):.6g}"

    if case.get("index_en", False):
        index_offset = out_h * out_w * 2 * 16
        positions = np.frombuffer(output_map, dtype=np.uint16, count=channels, offset=index_offset).copy()
        expected_positions = np.asarray(case["expected_positions"], dtype=np.uint16)
        ok = ok and np.array_equal(positions, expected_positions)
        detail += f" positions={positions.tolist()}"

    if case.get("check_tail", False):
        tail = packed_output[:, :, channels:]
        expected_tail = np.asarray(case["expected_tail"], dtype=dtype)
        ok = ok and np.all(tail == expected_tail)
        detail += f" padded_tail={tail.reshape(-1).tolist()}"

    got_flat = got.reshape(-1)
    expected_flat = expected.reshape(-1)
    got_preview = got_flat[:8].tolist()
    expected_preview = expected_flat[:8].tolist()
    suffix = f" ... ({got.size} values)" if got.size > 8 else ""
    print(
        f"{case['name']}: submit={submit_ret} got={got_preview} "
        f"expected={expected_preview}{suffix} {detail} {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def pool_reference(source, kernel, stride, method, padding=(0, 0, 0, 0)):
    source = np.asarray(source)
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_top, pad_right, pad_bottom, pad_left = padding
    padded = np.pad(
        source,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        constant_values=0,
    )
    out_h = (padded.shape[0] - kernel_h) // stride_h + 1
    out_w = (padded.shape[1] - kernel_w) // stride_w + 1
    result = np.empty((out_h, out_w, source.shape[2]), dtype=source.dtype)
    for y in range(out_h):
        for x in range(out_w):
            window = padded[
                y * stride_h:y * stride_h + kernel_h,
                x * stride_w:x * stride_w + kernel_w,
            ]
            if method == POOL_MAXIMUM:
                result[y, x] = np.max(window, axis=(0, 1))
            elif method == POOL_MINIMUM:
                result[y, x] = np.min(window, axis=(0, 1))
            else:
                result[y, x] = np.mean(window.astype(np.float32), axis=(0, 1))
    return result.astype(source.dtype)


def cases():
    base = (np.arange(4 * 4 * 8, dtype=np.float16).reshape(4, 4, 8) - 32) / 8
    result = {}
    for name, method in (
        ("fp16_average", POOL_AVERAGE),
        ("fp16_maximum", POOL_MAXIMUM),
        ("fp16_minimum", POOL_MINIMUM),
    ):
        result[name] = {
            "name": name,
            "dtype": np.float16,
            "precision": PRECISION_FP16,
            "rdma_width": RDMA_WIDTH_INT16,
            "channels": 8,
            "in_shape": (4, 4),
            "out_shape": (3, 3),
            "kernel": (2, 2),
            "stride": (1, 1),
            "method": method,
            "source": base,
            "expected": pool_reference(base, (2, 2), (1, 1), method),
            "atol": 0.0625 if method == POOL_AVERAGE else 0,
        }

    global_source = np.arange(3 * 5 * 8, dtype=np.float16).reshape(3, 5, 8) / 8
    result["global_average_3x5"] = {
        "name": "global_average_3x5",
        "dtype": np.float16,
        "precision": PRECISION_FP16,
        "rdma_width": RDMA_WIDTH_INT16,
        "channels": 8,
        "in_shape": (3, 5),
        "out_shape": (1, 1),
        "kernel": (3, 5),
        "stride": (3, 5),
        "method": POOL_AVERAGE,
        "source": global_source,
        "expected": pool_reference(global_source, (3, 5), (3, 5), POOL_AVERAGE),
        "atol": 0.0625,
    }

    index_source = np.zeros((2, 2, 8), dtype=np.float16)
    positions = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
    for channel, position in enumerate(positions):
        index_source.reshape(4, 8)[position, channel] = 100 + channel
    result["maximum_indices"] = {
        "name": "maximum_indices",
        "dtype": np.float16,
        "precision": PRECISION_FP16,
        "rdma_width": RDMA_WIDTH_INT16,
        "channels": 8,
        "in_shape": (2, 2),
        "out_shape": (1, 1),
        "kernel": (2, 2),
        "stride": (2, 2),
        "method": POOL_MAXIMUM,
        "source": index_source,
        "expected": np.arange(100, 108, dtype=np.float16).reshape(1, 1, 8),
        "index_en": True,
        "expected_positions": [0, 1, 8, 9, 0, 1, 8, 9],
    }

    precision_cases = (
        ("int8_maximum", np.int8, PRECISION_INT8, RDMA_WIDTH_INT8, 16, False),
        ("int16_maximum", np.int16, PRECISION_INT16, RDMA_WIDTH_INT16, 8, False),
        ("int32_maximum", np.int32, PRECISION_INT32, RDMA_WIDTH_INT32, 4, True),
        ("fp32_maximum", np.float32, PRECISION_FP32, RDMA_WIDTH_INT32, 4, True),
    )
    for name, dtype, precision, rdma_width, channels, unsupported in precision_cases:
        source = np.arange(4 * channels, dtype=dtype).reshape(2, 2, channels) + 1
        result[name] = {
            "name": name,
            "dtype": dtype,
            "precision": precision,
            "rdma_width": rdma_width,
            "channels": channels,
            "in_shape": (2, 2),
            "out_shape": (1, 1),
            "kernel": (2, 2),
            "stride": (2, 2),
            "method": POOL_MAXIMUM,
            "source": source,
            "expected": np.max(source, axis=(0, 1), keepdims=True),
            "expect_unsupported": unsupported,
        }

    kernel16_source = np.arange(16 * 16 * 8, dtype=np.float16).reshape(16, 16, 8)
    result["kernel_stride_16"] = {
        "name": "kernel_stride_16",
        "dtype": np.float16,
        "precision": PRECISION_FP16,
        "rdma_width": RDMA_WIDTH_INT16,
        "channels": 8,
        "in_shape": (16, 16),
        "out_shape": (1, 1),
        "kernel": (16, 16),
        "stride": (16, 16),
        "method": POOL_MAXIMUM,
        "source": kernel16_source,
        "expected": np.max(kernel16_source, axis=(0, 1), keepdims=True),
    }

    pad_source = np.arange(1, 8 * 8 * 8 + 1, dtype=np.float16).reshape(8, 8, 8)
    result["padding_7"] = {
        "name": "padding_7",
        "dtype": np.float16,
        "precision": PRECISION_FP16,
        "rdma_width": RDMA_WIDTH_INT16,
        "channels": 8,
        "in_shape": (8, 8),
        "out_shape": (15, 15),
        "kernel": (8, 8),
        "stride": (1, 1),
        "padding": (7, 7, 7, 7),
        "method": POOL_MAXIMUM,
        "source": pad_source,
        "expected": pool_reference(
            pad_source, (8, 8), (1, 1), POOL_MAXIMUM, (7, 7, 7, 7)
        ),
    }

    channels_source = np.asarray(
        [[[1, 50, 3], [20, 2, 60]], [[4, 5, 6], [7, 8, 9]]],
        dtype=np.float16,
    )
    result["three_channels"] = {
        "name": "three_channels",
        "dtype": np.float16,
        "precision": PRECISION_FP16,
        "rdma_width": RDMA_WIDTH_INT16,
        "channels": 3,
        "in_shape": (2, 2),
        "out_shape": (1, 1),
        "kernel": (2, 2),
        "stride": (2, 2),
        "method": POOL_MAXIMUM,
        "source": channels_source,
        "expected": [[[20, 50, 60]]],
        "tail_value": -99,
        "check_tail": True,
        "expected_tail": -99,
    }
    return result


def print_register_surface():
    print("PPU selector surface: average=0, maximum=1, minimum=2")
    print("INDEX_EN: local pool coordinate only; observed encoding=(row << 3) | column")
    print("No PPU selector: general ArgMax, sort, gather/scatter, WHERE, IEEE, int bitops")
    print("DPU_RDMA, not PPU: unpool/upsample register exists; working PoC unresolved")
    print("PPU flying=1 external RDMA path: exercised here; DPU-flying path: not submitted")


def main():
    available = cases()
    selected = sys.argv[1] if len(sys.argv) > 1 else "all"
    if selected == "--list":
        print("\n".join(available))
        return 0
    if selected == "--surface":
        print_register_surface()
        return 0
    if selected != "all":
        if selected not in available:
            raise SystemExit(f"unknown case {selected!r}; use --list")
        return run_pool_case(available[selected])

    print_register_surface()
    status = 0
    for name in available:
        process = subprocess.run([sys.executable, __file__, name], check=False)
        status |= process.returncode
    print("PPU POOLING MATRIX PASS" if status == 0 else "PPU POOLING MATRIX FAIL")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
