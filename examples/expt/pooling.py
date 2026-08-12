"""Exercise the RK3588 PPU pooling surface with decoded register writes.

Hardware results on RK3588:
  * Average, maximum, and minimum pooling are native PPU operations.
  * Global average pooling needs dimension-specific FP17 reciprocals; it does
    not need a host-side divide.
  * INDEX_EN writes three row and three column bits.  Coordinates alias modulo
    eight, and padding+stride does not preserve a simple window-local phase.
  * FP16 works.  PPU_RDMA_DATA_FORMAT selects storage width, not the PPU
    arithmetic dtype: 0/1/2/3 mean 4/8/16/32-bit input.  Signed INT8 and
    signed INT16 maximum and minimum pooling pass with widths 1 and 2.
    Integer average is exact with Q16 reciprocals; the FP16 reciprocal recipe
    is retained as an inexact negative control.
  * FP16 negative maximum pooling uses a low-halfword 0xFC00 padding value.
  * USE_CNT values 0 through 7 all retain include-padding average semantics
    in the distinguishable padded-average probe.
  * External 32-bit input uses RDMA width 3.  Tested PPU PROC_PRECISION 4/5
    combinations submit but write zero.
  * Kernel/stride 16, padding 7, and non-multiple channel counts work in the
    isolated probes below.  Pixels are still stored in 16-byte atoms.
  * Three fenced NPU-written passes reduce 8x8 to 1x1 for avg/max/min exactly.
  * Explicit PPU_RDMA line/surface strides work; nonzero notch is not an
    additive line-pitch control in that recipe.

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

# Despite the field name IN_PRECISION, RK3588 TRM 36.6.8 defines storage width:
# 0=4-bit, 1=8-bit, 2=16-bit, and 3=32-bit.
PPU_RDMA_WIDTH_INT4 = 0
PPU_RDMA_WIDTH_INT8 = 1
PPU_RDMA_WIDTH_INT16 = 2
PPU_RDMA_WIDTH_INT32 = 3

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
    output_precision,
    rdma_width,
    kernel,
    stride,
    padding=(0, 0, 0, 0),
    index_en=False,
    use_cnt=0,
    padding_value=(0, 0),
    notch_addr=0,
    recip_override=None,
    rdma_line_stride=None,
    rdma_surface_stride=None,
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
    if not 0 <= use_cnt <= 7:
        raise ValueError("PPU USE_CNT must fit its three-bit field")
    if not 0 <= notch_addr <= 0x1FFF:
        raise ValueError("PPU NOTCH_ADDR must fit its thirteen-bit field")
    operation_mode = (
        (int(index_en) << 30)
        | (notch_addr << 16)
        | (use_cnt << 5)
        | (1 << 4)
        | method
    )
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
        reciprocal_width, reciprocal_height = (
            recip_override
            if recip_override is not None
            else (fp17(1.0 / kernel_w), fp17(1.0 / kernel_h))
        )
        commands += [
            emit(reg.PPU, reg.PPU_RECIP_KERNEL_WIDTH, reciprocal_width),
            emit(reg.PPU, reg.PPU_RECIP_KERNEL_HEIGHT, reciprocal_height),
        ]
    commands += [
        emit(reg.PPU, reg.PPU_POOLING_PADDING_CFG, padding_cfg),
        emit(reg.PPU, reg.PPU_PADDING_VALUE_1_CFG, padding_value[0]),
        emit(reg.PPU, reg.PPU_PADDING_VALUE_2_CFG, padding_value[1]),
        emit(reg.PPU, reg.PPU_DST_BASE_ADDR, output_dma),
        emit(reg.PPU, reg.PPU_DST_SURF_STRIDE, out_area << 4),
        emit(reg.PPU, reg.PPU_DATA_FORMAT, (index_add << 4) | output_precision),
        emit(reg.PPU, reg.PPU_MISC_CTRL, 3),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_WIDTH, in_w - 1),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_HEIGHT, in_h - 1),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_CUBE_IN_CHANNEL, channels - 1),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_SRC_BASE_ADDR, input_dma),
        emit(
            reg.PPU_RDMA,
            reg.PPU_RDMA_SRC_LINE_STRIDE,
            rdma_line_stride if rdma_line_stride is not None else in_w << 4,
        ),
        emit(
            reg.PPU_RDMA,
            reg.PPU_RDMA_SRC_SURF_STRIDE,
            rdma_surface_stride
            if rdma_surface_stride is not None
            else in_h * in_w << 4,
        ),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_DATA_FORMAT, rdma_width),
        emit(reg.PPU_RDMA, reg.PPU_RDMA_OPERATION_ENABLE, 1),
        # PC enables the known PPU + PPU_RDMA pair.
        emit(reg.PC, reg.PC_OPERATION_ENABLE, POOL_ENABLE_MASK),
    ]
    return commands


def _pack_cube(buffer, source, dtype, tail_value=0):
    """Pack HWC values into the PPU's C1HWC2 surface order."""
    source = np.asarray(source, dtype=dtype)
    in_h, in_w, channels = source.shape
    atom_channels = 16 // np.dtype(dtype).itemsize
    surfaces = (channels + atom_channels - 1) // atom_channels
    packed = np.ndarray(
        (surfaces, in_h, in_w, atom_channels), dtype=dtype, buffer=buffer
    )
    packed.fill(tail_value)
    for channel in range(channels):
        packed[channel // atom_channels, :, :, channel % atom_channels] = source[:, :, channel]
    return packed


def _pack_cube_strided(
    buffer,
    source,
    dtype,
    line_stride,
    surface_stride,
    tail_value=0,
):
    """Pack HWC data with explicit PPU_RDMA line and surface byte strides."""
    source = np.asarray(source, dtype=dtype)
    in_h, in_w, channels = source.shape
    itemsize = np.dtype(dtype).itemsize
    atom_channels = 16 // itemsize
    surfaces = (channels + atom_channels - 1) // atom_channels
    if line_stride < in_w * 16 or line_stride & 0xF:
        raise ValueError("PPU line stride must be aligned and cover the row")
    if surface_stride < in_h * line_stride or surface_stride & 0xF:
        raise ValueError("PPU surface stride must cover all rows and be aligned")
    np.frombuffer(buffer, dtype=np.uint8).fill(0)
    for channel in range(channels):
        surface = channel // atom_channels
        atom_channel = channel % atom_channels
        for y in range(in_h):
            for x in range(in_w):
                offset = (
                    surface * surface_stride
                    + y * line_stride
                    + x * 16
                    + atom_channel * itemsize
                )
                np.frombuffer(
                    buffer,
                    dtype=dtype,
                    count=1,
                    offset=offset,
                )[0] = source[y, x, channel]
    return surfaces


def _unpack_cube(buffer, shape, channels, dtype):
    """Read a C1HWC2 PPU result back as logical HWC values."""
    out_h, out_w = shape
    atom_channels = 16 // np.dtype(dtype).itemsize
    surfaces = (channels + atom_channels - 1) // atom_channels
    packed = np.frombuffer(
        buffer,
        dtype=dtype,
        count=surfaces * out_h * out_w * atom_channels,
    ).reshape(surfaces, out_h, out_w, atom_channels).copy()
    output = np.empty((out_h, out_w, channels), dtype=dtype)
    for channel in range(channels):
        output[:, :, channel] = packed[
            channel // atom_channels, :, :, channel % atom_channels
        ]
    return output, packed


def run_pool_case(case):
    fd = open_rocket_device()
    task_map, task_bo = mem_allocate(fd, 4096)
    regcmd_map, regcmd_bo = mem_allocate(fd, 4096)
    input_map, input_bo = mem_allocate(fd, 4 * 1024 * 1024)
    output_map, output_bo = mem_allocate(fd, 4 * 1024 * 1024)

    input_dtype = np.dtype(case.get("input_dtype", case["dtype"]))
    output_dtype = np.dtype(case.get("output_dtype", case["dtype"]))
    in_h, in_w = case["in_shape"]
    out_h, out_w = case["out_shape"]
    channels = case["channels"]
    source = np.asarray(case["source"], dtype=input_dtype).reshape(
        in_h, in_w, channels
    )
    if "rdma_line_stride" in case or "rdma_surface_stride" in case:
        line_stride = case.get("rdma_line_stride", in_w << 4)
        surface_stride = case.get("rdma_surface_stride", in_h * line_stride)
        _pack_cube_strided(
            input_map,
            source,
            input_dtype,
            line_stride,
            surface_stride,
            case.get("tail_value", 0),
        )
    else:
        _pack_cube(input_map, source, input_dtype, case.get("tail_value", 0))
    output_map[:] = bytes(len(output_map))

    commands = make_pool_regcmds(
        input_bo.dma_addr,
        output_bo.dma_addr,
        case["in_shape"],
        case["out_shape"],
        channels,
        case["method"],
        case["output_precision"],
        case["rdma_width"],
        case["kernel"],
        case["stride"],
        case.get("padding", (0, 0, 0, 0)),
        case.get("index_en", False),
        case.get("use_cnt", 0),
        case.get("padding_value", (0, 0)),
        case.get("notch_addr", 0),
        case.get("recip_override"),
        case.get("rdma_line_stride"),
        case.get("rdma_surface_stride"),
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
    got, packed_output = _unpack_cube(
        output_map, case["out_shape"], channels, output_dtype
    )
    expected = np.asarray(case["expected"], dtype=output_dtype).reshape(got.shape)

    if "expected_inexact" in case:
        inexact = np.asarray(case["expected_inexact"], dtype=output_dtype).reshape(
            got.shape
        )
        ok = submit_ret == 0 and np.array_equal(got, inexact)
        exact_diff = np.max(np.abs(got.astype(np.int64) - expected.astype(np.int64)))
        detail = f"captured_inexact_integer_average max_exact_diff={exact_diff}"
    elif case.get("expect_unsupported", False):
        ok = submit_ret == 0 and np.all(got == 0) and np.any(expected != 0)
        detail = "writes zero: format unsupported" if ok else "unexpected result"
    else:
        tolerance = case.get("atol", 0.0)
        ok = submit_ret == 0 and np.allclose(got, expected, atol=tolerance, rtol=0)
        detail = f"max_abs_diff={np.max(np.abs(got.astype(np.float64) - expected)):.6g}"
        for alternate_name, alternate_values in case.get("alternate_expected", {}).items():
            alternate = np.asarray(alternate_values, dtype=output_dtype).reshape(got.shape)
            if submit_ret == 0 and np.allclose(got, alternate, atol=tolerance, rtol=0):
                ok = True
                detail = f"matched={alternate_name}"
                break

    if case.get("index_en", False):
        index_offset = out_h * out_w * 2 * 16
        index_buffer = memoryview(output_map)[index_offset:]
        positions, _ = _unpack_cube(
            index_buffer,
            case["out_shape"],
            channels,
            np.uint16,
        )
        expected_positions = np.asarray(
            case["expected_positions"], dtype=np.uint16
        ).reshape(positions.shape)
        ok = ok and np.array_equal(positions, expected_positions)
        detail += f" positions={positions.reshape(-1).tolist()}"

    if case.get("check_tail", False):
        atom_channels = 16 // output_dtype.itemsize
        used_in_last_surface = channels % atom_channels
        if used_in_last_surface == 0:
            tail = packed_output[:0]
        else:
            tail = packed_output[-1, :, :, used_in_last_surface:]
        expected_tail = np.asarray(case["expected_tail"], dtype=output_dtype)
        ok = ok and np.all(tail == expected_tail)
        detail += f" padded_tail={tail.reshape(-1).tolist()}"

    got_flat = got.reshape(-1)
    expected_flat = expected.reshape(-1)
    preview_count = case.get("preview_count", 8)
    got_preview = got_flat[:preview_count].tolist()
    expected_preview = expected_flat[:preview_count].tolist()
    suffix = f" ... ({got.size} values)" if got.size > preview_count else ""
    print(
        f"{case['name']}: submit={submit_ret} got={got_preview} "
        f"expected={expected_preview}{suffix} {detail} {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def _write_pool_task(task_map, regcmd_map, regcmd_bo, commands):
    """Write one known one-task Rocket PPU submission descriptor."""
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
    return tasks


def run_multi_pass_pool(method):
    """Chain three fenced PPU passes through NPU-written intermediate BOs."""
    fd = open_rocket_device()
    source = (
        np.arange(8 * 8 * 8, dtype=np.float16).reshape(8, 8, 8) - 200
    ) / 8
    input_map, input_bo = mem_allocate(fd, 4 * 1024 * 1024)
    _pack_cube(input_map, source, np.float16)
    pass_shapes = (((8, 8), (4, 4)), ((4, 4), (2, 2)), ((2, 2), (1, 1)))
    output_allocations = [mem_allocate(fd, 4 * 1024 * 1024) for _ in pass_shapes]
    task_allocations = [mem_allocate(fd, 4096) for _ in pass_shapes]
    regcmd_allocations = [mem_allocate(fd, 4096) for _ in pass_shapes]
    input_bos = [input_bo] + [allocation[1] for allocation in output_allocations[:-1]]
    submit_results = []
    try:
        for pass_index, ((in_shape, out_shape), input_pass_bo) in enumerate(
            zip(pass_shapes, input_bos)
        ):
            output_map, output_bo = output_allocations[pass_index]
            task_map, _ = task_allocations[pass_index]
            regcmd_map, regcmd_bo = regcmd_allocations[pass_index]
            output_map[:] = bytes(len(output_map))
            commands = make_pool_regcmds(
                input_pass_bo.dma_addr,
                output_bo.dma_addr,
                in_shape,
                out_shape,
                8,
                method,
                PRECISION_FP16,
                PPU_RDMA_WIDTH_INT16,
                (2, 2),
                (2, 2),
            )
            tasks = _write_pool_task(task_map, regcmd_map, regcmd_bo, commands)
            for bo in (regcmd_bo, input_pass_bo, output_bo):
                ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))
            submit_ret = rocket_submit(
                fd,
                tasks,
                in_bos=[regcmd_bo, input_pass_bo],
                out_bos=[output_bo],
            )
            submit_results.append(submit_ret)
            ioctl(
                fd,
                DRM_IOCTL_ROCKET_PREP_BO,
                drm_rocket_prep_bo(
                    handle=output_bo.handle,
                    timeout_ns=time.monotonic_ns() + 6_000_000_000,
                ),
            )
        final_map, _ = output_allocations[-1]
        got, _ = _unpack_cube(final_map, (1, 1), 8, np.float16)
    finally:
        os.close(fd)

    expected = source
    for _ in pass_shapes:
        expected = pool_reference(expected, (2, 2), (2, 2), method)
    tolerance = 0.0625 if method == POOL_AVERAGE else 0.0
    passed = all(result == 0 for result in submit_results) and np.allclose(
        got, expected, atol=tolerance, rtol=0
    )
    method_name = {
        POOL_AVERAGE: "average",
        POOL_MAXIMUM: "maximum",
        POOL_MINIMUM: "minimum",
    }[method]
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32))))
    print(
        f"multi_pass_{method_name}: submits={submit_results} got={got.reshape(-1)} "
        f"expected={expected.reshape(-1)} max_diff={max_diff:.6g} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


def validate_multi_pass_streams():
    """Validate all three decoded pass geometries for every pool selector."""
    checked = 0
    for method in (POOL_AVERAGE, POOL_MAXIMUM, POOL_MINIMUM):
        for in_shape, out_shape in (
            ((8, 8), (4, 4)),
            ((4, 4), (2, 2)),
            ((2, 2), (1, 1)),
        ):
            commands = make_pool_regcmds(
                0x10000000,
                0x10010000,
                in_shape,
                out_shape,
                8,
                method,
                PRECISION_FP16,
                PPU_RDMA_WIDTH_INT16,
                (2, 2),
                (2, 2),
            )
            decoded = {
                ((command >> 48) & 0xFFFF, command & 0xFFFF):
                (command >> 16) & 0xFFFFFFFF
                for command in commands
            }
            assert decoded[(reg.PPU, reg.PPU_DATA_CUBE_IN_WIDTH)] == in_shape[1] - 1
            assert decoded[(reg.PPU, reg.PPU_DATA_CUBE_OUT_WIDTH)] == out_shape[1] - 1
            assert decoded[(reg.PPU_RDMA, reg.PPU_RDMA_SRC_LINE_STRIDE)] == in_shape[1] << 4
            assert decoded[(reg.PPU, reg.PPU_DST_SURF_STRIDE)] == out_shape[0] * out_shape[1] << 4
            checked += 1
    print(f"offline multi-pass pooling validation PASS ({checked} streams)")
    return 0


def pool_reference(
    source, kernel, stride, method, padding=(0, 0, 0, 0), padding_value=0
):
    source = np.asarray(source)
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_top, pad_right, pad_bottom, pad_left = padding
    padded = np.pad(
        source,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        constant_values=padding_value,
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


def maximum_with_index_reference(
    source,
    kernel,
    stride,
    padding=(0, 0, 0, 0),
    padding_value=-np.inf,
):
    """Return maximum values and local PPU `(row << 3) | column` indices."""
    source = np.asarray(source)
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_top, pad_right, pad_bottom, pad_left = padding
    padded = np.pad(
        source,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        constant_values=padding_value,
    )
    out_h = (padded.shape[0] - kernel_h) // stride_h + 1
    out_w = (padded.shape[1] - kernel_w) // stride_w + 1
    values = np.empty((out_h, out_w, source.shape[2]), dtype=source.dtype)
    positions = np.empty((out_h, out_w, source.shape[2]), dtype=np.uint16)
    for y in range(out_h):
        for x in range(out_w):
            window = padded[
                y * stride_h:y * stride_h + kernel_h,
                x * stride_w:x * stride_w + kernel_w,
            ]
            for channel in range(source.shape[2]):
                flat_position = int(np.argmax(window[:, :, channel]))
                row, column = divmod(flat_position, kernel_w)
                values[y, x, channel] = window[row, column, channel]
                positions[y, x, channel] = (row << 3) | column
    return values, positions


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
            "output_precision": PRECISION_FP16,
            "rdma_width": PPU_RDMA_WIDTH_INT16,
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
        "output_precision": PRECISION_FP16,
        "rdma_width": PPU_RDMA_WIDTH_INT16,
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
        "output_precision": PRECISION_FP16,
        "rdma_width": PPU_RDMA_WIDTH_INT16,
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

    padded_index_source = (
        np.arange(4 * 4 * 8, dtype=np.float16).reshape(4, 4, 8) + 1
    )
    padded_index_values, padded_index_positions = maximum_with_index_reference(
        padded_index_source,
        (3, 3),
        (2, 2),
        (1, 1, 1, 1),
    )
    result["maximum_indices_padding_stride"] = {
        "name": "maximum_indices_padding_stride",
        "dtype": np.float16,
        "output_precision": PRECISION_FP16,
        "rdma_width": PPU_RDMA_WIDTH_INT16,
        "channels": 8,
        "in_shape": (4, 4),
        "out_shape": (2, 2),
        "kernel": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1, 1, 1),
        "padding_value": (0x0000FC00, 0),
        "method": POOL_MAXIMUM,
        "source": padded_index_source,
        "expected": padded_index_values,
        "index_en": True,
        # Values are ordinary local maxima, but the index row phase is not
        # reset for the lower output row in this padded/strided recipe.
        "expected_positions": np.asarray(
            [18] * 16 + [2] * 16, dtype=np.uint16
        ).reshape(2, 2, 8),
        "reference_positions": padded_index_positions,
        "index_limitation": "vertical phase is not window-local",
        "probe_only": True,
    }

    index16_source = np.zeros((16, 16, 8), dtype=np.float16)
    index16_coordinates = (
        (0, 7),
        (0, 8),
        (7, 15),
        (8, 0),
        (8, 8),
        (15, 7),
        (15, 8),
        (15, 15),
    )
    for channel, (row, column) in enumerate(index16_coordinates):
        index16_source[row, column, channel] = 100 + channel
    index16_values, index16_positions = maximum_with_index_reference(
        index16_source,
        (16, 16),
        (16, 16),
    )
    result["maximum_indices_kernel16"] = {
        "name": "maximum_indices_kernel16",
        "dtype": np.float16,
        "output_precision": PRECISION_FP16,
        "rdma_width": PPU_RDMA_WIDTH_INT16,
        "channels": 8,
        "in_shape": (16, 16),
        "out_shape": (1, 1),
        "kernel": (16, 16),
        "stride": (16, 16),
        "method": POOL_MAXIMUM,
        "source": index16_source,
        "expected": index16_values,
        "index_en": True,
        # INDEX_EN has only three row and three column bits.  Coordinates
        # beyond seven therefore alias modulo eight.
        "expected_positions": np.asarray(
            [((row & 7) << 3) | (column & 7)
             for row, column in index16_coordinates],
            dtype=np.uint16,
        ).reshape(1, 1, 8),
        "reference_positions": index16_positions,
        "index_limitation": "row/column coordinates alias modulo 8",
        "probe_only": True,
    }

    precision_cases = (
        ("int8_maximum", np.int8, PRECISION_INT8, PPU_RDMA_WIDTH_INT8, 16, False),
        ("int16_maximum", np.int16, PRECISION_INT16, PPU_RDMA_WIDTH_INT16, 8, False),
        ("int32_maximum", np.int32, PRECISION_INT32, PPU_RDMA_WIDTH_INT32, 4, True),
        ("fp32_maximum", np.float32, PRECISION_FP32, PPU_RDMA_WIDTH_INT32, 4, True),
    )
    for name, dtype, output_precision, rdma_width, channels, unsupported in precision_cases:
        lane = np.arange(channels, dtype=np.int32)
        source = np.stack(
            (-100 + lane, 50 - lane, -5 - lane, np.where(lane & 1, -120, 100 - lane))
        ).reshape(2, 2, channels).astype(dtype)
        result[name] = {
            "name": name,
            "dtype": dtype,
            "output_precision": output_precision,
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

    for prefix, dtype, output_precision, rdma_width, channels in (
        ("int8", np.int8, PRECISION_INT8, PPU_RDMA_WIDTH_INT8, 16),
        ("int16", np.int16, PRECISION_INT16, PPU_RDMA_WIDTH_INT16, 8),
    ):
        lane = np.arange(channels, dtype=np.int32)
        source = np.stack(
            (-40 + lane, 20 + lane, -8 + lane, 28 + lane)
        ).reshape(2, 2, channels).astype(dtype)
        for method_name, method in (
            ("average", POOL_AVERAGE),
            ("minimum", POOL_MINIMUM),
        ):
            name = f"{prefix}_{method_name}"
            result[name] = {
                "name": name,
                "dtype": dtype,
                "output_precision": output_precision,
                "rdma_width": rdma_width,
                "channels": channels,
                "in_shape": (2, 2),
                "out_shape": (1, 1),
                "kernel": (2, 2),
                "stride": (2, 2),
                "method": method,
                "source": source,
                "expected": pool_reference(source, (2, 2), (2, 2), method),
                "preview_count": channels if method == POOL_AVERAGE else 8,
            }
            if method == POOL_AVERAGE:
                result[name]["expected_inexact"] = np.asarray(
                    [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13][
                        :channels
                    ],
                    dtype=dtype,
                ).reshape(1, 1, channels)

    for prefix, dtype, output_precision, rdma_width, channels in (
        ("int8", np.int8, PRECISION_INT8, PPU_RDMA_WIDTH_INT8, 16),
        ("int16", np.int16, PRECISION_INT16, PPU_RDMA_WIDTH_INT16, 8),
    ):
        lane = np.arange(channels, dtype=np.int32)
        source = np.stack(
            (-40 + lane, 20 + lane, -8 + lane, 28 + lane)
        ).reshape(2, 2, channels).astype(dtype)
        name = f"{prefix}_average_recip_q16"
        result[name] = {
            "name": name,
            "dtype": dtype,
            "output_precision": output_precision,
            "rdma_width": rdma_width,
            "channels": channels,
            "in_shape": (2, 2),
            "out_shape": (1, 1),
            "kernel": (2, 2),
            "stride": (2, 2),
            "method": POOL_AVERAGE,
            "source": source,
            "expected": pool_reference(source, (2, 2), (2, 2), POOL_AVERAGE),
            # Integer PPU arithmetic follows the TRM's Q16 reciprocal wording.
            "recip_override": (0x8000, 0x8000),
            "preview_count": channels,
            "probe_only": True,
        }

        for geometry_name, in_shape, pattern in (
            ("3x3", (3, 3), (-4, -3, -2, -1, 0, 1, 2, 3, 4)),
            ("2x3", (2, 3), (-3, -2, -1, 1, 2, 3)),
        ):
            means = np.arange(channels, dtype=np.int32) - channels // 2
            source = (
                np.asarray(pattern, dtype=np.int32)[:, None] + means[None, :]
            ).reshape(in_shape[0], in_shape[1], channels).astype(dtype)
            kernel_h, kernel_w = in_shape
            name = f"{prefix}_average_recip_q16_{geometry_name}"
            result[name] = {
                "name": name,
                "dtype": dtype,
                "output_precision": output_precision,
                "rdma_width": rdma_width,
                "channels": channels,
                "in_shape": in_shape,
                "out_shape": (1, 1),
                "kernel": in_shape,
                "stride": in_shape,
                "method": POOL_AVERAGE,
                "source": source,
                "expected": means.astype(dtype).reshape(1, 1, channels),
                "recip_override": (
                    round(65536 / kernel_w),
                    round(65536 / kernel_h),
                ),
                "preview_count": channels,
                "probe_only": True,
            }

    # The vendor ChannelTile capture's 0x7030=1 means an 8-bit RDMA payload,
    # matching PPU PROC_PRECISION=0 (INT8); it is not an INT16-to-INT8 cast.
    vendor_token_source = np.arange(-16, 16, dtype=np.int8).reshape(1, 1, 32)
    result["vendor_int8_1x1_c32"] = {
        "name": "vendor_int8_1x1_c32",
        "dtype": np.int8,
        "output_precision": PRECISION_INT8,
        "rdma_width": PPU_RDMA_WIDTH_INT8,
        "channels": 32,
        "in_shape": (1, 1),
        "out_shape": (1, 1),
        "kernel": (1, 1),
        "stride": (1, 1),
        "method": POOL_MAXIMUM,
        "source": vendor_token_source,
        "expected": vendor_token_source,
    }

    kernel16_source = np.arange(16 * 16 * 8, dtype=np.float16).reshape(16, 16, 8)
    result["kernel_stride_16"] = {
        "name": "kernel_stride_16",
        "dtype": np.float16,
        "output_precision": PRECISION_FP16,
        "rdma_width": PPU_RDMA_WIDTH_INT16,
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
        "output_precision": PRECISION_FP16,
        "rdma_width": PPU_RDMA_WIDTH_INT16,
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

    negative_pad_source = -np.arange(1, 2 * 2 * 8 + 1, dtype=np.float16).reshape(2, 2, 8)
    result["negative_max_padding_neg_inf"] = {
        "name": "negative_max_padding_neg_inf",
        "dtype": np.float16,
        "output_precision": PRECISION_FP16,
        "rdma_width": PPU_RDMA_WIDTH_INT16,
        "channels": 8,
        "in_shape": (2, 2),
        "out_shape": (3, 3),
        "kernel": (2, 2),
        "stride": (1, 1),
        "padding": (1, 1, 1, 1),
        "padding_value": (0x0000FC00, 0),
        "method": POOL_MAXIMUM,
        "source": negative_pad_source,
        "expected": pool_reference(
            negative_pad_source,
            (2, 2),
            (1, 1),
            POOL_MAXIMUM,
            (1, 1, 1, 1),
            np.float16(-np.inf),
        ),
    }

    use_cnt_source = np.full((1, 1, 8), 8, dtype=np.float16)
    include_pad = np.full((2, 2, 8), 2, dtype=np.float16)
    exclude_pad = np.full((2, 2, 8), 8, dtype=np.float16)
    for use_cnt in range(8):
        name = f"average_padding_use_cnt_{use_cnt}"
        result[name] = {
            "name": name,
            "dtype": np.float16,
            "output_precision": PRECISION_FP16,
            "rdma_width": PPU_RDMA_WIDTH_INT16,
            "channels": 8,
            "in_shape": (1, 1),
            "out_shape": (2, 2),
            "kernel": (2, 2),
            "stride": (1, 1),
            "padding": (1, 1, 1, 1),
            "use_cnt": use_cnt,
            "method": POOL_AVERAGE,
            "source": use_cnt_source,
            "expected": include_pad,
            "alternate_expected": {"exclude_pad": exclude_pad},
            "atol": 0.0625,
            "probe_only": True,
        }

    channels_source = np.asarray(
        [[[1, 50, 3], [20, 2, 60]], [[4, 5, 6], [7, 8, 9]]],
        dtype=np.float16,
    )
    result["three_channels"] = {
        "name": "three_channels",
        "dtype": np.float16,
        "output_precision": PRECISION_FP16,
        "rdma_width": PPU_RDMA_WIDTH_INT16,
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

    strided_source = (
        np.arange(3 * 3 * 16, dtype=np.float16).reshape(3, 3, 16) - 40
    )
    for notch_addr in (0, 2):
        name = f"strided_input_notch_{notch_addr}"
        result[name] = {
            "name": name,
            "dtype": np.float16,
            "output_precision": PRECISION_FP16,
            "rdma_width": PPU_RDMA_WIDTH_INT16,
            "channels": 16,
            "in_shape": (3, 3),
            "out_shape": (2, 2),
            "kernel": (2, 2),
            "stride": (1, 1),
            "method": POOL_MAXIMUM,
            "source": strided_source,
            "expected": pool_reference(
                strided_source, (2, 2), (1, 1), POOL_MAXIMUM
            ),
            "rdma_line_stride": 5 << 4,
            "rdma_surface_stride": 16 << 4,
            "notch_addr": notch_addr,
            "probe_only": True,
        }
    return result


def print_register_surface():
    print("PPU selector surface: average=0, maximum=1, minimum=2")
    print("INDEX_EN: 3-bit row/column; aliases modulo 8 and padded/strided phase is not local")
    print("No PPU selector: general ArgMax, sort, gather/scatter, WHERE, IEEE, int bitops")
    print("DPU_RDMA, not PPU: unpool/upsample register exists; working PoC unresolved")
    print("PPU flying=1 external RDMA path: exercised here; DPU-flying path: not submitted")
    print("PPU_RDMA 0x7030 storage width: 4-bit=0, 8-bit=1, 16-bit=2, 32-bit=3")


def validate_decoded_streams():
    """Validate probe encodings and cube packing without opening the NPU device."""
    available = cases()
    for case in available.values():
        commands = make_pool_regcmds(
            0x10000000,
            0x10010000,
            case["in_shape"],
            case["out_shape"],
            case["channels"],
            case["method"],
            case["output_precision"],
            case["rdma_width"],
            case["kernel"],
            case["stride"],
            case.get("padding", (0, 0, 0, 0)),
            case.get("index_en", False),
            case.get("use_cnt", 0),
            case.get("padding_value", (0, 0)),
            case.get("notch_addr", 0),
            case.get("recip_override"),
            case.get("rdma_line_stride"),
            case.get("rdma_surface_stride"),
        )
        decoded = {
            ((command >> 48) & 0xFFFF, command & 0xFFFF):
            (command >> 16) & 0xFFFFFFFF
            for command in commands
        }
        assert decoded[(reg.PPU, reg.PPU_DATA_FORMAT)] & 0x7 == case["output_precision"]
        assert decoded[(reg.PPU_RDMA, reg.PPU_RDMA_DATA_FORMAT)] == case["rdma_width"]
        assert decoded[(reg.PPU_RDMA, reg.PPU_RDMA_SRC_LINE_STRIDE)] == case.get(
            "rdma_line_stride", case["in_shape"][1] * 16
        )
        assert decoded[(reg.PPU_RDMA, reg.PPU_RDMA_SRC_SURF_STRIDE)] == case.get(
            "rdma_surface_stride",
            case["in_shape"][0] * case["in_shape"][1] * 16,
        )
        assert decoded[(reg.PPU, reg.PPU_DST_SURF_STRIDE)] == case["out_shape"][0] * case["out_shape"][1] * 16
        assert decoded[(reg.PC, reg.PC_OPERATION_ENABLE)] == POOL_ENABLE_MASK
        if "expected_inexact" in case:
            assert not np.array_equal(case["expected_inexact"], case["expected"])

    vendor_token = available["vendor_int8_1x1_c32"]
    commands = make_pool_regcmds(
        0x10000000,
        0x10010000,
        vendor_token["in_shape"],
        vendor_token["out_shape"],
        vendor_token["channels"],
        vendor_token["method"],
        vendor_token["output_precision"],
        vendor_token["rdma_width"],
        vendor_token["kernel"],
        vendor_token["stride"],
    )
    decoded = {
        ((command >> 48) & 0xFFFF, command & 0xFFFF):
        (command >> 16) & 0xFFFFFFFF
        for command in commands
    }
    assert decoded[(reg.PPU, reg.PPU_DATA_CUBE_IN_CHANNEL)] == 31
    assert decoded[(reg.PPU, reg.PPU_DATA_CUBE_OUT_CHANNEL)] == 31
    assert decoded[(reg.PPU, reg.PPU_OPERATION_MODE_CFG)] == 0x11
    assert decoded[(reg.PPU, reg.PPU_DATA_FORMAT)] == 0x10
    assert decoded[(reg.PPU_RDMA, reg.PPU_RDMA_DATA_FORMAT)] == 1

    negative_pad = available["negative_max_padding_neg_inf"]
    commands = make_pool_regcmds(
        0x10000000,
        0x10010000,
        negative_pad["in_shape"],
        negative_pad["out_shape"],
        negative_pad["channels"],
        negative_pad["method"],
        negative_pad["output_precision"],
        negative_pad["rdma_width"],
        negative_pad["kernel"],
        negative_pad["stride"],
        negative_pad["padding"],
        padding_value=negative_pad["padding_value"],
    )
    decoded = {
        ((command >> 48) & 0xFFFF, command & 0xFFFF):
        (command >> 16) & 0xFFFFFFFF
        for command in commands
    }
    assert decoded[(reg.PPU, reg.PPU_PADDING_VALUE_1_CFG)] == 0xFC00
    assert decoded[(reg.PPU, reg.PPU_PADDING_VALUE_2_CFG)] == 0

    for use_cnt in range(8):
        case = available[f"average_padding_use_cnt_{use_cnt}"]
        commands = make_pool_regcmds(
            0x10000000,
            0x10010000,
            case["in_shape"],
            case["out_shape"],
            case["channels"],
            case["method"],
            case["output_precision"],
            case["rdma_width"],
            case["kernel"],
            case["stride"],
            case["padding"],
            use_cnt=use_cnt,
        )
        operation_mode = next(
            (command >> 16) & 0xFFFFFFFF
            for command in commands
            if ((command >> 48) & 0xFFFF, command & 0xFFFF)
            == (reg.PPU, reg.PPU_OPERATION_MODE_CFG)
        )
        assert (operation_mode >> 5) & 0x7 == use_cnt

    source = np.arange(-16, 16, dtype=np.int16).reshape(1, 1, 32)
    storage = bytearray(64)
    _pack_cube(storage, source, np.int16)
    unpacked, _ = _unpack_cube(storage, (1, 1), 32, np.int16)
    assert np.array_equal(unpacked, source)
    print(f"offline decoded-stream validation PASS ({len(available)} cases)")
    return 0


def main():
    available = cases()
    selected = sys.argv[1] if len(sys.argv) > 1 else "all"
    if selected == "--list":
        for name, case in available.items():
            suffix = " [isolated semantics probe]" if case.get("probe_only", False) else ""
            print(name + suffix)
        return 0
    if selected == "--validate":
        return validate_decoded_streams()
    if selected == "--validate-multi-pass":
        return validate_multi_pass_streams()
    if selected == "--surface":
        print_register_surface()
        return 0
    if selected in ("multi_pass_average", "multi_pass_maximum", "multi_pass_minimum"):
        return run_multi_pass_pool(
            {
                "multi_pass_average": POOL_AVERAGE,
                "multi_pass_maximum": POOL_MAXIMUM,
                "multi_pass_minimum": POOL_MINIMUM,
            }[selected]
        )
    if selected != "all":
        if selected not in available:
            raise SystemExit(f"unknown case {selected!r}; use --list")
        return run_pool_case(available[selected])

    print_register_surface()
    status = 0
    for name, case in available.items():
        if case.get("probe_only", False):
            continue
        process = subprocess.run([sys.executable, __file__, name], check=False)
        status |= process.returncode
    print("Opt-in isolated-semantics probes were not submitted")
    print("PPU POOLING MATRIX PASS" if status == 0 else "PPU POOLING MATRIX FAIL")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
