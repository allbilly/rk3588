from fcntl import ioctl
import os, mmap, sys
import ctypes
import numpy as np
import argparse


class reg:
    PC = 0x0100
    PPU = 0x4000
    PPU_RDMA = 0x8000
    REG_PC_OPERATION_ENABLE = 0x0008
    REG_PPU_S_POINTER = 0x6004
    REG_PPU_DATA_CUBE_IN_WIDTH = 0x600c
    REG_PPU_DATA_CUBE_IN_HEIGHT = 0x6010
    REG_PPU_DATA_CUBE_IN_CHANNEL = 0x6014
    REG_PPU_DATA_CUBE_OUT_WIDTH = 0x6018
    REG_PPU_DATA_CUBE_OUT_HEIGHT = 0x601c
    REG_PPU_DATA_CUBE_OUT_CHANNEL = 0x6020
    REG_PPU_OPERATION_MODE_CFG = 0x6024
    REG_PPU_POOLING_KERNEL_CFG = 0x6034
    REG_PPU_RECIP_KERNEL_WIDTH = 0x6038
    REG_PPU_RECIP_KERNEL_HEIGHT = 0x603c
    REG_PPU_DST_BASE_ADDR = 0x6070
    REG_PPU_DST_SURF_STRIDE = 0x607c
    REG_PPU_DATA_FORMAT = 0x6084
    REG_PPU_MISC_CTRL = 0x60dc
    REG_PPU_RDMA_RDMA_S_POINTER = 0x7004
    REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH = 0x700c
    REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT = 0x7010
    REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL = 0x7014
    REG_PPU_RDMA_RDMA_SRC_BASE_ADDR = 0x701c
    REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE = 0x7024
    REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE = 0x7028
    REG_PPU_RDMA_RDMA_DATA_FORMAT = 0x7030
    REG_PPU_RDMA_RDMA_OPERATION_ENABLE = 0x7038
    PPU_S_POINTER_POINTER_PP_MODE__SHIFT = 3
    PPU_S_POINTER_POINTER_PP_MODE__MASK = 0x8
    PPU_S_POINTER_EXECUTER_PP_EN__SHIFT = 2
    PPU_S_POINTER_EXECUTER_PP_EN__MASK = 0x4
    PPU_S_POINTER_POINTER_PP_EN__SHIFT = 1
    PPU_S_POINTER_POINTER_PP_EN__MASK = 0x2
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT = 3
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK = 0x8
    PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT = 2
    PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK = 0x4
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT = 1
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK = 0x2
    PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH__SHIFT = 0
    PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH__MASK = 0x1fff
    PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT__SHIFT = 0
    PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT__MASK = 0x1fff
    PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL__SHIFT = 0
    PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL__MASK = 0x1fff
    PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH__SHIFT = 0
    PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH__MASK = 0x1fff
    PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT__SHIFT = 0
    PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT__MASK = 0x1fff
    PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL__SHIFT = 0
    PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL__MASK = 0x1fff
    PPU_OPERATION_MODE_CFG_FLYING_MODE__SHIFT = 4
    PPU_OPERATION_MODE_CFG_FLYING_MODE__MASK = 0x10
    PPU_OPERATION_MODE_CFG_POOLING_METHOD__SHIFT = 0
    PPU_OPERATION_MODE_CFG_POOLING_METHOD__MASK = 0x3
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT__SHIFT = 20
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT__MASK = 0x00f00000
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH__SHIFT = 16
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH__MASK = 0x000f0000
    PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT__SHIFT = 8
    PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT__MASK = 0x00000f00
    PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH__SHIFT = 0
    PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH__MASK = 0x0000000f
    PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH__SHIFT = 0
    PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH__MASK = 0x0001ffff
    PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT__SHIFT = 0
    PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT__MASK = 0x0001ffff
    PPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT = 4
    PPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK = 0xfffffff0
    PPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT = 0
    PPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK = 0x0001ffff
    PPU_DATA_FORMAT_INDEX_ADD__SHIFT = 16
    PPU_DATA_FORMAT_INDEX_ADD__MASK = 0x1fff0000
    PPU_DATA_FORMAT_PROC_PRECISION__SHIFT = 0
    PPU_DATA_FORMAT_PROC_PRECISION__MASK = 0x00000003
    PPU_MISC_CTRL_BURST_LEN__SHIFT = 0
    PPU_MISC_CTRL_BURST_LEN__MASK = 0x0000000f
    PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH__SHIFT = 0
    PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH__MASK = 0x1fff
    PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT__SHIFT = 0
    PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT__MASK = 0x1fff
    PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL__SHIFT = 0
    PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL__MASK = 0x1fff
    PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT = 0
    PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK = 0xffffffff
    PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE__SHIFT = 0
    PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE__MASK = 0xffffffff
    PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE__SHIFT = 0
    PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE__MASK = 0xffffffff
    PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION__SHIFT = 0
    PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION__MASK = 0x3
    PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN__SHIFT = 0
    PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN__MASK = 0x1
    PC_OPERATION_ENABLE_RESERVED_0__SHIFT = 1
    PC_OPERATION_ENABLE_RESERVED_0__MASK = 0xfffffffe


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0x2
RKNPU_JOB_PINGPONG = 0x4

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


class rknpu_mem_destroy(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("obj_addr", ctypes.c_uint64),
    ]


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


DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_MEM_DESTROY = _IOWR("d", 0x44, ctypes.sizeof(rknpu_mem_destroy))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR("d", 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR("d", 0x40, ctypes.sizeof(rknpu_action))


def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(
            flags=flags,
            size=size,
    )
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    print(f"ret={ret}, handle={mem_create.handle}, obj_addr={mem_create.obj_addr:#x}, dma_addr={mem_create.dma_addr:#x}")
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    print(f"Memory mapped at offset={mem_map.offset:#x}")
    return buf, mem_create


def mem_destroy(fd, mem_create):
    return ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY,
                              rknpu_mem_destroy(handle=mem_create.handle, obj_addr=mem_create.obj_addr))


def submit(fd, task_obj_addr):
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
POOL_OPS = ("min", "max", "avg", "globalmin", "globalmax", "globalavg")
POOL_ENABLE_MASK = 0x60
POOL_INT_MASK = 0xc00
POOL_TASK_OP_IDX = 1
POOL_PC_ENABLE = 48
POOL_CHANNELS = 8
def mask_reg(val, shift, mask):
    return (int(val) << shift) & mask


def emit(target, addr, value):
    if target == reg.PC: target = 0x80
    return (((target + 1) & 0xffff) << 48) | ((int(value) & 0xffffffff) << 16) | (addr & 0xffff)


def field(name, value):
    return mask_reg(value, getattr(reg, f"{name}__SHIFT"), getattr(reg, f"{name}__MASK"))


def pack_f16_strided(buf, vals, stride_fp16=8):
    arr = np.asarray(vals, dtype=np.float16).reshape(-1).view(np.uint16)
    dst = (ctypes.c_uint16 * (len(arr) * stride_fp16)).from_buffer(buf)
    for i, value in enumerate(arr):
        dst[i * stride_fp16] = int(value)


def read_f16_strided(buf, n, stride_fp16=8):
    raw = np.frombuffer(buf, dtype=np.float16, count=n * stride_fp16)
    return raw[::stride_fp16].copy()


def align_up(value, align):
    return ((int(value) + align - 1) // align) * align


def pool2d_reference(x, op):
    if op.startswith("global"):
        window = x
        if op == "globalmax": return np.max(window, axis=(0, 1), keepdims=True).astype(np.float16)
        if op == "globalmin": return np.min(window, axis=(0, 1), keepdims=True).astype(np.float16)
        return np.mean(window.astype(np.float32), axis=(0, 1), keepdims=True).astype(np.float16)

    out_h, out_w = x.shape[0] - 1, x.shape[1] - 1
    out = np.empty((out_h, out_w) + x.shape[2:], dtype=np.float16)
    for y in range(out_h):
        for x0 in range(out_w):
            window = x[y:y + 2, x0:x0 + 2]
            if op == "max":
                out[y, x0] = np.max(window, axis=(0, 1))
            elif op == "min":
                out[y, x0] = np.min(window, axis=(0, 1))
            else:
                out[y, x0] = np.mean(window.astype(np.float32), axis=(0, 1)).astype(np.float16)
    return out


def ppu_pointer():
    return (field("PPU_S_POINTER_POINTER_PP_MODE", 1) |
                    field("PPU_S_POINTER_EXECUTER_PP_EN", 1) |
                    field("PPU_S_POINTER_POINTER_PP_EN", 1))


def ppu_rdma_pointer():
    return (field("PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE", 1) |
                    field("PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN", 1) |
                    field("PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN", 1))


def kernel_cfg(stride_h=0, stride_w=0, kernel_h=1, kernel_w=1):
    return (field("PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT", stride_h) |
                    field("PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH", stride_w) |
                    field("PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT", kernel_h) |
                    field("PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH", kernel_w))


def pooling_regs(op, input_dma=0x11110000, output_dma=0x22220000, in_h=4, in_w=4):
    if op not in POOL_OPS:
        raise ValueError(f"unknown pool op {op!r}; expected one of {', '.join(POOL_OPS)}")
    if in_h < 2 or in_w < 2:
        raise ValueError(f"pool input must be at least 2x2, got {in_h}x{in_w}")

    hw_op = "max" if op in ("min", "globalmin") else op
    direct_global = op in ("globalmax", "globalavg")
    max_pool = hw_op in ("max", "globalmax")
    in_h_field = in_h - 1
    in_w_field = in_w - 1
    out_h_field = 0 if direct_global else in_h - 2
    out_w_field = 0 if direct_global else in_w - 2
    channel_field = POOL_CHANNELS - 1
    width_stride = in_w
    src_surf_stride = width_stride * in_h
    dst_surf_stride = 1 if direct_global else in_w * (in_h - 1)
    index_add = 1 if direct_global else dst_surf_stride
    k_h = in_h_field if direct_global else 1
    k_w = in_w_field if direct_global else 1
    s_h = in_h_field if direct_global else 0
    s_w = in_w_field if direct_global else 0

    regs = [
        emit(reg.PPU, reg.REG_PPU_S_POINTER, ppu_pointer()),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_S_POINTER, ppu_rdma_pointer()),
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_IN_WIDTH, field("PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH", in_w_field)),
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_IN_HEIGHT, field("PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT", in_h_field)),
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_IN_CHANNEL, field("PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL", channel_field)),
    ]
    if op != "globalmax":
        regs += [
            emit(reg.PPU, reg.REG_PPU_DATA_CUBE_OUT_WIDTH, field("PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH", out_w_field)),
            emit(reg.PPU, reg.REG_PPU_DATA_CUBE_OUT_HEIGHT, field("PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT", out_h_field)),
        ]
    regs += [
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_OUT_CHANNEL, field("PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL", channel_field)),
        emit(reg.PPU, reg.REG_PPU_OPERATION_MODE_CFG,
                  field("PPU_OPERATION_MODE_CFG_FLYING_MODE", 1) |
                  field("PPU_OPERATION_MODE_CFG_POOLING_METHOD", int(max_pool))),
        emit(reg.PPU, reg.REG_PPU_POOLING_KERNEL_CFG, kernel_cfg(s_h, s_w, k_h, k_w)),
    ]
    if not max_pool:
        regs += [
            emit(reg.PPU, reg.REG_PPU_RECIP_KERNEL_WIDTH,
                      field("PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH", 30720)),
            emit(reg.PPU, reg.REG_PPU_RECIP_KERNEL_HEIGHT,
                      field("PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT", 30720)),
        ]

    regs += [
        emit(reg.PPU, reg.REG_PPU_DST_BASE_ADDR, field("PPU_DST_BASE_ADDR_DST_BASE_ADDR", output_dma // 16)),
        emit(reg.PPU, reg.REG_PPU_DST_SURF_STRIDE, field("PPU_DST_SURF_STRIDE_DST_SURF_STRIDE", dst_surf_stride)),
        emit(reg.PPU, reg.REG_PPU_DATA_FORMAT,
                  field("PPU_DATA_FORMAT_INDEX_ADD", index_add) |
                  field("PPU_DATA_FORMAT_PROC_PRECISION", 2)),
        emit(reg.PPU, reg.REG_PPU_MISC_CTRL, field("PPU_MISC_CTRL_BURST_LEN", 3)),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH,
                  field("PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH", in_w_field)),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT,
                  field("PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT", in_h_field)),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL,
                  field("PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL", channel_field)),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_SRC_BASE_ADDR,
                  field("PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR", input_dma)),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE,
                  field("PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE", width_stride)),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE,
                  field("PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE", src_surf_stride)),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_DATA_FORMAT,
                  field("PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION", 2)),
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_OPERATION_ENABLE,
                  field("PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN", 1)),
        emit(reg.PC, reg.REG_PC_OPERATION_ENABLE,
                  field("PC_OPERATION_ENABLE_RESERVED_0", POOL_PC_ENABLE)),
    ]
    return regs


def run_pool(op, in_h=4, in_w=4):
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    maps = []
    mems = []
    try:
        reset_npu(fd)
        task_map, task_mc = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        regcmd_map, regcmd_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
        input_bytes = align_up(in_h * in_w * POOL_CHANNELS * np.dtype(np.float16).itemsize, 4096)
        out_h = 1 if op in ("globalmax", "globalavg") else in_h - 1
        out_w = 1 if op in ("globalmax", "globalavg") else in_w - 1
        if op == "globalmin":
            out_h, out_w = in_h - 1, in_w - 1
        output_bytes = align_up(max(1, out_h * out_w * POOL_CHANNELS * np.dtype(np.float16).itemsize), 4096)
        input_map, input_mc = mem_allocate(fd, input_bytes, RKNPU_MEM_NON_CACHEABLE)
        output_map, output_mc = mem_allocate(fd, output_bytes, RKNPU_MEM_NON_CACHEABLE)
        maps = [task_map, regcmd_map, input_map, output_map]
        mems = [task_mc, regcmd_mc, input_mc, output_mc]

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
        regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

        x = (np.arange(in_h * in_w * POOL_CHANNELS, dtype=np.float16).reshape(in_h, in_w, POOL_CHANNELS) /
                  np.float16(8.0)).astype(np.float16)
        expected = pool2d_reference(x, op)
        input_for_op = -x if op in ("min", "globalmin") else x
        input_map[:input_for_op.nbytes] = input_for_op.reshape(-1).tobytes()
        output_map[:output_bytes] = b"\x00" * output_bytes

        regs = pooling_regs(op, input_mc.dma_addr, output_mc.dma_addr, in_h, in_w)
        for i, value in enumerate(regs):
            regcmd[i] = value

        tasks[0].flags = 0
        tasks[0].op_idx = POOL_TASK_OP_IDX
        tasks[0].enable_mask = POOL_ENABLE_MASK
        tasks[0].int_mask = POOL_INT_MASK
        tasks[0].int_clear = 0x1ffff
        tasks[0].int_status = 0
        tasks[0].regcfg_amount = len(regs)
        tasks[0].regcfg_offset = 0
        tasks[0].regcmd_addr = regcmd_mc.dma_addr

        ret = submit(fd, task_mc.obj_addr)
        read_elems = (in_h - 1) * (in_w - 1) * x.shape[2] if op == "globalmin" else expected.size
        got_raw = np.frombuffer(output_map, dtype=np.float16, count=read_elems).copy()
        if op == "globalmin":
            pooled = got_raw.reshape(in_h - 1, in_w - 1, x.shape[2])
            got = -np.max(pooled.astype(np.float32), axis=(0, 1), keepdims=True).astype(np.float16)
        elif op == "min":
            got = -got_raw.reshape(expected.shape)
        else:
            got = got_raw.reshape(expected.shape)
        decoded = (got.astype(np.float32) / x.shape[0]).astype(np.float16) if op == "globalavg" else got
        atol = 0.25 if op in ("avg", "globalavg") else 0.0
        ok = ret == 0 and np.allclose(decoded, expected, atol=atol)
        print(f"SUBMIT ret={ret}")
        print(f"op={op} input_shape={x.shape} output_shape={expected.shape} reg_count={len(regs)}")
        print(f"NPU output={got.reshape(-1)[:min(32, got.size)]}")
        if op == "globalavg":
            print(f"NPU decoded={decoded.reshape(-1)[:min(32, decoded.size)]}")
        print(f"expected={expected.reshape(-1)[:min(32, expected.size)]}")
        print(f"max_abs_diff={float(np.max(np.abs(decoded.astype(np.float32) - expected.astype(np.float32)))):.6f}")
        print(f"{op.upper()}POOL PASS" if ok else f"{op.upper()}POOL FAIL")
        return 0 if ok else 1
    finally:
        for mmap_obj in reversed(maps):
            mmap_obj.close()
        for mem_create in reversed(mems):
            mem_destroy(fd, mem_create)
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description="RK3588 PPU pool register streams from experimental/rknnops.h")
    parser.add_argument("--op", choices=POOL_OPS + ("all",), default="all")
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--dry", action="store_true", help="print register streams without submitting")
    args = parser.parse_args()

    if args.dry:
        ops = POOL_OPS if args.op == "all" else (args.op,)
        print("Pool dry run only: no /dev/dri open, no ioctl, no submit")
        for op in ops:
            regs = pooling_regs(op, in_h=args.height, in_w=args.width)
            print(f"op={op} reg_count={len(regs)}")
            print("first_regs:")
            for value in regs[:6]:
                print(f"  0x{value:016x}")
            print("last_regs:")
            for value in regs[-6:]:
                print(f"  0x{value:016x}")
        print("POOL DRY RUN PASS")
        return 0

    if args.op == "all":
        rc = 0
        for op in POOL_OPS:
            rc |= run_pool(op, args.height, args.width)
        return rc
    return run_pool(args.op, args.height, args.width)


if __name__ == "__main__":
    raise SystemExit(main())
