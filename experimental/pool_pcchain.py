#!/usr/bin/env python3
import argparse, ctypes, mmap, os
from fcntl import ioctl
import numpy as np

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0 << 1
RKNPU_JOB_NONBLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 0x4

POOL_OPS = ("min", "max", "avg", "globalmin", "globalmax", "globalavg")
POOL_ENABLE_MASK = 0x60
POOL_INT_MASK = 0xc00
POOL_TASK_OP_IDX = 1
POOL_PC_ENABLE = 48
POOL_CHANNELS = 8
TARGET_PC = 0x0081
TARGET_PC_REG = 0x0101
TARGET_VERSION = 0x0041
REG_BLOCK_QWORDS = 32
PC_BASE_ADDRESS = 0x0010
PC_REGISTER_AMOUNTS = 0x0014
PC_OPERATION_ENABLE = 0x0008


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


class rockchip:
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


def _IOWR(type_, nr, size):
  return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)


DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_MEM_DESTROY = _IOWR("d", 0x44, ctypes.sizeof(rknpu_mem_destroy))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR("d", 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR("d", 0x40, ctypes.sizeof(rknpu_action))


def mem_allocate(fd, size, flags=0):
  mem_create = rknpu_mem_create(flags=flags, size=size)
  ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
  mem_map = rknpu_mem_map(handle=mem_create.handle)
  ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
  buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
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
  return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))


def reg(val, shift, mask):
  return (int(val) << shift) & mask


def emit(target, addr, value):
  if target == rockchip.PC: target = 0x80
  return (((target + 1) & 0xffff) << 48) | ((int(value) & 0xffffffff) << 16) | (addr & 0xffff)


def raw_regcmd(target, addr, value):
  return (target << 48) | ((int(value) & 0xffffffff) << 16) | (addr & 0xffff)


def pc_base_address_value(regcmd_dma):
  return regcmd_dma & 0xfffffff0


def field(name, value):
  return reg(value, getattr(rockchip, f"{name}__SHIFT"), getattr(rockchip, f"{name}__MASK"))


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
    emit(rockchip.PPU, rockchip.REG_PPU_S_POINTER, ppu_pointer()),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_S_POINTER, ppu_rdma_pointer()),
    emit(rockchip.PPU, rockchip.REG_PPU_DATA_CUBE_IN_WIDTH, field("PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH", in_w_field)),
    emit(rockchip.PPU, rockchip.REG_PPU_DATA_CUBE_IN_HEIGHT, field("PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT", in_h_field)),
    emit(rockchip.PPU, rockchip.REG_PPU_DATA_CUBE_IN_CHANNEL, field("PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL", channel_field)),
  ]
  if op != "globalmax":
    regs += [
      emit(rockchip.PPU, rockchip.REG_PPU_DATA_CUBE_OUT_WIDTH, field("PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH", out_w_field)),
      emit(rockchip.PPU, rockchip.REG_PPU_DATA_CUBE_OUT_HEIGHT, field("PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT", out_h_field)),
    ]
  regs += [
    emit(rockchip.PPU, rockchip.REG_PPU_DATA_CUBE_OUT_CHANNEL, field("PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL", channel_field)),
    emit(rockchip.PPU, rockchip.REG_PPU_OPERATION_MODE_CFG,
         field("PPU_OPERATION_MODE_CFG_FLYING_MODE", 1) |
         field("PPU_OPERATION_MODE_CFG_POOLING_METHOD", int(max_pool))),
    emit(rockchip.PPU, rockchip.REG_PPU_POOLING_KERNEL_CFG, kernel_cfg(s_h, s_w, k_h, k_w)),
  ]
  if not max_pool:
    regs += [
      emit(rockchip.PPU, rockchip.REG_PPU_RECIP_KERNEL_WIDTH,
           field("PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH", 30720)),
      emit(rockchip.PPU, rockchip.REG_PPU_RECIP_KERNEL_HEIGHT,
           field("PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT", 30720)),
    ]

  regs += [
    emit(rockchip.PPU, rockchip.REG_PPU_DST_BASE_ADDR, field("PPU_DST_BASE_ADDR_DST_BASE_ADDR", output_dma // 16)),
    emit(rockchip.PPU, rockchip.REG_PPU_DST_SURF_STRIDE, field("PPU_DST_SURF_STRIDE_DST_SURF_STRIDE", dst_surf_stride)),
    emit(rockchip.PPU, rockchip.REG_PPU_DATA_FORMAT,
         field("PPU_DATA_FORMAT_INDEX_ADD", index_add) |
         field("PPU_DATA_FORMAT_PROC_PRECISION", 2)),
    emit(rockchip.PPU, rockchip.REG_PPU_MISC_CTRL, field("PPU_MISC_CTRL_BURST_LEN", 3)),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH,
         field("PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH", in_w_field)),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT,
         field("PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT", in_h_field)),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL,
         field("PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL", channel_field)),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_SRC_BASE_ADDR,
         field("PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR", input_dma)),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE,
         field("PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE", width_stride)),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE,
         field("PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE", src_surf_stride)),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_DATA_FORMAT,
         field("PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION", 2)),
    emit(rockchip.PPU_RDMA, rockchip.REG_PPU_RDMA_RDMA_OPERATION_ENABLE,
         field("PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN", 1)),
    emit(rockchip.PC, rockchip.REG_PC_OPERATION_ENABLE,
         field("PC_OPERATION_ENABLE_RESERVED_0", POOL_PC_ENABLE)),
  ]
  return regs


def pool_segment(op, input_dma, output_dma, next_dma, next_amount, height, width):
  regs = pooling_regs(op, input_dma, output_dma, height, width)
  body = regs[:-1]
  tail = [
    raw_regcmd(TARGET_PC_REG, PC_BASE_ADDRESS, pc_base_address_value(next_dma)) if next_dma else 0,
    raw_regcmd(TARGET_PC_REG, PC_REGISTER_AMOUNTS, next_amount if next_dma else 0),
    raw_regcmd(TARGET_VERSION, 0, 0),
    raw_regcmd(TARGET_PC, PC_OPERATION_ENABLE, field("PC_OPERATION_ENABLE_RESERVED_0", POOL_PC_ENABLE)),
  ]
  return body + tail


def output_elements(op, height, width):
  if op in ("globalmax", "globalavg"):
    return POOL_CHANNELS
  return (height - 1) * (width - 1) * POOL_CHANNELS


def expected_from_input(x, op):
  expected = pool2d_reference(x, op)
  if op == "globalavg":
    return (expected.astype(np.float32) * x.shape[0]).astype(np.float16)
  return expected


def decode_output(raw, op, height, width):
  if op == "globalmin":
    pooled = raw.reshape(height - 1, width - 1, POOL_CHANNELS)
    return -np.max(pooled.astype(np.float32), axis=(0, 1), keepdims=True).astype(np.float16)
  if op == "min":
    return -raw.reshape(height - 1, width - 1, POOL_CHANNELS)
  if op in ("globalmax", "globalavg"):
    return raw.reshape(1, 1, POOL_CHANNELS)
  return raw.reshape(height - 1, width - 1, POOL_CHANNELS)


def parse_ops(spec):
  ops = tuple(op.strip() for op in spec.split(",") if op.strip())
  bad = [op for op in ops if op not in POOL_OPS]
  if bad:
    raise ValueError(f"unknown pool ops {bad}; expected {POOL_OPS}")
  return ops


def run_pool_chain(ops, in_h=4, in_w=4):
  fd = os.open("/dev/dri/card1", os.O_RDWR)
  maps = []
  mems = []
  try:
    reset_npu(fd)
    task_map, task_mc = mem_allocate(fd, align_up(len(ops) * ctypes.sizeof(struct_rknpu_task), 4096),
                                     RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mc = mem_allocate(fd, len(ops) * REG_BLOCK_QWORDS * ctypes.sizeof(ctypes.c_uint64),
                                         RKNPU_MEM_NON_CACHEABLE)
    input_stride = align_up(in_h * in_w * POOL_CHANNELS * np.dtype(np.float16).itemsize, 4096)
    output_stride = align_up(max(output_elements(op, in_h, in_w) for op in ops) *
                             np.dtype(np.float16).itemsize, 4096)
    input_map, input_mc = mem_allocate(fd, len(ops) * input_stride, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mc = mem_allocate(fd, len(ops) * output_stride, RKNPU_MEM_NON_CACHEABLE)
    maps = [task_map, regcmd_map, input_map, output_map]
    mems = [task_mc, regcmd_mc, input_mc, output_mc]

    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    expected = []
    for task_idx, op in enumerate(ops):
      x = (np.arange(in_h * in_w * POOL_CHANNELS, dtype=np.float16).reshape(in_h, in_w, POOL_CHANNELS) /
           np.float16(8.0) + np.float16(task_idx)).astype(np.float16)
      input_for_op = -x if op in ("min", "globalmin") else x
      in_off = task_idx * input_stride
      out_off = task_idx * output_stride
      input_map[in_off:in_off + input_for_op.nbytes] = input_for_op.reshape(-1).tobytes()
      output_map[out_off:out_off + output_stride] = b"\x00" * output_stride
      expected.append(expected_from_input(x, op))

    body_lengths = []
    for op in ops:
      body_lengths.append(len(pooling_regs(op, 0, 0, in_h, in_w)) - 1)
    for task_idx, op in enumerate(ops):
      next_dma = regcmd_mc.dma_addr + (task_idx + 1) * REG_BLOCK_QWORDS * 8 if task_idx + 1 < len(ops) else 0
      next_amount = (body_lengths[task_idx + 1] + 2) // 2 if task_idx + 1 < len(ops) else 0
      regs = pool_segment(op,
                          input_mc.dma_addr + task_idx * input_stride,
                          output_mc.dma_addr + task_idx * output_stride,
                          next_dma, next_amount, in_h, in_w)
      base = task_idx * REG_BLOCK_QWORDS
      for i in range(REG_BLOCK_QWORDS):
        regcmd[base + i] = 0
      for i, value in enumerate(regs):
        regcmd[base + i] = value
      tasks[task_idx].flags = 0
      tasks[task_idx].op_idx = POOL_TASK_OP_IDX
      tasks[task_idx].enable_mask = POOL_ENABLE_MASK
      tasks[task_idx].int_mask = POOL_INT_MASK
      tasks[task_idx].int_clear = 0x1ffff
      tasks[task_idx].int_status = 0
      tasks[task_idx].regcfg_amount = body_lengths[task_idx]
      tasks[task_idx].regcfg_offset = 0
      tasks[task_idx].regcmd_addr = regcmd_mc.dma_addr + base * 8

    ret = submit(fd, task_mc.obj_addr)
    ok = ret == 0
    for task_idx, op in enumerate(ops):
      count = output_elements(op, in_h, in_w)
      raw = np.frombuffer(output_map, dtype=np.float16, count=count, offset=task_idx * output_stride).copy()
      got = decode_output(raw, op, in_h, in_w)
      want = expected[task_idx]
      task_ok = np.allclose(got, want, atol=0.25 if op in ("avg", "globalavg") else 0.0)
      ok = ok and task_ok
      print(f"task={task_idx} op={op} ok={task_ok} got={got.reshape(-1)[:8]} expected={want.reshape(-1)[:8]}")
    print("POOL PCCHAIN PASS" if ok else "POOL PCCHAIN FAIL")
    return 0 if ok else 1
  finally:
    for mmap_obj in reversed(maps):
      mmap_obj.close()
    for mem_create in reversed(mems):
      mem_destroy(fd, mem_create)
    os.close(fd)


def main():
  parser = argparse.ArgumentParser(description="Build a PC-chained RK3588 PPU pool task stream.")
  parser.add_argument("--ops", default=",".join(POOL_OPS))
  parser.add_argument("--height", type=int, default=4)
  parser.add_argument("--width", type=int, default=4)
  parser.add_argument("--dry", action="store_true")
  args = parser.parse_args()

  ops = parse_ops(args.ops)
  if args.dry:
    for i, op in enumerate(ops):
      regs = pool_segment(op, 0x11110000 + i * 0x10000, 0x22220000 + i * 0x10000, 0, 0, args.height, args.width)
      print(f"task={i} op={op} regs={len(regs)} first=0x{regs[0]:016x} last=0x{regs[-1]:016x}")
    print("POOL PCCHAIN DRY RUN PASS")
    return 0

  return run_pool_chain(ops, args.height, args.width)


if __name__ == "__main__":
  raise SystemExit(main())
