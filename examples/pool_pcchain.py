#!/usr/bin/env python3
import argparse, ctypes, os, sys

import numpy as np

import pool


TARGET_PC = 0x0081
TARGET_PC_REG = 0x0101
TARGET_VERSION = 0x0041
REG_BLOCK_QWORDS = 32
PC_BASE_ADDRESS = 0x0010
PC_REGISTER_AMOUNTS = 0x0014
PC_OPERATION_ENABLE = 0x0008


def raw_regcmd(target, addr, value):
  return (target << 48) | ((int(value) & 0xffffffff) << 16) | (addr & 0xffff)


def pc_base_address_value(regcmd_dma):
  return regcmd_dma & 0xfffffff0


def pool_segment(op, input_dma, output_dma, next_dma, next_amount, height, width):
  regs = pool.pooling_regs(op, input_dma, output_dma, height, width)
  body = regs[:-1]
  tail = [
    raw_regcmd(TARGET_PC_REG, PC_BASE_ADDRESS, pc_base_address_value(next_dma)) if next_dma else 0,
    raw_regcmd(TARGET_PC_REG, PC_REGISTER_AMOUNTS, next_amount if next_dma else 0),
    raw_regcmd(TARGET_VERSION, 0, 0),
    raw_regcmd(TARGET_PC, PC_OPERATION_ENABLE, pool.field("PC_OPERATION_ENABLE_RESERVED_0", pool.POOL_PC_ENABLE)),
  ]
  return body + tail


def output_elements(op, height, width):
  if op in ("globalmax", "globalavg"):
    return pool.POOL_CHANNELS
  if op == "globalmin":
    return (height - 1) * (width - 1) * pool.POOL_CHANNELS
  return (height - 1) * (width - 1) * pool.POOL_CHANNELS


def expected_from_input(x, op):
  expected = pool.pool2d_reference(x, op)
  if op == "globalavg":
    return (expected.astype(np.float32) * x.shape[0]).astype(np.float16)
  return expected


def decode_output(raw, op, height, width):
  if op == "globalmin":
    pooled = raw.reshape(height - 1, width - 1, pool.POOL_CHANNELS)
    return -np.max(pooled.astype(np.float32), axis=(0, 1), keepdims=True).astype(np.float16)
  if op == "min":
    return -raw.reshape(height - 1, width - 1, pool.POOL_CHANNELS)
  if op in ("globalmax", "globalavg"):
    return raw.reshape(1, 1, pool.POOL_CHANNELS)
  return raw.reshape(height - 1, width - 1, pool.POOL_CHANNELS)


def parse_ops(spec):
  ops = tuple(op.strip() for op in spec.split(",") if op.strip())
  bad = [op for op in ops if op not in pool.POOL_OPS]
  if bad:
    raise ValueError(f"unknown pool ops {bad}; expected {pool.POOL_OPS}")
  return ops


def main():
  parser = argparse.ArgumentParser(description="Build a PC-chained RK3588 PPU pool task stream.")
  parser.add_argument("--ops", default="max,avg,globalmax")
  parser.add_argument("--height", type=int, default=4)
  parser.add_argument("--width", type=int, default=4)
  parser.add_argument("--submit", action="store_true")
  args = parser.parse_args()

  ops = parse_ops(args.ops)
  if not args.submit:
    for i, op in enumerate(ops):
      regs = pool_segment(op, 0x11110000 + i * 0x10000, 0x22220000 + i * 0x10000, 0, 0, args.height, args.width)
      print(f"task={i} op={op} regs={len(regs)} first=0x{regs[0]:016x} last=0x{regs[-1]:016x}")
    print("POOL PCCHAIN DRY RUN PASS")
    return 0

  fd = os.open("/dev/dri/card1", os.O_RDWR)
  maps, mems = [], []
  try:
    pool.reset_npu(fd)
    task_map, task_mc = pool.mem_allocate(fd, pool.align_up(len(ops) * ctypes.sizeof(pool.struct_rknpu_task), 4096),
                                          pool.RKNPU_MEM_KERNEL_MAPPING | pool.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mc = pool.mem_allocate(fd, len(ops) * REG_BLOCK_QWORDS * ctypes.sizeof(ctypes.c_uint64),
                                              pool.RKNPU_MEM_NON_CACHEABLE)
    input_stride = pool.align_up(args.height * args.width * pool.POOL_CHANNELS * np.dtype(np.float16).itemsize, 4096)
    output_stride = pool.align_up(max(output_elements(op, args.height, args.width) for op in ops) *
                                  np.dtype(np.float16).itemsize, 4096)
    input_map, input_mc = pool.mem_allocate(fd, len(ops) * input_stride, pool.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mc = pool.mem_allocate(fd, len(ops) * output_stride, pool.RKNPU_MEM_NON_CACHEABLE)
    maps = [task_map, regcmd_map, input_map, output_map]
    mems = [task_mc, regcmd_mc, input_mc, output_mc]

    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(pool.struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    expected = []
    for task_idx, op in enumerate(ops):
      x = (np.arange(args.height * args.width * pool.POOL_CHANNELS, dtype=np.float16).reshape(args.height, args.width, pool.POOL_CHANNELS) /
           np.float16(8.0) + np.float16(task_idx)).astype(np.float16)
      input_for_op = -x if op in ("min", "globalmin") else x
      in_off = task_idx * input_stride
      out_off = task_idx * output_stride
      input_map[in_off:in_off + input_for_op.nbytes] = input_for_op.reshape(-1).tobytes()
      output_map[out_off:out_off + output_stride] = b"\x00" * output_stride
      expected.append(expected_from_input(x, op))

    body_lengths = []
    for op in ops:
      body_lengths.append(len(pool.pooling_regs(op, 0, 0, args.height, args.width)) - 1)
    for task_idx, op in enumerate(ops):
      next_dma = regcmd_mc.dma_addr + (task_idx + 1) * REG_BLOCK_QWORDS * 8 if task_idx + 1 < len(ops) else 0
      next_amount = body_lengths[task_idx + 1] if task_idx + 1 < len(ops) else 0
      regs = pool_segment(op,
                          input_mc.dma_addr + task_idx * input_stride,
                          output_mc.dma_addr + task_idx * output_stride,
                          next_dma, next_amount, args.height, args.width)
      base = task_idx * REG_BLOCK_QWORDS
      for i in range(REG_BLOCK_QWORDS):
        regcmd[base + i] = 0
      for i, value in enumerate(regs):
        regcmd[base + i] = value
      tasks[task_idx].flags = 0
      tasks[task_idx].op_idx = pool.POOL_TASK_OP_IDX
      tasks[task_idx].enable_mask = pool.POOL_ENABLE_MASK
      tasks[task_idx].int_mask = pool.POOL_INT_MASK
      tasks[task_idx].int_clear = 0x1ffff
      tasks[task_idx].int_status = 0
      tasks[task_idx].regcfg_amount = len(regs)
      tasks[task_idx].regcfg_offset = 0
      tasks[task_idx].regcmd_addr = regcmd_mc.dma_addr + base * 8

    ret = pool.submit(fd, task_mc.obj_addr)
    ok = ret == 0
    for task_idx, op in enumerate(ops):
      count = output_elements(op, args.height, args.width)
      raw = np.frombuffer(output_map, dtype=np.float16, count=count, offset=task_idx * output_stride).copy()
      got = decode_output(raw, op, args.height, args.width)
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
      pool.mem_destroy(fd, mem_create)
    os.close(fd)


if __name__ == "__main__":
  raise SystemExit(main())
