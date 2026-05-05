#!/usr/bin/env python3
import argparse, ctypes, os, sys
from fcntl import ioctl

import numpy as np

import pool


REG_BLOCK_QWORDS = 32


def parse_ops(spec):
  ops = tuple(op.strip() for op in spec.split(",") if op.strip())
  bad = [op for op in ops if op not in pool.POOL_OPS]
  if bad:
    raise ValueError(f"unknown pool ops {bad}; expected {pool.POOL_OPS}")
  return ops


def parse_core_ranges(spec, task_count):
  if spec == "split3":
    return {0: (0, 1), 1: (1, 1), 2: (2, max(0, task_count - 2))}
  if spec == "core0":
    return {0: (0, task_count)}
  ranges = {}
  for item in spec.split(","):
    core_s, start_s, count_s = item.split(":")
    ranges[int(core_s, 0)] = (int(start_s, 0), int(count_s, 0))
  return ranges


def output_elements(op, height, width):
  if op in ("globalmax", "globalavg"):
    return pool.POOL_CHANNELS
  return (height - 1) * (width - 1) * pool.POOL_CHANNELS


def decode_output(raw, op, height, width):
  if op == "globalmin":
    pooled = raw.reshape(height - 1, width - 1, pool.POOL_CHANNELS)
    return -np.max(pooled.astype(np.float32), axis=(0, 1), keepdims=True).astype(np.float16)
  if op == "min":
    return -raw.reshape(height - 1, width - 1, pool.POOL_CHANNELS)
  if op in ("globalmax", "globalavg"):
    return raw.reshape(1, 1, pool.POOL_CHANNELS)
  return raw.reshape(height - 1, width - 1, pool.POOL_CHANNELS)


def expected_from_input(x, op):
  expected = pool.pool2d_reference(x, op)
  if op == "globalavg":
    return (expected.astype(np.float32) * x.shape[0]).astype(np.float16)
  return expected


def submit_multicore(fd, task_obj_addr, task_count, core_ranges, core_mask):
  submit_struct = pool.rknpu_submit(
    flags=pool.RKNPU_JOB_PC | pool.RKNPU_JOB_BLOCK | pool.RKNPU_JOB_PINGPONG,
    timeout=6000,
    task_start=0,
    task_number=task_count,
    task_counter=0,
    priority=0,
    task_obj_addr=task_obj_addr,
    regcfg_obj_addr=0,
    task_base_addr=0,
    user_data=0,
    core_mask=core_mask,
    fence_fd=-1,
  )
  for i in range(5):
    submit_struct.subcore_task[i] = pool.rknpu_subcore_task(0, 0)
  for core, (start, count) in core_ranges.items():
    submit_struct.subcore_task[core] = pool.rknpu_subcore_task(start, count)
  return ioctl(fd, pool.DRM_IOCTL_RKNPU_SUBMIT, submit_struct), submit_struct


def main():
  parser = argparse.ArgumentParser(description="Build independent RK3588 PPU pool tasks for multicore submit experiments.")
  parser.add_argument("--ops", default="max,avg,globalmax")
  parser.add_argument("--height", type=int, default=4)
  parser.add_argument("--width", type=int, default=4)
  parser.add_argument("--core-ranges", default="split3")
  parser.add_argument("--core-mask", type=lambda x: int(x, 0), default=0x7)
  parser.add_argument("--dry", action="store_true")
  args = parser.parse_args()

  ops = parse_ops(args.ops)
  core_ranges = parse_core_ranges(args.core_ranges, len(ops))
  if args.dry:
    for task_idx, op in enumerate(ops):
      regs = pool.pooling_regs(op, 0x11110000 + task_idx * 0x10000, 0x22220000 + task_idx * 0x10000,
                               args.height, args.width)
      print(f"task={task_idx} op={op} regs={len(regs)} first=0x{regs[0]:016x} last=0x{regs[-1]:016x}")
    print(f"core_ranges={core_ranges} core_mask=0x{args.core_mask:x}")
    print("POOL MULTICORE DRY RUN PASS")
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

      regs = pool.pooling_regs(op,
                               input_mc.dma_addr + in_off,
                               output_mc.dma_addr + out_off,
                               args.height, args.width)
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

    ret, submit_struct = submit_multicore(fd, task_mc.obj_addr, len(ops), core_ranges, args.core_mask)
    print(f"submit ret={ret} core_mask=0x{submit_struct.core_mask:x} task_number={submit_struct.task_number}")
    ok = ret == 0
    for task_idx, op in enumerate(ops):
      count = output_elements(op, args.height, args.width)
      raw = np.frombuffer(output_map, dtype=np.float16, count=count, offset=task_idx * output_stride).copy()
      got = decode_output(raw, op, args.height, args.width)
      want = expected[task_idx]
      task_ok = np.allclose(got, want, atol=0.25 if op in ("avg", "globalavg") else 0.0)
      ok = ok and task_ok
      print(f"task={task_idx} op={op} ok={task_ok} got={got.reshape(-1)[:8]} expected={want.reshape(-1)[:8]}")
    print("POOL MULTICORE PASS" if ok else "POOL MULTICORE FAIL")
    return 0 if ok else 1
  finally:
    for mmap_obj in reversed(maps):
      mmap_obj.close()
    for mem_create in reversed(mems):
      pool.mem_destroy(fd, mem_create)
    os.close(fd)


if __name__ == "__main__":
  raise SystemExit(main())
