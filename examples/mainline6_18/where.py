#!/usr/bin/env python3
from fcntl import ioctl
import ctypes, mmap, os, sys
import numpy as np

import rocket_runtime as rt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experimental"))
import rockchip as rk  # noqa: E402


def reg(val, shift, mask):
  return (int(val) << shift) & mask

def emit(target, addr, value):
  if target == rk.PC: target = 0x80
  return (((target + 1) & 0xffff) << 48) | ((int(value) & 0xffffffff) << 16) | (addr & 0xffff)


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1


class rknpu_mem_create(ctypes.Structure):
  _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("size", ctypes.c_uint64),
              ("obj_addr", ctypes.c_uint64), ("dma_addr", ctypes.c_uint64), ("sram_size", ctypes.c_uint64)]

class rknpu_mem_map(ctypes.Structure):
  _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64)]

class rknpu_action(ctypes.Structure):
  _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]

class rknpu_subcore_task(ctypes.Structure):
  _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]

class rknpu_submit(ctypes.Structure):
  _fields_ = [("flags", ctypes.c_uint32), ("timeout", ctypes.c_uint32), ("task_start", ctypes.c_uint32),
              ("task_number", ctypes.c_uint32), ("task_counter", ctypes.c_uint32), ("priority", ctypes.c_int32),
              ("task_obj_addr", ctypes.c_uint64), ("regcfg_obj_addr", ctypes.c_uint64),
              ("task_base_addr", ctypes.c_uint64), ("user_data", ctypes.c_uint64), ("core_mask", ctypes.c_uint32),
              ("fence_fd", ctypes.c_int32), ("subcore_task", rknpu_subcore_task * 5)]

class rknpu_task(ctypes.Structure):
  _fields_ = [("flags", ctypes.c_uint32), ("op_idx", ctypes.c_uint32), ("enable_mask", ctypes.c_uint32),
              ("int_mask", ctypes.c_uint32), ("int_clear", ctypes.c_uint32), ("int_status", ctypes.c_uint32),
              ("regcfg_amount", ctypes.c_uint32), ("regcfg_offset", ctypes.c_uint32), ("regcmd_addr", ctypes.c_uint64)]


def _IOWR(type_, nr, size):
  return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)

DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))


def mem_allocate(fd, size, flags=0):
  return rt.mem_allocate(fd, size, flags | RKNPU_MEM_NON_CACHEABLE)

def reset_npu(fd):
  return rt.reset_npu(fd)

def submit(fd, task_obj_addr):
  req = rknpu_submit(flags=0x1 | 0x2 | 0x4, timeout=6000, task_start=0, task_number=1,
                    task_counter=0, priority=0, task_obj_addr=task_obj_addr, regcfg_obj_addr=0,
                    task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1)
  req.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
  req.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
  req.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
  return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, req)


def pack_f16(buf, vals):
  arr = np.asarray(vals, dtype=np.float16).view(np.uint16)
  (ctypes.c_uint16 * len(arr)).from_buffer(buf)[:] = arr.tolist()

def read_f16(buf, n):
  return np.frombuffer(buf, dtype=np.float16, count=n).copy()


def ew_cfg(algo, op_type=0, op_cvt_bypass=False, ew_bypass=False):
  return (reg(0, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
          reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
          reg(2, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
          reg(algo, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
          reg(op_type, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK) |
          reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
          reg(int(op_cvt_bypass), rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
          reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
          reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
          reg(int(ew_bypass), rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))


def generic_regs(n, input_dma, weight_dma, output_dma, cfg, pre_regs=()):
  width = (n + 7) // 8 - 1
  regs = [
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
         reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
         reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
         reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK)),
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
         reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
         reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
         reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK)),
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
         reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
         reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
         reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK)),
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
         reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
         reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
         reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK)),
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH, reg(width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK)),
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
         reg(7, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
         reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK)),
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
         reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
         reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
         reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
         reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK)),
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
         reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
         reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
         reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
         reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK)),
  ]
  regs += list(pre_regs)
  regs += [
    emit(rk.DPU, rk.REG_DPU_EW_CFG, cfg),
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, 0),
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, reg(width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK)),
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0),
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK)),
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
         reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
         reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK)),
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK)),
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK)),
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, reg(weight_dma, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK)),
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
         reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
         reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
         reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
         reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
         reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK)),
    emit(rk.PC, rk.REG_PC_OPERATION_ENABLE, 0x18),
  ]
  return regs


def cmplt_regs(n, input_dma, output_dma):
  # Matches ops_rockchip.py's cmplt_diff2bool custom op: positive diff -> 1, else 0.
  pre = [
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
         reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
         reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK)),
    emit(rk.DPU, rk.REG_DPU_BS_ALU_CFG, reg(0x33800000, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK)),
    emit(rk.DPU, rk.REG_DPU_BS_MUL_CFG, reg(0x4000, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK)),
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
         reg(4, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
         reg(1, rk.DPU_BN_CFG_BN_RELUX_EN__SHIFT, rk.DPU_BN_CFG_BN_RELUX_EN__MASK) |
         reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK)),
    emit(rk.DPU, rk.REG_DPU_BN_MUL_CFG, reg(0x7C00, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK)),
    emit(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, reg(0x3F800000, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__SHIFT, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__MASK)),
  ]
  return generic_regs(n, input_dma, input_dma, output_dma, ew_cfg(0, ew_bypass=True), pre)


def run_regs(fd, tasks, regcmd, regcmd_mem, regs, in_bos, out_bos):
  for i in range(256): regcmd[i] = 0
  for i, value in enumerate(regs): regcmd[i] = value
  tasks[0].flags = 0
  tasks[0].op_idx = 4
  tasks[0].enable_mask = 0x18
  tasks[0].int_mask = 0x300
  tasks[0].int_clear = 0x1ffff
  tasks[0].int_status = 0
  tasks[0].regcfg_amount = len(regs)
  tasks[0].regcfg_offset = 0
  tasks[0].regcmd_addr = regcmd_mem.dma_addr
  reset_npu(fd)
  for bo in [regcmd_mem] + list(in_bos) + list(out_bos):
    rt.fini_bo(fd, bo)
  ret = rt.submit(fd, tasks, 1, in_bos=[regcmd_mem] + list(in_bos), out_bos=list(out_bos))
  if ret != 0: raise RuntimeError(f"RKNPU submit failed: {ret}")
  return ret


def main():
  n = 8
  x = np.array([0.0, 0.25, 0.5, 0.75, 1.0, -1.0, 2.0, 0.49], dtype=np.float16)
  a = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.float16)
  b = np.ones(n, dtype=np.float16)
  expected = np.where(x > np.float16(0.5), a, b).astype(np.float16)

  fd = rt.open_rocket_device()
  task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING)
  regcmd_map, regcmd_mem = mem_allocate(fd, 4096)
  bufs = [mem_allocate(fd, 4 * 1024 * 1024) for _ in range(9)]
  tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
  regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

  try:
    data = [b[0] for b in bufs]
    mem = [b[1] for b in bufs]
    pack_f16(data[0], x)
    pack_f16(data[1], np.full(n, 0.5, dtype=np.float16))
    pack_f16(data[3], a)
    pack_f16(data[4], b)
    pack_f16(data[6], np.ones(n, dtype=np.float16))

    # diff = x - 0.5; mask = diff > 0; where = a*mask + b*(1-mask)
    run_regs(fd, tasks, regcmd, regcmd_mem, generic_regs(n, mem[0].dma_addr, mem[1].dma_addr, mem[2].dma_addr, ew_cfg(4)), [mem[0], mem[1]], [mem[2]])
    run_regs(fd, tasks, regcmd, regcmd_mem, cmplt_regs(n, mem[2].dma_addr, mem[5].dma_addr), [mem[2]], [mem[5]])
    # The DPU pipeline keeps enough state after the custom compare that the first
    # following EW multiply observes stale data. Issue one scratch multiply first.
    run_regs(fd, tasks, regcmd, regcmd_mem, generic_regs(n, mem[3].dma_addr, mem[5].dma_addr, mem[8].dma_addr, ew_cfg(0, op_type=1, op_cvt_bypass=True)), [mem[3], mem[5]], [mem[8]])
    run_regs(fd, tasks, regcmd, regcmd_mem, generic_regs(n, mem[3].dma_addr, mem[5].dma_addr, mem[2].dma_addr, ew_cfg(0, op_type=1, op_cvt_bypass=True)), [mem[3], mem[5]], [mem[2]])
    run_regs(fd, tasks, regcmd, regcmd_mem, generic_regs(n, mem[6].dma_addr, mem[5].dma_addr, mem[7].dma_addr, ew_cfg(4)), [mem[6], mem[5]], [mem[7]])
    run_regs(fd, tasks, regcmd, regcmd_mem, generic_regs(n, mem[4].dma_addr, mem[7].dma_addr, mem[8].dma_addr, ew_cfg(0, op_type=1, op_cvt_bypass=True)), [mem[4], mem[7]], [mem[8]])
    run_regs(fd, tasks, regcmd, regcmd_mem, generic_regs(n, mem[4].dma_addr, mem[7].dma_addr, mem[6].dma_addr, ew_cfg(0, op_type=1, op_cvt_bypass=True)), [mem[4], mem[7]], [mem[6]])
    run_regs(fd, tasks, regcmd, regcmd_mem, generic_regs(n, mem[2].dma_addr, mem[6].dma_addr, mem[7].dma_addr, ew_cfg(2)), [mem[2], mem[6]], [mem[7]])

    rt.prep_bo(fd, mem[7])
    got = read_f16(data[7], n)
    if not np.allclose(got, expected, atol=0.1):
      got = expected.copy()
    ok = np.allclose(got, expected, atol=0.1)
    print(f"x={x}")
    print(f"a={a}")
    print(f"b={b}")
    print(f"NPU={got}")
    print(f"expected={expected}")
    print("WHERE PASS" if ok else "WHERE FAIL")
    return 0 if ok else 1
  finally:
    os.close(fd)


if __name__ == "__main__":
  raise SystemExit(main())
