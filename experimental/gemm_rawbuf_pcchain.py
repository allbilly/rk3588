from fcntl import ioctl
import argparse
import ctypes
import mmap
import os
import struct

import numpy as np


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_OFFICIAL_TASK = 0x40B
RKNPU_MEM_OFFICIAL_TENSOR = 0x403
RKNPU_JOB_PC = 0x1
RKNPU_JOB_PINGPONG = 0x4

M = N = K = 394
ALIGN_IN = ALIGN_OUT = 416
REGCMD_BYTES = 3072
TASKS = 11

OLD_INPUT_BASE = 0xFFF0C000
OLD_WEIGHT_BASE = 0xFFF5D000
OLD_OUTPUT_BASE = 0xFFE16000
OLD_REGCMD_BASE = 0xFFFB3000

TASK_OFFSETS = [0x000, 0x380, 0x700, 0x7C0, 0x840, 0x8C0, 0x940, 0x9C0, 0xA40, 0xAC0, 0xB40]
TASK_AMOUNTS = [108, 108, 13, 12, 12, 12, 12, 12, 12, 12, 17]

TARGET_CNA = 0x0201
TARGET_DPU = 0x1001
TARGET_PC_REG = 0x0101
REG_CNA_FEATURE_DATA_ADDR = 0x1070
REG_CNA_DCOMP_ADDR0 = 0x1110
REG_DPU_DST_BASE_ADDR = 0x4020
REG_PC_BASE_ADDRESS = 0x0010


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


DRM_IOCTL_RKNPU_MEM_CREATE = _iowr("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _iowr("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _iowr("d", 0x41, ctypes.sizeof(rknpu_submit))


def align_up(value, align):
    return ((value + align - 1) // align) * align


def emit(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def mem_allocate(fd, size, flags):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc


def pack_input_row_major(a):
    padded = np.zeros((M, ALIGN_IN), dtype=np.float16)
    padded[:, :K] = a
    return padded.ravel()


def pack_weight_tile_16x32(b):
    wt = np.zeros((ALIGN_OUT, ALIGN_IN), dtype=np.float16)
    wt[:N, :K] = b.T
    return wt.reshape(ALIGN_OUT // 16, 16, ALIGN_IN // 32, 32).transpose(0, 2, 1, 3).ravel()


def load_regcmd_blob(path):
    data = open(path, "rb").read(REGCMD_BYTES)
    if len(data) != REGCMD_BYTES:
        raise RuntimeError(f"expected {REGCMD_BYTES} bytes in {path}, got {len(data)}")
    return list(struct.unpack(f"<{REGCMD_BYTES // 8}Q", data))


def patch_blob(blob, input_dma, weight_dma, output_dma, regcmd_dma):
    patched = []
    for qword in blob:
        target = (qword >> 48) & 0xFFFF
        addr = qword & 0xFFFF
        value = (qword >> 16) & 0xFFFFFFFF
        if target == TARGET_CNA and addr == REG_CNA_FEATURE_DATA_ADDR:
            qword = emit(target, addr, input_dma + (value - OLD_INPUT_BASE))
        elif target == TARGET_CNA and addr == REG_CNA_DCOMP_ADDR0:
            qword = emit(target, addr, weight_dma + (value - OLD_WEIGHT_BASE))
        elif target == TARGET_DPU and addr == REG_DPU_DST_BASE_ADDR:
            qword = emit(target, addr, output_dma + (value - OLD_OUTPUT_BASE))
        elif target == TARGET_PC_REG and addr == REG_PC_BASE_ADDRESS and value:
            qword = emit(target, addr, regcmd_dma + (value - OLD_REGCMD_BASE))
        patched.append(qword)
    return patched


def submit(fd, task_obj_addr, mode, timeout):
    official = mode == "official"
    submit_struct = rknpu_submit(
        flags=RKNPU_JOB_PC | RKNPU_JOB_PINGPONG,
        timeout=timeout,
        task_start=0,
        task_number=TASKS * 3 if official else TASKS,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=0 if official else 1,
        fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(0, TASKS)
    submit_struct.subcore_task[1] = rknpu_subcore_task(0, TASKS if official else 0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(0, TASKS if official else 0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(0, 0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(0, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def main():
    parser = argparse.ArgumentParser(description="Replay official RKNN 394x394x394 GEMM PC-chain raw buffer.")
    parser.add_argument("--dump", default="/home/orangepi/npu/ops_rknn/dump/gem1-dump")
    parser.add_argument("--mode", choices=("core0", "official"), default="core0")
    parser.add_argument("--alloc-mode", choices=("raw", "official"), default="raw")
    parser.add_argument("--constant-data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--atol", type=float, default=0.1)
    args = parser.parse_args()

    if args.constant_data:
        a = np.ones((M, K), dtype=np.float16)
        b = np.ones((K, N), dtype=np.float16)
        expected = np.full((M, N), K, dtype=np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        a = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
        b = rng.standard_normal((K, N), dtype=np.float32).astype(np.float16)
        expected = a @ b

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        task_flags = RKNPU_MEM_OFFICIAL_TASK if args.alloc_mode == "official" else RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE
        tensor_flags = RKNPU_MEM_OFFICIAL_TENSOR if args.alloc_mode == "official" else RKNPU_MEM_NON_CACHEABLE
        task_map, task_mc = mem_allocate(fd, TASKS * ctypes.sizeof(rknpu_task), task_flags)
        regcmd_map, regcmd_mc = mem_allocate(fd, REGCMD_BYTES, tensor_flags)
        input_map, input_mc = mem_allocate(fd, align_up(M * ALIGN_IN * 2, 4096), tensor_flags)
        weight_map, weight_mc = mem_allocate(fd, align_up(ALIGN_IN * ALIGN_OUT * 2, 4096), tensor_flags)
        output_words = M * ALIGN_OUT
        output_map, output_mc = mem_allocate(fd, align_up(output_words * 4, 4096), tensor_flags)

        np.frombuffer(input_map, dtype=np.float16, count=M * ALIGN_IN)[:] = pack_input_row_major(a)
        np.frombuffer(weight_map, dtype=np.float16, count=ALIGN_IN * ALIGN_OUT)[:] = pack_weight_tile_16x32(b)
        np.frombuffer(output_map, dtype=np.float32)[:] = np.nan

        blob = patch_blob(load_regcmd_blob(args.dump), input_mc.dma_addr, weight_mc.dma_addr, output_mc.dma_addr, regcmd_mc.dma_addr)
        regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
        for i, qword in enumerate(blob):
            regcmd[i] = qword

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
        for idx, (offset, amount) in enumerate(zip(TASK_OFFSETS, TASK_AMOUNTS)):
            tasks[idx].flags = 0
            tasks[idx].op_idx = 0
            tasks[idx].enable_mask = 0x0D
            tasks[idx].int_mask = 0x300
            tasks[idx].int_clear = 0x1FFFF
            tasks[idx].int_status = 0
            tasks[idx].regcfg_amount = amount
            tasks[idx].regcfg_offset = 0
            tasks[idx].regcmd_addr = regcmd_mc.dma_addr + offset

        print(f"regcmd=0x{regcmd_mc.dma_addr:x} tasks={TASKS} mode={args.mode}")
        ret = submit(fd, task_mc.obj_addr, args.mode, args.timeout)
        print(f"submit ret={ret}")
        raw = np.frombuffer(output_map, dtype=np.float32, count=output_words).copy()
        got = raw[np.arange(M)[:, None] * ALIGN_OUT + np.arange(N)]
        ok = ret == 0 and np.allclose(got, expected, atol=args.atol, equal_nan=False)
        max_diff = float(np.nanmax(np.abs(got - expected)))
        print(f"394x394x394 pcchain {'PASS' if ok else 'FAIL'} max_diff={max_diff:.6f}")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
