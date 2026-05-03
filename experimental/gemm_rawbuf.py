from fcntl import ioctl
import argparse
import ctypes
import mmap
import os

import numpy as np


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_OFFICIAL_TASK = 0x40B
RKNPU_MEM_OFFICIAL_TENSOR = 0x403
RKNPU_JOB_PC = 0x1
RKNPU_JOB_PINGPONG = 0x4

M = 424
K = 424
N = 424
ALIGN_IN = 448
ALIGN_OUT = 448
TASKS = 2
REGCFG_AMOUNT = 108
SEGMENT_QWORDS = 112
TILE_ROWS = (365, 59)

TARGET_CNA = 0x0201
TARGET_DPU = 0x1001
TARGET_PC_REG = 0x0101
TARGET_PC_OP = 0x0081
REG_CNA_FEATURE_DATA_ADDR = 0x1070
REG_CNA_DCOMP_ADDR0 = 0x1110
REG_DPU_DST_BASE_ADDR = 0x4020
REG_PC_BASE_ADDRESS = 0x0010
REG_PC_REGISTER_AMOUNTS = 0x0014
REG_PC_OPERATION_ENABLE = 0x0008


def align_up(value, align):
    return ((value + align - 1) // align) * align

CAPTURED_GEMM_REGCMD_BLOB = [
    0x02010000002a1040, 0x0201000000001104, 0x0201000000001100, 0x020120000120100c,
    0x10010000000e4004, 0x020120000120100c, 0x0201000004a01010, 0x0201000000091014,
    0x02010001016d1020, 0x020101bf01c01024, 0x0201000000011028, 0x02010000016d102c,
    0x0201000620001030, 0x0201000003801034, 0x0201010101c01038, 0x02010000002a1040,
    0x02010000000e1044, 0x02010000000b104c, 0x0201000100001050, 0x0201000100001054,
    0x0201000100001058, 0x020100010000105c, 0x0201000000001060, 0x0201000000001064,
    0x0201000000001068, 0x0201ffeb60001070, 0x0201000000001074, 0x0201000f000f1078,
    0x020100000038107c, 0x0201000000001080, 0x02010001016d1084, 0x0201000001c01088,
    0x0201000000001100, 0x0201000000001104, 0x0201fff130001110, 0x0201000000001140,
    0x0201000000001144, 0x0201000000001148, 0x020100000000114c, 0x0201000000001150,
    0x0201000000001154, 0x0201000000001158, 0x020100000000115c, 0x0201000000001160,
    0x0201000000001164, 0x0201000000001168, 0x020100000000116c, 0x0201000000001170,
    0x0201000000001174, 0x0201000000001178, 0x020100000000117c, 0x0201000000001180,
    0x0201000000001184, 0x0801000002013010, 0x0801016c00003014, 0x0801000001bf3018,
    0x080100000000301c, 0x0801000000003030, 0x1001000001e4400c, 0x1001a80000024010,
    0x1001000000004014, 0x1001ffd9a0004020, 0x1001000000104024, 0x1001000000004030,
    0x10010000016c4034, 0x1001006f006f4038, 0x100101bf01bf403c, 0x1001000000534040,
    0x1001000000004044, 0x1001000000004048, 0x100100000000404c, 0x10010000036e4050,
    0x1001000000004054, 0x1001000001bf4058, 0x1001016c0000405c, 0x1001000000534060,
    0x1001000000004064, 0x1001000000004068, 0x100100000000406c, 0x1001000003834070,
    0x1001000000004074, 0x1001000000014078, 0x100100000000407c, 0x1001000000004080,
    0x1001000000014084, 0x1001000000004088, 0x1001000000004090, 0x1001000000004094,
    0x1001000000004098, 0x100100000000409c, 0x10010000000040a0, 0x10010000000040a4,
    0x10010000000040a8, 0x10010000000040ac, 0x10010000004040c0, 0x10010000000040c4,
    0x1001000000004100, 0x1001000000004104, 0x1001000000004108, 0x100100000000410c,
    0x1001000000004110, 0x1001000000004114, 0x1001000000004118, 0x100100000000411c,
    0x1001000000004120, 0x1001000000004124, 0x1001000000004128, 0x100100000000412c,
    0x0101fff763800010, 0x0101000000370014, 0x0041000000000000, 0x00810000000d0008,
    0x02010000002a1040, 0x0201000000001104, 0x0201000000001100, 0x020120000120100c,
    0x10010000000e4004, 0x020120000120100c, 0x0201000003b01010, 0x0201000000091014,
    0x02010001003b1020, 0x020101bf01c01024, 0x0201000000011028, 0x02010000003b102c,
    0x0201000620001030, 0x0201000003801034, 0x0201010101c01038, 0x02010000002a1040,
    0x02010000000e1044, 0x02010000000b104c, 0x0201000100001050, 0x0201000100001054,
    0x0201000100001058, 0x020100010000105c, 0x0201000000001060, 0x0201000000001064,
    0x0201000000001068, 0x0201fff05d801070, 0x0201000000001074, 0x0201000f000f1078,
    0x020100000038107c, 0x0201000000001080, 0x02010001003b1084, 0x0201000001c01088,
    0x0201000000001100, 0x0201000000001104, 0x0201fff130001110, 0x0201000000001140,
    0x0201000000001144, 0x0201000000001148, 0x020100000000114c, 0x0201000000001150,
    0x0201000000001154, 0x0201000000001158, 0x020100000000115c, 0x0201000000001160,
    0x0201000000001164, 0x0201000000001168, 0x020100000000116c, 0x0201000000001170,
    0x0201000000001174, 0x0201000000001178, 0x020100000000117c, 0x0201000000001180,
    0x0201000000001184, 0x0801000002013010, 0x0801003a00003014, 0x0801000001bf3018,
    0x080100000000301c, 0x0801000000003030, 0x1001000001e4400c, 0x1001a80000024010,
    0x1001000000004014, 0x1001ffe39b004020, 0x1001000000104024, 0x1001000000004030,
    0x10010000003a4034, 0x1001006f006f4038, 0x100101bf01bf403c, 0x1001000000534040,
    0x1001000000004044, 0x1001000000004048, 0x100100000000404c, 0x10010000036e4050,
    0x1001000000004054, 0x1001000001bf4058, 0x1001003a0000405c, 0x1001000000534060,
    0x1001000000004064, 0x1001000000004068, 0x100100000000406c, 0x1001000003834070,
    0x1001000000004074, 0x1001000000014078, 0x100100000000407c, 0x1001000000004080,
    0x1001000000014084, 0x1001000000004088, 0x1001000000004090, 0x1001000000004094,
    0x1001000000004098, 0x100100000000409c, 0x10010000000040a0, 0x10010000000040a4,
    0x10010000000040a8, 0x10010000000040ac, 0x10010000004040c0, 0x10010000000040c4,
    0x1001000000004100, 0x1001000000004104, 0x1001000000004108, 0x100100000000410c,
    0x1001000000004110, 0x1001000000004114, 0x1001000000004118, 0x100100000000411c,
    0x1001000000004120, 0x1001000000004124, 0x1001000000004128, 0x100100000000412c,
    0x0000000000000000, 0x0101000000000014, 0x0041000000000000, 0x00810000000d0008,
]
PC_VERSION_QWORD = 0x0041000000000000


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


def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)


DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR("d", 0x41, ctypes.sizeof(rknpu_submit))


def mem_allocate(fd, size, flags):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc


def emit(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def patch_first(segment, target, addr, value):
    for i, qword in enumerate(segment):
        if ((qword >> 48) & 0xFFFF) == target and (qword & 0xFFFF) == addr:
            segment[i] = emit(target, addr, value)
            return
    raise RuntimeError(f"missing register target=0x{target:x} addr=0x{addr:x}")


def pack_input_nc1hwc2_c2_8(a):
    padded = np.zeros((M, ALIGN_IN), dtype=np.float16)
    padded[:, :K] = a
    return padded.reshape(M, ALIGN_IN // 8, 8).transpose(1, 0, 2).ravel()


def pack_weight_tile_16x32(b):
    wt = np.zeros((ALIGN_OUT, ALIGN_IN), dtype=np.float16)
    wt[:N, :K] = b.T
    return wt.reshape(ALIGN_OUT // 16, 16, ALIGN_IN // 32, 32).transpose(0, 2, 1, 3).ravel()


def submit(fd, task_obj_addr, timeout, mode):
    official = mode == "official"
    submit_struct = rknpu_submit(
        flags=RKNPU_JOB_PC | RKNPU_JOB_PINGPONG,
        timeout=timeout,
        task_start=0,
        task_number=6 if official else TASKS,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=0 if official else 1,
        fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(0, 2)
    submit_struct.subcore_task[1] = rknpu_subcore_task(0, 2 if official else 0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(0, 2 if official else 0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(0, 0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(0, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def main():
    parser = argparse.ArgumentParser(description="Replay the official RKNN 424x424x424 GEMM raw command blob.")
    parser.add_argument("--mode", choices=("official", "core0"), default="official")
    parser.add_argument("--alloc-mode", choices=("raw", "official"), default="raw")
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--constant-data", action="store_true", help="Use small constant inputs/weights to rule out random data issues.")
    args = parser.parse_args()

    if args.constant_data:
        a = np.full((M, K), 0.125, dtype=np.float16)
        b = np.full((K, N), 0.25, dtype=np.float16)
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
        input_bytes = M * ALIGN_IN * 2
        weight_bytes = ALIGN_IN * ALIGN_OUT * 2
        output_bytes = max(256, ((M - 1) * ALIGN_OUT + N) * 4)
        input_alloc = align_up(input_bytes, 4096)
        weight_alloc = align_up(weight_bytes, 4096)
        output_alloc = align_up(output_bytes, 4096)

        regcmd_map, regcmd_mc = mem_allocate(fd, TASKS * SEGMENT_QWORDS * ctypes.sizeof(ctypes.c_uint64), tensor_flags)
        input_map, input_mc = mem_allocate(fd, input_alloc, tensor_flags)
        weight_map, weight_mc = mem_allocate(fd, weight_alloc, tensor_flags)
        output_map, output_mc = mem_allocate(fd, output_alloc, tensor_flags)

        np.frombuffer(input_map, dtype=np.float16, count=M * ALIGN_IN)[:] = pack_input_nc1hwc2_c2_8(a)
        np.frombuffer(weight_map, dtype=np.float16, count=ALIGN_IN * ALIGN_OUT)[:] = pack_weight_tile_16x32(b)
        np.frombuffer(output_map, dtype=np.float32)[:] = np.nan

        blob = list(CAPTURED_GEMM_REGCMD_BLOB)
        for idx, row_start in enumerate((0, TILE_ROWS[0])):
            base = idx * SEGMENT_QWORDS
            segment = blob[base:base + SEGMENT_QWORDS]
            input_dma = input_mc.dma_addr + row_start * ALIGN_IN * 2
            output_dma = output_mc.dma_addr + row_start * ALIGN_OUT * 4
            next_dma = regcmd_mc.dma_addr + SEGMENT_QWORDS * 8 if idx == 0 else 0
            patch_first(segment, TARGET_CNA, REG_CNA_FEATURE_DATA_ADDR, input_dma)
            patch_first(segment, TARGET_CNA, REG_CNA_DCOMP_ADDR0, weight_mc.dma_addr)
            patch_first(segment, TARGET_DPU, REG_DPU_DST_BASE_ADDR, output_dma)
            tail = REGCFG_AMOUNT - 4
            segment[tail + 0] = emit(TARGET_PC_REG, REG_PC_BASE_ADDRESS, next_dma & 0xFFFFFFF0) if next_dma else 0
            segment[tail + 1] = emit(TARGET_PC_REG, REG_PC_REGISTER_AMOUNTS, 55 if next_dma else 0)
            segment[tail + 2] = 0x0041000000000000
            segment[tail + 3] = emit(TARGET_PC_OP, REG_PC_OPERATION_ENABLE, 0x0D)
            for i, qword in enumerate(segment):
                ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))[base + i] = qword

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
        for idx in range(TASKS):
            tasks[idx].flags = 0
            tasks[idx].op_idx = 0
            tasks[idx].enable_mask = 0x0D
            tasks[idx].int_mask = 0x300
            tasks[idx].int_clear = 0x1FFFF
            tasks[idx].int_status = 0
            tasks[idx].regcfg_amount = REGCFG_AMOUNT
            tasks[idx].regcfg_offset = 0
            tasks[idx].regcmd_addr = regcmd_mc.dma_addr + idx * SEGMENT_QWORDS * 8
            print(f"task[{idx}] regcmd=0x{tasks[idx].regcmd_addr:x} amount={REGCFG_AMOUNT}")

        ret = submit(fd, task_mc.obj_addr, args.timeout, args.mode)
        print(f"submit ret={ret}")
        raw = np.frombuffer(output_map, dtype=np.float32, count=((M - 1) * ALIGN_OUT + N)).copy()
        got = raw[np.arange(M)[:, None] * ALIGN_OUT + np.arange(N)]
        ok = ret == 0 and np.allclose(got, expected, atol=args.atol, equal_nan=False)
        md = float(np.nanmax(np.abs(got - expected))) if got.size else 0.0
        print(f"gemm rawbuf {'PASS' if ok else 'FAIL'} max_diff={md:.6f}")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
