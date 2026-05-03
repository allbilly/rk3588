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

M = N = K = 385
ALIGN_IN = ALIGN_OUT = 416
REGCFG_AMOUNT = 108
SEGMENT_QWORDS = 112

TARGET_CNA = 0x0201
TARGET_DPU = 0x1001
REG_CNA_FEATURE_DATA_ADDR = 0x1070
REG_CNA_DCOMP_ADDR0 = 0x1110
REG_DPU_DST_BASE_ADDR = 0x4020

CAPTURED_GEMM_385_REGCMD_BLOB = [
    0x02010000002a1040, 0x0201000000001104, 0x0201000000001100, 0x020120000120100c,
    0x10010000000e4004, 0x020120000120100c, 0x0201000005001010, 0x0201000000091014,
    0x0201000101811020, 0x0201019f01a01024, 0x0201000000011028, 0x020100000181102c,
    0x0201000548001030, 0x0201000003401034, 0x0201010101a01038, 0x02010000002a1040,
    0x02010000000d1044, 0x02010000000b104c, 0x0201000100001050, 0x0201000100001054,
    0x0201000100001058, 0x020100010000105c, 0x0201000000001060, 0x0201000000001064,
    0x0201000000001068, 0x0201fff140001070, 0x0201000000001074, 0x0201000f000f1078,
    0x020100000034107c, 0x0201000000001080, 0x0201000101811084, 0x0201000001a01088,
    0x0201000000001100, 0x0201000000001104, 0x0201fff630001110, 0x0201000000001140,
    0x0201000000001144, 0x0201000000001148, 0x020100000000114c, 0x0201000000001150,
    0x0201000000001154, 0x0201000000001158, 0x020100000000115c, 0x0201000000001160,
    0x0201000000001164, 0x0201000000001168, 0x020100000000116c, 0x0201000000001170,
    0x0201000000001174, 0x0201000000001178, 0x020100000000117c, 0x0201000000001180,
    0x0201000000001184, 0x0801000002013010, 0x0801018000003014, 0x08010000019f3018,
    0x080100000000301c, 0x0801000000003030, 0x1001000001e4400c, 0x1001a80000024010,
    0x1001000000004014, 0x1001ffe220004020, 0x1001000000104024, 0x1001000000004030,
    0x1001000001804034, 0x1001006700674038, 0x1001019f019f403c, 0x1001000000534040,
    0x1001000000004044, 0x1001000000004048, 0x100100000000404c, 0x10010000036e4050,
    0x1001000000004054, 0x10010000019f4058, 0x100101800000405c, 0x1001000000534060,
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


def mem_allocate(fd, size, flags):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc


def emit(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def patch_first(blob, target, addr, value):
    for i, qword in enumerate(blob):
        if ((qword >> 48) & 0xFFFF) == target and (qword & 0xFFFF) == addr:
            blob[i] = emit(target, addr, value)
            return
    raise RuntimeError(f"missing register target=0x{target:x} addr=0x{addr:x}")


def pack_input_nc1hwc2_c2_8(a):
    padded = np.zeros((M, ALIGN_IN), dtype=np.float16)
    padded[:, :K] = a
    return padded.reshape(M, ALIGN_IN // 8, 8).transpose(1, 0, 2).ravel()


def pack_input_row_major(a):
    padded = np.zeros((M, ALIGN_IN), dtype=np.float16)
    padded[:, :K] = a
    return padded.ravel()


def pack_weight_tile_16x32(b):
    wt = np.zeros((ALIGN_OUT, ALIGN_IN), dtype=np.float16)
    wt[:N, :K] = b.T
    return wt.reshape(ALIGN_OUT // 16, 16, ALIGN_IN // 32, 32).transpose(0, 2, 1, 3).ravel()


def submit(fd, task_obj_addr, timeout, mode, flags):
    official = mode == "official"
    submit_struct = rknpu_submit(
        flags=flags,
        timeout=timeout,
        task_start=0,
        task_number=3 if official else 1,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=0 if official else 1,
        fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(0, 1)
    submit_struct.subcore_task[1] = rknpu_subcore_task(0, 1 if official else 0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(0, 1 if official else 0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(0, 0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(0, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def main():
    parser = argparse.ArgumentParser(description="Replay the official RKNN 385x385x385 GEMM raw command blob.")
    parser.add_argument("--mode", choices=("official", "core0"), default="official")
    parser.add_argument("--alloc-mode", choices=("raw", "official"), default="raw")
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--submit-flags", type=lambda x: int(x, 0), default=RKNPU_JOB_PC | RKNPU_JOB_PINGPONG)
    parser.add_argument("--regcfg-amount", type=int, default=REGCFG_AMOUNT)
    parser.add_argument("--op-idx", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--enable-mask", type=lambda x: int(x, 0), default=0x0D)
    parser.add_argument("--print-sample", action="store_true")
    parser.add_argument("--decode", choices=("linear", "c2-4"), default="linear")
    parser.add_argument("--input-layout", choices=("nc1hwc2-c2-8", "row-major"), default="row-major")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--constant-data", action="store_true")
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
        task_map, task_mc = mem_allocate(fd, ctypes.sizeof(rknpu_task), task_flags)
        regcmd_map, regcmd_mc = mem_allocate(fd, SEGMENT_QWORDS * ctypes.sizeof(ctypes.c_uint64), tensor_flags)
        input_map, input_mc = mem_allocate(fd, align_up(M * ALIGN_IN * 2, 4096), tensor_flags)
        weight_map, weight_mc = mem_allocate(fd, align_up(ALIGN_IN * ALIGN_OUT * 2, 4096), tensor_flags)
        output_words = (M - 1) * ALIGN_OUT + N
        output_map, output_mc = mem_allocate(fd, align_up(max(256, output_words * 4), 4096), tensor_flags)

        input_pack = pack_input_row_major(a) if args.input_layout == "row-major" else pack_input_nc1hwc2_c2_8(a)
        np.frombuffer(input_map, dtype=np.float16, count=M * ALIGN_IN)[:] = input_pack
        np.frombuffer(weight_map, dtype=np.float16, count=ALIGN_IN * ALIGN_OUT)[:] = pack_weight_tile_16x32(b)
        np.frombuffer(output_map, dtype=np.float32)[:] = np.nan

        blob = list(CAPTURED_GEMM_385_REGCMD_BLOB)
        patch_first(blob, TARGET_CNA, REG_CNA_FEATURE_DATA_ADDR, input_mc.dma_addr)
        patch_first(blob, TARGET_CNA, REG_CNA_DCOMP_ADDR0, weight_mc.dma_addr)
        patch_first(blob, TARGET_DPU, REG_DPU_DST_BASE_ADDR, output_mc.dma_addr)
        regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
        for i, qword in enumerate(blob):
            regcmd[i] = qword

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
        tasks[0].flags = 0
        tasks[0].op_idx = args.op_idx
        tasks[0].enable_mask = args.enable_mask
        tasks[0].int_mask = 0x300
        tasks[0].int_clear = 0x1FFFF
        tasks[0].int_status = 0
        tasks[0].regcfg_amount = args.regcfg_amount
        tasks[0].regcfg_offset = 0
        tasks[0].regcmd_addr = regcmd_mc.dma_addr

        print(f"task[0] regcmd=0x{tasks[0].regcmd_addr:x} amount={tasks[0].regcfg_amount} flags=0x{args.submit_flags:x}")
        ret = submit(fd, task_mc.obj_addr, args.timeout, args.mode, args.submit_flags)
        print(f"submit ret={ret}")
        raw = np.frombuffer(output_map, dtype=np.float32, count=M * ALIGN_OUT).copy()
        if args.decode == "c2-4":
            got = raw[: ALIGN_OUT // 4 * M * 4].reshape(ALIGN_OUT // 4, M, 4).transpose(1, 0, 2).reshape(M, ALIGN_OUT)[:, :N].copy()
        else:
            got = raw[np.arange(M)[:, None] * ALIGN_OUT + np.arange(N)]
        ok = ret == 0 and np.allclose(got, expected, atol=args.atol, equal_nan=False)
        max_diff = float(np.nanmax(np.abs(got - expected)))
        if args.print_sample:
            print("sample", got[:2, :8])
            print("tail", got[-4:, :8])
            print("row_nonzero_tail", np.count_nonzero(got, axis=1)[-16:])
            print("nan_count", int(np.isnan(got).sum()), "nonzero_count", int(np.count_nonzero(got)))
            print("raw_nonzero_count", int(np.count_nonzero(raw)), "raw_size", raw.size)
        print(f"385x385x385 rawbuf {'PASS' if ok else 'FAIL'} max_diff={max_diff:.6f}")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
