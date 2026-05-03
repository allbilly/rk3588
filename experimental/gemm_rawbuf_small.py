from fcntl import ioctl
import ctypes
import mmap
import os

import numpy as np


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1

M = 2
N = 2
K = 2
ALIGN_IN = 32
ALIGN_OUT = 32
FP16_BYTES = 2
FP32_BYTES = 4

TARGET_CNA = 0x0201
TARGET_DPU = 0x1001
REG_CNA_FEATURE_DATA_ADDR = 0x1070
REG_CNA_DCOMP_ADDR0 = 0x1110
REG_DPU_DST_BASE_ADDR = 0x4020

CMDBUF_BLOB = [
    0x10010000000e4004, 0x020120000120100c, 0x0201000000301010, 0x0201000000091014,
    0x0201000100021020, 0x0201001f00201024, 0x0201000000011028, 0x020100000002102c,
    0x0201000008001030, 0x0201000000401034, 0x0201010100201038, 0x0201000000b11040,
    0x0201000000011044, 0x02010000000b104c, 0x0201000100001050, 0x0201000100001054,
    0x0201000100001058, 0x020100010000105c, 0x0201000000001070, 0x0201000f000f1078,
    0x020100000004107c, 0x0201000000001080, 0x0201000100021084, 0x0201000000201088,
    0x0201000000001110, 0x0801000002013010, 0x0801000100003014, 0x08010000001f3018,
    0x0801000000003030, 0x1001000001e4400c, 0x1001a80000024010, 0x1001000000004020,
    0x1001000000104024, 0x1001000000004030, 0x1001000000014034, 0x1001000700074038,
    0x1001001f001f403c, 0x1001000000534040, 0x10010000036e4050, 0x10010000001f4058,
    0x100100010000405c, 0x1001000000534060, 0x1001000003834070, 0x10010000004040c0,
    0x00810000000d0008,
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


class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]


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


DRM_IOCTL_RKNPU_ACTION = _IOWR("d", 0x40, ctypes.sizeof(rknpu_action))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR("d", 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR("d", 0x43, ctypes.sizeof(rknpu_mem_map))


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
    raise RuntimeError(f"missing target=0x{target:x} addr=0x{addr:x}")


def pack_weight_tile_16x32(b):
    wt = np.zeros((ALIGN_OUT, ALIGN_IN), dtype=np.float16)
    wt[:N, :K] = b.T
    return wt.reshape(ALIGN_OUT // 16, 16, ALIGN_IN // 32, 32).transpose(0, 2, 1, 3).ravel()


def submit(fd, task_obj_addr):
    submit_struct = rknpu_submit(
        flags=0x1,
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
    submit_struct.subcore_task[0] = rknpu_subcore_task(0, 1)
    submit_struct.subcore_task[1] = rknpu_subcore_task(1, 0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(2, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def main():
    a = np.array([[1, 2], [3, 4]], dtype=np.float16)
    b = np.array([[5, 6], [7, 8]], dtype=np.float16)
    expected = a @ b

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        regcmd_map, regcmd_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
        input_map, input_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
        weight_map, weight_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
        output_map, output_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)

        in_pack = np.zeros(M * ALIGN_IN, dtype=np.float16)
        in_pack.reshape(M, ALIGN_IN)[:, :K] = a
        np.frombuffer(input_map, dtype=np.float16, count=M * ALIGN_IN)[:] = in_pack
        np.frombuffer(weight_map, dtype=np.float16, count=ALIGN_IN * ALIGN_OUT)[:] = pack_weight_tile_16x32(b)
        np.frombuffer(output_map, dtype=np.float32)[:] = np.nan

        blob = list(CMDBUF_BLOB)
        patch_first(blob, TARGET_CNA, REG_CNA_FEATURE_DATA_ADDR, input_mc.dma_addr)
        patch_first(blob, TARGET_CNA, REG_CNA_DCOMP_ADDR0, weight_mc.dma_addr)
        patch_first(blob, TARGET_DPU, REG_DPU_DST_BASE_ADDR, output_mc.dma_addr)
        regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
        for i, qword in enumerate(blob):
            regcmd[i] = qword

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
        tasks[0].flags = 0
        tasks[0].op_idx = 4
        tasks[0].enable_mask = 0x18
        tasks[0].int_mask = 0x300
        tasks[0].int_clear = 0x1FFFF
        tasks[0].int_status = 0
        tasks[0].regcfg_amount = len(blob)
        tasks[0].regcfg_offset = 0
        tasks[0].regcmd_addr = regcmd_mc.dma_addr

        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))
        ret = submit(fd, task_mc.obj_addr)
        raw = np.frombuffer(output_map, dtype=np.float32, count=((M - 1) * ALIGN_OUT + N)).copy()
        got = raw[np.arange(M)[:, None] * ALIGN_OUT + np.arange(N)]
        ok = ret == 0 and np.allclose(got, expected, atol=0.1)
        print(f"submit ret={ret}")
        print(got)
        print(expected)
        print("gemm rawbuf small PASS" if ok else "gemm rawbuf small FAIL")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
