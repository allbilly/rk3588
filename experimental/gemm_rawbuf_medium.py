from fcntl import ioctl
import ctypes
import mmap
import os

import numpy as np


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1

M = N = K = 384
ALIGN_IN = ALIGN_OUT = 384
FP16_BYTES = 2
FP32_BYTES = 4

TARGET_CNA = 0x0201
TARGET_DPU = 0x1001
REG_CNA_FEATURE_DATA_ADDR = 0x1070
REG_CNA_DCOMP_ADDR0 = 0x1110
REG_DPU_DST_BASE_ADDR = 0x4020


def make_regs(input_dma, weight_dma, output_dma):
    # Local standalone copy of the known-good 384x384x384 register schedule.
    def e(target, reg_addr, value):
        return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

    return [
        e(TARGET_DPU, 0x4004, 0x0E),
        e(TARGET_CNA, 0x100C, 0x120),
        e(TARGET_CNA, 0x1010, 86 << 4),
        e(TARGET_CNA, 0x1014, 0x09),
        e(TARGET_CNA, 0x1020, (1 << 16) | M),
        e(TARGET_CNA, 0x1024, ((ALIGN_IN - 1) << 16) | ALIGN_IN),
        e(TARGET_CNA, 0x1028, 1),
        e(TARGET_CNA, 0x102C, M),
        e(TARGET_CNA, 0x1030, ALIGN_IN * FP16_BYTES * ALIGN_OUT),
        e(TARGET_CNA, 0x1034, ALIGN_IN * FP16_BYTES),
        e(TARGET_CNA, 0x1038, (1 << 24) | (1 << 16) | ALIGN_OUT),
        e(TARGET_CNA, 0x1040, ((12 - 9) << 4) | 9),
        e(TARGET_CNA, 0x1044, 12),
        e(TARGET_CNA, 0x104C, 0x0B),
        e(TARGET_CNA, 0x1050, 1 << 16),
        e(TARGET_CNA, 0x1054, 1 << 16),
        e(TARGET_CNA, 0x1058, 1 << 16),
        e(TARGET_CNA, 0x105C, 1 << 16),
        e(TARGET_CNA, 0x1070, input_dma),
        e(TARGET_CNA, 0x1078, (15 << 16) | 15),
        e(TARGET_CNA, 0x107C, 48),
        e(TARGET_CNA, 0x1080, 0),
        e(TARGET_CNA, 0x1084, (1 << 16) | M),
        e(TARGET_CNA, 0x1088, ALIGN_IN),
        e(TARGET_CNA, 0x1110, weight_dma),
        e(0x0801, 0x3010, (2 << 8) | 1),
        e(0x0801, 0x3014, (M - 1) << 16),
        e(0x0801, 0x3018, ALIGN_OUT - 1),
        e(0x0801, 0x3030, 0),
        e(TARGET_DPU, 0x400C, (15 << 5) | (2 << 1)),
        e(TARGET_DPU, 0x4010, (5 << 29) | (2 << 26) | 2),
        e(TARGET_DPU, 0x4020, output_dma),
        e(TARGET_DPU, 0x4024, ALIGN_OUT << 4),
        e(TARGET_DPU, 0x4030, 0),
        e(TARGET_DPU, 0x4034, M - 1),
        e(TARGET_DPU, 0x4038, 0),
        e(TARGET_DPU, 0x403C, ((ALIGN_OUT - 1) << 16) | (ALIGN_OUT - 1)),
        e(TARGET_DPU, 0x4040, 0x53),
        e(TARGET_DPU, 0x4050, 0x36E),
        e(TARGET_DPU, 0x4058, ALIGN_OUT - 1),
        e(TARGET_DPU, 0x405C, (M - 1) << 16),
        e(TARGET_DPU, 0x4060, 0x53),
        e(TARGET_DPU, 0x4070, 0x383),
        e(TARGET_DPU, 0x40C0, (ALIGN_OUT * 4) << 4),
        e(0x0081, 0x0008, 0x0D),
    ]

CMDBUF_BLOB = [
    0x02010000002a1040, 0x0201000000001104, 0x0201000000001100, 0x020120000120100c,
    0x10010000000e4004, 0x020120000120100c, 0x0201000000001010, 0x0201000000091014,
    0x0201000101801020, 0x0201017f01801024, 0x0201000000011028, 0x020100000180102c,
    0x0201000480001030, 0x0201000003001034, 0x0201010101801038, 0x02010000002a1040,
    0x02010000000c1044, 0x02010000000b104c, 0x0201000100001050, 0x0201000100001054,
    0x0201000100001058, 0x020100010000105c, 0x0201000000001060, 0x0201000000001064,
    0x0201000000001068, 0x0201fff140001070, 0x0201000000001074, 0x0201000f000f1078,
    0x020100000034107c, 0x0201000000001080, 0x0201000101801084, 0x0201000001801088,
    0x0201000000001100, 0x0201000000001104, 0x0201fff630001110, 0x0201000000001140,
    0x0201000000001144, 0x0201000000001148, 0x020100000000114c, 0x0201000000001150,
    0x0201000000001154, 0x0201000000001158, 0x020100000000115c, 0x0201000000001160,
    0x0201000000001164, 0x0201000000001168, 0x020100000000116c, 0x0201000000001170,
    0x0201000000001174, 0x0201000000001178, 0x020100000000117c, 0x0201000000001180,
    0x0201000000001184, 0x0801000002013010, 0x0801017f00003014, 0x0801000001803018,
    0x080100000000301c, 0x0801000000003030, 0x1001000001e4400c, 0x1001a80000024010,
    0x1001000000004014, 0x1001ffe220004020, 0x1001000000104024, 0x1001000000004030,
    0x10010000017f4034, 0x1001006700674038, 0x1001017f017f403c, 0x1001000000534040,
    0x1001000000004044, 0x1001000000004048, 0x100100000000404c, 0x10010000036e4050,
    0x1001000000004054, 0x10010000017f4058, 0x1001017f0000405c, 0x1001000000534060,
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


DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_ACTION = _IOWR("d", 0x40, ctypes.sizeof(rknpu_action))
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


def patch_first(blob, target, addr, value):
    for i, qword in enumerate(blob):
        if ((qword >> 48) & 0xFFFF) == target and (qword & 0xFFFF) == addr:
            blob[i] = emit(target, addr, value)
            return
    raise RuntimeError(f"missing target=0x{target:x} addr=0x{addr:x}")


def pack_input_row_major(a):
    padded = np.zeros((M, ALIGN_IN), dtype=np.float16)
    padded[:, :K] = a
    return padded.ravel()


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
    a = np.ones((M, K), dtype=np.float16)
    b = np.ones((K, N), dtype=np.float16)
    expected = np.full((M, N), K, dtype=np.float32)

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        regcmd_map, regcmd_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
        input_map, input_mc = mem_allocate(fd, align_up(M * ALIGN_IN * 2, 4096), RKNPU_MEM_NON_CACHEABLE)
        weight_map, weight_mc = mem_allocate(fd, align_up(ALIGN_IN * ALIGN_OUT * 2, 4096), RKNPU_MEM_NON_CACHEABLE)
        output_words = (M - 1) * ALIGN_OUT + N
        output_map, output_mc = mem_allocate(fd, align_up(max(256, output_words * 4), 4096), RKNPU_MEM_NON_CACHEABLE)

        np.frombuffer(input_map, dtype=np.float16, count=M * ALIGN_IN)[:] = pack_input_row_major(a)
        np.frombuffer(weight_map, dtype=np.float16, count=ALIGN_IN * ALIGN_OUT)[:] = pack_weight_tile_16x32(b)
        np.frombuffer(output_map, dtype=np.float32)[:] = np.nan

        blob = make_regs(input_mc.dma_addr, weight_mc.dma_addr, output_mc.dma_addr)
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
        raw = np.frombuffer(output_map, dtype=np.float32, count=output_words).copy()
        got = raw[np.arange(M)[:, None] * ALIGN_OUT + np.arange(N)]
        ok = ret == 0 and np.allclose(got, expected, atol=0.1)
        max_diff = float(np.max(np.abs(got - expected)))
        print(f"submit ret={ret}")
        print(f"384x384x384 rawbuf {'PASS' if ok else 'FAIL'} max_diff={max_diff:.6f}")
        return 0 if ok else 1
    finally:
        os.close(fd)


def align_up(value, align):
    return ((value + align - 1) // align) * align


if __name__ == "__main__":
    raise SystemExit(main())
