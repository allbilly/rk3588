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

CAPTURED_GEMM_394_REGCMD_HEX = (
    "4010b10000000102041100000000010200110000000001020c1020010020010204400e00000001100c10200100200102"
    "1010800200000102141009000000010220102700010001022410a0019f01010228100100000001022c10270000000102"
    "301000480500010234104003000001023810a001010101024010b1000000010244100d00000001024c100b0000000102"
    "5010000001000102541000000100010258100000010001025c1000000100010260100000000001026410000000000102"
    "6810000000000102701000c0f0ff0102741000000000010278100f000f0001027c103400000001028010000000000102"
    "84102700010001028810a0010000010200110000000001020411000000000102101100d0f5ff01024011000000000102"
    "441100000000010248110000000001024c11000000000102501100000000010254110000000001025811000000000102"
    "5c110000000001026011000000000102641100000000010268110000000001026c110000000001027011000000000102"
    "741100000000010278110000000001027c11000000000102801100000000010284110000000001021030010200000108"
    "143000002600010818309f01000001081c3000000000010830300000000001080c40e401000001101040020000a80110"
    "144000000000011020400060e1ff01102440100000000110304000000000011034402600000001103840670067000110"
    "3c409f019f0101104040530000000110444000000000011048400000000001104c4000000000011050406e0300000110"
    "544000000000011058409f01000001105c40000026000110604053000000011064400000000001106840000000000110"
    "6c400000000001107040830300000110744000000000011078400100000001107c400000000001108040000000000110"
    "844001000000011088400000000001109040000000000110944000000000011098400000000001109c40000000000110"
    "a040000000000110a440000000000110a840000000000110ac40000000000110c040400000000110c440000000000110"
    "0041000000000110044100000000011008410000000001100c4100000000011010410000000001101441000000000110"
    "18410000000001101c410000000001102041000000000110244100000000011028410000000001102c41000000000110"
    "10008033fbff01011400370000000101000000000000410008000d00000081004010b120000001020411000000000102"
    "00110000000001020c1020010020010204400e00000001100c1020010020010210108002000001021410090000000102"
    "20102700010001022410a0019f01010228100100000001022c1027000000010230100048050001023410400300000102"
    "3810a001010101024010b1200000010244100d00000001024c100b000000010250100000010001025410000001000102"
    "58100000010001025c100000010001026010000000000102641000000000010268100000000001027010c03ef1ff0102"
    "741000000000010278100f000f0001027c10340000000102801000000000010284102700010001028810a00100000102"
    "00110000000001020411000000000102101100d0f5ff0102401100000000010244110000000001024811000000000102"
    "4c110000000001025011000000000102541100000000010258110000000001025c110000000001026011000000000102"
    "641100000000010268110000000001026c11000000000102701100000000010274110000000001027811000000000102"
    "7c11000000000102801100000000010284110000000001021030010200000108143000002600010818309f0100000108"
    "1c3000000000010830300000000001080c40e401000001101040020000a8011014400000000001102040805de2ff0110"
    "24401000000001103040000000000110344026000000011038406700670001103c409f019f0101104040530000000110"
    "444000000000011048400000000001104c4000000000011050406e0300000110544000000000011058409f0100000110"
    "5c400000260001106040530000000110644000000000011068400000000001106c400000000001107040830300000110"
    "744000000000011078400100000001107c40000000000110804000000000011084400100000001108840000000000110"
    "9040000000000110944000000000011098400000000001109c40000000000110a040000000000110a440000000000110"
    "a840000000000110ac40000000000110c040400000000110c44000000000011000410000000001100441000000000110"
    "08410000000001100c410000000001101041000000000110144100000000011018410000000001101c41000000000110"
    "2041000000000110244100000000011028410000000001102c4100000000011010000037fbff01011400080000000101"
    "000000000000410008000d00000081004010b12000000102041100000000010200110000000001020c10200100200102"
    "04400e00000001104010b12000000102701080bdf1ff010284102700010001028810a00100000102101100d0f5ff0102"
    "2040005be3ff011058409f01000001105c400000260001101000c037fbff010114000700000001010000000000004100"
    "08000d000000810000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "000000000000000000000000000000004010b12000000102041100000000010200110000000001020c10200100200102"
    "04400e00000001107010403cf2ff010284102700010001028810a00100000102101100d0f5ff010220408058e4ff0110"
    "58409f01000001105c4000002600011010004038fbff01011400070000000101000000000000410008000d0000008100"
    "4010b12000000102041100000000010200110000000001020c1020010020010204400e0000000110701000bbf2ff0102"
    "84102700010001028810a00100000102101100d0f5ff010220400056e5ff011058409f01000001105c40000026000110"
    "1000c038fbff01011400070000000101000000000000410008000d00000081004010b120000001020411000000000102"
    "00110000000001020c1020010020010204400e00000001107010c039f3ff010284102700010001028810a00100000102"
    "101100d0f5ff010220408053e6ff011058409f01000001105c4000002600011010004039fbff01011400070000000101"
    "000000000000410008000d00000081004010b12000000102041100000000010200110000000001020c10200100200102"
    "04400e0000000110701080b8f3ff010284102700010001028810a00100000102101100d0f5ff010220400051e7ff0110"
    "58409f01000001105c400000260001101000c039fbff01011400070000000101000000000000410008000d0000008100"
    "4010b12000000102041100000000010200110000000001020c1020010020010204400e000000011070104037f4ff0102"
    "84102700010001028810a00100000102101100d0f5ff01022040804ee8ff011058409f01000001105c40000026000110"
    "1000403afbff01011400070000000101000000000000410008000d00000081004010b120000001020411000000000102"
    "00110000000001020c1020010020010204400e0000000110701000b6f4ff010284102700010001028810a00100000102"
    "101100d0f5ff01022040004ce9ff011058409f01000001105c400000260001101000c03afbff01011400070000000101"
    "000000000000410008000d00000081004010b12000000102041100000000010200110000000001020c10200100200102"
    "04400e00000001107010c034f5ff010284102700010001028810a00100000102101100d0f5ff010220408049eaff0110"
    "58409f01000001105c400000260001101000403bfbff010114000a0000000101000000000000410008000d0000008100"
    "4010b12000000102041100000000010200110000000001020c1020010020010204400e00000001101010500000000102"
    "20100400010001022c10040000000102701080b3f5ff010284100400010001028810a00100000102101100d0f5ff0102"
    "143000000300010820400047ebff0110344003000000011058409f01000001105c400000030001100000000000000000"
    "1400000000000101000000000000410008000d0000008100000000000000000000000000000000000000000000000000"
)


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


def load_regcmd_blob():
    data = bytes.fromhex(CAPTURED_GEMM_394_REGCMD_HEX)
    if len(data) != REGCMD_BYTES:
        raise RuntimeError(f"expected {REGCMD_BYTES} inline bytes, got {len(data)}")
    return [int.from_bytes(data[i:i + 8], "little") for i in range(0, len(data), 8)]


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

        blob = patch_blob(load_regcmd_blob(), input_mc.dma_addr, weight_mc.dma_addr, output_mc.dma_addr, regcmd_mc.dma_addr)
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
