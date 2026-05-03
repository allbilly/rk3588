from fcntl import ioctl
import ctypes
import mmap
import os

import numpy as np


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0x2
RKNPU_JOB_PINGPONG = 0x4

TASKS = 3
VALUES = [(3, 5), (7, 11), (13, 17)]
BODY_QWORDS = 16
SEGMENT_QWORDS = BODY_QWORDS + 4
REG_BLOCK_QWORDS = 32

TARGET_DPU = 0x1001
TARGET_RDMA = 0x2001
TARGET_PC = 0x0081
TARGET_PC_REG = 0x0101
TARGET_VERSION = 0x0041


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


def mem_allocate(fd, size, flags=0):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc


def emit(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def add_segment(input_dma, weight_dma, output_dma, next_dma, segment_amount, elements):
    width = (elements + 7) // 8 - 1
    body = [
        emit(TARGET_DPU, 0x4004, 0x0000000E),
        emit(TARGET_DPU, 0x400C, 0x000001E5),
        emit(TARGET_DPU, 0x4010, 0x48000002),
        emit(TARGET_DPU, 0x403C, 0x00070007),
        emit(TARGET_DPU, 0x4030, width),
        emit(TARGET_DPU, 0x4070, 0x108202C0),
        emit(TARGET_DPU, 0x4084, 0x00010001),
        emit(TARGET_RDMA, 0x5004, 0x0000000E),
        emit(TARGET_RDMA, 0x500C, width),
        emit(TARGET_RDMA, 0x5010, 0x00000000),
        emit(TARGET_RDMA, 0x5014, 0x00000007),
        emit(TARGET_RDMA, 0x5034, 0x40000008),
        emit(TARGET_DPU, 0x4020, output_dma),
        emit(TARGET_RDMA, 0x5018, input_dma),
        emit(TARGET_RDMA, 0x5038, weight_dma),
        emit(TARGET_RDMA, 0x5044, 0x00017849),
    ]
    tail = [
        emit(TARGET_PC_REG, 0x0010, next_dma & 0xFFFFFFF0) if next_dma else 0,
        emit(TARGET_PC_REG, 0x0014, segment_amount if next_dma else 0),
        emit(TARGET_VERSION, 0, 0),
        emit(TARGET_PC, 0x0008, 0x00000018),
    ]
    return body + tail


def submit(fd, task_obj_addr, flags, mode):
    official = mode == "official"
    submit_struct = rknpu_submit(
        flags=flags,
        timeout=6000,
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
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Three-task decoded ADD PC-chain probe.")
    parser.add_argument("--mode", choices=("core0", "official"), default="core0")
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG)
    parser.add_argument("--segment-elements", type=int, default=8, help="FP16 values per chained ADD segment.")
    args = parser.parse_args()
    if args.segment_elements <= 0 or args.segment_elements % 8:
        raise ValueError("--segment-elements must be a positive multiple of 8")

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        regcmd_map, regcmd_mc = mem_allocate(fd, TASKS * REG_BLOCK_QWORDS * ctypes.sizeof(ctypes.c_uint64), RKNPU_MEM_NON_CACHEABLE)
        byte_stride = max(4096, args.segment_elements * ctypes.sizeof(ctypes.c_uint16))
        elem_stride = byte_stride // ctypes.sizeof(ctypes.c_uint16)
        input_map, input_mc = mem_allocate(fd, TASKS * byte_stride, RKNPU_MEM_NON_CACHEABLE)
        weight_map, weight_mc = mem_allocate(fd, TASKS * byte_stride, RKNPU_MEM_NON_CACHEABLE)
        output_map, output_mc = mem_allocate(fd, TASKS * byte_stride, RKNPU_MEM_NON_CACHEABLE)

        regs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
        inputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), ctypes.POINTER(ctypes.c_uint16))
        weights = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), ctypes.POINTER(ctypes.c_uint16))
        outputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), ctypes.POINTER(ctypes.c_uint16))

        for task_idx, (aval, bval) in enumerate(VALUES):
            base_elem = task_idx * elem_stride
            for i in range(args.segment_elements):
                inputs[base_elem + i] = aval
                weights[base_elem + i] = bval
                outputs[base_elem + i] = 0

        for task_idx in range(TASKS):
            next_dma = regcmd_mc.dma_addr + (task_idx + 1) * REG_BLOCK_QWORDS * 8 if task_idx + 1 < TASKS else 0
            segment = add_segment(
                input_mc.dma_addr + task_idx * byte_stride,
                weight_mc.dma_addr + task_idx * byte_stride,
                output_mc.dma_addr + task_idx * byte_stride,
                next_dma,
                BODY_QWORDS,
                args.segment_elements,
            )
            base = task_idx * REG_BLOCK_QWORDS
            for i in range(REG_BLOCK_QWORDS):
                regs[base + i] = 0
            for i, value in enumerate(segment):
                regs[base + i] = value

            tasks[task_idx].flags = 0
            tasks[task_idx].op_idx = 4
            tasks[task_idx].enable_mask = 0x18
            tasks[task_idx].int_mask = 0x300
            tasks[task_idx].int_clear = 0x1FFFF
            tasks[task_idx].int_status = 0
            tasks[task_idx].regcfg_amount = SEGMENT_QWORDS
            tasks[task_idx].regcfg_offset = 0
            tasks[task_idx].regcmd_addr = regcmd_mc.dma_addr + base * 8

        ret = submit(fd, task_mc.obj_addr, args.flags, args.mode)
        got = [
            [outputs[task_idx * elem_stride + i] for i in range(args.segment_elements)]
            for task_idx in range(TASKS)
        ]
        expected = [[a + b] * args.segment_elements for a, b in VALUES]
        ok = ret == 0 and got == expected
        print(f"submit ret={ret} mode={args.mode} segment_elements={args.segment_elements} flags=0x{args.flags:x}")
        print("got_head", [row[:16] for row in got])
        print("expected_head", [row[:16] for row in expected])
        print("ADD PCCHAIN PASS" if ok else "ADD PCCHAIN FAIL")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
