from fcntl import ioctl
import ctypes
import mmap
import os

import rocket_runtime as rt


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0 << 1
RKNPU_JOB_NONBLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 0x4


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
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]


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


class struct_rknpu_task(ctypes.Structure):
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


def mem_allocate(fd, size, flags=0):
    return rt.mem_allocate(fd, size, flags)


def submit(fd, task_obj_addr):
    submit_struct = rknpu_submit(
        flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
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
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def regcmd(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def main():
    fd = rt.open_rocket_device()
    try:
        task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        regcmd_map, regcmd_mc = mem_allocate(fd, 1024, RKNPU_MEM_NON_CACHEABLE)
        input_map, input_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
        weight_map, weight_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
        output_map, output_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
        cmdbuf = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
        inputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), ctypes.POINTER(ctypes.c_uint16))
        weights = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), ctypes.POINTER(ctypes.c_uint16))
        outputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), ctypes.POINTER(ctypes.c_uint16))

        cmdbuf_blob = [
            0x10010000000e4004,
            0x20010000000e5004,
            0x1001000001e5400c,
            0x1001480000024010,
            0x100100070007403c,
            0x1001000000004030,
            0x1001108202c04070,
            0x200100000000500c,
            0x2001000000005010,
            0x2001000700075014,
            0x2001400000085034,
            regcmd(0x1001, 0x4020, output_mc.dma_addr),
            regcmd(0x2001, 0x5018, input_mc.dma_addr),
            regcmd(0x2001, 0x5038, weight_mc.dma_addr),
            0x2001000178495044,
            0x0081000000180008,
        ]
        for i, value in enumerate(cmdbuf_blob):
            cmdbuf[i] = value

        for i in range(8):
            inputs[i] = 3
            weights[i] = 5
            outputs[i] = 0

        tasks[0].flags = 0
        tasks[0].op_idx = 4
        tasks[0].enable_mask = 0x18
        tasks[0].int_mask = 0x300
        tasks[0].int_clear = 0x1FFFF
        tasks[0].int_status = 0
        tasks[0].regcfg_amount = len(cmdbuf_blob)
        tasks[0].regcfg_offset = 0
        tasks[0].regcmd_addr = regcmd_mc.dma_addr

        for bo in (regcmd_mc, input_mc, weight_mc, output_mc):
            rt.fini_bo(fd, bo)
        ret = rt.submit(fd, tasks, 1, in_bos=[regcmd_mc, input_mc, weight_mc], out_bos=[output_mc])
        rt.prep_bo(fd, output_mc)
        got = [outputs[i] for i in range(8)]
        print(f"SUBMIT ret={ret}")
        print(got)
        ok = ret == 0 and got == [8] * 8
        print("ADD RAWBUF PASS" if ok else "ADD RAWBUF FAIL")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
