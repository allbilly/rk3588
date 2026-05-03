from fcntl import ioctl
import ctypes
import mmap
import os
import sys

import numpy as np


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1

M = N = K = 384


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


def _iowr(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)


DRM_IOCTL_RKNPU_ACTION = _iowr("d", 0x40, ctypes.sizeof(rknpu_action))
DRM_IOCTL_RKNPU_SUBMIT = _iowr("d", 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_MEM_CREATE = _iowr("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _iowr("d", 0x43, ctypes.sizeof(rknpu_mem_map))


def align_up(value, align):
    return ((value + align - 1) // align) * align


def mem_allocate(fd, size, flags):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc


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
    submit_struct.subcore_task[3] = rknpu_subcore_task(0, 0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(0, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def main():
    sys.path.insert(0, "examples")
    import gemm

    align_in, align_out, _, pad_k = gemm._gemm_layout(M, N, K)
    a = np.ones((M, K), dtype=np.float16)
    b = np.ones((K, N), dtype=np.float16)
    expected = np.full((M, N), K, dtype=np.float32)

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        regcmd_map, regcmd_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
        input_map, input_mc = mem_allocate(fd, align_up(M * align_in * 2, 4096), RKNPU_MEM_NON_CACHEABLE)
        weight_map, weight_mc = mem_allocate(fd, align_up(align_in * align_out * 2, 4096), RKNPU_MEM_NON_CACHEABLE)
        output_words = (M - 1) * align_out + N
        output_map, output_mc = mem_allocate(fd, align_up(max(256, output_words * 4), 4096), RKNPU_MEM_NON_CACHEABLE)

        in_pack = np.zeros(align_in * M, dtype=np.float16)
        wt_pack = np.zeros(align_in * align_out, dtype=np.float16)
        pack_input = gemm.pack_input_row_major if pad_k else gemm.get_input_packer(M, N, K, align_in)
        pack_weight = gemm.get_weight_packer(M, N, K, align_in)
        pack_input(M, N, K, a, in_pack, align_in)
        pack_weight(M, N, K, b, wt_pack, align_in, align_out)
        np.frombuffer(input_map, dtype=np.float16, count=in_pack.size)[:] = in_pack
        np.frombuffer(weight_map, dtype=np.float16, count=wt_pack.size)[:] = wt_pack
        np.frombuffer(output_map, dtype=np.float32)[:] = np.nan

        regs = gemm.make_gemm_regs(M, N, K, input_mc.dma_addr, weight_mc.dma_addr, output_mc.dma_addr)
        regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
        for i, qword in enumerate(regs):
            regcmd[i] = qword

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
        tasks[0].flags = 0
        tasks[0].op_idx = 4
        tasks[0].enable_mask = 0x18
        tasks[0].int_mask = 0x300
        tasks[0].int_clear = 0x1FFFF
        tasks[0].int_status = 0
        tasks[0].regcfg_amount = len(regs)
        tasks[0].regcfg_offset = 0
        tasks[0].regcmd_addr = regcmd_mc.dma_addr

        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))
        ret = submit(fd, task_mc.obj_addr)
        raw = np.frombuffer(output_map, dtype=np.float32, count=output_words).copy()
        got = gemm.get_output_decoder(M, N, K, align_out)(M, N, K, raw, align_out)
        ok = ret == 0 and np.allclose(got, expected, atol=0.1)
        max_diff = float(np.max(np.abs(got - expected)))
        print(f"submit ret={ret}")
        print(f"{M}x{N}x{K}: {'PASS' if ok else 'FAIL'} (max_diff={max_diff:.4f})")
        return 0 if ok else 1
    finally:
        os.close(fd)
        os.close(gemm.fd)


if __name__ == "__main__":
    raise SystemExit(main())
