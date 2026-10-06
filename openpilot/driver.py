"""Small vendor RKNPU allocation/submission API using only Python's stdlib.

The layouts come from rknpu-ioctl.h for the running 0.9.8 driver. All submissions
are blocking. Closing a device frees its own buffers after the job completes.
"""
import ctypes as C
from fcntl import flock, ioctl, LOCK_EX, LOCK_UN
import glob
import mmap
import os
from pathlib import Path


class Create(C.Structure):
    _fields_ = [("handle", C.c_uint32), ("flags", C.c_uint32),
                ("size", C.c_uint64), ("obj_addr", C.c_uint64),
                ("dma_addr", C.c_uint64), ("sram_size", C.c_uint64)]


class Map(C.Structure):
    _fields_ = [("handle", C.c_uint32), ("reserved", C.c_uint32), ("offset", C.c_uint64)]


class Destroy(C.Structure):
    _fields_ = [("handle", C.c_uint32), ("reserved", C.c_uint32), ("obj_addr", C.c_uint64)]


class Sync(C.Structure):
    _fields_ = [("flags", C.c_uint32), ("reserved", C.c_uint32),
                ("obj_addr", C.c_uint64), ("offset", C.c_uint64), ("size", C.c_uint64)]


class Range(C.Structure):
    _fields_ = [("task_start", C.c_uint32), ("task_number", C.c_uint32)]


class Submit(C.Structure):
    _fields_ = [("flags", C.c_uint32), ("timeout", C.c_uint32),
                ("task_start", C.c_uint32), ("task_number", C.c_uint32),
                ("task_counter", C.c_uint32), ("priority", C.c_int32),
                ("task_obj_addr", C.c_uint64), ("iommu_domain_id", C.c_uint32),
                ("reserved", C.c_uint32), ("task_base_addr", C.c_uint64),
                ("hw_elapse_time", C.c_int64), ("core_mask", C.c_uint32),
                ("fence_fd", C.c_int32), ("subcore_task", Range * 5)]


class Task(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("flags", "op_idx", "enable_mask", "int_mask", "int_clear",
                 "int_status", "regcfg_amount", "regcfg_offset")] + [("regcmd_addr", C.c_uint64)]


def command(number, structure):
    return (3 << 30) | (C.sizeof(structure) << 16) | (ord("d") << 8) | number


def align(value, multiple=4096):
    return (value + multiple - 1) // multiple * multiple


class Buffer:
    def __init__(self, device, size, flags):
        self.device = device
        self.info = Create(size=size, flags=flags)
        ioctl(device.fd, command(0x42, Create), self.info)
        if self.info.size < size or self.info.dma_addr + self.info.size > 1 << 32:
            ioctl(device.fd, command(0x44, Destroy), Destroy(handle=self.info.handle, obj_addr=self.info.obj_addr))
            raise ValueError("Buffer cannot be addressed by the 32-bit NPU registers")
        mapping = Map(handle=self.info.handle)
        ioctl(device.fd, command(0x43, Map), mapping)
        self.mapping = mmap.mmap(device.fd, self.info.size, mmap.MAP_SHARED,
                                 mmap.PROT_READ | mmap.PROT_WRITE, offset=mapping.offset)
        self.size = self.info.size
        self.dma = self.info.dma_addr
        self.obj = self.info.obj_addr

    def write(self, data, offset=0):
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError("Write exceeds allocated NPU buffer")
        self.mapping[offset:offset + len(data)] = data

    def sync(self, direction):
        if self.info.flags & 2:  # RKNPU_MEM_CACHEABLE
            ioctl(self.device.fd, command(0x45, Sync),
                  Sync(flags=direction, obj_addr=self.obj, size=self.size))

    def close(self):
        self.mapping.close()
        ioctl(self.device.fd, command(0x44, Destroy), Destroy(handle=self.info.handle, obj_addr=self.obj))


class Device:
    def __init__(self, path=None):
        if C.sizeof(Task) != 40 or C.sizeof(Submit) != 104:
            raise RuntimeError("Unexpected vendor ioctl ABI; requires 64-bit Linux")
        if path is None:
            cards = [p for p in sorted(glob.glob("/dev/dri/card[0-9]*"))
                     if (Path("/sys/class/drm") / Path(p).name / "device/driver").resolve().name.lower() == "rknpu"]
            if len(cards) != 1:
                raise RuntimeError(f"Expected one RKNPU DRM card, found {cards}")
            path = cards[0]
        self.lock = open("/tmp/rk3588_npu_submit.lock", "a+b")
        flock(self.lock, LOCK_EX)
        self.fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        self.buffers = []
        self.busy = False

    def allocate(self, size, flags=0):
        if size <= 0:
            raise ValueError("NPU buffers must have positive sizes")
        buffer = Buffer(self, size, flags)
        self.buffers.append(buffer)
        return buffer

    def submit(self, task_buffer, settings):
        first, count = settings["task_start"], settings["task_number"]
        if not count:
            raise ValueError("Empty task submission")
        if not task_buffer.info.flags & 8:
            raise ValueError("Task descriptor buffer must have kernel mapping")
        if settings["flags"] not in (1, 5) or settings["core_mask"] != 1:
            raise ValueError("Only verified blocking, single-core PC submissions are supported")
        ranges = settings["subcores"]
        if len(ranges) != 5 or not ranges[0][1]:
            raise ValueError("Invalid subcore task layout")
        for start, number in ranges:
            if start < 0 or number < 0 or start + number > first + count:
                raise ValueError("Subcore range exceeds task descriptors")
            if (start + number) * C.sizeof(Task) > task_buffer.size:
                raise ValueError("Active task range exceeds its kernel mapped buffer")
        # RK3588 uses subcore_task[0] for core 0. RKNN's declared count may
        # include three identical core ranges; those unused descriptors do not
        # exist in its task BO (rknpu_job.c:rknpu_get_task_number).
        submit = Submit(flags=settings["flags"], timeout=settings["timeout"], task_start=first,
                        task_number=count, task_obj_addr=task_buffer.obj,
                        core_mask=1, fence_fd=-1)
        for index, (start, number) in enumerate(ranges):
            submit.subcore_task[index] = Range(start, number)
        self.busy = True
        try:
            ioctl(self.fd, command(0x41, Submit), submit)
        finally:
            # ioctl is blocking; never terminate a process with a running NPU job.
            self.busy = False
        return submit.hw_elapse_time

    def close(self):
        if self.busy:
            raise RuntimeError("Cannot free buffers while NPU is running")
        for buffer in reversed(self.buffers):
            buffer.close()
        self.buffers.clear()
        os.close(self.fd)
        flock(self.lock, LOCK_UN)
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
