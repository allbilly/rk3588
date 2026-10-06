"""Linux 6.18 Rocket UAPI and decoded-model adapter, using only the stdlib.

See include/uapi/drm/rocket_accel.h and drivers/accel/rocket/rocket_{job,gem}.c.
Rocket schedules tasks individually on one core per job. Vendor inline PC
chains must therefore be terminated while retaining their unit-enable word.
"""
import ctypes as C
import errno
from fcntl import ioctl
import mmap
import struct
import time


class CreateBO(C.Structure):
    _fields_ = [("size", C.c_uint32), ("handle", C.c_uint32),
                ("dma_address", C.c_uint64), ("offset", C.c_uint64)]


class PrepBO(C.Structure):
    _fields_ = [("handle", C.c_uint32), ("reserved", C.c_uint32),
                ("timeout_ns", C.c_int64)]


class FiniBO(C.Structure):
    _fields_ = [("handle", C.c_uint32), ("reserved", C.c_uint32)]


class CloseBO(C.Structure):
    _fields_ = [("handle", C.c_uint32), ("pad", C.c_uint32)]


class Task(C.Structure):
    _fields_ = [("regcmd", C.c_uint32), ("regcmd_count", C.c_uint32)]


class Job(C.Structure):
    _fields_ = [("tasks", C.c_uint64), ("in_bo_handles", C.c_uint64),
                ("out_bo_handles", C.c_uint64), ("task_count", C.c_uint32),
                ("task_struct_size", C.c_uint32), ("in_bo_handle_count", C.c_uint32),
                ("out_bo_handle_count", C.c_uint32)]


class Submit(C.Structure):
    _fields_ = [("jobs", C.c_uint64), ("job_count", C.c_uint32),
                ("job_struct_size", C.c_uint32), ("reserved", C.c_uint64)]


def command(number, structure, direction=1):
    return (direction << 30) | (C.sizeof(structure) << 16) | (ord("d") << 8) | number


CREATE_BO = command(0x40, CreateBO, 3)
SUBMIT = command(0x41, Submit)
PREP_BO = command(0x42, PrepBO)
FINI_BO = command(0x43, FiniBO)
GEM_CLOSE = command(0x09, CloseBO)


def prepare_commands(encoded, task):
    if len(encoded) != (task["regcfg_amount"] + 4) * 8:
        raise ValueError("Rocket task requires the complete four-word PC trailer")
    base, amount, version, enable = struct.unpack("<4Q", encoded[-32:])
    if (base != 0 and base & 0xffff00000000ffff != 0x0101000000000010) or \
            amount & 0xffff00000000ffff != 0x0101000000000014 or \
            version != 0x0041000000000000 or \
            enable != (0x0081 << 48 | task["enable_mask"] << 16 | 0x0008):
        raise ValueError("Unrecognized PC trailer; refusing an unsafe Rocket submission")
    # The kernel advances to the next task after its IRQ. Clearing both chain
    # fields prevents the PC engine from also advancing to the captured task.
    return encoded[:-32] + struct.pack("<4Q", 0x0101000000000010,
                                      0x0101000000000014, version, enable)


class Buffer:
    def __init__(self, device, size, flags=0):
        if size > 0xffffffff:
            raise ValueError("Rocket BO size exceeds its 32-bit UAPI field")
        self.device = device
        # CREATE_BO does not return the page-rounded size on Linux 6.18.
        self.size = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        if self.size > 0xffffffff:
            raise ValueError("Page-rounded Rocket BO size exceeds its 32-bit UAPI field")
        self.info = CreateBO(size=self.size)
        ioctl(device.fd, CREATE_BO, self.info)
        self.mapping = None
        try:
            if self.info.dma_address + self.size > 1 << 32:
                raise ValueError("Buffer cannot be addressed by the 32-bit NPU registers")
            self.mapping = mmap.mmap(device.fd, self.size, mmap.MAP_SHARED,
                                     mmap.PROT_READ | mmap.PROT_WRITE, offset=self.info.offset)
            self.dma = self.info.dma_address
            self.cpu_owned = False
            self.sync(2)
        except BaseException:
            # This new BO has never been submitted and is not registered yet.
            try:
                if self.mapping is not None:
                    self.mapping.close()
            finally:
                ioctl(device.fd, GEM_CLOSE, CloseBO(handle=self.info.handle))
            raise

    def write(self, data, offset=0):
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError("Write exceeds allocated NPU buffer")
        self.sync(2)
        self.mapping[offset:offset + len(data)] = data

    def sync(self, direction, timeout_ms=6000):
        if direction == 1:
            if self.cpu_owned:
                ioctl(self.device.fd, FINI_BO, FiniBO(handle=self.info.handle))
                self.cpu_owned = False
        elif direction == 2:
            if not self.cpu_owned:
                while True:
                    try:
                        ioctl(self.device.fd, PREP_BO, PrepBO(handle=self.info.handle,
                              timeout_ns=time.monotonic_ns() + timeout_ms * 1_000_000))
                        break
                    except OSError as error:
                        if error.errno not in (errno.ETIMEDOUT, errno.EBUSY, errno.EINTR):
                            raise
                        # A userspace observation timeout does not prove the
                        # NPU stopped. Keep waiting; never free a live job's BOs.
                self.cpu_owned = True
        else:
            raise ValueError("Invalid NPU buffer synchronization direction")

    def close(self):
        self.sync(2)
        self.mapping.close()
        ioctl(self.device.fd, GEM_CLOSE, CloseBO(handle=self.info.handle))


def submit(device, task_buffer, first, count, settings, buffers):
    if task_buffer not in buffers or any(buffer.device is not device for buffer in buffers):
        raise ValueError("Rocket submission requires this model's buffers on the same device")
    if first < 0 or count <= 0 or (first + count) * 40 > task_buffer.size:
        raise ValueError("Rocket task range exceeds the model's descriptor buffer")
    task_buffer.sync(2)
    tasks = (Task * count)()
    for index in range(count):
        # Keep the portable cache's vendor descriptor layout, but pass only
        # regcmd/count to Rocket. Unused repeated vendor core ranges are omitted.
        start = (first + index) * 40
        descriptor = struct.unpack("<8IQ", task_buffer.mapping[start:start + 40])
        # Linux 6.18 arms/tests only DPU_0/DPU_1 completion. A PPU-only task
        # would time out and reset the core; mixed units can finish too early.
        if not descriptor[2] & 0x08 or descriptor[2] & 0x60:
            raise ValueError("Stock Rocket requires DPU completion; standalone or mixed PPU tasks are unsupported")
        amount, offset, address = descriptor[6:9]
        if offset or address & 15 or not 0 < amount + 4 <= 0x10000 * 2:
            raise ValueError("Invalid Rocket register command range")
        if not any(buffer.dma <= address and address + (amount + 4) * 8 <= buffer.dma + buffer.size
                   for buffer in buffers):
            raise ValueError("Rocket command range exceeds the model's BOs")
        tasks[index] = Task(address, amount + 4)
    # Some coefficient BOs also contain commands and intermediate tensors.
    # Conservatively fence every model BO as read/write, once, rather than
    # placing duplicate handles in the kernel's reservation-lock arrays.
    handles = (C.c_uint32 * len(buffers))(*(buffer.info.handle for buffer in buffers))
    if len(set(handles)) != len(handles):
        raise ValueError("Duplicate Rocket buffer handles")
    job = Job(tasks=C.addressof(tasks), task_count=count, task_struct_size=C.sizeof(Task),
              out_bo_handles=C.addressof(handles), out_bo_handle_count=len(handles))
    jobs = (Job * 1)(job)
    request = Submit(jobs=C.addressof(jobs), job_count=1, job_struct_size=C.sizeof(Job))
    for buffer in buffers:
        buffer.sync(1)
    start = time.monotonic_ns()
    device.busy = True
    try:
        ioctl(device.fd, SUBMIT, request)
    except OSError:
        # Linux 6.18's single-job IOW ioctl returns errors before queueing.
        # A Python interruption can leave submission uncertain; keep its guard.
        device.busy = False
        raise
    # SUBMIT is asynchronous. Every BO has the same write fence, so waiting
    # on one completes the job before tensor copies, another job, or cleanup.
    # If a non-timeout wait fails, leave busy set to prevent unsafe cleanup.
    buffers[0].sync(2, timeout_ms=max(1, settings["timeout"]))
    device.busy = False
    return (time.monotonic_ns() - start) // 1000
