import glob
import ctypes
import mmap
import os
import time
from fcntl import ioctl


DRM_COMMAND_BASE = 0x40

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_CACHEABLE = 2
RKNPU_MEM_IOMMU = 0x10
RKNPU_MEM_NON_CONTIGUOUS = 1
RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT = 0x400
RKNPU_ACT_RESET = 1
RKNPU_MEM_SYNC_TO_DEVICE = 1
RKNPU_MEM_SYNC_FROM_DEVICE = 2


class drm_rocket_create_bo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("handle", ctypes.c_uint32),
        ("dma_address", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
    ]


class drm_rocket_prep_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("timeout_ns", ctypes.c_int64),
    ]


class drm_rocket_fini_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class drm_rocket_task(ctypes.Structure):
    _fields_ = [
        ("regcmd", ctypes.c_uint32),
        ("regcmd_count", ctypes.c_uint32),
    ]


class drm_rocket_job(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_uint64),
        ("in_bo_handles", ctypes.c_uint64),
        ("out_bo_handles", ctypes.c_uint64),
        ("task_count", ctypes.c_uint32),
        ("task_struct_size", ctypes.c_uint32),
        ("in_bo_handle_count", ctypes.c_uint32),
        ("out_bo_handle_count", ctypes.c_uint32),
    ]


class drm_rocket_submit(ctypes.Structure):
    _fields_ = [
        ("jobs", ctypes.c_uint64),
        ("job_count", ctypes.c_uint32),
        ("job_struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]


def _IOW(type_, nr, size):
    return (1 << 30) | (ord(type_) << 8) | nr | (size << 16)


def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)


DRM_IOCTL_ROCKET_CREATE_BO = _IOWR(
    "d", DRM_COMMAND_BASE + 0x00, ctypes.sizeof(drm_rocket_create_bo)
)
DRM_IOCTL_ROCKET_SUBMIT = _IOW(
    "d", DRM_COMMAND_BASE + 0x01, ctypes.sizeof(drm_rocket_submit)
)
DRM_IOCTL_ROCKET_PREP_BO = _IOW(
    "d", DRM_COMMAND_BASE + 0x02, ctypes.sizeof(drm_rocket_prep_bo)
)
DRM_IOCTL_ROCKET_FINI_BO = _IOW(
    "d", DRM_COMMAND_BASE + 0x03, ctypes.sizeof(drm_rocket_fini_bo)
)


class RocketBO:
    __slots__ = ("handle", "size", "dma_address", "offset")

    def __init__(self, handle, size, dma_address, offset):
        self.handle = int(handle)
        self.size = int(size)
        self.dma_address = int(dma_address)
        self.offset = int(offset)

    @property
    def dma_addr(self):
        return self.dma_address

    @property
    def obj_addr(self):
        return self.handle


def open_rocket_device():
    path = os.environ.get("ROCKET_DEVICE")
    if path:
        return os.open(path, os.O_RDWR)

    candidates = (
        sorted(glob.glob("/dev/accel/accel*"))
        + sorted(glob.glob("/dev/dri/renderD*"))
        + sorted(glob.glob("/dev/dri/card*"))
    )
    last_error = None
    for candidate in candidates:
        try:
            return os.open(candidate, os.O_RDWR)
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError("No Rocket device found")


def mem_allocate(fd, size, flags=0):
    bo = drm_rocket_create_bo(size=size)
    ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, bo)
    buf = mmap.mmap(fd, bo.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return buf, RocketBO(bo.handle, bo.size, bo.dma_address, bo.offset)


def prep_bo(fd, bo, timeout_ns=6_000_000_000):
    if timeout_ns > 0:
        timeout_ns = time.monotonic_ns() + timeout_ns
    ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, drm_rocket_prep_bo(handle=bo.handle, timeout_ns=timeout_ns))


def fini_bo(fd, bo):
    ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))


def mem_sync(fd, obj_addr, offset, size, flags):
    if flags == RKNPU_MEM_SYNC_TO_DEVICE:
        fini_bo(fd, RocketBO(obj_addr, 0, 0, 0))
    elif flags == RKNPU_MEM_SYNC_FROM_DEVICE:
        prep_bo(fd, RocketBO(obj_addr, 0, 0, 0))


def reset_npu(fd):
    return 0


def mem_destroy(fd, mem_create):
    return 0


def build_rocket_tasks(vendor_tasks, task_count):
    tasks = (drm_rocket_task * task_count)()
    for i in range(task_count):
        tasks[i].regcmd = int(vendor_tasks[i].regcmd_addr) & 0xFFFFFFFF
        tasks[i].regcmd_count = int(vendor_tasks[i].regcfg_amount)
    return tasks


def submit(fd, vendor_tasks, task_count=1, in_bos=(), out_bos=()):
    rocket_tasks = build_rocket_tasks(vendor_tasks, task_count)
    in_handles = (ctypes.c_uint32 * len(in_bos))(*(bo.handle for bo in in_bos)) if in_bos else None
    out_handles = (ctypes.c_uint32 * len(out_bos))(*(bo.handle for bo in out_bos)) if out_bos else None
    job = drm_rocket_job(
        tasks=ctypes.addressof(rocket_tasks),
        in_bo_handles=ctypes.addressof(in_handles) if in_handles is not None else 0,
        out_bo_handles=ctypes.addressof(out_handles) if out_handles is not None else 0,
        task_count=task_count,
        task_struct_size=ctypes.sizeof(drm_rocket_task),
        in_bo_handle_count=len(in_bos),
        out_bo_handle_count=len(out_bos),
    )
    jobs = (drm_rocket_job * 1)(job)
    submit_struct = drm_rocket_submit(
        jobs=ctypes.addressof(jobs),
        job_count=1,
        job_struct_size=ctypes.sizeof(drm_rocket_job),
    )
    return ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, submit_struct)
