from fcntl import flock, ioctl, LOCK_EX, LOCK_NB, LOCK_UN
import ctypes
import mmap
import os


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 6
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0x2
RKNPU_JOB_PINGPONG = 0x4
LOCK_PATH = "/tmp/rk3588_npu_submit.lock"


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


class rknpu_action(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("value", ctypes.c_uint32),
    ]


class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
    ]


class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("timeout", ctypes.c_uint32),
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
        ("task_counter", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64),
        ("iommu_domain_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("task_base_addr", ctypes.c_uint64),
        ("hw_elapse_time", ctypes.c_int64),
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
DRM_IOCTL_RKNPU_ACTION = _IOWR("d", 0x40, ctypes.sizeof(rknpu_action))


def open_device(path="/dev/dri/card1"):
    return os.open(path, os.O_RDWR)


def mem_allocate(fd, size, flags=0, verbose=False):
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    if ret != 0:
        raise OSError(ret, "RKNPU_MEM_CREATE failed")

    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    if ret != 0:
        raise OSError(ret, "RKNPU_MEM_MAP failed")

    buf = mmap.mmap(
        fd,
        mem_create.size,
        mmap.MAP_SHARED,
        mmap.PROT_READ | mmap.PROT_WRITE,
        offset=mem_map.offset,
    )
    if verbose:
        print(
            f"alloc size={size} flags=0x{flags:x} handle={mem_create.handle} "
            f"obj=0x{mem_create.obj_addr:x} dma=0x{mem_create.dma_addr:x}"
        )
    return buf, mem_create


def reset_npu(fd):
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))


def fill_submit(
    submit,
    task_obj_addr,
    total_tasks,
    core_ranges,
    core_mask=None,
    flags=RKNPU_JOB_PC | RKNPU_JOB_PINGPONG,
    timeout=6000,
    subcore_layout="direct",
):
    submit.flags = flags
    submit.timeout = timeout
    submit.task_start = 0
    submit.task_number = total_tasks
    submit.task_counter = 0
    submit.priority = 0
    submit.task_obj_addr = task_obj_addr
    submit.iommu_domain_id = 0
    submit.reserved = 0
    submit.task_base_addr = 0
    submit.hw_elapse_time = 0
    submit.core_mask = core_mask if core_mask is not None else sum(1 << core for core in core_ranges)
    submit.fence_fd = -1
    for i in range(5):
        submit.subcore_task[i] = rknpu_subcore_task(task_start=0, task_number=0)
    if subcore_layout == "direct":
        for core, (start, count) in core_ranges.items():
            if core < 0 or core >= 5:
                raise ValueError(f"core index {core} cannot fit subcore_task[5]")
            submit.subcore_task[core] = rknpu_subcore_task(task_start=start, task_number=count)
    elif subcore_layout == "rk3588-tricore-tail":
        for core, (start, count) in core_ranges.items():
            if core < 0 or core > 2:
                raise ValueError(f"RK3588 core index must be 0..2, got {core}")
            submit.subcore_task[core + 2] = rknpu_subcore_task(task_start=start, task_number=count)
    else:
        raise ValueError(f"unknown subcore layout: {subcore_layout}")
    return submit


def submit(
    fd,
    task_obj_addr,
    total_tasks,
    core_ranges,
    core_mask=None,
    flags=RKNPU_JOB_PC | RKNPU_JOB_PINGPONG,
    timeout=6000,
    subcore_layout="direct",
):
    submit_struct = fill_submit(
        rknpu_submit(),
        task_obj_addr=task_obj_addr,
        total_tasks=total_tasks,
        core_ranges=core_ranges,
        core_mask=core_mask,
        flags=flags,
        timeout=timeout,
        subcore_layout=subcore_layout,
    )
    lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        try:
            flock(lock_fd, LOCK_EX | LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another NPU experiment holds {LOCK_PATH}; refusing parallel submit") from exc
        return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct), submit_struct
    finally:
        flock(lock_fd, LOCK_UN)
        os.close(lock_fd)


def print_submit_layout(submit_struct):
    print(f"flags=0x{submit_struct.flags:x}")
    print(f"core_mask=0x{submit_struct.core_mask:x}")
    print(f"task_number={submit_struct.task_number}")
    for i in range(5):
        st = submit_struct.subcore_task[i]
        print(f"subcore_task[{i}]=({st.task_start},{st.task_number})")
