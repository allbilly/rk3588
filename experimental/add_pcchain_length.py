from fcntl import flock, ioctl, LOCK_EX, LOCK_NB, LOCK_UN
import argparse
import ctypes
import mmap
import os


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 6
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0 << 1
RKNPU_JOB_NONBLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 0x4

BODY_QWORDS = 16
SEGMENT_QWORDS = BODY_QWORDS + 4
REG_BLOCK_QWORDS = 32

TARGET_DPU = 0x1001
TARGET_RDMA = 0x2001
TARGET_PC = 0x0081
TARGET_PC_REG = 0x0101
TARGET_VERSION = 0x0041
LOCK_PATH = "/tmp/rk3588_npu_submit.lock"
DEFAULT_MAX_SAFE_SEGMENTS = 8


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


DRM_IOCTL_RKNPU_MEM_CREATE = _iowr("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _iowr("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _iowr("d", 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _iowr("d", 0x40, ctypes.sizeof(rknpu_action))


def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    return buf, mem_create


def emit(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def add_segment(input_dma, weight_dma, output_dma, next_dma, elements):
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
        emit(TARGET_RDMA, 0x5010, 0),
        emit(TARGET_RDMA, 0x5014, 0x00000007),
        emit(TARGET_RDMA, 0x5034, 0x40000008),
        emit(TARGET_DPU, 0x4020, output_dma),
        emit(TARGET_RDMA, 0x5018, input_dma),
        emit(TARGET_RDMA, 0x5038, weight_dma),
        emit(TARGET_RDMA, 0x5044, 0x00017849),
    ]
    tail = [
        emit(TARGET_PC_REG, 0x0010, next_dma & 0xFFFFFFF0) if next_dma else 0,
        emit(TARGET_PC_REG, 0x0014, BODY_QWORDS if next_dma else 0),
        emit(TARGET_VERSION, 0, 0),
        emit(TARGET_PC, 0x0008, 0x18),
    ]
    return body + tail


def submit(fd, task_obj_addr, task_count, timeout):
    submit_struct = rknpu_submit(
        flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
        timeout=timeout,
        task_start=0,
        task_number=task_count,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=1,
        fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(0, task_count)
    for i in range(1, 5):
        submit_struct.subcore_task[i] = rknpu_subcore_task(0, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def reset_npu(fd):
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))


def run_once(fd, task_count, elements, timeout):
    reset_npu(fd)
    byte_stride = max(4096, elements * ctypes.sizeof(ctypes.c_uint16))
    elem_stride = byte_stride // ctypes.sizeof(ctypes.c_uint16)
    task_map, task_mc = mem_allocate(fd, task_count * ctypes.sizeof(rknpu_task), RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mc = mem_allocate(fd, task_count * REG_BLOCK_QWORDS * ctypes.sizeof(ctypes.c_uint64), RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mc = mem_allocate(fd, task_count * byte_stride, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mc = mem_allocate(fd, task_count * byte_stride, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mc = mem_allocate(fd, task_count * byte_stride, RKNPU_MEM_NON_CACHEABLE)

    regs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
    inputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), ctypes.POINTER(ctypes.c_uint16))
    weights = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), ctypes.POINTER(ctypes.c_uint16))
    outputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), ctypes.POINTER(ctypes.c_uint16))

    for task_idx in range(task_count):
        aval = 3 + task_idx
        bval = 5 + 2 * task_idx
        base_elem = task_idx * elem_stride
        for i in range(elements):
            inputs[base_elem + i] = aval
            weights[base_elem + i] = bval
            outputs[base_elem + i] = 0

    for task_idx in range(task_count):
        base = task_idx * REG_BLOCK_QWORDS
        next_dma = regcmd_mc.dma_addr + (task_idx + 1) * REG_BLOCK_QWORDS * 8 if task_idx + 1 < task_count else 0
        segment = add_segment(
            input_mc.dma_addr + task_idx * byte_stride,
            weight_mc.dma_addr + task_idx * byte_stride,
            output_mc.dma_addr + task_idx * byte_stride,
            next_dma,
            elements,
        )
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

    try:
        ret = submit(fd, task_mc.obj_addr, task_count, timeout)
        if ret != 0:
            return False, f"submit_ret={ret}"

        for task_idx in range(task_count):
            expected = 8 + 3 * task_idx
            base_elem = task_idx * elem_stride
            for i in range(elements):
                got = int(outputs[base_elem + i])
                if got != expected:
                    return False, f"task={task_idx} index={i} got={got} expected={expected}"
        return True, "pass"
    finally:
        task_map.close()
        regcmd_map.close()
        input_map.close()
        weight_map.close()
        output_map.close()


def parse_lengths(args):
    if args.lengths:
        return [int(item, 0) for item in args.lengths.split(",")]
    lengths = []
    value = args.start
    while value <= args.max_segments:
        lengths.append(value)
        value *= 2
    if lengths[-1] != args.max_segments:
        lengths.append(args.max_segments)
    return lengths


def main():
    parser = argparse.ArgumentParser(description="Measure safe single-core ADD PC-chain length.")
    parser.add_argument("--device", default="/dev/dri/card1")
    parser.add_argument("--segment-elements", type=int, default=4096)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--max-segments", type=int, default=DEFAULT_MAX_SAFE_SEGMENTS)
    parser.add_argument("--lengths", help="Comma-separated segment counts. Overrides --start/--max-segments.")
    parser.add_argument("--timeout", type=int, default=10000)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument(
        "--allow-risky-length",
        action="store_true",
        help="Allow lengths above the currently verified safe boundary. May hang or wedge the downstream driver.",
    )
    args = parser.parse_args()
    if args.segment_elements <= 0 or args.segment_elements % 8:
        raise ValueError("--segment-elements must be a positive multiple of 8")
    lengths = parse_lengths(args)
    risky_lengths = [length for length in lengths if length > DEFAULT_MAX_SAFE_SEGMENTS]
    if risky_lengths and not args.allow_risky_length:
        raise RuntimeError(
            f"Refusing risky chain lengths {risky_lengths}. Last stable boundary is "
            f"{DEFAULT_MAX_SAFE_SEGMENTS} segments at 4096 FP16 values/segment. "
            "Pass --allow-risky-length only with physical reset access and no other NPU process running."
        )

    lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    fd = os.open(args.device, os.O_RDWR)
    try:
        try:
            flock(lock_fd, LOCK_EX | LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another NPU experiment holds {LOCK_PATH}; refusing parallel submit") from exc
        all_ok = True
        for task_count in lengths:
            ok, detail = run_once(fd, task_count, args.segment_elements, args.timeout)
            print(f"segments={task_count} elements={args.segment_elements} {'PASS' if ok else 'FAIL'} {detail}")
            all_ok &= ok
            if not ok and args.stop_on_fail:
                break
        return 0 if all_ok else 1
    finally:
        flock(lock_fd, LOCK_UN)
        os.close(fd)
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
