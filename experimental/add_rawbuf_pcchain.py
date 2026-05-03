from fcntl import ioctl
import argparse
import ctypes
import mmap
import os


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0x2
RKNPU_JOB_PINGPONG = 0x4

SEGMENTS = 3
VALUES = [(3, 5), (7, 11), (13, 17)]
REG_BLOCK_QWORDS = 32

TARGET_DPU = 0x1001
TARGET_RDMA = 0x2001
TARGET_PC = 0x0081
TARGET_PC_REG = 0x0101
TARGET_VERSION = 0x0041
REG_PC_BASE_ADDRESS = 0x0010
REG_PC_REGISTER_AMOUNTS = 0x0014
REG_PC_OPERATION_ENABLE = 0x0008


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
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    return buf, mem_create


def regcmd(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def add_body(input_dma, weight_dma, output_dma):
    return [
        0x10010000000e4004,
        0x1001000001e5400c,
        0x1001480000024010,
        0x100100070007403c,
        0x1001000000004030,
        0x1001108202c04070,
        0x1001000100014084,
        0x20010000000e5004,
        0x200100000000500c,
        0x2001000000005010,
        0x2001000700075014,
        0x2001400000085034,
        regcmd(TARGET_DPU, 0x4020, output_dma),
        regcmd(TARGET_RDMA, 0x5018, input_dma),
        regcmd(TARGET_RDMA, 0x5038, weight_dma),
        0x2001000178495044,
    ]


def amount_for(style, next_len):
    if style == "raw":
        return next_len + 4
    if style == "body":
        return next_len
    if style == "gemm":
        return (next_len + 1) // 2 + 1
    raise ValueError(style)


def pc_tail(task_idx, regcmd_dma, body_len, amount_style, pc_op_enable, fixed_amount=None, fixed_amounts=None):
    if task_idx + 1 < SEGMENTS:
        next_addr = regcmd_dma + (task_idx + 1) * REG_BLOCK_QWORDS * ctypes.sizeof(ctypes.c_uint64)
        if fixed_amounts is not None:
            amount = fixed_amounts[task_idx]
        else:
            amount = fixed_amount if fixed_amount is not None else amount_for(amount_style, body_len)
        return [
            regcmd(TARGET_PC_REG, REG_PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
            regcmd(TARGET_PC_REG, REG_PC_REGISTER_AMOUNTS, amount),
            regcmd(TARGET_VERSION, 0, 0),
            regcmd(TARGET_PC, REG_PC_OPERATION_ENABLE, pc_op_enable),
        ]
    return [
        0,
        regcmd(TARGET_PC_REG, REG_PC_REGISTER_AMOUNTS, 0),
        regcmd(TARGET_VERSION, 0, 0),
        regcmd(TARGET_PC, REG_PC_OPERATION_ENABLE, pc_op_enable),
    ]


def submit(fd, task_obj_addr, flags, timeout, task_number, mode):
    official = mode == "official"
    actual_task_number = task_number * 3 if official else task_number
    submit_struct = rknpu_submit(
        flags=flags,
        timeout=timeout,
        task_start=0,
        task_number=actual_task_number,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=0 if official else 1,
        fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_number)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=0, task_number=task_number if official else 0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=0, task_number=task_number if official else 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)


def main():
    parser = argparse.ArgumentParser(description="Single-core ADD rawbuf PC-chain probe for one logical output split into segments.")
    parser.add_argument("--amount-style", choices=("raw", "body", "gemm"), default="raw")
    parser.add_argument("--fixed-amount", type=int, default=None)
    parser.add_argument("--fixed-amounts", default=None, help="Comma-separated PC amounts for task0->1 and task1->2.")
    parser.add_argument("--regcmd-mode", choices=("offset", "absolute"), default="offset")
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG)
    parser.add_argument("--mode", choices=("core0", "official"), default="core0")
    parser.add_argument(
        "--allow-unsafe-submit",
        action="store_true",
        help="Allow official-style three-core submit. This is not used for PC-chain bring-up because it can race/lock this kernel.",
    )
    parser.add_argument("--pc-op-enable", type=lambda x: int(x, 0), default=0x18)
    parser.add_argument("--task-number", type=int, default=TASKS)
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--descriptor-amount", choices=("body", "segment"), default="body")
    args = parser.parse_args()
    if args.mode != "core0" and not args.allow_unsafe_submit:
        raise RuntimeError("Refusing non-core0 PC-chain submit. Use --allow-unsafe-submit only with physical reset access.")
    fixed_amounts = None
    if args.fixed_amounts is not None:
        fixed_amounts = [int(x, 0) for x in args.fixed_amounts.split(",")]
        if len(fixed_amounts) != SEGMENTS - 1:
            raise ValueError(f"--fixed-amounts must contain {SEGMENTS - 1} values")

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        task_map, task_mc = mem_allocate(fd, 1024, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        regcmd_map, regcmd_mc = mem_allocate(fd, SEGMENTS * REG_BLOCK_QWORDS * ctypes.sizeof(ctypes.c_uint64), RKNPU_MEM_NON_CACHEABLE)
        input_map, input_mc = mem_allocate(fd, SEGMENTS * 4096, RKNPU_MEM_NON_CACHEABLE)
        weight_map, weight_mc = mem_allocate(fd, SEGMENTS * 4096, RKNPU_MEM_NON_CACHEABLE)
        output_map, output_mc = mem_allocate(fd, SEGMENTS * 4096, RKNPU_MEM_NON_CACHEABLE)

        regs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
        inputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), ctypes.POINTER(ctypes.c_uint16))
        weights = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), ctypes.POINTER(ctypes.c_uint16))
        outputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), ctypes.POINTER(ctypes.c_uint16))

        body_len = None
        for task_idx, (aval, bval) in enumerate(VALUES):
            for i in range(8):
                inputs[task_idx * 2048 + i] = aval
                weights[task_idx * 2048 + i] = bval
                outputs[task_idx * 2048 + i] = 0

            body = add_body(
                input_mc.dma_addr + task_idx * 4096,
                weight_mc.dma_addr + task_idx * 4096,
                output_mc.dma_addr + task_idx * 4096,
            )
            body_len = len(body)
            segment = body + pc_tail(
                task_idx,
                regcmd_mc.dma_addr,
                body_len,
                args.amount_style,
                args.pc_op_enable,
                args.fixed_amount,
                fixed_amounts,
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
            tasks[task_idx].regcfg_amount = body_len if args.descriptor_amount == "body" else len(segment)
            if args.regcmd_mode == "offset":
                tasks[task_idx].regcfg_offset = base * ctypes.sizeof(ctypes.c_uint64)
                tasks[task_idx].regcmd_addr = regcmd_mc.dma_addr
            else:
                tasks[task_idx].regcfg_offset = 0
                tasks[task_idx].regcmd_addr = regcmd_mc.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)

        ret = submit(fd, task_mc.obj_addr, args.flags, args.timeout, args.task_number, args.mode)
        got = [[outputs[task_idx * 2048 + i] for i in range(8)] for task_idx in range(SEGMENTS)]
        expected = [[a + b] * 8 for a, b in VALUES]
        ok = ret == 0 and got == expected
        print(
            f"submit ret={ret} amount_style={args.amount_style} fixed_amount={args.fixed_amount} fixed_amounts={args.fixed_amounts} "
            f"regcmd_mode={args.regcmd_mode} descriptor_amount={args.descriptor_amount} flags=0x{args.flags:x} mode={args.mode}"
        )
        print("got", got)
        print("expected", expected)
        print("ADD RAWBUF PCCHAIN PASS" if ok else "ADD RAWBUF PCCHAIN FAIL")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
