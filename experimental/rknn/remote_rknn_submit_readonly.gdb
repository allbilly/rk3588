set pagination off
set breakpoint pending on
set confirm off

python
import gdb

IOC_NRBITS = 8
IOC_TYPEBITS = 8
IOC_SIZEBITS = 14
IOC_NRSHIFT = 0
IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
DRM_COMMAND_BASE = 0x40
DRM_IOCTL_RKNPU_SUBMIT_NR = DRM_COMMAND_BASE + 1
SUBMIT_STRUCT_SIZE = 104
TASK_STRUCT_SIZE = 40


def read_mem(addr, size):
    return gdb.selected_inferior().read_memory(addr, size).tobytes()


def read_u32(data, offset):
    return int.from_bytes(data[offset:offset + 4], "little")


def read_i32(data, offset):
    return int.from_bytes(data[offset:offset + 4], "little", signed=True)


def read_u64(data, offset):
    return int.from_bytes(data[offset:offset + 8], "little")


def decode_submit(addr):
    data = read_mem(addr, SUBMIT_STRUCT_SIZE)
    subcores = []
    for idx in range(5):
        base = 64 + idx * 8
        subcores.append((read_u32(data, base), read_u32(data, base + 4)))
    return {
        "flags": read_u32(data, 0),
        "timeout": read_u32(data, 4),
        "task_start": read_u32(data, 8),
        "task_number": read_u32(data, 12),
        "task_counter": read_u32(data, 16),
        "priority": read_i32(data, 20),
        "task_obj_addr": read_u64(data, 24),
        "regcfg_obj_addr": read_u64(data, 32),
        "task_base_addr": read_u64(data, 40),
        "user_data": read_u64(data, 48),
        "core_mask": read_u32(data, 56),
        "fence_fd": read_i32(data, 60),
        "subcores": subcores,
    }


def decode_tasks(task_obj_addr, count):
    if task_obj_addr == 0 or count <= 0:
        return []
    try:
        data = read_mem(task_obj_addr, count * TASK_STRUCT_SIZE)
    except gdb.MemoryError:
        gdb.write("  task object is not directly readable from userspace mapping\n")
        return []
    tasks = []
    for idx in range(count):
        base = idx * TASK_STRUCT_SIZE
        tasks.append({
            "flags": read_u32(data, base + 0),
            "op_idx": read_u32(data, base + 4),
            "enable_mask": read_u32(data, base + 8),
            "int_mask": read_u32(data, base + 12),
            "int_clear": read_u32(data, base + 16),
            "int_status": read_u32(data, base + 20),
            "regcfg_amount": read_u32(data, base + 24),
            "regcfg_offset": read_u32(data, base + 28),
            "regcmd_addr": read_u64(data, base + 32),
        })
    return tasks


class IoctlSubmitHook(gdb.Breakpoint):
    def __init__(self):
        super().__init__("ioctl", internal=False)
        self.calls = 0
        self.submits = 0

    def stop(self):
        self.calls += 1
        cmd = int(gdb.parse_and_eval("$x1"))
        nr = (cmd >> IOC_NRSHIFT) & ((1 << IOC_NRBITS) - 1)
        size = (cmd >> IOC_SIZESHIFT) & ((1 << IOC_SIZEBITS) - 1)
        if nr != DRM_IOCTL_RKNPU_SUBMIT_NR:
            return False
        self.submits += 1
        arg = int(gdb.parse_and_eval("$x2"))
        submit = decode_submit(arg)
        gdb.write(f"SUBMIT[{self.submits}] cmd=0x{cmd:08x} size={size} arg=0x{arg:x}\n")
        gdb.write(
            f"  flags=0x{submit['flags']:x} timeout={submit['timeout']} "
            f"task_start={submit['task_start']} task_number={submit['task_number']} "
            f"core_mask=0x{submit['core_mask']:x} fence_fd={submit['fence_fd']}\n")
        gdb.write(
            f"  task_obj_addr=0x{submit['task_obj_addr']:x} "
            f"regcfg_obj_addr=0x{submit['regcfg_obj_addr']:x} "
            f"task_base_addr=0x{submit['task_base_addr']:x}\n")
        for idx, (start, number) in enumerate(submit["subcores"]):
            gdb.write(f"  subcore_task[{idx}]={{task_start={start}, task_number={number}}}\n")
        tasks = decode_tasks(submit["task_obj_addr"], submit["task_number"])
        for idx, task in enumerate(tasks):
            qstart = -1 if submit["task_base_addr"] == 0 else (
                task["regcmd_addr"] - submit["task_base_addr"]) // 8
            gdb.write(
                f"  task[{idx}] flags=0x{task['flags']:x} op_idx={task['op_idx']} "
                f"enable=0x{task['enable_mask']:x} int=0x{task['int_mask']:x} "
                f"clear=0x{task['int_clear']:x} amount={task['regcfg_amount']} "
                f"regcmd_addr=0x{task['regcmd_addr']:x} qstart={qstart}\n")
        return False


IoctlSubmitHook()
gdb.write("remote_rknn_submit_readonly.gdb loaded\n")
end

run
