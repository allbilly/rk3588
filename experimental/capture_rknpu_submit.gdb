set pagination off
set breakpoint pending on

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
SUBCORE_ENTRIES = 5


def fmt(val):
    return f"0x{val:x}"


def decode_submit(addr):
    inf = gdb.selected_inferior()
    data = inf.read_memory(addr, SUBMIT_STRUCT_SIZE).tobytes()
    off = 0

    def read(size, signed=False):
        nonlocal off
        val = int.from_bytes(data[off:off + size], "little", signed=signed)
        off += size
        return val

    out = {}
    out["flags"] = read(4)
    out["timeout"] = read(4)
    out["task_start"] = read(4)
    out["task_number"] = read(4)
    out["task_counter"] = read(4)
    out["priority"] = read(4, signed=True)
    out["task_obj_addr"] = read(8)
    out["field32"] = read(4)
    out["field36"] = read(4)
    out["task_base_addr"] = read(8)
    out["field48"] = read(8)
    out["core_mask"] = read(4)
    out["fence_fd"] = read(4, signed=True)
    out["subcores"] = []
    for _ in range(SUBCORE_ENTRIES):
        out["subcores"].append((read(4), read(4)))
    return out


def print_tasks(task_obj_addr, count):
    if not task_obj_addr or not count:
        return
    inf = gdb.selected_inferior()
    for idx in range(count):
        base = task_obj_addr + idx * TASK_STRUCT_SIZE
        try:
            data = inf.read_memory(base, TASK_STRUCT_SIZE).tobytes()
        except gdb.MemoryError:
            print(f"  task[{idx}] unreadable at {fmt(base)}")
            break
        off = 0

        def read(size, signed=False):
            nonlocal off
            val = int.from_bytes(data[off:off + size], "little", signed=signed)
            off += size
            return val

        flags = read(4)
        op_idx = read(4)
        enable_mask = read(4)
        int_mask = read(4)
        int_clear = read(4)
        int_status = read(4)
        regcfg_amount = read(4)
        regcfg_offset = read(4)
        regcmd_addr = read(8)
        print(f"  task[{idx}]: flags=0x{flags:08x} op_idx={op_idx} enable_mask=0x{enable_mask:08x}")
        print(f"    int_mask=0x{int_mask:08x} int_clear=0x{int_clear:08x} int_status=0x{int_status:08x}")
        print(f"    regcfg_amount={regcfg_amount} regcfg_offset={regcfg_offset} regcmd_addr={fmt(regcmd_addr)}")


class Capture:
    calls = 0
    submits = 0

    @classmethod
    def handle(cls):
        cmd = int(gdb.parse_and_eval("$x1"))
        nr = (cmd >> IOC_NRSHIFT) & ((1 << IOC_NRBITS) - 1)
        size = (cmd >> IOC_SIZESHIFT) & ((1 << IOC_SIZEBITS) - 1)
        if nr != DRM_IOCTL_RKNPU_SUBMIT_NR:
            return
        cls.submits += 1
        arg = int(gdb.parse_and_eval("$x2"))
        print(f"\n=== DRM_IOCTL_RKNPU_SUBMIT #{cls.submits} cmd=0x{cmd:08x} size={size} arg={fmt(arg)} ===")
        try:
            submit = decode_submit(arg)
        except gdb.MemoryError:
            print("  submit struct unreadable")
            return
        print(f"  flags=0x{submit['flags']:08x} timeout={submit['timeout']} task_start={submit['task_start']} task_number={submit['task_number']}")
        print(f"  task_counter={submit['task_counter']} priority={submit['priority']} task_obj_addr={fmt(submit['task_obj_addr'])}")
        print(f"  field32=0x{submit['field32']:08x} field36=0x{submit['field36']:08x} task_base_addr={fmt(submit['task_base_addr'])} field48={fmt(submit['field48'])}")
        print(f"  core_mask=0x{submit['core_mask']:08x} fence_fd={submit['fence_fd']}")
        for idx, (start, number) in enumerate(submit["subcores"]):
            print(f"  subcore_task[{idx}]={{task_start={start}, task_number={number}}}")
        print_tasks(submit["task_obj_addr"], submit["task_number"])

end

break ioctl
commands
  silent
  python Capture.handle()
  continue
end

run
