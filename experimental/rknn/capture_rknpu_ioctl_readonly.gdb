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
DRM_IOCTL_RKNPU_MEM_CREATE_NR = DRM_COMMAND_BASE + 2
DRM_IOCTL_RKNPU_MEM_SYNC_NR = DRM_COMMAND_BASE + 5
SUBMIT_STRUCT_SIZE = 104
MEM_CREATE_STRUCT_SIZE = 40
MEM_SYNC_STRUCT_SIZE = 32
SUBCORE_ENTRIES = 5


def fmt(val):
    return f"0x{val:x}"


def read_mem(addr, size):
    return gdb.selected_inferior().read_memory(addr, size).tobytes()


def decode_submit(addr):
    data = read_mem(addr, SUBMIT_STRUCT_SIZE)
    off = 0

    def read(size, signed=False):
        nonlocal off
        val = int.from_bytes(data[off:off + size], "little", signed=signed)
        off += size
        return val

    out = {
        "flags": read(4),
        "timeout": read(4),
        "task_start": read(4),
        "task_number": read(4),
        "task_counter": read(4),
        "priority": read(4, signed=True),
        "task_obj_addr": read(8),
        "regcfg_obj_addr": read(8),
        "task_base_addr": read(8),
        "user_data": read(8),
        "core_mask": read(4),
        "fence_fd": read(4, signed=True),
        "subcores": [],
    }
    for _ in range(SUBCORE_ENTRIES):
        out["subcores"].append((read(4), read(4)))
    return out


def decode_mem_create(addr):
    data = read_mem(addr, MEM_CREATE_STRUCT_SIZE)
    off = 0

    def read(size):
        nonlocal off
        val = int.from_bytes(data[off:off + size], "little")
        off += size
        return val

    return {
        "handle": read(4),
        "flags": read(4),
        "size": read(8),
        "obj_addr": read(8),
        "dma_addr": read(8),
        "sram_size": read(8),
    }


def decode_mem_sync(addr):
    data = read_mem(addr, MEM_SYNC_STRUCT_SIZE)
    off = 0

    def read(size):
        nonlocal off
        val = int.from_bytes(data[off:off + size], "little")
        off += size
        return val

    return {
        "flags": read(4),
        "reserved": read(4),
        "obj_addr": read(8),
        "offset": read(8),
        "size": read(8),
    }


class Capture:
    calls = 0
    submits = 0
    mem_creates = []
    mem_syncs = []

    @classmethod
    def handle(cls):
        cmd = int(gdb.parse_and_eval("$x1"))
        nr = (cmd >> IOC_NRSHIFT) & ((1 << IOC_NRBITS) - 1)
        if nr not in (DRM_IOCTL_RKNPU_SUBMIT_NR, DRM_IOCTL_RKNPU_MEM_CREATE_NR, DRM_IOCTL_RKNPU_MEM_SYNC_NR):
            return
        cls.calls += 1
        arg = int(gdb.parse_and_eval("$x2"))
        size = (cmd >> IOC_SIZESHIFT) & ((1 << IOC_SIZEBITS) - 1)
        if nr == DRM_IOCTL_RKNPU_MEM_CREATE_NR:
            try:
                mem = decode_mem_create(arg)
            except gdb.MemoryError:
                print(f"MEM_CREATE #{len(cls.mem_creates) + 1} unreadable arg={fmt(arg)}")
                return
            cls.mem_creates.append(mem)
            print(f"MEM_CREATE #{len(cls.mem_creates)} size={mem['size']} flags=0x{mem['flags']:08x} handle=0x{mem['handle']:x} obj={fmt(mem['obj_addr'])} dma={fmt(mem['dma_addr'])} sram={mem['sram_size']}")
            return
        if nr == DRM_IOCTL_RKNPU_MEM_SYNC_NR:
            try:
                sync = decode_mem_sync(arg)
            except gdb.MemoryError:
                print(f"MEM_SYNC unreadable arg={fmt(arg)}")
                return
            cls.mem_syncs.append(sync)
            print(f"MEM_SYNC flags=0x{sync['flags']:08x} obj={fmt(sync['obj_addr'])} offset={sync['offset']} size={sync['size']}")
            return
        cls.submits += 1
        print(f"\n=== DRM_IOCTL_RKNPU_SUBMIT #{cls.submits} cmd=0x{cmd:08x} size={size} arg={fmt(arg)} ===")
        try:
            submit = decode_submit(arg)
        except gdb.MemoryError:
            print("  submit struct unreadable")
            return
        print(f"  flags=0x{submit['flags']:08x} timeout={submit['timeout']} task_start={submit['task_start']} task_number={submit['task_number']}")
        print(f"  task_counter={submit['task_counter']} priority={submit['priority']} task_obj_addr={fmt(submit['task_obj_addr'])}")
        print(f"  regcfg_obj_addr={fmt(submit['regcfg_obj_addr'])} task_base_addr={fmt(submit['task_base_addr'])} user_data={fmt(submit['user_data'])}")
        print(f"  core_mask=0x{submit['core_mask']:08x} fence_fd={submit['fence_fd']}")
        for idx, (start, number) in enumerate(submit["subcores"]):
            print(f"  subcore_task[{idx}]={{task_start={start}, task_number={number}}}")
        matches = [mem for mem in cls.mem_creates if mem["obj_addr"] == submit["task_obj_addr"]]
        print(f"  task_obj_mem_create_matches={len(matches)}")
        for mem in matches:
            print(f"    match size={mem['size']} handle=0x{mem['handle']:x} dma={fmt(mem['dma_addr'])} flags=0x{mem['flags']:08x}")
        sync_matches = [sync for sync in cls.mem_syncs if sync["obj_addr"] == submit["task_obj_addr"]]
        print(f"  task_obj_mem_sync_matches={len(sync_matches)}")
        for sync in sync_matches:
            print(f"    sync flags=0x{sync['flags']:08x} offset={sync['offset']} size={sync['size']}")

end

break ioctl
commands
  silent
  python Capture.handle()
  continue
end

run
