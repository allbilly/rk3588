set pagination off
set breakpoint pending on

python
import gdb
import os

IOC_NRBITS = 8
IOC_TYPEBITS = 8
IOC_SIZEBITS = 14
IOC_NRSHIFT = 0
IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
DRM_COMMAND_BASE = 0x40
DRM_IOCTL_RKNPU_ACTION_NR = DRM_COMMAND_BASE + 0
DRM_IOCTL_RKNPU_SUBMIT_NR = DRM_COMMAND_BASE + 1
DRM_IOCTL_RKNPU_MEM_CREATE_NR = DRM_COMMAND_BASE + 2
DRM_IOCTL_RKNPU_MEM_MAP_NR = DRM_COMMAND_BASE + 3
DRM_IOCTL_RKNPU_MEM_SYNC_NR = DRM_COMMAND_BASE + 5
ACTION_STRUCT_SIZE = 8
SUBMIT_STRUCT_SIZE = 104
MEM_CREATE_STRUCT_SIZE = 40
MEM_MAP_STRUCT_SIZE = 16
MEM_SYNC_STRUCT_SIZE = 32
SUBCORE_ENTRIES = 5
MMIO_SNAPSHOT = os.environ.get("RKNPU_MMIO_SNAPSHOT") == "1"
MMIO_SNAPSHOT_SCRIPT = "/home/orangepi/rk3588/experimental/rknn/snapshot_rknpu_mmio_readonly.py"
SYSFS_SNAPSHOT = os.environ.get("RKNPU_SYSFS_SNAPSHOT") == "1"
SYSFS_SNAPSHOT_SCRIPT = "/home/orangepi/rk3588/experimental/rknn/snapshot_rknpu_sysfs.py"


def snapshot_mmio(label):
    if MMIO_SNAPSHOT:
        gdb.execute(f"shell python3 {MMIO_SNAPSHOT_SCRIPT} --label {label}", to_string=False)
    if SYSFS_SNAPSHOT:
        gdb.execute(f"shell python3 {SYSFS_SNAPSHOT_SCRIPT} --label {label}", to_string=False)


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


def decode_action(addr):
    data = read_mem(addr, ACTION_STRUCT_SIZE)
    return {
        "flags": int.from_bytes(data[0:4], "little"),
        "value": int.from_bytes(data[4:8], "little"),
    }


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


def decode_mem_map(addr):
    data = read_mem(addr, MEM_MAP_STRUCT_SIZE)
    off = 0

    def read(size):
        nonlocal off
        val = int.from_bytes(data[off:off + size], "little")
        off += size
        return val

    return {
        "handle": read(4),
        "reserved": read(4),
        "offset": read(8),
    }


class Capture:
    calls = 0
    actions = []
    submits = 0
    mem_creates = []
    mem_maps = []
    mem_syncs = []

    @classmethod
    def handle(cls):
        cmd = int(gdb.parse_and_eval("$x1"))
        nr = (cmd >> IOC_NRSHIFT) & ((1 << IOC_NRBITS) - 1)
        if nr not in (DRM_IOCTL_RKNPU_ACTION_NR, DRM_IOCTL_RKNPU_SUBMIT_NR,
                      DRM_IOCTL_RKNPU_MEM_CREATE_NR, DRM_IOCTL_RKNPU_MEM_MAP_NR,
                      DRM_IOCTL_RKNPU_MEM_SYNC_NR):
            return
        cls.calls += 1
        arg = int(gdb.parse_and_eval("$x2"))
        size = (cmd >> IOC_SIZESHIFT) & ((1 << IOC_SIZEBITS) - 1)
        call_id = cls.calls
        kind = f"nr={nr}"
        if nr == DRM_IOCTL_RKNPU_ACTION_NR:
            try:
                action = decode_action(arg)
            except gdb.MemoryError:
                print(f"ACTION unreadable arg={fmt(arg)}")
                return
            cls.actions.append(action)
            kind = f"ACTION flags=0x{action['flags']:08x}"
            print(f"ACTION flags=0x{action['flags']:08x} value=0x{action['value']:08x}")
            IoctlReturn(call_id, kind)
            return
        if nr == DRM_IOCTL_RKNPU_MEM_CREATE_NR:
            try:
                mem = decode_mem_create(arg)
            except gdb.MemoryError:
                print(f"MEM_CREATE #{len(cls.mem_creates) + 1} unreadable arg={fmt(arg)}")
                return
            cls.mem_creates.append(mem)
            kind = f"MEM_CREATE #{len(cls.mem_creates)}"
            print(f"MEM_CREATE #{len(cls.mem_creates)} size={mem['size']} flags=0x{mem['flags']:08x} handle=0x{mem['handle']:x} obj={fmt(mem['obj_addr'])} dma={fmt(mem['dma_addr'])} sram={mem['sram_size']}")
            MemCreateReturn(call_id, kind, len(cls.mem_creates) - 1, arg)
            return
        if nr == DRM_IOCTL_RKNPU_MEM_MAP_NR:
            try:
                mem_map = decode_mem_map(arg)
            except gdb.MemoryError:
                print(f"MEM_MAP #{len(cls.mem_maps) + 1} unreadable arg={fmt(arg)}")
                return
            cls.mem_maps.append(mem_map)
            kind = f"MEM_MAP #{len(cls.mem_maps)}"
            print(f"MEM_MAP #{len(cls.mem_maps)} handle=0x{mem_map['handle']:x} reserved=0x{mem_map['reserved']:x} offset={fmt(mem_map['offset'])}")
            MemMapReturn(call_id, kind, len(cls.mem_maps) - 1, arg)
            return
        if nr == DRM_IOCTL_RKNPU_MEM_SYNC_NR:
            try:
                sync = decode_mem_sync(arg)
            except gdb.MemoryError:
                print(f"MEM_SYNC unreadable arg={fmt(arg)}")
                return
            cls.mem_syncs.append(sync)
            kind = f"MEM_SYNC flags=0x{sync['flags']:08x}"
            print(f"MEM_SYNC flags=0x{sync['flags']:08x} obj={fmt(sync['obj_addr'])} offset={sync['offset']} size={sync['size']}")
            IoctlReturn(call_id, kind)
            return
        cls.submits += 1
        kind = f"SUBMIT #{cls.submits}"
        snapshot_mmio(f"before_submit_{cls.submits}")
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
        IoctlReturn(call_id, kind)


class IoctlReturn(gdb.FinishBreakpoint):
    def __init__(self, call_id, kind):
        super().__init__(internal=True)
        self.call_id = call_id
        self.kind = kind

    def stop(self):
        try:
            ret = int(gdb.parse_and_eval("$x0"))
        except gdb.error:
            ret = 0
        if ret >= (1 << 63):
            ret -= 1 << 64
        print(f"IOCTL_RET #{self.call_id} {self.kind} ret={ret}")
        if self.kind.startswith("SUBMIT #"):
            snapshot_mmio(f"after_submit_{self.kind.split('#', 1)[1]}")
        return False


class MemCreateReturn(IoctlReturn):
    def __init__(self, call_id, kind, mem_index, arg):
        super().__init__(call_id, kind)
        self.mem_index = mem_index
        self.arg = arg

    def stop(self):
        try:
            ret = int(gdb.parse_and_eval("$x0"))
        except gdb.error:
            ret = 0
        if ret >= (1 << 63):
            ret -= 1 << 64
        try:
            mem = decode_mem_create(self.arg)
        except gdb.MemoryError:
            print(f"IOCTL_RET #{self.call_id} {self.kind} ret={ret}")
            print(f"MEM_CREATE_RET #{self.mem_index + 1} unreadable arg={fmt(self.arg)}")
            return False
        Capture.mem_creates[self.mem_index] = mem
        print(f"IOCTL_RET #{self.call_id} {self.kind} ret={ret}")
        print(f"MEM_CREATE_RET #{self.mem_index + 1} handle=0x{mem['handle']:x} obj={fmt(mem['obj_addr'])} dma={fmt(mem['dma_addr'])} sram={mem['sram_size']}")
        return False


class MemMapReturn(IoctlReturn):
    def __init__(self, call_id, kind, map_index, arg):
        super().__init__(call_id, kind)
        self.map_index = map_index
        self.arg = arg

    def stop(self):
        try:
            ret = int(gdb.parse_and_eval("$x0"))
        except gdb.error:
            ret = 0
        if ret >= (1 << 63):
            ret -= 1 << 64
        try:
            mem_map = decode_mem_map(self.arg)
        except gdb.MemoryError:
            print(f"IOCTL_RET #{self.call_id} {self.kind} ret={ret}")
            print(f"MEM_MAP_RET #{self.map_index + 1} unreadable arg={fmt(self.arg)}")
            return False
        Capture.mem_maps[self.map_index] = mem_map
        print(f"IOCTL_RET #{self.call_id} {self.kind} ret={ret}")
        print(f"MEM_MAP_RET #{self.map_index + 1} handle=0x{mem_map['handle']:x} reserved=0x{mem_map['reserved']:x} offset={fmt(mem_map['offset'])}")
        return False


class MmapCapture:
    calls = 0
    active_maps = []

    @classmethod
    def handle(cls):
        # AArch64 mmap(addr, length, prot, flags, fd, offset)
        addr = int(gdb.parse_and_eval("$x0"))
        length = int(gdb.parse_and_eval("$x1"))
        prot = int(gdb.parse_and_eval("$x2"))
        flags = int(gdb.parse_and_eval("$x3"))
        fd = int(gdb.parse_and_eval("$x4"))
        offset = int(gdb.parse_and_eval("$x5"))
        matching_maps = [
            (idx + 1, row)
            for idx, row in enumerate(Capture.mem_maps)
            if row["offset"] == offset and offset != 0
        ]
        if not matching_maps:
            return
        cls.calls += 1
        map_indices = ",".join(str(idx) for idx, _row in matching_maps)
        print(
            f"MMAP #{cls.calls} maps={map_indices} addr={fmt(addr)} length={length} "
            f"prot=0x{prot:x} flags=0x{flags:x} fd={fd} offset={fmt(offset)}"
        )
        MmapReturn(cls.calls, matching_maps, length, fd, offset)


class MmapReturn(gdb.FinishBreakpoint):
    def __init__(self, call_id, matching_maps, length, fd, offset):
        super().__init__(internal=True)
        self.call_id = call_id
        self.matching_maps = matching_maps
        self.length = length
        self.fd = fd
        self.offset = offset

    def stop(self):
        try:
            ret = int(gdb.parse_and_eval("$x0"))
        except gdb.error:
            ret = 0
        if ret >= (1 << 63):
            ret -= 1 << 64
        print(f"MMAP_RET #{self.call_id} addr={fmt(ret)}")
        if ret > 0:
            MmapCapture.active_maps.append({
                "call_id": self.call_id,
                "map_indices": tuple(idx for idx, _row in self.matching_maps),
                "addr": ret,
                "length": self.length,
                "fd": self.fd,
                "offset": self.offset,
            })
        return False


class MunmapCapture:
    calls = 0

    @classmethod
    def handle(cls):
        addr = int(gdb.parse_and_eval("$x0"))
        length = int(gdb.parse_and_eval("$x1"))
        matches = [
            row for row in MmapCapture.active_maps
            if row["addr"] == addr
        ]
        if not matches:
            return
        cls.calls += 1
        map_indices = ",".join(str(idx) for idx in matches[0]["map_indices"])
        print(
            f"MUNMAP #{cls.calls} maps={map_indices} addr={fmt(addr)} length={length} "
            f"expected_length={matches[0]['length']} fd={matches[0]['fd']} offset={fmt(matches[0]['offset'])}"
        )
        MunmapReturn(cls.calls)


class MunmapReturn(gdb.FinishBreakpoint):
    def __init__(self, call_id):
        super().__init__(internal=True)
        self.call_id = call_id

    def stop(self):
        try:
            ret = int(gdb.parse_and_eval("$x0"))
        except gdb.error:
            ret = 0
        if ret >= (1 << 63):
            ret -= 1 << 64
        print(f"MUNMAP_RET #{self.call_id} ret={ret}")
        return False


class CloseCapture:
    calls = 0

    @classmethod
    def handle(cls):
        fd = int(gdb.parse_and_eval("$x0"))
        # RKNN BO fds in the current traces are >= 4; fd 3 is /dev/dri/card1.
        if fd < 3:
            return
        if not Capture.mem_creates and not MmapCapture.active_maps:
            return
        cls.calls += 1
        print(f"CLOSE #{cls.calls} fd={fd}")
        CloseReturn(cls.calls)


class CloseReturn(gdb.FinishBreakpoint):
    def __init__(self, call_id):
        super().__init__(internal=True)
        self.call_id = call_id

    def stop(self):
        try:
            ret = int(gdb.parse_and_eval("$x0"))
        except gdb.error:
            ret = 0
        if ret >= (1 << 63):
            ret -= 1 << 64
        print(f"CLOSE_RET #{self.call_id} ret={ret}")
        return False

end

break ioctl
commands
  silent
  python Capture.handle()
  continue
end

break mmap
commands
  silent
  python MmapCapture.handle()
  continue
end

break munmap
commands
  silent
  python MunmapCapture.handle()
  continue
end

break close
commands
  silent
  python CloseCapture.handle()
  continue
end

run
