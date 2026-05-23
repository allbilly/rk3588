set pagination off
set breakpoint pending on
set confirm off

python
import gdb
import struct

g_base = None
g_regcmd_buf = None
g_regcmd_size = 0


def u64(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 8).tobytes(), "little")


def u32(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 4).tobytes(), "little")


def find_librknnrt_base():
    out = gdb.execute("info proc mappings", to_string=True)
    for line in out.splitlines():
        if "librknnrt" not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            return int(parts[0], 16)
        except ValueError:
            continue
    return None


def fmt_cmd(cmd):
    target = (cmd >> 48) & 0xffff
    value = (cmd >> 16) & 0xffffffff
    reg = cmd & 0xffff
    return target, reg, value


def find_regcmd_in_maps():
    pid = gdb.selected_inferior().pid
    try:
        with open(f"/proc/{pid}/maps") as f:
            lines = f.readlines()
    except:
        return None, 0

    candidates = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        addr_range = parts[0]
        perms = parts[1] if len(parts) > 1 else ""
        pathname = " ".join(parts[5:]) if len(parts) > 5 else ""
        start_s, end_s = addr_range.split("-")
        start = int(start_s, 16)
        end = int(end_s, 16)
        size = end - start

        if "rknpu" in pathname or "rknn" in pathname:
            continue
        if size < 256 or size > 0x100000:
            continue
        if "r" not in perms:
            continue

        try:
            inf = gdb.selected_inferior()
            first_word = int.from_bytes(inf.read_memory(start, 4).tobytes(), "little")
        except:
            continue

        candidates.append((start, size, pathname, first_word))

    return candidates


def dump_all_regcmd_buffers():
    pid = gdb.selected_inferior().pid
    inf = gdb.selected_inferior()

    try:
        with open(f"/proc/{pid}/maps") as f:
            lines = f.readlines()
    except Exception as e:
        gdb.write(f"Cannot read maps: {e}\n")
        return

    gdb.write(f"\n{'='*70}\n")
    gdb.write(f"Scanning memory for register command buffers\n")
    gdb.write(f"{'='*70}\n")

    all_cmds = []

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        addr_range = parts[0]
        perms = parts[1]
        pathname = " ".join(parts[5:]) if len(parts) > 5 else ""
        start_s, end_s = addr_range.split("-")
        start = int(start_s, 16)
        end = int(end_s, 16)
        size = end - start

        if "rw" not in perms:
            continue
        if size > 0x100000:
            continue

        try:
            data = bytes(inf.read_memory(start, size))
        except:
            continue

        for off in range(0, len(data) - 8, 8):
            cmd = struct.unpack_from("<Q", data, off)[0]
            target = (cmd >> 48) & 0xffff
            if target == 0 or target > 0x100:
                continue
            reg = cmd & 0xffff
            value = (cmd >> 16) & 0xffffffff
            if reg == 0 and value == 0:
                continue
            all_cmds.append((start + off, target, reg, value, cmd))

    if not all_cmds:
        gdb.write("No register commands found!\n")
        return

    by_target = {}
    for addr, target, reg, value, raw in all_cmds:
        by_target.setdefault(target, []).append((addr, reg, value, raw))

    gdb.write(f"\nFound {len(all_cmds)} register commands across {len(by_target)} targets\n")

    for target in sorted(by_target.keys()):
        cmds = by_target[target]
        gdb.write(f"\n  Target 0x{target:04x} ({len(cmds)} commands):\n")
        by_reg = {}
        for addr, reg, value, raw in cmds:
            by_reg.setdefault(reg, []).append((addr, value, raw))
        for reg in sorted(by_reg.keys()):
            entries = by_reg[reg]
            gdb.write(f"    reg 0x{reg:04x}: {len(entries)} writes\n")
            for addr, value, raw in entries[:4]:
                gdb.write(f"      @0x{addr:x} = 0x{value:08x}\n")
            if len(entries) > 4:
                gdb.write(f"      ... {len(entries)-4} more\n")


class InitHook(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknn_init", internal=False)

    def stop(self):
        global g_base
        base = find_librknnrt_base()
        if base:
            g_base = base
            gdb.write(f"librknnrt base=0x{base:x}\n")
        return False


class RunHook(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknn_run", internal=False)

    def stop(self):
        gdb.write(f"\nrknn_run hit - dumping register commands\n")
        dump_all_regcmd_buffers()
        return False


class DestroyHook(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknn_destroy", internal=False)

    def stop(self):
        gdb.write(f"\nrknn_destroy hit - dumping final register commands\n")
        dump_all_regcmd_buffers()
        return False


InitHook()
RunHook()
DestroyHook()
gdb.write("trace_dump_regcmds.gdb loaded\n")
end

run
