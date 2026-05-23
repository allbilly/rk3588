set pagination off
set breakpoint pending on
set confirm off

python
import gdb

HELPER_INDEXED_OFF = 0x002efd38
HELPER_PAIR_OFF = 0x002efce0
HELPER_BFI_OFF = 0x00100a40


def u64(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 8).tobytes(), "little")


def u32(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 4).tobytes(), "little")


def fmt_cmd(cmd):
    target = (cmd >> 48) & 0xffff
    value = (cmd >> 16) & 0xffffffff
    reg = cmd & 0xffff
    return f"cmd=0x{cmd:016x} target=0x{target:04x} reg=0x{reg:04x} value=0x{value:08x}"


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


class ReuseTrace:
    installed = False
    base = None

    @classmethod
    def install(cls):
        if cls.installed:
            return
        base = find_librknnrt_base()
        if base is None:
            return
        cls.base = base
        cls.installed = True
        gdb.write(f"librknnrt base=0x{base:x}\n")
        gdb.write(f"indexed helper=0x{base + HELPER_INDEXED_OFF:x}\n")
        gdb.write(f"pair helper=0x{base + HELPER_PAIR_OFF:x}\n")
        gdb.write(f"bfi helper=0x{base + HELPER_BFI_OFF:x}\n")
        HelperBreakpoint(f"*0x{base + HELPER_INDEXED_OFF:x}")
        HelperBreakpoint(f"*0x{base + HELPER_PAIR_OFF:x}")

    @classmethod
    def handle_stop(cls):
        pc = int(gdb.parse_and_eval("$pc"))
        lr = int(gdb.parse_and_eval("$lr"))
        if cls.base is None:
            return False
        if pc == cls.base + HELPER_INDEXED_OFF:
            cls.handle_indexed(lr)
            return True
        if pc == cls.base + HELPER_PAIR_OFF:
            cls.handle_pair(lr)
            return True
        return False

    @staticmethod
    def handle_indexed(lr):
        ctx = int(gdb.parse_and_eval("$x0"))
        index = int(gdb.parse_and_eval("$w1"))
        value = int(gdb.parse_and_eval("$w2"))
        gdb.write(f"\nreuse_indexed lr=0x{lr:x} ctx=0x{ctx:x} index={index} value=0x{value:08x}\n")
        try:
            base = u64(ctx + 0x288)
            vec_base = u64(base + 0x08)
            vec_count = u64(base + 0x28)
            slot = vec_base + index * 8
            old_cmd = u64(slot)
            gdb.write(f"  ctx+0x288=0x{base:x} vec_base=0x{vec_base:x} vec_count={vec_count} slot=0x{slot:x}\n")
            gdb.write(f"  old {fmt_cmd(old_cmd)}\n")
        except Exception as exc:
            gdb.write(f"  indexed pre-read failed: {exc}\n")

    @staticmethod
    def handle_pair(lr):
        ctx = int(gdb.parse_and_eval("$x0"))
        record = int(gdb.parse_and_eval("$x1"))
        value = int(gdb.parse_and_eval("$w2"))
        gdb.write(f"\nreuse_pair lr=0x{lr:x} ctx=0x{ctx:x} record=0x{record:x} value=0x{value:08x}\n")
        try:
            first = u64(record)
            second = u64(record + 8)
            gdb.write(f"  pair[0]=0x{first:x} pair[1]=0x{second:x}\n")
            if first:
                gdb.write(f"  pair[0]+4 old=0x{u32(first + 4):08x}\n")
            if second:
                old_cmd = u64(second)
                gdb.write(f"  old {fmt_cmd(old_cmd)}\n")
        except Exception as exc:
            gdb.write(f"  pair pre-read failed: {exc}\n")


class HelperBreakpoint(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, internal=False)

    def stop(self):
        ReuseTrace.handle_stop()
        return False


class SolibHook(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknn_init", internal=False)

    def stop(self):
        ReuseTrace.install()
        return False


SolibHook()
gdb.write("trace_librknnrt_reuse.gdb loaded; run an RKNN conv model that triggers F2 ChannelTile\n")
end

run
