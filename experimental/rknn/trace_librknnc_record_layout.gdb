set pagination off
set breakpoint pending on
set confirm off

python
import gdb


TOP_TILER_OFF = 0x002539c0
HIT_LIMIT = 3
PTR_MIN = 0x10000
PTR_MAX = 0x0001000000000000


def reg(name):
    try:
        return int(gdb.parse_and_eval("$" + name))
    except Exception:
        return 0


def read_mem(addr, size):
    return gdb.selected_inferior().read_memory(addr, size).tobytes()


def u64(addr):
    return int.from_bytes(read_mem(addr, 8), "little")


def u32(addr):
    return int.from_bytes(read_mem(addr, 4), "little")


def is_ptr(value):
    return PTR_MIN <= value < PTR_MAX


def find_librknnc_base():
    try:
        out = gdb.execute("info proc mappings", to_string=True)
    except gdb.error:
        return None
    base = None
    for line in out.splitlines():
        if "librknnc.so" not in line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            start = int(parts[0], 16)
        except ValueError:
            continue
        base = start if base is None else min(base, start)
    return base


def d32_line(addr, count):
    vals = []
    for i in range(count):
        try:
            vals.append(f"{u32(addr + i * 4):08x}")
        except Exception:
            if not vals:
                return "unreadable"
            break
    return " ".join(vals)


def q64_line(addr, count):
    vals = []
    for i in range(count):
        try:
            vals.append(f"+{i * 8:02x}=0x{u64(addr + i * 8):016x}")
        except Exception:
            if not vals:
                return "unreadable"
            break
    return " ".join(vals)


def vector_triple(holder):
    begin = u64(holder)
    end = u64(holder + 8)
    cap = u64(holder + 16)
    return begin, end, cap


def looks_like_vector(addr):
    try:
        begin, end, cap = vector_triple(addr)
    except Exception:
        return False
    if begin == 0 and end == 0 and cap == 0:
        return True
    if not (is_ptr(begin) and is_ptr(end) and is_ptr(cap)):
        return False
    if not (begin <= end <= cap):
        return False
    return (cap - begin) <= 0x4000


def dump_raw(label, addr, dwords=32, qwords=16):
    gdb.write(f"    {label} d32 {d32_line(addr, dwords)}\n")
    gdb.write(f"    {label} q64 {q64_line(addr, qwords)}\n")


def dump_nested_vector(label, holder, depth):
    try:
        begin, end, cap = vector_triple(holder)
    except Exception:
        gdb.write(f"    {label} vec unreadable\n")
        return
    span = max(0, end - begin)
    cap_span = max(0, cap - begin)
    gdb.write(
        f"    {label} vec begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} "
        f"bytes={span} cap_bytes={cap_span}\n"
    )
    if not begin or span <= 0 or span > 0x4000:
        return
    dump_raw(label + ".data", begin, min(96, max(1, span // 4)), min(48, max(1, span // 8)))
    if depth <= 0 or span % 8 != 0:
        return
    for i in range(min(8, span // 8)):
        try:
            ptr = u64(begin + i * 8)
        except Exception:
            continue
        if not is_ptr(ptr):
            continue
        if looks_like_vector(ptr):
            dump_nested_vector(f"{label}.ptr[{i}]@0x{ptr:x}", ptr, depth - 1)
        else:
            dump_raw(f"{label}.ptr[{i}]@0x{ptr:x}", ptr, 48, 24)


def dump_holder(name, holder):
    gdb.write(f"  holder {name}@0x{holder:x}\n")
    if looks_like_vector(holder):
        dump_nested_vector(name, holder, 2)
    else:
        dump_raw(name, holder, 32, 16)


class ExitBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace_librknnc_record_layout] process called exit; quitting gdb\n")
        gdb.execute("quit")
        return False


class ReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, hit_id, args):
        self.hit_id = hit_id
        self.args = args
        super().__init__(f"*0x{args['lr']:x}", internal=True)
        self.silent = True

    def stop(self):
        gdb.write(f"\n[top_tiler_record_return hit={self.hit_id} ret_x0=0x{reg('x0'):x}]\n")
        gdb.write(f"  dims_x6 {d32_line(self.args['x6'], 24)}\n")
        for name in ("x3", "x4", "x5", "x8"):
            dump_holder(name, self.args[name])
        self.enabled = False
        return False


class TopTilerBreakpoint(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.hits = 0

    def stop(self):
        self.hits += 1
        args = {name: reg(name) for name in ("x3", "x4", "x5", "x6", "x8", "lr")}
        gdb.write(f"\n[top_tiler_record_entry hit={self.hits} x6=0x{args['x6']:x} lr=0x{args['lr']:x}]\n")
        ReturnBreakpoint(self.hits, args)
        if self.hits >= HIT_LIMIT:
            self.enabled = False
        return False


class LoaderBreakpoint(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, internal=False)
        self.silent = True

    def stop(self):
        install()
        return False


installed = False


def install():
    global installed
    if installed:
        return
    base = find_librknnc_base()
    if base is None:
        return
    installed = True
    addr = base + TOP_TILER_OFF
    gdb.write(f"librknnc base=0x{base:x} top_tiler=0x{addr:x}\n")
    TopTilerBreakpoint(addr)


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBreakpoint(spec)
    except Exception:
        pass

try:
    ExitBreakpoint()
except Exception:
    pass

gdb.write("trace_librknnc_record_layout.gdb loaded\n")
end

run
