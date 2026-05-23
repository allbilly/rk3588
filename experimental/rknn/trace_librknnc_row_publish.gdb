set pagination off
set breakpoint pending on
set confirm off

python
import gdb

TRACE_POINTS = [
    ("row_record_write", 0x00257f00, 120),
    ("row_publish_begin", 0x00258444, 120),
    ("row_publish_slot", 0x002584b8, 120),
    ("row_publish_after_vectors", 0x002589d0, 120),
    ("row_publish_commit", 0x00258a14, 120),
]

PTR_MIN = 0x10000
PTR_MAX = 0x0001000000000000


def reg(name):
    try:
        return int(gdb.parse_and_eval("$" + name))
    except Exception:
        return 0


def read_mem(addr, size):
    return gdb.selected_inferior().read_memory(addr, size).tobytes()


def u32(addr):
    return int.from_bytes(read_mem(addr, 4), "little")


def u64(addr):
    return int.from_bytes(read_mem(addr, 8), "little")


def is_ptr(value):
    return PTR_MIN <= value < PTR_MAX


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


def vector_info(holder):
    try:
        begin = u64(holder)
        end = u64(holder + 8)
        cap = u64(holder + 16)
    except Exception:
        return None
    return begin, end, cap, max(0, end - begin)


def dump_vector(label, holder, max_bytes=96):
    info = vector_info(holder)
    if info is None:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    begin, end, cap, span = info
    gdb.write(f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if is_ptr(begin) and 0 < span <= max_bytes:
        gdb.write(f"    d32 {d32_line(begin, max(1, span // 4))}\n")


def dump_row(label, addr):
    if not is_ptr(addr):
        gdb.write(f"  {label}=0x{addr:x} not-ptr\n")
        return
    gdb.write(f"  {label}=0x{addr:x}\n")
    gdb.write(f"    header+0x000 d32 {d32_line(addr, 16)}\n")
    gdb.write(f"    scalars+0x178 d32 {d32_line(addr + 0x178, 16)}\n")
    gdb.write(f"    tail+0x188 d32 {d32_line(addr + 0x188, 20)}\n")
    for off in (0x40, 0x58, 0x70, 0x88, 0xa0, 0xb8, 0xd0, 0xe8, 0x100, 0x118, 0x130):
        dump_vector(f"row_vec+0x{off:x}", addr + off, max_bytes=64)


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


class RowTrace:
    @staticmethod
    def handle(name, hit):
        sp = reg("sp")
        x19 = reg("x19")
        x21 = reg("x21")
        x7 = reg("x7")
        gdb.write(f"\n[{name} hit={hit} sp=0x{sp:x} x19=0x{x19:x} x21=0x{x21:x} x7=0x{x7:x}]\n")
        gdb.write(
            "  regs "
            f"w0={reg('w0') & 0xffffffff} w1={reg('w1') & 0xffffffff} "
            f"w2={reg('w2') & 0xffffffff} w3={reg('w3') & 0xffffffff} "
            f"w19={reg('w19') & 0xffffffff} w21={reg('w21') & 0xffffffff} "
            f"w23={reg('w23') & 0xffffffff} w24={reg('w24') & 0xffffffff} "
            f"w28={reg('w28') & 0xffffffff}\n"
        )
        gdb.write(f"  stack_record_sp+0x7d8 d32 {d32_line(sp + 0x7d8, 8)}\n")
        gdb.write(f"  staged_sp+0x8c0 d32 {d32_line(sp + 0x8c0, 20)}\n")
        gdb.write(f"  staged_tail_sp+0x990 d32 {d32_line(sp + 0x990 - 8, 12)}\n")
        if name == "row_record_write" and is_ptr(x7):
            gdb.write(f"  x7_record_after_write_target d32 {d32_line(x7 - 8, 4)}\n")
        if name in ("row_publish_begin", "row_publish_slot"):
            try:
                begin = u64(x21 + 8)
                end = u64(x21 + 16)
                gdb.write(f"  dest_vector_at_x21: begin=0x{begin:x} end=0x{end:x} bytes={end - begin}\n")
                dump_row("dest_current_slot", begin)
            except Exception as exc:
                gdb.write(f"  dest_vector_at_x21 unreadable: {exc}\n")
        if name in ("row_publish_after_vectors", "row_publish_commit"):
            dump_row("published_row_x19", x19)
            try:
                next_ptr = u64(x21 + 8)
                gdb.write(f"  x21_next_after_commit_candidate=0x{next_ptr:x}\n")
            except Exception:
                pass


class ExitBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace] process exit; quitting\n")
        gdb.execute("quit")
        return False


class TracerBP(gdb.Breakpoint):
    def __init__(self, name, addr, limit):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.name = name
        self.limit = limit
        self.hits = 0

    def stop(self):
        self.hits += 1
        RowTrace.handle(self.name, self.hits)
        if self.hits >= self.limit:
            self.enabled = False
        return False


class LoaderBP(gdb.Breakpoint):
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
    gdb.write("librknnc base=0x%x\n" % base)
    for name, off, limit in TRACE_POINTS:
        addr = base + off
        gdb.write(f"  trace {name}=0x{addr:x}\n")
        TracerBP(name, addr, limit)


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBP(spec)
    except Exception:
        pass

try:
    ExitBreakpoint()
except Exception:
    pass

gdb.write("trace_librknnc_row_publish.gdb loaded\n")
end

run
