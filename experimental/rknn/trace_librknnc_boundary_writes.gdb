set pagination off
set breakpoint pending on
set confirm off

python
import gdb


TOP_TILER_RETURN_OFF = 0x00257140
BOUNDARY_LOOP_ENTRY_OFF = 0x002572c0
BOUNDARY_ZERO_OFF = 0x002572fc
BOUNDARY_WRITE_OFF = 0x00257330
BOUNDARY_DONE_OFF = 0x0025733c
HIT_LIMIT = 80
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


def vector_info(holder):
    try:
        begin = u64(holder)
        end = u64(holder + 8)
        cap = u64(holder + 16)
    except Exception:
        return None
    return begin, end, cap, max(0, end - begin)


def dump_vector(label, holder, words=16):
    info = vector_info(holder)
    if info is None:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    begin, end, cap, span = info
    gdb.write(f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if begin and span > 0:
        gdb.write(f"    d32 {d32_line(begin, min(words, max(1, span // 4)))}\n")


def dump_context(tag):
    sp = reg("sp")
    x2 = reg("x2")
    x3 = reg("x3")
    x5 = reg("x5")
    x7 = reg("x7")
    x8 = reg("x8")
    x19 = reg("x19")
    x20 = reg("x20")
    x21 = reg("x21")
    x22 = reg("x22")
    x23 = reg("x23")
    x24 = reg("x24")
    x25 = reg("x25")
    x26 = reg("x26")
    w0 = reg("x0") & 0xffffffff
    w1 = reg("x1") & 0xffffffff
    w9 = reg("x9") & 0xffffffff
    gdb.write(
        f"\n[{tag} pc=0x{reg('pc'):x} sp=0x{sp:x} "
        f"x19={x19} x20={x20} x22=0x{x22:x} x23={x23} x24=0x{x24:x} "
        f"x25={x25} x26=0x{x26:x} x21={x21} w0={w0} w1={w1} w9={w9}]\n"
    )
    gdb.write(f"  regs x2=0x{x2:x} x3=0x{x3:x} x5=0x{x5:x} x7=0x{x7:x} x8=0x{x8:x}\n")
    dump_vector("caller_x8_sp648", sp + 0x648, 24)
    dump_vector("source_record_vector_x8", x8, 24)
    if is_ptr(x5):
        try:
            rec = x5 + x20 * 8
            rb = u64(rec)
            re = u64(rec + 8)
            rc = u64(rec + 16)
            gdb.write(
                f"  selected_rec x5+x20*8=0x{rec:x}: "
                f"begin=0x{rb:x} end=0x{re:x} cap=0x{rc:x} bytes={max(0, re-rb)}\n"
            )
            if is_ptr(rb):
                gdb.write(f"    selected_rec.data d32 {d32_line(rb, 16)}\n")
        except Exception as exc:
            gdb.write(f"  selected_rec unreadable {exc}\n")
    if is_ptr(x2):
        gdb.write(f"  dest_x2 d32 {d32_line(x2, 16)}\n")
    if is_ptr(x3):
        gdb.write(f"  write_ptr_x3 d32 {d32_line(x3, 8)}\n")


class ExitBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace_librknnc_boundary_writes] process called exit; quitting gdb\n")
        gdb.execute("quit")
        return False


class TraceBreakpoint(gdb.Breakpoint):
    def __init__(self, name, addr, limit=HIT_LIMIT):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.name = name
        self.limit = limit
        self.hits = 0

    def stop(self):
        self.hits += 1
        dump_context(f"{self.name} hit={self.hits}")
        if self.hits >= self.limit:
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
    gdb.write(
        f"librknnc base=0x{base:x} return_site=0x{base + TOP_TILER_RETURN_OFF:x} "
        f"loop_entry=0x{base + BOUNDARY_LOOP_ENTRY_OFF:x} "
        f"zero=0x{base + BOUNDARY_ZERO_OFF:x} "
        f"write=0x{base + BOUNDARY_WRITE_OFF:x} "
        f"done=0x{base + BOUNDARY_DONE_OFF:x}\n"
    )
    TraceBreakpoint("return_site", base + TOP_TILER_RETURN_OFF, 3)
    TraceBreakpoint("boundary_loop_entry", base + BOUNDARY_LOOP_ENTRY_OFF)
    TraceBreakpoint("boundary_zero", base + BOUNDARY_ZERO_OFF)
    TraceBreakpoint("boundary_write", base + BOUNDARY_WRITE_OFF)
    TraceBreakpoint("boundary_done", base + BOUNDARY_DONE_OFF)


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBreakpoint(spec)
    except Exception:
        pass

try:
    ExitBreakpoint()
except Exception:
    pass

gdb.write("trace_librknnc_boundary_writes.gdb loaded\n")
end

run
