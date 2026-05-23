set pagination off
set breakpoint pending on
set confirm off

python
import gdb

TRACE_POINTS = [
    ("row_scan_a", 0x00635ce8, 80),
    ("row_scan_b", 0x01146cd0, 80),
    ("row_scan_c", 0x01707520, 80),
    ("row_republish_before", 0x00cac928, 80),
    ("row_republish_commit", 0x00cac94c, 80),
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


def dump_vec(label, holder, max_bytes=64):
    try:
        begin = u64(holder)
        end = u64(holder + 8)
        cap = u64(holder + 16)
    except Exception:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    span = max(0, end - begin)
    gdb.write(f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if is_ptr(begin) and 0 < span <= max_bytes:
        gdb.write(f"    d32 {d32_line(begin, max(1, span // 4))}\n")


def dump_row(label, addr):
    if not is_ptr(addr):
        gdb.write(f"  {label}=0x{addr:x} not-ptr\n")
        return
    gdb.write(f"  {label}=0x{addr:x}\n")
    gdb.write(f"    header d32 {d32_line(addr, 16)}\n")
    gdb.write(f"    tail178 d32 {d32_line(addr + 0x178, 12)}\n")
    for off in (0x40, 0x58, 0x70, 0xa0, 0xb8, 0xd0, 0x100, 0x118, 0x130):
        dump_vec(f"vec+0x{off:x}", addr + off)


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


class ConsumerTrace:
    @staticmethod
    def handle(name, hit):
        sp = reg("sp")
        gdb.write(f"\n[{name} hit={hit} sp=0x{sp:x} lr=0x{reg('lr'):x}]\n")
        gdb.write(
            "  regs "
            f"x0=0x{reg('x0'):x} x1=0x{reg('x1'):x} x2=0x{reg('x2'):x} "
            f"x3=0x{reg('x3'):x} x4=0x{reg('x4'):x} x5=0x{reg('x5'):x} "
            f"x23=0x{reg('x23'):x} x27=0x{reg('x27'):x} x28=0x{reg('x28'):x} "
            f"w1={reg('w1') & 0xffffffff} w2={reg('w2') & 0xffffffff} w20={reg('w20') & 0xffffffff}\n"
        )
        if name.startswith("row_scan"):
            # x0 is the current row pointer after add target, so dump the previous row candidate too.
            dump_row("scan_current_x0", reg("x0"))
            dump_row("scan_prev_x0_minus_1a8", reg("x0") - 0x1a8)
            for off in (736, 744, 1104, 1112):
                try:
                    gdb.write(f"  sp+{off}=0x{u64(sp+off):x}\n")
                except Exception:
                    pass
        else:
            dump_row("x19_row", reg("x19"))
            dump_row("x27_stack_row", reg("x27"))
            dump_vec("x23_vec", reg("x23"))
            try:
                dump_row("x23_current_end", u64(reg("x23") + 8))
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
        ConsumerTrace.handle(self.name, self.hits)
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

gdb.write("trace_librknnc_row_consumers.gdb loaded\n")
end

run
