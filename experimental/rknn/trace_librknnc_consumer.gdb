set pagination off
set breakpoint pending on
set confirm off

python
import gdb


TOP_TILER_OFF = 0x002539c0
TOP_TILER_RETURN_OFF = 0x00257140
CONSUMER_LOOP_OFF = 0x00257240
CONSUMER_LOOP_LIMIT = 18
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
    return u64(holder), u64(holder + 8), u64(holder + 16)


def dump_vector(label, holder, max_bytes=0x180):
    try:
        begin, end, cap = vector_triple(holder)
    except Exception:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    span = max(0, end - begin)
    gdb.write(f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if not begin or span <= 0:
        return
    dump = min(span, max_bytes)
    gdb.write(f"    d32 {d32_line(begin, max(1, dump // 4))}\n")
    gdb.write(f"    q64 {q64_line(begin, max(1, dump // 8))}\n")
    if span % 24 == 0:
        for i in range(min(10, span // 24)):
            rec = begin + i * 24
            try:
                r0, r1, r2 = u64(rec), u64(rec + 8), u64(rec + 16)
            except Exception:
                continue
            gdb.write(f"    rec24[{i}] +0=0x{r0:016x} +8=0x{r1:016x} +16=0x{r2:016x}\n")
            if is_ptr(r0):
                gdb.write(f"      rec24[{i}].ptr0 d32 {d32_line(r0, 24)}\n")


def dump_stack_vectors(sp):
    # These stack offsets come from the caller at librknnc.so+0x2570fc..0x257138.
    for label, off in (
        ("caller_x4_sp510", 0x510),
        ("caller_x5_sp530", 0x530),
        ("caller_x8_sp648", 0x648),
        ("outer_vec_sp5b0", 0x5b0),
        ("outer_vec_sp5d0", 0x5d0),
        ("outer_vec_sp5f0", 0x5f0),
        ("outer_vec_sp630", 0x630),
    ):
        dump_vector(label, sp + off)


class ExitBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace_librknnc_consumer] process called exit; quitting gdb\n")
        gdb.execute("quit")
        return False


class TopTilerEntryBreakpoint(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.hits = 0

    def stop(self):
        self.hits += 1
        gdb.write(
            f"\n[top_tiler_entry hit={self.hits} sp=0x{reg('sp'):x} "
            f"x3=0x{reg('x3'):x} x4=0x{reg('x4'):x} x5=0x{reg('x5'):x} "
            f"x6=0x{reg('x6'):x} x8=0x{reg('x8'):x} lr=0x{reg('lr'):x}]\n"
        )
        if self.hits >= HIT_LIMIT:
            self.enabled = False
        return False


class TopTilerReturnSiteBreakpoint(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.hits = 0

    def stop(self):
        self.hits += 1
        sp = reg("sp")
        gdb.write(f"\n[top_tiler_consumer_return_site hit={self.hits} sp=0x{sp:x}]\n")
        dump_stack_vectors(sp)
        if self.hits >= HIT_LIMIT:
            self.enabled = False
        return False


class ConsumerLoopBreakpoint(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.hits = 0

    def stop(self):
        self.hits += 1
        sp = reg("sp")
        gdb.write(
            f"\n[consumer_loop hit={self.hits} sp=0x{sp:x} "
            f"x6=0x{reg('x6'):x} x22=0x{reg('x22'):x} x27=0x{reg('x27'):x}]\n"
        )
        dump_stack_vectors(sp)
        if self.hits >= CONSUMER_LOOP_LIMIT:
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
        f"librknnc base=0x{base:x} top_tiler=0x{base + TOP_TILER_OFF:x} "
        f"return_site=0x{base + TOP_TILER_RETURN_OFF:x} "
        f"consumer_loop=0x{base + CONSUMER_LOOP_OFF:x}\n"
    )
    TopTilerEntryBreakpoint(base + TOP_TILER_OFF)
    TopTilerReturnSiteBreakpoint(base + TOP_TILER_RETURN_OFF)
    ConsumerLoopBreakpoint(base + CONSUMER_LOOP_OFF)


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBreakpoint(spec)
    except Exception:
        pass

try:
    ExitBreakpoint()
except Exception:
    pass

gdb.write("trace_librknnc_consumer.gdb loaded\n")
end

run
