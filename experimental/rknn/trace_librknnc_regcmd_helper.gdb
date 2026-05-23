set pagination off
set breakpoint pending on
set confirm off

python
import gdb


# Offsets are for experimental/rknn/librknnc.so:
# sha256 d499753a91065f0b52b2cdfa43c073645bcc37467c8e077372b302ccafd5d53c
TRACE_POINTS = [
    ("helper_callsite", 0x00636574, 120),
    ("helper_entry", 0x005aea40, 120),
    ("helper_after_mode", 0x005aeabc, 120),
    ("helper_dims_ready", 0x005aeb30, 120),
    ("helper_emit_call", 0x005aece4, 120),
    ("helper_publish16", 0x005aeda8, 120),
    ("helper_return", 0x005aedb8, 120),
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


def q64_line(addr, count):
    vals = []
    for i in range(count):
        try:
            vals.append(f"{u64(addr + i * 8):016x}")
        except Exception:
            if not vals:
                return "unreadable"
            break
    return " ".join(vals)


def dump_vec(label, holder, max_bytes=128):
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
        gdb.write(f"    q64 {q64_line(begin, max(1, span // 8))}\n")


def dump_task(label, addr):
    if not is_ptr(addr):
        gdb.write(f"  {label}=0x{addr:x} not-ptr\n")
        return
    gdb.write(f"  {label}=0x{addr:x}\n")
    gdb.write(f"    d32+0x000 {d32_line(addr, 32)}\n")
    for off in (0x40, 0x58, 0x70, 0x88, 0xa0, 0xb8, 0xd0, 0xe8):
        dump_vec(f"{label}+0x{off:x}", addr + off, 80)
    gdb.write(f"    d32+0x178 {d32_line(addr + 0x178, 20)}\n")


def dump_sp(sp):
    gdb.write(f"  sp d32+0x4c0 {d32_line(sp + 0x4c0, 40)}\n")
    gdb.write(f"  sp d32+0x600 {d32_line(sp + 0x600, 28)}\n")
    for off in (0x4c0, 0x600, 0x628, 0x630, 0x638, 0x650, 0x660):
        dump_vec(f"sp+0x{off:x}", sp + off, 96)


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


class ReturnBP(gdb.Breakpoint):
    def __init__(self, lr, tag):
        super().__init__(f"*0x{lr:x}", internal=True)
        self.silent = True
        self.tag = tag

    def stop(self):
        gdb.write(f"[return:{self.tag} pc=0x{reg('pc'):x}] w0=0x{reg('w0') & 0xffffffff:x} x0=0x{reg('x0'):x}\n")
        self.enabled = False
        return False


class TraceBP(gdb.Breakpoint):
    def __init__(self, name, addr, limit):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.silent = True
        self.name = name
        self.limit = limit
        self.hits = 0

    def stop(self):
        self.hits += 1
        sp = reg("sp")
        gdb.write(f"\n[{self.name} hit={self.hits} pc=0x{reg('pc'):x} lr=0x{reg('lr'):x} sp=0x{sp:x}]\n")
        gdb.write(
            "  regs "
            f"x0=0x{reg('x0'):x} x1=0x{reg('x1'):x} x2=0x{reg('x2'):x} x3=0x{reg('x3'):x} "
            f"x4=0x{reg('x4'):x} x5=0x{reg('x5'):x} x6=0x{reg('x6'):x} x7=0x{reg('x7'):x} "
            f"x19=0x{reg('x19'):x} x20=0x{reg('x20'):x} x21=0x{reg('x21'):x} x22=0x{reg('x22'):x} "
            f"w0=0x{reg('w0') & 0xffffffff:x} w1=0x{reg('w1') & 0xffffffff:x} "
            f"w2=0x{reg('w2') & 0xffffffff:x} w3=0x{reg('w3') & 0xffffffff:x} "
            f"w4=0x{reg('w4') & 0xffffffff:x} w5=0x{reg('w5') & 0xffffffff:x} "
            f"w6=0x{reg('w6') & 0xffffffff:x}\n"
        )
        if self.name == "helper_callsite":
            dump_task("call_task_x4", reg("x4"))
            dump_sp(sp)
            ReturnBP(reg("lr"), "helper_callsite")
        else:
            dump_task("helper_task_x21", reg("x21"))
            dump_task("helper_x4", reg("x4"))
            dump_sp(sp)
            if self.name == "helper_entry":
                ReturnBP(reg("lr"), "helper_entry")
        if self.hits >= self.limit:
            self.enabled = False
        return False


class ExitBP(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace] process exit; quitting\n")
        gdb.execute("quit")
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
    gdb.write(f"librknnc base=0x{base:x}\n")
    for name, off, limit in TRACE_POINTS:
        addr = base + off
        gdb.write(f"  trace {name}=0x{addr:x}\n")
        TraceBP(name, addr, limit)


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBP(spec)
    except Exception:
        pass

try:
    ExitBP()
except Exception:
    pass

gdb.write("trace_librknnc_regcmd_helper.gdb loaded\n")
end

run
