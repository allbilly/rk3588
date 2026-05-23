set pagination off
set breakpoint pending on
set confirm off

python
import gdb


POINTS = [
    ("after_k_loop", 0x00257350, 20),
    ("before_alloc_sp5d0", 0x00257374, 10),
    ("after_alloc_sp5d0", 0x00257388, 10),
    ("before_alloc_sp5f0", 0x00257388, 10),
    ("after_alloc_sp5f0", 0x002573b4, 10),
    ("before_copy_sp630", 0x002574bc, 10),
    ("after_copy_sp630", 0x002574e0, 10),
    ("later_loop_setup", 0x002575a8, 30),
    ("later_loop_scan", 0x002576a0, 80),
    ("later_tile_read", 0x002577f0, 80),
    ("later_tile_qbuild", 0x0025788c, 80),
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


def dump_int_vector(label, holder, max_words=24):
    info = vector_info(holder)
    if info is None:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    begin, end, cap, span = info
    gdb.write(f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if begin and 0 < span <= max_words * 4:
        gdb.write(f"    d32 {d32_line(begin, max(1, span // 4))}\n")


def dump_vec24(label, holder, max_recs=8):
    info = vector_info(holder)
    if info is None:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    begin, end, cap, span = info
    gdb.write(f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if not begin or span <= 0:
        return
    if span % 24 == 0:
        for i in range(min(max_recs, span // 24)):
            rec = begin + i * 24
            try:
                rb, re, rc = u64(rec), u64(rec + 8), u64(rec + 16)
            except Exception:
                continue
            gdb.write(f"    rec24[{i}] begin=0x{rb:x} end=0x{re:x} cap=0x{rc:x} bytes={max(0, re-rb)}\n")
            if is_ptr(rb) and 0 <= re - rb <= 0x100:
                gdb.write(f"      rec24[{i}].d32 {d32_line(rb, max(1, (re-rb)//4))}\n")
    elif span <= 0x100:
        gdb.write(f"    raw d32 {d32_line(begin, max(1, span // 4))}\n")


def dump_stack(tag):
    sp = reg("sp")
    gdb.write(
        f"\n[{tag} pc=0x{reg('pc'):x} sp=0x{sp:x} "
        f"x0=0x{reg('x0'):x} x1=0x{reg('x1'):x} x2=0x{reg('x2'):x} "
        f"x3=0x{reg('x3'):x} x4=0x{reg('x4'):x} x22=0x{reg('x22'):x} "
        f"w1={reg('x1') & 0xffffffff} w2={reg('x2') & 0xffffffff} "
        f"w3={reg('x3') & 0xffffffff} w20={reg('x20') & 0xffffffff}]\n"
    )
    for label, off in (
        ("caller_x8_sp648", 0x648),
        ("vec_sp5b0", 0x5b0),
        ("vec_sp5d0", 0x5d0),
        ("vec_sp5f0", 0x5f0),
        ("vec_sp610", 0x610),
        ("vec_sp630", 0x630),
    ):
        dump_vec24(label, sp + off)
    # Raw vector<int> pointers that appear in the later loop.
    if is_ptr(reg("x22")):
        dump_vec24("x22_vec24", reg("x22"))
    if is_ptr(reg("x1")):
        dump_int_vector("x1_int_vec", reg("x1"))
    if is_ptr(reg("x2")):
        dump_int_vector("x2_int_vec", reg("x2"))


class ExitBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace_librknnc_height_boundary] process called exit; quitting gdb\n")
        gdb.execute("quit")
        return False


class TraceBreakpoint(gdb.Breakpoint):
    def __init__(self, name, addr, limit):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.name = name
        self.limit = limit
        self.hits = 0

    def stop(self):
        self.hits += 1
        dump_stack(f"{self.name} hit={self.hits}")
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
    gdb.write("librknnc base=0x%x\n" % base)
    for name, off, limit in POINTS:
        gdb.write(f"trace {name}=0x{base + off:x} limit={limit}\n")
        TraceBreakpoint(name, base + off, limit)


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBreakpoint(spec)
    except Exception:
        pass

try:
    ExitBreakpoint()
except Exception:
    pass

gdb.write("trace_librknnc_height_boundary.gdb loaded\n")
end

run
