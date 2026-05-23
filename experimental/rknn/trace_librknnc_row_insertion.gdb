set pagination off
set breakpoint pending on
set confirm off

python
import gdb

# Trace points after row_final_stage (0x257ef8) to find tile table insertion
POINTS = [
    ("row_final_stage", 0x00257ef8, 80),
    ("post_stage_1", 0x00257f00, 80),
    ("post_stage_2", 0x00257f20, 80),
    ("post_stage_3", 0x00257f40, 80),
    ("post_stage_4", 0x00257f60, 80),
    ("post_stage_5", 0x00257f80, 80),
    ("post_stage_6", 0x00257fa0, 80),
    ("post_stage_7", 0x00257fc0, 80),
    ("post_stage_8", 0x00257fe0, 80),
    ("post_stage_9", 0x00258000, 80),
    ("post_stage_10", 0x00258020, 80),
    ("post_stage_11", 0x00258040, 80),
    ("post_stage_12", 0x00258060, 80),
    ("post_stage_13", 0x00258080, 80),
    ("post_stage_14", 0x002580a0, 80),
    ("post_stage_15", 0x002580c0, 80),
    ("post_stage_16", 0x002580e0, 80),
    ("post_stage_17", 0x00258100, 80),
    ("post_stage_18", 0x00258120, 80),
    ("post_stage_19", 0x00258140, 80),
    ("post_stage_20", 0x00258160, 80),
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


def vector_info(holder):
    try:
        begin = u64(holder)
        end = u64(holder + 8)
        cap = u64(holder + 16)
    except Exception:
        return None
    return begin, end, cap, max(0, end - begin)


def dump_vector(label, holder, max_words=24):
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
    if begin and span % 24 == 0:
        for i in range(min(max_recs, span // 24)):
            rec = begin + i * 24
            rb, re, rc = u64(rec), u64(rec + 8), u64(rec + 16)
            gdb.write(f"    rec24[{i}] begin=0x{rb:x} end=0x{re:x} cap=0x{rc:x} bytes={max(0, re-rb)}\n")
            if is_ptr(rb) and 0 <= re - rb <= 0x100:
                gdb.write(f"      d32 {d32_line(rb, max(1, (re-rb)//4))}\n")


def dump_tile_table(label, holder, max_recs=32):
    """Dump a potential tile table as a vector of records."""
    info = vector_info(holder)
    if info is None:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    begin, end, cap, span = info
    gdb.write(f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if begin and span > 0:
        rec_size = span // max(1, span // 32)  # Try to guess record size
        if span % rec_size == 0:
            for i in range(min(max_recs, span // rec_size)):
                rec = begin + i * rec_size
                gdb.write(f"    tile[{i}] d32 {d32_line(rec, min(16, rec_size))}\n")


def dump_context(tag):
    sp = reg("sp")
    gdb.write(
        f"\n[{tag} pc=0x{reg('pc'):x} sp=0x{sp:x} "
        f"x0=0x{reg('x0'):x} x1=0x{reg('x1'):x} x2=0x{reg('x2'):x} "
        f"x3=0x{reg('x3'):x} x4=0x{reg('x4'):x} x5=0x{reg('x5'):x} "
        f"x7=0x{reg('x7'):x} x22=0x{reg('x22'):x} "
        f"w0={reg('x0') & 0xffffffff} w1={reg('x1') & 0xffffffff} "
        f"w2={reg('x2') & 0xffffffff} w3={reg('x3') & 0xffffffff} "
        f"w19={reg('x19') & 0xffffffff} w21={reg('x21') & 0xffffffff} "
        f"w23={reg('x23') & 0xffffffff} w24={reg('x24') & 0xffffffff}]\n"
    )
    # Key stack locations from previous traces
    gdb.write(f"  staged_sp480 d32 {d32_line(sp + 0x1e0, 64)}\n")
    gdb.write(f"  row_header_sp650 d32 {d32_line(sp + 0x650, 32)}\n")
    gdb.write(f"  row_obj_sp6c0 d32 {d32_line(sp + 0x6c0, 96)}\n")
    # Look for tile table holders at various stack offsets
    dump_vec24("y_boundary_sp648", sp + 0x648, 4)
    dump_vector("row_vec_sp6c0", sp + 0x6c0)
    dump_vector("row_vec_sp6d0", sp + 0x6d0)
    dump_vector("row_vec_sp6e0", sp + 0x6e0)
    dump_vector("row_vec_sp700", sp + 0x700)
    dump_vector("row_vec_sp718", sp + 0x718)
    dump_vector("row_vec_sp760", sp + 0x760)
    # New: look for tile table at sp+0x780, sp+0x7a0, sp+0x7c0
    dump_vector("tile_table_sp780", sp + 0x780)
    dump_vector("tile_table_sp7a0", sp + 0x7a0)
    dump_vector("tile_table_sp7c0", sp + 0x7c0)
    dump_vector("tile_table_sp7e0", sp + 0x7e0)
    if is_ptr(reg("x22")):
        dump_vec24("x22_vec24", reg("x22"), 4)
    # Follow x0 as potential tile table pointer
    if is_ptr(reg("x0")):
        dump_vec24("x0_vec24", reg("x0"), 4)
    # Follow x1 as potential row object pointer
    if is_ptr(reg("x1")):
        gdb.write(f"  x1_obj d32 {d32_line(reg('x1'), 32)}\n")


class ExitBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace_librknnc_row_insertion] process called exit; quitting gdb\n")
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

gdb.write("trace_librknnc_row_insertion.gdb loaded\n")
end

run
