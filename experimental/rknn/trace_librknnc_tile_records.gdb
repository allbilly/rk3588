set pagination off
set breakpoint pending on
set confirm off

python
import gdb

# Key offsets from disassembly analysis
RECORD_WRITE = 0x00257f00    # stp w0, w23, [x7, #-8]; stp w28, w24, [x7]
VEC_ASSIGN = 0x00257f80      # subs x19, x0, x1 ; check for zero then alloc
VEC_ALLOC_DONE = 0x00257fac  # str x3, [sp, #2128] — output vector begin stored
VEC_ALLOC2_DONE = 0x0025801c # str x3, [sp, #2152] — second output vector

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


def dump_vec(label, holder, max_words=24):
    info = vector_info(holder)
    if info is None:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    begin, end, cap, span = info
    gdb.write(f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if begin and 0 < span <= max_words * 4:
        gdb.write(f"    d32 {d32_line(begin, max(1, span // 4))}\n")


# Track output vectors across hits
class RecordTrace:
    outputs = {}  # stack_sp -> list of (hit, output_vector_addr)

    @classmethod
    def handle_record_write(cls):
        sp = reg("sp")
        x7 = reg("x7")
        w0 = reg("x0") & 0xffffffff
        w23 = reg("x23") & 0xffffffff
        w28 = reg("x28") & 0xffffffff
        w24 = reg("x24") & 0xffffffff
        gdb.write(f"\n[RECORD_WRITE sp=0x{sp:x} x7=0x{x7:x} w0={w0} w23={w23} w28={w28} w24={w24}]\n")
        gdb.write(f"  Record at x7-8=(w0={w0}, w23={w23}) at x7=(w28={w28}, w24={w24})\n")
        # Dump record from stack
        if is_ptr(x7):
            gdb.write(f"  record[0..3] d32 {d32_line(x7 - 8, 4)}\n")
        # Also dump vec allocations at sp+2128, sp+2152
        dump_vec("output_vec_at_sp+2128", sp + 0x850)
        dump_vec("output_vec_at_sp+2152", sp + 0x868)

    @classmethod
    def handle_vec_alloc_done(cls):
        sp = reg("sp")
        x3 = reg("x3")
        gdb.write(f"\n[VEC_ALLOC_DONE sp=0x{sp:x} new_begin=0x{x3:x}]\n")
        dump_vec("output_vec_at_sp+2128", sp + 0x850)
        dump_vec("output_vec_at_sp+2152", sp + 0x868)
        dump_vec("output_vec_at_sp+2168", sp + 0x878)

    @classmethod
    def handle_vec_alloc2_done(cls):
        sp = reg("sp")
        x3 = reg("x3")
        gdb.write(f"\n[VEC_ALLOC2_DONE sp=0x{sp:x} new_begin=0x{x3:x}]\n")
        dump_vec("output_vec_at_sp+2128", sp + 0x850)
        dump_vec("output_vec_at_sp+2152", sp + 0x868)
        dump_vec("output_vec_at_sp+2168", sp + 0x878)


class ExitBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True
    def stop(self):
        gdb.write("[trace] process exit; quitting\n")
        gdb.execute("quit")
        return False


class TracerBP(gdb.Breakpoint):
    def __init__(self, name, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.name = name
        self.hits = 0
    def stop(self):
        self.hits += 1
        getattr(RecordTrace, f"handle_{self.name}")()
        if self.hits >= 400:
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
    for name, off in [
        ("record_write", RECORD_WRITE),
        ("vec_alloc_done", VEC_ALLOC_DONE),
        ("vec_alloc2_done", VEC_ALLOC2_DONE),
    ]:
        addr = base + off
        gdb.write(f"  trace {name}=0x{addr:x}\n")
        TracerBP(name, addr)


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBP(spec)
    except Exception:
        pass
try:
    ExitBreakpoint()
except Exception:
    pass
gdb.write("trace_librknnc_tile_records.gdb loaded\n")
end

run
