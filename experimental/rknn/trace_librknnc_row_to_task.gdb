set pagination off
set breakpoint pending on
set confirm off

python
import gdb


TRACE_POINTS = [
    ("counts_sum_start", 0x00634748, 60),
    ("count_bucket_select", 0x00634820, 80),
    ("row_base_selected", 0x00634840, 80),
    ("row_shape_filter", 0x00634874, 80),
    ("row_mode_filter", 0x00634884, 80),
    ("offset_inputs", 0x006348b8, 80),
    ("row_flag_check", 0x006348c8, 80),
    ("offset0_ready", 0x006348f0, 80),
    ("offset1_ready", 0x006349c4, 80),
    ("task_desc_begin", 0x00634c40, 80),
    ("task_desc_fields", 0x00634c70, 80),
    ("task_desc_vectors", 0x00634ce4, 80),
    ("task_desc_helper_call", 0x00636574, 80),
    ("task_desc_push16", 0x006358d8, 80),
    ("existing_row_copy", 0x00635fe8, 80),
    ("tile_helper_entry", 0x005aea40, 80),
    ("tile_helper_after_mode", 0x005aeabc, 80),
    ("tile_helper_dims_ready", 0x005aeb30, 80),
    ("tile_helper_emit_call", 0x005aece4, 80),
    ("tile_helper_publish16", 0x005aeda8, 80),
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


def dump_sp_word(sp, off):
    try:
        gdb.write(f"  sp+0x{off:x}: u32=0x{u32(sp + off):08x} u64=0x{u64(sp + off):016x}\n")
    except Exception:
        gdb.write(f"  sp+0x{off:x}: unreadable\n")


def dump_vec(label, holder, max_bytes=96):
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


def dump_counts(sp):
    try:
        ptr = u64(sp + 384)
        end = u64(sp + 392)
    except Exception:
        gdb.write("  counts: unreadable\n")
        return
    span = max(0, end - ptr)
    gdb.write(f"  counts@sp+0x180: begin=0x{ptr:x} end=0x{end:x} bytes={span}\n")
    if is_ptr(ptr) and 0 < span <= 128:
        gdb.write(f"    d32 {d32_line(ptr, max(1, span // 4))}\n")


def dump_row(label, addr):
    if not is_ptr(addr):
        gdb.write(f"  {label}=0x{addr:x} not-ptr\n")
        return
    gdb.write(f"  {label}=0x{addr:x}\n")
    gdb.write(f"    header d32 {d32_line(addr, 16)}\n")
    for off in (0x30, 0x40, 0x58, 0x70, 0xa0, 0xb8, 0xd0, 0x100, 0x118, 0x130):
        dump_vec(f"{label}+0x{off:x}", addr + off, 64)
    gdb.write(f"    scalars+0x148 d32 {d32_line(addr + 0x148, 28)}\n")


def dump_task(sp):
    base = sp + 0x4c0
    gdb.write(f"  task_desc@sp+0x4c0 d32 {d32_line(base, 32)}\n")
    for off in (0x40, 0x58, 0x70, 0x88, 0xa0, 0xb8, 0xd0, 0xe8):
        dump_vec(f"task+0x{off:x}", base + off, 64)


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


class RowToTaskTrace:
    @staticmethod
    def handle(name, hit):
        sp = reg("sp")
        gdb.write(f"\n[{name} hit={hit} pc=0x{reg('pc'):x} sp=0x{sp:x} lr=0x{reg('lr'):x}]\n")
        gdb.write(
            "  regs "
            f"x19=0x{reg('x19'):x} x20=0x{reg('x20'):x} x21=0x{reg('x21'):x} "
            f"x22=0x{reg('x22'):x} x25=0x{reg('x25'):x} x28=0x{reg('x28'):x} "
            f"w0={reg('w0') & 0xffffffff} w1={reg('w1') & 0xffffffff} "
            f"w2={reg('w2') & 0xffffffff} w3={reg('w3') & 0xffffffff} "
            f"w20={reg('w20') & 0xffffffff} w21={reg('w21') & 0xffffffff} "
            f"w22={reg('w22') & 0xffffffff} w28={reg('w28') & 0xffffffff}\n"
        )
        if name.startswith("tile_helper"):
            RowToTaskTrace.handle_tile_helper(name)
            return
        for off in (0xd0, 0xec, 0xf0, 0x100, 0x118, 0x134, 0x138, 0x13c,
                    0x140, 0x148, 0x150, 0x160, 0x164, 0x1c8, 0x1cc,
                    0x1ec, 0x1f8, 0x1fc, 0x2d0, 0x2d4, 0x2d8, 0x2dc,
                    0x378, 0x450, 0x458):
            dump_sp_word(sp, off)
        dump_counts(sp)
        dump_row("row_x20", reg("x20"))
        dump_row("row_x25", reg("x25"))
        dump_task(sp)
        if name == "task_desc_helper_call":
            dump_row("helper_arg_task_x4", reg("x4"))
        if name == "task_desc_push16":
            try:
                dest = u64(reg("x20") + 8)
                gdb.write(f"  push16_dest=0x{dest:x} d32 {d32_line(dest, 4)}\n")
            except Exception:
                pass

    @staticmethod
    def handle_tile_helper(name):
        sp = reg("sp")
        gdb.write(
            "  helper_regs "
            f"x21_task=0x{reg('x21'):x} x22_arg2=0x{reg('x22'):x} x23_arg0=0x{reg('x23'):x} "
            f"x24_arg3=0x{reg('x24'):x} w19={reg('w19') & 0xffffffff} "
            f"w21={reg('w21') & 0xffffffff} w25={reg('w25') & 0xffffffff} "
            f"w27={reg('w27') & 0xffffffff} w28={reg('w28') & 0xffffffff}\n"
        )
        dump_row("helper_task_x21", reg("x21"))
        for off in (0xa4, 0xc8, 0xcc, 0xd0, 0xd8, 0xe0, 0xf0, 0x100, 0x120,
                    0x130, 0x138):
            dump_sp_word(sp, off)
        if name in ("tile_helper_emit_call", "tile_helper_publish16"):
            gdb.write(f"  helper_vec_sp100 d32 {d32_line(sp + 0x100, 16)}\n")
            gdb.write(f"  helper_vec_sp120 d32 {d32_line(sp + 0x120, 16)}\n")
            try:
                dest_holder = u64(sp + 168)
                gdb.write(f"  helper_publish_holder=0x{dest_holder:x}\n")
                if is_ptr(dest_holder):
                    dump_vec("helper_publish_holder+0x50", dest_holder + 0x50, 128)
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
        RowToTaskTrace.handle(self.name, self.hits)
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

gdb.write("trace_librknnc_row_to_task.gdb loaded\n")
end

run
