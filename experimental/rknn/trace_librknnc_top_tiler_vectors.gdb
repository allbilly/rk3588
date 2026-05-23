set pagination off
set breakpoint pending on
set confirm off

python
import gdb

TOP_TILER_OFF = 0x002539c0
HIT_LIMIT = 3


def reg(name):
    try:
        return int(gdb.parse_and_eval("$" + name))
    except Exception:
        return 0


def u64(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 8).tobytes(), "little")


def u32(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 4).tobytes(), "little")


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


def dump_vector(name, holder):
    try:
        begin = u64(holder)
        end = u64(holder + 8)
        cap = u64(holder + 16)
    except Exception as exc:
        gdb.write(f"  vec {name}@0x{holder:x}: unreadable {exc}\n")
        return
    span = max(0, end - begin)
    gdb.write(f"  vec {name}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} cap=0x{cap:x} bytes={span}\n")
    if begin and span:
        gdb.write(f"    data {d32_line(begin, min(64, max(1, span // 4)))}\n")
    if name == "x8" and begin and span >= 8:
        for i in range(min(3, span // 8)):
            try:
                ptr = u64(begin + i * 8)
            except Exception:
                continue
            if 0x10000 <= ptr < 0x0001000000000000:
                gdb.write(f"    ptr[{i}]=0x{ptr:x} data {d32_line(ptr, 40)}\n")


class ReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, hit_id, args):
        self.hit_id = hit_id
        self.args = args
        super().__init__(f"*0x{args['lr']:x}", internal=True)
        self.silent = True

    def stop(self):
        gdb.write(f"\n[top_tiler_return hit={self.hit_id} ret_x0=0x{reg('x0'):x}]\n")
        gdb.write(f"  dims_x6 {d32_line(self.args['x6'], 20)}\n")
        for name in ("x3", "x4", "x5", "x8"):
            dump_vector(name, self.args[name])
        self.enabled = False
        return False


class TopTilerBreakpoint(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.hits = 0

    def stop(self):
        self.hits += 1
        args = {name: reg(name) for name in ("x3", "x4", "x5", "x6", "x8", "lr")}
        gdb.write(f"\n[top_tiler_entry hit={self.hits} x6=0x{args['x6']:x} lr=0x{args['lr']:x}]\n")
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

gdb.write("trace_librknnc_top_tiler_vectors.gdb loaded\n")
end

run
