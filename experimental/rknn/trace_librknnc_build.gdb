set pagination off
set breakpoint pending on
set confirm off

python
import gdb


# Offsets are for experimental/rknn/librknnc.so:
# sha256 d499753a91065f0b52b2cdfa43c073645bcc37467c8e077372b302ccafd5d53c
TRACE_POINTS = [
    ("param_extractor_mode_switch", 0x001cbc10, 80),
    ("split_planner", 0x0024c1d0, 24),
    ("top_tiler", 0x002539c0, 24),
    ("cna_bank_validator_a", 0x007d13dc, 64),
    ("cna_bank_validator_b", 0x007d1438, 64),
    ("reuse_update_a", 0x01531590, 80),
    ("reuse_update_b", 0x015477dc, 80),
    ("reuse_update_c", 0x01547820, 80),
    ("abc_or_channel_tile_emit_a", 0x015d0cc0, 80),
    ("abc_or_channel_tile_emit_b", 0x015d5ef0, 80),
    ("kc_or_alt_emit", 0x015e1b60, 80),
]


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
    best = None
    for line in out.splitlines():
        if "librknnc.so" not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            start = int(parts[0], 16)
        except ValueError:
            continue
        if best is None or start < best:
            best = start
    return best


def safe_words(addr, count=8):
    if addr == 0:
        return "null"
    vals = []
    for i in range(count):
        try:
            vals.append(f"+0x{i * 8:02x}=0x{u64(addr + i * 8):016x}")
        except Exception:
            if i == 0:
                return "unreadable"
            break
    return " ".join(vals)


def safe_dwords(addr, count=16):
    if addr == 0:
        return "null"
    vals = []
    for i in range(count):
        try:
            vals.append(f"{u32(addr + i * 4):08x}")
        except Exception:
            if i == 0:
                return "unreadable"
            break
    return " ".join(vals)


def dump_vector(prefix, holder, name):
    try:
        begin = u64(holder)
        end = u64(holder + 8)
        cap = u64(holder + 16)
    except Exception as exc:
        gdb.write(f"  {prefix} vec {name}@0x{holder:x}: unreadable {exc}\n")
        return
    span = max(0, end - begin)
    cap_span = max(0, cap - begin)
    gdb.write(
        f"  {prefix} vec {name}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} "
        f"cap=0x{cap:x} bytes={span} cap_bytes={cap_span} "
        f"n32={span // 4 if span % 4 == 0 else 'na'} "
        f"n64={span // 8 if span % 8 == 0 else 'na'}\n"
    )
    if begin and span:
        dwords = min(span // 4, 64) if span % 4 == 0 else 32
        qwords = min(span // 8, 32) if span % 8 == 0 else 16
        gdb.write(f"    d32 {safe_dwords(begin, dwords)}\n")
        gdb.write(f"    q64 {safe_words(begin, qwords)}\n")
        if name in ("x4", "x5", "x8") and span % 8 == 0:
            for i in range(min(span // 8, 6)):
                try:
                    ptr = u64(begin + i * 8)
                except Exception:
                    continue
                if 0x10000 <= ptr < 0x0001000000000000:
                    gdb.write(f"    ptr[{i}]@0x{ptr:x} d32 {safe_dwords(ptr, 24)}\n")
                    gdb.write(f"    ptr[{i}]@0x{ptr:x} q64 {safe_words(ptr, 12)}\n")


def dump_top_tiler_vectors(prefix, args):
    for name in ("x3", "x4", "x5", "x8"):
        dump_vector(prefix, args.get(name, 0), name)


def dump_regs():
    names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "lr"]
    return " ".join(f"{name}=0x{reg(name):x}" for name in names)


def capture_args():
    return {name: reg(name) for name in ("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "sp", "lr")}


def dump_arg_words(prefix, args, names=("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x8")):
    for name in names:
        ptr = args.get(name, 0)
        gdb.write(f"  {prefix} *{name}@0x{ptr:x}: {safe_words(ptr, 10)}\n")


class LibrknncTrace:
    installed = False
    base = None

    @classmethod
    def install(cls):
        if cls.installed:
            return
        base = find_librknnc_base()
        if base is None:
            return
        cls.base = base
        cls.installed = True
        gdb.write(f"librknnc base=0x{base:x}\n")
        for name, off, limit in TRACE_POINTS:
            addr = base + off
            gdb.write(f"trace {name}=0x{addr:x} limit={limit}\n")
            TraceBreakpoint(name, addr, limit)

    @classmethod
    def handle(cls, name, off, hit):
        pc = reg("pc")
        args = capture_args()
        gdb.write(f"\n[librknnc:{name} hit={hit} pc=0x{pc:x} off=0x{off:x}]\n")
        gdb.write("  " + dump_regs() + "\n")
        # Print first two object-like arguments as raw pointer-sized words. This
        # keeps the trace useful even before we know each C++ method signature.
        dump_arg_words("entry", args, ("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x8"))
        if name in ("split_planner", "top_tiler"):
            ReturnBreakpoint(name, args)
        try:
            bt = gdb.execute("bt 8", to_string=True)
            for line in bt.splitlines():
                gdb.write("  " + line + "\n")
        except Exception as exc:
            gdb.write(f"  bt failed: {exc}\n")


class ReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, name, args):
        self.name = name
        self.args = dict(args)
        super().__init__(f"*0x{args['lr']:x}", internal=False)
        self.silent = True

    def stop(self):
        gdb.write(f"\n[librknnc:{self.name}:return pc=0x{reg('pc'):x}]\n")
        gdb.write(f"  ret_x0=0x{reg('x0'):x} ret_x1=0x{reg('x1'):x} ret_w0=0x{reg('x0') & 0xffffffff:x}\n")
        dump_arg_words("return", self.args, ("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x8"))
        if self.name == "top_tiler":
            dump_top_tiler_vectors("return", self.args)
        self.enabled = False
        return False


class TraceBreakpoint(gdb.Breakpoint):
    def __init__(self, name, addr, limit):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.name = name
        self.off = addr - LibrknncTrace.base
        self.limit = limit
        self.hits = 0

    def stop(self):
        self.hits += 1
        if self.hits <= self.limit:
            LibrknncTrace.handle(self.name, self.off, self.hits)
        if self.hits == self.limit:
            gdb.write(f"[librknnc:{self.name}] hit limit reached, disabling breakpoint\n")
            self.enabled = False
        return False


class LoaderBreakpoint(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, internal=False)
        self.silent = True

    def stop(self):
        LibrknncTrace.install()
        return False


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBreakpoint(spec)
    except Exception as exc:
        gdb.write(f"loader breakpoint {spec} unavailable: {exc}\n")

gdb.write("trace_librknnc_build.gdb loaded; run an RKNN build process\n")
end

run
