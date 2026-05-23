set pagination off
set breakpoint pending on
set confirm off

python
import gdb


# Offsets are for experimental/rknn/librknnc.so:
# sha256 d499753a91065f0b52b2cdfa43c073645bcc37467c8e077372b302ccafd5d53c
TRACE_POINTS = [
    ("builder_entry", 0x01549c44, 16),
    ("payload_copy_before", 0x0154aed0, 32),
    ("payload_memcpy_call", 0x0154aee4, 32),
    ("payload_copy_after", 0x0154aee8, 32),
    ("wrapper_copy_before", 0x0154afe8, 16),
    ("wrapper_copy_after", 0x0154b03c, 16),
    ("builder_return", 0x0154b0d0, 16),
    ("caller_after_build", 0x0154b784, 16),
    ("caller_before_write", 0x0154b7b4, 16),
]

PATTERNS = {
    "conv2_setup": bytes.fromhex("10 10 f0 00 00 00 01 02"),
    "conv2_k_half": bytes.fromhex("10 10 f0 00 00 40 01 02"),
    "conv2_k_tile": bytes.fromhex("10 10 f0 00 00 50 01 02"),
}

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


def is_ptr(value):
    return PTR_MIN <= value < PTR_MAX


def scan_buffer(label, addr, size, extra=""):
    if not is_ptr(addr) or size <= 0:
        return False
    size = min(size, 2 << 20)
    try:
        data = read_mem(addr, size)
    except Exception as exc:
        gdb.write(f"  scan {label}: unreadable addr=0x{addr:x} size={size} {exc}\n")
        return False
    found = False
    for name, pat in PATTERNS.items():
        off = data.find(pat)
        if off < 0:
            continue
        found = True
        gdb.write(
            f"  qword {label}:{name}: buf=0x{addr:x} size={size} off=0x{off:x} "
            f"abs=0x{addr + off:x} {extra}\n"
        )
        start = max(0, off - 32)
        end = min(len(data), off + 96)
        gdb.write("    bytes " + data[start:end].hex(" ") + "\n")
    return found


def dump_qwords(label, addr, count):
    if not is_ptr(addr):
        gdb.write(f"  {label}=0x{addr:x} not-ptr\n")
        return
    vals = []
    for i in range(count):
        try:
            vals.append(f"{u64(addr + i * 8):016x}")
        except Exception:
            break
    gdb.write(f"  {label}@0x{addr:x}: {' '.join(vals) if vals else 'unreadable'}\n")


def dump_vec(label, holder):
    try:
        begin = u64(holder)
        end = u64(holder + 8)
        cap = u64(holder + 16)
    except Exception:
        gdb.write(f"  {label}@0x{holder:x}: unreadable\n")
        return
    gdb.write(
        f"  {label}@0x{holder:x}: begin=0x{begin:x} end=0x{end:x} "
        f"cap=0x{cap:x} bytes={max(0, end - begin)}\n"
    )


def dump_export_obj(label, obj):
    if not is_ptr(obj):
        gdb.write(f"  {label}=0x{obj:x} not-ptr\n")
        return
    gdb.write(f"  {label}=0x{obj:x}\n")
    dump_qwords(label + "+0x00", obj, 8)
    try:
        base = u64(obj + 8)
        size = u64(obj + 16)
        off = u64(obj + 40)
    except Exception:
        return
    gdb.write(f"    base=0x{base:x} size={size} payload_off=0x{off:x} payload=0x{base + off:x}\n")
    scan_buffer(label + ".base", base, size + off if size < (2 << 20) else size, "")
    scan_buffer(label + ".payload", base + off, size, "")


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
            f"  x0=0x{reg('x0'):x} x1=0x{reg('x1'):x} x2=0x{reg('x2'):x} "
            f"x3=0x{reg('x3'):x} x4=0x{reg('x4'):x} x7=0x{reg('x7'):x} "
            f"x19=0x{reg('x19'):x} x20=0x{reg('x20'):x} x21=0x{reg('x21'):x} "
            f"x22=0x{reg('x22'):x} x23=0x{reg('x23'):x} x24=0x{reg('x24'):x} "
            f"w0=0x{reg('w0') & 0xffffffff:x}\n"
        )
        if self.name == "builder_entry":
            dump_vec("compiler+0x108", reg("x0") + 0x108)
            dump_vec("compiler+0x188", reg("x0") + 0x188)
            dump_vec("compiler+0x1f0", reg("x0") + 0x1f0)
        elif self.name == "payload_copy_before":
            export_holder = u64(reg("sp") + 120)
            export_obj = u64(export_holder)
            dump_export_obj("export_obj_before_payload", export_obj)
            scan_buffer("staging_before_payload", reg("x1"), reg("x2"), "memcpy src")
        elif self.name == "payload_memcpy_call":
            export_holder = u64(reg("sp") + 120)
            export_obj = u64(export_holder)
            dump_export_obj("export_obj_at_payload_memcpy", export_obj)
            scan_buffer("payload_memcpy_src", reg("x1"), reg("x2"), "memcpy src")
            scan_buffer("payload_memcpy_dst", reg("x0"), reg("x2"), "memcpy dst")
        elif self.name == "payload_copy_after":
            export_holder = u64(reg("sp") + 120)
            export_obj = u64(export_holder)
            dump_export_obj("export_obj_after_payload", export_obj)
        elif self.name == "wrapper_copy_before":
            export_holder = u64(reg("sp") + 120)
            export_obj = u64(export_holder)
            dump_export_obj("export_obj_before_wrapper", export_obj)
        elif self.name == "wrapper_copy_after":
            export_holder = u64(reg("sp") + 120)
            export_obj = u64(export_holder)
            dump_export_obj("export_obj_after_wrapper", export_obj)
        elif self.name in ("caller_after_build", "caller_before_write"):
            dump_export_obj("caller_export_obj", u64(sp + 72))
        elif self.name == "builder_return":
            export_holder = u64(reg("sp") + 120)
            export_obj = u64(export_holder)
            dump_export_obj("export_obj_at_return", export_obj)
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

gdb.write("trace_librknnc_export_builder.gdb loaded\n")
end

run
