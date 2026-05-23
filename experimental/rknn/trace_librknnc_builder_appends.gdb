set pagination off
set breakpoint pending on
set confirm off

python
import gdb


# Offsets are for experimental/rknn/librknnc.so:
# sha256 d499753a91065f0b52b2cdfa43c073645bcc37467c8e077372b302ccafd5d53c
BUILDER_START = 0x01549c44
BUILDER_END = 0x0154b0d4

HELPERS = [
    ("align4", 0x01537840),
    ("append_indexed_u32", 0x01537e20),
    ("append_byte_tag", 0x01561100),
    ("append_u32", 0x015611f0),
    ("append_bytes", 0x015612e0),
    ("append_indexed_u32_b", 0x015615c0),
    ("append_varint_or_size", 0x01561fa0),
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


def u32(addr):
    return int.from_bytes(read_mem(addr, 4), "little")


def u16(addr):
    return int.from_bytes(read_mem(addr, 2), "little")


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


def stage_fields(obj):
    return {
        "obj": obj,
        "used": u64(obj + 48),
        "cursor": u64(obj + 64),
        "record_ptr": u64(obj + 72),
        "record_count": u32(obj + 80),
        "max_tag": u16(obj + 84),
        "base_used": u64(obj + 88),
        "dirty": read_mem(obj + 96, 1)[0],
        "capacity": u64(obj + 104),
        "gate": read_mem(obj + 112, 1)[0],
    }


def scan_addr(label, addr, size, extra):
    if not is_ptr(addr) or size <= 0:
        return False
    size = min(size, 2 << 20)
    try:
        data = read_mem(addr, size)
    except Exception:
        return False
    hit = False
    for name, pat in PATTERNS.items():
        off = data.find(pat)
        if off >= 0:
            hit = True
            gdb.write(
                f"  qword {label}:{name}: addr=0x{addr:x} size={size} "
                f"off=0x{off:x} abs=0x{addr + off:x} {extra}\n"
            )
    return hit


def scan_stage(label, obj, extra=""):
    if not is_ptr(obj):
        return False
    try:
        f = stage_fields(obj)
    except Exception:
        return False
    hit = False
    # The writer grows a payload in a malloc region and keeps a reverse cursor.
    # Try all plausible anchors; the correct one is easy to identify by offset.
    candidates = [
        ("cursor", f["cursor"], f["used"] + 0x1000),
        ("cursor_minus_used", f["cursor"] - f["used"], f["used"] + 0x1000),
        ("record_ptr", f["record_ptr"], f["used"] + 0x1000),
        ("record_ptr_minus_used", f["record_ptr"] - f["used"], f["used"] + 0x1000),
    ]
    for cname, addr, size in candidates:
        if scan_addr(label + "." + cname, addr, size, extra):
            hit = True
    if hit:
        gdb.write(
            f"  stage obj=0x{obj:x} used=0x{f['used']:x} cursor=0x{f['cursor']:x} "
            f"record_ptr=0x{f['record_ptr']:x} count={f['record_count']} "
            f"max_tag=0x{f['max_tag']:x} base_used=0x{f['base_used']:x} "
            f"dirty={f['dirty']} cap={f['capacity']} gate={f['gate']} {extra}\n"
        )
    return hit


class RetBP(gdb.Breakpoint):
    def __init__(self, addr, name, obj, callsite):
        super().__init__(f"*0x{addr:x}", internal=True)
        self.silent = True
        self.name = name
        self.obj = obj
        self.callsite = callsite

    def stop(self):
        if scan_stage(
            "after_" + self.name,
            self.obj,
            f"callsite=0x{self.callsite:x} ret_pc=0x{reg('pc'):x} x0=0x{reg('x0'):x}",
        ):
            gdb.write(f"[append-hit] {self.name} returned from builder callsite 0x{self.callsite:x}\n")
            bt = gdb.execute("bt 8", to_string=True)
            for line in bt.splitlines():
                gdb.write("  " + line + "\n")
        self.enabled = False
        return False


class HelperBP(gdb.Breakpoint):
    def __init__(self, name, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.silent = True
        self.name = name
        self.hits = 0

    def stop(self):
        self.hits += 1
        lr = reg("lr")
        if not (BASE + BUILDER_START <= lr <= BASE + BUILDER_END):
            return False
        obj = reg("x0")
        callsite = lr - 4
        if self.hits < 2000:
            RetBP(lr, self.name, obj, callsite)
        return False


class ExactMemcpyBP(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.silent = True

    def stop(self):
        gdb.write(
            f"\n[payload_memcpy_call pc=0x{reg('pc'):x}] dst=0x{reg('x0'):x} "
            f"src=0x{reg('x1'):x} size=0x{reg('x2'):x} x22=0x{reg('x22'):x}\n"
        )
        scan_addr("payload_src", reg("x1"), reg("x2"), "exact memcpy src")
        scan_stage("stage_at_payload_memcpy", reg("x22"), "exact memcpy x22")
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


BASE = None
installed = False


def install():
    global installed, BASE
    if installed:
        return
    BASE = find_librknnc_base()
    if BASE is None:
        return
    installed = True
    gdb.write(f"librknnc base=0x{BASE:x}\n")
    for name, off in HELPERS:
        gdb.write(f"  helper {name}=0x{BASE + off:x}\n")
        HelperBP(name, BASE + off)
    gdb.write(f"  payload_memcpy=0x{BASE + 0x0154aee4:x}\n")
    ExactMemcpyBP(BASE + 0x0154aee4)


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBP(spec)
    except Exception:
        pass

try:
    ExitBP()
except Exception:
    pass

gdb.write("trace_librknnc_builder_appends.gdb loaded\n")
end

run
