set pagination off
set breakpoint pending on
set confirm off

python
import gdb


PATTERNS = {
    "conv2_setup": bytes.fromhex("10 10 f0 00 00 00 01 02"),
    "conv2_k_half": bytes.fromhex("10 10 f0 00 00 40 01 02"),
    "conv2_k_tile": bytes.fromhex("10 10 f0 00 00 50 01 02"),
    "conv2_weight_size0": bytes.fromhex("30 10 00 10 0e 00 01 02"),
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


def is_ptr(value):
    return PTR_MIN <= value < PTR_MAX


def scan_buffer(label, addr, size, extra=""):
    if not is_ptr(addr) or size <= 0:
        return False
    size = min(size, 1 << 20)
    try:
        data = read_mem(addr, size)
    except Exception:
        return False
    found = False
    for name, pat in PATTERNS.items():
        off = data.find(pat)
        if off < 0:
            continue
        found = True
        gdb.write(
            f"\n[qword:{label}:{name}] buf=0x{addr:x} size={size} off=0x{off:x} "
            f"abs=0x{addr + off:x} {extra}\n"
        )
        start = max(0, off - 32)
        end = min(len(data), off + 96)
        gdb.write("  bytes " + data[start:end].hex(" ") + "\n")
        try:
            bt = gdb.execute("bt 12", to_string=True)
            for line in bt.splitlines():
                gdb.write("  " + line + "\n")
        except Exception as exc:
            gdb.write(f"  bt failed: {exc}\n")
    return found


class ScanBP(gdb.Breakpoint):
    def __init__(self, spec, kind):
        super().__init__(spec, internal=False)
        self.silent = True
        self.kind = kind
        self.hits = 0
        self.found = 0

    def stop(self):
        self.hits += 1
        hit = False
        if self.kind == "write":
            hit = scan_buffer("write", reg("x1"), reg("x2"), f"fd={reg('w0') & 0xffffffff}")
        elif self.kind == "fwrite":
            size = reg("x1") * reg("x2")
            hit = scan_buffer("fwrite", reg("x0"), size, f"size={reg('x1')} nmemb={reg('x2')}")
        elif self.kind == "ostream_write":
            hit = scan_buffer("ostream_write", reg("x1"), reg("x2"), f"ostream=0x{reg('x0'):x}")
        elif self.kind == "memcpy":
            hit = scan_buffer("memcpy_src", reg("x1"), reg("x2"), f"dst=0x{reg('x0'):x}")
        elif self.kind == "memmove":
            hit = scan_buffer("memmove_src", reg("x1"), reg("x2"), f"dst=0x{reg('x0'):x}")
        if hit:
            self.found += 1
        return False


class ExitBP(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace] process exit; quitting\n")
        gdb.execute("quit")
        return False


for spec, kind in (
    ("write", "write"),
    ("fwrite", "fwrite"),
    ("std::basic_ostream<char, std::char_traits<char> >::write(char const*, long)", "ostream_write"),
    ("memcpy", "memcpy"),
    ("memmove", "memmove"),
):
    try:
        ScanBP(spec, kind)
    except Exception as exc:
        gdb.write(f"breakpoint {spec} unavailable: {exc}\n")

try:
    ExitBP()
except Exception:
    pass

gdb.write("trace_librknnc_export_qwords.gdb loaded\n")
end

run
