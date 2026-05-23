set pagination off
set breakpoint pending on
set confirm off

python
import gdb

BUILDER_ENTRY = 0x01549c44
PAYLOAD_MEMCPY = 0x0154aee4

QW_SETUP = bytes.fromhex("10 10 f0 00 00 00 01 02")
QW_K_HALF = bytes.fromhex("10 10 f0 00 00 40 01 02")
QW_K_TILE = bytes.fromhex("10 10 f0 00 00 50 01 02")
PTR_MIN = 0x10000
PTR_MAX = 0x0001000000000000

def reg(name):
    try:
        return int(gdb.parse_and_eval("$" + name))
    except:
        return 0

def read_mem(addr, size):
    return gdb.selected_inferior().read_memory(addr, size).tobytes()

def u64(addr):
    return int.from_bytes(read_mem(addr, 8), "little")

def is_ptr(v):
    return PTR_MIN <= v < PTR_MAX

def find_base():
    out = gdb.execute("info proc mappings", to_string=True)
    for line in out.splitlines():
        if "librknnc.so" not in line:
            continue
        try:
            return int(line.split()[0], 16)
        except:
            pass
    return None

def scan_qwords(label, addr, size, extra=""):
    if not is_ptr(addr) or size <= 0:
        return False
    size = min(size, 2 << 20)
    try:
        data = read_mem(addr, size)
    except:
        return False
    hit = False
    for name, pat in [("setup", QW_SETUP), ("k_half", QW_K_HALF), ("k_tile", QW_K_TILE)]:
        off = data.find(pat)
        if off >= 0:
            hit = True
            gdb.write(f"  qword {label}:{name}: addr=0x{addr:x} off=0x{off:x} abs=0x{addr+off:x} {extra}\n")
    return hit

def dump_writer(label, obj, extra=""):
    if not is_ptr(obj):
        gdb.write(f"  {label}: obj=0x{obj:x} not-ptr {extra}\n")
        return
    used = u64(obj + 0x30)
    cursor = u64(obj + 0x40)
    gdb.write(f"  {label}: obj=0x{obj:x} used=0x{used:x} cursor=0x{cursor:x} {extra}\n")
    if is_ptr(cursor):
        scan_qwords(f"{label}.cursor", cursor, used + 0x100, "")

class BuilderEntryBP(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)

    def stop(self):
        gdb.write(f"\n[builder_entry pc=0x{reg('pc'):x} lr=0x{reg('lr'):x}]\n")
        x22 = reg("x22")
        gdb.write(f"  x22=0x{x22:x} x0=0x{reg('x0'):x} x1=0x{reg('x1'):x}\n")
        bt = gdb.execute("bt 8", to_string=True)
        for l in bt.splitlines():
            gdb.write("  " + l + "\n")
        dump_writer("entry", x22, "x22 at entry")
        try:
            for off, name in [(0x108, "vec108"), (0x188, "vec188"), (0x1f0, "ptr1f0")]:
                p = u64(x22 + off)
                p2 = u64(x22 + off + 8)
                gdb.write(f"  comp+0x{off:x} ({name}): 0x{p:x} 0x{p2:x}\n")
        except:
            pass
        return False

class AfterFirstAppendBP(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.silent = True
        self._hits = 0

    def stop(self):
        self._hits += 1
        if self._hits > 5:
            self.enabled = False
            return False
        x22 = reg("x22")
        gdb.write(f"\n[after_first_append pc=0x{reg('pc'):x}]\n")
        dump_writer("after_first_append", x22, "x22 after first helper")
        return False

class PayloadMemcpyBP(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.silent = True

    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.silent = True
        self._hits = 0

    def stop(self):
        self._hits += 1
        if self._hits > 5:
            self.enabled = False
            return False
        gdb.write(f"\n[payload_memcpy pc=0x{reg('pc'):x}] dst=0x{reg('x0'):x} src=0x{reg('x1'):x} size=0x{reg('x2'):x}\n")
        gdb.execute("bt 8")
        scan_qwords("payload_src", reg("x1"), reg("x2"), "")
        scan_qwords("payload_dst", reg("x0"), reg("x2"), "")
        return False


_installed = False

class LoaderBP(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, internal=False)
        self.silent = True

    def stop(self):
        global _installed
        if _installed:
            return False
        base = find_base()
        if base is None:
            return False
        _installed = True
        gdb.write(f"librknnc base=0x{base:x}\n")
        BuilderEntryBP(base + BUILDER_ENTRY)
        AfterFirstAppendBP(base + 0x1549d60)
        PayloadMemcpyBP(base + PAYLOAD_MEMCPY)
        gdb.write("breakpoints installed\n")
        return False


class ExitBP(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        gdb.write("[trace] process exit; quitting\n")
        gdb.execute("quit")
        return False


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBP(spec)
    except:
        pass

try:
    ExitBP()
except:
    pass

gdb.write("trace_librknnc_staging_source.gdb loaded\n")
end

run
