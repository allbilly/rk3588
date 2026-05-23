set pagination off
set breakpoint pending on
set confirm off

python
import gdb

# Only param_extractor mode switch
PARAM_EXTRACTOR = 0x001cbc10

def reg(name):
    try:
        return int(gdb.parse_and_eval("$" + name))
    except Exception:
        return 0

def u64(addr):
    return int.from_bytes(gdb.selected_inferior().read_memory(addr, 8).tobytes(), "little")

def u32(addr):
    return int.from_bytes(gdb.selected_inferior().read_memory(addr, 4).tobytes(), "little")

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

class ParamExtractorBP(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.hits = 0
    def stop(self):
        self.hits += 1
        x0 = reg("x0")
        x1 = reg("x1")
        x2 = reg("x2")
        x3 = reg("x3")
        sp = reg("sp")
        mode = x1 & 0xffffffff
        gdb.write(f"\n[PARAM_EXTRACTOR hit={self.hits} mode=0x{mode:x}({mode})]\n")
        gdb.write(f"  x0=0x{x0:x} x1=0x{x1:x} x2=0x{x2:x} x3=0x{x3:x} sp=0x{sp:x}\n")
        # Try to read conv-type identification from x0 (context pointer)
        if x0:
            try:
                gdb.write(f"  x0[0..7] q64: {safe_words(x0, 8)}\n")
                gdb.write(f"  x0[0..15] d32: {safe_dwords(x0, 16)}\n")
            except Exception:
                pass
        if x2 and x2 < 0x100000000000 and x2 > 0x10000:
            try:
                gdb.write(f"  x2[0..7] q64: {safe_words(x2, 8)}\n")
                gdb.write(f"  x2[0..15] d32: {safe_dwords(x2, 16)}\n")
            except Exception:
                pass
        # Print frame 1-4 to identify which conv is being processed
        try:
            bt = gdb.execute("bt 6", to_string=True)
            for line in bt.splitlines():
                gdb.write("  " + line + "\n")
        except Exception:
            pass
        if self.hits >= 40:
            gdb.write("[PARAM_EXTRACTOR] 40 hits, disabling\n")
            self.enabled = False
        return False

class LoaderBP(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, internal=False)
        self.silent = True
    def stop(self):
        base = find_librknnc_base()
        if base is not None:
            addr = base + PARAM_EXTRACTOR
            gdb.write(f"librknnc base=0x{base:x}, param_extractor=0x{addr:x}\n")
            ParamExtractorBP(addr)
        return False

for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBP(spec)
    except Exception:
        pass

gdb.write("trace_librknnc_param_extractor.gdb loaded\n")
end

run
