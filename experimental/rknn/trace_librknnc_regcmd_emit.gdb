set pagination off
set breakpoint pending on
set confirm off

python
import gdb


# Offsets are for experimental/rknn/librknnc.so:
# sha256 d499753a91065f0b52b2cdfa43c073645bcc37467c8e077372b302ccafd5d53c
TRACE_POINTS = [
    ("emit_loop_a_value_ready", 0x00c152f0, 800),
    ("emit_loop_a_append", 0x00c15300, 800),
    ("emit_loop_b_value_ready", 0x00c154c4, 800),
    ("emit_loop_b_append", 0x00c154d4, 800),
    ("emit_qword_append_late", 0x00c15ff4, 400),
]

TRACE_BPS = []

TARGETS = {
    0x0201: "CNA",
    0x0801: "CORE",
    0x1001: "DPU",
    0x0081: "PC",
    0x0101: "PCREG",
    0x0401: "PPU",
    0x2001: "RDMA",
}

KEY_REGS = {
    0x100c, 0x1010, 0x1020, 0x1024, 0x1028, 0x102c,
    0x1030, 0x1034, 0x1038, 0x1040, 0x1044, 0x1070,
    0x107c, 0x1080, 0x1088, 0x1100, 0x1104, 0x1110,
    0x3010, 0x3014, 0x3018, 0x4004, 0x400c, 0x4010,
    0x4020, 0x4030, 0x4034, 0x403c, 0x4058, 0x405c,
    0x4070,
}


def reg(name):
    try:
        return int(gdb.parse_and_eval("$" + name))
    except Exception:
        return 0


def read_mem(addr, size):
    return gdb.selected_inferior().read_memory(addr, size).tobytes()


def u16(addr):
    return int.from_bytes(read_mem(addr, 2), "little")


def u32(addr):
    return int.from_bytes(read_mem(addr, 4), "little")


def u64(addr):
    return int.from_bytes(read_mem(addr, 8), "little")


def decode(qword):
    target = (qword >> 48) & 0xffff
    value = (qword >> 16) & 0xffffffff
    reg_addr = qword & 0xffff
    return target, reg_addr, value


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


def format_qword(qword):
    target, reg_addr, value = decode(qword)
    name = TARGETS.get(target, f"target_{target:04x}")
    return f"q=0x{qword:016x} {name} reg=0x{reg_addr:04x} value=0x{value:08x}"


class TraceBP(gdb.Breakpoint):
    def __init__(self, name, addr, limit):
        super().__init__(f"*0x{addr:x}", internal=False)
        self.silent = True
        self.name = name
        self.limit = limit
        self.hits = 0
        self.printed = 0

    def stop(self):
        self.hits += 1
        sp = reg("sp")
        qword = None
        try:
            qword = u64(sp + 264)
        except Exception:
            pass
        if self.name == "emit_qword_append_late":
            try:
                qword = u64(sp + 384)
            except Exception:
                pass
        interesting = False
        if qword is not None:
            target, reg_addr, value = decode(qword)
            interesting = target in TARGETS and (reg_addr in KEY_REGS or reg_addr % 4 == 0)
        if interesting:
            self.printed += 1
            gdb.write(
                f"[{self.name} hit={self.hits} printed={self.printed} pc=0x{reg('pc'):x} "
                f"x21=0x{reg('x21'):x} x24=0x{reg('x24'):x} "
                f"dest=0x{reg('x1'):x} {format_qword(qword)} "
                f"sp+0x108:u16=0x{u16(sp + 266):04x} sp+0x10c:u32=0x{u32(sp + 268):08x}]\n"
            )
        if self.hits >= self.limit:
            self.enabled = False
        return False


class ExitBP(gdb.Breakpoint):
    def __init__(self):
        super().__init__("exit", internal=True)
        self.silent = True

    def stop(self):
        for bp in TRACE_BPS:
            gdb.write(f"[trace-summary] {bp.name} hits={bp.hits} printed={bp.printed}\n")
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
        TRACE_BPS.append(TraceBP(name, addr, limit))


for spec in ("dlopen", "__libc_dlopen_mode", "_dl_debug_state"):
    try:
        LoaderBP(spec)
    except Exception:
        pass

try:
    ExitBP()
except Exception:
    pass

gdb.write("trace_librknnc_regcmd_emit.gdb loaded\n")
end

run
