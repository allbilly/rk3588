set pagination off
set breakpoint pending on
set confirm off

python
import gdb
import struct

HELPER_INDEXED_OFF = 0x002efd38
HELPER_PAIR_OFF    = 0x002efce0
HELPER_BFI_OFF     = 0x00100a40

g_patches = []
g_pair_patches = []
g_base = None


def u64(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 8).tobytes(), "little")


def u32(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 4).tobytes(), "little")


def fmt_cmd(cmd):
    target = (cmd >> 48) & 0xffff
    value = (cmd >> 16) & 0xffffffff
    reg = cmd & 0xffff
    return f"t=0x{target:04x} r=0x{reg:04x} v=0x{value:08x}"


def find_librknnrt_base():
    out = gdb.execute("info proc mappings", to_string=True)
    for line in out.splitlines():
        if "librknnrt" not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            return int(parts[0], 16)
        except ValueError:
            continue
    return None


class IndexedBP(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, internal=False)

    def stop(self):
        global g_base
        ctx = int(gdb.parse_and_eval("$x0"))
        index = int(gdb.parse_and_eval("$w1"))
        value = int(gdb.parse_and_eval("$w2"))
        lr = int(gdb.parse_and_eval("$lr"))

        lr_off = lr - g_base if g_base else 0

        target_reg = ""
        old_cmd = 0
        try:
            base = u64(ctx + 0x288)
            vec_base = u64(base + 0x08)
            vec_offset = u64(base + 0x28)
            slot = vec_base + vec_offset + index * 8
            old_cmd = u64(slot)
            t = (old_cmd >> 48) & 0xffff
            r = old_cmd & 0xffff
            target_reg = f"0x{t:04x}:0x{r:04x}"
        except:
            target_reg = "unreadable"

        g_patches.append({
            "lr_off": lr_off,
            "index": index,
            "value": value,
            "target_reg": target_reg,
            "old_cmd": old_cmd,
        })

        gdb.write(f"P indexed lr=0x{lr_off:x} idx={index:5d} val=0x{value:08x} -> {target_reg}\n")
        return False


class PairBP(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, internal=False)

    def stop(self):
        global g_base
        ctx = int(gdb.parse_and_eval("$x0"))
        record = int(gdb.parse_and_eval("$x1"))
        value = int(gdb.parse_and_eval("$w2"))
        lr = int(gdb.parse_and_eval("$lr"))
        lr_off = lr - g_base if g_base else 0

        gdb.write(f"P pair    lr=0x{lr_off:x} rec=0x{record:x} val=0x{value:08x}\n")
        g_pair_patches.append({"lr_off": lr_off, "value": value})
        return False


class InitHook(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknn_init", internal=False)

    def stop(self):
        global g_base
        base = find_librknnrt_base()
        if base:
            g_base = base
            gdb.write(f"librknnrt base=0x{base:x}\n")
            IndexedBP(f"*0x{base + HELPER_INDEXED_OFF:x}")
            PairBP(f"*0x{base + HELPER_PAIR_OFF:x}")
        return False


class DestroyHook(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknn_destroy", internal=False)

    def stop(self):
        gdb.write(f"\n{'='*70}\n")
        gdb.write(f"SUMMARY: {len(g_patches)} indexed patches, {len(g_pair_patches)} pair patches\n")
        gdb.write(f"{'='*70}\n")

        by_target_reg = {}
        for p in g_patches:
            key = p["target_reg"]
            by_target_reg.setdefault(key, []).append(p)

        for key in sorted(by_target_reg.keys()):
            entries = by_target_reg[key]
            values = [e["value"] for e in entries]
            gdb.write(f"\n  {key} ({len(entries)} patches):\n")
            vals_str = " ".join(f"0x{v:08x}" for v in values[:8])
            if len(values) > 8:
                vals_str += " ..."
            gdb.write(f"    values: {vals_str}\n")
            gdb.write(f"    indices: {[e['index'] for e in entries[:8]]}\n")
            lr_str = " ".join(f"0x{e['lr_off']:x}" for e in entries[:8])
            gdb.write(f"    lr_offs: {lr_str}\n")

        by_lr = {}
        for p in g_patches:
            by_lr.setdefault(p["lr_off"], []).append(p)

        gdb.write(f"\n  By caller (lr offset):\n")
        for lr in sorted(by_lr.keys()):
            entries = by_lr[lr]
            gdb.write(f"    0x{lr:x}: {len(entries)} patches\n")

        return False


InitHook()
DestroyHook()
gdb.write("trace_patch_regtargets.gdb loaded\n")
end

run
