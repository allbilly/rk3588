set pagination off
set breakpoint pending on
set confirm off

python
import gdb
import struct
import time

HELPER_INDEXED_OFF = 0x002efd38
HELPER_PAIR_OFF    = 0x002efce0
HELPER_BFI_OFF     = 0x00100a40
ABC_T_OFF          = 0x00597828
KC_T_OFF           = 0x00598468
EMIT_5958c8_OFF    = 0x005958c8
EMIT_5db2d8_OFF    = 0x005db2d8
PARAM_EXTRACT_OFF  = 0x00384880
REUSE_UPDATER_OFF  = 0x00307198
REUSE_UPDATER2_OFF = 0x00312328
BANK_CONFIG_OFF    = 0x005c6b50

g_emit_log = []
g_patch_log = []
g_abc_t_count = 0
g_kc_t_count = 0
g_cmd_buf_cache = {}


def u64(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 8).tobytes(), "little")


def u32(addr):
    inf = gdb.selected_inferior()
    return int.from_bytes(inf.read_memory(addr, 4).tobytes(), "little")


def read_mem(addr, size):
    inf = gdb.selected_inferior()
    return bytes(inf.read_memory(addr, size))


def fmt_cmd(cmd):
    target = (cmd >> 48) & 0xffff
    value = (cmd >> 16) & 0xffffffff
    reg = cmd & 0xffff
    return f"target=0x{target:04x} reg=0x{reg:04x} value=0x{value:08x}"


def fmt_cmd_full(cmd):
    target = (cmd >> 48) & 0xffff
    value = (cmd >> 16) & 0xffffffff
    reg = cmd & 0xffff
    return f"cmd=0x{cmd:016x} {fmt_cmd(cmd)}"


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


def dump_cmd_buffer(base_ptr, offset_base, count, label=""):
    cmds = []
    for i in range(count):
        try:
            addr = base_ptr + offset_base + i * 8
            cmd = u64(addr)
            if cmd != 0:
                cmds.append((i, cmd, fmt_cmd_full(cmd)))
        except:
            pass
    if cmds:
        gdb.write(f"\n  === {label} ({len(cmds)} non-zero cmds) ===\n")
        for idx, cmd, desc in cmds:
            gdb.write(f"  [{idx:4d}] {desc}\n")
    return cmds


class ChannelTileTrace:
    installed = False
    base = None
    call_depth = 0
    in_abc_t = False
    in_kc_t = False
    tile_num = 0

    @classmethod
    def install(cls):
        if cls.installed:
            return
        base = find_librknnrt_base()
        if base is None:
            return
        cls.base = base
        cls.installed = True
        gdb.write(f"librknnrt base=0x{base:x}\n")

        bps = [
            (ABC_T_OFF, "abc_t"),
            (KC_T_OFF, "kc_t"),
            (HELPER_INDEXED_OFF, "indexed_patch"),
            (HELPER_PAIR_OFF, "pair_patch"),
            (EMIT_5958c8_OFF, "emit_5958c8"),
            (BANK_CONFIG_OFF, "bank_config"),
        ]

        for off, name in bps:
            addr = f"*0x{base + off:x}"
            FuncBreakpoint(addr, name)
            gdb.write(f"  BP {name} at 0x{base + off:x}\n")

    @classmethod
    def handle(cls, name):
        pc = int(gdb.parse_and_eval("$pc"))
        if cls.base is None:
            return False

        if name == "abc_t":
            return cls.handle_abc_t()
        elif name == "kc_t":
            return cls.handle_kc_t()
        elif name == "indexed_patch":
            return cls.handle_indexed()
        elif name == "pair_patch":
            return cls.handle_pair()
        elif name == "emit_5958c8":
            return cls.handle_emit()
        elif name == "bank_config":
            return cls.handle_bank_config()
        return False

    @classmethod
    def handle_abc_t(cls):
        cls.tile_num += 1
        x0 = int(gdb.parse_and_eval("$x0"))
        x1 = int(gdb.parse_and_eval("$x1"))
        x2 = int(gdb.parse_and_eval("$x2"))
        x3 = int(gdb.parse_and_eval("$x3"))
        x5 = int(gdb.parse_and_eval("$x5"))
        x6 = int(gdb.parse_and_eval("$x6"))
        x7 = int(gdb.parse_and_eval("$x7"))
        gdb.write(f"\n{'='*60}\n")
        gdb.write(f"ABC_T TILE #{cls.tile_num} ctx=0x{x0:x}\n")
        gdb.write(f"  args: x1=0x{x1:x} x2=0x{x2:x} x3=0x{x3:x} x5=0x{x5:x} x6={x6} x7={x7}\n")

        try:
            obj_ptr = u64(x2)
            if obj_ptr:
                byte_40 = u32(obj_ptr + 0x40) & 0xff
                byte_38 = u32(obj_ptr + 0x38) & 0xff
                gdb.write(f"  param_obj=[x2]=0x{obj_ptr:x} byte38=0x{byte_38:02x} byte40=0x{byte_40:02x}\n")
        except:
            pass

        try:
            target = u32(x0)
            atomic_c = u32(x0 + 0x28)
            gdb.write(f"  target=0x{target:08x} atomic_c_total={atomic_c}\n")
        except:
            pass

        return False

    @classmethod
    def handle_kc_t(cls):
        cls.tile_num += 1
        x0 = int(gdb.parse_and_eval("$x0"))
        x1 = int(gdb.parse_and_eval("$x1"))
        x2 = int(gdb.parse_and_eval("$x2"))
        gdb.write(f"\n{'='*60}\n")
        gdb.write(f"KC_T TILE #{cls.tile_num} ctx=0x{x0:x}\n")
        gdb.write(f"  args: x1=0x{x1:x} x2=0x{x2:x}\n")
        return False

    @classmethod
    def handle_indexed(cls):
        ctx = int(gdb.parse_and_eval("$x0"))
        index = int(gdb.parse_and_eval("$w1"))
        value = int(gdb.parse_and_eval("$w2"))
        lr = int(gdb.parse_and_eval("$lr"))
        gdb.write(f"\n  PATCH_INDEXED lr=0x{lr - cls.base:x} ctx=0x{ctx:x} index=0x{index:x} value=0x{value:08x}\n")

        try:
            base = u64(ctx + 0x288)
            vec_base = u64(base + 0x08)
            vec_offset = u64(base + 0x28)
            slot = vec_base + vec_offset + index * 8
            old_cmd = u64(slot)
            new_target = (old_cmd >> 48) & 0xffff
            new_reg = old_cmd & 0xffff
            new_value = value
            gdb.write(f"    vec@0x{base:x} base=0x{vec_base:x} off=0x{vec_offset:x}\n")
            gdb.write(f"    OLD: {fmt_cmd_full(old_cmd)}\n")
            gdb.write(f"    NEW: target=0x{new_target:04x} reg=0x{new_reg:04x} value=0x{new_value:08x}\n")
            entry = {
                "type": "indexed",
                "lr": lr - cls.base,
                "index": index,
                "old_cmd": old_cmd,
                "new_target": new_target,
                "new_reg": new_reg,
                "new_value": new_value,
            }
            g_patch_log.append(entry)
        except Exception as exc:
            gdb.write(f"    pre-read failed: {exc}\n")

        return False

    @classmethod
    def handle_pair(cls):
        ctx = int(gdb.parse_and_eval("$x0"))
        record = int(gdb.parse_and_eval("$x1"))
        value = int(gdb.parse_and_eval("$w2"))
        lr = int(gdb.parse_and_eval("$lr"))
        gdb.write(f"\n  PATCH_PAIR lr=0x{lr - cls.base:x} ctx=0x{ctx:x} record=0x{record:x} value=0x{value:08x}\n")

        try:
            first = u64(record)
            second = u64(record + 8)
            gdb.write(f"    pair[0]=0x{first:x} pair[1]=0x{second:x}\n")
            if first:
                gdb.write(f"    pair[0]+4 old=0x{u32(first + 4):08x}\n")
            if second:
                old_cmd = u64(second)
                gdb.write(f"    OLD: {fmt_cmd_full(old_cmd)}\n")
                entry = {
                    "type": "pair",
                    "lr": lr - cls.base,
                    "record": record,
                    "old_cmd": old_cmd,
                    "new_value": value,
                    "new_target": (old_cmd >> 48) & 0xffff,
                    "new_reg": old_cmd & 0xffff,
                }
                g_patch_log.append(entry)
        except Exception as exc:
            gdb.write(f"    pair pre-read failed: {exc}\n")

        return False

    @classmethod
    def handle_emit(cls):
        x0 = int(gdb.parse_and_eval("$x0"))
        x1 = int(gdb.parse_and_eval("$x1"))
        lr = int(gdb.parse_and_eval("$lr"))
        gdb.write(f"  EMIT_5958c8 lr=0x{lr - cls.base:x} x0=0x{x0:x} x1=0x{x1:x}\n")
        return False

    @classmethod
    def handle_bank_config(cls):
        gdb.write(f"\n  BANK_CONFIG called\n")
        return False


class FuncBreakpoint(gdb.Breakpoint):
    def __init__(self, spec, name):
        super().__init__(spec, internal=False)
        self.name = name

    def stop(self):
        return ChannelTileTrace.handle(self.name)


class SolibHook(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknn_init", internal=False)

    def stop(self):
        ChannelTileTrace.install()
        return False


class DumpAtDestroy(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknn_destroy", internal=False)

    def stop(self):
        gdb.write(f"\n{'='*60}\n")
        gdb.write(f"RKNN_DESTROY - dumping patch log ({len(g_patch_log)} entries)\n")
        gdb.write(f"{'='*60}\n")

        by_reg = {}
        for entry in g_patch_log:
            reg_key = f"0x{entry.get('new_target', 0):04x}:0x{entry.get('new_reg', 0):04x}"
            by_reg.setdefault(reg_key, []).append(entry)

        for reg_key in sorted(by_reg.keys()):
            entries = by_reg[reg_key]
            gdb.write(f"\n  Register {reg_key}: {len(entries)} patches\n")
            for e in entries[:5]:
                gdb.write(f"    lr=0x{e['lr']:x} value=0x{e.get('new_value', 0):08x} type={e['type']}\n")
            if len(entries) > 5:
                gdb.write(f"    ... and {len(entries) - 5} more\n")

        gdb.write(f"\n  Total ABC_T tiles: {ChannelTileTrace.tile_num}\n")

        return False


SolibHook()
DumpAtDestroy()
gdb.write("trace_channeltile_emit.gdb loaded\n")
gdb.write("Breakpoints: ABC_T, KC_T, indexed_patch, pair_patch, emit, bank_config\n")
gdb.write("Run a multi-tile RKNN model to capture ChannelTile register emissions\n")
end

run
