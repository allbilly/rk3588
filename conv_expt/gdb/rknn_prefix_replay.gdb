set pagination off
set confirm off
set breakpoint pending on

python
import gdb
import os

KEEP_TASKS = int(os.environ.get("KEEP_TASKS", "1"))
PREFIX_MODE = os.environ.get("PREFIX_MODE", "linear")
SUBMIT_STRUCT_SIZE = 104
DRM_COMMAND_BASE = 0x40
DRM_IOCTL_RKNPU_SUBMIT_NR = DRM_COMMAND_BASE + 1

def _decode_submit(addr):
    inf = gdb.selected_inferior()
    data = inf.read_memory(addr, SUBMIT_STRUCT_SIZE).tobytes()
    off = 0
    def read(size, signed=False):
        nonlocal off
        val = int.from_bytes(data[off:off + size], "little", signed=signed)
        off += size
        return val
    out = {
        "flags": read(4),
        "timeout": read(4),
        "task_start": read(4),
        "task_number": read(4),
        "task_counter": read(4),
        "priority": read(4, signed=True),
        "task_obj_addr": read(8),
        "regcfg_obj_addr": read(8),
        "task_base_addr": read(8),
        "user_data": read(8),
        "core_mask": read(4),
        "fence_fd": read(4, signed=True),
        "subcores": [],
    }
    for _ in range(5):
        out["subcores"].append((read(4), read(4)))
    return out

def _print_submit(label, data):
    print(f"{label} struct rknpu_submit:")
    print(f"  task_start={data['task_start']} task_number={data['task_number']} core_mask=0x{data['core_mask']:08x}")
    print(f"  task_obj_addr=0x{data['task_obj_addr']:x} regcfg_obj_addr=0x{data['regcfg_obj_addr']:x} task_base_addr=0x{data['task_base_addr']:x}")
    for idx, (start, number) in enumerate(data["subcores"]):
        print(f"  subcore_task[{idx}]={{task_start={start}, task_number={number}}}")

def _patch_submit(addr):
    inf = gdb.selected_inferior()
    inf.write_memory(addr + 8, (0).to_bytes(4, "little"))
    inf.write_memory(addr + 12, KEEP_TASKS.to_bytes(4, "little"))
    for idx in range(5):
        base = addr + 64 + idx * 8
        inf.write_memory(base, (0).to_bytes(4, "little"))
        if PREFIX_MODE == "subcores":
            number = 1 if idx < KEEP_TASKS else 0
        else:
            number = KEEP_TASKS if idx == 0 else 0
        inf.write_memory(base + 4, number.to_bytes(4, "little"))
    print(f"PATCHED submit prefix: mode={PREFIX_MODE} task_start=0 task_number={KEEP_TASKS}")

class PrefixSubmitPatch:
    submit_count = 0
    @classmethod
    def handle(cls):
        cmd = int(gdb.parse_and_eval("$x1"))
        nr = cmd & 0xff
        if nr != DRM_IOCTL_RKNPU_SUBMIT_NR:
            return
        cls.submit_count += 1
        addr = int(gdb.parse_and_eval("$x2"))
        print(f"DRM_IOCTL_RKNPU_SUBMIT #{cls.submit_count} arg=0x{addr:x}")
        before = _decode_submit(addr)
        _print_submit("before", before)
        _patch_submit(addr)
        after = _decode_submit(addr)
        _print_submit("after", after)

PrefixSubmitPatch
end

break ioctl
commands
  silent
  python PrefixSubmitPatch.handle()
  continue
end

break rknn_destroy
commands
  silent
  printf "break rknn_destroy: dumping live GEMs for prefix replay\n"
  shell mkdir -p "${DUMP_DIR:-dump/prefix_replay}"
  shell python3 dump.py "${DUMP_GEM:-5}" > "${DUMP_DIR:-dump/prefix_replay}/dump_gem${DUMP_GEM:-5}.txt"
  continue
end

run
quit
