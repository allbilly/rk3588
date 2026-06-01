set pagination off
set confirm off
set breakpoint pending on

python
import gdb
import os
from pathlib import Path

SUBMIT_STRUCT_SIZE = 104
TASK_STRUCT_SIZE = 40
DRM_COMMAND_BASE = 0x40
DRM_IOCTL_RKNPU_SUBMIT_NR = DRM_COMMAND_BASE + 1
OUT = Path(os.environ.get("REGCMD_DUMP", "/tmp/opencode/rknn_regcmd_dump.bin"))

def _read_submit(addr):
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

def _read_tasks(addr, count):
    inf = gdb.selected_inferior()
    tasks = []
    for idx in range(count):
        data = inf.read_memory(addr + idx * TASK_STRUCT_SIZE, TASK_STRUCT_SIZE).tobytes()
        values = []
        off = 0
        for size in (4, 4, 4, 4, 4, 4, 4, 4, 8):
            values.append(int.from_bytes(data[off:off + size], "little"))
            off += size
        tasks.append({
            "flags": values[0],
            "op_idx": values[1],
            "enable_mask": values[2],
            "int_mask": values[3],
            "int_clear": values[4],
            "int_status": values[5],
            "regcfg_amount": values[6],
            "regcfg_offset": values[7],
            "regcmd_addr": values[8],
        })
    return tasks

def _looks_like_tasks(data):
    if len(data) < TASK_STRUCT_SIZE * 3:
        return False
    expected = ((0x0d, 108), (0x0d, 104), (0x60, 26))
    for idx in range(3):
        off = idx * TASK_STRUCT_SIZE
        op_idx = int.from_bytes(data[off + 4:off + 8], "little")
        mask = int.from_bytes(data[off + 8:off + 12], "little")
        amount = int.from_bytes(data[off + 24:off + 28], "little")
        regcmd = int.from_bytes(data[off + 32:off + 40], "little")
        if op_idx != 1 or (mask, amount) != expected[idx] or regcmd == 0 or (regcmd & 0xf):
            return False
    return True

def _find_task_mapping():
    pid = gdb.selected_inferior().pid
    inf = gdb.selected_inferior()
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or "r" not in parts[1]:
                continue
            start_s, end_s = parts[0].split("-")
            start, end = int(start_s, 16), int(end_s, 16)
            if end - start > 1024 * 1024:
                continue
            try:
                data = inf.read_memory(start, end - start).tobytes()
            except gdb.MemoryError:
                continue
            for off in range(0, max(0, len(data) - TASK_STRUCT_SIZE * 3 + 1), 8):
                if _looks_like_tasks(data[off:off + TASK_STRUCT_SIZE * 3]):
                    return start + off
    return None

class DumpSubmitRegcmd:
    done = False

    @classmethod
    def handle(cls):
        if cls.done:
            return
        cmd = int(gdb.parse_and_eval("$x1"))
        if (cmd & 0xff) != DRM_IOCTL_RKNPU_SUBMIT_NR:
            return
        addr = int(gdb.parse_and_eval("$x2"))
        submit = _read_submit(addr)
        task_addr = _find_task_mapping()
        if task_addr is None:
            print("unable to find user task mapping")
            return
        tasks = _read_tasks(task_addr, submit["task_number"])
        if not tasks:
            return
        print(f"found task user mapping at 0x{task_addr:x}")
        for idx, task in enumerate(tasks):
            print(
                f"  task{idx}: amount={task['regcfg_amount']} mask=0x{task['enable_mask']:x} "
                f"int_status=0x{task['int_status']:x} regcmd_addr=0x{task['regcmd_addr']:x}"
            )
        base = task_addr - 0x1000
        amount = max(task["regcfg_amount"] for task in tasks)
        size = amount * 8
        data = gdb.selected_inferior().read_memory(base, size).tobytes()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(data)
        meta = OUT.with_suffix(OUT.suffix + ".txt")
        with meta.open("w") as f:
            f.write(f"task_number={submit['task_number']} core_mask=0x{submit['core_mask']:08x}\n")
            f.write("subcores=" + ",".join(f"({a},{b})" for a, b in submit["subcores"]) + "\n")
            f.write(f"task_user_addr=0x{task_addr:x} inferred_regcmd_user_addr=0x{base:x}\n")
            for idx, task in enumerate(tasks):
                f.write(
                    f"task{idx}: amount={task['regcfg_amount']} mask=0x{task['enable_mask']:x} "
                    f"int_status=0x{task['int_status']:x} regcmd_addr=0x{task['regcmd_addr']:x}\n"
                )
        print(f"dumped submit regcmd base=0x{base:x} amount={amount} bytes={size} to {OUT}")
        cls.done = True

DumpSubmitRegcmd
end

break ioctl
commands
  silent
  python DumpSubmitRegcmd.handle()
  continue
end

run
quit
