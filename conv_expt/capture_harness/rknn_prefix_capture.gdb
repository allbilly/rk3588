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


class PrefixSubmitPatch:
    submit_count = 0

    @classmethod
    def handle(cls):
        try:
            cmd = int(gdb.parse_and_eval("$x1"))
        except gdb.error:
            return
        nr = cmd & 0xff
        if nr != DRM_IOCTL_RKNPU_SUBMIT_NR:
            return
        cls.submit_count += 1
        addr = int(gdb.parse_and_eval("$x2"))
        _patch_submit(addr)


def _dump_gem(gem_n):
    dump_dir = os.environ.get("DUMP_DIR", "dump/prefix_replay")
    os.makedirs(dump_dir, exist_ok=True)
    dump_tool = os.environ.get("DUMP_TOOL", "/home/orangepi/rk3588/experimental/dump.py")
    out = f"{dump_dir}/dump_gem{gem_n}.txt"
    rc = os.system(f"python3 {dump_tool} {gem_n} > {out} 2>&1")
    print(f"  dumped gem={gem_n} -> {out} rc={rc}")


end

# Set the breakpoint on ioctl (resolved after shared library load)
break ioctl
commands
  silent
  python PrefixSubmitPatch.handle()
  continue
end

break rknn_destroy
commands
  silent
  printf "rknn_destroy: dumping live GEMs for prefix replay\n"
  python
import os
gems = os.environ.get("DUMP_GEMS", "1,2")
for g in gems.split(","):
    g = g.strip()
    if g:
        PrefixSubmitPatch.handle.__class__  # ensure class loaded
        os.environ["DUMP_GEM"] = g
        # inline dump call
        import subprocess
        dump_dir = os.environ.get("DUMP_DIR", "dump/prefix_replay")
        os.makedirs(dump_dir, exist_ok=True)
        dump_tool = os.environ.get("DUMP_TOOL", "/home/orangepi/rk3588/experimental/dump.py")
        out = f"{dump_dir}/dump_gem{g}.txt"
        rc = subprocess.run(["python3", dump_tool, g], capture_output=True, text=True, timeout=60)
        with open(out, "w") as f:
            f.write(rc.stdout)
            if rc.stderr:
                f.write("\n[stderr]\n")
                f.write(rc.stderr)
        print(f"  dumped gem={g} -> {out} rc={rc.returncode}")
  continue
end

run
quit
