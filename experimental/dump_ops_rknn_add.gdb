set pagination off
set breakpoint pending on
set confirm off

python
import gdb

SUBMIT_IOCTL = 0xc0686441
dumped = False
SUBMIT_STRUCT_SIZE = 104
TASK_STRUCT_SIZE = 40

def u32(data, offset):
    return int.from_bytes(data[offset:offset + 4], "little")

def u64(data, offset):
    return int.from_bytes(data[offset:offset + 8], "little")

def print_submit(addr):
    inf = gdb.selected_inferior()
    data = inf.read_memory(addr, SUBMIT_STRUCT_SIZE).tobytes()
    flags = u32(data, 0)
    task_number = u32(data, 12)
    task_obj_addr = u64(data, 24)
    core_mask = u32(data, 56)
    gdb.write(f"submit flags=0x{flags:x} task_number={task_number} task_obj_addr=0x{task_obj_addr:x} core_mask=0x{core_mask:x}\n")
    for i in range(5):
        off = 64 + i * 8
        gdb.write(f"  subcore_task[{i}]=({u32(data, off)},{u32(data, off + 4)})\n")
end

break ioctl
commands
  silent
  python
cmd = int(gdb.parse_and_eval("$x1"))
if cmd == SUBMIT_IOCTL and not globals()["dumped"]:
    globals()["dumped"] = True
    gdb.write("capturing ops_rknn ADD GEM buffers before submit\n")
    print_submit(int(gdb.parse_and_eval("$x2")))
    gdb.execute("shell python3 dump.py 1 2 3 4 5 6 > /tmp/ops_rknn_add_gems.txt 2>&1")
end
  continue
end

run
quit
