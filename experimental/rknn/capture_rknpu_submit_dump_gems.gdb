set pagination off
set breakpoint pending on

python
import gdb

IOC_NRBITS = 8
IOC_TYPEBITS = 8
IOC_SIZEBITS = 14
IOC_NRSHIFT = 0
IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
DRM_COMMAND_BASE = 0x40
DRM_IOCTL_RKNPU_SUBMIT_NR = DRM_COMMAND_BASE + 1
SUBMIT_STRUCT_SIZE = 104


def decode_submit(addr):
    data = gdb.selected_inferior().read_memory(addr, SUBMIT_STRUCT_SIZE).tobytes()
    return int.from_bytes(data[12:16], "little"), int.from_bytes(data[24:32], "little")


class Capture:
    dumps = 0

    @classmethod
    def handle(cls):
        cmd = int(gdb.parse_and_eval("$x1"))
        nr = (cmd >> IOC_NRSHIFT) & ((1 << IOC_NRBITS) - 1)
        if nr != DRM_IOCTL_RKNPU_SUBMIT_NR:
            return
        cls.dumps += 1
        arg = int(gdb.parse_and_eval("$x2"))
        try:
            task_number, task_obj_addr = decode_submit(arg)
        except gdb.MemoryError:
            task_number, task_obj_addr = 0, 0
        print(f"submit #{cls.dumps}: task_number={task_number} task_obj_addr=0x{task_obj_addr:x}")
        gdb.execute("shell python3 /tmp/dump_rknpu_task_gems.py 1 2 3 4 5 6", to_string=False)

end

break ioctl
commands
  silent
  python Capture.handle()
  continue
end

run
