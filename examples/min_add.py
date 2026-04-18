from fcntl import ioctl
import os, mmap
import ctypes
RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0

fd = os.open(f"/dev/dri/card1", os.O_RDWR)

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("obj_addr", ctypes.c_uint64),
        ("dma_addr", ctypes.c_uint64),
        ("sram_size", ctypes.c_uint64),
    ]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
    ]

class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("timeout", ctypes.c_uint32),
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
        ("task_counter", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64),
        ("regcfg_obj_addr", ctypes.c_uint64),
        ("task_base_addr", ctypes.c_uint64),
        ("user_data", ctypes.c_uint64),
        ("core_mask", ctypes.c_uint32),
        ("fence_fd", ctypes.c_int32),
        ("subcore_task", rknpu_subcore_task * 5),
    ]

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)
DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))

class struct_rknpu_task(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("op_idx", ctypes.c_uint32),
        ("enable_mask", ctypes.c_uint32),
        ("int_mask", ctypes.c_uint32),
        ("int_clear", ctypes.c_uint32),
        ("int_status", ctypes.c_uint32),
        ("regcfg_amount", ctypes.c_uint32),
        ("regcfg_offset", ctypes.c_uint32),
        ("regcmd_addr", ctypes.c_uint64),
    ]

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("obj_addr", ctypes.c_uint64),
        ("dma_addr", ctypes.c_uint64),
        ("sram_size", ctypes.c_uint64),
    ]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
    ]

class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("timeout", ctypes.c_uint32),
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
        ("task_counter", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64),
        ("regcfg_obj_addr", ctypes.c_uint64),
        ("task_base_addr", ctypes.c_uint64),
        ("user_data", ctypes.c_uint64),
        ("core_mask", ctypes.c_uint32),
        ("fence_fd", ctypes.c_int32),
        ("subcore_task", rknpu_subcore_task * 5),
    ]


def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(
        flags=flags, #0x10 | 0x2,  # KERNEL_MAPPING | NON_CACHEABLE
        size=size
    )
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    print(f"ret={ret}, handle={mem_create.handle}, obj_addr={mem_create.obj_addr:#x}, dma_addr={mem_create.dma_addr:#x}")
    
    # Map memory to access from userspace
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    print(f"Memory mapped at offset={mem_map.offset:#x}")
    return buf, mem_create

def submit(task_obj_addr):
    submit_struct = rknpu_submit(
        flags=0x1 | 0x2 | 0x4,  # RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG
        timeout=6000,
        task_start=0,
        task_number=1,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=1,
        fence_fd=-1,
    )
    # Initialize subcore_task array
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
    
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

npu_regs = [
  0x10010000000e4004, # 0
  0x20010000000e5004, # 1
  0x1001000001e5400c, # 2
  0x1001480000024010, # 3
  0x1001000000004014, # 4
  0x1001000000004020, # 5
  0x1001000000c04024, # 6
  0x1001000000094030, # 7
  0x1001000000004034, # 8
  0x1001000000004038, # 9
  0x100100070007403c, # 10
  0x1001000000534040, # 11
  0x1001000000004044, # 12
  0x1001000000004048, # 13
  0x100100000000404c, # 14
  0x1001000000024050, # 15
  0x1001000000004054, # 16
  0x1001000000074058, # 17
  0x100100000009405c, # 18
  0x1001000000534060, # 19
  0x1001000000004064, # 20
  0x1001000000004068, # 21
  0x100100000000406c, # 22
  0x1001108202c04070, # 23
  0x1001000000004074, # 24
  0x1001000000014078, # 25
  0x100100000000407c, # 26
  0x1001000000004080, # 27
  0x1001000100014084, # 28
  0x1001000000004088, # 29
  0x1001000000004090, # 30
  0x1001000000004094, # 31
  0x1001000000004098, # 32
  0x100100000000409c, # 33
  0x10010000000040a0, # 34
  0x10010000000040a4, # 35
  0x10010000000040a8, # 36
  0x10010000000040ac, # 37
  0x1001000000c040c0, # 38
  0x10010000000040c4, # 39
  0x1001000000004100, # 40
  0x1001000000004104, # 41
  0x1001000000004108, # 42
  0x100100000000410c, # 43
  0x1001000000004110, # 44
  0x1001000000004114, # 45
  0x1001000000004118, # 46
  0x100100000000411c, # 47
  0x1001000000004120, # 48
  0x1001000000004124, # 49
  0x1001000000004128, # 50
  0x100100000000412c, # 51
  0x200100000009500c, # 52
  0x2001000000005010, # 53
  0x2001000000075014, # 54
  0x2001000000005018, # 55
  0x200100000000501c, # 56
  0x2001000000005020, # 57
  0x2001000000005028, # 58
  0x200100000000502c, # 59
  0x2001400000085034, # 60
  0x2001000000005038, # 61
  0x2001000000c05040, # 62
  0x2001000178495044, # 63
  0x2001000000005048, # 64
  0x200100000020504c, # 65
  0x2001000000005064, # 66
  0x2001010101015068, # 67
  0x200100000020506c, # 68
  0x0000000000000000, # 69
  0x0101000000000014, # 70
  0x0041000000000000, # 71
  0x0081000000180008, # 72
]

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
inputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), ctypes.POINTER(ctypes.c_uint16))
weights = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), ctypes.POINTER(ctypes.c_uint16))
outputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), ctypes.POINTER(ctypes.c_uint16))

# update addr in npu_regs
npu_regs[55] = npu_regs[55] | ((input_mem_create.dma_addr & 0xFFFFFFFF) << 16)
npu_regs[61] = npu_regs[61] | ((weight_mem_create.dma_addr & 0xFFFFFFFF) << 16)
npu_regs[5] = npu_regs[5] | ((output_mem_create.dma_addr & 0xFFFFFFFF) << 16)

for i in range(len(npu_regs)):
    regcmd[i] = npu_regs[i]
for i in range(50):
    inputs[i] = 3
    weights[i] = 5

tasks[0].flags  = 0;
tasks[0].op_idx = 4;
tasks[0].enable_mask = 0x18;
tasks[0].int_mask = 0x300;
tasks[0].int_clear = 0x1ffff;
tasks[0].int_status = 0;
tasks[0].regcfg_amount = len(npu_regs)
tasks[0].regcfg_offset = 0;
tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

ret = submit(tasks_mem_create.obj_addr)
print(f"SUBMIT ret={ret}")

for i in range(50):
    print(outputs[i])