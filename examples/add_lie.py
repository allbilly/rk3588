import ctypes
import os 
import rockchip as rk
import mmap
import fcntl
_libc = ctypes.CDLL("libc.so.6")
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
_libc.munmap.restype = ctypes.c_int

class FileIOInterface:
    def __init__(self, path: str = "", flags: int = os.O_RDONLY, fd: int | None = None):
        self.path: str = path
        self.fd: int = fd or os.open(path, flags)
    
    def __del__(self):
        if hasattr(self, 'fd'):
            os.close(self.fd)
    
    def ioctl(self, request, arg):
        return fcntl.ioctl(self.fd, request, arg)
    
    def mmap(self, start, sz, prot, flags, offset):
        return FileIOInterface._mmap(start, sz, prot, flags, self.fd, offset)
    
    def read(self, size=None, binary=False, offset=None):
        if offset is not None:
            self.seek(offset)
        with open(self.fd, "rb" if binary else "r", closefd=False) as file:
            return file.read(size)
    
    def write(self, content, binary=False, offset=None):
        if offset is not None:
            self.seek(offset)
        with open(self.fd, "wb" if binary else "w", closefd=False) as file:
            file.write(content)
    
    def seek(self, offset):
        os.lseek(self.fd, offset, os.SEEK_SET)
    
    @staticmethod
    def _mmap(start, sz, prot, flags, fd, offset):
        x = _libc.mmap(start, sz, prot, flags, fd, offset)
        if x == 0xffffffffffffffff:
            raise OSError(f"Failed to mmap {sz} bytes: {os.strerror(ctypes.get_errno())}")
        return x

    @staticmethod
    def munmap(buf, sz):
        return _libc.munmap(buf, sz)

fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)

# Allocate regcmd, input, weights, output buffers and print their dma addresses

# npu_regs from addreg.cpp, as Python list of uint64_t
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

tasks_mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=1024, flags=rk.RKNPU_MEM_KERNEL_MAPPING | rk.RKNPU_MEM_NON_CACHEABLE)
tasks_mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=tasks_mem_create.handle, offset=0)
tasks_map = mmap.mmap(fd_ctl.fd, tasks_mem_create.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=tasks_mem_map.offset)
tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(tasks_map)), ctypes.POINTER(rk.struct_rknpu_task))


# Allocate regcmd buffer
regcmd_mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=1024, flags=rk.RKNPU_MEM_NON_CACHEABLE)
regcmd_mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=regcmd_mem_create.handle, offset=0)
regcmd_map = mmap.mmap(fd_ctl.fd, regcmd_mem_create.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=regcmd_mem_map.offset)
regcmd = (ctypes.c_uint64 * (regcmd_mem_create.size // 8)).from_buffer(regcmd_map)

# Allocate input buffer
input_mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=4194304, flags=rk.RKNPU_MEM_NON_CACHEABLE)
input_mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=input_mem_create.handle, offset=0)
input_map = mmap.mmap(fd_ctl.fd, input_mem_create.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=input_mem_map.offset)
inputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), ctypes.POINTER(ctypes.c_uint16))

# Allocate weights buffer
weights_mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=4194304, flags=rk.RKNPU_MEM_NON_CACHEABLE)
weights_mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=weights_mem_create.handle, offset=0)
weights_map = mmap.mmap(fd_ctl.fd, weights_mem_create.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=weights_mem_map.offset)
weights = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(weights_map)), ctypes.POINTER(ctypes.c_uint16))

# Allocate output buffer
output_mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=4194304, flags=rk.RKNPU_MEM_NON_CACHEABLE)
output_mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=output_mem_create.handle, offset=0)
output_map = mmap.mmap(fd_ctl.fd, output_mem_create.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=output_mem_map.offset)
outputs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), ctypes.POINTER(ctypes.c_uint16))

# Set input, weights, and output physical memory locations (32-bit address size)
input_dma = input_mem_create.dma_addr
weights_dma = weights_mem_create.dma_addr
output_dma = output_mem_create.dma_addr

npu_regs[55] = npu_regs[55] | ((input_dma & 0xFFFFFFFF) << 16)
npu_regs[61] = npu_regs[61] | ((weights_dma & 0xFFFFFFFF) << 16)
npu_regs[5] = npu_regs[5] | ((output_dma & 0xFFFFFFFF) << 16)
# Copy npu_regs into regcmd buffer
for i in range(len(npu_regs)):
    regcmd[i] = npu_regs[i]

for i in range(50):
    inputs[i] = 1
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

# Print all values of the first rknpu_task struct
print("tasks[0] values:")
print("  flags:", tasks[0].flags)
print("  op_idx:", tasks[0].op_idx)
print("  enable_mask:", tasks[0].enable_mask)
print("  int_mask:", tasks[0].int_mask)
print("  int_clear:", tasks[0].int_clear)
print("  int_status:", tasks[0].int_status)
print("  regcfg_amount:", tasks[0].regcfg_amount)
print("  regcfg_offset:", tasks[0].regcfg_offset)
print("  regcmd_addr:", hex(tasks[0].regcmd_addr) if hasattr(tasks[0], "regcmd_addr") else tasks[0].regcmd_addr)


submit_res = rk.struct_rknpu_submit(
        flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
        timeout=6000,
        task_start=0,
        task_number=1,
        task_counter=0,
        priority=0,
        task_obj_addr=tasks_mem_create.obj_addr,  # Placeholder, would be actual address in real code
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=1,
        fence_fd=-1,  
        subcore_task=(rk.struct_rknpu_subcore_task * 5)(
            rk.struct_rknpu_subcore_task(task_start=0, task_number=1),
            rk.struct_rknpu_subcore_task(task_start=1, task_number=0),
            rk.struct_rknpu_subcore_task(task_start=2, task_number=0),
        )
)

res = rk.DRM_IOCTL_RKNPU_SUBMIT(fd_ctl,   
        __payload=submit_res
)

for i in range(50):
    print(outputs[i])
