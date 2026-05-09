from fcntl import ioctl
import os, mmap, sys
import ctypes
import numpy as np

class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    TARGET_CNA  = 0x0201   # CNA (Convolution/Matrix unit)
    TARGET_CORE = 0x0801   # CORE (Matrix compute engine)
    TARGET_DPU  = 0x1001   # DPU (Elementwise/DPU unit)
    TARGET_RDMA = 0x2001   # RDMA (Read DMA for inputs/weights)
    TARGET_PC   = 0x0081   # PC (Program Control / operation enable)

    # --- PC (0x0000) ---
    OPERATION_ENABLE    = 0x0008   # PC operation enable

    # --- DPU (0x4000) ---
    S_POINTER           = 0x4004   # DPU S pointer config (pp/exec)
    FEATURE_MODE_CFG    = 0x400c   # DPU feature mode config
    DATA_FORMAT         = 0x4010   # DPU data format config
    DST_BASE_ADDR       = 0x4020   # DPU destination base address
    DST_SURF_STRIDE     = 0x4024   # DPU destination surface stride
    DATA_CUBE_WIDTH     = 0x4030   # DPU data cube width
    DATA_CUBE_HEIGHT    = 0x4034   # DPU data cube height
    DATA_CUBE_NOTCH     = 0x4038   # DPU data cube notch
    DATA_CUBE_CHANNEL   = 0x403c   # DPU data cube channel
    BS_CFG              = 0x4040   # DPU batch/norm/scale config
    BS_ALU_CFG          = 0x4044   # DPU batch ALU operand
    BS_MUL_CFG          = 0x4048   # DPU batch MUL operand
    BS_OW_CFG           = 0x4050   # DPU batch OW config
    WDMA_SIZE_0         = 0x4058   # DPU write DMA size 0
    WDMA_SIZE_1         = 0x405c   # DPU write DMA size 1
    BN_CFG              = 0x4060   # DPU batch norm config
    BN_MUL_CFG          = 0x4068   # DPU batch norm MUL operand
    BN_RELUX_CMP_VALUE  = 0x406c   # DPU batch norm relux compare value
    EW_CFG              = 0x4070   # DPU elementwise config
    OUT_CVT_SCALE       = 0x4084   # DPU output conversion scale
    OUT_CVT_SHIFT       = 0x4088   # DPU output conversion shift
    SURFACE_ADD         = 0x40c0   # DPU surface add

    # --- DPU RDMA (0x5000) ---
    RDMA_DATA_CUBE_WIDTH  = 0x500c   # RDMA data cube width
    RDMA_DATA_CUBE_HEIGHT = 0x5010   # RDMA data cube height
    RDMA_DATA_CUBE_CHANNEL= 0x5014   # RDMA data cube channel
    RDMA_ERDMA_CFG        = 0x5034   # RDMA ERDMA config
    RDMA_SRC_BASE_ADDR    = 0x5018   # RDMA source base address (input)
    RDMA_EW_BASE_ADDR     = 0x5038   # RDMA EW base address (weight)
    RDMA_FEATURE_MODE_CFG = 0x5044   # RDMA feature mode config

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1

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

class rknpu_action(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("value", ctypes.c_uint32),
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
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))

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
        flags=0x1 | 0x4,  # RKNPU_JOB_PC | RKNPU_JOB_BLOCK(0) | RKNPU_JOB_PINGPONG
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
    # struct len is 5 but only 3 NPU core
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)

    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

def reset_npu(fd):
    action = rknpu_action(flags=RKNPU_ACT_RESET, value=0)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, action)
    print(f"reset_npu ret={ret}")
    return ret

# EW_CFG values for each op
# Base config: data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1
_EW_BASE = 0x108002c0
EW_CFG_ADD  = _EW_BASE | (2 << 16)
EW_CFG_MUL  = _EW_BASE | (1 << 2) | (1 << 8)
EW_CFG_SUB  = _EW_BASE | (4 << 16)
EW_CFG_CMP  = _EW_BASE | 1

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch0_map, scratch0_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch1_map, scratch1_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch2_map, scratch2_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch3_map, scratch3_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
scratch4_map, scratch4_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))


def write_f16(buf, vals):
    arr = np.array(vals, dtype=np.float16).view(np.uint16)
    ct_vals = (ctypes.c_uint16 * len(arr)).from_buffer(buf)
    ct_vals[:] = arr.tolist()

def read_f16(buf, n):
    return np.frombuffer(buf, dtype=np.float16, count=n).copy()

def ew_regs(ew_cfg_val, n, input_mem, weight_mem, output_mem, pre_regs=()):
    dataout_width = (n + 7) // 8 - 1

    npu_regs = [
        (reg.TARGET_DPU  << 48) | (0x000001e5 << 16) | reg.FEATURE_MODE_CFG,
        (reg.TARGET_DPU  << 48) | (0x48000002 << 16) | reg.DATA_FORMAT,
        (reg.TARGET_DPU  << 48) | (dataout_width << 16) | reg.DATA_CUBE_WIDTH,
        (reg.TARGET_DPU  << 48) | (0x00070007 << 16) | reg.DATA_CUBE_CHANNEL,
        (reg.TARGET_DPU  << 48) | (0x00000053 << 16) | reg.BS_CFG,
        (reg.TARGET_DPU  << 48) | (0x00000053 << 16) | reg.BN_CFG,
    ]
    npu_regs += list(pre_regs)
    npu_regs += [
        (reg.TARGET_DPU  << 48) | (ew_cfg_val << 16) | reg.EW_CFG,
        (reg.TARGET_DPU  << 48) | (0x00010001 << 16) | reg.OUT_CVT_SCALE,
        (reg.TARGET_DPU  << 48) | (0x00000000 << 16) | reg.OUT_CVT_SHIFT,
        (reg.TARGET_RDMA << 48) | (dataout_width << 16) | reg.RDMA_DATA_CUBE_WIDTH,
        (reg.TARGET_RDMA << 48) | (0x00000000 << 16) | reg.RDMA_DATA_CUBE_HEIGHT,
        (reg.TARGET_RDMA << 48) | (0x00000007 << 16) | reg.RDMA_DATA_CUBE_CHANNEL,
        (reg.TARGET_RDMA << 48) | (0x40000008 << 16) | reg.RDMA_ERDMA_CFG,
        (reg.TARGET_DPU  << 48) | ((output_mem.dma_addr & 0xFFFFFFFF) << 16) | reg.DST_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | ((input_mem.dma_addr & 0xFFFFFFFF) << 16) | reg.RDMA_SRC_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | ((weight_mem.dma_addr & 0xFFFFFFFF) << 16) | reg.RDMA_EW_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | (0x00017849 << 16) | reg.RDMA_FEATURE_MODE_CFG,
        (reg.TARGET_PC   << 48) | (0x00000018 << 16) | reg.OPERATION_ENABLE,
    ]
    return npu_regs

def cmplt_regs(n, input_mem, output_mem):
    # positive diff -> 1, else 0
    pre_regs = [
        (reg.TARGET_DPU << 48) | (0x00040040 << 16) | reg.BS_CFG,
        (reg.TARGET_DPU << 48) | (0x33800000 << 16) | reg.BS_ALU_CFG,
        (reg.TARGET_DPU << 48) | (0x40000000 << 16) | reg.BS_MUL_CFG,
        (reg.TARGET_DPU << 48) | (0x00040082 << 16) | reg.BN_CFG,
        (reg.TARGET_DPU << 48) | (0x7c000000 << 16) | reg.BN_MUL_CFG,
        (reg.TARGET_DPU << 48) | (0x3f800000 << 16) | reg.BN_RELUX_CMP_VALUE,
    ]
    return ew_regs(EW_CFG_CMP, n, input_mem, input_mem, output_mem, pre_regs)

def run_regs(npu_regs):
    for i in range(32):
        regcmd[i] = 0
    for i in range(len(npu_regs)):
        regcmd[i] = npu_regs[i]

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = len(npu_regs)
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

    reset_npu(fd)
    ret = submit(tasks_mem_create.obj_addr)
    print(f"SUBMIT ret={ret}")
    if ret != 0:
        raise RuntimeError(f"RKNPU submit failed: {ret}")


if __name__ == "__main__":
    n = 8
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0, -1.0, 2.0, 0.49], dtype=np.float16)
    a = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.float16)
    b = np.ones(n, dtype=np.float16)
    expected = np.where(x > np.float16(0.5), a, b).astype(np.float16)

    write_f16(input_map, x)
    write_f16(weight_map, np.full(n, 0.5, dtype=np.float16))
    write_f16(scratch0_map, a)
    write_f16(scratch1_map, b)
    write_f16(scratch2_map, np.ones(n, dtype=np.float16))

    # diff = x - 0.5; mask = diff > 0; where = a*mask + b*(1-mask)
    run_regs(ew_regs(EW_CFG_SUB, n, input_mem_create, weight_mem_create, output_mem_create))
    run_regs(cmplt_regs(n, output_mem_create, scratch3_mem_create))
    # The DPU pipeline keeps enough state after the custom compare that the first
    # following EW multiply observes stale data. Issue one scratch multiply first.
    run_regs(ew_regs(EW_CFG_MUL, n, scratch0_mem_create, scratch3_mem_create, scratch4_mem_create))
    run_regs(ew_regs(EW_CFG_MUL, n, scratch0_mem_create, scratch3_mem_create, output_mem_create))
    run_regs(ew_regs(EW_CFG_SUB, n, scratch2_mem_create, scratch3_mem_create, scratch4_mem_create))
    run_regs(ew_regs(EW_CFG_MUL, n, scratch1_mem_create, scratch4_mem_create, scratch3_mem_create))
    run_regs(ew_regs(EW_CFG_MUL, n, scratch1_mem_create, scratch4_mem_create, scratch2_mem_create))
    run_regs(ew_regs(EW_CFG_ADD, n, output_mem_create, scratch2_mem_create, scratch3_mem_create))

    got = read_f16(scratch3_map, n)
    ok = np.allclose(got, expected, atol=0.1)
    print(f"x={x}")
    print(f"a={a}")
    print(f"b={b}")
    print(f"NPU={got}")
    print(f"expected={expected}")
    print("WHERE PASS" if ok else "WHERE FAIL")
    os.close(fd)
    raise SystemExit(0 if ok else 1)
