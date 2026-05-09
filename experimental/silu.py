#!/usr/bin/env python3
from fcntl import ioctl
import os, mmap, sys, math
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
    BS_OW_CFG           = 0x4050   # DPU batch OW config
    WDMA_SIZE_0         = 0x4058   # DPU write DMA size 0
    WDMA_SIZE_1         = 0x405c   # DPU write DMA size 1
    BN_CFG              = 0x4060   # DPU batch norm config
    BN_ALU_CFG          = 0x4064   # DPU batch norm ALU operand
    BN_MUL_CFG          = 0x4068   # DPU batch norm MUL operand
    EW_CFG              = 0x4070   # DPU elementwise config
    EW_CVT_SCALE_VALUE  = 0x4078   # DPU elementwise conversion scale
    OUT_CVT_SCALE       = 0x4084   # DPU output conversion scale
    SURFACE_ADD         = 0x40c0   # DPU surface add
    LUT_ACCESS_CFG      = 0x4100   # DPU LUT access config
    LUT_ACCESS_DATA     = 0x4104   # DPU LUT access data
    LUT_CFG             = 0x4108   # DPU LUT config
    LUT_INFO            = 0x410c   # DPU LUT table info
    LUT_LE_START        = 0x4110   # DPU LUT linear-exp start
    LUT_LO_END          = 0x411c   # DPU LUT linear-only end
    LUT_LO_SLOPE_SCALE  = 0x4128   # DPU LUT overflow slope scale
    LUT_LO_SLOPE_SHIFT  = 0x412c   # DPU LUT overflow slope shift

    # --- DPU RDMA (0x5000) ---
    RDMA_S_POINTER        = 0x5004   # RDMA S pointer config (pp/exec)
    RDMA_DATA_CUBE_WIDTH  = 0x500c   # RDMA data cube width
    RDMA_DATA_CUBE_HEIGHT = 0x5010   # RDMA data cube height
    RDMA_DATA_CUBE_CHANNEL= 0x5014   # RDMA data cube channel
    RDMA_ERDMA_CFG        = 0x5034   # RDMA ERDMA config
    RDMA_SRC_BASE_ADDR    = 0x5018   # RDMA source base address (input)
    RDMA_EW_BASE_ADDR     = 0x5038   # RDMA EW base address (weight)
    RDMA_FEATURE_MODE_CFG = 0x5044   # RDMA feature mode config
    RDMA_WEIGHT           = 0x5068   # RDMA LUT weight config

LUT_SIZE = 513
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

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=16384, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4096, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4096, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))


def emit(target, addr, value):
    return (target << 48) | ((int(value) & 0xffffffff) << 16) | (addr & 0xffff)

def write_f16(buf, vals):
    arr = np.asarray(vals, dtype=np.float16).view(np.uint16)
    (ctypes.c_uint16 * len(arr)).from_buffer(buf)[:] = arr.tolist()

def read_f16(buf, n):
    return np.frombuffer(buf, dtype=np.float16, count=n).copy()

def make_silu_lut():
    # Mirrors experimental/rknnops.h's alu_case_silu LUT construction.
    lut = [0] * (LUT_SIZE * 2)
    index_scale = 2824.0
    step = 32.0 / index_scale
    output_scale = 5664.8

    for i in range(LUT_SIZE):
        x = (LUT_SIZE - 1 - i) * step
        y = -x / (1.0 + math.exp(x))
        lut[i] = int(np.clip(round(y * output_scale), -32768, 32767))
    for i in range(LUT_SIZE):
        x = i * step
        y = x / (1.0 + math.exp(-x))
        lut[LUT_SIZE + i] = int(np.clip(round(y * output_scale), -32768, 32767))

    return lut, 0x6984, output_scale

def fill_lut_regs(lut):
    npu_regs = []
    for table_id, base in ((0, 0), (1, LUT_SIZE)):
        npu_regs.append(emit(reg.TARGET_DPU, reg.LUT_ACCESS_CFG, 0x00020000 | (table_id << 16)))
        for i in range(LUT_SIZE):
            v = int(lut[base + i])
            data = v & 0xffff
            if v < 0:
                data |= 0xffff0000
            npu_regs.append(emit(reg.TARGET_DPU, reg.LUT_ACCESS_DATA, data))
    return npu_regs

def silu_regs(n, input_dma, output_dma):
    lut, bn_mul_operand, output_scale = make_silu_lut()
    # rknnops.h's SiLU path uses one 16x8 DPU tile regardless of the requested
    # logical element count. The caller reads only the first n outputs.
    width = 15
    npu_regs = fill_lut_regs(lut)

    npu_regs += [
        (reg.TARGET_DPU  << 48) | (0x00000030 << 16) | reg.S_POINTER,
        (reg.TARGET_RDMA << 48) | (0x00000030 << 16) | reg.RDMA_S_POINTER,
        (reg.TARGET_DPU  << 48) | (0x000001e5 << 16) | reg.FEATURE_MODE_CFG,
        (reg.TARGET_DPU  << 48) | (0x48000002 << 16) | reg.DATA_FORMAT,
        (reg.TARGET_DPU  << 48) | ((output_dma & 0xFFFFFFFF) << 16) | reg.DST_BASE_ADDR,
        (reg.TARGET_DPU  << 48) | (0x00000100 << 16) | reg.DST_SURF_STRIDE,
        (reg.TARGET_DPU  << 48) | (width << 16) | reg.DATA_CUBE_WIDTH,
        (reg.TARGET_DPU  << 48) | (0x00070007 << 16) | reg.DATA_CUBE_CHANNEL,
        (reg.TARGET_DPU  << 48) | (0x00000053 << 16) | reg.BS_CFG,
        (reg.TARGET_DPU  << 48) | (0x00000002 << 16) | reg.BS_OW_CFG,
        (reg.TARGET_DPU  << 48) | (0x00000007 << 16) | reg.WDMA_SIZE_0,
        (reg.TARGET_DPU  << 48) | (width << 16) | reg.WDMA_SIZE_1,
        (reg.TARGET_DPU  << 48) | (0x00020040 << 16) | reg.BN_CFG,
        (reg.TARGET_DPU  << 48) | (0x80000000 << 16) | reg.BN_ALU_CFG,
        (reg.TARGET_DPU  << 48) | ((bn_mul_operand << 16) << 16) | reg.BN_MUL_CFG,
        (reg.TARGET_DPU  << 48) | (0x00000302 << 16) | reg.EW_CFG,
        (reg.TARGET_DPU  << 48) | (0x00000001 << 16) | reg.EW_CVT_SCALE_VALUE,
        (reg.TARGET_DPU  << 48) | (0x00010001 << 16) | reg.OUT_CVT_SCALE,
        (reg.TARGET_DPU  << 48) | (0x00000200 << 16) | reg.SURFACE_ADD,
        (reg.TARGET_DPU  << 48) | (0x00000000 << 16) | 0x40c4,
        (reg.TARGET_DPU  << 48) | (0x00000068 << 16) | reg.LUT_CFG,
        (reg.TARGET_DPU  << 48) | (0x00050500 << 16) | reg.LUT_INFO,
        (reg.TARGET_DPU  << 48) | (0xffffc000 << 16) | reg.LUT_LE_START,
        (reg.TARGET_DPU  << 48) | (0x00004000 << 16) | reg.LUT_LO_END,
        (reg.TARGET_DPU  << 48) | (0x40320000 << 16) | reg.LUT_LO_SLOPE_SCALE,
        (reg.TARGET_DPU  << 48) | (0x000001a0 << 16) | reg.LUT_LO_SLOPE_SHIFT,
        (reg.TARGET_RDMA << 48) | (width << 16) | reg.RDMA_DATA_CUBE_WIDTH,
        (reg.TARGET_RDMA << 48) | (0x00000007 << 16) | reg.RDMA_DATA_CUBE_CHANNEL,
        (reg.TARGET_RDMA << 48) | ((input_dma & 0xFFFFFFFF) << 16) | reg.RDMA_SRC_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | (0x00000001 << 16) | reg.RDMA_ERDMA_CFG,
        (reg.TARGET_RDMA << 48) | (0x00017849 << 16) | reg.RDMA_FEATURE_MODE_CFG,
        (reg.TARGET_RDMA << 48) | (0x01010101 << 16) | reg.RDMA_WEIGHT,
        (reg.TARGET_PC   << 48) | (0x00000018 << 16) | reg.OPERATION_ENABLE,
    ]
    return npu_regs, output_scale

def run_silu(a_vals):
    n = len(a_vals)
    npu_regs, output_scale = silu_regs(n, input_mem_create.dma_addr, output_mem_create.dma_addr)

    for i in range(2048):
        regcmd[i] = 0
    for i in range(len(npu_regs)):
        regcmd[i] = npu_regs[i]

    write_f16(input_map, a_vals)
    output_map[: n * 2] = b"\x00" * (n * 2)

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

    raw = read_f16(output_map, n)
    return raw, (raw.astype(np.float32) / output_scale).astype(np.float16)


if __name__ == "__main__":
    a = np.linspace(-3.0, 3.0, 8, dtype=np.float16)
    expected = (a.astype(np.float32) / (1.0 + np.exp(-a.astype(np.float32)))).astype(np.float16)

    raw, r_arr = run_silu(a)
    match = np.allclose(r_arr, expected, atol=0.01)
    print(f"input={a}")
    print(f"NPU raw={raw}")
    print(f"NPU decoded={r_arr}")
    print(f"expected={expected}")
    print(f"SILU {'PASS' if match else 'FAIL'}")

    os.close(fd)
    sys.exit(0 if match else 1)
