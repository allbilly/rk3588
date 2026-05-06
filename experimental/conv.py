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
    BS_OW_CFG           = 0x4050   # DPU batch OW config
    WDMA_SIZE_0         = 0x4058   # DPU write DMA size 0
    WDMA_SIZE_1         = 0x405c   # DPU write DMA size 1
    BN_CFG              = 0x4060   # DPU batch norm config
    EW_CFG              = 0x4070   # DPU elementwise config
    OUT_CVT_SCALE       = 0x4084   # DPU output conversion scale
    SURFACE_ADD         = 0x40c0   # DPU surface add

    # --- DPU RDMA (0x5000) ---
    RDMA_DATA_CUBE_WIDTH  = 0x500c   # RDMA data cube width
    RDMA_DATA_CUBE_HEIGHT = 0x5010   # RDMA data cube height
    RDMA_DATA_CUBE_CHANNEL= 0x5014   # RDMA data cube channel
    RDMA_ERDMA_CFG        = 0x5034   # RDMA ERDMA config
    RDMA_SRC_BASE_ADDR    = 0x5018   # RDMA source base address (input)
    RDMA_EW_BASE_ADDR     = 0x5038   # RDMA EW base address (weight)
    RDMA_FEATURE_MODE_CFG = 0x5044   # RDMA feature mode config

    # --- CNA (0x1000) ---
    CNA_CONV_CON1          = 0x100c   # CNA convolution control 1
    CNA_CONV_CON2          = 0x1010   # CNA convolution control 2 (grains)
    CNA_CONV_CON3          = 0x1014   # CNA convolution control 3 (stride)
    CNA_DATA_SIZE0         = 0x1020   # CNA input data size 0
    CNA_DATA_SIZE1         = 0x1024   # CNA input data size 1 (channel)
    CNA_DATA_SIZE2         = 0x1028   # CNA output data size 2
    CNA_DATA_SIZE3         = 0x102c   # CNA output data size 3 (atomics)
    CNA_WEIGHT_SIZE0       = 0x1030   # CNA weight total size
    CNA_WEIGHT_SIZE1       = 0x1034   # CNA weight per-kernel size
    CNA_WEIGHT_SIZE2       = 0x1038   # CNA weight dims (width/height/kernels)
    CNA_CBUF_CON0          = 0x1040   # CNA CBUF config 0 (banks)
    CNA_CBUF_CON1          = 0x1044   # CNA CBUF config 1 (entries)
    CNA_CVT_CON0           = 0x104c   # CNA convert config 0
    CNA_CVT_CON1           = 0x1050   # CNA convert config 1 (scale)
    CNA_CVT_CON2           = 0x1054   # CNA convert config 2 (scale)
    CNA_CVT_CON3           = 0x1058   # CNA convert config 3 (scale)
    CNA_CVT_CON4           = 0x105c   # CNA convert config 4 (scale)
    CNA_FEATURE_DATA_ADDR  = 0x1070   # CNA feature data base address
    CNA_DMA_CON0           = 0x1078   # CNA DMA control 0 (burst)
    CNA_DMA_CON1           = 0x107c   # CNA DMA control 1 (line stride)
    CNA_DMA_CON2           = 0x1080   # CNA DMA control 2 (surface stride)
    CNA_FC_DATA_SIZE0      = 0x1084   # CNA FC data size 0
    CNA_FC_DATA_SIZE1      = 0x1088   # CNA FC data size 1 (channel)
    CNA_DCOMP_ADDR0        = 0x1110   # CNA weight decompress address 0

    # --- CORE (0x3000) ---
    CORE_MISC_CFG          = 0x3010   # CORE misc config
    CORE_DATAOUT_SIZE_0    = 0x3014   # CORE dataout size 0 (height/width)
    CORE_DATAOUT_SIZE_1    = 0x3018   # CORE dataout size 1 (channel)
    CORE_RESERVED_3030     = 0x3030   # CORE reserved (must be zeroed)

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
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def write_regs(npu_regs, clear_count):
    for i in range(clear_count):
        regcmd[i] = 0
    for i in range(len(npu_regs)):
        regcmd[i] = npu_regs[i]

def setup_task(op_idx, enable_mask, reg_count):
    tasks[0].flags  = 0;
    tasks[0].op_idx = op_idx;
    tasks[0].enable_mask = enable_mask;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = reg_count
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

def conv_regs(input_dma, weight_dma, output_dma):
    return [
        (reg.TARGET_DPU  << 48) | (0x0000000e << 16) | reg.S_POINTER,
        (reg.TARGET_CNA  << 48) | (0x6000a120 << 16) | reg.CNA_CONV_CON1,
        (reg.TARGET_CNA  << 48) | (0x00000050 << 16) | reg.CNA_CONV_CON2,
        (reg.TARGET_CNA  << 48) | (0x00000009 << 16) | reg.CNA_CONV_CON3,
        (reg.TARGET_CNA  << 48) | (0x00040004 << 16) | reg.CNA_DATA_SIZE0,
        (reg.TARGET_CNA  << 48) | (0x00020008 << 16) | reg.CNA_DATA_SIZE1,
        (reg.TARGET_CNA  << 48) | (0x00000004 << 16) | reg.CNA_DATA_SIZE2,
        (reg.TARGET_CNA  << 48) | (0x00000010 << 16) | reg.CNA_DATA_SIZE3,
        (reg.TARGET_CNA  << 48) | (0x00000020 << 16) | reg.CNA_WEIGHT_SIZE0,
        (reg.TARGET_CNA  << 48) | (0x00000010 << 16) | reg.CNA_WEIGHT_SIZE1,
        (reg.TARGET_CNA  << 48) | (0x01010002 << 16) | reg.CNA_WEIGHT_SIZE2,
        (reg.TARGET_CNA  << 48) | (0x000000b1 << 16) | reg.CNA_CBUF_CON0,
        (reg.TARGET_CNA  << 48) | (0x00000010 << 16) | reg.CNA_CBUF_CON1,
        (reg.TARGET_CNA  << 48) | (0x00000001 << 16) | reg.CNA_CVT_CON0,
        (reg.TARGET_CNA  << 48) | (0x00010000 << 16) | reg.CNA_CVT_CON1,
        (reg.TARGET_CNA  << 48) | (0x00010000 << 16) | reg.CNA_CVT_CON2,
        (reg.TARGET_CNA  << 48) | (0x00010000 << 16) | reg.CNA_CVT_CON3,
        (reg.TARGET_CNA  << 48) | (0x00010000 << 16) | reg.CNA_CVT_CON4,
        (reg.TARGET_CNA  << 48) | ((input_dma & 0xffffffff) << 16) | reg.CNA_FEATURE_DATA_ADDR,
        (reg.TARGET_CNA  << 48) | (0x000f000f << 16) | reg.CNA_DMA_CON0,
        (reg.TARGET_CNA  << 48) | (0x00000004 << 16) | reg.CNA_DMA_CON1,
        (reg.TARGET_CNA  << 48) | (0x0000000c << 16) | reg.CNA_DMA_CON2,
        (reg.TARGET_CNA  << 48) | (0x00040004 << 16) | reg.CNA_FC_DATA_SIZE0,
        (reg.TARGET_CNA  << 48) | (0x00000008 << 16) | reg.CNA_FC_DATA_SIZE1,
        (reg.TARGET_CNA  << 48) | ((weight_dma & 0xffffffff) << 16) | reg.CNA_DCOMP_ADDR0,
        (reg.TARGET_CNA  << 48) | (0x00000007 << 16) | 0x1060,
        (reg.TARGET_CORE << 48) | (0x00000200 << 16) | reg.CORE_MISC_CFG,
        (reg.TARGET_CORE << 48) | (0x00030003 << 16) | reg.CORE_DATAOUT_SIZE_0,
        (reg.TARGET_CORE << 48) | (0x0000000f << 16) | reg.CORE_DATAOUT_SIZE_1,
        (reg.TARGET_CORE << 48) | (0x00000000 << 16) | reg.CORE_RESERVED_3030,
        (reg.TARGET_DPU  << 48) | (0x000001e4 << 16) | reg.FEATURE_MODE_CFG,
        (reg.TARGET_DPU  << 48) | (0x48000002 << 16) | reg.DATA_FORMAT,
        (reg.TARGET_DPU  << 48) | ((output_dma & 0xffffffff) << 16) | reg.DST_BASE_ADDR,
        (reg.TARGET_DPU  << 48) | (0x00000100 << 16) | reg.DST_SURF_STRIDE,
        (reg.TARGET_DPU  << 48) | (0x00000003 << 16) | reg.DATA_CUBE_WIDTH,
        (reg.TARGET_DPU  << 48) | (0x00000003 << 16) | reg.DATA_CUBE_HEIGHT,
        (reg.TARGET_DPU  << 48) | (0x0001000f << 16) | reg.DATA_CUBE_CHANNEL,
        (reg.TARGET_DPU  << 48) | (0x00000053 << 16) | reg.BS_CFG,
        (reg.TARGET_DPU  << 48) | (0x00000126 << 16) | reg.BS_OW_CFG,
        (reg.TARGET_DPU  << 48) | (0x0000000f << 16) | reg.WDMA_SIZE_0,
        (reg.TARGET_DPU  << 48) | (0x00030003 << 16) | reg.WDMA_SIZE_1,
        (reg.TARGET_DPU  << 48) | (0x00000053 << 16) | reg.BN_CFG,
        (reg.TARGET_DPU  << 48) | (0x00000383 << 16) | reg.EW_CFG,
        (reg.TARGET_DPU  << 48) | (0x00000001 << 16) | 0x4078,
        (reg.TARGET_DPU  << 48) | (0x00010001 << 16) | reg.OUT_CVT_SCALE,
        (reg.TARGET_DPU  << 48) | (0x00000200 << 16) | reg.SURFACE_ADD,
        (0x0001          << 48) | (0x00000000 << 16) | 0x40c4,
        (reg.TARGET_PC   << 48) | (0x0000000d << 16) | reg.OPERATION_ENABLE,
    ]

def pack_input(x):
    nchw = x.reshape(1, 3, 4, 4)
    return nchw.transpose(0, 2, 3, 1).reshape(-1)

def pack_weights(w):
    packed = np.zeros((2, 1, 1, 8), dtype=np.float16)
    packed[:, :, :, :3] = w.reshape(2, 3, 1, 1).transpose(0, 2, 3, 1)
    return packed.reshape(-1)

def unpack_output(raw):
    packed = raw[:128].reshape(1, 1, 1, 16, 8)
    return packed.transpose(0, 1, 4, 2, 3).reshape(1, 8, 1, 16)[:, :2].reshape(1, 2, 4, 4)

def run_conv():
    input_nchw = (np.arange(1 * 2 * 4 * 4, dtype=np.float16).reshape(1, 2, 4, 4) / np.float16(8.0)).astype(np.float16)
    input_padded = np.zeros((1, 3, 4, 4), dtype=np.float16)
    input_padded[:, :2] = input_nchw
    weight = np.array([
        1.0, -0.5, 0.0,
        0.25, 0.75, 0.0,
    ], dtype=np.float16).reshape(2, 3, 1, 1)

    input_packed = pack_input(input_padded)
    weight_packed = pack_weights(weight)
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), input_packed.ctypes.data, input_packed.nbytes)
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), weight_packed.ctypes.data, weight_packed.nbytes)

    npu_regs = conv_regs(input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)
    write_regs(npu_regs, 64)
    setup_task(1, 0x0d, len(npu_regs))

    reset_npu(fd)
    ret = submit(tasks_mem_create.obj_addr)
    out_raw = np.frombuffer(output_map, dtype=np.float16, count=128).copy()
    got = unpack_output(out_raw)
    expected = np.zeros((1, 2, 4, 4), dtype=np.float16)
    for o in range(2):
        for c in range(2):
            expected[:, o] += input_nchw[:, c] * weight[o, c, 0, 0]
    ok = ret == 0 and np.allclose(got, expected, atol=0.1)
    print(f"SUBMIT ret={ret}")
    print(f"NPU output={got.reshape(-1)[:16]}")
    print(f"expected={expected.reshape(-1)[:16]}")
    print(f"max_abs_diff={float(np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))):.6f}")
    print("CONV PASS" if ok else "CONV FAIL")
    return 0 if ok else 1

def dry_run():
    npu_regs = conv_regs(0, 0, 0)
    names = {reg.TARGET_CNA: "CNA", reg.TARGET_CORE: "CORE", reg.TARGET_DPU: "DPU", reg.TARGET_PC: "PC"}
    print(f"CONV dry run regs={len(npu_regs)}")
    for i, cmd in enumerate(npu_regs):
        target = (cmd >> 48) & 0xffff
        value = (cmd >> 16) & 0xffffffff
        addr = cmd & 0xffff
        print(f"  [{i:2d}] {names.get(target, f'0x{target:04x}')}[0x{addr:04x}] = 0x{value:08x}")
    print("CONV DRY RUN PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(dry_run() if "--dry" in sys.argv else run_conv())
