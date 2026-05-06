import os, mmap, ctypes, numpy as np, sys
from fcntl import ioctl

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
FP16_BYTES = 2

class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    TARGET_CNA  = 0x0201   # CNA (Convolution/Matrix unit)
    TARGET_CORE = 0x0801   # CORE (Matrix compute engine)
    TARGET_DPU  = 0x1001   # DPU (Elementwise/DPU unit)
    TARGET_RDMA = 0x2001   # RDMA (Read DMA for inputs/weights)
    TARGET_PC   = 0x0081   # PC (Program Control / operation enable)
    TARGET_PC_REG = 0x0101 # PC chain registers
    TARGET_VERSION = 0x0041

    # --- PC (0x0000) ---
    OPERATION_ENABLE    = 0x0008   # PC operation enable
    PC_BASE_ADDRESS     = 0x0010   # next regcmd DMA address for PC chain
    PC_REGISTER_AMOUNTS = 0x0014   # next regcmd fetch amount for PC chain

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
    CNA_CVT_CON5           = 0x1060   # CNA convert config 5 (mask)
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

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)
DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))

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

def npu_reset(fd):
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

def npu_submit(task_obj_addr, task_count=1, flags=0x1):
    npu_reset(fd)
    submit_struct = rknpu_submit(
        flags=flags,
        timeout=6000,
        task_start=0,
        task_number=task_count,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=1,
        fence_fd=-1,
    )
    # 3 cores
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

fd = os.open(f"/dev/dri/card1", os.O_RDWR)
task_map, tasks_mem_create = mem_allocate(fd, size=64*1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=512*1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
npu_tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
npu_regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

def make_conv_regs(in_dma, wt_dma, out_dma):
    npu_regs = [
        E(reg.TARGET_DPU,  reg.S_POINTER,
                ((1 << 3) |                           # DPU_S_POINTER_POINTER_PP_MODE
                (1 << 2)  |                           # DPU_S_POINTER_EXECUTER_PP_EN
                (1 << 1))                             # DPU_S_POINTER_POINTER_PP_EN
        ),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON1,
                ((3 << 29) |                          # CNA_CONV_CON1_PIXEL/LINE mode
                (10 << 12) |                          # CNA_CONV_CON1_PIXEL_CHANNELS
                (2 << 7)  |                           # CNA_CONV_CON1_PROC_PRECISION(2)=fp16
                (2 << 4))                             # CNA_CONV_CON1_IN_PRECISION(2)=fp16
        ),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON2,
                (5 << 4)                              # CNA_CONV_CON2_FEATURE_GRAINS
        ),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON3,
                ((1 << 3) |                           # CNA_CONV_CON3_OPERATION_ENABLE
                1)                                    # CNA_CONV_CON3_WEIGHT_REUSE
        ),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE0,
                ((4 << 16) |                          # CNA_DATA_SIZE0_WIDTH
                4)                                    # CNA_DATA_SIZE0_HEIGHT
        ),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE1,
                ((2 << 16) |                          # CNA_DATA_SIZE1_CHANNEL
                8)                                    # CNA_DATA_SIZE1_ATOMICS
        ),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE2,   4),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE3,   16),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE0, 32),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE1, 16),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE2,
                ((1 << 24) |                          # CNA_WEIGHT_SIZE2_KERNEL_WIDTH
                (1 << 16) |                           # CNA_WEIGHT_SIZE2_KERNEL_HEIGHT
                2)                                    # CNA_WEIGHT_SIZE2_KERNELS
        ),
        E(reg.TARGET_CNA,  reg.CNA_CBUF_CON0,
                ((11 << 4) |                          # CNA_CBUF_CON0_WEIGHT_BANK
                1)                                    # CNA_CBUF_CON0_DATA_BANK
        ),
        E(reg.TARGET_CNA,  reg.CNA_CBUF_CON1,    16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON0,     1),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON1,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON2,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON3,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON4,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_FEATURE_DATA_ADDR, in_dma & 0xFFFFFFFF),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON0,
                ((15 << 16) |                         # CNA_DMA_CON0_READ_BURST
                15)                                   # CNA_DMA_CON0_WEIGHT_BURST
        ),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON1,     4),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON2,     12),
        E(reg.TARGET_CNA,  reg.CNA_FC_DATA_SIZE0,
                ((4 << 16) |                          # CNA_FC_DATA_SIZE0_WIDTH
                4)                                    # CNA_FC_DATA_SIZE0_HEIGHT
        ),
        E(reg.TARGET_CNA,  reg.CNA_FC_DATA_SIZE1, 8),
        E(reg.TARGET_CNA,  reg.CNA_DCOMP_ADDR0,  wt_dma & 0xFFFFFFFF),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON5,     7),
        E(reg.TARGET_CORE, reg.CORE_MISC_CFG,
                (2 << 8)                              # CORE_MISC_CFG_PROC_PRECISION
        ),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_0,
                ((3 << 16) |                          # CORE_DATAOUT_SIZE_0_HEIGHT
                3)                                    # CORE_DATAOUT_SIZE_0_WIDTH
        ),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_1, 15),
        E(reg.TARGET_CORE, reg.CORE_RESERVED_3030,  0),
        E(reg.TARGET_DPU,  reg.FEATURE_MODE_CFG,
                ((15 << 5) |                          # DPU_FEATURE_MODE_CFG_BYPASS
                (2 << 1))                             # DPU_FEATURE_MODE_CFG_MODE
        ),
        E(reg.TARGET_DPU,  reg.DATA_FORMAT,
                ((2 << 29) |                          # DPU_DATA_FORMAT_OUT_PRECISION
                (2 << 26) |                           # DPU_DATA_FORMAT_PROC_PRECISION
                2)                                    # DPU_DATA_FORMAT_IN_PRECISION
        ),
        E(reg.TARGET_DPU,  reg.DST_BASE_ADDR,     out_dma & 0xFFFFFFFF),
        E(reg.TARGET_DPU,  reg.DST_SURF_STRIDE,
                (16 << 4)                             # DPU_DST_SURF_STRIDE
        ),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_WIDTH,   3),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_HEIGHT,  3),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_CHANNEL,
                ((1 << 16) |                          # DPU_DATA_CUBE_CHANNEL_CUBE
                15)                                   # DPU_DATA_CUBE_CHANNEL_ATOMICS
        ),
        E(reg.TARGET_DPU,  reg.BS_CFG,           0x53),
        E(reg.TARGET_DPU,  reg.BS_OW_CFG,        0x126),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_0,      15),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_1,
                ((3 << 16) |                          # DPU_WDMA_SIZE_1_HEIGHT
                3)                                    # DPU_WDMA_SIZE_1_WIDTH
        ),
        E(reg.TARGET_DPU,  reg.BN_CFG,           0x53),
        E(reg.TARGET_DPU,  reg.EW_CFG,           0x383),
        E(reg.TARGET_DPU,  0x4078,               1),
        E(reg.TARGET_DPU,  reg.OUT_CVT_SCALE,    (1 << 16) | 1),
        E(reg.TARGET_DPU,  reg.SURFACE_ADD,
                (32 << 4)                             # DPU_SURFACE_ADD
        ),
        E(0x0001,          0x40c4,               0),
        E(reg.TARGET_PC,   reg.OPERATION_ENABLE, 0x0d),
    ]
    return npu_regs

def write_regs_to_npu_task(task_regs):
    def enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len):
        enable_npu_units = E(reg.TARGET_PC, reg.OPERATION_ENABLE,
                            (6 << 1) |                           # PC_OPERATION_ENABLE_RESERVED_0 = units to enable(DPU/CNA/PPU), info not in TRM
                            1)                                   # PC_OPERATION_ENABLE_OP_EN

        if next_offset is None: return [ enable_npu_units ]
        next_addr = regcmd_mem_create.dma_addr + next_offset* ctypes.sizeof(ctypes.c_uint64)
        return [
            E(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
            E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(next_task_regs_len, 2) + 1),
            E(reg.TARGET_VERSION, 0, 0),
            enable_npu_units
        ]

    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + 4, 2)
    assert offset <= regcmd_mem_create.size // ctypes.sizeof(ctypes.c_uint64), "regcmd buffer too small"

    # chain mutilple tasks in one npu_submit
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            npu_regcmd[base + i] = qword
        next_task_regs_len = len(task_regs[idx + 1]) if idx + 1 < len(task_regs) else 0
        next_offset = offsets[idx + 1]               if idx + 1 < len(task_regs) else None

        tails = enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len)
        if len(task_regs) == 1 and regs[-1] == E(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x0d):
            tails = []
        for i, qword in enumerate(tails):
            npu_regcmd[base + len(regs) + i] = qword

        # write flag for npu_tasks
        npu_tasks[idx].flags = 0
        npu_tasks[idx].regcmd_addr = regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)
        npu_tasks[idx].regcfg_amount = len(regs)
        npu_tasks[idx].regcfg_offset = 0
        npu_tasks[idx].op_idx = 1
        npu_tasks[idx].enable_mask = 0x0d                  # Downstream raw CONV task descriptor mask.
        npu_tasks[idx].int_mask = (1 << 8) | (1 << 9)      # PC_INTERRUPT_MASK_DPU_0 | DPU_1.
        npu_tasks[idx].int_clear = 0x1ffff                 # Downstream RKNPU_INT_CLEAR clears all driver status bits.
        npu_tasks[idx].int_status = 0

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
    ct_inputs = (ctypes.c_uint16 * len(input_packed)).from_buffer(input_map)
    ct_weights = (ctypes.c_uint16 * len(weight_packed)).from_buffer(weight_map)
    ct_inputs[:] = input_packed.view(np.uint16).tolist()
    ct_weights[:] = weight_packed.view(np.uint16).tolist()

    task_regs = [make_conv_regs(input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)]
    assert len(task_regs) <= tasks_mem_create.size // ctypes.sizeof(struct_rknpu_task), "task buffer too small"

    write_regs_to_npu_task(task_regs)
    ret = npu_submit(tasks_mem_create.obj_addr, task_count=len(task_regs),
           flags=((1 << 0) |                          # RKNPU_JOB_PC
                  (1 << 1) |                          # RKNPU_JOB_BLOCK
                  (1 << 2)))                          # RKNPU_JOB_PINGPONG

    out_raw = np.frombuffer(output_map, dtype=np.float16, count=128).copy()
    got = unpack_output(out_raw)
    expected = np.zeros((1, 2, 4, 4), dtype=np.float16)
    for o in range(2):
        for c in range(2):
            expected[:, o] += input_nchw[:, c] * weight[o, c, 0, 0]

    ok = ret == 0 and np.allclose(got, expected, atol=0.1)
    md = np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))
    print(f"SUBMIT ret={ret}")
    print(f"NPU output={got.reshape(-1)[:16]}")
    print(f"expected={expected.reshape(-1)[:16]}")
    print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")
    return got

def dry_run():
    npu_regs = make_conv_regs(0, 0, 0)
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
    if "--dry" in sys.argv:
        raise SystemExit(dry_run())
    r = run_conv()
    os.close(fd)
