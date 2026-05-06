import os, mmap, ctypes, numpy as np, sys
from fcntl import ioctl

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
FP16_BYTES = 2
REGCMD_RESERVED = 16384

class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    CNA  = 0x0201   # CNA (Convolution/Matrix unit)
    CORE = 0x0801   # CORE (Matrix compute engine)
    DPU  = 0x1001   # DPU (Elementwise/DPU unit)
    RDMA = 0x2001   # RDMA (Read DMA for inputs/weights)
    PC   = 0x0081   # PC (Program Control / operation enable)
    PC_REG = 0x0101 # PC chain registers
    VERSION = 0x0041

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
    CNA_CVT_CON5           = 0x1180   # CNA convert config 5 (mask)
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
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=task_count * 2, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
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
        E(reg.DPU,  reg.S_POINTER,
                ((1 << 3) |                           # DPU_S_POINTER_POINTER_PP_MODE
                (1 << 2)  |                           # DPU_S_POINTER_EXECUTER_PP_EN
                (1 << 1))                             # DPU_S_POINTER_POINTER_PP_EN
        ),
        E(reg.CNA,  reg.CNA_CONV_CON1,
                ((3 << 29) |                          # CNA_CONV_CON1_PIXEL/LINE mode
                (10 << 12) |                          # CNA_CONV_CON1_PIXEL_CHANNELS
                (2 << 7)  |                           # CNA_CONV_CON1_PROC_PRECISION(2)=fp16
                (2 << 4))                             # CNA_CONV_CON1_IN_PRECISION(2)=fp16
        ),
        E(reg.CNA,  reg.CNA_CONV_CON2,
                (5 << 4)                              # CNA_CONV_CON2_FEATURE_GRAINS, fetch 5 input lines for 4x4 1x1 conv
        ),
        E(reg.CNA,  reg.CNA_CONV_CON3,
                ((1 << 3) |                           # CNA_CONV_CON3_OPERATION_ENABLE
                1)                                    # CNA_CONV_CON3_WEIGHT_REUSE
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE0,
                ((4 << 16) |                          # CNA_DATA_SIZE0_WIDTH
                4)                                    # CNA_DATA_SIZE0_HEIGHT
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE1,
                ((2 << 16) |                          # CNA_DATA_SIZE1_CHANNEL
                8)                                    # CNA_DATA_SIZE1_ATOMICS
        ),
        E(reg.CNA,  reg.CNA_DATA_SIZE2,   4),  # CNA_DATA_SIZE2_OUTPUT_WIDTH
        E(reg.CNA,  reg.CNA_DATA_SIZE3,   16), # CNA_DATA_SIZE3_OUTPUT_ATOMICS
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE0, 32), # CNA_WEIGHT_SIZE0_TOTAL_BYTES
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE1, 16), # CNA_WEIGHT_SIZE1_BYTES_PER_KERNEL
        E(reg.CNA,  reg.CNA_WEIGHT_SIZE2,
                ((1 << 24) |                          # CNA_WEIGHT_SIZE2_KERNEL_WIDTH
                (1 << 16) |                           # CNA_WEIGHT_SIZE2_KERNEL_HEIGHT
                2)                                    # CNA_WEIGHT_SIZE2_KERNELS
        ),
        E(reg.CNA,  reg.CNA_CBUF_CON0,
                ((11 << 4) |                          # CNA_CBUF_CON0_WEIGHT_BANK
                1)                                    # CNA_CBUF_CON0_DATA_BANK
        ),
        E(reg.CNA,  reg.CNA_CBUF_CON1,    16), # CNA_CBUF_CON1_ENTRIES
        E(reg.CNA,  reg.CNA_CVT_CON0,     1),  # CNA_CVT_CON0_ENABLE
        E(reg.CNA,  reg.CNA_CVT_CON1,
                (1 << 16)                             # CNA_CVT_CON1_SCALE
        ),
        E(reg.CNA,  reg.CNA_CVT_CON2,
                (1 << 16)                             # CNA_CVT_CON2_SCALE
        ),
        E(reg.CNA,  reg.CNA_CVT_CON3,
                (1 << 16)                             # CNA_CVT_CON3_SCALE
        ),
        E(reg.CNA,  reg.CNA_CVT_CON4,
                (1 << 16)                             # CNA_CVT_CON4_SCALE
        ),
        E(reg.CNA,  reg.CNA_FEATURE_DATA_ADDR, in_dma & 0xFFFFFFFF),
        E(reg.CNA,  reg.CNA_DMA_CON0,
                ((15 << 16) |                         # CNA_DMA_CON0_READ_BURST
                15)                                   # CNA_DMA_CON0_WEIGHT_BURST
        ),
        E(reg.CNA,  reg.CNA_DMA_CON1,     4),  # CNA_DMA_CON1_LINE_STRIDE
        E(reg.CNA,  reg.CNA_DMA_CON2,     12), # CNA_DMA_CON2_SURF_STRIDE
        E(reg.CNA,  reg.CNA_FC_DATA_SIZE0,
                ((4 << 16) |                          # CNA_FC_DATA_SIZE0_WIDTH
                4)                                    # CNA_FC_DATA_SIZE0_HEIGHT
        ),
        E(reg.CNA,  reg.CNA_FC_DATA_SIZE1, 8), # CNA_FC_DATA_SIZE1_ALIGNED_CHANNEL
        E(reg.CNA,  reg.CNA_DCOMP_ADDR0,  (wt_dma + REGCMD_RESERVED) & 0xFFFFFFFF),
        E(reg.CNA,  reg.CNA_CVT_CON5,     7),  # CNA_CVT_CON5_ACTIVE_LANE_MASK
        E(reg.CORE, reg.CORE_MISC_CFG,
                (2 << 8)                              # CORE_MISC_CFG_PROC_PRECISION
        ),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0,
                ((3 << 16) |                          # CORE_DATAOUT_SIZE_0_HEIGHT
                3)                                    # CORE_DATAOUT_SIZE_0_WIDTH
        ),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, 15), # CORE_DATAOUT_SIZE_1_CHANNEL
        E(reg.CORE, reg.CORE_RESERVED_3030,  0),  # CORE_RESERVED_3030_ZERO
        E(reg.DPU,  reg.FEATURE_MODE_CFG,
                ((15 << 5) |                          # DPU_FEATURE_MODE_CFG_BYPASS
                (2 << 1))                             # DPU_FEATURE_MODE_CFG_MODE
        ),
        E(reg.DPU,  reg.DATA_FORMAT,
                ((2 << 29) |                          # DPU_DATA_FORMAT_OUT_PRECISION
                (2 << 26) |                           # DPU_DATA_FORMAT_PROC_PRECISION
                2)                                    # DPU_DATA_FORMAT_IN_PRECISION
        ),
        E(reg.DPU,  reg.DST_BASE_ADDR,     out_dma & 0xFFFFFFFF),
        E(reg.DPU,  reg.DST_SURF_STRIDE,
                (16 << 4)                             # DPU_DST_SURF_STRIDE
        ),
        E(reg.DPU,  reg.DATA_CUBE_WIDTH,   3), # DPU_DATA_CUBE_WIDTH
        E(reg.DPU,  reg.DATA_CUBE_HEIGHT,  3), # DPU_DATA_CUBE_HEIGHT
        E(reg.DPU,  reg.DATA_CUBE_CHANNEL,
                ((1 << 16) |                          # DPU_DATA_CUBE_CHANNEL_CUBE
                15)                                   # DPU_DATA_CUBE_CHANNEL_ATOMICS
        ),
        E(reg.DPU,  reg.BS_CFG,
                ((1 << 6) |                           # DPU_BS_CFG_BS_RELU_BYPASS
                (1 << 4)  |                           # DPU_BS_CFG_BS_MUL_BYPASS
                (1 << 1)  |                           # DPU_BS_CFG_BS_ALU_BYPASS
                1)                                    # DPU_BS_CFG_BS_BYPASS
        ),
        E(reg.DPU,  reg.BS_OW_CFG,
                ((1 << 8) |                           # DPU_BS_OW_CFG_SIZE_E_2
                (1 << 5)  |                           # DPU_BS_OW_CFG_SIZE_E_1
                (1 << 2)  |                           # DPU_BS_OW_CFG_SIZE_E_0
                (1 << 1))                             # DPU_BS_OW_CFG_OD_BYPASS
        ),
        E(reg.DPU,  reg.WDMA_SIZE_0,      15), # DPU_WDMA_SIZE_0_CHANNEL
        E(reg.DPU,  reg.WDMA_SIZE_1,
                ((3 << 16) |                          # DPU_WDMA_SIZE_1_HEIGHT
                3)                                    # DPU_WDMA_SIZE_1_WIDTH
        ),
        E(reg.DPU,  reg.BN_CFG,
                ((1 << 6) |                           # DPU_BN_CFG_BN_RELU_BYPASS
                (1 << 4)  |                           # DPU_BN_CFG_BN_MUL_BYPASS
                (1 << 1)  |                           # DPU_BN_CFG_BN_ALU_BYPASS
                1)                                    # DPU_BN_CFG_BN_BYPASS
        ),
        E(reg.DPU,  reg.EW_CFG,
                ((1 << 9) |                           # DPU_EW_CFG_EW_RELU_BYPASS
                (1 << 8)  |                           # DPU_EW_CFG_EW_OP_CVT_BYPASS
                (1 << 7)  |                           # DPU_EW_CFG_EW_LUT_BYPASS
                (1 << 1)  |                           # DPU_EW_CFG_EW_OP_BYPASS
                1)                                    # DPU_EW_CFG_EW_BYPASS
        ),
        E(reg.DPU,  0x4078,               1),  # DPU_EW_CVT_SCALE_VALUE
        E(reg.DPU,  reg.OUT_CVT_SCALE,
                ((1 << 16) |                          # DPU_OUT_CVT_SCALE_OFFSET
                1)                                    # DPU_OUT_CVT_SCALE_SCALE
        ),
        E(reg.DPU,  reg.SURFACE_ADD,
                (32 << 4)                             # DPU_SURFACE_ADD
        ),
        E(0x0001,          0x40c4,               0),  # DPU_RESERVED_40C4_ZERO
        E(reg.PC,   reg.OPERATION_ENABLE,
                ((6 << 1) |                           # PC_OPERATION_ENABLE_RESERVED_0 enables DPU/CNA/PPU
                1)                                    # PC_OPERATION_ENABLE_OP_EN
        ),
    ]
    return npu_regs

def write_regs_to_npu_task(task_regs):
    assert len(task_regs) == 1, "conv.py currently submits one decoded conv task"
    regs = task_regs[0]
    for i in range(64):
        npu_regcmd[i] = 0
    for i, qword in enumerate(regs):
        npu_regcmd[i] = qword

    npu_tasks[0].flags = 0
    npu_tasks[0].op_idx = 1
    npu_tasks[0].enable_mask = 0x0d
    npu_tasks[0].int_mask = 0x300
    npu_tasks[0].int_clear = 0x1ffff
    npu_tasks[0].int_status = 0
    npu_tasks[0].regcfg_amount = len(regs)
    npu_tasks[0].regcfg_offset = 0
    npu_tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

def run_conv2d(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    np.random.seed(42)
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
    weight_in_channels = in_channels // groups if groups > 0 else in_channels

    inp = np.random.randn(batch, in_channels, in_h, in_w).astype(np.float16)
    wt = np.random.randn(out_channels, weight_in_channels, kernel_h, kernel_w).astype(np.float16)
    result = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float16)

    oc_per_group = out_channels // groups if groups > 0 else out_channels
    ic_per_group = weight_in_channels
    for o in range(out_channels):
        group = o // oc_per_group if groups > 1 else 0
        for c_local in range(ic_per_group):
            c = group * ic_per_group + c_local
            for i in range(kernel_h):
                for j in range(kernel_w):
                    result[:, o] += inp[:, c, i:i+out_h, j:j+out_w] * wt[o, c_local, i, j]

    return result, inp, wt

def run_conv():
    input_nchw = (np.arange(1 * 2 * 4 * 4, dtype=np.float16).reshape(1, 2, 4, 4) / np.float16(8.0)).astype(np.float16)
    input_padded = np.zeros((1, 3, 4, 4), dtype=np.float16)
    input_padded[:, :2] = input_nchw
    weight = np.array([
        1.0, -0.5, 0.0,
        0.25, 0.75, 0.0,
    ], dtype=np.float16).reshape(2, 3, 1, 1)

    # pack_input_nhwc
    input_packed = input_padded.reshape(1, 3, 4, 4).transpose(0, 2, 3, 1).reshape(-1)

    # pack_weights_ohwi_c8
    weight_packed = np.zeros((2, 1, 1, 8), dtype=np.float16)
    weight_packed[:, :, :, :3] = weight.reshape(2, 3, 1, 1).transpose(0, 2, 3, 1)
    weight_packed = weight_packed.reshape(-1)
    ct_inputs = (ctypes.c_uint16 * len(input_packed)).from_buffer(input_map)
    ct_weights = (ctypes.c_uint16 * len(weight_packed)).from_buffer(weight_map, REGCMD_RESERVED)
    ct_inputs[:] = input_packed.view(np.uint16).tolist()
    ct_weights[:] = weight_packed.view(np.uint16).tolist()

    task_regs = [make_conv_regs(input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)]
    assert len(task_regs) <= tasks_mem_create.size // ctypes.sizeof(struct_rknpu_task), "task buffer too small"

    write_regs_to_npu_task(task_regs)
    npu_reset(fd)
    ret = npu_submit(tasks_mem_create.obj_addr, task_count=len(task_regs),
           flags=((1 << 0) |                          # RKNPU_JOB_PC
                  (1 << 1) |                          # RKNPU_JOB_BLOCK
                  (1 << 2)))                          # RKNPU_JOB_PINGPONG

    out_raw = np.frombuffer(output_map, dtype=np.float16, count=128).copy()
    # unpack_output_nc1hwc2
    got = out_raw[:128].reshape(1, 1, 1, 16, 8).transpose(0, 1, 4, 2, 3).reshape(1, 8, 1, 16)[:, :2].reshape(1, 2, 4, 4)
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

if __name__ == "__main__":
    test_cases = [
        # 1x1 kernels
        (2, 2, 1, 1, (4, 4), 1, "default 1x1"),
        (1, 6, 1, 1, (4, 4), 1, "ic=1 oc=6 1x1"),
        (3, 3, 1, 1, (4, 4), 1, "ic=3 1x1"),
        (4, 2, 1, 1, (4, 4), 1, "ic=4 1x1"),
        (4, 4, 1, 1, (9, 9), 1, "1x1 9x9"),
        (8, 8, 1, 1, (5, 5), 1, "ic=8 oc=8 1x1 5x5"),
        (16, 16, 1, 1, (8, 8), 1, "1x1 ic=16 oc=16 8x8"),
        (16, 16, 1, 1, (32, 32), 1, "1x1 32x32"),
        (10, 20, 3, 3, (9, 9), 1, "test_ops simple_conv2d_nhwc shape"),

        # Non-1x1 kernels
        (4, 4, 3, 3, (9, 9), 1, "simple 3x3 9x9"),
        (16, 16, 3, 3, (9, 9), 1, "3x3 m4"),
        (2, 4, 3, 3, (6, 6), 1, "3x3 oc!=ic"),
        (1, 6, 3, 3, (5, 7), 1, "ic=1 oc=6 3x3"),
        (16, 16, 3, 3, (18, 18), 1, "16x16 3x3 18x18"),
        (2, 4, 2, 2, (5, 5), 1, "2x2 kernel"),
        (1, 32, 5, 5, (10, 10), 1, "5x5 kernel"),
        (8, 4, 4, 4, (10, 10), 1, "4x4 kernel"),
        (3, 3, 3, 3, (11, 28), 3, "dw 3x3 11x28"),
        (1, 6, 3, 1, (5, 7), 1, "ic=1 oc=6 3x1"),
        (3, 6, 1, 3, (5, 5), 1, "1x3 kernel"),

        # test_ops coverage
        (3, 6, 3, 3, (5, 7), 3, "test_ops _test_conv2d cin=3 3x3 g=3"),
        (3, 6, 2, 1, (5, 7), 1, "test_ops _test_conv2d cin=3 2x1"),
        (3, 6, 2, 3, (5, 7), 1, "test_ops _test_conv2d cin=3 2x3"),
        (3, 6, 3, 1, (5, 7), 1, "test_ops _test_conv2d cin=3 3x1"),
        (3, 6, 3, 5, (5, 7), 1, "test_ops _test_conv2d cin=3 3x5"),
        (1, 6, 3, 3, (5, 7), 1, "test_ops _test_conv2d cin=1 3x3"),
        (1, 6, 2, 1, (5, 7), 1, "test_ops _test_conv2d cin=1 2x1"),
        (1, 6, 2, 3, (5, 7), 1, "test_ops _test_conv2d cin=1 2x3"),
        (1, 6, 3, 1, (5, 7), 1, "test_ops _test_conv2d cin=1 3x1"),
        (1, 6, 3, 5, (5, 7), 1, "test_ops _test_conv2d cin=1 3x5"),
        (4, 2, 1, 1, (1, 1), 2, "test_ops simple_grouped_conv2d"),
        (4, 4, 1, 1, (1, 1), 2, "test_ops medium_grouped_conv2d"),
        (32, 32, 1, 1, (32, 32), 32, "test_ops depthwise_conv2d"),
        (15, 35, 3, 3, (5, 5), 5, "test_ops grouped_conv2d"),
    ]

    for in_c, out_c, kh, kw, input_hw, groups, desc in test_cases:
        print(f"\n{desc}:")
        r, inp, wt = run_conv2d(in_c, out_c, kh, kw, input_hw, groups)
        b, oc, oh, ow = r.shape
        expected = np.zeros((b, oc, oh, ow), dtype=np.float16)
        oc_per_group = out_c // groups if groups > 0 else out_c
        ic_per_group = in_c // groups if groups > 0 else in_c
        for o in range(out_c):
            group = o // oc_per_group if groups > 1 else 0
            for c_local in range(ic_per_group):
                c = group * ic_per_group + c_local
                for i in range(kh):
                    for j in range(kw):
                        expected[:, o] += inp[:, c, i:i+oh, j:j+ow] * wt[o, c_local, i, j]
        ok = np.allclose(r, expected, atol=0.1) and not np.any(np.isinf(r))
        md = np.max(np.abs(r - expected))
        print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")
        assert ok, f"test shape {in_c, out_c, kh, kw, input_hw, groups} failed"
    os.close(fd)
