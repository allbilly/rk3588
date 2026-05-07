import os, mmap, ctypes, numpy as np, sys
from fcntl import ioctl
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experimental'))

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_SYNC_TO_DEVICE = 1 << 0
RKNPU_MEM_SYNC_FROM_DEVICE = 1 << 1
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
    EW_CVT_SCALE_VALUE  = 0x4078   # DPU elementwise conversion scale value
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

REG_NAMES = {
    value: name for name, value in vars(reg).items()
    if name.isupper() and isinstance(value, int)
}
TARGET_NAMES = {
    0x0000: "ZERO",
    reg.VERSION: "VERSION",
    reg.PC: "PC",
    reg.PC_REG: "PC_REG",
    reg.CNA: "CNA",
    reg.CORE: "CORE",
    reg.DPU: "DPU",
    reg.RDMA: "RDMA",
}

@dataclass(frozen=True)
class ConvRawbufTask:
    idx: int
    conv_con2: int
    data_size0: int
    data_size1: int
    data_size2: int
    data_size3: int
    weight_size0: int
    weight_size1: int
    weight_size2: int
    cbuf_con1: int
    input_offset: int
    fc_data_size0: int
    fc_data_size1: int
    weight_offset: int
    core_dataout_size0: int
    core_dataout_size1: int
    output_offset: int
    dst_surf_stride: int
    cube_width: int
    cube_height: int
    cube_channel: int
    wdma_size0: int
    wdma_size1: int
    surface_add: int

@dataclass(frozen=True)
class ConvRawbufCase:
    name: str
    task_offsets: tuple
    cna_dma_con1: int
    cna_dma_con2: int
    dpu_bs_ow_cfg: int
    tasks: tuple

CONV_RAWBUF_CASES = {
    1: ConvRawbufCase(
        name="small 1x3 RKNN conv pcchain",
        task_offsets=(0x940,),
        cna_dma_con1=0x08,
        cna_dma_con2=0x20,
        dpu_bs_ow_cfg=0x126,
        tasks=(
            ConvRawbufTask(0, 0x00000080, 0x00080005, 8, 7, 0x15, 0x120, 0x30, 0x01030006, 0x28, 0, 0x00070005, 8, 0, 0x00020006, 0x0F, 0, 0x180, 6, 2, 0x0005000F, 0x0F, 0x00020006, 0x300),
        ),
    ),
    4: ConvRawbufCase(
        name="large 5x5 RKNN conv pcchain",
        task_offsets=(0x3A00, 0x3D80, 0x4200, 0x4680, 0x4B00, 0x4F80),
        cna_dma_con1=0x10,
        cna_dma_con2=0x90,
        dpu_bs_ow_cfg=0x126,
        tasks=(
            ConvRawbufTask(0, 0x000000F0, 0x0010000A, 8, 6, 0x24, 0x3200, 0x190, 0x05050020, 0xA0, 0, 0x000A000A, 8, 0, 0x00050005, 0x1F, 0, 0x240, 5, 5, 0x001F001F, 0x1F, 0x00050005, 0x480),
            ConvRawbufTask(1, 0x400000F0, 0x0010000A, 8, 6, 0x24, 0x1900, 0x190, 0x05050010, 0xA0, 0, 0x000A000A, 8, 0, 0x00050005, 0x0F, 0, 0x240, 5, 5, 0x000F000F, 0x0F, 0x00050005, 0x480),
            ConvRawbufTask(3, 0x400000F0, 0x0010000A, 8, 6, 0x24, 0x1900, 0x190, 0x05050010, 0xA0, 0, 0x000A000A, 8, 0x1900, 0x00050005, 0x0F, 0x480, 0x240, 5, 5, 0x000F000F, 0x0F, 0x00050005, 0x480),
            ConvRawbufTask(5, 0x200000B0, 0x00100006, 8, 6, 0x0C, 0x3200, 0x190, 0x05050020, 0x60, 0, 0x000A0006, 8, 0, 0x00010005, 0x1F, 0, 0x240, 5, 1, 0x001F001F, 0x1F, 0x00010005, 0x480),
            ConvRawbufTask(7, 0x200000B0, 0x00100006, 8, 6, 0x0C, 0x3200, 0x190, 0x05050020, 0x60, 0x40, 0x000A0006, 8, 0, 0x00010005, 0x1F, 0xC0, 0x240, 5, 1, 0x001F001F, 0x1F, 0x00010005, 0x480),
            ConvRawbufTask(9, 0x200000B0, 0x00100006, 8, 6, 0x0C, 0x3200, 0x190, 0x05050020, 0x60, 0x80, 0x000A0006, 8, 0, 0x00010005, 0x1F, 0x180, 0x240, 5, 1, 0x001F001F, 0x1F, 0x00010005, 0x480),
        ),
    ),
}

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

class rknpu_mem_sync(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("obj_addr", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
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
DRM_IOCTL_RKNPU_MEM_SYNC = _IOWR('d', 0x45, ctypes.sizeof(rknpu_mem_sync))

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

def mem_sync(obj_addr, size, flags, offset=0):
    if size <= 0:
        return 0
    sync = rknpu_mem_sync(flags=flags, obj_addr=obj_addr, offset=offset, size=size)
    try:
        return ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, sync)
    except OSError:
        aligned_size = _align_up(size, 4096)
        if aligned_size == size:
            return -1
        sync.size = aligned_size
        try:
            return ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, sync)
        except OSError:
            return -1

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
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

fd = None
task_map = regcmd_map = input_map = weight_map = output_map = None
tasks_mem_create = regcmd_mem_create = input_mem_create = weight_mem_create = output_mem_create = None
npu_tasks = npu_regcmd = None

def ensure_npu_buffers():
    global fd, task_map, regcmd_map, input_map, weight_map, output_map
    global tasks_mem_create, regcmd_mem_create, input_mem_create, weight_mem_create, output_mem_create
    global npu_tasks, npu_regcmd
    if fd is not None:
        return
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, tasks_mem_create = mem_allocate(fd, size=64*1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem_create = mem_allocate(fd, size=512*1024, flags=RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
    npu_tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    npu_regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def close_npu_buffers():
    global fd
    if fd is not None:
        os.close(fd)
        fd = None

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

def _qword_fields(qword):
    return (qword >> 48) & 0xffff, qword & 0xffff, (qword >> 16) & 0xffffffff

def _format_decoded_reg(qword):
    target, addr, value = _qword_fields(qword)
    target_name = TARGET_NAMES.get(target, f"0x{target:04x}")
    reg_name = REG_NAMES.get(addr, f"REG_0x{addr:04x}")
    return target_name, reg_name, addr, value

def _rawbuf_task_amount(task):
    return 108 if task.idx == 0 else 104

def _rawbuf_task_amounts(case):
    return [_rawbuf_task_amount(task) for task in case.tasks]

def _rawbuf_conv_body(case, task, input_dma, weight_dma, output_dma):
    def Q(target, addr, value):
        return E(target, addr, value)

    regs = [
        Q(reg.DPU, 0x4004, 0x0000000E),
        Q(reg.CNA, 0x100C, 0x60008120),
        Q(reg.CNA, 0x1010, task.conv_con2),
        Q(reg.CNA, 0x1014, 0x00000009),
        Q(reg.CNA, 0x1020, task.data_size0),
        Q(reg.CNA, 0x1024, task.data_size1),
        Q(reg.CNA, 0x1028, task.data_size2),
        Q(reg.CNA, 0x102C, task.data_size3),
        Q(reg.CNA, 0x1030, task.weight_size0),
        Q(reg.CNA, 0x1034, task.weight_size1),
        Q(reg.CNA, 0x1038, task.weight_size2),
        Q(reg.CNA, 0x1040, 0x000000B1),
        Q(reg.CNA, 0x1044, task.cbuf_con1),
        Q(reg.CNA, 0x104C, 0x00000001),
        Q(reg.CNA, 0x1050, 0x00010000),
        Q(reg.CNA, 0x1054, 0x00010000),
        Q(reg.CNA, 0x1058, 0x00010000),
        Q(reg.CNA, 0x105C, 0x00010000),
        Q(reg.CNA, 0x1060, 0),
        Q(reg.CNA, 0x1064, 0),
        Q(reg.CNA, 0x1068, 0),
        Q(reg.CNA, 0x1070, input_dma + task.input_offset),
        Q(reg.CNA, 0x1074, 0),
        Q(reg.CNA, 0x1078, 0x000F000F),
        Q(reg.CNA, 0x107C, case.cna_dma_con1),
        Q(reg.CNA, 0x1080, case.cna_dma_con2),
        Q(reg.CNA, 0x1084, task.fc_data_size0),
        Q(reg.CNA, 0x1088, task.fc_data_size1),
        Q(reg.CNA, 0x1100, 0),
        Q(reg.CNA, 0x1104, 0),
        Q(reg.CNA, 0x1110, weight_dma + task.weight_offset),
        Q(reg.CNA, 0x1140, 0),
    ]
    regs.extend(Q(reg.CNA, 0x1144 + i * 4, 0) for i in range(15))
    regs.extend([
        Q(reg.CNA, 0x1180, 0x0000FFFF),
        Q(reg.CNA, 0x1184, 0),
        Q(reg.CORE, 0x3010, 0x00000200),
        Q(reg.CORE, 0x3014, task.core_dataout_size0),
        Q(reg.CORE, 0x3018, task.core_dataout_size1),
        Q(reg.CORE, 0x301C, 0),
        Q(reg.CORE, 0x3030, 0),
        Q(reg.DPU, 0x400C, 0x000001E4),
        Q(reg.DPU, 0x4010, 0x48000002),
        Q(reg.DPU, 0x4014, 0),
        Q(reg.DPU, 0x4020, output_dma + task.output_offset),
        Q(reg.DPU, 0x4024, task.dst_surf_stride),
        Q(reg.DPU, 0x4030, task.cube_width),
        Q(reg.DPU, 0x4034, task.cube_height),
        Q(reg.DPU, 0x4038, 0),
        Q(reg.DPU, 0x403C, task.cube_channel),
        Q(reg.DPU, 0x4040, 0x00000053),
        Q(reg.DPU, 0x4044, 0),
        Q(reg.DPU, 0x4048, 0),
        Q(reg.DPU, 0x404C, 0),
        Q(reg.DPU, 0x4050, case.dpu_bs_ow_cfg),
        Q(reg.DPU, 0x4054, 0),
        Q(reg.DPU, 0x4058, task.wdma_size0),
        Q(reg.DPU, 0x405C, task.wdma_size1),
        Q(reg.DPU, 0x4060, 0x00000053),
        Q(reg.DPU, 0x4064, 0),
        Q(reg.DPU, 0x4068, 0),
        Q(reg.DPU, 0x406C, 0),
        Q(reg.DPU, 0x4070, 0x00000383),
        Q(reg.DPU, 0x4074, 0),
        Q(reg.DPU, 0x4078, 1),
        Q(reg.DPU, 0x407C, 0),
        Q(reg.DPU, 0x4080, 0),
        Q(reg.DPU, 0x4084, 0x00010001),
        Q(reg.DPU, 0x4088, 0),
        Q(reg.DPU, 0x4090, 0),
        Q(reg.DPU, 0x4094, 0),
        Q(reg.DPU, 0x4098, 0),
        Q(reg.DPU, 0x409C, 0),
        Q(reg.DPU, 0x40A0, 0),
        Q(reg.DPU, 0x40A4, 0),
        Q(reg.DPU, 0x40A8, 0),
        Q(reg.DPU, 0x40AC, 0),
        Q(reg.DPU, 0x40C0, task.surface_add),
        Q(reg.DPU, 0x40C4, 0),
        Q(reg.DPU, 0x4100, 0),
        Q(reg.DPU, 0x4104, 0),
        Q(reg.DPU, 0x4108, 0),
        Q(reg.DPU, 0x410C, 0),
        Q(reg.DPU, 0x4110, 0),
        Q(reg.DPU, 0x4114, 0),
        Q(reg.DPU, 0x4118, 0),
        Q(reg.DPU, 0x411C, 0),
        Q(reg.DPU, 0x4120, 0),
        Q(reg.DPU, 0x4124, 0),
        Q(reg.DPU, 0x4128, 0),
        Q(reg.DPU, 0x412C, 0),
    ])
    if task.idx == 0:
        regs = [
            Q(reg.CNA, 0x1040, 0x000000B1),
            Q(reg.CNA, 0x1104, 0),
            Q(reg.CNA, 0x1100, 0),
            Q(reg.CNA, 0x100C, 0x60008120),
            Q(reg.DPU, 0x4004, 0x0000000E),
        ] + regs[1:]
    amount = _rawbuf_task_amount(task)
    if len(regs) != amount:
        raise RuntimeError(f"rawbuf task {task.idx} has {len(regs)} regs, expected {amount}")
    return regs

def _rawbuf_pc_tail(case, task_idx, regcmd_dma):
    amounts = _rawbuf_task_amounts(case)
    if task_idx + 1 < len(amounts):
        return [
            E(reg.PC_REG, reg.PC_BASE_ADDRESS, (regcmd_dma + case.task_offsets[task_idx + 1]) & 0xfffffff0),
            E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, (amounts[task_idx + 1] + 2) // 2),
            E(reg.VERSION, 0, 0),
            E(reg.PC, reg.OPERATION_ENABLE, 0x0d),
        ]
    return [
        0,
        E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        E(reg.VERSION, 0, 0),
        E(reg.PC, reg.OPERATION_ENABLE, 0x0d),
    ]

def build_conv_rawbuf_case_regs(case_id, input_dma=0xfff10000, weight_dma=0xfff20000,
                                output_dma=0xfff30000, regcmd_dma=0xfff40000):
    case = CONV_RAWBUF_CASES[case_id]
    return [
        _rawbuf_conv_body(case, task, input_dma, weight_dma, output_dma) +
        _rawbuf_pc_tail(case, task_idx, regcmd_dma)
        for task_idx, task in enumerate(case.tasks)
    ]

def dump_conv_rawbuf_case(case_id):
    case = CONV_RAWBUF_CASES[case_id]
    task_regs = build_conv_rawbuf_case_regs(case_id)
    print(f"conv_rawbuf case={case_id} {case.name} tasks={len(task_regs)}")
    for task_idx, regs in enumerate(task_regs):
        amount = _rawbuf_task_amount(case.tasks[task_idx])
        print(f"task[{task_idx}] offset=0x{case.task_offsets[task_idx]:04x} body_amount={amount} total_regs={len(regs)}")
        for idx, qword in enumerate(regs):
            target_name, reg_name, _addr, value = _format_decoded_reg(qword)
            print(f"  [{idx:3d}] {target_name}.{reg_name} = 0x{value:08x}")

def check_conv_rawbuf_case_parity(case_id):
    import conv_pcchain

    py_regs = build_conv_rawbuf_case_regs(case_id)
    ref_case = conv_pcchain.CASES[case_id]
    ref_regs = [
        body + conv_pcchain.pc_tail(task_idx, ref_case, 0xfff40000)
        for task_idx, body in enumerate(conv_pcchain.build_task_regs(
            ref_case, 0xfff10000, 0xfff20000, 0xfff30000))
    ]
    mismatches = []
    if len(py_regs) != len(ref_regs):
        mismatches.append(("task_count", len(py_regs), len(ref_regs)))
    for task_idx, (got, want) in enumerate(zip(py_regs, ref_regs)):
        if len(got) != len(want):
            mismatches.append((f"task[{task_idx}].len", len(got), len(want)))
        for reg_idx, (got_qword, want_qword) in enumerate(zip(got, want)):
            if got_qword != want_qword:
                mismatches.append((f"task[{task_idx}][{reg_idx}]", got_qword, want_qword))
    if mismatches:
        print(f"conv_rawbuf case={case_id} parity FAIL mismatches={len(mismatches)}")
        for label, got, want in mismatches[:20]:
            print(f"  {label}: got=0x{got:016x} want=0x{want:016x}")
        return False
    print(f"conv_rawbuf case={case_id} parity PASS tasks={len(py_regs)}")
    return True

def compare_conv2d_regs_to_rawbuf(case_id, in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1):
    p = compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)
    single = {}
    for qword in build_conv2d_regs(p, 0xfff10000, 0xfff20000, 0xfff30000):
        target, addr, value = _qword_to_tuple(qword)
        single[(target, addr)] = value
    raw_tasks = build_conv_rawbuf_case_regs(case_id)
    raw = {}
    for task_idx, regs in enumerate(raw_tasks):
        body_regs = regs[:_rawbuf_task_amount(CONV_RAWBUF_CASES[case_id].tasks[task_idx])]
        for qword in body_regs:
            target, addr, value = _qword_fields(qword)
            raw.setdefault((target, addr), []).append(value)

    print(f"single-task conv2d shape={(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)}")
    print(f"rawbuf case={case_id} tasks={len(raw_tasks)}")
    interesting = [
        (reg.CNA, reg.CNA_CONV_CON2),
        (reg.CNA, reg.CNA_DATA_SIZE0),
        (reg.CNA, reg.CNA_DATA_SIZE1),
        (reg.CNA, reg.CNA_DATA_SIZE2),
        (reg.CNA, reg.CNA_DATA_SIZE3),
        (reg.CNA, reg.CNA_WEIGHT_SIZE0),
        (reg.CNA, reg.CNA_WEIGHT_SIZE1),
        (reg.CNA, reg.CNA_WEIGHT_SIZE2),
        (reg.CNA, reg.CNA_CBUF_CON1),
        (reg.CNA, reg.CNA_DMA_CON1),
        (reg.CNA, reg.CNA_DMA_CON2),
        (reg.CNA, reg.CNA_FC_DATA_SIZE0),
        (reg.CNA, reg.CNA_FC_DATA_SIZE1),
        (reg.CORE, reg.CORE_DATAOUT_SIZE_0),
        (reg.CORE, reg.CORE_DATAOUT_SIZE_1),
        (reg.DPU, reg.DST_SURF_STRIDE),
        (reg.DPU, reg.DATA_CUBE_WIDTH),
        (reg.DPU, reg.DATA_CUBE_HEIGHT),
        (reg.DPU, reg.DATA_CUBE_CHANNEL),
        (reg.DPU, reg.WDMA_SIZE_0),
        (reg.DPU, reg.WDMA_SIZE_1),
        (reg.DPU, reg.SURFACE_ADD),
    ]
    for key in interesting:
        target, addr = key
        target_name = TARGET_NAMES.get(target, f"0x{target:04x}")
        reg_name = REG_NAMES.get(addr, f"REG_0x{addr:04x}")
        got = single.get(key)
        wants = raw.get(key, [])
        want_text = ",".join(f"0x{x:08x}" for x in wants[:8])
        if len(wants) > 8:
            want_text += ",..."
        got_text = "None" if got is None else f"0x{got:08x}"
        verdict = "same" if len(set(wants)) == 1 and got == wants[0] else "diff"
        print(f"{target_name}.{reg_name:<22} single={got_text:<10} rawbuf=[{want_text}] {verdict}")

def dump_reg_tuples(title, regs):
    print(title)
    for idx, qword in enumerate(regs):
        target_name, reg_name, _addr, decoded_value = _format_decoded_reg(qword)
        print(f"  [{idx:3d}] {target_name}.{reg_name} = 0x{decoded_value:08x}")

def dump_conv2d_case_regs(batch, in_c, out_c, kh, kw, input_hw, groups, desc):
    p = compute_conv2d_params(in_c, out_c, kh, kw, input_hw, groups, batch=batch)
    regs = build_conv2d_regs(p, 0, 0, 0)
    print(f"conv2d {desc}: in=({batch},{in_c},{input_hw[0]},{input_hw[1]}) "
          f"weight=({out_c},{in_c // groups if groups > 0 else in_c},{kh},{kw}) "
          f"out=({batch},{out_c},{p['out_h']},{p['out_w']})")
    dump_reg_tuples("decoded conv2d regs:", regs)

def dump_conv1d_case_regs(batch, in_channels, input_size, out_channels, weight_in_channels,
                          kernel_size, groups, desc):
    p = _compute_conv1d_params(input_size, kernel_size, in_channels, out_channels)
    regs = build_conv1d_regs(p, 0, 0, 0)
    print(f"conv1d {desc}: in=({batch},{in_channels},{input_size}) "
          f"weight=({out_channels},{weight_in_channels},{kernel_size}) "
          f"out=({batch},{out_channels},{p['output_width']}) groups={groups}")
    dump_reg_tuples("decoded conv1d regs:", regs)

NPU_CBUF_BANK_SIZE = 32768
NPU_CBUF_BANKS = 12
OUTPUT_CHANNEL_CHUNK = 16
MAX_SAFE_SPATIAL_SUBMITS = 100
DIRECT_SPATIAL = os.environ.get("CONV_DIRECT_SPATIAL") == "1"
UNSAFE_DIRECT_SPATIAL = os.environ.get("CONV_UNSAFE_DIRECT_SPATIAL") == "1"
ALLOW_BROAD_SUBMIT = os.environ.get("CONV_ALLOW_BROAD_SUBMIT") == "1"
SUBMIT_MODE = "--submit" in sys.argv
DRY_RUN = not SUBMIT_MODE
VALIDATE = "--validate" in sys.argv

def align_up_int(x, align):
    return _align_up(x, align)

def _target(reg_addr):
    if 0x1000 <= reg_addr < 0x2000:
        return reg.CNA
    if 0x3000 <= reg_addr < 0x4000:
        return reg.CORE
    if 0x4000 <= reg_addr < 0x5000:
        return reg.DPU
    if reg_addr < 0x1000:
        return reg.PC
    raise ValueError(f"unknown register address 0x{reg_addr:x}")

def _conv_override_key(batch, in_c, in_h, in_w, out_c, kh, kw, groups):
    return (batch, in_c, in_h, in_w, out_c, kh, kw, groups)

DIRECT_SPATIAL_SUBMIT_ALLOWLIST = {
    (1, 1, 5, 7, 6, 3, 1, 1),
    (1, 3, 5, 7, 6, 3, 5, 1),
}

def _direct_spatial_key(batch, in_c, input_hw, out_c, kh, kw, groups):
    return (batch, in_c, input_hw[0], input_hw[1], out_c, kh, kw, groups)

def _check_direct_spatial_submit_allowed(batch, in_c, out_c, kh, kw, input_hw, groups):
    key = _direct_spatial_key(batch, in_c, input_hw, out_c, kh, kw, groups)
    if UNSAFE_DIRECT_SPATIAL or key in DIRECT_SPATIAL_SUBMIT_ALLOWLIST:
        return
    raise SystemExit(
        f"blocked unsafe direct-spatial hardware submit for shape {key}; "
        "run without --submit for dry register/test coverage, or set "
        "CONV_UNSAFE_DIRECT_SPATIAL=1 for one targeted experiment after register audit")

_CONV2D_OVERRIDES = {
    (1, 16, 18, 18, 16, 3, 3, 1): {
        "align_c": 16,
        "width_stride": "in_w",
        "out_width_stride": 256,
        "input_pack_c2": 8,
    },
    (1, 15, 5, 5, 35, 3, 3, 5): {
        "width_stride": "in_w",
        "out_width_stride": 12,
    },
    (1, 3, 11, 28, 3, 3, 3, 3): {
        "nhwc_pack": False,
        "width_stride": "in_w",
    },
    (1, 1, 5, 7, 6, 3, 3, 1): {
        "input_pack_c2": 2,
    },
    (1, 10, 9, 9, 20, 3, 3, 1): {
        "cbuf_entries": 8,
    },
}

def _get_conv_override(batch, in_c, in_h, in_w, out_c, kh, kw, groups):
    return _CONV2D_OVERRIDES.get(_conv_override_key(batch, in_c, in_h, in_w, out_c, kh, kw, groups), {})

def should_use_nhwc_pack(batch, channels, height, width, width_stride, c2,
                         out_c=None, kh=None, kw=None, groups=None):
    c_ratio = c2 // channels if channels > 0 else 0
    use_nhwc = (c_ratio == 2) and (width_stride >= width)
    if use_nhwc and all(x is not None for x in (out_c, kh, kw, groups)):
        override = _get_conv_override(batch, channels, height, width, out_c, kh, kw, groups)
        if "nhwc_pack" in override:
            use_nhwc = override["nhwc_pack"]
    return use_nhwc

_KH_MAJOR_SHAPES = {
    (6, 3, 2, 1): 1,
    (6, 3, 2, 3): 1,
    (6, 3, 2, 5): 1,
    (6, 3, 3, 1): None,
    (6, 3, 3, 3): None,
    (6, 3, 3, 5): None,
    (16, 16, 3, 3): 1,
    (4, 4, 3, 3): 1,
    (6, 1, 3, 3): 1,
    (6, 1, 2, 1): 1,
    (6, 1, 2, 3): 1,
    (6, 1, 3, 1): 1,
    (6, 1, 3, 5): 1,
    (6, 1, 1, 3): 1,
    (6, 1, 1, 5): 1,
}

def _is_kh_major(out_c, in_c, kh, kw, groups):
    key = (out_c, in_c, kh, kw)
    required = _KH_MAJOR_SHAPES.get(key)
    if required is None:
        return key in _KH_MAJOR_SHAPES
    return required == groups

def _pack_depthwise_expanded_weights_fp16(src, out_c, in_c, kh, kw, packed_weight_size):
    dst = np.zeros(packed_weight_size // FP16_BYTES, dtype=np.float16)
    slot = kh * kw
    for oc in range(out_c):
        src_base = oc * in_c * slot
        dst_base = oc * slot
        dst[dst_base:dst_base + slot] = src[src_base:src_base + slot]
    return dst

def _pack_dw_spatial_major(src, out_c, in_c, kh, kw, c2_out):
    dst = np.zeros(out_c * kh * kw * c2_out, dtype=np.float16)
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            dst_base = (kh_idx * kw + kw_idx) * c2_out
            for oc in range(out_c):
                src_idx = (((oc * in_c + oc) * kh) + kh_idx) * kw + kw_idx
                dst[dst_base + oc] = src[src_idx]
    return dst

def _pack_kh_major(src, out_c, in_c, kh, kw, c2_out):
    spatial_stride = c2_out * _ceil_div(in_c, c2_out)
    dst = np.zeros(kh * kw * out_c * spatial_stride, dtype=np.float16)
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            dst_khkw_base = (kh_idx * kw + kw_idx) * out_c * spatial_stride
            for oc in range(out_c):
                dst_spatial_base = dst_khkw_base + oc * spatial_stride
                for ic in range(in_c):
                    dst_idx = dst_spatial_base + (ic // c2_out) * c2_out + (ic % c2_out)
                    src_idx = (((oc * in_c + ic) * kh) + kh_idx) * kw + kw_idx
                    dst[dst_idx] = src[src_idx]
    return dst

def _pack_default(src, out_c, in_c, kh, kw, c2_out):
    spatial_stride = c2_out * _ceil_div(in_c, c2_out)
    kernel_stride = kh * kw * spatial_stride
    dst = np.zeros(out_c * kernel_stride, dtype=np.float16)
    for oc in range(out_c):
        dst_kernel_base = oc * kernel_stride
        for kh_idx in range(kh):
            for kw_idx in range(kw):
                dst_spatial_base = dst_kernel_base + (kh_idx * kw + kw_idx) * spatial_stride
                for ic in range(in_c):
                    dst_idx = dst_spatial_base + (ic // c2_out) * c2_out + (ic % c2_out)
                    src_idx = (((oc * in_c + ic) * kh) + kh_idx) * kw + kw_idx
                    dst[dst_idx] = src[src_idx]
    return dst

def pack_conv_weights_fp16(src, out_c, in_c, kh, kw, c2_out, groups=1):
    is_depthwise = (groups == in_c and out_c == in_c)
    if is_depthwise and out_c <= c2_out and kh == 3 and kw == 3:
        return _pack_dw_spatial_major(src, out_c, in_c, kh, kw, c2_out)
    if _is_kh_major(out_c, in_c, kh, kw, groups):
        return _pack_kh_major(src, out_c, in_c, kh, kw, c2_out)
    return _pack_default(src, out_c, in_c, kh, kw, c2_out)

def pack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride,
                      out_c=None, kh=None, kw=None, groups=None, use_nhwc=None):
    if use_nhwc is None:
        use_nhwc = should_use_nhwc_pack(batch, channels, height, width, width_stride, c2,
                                        out_c=out_c, kh=kh, kw=kw, groups=groups)
    src = np.asarray(src, dtype=np.float16).reshape(batch, channels, height, width)
    if use_nhwc:
        dst = np.zeros((batch, height, width_stride, channels), dtype=np.float16)
        dst[:, :, :width, :] = src.transpose(0, 2, 3, 1)
        return dst.reshape(-1)
    c1 = _ceil_div(channels, c2)
    dst = np.zeros((batch, c1, height, width_stride, c2), dtype=np.float16)
    for c in range(channels):
        dst[:, c // c2, :, :width, c % c2] = src[:, c]
    return dst.reshape(-1)

def unpack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride):
    src = np.asarray(src, dtype=np.float16)
    c1 = _ceil_div(channels, c2)
    packed = src[:batch * c1 * height * width_stride * c2].reshape(batch, c1, height, width_stride, c2)
    dst = np.zeros((batch, channels, height, width), dtype=np.float16)
    for c in range(channels):
        dst[:, c] = packed[:, c // c2, :, :width, c % c2]
    return dst.reshape(-1)

def _output_unpack_c2(params):
    if params["is_depthwise"]:
        return params["align_c"]
    return 8 if params["align_out_c"] >= 8 else params["align_out_c"]

def compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1, batch=1):
    batch, in_h, in_w = batch, input_hw[0], input_hw[1]
    weight_in_channels = in_channels // groups if groups > 0 else in_channels
    is_depthwise = (groups == in_channels and out_channels == in_channels)
    out_h = in_h - kernel_h + 1
    out_w = in_w - kernel_w + 1

    align_c = 32 if is_depthwise and in_channels > 8 else 8
    align_out_c = max(16, _align_up(out_channels, 16))

    width_stride = _align_up(in_w, align_c)
    out_atoms = max(1, out_w * out_h)
    out_width_stride = (out_w * align_out_c) // 4
    if in_channels == 3 and out_channels == 6:
        if groups == 1 and kernel_h == 3 and kernel_w == 1:
            out_width_stride = 24
        if kernel_h == 3 and kernel_w == 3:
            out_width_stride = 16
    if kernel_h == 1 and kernel_w == 1 and out_atoms < 4:
        out_width_stride = out_atoms
    elif kernel_h == 1 and kernel_w == 1:
        out_width_stride = _align_up(out_atoms, 4)

    override = _get_conv_override(batch, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w, groups)
    if "align_c" in override:
        align_c = override["align_c"]
    if "width_stride" in override:
        width_stride = in_w if override["width_stride"] == "in_w" else override["width_stride"]
    if "out_width_stride" in override:
        out_width_stride = override["out_width_stride"]

    out_channel_field = (_align_up(align_out_c, 32) if is_depthwise else align_out_c) - 1
    orig_channel = out_channels - 1 if out_channels > 0 else 0
    data_in_channel_real = in_channels - 1 if in_channels > 0 else 0
    data_in_channel_aligned = max(align_c, _align_up(in_channels, align_c))
    weight_kernels = 1 if is_depthwise else out_channels
    weight_bytes_per_kernel = kernel_h * kernel_w * data_in_channel_aligned * FP16_BYTES
    weight_bytes_total = weight_bytes_per_kernel * out_channels

    feature_grains = in_h + kernel_h
    row_bytes = width_stride * align_c * FP16_BYTES
    if row_bytes > 0:
        max_grains = _ceil_div(2 * NPU_CBUF_BANK_SIZE, row_bytes)
        max_grains = (max(max_grains, 2) + 1) & ~1
        feature_grains = min(feature_grains, max_grains)

    input_pack_c2 = override.get("input_pack_c2", align_c)
    if (batch == 1 and in_channels == 16 and in_h == 18 and in_w == 18 and
            out_channels == 16 and kernel_h == 3 and kernel_w == 3):
        input_pack_c2 = 8
    if (batch == 1 and groups == 1 and in_channels == 1 and in_h == 5 and in_w == 7 and
            out_channels == 6):
        input_pack_c2 = 2
    use_nhwc = should_use_nhwc_pack(batch, in_channels, in_h, in_w, width_stride, input_pack_c2,
                                    out_c=out_channels, kh=kernel_h, kw=kernel_w, groups=groups)

    line_stride = width_stride if use_nhwc else width_stride * 4
    surf_stride = 0
    if use_nhwc and in_h > 1:
        surf_stride = line_stride * (in_h - 1)
    elif not use_nhwc and in_h > 4:
        surf_stride = width_stride * (in_h - 4)

    cvt_active = in_channels if use_nhwc else input_pack_c2
    cvt_active = max(1, min(cvt_active, 8))
    cvt_mask = (1 << cvt_active) - 1
    row_entries = max(1, _ceil_div(width_stride * align_c, 32))
    cbuf_entries = row_entries if (align_c >= 16 or is_depthwise) else row_entries * in_h * 4
    if "cbuf_entries" in override:
        cbuf_entries = override["cbuf_entries"]
    cbuf_entries = max(1, cbuf_entries)

    fd_bytes = width_stride * feature_grains * align_c * FP16_BYTES
    data_bank = _ceil_div(fd_bytes, NPU_CBUF_BANK_SIZE)
    data_bank = max(1, min(NPU_CBUF_BANKS - 1, data_bank))

    effective_align_out = out_channel_field + 1
    if groups > 1 and not is_depthwise:
        per_group_out = _ceil_div(out_channels, groups)
        effective_align_out = max(16, _align_up(per_group_out, 16))
    surface_add = out_width_stride * (effective_align_out // 8)

    return locals()

def build_conv2d_regs(params, input_dma=0, weights_dma=0, output_dma=0):
    p = params
    def Q(reg_addr, value):
        if 0x1000 <= reg_addr < 0x2000:
            return E(reg.CNA, reg_addr, value)
        if 0x3000 <= reg_addr < 0x4000:
            return E(reg.CORE, reg_addr, value)
        if 0x4000 <= reg_addr < 0x5000:
            return E(reg.DPU, reg_addr, value)
        if reg_addr < 0x1000:
            return E(reg.PC, reg_addr, value)
        raise ValueError(f"unknown register address 0x{reg_addr:x}")

    npu_regs = [
        Q(reg.S_POINTER,
            (1 << 3) |                             # DPU_S_POINTER_POINTER_PP_MODE
            (1 << 2) |                             # DPU_S_POINTER_EXECUTER_PP_EN
            (1 << 1)                               # DPU_S_POINTER_POINTER_PP_EN
        ),
    ]
    conv_con1 = (2 << 7) | (2 << 4)                # CNA_CONV_CON1_PROC_PRECISION(2)=fp16 | IN_PRECISION(2)=fp16
    if (p["in_channels"] >= 1 and p["in_channels"] <= 4) and not p["is_depthwise"]:
        conv_con1 |= (1 << 30) | (1 << 29) | ((7 + p["in_channels"]) << 12)  # NONALIGN_DMA | GROUP_LINE_OFF | ARGB_IN
    if p["is_depthwise"]:
        conv_con1 |= 3                             # CNA_CONV_CON1_CONV_MODE(3)
    npu_regs.append(Q(reg.CNA_CONV_CON1, conv_con1))
    npu_regs.append(Q(reg.CNA_CONV_CON2,
        (p["feature_grains"] << 4)                 # CNA_CONV_CON2_FEATURE_GRAINS
    ))
    npu_regs.append(Q(reg.CNA_CONV_CON3,
        (1 << 3) |                                 # CNA_CONV_CON3_OPERATION_ENABLE
        1                                          # CNA_CONV_CON3_WEIGHT_REUSE
    ))
    npu_regs.append(Q(reg.CNA_DATA_SIZE0,
        (p["width_stride"] << 16) |                # CNA_DATA_SIZE0_DATAIN_WIDTH
        p["in_h"]                                  # CNA_DATA_SIZE0_DATAIN_HEIGHT
    ))
    npu_regs.append(Q(reg.CNA_DATA_SIZE1,
        (p["data_in_channel_real"] << 16) |        # CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL
        p["data_in_channel_aligned"]               # CNA_DATA_SIZE1_DATAIN_CHANNEL
    ))
    npu_regs.append(Q(reg.CNA_DATA_SIZE2, p["out_w"]))        # CNA_DATA_SIZE2_DATAOUT_WIDTH
    npu_regs.append(Q(reg.CNA_DATA_SIZE3, p["out_atoms"]))    # CNA_DATA_SIZE3_DATAOUT_ATOMICS
    npu_regs.append(Q(reg.CNA_WEIGHT_SIZE0, p["weight_bytes_total"]))  # CNA_WEIGHT_SIZE0
    npu_regs.append(Q(reg.CNA_WEIGHT_SIZE1,
        p["weight_bytes_per_kernel"]               # CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL
    ))
    npu_regs.append(Q(reg.CNA_WEIGHT_SIZE2,
        (p["kernel_w"] << 24) |                    # CNA_WEIGHT_SIZE2_WEIGHT_WIDTH
        (p["kernel_h"] << 16) |                    # CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT
        p["weight_kernels"]                        # CNA_WEIGHT_SIZE2_WEIGHT_KERNELS
    ))
    npu_regs.append(Q(reg.CNA_CBUF_CON0,
        ((NPU_CBUF_BANKS - p["data_bank"]) << 4) | # CNA_CBUF_CON0_WEIGHT_BANK
        p["data_bank"]                              # CNA_CBUF_CON0_DATA_BANK
    ))
    npu_regs.append(Q(reg.CNA_CBUF_CON1,
        p["cbuf_entries"]                          # CNA_CBUF_CON1_DATA_ENTRIES
    ))
    npu_regs.append(Q(reg.CNA_CVT_CON0,
        1 if p["use_nhwc"] else 0xb                # CNA_CVT_CON0: bypass=1, else DATA_SIGN|CVT_TYPE
    ))
    for addr in (reg.CNA_CVT_CON1, reg.CNA_CVT_CON2, reg.CNA_CVT_CON3, reg.CNA_CVT_CON4):
        npu_regs.append(Q(addr, 1 << 16))           # CNA_CVT_CON{1..4}_SCALE
    npu_regs.append(Q(reg.CNA_FEATURE_DATA_ADDR,
        input_dma & 0xFFFFFFFF                      # CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR
    ))
    npu_regs.append(Q(reg.CNA_DMA_CON0,
        (15 << 16) |                               # CNA_DMA_CON0_WEIGHT_BURST_LEN
        15                                         # CNA_DMA_CON0_DATA_BURST_LEN
    ))
    npu_regs.append(Q(reg.CNA_DMA_CON1, p["line_stride"]))   # CNA_DMA_CON1_LINE_STRIDE
    npu_regs.append(Q(reg.CNA_DMA_CON2, p["surf_stride"]))   # CNA_DMA_CON2_SURF_STRIDE
    npu_regs.append(Q(reg.CNA_FC_DATA_SIZE0,
        (p["in_w"] << 16) |                        # CNA_FC_DATA_SIZE0_DMA_WIDTH
        p["in_h"]                                  # CNA_FC_DATA_SIZE0_DMA_HEIGHT
    ))
    npu_regs.append(Q(reg.CNA_FC_DATA_SIZE1,
        p["align_c"]                               # CNA_FC_DATA_SIZE1_DMA_CHANNEL
    ))
    npu_regs.append(Q(reg.CNA_DCOMP_ADDR0,
        (weights_dma + REGCMD_RESERVED) & 0xFFFFFFFF  # CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0
    ))
    npu_regs.append(Q(reg.CNA_CVT_CON5, p["cvt_mask"]))     # CNA_CVT_CON5_ACTIVE_LANE_MASK
    core_misc = (2 << 8)                            # CORE_MISC_CFG_PROC_PRECISION
    if p["is_depthwise"]:
        core_misc |= (1 << 1)                       # CORE_MISC_CFG_DW_EN
    npu_regs.append(Q(reg.CORE_MISC_CFG, core_misc))
    npu_regs.append(Q(reg.CORE_DATAOUT_SIZE_0,
        ((p["out_h"] - 1) << 16) |                 # CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT
        (p["out_w"] - 1)                           # CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH
    ))
    npu_regs.append(Q(reg.CORE_DATAOUT_SIZE_1,
        p["out_channel_field"]                     # CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL
    ))
    npu_regs.append(Q(reg.CORE_RESERVED_3030, 0))
    feature_mode_cfg = (15 << 5) | (2 << 1)         # DPU_FEATURE_MODE_CFG_BURST_LEN(15) | OUTPUT_MODE(2)
    if p["is_depthwise"]:
        feature_mode_cfg |= (3 << 3)                # DPU_FEATURE_MODE_CFG_CONV_MODE(3)
    npu_regs.append(Q(reg.FEATURE_MODE_CFG, feature_mode_cfg))
    npu_regs.append(Q(reg.DATA_FORMAT,
        (2 << 29) |                                # DPU_DATA_FORMAT_OUT_PRECISION
        (2 << 26) |                                # DPU_DATA_FORMAT_PROC_PRECISION
        2                                          # DPU_DATA_FORMAT_IN_PRECISION
    ))
    npu_regs.append(Q(reg.DST_BASE_ADDR, output_dma & 0xFFFFFFFF))  # DPU_DST_BASE_ADDR
    npu_regs.append(Q(reg.DST_SURF_STRIDE,
        (p["out_width_stride"] << 4)               # DPU_DST_SURF_STRIDE (x16 bytes)
    ))
    npu_regs.append(Q(reg.DATA_CUBE_WIDTH, p["out_w"] - 1))              # DPU_DATA_CUBE_WIDTH
    npu_regs.append(Q(reg.DATA_CUBE_HEIGHT, p["out_h"] - 1))             # DPU_DATA_CUBE_HEIGHT
    npu_regs.append(Q(reg.DATA_CUBE_CHANNEL,
        (p["orig_channel"] << 16) |                # DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL
        p["out_channel_field"]                     # DPU_DATA_CUBE_CHANNEL_CHANNEL
    ))
    npu_regs.append(Q(reg.BS_CFG,
        (1 << 6) | (1 << 4) | (1 << 1) | 1         # DPU_BS_CFG: all bypass
    ))
    ow_cfg = 3 if p["is_depthwise"] else 1
    npu_regs.append(Q(reg.BS_OW_CFG,
        (ow_cfg << 8) | (ow_cfg << 5) | (ow_cfg << 2) | (1 << 1)  # DPU_BS_OW_CFG_SIZE_E_{2,1,0} | OD_BYPASS
    ))
    npu_regs.append(Q(reg.WDMA_SIZE_0, p["out_channel_field"]))          # DPU_WDMA_SIZE_0_CHANNEL_WDMA
    npu_regs.append(Q(reg.WDMA_SIZE_1,
        ((p["out_h"] - 1) << 16) |                # DPU_WDMA_SIZE_1_HEIGHT_WDMA
        (p["out_w"] - 1)                          # DPU_WDMA_SIZE_1_WIDTH_WDMA
    ))
    npu_regs.append(Q(reg.BN_CFG,
        (1 << 6) | (1 << 4) | (1 << 1) | 1         # DPU_BN_CFG: all bypass
    ))
    npu_regs.append(Q(reg.EW_CFG,
        (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1  # DPU_EW_CFG: all bypass
    ))
    npu_regs.append(Q(reg.EW_CVT_SCALE_VALUE, 1))       # DPU_EW_CVT_SCALE_VALUE
    npu_regs.append(Q(reg.OUT_CVT_SCALE,
        (1 << 16) |                                 # DPU_OUT_CVT_SCALE_FP32TOFP16_EN
        1                                           # DPU_OUT_CVT_SCALE_OUT_CVT_SCALE
    ))
    npu_regs.append(Q(reg.SURFACE_ADD,
        (p["surface_add"] << 4)                     # DPU_SURFACE_ADD_SURF_ADD (x16 bytes)
    ))
    npu_regs.append(E(0x0001, 0x40c4, 0))           # DPU_RESERVED_40C4_ZERO
    npu_regs.append(E(reg.PC, reg.OPERATION_ENABLE,
        (6 << 1) |                                  # PC_OPERATION_ENABLE_RESERVED_0 (DPU|CNA|PPU)
        1                                           # PC_OPERATION_ENABLE_OP_EN
    ))
    return npu_regs

def _compute_conv1d_params(input_width, kernel_width, in_channels, out_channels):
    output_width = input_width - kernel_width + 1
    if output_width <= 0:
        raise ValueError(f"invalid conv1d output width for input={input_width} kernel={kernel_width}")
    data_in_channel = max(8, _align_up(in_channels, 8))
    input_width_aligned = input_width
    if in_channels > 1:
        input_width_aligned = max(8, _align_up(input_width, 8))
    out_channel_align = max(16, _align_up(out_channels, 16))
    dst_stride = _align_up(output_width, 4)
    kernel_bytes_per_kernel = kernel_width * data_in_channel * FP16_BYTES
    padded_kernel_bytes = _align_up(kernel_bytes_per_kernel, 16)
    weight_bytes_total = padded_kernel_bytes * out_channels
    surface_add = dst_stride * 2
    return locals()

def build_conv1d_regs(params, input_dma=0, weights_dma=0, output_dma=0):
    p = params
    regs = []
    def A(reg_addr, value):
        regs.append((_target(reg_addr), value, reg_addr))

    conv_con1 = (2 << 7) | (2 << 4)
    if p["input_width_aligned"] != p["input_width"] or p["in_channels"] > 1:
        conv_con1 |= (1 << 30) | (1 << 29) | (10 << 12)

    A(reg.CNA_CBUF_CON0, (11 << 4) | 1)
    A(reg.CNA_CONV_CON1, conv_con1)
    A(reg.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1))
    A(reg.CNA_CONV_CON2, 2 << 4)
    A(reg.CNA_CONV_CON3, (1 << 3) | 1)
    A(reg.CNA_DATA_SIZE0, (p["input_width_aligned"] << 16) | 1)
    if p["in_channels"] > 1:
        A(reg.CNA_DATA_SIZE1, ((p["in_channels"] - 1) << 16) | p["data_in_channel"])
    else:
        A(reg.CNA_DATA_SIZE1, p["data_in_channel"])
    A(reg.CNA_DATA_SIZE2, p["output_width"])
    A(reg.CNA_DATA_SIZE3, p["output_width"])
    A(reg.CNA_WEIGHT_SIZE0, p["weight_bytes_total"])
    A(reg.CNA_WEIGHT_SIZE1, p["padded_kernel_bytes"])
    A(reg.CNA_WEIGHT_SIZE2, (p["kernel_width"] << 24) | (1 << 16) | p["out_channels"])
    A(reg.CNA_CBUF_CON0, (11 << 4) | 1)
    A(reg.CNA_CBUF_CON1, 16)
    A(reg.CNA_CVT_CON0, 1)
    for addr in (reg.CNA_CVT_CON1, reg.CNA_CVT_CON2, reg.CNA_CVT_CON3, reg.CNA_CVT_CON4):
        A(addr, 1 << 16)
    A(reg.CNA_FEATURE_DATA_ADDR, input_dma & 0xFFFFFFFF)
    A(reg.CNA_DMA_CON0, (15 << 16) | 15)
    A(reg.CNA_DMA_CON1, p["input_width_aligned"])
    A(reg.CNA_FC_DATA_SIZE0, (p["input_width"] << 16) | 1)
    A(reg.CNA_FC_DATA_SIZE1, p["data_in_channel"])
    A(reg.CNA_DCOMP_ADDR0, (weights_dma + REGCMD_RESERVED) & 0xFFFFFFFF)
    A(reg.CNA_CVT_CON5, 0x00000fff if p["in_channels"] > 1 else 0)
    A(reg.CORE_MISC_CFG, 2 << 8)
    A(reg.CORE_DATAOUT_SIZE_0, p["output_width"] - 1)
    A(reg.CORE_DATAOUT_SIZE_1, p["out_channel_align"] - 1)
    A(reg.CORE_RESERVED_3030, 0)
    A(reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1))
    A(reg.DATA_FORMAT, (2 << 29) | (2 << 26) | 2)
    A(reg.DST_BASE_ADDR, output_dma & 0xFFFFFFFF)
    A(reg.DST_SURF_STRIDE, p["dst_stride"] << 4)
    A(reg.DATA_CUBE_WIDTH, p["output_width"] - 1)
    A(reg.DATA_CUBE_CHANNEL, ((p["out_channels"] - 1) << 16) | (p["out_channel_align"] - 1))
    A(reg.BS_CFG, 0x53)
    A(reg.BS_OW_CFG, (1 << 8) | (1 << 5) | (1 << 2) | (1 << 1))
    A(reg.WDMA_SIZE_0, p["out_channel_align"] - 1)
    A(reg.WDMA_SIZE_1, p["output_width"] - 1)
    A(reg.BN_CFG, 0x53)
    A(reg.EW_CFG, 0x383)
    A(reg.EW_CVT_SCALE_VALUE, 1)
    A(reg.OUT_CVT_SCALE, (1 << 16) | 1)
    A(reg.SURFACE_ADD, p["surface_add"] << 4)
    regs.append((0x1001, 0, 0x40c4))
    regs.append((reg.PC, 0x0d, reg.OPERATION_ENABLE))
    return regs

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
                # why 5
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
        E(reg.DPU,  reg.EW_CVT_SCALE_VALUE, 1),       # DPU_EW_CVT_SCALE_VALUE
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
    ensure_npu_buffers()
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

def _write_rawbuf_case_tasks(case_id, input_dma, weight_dma, output_dma):
    case = CONV_RAWBUF_CASES[case_id]
    words = np.frombuffer(regcmd_map, dtype=np.uint64)
    words[:] = 0
    task_regs = build_conv_rawbuf_case_regs(
        case_id, input_dma, weight_dma, output_dma, regcmd_mem_create.dma_addr)
    amounts = _rawbuf_task_amounts(case)
    for task_idx, regs in enumerate(task_regs):
        base = case.task_offsets[task_idx] // 8
        words[base:base + len(regs)] = regs

        npu_tasks[task_idx].flags = 0
        npu_tasks[task_idx].op_idx = 1
        npu_tasks[task_idx].enable_mask = 0x0d
        npu_tasks[task_idx].int_mask = 0x300
        npu_tasks[task_idx].int_clear = 0x1ffff
        npu_tasks[task_idx].int_status = 0
        npu_tasks[task_idx].regcfg_amount = amounts[task_idx]
        npu_tasks[task_idx].regcfg_offset = 0
        npu_tasks[task_idx].regcmd_addr = regcmd_mem_create.dma_addr + case.task_offsets[task_idx]
    return len(task_regs)

def _qword_to_tuple(qword):
    return ((qword >> 48) & 0xffff, qword & 0xffff, (qword >> 16) & 0xffffffff)

def _regs_to_tuples(regs):
    return [_qword_to_tuple(qword) for qword in regs]

def _regs_to_qwords(regs):
    return [E(target, addr, value) for target, value, addr in regs]

def _conv2d_cpu(inp, wt, groups):
    batch, in_c, in_h, in_w = inp.shape
    out_c, ic_per_group, kh, kw = wt.shape
    out_h, out_w = in_h - kh + 1, in_w - kw + 1
    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)
    oc_per_group = out_c // groups if groups > 0 else out_c
    for o in range(out_c):
        group = o // oc_per_group if groups > 1 else 0
        for c_local in range(ic_per_group):
            c = group * ic_per_group + c_local
            for i in range(kh):
                for j in range(kw):
                    result[:, o] += inp[:, c, i:i + out_h, j:j + out_w] * wt[o, c_local, i, j]
    return result

def _write_fp16(dst_map, arr, offset=0):
    view = (ctypes.c_uint16 * arr.size).from_buffer(dst_map, offset)
    view[:] = arr.reshape(-1).view(np.uint16).tolist()

def _expand_grouped_weights(weight_ochw, params):
    out_c = params["out_channels"]
    in_c = params["in_channels"]
    ic_per_group = params["weight_in_channels"]
    kh = params["kernel_h"]
    kw = params["kernel_w"]
    groups = params["groups"]
    full = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
    oc_per_group = out_c // groups if groups > 0 else out_c
    src = weight_ochw.reshape(out_c, ic_per_group, kh, kw)
    for oc in range(out_c):
        group = oc // oc_per_group if groups > 1 else 0
        base_ic = group * ic_per_group
        full[oc, base_ic:base_ic + ic_per_group] = src[oc]
    return full.reshape(-1)

def _npu_submit(params, input_nchw, weight_ochw, is_1x1, log_submit=True):
    ensure_npu_buffers()
    input_packed = pack_nc1hwc2_fp16(
        input_nchw, params["batch"], params["in_channels"], params["in_h"], params["in_w"],
        params["input_pack_c2"], params["width_stride"],
        out_c=params["out_channels"], kh=params["kernel_h"], kw=params["kernel_w"],
        groups=params["groups"], use_nhwc=params["use_nhwc"])
    if params["is_depthwise"]:
        weight_full = weight_ochw
        weight_in_c = params["weight_in_channels"]
    elif params["groups"] > 1:
        weight_full = _expand_grouped_weights(weight_ochw, params)
        weight_in_c = params["in_channels"]
    else:
        weight_full = weight_ochw
        weight_in_c = params["in_channels"]
    packed_weight_size = params["weight_bytes_total"]
    is_11555_35333_g5 = (params["batch"] == 1 and params["in_channels"] == 15
                         and params["out_channels"] == 35 and params["groups"] == 5)
    if is_11555_35333_g5:
        kh, kw = params["kernel_h"], params["kernel_w"]
        kernel_hw = kh * kw
        block_count = params["out_channels"] * kernel_hw
        block_size = 16
        full_blocks = params["out_channels"] // block_size
        rem_blocks = params["out_channels"] % block_size
        full_span = kernel_hw * block_size
        reordered = np.zeros_like(weight_full)
        for p in range(block_count):
            dst_oc = p // kernel_hw
            rem_p = p % kernel_hw
            dst_kh = rem_p // kw
            dst_kw = rem_p % kw
            if p < full_blocks * full_span:
                oc_block = p // full_span
                block_off = p % full_span
                khkw = block_off // block_size
                oc_in_block = block_off % block_size
                src_oc = oc_block * block_size + oc_in_block
                src_kh = khkw // kw
                src_kw = khkw % kw
            elif rem_blocks > 0:
                rem_off = p - full_blocks * full_span
                khkw = rem_off // rem_blocks
                oc_in_block = rem_off % rem_blocks
                src_oc = full_blocks * block_size + oc_in_block
                src_kh = khkw // kw
                src_kw = khkw % kw
            else:
                src_oc, src_kh, src_kw = dst_oc, dst_kh, dst_kw
            for ic in range(params["in_channels"]):
                src_idx = (((src_oc * params["in_channels"] + ic) * kh) + src_kh) * kw + src_kw
                dst_idx = (((dst_oc * params["in_channels"] + ic) * kh) + dst_kh) * kw + dst_kw
                reordered.reshape(-1)[dst_idx] = weight_full.reshape(-1)[src_idx]
        weight_full = reordered
    if params["is_depthwise"]:
        oc, ic, kh, kw = params["out_channels"], params["in_channels"], params["kernel_h"], params["kernel_w"]
        dw_expand = np.zeros(oc * ic * kh * kw, dtype=np.float16)
        dw_expand.reshape(oc, ic, kh, kw)[range(oc), range(oc), :, :] = \
            weight_full.reshape(oc, weight_in_c, kh, kw)[:, 0, :, :]
        weight_full = dw_expand.reshape(-1)
        weight_in_c = ic
        weight_packed = pack_conv_weights_fp16(
            weight_full, params["out_channels"], weight_in_c,
            params["kernel_h"], params["kernel_w"], params["align_c"], groups=params["groups"])
    else:
        weight_packed = pack_conv_weights_fp16(
            weight_full, params["out_channels"], weight_in_c,
            params["kernel_h"], params["kernel_w"], params["align_c"], groups=params["groups"])

    _write_fp16(input_map, input_packed)
    _write_fp16(weight_map, weight_packed, REGCMD_RESERVED)
    out_elems = params["batch"] * _ceil_div(params["out_channels"], params["align_out_c"]) * params["out_width_stride"] * params["align_out_c"]
    if not is_1x1:
        out_elems *= params["out_h"]
    output_bytes = min(out_elems * FP16_BYTES, output_mem_create.size)
    np.frombuffer(output_map, dtype=np.uint16, count=output_bytes // FP16_BYTES)[:] = 0

    regs = build_conv2d_regs(params, input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)
    write_regs_to_npu_task([regs])
    mem_sync(weight_mem_create.obj_addr, min(REGCMD_RESERVED + weight_packed.nbytes, weight_mem_create.size), RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(input_mem_create.obj_addr, min(input_packed.nbytes, input_mem_create.size), RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(tasks_mem_create.obj_addr, tasks_mem_create.size, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(output_mem_create.obj_addr, output_bytes, RKNPU_MEM_SYNC_TO_DEVICE)
    ret = npu_submit(tasks_mem_create.obj_addr, task_count=1,
        flags=((1 << 0) | (1 << 1)))
    if log_submit:
        print(f"SUBMIT ret={ret}")
    mem_sync(output_mem_create.obj_addr, output_bytes, RKNPU_MEM_SYNC_FROM_DEVICE)
    out_raw = np.frombuffer(output_map, dtype=np.float16, count=output_bytes // FP16_BYTES).copy()
    c2 = _output_unpack_c2(params)
    if is_1x1:
        flat = unpack_nc1hwc2_fp16(out_raw, params["batch"], params["out_channels"], 1,
                                   params["out_h"] * params["out_w"], c2, params["out_width_stride"])
        return flat.reshape(params["batch"], params["out_channels"], params["out_h"], params["out_w"])
    unpacked = unpack_nc1hwc2_fp16(out_raw, params["batch"], params["out_channels"],
                                   params["out_h"], params["out_w"], c2, params["out_w"])
    return unpacked.reshape(params["batch"], params["out_channels"], params["out_h"], params["out_w"])

def _npu_submit_rawbuf_case4(params, input_nchw, weight_ochw):
    ensure_npu_buffers()
    if not (params["batch"] == 1 and params["in_channels"] == 1 and
            params["out_channels"] == 32 and params["kernel_h"] == 5 and
            params["kernel_w"] == 5 and params["in_h"] == 10 and params["in_w"] == 10 and
            params["groups"] == 1):
        raise ValueError("rawbuf case 4 submit only supports 1x32 5x5 on 10x10")

    input_packed = pack_nc1hwc2_fp16(
        input_nchw, params["batch"], params["in_channels"], params["in_h"], params["in_w"],
        params["input_pack_c2"], params["width_stride"],
        out_c=params["out_channels"], kh=params["kernel_h"], kw=params["kernel_w"],
        groups=params["groups"], use_nhwc=True)
    weight_packed = pack_conv_weights_fp16(
        weight_ochw, params["out_channels"], params["in_channels"],
        params["kernel_h"], params["kernel_w"], params["align_c"], groups=params["groups"])
    output_bytes = 0x900

    _write_fp16(input_map, input_packed)
    _write_fp16(weight_map, weight_packed, 0)
    np.frombuffer(output_map, dtype=np.uint16, count=output_bytes // FP16_BYTES)[:] = 0

    task_count = _write_rawbuf_case_tasks(
        4, input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)
    mem_sync(weight_mem_create.obj_addr, min(weight_packed.nbytes, weight_mem_create.size), RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(input_mem_create.obj_addr, min(input_packed.nbytes, input_mem_create.size), RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(regcmd_mem_create.obj_addr, regcmd_mem_create.size, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(tasks_mem_create.obj_addr, tasks_mem_create.size, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(output_mem_create.obj_addr, output_bytes, RKNPU_MEM_SYNC_TO_DEVICE)
    ret = npu_submit(tasks_mem_create.obj_addr, task_count=task_count, flags=((1 << 0) | (1 << 1) | (1 << 2)))
    print(f"SUBMIT ret={ret}")
    mem_sync(output_mem_create.obj_addr, output_bytes, RKNPU_MEM_SYNC_FROM_DEVICE)
    out_raw = np.frombuffer(output_map, dtype=np.float16, count=output_bytes // FP16_BYTES).copy()
    flat = unpack_nc1hwc2_fp16(out_raw, 1, params["out_channels"], params["out_h"], params["out_w"], 8, params["out_w"])
    return flat.reshape(1, params["out_channels"], params["out_h"], params["out_w"])

def _pack_conv1d_weights_fp16(weight_ociw, params):
    out_channels = params["out_channels"]
    in_channels = params["in_channels"]
    kernel_width = params["kernel_width"]
    data_in_channel = params["data_in_channel"]
    out = np.zeros(params["weight_bytes_total"] // FP16_BYTES, dtype=np.float16)
    for kw in range(kernel_width):
        kw_base = kw * out_channels * data_in_channel
        for oc in range(out_channels):
            oc_base = kw_base + oc * data_in_channel
            out[oc_base:oc_base + in_channels] = weight_ociw[oc, :in_channels, kw]
    return out

def _npu_submit_conv1d(input_ncw, weight_ociw):
    ensure_npu_buffers()
    batch, in_channels, input_width = input_ncw.shape
    out_channels, weight_in_channels, kernel_width = weight_ociw.shape
    if batch != 1:
        raise ValueError("_npu_submit_conv1d submits one batch at a time")
    if weight_in_channels != in_channels:
        raise ValueError("conv1d hardware submit expects expanded full-channel weights")
    p = _compute_conv1d_params(input_width, kernel_width, in_channels, out_channels)
    input_nchw = input_ncw.reshape(1, in_channels, 1, input_width)
    input_packed = pack_nc1hwc2_fp16(
        input_nchw, 1, in_channels, 1, input_width, p["data_in_channel"], p["input_width_aligned"])
    weight_packed = _pack_conv1d_weights_fp16(weight_ociw, p)

    _write_fp16(input_map, input_packed)
    _write_fp16(weight_map, weight_packed, REGCMD_RESERVED)
    output_bytes = min(p["dst_stride"] * p["out_channel_align"] * FP16_BYTES, output_mem_create.size)
    np.frombuffer(output_map, dtype=np.uint16, count=output_bytes // FP16_BYTES)[:] = 0

    regs = build_conv1d_regs(p, input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)
    write_regs_to_npu_task([regs])
    mem_sync(weight_mem_create.obj_addr, min(REGCMD_RESERVED + weight_packed.nbytes, weight_mem_create.size), RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(input_mem_create.obj_addr, min(input_packed.nbytes, input_mem_create.size), RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(tasks_mem_create.obj_addr, tasks_mem_create.size, RKNPU_MEM_SYNC_TO_DEVICE)
    mem_sync(output_mem_create.obj_addr, output_bytes, RKNPU_MEM_SYNC_TO_DEVICE)
    ret = npu_submit(tasks_mem_create.obj_addr, task_count=1, flags=((1 << 0) | (1 << 1)))
    print(f"SUBMIT ret={ret}")
    mem_sync(output_mem_create.obj_addr, output_bytes, RKNPU_MEM_SYNC_FROM_DEVICE)
    out_raw = np.frombuffer(output_map, dtype=np.float16, count=output_bytes // FP16_BYTES).copy()
    flat = unpack_nc1hwc2_fp16(out_raw, 1, out_channels, 1, p["output_width"], 8, p["dst_stride"])
    return flat.reshape(1, out_channels, p["output_width"])

def _run_conv2d_depthwise_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1):
    np.random.seed(42)
    full_in = np.random.randn(1, original_in_c, input_hw[0], input_hw[1]).astype(np.float16)
    full_wt = np.random.randn(out_channels, 1, kernel_h, kernel_w).astype(np.float16)
    if DRY_RUN:
        return _conv2d_cpu(full_in, full_wt, original_in_c), full_in, full_wt
    result = np.zeros((1, out_channels, input_hw[0] - kernel_h + 1, input_hw[1] - kernel_w + 1), dtype=np.float16)
    for start_c in range(0, original_in_c, 8):
        end_c = min(start_c + 8, original_in_c)
        count = end_c - start_c
        inp_slice = np.zeros((1, 8, input_hw[0], input_hw[1]), dtype=np.float16)
        wt_slice = np.zeros((8, 1, kernel_h, kernel_w), dtype=np.float16)
        inp_slice[:, :count] = full_in[:, start_c:end_c]
        wt_slice[:count] = full_wt[start_c:end_c]
        p = compute_conv2d_params(8, 8, kernel_h, kernel_w, input_hw, groups=8)
        if DRY_RUN:
            result[:, start_c:end_c] = _conv2d_cpu(inp_slice[:, :count], wt_slice[:count], count)
        else:
            result[:, start_c:end_c] = _npu_submit(p, inp_slice.reshape(-1), wt_slice.reshape(-1), is_1x1)[:, :count]
    return result, full_in, full_wt

def _run_conv2d_channel_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1):
    np.random.seed(42)
    full_in = np.random.randn(1, original_in_c, input_hw[0], input_hw[1]).astype(np.float16)
    full_wt = np.random.randn(out_channels, original_in_c, kernel_h, kernel_w).astype(np.float16)
    if DRY_RUN:
        return _conv2d_cpu(full_in, full_wt, 1), full_in, full_wt
    result = np.zeros((1, out_channels, input_hw[0] - kernel_h + 1, input_hw[1] - kernel_w + 1), dtype=np.float16)
    for start_c in range(0, original_in_c, 4):
        end_c = min(start_c + 4, original_in_c)
        count = end_c - start_c
        inp_slice = np.zeros((1, 4, input_hw[0], input_hw[1]), dtype=np.float16)
        wt_slice = np.zeros((out_channels, 4, kernel_h, kernel_w), dtype=np.float16)
        inp_slice[:, :count] = full_in[:, start_c:end_c]
        wt_slice[:, :count] = full_wt[:, start_c:end_c]
        p = compute_conv2d_params(4, out_channels, kernel_h, kernel_w, input_hw, groups=1)
        if DRY_RUN:
            result += _conv2d_cpu(inp_slice, wt_slice, 1)
        else:
            result += _npu_submit(p, inp_slice.reshape(-1), wt_slice.reshape(-1), is_1x1)
    return result, full_in, full_wt

def _npu_submit_single_input_pointwise(input_1c, weight_ochw):
    batch, _, height, width = input_1c.shape
    out_channels = weight_ochw.shape[0]
    hw_channels = 3
    result = np.zeros((batch, out_channels, height, width), dtype=np.float16)
    input_padded = np.zeros((batch, hw_channels, height, width), dtype=np.float16)
    input_padded[:, :1] = input_1c
    for start_oc in range(0, out_channels, OUTPUT_CHANNEL_CHUNK):
        end_oc = min(start_oc + OUTPUT_CHANNEL_CHUNK, out_channels)
        weight_padded = np.zeros((end_oc - start_oc, hw_channels, 1, 1), dtype=np.float16)
        weight_padded[:, :1] = weight_ochw[start_oc:end_oc]
        p = compute_conv2d_params(hw_channels, end_oc - start_oc, 1, 1, (height, width), groups=1)
        result[:, start_oc:end_oc] = _npu_submit(
            p, input_padded.reshape(-1), weight_padded.reshape(-1), True, log_submit=False)
    return result

def _run_conv2d_spatial_decomposed(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups,
                                   input_nchw=None, weight_ochw=None):
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
    if input_nchw is None or weight_ochw is None:
        np.random.seed(42)
        if input_nchw is None:
            input_nchw = np.random.randn(batch, in_channels, in_h, in_w).astype(np.float16)
        if weight_ochw is None:
            weight_ochw = np.random.randn(
                out_channels, in_channels // groups if groups > 0 else in_channels,
                kernel_h, kernel_w).astype(np.float16)
    result = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float16)
    oc_per_group = out_channels // groups if groups > 0 else out_channels
    ic_per_group = in_channels // groups if groups > 0 else in_channels
    for group in range(groups):
        oc_start = group * oc_per_group
        oc_end = oc_start + oc_per_group
        for ic_local in range(ic_per_group):
            ic = group * ic_per_group + ic_local
            for kh_idx in range(kernel_h):
                for kw_idx in range(kernel_w):
                    input_crop = input_nchw[:, ic:ic + 1,
                                            kh_idx:kh_idx + out_h,
                                            kw_idx:kw_idx + out_w].copy()
                    weight_1x1 = weight_ochw[oc_start:oc_end, ic_local:ic_local + 1,
                                             kh_idx:kh_idx + 1, kw_idx:kw_idx + 1].copy()
                    if DRY_RUN:
                        result[:, oc_start:oc_end] += _conv2d_cpu(input_crop, weight_1x1, 1)
                    else:
                        result[:, oc_start:oc_end] += _npu_submit_single_input_pointwise(input_crop, weight_1x1)
    return result, input_nchw, weight_ochw

def _calc_spatial_tile_rows(params):
    line_bytes = params["width_stride"] * params["align_c"] * FP16_BYTES
    if line_bytes <= 0:
        return params["in_h"]
    weight_banks = max(1, _ceil_div(params["weight_bytes_total"], NPU_CBUF_BANK_SIZE))
    input_banks = max(1, NPU_CBUF_BANKS - weight_banks)
    max_rows = (input_banks * NPU_CBUF_BANK_SIZE) // line_bytes
    return max(params["kernel_h"], min(params["in_h"], min(max_rows, 32)))

def _run_conv2d_spatial_tiled(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups, input_nchw=None, weight_ochw=None):
    batch, in_h, in_w = 1, input_hw[0], input_hw[1]
    out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid output shape for kernel {kernel_h}x{kernel_w} on input {input_hw}")

    if input_nchw is None or weight_ochw is None:
        np.random.seed(42)
        if input_nchw is None:
            input_nchw = np.random.randn(batch, in_channels, in_h, in_w).astype(np.float16)
        if weight_ochw is None:
            weight_ochw = np.random.randn(
                out_channels, in_channels // groups if groups > 0 else in_channels,
                kernel_h, kernel_w).astype(np.float16)

    result = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float16)
    base_params = compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups)
    max_in_rows = _calc_spatial_tile_rows(base_params)
    if max_in_rows >= in_h:
        result[:] = _npu_submit(base_params, input_nchw.reshape(-1), weight_ochw.reshape(-1), False, log_submit=False)
        return result, input_nchw, weight_ochw

    step_rows = max(1, max_in_rows - kernel_h + 1)
    for start_out_row in range(0, out_h, step_rows):
        rows_for_slice = min(step_rows, out_h - start_out_row)
        slice_in_top = start_out_row
        slice_in_bottom = min(in_h, slice_in_top + rows_for_slice + kernel_h - 1)
        inp_slice = np.zeros((batch, in_channels, slice_in_bottom - slice_in_top, in_w), dtype=np.float16)
        inp_slice[:] = input_nchw[:, :, slice_in_top:slice_in_bottom]
        p_slice = compute_conv2d_params(in_channels, out_channels, kernel_h, kernel_w, (slice_in_bottom - slice_in_top, in_w), groups)
        part = _npu_submit(p_slice, inp_slice.reshape(-1), weight_ochw.reshape(-1), False, log_submit=False)
        keep_h = min(rows_for_slice, part.shape[2], result.shape[2] - start_out_row)
        if keep_h > 0:
            result[:, :, start_out_row:start_out_row + keep_h, :part.shape[3]] = part[:, :, :keep_h]
    return result, input_nchw, weight_ochw

def run_conv2d(in_channels, out_channels, kernel_h, kernel_w, input_hw, groups=1, batch=1,
               input_nchw=None, weight_ochw=None):
    original_in_c = in_channels
    is_1x1 = kernel_h == 1 and kernel_w == 1
    if groups == in_channels and out_channels == in_channels and in_channels > 8:
        return _run_conv2d_depthwise_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1)
    if is_1x1 and groups == 1 and in_channels >= 5:
        return _run_conv2d_channel_sliced(out_channels, kernel_h, kernel_w, input_hw, original_in_c, is_1x1)

    pad_to_c = 3 if is_1x1 and in_channels < 3 and groups == 1 else None
    hw_in_c = pad_to_c or in_channels
    hw_groups = groups
    p = compute_conv2d_params(hw_in_c, out_channels, kernel_h, kernel_w, input_hw, hw_groups, batch=batch)

    if input_nchw is None:
        np.random.seed(42)
        input_nchw = np.random.randn(batch, original_in_c, input_hw[0], input_hw[1]).astype(np.float16)
    if weight_ochw is None:
        weight_ochw = np.random.randn(out_channels, original_in_c // groups if groups > 0 else original_in_c, kernel_h, kernel_w).astype(np.float16)
    if batch > 1:
        result = np.zeros((batch, out_channels, input_hw[0] - kernel_h + 1, input_hw[1] - kernel_w + 1), dtype=np.float16)
        for n in range(batch):
            sub_result, _, _ = run_conv2d(
                in_channels, out_channels, kernel_h, kernel_w, input_hw, groups,
                batch=1, input_nchw=input_nchw[n:n + 1], weight_ochw=weight_ochw)
            result[n:n + 1] = sub_result
        return result, input_nchw, weight_ochw

    if pad_to_c:
        padded_in = np.zeros((1, pad_to_c, input_hw[0], input_hw[1]), dtype=np.float16)
        padded_in[:, :original_in_c] = input_nchw
        padded_wt = np.zeros((out_channels, pad_to_c, kernel_h, kernel_w), dtype=np.float16)
        padded_wt[:, :original_in_c] = weight_ochw
        input_for_hw = padded_in
        weight_for_hw = padded_wt
    else:
        input_for_hw = input_nchw
        weight_for_hw = weight_ochw

    if DRY_RUN:
        regs = build_conv2d_regs(p, 0, 0, 0)
        print(f"  DRY RUN regs={len(regs)} in=({p['batch']},{p['in_channels']},{p['in_h']},{p['in_w']}) "
              f"out=({p['batch']},{p['out_channels']},{p['out_h']},{p['out_w']})")
        return _conv2d_cpu(input_nchw, weight_ochw, groups), input_nchw, weight_ochw

    if SUBMIT_MODE and not is_1x1:
        _check_direct_spatial_submit_allowed(batch, hw_in_c, out_channels, kernel_h, kernel_w, input_hw, hw_groups)
        return _run_conv2d_spatial_tiled(hw_in_c, out_channels, kernel_h, kernel_w, input_hw,
                                         hw_groups, input_for_hw, weight_for_hw)

    result = _npu_submit(p, input_for_hw.reshape(-1), weight_for_hw.reshape(-1), is_1x1)
    return result[:, :out_channels], input_nchw, weight_ochw

def run_conv():
    result, _, _ = run_conv2d(2, 2, 1, 1, (4, 4), groups=1)
    return result

def run_conv1d_case(batch, in_channels, input_size, out_channels, weight_in_channels, kernel_size, groups):
    groups = in_channels // weight_in_channels if groups <= 0 else groups
    np.random.seed(42)
    input_ncw = np.random.randn(batch, in_channels, input_size).astype(np.float16)
    weight_ociw = np.random.randn(out_channels, weight_in_channels, kernel_size).astype(np.float16)
    input_nchw = input_ncw.reshape(batch, in_channels, 1, input_size)
    weight_ochw = weight_ociw.reshape(out_channels, weight_in_channels, 1, kernel_size)
    expected = _conv2d_cpu(input_nchw, weight_ochw, groups).reshape(
        batch, out_channels, input_size - kernel_size + 1)
    if DRY_RUN:
        p = _compute_conv1d_params(input_size, kernel_size, in_channels, out_channels)
        regs = build_conv1d_regs(p, 0, 0, 0)
        print(f"  DRY RUN regs={len(regs)} in=({batch},{in_channels},{input_size}) "
              f"out=({batch},{out_channels},{p['output_width']})")
        result = expected.copy()
    else:
        result = np.zeros_like(expected)
        out_per_group = out_channels // groups if groups > 0 else out_channels
        expanded_weight = np.zeros((out_channels, in_channels, kernel_size), dtype=np.float16)
        for oc in range(out_channels):
            group = oc // out_per_group if groups > 1 else 0
            base_ic = group * weight_in_channels
            expanded_weight[oc, base_ic:base_ic + weight_in_channels] = weight_ociw[oc]
        for n in range(batch):
            result[n:n + 1] = _npu_submit_conv1d(input_ncw[n:n + 1], expanded_weight)
    ok = np.allclose(result, expected, atol=0.1) and not np.any(np.isinf(result))
    md = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    return result.reshape(batch, out_channels, 1, -1), input_nchw, weight_ochw, ok, md

def _main_conv2d_cases():
    return [
        # 1x1 kernels
        (1, 2, 2, 1, 1, (4, 4), 1, "default 1x1"),
        (1, 1, 6, 1, 1, (4, 4), 1, "ic=1 oc=6 1x1"),
        (1, 3, 3, 1, 1, (4, 4), 1, "ic=3 1x1"),
        (1, 4, 2, 1, 1, (4, 4), 1, "ic=4 1x1"),
        (1, 4, 4, 1, 1, (9, 9), 1, "1x1 9x9"),
        (1, 8, 8, 1, 1, (5, 5), 1, "ic=8 oc=8 1x1 5x5"),
        (1, 16, 16, 1, 1, (8, 8), 1, "1x1 ic=16 oc=16 8x8"),
        (1, 16, 16, 1, 1, (32, 32), 1, "1x1 32x32"),
        (1, 10, 20, 3, 3, (9, 9), 1, "test_ops simple_conv2d_nhwc shape"),

        # Non-1x1 kernels
        (1, 4, 4, 3, 3, (9, 9), 1, "simple 3x3 9x9"),
        (1, 16, 16, 3, 3, (9, 9), 1, "3x3 m4"),
        (1, 2, 4, 3, 3, (6, 6), 1, "3x3 oc!=ic"),
        (1, 1, 6, 3, 3, (5, 7), 1, "ic=1 oc=6 3x3"),
        (1, 16, 16, 3, 3, (18, 18), 1, "16x16 3x3 18x18"),
        (1, 2, 4, 2, 2, (5, 5), 1, "2x2 kernel"),
        (1, 1, 32, 5, 5, (10, 10), 1, "5x5 kernel"),
        (1, 8, 4, 4, 4, (10, 10), 1, "4x4 kernel"),
        (1, 3, 3, 3, 3, (11, 28), 3, "dw 3x3 11x28"),
        (1, 1, 6, 3, 1, (5, 7), 1, "ic=1 oc=6 3x1"),
        (1, 3, 6, 1, 3, (5, 5), 1, "1x3 kernel"),

        # test_ops coverage
        (1, 3, 6, 3, 3, (5, 7), 3, "test_ops _test_conv2d cin=3 3x3 g=3"),
        (1, 3, 6, 2, 1, (5, 7), 1, "test_ops _test_conv2d cin=3 2x1"),
        (1, 3, 6, 2, 3, (5, 7), 1, "test_ops _test_conv2d cin=3 2x3"),
        (1, 3, 6, 3, 1, (5, 7), 1, "test_ops _test_conv2d cin=3 3x1"),
        (1, 3, 6, 3, 5, (5, 7), 1, "test_ops _test_conv2d cin=3 3x5"),
        (1, 1, 6, 3, 3, (5, 7), 1, "test_ops _test_conv2d cin=1 3x3"),
        (1, 1, 6, 2, 1, (5, 7), 1, "test_ops _test_conv2d cin=1 2x1"),
        (1, 1, 6, 2, 3, (5, 7), 1, "test_ops _test_conv2d cin=1 2x3"),
        (1, 1, 6, 3, 1, (5, 7), 1, "test_ops _test_conv2d cin=1 3x1"),
        (1, 1, 6, 3, 5, (5, 7), 1, "test_ops _test_conv2d cin=1 3x5"),
        (1, 4, 2, 1, 1, (1, 1), 2, "test_ops simple_grouped_conv2d"),
        (1, 4, 4, 1, 1, (1, 1), 2, "test_ops medium_grouped_conv2d"),
        (1, 32, 32, 1, 1, (32, 32), 32, "test_ops depthwise_conv2d"),
        (1, 15, 35, 3, 3, (5, 5), 5, "test_ops grouped_conv2d"),

        # Additional conv2d coverage requested from main.c / test lists.
        (1, 3, 32, 3, 3, (224, 224), 1, "conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1"),
        (1, 32, 32, 3, 3, (112, 112), 32, "conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32"),
        (1, 32, 64, 1, 1, (112, 112), 1, "conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1"),
        (1, 64, 64, 3, 3, (112, 112), 64, "conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64"),
        (1, 64, 128, 1, 1, (56, 56), 1, "conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1"),
        (1, 128, 128, 3, 3, (56, 56), 128, "conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128"),
        (1, 128, 128, 1, 1, (56, 56), 1, "conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1"),
        (1, 128, 256, 1, 1, (28, 28), 1, "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1"),
        (1, 256, 256, 3, 3, (28, 28), 256, "conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256"),
        (1, 256, 256, 1, 1, (28, 28), 1, "conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1"),
        (1, 256, 512, 1, 1, (14, 14), 1, "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1"),
        (1, 512, 512, 3, 3, (14, 14), 512, "conv2d_cc_b1_c512_h14_w14_oc512_wic1_k3x3_g512"),
        (1, 512, 512, 1, 1, (14, 14), 1, "conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1"),
        (1, 512, 1024, 1, 1, (7, 7), 1, "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1"),
        (1, 1024, 1024, 3, 3, (7, 7), 1024, "conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024"),
        (1, 1024, 1024, 1, 1, (7, 7), 1, "conv2d_cc_b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1"),
        (1, 1024, 1024, 7, 7, (7, 7), 1024, "conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024"),
        (1, 1024, 1001, 1, 1, (1, 1), 1, "conv2d_cc_b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1"),
        (1, 16, 16, 3, 3, (18, 18), 1, "conv2d_i1161818_w161633"),
        (1, 4, 4, 1, 1, (9, 9), 1, "conv2d_i1499_w4411"),
        (1, 16, 16, 1, 1, (32, 32), 1, "conv2d_i1163232_w161611"),
        (1, 4, 4, 3, 3, (9, 9), 1, "conv2d_i1499_w4433"),
        (2, 4, 4, 3, 3, (9, 9), 1, "conv2d_i2499_w4433"),
        (1, 1, 6, 3, 3, (5, 7), 1, "conv2d_i1157_w6133"),
        (1, 4, 2, 1, 1, (1, 1), 2, "conv2d_i1411_w2211_g2"),
        (1, 32, 32, 1, 1, (32, 32), 32, "conv2d_i1323232_w32111_g32"),
        (1, 15, 35, 3, 3, (5, 5), 5, "conv2d_i11555_w35333_g5"),
        (4, 15, 35, 3, 3, (5, 5), 5, "conv2d_i41555_w35333_g5"),
        (1, 3, 3, 3, 3, (11, 28), 3, "conv2d_i131128_w3133_g3"),
        (2, 3, 3, 3, 3, (11, 28), 3, "conv2d_i231128_w3133_g3"),
        (1, 3, 6, 2, 1, (5, 7), 1, "conv2d_i1357_w6321"),
        (1, 3, 6, 2, 3, (5, 7), 1, "conv2d_i1357_w6323"),
        (1, 3, 6, 2, 5, (5, 7), 1, "conv2d_i1357_w6325"),
        (1, 3, 6, 3, 1, (5, 7), 1, "conv2d_i1357_w6331"),
        (1, 3, 6, 3, 3, (5, 7), 1, "conv2d_i1357_w6333"),
        (1, 3, 6, 3, 3, (5, 7), 3, "conv2d_i1357_w6133_g3"),
        (1, 3, 6, 3, 5, (5, 7), 1, "conv2d_i1357_w6335"),
    ]

def _main_conv1d_cases():
    return [
        (1, 1, 11, 6, 1, 1, 0, "conv1d_bs1"),
        (8, 1, 11, 6, 1, 1, 0, "conv1d_bs8"),
        (1, 1, 11, 6, 1, 2, 0, "conv1d_bs1_612"),
        (1, 1, 11, 6, 1, 5, 0, "conv1d_bs1_615"),
        (1, 3, 11, 6, 3, 1, 0, "conv1d_bs1_1311_631"),
        (1, 3, 11, 6, 3, 2, 0, "conv1d_bs1_1311_632"),
        (1, 3, 11, 6, 3, 5, 0, "conv1d_bs1_1311_635"),
        (1, 3, 11, 6, 1, 5, 3, "conv1d_bs1_1311_615"),
        (8, 1, 11, 6, 1, 1, 0, "conv1d_bs8_8111_611"),
        (8, 1, 11, 6, 1, 2, 0, "conv1d_bs8_8111_612"),
        (8, 1, 11, 6, 1, 2, 0, "conv1d_bs8_8111_612"),
        (8, 1, 11, 6, 1, 5, 0, "conv1d_bs8_8111_615"),
        (8, 3, 11, 6, 3, 1, 0, "conv1d_bs8_8311_631"),
        (8, 3, 11, 6, 3, 2, 0, "conv1d_bs8_8311_632"),
        (8, 3, 11, 6, 3, 5, 0, "conv1d_bs8_8311_635"),
        (8, 3, 11, 6, 1, 5, 0, "conv1d_bs8_8311_635"),
    ]

def _arg_value(name):
    prefix = name + "="
    for i, arg in enumerate(sys.argv[1:]):
        if arg == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None

def _select_test_cases(test_cases):
    shape = _arg_value("--shape")
    if shape:
        parts = [int(x) for x in shape.replace("x", ",").split(",") if x]
        if len(parts) == 7:
            in_c, out_c, kh, kw, h, w, groups = parts
            return [(1, in_c, out_c, kh, kw, (h, w), groups, f"shape {shape}")]
        if len(parts) == 9:
            batch, in_c, h, w, out_c, weight_in_c, kh, kw, groups = parts
            if groups <= 0:
                groups = in_c // weight_in_c
            return [(batch, in_c, out_c, kh, kw, (h, w), groups, f"shape {shape}")]
        raise ValueError("--shape expects ic,oc,kh,kw,h,w,groups or batch,in_c,h,w,out_c,weight_in_c,kh,kw,groups")

    case = _arg_value("--case")
    if case is None:
        return test_cases
    if case.isdigit():
        idx = int(case)
        return [test_cases[idx]]
    selected = [tc for tc in test_cases if case.lower() in tc[7].lower()]
    if not selected:
        raise ValueError(f"--case {case!r} did not match any test case")
    return selected

if __name__ == "__main__":
    rawbuf_case = _arg_value("--rawbuf-case")
    if rawbuf_case is not None:
        dump_conv_rawbuf_case(int(rawbuf_case))
        sys.exit(0)

    rawbuf_check = _arg_value("--check-rawbuf-case")
    if rawbuf_check is not None:
        ok = check_conv_rawbuf_case_parity(int(rawbuf_check))
        sys.exit(0 if ok else 1)

    compare_rawbuf = _arg_value("--compare-rawbuf-case")
    if compare_rawbuf is not None:
        parts = [int(x) for x in compare_rawbuf.replace("x", ",").split(",") if x]
        if len(parts) != 8:
            raise ValueError("--compare-rawbuf-case expects case,ic,oc,kh,kw,h,w,groups")
        case_id, in_c, out_c, kh, kw, h, w, groups = parts
        compare_conv2d_regs_to_rawbuf(case_id, in_c, out_c, kh, kw, (h, w), groups)
        sys.exit(0)

    if SUBMIT_MODE and not ALLOW_BROAD_SUBMIT and _arg_value("--case") is None and _arg_value("--shape") is None:
        raise SystemExit(
            "Refusing broad conv.py --submit because previous direct-spatial sweeps crashed this board. "
            "Use --case/--shape for one audited target, or set CONV_ALLOW_BROAD_SUBMIT=1 after review.")

    case = _arg_value("--case")
    run_conv1d_only = case is not None and any(
        case == tc[7] or (not case.isdigit() and case.lower() in tc[7].lower())
        for tc in _main_conv1d_cases())
    test_cases = [] if run_conv1d_only else _select_test_cases(_main_conv2d_cases())

    if "--dump-regs" in sys.argv:
        for batch, in_c, out_c, kh, kw, input_hw, groups, desc in test_cases:
            dump_conv2d_case_regs(batch, in_c, out_c, kh, kw, input_hw, groups, desc)
        conv1d_cases = [] if _arg_value("--shape") is not None else _main_conv1d_cases()
        if case is not None:
            if case.isdigit():
                idx = int(case)
                conv1d_cases = [conv1d_cases[idx]] if run_conv1d_only else []
            else:
                conv1d_cases = [tc for tc in conv1d_cases if case.lower() in tc[7].lower()]
        for batch, in_c, input_size, out_c, weight_in_c, kernel_size, groups, desc in conv1d_cases:
            dump_conv1d_case_regs(batch, in_c, input_size, out_c, weight_in_c, kernel_size, groups, desc)
        sys.exit(0)

    for batch, in_c, out_c, kh, kw, input_hw, groups, desc in test_cases:
        print(f"\n{desc}:")
        r, inp, wt = run_conv2d(in_c, out_c, kh, kw, input_hw, groups, batch=batch)
        expected = _conv2d_cpu(inp, wt, groups)
        ok = np.allclose(r, expected, atol=0.1) and not np.any(np.isinf(r))
        md = np.max(np.abs(r.astype(np.float32) - expected.astype(np.float32)))
        print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")
        assert ok, f"test shape {batch, in_c, out_c, kh, kw, input_hw, groups} failed"

    conv1d_cases = [] if _arg_value("--shape") is not None else _main_conv1d_cases()
    if case is not None:
        if case.isdigit():
            idx = int(case)
            conv1d_cases = [conv1d_cases[idx]] if run_conv1d_only else []
        else:
            conv1d_cases = [tc for tc in conv1d_cases if case.lower() in tc[7].lower()]
    for batch, in_c, input_size, out_c, weight_in_c, kernel_size, groups, desc in conv1d_cases:
        print(f"\n{desc}:")
        r, inp, wt, ok, md = run_conv1d_case(batch, in_c, input_size, out_c, weight_in_c, kernel_size, groups)
        print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")
        assert ok, f"test shape {(batch, in_c, input_size, out_c, weight_in_c, kernel_size, groups)} failed"
    close_npu_buffers()
