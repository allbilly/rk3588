import os, mmap, sys, ctypes, numpy as np
from fcntl import ioctl

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 1 << 2

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
        ("iommu_domain_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("task_base_addr", ctypes.c_uint64),
        ("hw_elapse_time", ctypes.c_int64),
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
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    return buf, mem_create

def npu_reset(fd):
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

def npu_submit(task_obj_addr, task_count=1, flags=0x1 | 0x2 | 0x4):
    npu_reset(fd)
    submit_struct = rknpu_submit(
        flags=flags, timeout=6000, task_start=0, task_number=task_count,
        task_counter=0, priority=0, task_obj_addr=task_obj_addr,
        iommu_domain_id=0, reserved=0, task_base_addr=0, hw_elapse_time=0, core_mask=1, fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
npu_tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
npu_regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def _align_up(x, align):
    return (x + align - 1) // align * align

def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

# EW_CFG values for each op
# Base config: data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1
_EW_BASE = 0x108002c0
EW_CFG_ADD  = _EW_BASE | (2 << 16)
EW_CFG_MUL  = _EW_BASE | (1 << 2) | (1 << 8)
EW_CFG_SUB  = _EW_BASE | (4 << 16)
EW_CFG_MAX  = _EW_BASE
EW_CFG_NEG  = EW_CFG_MUL

def write_regs_to_npu_task(npu_regs):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())

    tail = [
        E(reg.PC_REG, 0x0010, 0),
        E(reg.PC_REG, 0x0014, 0),
        E(reg.VERSION, 0, 0),
        E(reg.PC, reg.OPERATION_ENABLE, 0x18),
    ]
    for i, qword in enumerate(npu_regs):
        npu_regcmd[i] = qword
    for i, qword in enumerate(tail):
        npu_regcmd[len(npu_regs) + i] = qword

    npu_tasks[0].regcmd_addr = regcmd_mem_create.dma_addr
    npu_tasks[0].regcfg_amount = len(npu_regs)
    npu_tasks[0].op_idx = 4
    npu_tasks[0].enable_mask = 0x18
    npu_tasks[0].int_mask = (1 << 8) | (1 << 9)
    npu_tasks[0].int_clear = 0x1ffff

def run_op(ew_cfg_val, a_vals, b_vals, neg_op=False, fdiv_op=False):
    n = len(a_vals)
    dataout_width = (n + 7) // 8 - 1

    npu_regs = [
        E(reg.DPU,  reg.FEATURE_MODE_CFG,
            ((15 << 5) |                          # DPU_FEATURE_MODE_CFG_BURST_LEN
             (2 << 1)  |                          # DPU_FEATURE_MODE_CFG_OUTPUT_MODE
             1)                                    # DPU_FEATURE_MODE_CFG_FLYING_MODE
        ),
        E(reg.DPU,  reg.DATA_FORMAT,
            ((2 << 29) |                          # DPU_DATA_FORMAT_OUT_PRECISION(fp16)
             (2 << 26) |                          # DPU_DATA_FORMAT_IN_PRECISION(fp16)
             2)                                    # DPU_DATA_FORMAT_PROC_PRECISION(fp16)
        ),
        E(reg.DPU,  reg.DATA_CUBE_WIDTH, dataout_width),   # DPU_DATA_CUBE_WIDTH_WIDTH
        E(reg.DPU,  reg.DATA_CUBE_HEIGHT, 0),               # DPU_DATA_CUBE_HEIGHT_HEIGHT
        E(reg.DPU,  reg.DATA_CUBE_NOTCH, 0),                # DPU_DATA_CUBE_NOTCH
        E(reg.DPU,  reg.DATA_CUBE_CHANNEL,
            ((7 << 16) |                          # DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL
             7)                                    # DPU_DATA_CUBE_CHANNEL_CHANNEL
        ),
        E(reg.DPU,  reg.EW_CFG, ew_cfg_val),      # DPU_EW_CFG (set per op)
        E(reg.DPU,  reg.OUT_CVT_SCALE,
            (1 if fdiv_op else                     # DPU_OUT_CVT_SCALE_OUT_CVT_SCALE
             ((1 << 16) | 1))                      # DPU_OUT_CVT_SCALE_FP32TOFP16_EN | scale
        ),
        E(reg.RDMA, reg.RDMA_DATA_CUBE_WIDTH, dataout_width),  # RDMA width
        E(reg.RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),             # RDMA height
        E(reg.RDMA, reg.RDMA_DATA_CUBE_CHANNEL,
            7                                     # RDMA channel (8 elements per atom)
        ),
        E(reg.RDMA, reg.RDMA_ERDMA_CFG,
            ((1 << 30) |                          # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE
             (2 << 2))                            # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE
        ),
        E(reg.DPU,  reg.DST_BASE_ADDR, output_mem_create.dma_addr ),
        E(reg.RDMA, reg.RDMA_SRC_BASE_ADDR, input_mem_create.dma_addr ),
        E(reg.RDMA, reg.RDMA_EW_BASE_ADDR, weight_mem_create.dma_addr ),
        E(reg.RDMA, reg.RDMA_FEATURE_MODE_CFG,
            ((2 << 15) |                          # DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(fp16)
             (15 << 11) |                         # DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN
             (2 << 5)  |                          # DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(fp16)
             ((not fdiv_op) << 3) |               # DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN
             1)                                    # DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE
        ),
    ]

    write_regs_to_npu_task(npu_regs)

    a_packed = np.array(a_vals, dtype=np.float16).view(np.uint16)
    ct_inputs = (ctypes.c_uint16 * n).from_buffer(input_map)
    ct_inputs[:] = a_packed.tolist()

    w_vals = np.full(n, -1.0, dtype=np.float16) if neg_op else np.array(b_vals, dtype=np.float16)
    w_packed = w_vals.view(np.uint16)
    ct_weights = (ctypes.c_uint16 * n).from_buffer(weight_map)
    ct_weights[:] = w_packed.tolist()

    npu_submit(tasks_mem_create.obj_addr)

    return np.frombuffer(output_map, dtype=np.float16, count=n).copy().tolist()


OPS = {
    "ADD":  (EW_CFG_ADD, lambda a, b: a + b,       {}),
    "MUL":  (EW_CFG_MUL, lambda a, b: a * b,       {}),
    "SUB":  (EW_CFG_SUB, lambda a, b: a - b,       {}),
    "MAX":  (EW_CFG_MAX, lambda a, b: np.maximum(a, b), {}),
    "NEG":  (EW_CFG_NEG, lambda a, _: -a,          {"neg_op": True}),
    "FDIV": (_EW_BASE | (3 << 16) | (1 << 8), lambda a, b: a / b, {"fdiv_op": True}),
}

if __name__ == "__main__":
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"

    if mode == "ALL":
        run_list = list(OPS.items())
    elif mode in OPS:
        run_list = [(mode, OPS[mode])]
    else:
        print(f"Unknown mode '{mode}'. Options: ALL, {', '.join(OPS.keys())}")
        sys.exit(1)

    np.random.seed(42)
    for op_name, (ew_cfg_val, expected_fn, kw) in run_list:
        for n in [1, 2, 3, 4, 5, 8, 16, 32]:
            a_vals = np.random.uniform(-5, 5, n).astype(np.float16)
            if op_name == "FDIV":
                b_vals = np.random.uniform(0.5, 5, n).astype(np.float16)
            elif op_name == "MAX":
                b_vals = np.random.uniform(-5, 5, n).astype(np.float16)
            else:
                b_vals = np.random.uniform(-5, 5, n).astype(np.float16)

            r = run_op(ew_cfg_val=ew_cfg_val, a_vals=a_vals, b_vals=b_vals, **kw)
            r_arr = np.array(r, dtype=np.float16)
            expected = expected_fn(a_vals, b_vals)
            match = np.allclose(r_arr, expected, atol=0.1)
            md = np.max(np.abs(r_arr.astype(np.float64) - expected.astype(np.float64)))
            ok = "PASS" if match else "FAIL"
            print(f"{op_name:4s} n={n:2d}: {ok}  max_diff={md:.6f}")
            assert match, f"{op_name} n={n} failed"

    os.close(fd)
