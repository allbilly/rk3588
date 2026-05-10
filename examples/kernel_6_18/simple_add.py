from fcntl import ioctl
import glob
import os, mmap, sys, time
import ctypes
import numpy as np

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

class drm_rocket_create_bo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("handle", ctypes.c_uint32),
        ("dma_address", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
    ]

class drm_rocket_prep_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("timeout_ns", ctypes.c_int64),
    ]

class drm_rocket_fini_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]

class drm_rocket_task(ctypes.Structure):
    _fields_ = [
        ("regcmd", ctypes.c_uint32),
        ("regcmd_count", ctypes.c_uint32),
    ]

class drm_rocket_job(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_uint64),
        ("in_bo_handles", ctypes.c_uint64),
        ("out_bo_handles", ctypes.c_uint64),
        ("task_count", ctypes.c_uint32),
        ("task_struct_size", ctypes.c_uint32),
        ("in_bo_handle_count", ctypes.c_uint32),
        ("out_bo_handle_count", ctypes.c_uint32),
    ]

class drm_rocket_submit(ctypes.Structure):
    _fields_ = [
        ("jobs", ctypes.c_uint64),
        ("job_count", ctypes.c_uint32),
        ("job_struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]

def _IOW(type_, nr, size):
    return (1 << 30) | (ord(type_) << 8) | nr | (size << 16)

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)

DRM_IOCTL_ROCKET_CREATE_BO = _IOWR('d', 0x40, ctypes.sizeof(drm_rocket_create_bo))
DRM_IOCTL_ROCKET_SUBMIT = _IOW('d', 0x41, ctypes.sizeof(drm_rocket_submit))
DRM_IOCTL_ROCKET_PREP_BO = _IOW('d', 0x42, ctypes.sizeof(drm_rocket_prep_bo))
DRM_IOCTL_ROCKET_FINI_BO = _IOW('d', 0x43, ctypes.sizeof(drm_rocket_fini_bo))

class RocketBO:
    __slots__ = ("handle", "size", "dma_address", "offset")
    def __init__(self, handle, size, dma_address, offset):
        self.handle = int(handle)
        self.size = int(size)
        self.dma_address = int(dma_address)
        self.offset = int(offset)

    @property
    def dma_addr(self):
        return self.dma_address

    @property
    def obj_addr(self):
        return self.handle

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

def open_rocket_device():
    path = os.environ.get("ROCKET_DEVICE")
    if path:
        return os.open(path, os.O_RDWR)
    candidates = (sorted(glob.glob("/dev/accel/accel*")) + sorted(glob.glob("/dev/dri/renderD*")) + sorted(glob.glob("/dev/dri/card*")))
    for c in candidates:
        try: return os.open(c, os.O_RDWR)
        except OSError: pass
    raise FileNotFoundError("No Rocket device found")

fd = open_rocket_device()

def mem_allocate(fd, size, flags=0):
    bo = drm_rocket_create_bo(size=size)
    ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, bo)
    buf = mmap.mmap(fd, bo.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return buf, RocketBO(bo.handle, bo.size, bo.dma_address, bo.offset)

def rocket_submit(fd, vendor_tasks, task_count=1, in_bos=(), out_bos=()):
    rocket_tasks = (drm_rocket_task * task_count)()
    for i in range(task_count):
        rocket_tasks[i].regcmd = int(vendor_tasks[i].regcmd_addr) & 0xFFFFFFFF
        rocket_tasks[i].regcmd_count = int(vendor_tasks[i].regcfg_amount)
    in_handles = (ctypes.c_uint32 * len(in_bos))(*(bo.handle for bo in in_bos)) if in_bos else None
    out_handles = (ctypes.c_uint32 * len(out_bos))(*(bo.handle for bo in out_bos)) if out_bos else None
    job = drm_rocket_job(
        tasks=ctypes.addressof(rocket_tasks),
        in_bo_handles=ctypes.addressof(in_handles) if in_handles is not None else 0,
        out_bo_handles=ctypes.addressof(out_handles) if out_handles is not None else 0,
        task_count=task_count,
        task_struct_size=ctypes.sizeof(drm_rocket_task),
        in_bo_handle_count=len(in_bos),
        out_bo_handle_count=len(out_bos),
    )
    jobs = (drm_rocket_job * 1)(job)
    submit_struct = drm_rocket_submit(
        jobs=ctypes.addressof(jobs),
        job_count=1,
        job_struct_size=ctypes.sizeof(drm_rocket_job),
    )
    return ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, submit_struct)

task_map, tasks_mem_create = mem_allocate(fd, size=1024)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=1024)
input_map, input_mem_create = mem_allocate(fd, size=4194304)
weight_map, weight_mem_create = mem_allocate(fd, size=4194304)
output_map, output_mem_create = mem_allocate(fd, size=4194304)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def run_op(ew_cfg_val, a_vals, b_vals, neg_op=False, fdiv_op=False):
    n = len(a_vals)
    dataout_width = (n + 7) // 8 - 1

    pc_tail = [
        (reg.TARGET_PC_REG << 48) | (0 << 16) | reg.PC_BASE_ADDRESS,
        (reg.TARGET_PC_REG << 48) | (0 << 16) | reg.PC_REGISTER_AMOUNTS,
        (reg.TARGET_VERSION << 48) | (0 << 16) | 0,
    ]
    npu_regs = [
        0x1001000001e5400c,
        0x1001480000024010,
        0x100100070007403c,
        0x1001000000004030,
        0x1001108202c04070,
        0x200100000000500c,
        0x2001000000005010,
        0x2001000700075014,
        0x2001400000085034,
        (reg.TARGET_DPU  << 48) | ((output_mem_create.dma_addr & 0xFFFFFFFF) << 16) | reg.DST_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | ((input_mem_create.dma_addr & 0xFFFFFFFF) << 16) | reg.RDMA_SRC_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | ((weight_mem_create.dma_addr & 0xFFFFFFFF) << 16) | reg.RDMA_EW_BASE_ADDR,
        0x2001000178495044,
    ] + pc_tail + [
        0x0081000000180008,
    ]

    for i in range(16):
        regcmd[i] = 0
    for i in range(len(npu_regs)):
        regcmd[i] = npu_regs[i]

    a_packed = np.array(a_vals, dtype=np.uint16).view(np.uint16)
    ct_inputs = (ctypes.c_uint16 * n).from_buffer(input_map)
    ct_inputs[:] = a_packed.tolist()

    w_vals = np.full(n, 0xffff, dtype=np.uint16) if neg_op else np.array(b_vals, dtype=np.uint16)
    w_packed = w_vals.view(np.uint16)
    ct_weights = (ctypes.c_uint16 * n).from_buffer(weight_map)
    ct_weights[:] = w_packed.tolist()

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = len(npu_regs)
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

    for bo in (regcmd_mem_create, input_mem_create, weight_mem_create, output_mem_create):
        ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))
    ret = rocket_submit(fd, tasks, 1,
        in_bos=[regcmd_mem_create, input_mem_create, weight_mem_create],
        out_bos=[output_mem_create])
    ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, drm_rocket_prep_bo(handle=output_mem_create.handle, timeout_ns=time.monotonic_ns() + 6000000000))
    print(f"SUBMIT ret={ret}")

    return np.frombuffer(output_map, dtype=np.uint16, count=n).copy().tolist()


_EW_BASE = 0x108002c0
EW_CFG_ADD  = _EW_BASE | (2 << 16)

OPS = {
    "ADD":  (EW_CFG_ADD, lambda a, b: a + b,       {}),
}

if __name__ == "__main__":
    a = np.array([3, 3, 3, 3, 3, 3, 3, 3], dtype=np.uint16)
    b = np.array([5, 5, 5, 5, 5, 5, 5, 5], dtype=np.uint16)

    b_vals_map = {"ADD": b}

    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"

    if mode == "ALL":
        run_list = list(OPS.items())
    elif mode in OPS:
        run_list = [(mode, OPS[mode])]
    else:
        print(f"Unknown mode '{mode}'. Options: ALL, {', '.join(OPS.keys())}")
        sys.exit(1)

    for op_name, (ew_cfg_val, expected_fn, kw) in run_list:
        r = run_op(ew_cfg_val=ew_cfg_val, a_vals=a, b_vals=b_vals_map[op_name], **kw)
        r_arr = np.array(r, dtype=np.uint16)
        expected = expected_fn(a, b_vals_map[op_name])
        match = np.allclose(r_arr, expected, atol=0.1)
        print(f"{op_name:4s} NPU={r_arr} expected={expected} {'PASS' if match else 'FAIL'}")

    os.close(fd)
