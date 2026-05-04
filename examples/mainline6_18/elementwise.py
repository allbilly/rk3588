import glob
import os, mmap, sys
import ctypes
import time
from fcntl import ioctl
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
    RDMA_EW_BASE_ADDR    = 0x5038   # RDMA EW base address (weight)
    RDMA_FEATURE_MODE_CFG = 0x5044   # RDMA feature mode config
    RDMA_S_POINTER       = 0x5004   # RDMA S pointer config

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

DRM_COMMAND_BASE = 0x40
DRM_ROCKET_CREATE_BO = 0x00
DRM_ROCKET_SUBMIT = 0x01
DRM_ROCKET_PREP_BO = 0x02
DRM_ROCKET_FINI_BO = 0x03


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


DRM_IOCTL_ROCKET_CREATE_BO = _IOWR(
    "d", DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO, ctypes.sizeof(drm_rocket_create_bo)
)
DRM_IOCTL_ROCKET_SUBMIT = _IOW(
    "d", DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT, ctypes.sizeof(drm_rocket_submit)
)
DRM_IOCTL_ROCKET_PREP_BO = _IOW(
    "d", DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO, ctypes.sizeof(drm_rocket_prep_bo)
)
DRM_IOCTL_ROCKET_FINI_BO = _IOW(
    "d", DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO, ctypes.sizeof(drm_rocket_fini_bo)
)


def open_rocket_device():
    if path := os.environ.get("ROCKET_DEVICE"):
        return os.open(path, os.O_RDWR)
    candidates = (
        sorted(glob.glob("/dev/accel/accel*"))
        + sorted(glob.glob("/dev/dri/renderD*"))
        + sorted(glob.glob("/dev/dri/card*"))
    )
    last_error = None
    for path in candidates:
        try:
            return os.open(path, os.O_RDWR)
        except OSError as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise FileNotFoundError("No /dev/accel/accel*, /dev/dri/renderD*, or /dev/dri/card* device found")


fd = open_rocket_device()

def mem_allocate(fd, size):
    bo = drm_rocket_create_bo(size=size)
    ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, bo)
    buf = mmap.mmap(fd, bo.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    print(f"BO handle={bo.handle}, dma={bo.dma_address:#x}, offset={bo.offset:#x}, size={bo.size}")
    return buf, bo


def prep_bo(bo, timeout_ns=6_000_000_000):
    if timeout_ns > 0:
        timeout_ns = time.monotonic_ns() + timeout_ns
    ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, drm_rocket_prep_bo(handle=bo.handle, timeout_ns=timeout_ns))


def fini_bo(bo):
    ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))


def submit(regcmd_bo, regcmd_count, in_bos, out_bos):
    tasks = (drm_rocket_task * 1)(
        drm_rocket_task(regcmd=regcmd_bo.dma_address & 0xFFFFFFFF, regcmd_count=regcmd_count)
    )
    in_handles = (ctypes.c_uint32 * len(in_bos))(*(bo.handle for bo in in_bos))
    out_handles = (ctypes.c_uint32 * len(out_bos))(*(bo.handle for bo in out_bos))
    jobs = (drm_rocket_job * 1)(
        drm_rocket_job(
            tasks=ctypes.addressof(tasks),
            in_bo_handles=ctypes.addressof(in_handles),
            out_bo_handles=ctypes.addressof(out_handles),
            task_count=1,
            task_struct_size=ctypes.sizeof(drm_rocket_task),
            in_bo_handle_count=len(in_bos),
            out_bo_handle_count=len(out_bos),
        )
    )
    submit_struct = drm_rocket_submit(
        jobs=ctypes.addressof(jobs),
        job_count=1,
        job_struct_size=ctypes.sizeof(drm_rocket_job),
    )
    return ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, submit_struct)

# EW_CFG values for each op
# Base config: data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1
_EW_BASE = 0x108002c0
EW_CFG_ADD  = _EW_BASE | (2 << 16)
EW_CFG_MUL  = _EW_BASE | (1 << 2) | (1 << 8)
EW_CFG_SUB  = _EW_BASE | (4 << 16)
EW_CFG_MAX  = _EW_BASE
EW_CFG_NEG  = EW_CFG_MUL

regcmd_map, regcmd_mem_create = mem_allocate(fd, size=4096)
input_map, input_mem_create = mem_allocate(fd, size=4096)
weight_map, weight_mem_create = mem_allocate(fd, size=4096)
output_map, output_mem_create = mem_allocate(fd, size=4096)

regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))


def run_op(ew_cfg_val, a_vals, b_vals, neg_op=False, fdiv_op=False):
    n = len(a_vals)
    dataout_width = (n + 7) // 8 - 1

    for bo in (regcmd_mem_create, input_mem_create, weight_mem_create, output_mem_create):
        prep_bo(bo)

    out_cvt = 1 if fdiv_op else 0x10001
    feat_cfg = 0x00017841 if fdiv_op else 0x00017849
    npu_regs = [
        (reg.TARGET_DPU  << 48) | (0x0000000e << 16) | reg.S_POINTER,
        (reg.TARGET_DPU  << 48) | (0x000001e5 << 16) | reg.FEATURE_MODE_CFG,
        (reg.TARGET_DPU  << 48) | (0x48000002 << 16) | reg.DATA_FORMAT,
        (reg.TARGET_DPU  << 48) | (dataout_width << 16) | reg.DATA_CUBE_WIDTH,
        (reg.TARGET_DPU  << 48) | (0x00070007 << 16) | reg.DATA_CUBE_CHANNEL,
        (reg.TARGET_DPU  << 48) | (ew_cfg_val << 16) | reg.EW_CFG,
        (reg.TARGET_DPU  << 48) | (out_cvt << 16) | reg.OUT_CVT_SCALE,
        (reg.TARGET_RDMA << 48) | (0x0000000e << 16) | reg.RDMA_S_POINTER,
        (reg.TARGET_RDMA << 48) | (dataout_width << 16) | reg.RDMA_DATA_CUBE_WIDTH,
        (reg.TARGET_RDMA << 48) | (0x00000000 << 16) | reg.RDMA_DATA_CUBE_HEIGHT,
        (reg.TARGET_RDMA << 48) | (0x00000007 << 16) | reg.RDMA_DATA_CUBE_CHANNEL,
        (reg.TARGET_RDMA << 48) | (0x40000008 << 16) | reg.RDMA_ERDMA_CFG,
        (reg.TARGET_DPU  << 48) | ((output_mem_create.dma_address & 0xFFFFFFFF) << 16) | reg.DST_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | ((input_mem_create.dma_address & 0xFFFFFFFF) << 16) | reg.RDMA_SRC_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | ((weight_mem_create.dma_address & 0xFFFFFFFF) << 16) | reg.RDMA_EW_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | (feat_cfg << 16) | reg.RDMA_FEATURE_MODE_CFG,
        (reg.TARGET_PC   << 48) | (0x00000018 << 16) | reg.OPERATION_ENABLE,
    ]

    for i in range(512):
        regcmd[i] = 0
    for i in range(len(npu_regs)):
        regcmd[i] = npu_regs[i]

    a_packed = np.array(a_vals, dtype=np.float16).view(np.uint16)
    ct_inputs = (ctypes.c_uint16 * n).from_buffer(input_map)
    ct_inputs[:] = a_packed.tolist()

    w_vals = np.full(n, -1.0, dtype=np.float16) if neg_op else np.array(b_vals, dtype=np.float16)
    w_packed = w_vals.view(np.uint16)
    ct_weights = (ctypes.c_uint16 * n).from_buffer(weight_map)
    ct_weights[:] = w_packed.tolist()

    ct_outputs = (ctypes.c_uint16 * n).from_buffer(output_map)
    ct_outputs[:] = [0] * n

    for bo in (regcmd_mem_create, input_mem_create, weight_mem_create, output_mem_create):
        fini_bo(bo)

    ret = submit(
        regcmd_mem_create,
        len(npu_regs),
        in_bos=(regcmd_mem_create, input_mem_create, weight_mem_create),
        out_bos=(output_mem_create,),
    )
    prep_bo(output_mem_create)
    print(f"SUBMIT ret={ret}")

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
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float16)
    b = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float16)
    c = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float16)

    b_vals_map = {"ADD": b, "MUL": b, "SUB": b, "MAX": c, "NEG": b, "FDIV": c}

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
        r_arr = np.array(r, dtype=np.float16)
        expected = expected_fn(a, b_vals_map[op_name])
        match = np.allclose(r_arr, expected, atol=0.1)
        print(f"{op_name:4s} NPU={r_arr} expected={expected} {'PASS' if match else 'FAIL'}")

    os.close(fd)
