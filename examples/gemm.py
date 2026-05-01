from fcntl import ioctl
import os, mmap, sys
import ctypes
import numpy as np

class reg:
    TARGET_CNA  = 0x0201
    TARGET_CORE = 0x0801
    TARGET_DPU  = 0x1001
    TARGET_RDMA = 0x2001
    TARGET_PC   = 0x0081

    OPERATION_ENABLE    = 0x0008
    S_POINTER           = 0x4004
    FEATURE_MODE_CFG    = 0x400c
    DATA_FORMAT         = 0x4010
    DST_BASE_ADDR       = 0x4020
    DST_SURF_STRIDE     = 0x4024
    DATA_CUBE_WIDTH     = 0x4030
    DATA_CUBE_HEIGHT    = 0x4034
    DATA_CUBE_NOTCH     = 0x4038
    DATA_CUBE_CHANNEL   = 0x403c
    BS_CFG              = 0x4040
    BS_OW_CFG           = 0x4050
    WDMA_SIZE_0         = 0x4058
    WDMA_SIZE_1         = 0x405c
    BN_CFG              = 0x4060
    EW_CFG              = 0x4070
    OUT_CVT_SCALE       = 0x4084
    SURFACE_ADD         = 0x40c0

    RDMA_DATA_CUBE_WIDTH  = 0x500c
    RDMA_DATA_CUBE_HEIGHT = 0x5010
    RDMA_DATA_CUBE_CHANNEL= 0x5014
    RDMA_ERDMA_CFG        = 0x5034
    RDMA_SRC_BASE_ADDR    = 0x5018
    RDMA_EW_BASE_ADDR    = 0x5038
    RDMA_FEATURE_MODE_CFG = 0x5044
    RDMA_DMA_MAP          = 0x504c

    CNA_CONV_CON1          = 0x100c
    CNA_CONV_CON2          = 0x1010
    CNA_CONV_CON3          = 0x1014
    CNA_DATA_SIZE0         = 0x1020
    CNA_DATA_SIZE1         = 0x1024
    CNA_DATA_SIZE2         = 0x1028
    CNA_DATA_SIZE3         = 0x102c
    CNA_WEIGHT_SIZE0       = 0x1030
    CNA_WEIGHT_SIZE1       = 0x1034
    CNA_WEIGHT_SIZE2       = 0x1038
    CNA_CBUF_CON0          = 0x1040
    CNA_CBUF_CON1          = 0x1044
    CNA_CVT_CON0           = 0x104c
    CNA_CVT_CON1           = 0x1050
    CNA_CVT_CON2           = 0x1054
    CNA_CVT_CON3           = 0x1058
    CNA_CVT_CON4           = 0x105c
    CNA_FEATURE_DATA_ADDR  = 0x1070
    CNA_DMA_CON0           = 0x1078
    CNA_DMA_CON1           = 0x107c
    CNA_DMA_CON2           = 0x1080
    CNA_FC_DATA_SIZE0      = 0x1084
    CNA_FC_DATA_SIZE1      = 0x1088
    CNA_DCOMP_ADDR0        = 0x1110

    CORE_MISC_CFG          = 0x3010
    CORE_DATAOUT_SIZE_0    = 0x3014
    CORE_DATAOUT_SIZE_1    = 0x3018
    CORE_RESERVED_3030     = 0x3030

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1

fd = os.open("/dev/dri/card1", os.O_RDWR)

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
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    return buf, mem_create

def submit(task_obj_addr):
    submit_struct = rknpu_submit(
        flags=0x1,  # PC | BLOCK (NOT NONBLOCK: ioctl would return before NPU finishes)
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
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

def reset_npu(fd):
    action = rknpu_action(flags=RKNPU_ACT_RESET, value=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, action)

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=8192, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

_bufs = [task_map, regcmd_map, input_map, weight_map, output_map]

def reopen_device():
    global fd, task_map, tasks_mem_create, regcmd_map, regcmd_mem_create
    global input_map, input_mem_create, weight_map, weight_mem_create
    global output_map, output_mem_create, tasks, regcmd, _bufs
    for buf in _bufs:
        buf.close()
    os.close(fd)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem_create = mem_allocate(fd, size=8192, flags=RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    _bufs = [task_map, regcmd_map, input_map, weight_map, output_map]

# ---------------------------------------------------------------------------
# Packing dispatch: table of (m,n,k) -> (pack_input, pack_weight, decode_output)
# ---------------------------------------------------------------------------

def pack_input_row_major(m, n, k, a_matrix, in_pack, align_in):
    in_pack.reshape(m, align_in)[:, :k] = a_matrix[:, :k]

def pack_weight_row_major(m, n, k, b_matrix, wt_pack, align_in, align_out):
    wt_pack.reshape(align_out, align_in)[:n, :k] = b_matrix[:, :n].T

def pack_input_c2_8(m, n, k, a_matrix, in_pack, align_in):
    in_pack[:] = a_matrix[:, :k].reshape(m, -1, 8).transpose(1, 0, 2).ravel()

def pack_weight_tile_16x32(m, n, k, b_matrix, wt_pack, align_in, align_out):
    wt = np.zeros((align_out, align_in), dtype=np.float16)
    wt[:n, :k] = b_matrix.T[:n, :k]
    wt_pack[:] = wt.reshape(align_out // 16, 16, align_in // 32, 32
        ).transpose(0, 2, 1, 3).ravel()



def decode_output_linear(m, n, k, raw, align_out):
    row_start = np.arange(m) * align_out
    return raw[row_start[:, None] + np.arange(n)]

def decode_output_c2_4(m, n, k, raw, align_out):
    return raw[:n // 4 * m * 4].reshape(n // 4, m, 4).transpose(1, 0, 2).reshape(m, n).copy()

# Hardware constants
C2_INPUT = 8   # NC1HWC2 channel group size for input (derived from ATOMIC_K_SIZE=32)
C2_OUTPUT = 4  # NC1HWC2 channel group size for output (DPU formatter atomic width)

def _uses_c2_input(m, n, k):
    return (m, n, k) in ((64, 64, 64), (256, 256, 256))

def get_input_packer(m, n, k, align_in):
    if _uses_c2_input(m, n, k):
        return pack_input_c2_8
    return pack_input_row_major

def get_weight_packer(m, n, k, align_in):
    return pack_weight_tile_16x32

def get_output_decoder(m, n, k, align_out):
    return decode_output_c2_4 if _uses_c2_input(m, n, k) else decode_output_linear

# ---------------------------------------------------------------------------
# Register programming
# ---------------------------------------------------------------------------

def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma):
    align_in = max(32, ((k + 31) // 32) * 32)
    align_out = max(32, ((n + 31) // 32) * 32)
    if align_in < align_out:
        align_in = align_out  # pad K to match N alignment; CBUF readback issue at specific align_in values

    eff_k = k
    if align_in > max(32, ((k + 31) // 32) * 32):  # only when padding was actually applied
        eff_k = align_in  # use padded K for line_stride/surf_stride/notch calculations

    # GROUP_LINE_OFF = 0 when K==N (square matmul, CSC line offset would alias); else 1
    no_group_line_off = (k == n) and (align_in >= 64)

    # CDMA line_stride register (CNA_DMA_CON1): byte_stride = register << 5 (× ATOM_CUBE_SIZE=32).
    # For packed data (is_line_packed): byte_stride == ATOM_CUBE_SIZE * cube_width = 32 bytes.
    # For C2=8 NC1HWC2: 4 atoms = 128 bytes (matches M*C2/8 elements per C1-plane chunk for these shapes).
    # For row-major: burst covers ceil(eff_k/32) atoms of 32 bytes each.
    line_stride = 4
    if 32 < eff_k < 512 and not _uses_c2_input(m, n, k):
        line_stride = min(13, (eff_k + 31) // 32) * 4

    # surf_stride: non-zero only for C2=8 NC1HWC2 shapes, where surface groups
    # span multiple C1 planes in CBUF. For all other shapes: 0 (contiguous).
    surf_groups = m // 4
    surf_stride = 0
    if align_in >= 64 and _uses_c2_input(m, n, k):
        surf_stride = line_stride * (surf_groups - 1) + int(surf_groups == 0)

    input_bytes = 1 * m * align_in * 2
    data_bank = max(1, min(11, (input_bytes + 32767) // 32768))  # ceiling / 32768; +32767 NOT +32768 (extra bank at exact boundary)
    data_entries = (1 * align_in + 31) // 32

    # dst_surf_stride = align_out when k==n (no_group_line_off), else 1
    # is_matmul_64/256 special cases are redundant: align_out==64 for 64x64, 256 for 256x256
    dst_surf_stride = align_out if no_group_line_off else 1

    # feature_grains = CBUF capacity / bytes_per_line, derived from 2 banks (65536 bytes) / (align_in * 2)
    feature_grains = m + 1
    if eff_k > 7872:
        feature_grains = 2
    elif m > 80:
        denom = align_in * 2
        grains = (2 * 32768 + denom - 1) // denom
        grains = (grains + 1) & ~1
        feature_grains = max(80, grains)

    notch_blocks = min(13, align_out // 32)
    notch_val = 8 * notch_blocks - 1
    # Zero notch for square matmuls (no inter-atom padding needed);
    # also zero for very wide lines (K>7872) where CBUF is fully utilized per line
    if (k == n and align_out >= 64) or eff_k > 7872:
        notch_val = 0

    def E(target, reg_addr, value):
        return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

    conv_con1 = (2 << 7) | (2 << 4)
    if not no_group_line_off:
        conv_con1 |= 1 << 29

    npu_regs = [
        E(reg.TARGET_DPU,  reg.S_POINTER,       (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON1,    conv_con1),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON2,    feature_grains << 4),
        E(reg.TARGET_CNA,  reg.CNA_CONV_CON3,    (1 << 3) | 1),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE0,   (1 << 16) | m),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE1,   ((align_in - 1) << 16) | align_in),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE2,   1),
        E(reg.TARGET_CNA,  reg.CNA_DATA_SIZE3,   m),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE0, align_in * 2 * align_out),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE1, align_in * 2),
        E(reg.TARGET_CNA,  reg.CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | align_out),
        E(reg.TARGET_CNA,  reg.CNA_CBUF_CON0,    ((12 - data_bank) << 4) | data_bank),
        E(reg.TARGET_CNA,  reg.CNA_CBUF_CON1,    data_entries),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON0,     (1 << 3) | (1 << 1) | 1),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON1,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON2,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON3,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_CVT_CON4,     1 << 16),
        E(reg.TARGET_CNA,  reg.CNA_FEATURE_DATA_ADDR, in_dma & 0xFFFFFFFF),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON0,     (15 << 16) | 15),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON1,     line_stride),
        E(reg.TARGET_CNA,  reg.CNA_DMA_CON2,     surf_stride),
        E(reg.TARGET_CNA,  reg.CNA_FC_DATA_SIZE0, (1 << 16) | m),
        E(reg.TARGET_CNA,  reg.CNA_FC_DATA_SIZE1, align_in),
        # No REGCMD_RESERVED offset — regcmds in separate buffer, weights start at offset 0
        E(reg.TARGET_CNA,  reg.CNA_DCOMP_ADDR0,  wt_dma & 0xFFFFFFFF),
        E(reg.TARGET_CORE, reg.CORE_MISC_CFG,       (2 << 8) | 1),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_0, ((m - 1) << 16) | 0),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_1, align_out - 1),
        E(reg.TARGET_CORE, reg.CORE_RESERVED_3030,  0),
        E(reg.TARGET_DPU,  reg.FEATURE_MODE_CFG,  (15 << 5) | (2 << 1)),
        E(reg.TARGET_DPU,  reg.DATA_FORMAT,       (5 << 29) | (2 << 26) | 2),
        E(reg.TARGET_DPU,  reg.DST_BASE_ADDR,     out_dma & 0xFFFFFFFF),
        E(reg.TARGET_DPU,  reg.DST_SURF_STRIDE,   dst_surf_stride << 4),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_WIDTH,   0),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_HEIGHT,  m - 1),
        # Both halves of NOTCH and CHANNEL use same value (hi=lo) for matmul
        E(reg.TARGET_DPU,  reg.DATA_CUBE_NOTCH,   (notch_val << 16) | notch_val),
        E(reg.TARGET_DPU,  reg.DATA_CUBE_CHANNEL, ((align_out - 1) << 16) | (align_out - 1)),
        E(reg.TARGET_DPU,  reg.BS_CFG,            0x00000053),
        E(reg.TARGET_DPU,  reg.BS_OW_CFG,         0x0000036e),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_0,       align_out - 1),
        E(reg.TARGET_DPU,  reg.WDMA_SIZE_1,       ((m - 1) << 16) | 0),
        E(reg.TARGET_DPU,  reg.BN_CFG,            0x00000053),
        E(reg.TARGET_DPU,  reg.EW_CFG,            0x00000383),
        E(reg.TARGET_DPU,  reg.SURFACE_ADD,       (dst_surf_stride * 4) << 4),
        E(reg.TARGET_PC,   reg.OPERATION_ENABLE,  0x0000000d),
    ]
    return npu_regs

# ---------------------------------------------------------------------------
# Main GEMM execution
# ---------------------------------------------------------------------------

def run_gemm(m, n, k, a_matrix, b_matrix):
    align_in = max(32, ((k + 31) // 32) * 32)
    align_out = max(32, ((n + 31) // 32) * 32)
    pad_k = align_in < align_out
    if pad_k:
        align_in = align_out

    in_pack = np.zeros(align_in * m, dtype=np.float16)
    wt_pack = np.zeros(align_in * align_out, dtype=np.float16)

    pack_input = get_input_packer(m, n, k, align_in)
    pack_weight = get_weight_packer(m, n, k, align_in)

    if pad_k:
        pack_input = pack_input_row_major

    pack_input(m, n, k, a_matrix, in_pack, align_in)
    pack_weight(m, n, k, b_matrix, wt_pack, align_in, align_out)

    ct_inputs = (ctypes.c_uint16 * len(in_pack)).from_buffer(input_map)
    ct_weights = (ctypes.c_uint16 * len(wt_pack)).from_buffer(weight_map)
    ct_inputs[:] = in_pack.view(np.uint16).tolist()
    ct_weights[:] = wt_pack.view(np.uint16).tolist()

    out_nbytes = max(256, (m - 1) * align_out * 4 + n * 4)
    npu_regs = make_gemm_regs(m, n, k,
        input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)

    if "--dry" in sys.argv:
        print(f"\n=== GEMM {m}x{n}x{k} DRY RUN ===")
        target_names = {0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU", 0x2001: "RDMA", 0x0081: "PC"}
        for i, v in enumerate(npu_regs):
            tgt = (v >> 48) & 0xFFFF
            ra = v & 0xFFFF
            val = (v >> 16) & 0xFFFFFFFF
            print(f"  [{i:3d}] {target_names.get(tgt, f'0x{tgt:04x}')}[0x{ra:04x}] = 0x{val:08x}")
        return None

    for i in range(len(npu_regs)):
        regcmd[i] = npu_regs[i]
    for i in range(len(npu_regs), 64):
        regcmd[i] = 0

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;        # GEMM pipeline — NOT conv pipeline (op_idx=1, enable_mask=0xd)
    tasks[0].enable_mask = 0x18;# CSC | CMAC_A — enable mask for matmul sub-blocks
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = len(npu_regs)
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

    reset_npu(fd)
    ret = submit(tasks_mem_create.obj_addr)

    raw = np.frombuffer(output_map, dtype=np.float32, count=out_nbytes // 4).copy()

    decode_output = get_output_decoder(m, n, k, align_out)
    out = decode_output(m, n, k, raw, align_out)
    return out

if __name__ == "__main__":
    test_cases = [
        (2, 2, 1,
        np.array([[1, 2], [3, 4]], dtype=np.float16),
        np.array([[5, 6], [7, 8]], dtype=np.float16)),
    ]
    np.random.seed(42)
    for mnk in [(8, 8, 8), (9, 9, 9), (64, 64, 64), (256, 256, 256)]:
        m, n, k = mnk
        a = np.random.randn(m, k).astype(np.float16)
        b = np.random.randn(k, n).astype(np.float16)
        test_cases.append((m, n, k, a, b))

    for m, n, k, a, b in test_cases:
        print(f"\n{m}x{n}x{k}:")
        r = run_gemm(m, n, k, a, b)
        if r is None:
            continue
        expected = a @ b
        ok = np.allclose(r, expected, atol=0.1)
        md = np.max(np.abs(r - expected))
        print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")

    os.close(fd)
