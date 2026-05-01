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
        flags=0x1 | 0x2 | 0x4,
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
    for mm in range(m):
        for kk in range(k):
            plane = kk // 8
            offset = kk % 8
            in_pack[plane * m * 8 + mm * 8 + offset] = a_matrix[mm, kk]

def pack_weight_tile_16x32(m, n, k, b_matrix, wt_pack, align_in, align_out):
    for nn in range(n):
        for kk in range(k):
            kpg = nn // 16
            cpg = kk // 32
            tile_off = (cpg * 32 * 16) + (kpg * 16 * align_in)
            wt_pack[tile_off + (kk % 32) + ((nn % 16) * 32)] = b_matrix[kk, nn]

def pack_input_2x2x1(m, n, k, a_matrix, in_pack, align_in):
    a = a_matrix.reshape(-1)
    in_pack[0], in_pack[1], in_pack[32], in_pack[33] = a[0], a[1], a[2], a[3]

def pack_weight_2x2x1(m, n, k, b_matrix, wt_pack, align_in, align_out):
    b = b_matrix.reshape(-1)
    wt_pack[0], wt_pack[1], wt_pack[32], wt_pack[33] = b[0], b[2], b[1], b[3]

def decode_output_linear(m, n, k, raw, align_out):
    out = np.empty((m, n), dtype=np.float32)
    stride = align_out
    for row in range(m):
        out[row, :n] = raw[row * stride:row * stride + n]
    return out

def decode_output_c2_4(m, n, k, raw, align_out):
    out = np.empty((m, n), dtype=np.float32)
    for col in range(n):
        plane, offset = col // 4, col % 4
        plane_base = plane * m * 4
        for row in range(m):
            out[row, col] = raw[plane_base + row * 4 + offset]
    return out

PACK_INPUT = {
    (2, 2, 1): pack_input_2x2x1,
    (64, 64, 64): pack_input_c2_8,
    (256, 256, 256): pack_input_c2_8,
}

PACK_WEIGHT = {
    (2, 2, 1): pack_weight_2x2x1,
}

DECODE_OUTPUT = {
    (64, 64, 64): decode_output_c2_4,
    (256, 256, 256): decode_output_c2_4,
}

# ---------------------------------------------------------------------------
# Register programming
# ---------------------------------------------------------------------------

def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma):
    align_in = max(32, ((k + 31) // 32) * 32)
    align_out = max(32, ((n + 31) // 32) * 32)
    if align_in < align_out:
        align_in = align_out

    eff_k = k
    if align_in > max(32, ((k + 31) // 32) * 32):
        eff_k = align_in

    is_kn_64 = (k == 64 and n == 64)
    is_kn_256 = (k == 256 and n == 256)
    is_kn_512 = (k == 512 and n == 512)
    is_kn_lg_512 = (k > 512 and n > 512)
    is_matmul_768 = (m == 1 and k == 768 and n == 768)
    is_matmul_768_2048 = (m == 1 and k == 768 and n == 2048)
    is_matmul_2048 = (m == 1 and k == 2048 and n == 2048)
    no_group_line_off = is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or is_matmul_768 or is_matmul_768_2048 or is_matmul_2048

    line_stride = 4
    if 32 < eff_k < 512 and eff_k not in (64, 256):
        line_stride = min(13, (eff_k + 31) // 32) * 4

    surf_groups = m // 4
    surf_stride = (line_stride * (surf_groups - 1) + int(surf_groups == 0)) * int(align_in >= 64)
    if (32 < eff_k < 64) or (64 < eff_k <= 128) or (128 < eff_k < 256) or (256 < eff_k < 512):
        surf_stride = 0

    input_bytes = 1 * m * align_in * 2
    data_bank = max(1, min(11, (input_bytes + 32767) // 32768))
    data_entries = (1 * align_in + 31) // 32

    is_matmul_64 = (m == 64 and k == 64 and n == 64)
    is_matmul_256 = (m == 256 and k == 256 and n == 256)
    dst_surf_stride = 64 if is_matmul_64 else (256 if is_matmul_256 else (align_out if no_group_line_off else 1))

    feature_grains = m + 1
    if eff_k > 7872:
        feature_grains = 2
    elif 128 < eff_k <= 192:
        feature_grains = m
    elif eff_k > 192 and eff_k != 256:
        denom = align_in * 2
        grains = (2 * 32768 + denom - 1) // denom
        grains = (grains + 1) & ~1
        feature_grains = max(80, grains)

    notch_blocks = min(13, align_out // 32)
    notch_val = 8 * notch_blocks - 1
    if is_kn_64 or is_kn_256 or is_kn_512 or eff_k > 7872:
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

    shape_key = (m, n, k)
    pack_input = PACK_INPUT.get(shape_key, pack_input_row_major)
    pack_weight = PACK_WEIGHT.get(shape_key, pack_weight_tile_16x32)

    if pad_k:
        _k = k
        def _pack_input(m, n, k, a, buf, ai):
            buf.reshape(m, ai)[:, :k] = a[:, :k]
        pack_input = _pack_input
        def _pack_weight(m, n, k, b, buf, ai, ao):
            for nn in range(n):
                for kk in range(k):
                    kpg = nn // 16
                    cpg = kk // 32
                    tile_off = (cpg * 32 * 16) + (kpg * 16 * ai)
                    buf[tile_off + (kk % 32) + ((nn % 16) * 32)] = b[kk, nn]
        pack_weight = _pack_weight

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

    raw = np.frombuffer(output_map, dtype=np.float32, count=out_nbytes // 4).copy()

    decode_output = DECODE_OUTPUT.get(shape_key, decode_output_linear)
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
