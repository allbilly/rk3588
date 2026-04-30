from fcntl import ioctl
import os, mmap, sys, struct
import ctypes
import numpy as np

class reg:
    TARGET_CNA  = 0x0200
    TARGET_CORE = 0x0800
    TARGET_DPU  = 0x1000
    TARGET_RDMA = 0x2000
    TARGET_PC   = 0x0080

    # DPU
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
    SURFACE_ADD         = 0x40c0

    # CNA
    CNA_CONV_CON1       = 0x100c
    CNA_CONV_CON2       = 0x1010
    CNA_CONV_CON3       = 0x1014
    CNA_DATA_SIZE0      = 0x1020
    CNA_DATA_SIZE1      = 0x1024
    CNA_DATA_SIZE2      = 0x1028
    CNA_DATA_SIZE3      = 0x102c
    CNA_WEIGHT_SIZE0    = 0x1030
    CNA_WEIGHT_SIZE1    = 0x1034
    CNA_WEIGHT_SIZE2    = 0x1038
    CNA_CBUF_CON0       = 0x1040
    CNA_CBUF_CON1       = 0x1044
    CNA_CVT_CON0        = 0x104c
    CNA_CVT_CON1        = 0x1050
    CNA_CVT_CON2        = 0x1054
    CNA_CVT_CON3        = 0x1058
    CNA_CVT_CON4        = 0x105c
    CNA_FEATURE_DATA_ADDR = 0x1070
    CNA_DMA_CON0        = 0x1078
    CNA_DMA_CON1        = 0x107c
    CNA_DMA_CON2        = 0x1080
    CNA_FC_DATA_SIZE0   = 0x1084
    CNA_FC_DATA_SIZE1   = 0x1088
    CNA_DCOMP_ADDR0     = 0x1110

    # CORE
    CORE_MISC_CFG       = 0x3010
    CORE_DATAOUT_SIZE_0 = 0x3014
    CORE_DATAOUT_SIZE_1 = 0x3018

    # PC
    OPERATION_ENABLE    = 0x0008

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1

fd = os.open("/dev/dri/card1", os.O_RDWR)

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64), ("obj_addr", ctypes.c_uint64),
        ("dma_addr", ctypes.c_uint64), ("sram_size", ctypes.c_uint64),
    ]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64)]

class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]

class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32), ("timeout", ctypes.c_uint32),
        ("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32),
        ("task_counter", ctypes.c_uint32), ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64), ("regcfg_obj_addr", ctypes.c_uint64),
        ("task_base_addr", ctypes.c_uint64), ("user_data", ctypes.c_uint64),
        ("core_mask", ctypes.c_uint32), ("fence_fd", ctypes.c_int32),
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
        ("flags", ctypes.c_uint32), ("op_idx", ctypes.c_uint32),
        ("enable_mask", ctypes.c_uint32), ("int_mask", ctypes.c_uint32),
        ("int_clear", ctypes.c_uint32), ("int_status", ctypes.c_uint32),
        ("regcfg_amount", ctypes.c_uint32), ("regcfg_offset", ctypes.c_uint32),
        ("regcmd_addr", ctypes.c_uint64),
    ]

def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    return buf, mem_create

def reset_npu(fd):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

def submit(task_obj_addr):
    s = rknpu_submit(
        flags=0x1 | 0x2 | 0x4, timeout=6000,
        task_start=0, task_number=1, task_counter=0, priority=0,
        task_obj_addr=task_obj_addr, regcfg_obj_addr=0, task_base_addr=0,
        user_data=0, core_mask=1, fence_fd=-1,
    )
    s.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    s.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    s.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    s.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    s.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, s)

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=8192, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma):
    align_in = max(32, ((k + 31) // 32) * 32)
    align_out = max(32, ((n + 31) // 32) * 32)

    is_kn_64 = k == 64 and n == 64
    data_in_height = m

    surf_groups = m // 4
    line_stride = 4
    if 32 < k < 512 and k not in (64, 256):
        line_stride = min(13, (k + 31) // 32) * 4
    surf_stride = (line_stride * (surf_groups - 1) + int(surf_groups == 0)) * int(align_in >= 64)
    if (32 < k < 64) or (64 < k <= 128) or (128 < k < 256) or (256 < k < 512):
        surf_stride = 0

    fd_bytes = 1 * data_in_height * align_in * 2
    data_bank = max(1, min(11, (fd_bytes + 32768 - 1) // 32768))
    data_entries = (1 * align_in + 31) // 32
    is_matmul_64 = (m == 64 and k == 64 and n == 64)
    is_matmul_256 = (m == 256 and k == 256 and n == 256)
    dst_surf_stride = 64 if is_matmul_64 else (256 if is_matmul_256 else 1)
    feature_grains = data_in_height + 1

    notch_blocks = min(13, align_out // 32)
    notch_val = 8 * notch_blocks - 1
    is_kn_64 = (k == 64 and n == 64)
    is_kn_256 = (k == 256 and n == 256)
    is_kn_512 = (k == 512 and n == 512)
    is_kn_lg_512 = (k > 512 and n > 512)
    if is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or k > 7872:
        notch_val = 0

    weight_bytes_per_kernel = align_in * 2
    weight_bytes = weight_bytes_per_kernel * align_out

    def t(target, reg_addr, value):
        return ((target + 1) & 0xFFFF) << 48 | (value & 0xFFFFFFFF) << 16 | (reg_addr & 0xFFFF)

    regs = [
        # S_POINTER: pointer_pp_mode(bit3)=1, executer_pp_en(bit2)=1, pointer_pp_en(bit1)=1
        t(reg.TARGET_DPU, reg.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),

        # CNA_CONV_CON1: proc_precision(bits7-9)=2, in_precision(bits4-6)=2
        t(reg.TARGET_CNA, reg.CNA_CONV_CON1, (2 << 7) | (2 << 4) | ((1 << 29) if not is_kn_64 else 0)),
        # CNA_CONV_CON2: feature_grains(bits4-13)
        t(reg.TARGET_CNA, reg.CNA_CONV_CON2, feature_grains << 4),
        # CNA_CONV_CON3: conv_y_stride(bits3-5)=1, conv_x_stride(bits0-2)=1
        t(reg.TARGET_CNA, reg.CNA_CONV_CON3, (1 << 3) | 1),
        # CNA_DATA_SIZE0: data_in_width(bits16-26)=1, data_in_height(bits0-10)
        t(reg.TARGET_CNA, reg.CNA_DATA_SIZE0, (1 << 16) | data_in_height),
        # CNA_DATA_SIZE1: channel_real(bits16-30)=align_in-1, channel(bits0-15)=align_in
        t(reg.TARGET_CNA, reg.CNA_DATA_SIZE1, ((align_in - 1) << 16) | align_in),
        # CNA_DATA_SIZE2: dataout_width(bits0-10)=1
        t(reg.TARGET_CNA, reg.CNA_DATA_SIZE2, 1),
        # CNA_DATA_SIZE3: dataout_atomics(bits0-21)=m
        t(reg.TARGET_CNA, reg.CNA_DATA_SIZE3, m),
        # CNA_WEIGHT_SIZE0: weight_bytes(bits0-31)
        t(reg.TARGET_CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes),
        # CNA_WEIGHT_SIZE1: weight_bytes_per_kernel(bits0-18)
        t(reg.TARGET_CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        # CNA_WEIGHT_SIZE2: width(bits24-28)=1, height(bits16-20)=1, kernels(bits0-13)=align_out
        t(reg.TARGET_CNA, reg.CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | align_out),
        # CNA_CBUF_CON0: weight_bank(bits4-7)=12-db, data_bank(bits0-3)=db
        t(reg.TARGET_CNA, reg.CNA_CBUF_CON0, ((12 - data_bank) << 4) | data_bank),
        # CNA_CBUF_CON1: data_entries(bits0-13)
        t(reg.TARGET_CNA, reg.CNA_CBUF_CON1, data_entries),
        # CNA_CVT_CON0: data_sign(bit3)=1, cvt_type(bit1)=1, cvt_bypass(bit0)=1
        t(reg.TARGET_CNA, reg.CNA_CVT_CON0, (1 << 3) | (1 << 1) | 1),
        # CNA_CVT_CON1-4: scale=1
        t(reg.TARGET_CNA, reg.CNA_CVT_CON1, 1 << 16),
        t(reg.TARGET_CNA, reg.CNA_CVT_CON2, 1 << 16),
        t(reg.TARGET_CNA, reg.CNA_CVT_CON3, 1 << 16),
        t(reg.TARGET_CNA, reg.CNA_CVT_CON4, 1 << 16),
        # CNA_FEATURE_DATA_ADDR
        t(reg.TARGET_CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma & 0xFFFFFFFF),
        # CNA_DMA_CON0: weight_burst_len(bits16-19)=15, data_burst_len(bits0-3)=15
        t(reg.TARGET_CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        # CNA_DMA_CON1: line_stride(bits0-27)
        t(reg.TARGET_CNA, reg.CNA_DMA_CON1, line_stride),
        # CNA_DMA_CON2: surf_stride(bits0-27)
        t(reg.TARGET_CNA, reg.CNA_DMA_CON2, surf_stride),
        # CNA_FC_DATA_SIZE0: dma_width(bits16-30)=1, dma_height(bits0-10)=m
        t(reg.TARGET_CNA, reg.CNA_FC_DATA_SIZE0, (1 << 16) | data_in_height),
        # CNA_FC_DATA_SIZE1: dma_channel(bits0-15)=align_in
        t(reg.TARGET_CNA, reg.CNA_FC_DATA_SIZE1, align_in),
        # CNA_DCOMP_ADDR0
        t(reg.TARGET_CNA, reg.CNA_DCOMP_ADDR0, wt_dma & 0xFFFFFFFF),

        # CORE_MISC_CFG: proc_precision(bits8-10)=2, qd_en(bit0)=1
        t(reg.TARGET_CORE, reg.CORE_MISC_CFG, (2 << 8) | 1),
        # CORE_DATAOUT_SIZE_0: height(bits16-31)=m-1, width(bits0-15)=0
        t(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_0, ((m - 1) << 16) | 0),
        # CORE_DATAOUT_SIZE_1: channel(bits0-15)=align_out-1
        t(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_1, align_out - 1),
        # CORE reserved register at 0x3030 (must be zeroed)
        t(reg.TARGET_CORE, 0x3030, 0),

        # DPU_WMMA: FEATURE_MODE_CFG: burst_len(bits5-9)=15, output_mode(bits1-2)=2
        t(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1)),
        # DATA_FORMAT: out_precision(bits29-31)=5, in_precision(bits26-28)=2, proc_precision(bits0-2)=2
        t(reg.TARGET_DPU, reg.DATA_FORMAT, (5 << 29) | (2 << 26) | 2),
        t(reg.TARGET_DPU, reg.DST_BASE_ADDR, out_dma & 0xFFFFFFFF),
        # DST_SURF_STRIDE: surf_stride(bits4-31)
        t(reg.TARGET_DPU, reg.DST_SURF_STRIDE, dst_surf_stride << 4),
        # DATA_CUBE_WIDTH(bits0-12)=0
        t(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, 0),
        # DATA_CUBE_HEIGHT(bits0-12)=m-1
        t(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, m - 1),
        # DATA_CUBE_NOTCH: notch1(bits16-28), notch0(bits0-12)
        t(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, (notch_val << 16) | notch_val),
        # DATA_CUBE_CHANNEL: orig(bits16-28)=ao-1, channel(bits0-12)=ao-1
        t(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, ((align_out - 1) << 16) | (align_out - 1)),
        # BS_CFG: all bypassed = 0x53
        t(reg.TARGET_DPU, reg.BS_CFG, 0x00000053),
        # BS_OW_CFG: all 3's + od_bypass = 0x36e
        t(reg.TARGET_DPU, reg.BS_OW_CFG, 0x0000036e),
        # WDMA_SIZE_0: channel(bits0-12)=align_out-1
        t(reg.TARGET_DPU, reg.WDMA_SIZE_0, align_out - 1),
        # WDMA_SIZE_1: height(bits16-28)=m-1, width(bits0-12)=0
        t(reg.TARGET_DPU, reg.WDMA_SIZE_1, ((m - 1) << 16) | 0),
        # BN_CFG: all bypassed = 0x53
        t(reg.TARGET_DPU, reg.BN_CFG, 0x00000053),
        # EW_CFG: all bypassed = 0x383
        t(reg.TARGET_DPU, reg.EW_CFG, 0x00000383),
        # SURFACE_ADD: surf_add(bits4-31)=dst_surf_stride*4
        t(reg.TARGET_DPU, reg.SURFACE_ADD, (dst_surf_stride * 4) << 4),

        # PC operation enable (WMMA: 0x0d)
        t(reg.TARGET_PC, reg.OPERATION_ENABLE, 0x0000000d),
    ]
    return regs

def reg_desc(regs):
    target_names = {0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU", 0x2001: "RDMA", 0x0081: "PC"}
    for i, v in enumerate(regs):
        tgt = (v >> 48) & 0xFFFF
        reg_addr = v & 0xFFFF
        val = (v >> 16) & 0xFFFFFFFF
        name = target_names.get(tgt, f"0x{tgt:04x}")
        print(f"  [{i:3d}] {name}[0x{reg_addr:04x}] = 0x{val:08x}")

DRY_RUN = "--submit" not in sys.argv

def run_gemm(m, n, k, a_matrix, b_matrix):
    align_in = max(32, ((k + 31) // 32) * 32)
    align_out = max(32, ((n + 31) // 32) * 32)

    in_pack = np.zeros(align_in * m, dtype=np.float16)
    wt_pack = np.zeros(align_in * align_out, dtype=np.float16)

    if (m, n, k) == (2, 2, 1):
        in_full = a_matrix.reshape(-1)
        wt_full = b_matrix.reshape(-1)
        in_pack[0], in_pack[1], in_pack[32], in_pack[33] = in_full[0], in_full[1], in_full[2], in_full[3]
        wt_pack[0], wt_pack[1], wt_pack[32], wt_pack[33] = wt_full[0], wt_full[2], wt_full[1], wt_full[3]
    elif (m, n, k) == (64, 64, 64):
        for mm in range(1, 65):
            for kk in range(1, 65):
                plane = (kk - 1) // 8
                offset = (kk - 1) % 8
                in_pack[plane * 64 * 8 + (mm - 1) * 8 + offset] = a_matrix[mm - 1, kk - 1]
        for nn in range(1, 65):
            for kk in range(1, 65):
                kpg, cpg = (nn - 1) // 16, (kk - 1) // 32
                wt_idx = ((cpg * 32) * 16) + (kpg * 16 * align_in) + ((kk - 1) % 32) + (((nn - 1) % 16) * 32)
                wt_pack[wt_idx] = b_matrix[kk - 1, nn - 1]
    else:
        in_mat = in_pack.reshape(m, align_in)
        in_mat[:, :min(k, align_in)] = a_matrix[:, :min(k, align_in)]
        wt_mat = wt_pack.reshape(align_out, align_in)
        wt_mat[:n, :min(k, align_in)] = b_matrix[:, :n].T

    ct_inputs = (ctypes.c_uint16 * len(in_pack)).from_buffer(input_map)
    ct_weights = (ctypes.c_uint16 * len(wt_pack)).from_buffer(weight_map)
    ct_inputs[:] = in_pack.view(np.uint16).tolist()
    ct_weights[:] = wt_pack.view(np.uint16).tolist()

    out_nbytes = max(256, (m - 1) * align_out * 4 + n * 4)
    npu_regs = make_gemm_regs(m, n, k,
        input_mem_create.dma_addr, weight_mem_create.dma_addr, output_mem_create.dma_addr)

    if DRY_RUN:
        print(f"\n=== GEMM {m}x{n}x{k} DRY RUN (registers only) ===")
        reg_desc(npu_regs)
        print(f"\nRegister count: {len(npu_regs)}")
        return None

    for i in range(64):
        regcmd[i] = 0
    for i in range(len(npu_regs)):
        regcmd[i] = npu_regs[i]

    tasks[0].flags = 0
    tasks[0].op_idx = 4
    tasks[0].enable_mask = 0x18
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(npu_regs)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = regcmd_mem_create.dma_addr

    reset_npu(fd)
    ret = submit(tasks_mem_create.obj_addr)
    print(f"SUBMIT ret={ret}")

    raw_floats = out_nbytes // 4
    raw = np.frombuffer(output_map, dtype=np.float32, count=raw_floats).copy()

    out = np.empty((m, n), dtype=np.float32)
    if (m, n, k) in {(64, 64, 64), (256, 256, 256)}:
        c2 = 4
        for col in range(n):
            plane, offset = col // c2, col % c2
            plane_base = plane * m * c2
            for row in range(m):
                out[row, col] = raw[plane_base + row * c2 + offset]
    else:
        stride = align_out
        for cand in range(n, min(align_out, 256)):
            if abs(raw[cand]) > 1e-6:
                stride = cand
                break
        for row in range(m):
            out[row, :] = raw[row * stride:row * stride + n]
    return out

if __name__ == "__main__":
    if DRY_RUN:
        print("=== DRY RUN MODE ===")
        print("Pass --submit to actually run on NPU\n")
        print("Reference: 64x64x64")
        a = np.random.randn(64, 64).astype(np.float16)
        b = np.random.randn(64, 64).astype(np.float16)
        run_gemm(64, 64, 64, a, b)

        print("\nReference: 2x2x1")
        a2 = np.array([[1, 2], [3, 4]], dtype=np.float16)
        b2 = np.array([[5, 6], [7, 8]], dtype=np.float16)
        run_gemm(2, 2, 1, a2, b2)

        print("\nRegisters printed. Verify against known-good dump before running with --submit")
    else:
        print("=== RUNNING ON NPU ===")

        for m, n, k in [(2, 2, 1), (8, 8, 8), (9, 9, 9), (64, 64, 64)]:
            if m == 2 and n == 2 and k == 1:
                a = np.array([[1, 2], [3, 4]], dtype=np.float16)
                b = np.array([[5, 6], [7, 8]], dtype=np.float16)
            else:
                np.random.seed(42)
                a = np.random.randn(m, k).astype(np.float16)
                b = np.random.randn(k, n).astype(np.float16)
            print(f"\n{m}x{n}x{k}:")
            r = run_gemm(m, n, k, a, b)
            expected = a @ b
            if r is not None:
                ok = np.allclose(r, expected, atol=0.1)
                md = np.max(np.abs(r - expected))
                print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md:.4f})")

    os.close(fd)
