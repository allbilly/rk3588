"""NPU GEMM for first-principles CONV fallback (1x1 and im2col spatial).

Self-contained (no import from conv.py) to avoid circular deps / keep conv lean.
"""
import os, mmap, ctypes
from fcntl import ioctl
import numpy as np

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 0 << 1
FP16_BYTES = 2
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
MIN_CHANNEL_TILE = 32
RK_LINE_STRIDE_GROUP_CAP = 13
RK_MIN_WIDE_FEATURE_GRAINS = 80
PC_CHAIN_TAIL_QWORDS = 4
GEMM_INPUT_BANKS = RK_CBUF_BANKS - 2
GEMM_MAX_ALIGN_IN = RK_CBUF_BANKS * MIN_CHANNEL_TILE
SPATIAL_ROWS_PER_CHUNK = 2048


class reg:
    CNA = 0x0201; CORE = 0x0801; DPU = 0x1001; PC = 0x0081; PC_REG = 0x0101; VERSION = 0x0041
    OPERATION_ENABLE = 0x0008; PC_BASE_ADDRESS = 0x0010; PC_REGISTER_AMOUNTS = 0x0014
    S_POINTER = 0x4004; FEATURE_MODE_CFG = 0x400c; DATA_FORMAT = 0x4010
    DST_BASE_ADDR = 0x4020; DST_SURF_STRIDE = 0x4024
    DATA_CUBE_WIDTH = 0x4030; DATA_CUBE_HEIGHT = 0x4034; DATA_CUBE_NOTCH = 0x4038
    DATA_CUBE_CHANNEL = 0x403c; BS_CFG = 0x4040; BS_OW_CFG = 0x4050
    WDMA_SIZE_0 = 0x4058; WDMA_SIZE_1 = 0x405c; BN_CFG = 0x4060; EW_CFG = 0x4070
    OUT_CVT_SCALE = 0x4084; SURFACE_ADD = 0x40c0
    CNA_CONV_CON1 = 0x100c; CNA_CONV_CON2 = 0x1010; CNA_CONV_CON3 = 0x1014
    CNA_DATA_SIZE0 = 0x1020; CNA_DATA_SIZE1 = 0x1024; CNA_DATA_SIZE2 = 0x1028
    CNA_DATA_SIZE3 = 0x102c; CNA_WEIGHT_SIZE0 = 0x1030; CNA_WEIGHT_SIZE1 = 0x1034
    CNA_WEIGHT_SIZE2 = 0x1038; CNA_CBUF_CON0 = 0x1040; CNA_CBUF_CON1 = 0x1044
    CNA_CVT_CON0 = 0x104c; CNA_CVT_CON1 = 0x1050; CNA_CVT_CON2 = 0x1054
    CNA_CVT_CON3 = 0x1058; CNA_CVT_CON4 = 0x105c
    CNA_FEATURE_DATA_ADDR = 0x1070; CNA_DMA_CON0 = 0x1078; CNA_DMA_CON1 = 0x107c
    CNA_DMA_CON2 = 0x1080; CNA_FC_DATA_SIZE0 = 0x1084; CNA_FC_DATA_SIZE1 = 0x1088
    CNA_DCOMP_ADDR0 = 0x1110
    CORE_MISC_CFG = 0x3010; CORE_DATAOUT_SIZE_0 = 0x3014; CORE_DATAOUT_SIZE_1 = 0x3018
    CORE_RESERVED_3030 = 0x3030


class rknpu_mem_create(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("size", ctypes.c_uint64),
                ("obj_addr", ctypes.c_uint64), ("dma_addr", ctypes.c_uint64), ("sram_size", ctypes.c_uint64)]


class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64)]


class rknpu_mem_destroy(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("obj_addr", ctypes.c_uint64)]


class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]


class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]


class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32), ("timeout", ctypes.c_uint32), ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32), ("task_counter", ctypes.c_uint32), ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64), ("iommu_domain_id", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("task_base_addr", ctypes.c_uint64), ("hw_elapse_time", ctypes.c_int64), ("core_mask", ctypes.c_uint32),
        ("fence_fd", ctypes.c_int32), ("subcore_task", rknpu_subcore_task * 5),
    ]


class struct_rknpu_task(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("op_idx", ctypes.c_uint32), ("enable_mask", ctypes.c_uint32),
                ("int_mask", ctypes.c_uint32), ("int_clear", ctypes.c_uint32), ("int_status", ctypes.c_uint32),
                ("regcfg_amount", ctypes.c_uint32), ("regcfg_offset", ctypes.c_uint32), ("regcmd_addr", ctypes.c_uint64)]


def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)


DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_MEM_DESTROY = _IOWR('d', 0x44, ctypes.sizeof(rknpu_mem_destroy))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))


def _ceil_div(x, y):
    return (x + y - 1) // y


def _align_up(x, align):
    return _ceil_div(x, align) * align


def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr


def mem_allocate(fd, size, flags=0):
    create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, create)
    mapping = rknpu_mem_map(handle=create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mapping)
    return mmap.mmap(fd, create.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                     offset=mapping.offset), create


def mem_destroy(fd, mem_create):
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY,
          rknpu_mem_destroy(handle=mem_create.handle, obj_addr=mem_create.obj_addr))


def close_allocations(fd, allocations):
    for mapping, mem in allocations:
        try:
            mapping.close()
        finally:
            mem_destroy(fd, mem)


def post_submit_reset(fd):
    for flags in (1, 6, 1):
        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=flags, value=0))


def _gemm_layout(m, n, k):
    aligned_k = max(MIN_CHANNEL_TILE, _align_up(k, MIN_CHANNEL_TILE))
    align_out = max(MIN_CHANNEL_TILE, _align_up(n, MIN_CHANNEL_TILE))
    align_in = max(aligned_k, align_out)
    eff_k = align_in if align_in != aligned_k else k
    return align_in, align_out, eff_k


def _gemm_output_indices(m, n, align_out):
    # FP32 DPU surface: m rows × align_out contiguous useful cols 0..n-1
    return (np.arange(m, dtype=np.int64) * align_out)[:, None] + np.arange(n, dtype=np.int64)[None, :]


def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma):
    align_in, align_out, eff_k = _gemm_layout(m, n, k)
    input_row_bytes = align_in * FP16_BYTES
    even_rows = (_ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1
    feature_grains = max(RK_MIN_WIDE_FEATURE_GRAINS, even_rows)
    data_banks = int(np.clip(_ceil_div(m * input_row_bytes, CBUF_BANK_SIZE), 1, RK_CBUF_BANKS - 1))
    line_stride = 4 * min(_ceil_div(eff_k, MIN_CHANNEL_TILE), RK_LINE_STRIDE_GROUP_CAP)
    notch_val = 8 * min(align_out // MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP) - 1
    return [
        E(reg.DPU, reg.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.CNA, reg.CNA_CONV_CON1, (2 << 4) | (2 << 7) | (1 << 29)),
        E(reg.CNA, reg.CNA_CONV_CON2, feature_grains << 4),
        E(reg.CNA, reg.CNA_CONV_CON3, (1 << 3) | 1),
        E(reg.CNA, reg.CNA_DATA_SIZE0, (1 << 16) | m),
        E(reg.CNA, reg.CNA_DATA_SIZE1, ((align_in - 1) << 16) | align_in),
        E(reg.CNA, reg.CNA_DATA_SIZE2, 1),
        E(reg.CNA, reg.CNA_DATA_SIZE3, m),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, input_row_bytes * align_out),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, input_row_bytes),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | align_out),
        E(reg.CNA, reg.CNA_CBUF_CON0, ((RK_CBUF_BANKS - data_banks) << 4) | data_banks),
        E(reg.CNA, reg.CNA_CBUF_CON1, _ceil_div(align_in, MIN_CHANNEL_TILE)),
        E(reg.CNA, reg.CNA_CVT_CON0, (1 << 3) | (1 << 1) | 1),
        E(reg.CNA, reg.CNA_CVT_CON1, 1 << 16), E(reg.CNA, reg.CNA_CVT_CON2, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON3, 1 << 16), E(reg.CNA, reg.CNA_CVT_CON4, 1 << 16),
        E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        E(reg.CNA, reg.CNA_DMA_CON1, line_stride), E(reg.CNA, reg.CNA_DMA_CON2, 0),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, (1 << 16) | m),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, align_in), E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CORE, reg.CORE_MISC_CFG, (2 << 8) | 1),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, ((m - 1) << 16)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, align_out - 1), E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1)),
        E(reg.DPU, reg.DATA_FORMAT, (5 << 29) | (2 << 26) | 2),
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma), E(reg.DPU, reg.DST_SURF_STRIDE, 1 << 4),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, 0), E(reg.DPU, reg.DATA_CUBE_HEIGHT, m - 1),
        E(reg.DPU, reg.DATA_CUBE_NOTCH, (notch_val << 16) | notch_val),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL, ((align_out - 1) << 16) | (align_out - 1)),
        E(reg.DPU, reg.BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.BS_OW_CFG, (3 << 8) | (3 << 5) | (3 << 2) | (1 << 1)),
        E(reg.DPU, reg.WDMA_SIZE_0, align_out - 1),
        E(reg.DPU, reg.WDMA_SIZE_1, ((m - 1) << 16)),
        E(reg.DPU, reg.BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
        E(reg.DPU, reg.OUT_CVT_SCALE, 0),
        E(reg.DPU, reg.SURFACE_ADD, (1 * 4) << 4),
    ]


def _pack_gemm_input(a_matrix, m, k, align_in):
    packed = np.zeros((m, align_in), dtype=np.float16)
    packed[:, :k] = a_matrix[:, :k]
    return packed.ravel().view(np.uint16)


def _pack_gemm_weight(b_matrix, n, k, align_in, align_out):
    weight = np.zeros((align_out, align_in), dtype=np.float16)
    weight[:n, :k] = b_matrix.T[:n, :k]
    packed = weight.reshape(align_out // 16, 16, align_in // 32, 32).transpose(0, 2, 1, 3).ravel()
    return packed.view(np.uint16)


def _write_gemm_tasks(task_map, regcmd_map, regcmd_mem, task_regs):
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    offsets, offset = [], 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + PC_CHAIN_TAIL_QWORDS, 2)
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            next_addr = regcmd_mem.dma_addr + offsets[idx + 1] * 8
            tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xfffffff0),
                    E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(len(task_regs[idx + 1]), 2) + 1),
                    E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
        else:
            tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, 0), E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                    E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
        for i, qword in enumerate(tail):
            regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 0
        tasks[idx].enable_mask = 0xd
        tasks[idx].int_mask = 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = len(regs)
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * 8


def _npu_submit_gemm(fd, task_obj_addr, task_count):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))
    # Match examples/gemm.py: PC | BLOCK | PINGPONG
    flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | (1 << 2)
    submit = rknpu_submit(flags=flags, timeout=6000, task_start=0,
                          task_number=task_count, task_counter=0, priority=0, task_obj_addr=task_obj_addr,
                          iommu_domain_id=0, reserved=0, task_base_addr=0, hw_elapse_time=0,
                          core_mask=1, fence_fd=-1)
    submit.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit)


def run_gemm_matrix(a_matrix, b_matrix, out_h, out_w):
    """a: M×K, b: K×N → NCHW (1,N,out_h,out_w) FP16. M must equal out_h*out_w."""
    m, k = a_matrix.shape
    n = b_matrix.shape[1]
    if b_matrix.shape[0] != k or m != out_h * out_w:
        raise ValueError("GEMM shape mismatch")
    align_in, align_out, _ = _gemm_layout(m, n, k)
    input_row_bytes = align_in * FP16_BYTES
    row_stride_bytes = align_out * 4
    input_flat = np.ascontiguousarray(_pack_gemm_input(a_matrix, m, k, align_in))
    weight_flat = np.ascontiguousarray(_pack_gemm_weight(b_matrix, n, k, align_in, align_out))
    m_tile = GEMM_INPUT_BANKS * CBUF_BANK_SIZE // input_row_bytes if align_in <= GEMM_MAX_ALIGN_IN else 1
    n_tiles = _ceil_div(m, m_tile)
    out_bytes = max(4096, m * row_stride_bytes)
    placeholder = [make_gemm_regs(min(m_tile, m - i * m_tile), n, k, 0, 0, 0) for i in range(n_tiles)]
    regcmd_qwords = sum(_align_up(len(r) + PC_CHAIN_TAIL_QWORDS, 2) for r in placeholder)

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, max(4096, n_tiles * ctypes.sizeof(struct_rknpu_task)),
                                      RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, max(4096, regcmd_qwords * 8), RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, max(4096, input_flat.nbytes), RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, max(4096, weight_flat.nbytes), RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, out_bytes, RKNPU_MEM_NON_CACHEABLE)
    try:
        ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), input_flat.ctypes.data, input_flat.nbytes)
        ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), weight_flat.ctypes.data, weight_flat.nbytes)
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, len(output_map))
        task_regs = []
        for start in range(0, m, m_tile):
            tile_m = min(m_tile, m - start)
            task_regs.append(make_gemm_regs(
                tile_m, n, k,
                input_mem.dma_addr + start * input_row_bytes,
                weight_mem.dma_addr,
                output_mem.dma_addr + start * row_stride_bytes))
        _write_gemm_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        if _npu_submit_gemm(fd, task_mem.obj_addr, len(task_regs)) < 0:
            raise RuntimeError("gemm npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float32, count=m * align_out).copy()
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    result = out_raw[_gemm_output_indices(m, n, align_out)].reshape(out_h, out_w, n)
    return result.transpose(2, 0, 1)[None].astype(np.float16), len(task_regs)


def run_pointwise_gemm(s, inp, wt):
    """1x1 CONV as GEMM: M=H*W, K=in_c, N=out_c. Chunk tall maps to keep BOs small."""
    b = wt[:, :, 0, 0].T.copy()
    got = np.zeros((s["batch"], s["out_c"], s["in_h"], s["in_w"]), dtype=np.float16)
    tasks = 0
    # Cap rows so input BO stays under ~2MB (align_in≈max(K,N) padded).
    align_guess = max(MIN_CHANNEL_TILE, _align_up(max(s["in_c"], s["out_c"]), MIN_CHANNEL_TILE))
    max_rows = max(1, (2 * 1024 * 1024) // max(1, s["in_w"] * align_guess * FP16_BYTES))
    for n in range(s["batch"]):
        for y0 in range(0, s["in_h"], max_rows):
            y1 = min(s["in_h"], y0 + max_rows)
            band = inp[n, :, y0:y1, :]
            m = (y1 - y0) * s["in_w"]
            a = band.transpose(1, 2, 0).reshape(m, s["in_c"])
            tile, t = run_gemm_matrix(a, b, y1 - y0, s["in_w"])
            got[n, :, y0:y1, :] = tile[0]
            tasks += t
    return got, tasks, "pointwise_gemm"


def _im2col(inp_chw, kh, kw, stride, out_h, out_w):
    cols = np.zeros((out_h * out_w, inp_chw.shape[0] * kh * kw), dtype=np.float16)
    idx = 0
    for oh in range(out_h):
        for ow in range(out_w):
            patch = inp_chw[:, oh * stride:oh * stride + kh, ow * stride:ow * stride + kw]
            cols[idx] = patch.reshape(-1)
            idx += 1
    return cols


def run_spatial_gemm(s, inp, wt):
    """Spatial CONV via im2col + GEMM, chunked by output rows."""
    stride = s.get("stride", 1)
    out_h = (s["in_h"] - s["kh"]) // stride + 1
    out_w = (s["in_w"] - s["kw"]) // stride + 1
    k = s["in_c"] * s["kh"] * s["kw"]
    b = wt.reshape(s["out_c"], k).T.copy()
    got = np.zeros((s["batch"], s["out_c"], out_h, out_w), dtype=np.float16)
    tasks = 0
    rows_per = max(1, SPATIAL_ROWS_PER_CHUNK // max(1, out_w))
    for n in range(s["batch"]):
        for y0 in range(0, out_h, rows_per):
            y1 = min(out_h, y0 + rows_per)
            in_y0 = y0 * stride
            in_y1 = (y1 - 1) * stride + s["kh"]
            band = inp[n, :, in_y0:in_y1, :]
            cols = _im2col(band, s["kh"], s["kw"], stride, y1 - y0, out_w)
            tile, t = run_gemm_matrix(cols, b, y1 - y0, out_w)
            got[n, :, y0:y1, :] = tile[0]
            tasks += t
    return got, tasks, "spatial_gemm"
