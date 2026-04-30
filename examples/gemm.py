from fcntl import ioctl
import os, mmap, ctypes, struct, math
import numpy as np

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
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc

def submit(task_obj_addr):
    s = rknpu_submit(
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
    s.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    s.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    s.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    s.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    s.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, s)

def reset_npu(fd):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

def emit(target, value, addr):
    return ((target & 0xFFFF) << 48) | ((value & 0xFFFFFFFF) << 16) | (addr & 0xFFFF)

def mv_address(mv):
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))

def _align_up(v, a):
    return ((v + a - 1) // a) * a if a > 0 else v

def _wmma_params(m, n, k):
    m = max(1, m); n = max(1, n); k = max(1, k)
    align_in = max(32, _align_up(k, 32))
    align_out = max(32, _align_up(n, 32))
    data_in_width = 1; data_in_height = m
    dataout_width = 1; dataout_height = m
    out_width_stride = 1
    is_kn_64 = (k == 64 and n == 64)
    is_kn_256 = (k == 256 and n == 256)
    is_kn_512 = (k == 512 and n == 512)
    is_kn_lg_512 = (k > 512 and n > 512)
    is_matmul_64 = (m == 64 and n == 64 and k == 64)
    is_matmul_256 = (m == 256 and n == 256 and k == 256)
    feature_grains = data_in_height + 1
    if k > 7872:
        feature_grains = 2
    elif 128 < k <= 192:
        feature_grains = data_in_height
    elif k > 192 and k != 256:
        denom = align_in * 2
        grains = max(80, ((2 * 32768 + denom - 1) // denom))
        grains = (grains + 1) & ~1
        feature_grains = grains
    weight_bytes_per_kernel = align_in * 2
    fd_bytes = data_in_width * data_in_height * align_in * 2
    data_bank = max(1, min(11, int((fd_bytes + 32767) // 32768)))
    line_stride = data_in_width * 4
    if 32 < k < 512 and k not in (64, 256):
        line_stride = min(13, (k + 31) // 32) * 4
    surf_groups = data_in_height // 4
    surf_stride = (line_stride * (surf_groups - 1) + int(surf_groups == 0)) * int(align_in >= 64)
    if (32 < k < 64) or (64 < k <= 128) or (128 < k < 256) or (256 < k < 512):
        surf_stride = 0
    dst_surf_stride = 64 if is_matmul_64 else (256 if is_matmul_256 else out_width_stride)
    notch_blocks = min(13, align_out // 32)
    notch_val = 8 * notch_blocks - 1
    if is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or k > 7872:
        notch_val = 0
    return {
        "m": m, "n": n, "k": k, "align_in": align_in, "align_out": align_out,
        "data_in_width": data_in_width, "data_in_height": data_in_height,
        "dataout_width": dataout_width, "dataout_height": dataout_height,
        "feature_grains": feature_grains, "weight_bytes_per_kernel": weight_bytes_per_kernel,
        "data_bank": data_bank, "line_stride": line_stride, "surf_stride": surf_stride,
        "dst_surf_stride": dst_surf_stride, "notch_val": notch_val,
    }

def pack_input_fp16(a_matrix, p):
    m, k = p["m"], p["k"]
    align_in = p["align_in"]
    dst = np.zeros(align_in * m, dtype=np.float16)
    if (m, k) == (64, 64):
        for mm in range(1, m + 1):
            for kk in range(1, k + 1):
                plane = (kk - 1) // 8
                offset = (kk - 1) % 8
                dst[plane * m * 8 + (mm - 1) * 8 + offset] = a_matrix[mm - 1, kk - 1]
    else:
        d2 = dst.reshape(m, align_in)
        d2[:, :k] = a_matrix
    return dst

def pack_weight_fp16(b_matrix, p):
    n, k = p["n"], p["k"]
    align_in = p["align_in"]
    align_out = p["align_out"]
    dst = np.zeros(align_out * align_in, dtype=np.float16)
    if (n, k) == (64, 64):
        for nn in range(1, n + 1):
            for kk in range(1, k + 1):
                kpg = (nn - 1) // 16
                cpg = (kk - 1) // 32
                idx = ((cpg * 32) * 16) + (kpg * 16 * align_in) + ((kk - 1) % 32) + (((nn - 1) % 16) * 32)
                dst[idx] = b_matrix[kk - 1, nn - 1]
    else:
        d2 = dst.reshape(align_out, align_in)
        d2[:n, :k] = b_matrix.T
    return dst

def unpack_output_fp32(raw, p):
    m, n = p["m"], p["n"]
    if (m, n) in [(64, 64), (256, 256)]:
        c2 = 4
        out = np.empty((m, n), dtype=np.float32)
        for col in range(n):
            plane = col // c2
            offset = col % c2
            plane_base = plane * m * c2
            for row in range(m):
                out[row, col] = raw[plane_base + row * c2 + offset]
        return out
    else:
        stride = p["align_out"]
        out = np.empty((m, n), dtype=np.float32)
        for row in range(m):
            out[row, :] = raw[row * stride : row * stride + n]
        return out

def run_matmul(a_matrix, b_matrix):
    m, k = a_matrix.shape
    kn, n = b_matrix.shape
    assert kn == k

    p = _wmma_params(m, n, k)

    input_packed = pack_input_fp16(a_matrix, p)
    weight_packed = pack_weight_fp16(b_matrix, p)

    input_bytes = len(input_packed) * 2
    weight_bytes = len(weight_packed) * 2
    out_stride = p["align_out"] * 4
    output_bytes = max(0x100, (p["m"] - 1) * out_stride + p["n"] * 4)
    cmd_buf_size = 16384

    reset_npu(fd)
    task_map, task_mc = mem_allocate(fd, cmd_buf_size, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    cmd_map, cmd_mc = mem_allocate(fd, cmd_buf_size, RKNPU_MEM_NON_CACHEABLE)
    in_map, in_mc = mem_allocate(fd, max(input_bytes, 4096), RKNPU_MEM_NON_CACHEABLE)
    wt_map, wt_mc = mem_allocate(fd, max(weight_bytes, 4096), RKNPU_MEM_NON_CACHEABLE)
    out_map, out_mc = mem_allocate(fd, max(output_bytes, 4096), RKNPU_MEM_NON_CACHEABLE)

    in_mv = memoryview(bytearray(input_packed.tobytes()))
    wt_mv = memoryview(bytearray(weight_packed.tobytes()))
    ctypes.memmove(mv_address(in_map), mv_address(in_mv), input_bytes)
    ctypes.memmove(mv_address(wt_map), mv_address(wt_mv), weight_bytes)

    in_verify = np.frombuffer(in_map.read(input_bytes), dtype=np.float16).copy()
    in_map.seek(0)
    wt_verify = np.frombuffer(wt_map.read(weight_bytes), dtype=np.float16).copy()
    wt_map.seek(0)
    print(f"  in packed non_zero={np.count_nonzero(in_verify)} weight non_zero={np.count_nonzero(wt_verify)}")
    print(f"  expected in[0:8]={a_matrix[0,:8]}")
    print(f"  actual in packed first 16={in_verify[:16]}")

    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)),
                        ctypes.POINTER(struct_rknpu_task))
    regs = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(cmd_map)),
                       ctypes.POINTER(ctypes.c_uint64))

    feature_addr = in_mc.dma_addr & 0xFFFFFFFF
    weight_addr = wt_mc.dma_addr & 0xFFFFFFFF
    dst_addr = out_mc.dma_addr & 0xFFFFFFFF

    is_kn_64 = (p["k"] == 64 and p["n"] == 64)
    is_kn_256 = (p["k"] == 256 and p["n"] == 256)
    is_kn_512 = (p["k"] == 512 and p["n"] == 512)
    is_kn_lg_512 = (p["k"] > 512 and p["n"] > 512)
    is_m_1_kn_768 = (p["m"] == 1 and p["k"] == 768 and p["n"] == 768)
    is_m_1_k768_n2048 = (p["m"] == 1 and p["k"] == 768 and p["n"] == 2048)
    is_m_1_kn_2048 = (p["m"] == 1 and p["k"] == 2048 and p["n"] == 2048)

    conv_con1 = (2 << 7) | (2 << 4)
    if not (is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or
            is_m_1_kn_768 or is_m_1_k768_n2048 or is_m_1_kn_2048):
        conv_con1 |= (1 << 29)

    weight_bytes_total = p["weight_bytes_per_kernel"] * p["align_out"]
    data_entries = (p["data_in_width"] * p["align_in"] + 31) // 32

    npu_regs = [
        emit(0x1001, (1 << 3) | (1 << 2) | (1 << 1), 0x4004),
        emit(0x0201, conv_con1, 0x100c),
        emit(0x0201, p["feature_grains"], 0x1010),
        emit(0x0201, (1 << 16) | 1, 0x1014),
        emit(0x0201, p["data_in_width"] | (p["data_in_height"] << 16), 0x1020),
        emit(0x0201, (p["align_in"] - 1) | (p["align_in"] << 16), 0x1024),
        emit(0x0201, p["dataout_width"], 0x1028),
        emit(0x0201, p["dataout_width"] * p["dataout_height"], 0x102c),
        emit(0x0201, weight_bytes_total, 0x1030),
        emit(0x0201, p["weight_bytes_per_kernel"], 0x1034),
        emit(0x0201, 1 | (1 << 16) | (p["align_out"] << 24), 0x1038),
        emit(0x0201, ((12 - p["data_bank"]) << 4) | p["data_bank"], 0x1040),
        emit(0x0201, data_entries, 0x1044),
        emit(0x0201, (1 << 3) | (1 << 2) | 1, 0x104c),
        emit(0x0201, 1 << 16, 0x1050),
        emit(0x0201, 1 << 16, 0x1054),
        emit(0x0201, 1 << 16, 0x1058),
        emit(0x0201, 1 << 16, 0x105c),
        emit(0x0201, feature_addr, 0x1070),
        emit(0x0201, (15 << 16) | 15, 0x1078),
        emit(0x0201, p["line_stride"], 0x107c),
        emit(0x0201, p["surf_stride"], 0x1080),
        emit(0x0201, p["data_in_width"] | (p["data_in_height"] << 16), 0x1084),
        emit(0x0201, p["align_in"], 0x1088),
        emit(0x0201, weight_addr, 0x1110),
        emit(0x0801, (2 << 8) | 1, 0x3010),
        emit(0x0801, (p["dataout_height"] - 1) | ((p["dataout_width"] - 1) << 16), 0x3014),
        emit(0x0801, p["align_out"] - 1, 0x3018),
        emit(0x1001, (15 << 5) | (2 << 1), 0x400c),
        emit(0x1001, (5 << 29) | (2 << 26) | 2, 0x4010),
        emit(0x1001, dst_addr, 0x4020),
        emit(0x1001, p["dst_surf_stride"], 0x4024),
        emit(0x1001, p["dataout_width"] - 1, 0x4030),
        emit(0x1001, p["dataout_height"] - 1, 0x4034),
        emit(0x1001, p["notch_val"] | (p["notch_val"] << 16), 0x4038),
        emit(0x1001, (p["align_out"] - 1) | ((p["align_out"] - 1) << 16), 0x403c),
        emit(0x1001, (1 << 6) | (1 << 4) | (1 << 1) | 1, 0x4040),
        emit(0x1001, (3 << 8) | (3 << 5) | (3 << 2) | (1 << 1), 0x4050),
        emit(0x1001, p["align_out"] - 1, 0x4058),
        emit(0x1001, (p["dataout_height"] - 1) | ((p["dataout_width"] - 1) << 16), 0x405c),
        emit(0x1001, (1 << 6) | (1 << 4) | (1 << 1) | 1, 0x4060),
        emit(0x1001, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1, 0x4070),
        emit(0x1001, p["dst_surf_stride"] * 4, 0x40c0),
        emit(0x0081, 0x0d, 0x0008),
    ]

    for i in range(len(npu_regs)):
        regs[i] = npu_regs[i]

    tasks[0].flags = 0
    tasks[0].op_idx = 4
    tasks[0].enable_mask = 0x18
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(npu_regs)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = cmd_mc.dma_addr

    ret = submit(task_mc.obj_addr)
    print(f"SUBMIT ret={ret}")
    print(f"  M={p['m']} N={p['n']} K={p['k']}")
    print(f"  in_dma=0x{in_mc.dma_addr:x}, wt_dma=0x{wt_mc.dma_addr:x}, out_dma=0x{out_mc.dma_addr:x}")

    out_mv = memoryview(bytearray(output_bytes))
    ctypes.memmove(mv_address(out_mv), ctypes.addressof(ctypes.c_char.from_buffer(out_map)), output_bytes)
    raw = np.frombuffer(out_mv.tobytes(), dtype=np.float32).copy()
    non_zero = np.count_nonzero(raw)
    print(f"  raw min={raw.min():.6f} max={raw.max():.6f} non_zero={non_zero}/{len(raw)}")
    if non_zero > 0:
        print(f"  first non-zero indices: {np.nonzero(raw[:100])[0][:10]}")
        print(f"  raw[:16]={raw[:16]}")
    return unpack_output_fp32(raw, p), a_matrix, b_matrix


if __name__ == "__main__":
    np.random.seed(42)
    M, N, K = 64, 64, 64
    a = np.random.randn(M, K).astype(np.float16)
    b = np.random.randn(K, N).astype(np.float16)

    result, a_mat, b_mat = run_matmul(a, b)
    expected = np.dot(a_mat.astype(np.float32), b_mat.astype(np.float32))
    match = np.allclose(result, expected, atol=0.5)
    print(f"GEMM {M}x{N}x{K}: {'PASS' if match else 'FAIL'}")
    if not match:
        max_err = np.max(np.abs(result - expected))
        print(f"  max_err={max_err:.6f}")
        print(f"  NPU[0,0]={result[0,0]:.6f} CPU[0,0]={expected[0,0]:.6f}")
