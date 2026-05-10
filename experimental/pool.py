from fcntl import ioctl
import os, mmap, sys
import ctypes
import numpy as np
import argparse

class reg:
    # --- Stream/Target IDs (shifted into bits 48-63) ---
    PC = 0x0100     # PC (Program Control / operation enable)
    PC_REG = 0x0100 # PC chain registers
    VERSION = 0x0040
    PPU = 0x4000    # PPU (Pooling Processing Unit)
    PPU_RDMA = 0x8000 # PPU RDMA (Read DMA for pooling)

    # --- PC (0x0000) ---
    REG_PC_OPERATION_ENABLE    = 0x0008   # PC operation enable
    REG_PC_BASE_ADDRESS        = 0x0010   # next regcmd DMA address for PC chain
    REG_PC_REGISTER_AMOUNTS    = 0x0014   # next regcmd fetch amount for PC chain

    # --- PPU (0x6000) ---
    REG_PPU_S_POINTER            = 0x6004   # PPU S pointer config (pp/exec)
    REG_PPU_DATA_CUBE_IN_WIDTH   = 0x600c   # PPU input data cube width
    REG_PPU_DATA_CUBE_IN_HEIGHT  = 0x6010   # PPU input data cube height
    REG_PPU_DATA_CUBE_IN_CHANNEL = 0x6014   # PPU input data cube channel
    REG_PPU_DATA_CUBE_OUT_WIDTH  = 0x6018   # PPU output data cube width
    REG_PPU_DATA_CUBE_OUT_HEIGHT = 0x601c   # PPU output data cube height
    REG_PPU_DATA_CUBE_OUT_CHANNEL= 0x6020   # PPU output data cube channel
    REG_PPU_OPERATION_MODE_CFG   = 0x6024   # PPU operation mode config (pool method, flying)
    REG_PPU_POOLING_KERNEL_CFG   = 0x6034   # PPU pooling kernel config (size/stride)
    REG_PPU_RECIP_KERNEL_WIDTH   = 0x6038   # PPU reciprocal kernel width (avg pool)
    REG_PPU_RECIP_KERNEL_HEIGHT  = 0x603c   # PPU reciprocal kernel height (avg pool)
    REG_PPU_DST_BASE_ADDR        = 0x6070   # PPU destination base address
    REG_PPU_DST_SURF_STRIDE      = 0x607c   # PPU destination surface stride
    REG_PPU_DATA_FORMAT          = 0x6084   # PPU data format config
    REG_PPU_MISC_CTRL            = 0x60dc   # PPU misc control (burst length)

    # --- PPU RDMA (0x7000) ---
    REG_PPU_RDMA_RDMA_S_POINTER          = 0x7004   # PPU RDMA S pointer config
    REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH      = 0x700c   # PPU RDMA input cube width
    REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT     = 0x7010   # PPU RDMA input cube height
    REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL    = 0x7014   # PPU RDMA input cube channel
    REG_PPU_RDMA_RDMA_SRC_BASE_ADDR      = 0x701c   # PPU RDMA source base address
    REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE    = 0x7024   # PPU RDMA source line stride
    REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE    = 0x7028   # PPU RDMA source surface stride
    REG_PPU_RDMA_RDMA_DATA_FORMAT        = 0x7030   # PPU RDMA data format config
    REG_PPU_RDMA_RDMA_OPERATION_ENABLE   = 0x7038   # PPU RDMA operation enable

RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0 << 1
RKNPU_JOB_NONBLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 0x4
FP16_BYTES = 2
PC_CHAIN_TAIL_QWORDS = 4
TASK_BUF_SIZE = 64 * 1024
REGCMD_BUF_SIZE = 512 * 1024
INPUT_BUF_SIZE = 4 * 1024 * 1024
OUTPUT_BUF_SIZE = 4 * 1024 * 1024

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

class rknpu_mem_destroy(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("obj_addr", ctypes.c_uint64),
    ]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]

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


class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]

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


DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_MEM_DESTROY = _IOWR("d", 0x44, ctypes.sizeof(rknpu_mem_destroy))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR("d", 0x41, ctypes.sizeof(rknpu_submit))
DRM_IOCTL_RKNPU_ACTION = _IOWR("d", 0x40, ctypes.sizeof(rknpu_action))

def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(
            flags=flags,
            size=size,
    )
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    print(f"ret={ret}, handle={mem_create.handle}, obj_addr={mem_create.obj_addr:#x}, dma_addr={mem_create.dma_addr:#x}")
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    print(f"Memory mapped at offset={mem_map.offset:#x}")
    return buf, mem_create

def mem_destroy(fd, mem_create):
    return ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY,
                              rknpu_mem_destroy(handle=mem_create.handle, obj_addr=mem_create.obj_addr))

def submit(fd, task_obj_addr, task_count=1):
    submit_struct = rknpu_submit(
        flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
        timeout=6000,
        task_start=0,
        task_number=task_count,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        iommu_domain_id=0,
        reserved=0,
        task_base_addr=0,
        hw_elapse_time=0,
        core_mask=1,
        fence_fd=-1,
    )
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

def reset_npu(fd):
    action = rknpu_action(flags=RKNPU_ACT_RESET, value=0)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, action)
    print(f"reset_npu ret={ret}")
    return ret

task_map, task_mc = mem_allocate(fd, TASK_BUF_SIZE, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mc = mem_allocate(fd, REGCMD_BUF_SIZE, RKNPU_MEM_NON_CACHEABLE)
input_map, input_mc = mem_allocate(fd, INPUT_BUF_SIZE, RKNPU_MEM_NON_CACHEABLE)
output_map, output_mc = mem_allocate(fd, OUTPUT_BUF_SIZE, RKNPU_MEM_NON_CACHEABLE)

POOL_OPS = ("min", "max", "avg", "globalmin", "globalmax", "globalavg")
POOL_ENABLE_MASK = 0x60
POOL_INT_MASK = 0xc00
POOL_TASK_OP_IDX = 1
POOL_PC_ENABLE = 48
POOL_CHANNELS = 8
def emit(target, addr, value):
    if target == reg.PC and addr == reg.REG_PC_OPERATION_ENABLE: target = 0x80
    return (((target + 1) & 0xffff) << 48) | ((int(value) & 0xffffffff) << 16) | (addr & 0xffff)

def pc_amount(reg_count):
    return (int(reg_count) + 1) // 2 + 1


def pack_f16_strided(buf, vals, stride_fp16=8):
    arr = np.asarray(vals, dtype=np.float16).reshape(-1).view(np.uint16)
    dst = (ctypes.c_uint16 * (len(arr) * stride_fp16)).from_buffer(buf)
    for i, value in enumerate(arr):
        dst[i * stride_fp16] = int(value)

def read_f16_strided(buf, n, stride_fp16=8):
    raw = np.frombuffer(buf, dtype=np.float16, count=n * stride_fp16)
    return raw[::stride_fp16].copy()

def align_up(value, align):
    return ((int(value) + align - 1) // align) * align

def pool2d_reference(x, op):
    if op.startswith("global"):
        window = x
        if op == "globalmax": return np.max(window, axis=(0, 1), keepdims=True).astype(np.float16)
        if op == "globalmin": return np.min(window, axis=(0, 1), keepdims=True).astype(np.float16)
        return np.mean(window.astype(np.float32), axis=(0, 1), keepdims=True).astype(np.float16)

    out_h, out_w = x.shape[0] - 1, x.shape[1] - 1
    out = np.empty((out_h, out_w) + x.shape[2:], dtype=np.float16)
    for y in range(out_h):
        for x0 in range(out_w):
            window = x[y:y + 2, x0:x0 + 2]
            if op == "max":
                out[y, x0] = np.max(window, axis=(0, 1))
            elif op == "min":
                out[y, x0] = np.min(window, axis=(0, 1))
            else:
                out[y, x0] = np.mean(window.astype(np.float32), axis=(0, 1)).astype(np.float16)
    return out

def pool_input(op, in_h, in_w):
    seed = 0x504f4f4c ^ (POOL_OPS.index(op) << 16) ^ (in_h << 4) ^ in_w
    rng = np.random.default_rng(seed)
    return rng.uniform(-8.0, 8.0, (in_h, in_w, POOL_CHANNELS)).astype(np.float16)


def pooling_regs(op, input_dma=0x11110000, output_dma=0x22220000, in_h=4, in_w=4):
    if op not in POOL_OPS:
        raise ValueError(f"unknown pool op {op!r}; expected one of {', '.join(POOL_OPS)}")
    if in_h < 2 or in_w < 2:
        raise ValueError(f"pool input must be at least 2x2, got {in_h}x{in_w}")

    hw_op = "max" if op in ("min", "globalmin") else op
    direct_global = op in ("globalmax", "globalavg")
    max_pool = hw_op in ("max", "globalmax")
    in_h_field = in_h - 1
    in_w_field = in_w - 1
    out_h_field = 0 if direct_global else in_h - 2
    out_w_field = 0 if direct_global else in_w - 2
    channel_field = POOL_CHANNELS - 1
    width_stride = in_w * POOL_CHANNELS * FP16_BYTES
    src_surf_stride = width_stride * in_h
    dst_surf_stride = 1 if direct_global else in_w * (in_h - 1)
    index_add = 1 if direct_global else dst_surf_stride
    k_h = in_h_field if direct_global else 1
    k_w = in_w_field if direct_global else 1
    s_h = in_h_field if direct_global else 0
    s_w = in_w_field if direct_global else 0

    npu_regs = [
        emit(reg.PPU, reg.REG_PPU_S_POINTER,
             (1 << 3) |                           # PPU_S_POINTER_POINTER_PP_MODE
             (1 << 2) |                           # PPU_S_POINTER_EXECUTER_PP_EN
             (1 << 1)),                           # PPU_S_POINTER_POINTER_PP_EN
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_S_POINTER,
             (1 << 3) |                           # PPU_RDMA_S_POINTER_POINTER_PP_MODE
             (1 << 2) |                           # PPU_RDMA_S_POINTER_EXECUTER_PP_EN
             (1 << 1)),                           # PPU_RDMA_S_POINTER_POINTER_PP_EN
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_IN_WIDTH, in_w_field),  # input width minus 1
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_IN_HEIGHT, in_h_field),  # input height minus 1
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_IN_CHANNEL, channel_field),  # input channels minus 1
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_OUT_WIDTH, out_w_field),  # output width minus 1
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_OUT_HEIGHT, out_h_field),  # output height minus 1
        emit(reg.PPU, reg.REG_PPU_DATA_CUBE_OUT_CHANNEL, channel_field),  # output channels minus 1
        emit(reg.PPU, reg.REG_PPU_OPERATION_MODE_CFG,
             (1 << 4) |                           # flying mode
             int(max_pool)),                      # pooling method
        emit(reg.PPU, reg.REG_PPU_POOLING_KERNEL_CFG,
             (s_h << 20) |                        # kernel stride height
             (s_w << 16) |                        # kernel stride width
             (k_h << 8) |                         # kernel height
             k_w),                                # kernel width
        emit(reg.PPU, reg.REG_PPU_DST_BASE_ADDR,
             (output_dma // 16) << 4),            # output base address
        emit(reg.PPU, reg.REG_PPU_DST_SURF_STRIDE, dst_surf_stride),  # output surface stride
        emit(reg.PPU, reg.REG_PPU_DATA_FORMAT,
             (index_add << 16) |                  # index_add
             2),                                  # fp16
        emit(reg.PPU, reg.REG_PPU_MISC_CTRL,
             3),                                  # burst length
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH,
             in_w_field),                         # RDMA input width minus 1
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT,
             in_h_field),                         # RDMA input height minus 1
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL,
             channel_field),                      # RDMA input channels minus 1
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_SRC_BASE_ADDR,
             input_dma),                          # RDMA source base address
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE,
             width_stride),                       # RDMA line stride
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE,
             src_surf_stride),                    # RDMA surface stride
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_DATA_FORMAT,
             2),                                  # RDMA fp16
        emit(reg.PPU_RDMA, reg.REG_PPU_RDMA_RDMA_OPERATION_ENABLE,
             1),                                  # RDMA enable
        emit(reg.PPU, reg.REG_PPU_RECIP_KERNEL_WIDTH,
             0 if max_pool else 30720),           # avg only
        emit(reg.PPU, reg.REG_PPU_RECIP_KERNEL_HEIGHT,
             0 if max_pool else 30720),           # avg only
    ]
    return npu_regs


def write_regs_to_npu_task(task_regs):
    def enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len):
        enable_npu_units = emit(reg.PC, reg.REG_PC_OPERATION_ENABLE,
                                POOL_PC_ENABLE << 1)             # PC_OPERATION_ENABLE_RESERVED_0

        if next_offset is None:
            return [
                emit(reg.PC_REG, reg.REG_PC_BASE_ADDRESS, 0),    # PC_BASE_ADDRESS = 0 (end chain)
                emit(reg.PC_REG, reg.REG_PC_REGISTER_AMOUNTS, 0), # PC_REGISTER_AMOUNTS = 0 (end chain)
                emit(reg.VERSION, 0, 0),                         # VERSION (OP_40 equivalent)
                enable_npu_units,
            ]
        next_addr = regcmd_mc.dma_addr + next_offset * ctypes.sizeof(ctypes.c_uint64)
        return [
            emit(reg.PC_REG, reg.REG_PC_BASE_ADDRESS, next_addr & 0xfffffff0),
            emit(reg.PC_REG, reg.REG_PC_REGISTER_AMOUNTS, pc_amount(next_task_regs_len)),
            emit(reg.VERSION, 0, 0),
            enable_npu_units,
        ]

    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())
    npu_tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    npu_regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += align_up(len(regs) + PC_CHAIN_TAIL_QWORDS, 2)
    if offset > regcmd_mc.size // ctypes.sizeof(ctypes.c_uint64):
        raise ValueError("regcmd buffer too small")
    if len(task_regs) > task_mc.size // ctypes.sizeof(struct_rknpu_task):
        raise ValueError("task buffer too small")

    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, value in enumerate(regs):
            npu_regcmd[base + i] = value
        next_offset = offsets[idx + 1] if idx + 1 < len(task_regs) else None
        next_task_regs_len = len(task_regs[idx + 1]) if idx + 1 < len(task_regs) else 0
        tails = enable_npu_units_and_set_next_pc_addr(next_offset, next_task_regs_len)
        for i, value in enumerate(tails):
            npu_regcmd[base + len(regs) + i] = value

        npu_tasks[idx].flags = 0
        npu_tasks[idx].op_idx = POOL_TASK_OP_IDX
        npu_tasks[idx].enable_mask = POOL_ENABLE_MASK
        npu_tasks[idx].int_mask = POOL_INT_MASK
        npu_tasks[idx].int_clear = 0x1ffff
        npu_tasks[idx].int_status = 0
        npu_tasks[idx].regcfg_amount = len(regs)
        npu_tasks[idx].regcfg_offset = 0
        npu_tasks[idx].regcmd_addr = regcmd_mc.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)


def output_shape(op, in_h, in_w):
    if op in ("globalmax", "globalavg"):
        return 1, 1
    return in_h - 1, in_w - 1


def output_elements(op, in_h, in_w):
    out_h, out_w = output_shape(op, in_h, in_w)
    if op == "globalmin":
        out_h, out_w = in_h - 1, in_w - 1
    return out_h * out_w * POOL_CHANNELS


def read_pool_output(output_map, op, expected_shape, in_h, in_w):
    if op == "globalmin":
        count = (in_h - 1) * (in_w - 1) * POOL_CHANNELS
        pooled = np.frombuffer(output_map, dtype=np.float16, count=count).copy().reshape(in_h - 1, in_w - 1, POOL_CHANNELS)
        return -np.max(pooled.astype(np.float32), axis=(0, 1), keepdims=True).astype(np.float16)
    if op in ("globalmax", "globalavg"):
        return np.frombuffer(output_map, dtype=np.float16, count=POOL_CHANNELS).copy().reshape(expected_shape)
    got = np.frombuffer(output_map, dtype=np.float16, count=np.prod(expected_shape)).copy().reshape(expected_shape)
    return -got if op == "min" else got


def pool_task_tiles(op, in_h, in_w):
    if op.startswith("global"):
        return [(0, in_h)]
    max_output_rows = max(1, min(8191 // in_w, 8191))
    tiles = []
    for out_start in range(0, in_h - 1, max_output_rows):
        out_rows = min(max_output_rows, in_h - 1 - out_start)
        tiles.append((out_start, out_rows + 1))
    return tiles


def run_pool(op, in_h=4, in_w=4, reset=True):
    if reset:
        reset_npu(fd)
    tiles = pool_task_tiles(op, in_h, in_w)
    regcmd_qwords = sum(align_up(len(pooling_regs(op, in_h=tile_h, in_w=in_w)), 2) + PC_CHAIN_TAIL_QWORDS
                        for _, tile_h in tiles)
    input_bytes = align_up(in_h * in_w * POOL_CHANNELS * FP16_BYTES, 4096)
    if op in ("globalmax", "globalavg"):
        output_elems = POOL_CHANNELS
    else:
        output_elems = (in_h - 1) * (in_w - 1) * POOL_CHANNELS
    output_bytes = align_up(max(1, output_elems * FP16_BYTES), 4096)
    if len(tiles) * ctypes.sizeof(struct_rknpu_task) > task_mc.size:
        raise ValueError("task buffer too small")
    if regcmd_qwords * ctypes.sizeof(ctypes.c_uint64) > regcmd_mc.size:
        raise ValueError("regcmd buffer too small")
    if input_bytes > input_mc.size:
        raise ValueError("input buffer too small")
    if output_bytes > output_mc.size:
        raise ValueError("output buffer too small")

    x = pool_input(op, in_h, in_w)
    expected = pool2d_reference(x, op)
    input_for_op = -x if op in ("min", "globalmin") else x
    input_map[:input_for_op.nbytes] = input_for_op.reshape(-1).tobytes()
    output_map[:output_bytes] = b"\x00" * output_bytes

    task_regs = []
    for out_start, tile_h in tiles:
        input_dma = input_mc.dma_addr + out_start * in_w * POOL_CHANNELS * FP16_BYTES
        output_dma = output_mc.dma_addr if op.startswith("global") else output_mc.dma_addr + out_start * (in_w - 1) * POOL_CHANNELS * FP16_BYTES
        task_regs.append(pooling_regs(op, input_dma, output_dma, tile_h, in_w))
    write_regs_to_npu_task(task_regs)

    ret = submit(fd, task_mc.obj_addr, len(task_regs))
    got = read_pool_output(output_map, op, expected.shape, in_h, in_w)
    decoded = (got.astype(np.float32) / x.shape[0]).astype(np.float16) if op == "globalavg" else got
    atol = 0.25 if op in ("avg", "globalavg") else 0.0
    ok = ret == 0 and np.allclose(decoded, expected, atol=atol)
    print(f"op={op} input_shape={x.shape} output_shape={expected.shape} tasks={len(task_regs)} reg_count={sum(len(regs) for regs in task_regs)}")
    print(f"NPU output={got.reshape(-1)[:min(32, got.size)]}")
    if op == "globalavg":
        print(f"NPU decoded={decoded.reshape(-1)[:min(32, decoded.size)]}")
    print(f"expected={expected.reshape(-1)[:min(32, expected.size)]}")
    print(f"max_abs_diff={float(np.max(np.abs(decoded.astype(np.float32) - expected.astype(np.float32)))):.6f}")
    print(f"{op.upper()}POOL PASS" if ok else f"{op.upper()}POOL FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="RK3588 PPU pool register streams from experimental/rknnops.h")
    parser.add_argument("--op", choices=POOL_OPS + ("all",), default="all")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--dry", action="store_true", help="print register streams without submitting")
    args = parser.parse_args()

    if args.dry:
        ops = POOL_OPS if args.op == "all" else (args.op,)
        height = args.height or 4
        width = args.width or 4
        print("Pool dry run only: no /dev/dri open, no ioctl, no submit")
        for op in ops:
            tiles = pool_task_tiles(op, height, width)
            regs = [pooling_regs(op, 0x11110000 + i * 0x10000, 0x22220000 + i * 0x10000, tile_h, width)
                    for i, (_, tile_h) in enumerate(tiles)]
            print(f"op={op} tasks={len(regs)} reg_count={sum(len(tile_regs) for tile_regs in regs)}")
            print("first_regs:")
            for value in regs[0][:6]:
                print(f"  0x{value:016x}")
            print("last_regs:")
            for value in regs[-1][-6:]:
                print(f"  0x{value:016x}")
        print("POOL DRY RUN PASS")
        return 0

    ops = POOL_OPS if args.op == "all" else (args.op,)
    tests = []
    if args.height or args.width:
        tests = [(op, args.height or 4, args.width or 4) for op in ops]
    else:
        tests += [(op, 4, 4) for op in ops]
        tests += [(op, 9000, 4) for op in ops if op in ("min", "max", "avg")]
    for test_idx, (op, height, width) in enumerate(tests):
        print(f"\n{op} {height}x{width}:")
        rc = run_pool(op, height, width, reset=(test_idx == 0))
        assert rc == 0, f"{op} pool shape {(height, width)} failed"
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        os.close(fd)
