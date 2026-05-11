from fcntl import ioctl
import ast
import glob
import os, mmap, sys, time, struct
import ctypes
import numpy as np

FEATURE_ATOMIC_SIZE = 16
WEIGHT_ATOMIC_SIZE = 32
CBUF_ENTRY_SIZE = 128
CBUF_ENTRIES_PER_BANK = 256
CBUF_BANKS = 12
BPE = 1

INPUT_ZERO_POINT = 0
WEIGHT_ZERO_POINT = 0
OUTPUT_ZERO_POINT = 0
INPUT_SCALE = 1.0 / 255.0
WEIGHT_SCALE = 1.0 / 255.0
OUTPUT_SCALE = 1.0 / 255.0

input_width = 0
input_height = 0
input_channels = 0
output_channels = 0
weights_height = 0
weights_width = 0
output_width = 0
output_height = 0
ADDITION_INPUT = False
ADD_TENSOR = -1
DEPTHWISE = False
GROUPS = 1
weight_input_channels = 0

def load_conv_legacy_shapes():
    def literal_dict_call(node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "dict":
            return None
        if node.args:
            return None
        out = {}
        for kw in node.keywords:
            if kw.arg is None:
                return None
            out[kw.arg] = ast.literal_eval(kw.value)
        return out

    conv_path = os.path.join(os.path.dirname(__file__), "conv_legacy.py")
    with open(conv_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=conv_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "shapes" in names and isinstance(node.value, ast.List):
                shapes = []
                for elt in node.value.elts:
                    shape = literal_dict_call(elt)
                    if shape is not None:
                        shapes.append(shape)
                return shapes
    raise RuntimeError("Could not find literal shapes list in conv_legacy.py")

def use_shape(shape):
    global input_width, input_height, input_channels, output_channels, weights_height, weights_width, output_width, output_height, ADDITION_INPUT, ADD_TENSOR, DEPTHWISE, GROUPS, weight_input_channels
    DEPTHWISE = shape["groups"] == shape["in_c"] == shape["out_c"] and shape["weight_in_c"] == 1
    if shape["in_c"] % shape["groups"] != 0 or shape["out_c"] % shape["groups"] != 0:
        raise ValueError(f"unsupported simple_conv shape: {shape['name']}")
    if not DEPTHWISE and shape["weight_in_c"] != shape["in_c"] // shape["groups"]:
        raise ValueError(f"unsupported simple_conv shape: {shape['name']}")
    GROUPS = shape["groups"]
    weight_input_channels = shape["weight_in_c"]
    input_width = shape["in_w"]
    input_height = shape["in_h"]
    input_channels = shape["in_c"]
    output_channels = shape["out_c"]
    weights_height = shape["kh"]
    weights_width = shape["kw"]
    output_width = input_width - weights_width + 1
    output_height = input_height - weights_height + 1
    ADDITION_INPUT = shape.get("addition_input", False)
    ADD_TENSOR = shape.get("add_tensor", -1)

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

    # --- CNA (0x1000) ---
    CNA_CONV_CON1       = 0x100c   # CNA convolution control 1
    CNA_CONV_CON2       = 0x1010   # CNA convolution control 2 (grains)
    CNA_CONV_CON3       = 0x1014   # CNA convolution control 3 (stride)
    CNA_DATA_SIZE0      = 0x1020   # CNA input data size 0
    CNA_DATA_SIZE1      = 0x1024   # CNA input data size 1 (channel)
    CNA_DATA_SIZE2      = 0x1028   # CNA output data size 2
    CNA_DATA_SIZE3      = 0x102c   # CNA output data size 3 (atomics)
    CNA_WEIGHT_SIZE0    = 0x1030   # CNA weight total size
    CNA_WEIGHT_SIZE1    = 0x1034   # CNA weight per-kernel size
    CNA_WEIGHT_SIZE2    = 0x1038   # CNA weight dims (width/height/kernels)
    CNA_CBUF_CON0       = 0x1040   # CNA CBUF config 0 (banks)
    CNA_CBUF_CON1       = 0x1044   # CNA CBUF config 1 (entries)
    CNA_CVT_CON0        = 0x104c   # CNA convert config 0
    CNA_CVT_CON1        = 0x1050   # CNA convert config 1 (scale/offset 0)
    CNA_CVT_CON2        = 0x1054   # CNA convert config 2 (scale/offset 1)
    CNA_CVT_CON3        = 0x1058   # CNA convert config 3 (scale/offset 2)
    CNA_CVT_CON4        = 0x105c   # CNA convert config 4 (scale/offset 3)
    CNA_PAD_CON0        = 0x1068   # CNA padding config 0 (left/top)
    CNA_FEATURE_DATA_ADDR = 0x1070 # CNA feature data base address
    CNA_DMA_CON0        = 0x1078   # CNA DMA control 0 (burst)
    CNA_DMA_CON1        = 0x107c   # CNA DMA control 1 (line stride)
    CNA_DMA_CON2        = 0x1080   # CNA DMA control 2 (surface stride)
    CNA_FC_DATA_SIZE0   = 0x1084   # CNA FC data size 0
    CNA_FC_DATA_SIZE1   = 0x1088   # CNA FC data size 1 (channel)
    CNA_DCOMP_ADDR0     = 0x1110   # CNA weight decompress address 0
    CNA_CVT_CON5        = 0x1180   # CNA convert config 5 (mask)
    CNA_PAD_CON1        = 0x1184   # CNA padding config 1 (pad value)

    # --- CORE (0x3000) ---
    CORE_MISC_CFG       = 0x3010   # CORE misc config
    CORE_DATAOUT_SIZE_0 = 0x3014   # CORE dataout size 0 (height/width)
    CORE_DATAOUT_SIZE_1 = 0x3018   # CORE dataout size 1 (channel)
    CORE_CLIP_TRUNCATE  = 0x301c   # CORE clip truncate config
    CORE_RESERVED_3030  = 0x3030   # CORE reserved (write 0 to clear stale state)

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
    BS_CFG              = 0x4040   # DPU batch/scale config
    BS_ALU_CFG          = 0x4044   # DPU batch/scale ALU config
    BS_MUL_CFG          = 0x4048   # DPU batch/scale multiply config
    BS_RELUX_CMP_VALUE  = 0x404c   # DPU batch/scale ReLU-X compare value
    BS_OW_CFG           = 0x4050   # DPU batch OW config
    BS_OW_OP            = 0x4054   # DPU batch OW operation
    WDMA_SIZE_0         = 0x4058   # DPU write DMA size 0
    WDMA_SIZE_1         = 0x405c   # DPU write DMA size 1
    BN_CFG              = 0x4060   # DPU batch norm config
    EW_CFG              = 0x4070   # DPU elementwise config
    EW_CVT_SCALE_VALUE  = 0x4078   # DPU EW convert scale value
    OUT_CVT_OFFSET      = 0x4080   # DPU output conversion offset
    OUT_CVT_SCALE       = 0x4084   # DPU output conversion scale
    OUT_CVT_SHIFT       = 0x4088   # DPU output conversion shift
    SURFACE_ADD         = 0x40c0   # DPU surface add

    # --- DPU RDMA (0x5000) ---
    RDMA_S_POINTER      = 0x5004   # RDMA S pointer config
    RDMA_DATA_CUBE_WIDTH  = 0x500c # RDMA data cube width
    RDMA_DATA_CUBE_HEIGHT = 0x5010 # RDMA data cube height
    RDMA_DATA_CUBE_CHANNEL= 0x5014 # RDMA data cube channel
    RDMA_BRDMA_CFG        = 0x501c # RDMA BRDMA config
    RDMA_BS_BASE_ADDR     = 0x5020 # RDMA batch/scale base address (bias)
    RDMA_ERDMA_CFG        = 0x5034 # RDMA ERDMA config
    RDMA_FEATURE_MODE_CFG = 0x5044 # RDMA feature mode config
    RDMA_WEIGHT           = 0x5068 # RDMA weight config

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
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=4096)
input_map, input_mem_create = mem_allocate(fd, size=4194304)
weight_map, weight_mem_create = mem_allocate(fd, size=4194304)
bias_map, bias_mem_create = mem_allocate(fd, size=4194304)
output_map, output_mem_create = mem_allocate(fd, size=4194304)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align(x, a):
    return _ceil_div(x, a) * a

def _field(mask, shift, val):
    return (int(val) << shift) & mask

def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

def fui(f):
    return struct.unpack('I', struct.pack('f', f))[0]

def calc_line_stride(width):
    return width * FEATURE_ATOMIC_SIZE * BPE

def calc_input_size(input_width, input_height, input_channels):
    input_channels_1 = _ceil_div(input_channels, FEATURE_ATOMIC_SIZE) * 2
    return input_width * input_height * input_channels_1 * FEATURE_ATOMIC_SIZE

def calc_weight_size(weights_width, weights_height, input_channels, output_channels):
    wc = _align(max(input_channels, FEATURE_ATOMIC_SIZE), WEIGHT_ATOMIC_SIZE)
    oc = 1 if DEPTHWISE else _align(output_channels, 2)
    return weights_width * weights_height * oc * wc * 2

def calc_raw_output_size(output_width, output_height, output_channels):
    oc1 = _ceil_div(output_channels, FEATURE_ATOMIC_SIZE) * 2
    return output_width * output_height * oc1 * FEATURE_ATOMIC_SIZE

def calc_entries_per_slice(input_width, input_channels):
    atomics_per_entry = CBUF_ENTRY_SIZE // FEATURE_ATOMIC_SIZE
    total_c_atomics = _ceil_div(input_channels * BPE, FEATURE_ATOMIC_SIZE)
    last_c_atomics = total_c_atomics % atomics_per_entry
    int_c_entries = (total_c_atomics // atomics_per_entry) * input_width
    if last_c_atomics == 3:
        frac_c_entries = input_width
    else:
        frac_c_entries = _ceil_div(last_c_atomics * input_width, atomics_per_entry)
    return int_c_entries + frac_c_entries

def calc_input_banks(input_width, input_height, input_channels):
    return _ceil_div(calc_entries_per_slice(input_width, input_channels) * input_height, CBUF_ENTRIES_PER_BANK)

def calc_weights_banks(weights_width, weights_height, input_channels, output_channels):
    bytes_ = weights_width * weights_height * input_channels * BPE
    if not DEPTHWISE:
        bytes_ *= output_channels
    return _ceil_div(_ceil_div(bytes_, CBUF_ENTRY_SIZE), CBUF_ENTRIES_PER_BANK) + 1

def split_tasks():
    entries_per_slice = calc_entries_per_slice(input_width, input_channels)
    input_banks_required = calc_input_banks(input_width, input_height, input_channels)
    weights_banks_required = calc_weights_banks(weights_width, weights_height, input_channels, output_channels)
    available_input_banks = CBUF_BANKS - weights_banks_required
    available_weights_banks = weights_banks_required

    if input_banks_required <= available_input_banks:
        return [dict(input_height=input_height, output_height=output_height,
                     input_offset=0, output_offset=0, input_banks=input_banks_required,
                     weight_banks=CBUF_BANKS - input_banks_required)]

    if weights_banks_required + 1 >= CBUF_BANKS:
        available_input_banks = 7
        available_weights_banks = CBUF_BANKS - available_input_banks

    available_slices = (CBUF_ENTRIES_PER_BANK * available_input_banks) // entries_per_slice
    slices = [dict(top_slice=0, bottom_slice=available_slices - 1)]
    s = weights_height - 1
    while s < input_height:
        prev = slices[-1]
        while s <= prev["bottom_slice"]:
            s += 1
        if s > prev["bottom_slice"]:
            s -= 1
        top_slice = min(s, prev["bottom_slice"]) - (weights_height - 1) + 1
        bottom_slice = top_slice + available_slices - 1
        if bottom_slice >= input_height - 1:
            slices.append(dict(top_slice=top_slice, bottom_slice=input_height - 1))
            break
        s = top_slice + weights_height - 1
        slices.append(dict(top_slice=top_slice, bottom_slice=bottom_slice))

    output_height_processed = 0
    tasks_out = []
    for idx, sl in enumerate(slices):
        task_input_height = min(sl["bottom_slice"], input_height - 1) - sl["top_slice"] + 1
        if task_input_height < weights_height:
            continue
        task_output_height = task_input_height - weights_height + 1
        tasks_out.append(dict(input_height=task_input_height, output_height=task_output_height,
                              input_offset=calc_line_stride(input_width) * sl["top_slice"],
                              output_offset=calc_line_stride(output_width) * output_height_processed,
                              input_banks=available_input_banks, weight_banks=available_weights_banks,
                              weight_reuse=idx > 0 and weights_banks_required + 1 < CBUF_BANKS))
        output_height_processed += task_output_height
    return tasks_out

def pack_input(input_nhwc):
    raw = np.zeros(calc_input_size(input_width, input_height, input_channels), dtype=np.uint8)
    if input_channels == 1:
        n = 0
        for x in range(input_width):
            for y in range(max(input_height, FEATURE_ATOMIC_SIZE)):
                raw[n] = input_nhwc[0, y, x, 0] if y < input_height else INPUT_ZERO_POINT
                n += 1
        return raw

    n = 0
    for u in range(_ceil_div(input_channels, FEATURE_ATOMIC_SIZE)):
        for x in range(input_width):
            for y in range(input_height):
                for c in range(FEATURE_ATOMIC_SIZE):
                    ic = c + u * FEATURE_ATOMIC_SIZE
                    if ic < input_channels:
                        raw[n] = (int(input_nhwc[0, y, x, ic]) - 0x80) & 0xFF
                    else:
                        raw[n] = (INPUT_ZERO_POINT - 0x80) & 0xFF
                    n += 1
    return raw

def pack_weights(weight_ohwi):
    ic = max(input_channels, FEATURE_ATOMIC_SIZE)
    oc = 1 if DEPTHWISE else _align(output_channels, 2)
    ic_groups = WEIGHT_ATOMIC_SIZE * (2 if DEPTHWISE else 1)
    out = np.zeros(calc_weight_size(weights_width, weights_height, input_channels, output_channels), dtype=np.uint8)
    n = 0
    for oo in range(_ceil_div(oc, WEIGHT_ATOMIC_SIZE)):
        for ii in range(_ceil_div(ic, ic_groups)):
            for x in range(weights_width):
                for y in range(weights_height):
                    for oi in range(min(oc - oo * WEIGHT_ATOMIC_SIZE, WEIGHT_ATOMIC_SIZE)):
                        for ij in range(min(ic, ic_groups)):
                            oc_idx = oo * WEIGHT_ATOMIC_SIZE + oi
                            ic_idx = ii * ic_groups + ij
                            if oc_idx >= output_channels:
                                out[n] = 0
                            elif ic_idx >= input_channels:
                                out[n] = (WEIGHT_ZERO_POINT - 0x80) & 0xFF
                            else:
                                if DEPTHWISE:
                                    out[n] = (int(weight_ohwi[0, x, y, ic_idx]) - 0x80) & 0xFF
                                else:
                                    out[n] = (int(weight_ohwi[oc_idx, x, y, ic_idx]) - 0x80) & 0xFF
                            n += 1
    return out

def pack_biases(biases, weight_ohwi):
    packed = np.zeros(output_channels, dtype=np.int32)
    for oc in range(output_channels):
        correction = 0
        for wx in range(weights_width):
            for wy in range(weights_height):
                if DEPTHWISE:
                    correction += (int(weight_ohwi[0, wx, wy, oc]) - WEIGHT_ZERO_POINT) * (INPUT_ZERO_POINT - 0x80)
                else:
                    for ic in range(input_channels):
                        correction += (int(weight_ohwi[oc, wx, wy, ic]) - WEIGHT_ZERO_POINT) * (INPUT_ZERO_POINT - 0x80)
        packed[oc] = int(biases[oc]) - correction
    return packed

def build_conv_regs(input_dma, weight_dma, bias_dma, output_dma, task=None):
    input_width_real = globals()["input_width"]
    input_channels_real = globals()["input_channels"]
    output_channels_real = globals()["output_channels"]
    depthwise = DEPTHWISE
    
    input_channels = _align(max(input_channels_real, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE)
    output_channels = _align(max(output_channels_real, 32), 32)
    if depthwise and output_channels_real <= 32:
        output_channels *= 2
    if depthwise:
        output_channels = _align(output_channels, 64)
    task_input_height = task["input_height"] if task is not None else input_height
    task_output_height = task["output_height"] if task is not None else output_height
    task_input_offset = task["input_offset"] if task is not None else 0
    task_output_offset = task["output_offset"] if task is not None else 0
    output_surface_stride = calc_line_stride(output_width) * output_height // FEATURE_ATOMIC_SIZE
    
    input_width = input_width_real
    input_line_stride = calc_line_stride(input_width) // 4
    input_surface_stride = int(input_line_stride * (input_height / 4 - 1))
    if input_channels_real == 1 and (output_channels_real > 1 or ADDITION_INPUT or ADD_TENSOR != -1):
        input_width = max(input_width_real, FEATURE_ATOMIC_SIZE)
        input_line_stride = max(calc_line_stride(input_width_real) // FEATURE_ATOMIC_SIZE, FEATURE_ATOMIC_SIZE)
        input_surface_stride = input_line_stride * (task_input_height - 1)
        if input_channels == 32 and op.input_width == 80:
            input_line_stride *= 4
            input_surface_stride = int(input_line_stride * (task_input_height / 4 - 1))
        else:
            input_surface_stride = int(input_line_stride * (task_input_height - 1))
    if input_width == 8 and (ADDITION_INPUT or ADD_TENSOR != -1):
        input_line_stride //= 2
        input_surface_stride = 112
    
    weights_kernels = 1 if depthwise else _align(output_channels_real, 2)
    if task is None:
        input_banks_required = calc_input_banks(input_width_real, input_height, input_channels_real)
        weights_banks_required = calc_weights_banks(weights_width, weights_height, input_channels_real, output_channels_real)
        input_banks = input_banks_required
        weight_banks = CBUF_BANKS - input_banks
        if input_banks_required + weights_banks_required > CBUF_BANKS:
            input_banks = 7
            weight_banks = CBUF_BANKS - input_banks
    else:
        input_banks = task["input_banks"]
        weight_banks = task["weight_banks"]
    atomic_count = output_width * task_output_height
    surfaces_per_row = output_width * output_height * 2
    if depthwise:
        surfaces_per_row *= 2
    pad_con1 = 0xffff8080 if (weights_width >= 3 and INPUT_ZERO_POINT == 0) else INPUT_ZERO_POINT - 0x80
    if ADDITION_INPUT or ADD_TENSOR != -1:
        pad_con1 = 0xffffff80

    scale = 1
    coff = 0
    if input_channels_real == 1:
        scale = 32388 if ADDITION_INPUT or ADD_TENSOR != -1 else 16384
        coff = 65408
        trunc = 15 if ADDITION_INPUT or ADD_TENSOR != -1 else 14

    if input_channels_real == 1:
        input_data_entries = input_width * task_input_height
    elif input_width == 40 and input_channels_real == 40:
        input_data_entries = 40
    else:
        input_data_entries = _ceil_div(
            input_width * 2 * _ceil_div(input_channels_real, FEATURE_ATOMIC_SIZE), 8)

    conv_scale = (INPUT_SCALE * WEIGHT_SCALE) / OUTPUT_SCALE
    scale_bits = fui(conv_scale)
    shift = 127 + 31 - 32 - (scale_bits >> 23) + 16
    out_scale = ((scale_bits >> 9) & 0x7fff) + 1
    if out_scale < 1 << 14:
        out_scale |= 1 << 14

    return [
        E(reg.TARGET_DPU, reg.S_POINTER,
          (1 << 3) | (1 << 2) | (1 << 1)),                              # DPU PP mode/enable
        E(reg.TARGET_RDMA, reg.RDMA_S_POINTER,
          (1 << 3) | (1 << 2) | (1 << 1)),                              # RDMA PP mode/enable
        E(reg.TARGET_CNA, reg.CNA_CONV_CON1,
          (((1 << 30) | (1 << 29) | (8 << 12)) if input_channels_real == 1 else 0) |
          (3 if depthwise else 0)),                                     # CNA_CONV_CON1
        E(reg.TARGET_CNA, reg.CNA_CONV_CON2, (50 + 1 + 1) << 4),        # FEATURE_GRAINS
        E(reg.TARGET_CNA, reg.CNA_CONV_CON3, (1 << 3) | 1),             # stride Y/X
        E(reg.TARGET_CNA, reg.CNA_DATA_SIZE0, (input_width << 16) | task_input_height),
        E(reg.TARGET_CNA, reg.CNA_DATA_SIZE1, ((input_channels_real - 1) << 16) | input_channels),
        E(reg.TARGET_CNA, reg.CNA_DATA_SIZE2, output_width),
        E(reg.TARGET_CNA, reg.CNA_DATA_SIZE3, atomic_count),
        E(reg.TARGET_CNA, reg.CNA_WEIGHT_SIZE0, weights_width * weights_height * input_channels * weights_kernels),
        E(reg.TARGET_CNA, reg.CNA_WEIGHT_SIZE1, weights_width * weights_height * input_channels),
        E(reg.TARGET_CNA, reg.CNA_WEIGHT_SIZE2, (weights_width << 24) | (weights_height << 16) | weights_kernels),
        E(reg.TARGET_CNA, reg.CNA_CBUF_CON0, ((1 << 13) if task is not None and task.get("weight_reuse") else 0) | (weight_banks << 4) | input_banks),
        E(reg.TARGET_CNA, reg.CNA_CBUF_CON1, input_data_entries),
        E(reg.TARGET_CNA, reg.CNA_CVT_CON0, 
            (1 << 3) | (1 << 1) | 1 
            if not input_channels_real == 1 else 
            ((trunc << 22) | (trunc << 16) | (trunc << 10) | (trunc << 4))),
        E(reg.TARGET_CNA, reg.CNA_CVT_CON1, (scale << 16) | coff),
        E(reg.TARGET_CNA, reg.CNA_CVT_CON2, (scale << 16) | coff),
        E(reg.TARGET_CNA, reg.CNA_CVT_CON3, (scale << 16) | coff),
        E(reg.TARGET_CNA, reg.CNA_CVT_CON4, (scale << 16) | coff),
        E(reg.TARGET_CNA, reg.CNA_PAD_CON0, 0),
        E(reg.TARGET_CNA, reg.CNA_FEATURE_DATA_ADDR, (input_dma + task_input_offset) & 0xFFFFFFFF),
        E(reg.TARGET_CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        E(reg.TARGET_CNA, reg.CNA_DMA_CON1, input_line_stride),
        E(reg.TARGET_CNA, reg.CNA_DMA_CON2, input_surface_stride),
        E(reg.TARGET_CNA, reg.CNA_FC_DATA_SIZE0, (input_width_real << 16) | task_input_height),
        E(reg.TARGET_CNA, reg.CNA_FC_DATA_SIZE1, input_channels),
        E(reg.TARGET_CNA, reg.CNA_DCOMP_ADDR0, weight_dma & 0xFFFFFFFF),
        E(reg.TARGET_CNA, reg.CNA_CVT_CON5, 0xffff if input_channels_real == 1 else 0),
        E(reg.TARGET_CNA, reg.CNA_PAD_CON1, pad_con1),
        E(reg.TARGET_CORE, reg.CORE_MISC_CFG, 1 | (2 if depthwise else 0)), # QD_EN, DW_EN
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_0, ((task_output_height - 1) << 16) | (output_width - 1)),
        E(reg.TARGET_CORE, reg.CORE_DATAOUT_SIZE_1, output_channels - 1),
        E(reg.TARGET_CORE, reg.CORE_CLIP_TRUNCATE, 0),
        E(reg.TARGET_CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, (15 << 5) | ((3 if depthwise else 0) << 3) | (2 << 1)),  # burst len, output mode
        E(reg.TARGET_DPU, reg.DATA_FORMAT, 0),
        E(reg.TARGET_DPU, reg.DST_BASE_ADDR, (output_dma + task_output_offset) & 0xFFFFFFFF),
        E(reg.TARGET_DPU, reg.DST_SURF_STRIDE, output_surface_stride << 4),
        E(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, output_width - 1),
        E(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, task_output_height - 1),
        E(reg.TARGET_DPU, reg.DATA_CUBE_NOTCH, 0),
        E(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, ((output_channels_real - 1) << 16) | (output_channels - 1)),
        E(reg.TARGET_DPU, reg.BS_CFG, (2 << 16) | (1 << 8) | (1 << 6) | (1 << 4)),
        E(reg.TARGET_DPU, reg.BS_ALU_CFG, 0),
        E(reg.TARGET_DPU, reg.BS_MUL_CFG, 0),
        E(reg.TARGET_DPU, reg.BS_RELUX_CMP_VALUE, 0),
        E(reg.TARGET_DPU, reg.BS_OW_CFG,
          ((3 if depthwise else 1) << 8) | ((3 if depthwise else 1) << 5) | ((3 if depthwise else 1) << 2)),
        E(reg.TARGET_DPU, reg.BS_OW_OP, 0x80 - WEIGHT_ZERO_POINT),
        E(reg.TARGET_DPU, reg.WDMA_SIZE_0, output_channels - 1),
        E(reg.TARGET_DPU, reg.WDMA_SIZE_1, ((task_output_height - 1) << 16) | (output_width - 1)),
        E(reg.TARGET_DPU, reg.BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.TARGET_DPU, reg.EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
        E(reg.TARGET_DPU, reg.EW_CVT_SCALE_VALUE, 1),
        E(reg.TARGET_DPU, reg.OUT_CVT_OFFSET, OUTPUT_ZERO_POINT - 0x80),
        E(reg.TARGET_DPU, reg.OUT_CVT_SCALE, out_scale),
        E(reg.TARGET_DPU, reg.OUT_CVT_SHIFT, shift - 1),
        E(reg.TARGET_DPU, reg.SURFACE_ADD, surfaces_per_row << 4),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, output_width - 1),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, task_output_height - 1),
        E(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, output_channels - 1),
        E(reg.TARGET_RDMA, reg.RDMA_BRDMA_CFG, 1 << 1),
        E(reg.TARGET_RDMA, reg.RDMA_BS_BASE_ADDR, bias_dma & 0xFFFFFFFF),
        E(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, 1),
        E(reg.TARGET_RDMA, reg.RDMA_FEATURE_MODE_CFG, (15 << 11) | (1 << 4) | ((3 if depthwise else 0) << 1)),
        E(reg.TARGET_RDMA, reg.RDMA_WEIGHT, (1 << 24) | (1 << 16) | (1 << 8) | 1),
        0,
        E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
        E(reg.TARGET_VERSION, 0, 0),
        E(reg.TARGET_PC, reg.OPERATION_ENABLE, (14 << 1) | 1),
    ]

def write_buffers(input_nhwc, weight_ohwi, biases):
    packed_input = pack_input(input_nhwc)
    packed_weight = pack_weights(weight_ohwi)
    packed_bias = pack_biases(biases, weight_ohwi)

    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), 0, input_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), 0, weight_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(bias_map)), 0, bias_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), packed_input.ctypes.data, packed_input.nbytes)
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), packed_weight.ctypes.data, packed_weight.nbytes)
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(bias_map)), packed_bias.ctypes.data, packed_bias.nbytes)

def read_output():
    raw = np.frombuffer(output_map, dtype=np.uint8, count=calc_raw_output_size(output_width, output_height, output_channels)).copy()
    raw = raw.reshape(-1, output_width, output_height, FEATURE_ATOMIC_SIZE)
    out = np.zeros((1, output_height, output_width, output_channels), dtype=np.uint8)
    for oc in range(output_channels):
        c = oc % FEATURE_ATOMIC_SIZE
        g = oc // FEATURE_ATOMIC_SIZE
        for y in range(output_height):
            for x in range(output_width):
                out[0, y, x, oc] = (int(raw[g, x, y, c]) + 0x80) & 0xFF
    return out

def expected_output(input_nhwc, weight_ohwi, biases):
    expected = np.zeros((input_nhwc.shape[0], output_height, output_width, output_channels), dtype=np.int64)
    input_per_group = input_channels // GROUPS
    output_per_group = output_channels // GROUPS
    for batch_index in range(input_nhwc.shape[0]):
        for oc in range(output_channels):
            expected[batch_index, :, :, oc] = int(biases[oc])
            group = oc // output_per_group if not DEPTHWISE else oc
            input_start = group * input_per_group if not DEPTHWISE else oc
            input_end = input_start + input_per_group if not DEPTHWISE else oc + 1
            for wy in range(weights_height):
                for wx in range(weights_width):
                    if DEPTHWISE:
                        expected[batch_index, :, :, oc] += (
                            input_nhwc[batch_index, wy:wy + output_height, wx:wx + output_width, oc].astype(np.int64) *
                            int(weight_ohwi[0, wx, wy, oc]))
                    else:
                        for ic in range(input_start, input_end):
                            expected[batch_index, :, :, oc] += (
                                input_nhwc[batch_index, wy:wy + output_height, wx:wx + output_width, ic].astype(np.int64) *
                                int(weight_ohwi[oc, wx, wy, ic - input_start]))
    conv_scale = (INPUT_SCALE * WEIGHT_SCALE) / OUTPUT_SCALE
    return np.clip(np.round(expected.astype(np.float64) * conv_scale + OUTPUT_ZERO_POINT), 0, 255).astype(np.uint8)

def run_conv2d_grouped(input_nhwc, weight_ohwi, biases):
    global input_channels, output_channels, GROUPS, DEPTHWISE

    real_input_channels = input_channels
    real_output_channels = output_channels
    real_groups = GROUPS
    real_depthwise = DEPTHWISE
    input_per_group = real_input_channels // real_groups
    output_per_group = real_output_channels // real_groups
    outputs = []

    try:
        GROUPS = 1
        DEPTHWISE = False
        input_channels = input_per_group
        output_channels = output_per_group
        for group in range(real_groups):
            input_start = group * input_per_group
            input_end = input_start + input_per_group
            output_start = group * output_per_group
            output_end = output_start + output_per_group
            outputs.append(run_conv2d_shape(
                input_nhwc[:, :, :, input_start:input_end],
                weight_ohwi[output_start:output_end],
                biases[output_start:output_end]))
        return np.concatenate(outputs, axis=3)
    finally:
        input_channels = real_input_channels
        output_channels = real_output_channels
        GROUPS = real_groups
        DEPTHWISE = real_depthwise

def run_conv2d_depthwise_chunks(input_nhwc, weight_ohwi, biases):
    global input_channels, output_channels, GROUPS

    real_input_channels = input_channels
    real_output_channels = output_channels
    real_groups = GROUPS
    outputs = []

    try:
        for start in range(0, real_input_channels, 64):
            end = min(start + 64, real_input_channels)
            input_channels = end - start
            output_channels = end - start
            GROUPS = end - start
            outputs.append(run_conv2d_shape(
                input_nhwc[:, :, :, start:end],
                weight_ohwi[:, :, :, start:end],
                biases[start:end]))
        return np.concatenate(outputs, axis=3)
    finally:
        input_channels = real_input_channels
        output_channels = real_output_channels
        GROUPS = real_groups

def run_conv2d_im2col(input_nhwc, weight_ohwi, biases):
    global input_width, input_height, input_channels, weights_width, weights_height, output_width, output_height, DEPTHWISE, GROUPS

    real_input_width = input_width
    real_input_height = input_height
    real_input_channels = input_channels
    real_weights_width = weights_width
    real_weights_height = weights_height
    real_output_width = output_width
    real_output_height = output_height
    real_depthwise = DEPTHWISE
    real_groups = GROUPS

    flat_c = real_weights_height * real_weights_width * real_input_channels
    flat_c_aligned = _align(flat_c, FEATURE_ATOMIC_SIZE)
    im2col = np.full((1, real_output_height, real_output_width, flat_c_aligned), INPUT_ZERO_POINT, dtype=np.uint8)
    flat = 0
    for ic in range(real_input_channels):
        for wy in range(real_weights_height):
            for wx in range(real_weights_width):
                im2col[0, :, :, flat] = input_nhwc[0, wy:wy + real_output_height, wx:wx + real_output_width, ic]
                flat += 1

    weight_1x1 = np.full((output_channels, 1, 1, flat_c_aligned), WEIGHT_ZERO_POINT, dtype=np.uint8)
    if real_depthwise:
        for channel in range(real_input_channels):
            flat = channel * real_weights_height * real_weights_width
            for wy in range(real_weights_height):
                for wx in range(real_weights_width):
                    weight_1x1[channel, 0, 0, flat] = weight_ohwi[0, wx, wy, channel]
                    flat += 1
    else:
        for oc in range(output_channels):
            flat = 0
            for ic in range(real_input_channels):
                for wy in range(real_weights_height):
                    for wx in range(real_weights_width):
                        weight_1x1[oc, 0, 0, flat] = weight_ohwi[oc, wx, wy, ic]
                        flat += 1

    input_width = real_output_width
    input_height = real_output_height
    input_channels = flat_c_aligned
    weights_width = 1
    weights_height = 1
    output_width = real_output_width
    output_height = real_output_height
    DEPTHWISE = False
    GROUPS = 1
    try:
        return run_conv2d_shape(im2col, weight_1x1, biases)
    finally:
        input_width = real_input_width
        input_height = real_input_height
        input_channels = real_input_channels
        weights_width = real_weights_width
        weights_height = real_weights_height
        output_width = real_output_width
        output_height = real_output_height
        DEPTHWISE = real_depthwise
        GROUPS = real_groups

def run_conv2d_shape(input_nhwc, weight_ohwi, biases):
    if input_nhwc.shape[0] > 1:
        return np.concatenate([
            run_conv2d_shape(input_nhwc[batch_index:batch_index + 1], weight_ohwi, biases)
            for batch_index in range(input_nhwc.shape[0])
        ], axis=0)

    if GROUPS != 1 and not DEPTHWISE:
        return run_conv2d_grouped(input_nhwc, weight_ohwi, biases)

    if DEPTHWISE and input_channels > 64 and (input_channels % 64) != 0:
        return run_conv2d_depthwise_chunks(input_nhwc, weight_ohwi, biases)

    if ((weights_width != 1 or weights_height != 1) and input_channels <= 3) or \
       (input_channels == 1 and (output_height == 1 or output_width == 1)):
        return run_conv2d_im2col(input_nhwc, weight_ohwi, biases)

    write_buffers(input_nhwc, weight_ohwi, biases)

    task_list = split_tasks()
    reg_lists = [build_conv_regs(input_mem_create.dma_addr, weight_mem_create.dma_addr, bias_mem_create.dma_addr, output_mem_create.dma_addr, task)
                 for task in task_list]
    offsets = []
    regcmd_offset = 0
    for regs in reg_lists:
        offsets.append(regcmd_offset)
        regcmd_offset += _align(len(regs), 8)
    for task_index, regs in enumerate(reg_lists[:-1]):
        next_regs = reg_lists[task_index + 1]
        next_addr = regcmd_mem_create.dma_addr + offsets[task_index + 1] * ctypes.sizeof(ctypes.c_uint64)
        regs_to_fetch = len(next_regs) - 4
        regs_to_fetch = _align(regs_to_fetch // 2, 2)
        regs[-4] = E(reg.TARGET_PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0)
        regs[-3] = E(reg.TARGET_PC_REG, reg.PC_REGISTER_AMOUNTS, regs_to_fetch)

    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, regcmd_map.size())
    for task_index, npu_regs in enumerate(reg_lists):
        regcmd_offset = offsets[task_index]
        for i, qword in enumerate(npu_regs):
            regcmd[regcmd_offset + i] = qword

        tasks[task_index].flags  = 0;
        tasks[task_index].op_idx = 1;
        tasks[task_index].enable_mask = 0xd;
        tasks[task_index].int_mask = 0x300;
        tasks[task_index].int_clear = 0x1ffff;
        tasks[task_index].int_status = 0;
        tasks[task_index].regcfg_amount = len(npu_regs)
        tasks[task_index].regcfg_offset = 0;
        tasks[task_index].regcmd_addr = regcmd_mem_create.dma_addr + regcmd_offset * ctypes.sizeof(ctypes.c_uint64)

    for bo in (regcmd_mem_create, input_mem_create, weight_mem_create, bias_mem_create, output_mem_create):
        ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))
    ret = rocket_submit(fd, tasks, len(task_list),
        in_bos=[input_mem_create, regcmd_mem_create, weight_mem_create, bias_mem_create],
        out_bos=[output_mem_create])
    ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, drm_rocket_prep_bo(handle=output_mem_create.handle, timeout_ns=time.monotonic_ns() + 6000000000))
    print(f"SUBMIT ret={ret}")

    return read_output()

if __name__ == "__main__":
    shapes = load_conv_legacy_shapes()
    name_width = max(len(shape["name"]) for shape in shapes) if shapes else 1
    passed = failed = skipped = 0

    print(f"Running {len(shapes)} shapes extracted from conv_legacy.py ...")
    for shape in shapes:
        print(f"  {shape['name']:<{name_width}s} ...", end=" ")
        sys.stdout.flush()
        try:
            use_shape(shape)
        except ValueError as e:
            failed += 1
            print(f"FAIL ({e})")
            raise

        np.random.seed(42)
        input_nhwc = np.random.randint(0, 256, (shape["batch"], input_height, input_width, input_channels), dtype=np.uint8)
        weight_shape = ((1, weights_width, weights_height, input_channels) if DEPTHWISE else
                        (output_channels, weights_width, weights_height, weight_input_channels))
        weight_ohwi = np.random.randint(0, 256, weight_shape, dtype=np.uint8)
        biases = np.random.randint(-128, 128, (output_channels,), dtype=np.int32)

        result = run_conv2d_shape(input_nhwc, weight_ohwi, biases)
        expected = expected_output(input_nhwc, weight_ohwi, biases)
        diff = int(np.max(np.abs(result.astype(np.int32) - expected.astype(np.int32))))
        ok = diff <= 1
        passed += 1 if ok else 0
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} (max_diff={diff})")
        assert ok, f"{shape['name']} failed (max_diff={diff})"

    print(f"conv_legacy-derived simple_conv matrix: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    os.close(fd)
    sys.exit(0 if failed == 0 else 1)
