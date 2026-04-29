from fcntl import ioctl
import os, mmap, sys
import ctypes, struct
import numpy as np
RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1

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
    mem_create = rknpu_mem_create(
        flags=flags, #0x10 | 0x2,  # KERNEL_MAPPING | NON_CACHEABLE
        size=size
    )
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    print(f"ret={ret}, handle={mem_create.handle}, obj_addr={mem_create.obj_addr:#x}, dma_addr={mem_create.dma_addr:#x}")
    
    # Map memory to access from userspace
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    buf = mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset)
    print(f"Memory mapped at offset={mem_map.offset:#x}")
    return buf, mem_create

def submit(task_obj_addr):
    submit_struct = rknpu_submit(
        flags=0x1 | 0x2 | 0x4,  # RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG
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
    # struct len is 5 but only 3 NPU core
    submit_struct.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
    submit_struct.subcore_task[1] = rknpu_subcore_task(task_start=1, task_number=0)
    submit_struct.subcore_task[2] = rknpu_subcore_task(task_start=2, task_number=0)
    submit_struct.subcore_task[3] = rknpu_subcore_task(task_start=0, task_number=0)
    submit_struct.subcore_task[4] = rknpu_subcore_task(task_start=0, task_number=0)
    
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit_struct)

def reset_npu(fd):
    action = rknpu_action(flags=RKNPU_ACT_RESET, value=0)
    ret = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, action)
    print(f"reset_npu ret={ret}")
    return ret

# EW_CFG values for each op
# Base config: data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1
_EW_BASE = 0x108002c0
EW_CFG_ADD  = _EW_BASE | (2 << 16)
EW_CFG_MUL  = _EW_BASE | (1 << 2) | (1 << 8)
EW_CFG_SUB  = _EW_BASE | (4 << 16)
EW_CFG_MAX  = _EW_BASE
EW_CFG_NEG  = EW_CFG_MUL

task_map, tasks_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
regcmd_map, regcmd_mem_create = mem_allocate(fd, size=1024, flags=RKNPU_MEM_NON_CACHEABLE)
input_map, input_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
weight_map, weight_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)
output_map, output_mem_create = mem_allocate(fd, size=4194304, flags=RKNPU_MEM_NON_CACHEABLE)

tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))


def run_op(ew_cfg_val, a_vals, b_vals, neg_op=False, fdiv_op=False):
    n = len(a_vals)
    dataout_width = (n + 7) // 8 - 1

    out_cvt = 1 if fdiv_op else 0x10001
    feat_cfg = 0x00017841 if fdiv_op else 0x00017849
    npu_regs = [
        (0x1001 << 48) | (0x000001e5 << 16) | 0x400c,
        (0x1001 << 48) | (0x48000002 << 16) | 0x4010,
        (0x1001 << 48) | (dataout_width << 16) | 0x4030,
        (0x1001 << 48) | (0x00070007 << 16) | 0x403c,
        (0x1001 << 48) | (ew_cfg_val << 16) | 0x4070,
        (0x1001 << 48) | (out_cvt << 16) | 0x4084,
        (0x2001 << 48) | (dataout_width << 16) | 0x500c,
        (0x2001 << 48) | (0x00000000 << 16) | 0x5010,
        (0x2001 << 48) | (0x00000007 << 16) | 0x5014,
        (0x2001 << 48) | (0x40000008 << 16) | 0x5034,
        (0x1001 << 48) | ((output_mem_create.dma_addr & 0xFFFFFFFF) << 16) | 0x4020,
        (0x2001 << 48) | ((input_mem_create.dma_addr & 0xFFFFFFFF) << 16) | 0x5018,
        (0x2001 << 48) | ((weight_mem_create.dma_addr & 0xFFFFFFFF) << 16) | 0x5038,
        (0x2001 << 48) | (feat_cfg << 16) | 0x5044,
        (0x0081 << 48) | (0x00000018 << 16) | 0x0008,
    ]

    for i in range(16):
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
