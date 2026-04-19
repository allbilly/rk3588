#!/usr/bin/env python3
import os
import sys
import mmap
import ctypes
import struct
import numpy as np
import fcntl
import math
from enum import Enum, auto
import rockchip as rk

_libc = ctypes.CDLL("libc.so.6")
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
_libc.munmap.restype = ctypes.c_int

def getenv(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def mv_address(mv: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))

class Ops(Enum):
    ADD = auto()
    MUL = auto()
    NEG = auto()
    MAX = auto()
    EXP2 = auto()
    CMPLT = auto()
    CMPEQ = auto()
    FDIV = auto()
    SUB = auto()
    WMMA = auto()
    TRUNC = auto()
    CUSTOM = auto()

class FileIOInterface:
    def __init__(self, path: str = "", flags: int = os.O_RDONLY, fd: int | None = None):
        self.path: str = path
        self.fd: int = fd or os.open(path, flags)
    
    def __del__(self):
        if hasattr(self, 'fd'):
            os.close(self.fd)
    
    def ioctl(self, request, arg):
        return fcntl.ioctl(self.fd, request, arg)
    
    def mmap(self, start, sz, prot, flags, offset):
        return FileIOInterface._mmap(start, sz, prot, flags, self.fd, offset)
    
    @staticmethod
    def _mmap(start, sz, prot, flags, fd, offset):
        x = _libc.mmap(start, sz, prot, flags, fd, offset)
        if x == 0xffffffffffffffff:
            raise OSError(f"Failed to mmap {sz} bytes: {os.strerror(ctypes.get_errno())}")
        return x

    @staticmethod
    def munmap(buf, sz):
        return _libc.munmap(buf, sz)

class HCQBuffer:
    def __init__(self, va_addr: int, size: int, meta=None):
        self.va_addr = va_addr
        self.size = size
        self.meta = meta
    
    def offset(self, offset: int = 0, size: int | None = None):
        return HCQBuffer(self.va_addr + offset, size or (self.size - offset), meta=self.meta)

class RockchipProgram:
    def __init__(self):
        self.fd_ctl = FileIOInterface("/dev/dri/card1", os.O_RDWR)
        
        self.hardware_ops = {
            Ops.ADD: 2,
            Ops.MUL: 0,
            Ops.NEG: 0,
            Ops.MAX: 0,
            Ops.EXP2: 0,
            Ops.CMPLT: 0,
            Ops.CMPEQ: 0,
            Ops.FDIV: 3,
            Ops.SUB: 4,
            Ops.WMMA: 0,
            Ops.TRUNC: 0,
            Ops.CUSTOM: 0,
        }
        
        self.cmd_buf_size = 16384
        self.lut_size = 513
        self.lut_enable = False
        self.q = []
    
    def reset_npu(self):
        rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, __payload=rk.struct_rknpu_action(
            flags=rk.RKNPU_ACT_RESET, value=0))
    
    def create_flink_name(self, handle: int, name: str, virt_address: int | None = None,
                          obj_addr: int | None = None, dma_address: int | None = None) -> int:
        flink_req = rk.struct_drm_gem_flink(handle=handle, name=0)
        result = rk.DRM_IOCTL_GEM_FLINK(self.fd_ctl, __payload=flink_req)
        return flink_req.name
    
    def _gpu_alloc(self, size: int, flags, name: str) -> HCQBuffer:
        mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(
            self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
        mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem_create.handle, offset=0)
        va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE,
                                   mmap.MAP_SHARED, mem_map.offset)
        mem_create.flink_name = self.create_flink_name(
            mem_create.handle, name, virt_address=va_addr,
            obj_addr=mem_create.obj_addr, dma_address=mem_create.dma_addr)
        
        return HCQBuffer(va_addr=va_addr, size=size, meta=mem_create)
    
    def _gpu_sync(self, buf: HCQBuffer, flags: int) -> None:
        if not getenv("ROCKCHIP_MEM_SYNC", 0):
            return
        rk.DRM_IOCTL_RKNPU_MEM_SYNC(
            self.fd_ctl, __payload=rk.struct_rknpu_mem_sync(
                flags=flags, reserved=0, obj_addr=buf.meta.obj_addr,
                offset=0, size=buf.size))
    
    def _gpu_free(self, buf: HCQBuffer) -> None:
        FileIOInterface.munmap(buf.va_addr, buf.size)
        rk.DRM_IOCTL_RKNPU_MEM_DESTROY(
            self.fd_ctl, __payload=rk.struct_rknpu_mem_destroy(
                handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr))
    
    def _gpu_free_multiple(self, buf_list) -> None:
        for buf in buf_list:
            self._gpu_free(buf)
    
    def reg(self, val, shift, mask):
        return ((val) << shift) & mask
    
    def emit_raw(self, target, reg_addr, value):
        target = target + 0x1
        packed = ((target & 0xFFFF) << 48) | ((value & 0xFFFFFFFF) << 16) | (reg_addr & 0xFFFF)
        self.q.append(packed)
    
    def check_lut_enable(self, op, arg):
        return op in (Ops.EXP2, Ops.TRUNC) or (op is Ops.CUSTOM and arg == "silu")
    
    def boilerplate(self, op, size, arg, feature_addr=0, weight_addr=0,
                   dst_addr=0, wmma_meta=None):
        burst_len = 15
        output_mode = 2
        flying_mode = 1
        channel = 7
        dataout_height = 0
        dataout_width = math.ceil(size / ((dataout_height + 1) * (channel + 1))) - 1
        
        precision_float16 = 2
        ew_cvt_type = 0
        ew_data_mode = 1
        ew_data_size = 2
        ew_relu_bypass = arg != "relu"
        ew_alu_algo = self.hardware_ops.get(op, 0)
        ew_op_src = 1
        erdma_data_size_16bit = 2
        
        if self.lut_enable:
            ew_data_mode = 0
            ew_data_size = 0
            ew_op_src = 0
        
        self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
            self.reg(burst_len, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT,
                    rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
            self.reg(output_mode, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT,
                    rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
            self.reg(flying_mode, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT,
                    rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
        
        self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
            self.reg(precision_float16, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT,
                    rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
            self.reg(precision_float16, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT,
                    rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
            self.reg(precision_float16, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT,
                    rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
        
        self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
            self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT,
                    rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
            self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT,
                    rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
        
        self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
            self.reg(dataout_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT,
                    rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
        
        is_mul_op = 1 if op == Ops.MUL else 0
        is_cvt_bypass = 1 if (op in [Ops.MUL, Ops.FDIV] or self.lut_enable) else 0
        is_lut_bypass = 0 if self.lut_enable else 1
        is_op_bypass = 1 if self.lut_enable else 0
        is_ew_bypass = 1 if arg in ["cmplt_diff2bool", "cmpeq_diff_zero_to_nan_to_32800",
                                     "cmpeq_32800_to_bool"] else 0
        
        self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
            self.reg(ew_cvt_type, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT,
                    rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
            self.reg(ew_data_mode, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT,
                    rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
            self.reg(ew_data_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT,
                    rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
            self.reg(ew_alu_algo, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT,
                    rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
            self.reg(is_mul_op, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT,
                    rk.DPU_EW_CFG_EW_OP_TYPE__MASK) |
            self.reg(ew_relu_bypass, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT,
                    rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
            self.reg(is_cvt_bypass, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT,
                    rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
            self.reg(is_lut_bypass, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT,
                    rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
            self.reg(ew_op_src, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT,
                    rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
            self.reg(is_op_bypass, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT,
                    rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
            self.reg(is_ew_bypass, rk.DPU_EW_CFG_EW_BYPASS__SHIFT,
                    rk.DPU_EW_CFG_EW_BYPASS__MASK))
        
        if op == Ops.FDIV:
            self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
                self.reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT,
                        rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
        
        self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
            self.reg(dataout_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT,
                    rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
        self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
            self.reg(dataout_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT,
                    rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
        self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
            self.reg(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT,
                    rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
        self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
            self.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT,
                    rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
            self.reg(erdma_data_size_16bit, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT,
                    rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    
    def submit(self, uop):
        self.q.append(0x2001000178495044)
        self.q.append(0x0081000000180008)
        
        tasks = ctypes.cast(self.task_buf.va_addr,
                           ctypes.POINTER(rk.struct_rknpu_task * 128)).contents
        
        assert len(self.q) <= self.cmd_buf_size, "Command buffer overflow"
        
        regcmd = ctypes.cast(self.cmd_buf.va_addr,
                            ctypes.POINTER(ctypes.c_uint64 * self.cmd_buf_size)).contents
        
        for i in range(len(self.q)):
            regcmd[i] = self.q[i]
            print(hex(regcmd[i]))
        
        tasks[0].flags = 0
        tasks[0].op_idx = 4
        tasks[0].enable_mask = 0x18
        tasks[0].int_mask = 0x300
        tasks[0].int_clear = 0x1ffff
        tasks[0].int_status = 0
        tasks[0].regcfg_amount = len(self.q)
        tasks[0].regcfg_offset = 0
        tasks[0].regcmd_addr = self.cmd_buf.meta.dma_addr
        
        submit_res = rk.struct_rknpu_submit(
            flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
            timeout=6000,
            task_start=0,
            task_number=1,
            task_counter=0,
            priority=0,
            task_obj_addr=self.task_buf.meta.obj_addr,
            regcfg_obj_addr=0,
            task_base_addr=0,
            user_data=0,
            core_mask=1,
            fence_fd=-1,
            subcore_task=(rk.struct_rknpu_subcore_task * 5)(
                rk.struct_rknpu_subcore_task(task_start=0, task_number=1),
                rk.struct_rknpu_subcore_task(task_start=1, task_number=0),
                rk.struct_rknpu_subcore_task(task_start=2, task_number=0),
            )
        )
        
        res = rk.DRM_IOCTL_RKNPU_SUBMIT(self.fd_ctl, __payload=submit_res)
        return res
    
    def run(self, op, arg, src_values):
        self.reset_npu()
        self.q = []
        self.lut_enable = self.check_lut_enable(op, arg)
        
        input1 = np.asarray(src_values[0], dtype=np.float16)
        input2 = np.asarray(src_values[1], dtype=np.float16)
        
        if len(input1) != len(input2):
            raise ValueError(f"Input arrays must be same size: {len(input1)} vs {len(input2)}")
        
        src = memoryview(bytearray(input1.tobytes()))
        src2 = memoryview(bytearray(input2.tobytes()))
        
        print(f"Input size: {len(input1)} elements, {src.nbytes} bytes")
        
        self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task_buf")
        self.cmd_buf = self._gpu_alloc(self.cmd_buf_size, 0, name="cmd_buf")
        self.input_buf = self._gpu_alloc(src.nbytes, 0, name="input")
        self.weight_buf = self._gpu_alloc(src2.nbytes, 0, name="weight")
        self.output_buf = self._gpu_alloc(src.nbytes, 0, name="output")
        
        try:
            print("Copying data to device...")
            ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
            ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
            self._gpu_sync(self.input_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
            self._gpu_sync(self.weight_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
            
            print(f"Configuring {op.name} operation...")
            self.boilerplate(op=op, size=len(input1), arg=arg)
            
            self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
                self.reg(self.output_buf.meta.dma_addr,
                        rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT,
                        rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))

            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
                self.reg(self.input_buf.meta.dma_addr,
                        rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT,
                        rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
                self.reg(self.weight_buf.meta.dma_addr,
                        rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT,
                        rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
            
            print("Submitting to NPU...")
            res = self.submit(op)
            print(f"Submit result: {res}")
            
            print("Reading results...")
            self._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
            dst = memoryview(bytearray(self.output_buf.size))
            ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
            result = struct.unpack(f'<{self.output_buf.size // 2}e', dst.tobytes())
            
            return list(result)
            
        finally:
            print("Cleaning up buffers...")
            self._gpu_free_multiple([self.task_buf, self.cmd_buf, self.input_buf,
                                    self.weight_buf, self.output_buf])


if __name__ == "__main__":
    print("=" * 60)
    print("Rockchip NPU Standalone ADD Test")
    print("=" * 60)
    
    try:
        print("\nInitializing RockchipProgram...")
        prog = RockchipProgram()
        print("Program initialized")
        
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float16)
        b = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float16)
        
        print(f"\nTest Data:")
        print(f"  Input A: {a}")
        print(f"  Input B: {b}")
        print(f"  Expected (A + B): {a + b}")
        
        print(f"\nRunning ADD operation on NPU...")
        result = prog.run(Ops.ADD, None, [a, b])
        result_arr = np.array(result, dtype=np.float16)
        
        print(f"\nResults:")
        print(f"  NPU Output: {result_arr}")
        print(f"  CPU Reference: {a + b}")
        
        expected = a + b
        match = np.allclose(result_arr, expected, atol=0.01)
        print(f"\n{'=' * 60}")
        if match:
            print("TEST PASSED: Results match within tolerance")
        else:
            print("TEST FAILED: Results do not match")
            print(f"  Difference: {np.abs(result_arr - expected)}")
        print(f"{'=' * 60}")
        
        print("\n" + "=" * 60)
        print("Testing MUL operation...")
        print("=" * 60)
        result_mul = prog.run(Ops.MUL, None, [a, b])
        result_mul_arr = np.array(result_mul, dtype=np.float16)
        expected_mul = a * b
        
        print(f"  NPU Output: {result_mul_arr}")
        
        match_mul = np.allclose(result_mul_arr, expected_mul, atol=0.01)
        if match_mul:
            print("MUL TEST PASSED")
        else:
            print("MUL TEST FAILED")
            print(f"  Difference: {np.abs(result_mul_arr - expected_mul)}")
        
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
