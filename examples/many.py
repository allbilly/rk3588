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

    def fill_lut(self, lut):
        for table_id, base in ((0, 0), (1, self.lut_size)):
            self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG,
                self.reg(1, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__SHIFT,
                        rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__MASK) |
                self.reg(table_id, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__SHIFT,
                        rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__MASK) |
                self.reg(0, rk.DPU_LUT_ACCESS_CFG_LUT_ADDR__SHIFT,
                        rk.DPU_LUT_ACCESS_CFG_LUT_ADDR__MASK))
            for i in range(self.lut_size):
                self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA,
                    self.reg(lut[base + i], rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT,
                            rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK))
    
    def _align_up(self, val: int, align: int) -> int:
        if align <= 0: return val
        return ((val + align - 1) // align) * align

    def _dtype_from_code(self, code: int):
        if code == 0: return np.float16
        if code == 1: return np.float32
        raise RuntimeError(f"dtype_code_{code}")

    def _wmma_params(self, m: int, n: int, k: int) -> dict[str, int]:
        m = max(1, m)
        n = max(1, n)
        k = max(1, k)
        align_in = max(32, self._align_up(k, 32))
        align_out = max(32, self._align_up(n, 32))
        data_in_width, data_in_height = 1, m
        dataout_width, dataout_height = 1, m
        out_width_stride = 1
        is_kn_64 = k == 64 and n == 64
        is_kn_256 = k == 256 and n == 256
        is_kn_512 = k == 512 and n == 512
        is_kn_lg_512 = k > 512 and n > 512
        is_matmul_64 = m == 64 and k == 64 and n == 64
        is_matmul_256 = m == 256 and k == 256 and n == 256
        feature_grains = data_in_height + 1
        if k > 7872:
            feature_grains = 2
        elif 128 < k <= 192:
            feature_grains = data_in_height
        elif k > 192 and k != 256:
            denom = align_in * 2
            grains = (2 * 32768 + denom - 1) // denom
            grains = (grains + 1) & ~1
            feature_grains = max(80, grains)
        weight_bytes_per_kernel = align_in * 2
        fd_bytes = data_in_width * data_in_height * align_in * 2
        data_bank = max(1, min(11, (fd_bytes + 32768 - 1) // 32768))
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

    def check_lut_enable(self, op, arg):
        return op in (Ops.EXP2, Ops.TRUNC) or (op is Ops.CUSTOM and arg == "silu")
    
    def boilerplate(self, op, size, arg, feature_addr=0, weight_addr=0,
                   dst_addr=0, wmma_meta=None):
        if op is Ops.WMMA:
            p = wmma_meta if wmma_meta is not None else self._wmma_params(2, 2, 2)
            self.emit_raw(rk.DPU, rk.REG_DPU_S_POINTER,
                self.reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
                self.reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
                self.reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))

            is_kn_64 = p["k"] == 64 and p["n"] == 64
            is_kn_256 = p["k"] == 256 and p["n"] == 256
            is_kn_512 = p["k"] == 512 and p["n"] == 512
            is_kn_lg_512 = p["k"] > 512 and p["n"] > 512
            is_m_1_kn_768 = p["m"] == 1 and p["k"] == 768 and p["n"] == 768
            is_m_1_k768_n2048 = p["m"] == 1 and p["k"] == 768 and p["n"] == 2048
            is_m_1_kn_2048 = p["m"] == 1 and p["k"] == 2048 and p["n"] == 2048
            conv_con1 = self.reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) | \
                        self.reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK)
            if not (is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or is_m_1_kn_768 or is_m_1_k768_n2048 or is_m_1_kn_2048):
                conv_con1 |= self.reg(1, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK)
            self.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON1, conv_con1)
            self.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON2,
                self.reg(p["feature_grains"], rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT, rk.CNA_CONV_CON2_FEATURE_GRAINS__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON3,
                self.reg(1, rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_Y_STRIDE__MASK) |
                self.reg(1, rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_X_STRIDE__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE0,
                self.reg(p["data_in_width"], rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__MASK) |
                self.reg(p["data_in_height"], rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE1,
                self.reg(p["align_in"]-1, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__MASK) |
                self.reg(p["align_in"], rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE2,
                self.reg(p["dataout_width"], rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__SHIFT, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE3,
                self.reg(p["dataout_width"]*p["dataout_height"], rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__SHIFT, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0,
                self.reg(p["weight_bytes_per_kernel"]*p["align_out"], rk.CNA_WEIGHT_SIZE0_WEIGHT_BYTES__SHIFT, rk.CNA_WEIGHT_SIZE0_WEIGHT_BYTES__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1,
                self.reg(p["weight_bytes_per_kernel"], rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__SHIFT,
                         rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2,
                self.reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK) |
                self.reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK) |
                self.reg(p["align_out"], rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_CBUF_CON0,
                self.reg(12-p["data_bank"], rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
                self.reg(p["data_bank"], rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_CBUF_CON1,
                self.reg((p["data_in_width"]*p["align_in"]+31)//32, rk.CNA_CBUF_CON1_DATA_ENTRIES__SHIFT,
                         rk.CNA_CBUF_CON1_DATA_ENTRIES__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON0,
                self.reg(1, rk.CNA_CVT_CON0_DATA_SIGN__SHIFT, rk.CNA_CVT_CON0_DATA_SIGN__MASK) |
                self.reg(1, rk.CNA_CVT_CON0_CVT_TYPE__SHIFT, rk.CNA_CVT_CON0_CVT_TYPE__MASK) |
                self.reg(1, rk.CNA_CVT_CON0_CVT_BYPASS__SHIFT, rk.CNA_CVT_CON0_CVT_BYPASS__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON1,
                self.reg(1, rk.CNA_CVT_CON1_CVT_SCALE0__SHIFT, rk.CNA_CVT_CON1_CVT_SCALE0__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON2,
                self.reg(1, rk.CNA_CVT_CON2_CVT_SCALE1__SHIFT, rk.CNA_CVT_CON2_CVT_SCALE1__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON3,
                self.reg(1, rk.CNA_CVT_CON3_CVT_SCALE2__SHIFT, rk.CNA_CVT_CON3_CVT_SCALE2__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON4,
                self.reg(1, rk.CNA_CVT_CON4_CVT_SCALE3__SHIFT, rk.CNA_CVT_CON4_CVT_SCALE3__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR,
                self.reg(feature_addr, rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__SHIFT,
                          rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__MASK))

            self.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON0,
                self.reg(15, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__MASK) |
                self.reg(15, rk.CNA_DMA_CON0_DATA_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_DATA_BURST_LEN__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON1,
                self.reg(p["line_stride"], rk.CNA_DMA_CON1_LINE_STRIDE__SHIFT, rk.CNA_DMA_CON1_LINE_STRIDE__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON2,
                self.reg(p["surf_stride"], rk.CNA_DMA_CON2_SURF_STRIDE__SHIFT, rk.CNA_DMA_CON2_SURF_STRIDE__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0,
                self.reg(p["data_in_width"], rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__MASK) |
                self.reg(p["data_in_height"], rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1,
                self.reg(p["align_in"], rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__SHIFT, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__MASK))
            self.emit_raw(rk.CNA, rk.REG_CNA_DCOMP_ADDR0,
                self.reg(weight_addr, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__SHIFT,
                          rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK))

            self.emit_raw(rk.CORE, rk.REG_CORE_MISC_CFG,
                self.reg(2, rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT, rk.CORE_MISC_CFG_PROC_PRECISION__MASK) |
                self.reg(1, rk.CORE_MISC_CFG_QD_EN__SHIFT, rk.CORE_MISC_CFG_QD_EN__MASK))
            self.emit_raw(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0,
                self.reg(p["dataout_height"]-1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK) |
                self.reg(p["dataout_width"]-1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK))
            self.emit_raw(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1,
                self.reg(p["align_out"]-1, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__SHIFT, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK))

            self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
                self.reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
                self.reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
                self.reg(5, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
                self.reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
                self.reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
                self.reg(dst_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT,
                          rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
                self.reg(p["dst_surf_stride"], rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
                self.reg(p["dataout_width"]-1, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
                self.reg(p["dataout_height"]-1, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,
                self.reg(p["notch_val"], rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__MASK) |
                self.reg(p["notch_val"], rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
                self.reg(p["align_out"]-1, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
                self.reg(p["align_out"]-1, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
                self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
                self.reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
                self.reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
                self.reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_OW_CFG,
                self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_2__MASK) |
                self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_1__MASK) |
                self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_0__MASK) |
                self.reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
                self.reg(p["align_out"]-1, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
                self.reg(p["dataout_height"]-1, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
                self.reg(p["dataout_width"]-1, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
                self.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
                self.reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
                self.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
                self.reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
                self.reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
                self.reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
                self.reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
                self.reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
                self.reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_SURFACE_ADD,
                self.reg(p["dst_surf_stride"]*4, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
            return

        if self.lut_enable:
            lut = [0] * self.lut_size * 2
            index_shift = 5
            index_scale = 0.0
            if op is Ops.EXP2:
                x_min, x_max = -2.0, 2.0
                step = (x_max - x_min) / (len(lut) - 1)
                index_scale = (1 << index_shift) / step
                max_val = max(2.0 ** x_min, 2.0 ** x_max)
                self.inv_scale = 1.0 / max_val if max_val > 1.0 else 1.0
                for i in range(len(lut)):
                    x = x_min + i * step
                    y = (2.0 ** x) * self.inv_scale
                    q = int(math.floor((y + 1.0) * 2**14 + 0.5))
                    lut[i] = np.clip(q, 0, 32767)
            elif op is Ops.CUSTOM and arg == "silu":
                x_min, x_max = 0, 5.8
                step = (x_max - x_min) / (self.lut_size - 1)
                index_scale = (1 << index_shift) / step
                max_val = max(x_min / (1.0 + math.exp(-x_min)), x_max / (1.0 + math.exp(-x_max)))
                self.inv_scale = 1.0 / max_val if max_val > 1.0 else 1.0
                for i in range(self.lut_size * 2):
                    x = (i - self.lut_size + (i < self.lut_size)) * step
                    y = x / (1.0 + math.exp(-x)) * self.inv_scale
                    q = int(math.floor(y * (2**15 - 1) + 0.5)) if y >= 0.0 else int(math.ceil(y * (2**15 - 1) - 0.5))
                    lut[i] = np.clip(q, -32768, 32767)
            elif op is Ops.TRUNC:
                max_val = 1 << 14
                for table_id in range(2):
                    base = table_id * self.lut_size
                    for i in range(self.lut_size):
                        lut[base + i] = 0 if (i % 2 == 0) else max_val
            bn_mul_operand = int(np.float16(index_scale).view(np.int16)) if index_scale != 0 else 0x3C00

            self.fill_lut(lut)
            self.emit_raw(rk.DPU, rk.REG_DPU_LUT_CFG,
                self.reg(1, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK) |
                self.reg(1, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK) |
                self.reg(2, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__SHIFT, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__MASK))
            index_select = 14 if op is Ops.TRUNC else 5
            self.emit_raw(rk.DPU, rk.REG_DPU_LUT_INFO,
                self.reg(index_select, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__MASK) |
                self.reg(index_select, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK))
            if op is Ops.TRUNC:
                self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START,
                    self.reg(0x00000000, rk.DPU_LUT_LE_START_LUT_LE_START__SHIFT, rk.DPU_LUT_LE_START_LUT_LE_START__MASK))
                self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_END,
                    self.reg(0x44000000, rk.DPU_LUT_LE_END_LUT_LE_END__SHIFT, rk.DPU_LUT_LE_END_LUT_LE_END__MASK))
                self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_START,
                    self.reg(0x44000000, rk.DPU_LUT_LO_START_LUT_LO_START__SHIFT, rk.DPU_LUT_LO_START_LUT_LO_START__MASK))
                self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END,
                    self.reg(0x44800000, rk.DPU_LUT_LO_END_LUT_LO_END__SHIFT, rk.DPU_LUT_LO_END_LUT_LO_END__MASK))
            else:
                self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START,
                    self.reg(0xffffc000, rk.DPU_LUT_LE_START_LUT_LE_START__SHIFT, rk.DPU_LUT_LE_START_LUT_LE_START__MASK))
                self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END,
                    self.reg(0x00004000, rk.DPU_LUT_LO_END_LUT_LO_END__SHIFT, rk.DPU_LUT_LO_END_LUT_LO_END__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE,
                self.reg(23107, rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__SHIFT,
                        rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT,
                self.reg(22, rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__SHIFT,
                        rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
                self.reg(2, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
                self.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
                self.reg(bn_mul_operand, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))

        elif op is Ops.CUSTOM and arg == "cmplt_diff2bool":
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
                self.reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
                self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
                self.reg(0x33800000, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
                self.reg(0x4000, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
                self.reg(4, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
                self.reg(1, rk.DPU_BN_CFG_BN_RELUX_EN__SHIFT, rk.DPU_BN_CFG_BN_RELUX_EN__MASK) |
                self.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
                self.reg(0x7C00, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE,
                self.reg(0x3F800000, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__SHIFT, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__MASK))
        elif op is Ops.CUSTOM and arg == "cmpeq_diff_zero_to_nan_to_32800":
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
                self.reg(2, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
                self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
                self.reg(0x7C00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
                self.reg(1, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))
        elif op is Ops.CUSTOM and arg == "cmpeq_32800_to_bool":
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
                self.reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
                self.reg(0, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
                self.reg(0x47001F00, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
                self.reg(0x3C00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
                self.reg(0, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))

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
        if uop is not Ops.WMMA:
            self.q.append(0x2001000178495044)
        self.q.append(0x00810000000d0008 if uop is Ops.WMMA else 0x0081000000180008)
        
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

    def _run_wmma_matmul(self, a_matrix: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
        m, k = a_matrix.shape
        if b_matrix.shape[0] != k: raise RuntimeError("k_mismatch")
        n = int(b_matrix.shape[1])
        wmma_meta = self._wmma_params(int(m), int(n), int(k))
        in_pack = np.zeros(wmma_meta["align_in"] * wmma_meta["m"], dtype=np.float16)
        wt_pack = np.zeros(wmma_meta["align_out"] * wmma_meta["align_in"], dtype=np.float16)
        if (wmma_meta["m"], wmma_meta["n"], wmma_meta["k"]) == (64, 64, 64):
            for mm in range(1, 65):
                for kk in range(1, 65):
                    plane = (kk - 1) // 8
                    offset = (kk - 1) % 8
                    in_pack[plane * 64 * 8 + (mm - 1) * 8 + offset] = a_matrix[mm - 1, kk - 1]
            for nn in range(1, 65):
                for kk in range(1, 65):
                    kpg, cpg = (nn - 1) // 16, (kk - 1) // 32
                    wt_idx = ((cpg * 32) * 16) + (kpg * 16 * wmma_meta["align_in"]) + ((kk - 1) % 32) + (((nn - 1) % 16) * 32)
                    wt_pack[wt_idx] = b_matrix[kk - 1, nn - 1]
        else:
            in_pack = in_pack.reshape(wmma_meta["m"], wmma_meta["align_in"])
            wt_pack = wt_pack.reshape(wmma_meta["align_out"], wmma_meta["align_in"])
            in_pack[:, :wmma_meta["k"]] = a_matrix
            wt_pack[:wmma_meta["n"], :wmma_meta["k"]] = b_matrix.T
            in_pack = in_pack.reshape(-1)
            wt_pack = wt_pack.reshape(-1)
        src = memoryview(bytearray(in_pack.tobytes()))
        src2 = memoryview(bytearray(wt_pack.tobytes()))
        self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task_buf")
        self.cmd_buf = self._gpu_alloc(self.cmd_buf_size, 0, name="cmd_buf")
        self.input_buf = self._gpu_alloc(src.nbytes, 0, name="input")
        self.weight_buf = self._gpu_alloc(src2.nbytes, 0, name="weight")
        out_stride = wmma_meta["align_out"] * 4
        out_nbytes = max(0x100, (wmma_meta["m"] - 1) * out_stride + wmma_meta["n"] * 4)
        self.output_buf = self._gpu_alloc(out_nbytes, 0, name="output")
        try:
            ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
            ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
            self._gpu_sync(self.input_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
            self._gpu_sync(self.weight_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
            self.q, self.lut_enable = [], False
            self.boilerplate(op=Ops.WMMA, size=int(m * k), arg=None,
                feature_addr=self.input_buf.meta.dma_addr, weight_addr=self.weight_buf.meta.dma_addr,
                dst_addr=self.output_buf.meta.dma_addr, wmma_meta=wmma_meta)
            self.submit(Ops.WMMA)
            self._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
            dst = memoryview(bytearray(self.output_buf.size))
            ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
            raw = np.frombuffer(dst.tobytes(), dtype=np.float32)
            out = np.empty((m, n), dtype=np.float32)
            if (wmma_meta["m"], wmma_meta["n"], wmma_meta["k"]) in {(64, 64, 64), (256, 256, 256)}:
                c2 = 4
                for col in range(n):
                    plane, offset = col // c2, col % c2
                    plane_base = plane * m * c2
                    for row in range(m):
                        out[row, col] = raw[plane_base + row * c2 + offset]
            else:
                stride = wmma_meta["align_out"]
                for row in range(m):
                    out[row, :] = raw[row * stride:row * stride + n]
            return out
        finally:
            self._gpu_free_multiple([self.task_buf, self.cmd_buf, self.input_buf,
                                     self.weight_buf, self.output_buf])

    def run_wmma(self, a_matrix: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
        self.reset_npu()
        self.q = []
        self.lut_enable = False
        return self._run_wmma_matmul(a_matrix, b_matrix)

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

            if self.lut_enable:
                raw = np.rint(np.array(result, dtype=np.float32))
                if op is Ops.EXP2:
                    result = ((raw.astype(np.uint16) / 2**14) - 1.0) / self.inv_scale
                elif arg == "silu":
                    result = raw.astype(np.int16) / (2**15 - 1) / self.inv_scale
            
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

        print("\n" + "=" * 60)
        print("Testing SUB operation...")
        print("=" * 60)
        result_sub = prog.run(Ops.SUB, None, [a, b])
        result_sub_arr = np.array(result_sub, dtype=np.float16)
        expected_sub = a - b

        print(f"  NPU Output: {result_sub_arr}")

        match_sub = np.allclose(result_sub_arr, expected_sub, atol=0.01)
        if match_sub:
            print("SUB TEST PASSED")
        else:
            print("SUB TEST FAILED")
            print(f"  Difference: {np.abs(result_sub_arr - expected_sub)}")

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
