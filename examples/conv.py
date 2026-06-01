import os, mmap, sys, ctypes, argparse
from fcntl import ioctl
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conv_expt import conv_tile_planner as planner  # noqa: E402


RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CONTIGUOUS = 1
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_BLOCK = 0 << 1
FP16_BYTES = 2
FP32_BYTES = 4
FP16_ATOM_ELEMENTS = 16
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_MAX_CONV_FLAT_STRIDE = 992
UNPACK_C2 = FP16_ATOM_ELEMENTS // FP16_BYTES
PC_CHAIN_TAIL_QWORDS = 4
EXACT11_BYK_SHAPE = "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid"
H40_SPATIAL_BY_Y_SHAPE = "b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid"
EXACT11_BYK_AMOUNTS = (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
EXACT11_BYK_MASKS = (0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
EXACT11_BYK_PC_AMOUNTS = (0, 0x1000e, 0, 0x1000e, 0, 0x2000e, 0, 0x2000e, 0, 0x2000e)
EXACT11_BYK_ROLES = ("setup_body", "k_half_body0", "aux0", "k_half_body1", "aux1", "k_tile_body0", "aux2", "k_tile_body1", "aux3", "k_tile_body2", "aux4")
EXACT11_BYK_TAIL_CLASSES = ("ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING", "ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING", "ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING", "ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING", "ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING")
EXACT11_BYK_TAIL_VALUES = ("0,0,0,13", "0,65550,0,13,0,0,0,0", "0,0,0,96,0,0", "0,65550,0,13,0,0,0,0", "0,0,0,96,0,0", "0,131086,0,13,0,0,0,0", "0,0,0,96,0,0", "0,131086,0,13,0,0,0,0", "0,0,0,96,0,0", "0,131086,0,13,0,0,0,0")
H40_EXACT17_AMOUNTS = (108, 108, 104, 104, 26, 104, 104, 26, 104, 104, 26, 104, 104, 26, 104, 104, 26)
H40_EXACT17_MASKS = (0x0d, 0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x0d, 0x60, 0x0d, 0x0d, 0x60, 0x0d, 0x0d, 0x60, 0x0d, 0x0d, 0x60)
H40_EXACT17_PC_AMOUNTS = (55, 0, 0x10035, 0x1000e, 0, 0x10035, 0x1000e, 0, 0x20035, 0x2000e, 0, 0x20035, 0x2000e, 0, 0x20035, 0x2000e)
H40_EXACT17_Y_WINDOWS = ((0, 23), (21, 19))
H160_SPATIAL_BY_Y_SHAPE = "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid"
H160_SPATIAL_BY_Y_ROLES = ("setup", "setup", "setup")
H160_SPATIAL_BY_Y_SETUP_WINDOWS = ((0, 64), (62, 64), (124, 36))
H160_SETUP3_AMOUNTS = (108, 108, 17)
H160_SETUP3_CORE_S_POINTER = (8140 << 17) | (1 << 16) | (243 << 6) | (1 << 4) | (1 << 2) | (1 << 1) | 1
H160_SETUP3_PC_BOOTSTRAP_VERSIONS = (0x000A0000, 0x00020000, 0x000A0000)
PREFIX_BY_Y_SHAPES = {
    "b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid",
    "b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid",
    "b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid",
}
SETUP2_CLOSURE_SHAPES = {"b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid"}
PREFIX_BY_K_SHAPES = {EXACT11_BYK_SHAPE, "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid"}
POINTWISE_EXACT11_BYK_SHAPES = {
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1",
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1",
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1",
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1",
}
POINTWISE_EXACT11_BYK_WINDOWS = {
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1": ((0, 10, 0x0a0), (10, 9, 0x090), (19, 9, 0x090)),
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1": ((0, 10, 0x0a0), (10, 9, 0x090), (19, 9, 0x090)),
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
}
POINTWISE_EXACT11_BYK_CBUF0 = {
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1": 0x057,
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1": 0x084,
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1": 0x057,
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1": 0x084,
}
POINTWISE_EXACT11_BYK_DMA2 = {
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1": 0x2a0,
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1": 0x08c,
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1": 0x2a0,
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1": 0x08c,
}
POINTWISE_YK_SHAPES = {"b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid"}
LOCAL_POINTWISE_YK_SHAPES = {
    "b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid",
    "conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1",
    "conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1",
    "conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1",
    "b1_c32_h112_w112_oc64_wic32_k1x1_g1",
    "b1_c64_h56_w56_oc128_wic64_k1x1_g1",
    "b1_c128_h56_w56_oc128_wic128_k1x1_g1",
    "b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid",
}
POINTWISE_CHAINED_Y_SHAPES = {"conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1"}
GROUPED_SERIAL_SHAPES = {
    "conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2",
    "b1_c4_h1_w1_oc4_wic2_k1x1_g2",
    "conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3",
    "conv2d_b1_c4_h5_w5_oc4_wic2_k3x3_g2",
    "conv2d_b1_c4_h5_w5_oc8_wic2_k3x3_g2",
    "conv2d_b1_c4_h5_w5_oc12_wic2_k3x3_g2",
    "conv2d_b1_c6_h5_w5_oc6_wic2_k3x3_g3",
    "conv2d_b1_c6_h5_w5_oc12_wic2_k3x3_g3",
    "conv2d_b1_c6_h5_w5_oc18_wic2_k3x3_g3",
    "conv2d_b1_c15_h5_w5_oc20_wic3_k3x3_g5",
    "conv2d_b1_c15_h5_w5_oc25_wic3_k3x3_g5",
    "conv2d_b1_c15_h5_w5_oc30_wic3_k3x3_g5",
    "conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5",
    "conv2d_b1_c15_h5_w5_oc40_wic3_k3x3_g5",
    "b1_c3_h1_w11_oc6_wic1_k1x5_g3",
    "b8_c3_h1_w11_oc6_wic1_k1x5_g3",
    "conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5",
}
LOCAL_TILE_REPLAY_SHAPES = POINTWISE_YK_SHAPES | LOCAL_POINTWISE_YK_SHAPES | {
    "conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1",
    "conv2d_b1_c96_h56_w56_oc32_wic96_k1x1_g1",
    "conv2d_b1_c96_h56_w56_oc16_wic96_k1x1_g1",
    "conv2d_b1_c64_h56_w56_oc24_wic64_k1x1_g1",
    "conv2d_b1_c128_h56_w56_oc24_wic128_k1x1_g1",
    "conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1",
    "conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1",
    "conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1",
    "conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1",
    "conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1",
    "b1_c96_h75_w75_oc24_wic96_k1x1_g1_s1_pvalid",
    "b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid",
    "b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid",
    "b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid",
    "b1_c96_h20_w20_oc12_wic96_k1x1_g1_s1_pvalid",
    "b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid",
    "b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid",
    "conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1",
    "b1_c3_h224_w224_oc32_wic3_k3x3_g1",
    "b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid",
}
KNOWN_BAD_SPATIAL_SETUP_SHAPES = {
    "b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid",
    "b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid",
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid",
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid",
}
class reg:
    CNA = 0x0201; CORE = 0x0801; DPU = 0x1001; PC = 0x0081; PC_REG = 0x0101; VERSION = 0x0041
    OPERATION_ENABLE = 0x0008; PC_BASE_ADDRESS = 0x0010; PC_REGISTER_AMOUNTS = 0x0014
    S_POINTER = 0x4004; FEATURE_MODE_CFG = 0x400c; DATA_FORMAT = 0x4010; DST_BASE_ADDR = 0x4020
    DST_SURF_STRIDE = 0x4024; DATA_CUBE_WIDTH = 0x4030; DATA_CUBE_HEIGHT = 0x4034; DATA_CUBE_NOTCH = 0x4038
    DATA_CUBE_CHANNEL = 0x403c; BS_CFG = 0x4040; BS_OW_CFG = 0x4050; WDMA_SIZE_0 = 0x4058; WDMA_SIZE_1 = 0x405c
    BN_CFG = 0x4060; EW_CFG = 0x4070; EW_CVT_SCALE_VALUE = 0x4078; OUT_CVT_SCALE = 0x4084; SURFACE_ADD = 0x40c0
    CNA_CONV_CON1 = 0x100c; CNA_CONV_CON2 = 0x1010; CNA_CONV_CON3 = 0x1014; CNA_DATA_SIZE0 = 0x1020
    CNA_DATA_SIZE1 = 0x1024; CNA_DATA_SIZE2 = 0x1028; CNA_DATA_SIZE3 = 0x102c; CNA_WEIGHT_SIZE0 = 0x1030
    CNA_WEIGHT_SIZE1 = 0x1034; CNA_WEIGHT_SIZE2 = 0x1038; CNA_CBUF_CON0 = 0x1040; CNA_CBUF_CON1 = 0x1044
    CNA_CVT_CON0 = 0x104c; CNA_CVT_CON1 = 0x1050; CNA_CVT_CON2 = 0x1054; CNA_CVT_CON3 = 0x1058
    CNA_CVT_CON4 = 0x105c; CNA_CVT_CON5 = 0x1180; CNA_FEATURE_DATA_ADDR = 0x1070; CNA_DMA_CON0 = 0x1078
    CNA_DMA_CON1 = 0x107c; CNA_DMA_CON2 = 0x1080; CNA_FC_DATA_SIZE0 = 0x1084; CNA_FC_DATA_SIZE1 = 0x1088
    CNA_DCOMP_ADDR0 = 0x1110; CORE_S_POINTER = 0x3004; CORE_MISC_CFG = 0x3010; CORE_DATAOUT_SIZE_0 = 0x3014
    CORE_DATAOUT_SIZE_1 = 0x3018; CORE_RESERVED_3030 = 0x3030

EXACT11_BODY_ZERO_KEYS = (
    (reg.CNA, 0x1060), (reg.CNA, 0x1064), (reg.CNA, 0x1068), (reg.CNA, 0x1074),
    (reg.CNA, 0x1100), (reg.CNA, 0x1104), *[(reg.CNA, addr) for addr in range(0x1140, 0x1180, 4)],
    (reg.CNA, 0x1184), (reg.CORE, 0x301c), (reg.DPU, 0x4014), (reg.DPU, 0x4044), (reg.DPU, 0x4048),
    (reg.DPU, 0x404c), (reg.DPU, 0x4054), (reg.DPU, 0x4064), (reg.DPU, 0x4068), (reg.DPU, 0x406c),
    (reg.DPU, 0x4074), (reg.DPU, 0x407c), (reg.DPU, 0x4080), (reg.DPU, 0x4088),
    *[(reg.DPU, addr) for addr in range(0x4090, 0x40b0, 4)], (reg.DPU, 0x40c4),
    *[(reg.DPU, addr) for addr in range(0x4100, 0x4130, 4)],
)

H160_SETUP3_FULL_ROW_OMIT_KEYS = ((reg.CNA, reg.CNA_CVT_CON5), (reg.CORE, reg.CORE_RESERVED_3030), (reg.DPU, reg.DATA_CUBE_NOTCH))
H160_SETUP3_SHORT_ROW_KEYS = (
    (reg.CNA, reg.CNA_CBUF_CON0), (reg.CNA, reg.CNA_CONV_CON1), (reg.DPU, reg.S_POINTER),
    (reg.CNA, reg.CNA_DATA_SIZE0), (reg.CNA, reg.CNA_DATA_SIZE3), (reg.CNA, reg.CNA_CBUF_CON0),
    (reg.CNA, reg.CNA_FEATURE_DATA_ADDR), (reg.CNA, reg.CNA_FC_DATA_SIZE0), (reg.CNA, reg.CNA_FC_DATA_SIZE1),
    (reg.CNA, reg.CNA_DCOMP_ADDR0), (reg.CORE, reg.CORE_DATAOUT_SIZE_0), (reg.DPU, reg.DST_BASE_ADDR),
    (reg.DPU, reg.DATA_CUBE_HEIGHT), (reg.DPU, reg.WDMA_SIZE_0), (reg.DPU, reg.WDMA_SIZE_1),
)

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


DRM_IOCTL_RKNPU_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create)); DRM_IOCTL_RKNPU_MEM_MAP = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_MEM_DESTROY = _IOWR('d', 0x44, ctypes.sizeof(rknpu_mem_destroy))
DRM_IOCTL_RKNPU_SUBMIT = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit)); DRM_IOCTL_RKNPU_ACTION = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))


LIST_SHAPES = "supported b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid\nsupported conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1\nsupported conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1\nsupported conv2d_b1_c96_h56_w56_oc32_wic96_k1x1_g1\nsupported conv2d_b1_c96_h56_w56_oc16_wic96_k1x1_g1\nsupported conv2d_b1_c64_h56_w56_oc24_wic64_k1x1_g1\nsupported conv2d_b1_c128_h56_w56_oc24_wic128_k1x1_g1\nsupported conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1\ntry-NONE  b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid\ntry-NONE  b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid\ntry-NONE  conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1\ndisabled  conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1\ndisabled  b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid".splitlines()

def _ceil_div(x, y): return (x + y - 1) // y

def _align_up(x, align): return _ceil_div(x, align) * align

def shape_from_name(name):
    core = name[7:] if name.startswith("conv2d_") else name
    fields = core.split("_")
    vals = {field[:3] if field.startswith("wic") else field[0]: field for field in fields}
    try:
        kh, kw = (int(x) for x in vals["k"][1:].split("x"))
        return dict(name=name, batch=int(vals["b"][1:]), in_c=int(vals["c"][1:]), in_h=int(vals["h"][1:]),
                    in_w=int(vals["w"][1:]), out_c=int(vals["o"][2:]), weight_in_c=int(vals["wic"][3:]),
                    kh=kh, kw=kw, groups=int(vals["g"][1:]), stride=int(vals.get("s", "s1")[1:]))
    except (KeyError, ValueError):
        raise ValueError("expected encoded shape like b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid")

def E(target, reg_addr, value): return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

def _pairs(target, items): return tuple(E(target, addr, value) for addr, value in items)

def _zero_range(start, end): return tuple((addr, 0) for addr in range(start, end + 1, 4))

def setup_full_reg_qwords():
    cna = (
        (0x1040, 0xb1), (0x1104, 0), (0x1100, 0), (0x100c, 0x120),
        (0x100c, 0x120), (0x1010, 0x110), (0x1014, 9), (0x1020, 0xe000e),
        (0x1024, 0x1f0020), (0x1028, 0xc), (0x102c, 0x90), (0x1030, 0x12000),
        (0x1034, 0x240), (0x1038, 0x3030080), (0x1040, 0xb1), (0x1044, 0xe),
        (0x104c, 0xb), (0x1050, 0x10000), (0x1054, 0x10000), (0x1058, 0x10000),
        (0x105c, 0x10000), (0x1060, 0), (0x1064, 0), (0x1068, 0), (0x1070, 0),
        (0x1074, 0), (0x1078, 0xf000f), (0x107c, 0x38), (0x1080, 0x8c),
        (0x1084, 0xe000e), (0x1088, 0x20), (0x1100, 0), (0x1104, 0), (0x1110, 0),
        *_zero_range(0x1140, 0x117c), (0x1180, 0), (0x1184, 0),
    )
    core = ((0x3010, 0x200), (0x3014, 0xb000b), (0x3018, 0x7f), (0x301c, 0), (0x3030, 0))
    dpu = (
        (0x400c, 0x1e4), (0x4010, 0x48000002), (0x4014, 0), (0x4020, 0),
        (0x4024, 0x900), (0x4030, 0xb), (0x4034, 0xb), (0x4038, 0),
        (0x403c, 0x7f007f), (0x4040, 0x53), (0x4044, 0), (0x4048, 0),
        (0x404c, 0), (0x4050, 0x126), (0x4054, 0), (0x4058, 0x7f),
        (0x405c, 0xb000b), (0x4060, 0x53), (0x4064, 0), (0x4068, 0),
        (0x406c, 0), (0x4070, 0x383), (0x4074, 0), (0x4078, 1), (0x407c, 0),
        (0x4080, 0), (0x4084, 0x10001), (0x4088, 0), *_zero_range(0x4090, 0x409c),
        *_zero_range(0x40a0, 0x40ac), (0x40c0, 0x1200), (0x40c4, 0), *_zero_range(0x4100, 0x412c),
    )
    return _pairs(reg.CNA, cna[:4]) + (E(reg.DPU, 0x4004, 0xe),) + _pairs(reg.CNA, cna[4:]) + _pairs(reg.CORE, core) + _pairs(reg.DPU, dpu)

def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c

def _conv_align_c(in_c, groups, out_c):
    if not _is_depthwise(in_c, out_c, groups) and (groups > 1 or in_c > 4):
        return 16
    return max(8, min(1 << (max(1, in_c) - 1).bit_length(), 32 if _is_depthwise(in_c, out_c, groups) else 16))

def _conv_input_pack_c2(in_c, groups, out_c, align_c):
    if in_c == 1:
        return 2
    if _is_depthwise(in_c, out_c, groups) or groups > 1 or in_c > 4:
        return 8
    return align_c

def _conv_params(s):
    in_c, in_h, in_w, out_c, kh, kw, groups = s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["kh"], s["kw"], s["groups"]
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    is_spatial = kh != 1 or kw != 1
    out_h, out_w = in_h - kh + 1, in_w - kw + 1
    align_c = _conv_align_c(in_c, groups, out_c)
    align_out_c = max(32 if groups == 1 and kh == 1 and kw == 1 and in_c >= 64 else 16, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_atoms = max(1, out_h * out_w)
    out_width_stride = out_atoms if not is_spatial else _align_up(out_atoms, 4)
    input_pack_c2 = _conv_input_pack_c2(in_c, groups, out_c, align_c)
    use_nhwc = not is_depthwise and not (groups > 1 and is_spatial) and in_c < input_pack_c2
    return dict(is_depthwise=is_depthwise, is_spatial=is_spatial, out_h=out_h, out_w=out_w, align_c=align_c,
                align_out_c=align_out_c, width_stride=width_stride, out_width_stride=out_width_stride,
                input_pack_c2=input_pack_c2, use_nhwc=use_nhwc)

def _dma_strides(in_h, width_stride, use_nhwc_pack):
    if use_nhwc_pack:
        return width_stride, width_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, width_stride * (in_h - 4) if in_h > 4 else 0

def _cbuf_entries(width_stride, align_c, in_h, is_depthwise):
    row_entries = max(1, _ceil_div(width_stride * align_c, 2 * FP16_ATOM_ELEMENTS))
    return row_entries if align_c >= 16 or is_depthwise else row_entries * in_h * 4

def _feature_grains(row_bytes, floor_grains, use_nhwc_pack=False, is_spatial=False, is_depthwise=False):
    if use_nhwc_pack and is_spatial:
        return floor_grains
    if is_depthwise and is_spatial:
        return min(13, floor_grains)
    even_rows_per_two_banks = (_ceil_div(2 * CBUF_BANK_SIZE, row_bytes) + 1) & ~1
    return min(floor_grains, even_rows_per_two_banks)

def _data_bank(width_stride, feature_grains, align_c, use_nhwc_pack=False, is_spatial=False, is_depthwise=False):
    if is_spatial and (use_nhwc_pack or is_depthwise):
        return RK_CBUF_BANKS - 1
    return int(np.clip(_ceil_div(width_stride * feature_grains * align_c * FP16_BYTES, CBUF_BANK_SIZE), 1, RK_CBUF_BANKS - 1))

def _is_pointwise_wide(s):
    return s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1 and s["in_c"] >= 64

def _pack_pointwise_wide(weight, out_c, in_c):
    aligned_out_c = max(32, _align_up(out_c, 16))
    aligned_in_c = in_c
    rows = np.zeros((aligned_out_c, aligned_in_c), dtype=np.float16)
    rows[:out_c, :in_c] = weight[:out_c, :in_c, 0, 0]
    return np.concatenate([rows[oc:oc + 16, ic:ic + 32].ravel()
                           for oc in range(0, aligned_out_c, 16) for ic in range(0, aligned_in_c, 32)])

def _pack_kh_major(weight, out_c, in_c, kh, kw, c2_out):
    aligned_in_c = c2_out * _ceil_div(in_c, c2_out)
    padded = np.zeros((out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:, :in_c] = weight
    if kh != 1 or kw != 1:
        return np.concatenate([padded[oc:oc + 16, ic:ic + 32, y, x].ravel()
                               for oc in range(0, out_c, 16) for ic in range(0, aligned_in_c, 32)
                               for y in range(kh) for x in range(kw)])
    return np.concatenate([padded[oc:oc + 16].transpose(2, 3, 0, 1).ravel() for oc in range(0, out_c, 16)])

def pack_weights(weight_full, s, p):
    if _is_pointwise_wide(s):
        return _pack_pointwise_wide(weight_full, s["out_c"], s["in_c"])
    return _pack_kh_major(weight_full, s["out_c"], s["in_c"], s["kh"], s["kw"], p["align_c"])

def pack_input(input_nchw, p):
    in_c, in_h, in_w = input_nchw.shape
    if p["use_nhwc"]:
        out = np.zeros((in_h, p["width_stride"], in_c), dtype=np.float16)
        out[:, :in_w] = input_nchw.transpose(1, 2, 0)
        return out.ravel()
    c2 = p["input_pack_c2"]
    c1 = _ceil_div(in_c, c2)
    padded = np.zeros((c1 * c2, in_h, p["width_stride"]), dtype=np.float16)
    padded[:in_c, :, :in_w] = input_nchw
    return padded.reshape(c1, c2, in_h, p["width_stride"]).transpose(0, 2, 3, 1).ravel()

def unpack_output(out_raw, out_c, out_h, out_w, out_width_stride, c2):
    c1 = out_raw.size // (out_width_stride * c2)
    packed = out_raw.reshape(1, c1, 1, out_width_stride, c2)
    return packed[0, :, 0, :out_h * out_w, :].transpose(0, 2, 1).reshape(c1 * c2, out_h * out_w)[:out_c].reshape(out_c, out_h, out_w)

def make_regs(s, p, in_dma, wt_dma, out_dma, out_fp16):
    if s["name"] == "b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid":
        return patch_regs(setup_full_reg_qwords(), {
            (reg.CNA, reg.CNA_FEATURE_DATA_ADDR): in_dma,
            (reg.CNA, reg.CNA_DCOMP_ADDR0): wt_dma,
            (reg.DPU, reg.DST_BASE_ADDR): out_dma,
        })
    in_c, in_h, in_w, out_c, kh, kw, groups = s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["kh"], s["kw"], s["groups"]
    align_c, align_out_c = p["align_c"], p["align_out_c"]
    out_h, out_w, is_spatial = p["out_h"], p["out_w"], p["is_spatial"]
    data_in_channel_aligned = _align_up(in_c, align_c)
    weight_bytes_per_kernel = kh * kw * data_in_channel_aligned * FP16_BYTES
    feature_grains = _feature_grains(p["width_stride"] * data_in_channel_aligned * FP16_BYTES, in_h + kh, p["use_nhwc"], is_spatial, False)
    data_bank = _data_bank(p["width_stride"], feature_grains, data_in_channel_aligned, p["use_nhwc"], is_spatial, False)
    out_precision = 2 if out_fp16 else 5
    size_e = 1 if out_fp16 else 3
    out_channel_field = align_out_c - 1
    cvt_con0 = 0x0b if is_spatial and not p["is_depthwise"] else 1
    cvt_con5 = ((1 << in_c) if p["use_nhwc"] else p["input_pack_c2"]) - 1
    conv_con1 = (2 << 4) | (2 << 7) | (((1 << 30) | (1 << 29) | ((7 + in_c) << 12)) if p["use_nhwc"] and in_c <= 4 else 0)
    regs = [
        E(reg.DPU, reg.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.CNA, reg.CNA_CONV_CON1, conv_con1),
        E(reg.CNA, reg.CNA_CONV_CON2, feature_grains << 4),
        E(reg.CNA, reg.CNA_CONV_CON3, (1 << 3) | 1),
        E(reg.CNA, reg.CNA_DATA_SIZE0, (p["width_stride"] << 16) | in_h),
        E(reg.CNA, reg.CNA_DATA_SIZE1, ((in_c - 1) << 16) | data_in_channel_aligned),
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_w * out_h),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_per_kernel * out_c),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2, (kw << 24) | (kh << 16) | out_c),
        E(reg.CNA, reg.CNA_CBUF_CON0, ((RK_CBUF_BANKS - data_bank) << 4) | data_bank),
        E(reg.CNA, reg.CNA_CBUF_CON1, _cbuf_entries(p["width_stride"], data_in_channel_aligned, in_h, False)),
        E(reg.CNA, reg.CNA_CVT_CON0, cvt_con0), E(reg.CNA, reg.CNA_CVT_CON1, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON2, 1 << 16), E(reg.CNA, reg.CNA_CVT_CON3, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON4, 1 << 16), E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        E(reg.CNA, reg.CNA_DMA_CON1, _dma_strides(in_h, p["width_stride"], p["use_nhwc"])[0]),
        E(reg.CNA, reg.CNA_DMA_CON2, _dma_strides(in_h, p["width_stride"], p["use_nhwc"])[1]),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, (in_w << 16) | in_h),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, data_in_channel_aligned), E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, cvt_con5),
        E(reg.CORE, reg.CORE_MISC_CFG, (2 << 8) | is_spatial), E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field), E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1)), E(reg.DPU, reg.DATA_FORMAT, (out_precision << 29) | (2 << 26) | 2),
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma), E(reg.DPU, reg.DST_SURF_STRIDE, p["out_width_stride"] << 4),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1), E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1), E(reg.DPU, reg.DATA_CUBE_NOTCH, 0),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL, ((out_c - 1) << 16) | out_channel_field),
        E(reg.DPU, reg.BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.BS_OW_CFG, (size_e << 8) | (size_e << 5) | (size_e << 2) | (1 << 1)),
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field), E(reg.DPU, reg.WDMA_SIZE_1, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.DPU, reg.BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1), E(reg.DPU, reg.EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1), E(reg.DPU, reg.OUT_CVT_SCALE, ((1 << 16) | 1) if out_fp16 else 0),
        E(reg.DPU, reg.SURFACE_ADD, (p["out_width_stride"] * max(2, align_out_c // 16)) << 4),
    ]
    return regs

def make_y_tile_regs(s, p, row, in_dma, wt_dma, out_dma, input_off):
    tile_shape = dict(s, in_h=row["input_h"])
    tile_p = _conv_params(tile_shape)
    regs = make_regs(tile_shape, tile_p, in_dma + input_off, wt_dma, out_dma + row["y_start"] * p["out_w"] * 16, True)
    data_bank = RK_CBUF_BANKS - 1
    patches = {
        (reg.CNA, reg.CNA_CBUF_CON0): (int(row["weight_reuse"]) << 13) | ((RK_CBUF_BANKS - data_bank) << 4) | data_bank,
        (reg.DPU, reg.DST_SURF_STRIDE): p["out_width_stride"] << 4,
        (reg.DPU, reg.SURFACE_ADD): (p["out_width_stride"] * max(2, p["align_out_c"] // 16)) << 4,
    }
    if s["name"] not in PREFIX_BY_Y_SHAPES and s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1 and s["in_c"] % 32:
        patches[(reg.CNA, reg.CNA_CONV_CON2)] = 9 << 4
    if s["name"] in SETUP2_CLOSURE_SHAPES:
        conv2 = (_first_qword(regs, (reg.CNA, reg.CNA_CONV_CON2)) >> 16) & 0xffffffff
        patches[(reg.CNA, reg.CNA_CONV_CON2)] = max(conv2, 52 << 4)
        patches[(reg.CNA, reg.CNA_CVT_CON5)] = 0
    return patch_regs(regs, patches)

def make_k_tile_regs(s, p, row, in_dma, wt_dma, out_dma):
    tile_shape = dict(s, out_c=row["oc_count"])
    tile_p = _conv_params(tile_shape)
    regs = make_regs(tile_shape, tile_p, in_dma + row["feature_off"], wt_dma + row["weight_off"],
                     out_dma + row["output_off"], True)
    conv2 = (_first_qword(regs, (reg.CNA, reg.CNA_CONV_CON2)) >> 16) & 0xffffffff
    patches = {
        (reg.CNA, reg.CNA_CONV_CON2): 0x50000000 | (conv2 & 0x3ff0),
        (reg.DPU, reg.SURFACE_ADD): (p["out_width_stride"] * 2) << 4,
    }
    if s["in_h"] == 14 and s["in_c"] == 160 and s["out_c"] == 320:
        patches[(reg.CNA, reg.CNA_CONV_CON2)] = 0x500000f0
        patches[(reg.CNA, reg.CNA_CBUF_CON0)] = 0xa2
    return patch_regs(regs, patches)

def make_yk_pointwise_regs(s, p, row, oc_start, oc_count, in_dma, wt_dma, out_dma):
    tile_shape = dict(s, in_h=row["input_h"], out_c=oc_count)
    tile_p = _conv_params(tile_shape)
    feature_off = row["y_start"] * p["width_stride"] * p["input_pack_c2"] * FP16_BYTES
    weight_off = oc_start * s["in_c"] * FP16_BYTES
    output_off = oc_start * p["out_width_stride"] * FP16_BYTES + row["y_start"] * p["out_w"] * UNPACK_C2 * FP16_BYTES
    regs = make_regs(tile_shape, tile_p, in_dma + feature_off, wt_dma + weight_off, out_dma + output_off, True)
    patches = {
        (reg.CNA, reg.CNA_CBUF_CON0): (int(row["weight_reuse"]) << 13) | 0x1b,
        (reg.DPU, reg.DST_SURF_STRIDE): p["out_width_stride"] << 4,
        (reg.DPU, reg.SURFACE_ADD): (p["out_width_stride"] * max(2, tile_p["align_out_c"] // 16)) << 4,
    }
    return patch_regs(regs, patches)

def make_local_k_tile_regs(s, p, in_dma, wt_dma, out_dma):
    regs = make_regs(s, p, in_dma, wt_dma, out_dma, True)
    conv2 = 0x500000a0 if s["in_h"] == 7 else 0x500000f0
    patches = {(reg.CNA, reg.CNA_CONV_CON2): conv2}
    if s["in_h"] == 14:
        patches[(reg.CNA, reg.CNA_CBUF_CON0)] = 0xa2
    return patch_regs(regs, patches)

def patch_regs(regs, values):
    patched = []
    for qword in regs:
        key = (qword >> 48, qword & 0xffff)
        patched.append(E(key[0], key[1], values[key]) if key in values else qword)
    return patched

def mem_allocate(fd, size, flags=0):
    mem_create = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mem_create)
    mem_map = rknpu_mem_map(handle=mem_create.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mem_map)
    return mmap.mmap(fd, mem_create.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mem_map.offset), mem_create

def mem_destroy(fd, mem_create):
    return ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY,
                 rknpu_mem_destroy(handle=mem_create.handle, obj_addr=mem_create.obj_addr))

def close_allocations(fd, allocations):
    for mapped, mem_create in reversed(allocations):
        try:
            mapped.close()
        except BufferError:
            pass
        finally:
            mem_destroy(fd, mem_create)

def write_tasks(task_map, regcmd_map, regcmd_mem, task_regs):
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += _align_up(len(regs) + PC_CHAIN_TAIL_QWORDS, 2)
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            next_addr = regcmd_mem.dma_addr + offsets[idx + 1] * ctypes.sizeof(ctypes.c_uint64)
            tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr & 0xfffffff0),
                    E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, _ceil_div(len(task_regs[idx + 1]), 2) + 1),
                    E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
        else:
            tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, 0), E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                    E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
        for i, qword in enumerate(tail):
            regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = 0xd
        tasks[idx].int_mask = 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = len(regs)
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)

def _exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma, y_start=0, input_h=None, conv2_low=None):
    p = _conv_params(s)
    input_h = s["in_h"] if input_h is None else input_h
    tile_shape = dict(s, in_h=input_h, out_c=oc_count)
    tile_p = _conv_params(tile_shape)
    feature_off = y_start * p["width_stride"] * p["input_pack_c2"] * FP16_BYTES
    weight_off = oc_start * s["kh"] * s["kw"] * s["in_c"] * FP16_BYTES
    output_off = oc_start * p["out_width_stride"] * FP16_BYTES + y_start * p["out_w"] * UNPACK_C2 * FP16_BYTES
    regs = make_regs(tile_shape, tile_p, in_dma + feature_off, wt_dma + weight_off, out_dma + output_off, True)
    family_bits = {"setup": 0, "y_tile": 0x20000000, "k_half": 0x40000000, "k_tile": 0x50000000}[family]
    if conv2_low is None:
        conv2_low = 0x0c0 if s["name"] == H40_SPATIAL_BY_Y_SHAPE else (0x0a0 if s["in_h"] == 7 else 0x0f0)
    cbuf0 = 0x039 if s["name"] == H40_SPATIAL_BY_Y_SHAPE else 0x0a2
    patches = {
        (reg.CNA, reg.CNA_CONV_CON2): family_bits | conv2_low,
        (reg.CNA, reg.CNA_DATA_SIZE1): 0x1f00a0,
        (reg.CNA, reg.CNA_CBUF_CON0): cbuf0,
        (reg.CNA, reg.CNA_CVT_CON5): 0,
        (reg.CORE, reg.CORE_MISC_CFG): 0x200,
        (reg.DPU, reg.DST_SURF_STRIDE): p["out_width_stride"] << 4,
        (reg.DPU, reg.SURFACE_ADD): (p["out_width_stride"] * 2) << 4,
    }
    if s["name"] == H40_SPATIAL_BY_Y_SHAPE:
        patches[(reg.CNA, reg.CNA_DMA_CON2)] = 0x5a0
    values = {(qword >> 48, qword & 0xffff): (qword >> 16) & 0xffffffff for qword in patch_regs(regs, patches)}
    values.update({key: 0 for key in EXACT11_BODY_ZERO_KEYS if key not in values})
    cna_order = [reg.CNA_CONV_CON1, reg.CNA_CONV_CON2, reg.CNA_CONV_CON3, reg.CNA_DATA_SIZE0, reg.CNA_DATA_SIZE1,
                 reg.CNA_DATA_SIZE2, reg.CNA_DATA_SIZE3, reg.CNA_WEIGHT_SIZE0, reg.CNA_WEIGHT_SIZE1, reg.CNA_WEIGHT_SIZE2,
                 reg.CNA_CBUF_CON0, reg.CNA_CBUF_CON1, reg.CNA_CVT_CON0, reg.CNA_CVT_CON1, reg.CNA_CVT_CON2,
                 reg.CNA_CVT_CON3, reg.CNA_CVT_CON4, 0x1060, 0x1064, 0x1068, reg.CNA_FEATURE_DATA_ADDR, 0x1074,
                 reg.CNA_DMA_CON0, reg.CNA_DMA_CON1, reg.CNA_DMA_CON2, reg.CNA_FC_DATA_SIZE0, reg.CNA_FC_DATA_SIZE1,
                 0x1100, 0x1104, reg.CNA_DCOMP_ADDR0, *range(0x1140, 0x1180, 4), reg.CNA_CVT_CON5, 0x1184]
    dpu_order = [reg.FEATURE_MODE_CFG, reg.DATA_FORMAT, 0x4014, reg.DST_BASE_ADDR, reg.DST_SURF_STRIDE,
                 reg.DATA_CUBE_WIDTH, reg.DATA_CUBE_HEIGHT, reg.DATA_CUBE_NOTCH, reg.DATA_CUBE_CHANNEL, reg.BS_CFG,
                 0x4044, 0x4048, 0x404c, reg.BS_OW_CFG, 0x4054, reg.WDMA_SIZE_0, reg.WDMA_SIZE_1, reg.BN_CFG,
                 0x4064, 0x4068, 0x406c, reg.EW_CFG, 0x4074, reg.EW_CVT_SCALE_VALUE, 0x407c, 0x4080,
                 reg.OUT_CVT_SCALE, 0x4088, *range(0x4090, 0x40b0, 4), reg.SURFACE_ADD, 0x40c4, *range(0x4100, 0x4130, 4)]
    key_order = [(reg.DPU, reg.S_POINTER), *[(reg.CNA, addr) for addr in cna_order],
                 (reg.CORE, reg.CORE_MISC_CFG), (reg.CORE, reg.CORE_DATAOUT_SIZE_0),
                 (reg.CORE, reg.CORE_DATAOUT_SIZE_1), (reg.CORE, 0x301c), (reg.CORE, reg.CORE_RESERVED_3030),
                 *[(reg.DPU, addr) for addr in dpu_order]]
    ordered = [E(target, addr, values[(target, addr)]) for target, addr in key_order]
    if family == "setup":
        prelude = (E(reg.CNA, reg.CNA_CBUF_CON0, cbuf0), E(reg.CNA, 0x1104, 0),
                   E(reg.CNA, 0x1100, 0), E(reg.CNA, reg.CNA_CONV_CON1, 0x120))
        return list(prelude) + ordered
    return ordered

def _exact11_aux_regs(s, out_dma, aux_dma=None):
    channel_grain = min(_align_up(s["out_c"], 16), 32) - 1
    aux_dma = out_dma + 0x12e000 if aux_dma is None else aux_dma
    return [
        E(0x4001, 0x6004, 0xe), E(0x8001, 0x7004, 0xe), E(0x4001, 0x600c, 0), E(0x4001, 0x6010, 0),
        E(0x4001, 0x6014, channel_grain), E(0x4001, 0x6018, 0), E(0x4001, 0x601c, 0),
        E(0x4001, 0x6020, channel_grain), E(0x4001, 0x6024, 0x11), E(0x4001, 0x6034, 0),
        E(0x4001, 0x6038, 0), E(0x4001, 0x603c, 0), E(0x4001, 0x6040, 0),
        E(0x4001, 0x6044, 0), E(0x4001, 0x6048, 0), E(0x4001, 0x6070, aux_dma),
        E(0x4001, 0x607c, 0x10), E(0x4001, 0x6084, 0x10), E(0x4001, 0x60dc, 0x3),
        E(0x8001, 0x700c, 0), E(0x8001, 0x7010, 0), E(0x8001, 0x7014, channel_grain), E(0x8001, 0x701c, aux_dma + 0x400),
        E(0x8001, 0x7024, 0x10), E(0x8001, 0x7028, 0x10), E(0x8001, 0x7030, 1),
    ]

def _exact11_task_regs(s, in_dma, wt_dma, out_dma):
    half = _align_up(s["out_c"] // 2, 16)
    rows = [_exact11_body_regs(s, "setup", 0, s["out_c"], in_dma, wt_dma, out_dma),
            _exact11_body_regs(s, "k_half", 0, half, in_dma, wt_dma, out_dma), _exact11_aux_regs(s, out_dma),
            _exact11_body_regs(s, "k_half", half, s["out_c"] - half, in_dma, wt_dma, out_dma), _exact11_aux_regs(s, out_dma)]
    for oc_start, oc_count in ((0, 112), (112, 112), (224, 96)):
        rows.append(_exact11_body_regs(s, "k_tile", oc_start, oc_count, in_dma, wt_dma, out_dma))
        if oc_start != 224:
            rows.append(_exact11_aux_regs(s, out_dma))
    rows.append(_exact11_aux_regs(s, out_dma))
    if tuple(len(row) for row in rows) != EXACT11_BYK_AMOUNTS:
        raise RuntimeError("exact11 BY_K row amounts changed")
    return rows

def _pointwise_exact11_byk_task_regs(s, in_dma, wt_dma, out_dma):
    if s["name"] not in POINTWISE_EXACT11_BYK_SHAPES:
        raise ValueError("pointwise exact11 BY_K rows are scoped only to RKNN-prefix proven shapes")
    half = _align_up(s["out_c"] // 2, 16)
    cbuf0 = POINTWISE_EXACT11_BYK_CBUF0[s["name"]]
    full_dma2 = POINTWISE_EXACT11_BYK_DMA2[s["name"]]
    aux_dma = wt_dma + _align_up(s["out_c"], 16) * _align_up(s["in_c"], 16) * FP16_BYTES
    def pointwise_row(family, oc_start, oc_count, y_start=0, input_h=None, conv2_low=0x0a0):
        return patch_regs(_exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma,
                                             y_start, input_h, conv2_low),
                          {(reg.CNA, reg.CNA_CBUF_CON0): cbuf0, (reg.CNA, reg.CNA_DATA_SIZE1): ((s["in_c"] // 2 - 1) << 16) | s["in_c"],
                           (reg.CNA, reg.CNA_CVT_CON0): 0xb, (reg.CNA, reg.CNA_DMA_CON2): full_dma2})
    rows = [pointwise_row("setup", 0, s["out_c"]),
            pointwise_row("k_half", 0, half), _exact11_aux_regs(s, out_dma, aux_dma),
            pointwise_row("k_half", half, s["out_c"] - half), _exact11_aux_regs(s, out_dma, aux_dma)]
    for y_start, input_h, conv2_low in POINTWISE_EXACT11_BYK_WINDOWS[s["name"]]:
        rows.append(pointwise_row("y_tile", 0, s["out_c"], y_start, input_h, conv2_low))
        if y_start != POINTWISE_EXACT11_BYK_WINDOWS[s["name"]][-1][0]:
            rows.append(_exact11_aux_regs(s, out_dma, aux_dma))
    rows.append(_exact11_aux_regs(s, out_dma, aux_dma))
    if tuple(len(row) for row in rows) != EXACT11_BYK_AMOUNTS:
        raise RuntimeError("pointwise exact11 BY_K row amounts changed")
    return rows

def write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs):
    offsets = (0, 112, 224, 256, 368, 400, 512, 544, 656, 688, 800)
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            classes = EXACT11_BYK_TAIL_CLASSES[idx].split(",")
            values = [int(value) for value in EXACT11_BYK_TAIL_VALUES[idx].split(",")]
            for i, (cls, value) in enumerate(zip(classes, values)):
                if cls == "ZERO_PADDING":
                    qword = 0
                elif cls == "PC_BASE_ADDRESS":
                    qword = E(reg.PC_REG, reg.PC_BASE_ADDRESS, (regcmd_mem.dma_addr + offsets[idx + 1] * 8) & 0xfffffff0)
                elif cls == "PC_REGISTER_AMOUNTS":
                    qword = E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, value)
                elif cls == "VERSION":
                    qword = E(reg.VERSION, 0, value)
                else:
                    qword = E(reg.PC, reg.OPERATION_ENABLE, value)
                regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = EXACT11_BYK_MASKS[idx]
        tasks[idx].int_mask = 0xc00 if tasks[idx].enable_mask == 0x60 else 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = len(regs)
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * 8

def _h40_exact17_task_regs(s, in_dma, wt_dma, out_dma):
    if s["name"] != H40_SPATIAL_BY_Y_SHAPE:
        raise ValueError("h40 exact17 is scoped only to the prefix-proven spatial h40 shape")
    half = _align_up(s["out_c"] // 2, 16)
    groups = [("setup", 0, s["out_c"]), ("k_half", 0, half), ("k_half", half, s["out_c"] - half),
              ("k_tile", 0, 112), ("k_tile", 112, 112), ("k_tile", 224, 96)]
    rows = []
    for group_idx, (family, oc_start, oc_count) in enumerate(groups):
        for y_start, input_h in H40_EXACT17_Y_WINDOWS:
            rows.append(_exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma, y_start, input_h))
        if group_idx >= 1:
            rows.append(_exact11_aux_regs(s, out_dma))
    if tuple(len(row) for row in rows) != H40_EXACT17_AMOUNTS:
        raise RuntimeError("h40 exact17 row amounts changed")
    return rows

def write_h40_exact17_tasks(task_map, regcmd_map, regcmd_mem, task_regs):
    offsets, _bytes = regcmd_layout_from_amounts(H40_EXACT17_AMOUNTS)
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            next_addr = (regcmd_mem.dma_addr + offsets[idx + 1] * 8) & 0xfffffff0
            tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr), E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, H40_EXACT17_PC_AMOUNTS[idx]),
                    E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, H40_EXACT17_MASKS[idx])]
            if H40_EXACT17_MASKS[idx] == 0x60:
                tail = [0, E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0), E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, 0x60), 0, 0]
            elif idx in {1}:
                tail = [0, E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0), E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, 0x0d)]
            elif idx >= 2:
                tail += [0, 0, 0, 0]
            for i, qword in enumerate(tail):
                regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = H40_EXACT17_MASKS[idx]
        tasks[idx].int_mask = 0xc00 if tasks[idx].enable_mask == 0x60 else 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = len(regs)
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * 8

def _first_qword(regs, key):
    for qword in regs:
        if (qword >> 48, qword & 0xffff) == key:
            return qword
    raise KeyError(key)

def h160_setup3_layout():
    offsets, qoff = [], 0
    for amount in H160_SETUP3_AMOUNTS:
        offsets.append(qoff)
        qoff += _align_up(amount + PC_CHAIN_TAIL_QWORDS, 8)
    pc_amounts = tuple(0 if idx + 1 == len(H160_SETUP3_AMOUNTS) else _ceil_div(H160_SETUP3_AMOUNTS[idx + 1], 2) + 1
                       for idx in range(len(H160_SETUP3_AMOUNTS)))
    return tuple(offsets), pc_amounts

def make_h160_setup3_reg_rows(task_regs, regcmd_dma):
    offsets, pc_amounts = h160_setup3_layout()
    full_rows = []
    for regs in task_regs[:2]:
        full_rows.append([qword for qword in regs if (qword >> 48, qword & 0xffff) not in H160_SETUP3_FULL_ROW_OMIT_KEYS])
    row0_prefix = [E(reg.CORE, reg.CORE_S_POINTER, H160_SETUP3_CORE_S_POINTER)]
    row0_prefix += [E(reg.VERSION, 0, value) for value in H160_SETUP3_PC_BOOTSTRAP_VERSIONS]
    row0_prefix += [_first_qword(task_regs[0], (reg.CNA, reg.CNA_CBUF_CON0)), _first_qword(task_regs[0], (reg.CNA, reg.CNA_CONV_CON1))]
    row1_prefix = [_first_qword(task_regs[1], (reg.CNA, reg.CNA_CBUF_CON0)), _first_qword(task_regs[1], (reg.CNA, reg.CNA_CONV_CON1))]
    row0_tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, (regcmd_dma + offsets[1] * 8) & 0xfffffff0),
                 E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, pc_amounts[0]), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
    row1_tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, (regcmd_dma + offsets[2] * 8) & 0xfffffff0),
                 E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, pc_amounts[1]), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
    row2 = [_first_qword(task_regs[2], key) for key in H160_SETUP3_SHORT_ROW_KEYS] + [E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
    return tuple(tuple(row) for row in (row0_prefix + full_rows[0] + row0_tail, row1_prefix + full_rows[1] + row1_tail, row2))

def setup2_closure_layout():
    return (0, _align_up(108 + PC_CHAIN_TAIL_QWORDS, 8)), (55, 0)

def make_setup2_closure_reg_rows(task_regs, regcmd_dma):
    offsets, pc_amounts = setup2_closure_layout()
    full_rows = [[qword for qword in regs if (qword >> 48, qword & 0xffff) not in H160_SETUP3_FULL_ROW_OMIT_KEYS]
                 for regs in task_regs]
    row0_prefix = [E(reg.CORE, reg.CORE_S_POINTER, H160_SETUP3_CORE_S_POINTER)]
    row0_prefix += [E(reg.VERSION, 0, value) for value in H160_SETUP3_PC_BOOTSTRAP_VERSIONS]
    row0_prefix += [_first_qword(task_regs[0], (reg.CNA, reg.CNA_CBUF_CON0)), _first_qword(task_regs[0], (reg.CNA, reg.CNA_CONV_CON1))]
    row1_prefix = [_first_qword(task_regs[1], (reg.CNA, reg.CNA_CBUF_CON0)), _first_qword(task_regs[1], (reg.CNA, reg.CNA_CONV_CON1))]
    row0_tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, (regcmd_dma + offsets[1] * 8) & 0xfffffff0),
                 E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, pc_amounts[0]), E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
    row1_tail = [E(reg.PC_REG, reg.PC_BASE_ADDRESS, 0), E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0),
                 E(reg.PC, reg.OPERATION_ENABLE, (6 << 1) | 1)]
    return tuple(tuple(row) for row in (row0_prefix + full_rows[0] + row0_tail, row1_prefix + full_rows[1] + row1_tail))

def patch_h160_setup3_regs(task_regs):
    patched = []
    for idx, regs in enumerate(task_regs):
        patches = {
            (reg.CNA, reg.CNA_CBUF_CON0): (0x2000 if idx else 0) | 0x2a,
            (reg.CNA, reg.CNA_CONV_CON2): 15 << 4,
            (reg.CNA, reg.CNA_DMA_CON2): 0x6180,
            (reg.CORE, reg.CORE_MISC_CFG): 2 << 8,
            (reg.DPU, reg.SURFACE_ADD): 0xc3080,
        }
        patched.append(patch_regs(regs, patches))
    return tuple(patched)

def write_h160_setup3_tasks(task_map, regcmd_map, regcmd_mem, task_regs):
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    task_regs = patch_h160_setup3_regs(task_regs)
    rows = make_h160_setup3_reg_rows(task_regs, regcmd_mem.dma_addr)
    offsets, pc_amounts = h160_setup3_layout()
    if tuple(len(row) for row in rows) != (53, 49, 16) or pc_amounts != (55, 10, 0):
        raise RuntimeError("h160 setup3 closure generation failed preflight")
    for idx, row in enumerate(rows):
        for qword_idx, qword in enumerate(row):
            regcmd[offsets[idx] + qword_idx] = qword
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = 0x0d
        tasks[idx].int_mask = 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = H160_SETUP3_AMOUNTS[idx]
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + offsets[idx] * 8
    spans = tuple((offsets[idx + 1] - offsets[idx]) * 8 for idx in range(len(offsets) - 1)) + (_align_up(H160_SETUP3_AMOUNTS[-1] + PC_CHAIN_TAIL_QWORDS, 8) * 8,)
    return rows, offsets, spans, pc_amounts

def write_setup2_closure_tasks(task_map, regcmd_map, regcmd_mem, task_regs):
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    rows = make_setup2_closure_reg_rows(task_regs, regcmd_mem.dma_addr)
    offsets, pc_amounts = setup2_closure_layout()
    if tuple(len(row) for row in rows) != (53, 49) or pc_amounts != (55, 0):
        raise RuntimeError("setup2 closure generation failed preflight")
    for idx, row in enumerate(rows):
        for qword_idx, qword in enumerate(row):
            regcmd[offsets[idx] + qword_idx] = qword
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = 0x0d
        tasks[idx].int_mask = 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].int_status = 0x200 if idx == 1 else 0
        tasks[idx].regcfg_amount = 108
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + offsets[idx] * 8
    return rows, offsets, pc_amounts

def npu_submit(fd, task_obj_addr, task_count, core_mask=1, subcores=None):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))
    submit = rknpu_submit(flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK, timeout=6000, task_start=0, task_number=task_count,
                          task_counter=0, priority=0, task_obj_addr=task_obj_addr, iommu_domain_id=0,
                          reserved=0, task_base_addr=0, hw_elapse_time=0, core_mask=core_mask, fence_fd=-1)
    if subcores is None:
        subcores = ((0, task_count), (task_count, 0), (task_count, 0), (0, 0), (0, 0))
    for idx, (task_start, task_number) in enumerate(subcores):
        submit.subcore_task[idx] = rknpu_subcore_task(task_start=task_start, task_number=task_number)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit)

def post_submit_reset(fd):
    for reset_flag in (RKNPU_ACT_RESET, 6, RKNPU_ACT_RESET):
        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=reset_flag, value=0))

def validate_phase_a_shape(s):
    rows = planner.descriptor_rows_for_shape(s)
    families = {row["family"] for row in rows}
    if len(rows) == 1 and families == {"setup"} and rows[0]["split_method"] == "NONE":
        if s["name"] in KNOWN_BAD_SPATIAL_SETUP_SHAPES:
            raise ValueError("spatial setup/NONE path is known numerically wrong for this shape; no allocation or submit attempted")
        if _is_pointwise_wide(s):
            raise ValueError("pointwise-wide NONE needs RKNN 108-row closure; no allocation or submit attempted")
        return rows
    if rows and rows[0]["split_method"] == "BY_K":
        raise ValueError("BY_K/k_tile needs RKNN 108/104/26-row closure; no allocation or submit attempted")
    if rows and rows[0]["split_method"] == "BY_Y":
        if _is_pointwise_wide(s) and s["name"] not in LOCAL_TILE_REPLAY_SHAPES and s["name"] not in POINTWISE_CHAINED_Y_SHAPES:
            raise ValueError("pointwise-wide BY_Y needs proven row closure; no allocation or submit attempted")
        if (s["name"] not in PREFIX_BY_Y_SHAPES and len(families) == 1 and families == {"y_tile"} and
                s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1):
            return rows
        if s["name"] in SETUP2_CLOSURE_SHAPES:
            return rows
        raise ValueError("BY_Y/y_tile needs RKNN 108-row closure except proven pointwise row tiling; no allocation or submit attempted")
    if rows and rows[0]["split_method"] == "BY_YK":
        raise ValueError("BY_YK is disabled before allocation; mixed Y/K setup and k_half semantics are unresolved")
    else:
        raise ValueError(f"Phase A supports only one setup/NONE descriptor; got split={rows[0]['split_method'] if rows else 'none'} families={sorted(families)} rows={len(rows)}")

def compute_expected(inp, wt, s):
    out_h, out_w = s["in_h"] - s["kh"] + 1, s["in_w"] - s["kw"] + 1
    expected = np.zeros((s["batch"], s["out_c"], out_h, out_w))
    i64, w64 = inp.astype(np.float64), wt.astype(np.float64)
    for n in range(s["batch"]):
        for g in range(s["groups"]):
            ic0 = g * s["in_c"] // s["groups"]
            ic1 = (g + 1) * s["in_c"] // s["groups"]
            oc0 = g * s["out_c"] // s["groups"]
            oc1 = (g + 1) * s["out_c"] // s["groups"]
            for oc in range(oc0, oc1):
                for ic in range(ic0, ic1):
                    for i in range(s["kh"]):
                        for j in range(s["kw"]):
                            expected[n, oc] += i64[n, ic, i:i + out_h, j:j + out_w] * w64[oc, ic - ic0, i, j]
    return expected

def regcmd_layout_from_amounts(amounts):
    offsets, qoff = [], 0
    for amount in amounts:
        offsets.append(qoff)
        qoff += _align_up(amount + PC_CHAIN_TAIL_QWORDS, 8)
    return tuple(offsets), (offsets[-1] + amounts[-1]) * 8
def regcmd_alloc_bytes(required): return max(4096, _align_up(required, 4096))

def dry_run_exact11_byk(s):
    rows = planner.descriptor_rows_for_shape(s)
    if ((s["name"] not in {EXACT11_BYK_SHAPE, *POINTWISE_EXACT11_BYK_SHAPES}) or
            rows[0]["split_method"] != "BY_K"):
        raise ValueError("exact11 BY_K dry-run is scoped only to RKNN-prefix proven shapes")
    p = _conv_params(s)
    hw_out_fp16 = p["is_spatial"] or s["out_c"] >= 128 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE or _is_pointwise_wide(s)
    c2 = UNPACK_C2 if hw_out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
    input_bytes = _ceil_div(s["in_c"], p["input_pack_c2"]) * p["input_pack_c2"] * s["in_h"] * p["width_stride"] * FP16_BYTES
    weight_bytes = _align_up(s["out_c"], 16) * _align_up(s["in_c"], p["align_c"]) * s["kh"] * s["kw"] * FP16_BYTES
    output_bytes = _ceil_div(p["align_out_c"], c2) * p["out_width_stride"] * c2 * (FP16_BYTES if hw_out_fp16 else FP32_BYTES)
    offsets, regcmd_bytes = regcmd_layout_from_amounts(EXACT11_BYK_AMOUNTS)
    print(f"dry_run=exact11_byk shape={s['name']} tasks={len(EXACT11_BYK_AMOUNTS)} regcmd_bytes={regcmd_bytes} regcmd_alloc={regcmd_alloc_bytes(regcmd_bytes)}")
    print("amounts=" + ",".join(str(v) for v in EXACT11_BYK_AMOUNTS))
    print("masks=" + ",".join(hex(v) for v in EXACT11_BYK_MASKS))
    print("offsets=" + ",".join(str(v) for v in offsets))
    print("pc_amounts=" + ",".join(hex(v) for v in EXACT11_BYK_PC_AMOUNTS))
    if s["name"] in POINTWISE_EXACT11_BYK_SHAPES:
        roles = ("setup_body", "k_half_body0", "aux0", "k_half_body1", "aux1", "y_tile_body0", "aux2", "y_tile_body1", "aux3", "y_tile_body2", "aux4")
        print("roles=" + ",".join(roles))
        print("y_windows=" + ",".join(f"{y}+{h}:0x{conv2:x}" for y, h, conv2 in POINTWISE_EXACT11_BYK_WINDOWS[s["name"]]))
    else:
        print("roles=" + ",".join(EXACT11_BYK_ROLES))
    print("tail_classes=" + "|".join(EXACT11_BYK_TAIL_CLASSES))
    print("tail_values=" + "|".join(EXACT11_BYK_TAIL_VALUES))
    print(f"bo_bytes=task:{len(EXACT11_BYK_AMOUNTS) * ctypes.sizeof(struct_rknpu_task)},regcmd:{regcmd_bytes},"
          f"input:{input_bytes},weight:{weight_bytes},output:{output_bytes}")
    print("status=no_drm_no_submit")

def _h160_spatial_by_y_rows(s, p):
    feature_row_bytes = p["width_stride"] * p["input_pack_c2"] * FP16_BYTES
    output_row_bytes = p["out_w"] * UNPACK_C2 * FP16_BYTES
    for y_start, input_h in H160_SPATIAL_BY_Y_SETUP_WINDOWS:
        output_h = input_h - s["kh"] + 1
        yield dict(family="setup", y_start=y_start, input_h=input_h, output_h=output_h, oc_start=0, oc_count=s["out_c"],
                   feature_off=y_start * feature_row_bytes, weight_off=0,
                   output_off=y_start * output_row_bytes, dataout_atomics=output_h * p["out_w"],
                   core_size0=((output_h - 1) << 16) | (p["out_w"] - 1))

def dry_run_h160_spatial_by_y(s):
    if s["name"] != H160_SPATIAL_BY_Y_SHAPE:
        raise ValueError("h160 spatial BY_Y dry-run is scoped only to b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid")
    p = _conv_params(s)
    rows = tuple(_h160_spatial_by_y_rows(s, p))
    print(f"dry_run=h160_spatial_by_y shape={s['name']} rows={len(rows)} status=no_drm_no_submit")
    print("roles=" + ",".join(row["family"] for row in rows))
    print("input_h=" + ",".join(str(row["input_h"]) for row in rows))
    print("output_h=" + ",".join(str(row["output_h"]) for row in rows))
    for idx, row in enumerate(rows):
        print(f"row={idx:02d} family={row['family']} y={row['y_start']} oc={row['oc_start']}+{row['oc_count']} "
              f"feature_off=0x{row['feature_off']:x} weight_off=0x{row['weight_off']:x} output_off=0x{row['output_off']:x} "
              f"dataout_atomics={row['dataout_atomics']} core_size0=0x{row['core_size0']:x}")

def compute_expected_vectorized(inp, wt, s):
    windows = np.lib.stride_tricks.sliding_window_view(inp[0], (s["kh"], s["kw"]), axis=(1, 2))
    return np.einsum("cyxij,ocij->oyx", windows.astype(np.float32), wt.astype(np.float32), optimize=True)[None]

def run_h160_setup3_shape(s):
    if s["name"] != H160_SPATIAL_BY_Y_SHAPE:
        raise ValueError("--allow-h160-setup3-submit is scoped only to b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid")
    p = _conv_params(s)
    rows = tuple(dict(row, weight_reuse=idx > 0) for idx, row in enumerate(_h160_spatial_by_y_rows(s, p)))
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = pack_weights(wt, s, p).view(np.uint16)
    c2 = UNPACK_C2
    out_count = _ceil_div(p["align_out_c"], c2) * p["out_width_stride"] * c2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    output_flags = RKNPU_MEM_NON_CONTIGUOUS if output_bytes > 4 * 1024 * 1024 else RKNPU_MEM_NON_CACHEABLE
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), output_flags)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = [make_y_tile_regs(s, p, row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, row["feature_off"]) for row in rows]
        emitted_rows, offsets, spans, pc_amounts = write_h160_setup3_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        subcores = ((0, 3), (0, 0), (0, 0), (0, 0), (0, 0))
        print(f"guarded_h160_setup3_submit tasks=3 amounts={';'.join(str(v) for v in H160_SETUP3_AMOUNTS)} "
              f"spans={';'.join(hex(v) for v in spans)} pc_amounts={';'.join(str(v) for v in pc_amounts)} "
              f"rows={';'.join(str(len(row)) for row in emitted_rows)} offsets={';'.join(str(v) for v in offsets)} "
              "subcores=(0,3),(0,0),(0,0),(0,0),(0,0)")
        if npu_submit(fd, task_mem.obj_addr, 3, core_mask=0, subcores=subcores) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], c2).reshape(1, s["out_c"], p["out_h"], p["out_w"])
    expected = compute_expected_vectorized(inp, wt, s)
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} guarded=h160_setup3 tasks=3 amounts={';'.join(str(v) for v in H160_SETUP3_AMOUNTS)} "
          f"subcores=(0,3),(0,0),(0,0),(0,0),(0,0) {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def run_exact11_byk_shape(s):
    if s["name"] not in PREFIX_BY_K_SHAPES:
        raise ValueError("exact11 BY_K submit is scoped only to RKNN-proven h14/h7 shapes")
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = pack_weights(wt, s, p).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 8192, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = _exact11_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        subcores = ((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))
        print("exact11_byk_submit tasks=11 submit_task_number=3 amounts=" + ";".join(str(v) for v in EXACT11_BYK_AMOUNTS) +
              " masks=" + ";".join(hex(v) for v in EXACT11_BYK_MASKS) + " subcores=(0,1),(0,1),(0,1),(0,0),(0,0)")
        if npu_submit(fd, task_mem.obj_addr, 3, core_mask=0, subcores=subcores) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], UNPACK_C2).reshape(1, s["out_c"], p["out_h"], p["out_w"])
    expected = compute_expected_vectorized(inp, wt, s)
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} guarded=exact11_byk tasks=11 submit_tasks=3 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_exact11_byk_oc=" + ";".join(
            f"{start}:{float(np.max(np.abs(got[:, start:start + 32].astype(np.float32) - expected[:, start:start + 32]))):.4f}"
            for start in range(0, s["out_c"], 32)))
        print(f"debug_exact11_byk_abs got={float(np.max(np.abs(got.astype(np.float32)))):.4f} expected={float(np.max(np.abs(expected))):.4f}")
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def run_pointwise_exact11_byk_shape(s):
    if s["name"] not in POINTWISE_EXACT11_BYK_SHAPES:
        raise ValueError("pointwise exact11 BY_K submit is scoped only to RKNN-prefix proven pointwise shapes")
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = pack_weights(wt, s, p).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 8192, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = _pointwise_exact11_byk_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        subcores = ((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))
        print("pointwise_exact11_byk_submit tasks=11 submit_task_number=3 amounts=" + ";".join(str(v) for v in EXACT11_BYK_AMOUNTS) +
              " masks=" + ";".join(hex(v) for v in EXACT11_BYK_MASKS) + " subcores=(0,1),(0,1),(0,1),(0,0),(0,0)")
        if npu_submit(fd, task_mem.obj_addr, 3, core_mask=0, subcores=subcores) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], UNPACK_C2).reshape(1, s["out_c"], p["out_h"], p["out_w"])
    expected = compute_expected_vectorized(inp, wt, s)
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} guarded=pointwise_exact11_byk tasks=11 submit_tasks=3 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_pointwise_exact11_byk_y=" + ";".join(
            f"{y}:{float(np.max(np.abs(got[:, :, y:y + h].astype(np.float32) - expected[:, :, y:y + h]))):.4f}"
            for y, h, _ in POINTWISE_EXACT11_BYK_WINDOWS[s["name"]]))
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def run_h40_exact17_shape(s):
    if s["name"] != H40_SPATIAL_BY_Y_SHAPE:
        raise ValueError("h40 exact17 submit is scoped only to the prefix-proven spatial h40 shape")
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = pack_weights(wt, s, p).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 16384, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = _h40_exact17_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        write_h40_exact17_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        subcores = ((0, 2), (0, 2), (0, 2), (0, 0), (0, 0))
        print("h40_exact17_submit tasks=17 submit_task_number=6 amounts=" + ";".join(str(v) for v in H40_EXACT17_AMOUNTS) +
              " subcores=(0,2),(0,2),(0,2),(0,0),(0,0)")
        if npu_submit(fd, task_mem.obj_addr, 6, core_mask=0, subcores=subcores) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], UNPACK_C2).reshape(1, s["out_c"], p["out_h"], p["out_w"])
    expected = compute_expected_vectorized(inp, wt, s)
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} guarded=h40_exact17 tasks=17 submit_tasks=6 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def _run_single_tile_shape(s, inp, wt):
    p = _conv_params(s)
    hw_out_fp16 = p["is_spatial"] or s["out_c"] >= 128 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE or _is_pointwise_wide(s)
    read_dtype = np.float16 if hw_out_fp16 else np.float32
    c2 = UNPACK_C2 if hw_out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
    input_flat = pack_input(inp, p).view(np.uint16)
    weight_flat = pack_weights(wt, s, p).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], c2) * p["out_width_stride"] * c2
    output_bytes = out_count * np.dtype(read_dtype).itemsize
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, max(4096, input_flat.nbytes), RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, max(4096, weight_flat.nbytes), RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4096, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        write_tasks(task_map, regcmd_map, regcmd_mem, [make_regs(s, p, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, hw_out_fp16)])
        if npu_submit(fd, task_mem.obj_addr, 1) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=read_dtype, count=out_count).copy()
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    return unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], c2)

def run_grouped_serial_shape(s):
    if s["groups"] <= 1:
        raise ValueError("grouped serial path requires groups > 1")
    if s["groups"] > 16 or s["batch"] * s["groups"] > 64:
        raise ValueError("large grouped/depthwise serial path is fenced before allocation; no submit attempted")
    if s["in_c"] % s["groups"] or s["out_c"] % s["groups"]:
        raise ValueError("grouped serial path requires divisible input/output channels")
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    expected = compute_expected(inp, wt, s)
    p = _conv_params(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    in_per_group = s["in_c"] // s["groups"]
    out_per_group = s["out_c"] // s["groups"]
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 128 * 1024, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    try:
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        input_base = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
        weight_base = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
        task_regs, task_meta = [], []
        input_offset = weight_offset = output_offset = 0
        for n in range(s["batch"]):
            for g in range(s["groups"]):
                ic0 = g * in_per_group
                oc0 = g * out_per_group
                tile_shape = dict(s, name=s["name"] + "_groupserial", batch=1, in_c=in_per_group,
                                  out_c=out_per_group, weight_in_c=in_per_group, groups=1)
                tile_p = _conv_params(tile_shape)
                hw_out_fp16 = tile_p["is_spatial"] or tile_shape["out_c"] >= 128 or tile_p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE or _is_pointwise_wide(tile_shape)
                read_dtype = np.float16 if hw_out_fp16 else np.float32
                c2 = UNPACK_C2 if hw_out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
                input_flat = pack_input(inp[n, ic0:ic0 + in_per_group], tile_p).view(np.uint16)
                weight_flat = pack_weights(wt[oc0:oc0 + out_per_group], tile_shape, tile_p).view(np.uint16)
                out_count = _ceil_div(tile_p["align_out_c"], c2) * tile_p["out_width_stride"] * c2
                input_offset = _align_up(input_offset, 16)
                weight_offset = _align_up(weight_offset, 16)
                output_offset = _align_up(output_offset, 16)
                ctypes.memmove(input_base + input_offset, input_flat.ctypes.data, input_flat.nbytes)
                ctypes.memmove(weight_base + weight_offset, weight_flat.ctypes.data, weight_flat.nbytes)
                task_regs.append(make_regs(tile_shape, tile_p, input_mem.dma_addr + input_offset,
                                           weight_mem.dma_addr + weight_offset, output_mem.dma_addr + output_offset,
                                           hw_out_fp16))
                task_meta.append((n, oc0, out_count, read_dtype, out_per_group, tile_p, c2, output_offset))
                input_offset += input_flat.nbytes
                weight_offset += weight_flat.nbytes
                output_offset += out_count * np.dtype(read_dtype).itemsize
        write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        if npu_submit(fd, task_mem.obj_addr, len(task_regs)) < 0:
            raise RuntimeError("npu_submit failed")
        output_base = ctypes.addressof(ctypes.c_char.from_buffer(output_map))
        for n, oc0, out_count, read_dtype, tile_out_c, tile_p, c2, offset in task_meta:
            out_raw = np.frombuffer(output_map, dtype=read_dtype, count=out_count, offset=offset).copy()
            got[n, oc0:oc0 + tile_out_c] = unpack_output(out_raw, tile_out_c, tile_p["out_h"], tile_p["out_w"], tile_p["out_width_stride"], c2)
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    max_diff = float(np.max(np.abs(got.astype(np.float64) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} grouped_serial submits={s['batch'] * s['groups']} {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def run_pointwise_chained_y_shape(s):
    if s["name"] not in POINTWISE_CHAINED_Y_SHAPES:
        raise ValueError("pointwise chained-Y path is scoped to proven row-window shapes")
    p = _conv_params(s)
    specs = _tile_replay_specs(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_bytes = _ceil_div(s["in_c"], p["input_pack_c2"]) * p["input_pack_c2"] * s["in_h"] * p["width_stride"] * FP16_BYTES
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, max(4 * 1024 * 1024, input_bytes), RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    try:
        weight_flat = pack_weights(wt, s, p).view(np.uint16)
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        input_base = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
        input_offset = 0
        task_regs = []
        for y_start, input_h, _output_h, _oc_start, _oc_count in specs:
            tile_shape = dict(s, in_h=input_h)
            tile_flat = pack_input(inp[0, :, y_start:y_start + input_h, :], _conv_params(tile_shape)).view(np.uint16)
            ctypes.memmove(input_base + input_offset, tile_flat.ctypes.data, tile_flat.nbytes)
            row = {"y_start": y_start, "input_h": input_h, "weight_reuse": bool(task_regs)}
            task_regs.append(make_y_tile_regs(s, p, row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, input_offset))
            input_offset = _align_up(input_offset + tile_flat.nbytes, 16)
        write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        if npu_submit(fd, task_mem.obj_addr, len(task_regs)) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], UNPACK_C2).reshape(1, s["out_c"], p["out_h"], p["out_w"])
    expected = compute_expected_vectorized(inp, wt, s)
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} chained_y tiles={len(task_regs)} hw_out=fp16 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def run_pointwise_yk_shape(s):
    if s["name"] not in POINTWISE_YK_SHAPES:
        raise ValueError("pointwise YK runtime is scoped to the prefix-proven h40 pointwise shape")
    p = _conv_params(s)
    rows = planner.descriptor_rows_for_shape(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    try:
        for row in rows:
            for oc_start, oc_count in ((0, 32), (32, s["out_c"] - 32)):
                tile_shape = dict(s, in_h=row["input_h"], out_c=oc_count)
                tile_p = _conv_params(tile_shape)
                tile_in = inp[0, :, row["y_start"]:row["y_start"] + row["input_h"], :]
                tile_wt = wt[oc_start:oc_start + oc_count]
                input_flat = pack_input(tile_in, tile_p).view(np.uint16)
                weight_flat = pack_weights(tile_wt, tile_shape, tile_p).view(np.uint16)
                out_count = _ceil_div(tile_p["align_out_c"], UNPACK_C2) * tile_p["out_width_stride"] * UNPACK_C2
                (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
                (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
                ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
                local_shape = dict(tile_shape, name=tile_shape["name"] + "_tilelocal")
                local_row = {"y_start": 0, "input_h": row["input_h"], "weight_reuse": False}
                task_regs = [make_y_tile_regs(local_shape, tile_p, local_row, input_mem.dma_addr,
                                              weight_mem.dma_addr, output_mem.dma_addr, 0)]
                write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
                if npu_submit(fd, task_mem.obj_addr, 1) < 0:
                    raise RuntimeError("npu_submit failed")
                out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
                tile_got = unpack_output(out_raw, oc_count, tile_p["out_h"], tile_p["out_w"], tile_p["out_width_stride"], UNPACK_C2)
                got[0, oc_start:oc_start + oc_count, row["y_start"]:row["y_start"] + tile_p["out_h"]] = tile_got
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    expected = compute_expected_vectorized(inp, wt, s)
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} split=BY_YK family=pointwise_yk tasks=4 submits=4 hw_out=fp16 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_pointwise_yk_oc0_31=%.4f oc32_39=%.4f" % (
            float(np.max(np.abs(got[:, :32].astype(np.float32) - expected[:, :32]))),
            float(np.max(np.abs(got[:, 32:].astype(np.float32) - expected[:, 32:])))))
        print("debug_pointwise_yk_y0_34=%.4f y35_39=%.4f" % (
            float(np.max(np.abs(got[:, :, :35].astype(np.float32) - expected[:, :, :35]))),
            float(np.max(np.abs(got[:, :, 35:].astype(np.float32) - expected[:, :, 35:])))))
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def _tile_replay_specs(s):
    p = _conv_params(s)
    if s["name"] in PREFIX_BY_K_SHAPES:
        return [(0, s["in_h"], p["out_h"], oc, min(32, s["out_c"] - oc)) for oc in range(0, s["out_c"], 32)]
    if s["name"] == "b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid":
        rows = planner.descriptor_rows_for_shape(s)
        return [(row["y_start"], row["input_h"], row["output_h"], oc, min(32, s["out_c"] - oc))
                for row in rows for oc in range(0, s["out_c"], 32)]
    if s["name"] in POINTWISE_YK_SHAPES:
        rows = planner.descriptor_rows_for_shape(s)
        return [(row["y_start"], row["input_h"], row["output_h"], oc, count)
                for row in rows for oc, count in ((0, 32), (32, s["out_c"] - 32))]
    if s["name"] in LOCAL_POINTWISE_YK_SHAPES:
        rows = [row for row in planner.descriptor_rows_for_shape(s) if row["family"] == "setup"]
        specs = []
        for idx, row in enumerate(rows):
            oc_start = sum(prev["oc_count"] for prev in rows[:idx]
                           if prev["y_start"] == row["y_start"] and prev["input_h"] == row["input_h"])
            specs.append((row["y_start"], row["input_h"], row["output_h"], oc_start, row["oc_count"]))
        return specs
    if s["name"] in {"conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1",
                     "conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1",
                     "conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1",
                     "conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1"}:
        return [(0, 22, 22, 0, s["out_c"]), (22, 6, 6, 0, s["out_c"])]
    if s["name"] == "b1_c96_h75_w75_oc24_wic96_k1x1_g1_s1_pvalid":
        return [(0, 22, 22, 0, s["out_c"]), (22, 22, 22, 0, s["out_c"]),
                (44, 22, 22, 0, s["out_c"]), (66, 9, 9, 0, s["out_c"])]
    if _is_pointwise_wide(s):
        rows = planner.descriptor_rows_for_shape(s)
        return [(row["y_start"], row["input_h"], row["output_h"], 0, s["out_c"]) for row in rows]
    if s["name"] == "b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid":
        rows = planner.descriptor_rows_for_shape(s)
        return [(row["y_start"], row["input_h"], row["output_h"], 0, s["out_c"]) for row in rows]
    if s["name"] in {"conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1",
                     "b1_c3_h224_w224_oc32_wic3_k3x3_g1"}:
        return [(0, 50, 48, 0, s["out_c"]), (48, 50, 48, 0, s["out_c"]),
                (96, 50, 48, 0, s["out_c"]), (144, 50, 48, 0, s["out_c"]),
                (192, 32, 30, 0, s["out_c"])]
    raise ValueError("no local tile replay spec for shape")

def allow_local_tile_replay(s):
    return s["name"] in LOCAL_TILE_REPLAY_SHAPES

def run_local_tile_replay_shape(s):
    if not allow_local_tile_replay(s):
        raise ValueError("local tile replay is scoped to prefix-proven shapes")
    p = _conv_params(s)
    specs = _tile_replay_specs(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    try:
        for y_start, input_h, output_h, oc_start, oc_count in specs:
            hw_oc = 32 if _is_pointwise_wide(s) and oc_count < 32 else oc_count
            tile_shape = dict(s, name=s["name"] + "_tilelocal", in_h=input_h, out_c=hw_oc)
            tile_p = _conv_params(tile_shape)
            tile_in = inp[0, :, y_start:y_start + input_h, :]
            tile_wt = np.zeros((hw_oc, s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
            tile_wt[:oc_count] = wt[oc_start:oc_start + oc_count]
            input_flat = pack_input(tile_in, tile_p).view(np.uint16)
            weight_flat = pack_weights(tile_wt, tile_shape, tile_p).view(np.uint16)
            out_count = _ceil_div(tile_p["align_out_c"], UNPACK_C2) * tile_p["out_width_stride"] * UNPACK_C2
            (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
            (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
            ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
            if y_start or output_h < p["out_h"]:
                local_row = {"y_start": 0, "input_h": input_h, "weight_reuse": False}
                task_regs = [make_y_tile_regs(tile_shape, tile_p, local_row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, 0)]
            else:
                task_regs = [make_regs(tile_shape, tile_p, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, True)]
            write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
            if npu_submit(fd, task_mem.obj_addr, 1) < 0:
                raise RuntimeError("npu_submit failed")
            out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
            tile_got = unpack_output(out_raw, hw_oc, output_h, p["out_w"], tile_p["out_width_stride"], UNPACK_C2)[:oc_count]
            got[0, oc_start:oc_start + oc_count, y_start:y_start + output_h] = tile_got
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    expected = compute_expected_vectorized(inp, wt, s)
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} local_tile_replay tiles={len(specs)} hw_out=fp16 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def run_shape(s):
    rows = validate_phase_a_shape(s)
    p = _conv_params(s)
    hw_out_fp16 = p["is_spatial"] or s["out_c"] >= 128 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE or _is_pointwise_wide(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    weight_flat = pack_weights(wt, s, p).view(np.uint16)
    read_dtype = np.float16 if hw_out_fp16 else np.float32
    c2 = UNPACK_C2 if hw_out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
    out_count = _ceil_div(p["align_out_c"], c2) * p["out_width_stride"] * c2
    output_bytes = out_count * np.dtype(read_dtype).itemsize
    output_flags = RKNPU_MEM_NON_CONTIGUOUS if output_bytes > 4 * 1024 * 1024 else RKNPU_MEM_NON_CACHEABLE
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), output_flags)
    try:
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        if rows[0]["split_method"] == "BY_Y":
            task_regs = []
            if s["name"] in POINTWISE_YK_SHAPES:
                input_flat = pack_input(inp[0], p).view(np.uint16)
                (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
                for row in rows:
                    for oc_start, oc_count in ((0, 32), (32, s["out_c"] - 32)):
                        task_regs.append(make_yk_pointwise_regs(s, p, row, oc_start, oc_count,
                                                                 input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr))
            elif s["name"] in PREFIX_BY_Y_SHAPES and not p["use_nhwc"]:
                input_flat = pack_input(inp[0], p).view(np.uint16)
                (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
                for row in rows:
                    task_regs.append(make_y_tile_regs(s, p, row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, row["feature_off"]))
            else:
                input_base = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
                input_offset = 0
                for row in rows:
                    tile_shape = dict(s, in_h=row["input_h"])
                    tile_flat = pack_input(inp[0, :, row["y_start"]:row["y_start"] + row["input_h"], :], _conv_params(tile_shape)).view(np.uint16)
                    ctypes.memmove(input_base + input_offset, tile_flat.ctypes.data, tile_flat.nbytes)
                    task_regs.append(make_y_tile_regs(s, p, row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, input_offset))
                    input_offset = _align_up(input_offset + tile_flat.nbytes, 16)
        else:
            input_flat = pack_input(inp[0], p).view(np.uint16)
            (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
            if rows[0]["split_method"] == "BY_K":
                task_regs = [make_k_tile_regs(s, p, row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr) for row in rows]
            else:
                task_regs = [make_regs(s, p, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, hw_out_fp16)]
        submit_count = len(task_regs)
        if s["name"] in SETUP2_CLOSURE_SHAPES and s["name"] not in POINTWISE_YK_SHAPES:
            emitted_rows, offsets, pc_amounts = write_setup2_closure_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
            print(f"setup2_closure_submit tasks=2 amounts=108;108 rows={';'.join(str(len(row)) for row in emitted_rows)} "
                  f"offsets={';'.join(str(v) for v in offsets)} pc_amounts={';'.join(str(v) for v in pc_amounts)}")
        else:
            write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        subcores = None
        core_mask = 1
        if s["name"] in PREFIX_BY_K_SHAPES:
            core_mask = 0
            subcores = ((0, submit_count), (0, 0), (0, 0), (0, 0), (0, 0))
        elif s["name"] in PREFIX_BY_Y_SHAPES:
            core_mask = 0
            subcores = ((0, submit_count), (0, 0), (0, 0), (0, 0), (0, 0))
        if npu_submit(fd, task_mem.obj_addr, submit_count, core_mask=core_mask, subcores=subcores) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=read_dtype, count=out_count).copy()
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], c2).reshape(1, s["out_c"], p["out_h"], p["out_w"])
    expected = compute_expected_vectorized(inp, wt, s) if s["groups"] == 1 else compute_expected(inp, wt, s)
    atol = 0.12 if hw_out_fp16 else 0.1
    max_diff = float(np.max(np.abs(got.astype(np.float64) - expected)))
    ok = bool(np.allclose(got, expected, atol=atol))
    print(f"shape={s['name']} split={rows[0]['split_method']} family={rows[0]['family']} tasks={len(task_regs)} submit_tasks={submit_count} regs={';'.join(str(len(regs)) for regs in task_regs)} hw_out={'fp16' if hw_out_fp16 else 'fp32'} {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print(f"debug_output_absmax={float(np.max(np.abs(got.astype(np.float64)))):.4f} expected_absmax={float(np.max(np.abs(expected))):.4f}")
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Planner-driven Phase A CONV submit path")
    parser.add_argument("shape", nargs="?", help="supported shape name")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run-exact11-byk", action="store_true", help="print exact-11 BY_K metadata without DRM")
    parser.add_argument("--dry-run-h160-spatial-by-y", action="store_true", help="print h160 spatial BY_Y executable rows without DRM")
    parser.add_argument("--allow-h160-setup3-submit", action="store_true", help="run the guarded h160 setup3 closure attempt")
    parser.add_argument("--allow-exact11-byk-submit", action="store_true", help="run the guarded exact-11 BY_K h14 attempt")
    parser.add_argument("--allow-pointwise-exact11-byk-submit", action="store_true", help="run the guarded pointwise exact-11 BY_K attempt")
    args = parser.parse_args(argv)
    if args.list:
        print("encoded shape syntax: [conv2d_]bN_cN_hN_wN_ocN_wicN_kHxW_gN[_sN][_pvalid]")
        print("known examples:")
        print("\n".join(LIST_SHAPES))
        return 0
    if not args.shape:
        print("error: shape is required unless --list is used", file=sys.stderr)
        return 1
    try:
        s = shape_from_name(args.shape)
        if args.dry_run_exact11_byk:
            dry_run_exact11_byk(s)
        elif args.dry_run_h160_spatial_by_y:
            dry_run_h160_spatial_by_y(s)
        elif args.allow_exact11_byk_submit or s["name"] in PREFIX_BY_K_SHAPES:
            run_exact11_byk_shape(s)
        elif args.allow_pointwise_exact11_byk_submit or s["name"] in POINTWISE_EXACT11_BYK_SHAPES:
            run_pointwise_exact11_byk_shape(s)
        elif s["name"] == H40_SPATIAL_BY_Y_SHAPE:
            run_h40_exact17_shape(s)
        elif args.allow_h160_setup3_submit or s["name"] == H160_SPATIAL_BY_Y_SHAPE:
            run_h160_setup3_shape(s)
        elif s["name"] in GROUPED_SERIAL_SHAPES:
            run_grouped_serial_shape(s)
        elif s["groups"] > 1:
            raise ValueError("grouped/depthwise native and serial paths are fenced after post-health timeout; no submit attempted")
        elif s["name"] in POINTWISE_CHAINED_Y_SHAPES:
            run_pointwise_chained_y_shape(s)
        elif allow_local_tile_replay(s):
            run_local_tile_replay_shape(s)
        elif s["name"] in POINTWISE_YK_SHAPES:
            run_pointwise_yk_shape(s)
        else:
            run_shape(s)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
