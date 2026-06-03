import os, mmap, sys, ctypes, argparse, re
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
C256_H2_OC64_EXACT11_SHAPE = "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid"
H40_SPATIAL_BY_Y_SHAPE = "b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid"
EXACT11_BYK_AMOUNTS = (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
EXACT11_BYK_MASKS = (0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
EXACT11_BYK_PC_AMOUNTS = (0, 0x1000e, 0, 0x1000e, 0, 0x2000e, 0, 0x2000e, 0, 0x2000e)
EXACT11_BYK_ROLES = ("setup_body", "k_half_body0", "aux0", "k_half_body1", "aux1", "k_tile_body0", "aux2", "k_tile_body1", "aux3", "k_tile_body2", "aux4")
EXACT11_BYK_TAIL_CLASSES = ("ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING", "ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING", "ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING", "ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING", "ZERO_PADDING,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING", "PC_BASE_ADDRESS,PC_REGISTER_AMOUNTS,VERSION,PC_OPERATION_ENABLE,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING,ZERO_PADDING")
EXACT11_BYK_TAIL_VALUES = ("0,0,0,13", "0,65550,0,13,0,0,0,0", "0,0,0,96,0,0", "0,65550,0,13,0,0,0,0", "0,0,0,96,0,0", "0,131086,0,13,0,0,0,0", "0,0,0,96,0,0", "0,131086,0,13,0,0,0,0", "0,0,0,96,0,0", "0,131086,0,13,0,0,0,0")
EXACT_BYK_BODY_AMOUNT = 104
EXACT_BYK_SETUP_AMOUNT = 108
EXACT_BYK_AUX_AMOUNT = 26
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
PREFIX_BY_K_SHAPES = {
    EXACT11_BYK_SHAPE,
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid",
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid",
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid",
    "b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid",
    "b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid",
    "b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid",
    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid",
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid",
    "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid",
    "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1",
    "b1_c512_h7_w7_oc1024_wic512_k1x1_g1",
    C256_H2_OC64_EXACT11_SHAPE,
    "b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid",
}
POINTWISE_EXACT11_BYK_SHAPES = {
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1",
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1",
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1",
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1",
    "b1_c512_h14_w14_oc32_wic512_k1x1_g1",
    "b1_c512_h14_w14_oc112_wic512_k1x1_g1",
    "b1_c512_h14_w14_oc512_wic512_k1x1_g1",
    "conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1",
    "b1_c384_h14_w14_oc96_wic384_k1x1_g1",
    "b1_c480_h14_w14_oc96_wic480_k1x1_g1",
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1",
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1",
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1",
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1",
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid",
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid",
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid",
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid",
    "b1_c576_h14_w14_oc96_wic576_k1x1_g1",
}
POINTWISE_EXACT11_BYK_WINDOWS = {
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1": ((0, 10, 0x0a0), (10, 9, 0x090), (19, 9, 0x090)),
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1": ((0, 10, 0x0a0), (10, 9, 0x090), (19, 9, 0x090)),
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c512_h14_w14_oc32_wic512_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c512_h14_w14_oc112_wic512_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c512_h14_w14_oc512_wic512_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c384_h14_w14_oc96_wic384_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c480_h14_w14_oc96_wic480_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
    "b1_c576_h14_w14_oc96_wic576_k1x1_g1": ((0, 5, 0x060), (5, 5, 0x060), (10, 4, 0x050)),
}
POINTWISE_EXACT11_BYK_CBUF0 = {
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1": 0x057,
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1": 0x084,
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1": 0x057,
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1": 0x084,
    "b1_c512_h14_w14_oc32_wic512_k1x1_g1": 0x057,
    "b1_c512_h14_w14_oc112_wic512_k1x1_g1": 0x057,
    "b1_c512_h14_w14_oc512_wic512_k1x1_g1": 0x057,
    "conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1": 0x057,
    "b1_c384_h14_w14_oc96_wic384_k1x1_g1": 0x057,
    "b1_c480_h14_w14_oc96_wic480_k1x1_g1": 0x057,
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1": 0x057,
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1": 0x057,
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1": 0x057,
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1": 0x057,
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid": 0x057,
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid": 0x057,
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid": 0x057,
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid": 0x057,
    "b1_c576_h14_w14_oc96_wic576_k1x1_g1": 0x057,
}
POINTWISE_EXACT11_BYK_DMA2 = {
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1": 0x2a0,
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1": 0x08c,
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1": 0x2a0,
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1": 0x08c,
    "b1_c512_h14_w14_oc32_wic512_k1x1_g1": 0x08c,
    "b1_c512_h14_w14_oc112_wic512_k1x1_g1": 0x08c,
    "b1_c512_h14_w14_oc512_wic512_k1x1_g1": 0x08c,
    "conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1": 0x08c,
    "b1_c384_h14_w14_oc96_wic384_k1x1_g1": 0x08c,
    "b1_c480_h14_w14_oc96_wic480_k1x1_g1": 0x08c,
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1": 0x08c,
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1": 0x08c,
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1": 0x08c,
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1": 0x08c,
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid": 0x08c,
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid": 0x08c,
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid": 0x08c,
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid": 0x08c,
    "b1_c576_h14_w14_oc96_wic576_k1x1_g1": 0x08c,
}
POINTWISE_EXACT11_BYK_DATA_SIZE1 = {
    "b1_c512_h14_w14_oc32_wic512_k1x1_g1": 0x003f0200,
    "b1_c512_h14_w14_oc112_wic512_k1x1_g1": 0x003f0200,
    "b1_c512_h14_w14_oc512_wic512_k1x1_g1": 0x003f0200,
    "conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1": 0x003f0200,
    "b1_c384_h14_w14_oc96_wic384_k1x1_g1": 0x00bf0180,
    "b1_c480_h14_w14_oc96_wic480_k1x1_g1": 0x00ef01e0,
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1": 0x01070210,
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1": 0x01070210,
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1": 0x01070210,
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1": 0x01070210,
    "b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid": 0x01070210,
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid": 0x01070210,
    "b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid": 0x01070210,
    "b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid": 0x01070210,
    "b1_c576_h14_w14_oc96_wic576_k1x1_g1": 0x011f0240,
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
POINTWISE_SETUP108_COMPACT_WEIGHT_SHAPES = {
    "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid",
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid",
}
CRASH_FENCED_SHAPES = {
    "b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid",
}
C256_H2_OC64_EXACT11_SHAPE = "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid"
C256_H2_OC64_EXACT11_OUT_C = (64, 32, None, 32, None, 32, None, 16, None, 16, None)
C256_H2_OC64_EXACT11_WEIGHT_SIZE0 = (0x8000, 0x4000, None, 0x4000, None, 0x4000, None, 0x2000, None, 0x2000, None)
C256_H2_OC64_EXACT11_DST_OFFSETS = (0x0, 0x0, None, 0x100, None, 0x0, None, 0x100, None, 0x180, None)
C576_H19_OC12_EXACT12_SHAPE = "b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid"
C576_H19_OC12_EXACT12_OUT_C = (12, 12, None, 4, None, 3, None, 2, None, 1, None, 2, None)[:11]  # 12-task row out_c (placeholders; we patch per row)
# Actual body field values from the c576_h19_oc12_s1pvalid_keep1_gem2 dump (NOT c256_h2_oc64 guesses):
C576_H19_OC12_CBUF0 = 0x1b                   # CBUF_CON0: WEIGHT_BANK(1) | DATA_BANK(11)
C576_H19_OC12_CBUF0_REUSE = 0x201b           # CBUF_CON0 with WEIGHT_REUSE(1) for k_half
C576_H19_OC12_DATA_SIZE1 = 0x3f0240          # DATAIN_CHANNEL_REAL(63) | DATAIN_CHANNEL(576) - actual from dump
C576_H19_OC12_DMA_CON2 = 0x011d              # SURF_STRIDE(285) - actual from dump
C576_H19_OC12_CVT_CON0 = 0x000b              # DATA_SIGN(1) | CVT_TYPE(1) | CVT_BYPASS(1) - actual from dump
C576_H19_OC12_SETUP_WEIGHT_SIZE0 = 0x3600    # 12*576*2 fp16 = 13824 (all 12 OC of weight)
C576_H19_OC12_K_HALF_WEIGHT_SIZE0 = 0x3600   # same as setup; k_half covers all 12 OC
C576_H19_OC12_K_TILE_WEIGHT_SIZE0 = 0x3600   # all k_tile rows use the full 12-OC weight (unused OC bits are ignored)
C576_H19_OC12_K_TILE_SPLITS = ((0, 4), (4, 3), (7, 2), (9, 1), (10, 2))  # 4+3+2+1+2=12
C576_H19_OC12_EXACT12_AMOUNTS = (108, 108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
C576_H19_OC12_EXACT12_MASKS = (0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
# Per-row out_c (used for body field patches); (None, 'aux') means skip (aux row)
C576_H19_OC12_EXACT12_ROW_OUT_C = (12, 12, 12, None, 12, None, 12, None, 12, None, 12, None)
C576_H19_OC12_EXACT12_ROW_CBUF0 = (C576_H19_OC12_CBUF0, C576_H19_OC12_CBUF0_REUSE, C576_H19_OC12_CBUF0, None, C576_H19_OC12_CBUF0, None, C576_H19_OC12_CBUF0, None, C576_H19_OC12_CBUF0, None, C576_H19_OC12_CBUF0, None)


POINTWISE_EXACT11_CHAIN_COMPACT_WEIGHT_SHAPES = {
    "b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid",
}
POINTWISE_EXACT11_COMPACT_WEIGHT_SHAPES = set()
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
    "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid",
}
KNOWN_BAD_SPATIAL_SETUP_SHAPES = {
    # c16_h80_oc64 promoted via EXACT11 BY_K closure (CBUF0=0x57, CONV2_LOW=0x1a0)
    # c16_h80_oc128 (3x3) needs custom 4-task structure (1 preamble + 3 y-tile computes)
    # c16_h80_oc128 (5x5) still fenced (different kernel)
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid",
    "b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid",
    "b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid",
}
DEPTHWISE_BODY_SHAPES = set()  # emptied: single-task rewrite needs more capture work; leave shapes fenced for now
DEPTHWISE_SETUP108_SHAPES = set()
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

CBUF0_OVERRIDES = {
    "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid": 0x0a2,
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid": 0x0a2,
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid": 0x057,
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid": 0x057,
    "b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid": 0x057,
    # 1x1 pointwise family: cbuf0=0xb1 (per live rknn_runtime c64_h1_oc128 capture)
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid": 0x0b1,
    "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid": 0x02a,
    "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid": 0x0b1,
    "b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid": 0x0b1,
    "b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid": 0x0b1,
    "b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid": 0x0b1,
    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x0b1,
    "b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid": 0x0b1,
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid": 0x084,
    "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1": 0x0a2,
    "b1_c512_h7_w7_oc1024_wic512_k1x1_g1": 0x0a2,
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1": 0x093,
}
DATA_SIZE1_OVERRIDES = {
    "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid": 0x1f00a0,
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid": 0x1f00a0,
    # c16_h80_oc64: natural formula gives 0x000F0010 (different from c160's 0x1f00a0)
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid": 0x000F0010,
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid": 0x000F0010,
    "b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid": 0x000F0010,
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid":   0x003f0040,
    "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid": 0x003f00c0,
    "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid":   0x003f0100,
    "b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid":   0x003f0100,
    "b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid":   0x003f0100,
    "b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid":   0x003f0080,
    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid":   0x003f0080,
    "b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid":   0x003f0080,
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid":  0x00270028,
    "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1": 0x003f0200,
    "b1_c512_h7_w7_oc1024_wic512_k1x1_g1":            0x003f0200,
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1":              0x003f0340,
}
CBUF1_OVERRIDES = {
    # c160_h14 and c160_h7 fall through to make_regs default (_cbuf_entries)
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid": 0x032,
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1": 0x0b6,
}
WEIGHT_SIZE0_OVERRIDES = {
    # 1x1 pointwise c64_h1_oc128: per-family weight sizes from live capture
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "setup"):  0x4000,
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_half"): 0x2000,
    ("b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid", "setup"): 0x6400,
    ("b1_c832_h7_w7_oc48_wic832_k1x1_g1", "setup"): 0x13800,
    # c16_h80_oc128 (3x3): setup=128*16*2=0x1000, k_half=64*16*2=0x800
}
WEIGHT_SIZE1_OVERRIDES = {
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid": 0x050,
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1": 0x680,
}
WEIGHT_SIZE2_OVERRIDES = {}
CVT_CON0_OVERRIDES = {
    # 1x1 pointwise c64_h1_oc128: live capture shows CVT_CON0=0xb even though kh=kw=1
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid": 0x000b,
    "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid": 0x000b,
    "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid": 0x000b,
    "b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid": 0x000b,
    "b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid": 0x000b,
    "b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid": 0x000b,
    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x000b,
    "b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid": 0x000b,
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid": 0x000b,
    "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1": 0x000b,
    "b1_c512_h7_w7_oc1024_wic512_k1x1_g1": 0x000b,
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1": 0x000b,
}

DEPTHWISE_OVERRIDES = {
    "conv_con1": 0x123,
    "conv2_low": 0x90,
    "data_size1": 0x003f0100,
    "cbuf0": 0xa2,
    "cbuf_con1": 0x50,
    "weight_size0": 0x1200,
    "weight_size1": 0x1200,
    "weight_size2": 0x03030001,
    "dma_con2": 0x3c,
    "feature_addr_padding": 0x0,
}

FC_DATA_SIZE1_OVERRIDES = {
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid": 0x028,
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1": 0x340,
}
DMA_CON2_OVERRIDES = {
    # 1x1 pointwise c64_h1_oc128: live capture shows DMA_CON2=0x0ffffffd (huge aux buffer)
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid": 0x0ffffffd,
    "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid": 0x02a0,
    "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid": 0x0ffffffc,
    "b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid": 0x0ffffffd,
    "b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid": 0x0ffffffd,
    "b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid": 0x0ffffffd,
    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x0ffffffd,
    "b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid": 0x0ffffffc,
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid": 0x05a0,
    "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1": 0x0015,
    "b1_c512_h7_w7_oc1024_wic512_k1x1_g1": 0x0015,
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1": 0x0015,
}
KT_FAMILY_BITS_OVERRIDES = {
}
# Per-shape k_tile OC splits (must sum to s["out_c"]; each oc_count is rounded up to 16)
# c160_h14/h7 fall through to the default `((0, 112), (112, 112), (224, 96))` for oc=320
# c16_h80_oc64 falls through to the same default which writes beyond oc=64 but the NPU masks
# The K_SPLITS dict overrides the default for shapes where the default is wrong (e.g. oc=128 needs (0, 32), (32, 64), (64, 32))
KT_TILE_SPLITS = {
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid": ((0, 48), (48, 48), (96, 32)),
    "b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid": ((0, 48), (48, 48), (96, 32)),
    "b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid": ((0, 96), (96, 96), (192, 64)),
    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": ((0, 96), (96, 96), (192, 64)),
    "b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid": ((0, 96), (96, 96), (192, 64)),
    # 1x1 pointwise: 3 k_tiles (per live c64_h1_oc128 capture)
    # Note: first k_tile covers 64 OC (overlaps with k_half)
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid":   ((0, 64), (48, 48), (96, 32)),
    "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid": ((0, 32), (32, 32), (64, 32)),
    "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1": ((0, 352), (352, 336), (688, 336)),
    "b1_c512_h7_w7_oc1024_wic512_k1x1_g1":           ((0, 352), (352, 336), (688, 336)),
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1":             ((0, 16), (16, 16), (32, 16)),
    "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid":   ((0, 32), (32, 16), (48, 16)),
    # 12-task c576_h19_oc12: 1 k_half + 5 k_tiles (k_splits 4+3+2+1+2=12), 5 aux rows
    C576_H19_OC12_EXACT12_SHAPE: ((0, 4), (4, 3), (7, 2), (9, 1), (10, 2)),
}
CONV2_LOW_OVERRIDES = {
    # c160_h14 and c160_h7 fall through to the default logic below (0x0f0 / 0x0a0)
    # c16_h80_oc64 spatial 3x3 with in_h=80 uses conv2_low=0x1a0 (per live RKNN capture)
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid": 0x1a0,
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid": 0x1a0,
    "b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid": 0x1a0,
    # 1x1 pointwise family: conv2_low=0x20 (per live c64_h1_oc128 capture)
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid": 0x020,
    "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid": 0x090,
    "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid": 0x030,
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid": 0x160,
    "b1_c832_h7_w7_oc48_wic832_k1x1_g1": 0x008,
    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x060,
    "b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid": 0x008,
}
# 1x1 with h*w=1: per-family DST_BASE offsets (per c64_h1_oc128 live capture)
# Each k_half/k_tile writes oc_start * 2 bytes (per fp16, c2-packed) into output
# k_halves use 0, 0x80 (=64*2); k_tiles use 0, 0x60 (=48*2), 0xc0 (=96*2)
# We pass a (name, family, oc_start) -> byte_offset dict
DST_OFFSETS_OVERRIDES = {
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "setup"):  0x00,  # oc_start=0
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_half", 0):  0x00,
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_half", 64): 0x80,
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_tile", 0):  0x00,
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_tile", 48): 0x60,
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_tile", 96): 0xc0,
}



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
    # Short descriptive form: conv2d_<ic>x<oc>_<kh>x<kw>_<h>x<w>[_g<g>][_b]
    #   e.g. conv2d_3x6_1x3_5x5, conv2d_4x4_1x1_1x1_g2
    short = re.match(
        r"^conv2d_(?P<ic>\d+)x(?P<oc>\d+)_(?P<kh>\d+)x(?P<kw>\d+)_(?P<h>\d+)x(?P<w>\d+)(?:_g(?P<g>\d+))?(?:_b)?$",
        name,
    )
    if short:
        s = short.groupdict()
        return dict(
            name=name,
            batch=1,
            in_c=int(s["ic"]),
            in_h=int(s["h"]),
            in_w=int(s["w"]),
            out_c=int(s["oc"]),
            weight_in_c=int(s["ic"]) // int(s.get("g") or 1),
            kh=int(s["kh"]),
            kw=int(s["kw"]),
            groups=int(s["g"] or 1),
            stride=1,
        )
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

def _pack_pointwise_compact_weight(weight, out_c, in_c):
    blocks = []
    for oc in range(0, out_c, 16):
        oc_count = min(16, out_c - oc)
        for ic in range(0, in_c, 32):
            blocks.append(weight[oc:oc + oc_count, ic:ic + 32, 0, 0].ravel())
    return np.concatenate(blocks).astype(np.float16)

def _pack_depthwise_compact_weight(weight, out_c, kh, kw):
    return weight[:out_c, 0, :kh, :kw].reshape(out_c, kh, kw).ravel().astype(np.float16)

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

def make_regs(s, p, in_dma, wt_dma, out_dma, out_fp16, full_data_bank=False):
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
    if full_data_bank:
        data_bank = RK_CBUF_BANKS - 1
    else:
        data_bank = _data_bank(p["width_stride"], feature_grains, data_in_channel_aligned, p["use_nhwc"], is_spatial, False)
    out_precision = 2 if out_fp16 else 5
    size_e = 1 if out_fp16 else 3
    out_channel_field = align_out_c - 1
    cvt_con0 = 0x0b if is_spatial and not p["is_depthwise"] else 1
    cvt_con5 = ((1 << in_c) - 1) if p["use_nhwc"] else 0
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
        E(reg.CORE, reg.CORE_MISC_CFG, 2 << 8), E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, ((out_h - 1) << 16) | (out_w - 1)),
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
        E(reg.DPU, reg.SURFACE_ADD, (p["out_width_stride"] * 2) << 4),
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
        (reg.DPU, reg.SURFACE_ADD): (p["out_width_stride"] * 2) << 4,
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
        (reg.DPU, reg.SURFACE_ADD): (p["out_width_stride"] * 2) << 4,
    }
    return patch_regs(regs, patches)

def make_local_k_tile_regs(s, p, in_dma, wt_dma, out_dma):
    regs = make_regs(s, p, in_dma, wt_dma, out_dma, True)
    conv2 = 0x500000a0 if s["in_h"] == 7 else 0x500000f0
    patches = {(reg.CNA, reg.CNA_CONV_CON2): conv2}
    if s["in_h"] == 14:
        patches[(reg.CNA, reg.CNA_CBUF_CON0)] = 0xa2
    return patch_regs(regs, patches)

# Depthwise body uses a distinct register layout: family_bits=0x10000000 (depthwise setup)
# or 0x50000000 (depthwise k_tile), plus specific overrides for conv_con1, conv2_low,
# cbuf0, weight sizes, dma_con2. Captured from rknn_runtime
# b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid at GEM 2:0x2000:768.
def _depthwise_body_patches(s, family, oc_count=0):
    surface_add = (oc_count * 2) << 4 if oc_count else (s["out_c"] * 2) << 4
    family_bits = 0x50000000 if family == "k_tile" else 0x10000000
    return {
        (reg.CNA, reg.CNA_CONV_CON1): DEPTHWISE_OVERRIDES["conv_con1"],
        (reg.CNA, reg.CNA_CONV_CON2): family_bits | DEPTHWISE_OVERRIDES["conv2_low"],
        (reg.CNA, reg.CNA_DATA_SIZE1): DEPTHWISE_OVERRIDES["data_size1"],
        (reg.CNA, reg.CNA_CBUF_CON0): DEPTHWISE_OVERRIDES["cbuf0"],
        (reg.CNA, reg.CNA_CBUF_CON1): DEPTHWISE_OVERRIDES["cbuf_con1"],
        (reg.CNA, reg.CNA_WEIGHT_SIZE0): DEPTHWISE_OVERRIDES["weight_size0"],
        (reg.CNA, reg.CNA_WEIGHT_SIZE1): DEPTHWISE_OVERRIDES["weight_size1"],
        (reg.CNA, reg.CNA_WEIGHT_SIZE2): DEPTHWISE_OVERRIDES["weight_size2"],
        (reg.CNA, reg.CNA_DMA_CON2): DEPTHWISE_OVERRIDES["dma_con2"],
        (reg.CNA, reg.CNA_CVT_CON5): 0,
        (reg.CORE, reg.CORE_MISC_CFG): 0x200,
        (reg.DPU, reg.SURFACE_ADD): surface_add,
    }

def make_depthwise_setup_regs(s, p, in_dma, wt_dma, out_dma):
    regs = make_regs(s, p, in_dma, wt_dma, out_dma, True)
    return patch_regs(regs, _depthwise_body_patches(s, "setup"))

def make_depthwise_setup108_regs(s, p, in_dma, wt_dma, out_dma):
    if s["name"] != "b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid":
        raise ValueError("depthwise setup108 replay is scoped to the prefix-proven c96_h20 shape")
    regs = _exact11_body_regs(s, "setup", 0, s["out_c"], in_dma, wt_dma, out_dma, conv2_low=0x120)
    regs = [qword for qword in regs if (qword >> 48, qword & 0xffff) not in {
        (reg.CORE, reg.CORE_RESERVED_3030),
        (reg.DPU, 0x40c4),
    }]
    regs.extend((E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0), E(reg.VERSION, 0, 0)))
    return patch_regs(regs, {
        (reg.CNA, reg.CNA_CONV_CON1): 0x123,
        (reg.CNA, reg.CNA_CONV_CON2): 0x120,
        (reg.CNA, reg.CNA_DATA_SIZE1): 0x001f0060,
        (reg.CNA, reg.CNA_WEIGHT_SIZE0): 0x06c0,
        (reg.CNA, reg.CNA_WEIGHT_SIZE1): 0x06c0,
        (reg.CNA, reg.CNA_WEIGHT_SIZE2): 0x03030001,
        (reg.CNA, reg.CNA_CBUF_CON0): 0x0093,
        (reg.CNA, reg.CNA_CBUF_CON1): 0x003c,
        (reg.CNA, reg.CNA_CVT_CON0): 0x000b,
        (reg.CORE, reg.CORE_MISC_CFG): 0x202,
        (reg.DPU, reg.FEATURE_MODE_CFG): 0x1fc,
        (reg.DPU, reg.BS_OW_CFG): 0x36e,
        (reg.DPU, reg.DST_SURF_STRIDE): 0x1440,
        (reg.DPU, reg.SURFACE_ADD): 0x5100,
    })

def make_depthwise_k_tile_regs(s, p, row, in_dma, wt_dma, out_dma):
    tile_shape = dict(s, out_c=row["oc_count"])
    tile_p = _conv_params(tile_shape)
    regs = make_regs(tile_shape, tile_p, in_dma + row["feature_off"], wt_dma + row["weight_off"],
                     out_dma + row["output_off"], True)
    return patch_regs(regs, _depthwise_body_patches(s, "k_tile", oc_count=row["oc_count"]))

def make_depthwise_y_tile_regs(s, p, row, in_dma, wt_dma, out_dma, input_off):
    tile_shape = dict(s, in_h=row["input_h"])
    tile_p = _conv_params(tile_shape)
    regs = make_regs(tile_shape, tile_p, in_dma + input_off, wt_dma, out_dma + row["y_start"] * p["out_w"] * 16, True)
    return patch_regs(regs, _depthwise_body_patches(s, "y_tile"))

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
    # 1x1 h*w=1 override: per-family per-oc_start output offset (live c64_h1_oc128 capture)
    if (s["name"], family, oc_start) in DST_OFFSETS_OVERRIDES:
        output_off = DST_OFFSETS_OVERRIDES[(s["name"], family, oc_start)]
    regs = make_regs(tile_shape, tile_p, in_dma + feature_off, wt_dma + weight_off, out_dma + output_off, True)
    family_bits_default = {"setup": 0, "y_tile": 0x20000000, "k_half": 0x40000000, "k_tile": 0x50000000}
    family_bits = (KT_FAMILY_BITS_OVERRIDES.get(s["name"], {}).get(family) or family_bits_default[family])
    if conv2_low is None:
        conv2_low = CONV2_LOW_OVERRIDES.get(s["name"], 0x0c0 if s["name"] == H40_SPATIAL_BY_Y_SHAPE else (0x0a0 if s["in_h"] == 7 else 0x0f0))
    cbuf0 = CBUF0_OVERRIDES.get(s["name"], 0x039 if s["name"] == H40_SPATIAL_BY_Y_SHAPE else 0x0a2)
    # Per-family per-shape override tables: lookup key is (name, family)
    family_key = (s["name"], family)
    if family == "k_half":
        # k_half weight size is for the k-half oc_count, not the full oc
        kh_weight_size0 = oc_count * s["in_c"] * s["kh"] * s["kw"] * FP16_BYTES
    else:
        kh_weight_size0 = None
    patches = {
        (reg.CNA, reg.CNA_CONV_CON2): family_bits | conv2_low,
        (reg.CNA, reg.CNA_DATA_SIZE1): DATA_SIZE1_OVERRIDES.get(family_key, DATA_SIZE1_OVERRIDES.get(s["name"], 0x1f00a0)),
        (reg.CNA, reg.CNA_CBUF_CON0): cbuf0,
        **{key: value for key, value in [
            ((reg.CNA, reg.CNA_CBUF_CON1), CBUF1_OVERRIDES.get(family_key, CBUF1_OVERRIDES.get(s["name"]))),
            ((reg.CNA, reg.CNA_WEIGHT_SIZE0), kh_weight_size0 if kh_weight_size0 is not None else WEIGHT_SIZE0_OVERRIDES.get(family_key, WEIGHT_SIZE0_OVERRIDES.get(s["name"]))),
            ((reg.CNA, reg.CNA_WEIGHT_SIZE1), WEIGHT_SIZE1_OVERRIDES.get(family_key, WEIGHT_SIZE1_OVERRIDES.get(s["name"]))),
            ((reg.CNA, reg.CNA_CVT_CON0), CVT_CON0_OVERRIDES.get(family_key, CVT_CON0_OVERRIDES.get(s["name"]))),
            ((reg.CNA, reg.CNA_FC_DATA_SIZE1), FC_DATA_SIZE1_OVERRIDES.get(family_key, FC_DATA_SIZE1_OVERRIDES.get(s["name"]))),
        ] if value is not None},
        (reg.CNA, reg.CNA_CVT_CON5): 0,
        (reg.CORE, reg.CORE_MISC_CFG): 0x200,
        (reg.DPU, reg.DST_SURF_STRIDE): p["out_width_stride"] << 4,
        (reg.DPU, reg.SURFACE_ADD): (p["out_width_stride"] * 2) << 4,
        (reg.CNA, reg.CNA_DMA_CON2): DMA_CON2_OVERRIDES.get(s["name"], _dma_strides(s["in_h"], p["width_stride"], p["use_nhwc"])[1]),
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
    layout = exact_byk_legacy_layout_check(s)
    if s["name"] == C256_H2_OC64_EXACT11_SHAPE:
        rows = _c256_h2_oc64_exact11_task_regs(s, in_dma, wt_dma, out_dma)
        if tuple(len(row) for row in rows) != layout["amounts"]:
            raise RuntimeError("c256_h2_oc64 exact11 BY_K row amounts changed")
        return rows
    if s["name"] == "b1_c832_h7_w7_oc48_wic832_k1x1_g1":
        setup0 = _exact11_body_regs(s, "setup", 0, s["out_c"], in_dma, wt_dma, out_dma, input_h=7, conv2_low=0x08)
        setup1 = patch_regs(
            _exact11_body_regs(s, "k_tile", 0, s["out_c"], in_dma, wt_dma, out_dma, y_start=0, input_h=4, conv2_low=0x08),
            {(reg.CNA, reg.CNA_CONV_CON2): 0x10000005,
             (reg.CNA, reg.CNA_WEIGHT_SIZE0): s["out_c"] * s["in_c"] * FP16_BYTES,
             (reg.CNA, reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | s["out_c"],
             (reg.CORE, reg.CORE_DATAOUT_SIZE_1): s["out_c"] - 1,
             (reg.DPU, reg.DATA_CUBE_CHANNEL): ((s["out_c"] - 1) << 16) | (s["out_c"] - 1),
             (reg.DPU, reg.WDMA_SIZE_0): s["out_c"] - 1})
        setup2 = patch_regs(
            _exact11_body_regs(s, "k_tile", 0, s["out_c"], in_dma, wt_dma, out_dma, y_start=4, input_h=3, conv2_low=0x08),
            {(reg.CNA, reg.CNA_CONV_CON2): 0x10000004,
             (reg.CNA, reg.CNA_WEIGHT_SIZE0): s["out_c"] * s["in_c"] * FP16_BYTES,
             (reg.CNA, reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | s["out_c"],
             (reg.CORE, reg.CORE_DATAOUT_SIZE_1): s["out_c"] - 1,
             (reg.DPU, reg.DATA_CUBE_CHANNEL): ((s["out_c"] - 1) << 16) | (s["out_c"] - 1),
             (reg.DPU, reg.WDMA_SIZE_0): s["out_c"] - 1})
        rows = [setup0, setup1, _exact11_aux_regs(s, out_dma),
                setup2, _exact11_aux_regs(s, out_dma)]
        splits = exact_byk_splits(s)
        last_oc_start = splits[-1][0]
        for oc_start, oc_count in splits:
            rows.append(patch_regs(
                _exact11_body_regs(s, "k_tile", oc_start, oc_count, in_dma, wt_dma, out_dma, input_h=7, conv2_low=0x08),
                {(reg.CORE, reg.CORE_DATAOUT_SIZE_1): oc_count - 1,
                 (reg.DPU, reg.DATA_CUBE_CHANNEL): ((oc_count - 1) << 16) | (oc_count - 1),
                 (reg.DPU, reg.WDMA_SIZE_0): oc_count - 1,
                 (reg.DPU, reg.DST_BASE_ADDR): out_dma + oc_start * 0x68}))
            if oc_start != last_oc_start:
                rows.append(_exact11_aux_regs(s, out_dma))
        rows.append(_exact11_aux_regs(s, out_dma))
        if tuple(len(row) for row in rows) != layout["amounts"]:
            raise RuntimeError("c832 h7 exact11 BY_K row amounts changed")
        return rows
    half = _align_up(s["out_c"] // 2, 16)
    rows = [_exact11_body_regs(s, "setup", 0, s["out_c"], in_dma, wt_dma, out_dma),
            _exact11_body_regs(s, "k_half", 0, half, in_dma, wt_dma, out_dma), _exact11_aux_regs(s, out_dma),
            _exact11_body_regs(s, "k_half", half, s["out_c"] - half, in_dma, wt_dma, out_dma), _exact11_aux_regs(s, out_dma)]
    splits = exact_byk_splits(s)
    last_oc_start = splits[-1][0]
    for oc_start, oc_count in splits:
        rows.append(_exact11_body_regs(s, "k_tile", oc_start, oc_count, in_dma, wt_dma, out_dma))
        if oc_start != last_oc_start:
            rows.append(_exact11_aux_regs(s, out_dma))
    rows.append(_exact11_aux_regs(s, out_dma))
    if tuple(len(row) for row in rows) != layout["amounts"]:
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
                          {(reg.CNA, reg.CNA_CBUF_CON0): cbuf0,
                           (reg.CNA, reg.CNA_DATA_SIZE1): POINTWISE_EXACT11_BYK_DATA_SIZE1.get(s["name"], ((s["in_c"] // 2 - 1) << 16) | s["in_c"]),
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

def write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs, layout):
    offsets = layout["offsets"]
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            classes = exact_byk_tail_classes(layout["roles"][idx], idx)
            for i, cls in enumerate(classes):
                if cls == "ZERO_PADDING":
                    qword = 0
                elif cls == "PC_BASE_ADDRESS":
                    qword = E(reg.PC_REG, reg.PC_BASE_ADDRESS, (regcmd_mem.dma_addr + offsets[idx + 1] * 8) & 0xfffffff0)
                elif cls == "PC_REGISTER_AMOUNTS":
                    qword = E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, layout["pc_amounts"][idx])
                elif cls == "VERSION":
                    qword = E(reg.VERSION, 0, 0)
                else:
                    qword = E(reg.PC, reg.OPERATION_ENABLE, layout["masks"][idx])
                regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = layout["masks"][idx]
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
    if s["name"] in CRASH_FENCED_SHAPES:
        raise ValueError("shape is crash-fenced after c256_h2 setup108 generalization reboot; capture exact RKNN GEM1/GEM2 closure before submit")
    rows = planner.descriptor_rows_for_shape(s)
    families = {row["family"] for row in rows}
    if len(rows) == 1 and families == {"setup"} and rows[0]["split_method"] == "NONE":
        if s["name"] in KNOWN_BAD_SPATIAL_SETUP_SHAPES:
            raise ValueError("spatial setup/NONE path is fenced as known numerically wrong for this shape; no allocation or submit attempted")
        if _is_pointwise_wide(s):
            raise ValueError("pointwise-wide NONE is fenced pending RKNN 108-row closure; no allocation or submit attempted")
        return rows
    if rows and rows[0]["split_method"] == "BY_K":
        raise ValueError("BY_K/k_tile is fenced pending RKNN 108/104/26-row closure; no allocation or submit attempted")
    if rows and rows[0]["split_method"] == "BY_Y":
        if _is_pointwise_wide(s) and s["name"] not in LOCAL_TILE_REPLAY_SHAPES and s["name"] not in POINTWISE_CHAINED_Y_SHAPES:
            raise ValueError("pointwise-wide BY_Y is fenced pending proven row closure; no allocation or submit attempted")
        if (s["name"] not in PREFIX_BY_Y_SHAPES and len(families) == 1 and families == {"y_tile"} and
                s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1):
            return rows
        if s["name"] in SETUP2_CLOSURE_SHAPES:
            return rows
        raise ValueError("BY_Y/y_tile is fenced pending RKNN 108-row closure except proven pointwise row tiling; no allocation or submit attempted")
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

def exact_byk_splits(s):
    return KT_TILE_SPLITS.get(s["name"], ((0, 112), (112, 112), (224, 96)))

def exact_byk_layout(s):
    splits = exact_byk_splits(s)
    roles = ["setup_body", "k_half_body0", "aux0", "k_half_body1", "aux1"]
    for idx, _split in enumerate(splits):
        roles.append(f"k_tile_body{idx}")
        if idx + 1 < len(splits):
            roles.append(f"aux{idx + 2}")
    roles.append(f"aux{len(splits) + 1}")
    amounts = tuple(EXACT_BYK_SETUP_AMOUNT if role == "setup_body"
                    else EXACT_BYK_AUX_AMOUNT if role.startswith("aux")
                    else EXACT_BYK_BODY_AMOUNT for role in roles)
    masks = tuple(0x60 if role.startswith("aux") else 0x0d for role in roles)
    pc_amounts = tuple(0 if role.startswith("aux") or role == "setup_body"
                       else 0x1000e if role.startswith("k_half")
                       else 0x2000e for role in roles[:-1])
    offsets = []
    qoff = 0
    for idx, amount in enumerate(amounts):
        offsets.append(qoff)
        if idx + 1 == len(amounts):
            break
        tail_qwords = 4 if idx == 0 else 6 if roles[idx].startswith("aux") else 8
        qoff += _align_up(amount + tail_qwords, 8)
    return {
        "roles": tuple(roles),
        "amounts": amounts,
        "masks": masks,
        "pc_amounts": pc_amounts,
        "offsets": tuple(offsets),
    }

def exact_byk_tail_classes(role, idx):
    if idx == 0:
        return ("ZERO_PADDING", "PC_REGISTER_AMOUNTS", "VERSION", "PC_OPERATION_ENABLE")
    if role.startswith("aux"):
        return ("ZERO_PADDING", "PC_REGISTER_AMOUNTS", "VERSION", "PC_OPERATION_ENABLE", "ZERO_PADDING", "ZERO_PADDING")
    return ("PC_BASE_ADDRESS", "PC_REGISTER_AMOUNTS", "VERSION", "PC_OPERATION_ENABLE",
            "ZERO_PADDING", "ZERO_PADDING", "ZERO_PADDING", "ZERO_PADDING")

def exact_byk_tail_value(cls, layout, idx):
    if cls == "PC_REGISTER_AMOUNTS":
        return layout["pc_amounts"][idx]
    if cls == "PC_OPERATION_ENABLE":
        return layout["masks"][idx]
    return 0

def exact_byk_legacy_layout_check(s):
    layout = exact_byk_layout(s)
    if len(exact_byk_splits(s)) == 3:
        assert layout["amounts"] == EXACT11_BYK_AMOUNTS
        assert layout["masks"] == EXACT11_BYK_MASKS
        assert layout["pc_amounts"] == EXACT11_BYK_PC_AMOUNTS
        assert layout["roles"] == EXACT11_BYK_ROLES
        assert layout["offsets"] == (0, 112, 224, 256, 368, 400, 512, 544, 656, 688, 800)
    elif s["name"] == C576_H19_OC12_EXACT12_SHAPE:
        # 12-task layout: 1 setup + 1 k_half (both 108q) + 5 k_tile (104q) + 5 aux (26q)
        # Build the layout manually (1-k_half + 5-k_tile structure, not 2-k_half default)
        return c576_h19_oc12_exact12_layout()
    return layout


def c576_h19_oc12_exact12_layout():
    """Shape-specific layout for c576_h19_oc12: 1 k_half + 5 k_tiles + 5 aux (12 tasks total)."""
    roles = ("setup_body", "k_half_body0", "k_tile_body0", "aux0",
             "k_tile_body1", "aux1", "k_tile_body2", "aux2",
             "k_tile_body3", "aux3", "k_tile_body4", "aux4")
    amounts = C576_H19_OC12_EXACT12_AMOUNTS
    masks = C576_H19_OC12_EXACT12_MASKS
    # pc_amounts is for the first 11 roles (last role is "aux4" which has no PC header)
    pc_amounts = (0, 0x1000e, 0x2000e, 0, 0x2000e, 0, 0x2000e, 0, 0x2000e, 0, 0x2000e)
    offsets = []
    qoff = 0
    for idx, amount in enumerate(amounts):
        offsets.append(qoff)
        if idx + 1 == len(amounts):
            break
        if idx == 0:
            tail_qwords = 4
        elif roles[idx].startswith("aux"):
            tail_qwords = 6
        elif roles[idx].startswith("k_half"):
            tail_qwords = 4  # k_half has 4-qword tail (PC_BASE_ADDRESS header only)
        else:
            tail_qwords = 8  # k_tile has 8-qword tail
        qoff += _align_up(amount + tail_qwords, 8)
    return {
        "roles": roles,
        "amounts": amounts,
        "masks": masks,
        "pc_amounts": pc_amounts,
        "offsets": tuple(offsets),
    }

def dry_run_exact11_byk(s):
    rows = planner.descriptor_rows_for_shape(s)
    if ((s["name"] not in PREFIX_BY_K_SHAPES and s["name"] not in POINTWISE_EXACT11_BYK_SHAPES) or
            rows[0]["split_method"] != "BY_K"):
        raise ValueError("exact11 BY_K dry-run is scoped only to RKNN-prefix proven shapes")
    layout = exact_byk_legacy_layout_check(s)
    p = _conv_params(s)
    hw_out_fp16 = p["is_spatial"] or s["out_c"] >= 128 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE or _is_pointwise_wide(s)
    c2 = UNPACK_C2 if hw_out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
    input_bytes = _ceil_div(s["in_c"], p["input_pack_c2"]) * p["input_pack_c2"] * s["in_h"] * p["width_stride"] * FP16_BYTES
    weight_bytes = _align_up(s["out_c"], 16) * _align_up(s["in_c"], p["align_c"]) * s["kh"] * s["kw"] * FP16_BYTES
    output_bytes = _ceil_div(p["align_out_c"], c2) * p["out_width_stride"] * c2 * (FP16_BYTES if hw_out_fp16 else FP32_BYTES)
    regcmd_bytes = (layout["offsets"][-1] + layout["amounts"][-1]) * 8
    print(f"dry_run=exact11_byk shape={s['name']} tasks={len(layout['amounts'])} regcmd_bytes={regcmd_bytes} regcmd_alloc={regcmd_alloc_bytes(regcmd_bytes)}")
    print("amounts=" + ",".join(str(v) for v in layout["amounts"]))
    print("masks=" + ",".join(hex(v) for v in layout["masks"]))
    print("offsets=" + ",".join(str(v) for v in layout["offsets"]))
    print("pc_amounts=" + ",".join(hex(v) for v in layout["pc_amounts"]))
    if s["name"] in POINTWISE_EXACT11_BYK_SHAPES:
        roles = ("setup_body", "k_half_body0", "aux0", "k_half_body1", "aux1", "y_tile_body0", "aux2", "y_tile_body1", "aux3", "y_tile_body2", "aux4")
        print("roles=" + ",".join(roles))
        print("y_windows=" + ",".join(f"{y}+{h}:0x{conv2:x}" for y, h, conv2 in POINTWISE_EXACT11_BYK_WINDOWS[s["name"]]))
    else:
        print("roles=" + ",".join(layout["roles"]))
    print("tail_classes=" + "|".join(",".join(exact_byk_tail_classes(role, idx))
                                      for idx, role in enumerate(layout["roles"][:-1])))
    print("tail_values=" + "|".join(",".join(str(exact_byk_tail_value(cls, layout, idx))
                                             for cls in exact_byk_tail_classes(role, idx))
                                     for idx, role in enumerate(layout["roles"][:-1])))
    print(f"bo_bytes=task:{len(layout['amounts']) * ctypes.sizeof(struct_rknpu_task)},regcmd:{regcmd_bytes},"
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
    layout = exact_byk_legacy_layout_check(s)
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    if s["name"] in POINTWISE_EXACT11_COMPACT_WEIGHT_SHAPES:
        weight_flat = _pack_pointwise_wide(wt, s["out_c"], s["in_c"]).view(np.uint16)
    else:
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
        write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs, layout)
        subcores = ((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))
        print(f"exact11_byk_submit tasks={len(layout['amounts'])} submit_task_number=3 amounts=" + ";".join(str(v) for v in layout["amounts"]) +
              " masks=" + ";".join(hex(v) for v in layout["masks"]) + " subcores=(0,1),(0,1),(0,1),(0,0),(0,0)")
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
    print(f"shape={s['name']} guarded=exact11_byk tasks={len(layout['amounts'])} submit_tasks=3 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_exact11_byk_oc=" + ";".join(
            f"{start}:{float(np.max(np.abs(got[:, start:start + 32].astype(np.float32) - expected[:, start:start + 32]))):.4f}"
            for start in range(0, s["out_c"], 32)))
        print(f"debug_exact11_byk_abs got={float(np.max(np.abs(got.astype(np.float32)))):.4f} expected={float(np.max(np.abs(expected))):.4f}")
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")


def run_pointwise_exact11_byk_shape(s):
    if s["name"] not in POINTWISE_EXACT11_BYK_SHAPES:
        raise ValueError("pointwise exact11 BY_K submit is scoped only to RKNN-prefix proven pointwise shapes")
    layout = exact_byk_legacy_layout_check(s)
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    if s["name"] in POINTWISE_EXACT11_COMPACT_WEIGHT_SHAPES:
        weight_flat = _pack_pointwise_wide(wt, s["out_c"], s["in_c"]).view(np.uint16)
    else:
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
        write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs, layout)
        subcores = ((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))
        print(f"pointwise_exact11_byk_submit tasks={len(layout['amounts'])} submit_task_number=3 amounts=" + ";".join(str(v) for v in layout["amounts"]) +
              " masks=" + ";".join(hex(v) for v in layout["masks"]) + " subcores=(0,1),(0,1),(0,1),(0,0),(0,0)")
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
    print(f"shape={s['name']} guarded=pointwise_exact11_byk tasks={len(layout['amounts'])} submit_tasks=3 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_pointwise_exact11_byk_y=" + ";".join(
            f"{y}:{float(np.max(np.abs(got[:, :, y:y + h].astype(np.float32) - expected[:, :, y:y + h]))):.4f}"
            for y, h, _ in POINTWISE_EXACT11_BYK_WINDOWS[s["name"]]))
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def run_pointwise_setup108_compact_weight_shape(s):
    if s["name"] not in POINTWISE_SETUP108_COMPACT_WEIGHT_SHAPES:
        raise ValueError("pointwise setup108 compact-weight submit is scoped only to RKNN-prefix proven shapes")
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = _pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = [_exact11_body_regs(s, "setup", 0, s["out_c"], input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)]
        if len(task_regs[0]) != EXACT_BYK_SETUP_AMOUNT:
            raise RuntimeError("pointwise setup108 row amount changed")
        write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        subcores = ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))
        print(f"pointwise_setup108_compact_weight_submit tasks=1 amount={len(task_regs[0])} weight_bytes={weight_flat.nbytes} subcores=(0,1),(0,0),(0,0),(0,0),(0,0)")
        if npu_submit(fd, task_mem.obj_addr, 1, core_mask=0, subcores=subcores) < 0:
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
    print(f"shape={s['name']} guarded=pointwise_setup108_compact_weight tasks=1 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_pointwise_setup108_compact_weight_ch=" + ";".join(
            f"{idx}:{float(np.max(np.abs(got[:, idx].astype(np.float32) - expected[:, idx]))):.4f}"
            for idx in range(s["out_c"])))
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def _pointwise_exact11_chain_compact_weight_task_regs(s, in_dma, wt_dma, out_dma):
    if s["name"] != "b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid":
        raise ValueError("pointwise exact11 compact-weight chain is scoped only to c256_h3")

    def row(family, y_start, input_h, conv2):
        regs = _exact11_body_regs(s, family, 0, s["out_c"], in_dma, wt_dma, out_dma, y_start, input_h)
        return patch_regs(regs, {
            (reg.CNA, reg.CNA_CONV_CON2): conv2,
            (reg.CNA, reg.CNA_DATA_SIZE1): 0x003f0100,
            (reg.CNA, reg.CNA_CBUF_CON0): 0x0b1,
            (reg.CNA, reg.CNA_CVT_CON0): 0x000b,
            (reg.CNA, reg.CNA_DMA_CON2): 0x0ffffffd,
        })

    aux_dma = wt_dma + 0x3000
    rows = [
        row("setup", 0, 3, 0x00000040),
        row("k_half", 0, 2, 0x10000030), _exact11_aux_regs(s, out_dma, aux_dma),
        row("k_half", 2, 1, 0x10000020), _exact11_aux_regs(s, out_dma, aux_dma),
        row("y_tile", 0, 1, 0x20000020), _exact11_aux_regs(s, out_dma, aux_dma),
        row("y_tile", 1, 1, 0x20000020), _exact11_aux_regs(s, out_dma, aux_dma),
        row("y_tile", 2, 1, 0x20000020), _exact11_aux_regs(s, out_dma, aux_dma),
    ]
    if tuple(len(row_regs) for row_regs in rows) != EXACT11_BYK_AMOUNTS:
        raise RuntimeError("pointwise exact11 compact-weight chain row amounts changed")
    return rows

def _c256_h2_oc64_exact11_task_regs(s, in_dma, wt_dma, out_dma):
    if s["name"] != C256_H2_OC64_EXACT11_SHAPE:
        raise ValueError("c256_h2_oc64 exact11 rows are scoped only to the crash-fenced oc64 shape")

    def row(family, oc_start, oc_count, weight_size0):
        regs = _exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma, input_h=2, conv2_low=0x30)
        return patch_regs(regs, {
            (reg.CNA, reg.CNA_DATA_SIZE1): 0x003f0100,
            (reg.CNA, reg.CNA_CBUF_CON0): 0x0b1,
            (reg.CNA, reg.CNA_CVT_CON0): 0x000b,
            (reg.CNA, reg.CNA_DMA_CON2): 0x0ffffffc,
            (reg.CNA, reg.CNA_WEIGHT_SIZE0): weight_size0,
            (reg.CNA, reg.CNA_WEIGHT_SIZE1): weight_size0 >> 8,
            (reg.CNA, reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | oc_count,
            (reg.CORE, reg.CORE_DATAOUT_SIZE_1): oc_count - 1,
            (reg.DPU, reg.DATA_CUBE_CHANNEL): ((oc_count - 1) << 16) | (oc_count - 1),
            (reg.DPU, reg.WDMA_SIZE_0): oc_count - 1,
        })

    aux_dma = wt_dma + 0x8000
    rows = [
        row("setup", 0, 64, 0x8000),
        row("k_half", 0, 32, 0x4000), _exact11_aux_regs(s, out_dma, aux_dma),
        row("k_half", 32, 32, 0x4000), _exact11_aux_regs(s, out_dma, aux_dma),
        row("k_tile", 0, 32, 0x4000), _exact11_aux_regs(s, out_dma, aux_dma),
        row("k_tile", 32, 16, 0x2000), _exact11_aux_regs(s, out_dma, aux_dma),
        row("k_tile", 48, 16, 0x2000), _exact11_aux_regs(s, out_dma, aux_dma),
    ]
    if tuple(len(row_regs) for row_regs in rows) != EXACT11_BYK_AMOUNTS:
        raise RuntimeError("c256_h2_oc64 exact11 row amounts changed")
    return rows

def dry_run_c256_h2_oc64_exact11(s):
    if s["name"] != C256_H2_OC64_EXACT11_SHAPE:
        raise ValueError("c256_h2_oc64 exact11 dry-run is scoped only to the crash-fenced oc64 shape")
    layout = exact_byk_legacy_layout_check(s)
    in_dma, wt_dma, out_dma = 0x10000000, 0x20000000, 0x30000000
    rows = _c256_h2_oc64_exact11_task_regs(s, in_dma, wt_dma, out_dma)
    families = ("setup", "k_half", "ppu_pdp", "k_half", "ppu_pdp", "k_tile", "ppu_pdp", "k_tile", "ppu_pdp", "k_tile", "ppu_pdp")
    out_c = []
    weight_size0 = []
    dst_offsets = []
    for family, regs_row in zip(families, rows):
        values = {(qword >> 48, qword & 0xffff): (qword >> 16) & 0xffffffff for qword in regs_row}
        if family == "ppu_pdp":
            out_c.append(None)
            weight_size0.append(None)
            dst_offsets.append(None)
            continue
        out_c.append((values[(reg.CORE, reg.CORE_DATAOUT_SIZE_1)] & 0xffff) + 1)
        weight_size0.append(values[(reg.CNA, reg.CNA_WEIGHT_SIZE0)])
        dst_offsets.append(values[(reg.DPU, reg.DST_BASE_ADDR)] - out_dma)
        if values[(reg.CNA, reg.CNA_CBUF_CON0)] != 0x0b1:
            raise RuntimeError("c256_h2_oc64 cbuf0 preflight mismatch")
        if values[(reg.CNA, reg.CNA_DATA_SIZE1)] != 0x003f0100:
            raise RuntimeError("c256_h2_oc64 data_size1 preflight mismatch")
        if values[(reg.CNA, reg.CNA_DMA_CON2)] != 0x0ffffffc:
            raise RuntimeError("c256_h2_oc64 dma2 preflight mismatch")
    if tuple(out_c) != C256_H2_OC64_EXACT11_OUT_C:
        raise RuntimeError("c256_h2_oc64 out_c preflight mismatch")
    if tuple(weight_size0) != C256_H2_OC64_EXACT11_WEIGHT_SIZE0:
        raise RuntimeError("c256_h2_oc64 weight_size0 preflight mismatch")
    if tuple(dst_offsets) != C256_H2_OC64_EXACT11_DST_OFFSETS:
        raise RuntimeError("c256_h2_oc64 dst offset preflight mismatch")
    print(f"dry_run=c256_h2_oc64_exact11 shape={s['name']} status=no_drm_no_submit")
    print("amounts=" + ",".join(str(value) for value in layout["amounts"]))
    print("masks=" + ",".join(hex(value) for value in layout["masks"]))
    print("offsets=" + ",".join(str(value) for value in layout["offsets"]))
    print("families=" + ",".join(families))
    print("out_c=" + ",".join("none" if value is None else str(value) for value in out_c))
    print("weight_size0=" + ",".join("none" if value is None else hex(value) for value in weight_size0))
    print("dst_offsets=" + ",".join("none" if value is None else hex(value) for value in dst_offsets))
    print("consts=cbuf0:0xb1,data_size1:0x003f0100,dma2:0x0ffffffc")

def run_c256_h2_oc64_exact11_shape(s):
    if s["name"] != C256_H2_OC64_EXACT11_SHAPE:
        raise ValueError("c256_h2_oc64 exact11 run is scoped only to the RKNN-proven c256_h2_oc64 shape")
    layout = exact_byk_legacy_layout_check(s)
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = _pack_pointwise_wide(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    regcmd_bytes = (layout["offsets"][-1] + layout["amounts"][-1]) * ctypes.sizeof(ctypes.c_uint64)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, regcmd_alloc_bytes(regcmd_bytes), RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = _c256_h2_oc64_exact11_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs, layout)
        subcores = ((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))
        print(f"c256_h2_oc64_exact11_submit tasks={len(layout['amounts'])} submit_task_number=3 amounts=" + ";".join(str(v) for v in layout["amounts"]) +
              " masks=" + ";".join(hex(v) for v in layout["masks"]) + " subcores=(0,1),(0,1),(0,1),(0,0),(0,0)")
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
    print(f"shape={s['name']} guarded=c256_h2_oc64_exact11 tasks={len(layout['amounts'])} submit_tasks=3 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_c256_h2_oc64_exact11_oc=" + ";".join(
            f"{start}:{float(np.max(np.abs(got[:, start:start + 16].astype(np.float32) - expected[:, start:start + 16]))):.4f}"
            for start in range(0, s["out_c"], 16)))
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")

def _c576_h19_oc12_exact12_task_regs(s, in_dma, wt_dma, out_dma):
    """Emit the 12-task 108/108/104/26... closure for c576_h19_oc12 (Attack H).

    Modeled on c256_h2_oc64 EXACT11 path with 5 k_tiles (k_splits 4+3+2+1+2=12)
    instead of 3. k_half row needs an explicit 4-qword prelude to reach 108q.

    Body field constants are from the c576_h19_oc12_s1pvalid_keep1_gem2 dump
    captured at /home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_oc12_s1pvalid_keep1_gem2/dump_gem2.txt:
    CBUF0=0x1b, DATA_SIZE1=0x3f0240, DMA_CON2=0x011d, CVT_CON0=0x000b.
    """
    if s["name"] != C576_H19_OC12_EXACT12_SHAPE:
        raise ValueError("c576_h19_oc12 exact12 rows are scoped only to the RKNN-proven c576_h19_oc12 shape")

    # Per-row y_start (in rows) and y_count derived from the GEM2 dump.
    # The 7 CNA tasks write 7 distinct (y_count wide x y_count tall) sub-regions
    # of the output, with the 5 k_tiles splitting the (oc, y) plane.
    # FE and DST offsets use y_start * 304 (= in_w * UNPACK_C2 * FP16_BYTES = 19*8*2)
    # which matches the GEM2 dump byte offsets 0/0x1300/0/0xbe0/0/0x850/0xf70.
    # Per-task y_counts (= DS0 w = DATA_CUBE_HEIGHT+1) come from the dump:
    #   setup=16, k_half=3, k_tile 0..4 = 10, 9, 7, 6, 6
    # Per-task CONV_CON2 (FEATURE_GRAINS in bits 0:11, sbsize in bits 16:23) from the dump:
    #   setup=0x80, k_half=0x40, k_tile 0=0x100080, k_tile 1=0x100080,
    #   k_tile 2=0x200070, k_tile 3=0x200060, k_tile 4=0x200060
    per_task_y_offset_rows = (0, 16, 0, 10, 0, 7, 13)
    per_task_y_count = (16, 3, 10, 9, 7, 6, 6)
    per_task_conv2 = (0x80, 0x40, 0x100080, 0x100080, 0x200070, 0x200060, 0x200060)
    p = _conv_params(s)

    def row(family, oc_start, oc_count, cbuf0_val, y_offset=0, y_count=19, conv2=0):
        # y_offset is in ROWS; the byte delta = y_offset * 304 (applied below).
        # y_count overrides the per-row DS0 w and the output DATA_CUBE_HEIGHT.
        # conv2 overrides the per-row CONV_CON2 (FEATURE_GRAINS|sbsize).
        body = _exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma, input_h=19)
        # DS0 w encodes the conv window width (= y_count for this shape).
        # FC_DATA_SIZE0 mirrors DS0 w (used by FC for op 1 = "pointwise").
        ds0_packed = (19 << 16) | y_count
        fc_ds0_packed = (19 << 16) | y_count
        # DATA_CUBE_HEIGHT = y_count - 1; WDMA_SIZE_1 = (h<<16) | (out_w-1)
        cube_h_minus_1 = y_count - 1
        wdma_h = cube_h_minus_1
        wdma_w = 18  # out_w - 1 = 19 - 1
        # Aligned OC = 16 (dump shows DATA_CUBE_CHANNEL = (11<<16)|15 and WDMA_SIZE_0=15)
        aligned_oc_minus_1 = 15
        # weight_bytes_per_kernel = 1*1*576*2 = 1152 = 0x480 (dump WEIGHT_SIZE1)
        weight_bytes_per_kernel = 0x480
        # DATA_SIZE3 = out_w * align_out_c = 19 * 16 = 304 = 0x130 (dump)
        data_size3 = 0x130
        patches = {
            (reg.CNA, reg.CNA_DATA_SIZE0): ds0_packed,
            (reg.CNA, reg.CNA_DATA_SIZE3): data_size3,
            (reg.CNA, reg.CNA_FC_DATA_SIZE0): fc_ds0_packed,
            (reg.CNA, reg.CNA_DATA_SIZE1): C576_H19_OC12_DATA_SIZE1,
            (reg.CNA, reg.CNA_CBUF_CON0): cbuf0_val,
            (reg.CNA, reg.CNA_CVT_CON0): C576_H19_OC12_CVT_CON0,
            (reg.CNA, reg.CNA_DMA_CON2): C576_H19_OC12_DMA_CON2,
            (reg.CNA, reg.CNA_WEIGHT_SIZE0): C576_H19_OC12_K_TILE_WEIGHT_SIZE0,
            (reg.CNA, reg.CNA_WEIGHT_SIZE1): weight_bytes_per_kernel,
            (reg.CNA, reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | oc_count,
            (reg.CORE, reg.CORE_DATAOUT_SIZE_0): ((cube_h_minus_1) << 16) | 18,  # (y_count-1) | (out_w-1) - matches dump
            (reg.CORE, reg.CORE_DATAOUT_SIZE_1): aligned_oc_minus_1,  # 15 not 11 (aligned)
            (reg.DPU, reg.DATA_CUBE_HEIGHT): cube_h_minus_1,
            (reg.DPU, reg.DATA_CUBE_CHANNEL): (aligned_oc_minus_1 << 16) | aligned_oc_minus_1,
            (reg.DPU, reg.WDMA_SIZE_0): aligned_oc_minus_1,
            (reg.DPU, reg.WDMA_SIZE_1): (wdma_h << 16) | wdma_w,
            (reg.DPU, reg.DST_SURF_STRIDE): 0x16c0,  # dump value
        }
        if conv2 != 0:
            # Override CONV_CON2 to match the dump (sbsize + feature_grains encoding)
            patches[(reg.CNA, reg.CNA_CONV_CON2)] = conv2
        if y_offset != 0:
            # y_offset is in ROWS; byte delta = y_offset * 304 = y_offset * in_w * UNPACK_C2 * FP16_BYTES
            byte_delta = y_offset * p["out_w"] * UNPACK_C2 * FP16_BYTES
            patches[(reg.CNA, reg.CNA_FEATURE_DATA_ADDR)] = in_dma + byte_delta
            patches[(reg.DPU, reg.DST_BASE_ADDR)] = out_dma + byte_delta
        return patch_regs(body, patches)

    aux_dma = wt_dma + 0x3600  # weight is 0x3600, aux goes after
    k_half_prelude = (
        E(reg.CNA, reg.CNA_CBUF_CON0, C576_H19_OC12_CBUF0),
        E(reg.CNA, 0x1104, 0),
        E(reg.CNA, 0x1100, 0),
        E(reg.CNA, reg.CNA_CONV_CON1, 0x120),
    )
    # task 0 = setup (y_offset=0, y_count=16, conv2=0x80)
    # task 1 = k_half (y_offset=16, y_count=3, conv2=0x40)
    # tasks 2..11 = (k_tile, aux) pairs; per-task conv2 from per_task_conv2
    rows = [
        row("setup", 0, 12, C576_H19_OC12_CBUF0, y_offset=per_task_y_offset_rows[0], y_count=per_task_y_count[0], conv2=per_task_conv2[0]),  # task 0: 108q
        list(k_half_prelude) + row("k_half", 0, 12, C576_H19_OC12_CBUF0_REUSE, y_offset=per_task_y_offset_rows[1], y_count=per_task_y_count[1], conv2=per_task_conv2[1]),  # task 1: 108q
    ]
    # The k_tiles all use the full 12 OC (per the dump: CORE_DATAOUT_SIZE_1=11 for all 7 CNA tasks).
    # The K_TILE_SPLITS (kept for layout compatibility) is not used for (oc_start, oc_count);
    # the (oc_count, kerns) is always 12 and the y dimension is the only thing that varies per tile.
    for k_tile_idx, _ in enumerate(C576_H19_OC12_K_TILE_SPLITS):
        y_off = per_task_y_offset_rows[2 + k_tile_idx]  # tasks 2..6
        y_cnt = per_task_y_count[2 + k_tile_idx]
        cv2 = per_task_conv2[2 + k_tile_idx]
        rows.append(row("k_tile", 0, 12, C576_H19_OC12_CBUF0, y_offset=y_off, y_count=y_cnt, conv2=cv2))  # 104q k_tile
        rows.append(_exact11_aux_regs(s, out_dma, aux_dma))                                                  # 26q aux
    # Validate amounts match the expected 12-task sequence
    actual_amounts = tuple(len(r) for r in rows)
    if actual_amounts != C576_H19_OC12_EXACT12_AMOUNTS:
        raise RuntimeError(f"c576_h19_oc12 exact12 row amounts mismatch: {actual_amounts} vs {C576_H19_OC12_EXACT12_AMOUNTS}")
    return rows


def dry_run_c576_h19_oc12_exact12(s):
    if s["name"] != C576_H19_OC12_EXACT12_SHAPE:
        raise ValueError("c576_h19_oc12 exact12 dry-run is scoped only to the RKNN-proven c576_h19_oc12 shape")
    in_dma, wt_dma, out_dma = 0x10000000, 0x20000000, 0x30000000
    rows = _c576_h19_oc12_exact12_task_regs(s, in_dma, wt_dma, out_dma)
    families = ("setup", "k_half", "k_tile", "ppu_pdp", "k_tile", "ppu_pdp",
                "k_tile", "ppu_pdp", "k_tile", "ppu_pdp", "k_tile", "ppu_pdp")
    out_c = []
    for family, regs_row in zip(families, rows):
        if family == "ppu_pdp":
            out_c.append(None)
            continue
        values = {(qword >> 48, qword & 0xffff): (qword >> 16) & 0xffffffff for qword in regs_row}
        out_c.append((values[(reg.CORE, reg.CORE_DATAOUT_SIZE_1)] & 0xffff) + 1)
        cbuf0 = values[(reg.CNA, reg.CNA_CBUF_CON0)]
        data_size1 = values[(reg.CNA, reg.CNA_DATA_SIZE1)]
        if cbuf0 not in (C576_H19_OC12_CBUF0, C576_H19_OC12_CBUF0_REUSE):
            raise RuntimeError(f"c576_h19_oc12 cbuf0 preflight mismatch at {family}: {cbuf0:#x}")
        if data_size1 != C576_H19_OC12_DATA_SIZE1:
            raise RuntimeError(f"c576_h19_oc12 data_size1 preflight mismatch at {family}: {data_size1:#x}")
    if tuple(out_c) != C576_H19_OC12_EXACT12_ROW_OUT_C:
        raise RuntimeError(f"c576_h19_oc12 out_c preflight mismatch: {out_c} vs {C576_H19_OC12_EXACT12_ROW_OUT_C}")
    print(f"dry_run=c576_h19_oc12_exact12 shape={s['name']} status=no_drm_no_submit")
    print("amounts=" + ",".join(str(len(r)) for r in rows))
    print("masks=" + ",".join(hex(v) for v in C576_H19_OC12_EXACT12_MASKS))
    print("families=" + ",".join(families))
    print("out_c=" + ",".join("none" if v is None else str(v) for v in out_c))
    print("k_tile_splits=4+3+2+1+2=12 (oc=12)")
    print(f"consts=cbuf0:0x{C576_H19_OC12_CBUF0:x} reuse:0x{C576_H19_OC12_CBUF0_REUSE:x} data_size1:0x{C576_H19_OC12_DATA_SIZE1:x} dma2:0x{C576_H19_OC12_DMA_CON2:x} cvt0:0x{C576_H19_OC12_CVT_CON0:x} weight0:0x{C576_H19_OC12_K_TILE_WEIGHT_SIZE0:x}")


def run_c576_h19_oc12_exact12_shape(s):
    if s["name"] != C576_H19_OC12_EXACT12_SHAPE:
        raise ValueError("c576_h19_oc12 exact12 run is scoped only to the RKNN-proven c576_h19_oc12 shape")
    layout = exact_byk_legacy_layout_check(s)
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = _pack_pointwise_wide(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    regcmd_bytes = (layout["offsets"][-1] + layout["amounts"][-1]) * ctypes.sizeof(ctypes.c_uint64)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, regcmd_alloc_bytes(regcmd_bytes), RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = _c576_h19_oc12_exact12_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs, layout)
        subcores = ((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))
        print(f"c576_h19_oc12_exact12_submit tasks={len(layout['amounts'])} submit_task_number=3 amounts=" + ";".join(str(v) for v in layout["amounts"]) +
              " masks=" + ";".join(hex(v) for v in layout["masks"]) + " subcores=(0,1),(0,1),(0,1),(0,0),(0,0)")
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
    print(f"shape={s['name']} guarded=c576_h19_oc12_exact12 tasks={len(layout['amounts'])} submit_tasks=3 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_c576_h19_oc12_oc=" + ";".join(
            f"{start}:{float(np.max(np.abs(got[:, start:start + 2].astype(np.float32) - expected[:, start:start + 2]))):.4f}"
            for start in range(0, s["out_c"], 2)))
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")




def run_pointwise_exact11_chain_compact_weight_shape(s):
    if s["name"] not in POINTWISE_EXACT11_CHAIN_COMPACT_WEIGHT_SHAPES:
        raise ValueError("pointwise exact11 compact-weight chain is scoped only to RKNN-prefix proven shapes")
    layout = exact_byk_legacy_layout_check(s)
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = _pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    regcmd_bytes = (layout["offsets"][-1] + layout["amounts"][-1]) * ctypes.sizeof(ctypes.c_uint64)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, regcmd_alloc_bytes(regcmd_bytes), RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = _pointwise_exact11_chain_compact_weight_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, task_regs, layout)
        subcores = ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))
        print(f"pointwise_exact11_chain_compact_weight_submit tasks={len(task_regs)} submit_task_number=1 weight_bytes={weight_flat.nbytes} subcores=(0,1),(0,0),(0,0),(0,0),(0,0)")
        if npu_submit(fd, task_mem.obj_addr, 1, core_mask=0, subcores=subcores) < 0:
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
    print(f"shape={s['name']} guarded=pointwise_exact11_chain_compact_weight tasks={len(layout['amounts'])} submit_tasks=1 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_pointwise_exact11_chain_compact_weight_ch=" + ";".join(
            f"{idx}:{float(np.max(np.abs(got[:, idx].astype(np.float32) - expected[:, idx]))):.4f}"
            for idx in range(s["out_c"])))
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
    if s["batch"] * s["groups"] > 256:
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
                hw_oc = out_per_group if out_per_group >= 2 else 2
                tile_shape = dict(s, name=s["name"] + "_groupserial", batch=1, in_c=in_per_group,
                                  out_c=hw_oc, weight_in_c=in_per_group, groups=1)
                tile_p = _conv_params(tile_shape)
                hw_out_fp16 = tile_p["is_spatial"] or tile_shape["out_c"] >= 128 or tile_p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE or _is_pointwise_wide(tile_shape)
                read_dtype = np.float16 if hw_out_fp16 else np.float32
                c2 = UNPACK_C2 if hw_out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
                input_flat = pack_input(inp[n, ic0:ic0 + in_per_group], tile_p).view(np.uint16)
                if hw_oc > out_per_group:
                    tile_wt = np.zeros((hw_oc, s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
                    tile_wt[:out_per_group] = wt[oc0:oc0 + out_per_group]
                else:
                    tile_wt = wt[oc0:oc0 + out_per_group]
                weight_flat = pack_weights(tile_wt, tile_shape, tile_p).view(np.uint16)
                out_count = _ceil_div(tile_p["align_out_c"], c2) * tile_p["out_width_stride"] * c2
                input_offset = _align_up(input_offset, 16)
                weight_offset = _align_up(weight_offset, 16)
                output_offset = _align_up(output_offset, 16)
                ctypes.memmove(input_base + input_offset, input_flat.ctypes.data, input_flat.nbytes)
                ctypes.memmove(weight_base + weight_offset, weight_flat.ctypes.data, weight_flat.nbytes)
                task_regs.append(make_regs(tile_shape, tile_p, input_mem.dma_addr + input_offset,
                                           weight_mem.dma_addr + weight_offset, output_mem.dma_addr + output_offset,
                                           hw_out_fp16))
                task_meta.append((n, oc0, out_count, read_dtype, out_per_group, tile_p, c2, output_offset, hw_oc))
                input_offset += input_flat.nbytes
                weight_offset += weight_flat.nbytes
                output_offset += out_count * np.dtype(read_dtype).itemsize
        write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        if npu_submit(fd, task_mem.obj_addr, len(task_regs)) < 0:
            raise RuntimeError("npu_submit failed")
        output_base = ctypes.addressof(ctypes.c_char.from_buffer(output_map))
        for n, oc0, out_count, read_dtype, tile_out_c, tile_p, c2, offset, hw_oc in task_meta:
            out_raw = np.frombuffer(output_map, dtype=read_dtype, count=out_count, offset=offset).copy()
            got[n, oc0:oc0 + tile_out_c] = unpack_output(out_raw, hw_oc, tile_p["out_h"], tile_p["out_w"], tile_p["out_width_stride"], c2)[:tile_out_c]
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

def run_depthwise_shape(s):
    if not _is_depthwise(s["in_c"], s["out_c"], s["groups"]):
        raise ValueError("depthwise path requires in_c == out_c == groups")
    rows = planner.descriptor_rows_for_shape(s)
    # only setup/NONE rows are safe to submit one-tile-at-a-time; multi-row
    # BY_Y/BY_K/BY_YK depthwise closures still need RKNN 108-row semantics
    if len(rows) > 1 and rows[0]["split_method"] != "NONE":
        if (rows[0]["split_method"] in ("BY_Y", "BY_K", "BY_YK") and
                s["name"] not in DEPTHWISE_BODY_SHAPES and s["name"] not in DEPTHWISE_SETUP108_SHAPES):
            raise ValueError(f"depthwise {rows[0]['split_method']} closure is unfenced without DEPTHWISE_BODY_SHAPES membership; no allocation or submit attempted")
    methods = {row["split_method"] for row in rows}
    families = {row["family"] for row in rows}
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    expected = compute_expected(inp, wt, s)
    p = _conv_params(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, 256 * 1024, RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    try:
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        if s["name"] in DEPTHWISE_SETUP108_SHAPES:
            weight_flat = _pack_depthwise_compact_weight(wt, s["out_c"], s["kh"], s["kw"]).view(np.uint16)
        else:
            weight_flat = pack_weights(wt, s, p).view(np.uint16)
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        input_base = ctypes.addressof(ctypes.c_char.from_buffer(input_map))
        weight_base = ctypes.addressof(ctypes.c_char.from_buffer(weight_map))
        task_regs = []
        if s["name"] in DEPTHWISE_SETUP108_SHAPES:
            input_flat = pack_input(inp[0], p).view(np.uint16)
            (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
            task_regs.append(make_depthwise_setup108_regs(s, p, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr))
        elif len(rows) == 1 and rows[0]["family"] == "setup" and rows[0]["split_method"] == "NONE":
            input_flat = pack_input(inp[0], p).view(np.uint16)
            (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
            task_regs.append(make_regs(s, p, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, True))
        elif methods == {"BY_Y"} and families == {"y_tile"}:
            input_offset = 0
            for row in rows:
                tile_shape = dict(s, in_h=row["input_h"])
                tile_p = _conv_params(tile_shape)
                tile_flat = pack_input(inp[0, :, row["y_start"]:row["y_start"] + row["input_h"], :], tile_p).view(np.uint16)
                ctypes.memmove(input_base + input_offset, tile_flat.ctypes.data, tile_flat.nbytes)
                local_row = {"y_start": row["y_start"], "input_h": row["input_h"], "weight_reuse": bool(task_regs)}
                task_regs.append(make_depthwise_y_tile_regs(s, p, local_row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, input_offset))
                input_offset = _align_up(input_offset + tile_flat.nbytes, 16)
        elif methods == {"BY_K"} and families == {"k_tile"}:
            for row in rows:
                tile_shape = dict(s, out_c=row["oc_count"])
                tile_p = _conv_params(tile_shape)
                tile_wt = np.zeros((row["oc_count"], s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
                tile_wt[:min(s["out_c"], row["oc_count"])] = wt[:min(s["out_c"], row["oc_count"])]
                weight_flat = pack_weights(tile_wt, tile_shape, tile_p).view(np.uint16)
                (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
                input_flat = pack_input(inp[0], p).view(np.uint16)
                (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
                task_regs.append(make_depthwise_k_tile_regs(s, p, row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr))
        else:
            raise ValueError(f"unsupported depthwise layout: methods={methods} families={families} rows={len(rows)}")
        write_tasks(task_map, regcmd_map, regcmd_mem, task_regs)
        if npu_submit(fd, task_mem.obj_addr, len(task_regs)) < 0:
            raise RuntimeError("npu_submit failed")
        out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        got[0] = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], UNPACK_C2)
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                               (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    max_diff = float(np.max(np.abs(got.astype(np.float64) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} depthwise split={rows[0]['split_method']} families={sorted(families)} tasks={len(task_regs)} {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
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
    if s["name"] == "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid":
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
    parser.add_argument("--dry-run-c256-h2-oc64-exact11", action="store_true", help="print c256_h2_oc64 exact-11 metadata without DRM")
    parser.add_argument("--dry-run-c576-h19-oc12-exact12", action="store_true", help="print c576_h19_oc12 exact-12 metadata without DRM")
    parser.add_argument("--dry-run-h160-spatial-by-y", action="store_true", help="print h160 spatial BY_Y executable rows without DRM")
    parser.add_argument("--allow-h160-setup3-submit", action="store_true", help="run the guarded h160 setup3 closure attempt")
    parser.add_argument("--allow-exact11-byk-submit", action="store_true", help="run the guarded exact-11 BY_K h14 attempt")
    parser.add_argument("--allow-pointwise-exact11-byk-submit", action="store_true", help="run the guarded pointwise exact-11 BY_K attempt")
    parser.add_argument("--allow-c256-h2-oc64-exact11-submit", action="store_true", help="run the guarded c256_h2_oc64 exact-11 attempt")
    parser.add_argument("--allow-c576-h19-oc12-exact12-submit", action="store_true", help="run the guarded c576_h19_oc12 exact-12 attempt")
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
        if args.dry_run_c256_h2_oc64_exact11:
            dry_run_c256_h2_oc64_exact11(s)
            return 0
        if args.dry_run_c576_h19_oc12_exact12:
            dry_run_c576_h19_oc12_exact12(s)
            return 0
        if s["name"] == C256_H2_OC64_EXACT11_SHAPE and args.allow_c256_h2_oc64_exact11_submit:
            run_c256_h2_oc64_exact11_shape(s)
            return 0
        if s["name"] == C576_H19_OC12_EXACT12_SHAPE and args.allow_c576_h19_oc12_exact12_submit:
            run_c576_h19_oc12_exact12_shape(s)
            return 0
        if s["name"] in CRASH_FENCED_SHAPES:
            raise ValueError("shape is crash-fenced after c256_h2 setup108 generalization reboot; capture exact RKNN GEM1/GEM2 closure before submit")
        if args.dry_run_exact11_byk:
            dry_run_exact11_byk(s)
        elif args.dry_run_h160_spatial_by_y:
            dry_run_h160_spatial_by_y(s)
        elif args.allow_exact11_byk_submit or s["name"] in PREFIX_BY_K_SHAPES:
            run_exact11_byk_shape(s)
        elif args.allow_pointwise_exact11_byk_submit or s["name"] in POINTWISE_EXACT11_BYK_SHAPES:
            run_pointwise_exact11_byk_shape(s)
        elif s["name"] in POINTWISE_EXACT11_CHAIN_COMPACT_WEIGHT_SHAPES:
            run_pointwise_exact11_chain_compact_weight_shape(s)
        elif s["name"] in POINTWISE_SETUP108_COMPACT_WEIGHT_SHAPES:
            run_pointwise_setup108_compact_weight_shape(s)
        elif s["name"] == H40_SPATIAL_BY_Y_SHAPE:
            run_h40_exact17_shape(s)
        elif args.allow_h160_setup3_submit or s["name"] == H160_SPATIAL_BY_Y_SHAPE:
            run_h160_setup3_shape(s)
        elif s["name"] in GROUPED_SERIAL_SHAPES:
            run_grouped_serial_shape(s)
        elif s["groups"] > 1 and _is_depthwise(s["in_c"], s["out_c"], s["groups"]):
            rows_dw = planner.descriptor_rows_for_shape(s)
            if len(rows_dw) > 1:
                run_depthwise_shape(s)
            else:
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
