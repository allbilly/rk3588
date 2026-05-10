"""
Mesa-Rocket-equivalent convolution in one Python file.

Port of ref/mesa/src/gallium/drivers/rocket/ rkt_ml.c, rkt_coefs.c,
rkt_task.c, rkt_regcmd.c, and rkt_ml.h into Python with NHWC uint8
quantized semantics matching the Teflon/TFLite contract.

Byte-for-byte BO and regcmd dumps against Mesa are the validation
strategy before any numerical comparison.
"""

import ast, ctypes, glob, mmap, os, struct, sys, time
from dataclasses import dataclass, field
from fcntl import ioctl
from math import ceil, log, fabs
import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------
CBUF_BANK_SIZE = 32768
CBUF_BANKS = 12
CBUF_ENTRIES_PER_BANK = 256
CBUF_ENTRY_SIZE = CBUF_BANK_SIZE // CBUF_ENTRIES_PER_BANK
FEATURE_ATOMIC_SIZE = 16
WEIGHT_ATOMIC_SIZE = 32
ATOMIC_K_SIZE = 16
BPE = 1                     # bytes per element for uint8

# ---------------------------------------------------------------------------
# Target IDs (from rkt_regcmd.c emit() -> rkt_get_target(reg) + 0x1)
# ---------------------------------------------------------------------------
TGT_PC      = 0x0081
TGT_PC_REG  = 0x0101
TGT_CNA     = 0x0201
TGT_CORE    = 0x0801
TGT_DPU     = 0x1001
TGT_DPU_RDMA= 0x2001
TGT_VERSION = 0x0041

# ---------------------------------------------------------------------------
# Register addresses (from build/src/gallium/drivers/rocket/rkt_registers.h)
# ---------------------------------------------------------------------------
REG_PC_OPERATION_ENABLE      = 0x0008
REG_PC_BASE_ADDRESS          = 0x0010
REG_PC_REGISTER_AMOUNTS      = 0x0014

REG_CNA_CONV_CON1            = 0x100c
REG_CNA_CONV_CON2            = 0x1010
REG_CNA_CONV_CON3            = 0x1014
REG_CNA_DATA_SIZE0           = 0x1020
REG_CNA_DATA_SIZE1           = 0x1024
REG_CNA_DATA_SIZE2           = 0x1028
REG_CNA_DATA_SIZE3           = 0x102c
REG_CNA_WEIGHT_SIZE0         = 0x1030
REG_CNA_WEIGHT_SIZE1         = 0x1034
REG_CNA_WEIGHT_SIZE2         = 0x1038
REG_CNA_CBUF_CON0            = 0x1040
REG_CNA_CBUF_CON1            = 0x1044
REG_CNA_CVT_CON0             = 0x104c
REG_CNA_CVT_CON1             = 0x1050
REG_CNA_CVT_CON2             = 0x1054
REG_CNA_CVT_CON3             = 0x1058
REG_CNA_CVT_CON4             = 0x105c
REG_CNA_FC_CON0              = 0x1060
REG_CNA_FC_CON1              = 0x1064
REG_CNA_PAD_CON0             = 0x1068
REG_CNA_FEATURE_DATA_ADDR    = 0x1070
REG_CNA_FC_CON2              = 0x1074
REG_CNA_DMA_CON0             = 0x1078
REG_CNA_DMA_CON1             = 0x107c
REG_CNA_DMA_CON2             = 0x1080
REG_CNA_FC_DATA_SIZE0        = 0x1084
REG_CNA_FC_DATA_SIZE1        = 0x1088
REG_CNA_DCOMP_CTRL           = 0x1100
REG_CNA_DCOMP_REGNUM         = 0x1104
REG_CNA_DCOMP_ADDR0          = 0x1110
REG_CNA_DCOMP_AMOUNT0        = 0x1140
REG_CNA_DCOMP_AMOUNT1        = 0x1144
REG_CNA_DCOMP_AMOUNT2        = 0x1148
REG_CNA_DCOMP_AMOUNT3        = 0x114c
REG_CNA_DCOMP_AMOUNT4        = 0x1150
REG_CNA_DCOMP_AMOUNT5        = 0x1154
REG_CNA_DCOMP_AMOUNT6        = 0x1158
REG_CNA_DCOMP_AMOUNT7        = 0x115c
REG_CNA_DCOMP_AMOUNT8        = 0x1160
REG_CNA_DCOMP_AMOUNT9        = 0x1164
REG_CNA_DCOMP_AMOUNT10       = 0x1168
REG_CNA_DCOMP_AMOUNT11       = 0x116c
REG_CNA_DCOMP_AMOUNT12       = 0x1170
REG_CNA_DCOMP_AMOUNT13       = 0x1174
REG_CNA_DCOMP_AMOUNT14       = 0x1178
REG_CNA_DCOMP_AMOUNT15       = 0x117c
REG_CNA_CVT_CON5             = 0x1180
REG_CNA_PAD_CON1             = 0x1184

REG_CORE_MISC_CFG            = 0x3010
REG_CORE_DATAOUT_SIZE_0      = 0x3014
REG_CORE_DATAOUT_SIZE_1      = 0x3018
REG_CORE_CLIP_TRUNCATE       = 0x301c
REG_CORE_RESERVED_3030       = 0x3030   # write 0 to clear stale state

REG_DPU_S_POINTER            = 0x4004
REG_DPU_FEATURE_MODE_CFG     = 0x400c
REG_DPU_DATA_FORMAT          = 0x4010
REG_DPU_OFFSET_PEND          = 0x4014
REG_DPU_DST_BASE_ADDR        = 0x4020
REG_DPU_DST_SURF_STRIDE      = 0x4024
REG_DPU_DATA_CUBE_WIDTH      = 0x4030
REG_DPU_DATA_CUBE_HEIGHT     = 0x4034
REG_DPU_DATA_CUBE_NOTCH_ADDR = 0x4038
REG_DPU_DATA_CUBE_CHANNEL    = 0x403c
REG_DPU_BS_CFG               = 0x4040
REG_DPU_BS_ALU_CFG           = 0x4044
REG_DPU_BS_MUL_CFG           = 0x4048
REG_DPU_BS_RELUX_CMP_VALUE   = 0x404c
REG_DPU_BS_OW_CFG            = 0x4050
REG_DPU_BS_OW_OP             = 0x4054
REG_DPU_WDMA_SIZE_0          = 0x4058
REG_DPU_WDMA_SIZE_1          = 0x405c
REG_DPU_BN_CFG               = 0x4060
REG_DPU_BN_ALU_CFG           = 0x4064
REG_DPU_BN_MUL_CFG           = 0x4068
REG_DPU_BN_RELUX_CMP_VALUE   = 0x406c
REG_DPU_EW_CFG               = 0x4070
REG_DPU_EW_CVT_OFFSET_VALUE  = 0x4074
REG_DPU_EW_CVT_SCALE_VALUE   = 0x4078
REG_DPU_EW_RELUX_CMP_VALUE   = 0x407c
REG_DPU_OUT_CVT_OFFSET       = 0x4080
REG_DPU_OUT_CVT_SCALE        = 0x4084
REG_DPU_OUT_CVT_SHIFT        = 0x4088
REG_DPU_EW_OP_VALUE_0        = 0x4090
REG_DPU_EW_OP_VALUE_1        = 0x4094
REG_DPU_EW_OP_VALUE_2        = 0x4098
REG_DPU_EW_OP_VALUE_3        = 0x409c
REG_DPU_EW_OP_VALUE_4        = 0x40a0
REG_DPU_EW_OP_VALUE_5        = 0x40a4
REG_DPU_EW_OP_VALUE_6        = 0x40a8
REG_DPU_EW_OP_VALUE_7        = 0x40ac
REG_DPU_SURFACE_ADD          = 0x40c0
REG_DPU_LUT_ACCESS_CFG       = 0x40d0
REG_DPU_LUT_ACCESS_DATA      = 0x40d4
REG_DPU_LUT_CFG              = 0x40d8
REG_DPU_LUT_INFO             = 0x40dc
REG_DPU_LUT_LE_START         = 0x40e0
REG_DPU_LUT_LE_END           = 0x40e4
REG_DPU_LUT_LO_START         = 0x40e8
REG_DPU_LUT_LO_END           = 0x40ec
REG_DPU_LUT_LE_SLOPE_SCALE   = 0x40f0
REG_DPU_LUT_LE_SLOPE_SHIFT   = 0x40f4
REG_DPU_LUT_LO_SLOPE_SCALE   = 0x40f8
REG_DPU_LUT_LO_SLOPE_SHIFT   = 0x40fc

REG_DPU_RDMA_RDMA_S_POINTER           = 0x5004
REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH     = 0x500c
REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT    = 0x5010
REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL   = 0x5014
REG_DPU_RDMA_RDMA_SRC_BASE_ADDR       = 0x5018
REG_DPU_RDMA_RDMA_BRDMA_CFG           = 0x501c
REG_DPU_RDMA_RDMA_BS_BASE_ADDR        = 0x5020
REG_DPU_RDMA_RDMA_NRDMA_CFG           = 0x5028
REG_DPU_RDMA_RDMA_BN_BASE_ADDR        = 0x502c
REG_DPU_RDMA_RDMA_ERDMA_CFG           = 0x5034
REG_DPU_RDMA_RDMA_EW_BASE_ADDR        = 0x5038
REG_DPU_RDMA_RDMA_EW_SURF_STRIDE      = 0x5040
REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG    = 0x5044
REG_DPU_RDMA_RDMA_SRC_DMA_CFG         = 0x5048
REG_DPU_RDMA_RDMA_SURF_NOTCH          = 0x504c
REG_DPU_RDMA_RDMA_PAD_CFG             = 0x5064
REG_DPU_RDMA_RDMA_WEIGHT              = 0x5068
REG_DPU_RDMA_RDMA_EW_SURF_NOTCH       = 0x506c

# ---------------------------------------------------------------------------
# Register field macro helpers (transliterated from rkt_registers.h)
# ---------------------------------------------------------------------------
def _field(mask, shift, val):
    return ((val) << shift) & mask

def _mask_hi_lo(hi, lo):
    return ((1 << (hi - lo + 1)) - 1) << lo

def _set_bit(pos, val):
    return (val & 1) << pos

# --- PC ---
PC_OPERATION_ENABLE_RESERVED_0 = lambda v: _field(0xfffffffe, 1, v)
PC_OPERATION_ENABLE_OP_EN      = lambda v: _field(0x00000001, 0, v)
PC_BASE_ADDRESS_PC_SOURCE_ADDR = lambda v: _field(0xfffffff0, 4, v)
PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT = lambda v: _field(0x0000ffff, 0, v)
PC_TASK_CON_TASK_COUNT_CLEAR   = lambda v: _field(0x00002000, 13, v)
PC_TASK_CON_TASK_NUMBER        = lambda v: _field(0x00000fff, 0, v)

# --- CNA ---
CNA_CONV_CON1_NONALIGN_DMA     = lambda v: _field(0x40000000, 30, v)
CNA_CONV_CON1_GROUP_LINE_OFF   = lambda v: _field(0x20000000, 29, v)
CNA_CONV_CON1_ARGB_IN          = lambda v: _field(0x0000f000, 12, v)
CNA_CONV_CON1_PROC_PRECISION   = lambda v: _field(0x00000380, 7, v)
CNA_CONV_CON1_IN_PRECISION     = lambda v: _field(0x00000070, 4, v)
CNA_CONV_CON1_CONV_MODE        = lambda v: _field(0x0000000f, 0, v)
CNA_CONV_CON2_FEATURE_GRAINS   = lambda v: _field(0x00003ff0, 4, v)
CNA_CONV_CON3_CONV_X_STRIDE    = lambda v: _field(0x00000007, 0, v)
CNA_CONV_CON3_CONV_Y_STRIDE    = lambda v: _field(0x00000038, 3, v)
CNA_DATA_SIZE0_DATAIN_WIDTH    = lambda v: _field(0x07ff0000, 16, v)
CNA_DATA_SIZE0_DATAIN_HEIGHT   = lambda v: _field(0x000007ff, 0, v)
CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL = lambda v: _field(0x3fff0000, 16, v)
CNA_DATA_SIZE1_DATAIN_CHANNEL  = lambda v: _field(0x0000ffff, 0, v)
CNA_DATA_SIZE2_DATAOUT_WIDTH   = lambda v: _field(0x000007ff, 0, v)
CNA_DATA_SIZE3_DATAOUT_ATOMICS = lambda v: _field(0x003fffff, 0, v)
CNA_WEIGHT_SIZE0_WEIGHT_BYTES  = lambda v: _field(0xffffffff, 0, v)
CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL = lambda v: _field(0x0007ffff, 0, v)
CNA_WEIGHT_SIZE2_WEIGHT_WIDTH  = lambda v: _field(0x1f000000, 24, v)
CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT = lambda v: _field(0x001f0000, 16, v)
CNA_WEIGHT_SIZE2_WEIGHT_KERNELS= lambda v: _field(0x00003fff, 0, v)
CNA_CBUF_CON0_WEIGHT_BANK      = lambda v: _field(0x000000f0, 4, v)
CNA_CBUF_CON0_DATA_BANK        = lambda v: _field(0x0000000f, 0, v)
CNA_CBUF_CON0_WEIGHT_REUSE     = lambda v: _field(0x00002000, 13, v)
CNA_CBUF_CON1_DATA_ENTRIES     = lambda v: _field(0x00003fff, 0, v)
CNA_CVT_CON0_DATA_SIGN         = lambda v: _field(0x00000008, 3, v)
CNA_CVT_CON0_CVT_TYPE          = lambda v: _field(0x00000002, 1, v)
CNA_CVT_CON0_CVT_BYPASS        = lambda v: _field(0x00000001, 0, v)
CNA_CVT_CON0_CVT_TRUNCATE_0    = lambda v: _field(0x000003f0, 4, v)
CNA_CVT_CON0_CVT_TRUNCATE_1    = lambda v: _field(0x0000fc00, 10, v)
CNA_CVT_CON0_CVT_TRUNCATE_2    = lambda v: _field(0x003f0000, 16, v)
CNA_CVT_CON0_CVT_TRUNCATE_3    = lambda v: _field(0x0fc00000, 22, v)
CNA_CVT_CON1_CVT_SCALE0        = lambda v: _field(0xffff0000, 16, v)
CNA_CVT_CON1_CVT_OFFSET0       = lambda v: _field(0x0000ffff, 0, v)
CNA_CVT_CON2_CVT_SCALE1        = lambda v: _field(0xffff0000, 16, v)
CNA_CVT_CON2_CVT_OFFSET1       = lambda v: _field(0x0000ffff, 0, v)
CNA_CVT_CON3_CVT_SCALE2        = lambda v: _field(0xffff0000, 16, v)
CNA_CVT_CON3_CVT_OFFSET2       = lambda v: _field(0x0000ffff, 0, v)
CNA_CVT_CON4_CVT_SCALE3        = lambda v: _field(0xffff0000, 16, v)
CNA_CVT_CON4_CVT_OFFSET3       = lambda v: _field(0x0000ffff, 0, v)
CNA_DMA_CON0_WEIGHT_BURST_LEN  = lambda v: _field(0x000f0000, 16, v)
CNA_DMA_CON0_DATA_BURST_LEN    = lambda v: _field(0x0000000f, 0, v)
CNA_DMA_CON1_LINE_STRIDE       = lambda v: _field(0x0fffffff, 0, v)
CNA_DMA_CON2_SURF_STRIDE       = lambda v: _field(0x0fffffff, 0, v)
CNA_FC_DATA_SIZE0_DMA_WIDTH    = lambda v: _field(0x3fff0000, 16, v)
CNA_FC_DATA_SIZE0_DMA_HEIGHT   = lambda v: _field(0x000007ff, 0, v)
CNA_FC_DATA_SIZE1_DMA_CHANNEL  = lambda v: _field(0x0000ffff, 0, v)
CNA_PAD_CON0_PAD_LEFT          = lambda v: _field(0x000000f0, 4, v)
CNA_PAD_CON0_PAD_TOP           = lambda v: _field(0x0000000f, 0, v)
CNA_PAD_CON1_PAD_VALUE         = lambda v: _field(0xffffffff, 0, v)
CNA_CVT_CON5_PER_CHANNEL_CVT_EN= lambda v: _field(0xffffffff, 0, v)
CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0 = lambda v: _field(0xffffffff, 0, v)
CNA_DCOMP_REGNUM_DCOMP_REGNUM  = lambda v: _field(0xffffffff, 0, v)

# --- CORE ---
CORE_MISC_CFG_QD_EN            = lambda v: _field(0x00000001, 0, v)
CORE_MISC_CFG_DW_EN            = lambda v: _field(0x00000002, 1, v)
CORE_MISC_CFG_PROC_PRECISION   = lambda v: _field(0x00000700, 8, v)
CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT = lambda v: _field(0xffff0000, 16, v)
CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH  = lambda v: _field(0x0000ffff, 0, v)
CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL= lambda v: _field(0x0000ffff, 0, v)
CORE_CLIP_TRUNCATE_CLIP_TRUNCATE    = lambda v: _field(0x0000001f, 0, v)

# --- DPU ---
DPU_S_POINTER_POINTER_PP_MODE  = lambda v: _field(0x00000008, 3, v)
DPU_S_POINTER_EXECUTER_PP_EN   = lambda v: _field(0x00000004, 2, v)
DPU_S_POINTER_POINTER_PP_EN    = lambda v: _field(0x00000002, 1, v)
DPU_FEATURE_MODE_CFG_BURST_LEN = lambda v: _field(0x000001e0, 5, v)
DPU_FEATURE_MODE_CFG_CONV_MODE = lambda v: _field(0x00000018, 3, v)
DPU_FEATURE_MODE_CFG_OUTPUT_MODE = lambda v: _field(0x00000006, 1, v)
DPU_DATA_FORMAT_OUT_PRECISION  = lambda v: _field(0xe0000000, 29, v)
DPU_DATA_FORMAT_IN_PRECISION   = lambda v: _field(0x1c000000, 26, v)
DPU_DATA_FORMAT_PROC_PRECISION = lambda v: _field(0x00000007, 0, v)
DPU_DST_SURF_STRIDE_DST_SURF_STRIDE = lambda v: _field(0xfffffff0, 4, v)
DPU_DATA_CUBE_WIDTH_WIDTH      = lambda v: _field(0x00001fff, 0, v)
DPU_DATA_CUBE_HEIGHT_HEIGHT    = lambda v: _field(0x00001fff, 0, v)
DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL = lambda v: _field(0x1fff0000, 16, v)
DPU_DATA_CUBE_CHANNEL_CHANNEL  = lambda v: _field(0x00001fff, 0, v)
DPU_BS_CFG_BS_ALU_ALGO         = lambda v: _field(0x000f0000, 16, v)
DPU_BS_CFG_BS_ALU_SRC          = lambda v: _field(0x00000100, 8, v)
DPU_BS_CFG_BS_RELU_BYPASS      = lambda v: _field(0x00000040, 6, v)
DPU_BS_CFG_BS_MUL_BYPASS       = lambda v: _field(0x00000010, 4, v)
DPU_BS_CFG_BS_ALU_BYPASS       = lambda v: _field(0x00000002, 1, v)
DPU_BS_CFG_BS_BYPASS           = lambda v: _field(0x00000001, 0, v)
DPU_BS_OW_CFG_SIZE_E_2         = lambda v: _field(0x00000700, 8, v)
DPU_BS_OW_CFG_SIZE_E_1         = lambda v: _field(0x000000e0, 5, v)
DPU_BS_OW_CFG_SIZE_E_0         = lambda v: _field(0x0000001c, 2, v)
DPU_BS_OW_CFG_OD_BYPASS        = lambda v: _field(0x00000002, 1, v)
DPU_BS_OW_OP_OW_OP             = lambda v: _field(0x0000ffff, 0, v)
DPU_WDMA_SIZE_0_CHANNEL_WDMA   = lambda v: _field(0x00001fff, 0, v)
DPU_WDMA_SIZE_1_HEIGHT_WDMA    = lambda v: _field(0x1fff0000, 16, v)
DPU_WDMA_SIZE_1_WIDTH_WDMA     = lambda v: _field(0x00001fff, 0, v)
DPU_BN_CFG_BN_RELU_BYPASS      = lambda v: _field(0x00000040, 6, v)
DPU_BN_CFG_BN_MUL_BYPASS       = lambda v: _field(0x00000010, 4, v)
DPU_BN_CFG_BN_ALU_BYPASS       = lambda v: _field(0x00000002, 1, v)
DPU_BN_CFG_BN_BYPASS           = lambda v: _field(0x00000001, 0, v)
DPU_EW_CFG_EW_RELU_BYPASS      = lambda v: _field(0x00000200, 9, v)
DPU_EW_CFG_EW_OP_CVT_BYPASS    = lambda v: _field(0x00000100, 8, v)
DPU_EW_CFG_EW_LUT_BYPASS       = lambda v: _field(0x00000080, 7, v)
DPU_EW_CFG_EW_OP_BYPASS        = lambda v: _field(0x00000002, 1, v)
DPU_EW_CFG_EW_BYPASS           = lambda v: _field(0x00000001, 0, v)
DPU_EW_CFG_EW_CVT_TYPE         = lambda v: _field(0x00000400, 10, v)
DPU_EW_CFG_EW_DATA_MODE        = lambda v: _field(0x00000800, 11, v)
DPU_EW_CFG_EDATA_SIZE          = lambda v: _field(0x00001000, 12, v)
DPU_EW_CFG_EW_ALU_ALGO         = lambda v: _field(0x000f0000, 16, v)
DPU_EW_CFG_EW_OP_SRC           = lambda v: _field(0x00100000, 20, v)
DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT = lambda v: _field(0x003f0000, 16, v)
DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE = lambda v: _field(0x0000ffff, 0, v)
DPU_OUT_CVT_SCALE_OUT_CVT_SCALE= lambda v: _field(0x0000ffff, 0, v)
DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT= lambda v: _field(0x00000fff, 0, v)
DPU_SURFACE_ADD_SURF_ADD       = lambda v: _field(0xfffffff0, 4, v)

# --- DPU RDMA ---
DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE  = lambda v: _field(0x00000008, 3, v)
DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN   = lambda v: _field(0x00000004, 2, v)
DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN    = lambda v: _field(0x00000002, 1, v)
DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN = lambda v: _field(0x00007800, 11, v)
DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE  = lambda v: _field(0x00000700, 8, v)
DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE = lambda v: _field(0x00000010, 4, v)
DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE = lambda v: _field(0x00000006, 1, v)
DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE  = lambda v: _field(0xc0000000, 30, v)
DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE  = lambda v: _field(0x0000000c, 2, v)
DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE    = lambda v: _field(0x00000001, 0, v)
DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE   = lambda v: _field(0x0000001e, 1, v)
DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE = lambda v: _field(0xfffffff0, 4, v)
DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR = lambda v: _field(0xfffffff0, 4, v)
DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH= lambda v: _field(0xfffffff0, 4, v)
DPU_RDMA_RDMA_WEIGHT_E_WEIGHT   = lambda v: _field(0xff000000, 24, v)
DPU_RDMA_RDMA_WEIGHT_N_WEIGHT   = lambda v: _field(0x00ff0000, 16, v)
DPU_RDMA_RDMA_WEIGHT_B_WEIGHT   = lambda v: _field(0x0000ff00, 8, v)
DPU_RDMA_RDMA_WEIGHT_M_WEIGHT   = lambda v: _field(0x000000ff, 0, v)
DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH   = lambda v: _field(0x00001fff, 0, v)
DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT = lambda v: _field(0x00001fff, 0, v)
DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL = lambda v: _field(0x00001fff, 0, v)
DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_MODE    = lambda v: _field(0xc0000000, 30, v)
DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SIZE    = lambda v: _field(0x3c000000, 26, v)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ceil_div(x, y):
    return (x + y - 1) // y

def _align(x, a):
    return _ceil_div(x, a) * a

def E(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

def fui(f):
    return struct.unpack('I', struct.pack('f', f))[0]

# ---------------------------------------------------------------------------
# Rocket ioctl helpers (from conv.py)
# ---------------------------------------------------------------------------
DRM_COMMAND_BASE = 0x40

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

def _IOW(type_, nr, size):
    return (1 << 30) | (ord(type_) << 8) | nr | (size << 16)

def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)

DRM_IOCTL_ROCKET_CREATE_BO = _IOWR('d', DRM_COMMAND_BASE + 0x00, ctypes.sizeof(drm_rocket_create_bo))
DRM_IOCTL_ROCKET_SUBMIT    = _IOW('d', DRM_COMMAND_BASE + 0x01, ctypes.sizeof(drm_rocket_submit))
DRM_IOCTL_ROCKET_PREP_BO   = _IOW('d', DRM_COMMAND_BASE + 0x02, ctypes.sizeof(drm_rocket_prep_bo))
DRM_IOCTL_ROCKET_FINI_BO   = _IOW('d', DRM_COMMAND_BASE + 0x03, ctypes.sizeof(drm_rocket_fini_bo))

class RocketBO:
    __slots__ = ("handle", "size", "dma_address", "offset")
    def __init__(self, handle, size, dma_address, offset):
        self.handle = int(handle)
        self.size = int(size)
        self.dma_address = int(dma_address)
        self.offset = int(offset)

def _rocket_mem_allocate(fd, size):
    bo = drm_rocket_create_bo(size=size)
    ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, bo)
    buf = mmap.mmap(fd, bo.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.offset)
    return buf, RocketBO(bo.handle, bo.size, bo.dma_address, bo.offset)

def _rocket_prep_bo(fd, bo, timeout_ns=6_000_000_000):
    if timeout_ns > 0:
        timeout_ns = time.monotonic_ns() + timeout_ns
    ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, drm_rocket_prep_bo(handle=bo.handle, timeout_ns=timeout_ns))

def _rocket_fini_bo(fd, bo):
    ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle=bo.handle))

def _open_rocket_device():
    path = os.environ.get("ROCKET_DEVICE")
    if path:
        return os.open(path, os.O_RDWR)
    candidates = sorted(glob.glob("/dev/accel/accel*")) + sorted(glob.glob("/dev/dri/renderD*"))
    for c in candidates:
        try:
            return os.open(c, os.O_RDWR)
        except OSError:
            pass
    raise FileNotFoundError("No Rocket device found")

# ---------------------------------------------------------------------------
# Data structures matching Mesa struct rkt_operation and struct split_task
# ---------------------------------------------------------------------------
@dataclass
class SplitTask:
    num: int = 0
    top_slice: int = 0
    bottom_slice: int = 0
    num_overlap_slices: int = 0
    num_retain_slices: int = 0
    convolutions: int = 0

    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0

    stride_x: int = 1
    stride_y: int = 1

    input_width: int = 0
    input_height: int = 0
    input_channels: int = 0
    input_channels_real: int = 0
    input_zero_point: int = 0
    input_scale: float = 1.0
    input_data_entries: int = 0
    input_line_stride: int = 0
    input_surface_stride: int = 0
    input_offset: int = 0

    output_width: int = 0
    output_height: int = 0
    output_channels: int = 0
    output_channels_real: int = 0
    output_zero_point: int = 0
    output_scale: float = 1.0
    output_surface_stride: int = 0
    output_offset: int = 0

    weights_width: int = 0
    weights_height: int = 0
    weights_kernels: int = 0
    weights_zero_point: int = 0
    weights_scale: float = 1.0

    input_banks: int = 0
    weights_banks: int = 0

    atomic_count: int = 0
    surfaces_per_row: int = 0

    regcfg_amount: int = 0
    regcfg_addr: int = 0

@dataclass
class Operation:
    depthwise: bool = False
    reuse_weights_cbuf: bool = False
    truncate_bits: int = 0

    padding_top: int = 0
    padding_bottom: int = 0
    padding_left: int = 0
    padding_right: int = 0
    stride: int = 1

    addition_input: bool = False
    add_tensor: int = -1
    addition_offset: int = 0
    addition_scale: float = 1.0

    input_index: int = 0
    input_width: int = 0
    input_height: int = 0
    input_channels: int = 0
    input_zero_point: int = 0
    input_scale: float = 1.0

    output_index: int = 0
    output_width: int = 0
    output_height: int = 0
    output_channels: int = 0
    output_zero_point: int = 0
    output_scale: float = 1.0

    weights_width: int = 0
    weights_height: int = 0
    weights_zero_point: int = 0
    weights_scale: float = 1.0

    tasks: list = field(default_factory=list)

    # BO handles and DMA addresses (set during create_bos)
    input_bo_handle: int = 0
    input_bo_addr: int = 0
    input_bo_size: int = 0
    weight_bo_handle: int = 0
    weight_bo_addr: int = 0
    weight_bo_size: int = 0
    bias_bo_handle: int = 0
    bias_bo_addr: int = 0
    bias_bo_size: int = 0
    output_bo_handle: int = 0
    output_bo_addr: int = 0
    output_bo_size: int = 0
    regcmd_bo_handle: int = 0
    regcmd_bo_addr: int = 0
    regcmd_bo_size: int = 0

# ---------------------------------------------------------------------------
# is_depthwise (rkt_ml.c:rkt_is_depthwise)
# ---------------------------------------------------------------------------
def is_depthwise(in_c, out_c, groups, depthwise_flag):
    return depthwise_flag and in_c > 1 and out_c > 1

def is_depthwise_op(poperation, in_channels, out_channels):
    return poperation.conv.depthwise and in_channels > 1 and out_channels > 1

# ---------------------------------------------------------------------------
# Input packing (rkt_ml.c:rkt_ml_subgraph_invoke input conversion)
# ---------------------------------------------------------------------------
def calc_input_size(input_width, input_height, input_channels):
    """Compute the Rocket tensor BO size for an input tensor."""
    input_channels_1 = _ceil_div(input_channels, FEATURE_ATOMIC_SIZE) * 2
    input_channels_2 = FEATURE_ATOMIC_SIZE
    return input_width * input_height * input_channels_1 * input_channels_2

def pack_input_like_mesa(input_nhwc, input_width, input_height, input_channels,
                         input_zero_point, output_channels, addition_input, add_tensor):
    """
    Convert NHWC uint8 input into Rocket hardware tensor format.
    Mirrors rkt_ml.c::rkt_ml_subgraph_invoke() input conversion.
    """
    feat_size = FEATURE_ATOMIC_SIZE
    raw = np.zeros(calc_input_size(input_width, input_height, input_channels), dtype=np.uint8)

    single_channel = (output_channels == 1 and input_channels == 1
                      and not addition_input and add_tensor == -1)
    if single_channel:
        raw[:input_nhwc.size] = input_nhwc.ravel()
        return raw

    if input_channels == 1:
        n = 0
        for x in range(input_width):
            for y in range(max(input_height, feat_size)):
                if y < input_height:
                    raw[n] = input_nhwc[0, y, x, 0]
                else:
                    raw[n] = input_zero_point
                n += 1
        return raw

    channels_per_atom = feat_size
    n = 0
    for u in range(_ceil_div(input_channels, channels_per_atom)):
        for x in range(input_width):
            for y in range(input_height):
                for c in range(channels_per_atom):
                    ic = c + u * channels_per_atom
                    if ic < input_channels:
                        raw[n] = (int(input_nhwc[0, y, x, ic]) - 0x80) & 0xFF
                    else:
                        raw[n] = (input_zero_point - 0x80) & 0xFF
                    n += 1
    return raw

# ---------------------------------------------------------------------------
# Weight packing (rkt_coefs.c:rkt_fill_weights)
# ---------------------------------------------------------------------------
def calc_weight_size(weights_width, weights_height, input_channels, output_channels,
                     depthwise):
    wc = _align(max(input_channels, FEATURE_ATOMIC_SIZE), WEIGHT_ATOMIC_SIZE)
    oc = _align(output_channels, 2) if not depthwise else 1
    return weights_width * weights_height * oc * wc * 2

def fill_weights_like_mesa(weight_ohwi, weights_width, weights_height,
                           input_channels, output_channels, zero_point, depthwise):
    """
    Pack quantized uint8 weights into Rocket hardware layout.
    Mirrors rkt_coefs.c::rkt_fill_weights().
    """
    ic = max(input_channels, FEATURE_ATOMIC_SIZE)
    oc = _align(output_channels, 2) if not depthwise else 1
    ic_groups = WEIGHT_ATOMIC_SIZE
    if depthwise:
        ic_groups *= 2

    ic1 = _ceil_div(ic, ic_groups)
    ic2 = min(ic, ic_groups)
    oc1 = _ceil_div(oc, WEIGHT_ATOMIC_SIZE)

    out_size = weights_width * weights_height * ic1 * ic_groups * oc1 * WEIGHT_ATOMIC_SIZE
    if depthwise:
        out_size = weights_width * weights_height * ic1 * ic_groups * oc1 * WEIGHT_ATOMIC_SIZE
    if not depthwise:
        out_size = weights_width * weights_height * oc * _align(ic, WEIGHT_ATOMIC_SIZE) * 2

    # Actual Mesa size formula
    wc = _align(ic, WEIGHT_ATOMIC_SIZE)
    total_size = weights_width * weights_height * oc * wc * 2

    out = np.zeros(total_size, dtype=np.uint8)
    n = 0

    for oo in range(_ceil_div(oc, WEIGHT_ATOMIC_SIZE)):
        for ii in range(ic1):
            for x in range(weights_width):
                for y in range(weights_height):
                    for oi in range(min(oc - oo * WEIGHT_ATOMIC_SIZE, WEIGHT_ATOMIC_SIZE)):
                        for ij in range(ic2):
                            oc_idx = oo * WEIGHT_ATOMIC_SIZE + oi
                            ic_idx = ii * ic_groups + ij

                            if output_channels > 2 and oc_idx >= _align(output_channels, 2):
                                continue

                            if oc_idx >= output_channels:
                                out[n] = 0
                                n += 1
                            elif ic_idx >= input_channels:
                                if ij < 16 or (input_channels % 32) > 16:
                                    out[n] = (zero_point - 0x80) & 0xFF
                                    n += 1
                                else:
                                    continue
                            else:
                                val = weight_ohwi[oc_idx, x, y, ic_idx]
                                out[n] = (int(val) - 0x80) & 0xFF
                                n += 1
    return out

# ---------------------------------------------------------------------------
# Bias correction (rkt_coefs.c)
# ---------------------------------------------------------------------------
def calculate_bias_correction(weights_ohwi, weights_width, weights_height,
                              input_channels, output_channels,
                              input_zero_point, weight_zero_point, depthwise):
    correction = np.zeros(output_channels, dtype=np.int64)
    if depthwise:
        for oc in range(output_channels):
            s = 0
            for x in range(weights_width):
                for y in range(weights_height):
                    s += (int(weights_ohwi[0, x, y, oc]) - weight_zero_point) * \
                         (input_zero_point - 0x80)
            correction[oc] = s
    else:
        for oc in range(output_channels):
            s = 0
            for x in range(weights_width):
                for y in range(weights_height):
                    for ic in range(input_channels):
                        s += (int(weights_ohwi[oc, x, y, ic]) - weight_zero_point) * \
                             (input_zero_point - 0x80)
            correction[oc] = s
    return correction

def rkt_fill_biases_like_mesa(biases_in, output_channels, weights_ohwi,
                              weights_width, weights_height, input_channels,
                              input_zero_point, weight_zero_point, depthwise,
                              weight_scale):
    """
    Port of rkt_coefs.c::rkt_fill_biases().
    Returns (biases_packed, truncate_bits).
    """
    correction = calculate_bias_correction(
        weights_ohwi, weights_width, weights_height, input_channels,
        output_channels, input_zero_point, weight_zero_point, depthwise)

    truncate_bits = 0
    scale_bits = fui(weight_scale)
    known_truncate_scales = [
        0x3a88323f, 0x3c0060de, 0x3c06022d, 0x3c1642e3,
        0x3c1e3f51, 0x3c5c8aa8, 0x3c615e93, 0x3c7326a2,
        0x3c783013, 0x3d1748e6, 0x3d282992, 0x3d2e87ae,
        0x3d77f5f6, 0x3a9a5956, 0x3caebc56,
    ]
    if scale_bits in known_truncate_scales:
        truncate_bits = 1

    biases = np.zeros(output_channels, dtype=np.int32)
    for oc in range(output_channels):
        corr = correction[oc]
        biases[oc] = (biases_in[oc] - corr) // (1 << truncate_bits)
    return biases, truncate_bits

# ---------------------------------------------------------------------------
# CBUF calculations (rkt_task.c)
# ---------------------------------------------------------------------------
def calc_line_stride(width):
    return width * ATOMIC_K_SIZE * BPE

def calc_entries_per_slice(input_width, input_channels):
    bpe = BPE
    atomics_per_entry = CBUF_ENTRY_SIZE // FEATURE_ATOMIC_SIZE
    total_c_atomics = _ceil_div(input_channels * bpe, FEATURE_ATOMIC_SIZE)
    last_c_atomics = total_c_atomics % atomics_per_entry
    int_c_entries = (total_c_atomics // atomics_per_entry) * input_width
    if last_c_atomics == 3:
        frac_c_entries = input_width
    else:
        frac_c_entries = _ceil_div(last_c_atomics * input_width, atomics_per_entry)
    return int_c_entries + frac_c_entries

def calc_input_banks(input_width, input_height, input_channels):
    eps = calc_entries_per_slice(input_width, input_channels)
    return _ceil_div(eps * input_height, CBUF_ENTRIES_PER_BANK)

def calc_weights_banks(weights_width, weights_height, input_channels,
                       output_channels, depthwise):
    bpe = BPE
    bytes_ = weights_width * weights_height * input_channels * bpe
    if not depthwise:
        bytes_ *= output_channels
    entries = _ceil_div(bytes_, CBUF_ENTRY_SIZE)
    banks = _ceil_div(entries, CBUF_ENTRIES_PER_BANK)
    banks += 1  # Mesa adds one extra bank
    return banks

# ---------------------------------------------------------------------------
# fill_task (rkt_task.c::fill_task)
# ---------------------------------------------------------------------------
def fill_task(op, task):
    task.stride_x = op.stride
    task.stride_y = op.stride

    task.input_width = op.input_width
    if task.input_width == 8 and (op.addition_input or op.add_tensor != -1):
        task.input_width *= 2

    task.input_height = op.input_height
    task.input_channels = _align(max(op.input_channels, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE)
    task.input_channels_real = op.input_channels
    task.input_zero_point = op.input_zero_point
    task.input_scale = op.input_scale

    task.output_width = op.output_width
    task.output_height = op.output_height
    task.output_channels_real = op.output_channels
    task.output_channels = _align(max(op.output_channels, 32), 32)
    if op.depthwise:
        if task.output_channels_real <= 32:
            task.output_channels *= 2
        task.output_channels = _align(task.output_channels, 64)

    task.output_zero_point = op.output_zero_point
    task.output_scale = op.output_scale

    if task.input_channels_real == 1 and \
       (task.output_channels_real > 1 or op.addition_input or op.add_tensor != -1):
        task.input_width = max(task.input_width, FEATURE_ATOMIC_SIZE)
        task.input_line_stride = max(calc_line_stride(op.input_width) // FEATURE_ATOMIC_SIZE,
                                     FEATURE_ATOMIC_SIZE)
        if op.input_channels == 32 and op.input_width == 80:
            task.input_line_stride *= 4
            task.input_surface_stride = int(task.input_line_stride * (task.input_height / 4 - 1))
        else:
            task.input_surface_stride = int(task.input_line_stride * (task.input_height - 1))
    else:
        task.input_line_stride = calc_line_stride(op.input_width) // 4
        task.input_surface_stride = int(task.input_line_stride * (task.input_height / 4 - 1))

    if task.input_width == 8 and (op.addition_input or op.add_tensor != -1):
        task.input_line_stride //= 2
        task.input_surface_stride = 112

    output_line_stride = calc_line_stride(op.output_width)
    task.output_surface_stride = output_line_stride * task.output_height // FEATURE_ATOMIC_SIZE

    if task.input_channels_real == 1:
        task.input_data_entries = task.input_width * task.input_height
    elif task.input_width == 40 and task.input_channels_real == 40:
        task.input_data_entries = 40
    else:
        task.input_data_entries = _ceil_div(
            task.input_width * 2 * _ceil_div(task.input_channels_real, FEATURE_ATOMIC_SIZE), 8)

    task.weights_width = op.weights_width
    task.weights_height = op.weights_height
    task.weights_zero_point = op.weights_zero_point
    task.weights_scale = op.weights_scale

    task.weights_kernels = 1 if op.depthwise else _align(op.output_channels, 2)

    task.surfaces_per_row = task.output_width * task.output_height * 2
    if op.depthwise:
        task.surfaces_per_row *= 2

# ---------------------------------------------------------------------------
# Task splitting (rkt_task.c::rkt_split_tasks)
# ---------------------------------------------------------------------------
def split_tasks_like_mesa(op):
    op.tasks = []

    entries_per_slice = calc_entries_per_slice(op.input_width, op.input_channels)
    input_banks_required = calc_input_banks(op.input_width, op.input_height, op.input_channels)
    weights_banks_required = calc_weights_banks(
        op.weights_width, op.weights_height, op.input_channels, op.output_channels, op.depthwise)
    available_weights_banks = weights_banks_required
    available_input_banks = CBUF_BANKS - weights_banks_required

    pad_top = op.padding_top
    pad_bottom = op.padding_bottom
    pad_left = op.padding_left
    pad_right = op.padding_right

    if weights_banks_required + 1 < CBUF_BANKS:
        op.reuse_weights_cbuf = True
    else:
        op.reuse_weights_cbuf = False
        available_input_banks = 7
        available_weights_banks = CBUF_BANKS - available_input_banks

    if input_banks_required <= available_input_banks:
        t = SplitTask()
        t.num = 0
        fill_task(op, t)
        t.input_banks = input_banks_required
        t.weights_banks = CBUF_BANKS - t.input_banks
        t.input_height = op.input_height
        t.pad_top = pad_top
        t.pad_bottom = pad_bottom
        t.pad_left = pad_left
        t.pad_right = pad_right
        t.atomic_count = t.output_width * t.output_height
        op.tasks.append(t)
        return

    available_slices = (CBUF_ENTRIES_PER_BANK * available_input_banks) // entries_per_slice

    t = SplitTask()
    t.num = 0
    fill_task(op, t)
    t.input_banks = available_input_banks
    t.weights_banks = available_weights_banks
    t.top_slice = 0
    t.bottom_slice = available_slices - 1
    t.pad_top = pad_top
    t.pad_left = pad_left
    t.pad_right = pad_right
    op.tasks.append(t)

    s = op.weights_height - pad_top - 1
    while s < op.input_height:
        prev = op.tasks[-1]
        while s <= prev.bottom_slice:
            s += op.stride
        if s > prev.bottom_slice:
            s -= op.stride

        t = SplitTask()
        t.num = len(op.tasks)
        fill_task(op, t)
        t.top_slice = min(s, prev.bottom_slice) - (op.weights_height - 1) + op.stride
        t.bottom_slice = t.top_slice + available_slices - 1
        t.pad_left = pad_left
        t.pad_right = pad_right

        if t.bottom_slice >= op.input_height - 1:
            t.bottom_slice = op.input_height - 1
            t.pad_bottom = pad_bottom
            op.tasks.append(t)
            break

        s = t.top_slice + op.weights_height - 1
        op.tasks.append(t)

    last = op.tasks[-1]
    if last.top_slice >= op.input_height or \
       last.bottom_slice >= (op.input_height + pad_bottom):
        op.tasks.pop()

    for i in range(1, len(op.tasks)):
        prev = op.tasks[i - 1]
        cur = op.tasks[i]
        if prev.bottom_slice >= cur.top_slice:
            cur.num_overlap_slices = prev.bottom_slice - cur.top_slice + 1
            prev.num_retain_slices = cur.num_overlap_slices
        else:
            cur.num_overlap_slices = 0
            prev.num_retain_slices = 0

    output_height_processed = 0
    for i, cur in enumerate(op.tasks):
        s = cur.top_slice + (op.weights_height - 1) - cur.pad_top
        conv_count = 0
        while s <= cur.bottom_slice + cur.pad_bottom:
            s += op.stride
            conv_count += 1
        cur.convolutions = conv_count

        cur.bottom_slice = min(cur.bottom_slice, op.input_height - 1)
        cur.input_height = cur.bottom_slice - cur.top_slice + 1

        cur.output_width = (cur.input_width + cur.pad_left + cur.pad_right -
                            op.weights_width) // op.stride + 1
        cur.output_height = (cur.input_height + cur.pad_top + cur.pad_bottom -
                             op.weights_height) // op.stride + 1
        cur.atomic_count = cur.output_width * cur.output_height

        cur.input_offset = calc_line_stride(op.input_width) * cur.top_slice
        cur.output_offset = calc_line_stride(op.output_width) * output_height_processed

        cur.input_banks = available_input_banks
        cur.weights_banks = available_weights_banks

        output_height_processed += cur.output_height

# ---------------------------------------------------------------------------
# Output size (rkt_ml.c::calc_raw_output_size)
# ---------------------------------------------------------------------------
def calc_raw_output_size(output_width, output_height, output_channels):
    oc1 = _ceil_div(output_channels, FEATURE_ATOMIC_SIZE) * 2
    oc2 = FEATURE_ATOMIC_SIZE
    return output_width * output_height * oc1 * oc2

# ---------------------------------------------------------------------------
# Register emission (rkt_regcmd.c::fill_first_regcmd)
# ---------------------------------------------------------------------------
def emit_regcmd_like_mesa(op, task_num, input_phys_addr, output_phys_addr,
                          weight_phys_addr, bias_phys_addr):
    task = op.tasks[task_num]
    num_tasks = len(op.tasks)
    offset = task.output_zero_point - 0x80
    regs = []

    def emit(addr, value):
        target = _get_target(addr) + 0x1
        regs.append(E(target, addr, value))

    def emit_raw(target, addr, value):
        regs.append(E(target, addr, value))

    con0 = (CNA_CBUF_CON0_WEIGHT_BANK(task.weights_banks) |
            CNA_CBUF_CON0_DATA_BANK(task.input_banks))
    if task_num > 0 and op.reuse_weights_cbuf:
        con0 |= CNA_CBUF_CON0_WEIGHT_REUSE(1)

    emit(REG_CNA_CBUF_CON0, con0)
    emit(REG_CNA_DCOMP_REGNUM, CNA_DCOMP_REGNUM_DCOMP_REGNUM(0))
    emit(REG_CNA_DCOMP_CTRL, 0)

    con1 = 0
    if task.input_channels_real == 1:
        con1 |= (CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) |
                 CNA_CONV_CON1_ARGB_IN(8))
    if op.depthwise:
        con1 |= CNA_CONV_CON1_CONV_MODE(3)

    emit(REG_CNA_CONV_CON1, con1)

    emit(REG_DPU_S_POINTER,
         DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) |
         DPU_S_POINTER_POINTER_PP_EN(1))

    emit(REG_DPU_RDMA_RDMA_S_POINTER,
         DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
         DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
         DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1))

    emit(REG_CNA_CONV_CON1, con1)
    emit(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(50 + task.stride_y + 1))
    emit(REG_CNA_CONV_CON3,
         CNA_CONV_CON3_CONV_X_STRIDE(task.stride_x) |
         CNA_CONV_CON3_CONV_Y_STRIDE(task.stride_y))
    emit(REG_CNA_DATA_SIZE0,
         CNA_DATA_SIZE0_DATAIN_WIDTH(task.input_width) |
         CNA_DATA_SIZE0_DATAIN_HEIGHT(task.input_height))
    emit(REG_CNA_DATA_SIZE1,
         CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(task.input_channels_real - 1) |
         CNA_DATA_SIZE1_DATAIN_CHANNEL(task.input_channels))
    emit(REG_CNA_DATA_SIZE2,
         CNA_DATA_SIZE2_DATAOUT_WIDTH(task.output_width))
    emit(REG_CNA_DATA_SIZE3,
         CNA_DATA_SIZE3_DATAOUT_ATOMICS(task.atomic_count))
    emit(REG_CNA_WEIGHT_SIZE0,
         CNA_WEIGHT_SIZE0_WEIGHT_BYTES(
             task.weights_width * task.weights_height *
             task.input_channels * task.weights_kernels))
    emit(REG_CNA_WEIGHT_SIZE1,
         CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(
             task.weights_width * task.weights_height * task.input_channels))
    emit(REG_CNA_WEIGHT_SIZE2,
         CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(task.weights_width) |
         CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(task.weights_height) |
         CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(task.weights_kernels))

    emit(REG_CNA_CBUF_CON0, con0)
    emit(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(task.input_data_entries))

    if task.input_channels_real == 1:
        trunc = 14
        scale = 16384
        coff = 65408

        if op.addition_input or op.add_tensor != -1:
            trunc = 15
            scale = 32388

        emit(REG_CNA_CVT_CON0,
             CNA_CVT_CON0_CVT_TRUNCATE_3(trunc) |
             CNA_CVT_CON0_CVT_TRUNCATE_2(trunc) |
             CNA_CVT_CON0_CVT_TRUNCATE_1(trunc) |
             CNA_CVT_CON0_CVT_TRUNCATE_0(trunc))
        emit(REG_CNA_CVT_CON1,
             CNA_CVT_CON1_CVT_SCALE0(scale) | CNA_CVT_CON1_CVT_OFFSET0(coff))
        emit(REG_CNA_CVT_CON2,
             CNA_CVT_CON2_CVT_SCALE1(scale) | CNA_CVT_CON2_CVT_OFFSET1(coff))
        emit(REG_CNA_CVT_CON3,
             CNA_CVT_CON3_CVT_SCALE2(scale) | CNA_CVT_CON3_CVT_OFFSET2(coff))
        emit(REG_CNA_CVT_CON4,
             CNA_CVT_CON4_CVT_SCALE3(scale) | CNA_CVT_CON4_CVT_OFFSET3(coff))
    else:
        emit(REG_CNA_CVT_CON0,
             CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) |
             CNA_CVT_CON0_CVT_BYPASS(1))
        emit(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1))
        emit(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1))
        emit(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1))
        emit(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1))

    emit(REG_CNA_FC_CON0, 0)
    emit(REG_CNA_FC_CON1, 0)
    emit(REG_CNA_PAD_CON0,
         CNA_PAD_CON0_PAD_LEFT(task.pad_left) |
         CNA_PAD_CON0_PAD_TOP(task.pad_top))

    feat_addr = input_phys_addr + task.input_offset
    emit(REG_CNA_FEATURE_DATA_ADDR, feat_addr & 0xFFFFFFFF)

    emit(REG_CNA_FC_CON2, 0)
    emit(REG_CNA_DMA_CON0,
         CNA_DMA_CON0_WEIGHT_BURST_LEN(15) |
         CNA_DMA_CON0_DATA_BURST_LEN(15))
    emit(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(task.input_line_stride))
    emit(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(task.input_surface_stride))

    emit(REG_CNA_FC_DATA_SIZE0,
         CNA_FC_DATA_SIZE0_DMA_WIDTH(op.input_width) |
         CNA_FC_DATA_SIZE0_DMA_HEIGHT(task.input_height))
    emit(REG_CNA_FC_DATA_SIZE1,
         CNA_FC_DATA_SIZE1_DMA_CHANNEL(task.input_channels))
    emit(REG_CNA_DCOMP_CTRL, 0)
    emit(REG_CNA_DCOMP_REGNUM, 0)
    emit(REG_CNA_DCOMP_ADDR0, CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0(weight_phys_addr & 0xFFFFFFFF))

    for regoff in [0x40, 0x44, 0x48, 0x4c, 0x50, 0x54, 0x58, 0x5c,
                   0x60, 0x64, 0x68, 0x6c, 0x70, 0x74, 0x78, 0x7c]:
        emit(0x1110 + regoff, 0)

    if task.input_channels_real == 1:
        emit(REG_CNA_CVT_CON5, CNA_CVT_CON5_PER_CHANNEL_CVT_EN(65535))
    else:
        emit(REG_CNA_CVT_CON5, CNA_CVT_CON5_PER_CHANNEL_CVT_EN(0))

    w3 = task.weights_width >= 3
    izp = task.input_zero_point
    if w3 and izp == 0:
        pad_con1 = 0xffff8080
    else:
        pad_con1 = izp - 0x80

    if op.addition_input or op.add_tensor != -1:
        pad_con1 = 0xffffff80

    if op.depthwise and izp == 0x8b:
        pad_con1 = 0x0b0b

    emit(REG_CNA_PAD_CON1, CNA_PAD_CON1_PAD_VALUE(pad_con1))

    misc_cfg = CORE_MISC_CFG_QD_EN(1)
    if op.depthwise:
        misc_cfg |= CORE_MISC_CFG_DW_EN(1)
    emit(REG_CORE_MISC_CFG, misc_cfg)
    emit(REG_CORE_DATAOUT_SIZE_0,
         CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(task.output_height - 1) |
         CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(task.output_width - 1))
    emit(REG_CORE_DATAOUT_SIZE_1,
         CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(task.output_channels - 1))
    emit(REG_CORE_CLIP_TRUNCATE,
         CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(op.truncate_bits))
    emit_raw(TGT_CORE | 0x1, 0x3030, 0)

    feat_mode_cfg = (DPU_FEATURE_MODE_CFG_BURST_LEN(15) |
                     DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2))
    if op.depthwise:
        feat_mode_cfg |= DPU_FEATURE_MODE_CFG_CONV_MODE(3)
    emit(REG_DPU_FEATURE_MODE_CFG, feat_mode_cfg)

    emit(REG_DPU_DATA_FORMAT, 0)
    emit(REG_DPU_OFFSET_PEND, 0)

    dst_addr = output_phys_addr + task.output_offset
    emit(REG_DPU_DST_BASE_ADDR, dst_addr & 0xFFFFFFFF)
    emit(REG_DPU_DST_SURF_STRIDE,
         DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(task.output_surface_stride))
    emit(REG_DPU_DATA_CUBE_WIDTH,
         DPU_DATA_CUBE_WIDTH_WIDTH(task.output_width - 1))
    emit(REG_DPU_DATA_CUBE_HEIGHT,
         DPU_DATA_CUBE_HEIGHT_HEIGHT(task.output_height - 1))
    emit(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0)
    emit(REG_DPU_DATA_CUBE_CHANNEL,
         DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(task.output_channels_real - 1) |
         DPU_DATA_CUBE_CHANNEL_CHANNEL(task.output_channels - 1))

    emit(REG_DPU_BS_CFG,
         DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) |
         DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1))
    emit(REG_DPU_BS_ALU_CFG, 0)
    emit(REG_DPU_BS_MUL_CFG, 0)
    emit(REG_DPU_BS_RELUX_CMP_VALUE, 0)

    if op.depthwise:
        emit(REG_DPU_BS_OW_CFG,
             DPU_BS_OW_CFG_SIZE_E_2(3) | DPU_BS_OW_CFG_SIZE_E_1(3) |
             DPU_BS_OW_CFG_SIZE_E_0(3))
    else:
        emit(REG_DPU_BS_OW_CFG,
             DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) |
             DPU_BS_OW_CFG_SIZE_E_0(1))

    emit(REG_DPU_BS_OW_OP, DPU_BS_OW_OP_OW_OP(0x80 - task.weights_zero_point))

    emit(REG_DPU_WDMA_SIZE_0,
         DPU_WDMA_SIZE_0_CHANNEL_WDMA(task.output_channels - 1))
    emit(REG_DPU_WDMA_SIZE_1,
         DPU_WDMA_SIZE_1_HEIGHT_WDMA(task.output_height - 1) |
         DPU_WDMA_SIZE_1_WIDTH_WDMA(task.output_width - 1))
    emit(REG_DPU_BN_CFG,
         DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
         DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1))
    emit(REG_DPU_BN_ALU_CFG, 0)
    emit(REG_DPU_BN_MUL_CFG, 0)
    emit(REG_DPU_BN_RELUX_CMP_VALUE, 0)

    if op.add_tensor != -1:
        emit(REG_DPU_EW_CFG,
             DPU_EW_CFG_EW_CVT_TYPE(1) | DPU_EW_CFG_EW_DATA_MODE(1) |
             DPU_EW_CFG_EDATA_SIZE(1) | DPU_EW_CFG_EW_ALU_ALGO(2) |
             DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) |
             DPU_EW_CFG_EW_OP_SRC(1))
        emit(REG_DPU_EW_CVT_OFFSET_VALUE, op.addition_offset)

        add_scale_map = {
            0.090192: 299.671889248, 0.399250: 1326.499209406,
            0.364902: 780.34375,     0.422037: 715.5625,
            0.213016: 564.6875,      0.244231: 499.796875,
            0.283416: 488.203125,    0.171151: 602.90625,
            0.164588: 271.921875,    0.204098: 262.90625,
            0.116532: 450.140625,    0.134499: 212.1953125,
            0.220141: 368.28125,     0.094560: 416.421875,
            0.093230: 305.421875,    0.100618: 313.671875,
        }
        add_scale = 0.0
        for k, v in add_scale_map.items():
            if fabs(op.addition_scale - k) < 0.00001:
                add_scale = v
                break

        add_scale_bits = fui(add_scale)
        add_shift = 127 + 31 - 32 - (add_scale_bits >> 23) + 16
        scale = ((add_scale_bits >> 9) & 0x7fff)
        if scale < 1 << 14:
            scale |= 1 << 14

        emit(REG_DPU_EW_CVT_SCALE_VALUE,
             DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT(add_shift - 1) |
             DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(scale))
        emit(REG_DPU_EW_RELUX_CMP_VALUE, 0x0)

        add_scale_out = {
            0.213016: (0x4, 25914, 24), 0.244231: (0x1, 28927, 24),
            0.283416: (0x6, 26050, 24), 0.171151: (0xfffffffd, 28937, 24),
            0.164588: (0x1, 24877, 23), 0.204098: (0x0, 23272, 23),
            0.116532: (0xfffffff8, 32292, 24), 0.134499: (0xfffffffb, 24153, 23),
            0.220141: (0xb, 27655, 24), 0.094560: (0x5, 20432, 23),
            0.093230: (0xffffffff, 25449, 23), 0.100618: (offset, 16874, 23),
            0.422037: (0x1, 22559, 24), 0.364902: (0x4, 18589, 24),
        }
        out_off, out_scale, out_shift = 0x6, 27676, 25
        for k, (o, s, sh) in add_scale_out.items():
            if fabs(op.addition_scale - k) < 0.00001:
                out_off, out_scale, out_shift = o, s, sh
                break
        emit(REG_DPU_OUT_CVT_OFFSET, out_off)
        emit(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(out_scale))
        emit(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(out_shift))
    else:
        emit(REG_DPU_EW_CFG,
             DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
             DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
             DPU_EW_CFG_EW_BYPASS(1))
        emit(REG_DPU_EW_CVT_OFFSET_VALUE, 0)
        emit(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1))
        emit(REG_DPU_EW_RELUX_CMP_VALUE, 0)
        emit(REG_DPU_OUT_CVT_OFFSET, offset)

        conv_scale = (task.input_scale * task.weights_scale) / task.output_scale
        scale_bits = fui(conv_scale)
        shift = 127 + 31 - 32 - (scale_bits >> 23) + 16
        if op.truncate_bits > 0:
            shift -= 1
        scale = ((scale_bits >> 9) & 0x7fff) + 1
        if scale < 1 << 14:
            scale |= 1 << 14

        emit(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale))
        emit(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1))

    for r in (REG_DPU_EW_OP_VALUE_0, REG_DPU_EW_OP_VALUE_1, REG_DPU_EW_OP_VALUE_2,
              REG_DPU_EW_OP_VALUE_3, REG_DPU_EW_OP_VALUE_4, REG_DPU_EW_OP_VALUE_5,
              REG_DPU_EW_OP_VALUE_6, REG_DPU_EW_OP_VALUE_7):
        emit(r, 0)

    emit(REG_DPU_SURFACE_ADD,
         DPU_SURFACE_ADD_SURF_ADD(task.surfaces_per_row))
    emit_raw(TGT_DPU | 0x1, 0x40c4, 0)
    emit(REG_DPU_LUT_ACCESS_CFG, 0)
    emit(REG_DPU_LUT_ACCESS_DATA, 0)
    emit(REG_DPU_LUT_CFG, 0)
    emit(REG_DPU_LUT_INFO, 0)
    emit(REG_DPU_LUT_LE_START, 0)
    emit(REG_DPU_LUT_LE_END, 0)
    emit(REG_DPU_LUT_LO_START, 0)
    emit(REG_DPU_LUT_LO_END, 0)
    emit(REG_DPU_LUT_LE_SLOPE_SCALE, 0)
    emit(REG_DPU_LUT_LE_SLOPE_SHIFT, 0)
    emit(REG_DPU_LUT_LO_SLOPE_SCALE, 0)
    emit(REG_DPU_LUT_LO_SLOPE_SHIFT, 0)

    emit(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
         DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(task.output_width - 1))
    emit(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
         DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(task.output_height - 1))
    emit(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
         DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(task.output_channels - 1))

    if op.add_tensor != -1:
        emit(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
             (output_phys_addr + task.output_offset) & 0xFFFFFFFF)
    else:
        emit(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0)

    emit(REG_DPU_RDMA_RDMA_BRDMA_CFG,
         DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(1))
    emit(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, bias_phys_addr & 0xFFFFFFFF)
    emit(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0)
    emit(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0)

    ew_stride = max(op.output_width * op.output_height, 12)
    if op.add_tensor != -1:
        emit(REG_DPU_RDMA_RDMA_ERDMA_CFG,
             DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) |
             DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(1))
        ew_base_offset = op.output_width * op.output_height * ATOMIC_K_SIZE
        emit(REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
             ((output_phys_addr + task.output_offset + ew_base_offset) & 0xFFFFFFFF))
        emit(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
             DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(ew_stride))
    else:
        emit(REG_DPU_RDMA_RDMA_ERDMA_CFG,
             DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1))
        emit(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
        emit(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0)

    rdma_feat = DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15)
    if op.add_tensor != -1:
        rdma_feat |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(5)
    else:
        rdma_feat |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1)
    if op.depthwise:
        rdma_feat |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(3)

    emit(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feat)
    emit(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0)

    surf_notch = ew_stride + task.output_width * (op.output_height - task.output_height)
    if op.input_width == 3:
        surf_notch = 15

    if op.add_tensor != -1:
        emit(REG_DPU_RDMA_RDMA_SURF_NOTCH,
             DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(surf_notch))
    else:
        emit(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0)

    emit(REG_DPU_RDMA_RDMA_PAD_CFG, 0)
    emit(REG_DPU_RDMA_RDMA_WEIGHT,
         DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
         DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1))

    if op.add_tensor != -1:
        emit(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
             DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(surf_notch))
    else:
        emit(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0x0)

    if num_tasks == 1:
        regs.append(0x0)  # placeholder for single-task mode
    else:
        emit(REG_PC_BASE_ADDRESS, 0)  # placeholder, patched later
    emit(REG_PC_REGISTER_AMOUNTS, 0)  # placeholder, patched later
    regs.append(E(TGT_VERSION, 0, 0))  # version sentinel
    emit_raw(TGT_PC, REG_PC_OPERATION_ENABLE,
             PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1))

    return regs

def _get_target(reg_addr):
    """Determine target from register address, matching rkt_get_target()."""
    if reg_addr < 0x1000:
        return 0x100  # PC
    elif reg_addr < 0x2000:
        return 0x200  # CNA
    elif reg_addr < 0x3000:
        return 0x200  # CNA (upper CNA regs)
    elif reg_addr < 0x4000:
        return 0x800  # CORE
    elif reg_addr < 0x5000:
        return 0x1000  # DPU
    elif reg_addr < 0x6000:
        return 0x2000  # DPU_RDMA
    else:
        return 0x400  # default

# ---------------------------------------------------------------------------
# PC patching and compilation (rkt_ml.c::compile_operation)
# ---------------------------------------------------------------------------
def compile_operation_like_mesa(op, regcmd_phys_addr):
    num_tasks = len(op.tasks)
    regcmds = []
    for i in range(num_tasks):
        regs = emit_regcmd_like_mesa(op, i,
                                     op.input_bo_addr, op.output_bo_addr,
                                     op.weight_bo_addr, op.bias_bo_addr)
        regcmds.append(regs)

    regcmd_offset = 0
    for i in range(num_tasks):
        size_qwords = len(regcmds[i])
        if i < num_tasks - 1:
            next_size = len(regcmds[i + 1])
            next_addr = regcmd_phys_addr + regcmd_offset + _align(size_qwords * 8, 64)
            regs_to_fetch = next_size - 4
            regs_to_fetch = _align(regs_to_fetch // 2, 2)
            regcmds[i][-4] = E(TGT_PC_REG, REG_PC_BASE_ADDRESS,
                               (next_addr & 0xFFFFFFF0))
            regcmds[i][-3] = E(TGT_PC_REG, REG_PC_REGISTER_AMOUNTS,
                               PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(regs_to_fetch))

        op.tasks[i].regcfg_amount = size_qwords
        op.tasks[i].regcfg_addr = regcmd_phys_addr + regcmd_offset
        regcmd_offset += _align(size_qwords * 8, 64)

    return regcmds

# ---------------------------------------------------------------------------
# Output conversion (rkt_ml.c::rkt_ml_subgraph_read_outputs)
# ---------------------------------------------------------------------------
def read_output_like_mesa(raw_output, output_width, output_height, output_channels):
    raw = raw_output.reshape(-1, output_width or 1, output_height or 1, FEATURE_ATOMIC_SIZE)
    out = np.zeros((output_height, output_width, output_channels), dtype=np.uint8)
    for oc in range(output_channels):
        c = oc % FEATURE_ATOMIC_SIZE
        g = oc // FEATURE_ATOMIC_SIZE
        for y in range(output_height):
            for x in range(output_width):
                out[y, x, oc] = (int(raw[g, x, y, c]) + 0x80) & 0xFF
    return out

# ---------------------------------------------------------------------------
# Submit (rkt_ml.c::rkt_ml_subgraph_invoke)
# ---------------------------------------------------------------------------
def submit_operation_like_mesa(fd, op, input_bo, weight_bo, bias_bo, output_bo,
                               regcmd_bo, regcmds):
    num_tasks = len(op.tasks)
    # include regcmd, weight, and optional bias in input handles
    num_inputs = 1  # input tensor
    extra_in = [regcmd_bo, weight_bo, bias_bo]
    if op.add_tensor != -1:
        extra_in.append(output_bo)
    num_inputs += len(extra_in)

    in_handles = (ctypes.c_uint32 * num_inputs)()
    in_handles[0] = input_bo.handle
    for j, bo in enumerate(extra_in):
        in_handles[1 + j] = bo.handle
    out_handles = (ctypes.c_uint32 * 1)(output_bo.handle)

    # fini_bo on all BOs -> makes them IDLE for kernel submission
    for bo in (regcmd_bo, input_bo, weight_bo, bias_bo, output_bo):
        try:
            _rocket_fini_bo(fd, bo)
        except OSError:
            pass

    if op.reuse_weights_cbuf:
        tasks = (drm_rocket_task * num_tasks)()
        for i in range(num_tasks):
            tasks[i].regcmd = op.tasks[i].regcfg_addr & 0xFFFFFFFF
            tasks[i].regcmd_count = op.tasks[i].regcfg_amount
        job = drm_rocket_job(
            tasks=ctypes.addressof(tasks),
            in_bo_handles=ctypes.addressof(in_handles) if num_inputs > 0 else 0,
            out_bo_handles=ctypes.addressof(out_handles),
            task_count=num_tasks,
            task_struct_size=ctypes.sizeof(drm_rocket_task),
            in_bo_handle_count=num_inputs,
            out_bo_handle_count=1,
        )
        jobs_array = (drm_rocket_job * 1)(job)
        submit = drm_rocket_submit(
            jobs=ctypes.addressof(jobs_array),
            job_count=1,
            job_struct_size=ctypes.sizeof(drm_rocket_job),
        )
        ret = ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, submit)
        _rocket_prep_bo(fd, output_bo)
        return ret
    else:
        # Submit one task per job (parallelism)
        job_refs = []
        for i in range(num_tasks):
            ktask = (drm_rocket_task * 1)()
            ktask[0].regcmd = op.tasks[i].regcfg_addr & 0xFFFFFFFF
            ktask[0].regcmd_count = op.tasks[i].regcfg_amount
            job = drm_rocket_job(
                tasks=ctypes.addressof(ktask),
                in_bo_handles=ctypes.addressof(in_handles) if num_inputs > 0 else 0,
                out_bo_handles=ctypes.addressof(out_handles),
                task_count=1,
                task_struct_size=ctypes.sizeof(drm_rocket_task),
                in_bo_handle_count=num_inputs,
                out_bo_handle_count=1,
            )
            job_refs.append((ktask, job))
        # Create a single submit with all jobs
        jobs_array = (drm_rocket_job * num_tasks)()
        for i, (_, j) in enumerate(job_refs):
            jobs_array[i] = j
        submit = drm_rocket_submit(
            jobs=ctypes.addressof(jobs_array),
            job_count=num_tasks,
            job_struct_size=ctypes.sizeof(drm_rocket_job),
        )
        ret = ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, submit)
        _rocket_prep_bo(fd, output_bo)
        return ret

# ---------------------------------------------------------------------------
# Create operation from TFLite-like arguments
# ---------------------------------------------------------------------------
def create_operation_from_args(input_nhwc, weight_ohwi, biases_in,
                                input_zero_point, input_scale,
                                weight_zero_point, weight_scale,
                                output_zero_point, output_scale,
                                pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
                                stride=1, depthwise=False,
                                add_tensor=-1, addition_input=False,
                                addition_offset=0, addition_scale=1.0):
    """
    Create an Operation from NHWC uint8 tensors.
    Mirrors rkt_ml.c::lower_convolution().
    """
    _, in_h, in_w, in_c = input_nhwc.shape
    oc, kw, kh, ic = weight_ohwi.shape
    output_channels = ic if depthwise else oc

    op = Operation(
        depthwise=depthwise,
        padding_top=pad_top, padding_bottom=pad_bottom,
        padding_left=pad_left, padding_right=pad_right,
        stride=stride,
        input_width=in_w, input_height=in_h, input_channels=in_c,
        input_zero_point=input_zero_point, input_scale=input_scale,
        output_width=(in_w + pad_left + pad_right - kw) // stride + 1,
        output_height=(in_h + pad_top + pad_bottom - kh) // stride + 1,
        output_channels=output_channels,
        output_zero_point=output_zero_point, output_scale=output_scale,
        weights_width=kw, weights_height=kh,
        weights_zero_point=weight_zero_point, weights_scale=weight_scale,
        add_tensor=add_tensor, addition_input=addition_input,
        addition_offset=addition_offset, addition_scale=addition_scale,
    )
    return op

# ---------------------------------------------------------------------------
# Allocate buffers and set up all BOs
# ---------------------------------------------------------------------------
def setup_bos(fd, op, input_data_nhwc, weight_ohwi, biases_data):
    """Allocate input, weight, bias, output, and regcmd BOs for an operation."""

    input_size = calc_input_size(op.input_width, op.input_height, op.input_channels)
    weight_size = calc_weight_size(op.weights_width, op.weights_height,
                                    op.input_channels, op.output_channels, op.depthwise)
    output_size = calc_raw_output_size(op.output_width, op.output_height, op.output_channels)

    input_buf, input_bo = _rocket_mem_allocate(fd, input_size)
    weight_buf, weight_bo = _rocket_mem_allocate(fd, weight_size)
    bias_buf, bias_bo = _rocket_mem_allocate(fd, op.output_channels * 4)
    output_buf, output_bo = _rocket_mem_allocate(fd, output_size)
    regcmd_buf, regcmd_bo = _rocket_mem_allocate(fd, 512 * 1024)

    packed_input = pack_input_like_mesa(
        input_data_nhwc, op.input_width, op.input_height, op.input_channels,
        op.input_zero_point, op.output_channels, op.addition_input, op.add_tensor)
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(input_buf)),
                   packed_input.ctypes.data, packed_input.nbytes)

    packed_weight = fill_weights_like_mesa(
        weight_ohwi, op.weights_width, op.weights_height,
        op.input_channels, op.output_channels, op.weights_zero_point, op.depthwise)
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(weight_buf)),
                   packed_weight.ctypes.data, packed_weight.nbytes)

    packed_bias, truncate_bits = rkt_fill_biases_like_mesa(
        biases_data, op.output_channels, weight_ohwi,
        op.weights_width, op.weights_height, op.input_channels,
        op.input_zero_point, op.weights_zero_point, op.depthwise,
        op.weights_scale)
    op.truncate_bits = truncate_bits
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(bias_buf)),
                   packed_bias.ctypes.data, packed_bias.nbytes)

    op.input_bo_handle = input_bo.handle
    op.input_bo_addr = input_bo.dma_address
    op.input_bo_size = input_size
    op.weight_bo_handle = weight_bo.handle
    op.weight_bo_addr = weight_bo.dma_address
    op.weight_bo_size = weight_size
    op.bias_bo_handle = bias_bo.handle
    op.bias_bo_addr = bias_bo.dma_address
    op.bias_bo_size = op.output_channels * 4
    op.output_bo_handle = output_bo.handle
    op.output_bo_addr = output_bo.dma_address
    op.output_bo_size = output_size
    op.regcmd_bo_handle = regcmd_bo.handle
    op.regcmd_bo_addr = regcmd_bo.dma_address
    op.regcmd_bo_size = 512 * 1024

    return (input_buf, input_bo), (weight_buf, weight_bo), (bias_buf, bias_bo), \
           (output_buf, output_bo), (regcmd_buf, regcmd_bo)

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_mesa_conv(fd, input_nhwc, weight_ohwi, biases,
                   input_zero_point=0, input_scale=1.0/255,
                   weight_zero_point=0, weight_scale=1.0/255,
                   output_zero_point=0, output_scale=1.0/255,
                   pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
                   stride=1, depthwise=False, _allow_im2col=True):
    """
    Run one convolution through the Mesa-equivalent pipeline.
    Returns output NHWC uint8 tensor.
    """

    _, in_h, in_w, in_c = input_nhwc.shape
    oc, kw, kh, _ = weight_ohwi.shape
    out_h = (in_h + pad_top + pad_bottom - kh) // stride + 1
    out_w = (in_w + pad_left + pad_right - kw) // stride + 1

    use_im2col = (
        _allow_im2col and stride == 1 and pad_top == 0 and pad_bottom == 0 and
        pad_left == 0 and pad_right == 0 and
        (((kw != 1 or kh != 1) and in_c <= 3) or
         (in_c == 1 and (out_h == 1 or out_w == 1)))
    )
    if use_im2col:
        result, op = run_mesa_conv_im2col(
            fd, input_nhwc, weight_ohwi, biases,
            input_zero_point=input_zero_point, input_scale=input_scale,
            weight_zero_point=weight_zero_point, weight_scale=weight_scale,
            output_zero_point=output_zero_point, output_scale=output_scale,
            depthwise=depthwise)
        return result, op

    if depthwise and input_nhwc.shape[3] > 64 and (input_nhwc.shape[3] % 64) != 0:
        outputs = []
        for start in range(0, input_nhwc.shape[3], 64):
            end = min(start + 64, input_nhwc.shape[3])
            chunk_out, _ = run_mesa_conv(
                fd, input_nhwc[:, :, :, start:end], weight_ohwi[:, :, :, start:end],
                biases[start:end], input_zero_point=input_zero_point,
                input_scale=input_scale, weight_zero_point=weight_zero_point,
                weight_scale=weight_scale, output_zero_point=output_zero_point,
                output_scale=output_scale, pad_top=pad_top, pad_bottom=pad_bottom,
                pad_left=pad_left, pad_right=pad_right, stride=stride, depthwise=depthwise,
                _allow_im2col=False)
            outputs.append(chunk_out)

        op = create_operation_from_args(
            input_nhwc, weight_ohwi, biases,
            input_zero_point, input_scale,
            weight_zero_point, weight_scale,
            output_zero_point, output_scale,
            pad_top=pad_top, pad_bottom=pad_bottom,
            pad_left=pad_left, pad_right=pad_right,
            stride=stride, depthwise=depthwise)
        return np.concatenate(outputs, axis=2), op

    op = create_operation_from_args(
        input_nhwc, weight_ohwi, biases,
        input_zero_point, input_scale,
        weight_zero_point, weight_scale,
        output_zero_point, output_scale,
        pad_top=pad_top, pad_bottom=pad_bottom,
        pad_left=pad_left, pad_right=pad_right,
        stride=stride, depthwise=depthwise)

    split_tasks_like_mesa(op)

    bos = setup_bos(fd, op, input_nhwc, weight_ohwi, biases)
    (input_buf, input_bo), (weight_buf, weight_bo), (bias_buf, bias_bo), \
        (output_buf, output_bo), (regcmd_buf, regcmd_bo) = bos

    regcmds = compile_operation_like_mesa(op, op.regcmd_bo_addr)

    total_qwords = sum(_align(len(r), 2) for r in regcmds)
    flat_regs = []
    for i, r in enumerate(regcmds):
        offset = sum(_align(len(regcmds[j]), 2) for j in range(i))
        for q in r:
            flat_regs.append(q)
        pad = _align(len(r), 2) - len(r)
        for _ in range(pad):
            flat_regs.append(0)

    regcmd_array = (ctypes.c_uint64 * len(flat_regs))(*flat_regs)
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_buf)),
                   ctypes.addressof(regcmd_array), len(flat_regs) * 8)

    submit_operation_like_mesa(fd, op, input_bo, weight_bo, bias_bo,
                                output_bo, regcmd_bo, regcmds)

    _rocket_prep_bo(fd, output_bo)
    raw_output = np.frombuffer(output_buf, dtype=np.uint8, count=op.output_bo_size).copy()

    result = read_output_like_mesa(raw_output, op.output_width, op.output_height,
                                    op.output_channels)

    return result, op


def run_mesa_conv_im2col(fd, input_nhwc, weight_ohwi, biases,
                         input_zero_point=0, input_scale=1.0/255,
                         weight_zero_point=0, weight_scale=1.0/255,
                         output_zero_point=0, output_scale=1.0/255,
                         depthwise=False):
    """Lower fragile spatial/1D cases to a 1x1 NPU conv; no CPU convolution."""
    _, in_h, in_w, in_c = input_nhwc.shape
    oc, kw, kh, _ = weight_ohwi.shape
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    output_channels = in_c if depthwise else oc
    flat_c = kh * kw * in_c
    flat_c_aligned = _align(flat_c, FEATURE_ATOMIC_SIZE)

    def build_im2col(row_start, row_end):
        tile_h = row_end - row_start
        im2col = np.full((1, tile_h, out_w, flat_c_aligned), input_zero_point, dtype=np.uint8)
        flat = 0
        for ic in range(in_c):
            for ky in range(kh):
                for kx in range(kw):
                    src_y = row_start + ky
                    im2col[0, :, :, flat] = input_nhwc[0, src_y:src_y + tile_h, kx:kx + out_w, ic]
                    flat += 1
        return im2col

    weight_1x1 = np.full((output_channels, 1, 1, flat_c_aligned), weight_zero_point, dtype=np.uint8)
    if depthwise:
        for channel in range(in_c):
            flat = channel * kh * kw
            for ky in range(kh):
                for kx in range(kw):
                    weight_1x1[channel, 0, 0, flat] = weight_ohwi[0, kx, ky, channel]
                    flat += 1
    else:
        for out_channel in range(oc):
            flat = 0
            for ic in range(in_c):
                for ky in range(kh):
                    for kx in range(kw):
                        weight_1x1[out_channel, 0, 0, flat] = weight_ohwi[out_channel, kx, ky, ic]
                        flat += 1

    weights_banks = calc_weights_banks(1, 1, flat_c_aligned, output_channels, False)
    available_input_banks = CBUF_BANKS - weights_banks
    entries_per_slice = calc_entries_per_slice(out_w, flat_c_aligned)
    tile_h = max(1, (CBUF_ENTRIES_PER_BANK * available_input_banks) // entries_per_slice)
    tile_h = min(out_h, tile_h)

    outputs = []
    last_op = None
    for row_start in range(0, out_h, tile_h):
        row_end = min(row_start + tile_h, out_h)
        im2col = build_im2col(row_start, row_end)
        result, last_op = run_mesa_conv(
            fd, im2col, weight_1x1, biases,
            input_zero_point=input_zero_point, input_scale=input_scale,
            weight_zero_point=weight_zero_point, weight_scale=weight_scale,
            output_zero_point=output_zero_point, output_scale=output_scale,
            stride=1, depthwise=False, _allow_im2col=False)
        outputs.append(result[:, :, :output_channels])

    return np.concatenate(outputs, axis=0), last_op


# ---------------------------------------------------------------------------
# CPU quantized reference
# ---------------------------------------------------------------------------
def compute_expected_quantized(input_nhwc, weight_ohwi, biases,
                                input_zero_point, weight_zero_point,
                                output_zero_point,
                                pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
                                stride=1, depthwise=False):
    """Simulate quantized convolution on CPU for comparison."""
    _, in_h, in_w, in_c = input_nhwc.shape
    oc, kw, kh, ic = weight_ohwi.shape
    output_channels = ic if depthwise else oc

    out_h = (in_h + pad_top + pad_bottom - kh) // stride + 1
    out_w = (in_w + pad_left + pad_right - kw) // stride + 1

    input_s = input_nhwc.astype(np.int32) - input_zero_point
    weight_s = weight_ohwi.astype(np.int32) - weight_zero_point

    padded = np.zeros((1, in_h + pad_top + pad_bottom, in_w + pad_left + pad_right, in_c), dtype=np.int32)
    padded[:, pad_top:pad_top + in_h, pad_left:pad_left + in_w, :] = input_s

    windows = np.lib.stride_tricks.sliding_window_view(
        padded[0], (kh, kw), axis=(0, 1))[::stride, ::stride]
    # sliding_window_view yields H/W windows as (out_h, out_w, C, kh, kw).
    if depthwise:
        expected = np.einsum('yxcab,bac->yxc', windows[:, :, :output_channels], weight_s[0], dtype=np.int64)
        expected = expected[np.newaxis, :, :, :] + biases.reshape(1, 1, 1, output_channels)
    else:
        expected = np.einsum('yxcab,obac->yxo', windows, weight_s, dtype=np.int64)
        expected = expected[np.newaxis, :, :, :] + biases.reshape(1, 1, 1, oc)

    input_scale = 1.0 / 255
    weight_scale = 1.0 / 255
    output_scale = 1.0 / 255
    conv_scale = (input_scale * weight_scale) / output_scale
    expected_f = expected.astype(np.float64) * conv_scale + output_zero_point
    expected_f = np.clip(np.round(expected_f), 0, 255).astype(np.uint8)
    return expected_f


def load_conv_py_shapes():
    """Extract the literal shapes list from sibling conv.py without importing it."""
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

    conv_path = os.path.join(os.path.dirname(__file__), "conv.py")
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
    raise RuntimeError("Could not find literal shapes list in conv.py")


def conv_py_shape_to_mesa_case(shape):
    """Translate one conv.py FP16 shape row into Mesa-style quantized semantics."""
    batch = shape["batch"]
    in_c = shape["in_c"]
    out_c = shape["out_c"]
    weight_in_c = shape["weight_in_c"]
    kh = shape["kh"]
    kw = shape["kw"]
    groups = shape["groups"]

    if batch != 1:
        return None, "batch>1 unsupported by Mesa clone runner"
    if shape["in_h"] < kh or shape["in_w"] < kw:
        return None, "invalid output shape"

    depthwise = groups == in_c == out_c and weight_in_c == 1
    if groups != 1 and not depthwise:
        return None, "arbitrary grouped convolution is outside Mesa clone scope"

    input_shape = (1, shape["in_h"], shape["in_w"], in_c)
    if depthwise:
        weight_shape = (1, kw, kh, in_c)
        bias_shape = (in_c,)
    else:
        if weight_in_c != in_c:
            return None, "weight_in_c does not match normal input channels"
        weight_shape = (out_c, kw, kh, in_c)
        bias_shape = (out_c,)

    return dict(name=shape["name"], input_shape=input_shape, weight_shape=weight_shape,
                bias_shape=bias_shape, pad=0, stride=1, depthwise=depthwise), None


def run_quantized_case(fd, shape):
    np.random.seed(42)
    input_nhwc = np.random.randint(0, 256, shape["input_shape"], dtype=np.uint8)
    weight_ohwi = np.random.randint(0, 256, shape["weight_shape"], dtype=np.uint8)
    biases = np.random.randint(-128, 128, shape["bias_shape"], dtype=np.int32)

    result, op = run_mesa_conv(
        fd, input_nhwc, weight_ohwi, biases,
        pad_top=shape["pad"], pad_bottom=shape["pad"],
        pad_left=shape["pad"], pad_right=shape["pad"],
        stride=shape["stride"], depthwise=shape["depthwise"])

    expected = compute_expected_quantized(
        input_nhwc, weight_ohwi, biases, 0, 0, 0,
        pad_top=shape["pad"], pad_bottom=shape["pad"],
        pad_left=shape["pad"], pad_right=shape["pad"],
        stride=shape["stride"], depthwise=shape["depthwise"])

    diff = int(np.max(np.abs(result.astype(np.int32) - expected[0].astype(np.int32))))
    return diff


# ---------------------------------------------------------------------------
# Shape runner
# ---------------------------------------------------------------------------
def shape_matrix_runner(fd):
    """Run a list of known TFLite conv shapes through the Mesa-equivalent pipeline."""

    shapes = [
        dict(name="3x3_valid", input_shape=(1, 4, 4, 3), weight_shape=(6, 3, 3, 3),
             bias_shape=(6,), pad=0, stride=1, depthwise=False),
        dict(name="3x3_stride2", input_shape=(1, 8, 8, 3), weight_shape=(6, 3, 3, 3),
             bias_shape=(6,), pad=0, stride=2, depthwise=False),
        dict(name="depthwise_3x3", input_shape=(1, 8, 8, 4), weight_shape=(1, 3, 3, 4),
             bias_shape=(4,), pad=0, stride=1, depthwise=True),
        dict(name="depthwise_stride2", input_shape=(1, 8, 8, 4), weight_shape=(1, 3, 3, 4),
             bias_shape=(4,), pad=0, stride=2, depthwise=True),
        dict(name="pad_same_3x3", input_shape=(1, 4, 4, 3), weight_shape=(6, 3, 3, 3),
             bias_shape=(6,), pad=1, stride=1, depthwise=False),
        dict(name="large_pointwise", input_shape=(1, 7, 7, 1024), weight_shape=(2048, 1, 1, 1024),
             bias_shape=(2048,), pad=0, stride=1, depthwise=False),
    ]

    failed = 0
    for s in shapes:
        print(f"Running {s['name']} ...", end=" ")
        sys.stdout.flush()

        try:
            diff = run_quantized_case(fd, s)
            ok = diff <= 1
            status = "PASS" if ok else f"FAIL (max_diff={diff})"
            print(f" {status}")
            failed += 0 if ok else 1
        except Exception as e:
            print(f" ERROR: {e}")
            failed += 1
    print(f"Built-in matrix: {len(shapes) - failed} PASS, {failed} FAIL")


def conv_py_shape_matrix_runner(fd):
    """Run every conv.py test row that has Mesa-equivalent quantized semantics."""
    conv_shapes = load_conv_py_shapes()
    cases = []
    skipped = []
    for s in conv_shapes:
        case, reason = conv_py_shape_to_mesa_case(s)
        if case is None:
            skipped.append((s["name"], reason))
        else:
            cases.append(case)

    name_width = max(len(s["name"]) for s in conv_shapes) if conv_shapes else 1
    passed = failed = 0
    print(f"Running {len(cases)} Mesa-equivalent cases extracted from conv.py ({len(skipped)} skipped) ...")
    for case in cases:
        print(f"  {case['name']:<{name_width}s} ...", end=" ")
        sys.stdout.flush()
        try:
            diff = run_quantized_case(fd, case)
            ok = diff <= 1
            if ok:
                passed += 1
                print("PASS")
            else:
                failed += 1
                print(f"FAIL (max_diff={diff})")
        except Exception as e:
            failed += 1
            print(f"ERROR: {e}")

    if skipped:
        print("Skipped conv.py rows outside this Mesa-clone contract:")
        for name, reason in skipped:
            print(f"  {name:<{name_width}s} SKIP ({reason})")

    print(f"conv.py-derived Mesa matrix: {passed} PASS, {failed} FAIL, {len(skipped)} SKIP")
    return failed


if __name__ == "__main__":
    fd = _open_rocket_device()
    shape_matrix_runner(fd)
    if "--conv-py-shapes" in sys.argv or "--all" in sys.argv:
        conv_py_shape_matrix_runner(fd)
    os.close(fd)
