#!/usr/bin/env python3
"""Standalone first-principles FP16 CONV for RK3588 NPU (gemm.py style, <1000 lines).

CBUF planner emits NONE/BY_Y/BY_K/BY_YK from geometry only (Mesa slice/bank formulas).
BY_YK is an independent Y×K cartesian product of local tiles (no shape-name tables).
Each tile: open → alloc → pack → 1-task submit → unpack → reset → close.
"""
import os, mmap, sys, ctypes, argparse
from fcntl import ioctl
import numpy as np

# Allow `python3 conv_grok/conv.py` and `python3 -m conv_grok.conv` alike.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

RKNPU_MEM_KERNEL_MAPPING = 8
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
MIN_HW_OC = 2
POINTWISE_WIDE_MIN_OC = 32
DW_MIN_TILE_H = 10          # Mesa depthwise spatial minimum output rows per tile
DW_MAX_INPUT_ROWS = 15      # Mesa depthwise CBUF feature-row ceiling
DW_PLANNER_INPUT_BANKS = 7  # fallback data banks when weight fills CBUF
LARGE_IC_FG_THRESHOLD = 64  # apply feature-grain cap above this IC
# CNA_CONV_CON3 CONV_{X,Y}_STRIDE are 3-bit fields (registers.xml) → stride ∈ [1, 7]
MAX_CONV_STRIDE = 7

# Regression / demo shape lists for CLI only — planner never reads these.
DEFAULT_SMOKE = (
    "conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1",
    "b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid",
)
# Stress shapes for CLI --extra-hard (planner is geometry-only).
EXTRA_HARD_SHAPES = (
    # RGB first-layer NHWC (tall Y)
    "b1_c3_h224_w224_oc32_wic3_k3x3_g1_s1_pvalid",
    "b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid",
    "b1_c3_h384_w384_oc32_wic3_k3x3_g1_s1_pvalid",
    "b1_c3_h448_w448_oc16_wic3_k3x3_g1_s1_pvalid",
    "b1_c3_h256_w128_oc32_wic3_k3x3_g1_s1_pvalid",
    # Mid-IC tall spatial (non-NHWC)
    "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid",
    "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid",
    "b1_c40_h80_w80_oc80_wic40_k3x3_g1_s1_pvalid",
    "b1_c24_h120_w120_oc48_wic24_k3x3_g1_s1_pvalid",
    # Odd-IC / fat / tiny spatial
    "b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid",
    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid",
    "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid",
    "b1_c192_h7_w7_oc192_wic192_k3x3_g1_s1_pvalid",
    # Large pointwise
    "b1_c64_h112_w112_oc128_wic64_k1x1_g1_s1_pvalid",
    "b1_c256_h28_w28_oc512_wic256_k1x1_g1_s1_pvalid",
    "b1_c512_h14_w14_oc256_wic512_k1x1_g1_s1_pvalid",
    "b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1_s1_pvalid",
    "b1_c1280_h10_w10_oc256_wic1280_k1x1_g1_s1_pvalid",
    "b1_c960_h10_w10_oc160_wic960_k1x1_g1_s1_pvalid",
    "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid",
    # Stride / asymmetric / batch
    "b1_c64_h56_w56_oc128_wic64_k3x3_g1_s2_pvalid",
    "b1_c32_h56_w56_oc64_wic32_k3x3_g1_s2_pvalid",
    "b1_c16_h64_w64_oc32_wic16_k5x5_g1_s2_pvalid",
    "b1_c32_h28_w28_oc64_wic32_k5x5_g1_s1_pvalid",
    "b1_c16_h9_w17_oc32_wic16_k3x5_g1_s1_pvalid",
    "b1_c32_h17_w9_oc64_wic32_k5x3_g1_s1_pvalid",
    "b1_c64_h14_w14_oc128_wic64_k1x7_g1_s1_pvalid",
    "b2_c32_h28_w28_oc64_wic32_k3x3_g1_s1_pvalid",
    # Grouped (non-DW)
    "b1_c64_h28_w28_oc64_wic16_k3x3_g4_s1_pvalid",
    "b1_c96_h28_w28_oc96_wic32_k3x3_g3_s1_pvalid",
    # Depthwise (wic1; keep mid-size — huge DW is slow serial)
    "b1_c128_h56_w56_oc128_wic1_k3x3_g128",
    "b1_c256_h28_w28_oc256_wic1_k5x5_g256",
    "b1_c64_h112_w112_oc64_wic1_k3x3_g64",
    "b1_c96_h56_w56_oc96_wic1_k5x5_g96",
    # --- round-2 stress ---
    # RGB + stride / fat first-layer
    "b1_c3_h224_w224_oc64_wic3_k7x7_g1_s2_pvalid",
    "b1_c3_h512_w512_oc32_wic3_k3x3_g1_s2_pvalid",
    "b1_c4_h192_w192_oc32_wic4_k3x3_g1_s1_pvalid",
    # Tall / wide mid-IC spatial
    "b1_c48_h96_w96_oc96_wic48_k3x3_g1_s1_pvalid",
    "b1_c80_h40_w40_oc160_wic80_k3x3_g1_s1_pvalid",
    "b1_c32_h200_w50_oc64_wic32_k3x3_g1_s1_pvalid",
    "b1_c64_h8_w8_oc128_wic64_k3x3_g1_s1_pvalid",
    # Fat pointwise / odd IC
    "b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid",
    "b1_c768_h14_w14_oc128_wic768_k1x1_g1_s1_pvalid",
    "b1_c1536_h7_w7_oc256_wic1536_k1x1_g1_s1_pvalid",
    "b1_c72_h28_w28_oc144_wic72_k1x1_g1_s1_pvalid",
    # Stride-2 / 7x7 / tiny H
    "b1_c128_h28_w28_oc256_wic128_k3x3_g1_s2_pvalid",
    "b1_c64_h28_w28_oc128_wic64_k7x7_g1_s1_pvalid",
    "b1_c256_h2_w2_oc256_wic256_k1x1_g1_s1_pvalid",
    "b1_c128_h1_w1_oc256_wic128_k1x1_g1_s1_pvalid",
    # Batch / grouped / DW stride
    "b4_c16_h32_w32_oc32_wic16_k3x3_g1_s1_pvalid",
    "b1_c128_h28_w28_oc128_wic32_k3x3_g4_s1_pvalid",
    "b1_c32_h56_w56_oc32_wic1_k3x3_g32_s2_pvalid",
    "b1_c192_h28_w28_oc192_wic1_k3x3_g192",
    # --- round-3 stress ---
    # RGB / small-IC (avoid h>=512 — compute_expected RAM)
    "b1_c3_h112_w224_oc32_wic3_k5x5_g1_s1_pvalid",
    "b1_c1_h256_w256_oc16_wic1_k3x3_g1_s1_pvalid",
    # Mid spatial pressure
    "b1_c56_h56_w56_oc112_wic56_k3x3_g1_s1_pvalid",
    "b1_c96_h48_w48_oc192_wic96_k3x3_g1_s1_pvalid",
    "b1_c112_h28_w28_oc224_wic112_k3x3_g1_s1_pvalid",
    "b1_c144_h20_w20_oc288_wic144_k3x3_g1_s1_pvalid",
    # Pointwise (y_step can be <6 — exercises _windows_from_step)
    "b1_c320_h28_w28_oc640_wic320_k1x1_g1_s1_pvalid",
    "b1_c576_h14_w14_oc96_wic576_k1x1_g1_s1_pvalid",
    "b1_c832_h7_w7_oc128_wic832_k1x1_g1_s1_pvalid",
    # Stride-3 / 1xN / NHx1
    "b1_c32_h48_w48_oc64_wic32_k3x3_g1_s3_pvalid",
    "b1_c64_h21_w21_oc128_wic64_k7x1_g1_s1_pvalid",
    "b1_c64_h21_w21_oc128_wic64_k1x7_g1_s1_pvalid",
    "b1_c16_h33_w17_oc32_wic16_k5x3_g1_s2_pvalid",
    # Grouped / DW mid / batch
    "b1_c160_h28_w28_oc160_wic40_k3x3_g4_s1_pvalid",
    "b1_c48_h56_w56_oc48_wic1_k5x5_g48",
    "b8_c8_h16_w16_oc16_wic8_k3x3_g1_s1_pvalid",
    # --- round-4 stress ---
    # Small y_step pointwise / odd H
    "b1_c384_h28_w28_oc384_wic384_k1x1_g1_s1_pvalid",
    "b1_c192_h36_w36_oc384_wic192_k1x1_g1_s1_pvalid",
    "b1_c448_h14_w14_oc224_wic448_k1x1_g1_s1_pvalid",
    # Spatial mid + stride / asymmetric
    "b1_c64_h64_w64_oc128_wic64_k3x3_g1_s2_pvalid",
    "b1_c36_h72_w72_oc72_wic36_k3x3_g1_s1_pvalid",
    "b1_c20_h100_w40_oc40_wic20_k3x3_g1_s1_pvalid",
    "b1_c48_h28_w56_oc96_wic48_k5x5_g1_s1_pvalid",
    "b1_c32_h45_w45_oc64_wic32_k3x3_g1_s3_pvalid",
    # Fat first-layer / tiny map
    "b1_c3_h192_w192_oc48_wic3_k5x5_g1_s2_pvalid",
    "b1_c256_h4_w4_oc512_wic256_k3x3_g1_s1_pvalid",
    "b1_c96_h5_w5_oc192_wic96_k3x3_g1_s1_pvalid",
    # Grouped / DW / batch
    "b1_c120_h28_w28_oc120_wic24_k3x3_g5_s1_pvalid",
    "b1_c80_h28_w28_oc80_wic1_k3x3_g80_s2_pvalid",
    "b2_c64_h28_w28_oc128_wic64_k1x1_g1_s1_pvalid",
    "b1_c16_h64_w32_oc32_wic16_k7x7_g1_s1_pvalid",
    # --- round-5 stress ---
    # Odd IC / non-32-aligned
    "b1_c17_h56_w56_oc34_wic17_k3x3_g1_s1_pvalid",
    "b1_c33_h28_w28_oc66_wic33_k3x3_g1_s1_pvalid",
    "b1_c88_h28_w28_oc176_wic88_k1x1_g1_s1_pvalid",
    "b1_c240_h14_w14_oc480_wic240_k1x1_g1_s1_pvalid",
    # Stride-4 / large-k vs map / 1x9
    "b1_c32_h64_w64_oc64_wic32_k3x3_g1_s4_pvalid",
    "b1_c16_h28_w28_oc32_wic16_k9x9_g1_s1_pvalid",
    "b1_c32_h28_w28_oc64_wic32_k1x9_g1_s1_pvalid",
    "b1_c32_h28_w28_oc64_wic32_k9x1_g1_s1_pvalid",
    # Tall thin / wide short / RGB s3
    "b1_c24_h128_w32_oc48_wic24_k3x3_g1_s1_pvalid",
    "b1_c24_h32_w128_oc48_wic24_k3x3_g1_s1_pvalid",
    "b1_c3_h160_w160_oc32_wic3_k3x3_g1_s3_pvalid",
    "b1_c4_h96_w96_oc16_wic4_k5x5_g1_s2_pvalid",
    # Pointwise pressure / tiny spatial fat IC
    "b1_c672_h14_w14_oc112_wic672_k1x1_g1_s1_pvalid",
    "b1_c288_h20_w20_oc288_wic288_k1x1_g1_s1_pvalid",
    "b1_c512_h3_w3_oc512_wic512_k1x1_g1_s1_pvalid",
    "b1_c192_h6_w6_oc384_wic192_k3x3_g1_s1_pvalid",
    # Grouped g2/g8 / DW k7 / batch
    "b1_c64_h40_w40_oc64_wic32_k3x3_g2_s1_pvalid",
    "b1_c128_h20_w20_oc128_wic16_k3x3_g8_s1_pvalid",
    "b1_c64_h28_w28_oc64_wic1_k7x7_g64",
    "b3_c32_h24_w24_oc64_wic32_k3x3_g1_s1_pvalid",
    # --- round-6 stress ---
    # NHWC-ish small-IC / odd C (c5..c7 must be C-major, not NHWC)
    "b1_c5_h128_w128_oc32_wic5_k3x3_g1_s1_pvalid",
    "b1_c6_h64_w64_oc24_wic6_k3x3_g1_s1_pvalid",
    "b1_c7_h64_w64_oc28_wic7_k3x3_g1_s2_pvalid",
    "b1_c1_h128_w128_oc8_wic1_k5x5_g1_s2_pvalid",
    # Asymmetric + stride / odd map
    "b1_c32_h35_w63_oc64_wic32_k3x5_g1_s2_pvalid",
    "b1_c48_h63_w35_oc96_wic48_k5x3_g1_s1_pvalid",
    "b1_c16_h40_w40_oc32_wic16_k5x1_g1_s2_pvalid",
    "b1_c16_h40_w40_oc32_wic16_k1x5_g1_s2_pvalid",
    "b1_c64_h11_w11_oc128_wic64_k3x3_g1_s1_pvalid",
    # Pointwise / fat OC
    "b1_c416_h14_w14_oc208_wic416_k1x1_g1_s1_pvalid",
    "b1_c896_h7_w7_oc128_wic896_k1x1_g1_s1_pvalid",
    "b1_c160_h28_w28_oc640_wic160_k1x1_g1_s1_pvalid",
    "b1_c96_h56_w56_oc24_wic96_k1x1_g1_s1_pvalid",
    # Mid spatial pressure
    "b1_c28_h84_w84_oc56_wic28_k3x3_g1_s1_pvalid",
    "b1_c112_h20_w20_oc224_wic112_k3x3_g1_s1_pvalid",
    "b1_c8_h96_w96_oc32_wic8_k5x5_g1_s1_pvalid",
    # Grouped g6 / DW s3 / batch
    "b1_c96_h28_w28_oc96_wic16_k3x3_g6_s1_pvalid",
    "b1_c48_h40_w40_oc48_wic1_k3x3_g48_s3_pvalid",
    "b5_c16_h20_w20_oc32_wic16_k3x3_g1_s1_pvalid",
    "b1_c3_h96_w192_oc24_wic3_k7x7_g1_s2_pvalid",
    # --- round-7 stress ---
    # NHWC c2 / just-above-8 / odd OC
    "b1_c2_h160_w160_oc16_wic2_k3x3_g1_s1_pvalid",
    "b1_c9_h56_w56_oc18_wic9_k3x3_g1_s1_pvalid",
    "b1_c12_h48_w48_oc20_wic12_k3x3_g1_s1_pvalid",
    "b1_c3_h128_w128_oc12_wic3_k5x5_g1_s2_pvalid",
    # Stride-5 / large-k / extreme aspect
    "b1_c32_h50_w50_oc64_wic32_k3x3_g1_s5_pvalid",
    "b1_c16_h24_w24_oc32_wic16_k11x11_g1_s1_pvalid",
    "b1_c24_h15_w100_oc48_wic24_k3x3_g1_s1_pvalid",
    "b1_c24_h100_w15_oc48_wic24_k3x3_g1_s1_pvalid",
    "b1_c32_h28_w28_oc64_wic32_k5x7_g1_s1_pvalid",
    # Pointwise / mid spatial
    "b1_c352_h20_w20_oc176_wic352_k1x1_g1_s1_pvalid",
    "b1_c704_h10_w10_oc128_wic704_k1x1_g1_s1_pvalid",
    "b1_c44_h56_w56_oc88_wic44_k3x3_g1_s1_pvalid",
    "b1_c176_h14_w14_oc352_wic176_k3x3_g1_s1_pvalid",
    "b1_c10_h80_w80_oc40_wic10_k5x5_g1_s1_pvalid",
    # Grouped g7 / DW / batch / RGB s3 k5
    "b1_c112_h28_w28_oc112_wic16_k3x3_g7_s1_pvalid",
    "b1_c96_h28_w28_oc96_wic1_k5x5_g96_s2_pvalid",
    "b6_c8_h16_w16_oc16_wic8_k3x3_g1_s1_pvalid",
    "b1_c3_h120_w120_oc32_wic3_k5x5_g1_s3_pvalid",
    "b1_c64_h7_w7_oc256_wic64_k1x1_g1_s1_pvalid",
    # --- round-8 stress ---
    # Edge IC / NHWC / C-major boundary
    "b1_c2_h96_w192_oc8_wic2_k5x5_g1_s2_pvalid",
    "b1_c11_h40_w40_oc22_wic11_k3x3_g1_s1_pvalid",
    "b1_c15_h32_w32_oc30_wic15_k3x3_g1_s1_pvalid",
    "b1_c4_h256_w64_oc16_wic4_k3x3_g1_s1_pvalid",
    # Stride / kernel combos
    "b1_c32_h60_w60_oc64_wic32_k5x5_g1_s3_pvalid",
    "b1_c16_h36_w36_oc32_wic16_k7x3_g1_s2_pvalid",
    "b1_c16_h36_w36_oc32_wic16_k3x7_g1_s2_pvalid",
    "b1_c48_h18_w18_oc96_wic48_k7x7_g1_s1_pvalid",
    "b1_c8_h48_w48_oc16_wic8_k1x11_g1_s1_pvalid",
    # Pointwise / spatial mid
    "b1_c480_h14_w14_oc160_wic480_k1x1_g1_s1_pvalid",
    "b1_c224_h28_w28_oc448_wic224_k1x1_g1_s1_pvalid",
    "b1_c52_h40_w40_oc104_wic52_k3x3_g1_s1_pvalid",
    "b1_c136_h16_w16_oc272_wic136_k3x3_g1_s1_pvalid",
    "b1_c18_h72_w72_oc36_wic18_k5x5_g1_s1_pvalid",
    # Grouped / DW / batch / tiny
    "b1_c144_h20_w20_oc144_wic24_k3x3_g6_s1_pvalid",
    "b1_c32_h64_w64_oc32_wic1_k3x3_g32_s4_pvalid",
    "b4_c32_h16_w16_oc64_wic32_k1x1_g1_s1_pvalid",
    "b1_c256_h5_w5_oc128_wic256_k3x3_g1_s1_pvalid",
    "b1_c3_h80_w320_oc16_wic3_k3x3_g1_s1_pvalid",
    # --- round-9 stress ---
    # Odd / boundary IC + NHWC tall
    "b1_c13_h48_w48_oc26_wic13_k3x3_g1_s1_pvalid",
    "b1_c19_h36_w36_oc38_wic19_k3x3_g1_s1_pvalid",
    "b1_c2_h224_w112_oc16_wic2_k3x3_g1_s2_pvalid",
    "b1_c4_h64_w256_oc32_wic4_k5x5_g1_s1_pvalid",
    # Stride / asymmetric / large-k
    "b1_c32_h42_w42_oc64_wic32_k3x3_g1_s6_pvalid",
    "b1_c24_h48_w48_oc48_wic24_k9x3_g1_s1_pvalid",
    "b1_c24_h48_w48_oc48_wic24_k3x9_g1_s1_pvalid",
    "b1_c16_h32_w32_oc32_wic16_k11x1_g1_s1_pvalid",
    "b1_c40_h24_w24_oc80_wic40_k5x5_g1_s2_pvalid",
    # Pointwise / spatial
    "b1_c608_h10_w10_oc96_wic608_k1x1_g1_s1_pvalid",
    "b1_c336_h16_w16_oc168_wic336_k1x1_g1_s1_pvalid",
    "b1_c68_h32_w32_oc136_wic68_k3x3_g1_s1_pvalid",
    "b1_c104_h24_w24_oc208_wic104_k3x3_g1_s1_pvalid",
    "b1_c14_h64_w64_oc28_wic14_k5x5_g1_s1_pvalid",
    # Grouped / DW / batch / 1x1 map
    "b1_c168_h14_w14_oc168_wic28_k3x3_g6_s1_pvalid",
    "b1_c40_h48_w48_oc40_wic1_k5x5_g40_s2_pvalid",
    "b2_c48_h28_w28_oc96_wic48_k3x3_g1_s1_pvalid",
    "b1_c512_h1_w1_oc256_wic512_k1x1_g1_s1_pvalid",
    "b1_c3_h288_w96_oc24_wic3_k7x7_g1_s2_pvalid",
    # --- round-10 stress ---
    # Odd IC / NHWC / C-major
    "b1_c21_h40_w40_oc42_wic21_k3x3_g1_s1_pvalid",
    "b1_c25_h28_w28_oc50_wic25_k3x3_g1_s1_pvalid",
    "b1_c2_h128_w256_oc8_wic2_k3x3_g1_s1_pvalid",
    "b1_c5_h96_w48_oc20_wic5_k5x5_g1_s2_pvalid",
    # Stride / kernel extremes
    "b1_c32_h56_w56_oc64_wic32_k3x3_g1_s7_pvalid",
    "b1_c16_h40_w40_oc32_wic16_k7x5_g1_s1_pvalid",
    "b1_c16_h40_w40_oc32_wic16_k5x7_g1_s1_pvalid",
    "b1_c8_h28_w28_oc16_wic8_k13x13_g1_s1_pvalid",
    "b1_c48_h32_w16_oc96_wic48_k3x3_g1_s2_pvalid",
    # Pointwise / spatial mid
    "b1_c736_h7_w7_oc128_wic736_k1x1_g1_s1_pvalid",
    "b1_c272_h20_w20_oc136_wic272_k1x1_g1_s1_pvalid",
    "b1_c76_h28_w28_oc152_wic76_k3x3_g1_s1_pvalid",
    "b1_c148_h12_w12_oc296_wic148_k3x3_g1_s1_pvalid",
    "b1_c22_h56_w56_oc44_wic22_k5x5_g1_s1_pvalid",
    # Grouped / DW / batch / fat OC
    "b1_c180_h16_w16_oc180_wic30_k3x3_g6_s1_pvalid",
    "b1_c56_h32_w32_oc56_wic1_k3x3_g56_s3_pvalid",
    "b3_c64_h14_w14_oc128_wic64_k1x1_g1_s1_pvalid",
    "b1_c128_h8_w8_oc512_wic128_k1x1_g1_s1_pvalid",
    "b1_c3_h160_w80_oc32_wic3_k3x3_g1_s2_pvalid",
    # --- round-11 stress ---
    # Odd IC / NHWC / C-major
    "b1_c23_h36_w36_oc46_wic23_k3x3_g1_s1_pvalid",
    "b1_c27_h32_w32_oc54_wic27_k3x3_g1_s1_pvalid",
    "b1_c6_h112_w56_oc24_wic6_k3x3_g1_s1_pvalid",
    "b1_c4_h48_w192_oc16_wic4_k7x7_g1_s2_pvalid",
    # Stride / kernel
    "b1_c32_h64_w32_oc64_wic32_k5x5_g1_s4_pvalid",
    "b1_c24_h36_w36_oc48_wic24_k9x5_g1_s1_pvalid",
    "b1_c24_h36_w36_oc48_wic24_k5x9_g1_s1_pvalid",
    "b1_c16_h30_w30_oc32_wic16_k1x13_g1_s1_pvalid",
    "b1_c64_h20_w40_oc128_wic64_k3x3_g1_s1_pvalid",
    # Pointwise / spatial
    "b1_c800_h7_w7_oc160_wic800_k1x1_g1_s1_pvalid",
    "b1_c400_h14_w14_oc200_wic400_k1x1_g1_s1_pvalid",
    "b1_c84_h24_w24_oc168_wic84_k3x3_g1_s1_pvalid",
    "b1_c156_h10_w10_oc312_wic156_k3x3_g1_s1_pvalid",
    "b1_c26_h48_w48_oc52_wic26_k5x5_g1_s1_pvalid",
    # Grouped / DW / batch
    "b1_c192_h14_w14_oc192_wic32_k3x3_g6_s1_pvalid",
    "b1_c72_h28_w28_oc72_wic1_k7x7_g72",
    "b4_c24_h20_w20_oc48_wic24_k3x3_g1_s1_pvalid",
    "b1_c96_h4_w4_oc384_wic96_k1x1_g1_s1_pvalid",
    "b1_c3_h200_w100_oc16_wic3_k5x5_g1_s2_pvalid",
    # --- round-12 stress ---
    # Odd IC / NHWC / C-major
    "b1_c29_h28_w28_oc58_wic29_k3x3_g1_s1_pvalid",
    "b1_c31_h24_w24_oc62_wic31_k3x3_g1_s1_pvalid",
    "b1_c7_h80_w160_oc28_wic7_k3x3_g1_s1_pvalid",
    "b1_c2_h64_w64_oc32_wic2_k7x7_g1_s2_pvalid",
    # Stride / kernel
    "b1_c32_h48_w96_oc64_wic32_k3x3_g1_s3_pvalid",
    "b1_c20_h40_w40_oc40_wic20_k11x3_g1_s1_pvalid",
    "b1_c20_h40_w40_oc40_wic20_k3x11_g1_s1_pvalid",
    "b1_c16_h28_w28_oc32_wic16_k13x1_g1_s1_pvalid",
    "b1_c56_h20_w40_oc112_wic56_k5x5_g1_s1_pvalid",
    # Pointwise / spatial
    "b1_c864_h7_w7_oc96_wic864_k1x1_g1_s1_pvalid",
    "b1_c432_h12_w12_oc216_wic432_k1x1_g1_s1_pvalid",
    "b1_c92_h20_w20_oc184_wic92_k3x3_g1_s1_pvalid",
    "b1_c164_h8_w8_oc328_wic164_k3x3_g1_s1_pvalid",
    "b1_c30_h40_w40_oc60_wic30_k5x5_g1_s2_pvalid",
    # Grouped / DW / batch
    "b1_c216_h12_w12_oc216_wic36_k3x3_g6_s1_pvalid",
    "b1_c64_h40_w40_oc64_wic1_k5x5_g64_s2_pvalid",
    "b2_c80_h16_w16_oc160_wic80_k1x1_g1_s1_pvalid",
    "b1_c192_h3_w3_oc384_wic192_k1x1_g1_s1_pvalid",
    "b1_c3_h112_w336_oc24_wic3_k3x3_g1_s1_pvalid",
    # --- round-13 stress ---
    # Odd IC / NHWC / C-major
    "b1_c35_h28_w28_oc70_wic35_k3x3_g1_s1_pvalid",
    "b1_c37_h20_w20_oc74_wic37_k3x3_g1_s1_pvalid",
    "b1_c5_h64_w128_oc16_wic5_k3x3_g1_s1_pvalid",
    "b1_c1_h96_w192_oc8_wic1_k3x3_g1_s2_pvalid",
    # Stride / kernel
    "b1_c32_h72_w36_oc64_wic32_k5x5_g1_s3_pvalid",
    "b1_c24_h32_w32_oc48_wic24_k7x9_g1_s1_pvalid",
    "b1_c24_h32_w32_oc48_wic24_k9x7_g1_s1_pvalid",
    "b1_c16_h36_w36_oc32_wic16_k15x1_g1_s1_pvalid",
    "b1_c40_h36_w18_oc80_wic40_k3x3_g1_s2_pvalid",
    # Pointwise / spatial
    "b1_c928_h7_w7_oc128_wic928_k1x1_g1_s1_pvalid",
    "b1_c496_h10_w10_oc248_wic496_k1x1_g1_s1_pvalid",
    "b1_c100_h18_w18_oc200_wic100_k3x3_g1_s1_pvalid",
    "b1_c172_h8_w8_oc344_wic172_k3x3_g1_s1_pvalid",
    "b1_c34_h36_w36_oc68_wic34_k5x5_g1_s1_pvalid",
    # Grouped / DW / batch
    "b1_c240_h10_w10_oc240_wic40_k3x3_g6_s1_pvalid",
    "b1_c80_h24_w24_oc80_wic1_k3x3_g80_s4_pvalid",
    "b5_c32_h12_w12_oc64_wic32_k3x3_g1_s1_pvalid",
    "b1_c320_h4_w4_oc160_wic320_k1x1_g1_s1_pvalid",
    "b1_c3_h240_w80_oc32_wic3_k5x5_g1_s2_pvalid",
    # --- round-14 stress ---
    # Odd IC / NHWC / C-major
    "b1_c39_h24_w24_oc78_wic39_k3x3_g1_s1_pvalid",
    "b1_c41_h20_w20_oc82_wic41_k3x3_g1_s1_pvalid",
    "b1_c6_h80_w40_oc24_wic6_k5x5_g1_s1_pvalid",
    "b1_c4_h160_w80_oc16_wic4_k3x3_g1_s2_pvalid",
    # Stride / kernel
    "b1_c32_h54_w54_oc64_wic32_k3x3_g1_s7_pvalid",  # HW max stride=7 (3-bit CON3)
    "b1_c20_h36_w36_oc40_wic20_k11x5_g1_s1_pvalid",
    "b1_c20_h36_w36_oc40_wic20_k5x11_g1_s1_pvalid",
    "b1_c16_h32_w32_oc32_wic16_k1x15_g1_s1_pvalid",
    "b1_c48_h28_w56_oc96_wic48_k3x3_g1_s2_pvalid",
    # Pointwise / spatial
    "b1_c992_h7_w7_oc128_wic992_k1x1_g1_s1_pvalid",
    "b1_c544_h10_w10_oc136_wic544_k1x1_g1_s1_pvalid",
    "b1_c108_h16_w16_oc216_wic108_k3x3_g1_s1_pvalid",
    "b1_c184_h8_w8_oc368_wic184_k3x3_g1_s1_pvalid",
    "b1_c38_h32_w32_oc76_wic38_k5x5_g1_s1_pvalid",
    # Grouped / DW / batch
    "b1_c252_h10_w10_oc252_wic42_k3x3_g6_s1_pvalid",
    "b1_c48_h36_w36_oc48_wic1_k5x5_g48_s3_pvalid",
    "b3_c40_h16_w16_oc80_wic40_k3x3_g1_s1_pvalid",
    "b1_c384_h2_w2_oc192_wic384_k1x1_g1_s1_pvalid",
    "b1_c3_h96_w288_oc24_wic3_k7x7_g1_s1_pvalid",
    # --- round-15 stress ---
    # Odd IC / NHWC / C-major
    "b1_c43_h24_w24_oc86_wic43_k3x3_g1_s1_pvalid",
    "b1_c45_h20_w20_oc90_wic45_k3x3_g1_s1_pvalid",
    "b1_c7_h96_w48_oc28_wic7_k5x5_g1_s2_pvalid",
    "b1_c2_h192_w96_oc16_wic2_k3x3_g1_s1_pvalid",
    # Stride / kernel (stride ≤7)
    "b1_c32_h49_w49_oc64_wic32_k3x3_g1_s7_pvalid",
    "b1_c24_h40_w40_oc48_wic24_k13x3_g1_s1_pvalid",
    "b1_c24_h40_w40_oc48_wic24_k3x13_g1_s1_pvalid",
    "b1_c16_h40_w40_oc32_wic16_k17x1_g1_s1_pvalid",
    "b1_c56_h24_w48_oc112_wic56_k5x5_g1_s2_pvalid",
    # Pointwise / spatial
    "b1_c1056_h7_w7_oc96_wic1056_k1x1_g1_s1_pvalid",
    "b1_c560_h10_w10_oc140_wic560_k1x1_g1_s1_pvalid",
    "b1_c116_h14_w14_oc232_wic116_k3x3_g1_s1_pvalid",
    "b1_c188_h8_w8_oc376_wic188_k3x3_g1_s1_pvalid",
    "b1_c42_h28_w28_oc84_wic42_k5x5_g1_s1_pvalid",
    # Grouped / DW / batch
    "b1_c264_h10_w10_oc264_wic44_k3x3_g6_s1_pvalid",
    "b1_c96_h20_w20_oc96_wic1_k3x3_g96_s5_pvalid",
    "b4_c48_h12_w12_oc96_wic48_k1x1_g1_s1_pvalid",
    "b1_c448_h3_w3_oc224_wic448_k1x1_g1_s1_pvalid",
    "b1_c3_h128_w256_oc32_wic3_k5x5_g1_s2_pvalid",
)
LIST_SHAPES = (
    ("smoke", "conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1"),
    ("smoke", "b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid"),
    ("try", "b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid"),
    ("try", "b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid"),
    ("try", "conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1"),
    ("try", "conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1"),
    ("extra", "b1_c3_h384_w384_oc32_wic3_k3x3_g1_s1_pvalid"),
    ("extra", "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid"),
    ("extra", "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid"),
    ("extra", "b1_c1280_h10_w10_oc256_wic1280_k1x1_g1_s1_pvalid"),
)


class reg:
    CNA = 0x0201; CORE = 0x0801; DPU = 0x1001; PC = 0x0081; PC_REG = 0x0101; VERSION = 0x0041
    OPERATION_ENABLE = 0x0008; PC_BASE_ADDRESS = 0x0010; PC_REGISTER_AMOUNTS = 0x0014
    S_POINTER = 0x4004; FEATURE_MODE_CFG = 0x400c; DATA_FORMAT = 0x4010
    DST_BASE_ADDR = 0x4020; DST_SURF_STRIDE = 0x4024
    DATA_CUBE_WIDTH = 0x4030; DATA_CUBE_HEIGHT = 0x4034; DATA_CUBE_NOTCH = 0x4038
    DATA_CUBE_CHANNEL = 0x403c; BS_CFG = 0x4040; BS_OW_CFG = 0x4050
    WDMA_SIZE_0 = 0x4058; WDMA_SIZE_1 = 0x405c; BN_CFG = 0x4060; EW_CFG = 0x4070
    EW_CVT_SCALE_VALUE = 0x4078; OUT_CVT_SCALE = 0x4084; SURFACE_ADD = 0x40c0
    CNA_CONV_CON1 = 0x100c; CNA_CONV_CON2 = 0x1010; CNA_CONV_CON3 = 0x1014
    CNA_DATA_SIZE0 = 0x1020; CNA_DATA_SIZE1 = 0x1024; CNA_DATA_SIZE2 = 0x1028
    CNA_DATA_SIZE3 = 0x102c; CNA_WEIGHT_SIZE0 = 0x1030; CNA_WEIGHT_SIZE1 = 0x1034
    CNA_WEIGHT_SIZE2 = 0x1038; CNA_CBUF_CON0 = 0x1040; CNA_CBUF_CON1 = 0x1044
    CNA_CVT_CON0 = 0x104c; CNA_CVT_CON1 = 0x1050; CNA_CVT_CON2 = 0x1054
    CNA_CVT_CON3 = 0x1058; CNA_CVT_CON4 = 0x105c; CNA_CVT_CON5 = 0x1180
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


def shape_from_name(name):
    core = name[7:] if name.startswith("conv2d_") else name
    fields = core.split("_")
    vals = {field[:3] if field.startswith("wic") else field[0]: field for field in fields}
    try:
        kh, kw = (int(x) for x in vals["k"][1:].split("x"))
        stride = int(vals.get("s", "s1")[1:])
        if not 1 <= stride <= MAX_CONV_STRIDE:
            raise ValueError(f"stride={stride} exceeds HW max {MAX_CONV_STRIDE} (CNA_CONV_CON3 3-bit)")
        return dict(name=name, batch=int(vals["b"][1:]), in_c=int(vals["c"][1:]), in_h=int(vals["h"][1:]),
                    in_w=int(vals["w"][1:]), out_c=int(vals["o"][2:]), weight_in_c=int(vals["wic"][3:]),
                    kh=kh, kw=kw, groups=int(vals["g"][1:]), stride=stride)
    except (KeyError, ValueError) as e:
        if isinstance(e, ValueError) and "stride=" in str(e):
            raise
        raise ValueError("expected encoded shape like b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid") from e


def make_shape(batch=1, in_c=1, in_h=1, in_w=None, out_c=1, weight_in_c=None,
               kh=1, kw=None, groups=1, stride=1, pvalid=False, name=None):
    """Build a shape dict from fields (any geometry — not limited to the 217 regression set)."""
    if kw is None:
        kw = kh
    if in_w is None:
        in_w = in_h
    if weight_in_c is None:
        weight_in_c = in_c
    parts = [f"b{batch}", f"c{in_c}", f"h{in_h}", f"w{in_w}", f"oc{out_c}", f"wic{weight_in_c}",
             f"k{kh}x{kw}", f"g{groups}"]
    if stride != 1:
        parts.append(f"s{stride}")
    if pvalid:
        parts.append("pvalid")
    s = shape_from_name("_".join(parts))
    if name:
        s["name"] = name
    return s


def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c


def _is_pointwise_wide(s):
    # IC>=32: 1x1 uses 32-channel weight atoms (c40 fails with kh-major pack).
    return s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1 and s["in_c"] >= POINTWISE_WIDE_MIN_OC


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


def should_use_nhwc_pack(channels, c2):
    # NHWC only when pack width is 2× channels (c1/c2/c3/c4). c5..c7 must use C-major.
    return channels > 0 and (c2 // channels == 2 or (channels == 2 and c2 // channels == 4))


def _conv_params(s):
    in_c, in_h, in_w, out_c = s["in_c"], s["in_h"], s["in_w"], s["out_c"]
    kh, kw, groups, stride = s["kh"], s["kw"], s["groups"], s.get("stride", 1)
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    is_spatial = kh != 1 or kw != 1
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    align_c = _conv_align_c(in_c, groups, out_c)
    align_out_c = max(32 if _is_pointwise_wide(s) else 16, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_atoms = max(1, out_h * out_w)
    out_width_stride = out_atoms if not is_spatial else _align_up(out_atoms, 4)
    input_pack_c2 = _conv_input_pack_c2(in_c, groups, out_c, align_c)
    use_nhwc = (not is_depthwise and not (groups > 1 and is_spatial)
                and should_use_nhwc_pack(in_c, input_pack_c2))
    return dict(is_depthwise=is_depthwise, is_spatial=is_spatial, out_h=out_h, out_w=out_w, align_c=align_c,
                align_out_c=align_out_c, width_stride=width_stride, out_width_stride=out_width_stride,
                input_pack_c2=input_pack_c2, use_nhwc=use_nhwc, stride=stride)


def _dma_strides(in_h, width_stride, use_nhwc_pack):
    # Surf stride is width_stride*(in_h-4) even when in_h<4 (wraps as uint32, e.g. h3→0xfffffffd).
    # Clamping to 0 breaks tiny spatial maps (c128/h3 native FAIL ~155).
    if use_nhwc_pack:
        return width_stride, width_stride * (in_h - 1) if in_h > 1 else 0
    return width_stride * 4, (width_stride * (in_h - 4)) & 0xffffffff


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


def _pointwise_data_in_c(in_c):
    return max(32, _align_up(in_c, 32))


def _planner_in_c(in_c, groups, p):
    """IC align for CBUF planner (32-atom pointwise vs conv align_c)."""
    if groups == 1 and not p["is_spatial"] and in_c >= POINTWISE_WIDE_MIN_OC:
        return _pointwise_data_in_c(in_c)
    return _align_up(in_c, p["align_c"])


def _pack_pointwise_wide(weight, out_c, in_c):
    # Pad IC to 32-atoms; IC-block-major layout. Unpadded IC (e.g. 72→80 reg vs 72 pack) fails.
    aligned_in_c = _pointwise_data_in_c(in_c)
    aligned_out_c = max(32, _align_up(out_c, 16))
    padded = np.zeros((aligned_out_c, aligned_in_c), dtype=np.float16)
    padded[:out_c, :in_c] = weight[:out_c, :in_c, 0, 0]
    return np.concatenate([
        padded[oc:oc + 16].reshape(-1, aligned_in_c // 32, 32).transpose(1, 0, 2).ravel()
        for oc in range(0, aligned_out_c, 16)
    ])


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
    wic = s.get("weight_in_c", s["in_c"])
    return _pack_kh_major(weight_full, s["out_c"], wic, s["kh"], s["kw"], p["align_c"])


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
    in_c, in_h, in_w, out_c = s["in_c"], s["in_h"], s["in_w"], s["out_c"]
    kh, kw = s["kh"], s["kw"]
    align_c, align_out_c = p["align_c"], p["align_out_c"]
    out_h, out_w, is_spatial = p["out_h"], p["out_w"], p["is_spatial"]
    data_in_c = _pointwise_data_in_c(in_c) if _is_pointwise_wide(s) else _align_up(in_c, align_c)
    weight_bytes_per_kernel = kh * kw * data_in_c * FP16_BYTES
    if full_data_bank and not is_spatial:
        # Do not clamp grains below tile H (clamp caused c960 wrong output).
        feature_grains = in_h
        wt_banks = max(1, _ceil_div(weight_bytes_per_kernel * out_c, CBUF_BANK_SIZE))
        data_bank = max(1, min(RK_CBUF_BANKS - 1, RK_CBUF_BANKS - wt_banks))
    else:
        feature_grains = _feature_grains(p["width_stride"] * data_in_c * FP16_BYTES, in_h + kh,
                                         p["use_nhwc"], is_spatial, False)
        if full_data_bank:
            wt_banks = max(1, _ceil_div(weight_bytes_per_kernel * out_c, CBUF_BANK_SIZE))
            data_bank = max(1, min(RK_CBUF_BANKS - 1, RK_CBUF_BANKS - wt_banks))
        else:
            data_bank = _data_bank(
                p["width_stride"], feature_grains, data_in_c, p["use_nhwc"], is_spatial, False)
    out_precision = 2 if out_fp16 else 5
    size_e = 1 if out_fp16 else 3
    out_channel_field = align_out_c - 1
    stride = s.get("stride", 1)
    if not 1 <= stride <= MAX_CONV_STRIDE:
        raise ValueError(f"stride={stride} exceeds HW max {MAX_CONV_STRIDE} (CNA_CONV_CON3 3-bit)")
    cvt_con0 = 0x0b if is_spatial and not p["is_depthwise"] else 1
    cvt_con5 = ((1 << in_c) - 1) if p["use_nhwc"] else 0
    conv_con1 = (2 << 4) | (2 << 7)
    if p["use_nhwc"] and in_c <= 4:
        conv_con1 |= (1 << 30) | (1 << 29) | ((7 + in_c) << 12)
    line_stride, surf_stride = _dma_strides(in_h, p["width_stride"], p["use_nhwc"])
    return [
        E(reg.DPU, reg.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        E(reg.CNA, reg.CNA_CONV_CON1, conv_con1),
        E(reg.CNA, reg.CNA_CONV_CON2, feature_grains << 4),
        E(reg.CNA, reg.CNA_CONV_CON3, (stride << 3) | stride),
        E(reg.CNA, reg.CNA_DATA_SIZE0, (p["width_stride"] << 16) | in_h),
        E(reg.CNA, reg.CNA_DATA_SIZE1, ((in_c - 1) << 16) | data_in_c),
        E(reg.CNA, reg.CNA_DATA_SIZE2, out_w),
        E(reg.CNA, reg.CNA_DATA_SIZE3, out_w * out_h),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE0, weight_bytes_per_kernel * out_c),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE1, weight_bytes_per_kernel),
        E(reg.CNA, reg.CNA_WEIGHT_SIZE2, (kw << 24) | (kh << 16) | out_c),
        E(reg.CNA, reg.CNA_CBUF_CON0, ((RK_CBUF_BANKS - data_bank) << 4) | data_bank),
        E(reg.CNA, reg.CNA_CBUF_CON1, _cbuf_entries(p["width_stride"], data_in_c, in_h, False)),
        E(reg.CNA, reg.CNA_CVT_CON0, cvt_con0), E(reg.CNA, reg.CNA_CVT_CON1, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON2, 1 << 16), E(reg.CNA, reg.CNA_CVT_CON3, 1 << 16),
        E(reg.CNA, reg.CNA_CVT_CON4, 1 << 16), E(reg.CNA, reg.CNA_FEATURE_DATA_ADDR, in_dma),
        E(reg.CNA, reg.CNA_DMA_CON0, (15 << 16) | 15),
        E(reg.CNA, reg.CNA_DMA_CON1, line_stride), E(reg.CNA, reg.CNA_DMA_CON2, surf_stride),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE0, (in_w << 16) | in_h),
        E(reg.CNA, reg.CNA_FC_DATA_SIZE1, data_in_c), E(reg.CNA, reg.CNA_DCOMP_ADDR0, wt_dma),
        E(reg.CNA, reg.CNA_CVT_CON5, cvt_con5),
        E(reg.CORE, reg.CORE_MISC_CFG, (2 << 8) | is_spatial),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_0, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.CORE, reg.CORE_DATAOUT_SIZE_1, out_channel_field), E(reg.CORE, reg.CORE_RESERVED_3030, 0),
        E(reg.DPU, reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1)),
        E(reg.DPU, reg.DATA_FORMAT, (out_precision << 29) | (2 << 26) | 2),
        E(reg.DPU, reg.DST_BASE_ADDR, out_dma), E(reg.DPU, reg.DST_SURF_STRIDE, p["out_width_stride"] << 4),
        E(reg.DPU, reg.DATA_CUBE_WIDTH, out_w - 1), E(reg.DPU, reg.DATA_CUBE_HEIGHT, out_h - 1),
        E(reg.DPU, reg.DATA_CUBE_NOTCH, 0),
        E(reg.DPU, reg.DATA_CUBE_CHANNEL, ((out_c - 1) << 16) | out_channel_field),
        E(reg.DPU, reg.BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.BS_OW_CFG, (size_e << 8) | (size_e << 5) | (size_e << 2) | (1 << 1)),
        E(reg.DPU, reg.WDMA_SIZE_0, out_channel_field),
        E(reg.DPU, reg.WDMA_SIZE_1, ((out_h - 1) << 16) | (out_w - 1)),
        E(reg.DPU, reg.BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
        E(reg.DPU, reg.EW_CVT_SCALE_VALUE, 1),
        E(reg.DPU, reg.OUT_CVT_SCALE, ((1 << 16) | 1) if out_fp16 else 0),
        E(reg.DPU, reg.SURFACE_ADD, (p["out_width_stride"] * 2) << 4),
    ]


# --- compact CBUF planner (formula-only; BY_YK = Y×K cartesian local tiles) ---

def _mesa_entries_per_slice(input_width, input_channels):
    atomics_per_entry = CBUF_ENTRY_BYTES // 16
    total_c_atomics = _ceil_div(input_channels * FP16_BYTES, 16)
    last_c_atomics = total_c_atomics % atomics_per_entry
    int_c_entries = (total_c_atomics // atomics_per_entry) * input_width
    frac = input_width if last_c_atomics == 3 else _ceil_div(last_c_atomics * input_width, atomics_per_entry)
    return int_c_entries + frac


def _mesa_weight_banks(kw, kh, in_c, out_c, depthwise):
    wt_bytes = kh * kw * in_c * FP16_BYTES
    if not depthwise:
        wt_bytes *= out_c
    return _ceil_div(_ceil_div(wt_bytes, CBUF_ENTRY_BYTES), CBUF_ENTRIES_PER_BANK) + 1


def _mesa_output_tile_h(in_w, out_h, in_c, out_c, kh, kw, stride, depthwise, input_banks=None):
    """Max output rows per tile from CBUF entry capacity (Mesa mesa_entries_per_slice)."""
    if input_banks is None:
        wt_banks = _mesa_weight_banks(kw, kh, in_c, out_c, depthwise)
        input_banks = RK_CBUF_BANKS - wt_banks if wt_banks + 1 < RK_CBUF_BANKS else 7
    eps = max(1, _mesa_entries_per_slice(in_w, in_c))
    slices = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // eps)
    return min(out_h, max(1, (slices - kh) // stride + 1))


def _pointwise_oc_tile_c(in_c, weight_banks=1):
    # Cap OC so one K-tile weight fits in `weight_banks` CBUF banks (32-aligned IC).
    data_in = _pointwise_data_in_c(in_c) if in_c >= POINTWISE_WIDE_MIN_OC else max(1, in_c)
    max_tile = (max(1, weight_banks) * CBUF_BANK_SIZE) // (data_in * FP16_BYTES)
    return 32 if max_tile >= 32 else 16 if max_tile >= 16 else 8 if max_tile >= 8 else 4


def _planner_input_banks(tile_weight_banks):
    return RK_CBUF_BANKS - tile_weight_banks if tile_weight_banks + 1 < RK_CBUF_BANKS else DW_PLANNER_INPUT_BANKS


def _mesa_max_y_rows(width, out_h, in_c, oc_tile, kh, kw, stride, depthwise, input_banks, row_bytes=None, grain_floor=None):
    """Max output rows per Y tile from Mesa slice capacity (+ optional feature-grain cap)."""
    tile_h = _mesa_output_tile_h(width, out_h, in_c, oc_tile, kh, kw, stride, depthwise, input_banks=input_banks)
    if row_bytes is not None and grain_floor is not None and in_c > LARGE_IC_FG_THRESHOLD:
        tile_h = min(tile_h, _feature_grains(row_bytes, grain_floor, False, not depthwise, depthwise) + 1)
    return tile_h


def _entry_cap_y_step(width, aligned_in, kh, stride, input_banks, out_h, strict_headroom=False):
    """Y rows allowed by CBUF entry capacity (mesa_entries_per_slice)."""
    eps = max(1, _mesa_entries_per_slice(width, aligned_in))
    max_input_h = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // eps)
    if strict_headroom and max_input_h < out_h:
        max_input_h = max(1, max_input_h - 1)
    return max(1, (max_input_h - kh) // stride + 1)


def _depthwise_spatial_y_step(s, p):
    in_c, kh, kw, in_w = s["in_c"], s["kh"], s["kw"], s["in_w"]
    stride, out_h = s.get("stride", 1), p["out_h"]
    row_bytes = in_w * _conv_align_c(in_c, in_c, in_c) * FP16_BYTES
    tile_h = _mesa_max_y_rows(in_w, out_h, in_c, in_c, kh, kw, stride, True, DW_PLANNER_INPUT_BANKS,
                              row_bytes=row_bytes, grain_floor=out_h + kh)
    if tile_h < out_h:
        tile_h = max(DW_MIN_TILE_H, tile_h)
    tile_h = min(tile_h, min(DW_MAX_INPUT_ROWS, out_h + kh - 1) - kh + 1)
    return tile_h if tile_h == out_h or tile_h % 2 == 0 else max(1, tile_h - 1)


def _compute_k_step(s, p):
    in_c, out_c, kh, kw, groups = s["in_c"], s["out_c"], s["kh"], s["kw"], s["groups"]
    is_dw, is_spatial = p["is_depthwise"], p["is_spatial"]
    aligned_in = _planner_in_c(in_c, groups, p)
    weight_banks = _ceil_div(kh * kw * aligned_in * FP16_BYTES * (1 if is_dw else out_c), CBUF_BANK_SIZE)
    k_step = out_c
    if is_dw and is_spatial:
        k_step = min(32, out_c)
    elif is_spatial and groups == 1 and not is_dw:
        row_bytes = p["width_stride"] * aligned_in * FP16_BYTES
        feature_rows = _feature_grains(row_bytes, s["in_h"] + kh, False, True, False)
        if weight_banks > 3 or feature_rows < s["in_h"]:
            k_step = 32 if out_c >= 32 else out_c
    elif not is_spatial and groups == 1:
        # Default: 1 weight bank (Y-split handles oversized features).
        # Upgrade to 2 banks when full-H features still fit in remaining data banks.
        pw_oc = _pointwise_oc_tile_c(in_c, 1)
        row_bytes = p["width_stride"] * aligned_in * FP16_BYTES
        feat_banks = _ceil_div(row_bytes * s["in_h"], CBUF_BANK_SIZE)
        if feat_banks < (RK_CBUF_BANKS - 2):
            pw_oc = max(pw_oc, _pointwise_oc_tile_c(in_c, 2))
        if weight_banks > 3 or out_c > pw_oc:
            k_step = pw_oc
    return min(k_step, out_c)


def _compute_y_step(s, k_step, p):
    in_c, in_h, in_w = s["in_c"], s["in_h"], s["in_w"]
    out_c, kh, kw, groups = s["out_c"], s["kh"], s["kw"], s["groups"]
    stride = s.get("stride", 1)
    is_spatial, is_dw, out_h = p["is_spatial"], p["is_depthwise"], p["out_h"]
    if is_dw and is_spatial:
        return _depthwise_spatial_y_step(s, p)

    aligned_in = _planner_in_c(in_c, groups, p)
    wide_pw = groups == 1 and not is_spatial and in_c >= POINTWISE_WIDE_MIN_OC
    tile_aligned = aligned_in if wide_pw else _align_up(k_step if is_dw else in_c, p["align_c"])
    row_bytes = p["width_stride"] * tile_aligned * FP16_BYTES
    y_step = out_h
    tile_wb = max(1, _ceil_div(kh * kw * aligned_in * FP16_BYTES * (k_step if not is_dw else 1), CBUF_BANK_SIZE))
    remaining = max(1, RK_CBUF_BANKS - tile_wb)
    fg = _feature_grains(row_bytes, in_h + kh, False, is_spatial, is_dw)
    data_banks_needed = _ceil_div(row_bytes * fg, CBUF_BANK_SIZE)
    if data_banks_needed > remaining:
        y_step = max(1, out_h * remaining // max(1, data_banks_needed))

    if not is_spatial:
        if in_c <= 4 and not is_dw and p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE:
            y_step = min(y_step, max(1, RK_MAX_CONV_FLAT_STRIDE // p["out_w"]))
        elif groups == 1 and not is_dw:
            y_step = min(y_step, _mesa_max_y_rows(p["width_stride"], out_h, aligned_in, k_step, kh, kw, stride,
                                                 False, remaining, row_bytes=row_bytes, grain_floor=out_h + kh))
            while y_step > 1 and wide_pw and _ceil_div(row_bytes * y_step, CBUF_BANK_SIZE) >= remaining:
                y_step -= 1
            y_step = min(y_step, _entry_cap_y_step(p["width_stride"], aligned_in, kh, stride, remaining, out_h,
                                                   strict_headroom=wide_pw))
    else:
        # Spatial (all layouts): Mesa entry capacity — was missing for non-NHWC (c8/c16_h160).
        input_banks = _planner_input_banks(tile_wb)
        y_step = min(y_step, _entry_cap_y_step(p["width_stride"], aligned_in, kh, stride, input_banks, out_h))
        # Input-row CBUF capacity → output rows (must honor stride / kh).
        if p["use_nhwc"]:
            # NHWC: theoretical full-bank grains overshoot HW (~2x); also divide by stride.
            safe_banks = max(1, (RK_CBUF_BANKS - 1) // 2)
            max_in_rows = max(1, (safe_banks * CBUF_BANK_SIZE) // max(1, row_bytes))
            head = 2 * kh  # RGB first-layer / large-k headroom
        else:
            max_in_rows = _feature_grains(row_bytes, in_h + kh, False, True, False)
            head = 2 * kh if max(kh, kw) >= 5 else kh
        y_step = min(y_step, max(1, (max_in_rows - head) // stride + 1))
        # Tall / fat-output spatial CBUF pressure (geometry-only).
        if out_h > 32 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE:
            tall_cap = 32 if p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE else 50
            y_step = min(y_step, tall_cap)
    return max(1, y_step)


def _windows_from_step(total, step, min_tail=6):
    """Partition [0,total) into windows; merge away a tiny final slice (<min_tail).

    When step < min_tail, tail-merge cannot help (tail is always < step); use plain
    steps. Otherwise shrinking must keep n >= 1 (step=5,tail=1 → n=0 OOMed c320).
    """
    if step >= total:
        return [(0, total)]
    wins, start = [], 0
    while start < total:
        remain = total - start
        if remain <= step:
            wins.append((start, remain))
            break
        tail = remain % step
        if step < min_tail or tail == 0 or tail >= min_tail or remain <= step + min_tail:
            n = step
        else:
            n = max(1, step - (min_tail - tail))
        wins.append((start, n))
        start += n
    return wins


def plan_local_serial_rows(s):
    """Emit independent local-serial tiles. BY_YK = Y×K cartesian product."""
    kh, kw, groups = s["kh"], s["kw"], s["groups"]
    stride = s.get("stride", 1)
    p = _conv_params(s)
    out_h, out_c, in_c = p["out_h"], s["out_c"], s["in_c"]
    k_step = _compute_k_step(s, p)
    y_step = _compute_y_step(s, k_step, p)
    if k_step < out_c and y_step < out_h:
        split = "BY_YK"
    elif k_step < out_c:
        split = "BY_K"
    elif y_step < out_h:
        split = "BY_Y"
    else:
        split = "NONE"
    y_wins = _windows_from_step(out_h, y_step)
    k_wins = _windows_from_step(out_c, k_step)

    def row(family, y_start, output_h, k_start, oc_count):
        input_h = min((output_h - 1) * stride + kh, s["in_h"] - y_start * stride)
        return dict(split_method=split, family=family, y_start=y_start, input_h=input_h,
                    output_h=output_h, k_start=k_start, oc_count=oc_count)

    if split == "NONE":
        rows = [row("setup", 0, out_h, 0, out_c)]
    elif split == "BY_Y":
        rows = [row("y_tile", ys, oh, 0, out_c) for ys, oh in y_wins]
    elif split == "BY_K":
        rows = [row("k_tile", 0, out_h, ks, oc) for ks, oc in k_wins]
    else:
        rows = [row("yk_tile", ys, oh, ks, oc) for ys, oh in y_wins for ks, oc in k_wins]
    return rows, p, y_step, k_step


def _depthwise_channel_step(out_c):
    return min(POINTWISE_WIDE_MIN_OC, out_c)


def plan_shape(s):
    """Geometry-only plan: split / steps / row count (no shape-name tables)."""
    if _is_depthwise(s["in_c"], s["out_c"], s["groups"]):
        rows, p, y_step, k_step = plan_depthwise_rows(s)
        est_tasks = s["batch"] * len(rows)
        return dict(rows=rows, p=p, y_step=y_step, k_step=k_step, split="depthwise",
                    path="depthwise_serial", est_tasks=est_tasks)
    rows, p, y_step, k_step = plan_local_serial_rows(s)
    split = rows[0]["split_method"]
    path = "grouped_serial" if s["groups"] != 1 else split
    return dict(rows=rows, p=p, y_step=y_step, k_step=k_step, split=split,
                path=path, est_tasks=s["batch"] * len(rows))


def plan_depthwise_rows(s):
    """Batched depthwise spatial: C windows × Y windows (groups=in_c=out_c per tile).

    ponytail: planner only — runtime still uses per-channel serial until DW regs/pack match conv_tiles.
    """
    p = _conv_params(s)
    out_h, out_c = p["out_h"], s["out_c"]
    c_step = _depthwise_channel_step(out_c)
    y_step = _compute_y_step(s, c_step, p)
    y_wins = _windows_from_step(out_h, y_step)
    c_wins = _windows_from_step(out_c, c_step, min_tail=1)
    stride = s.get("stride", 1)
    rows = []
    for ch_start, ch_count in c_wins:
        for y_start, output_h in y_wins:
            input_h = min((output_h - 1) * stride + s["kh"], s["in_h"] - y_start * stride)
            rows.append(dict(ch_start=ch_start, ch_count=ch_count, y_start=y_start,
                             input_h=input_h, output_h=output_h))
    return rows, p, y_step, c_step


# --- DRM / submit ---

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
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = 0xd
        tasks[idx].int_mask = 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = len(regs)
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * 8


def npu_submit(fd, task_obj_addr, task_count, core_mask=1):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))
    submit = rknpu_submit(flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK, timeout=6000, task_start=0, task_number=task_count,
                          task_counter=0, priority=0, task_obj_addr=task_obj_addr, iommu_domain_id=0,
                          reserved=0, task_base_addr=0, hw_elapse_time=0, core_mask=core_mask, fence_fd=-1)
    submit.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=task_count)
    submit.subcore_task[1] = rknpu_subcore_task(task_start=task_count, task_number=0)
    submit.subcore_task[2] = rknpu_subcore_task(task_start=task_count, task_number=0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, submit)


def post_submit_reset(fd):
    for reset_flag in (RKNPU_ACT_RESET, 6, RKNPU_ACT_RESET):
        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=reset_flag, value=0))


def hw_oc_for(s, oc_count):
    if _is_pointwise_wide(s) and oc_count < POINTWISE_WIDE_MIN_OC:
        return POINTWISE_WIDE_MIN_OC
    return max(MIN_HW_OC, oc_count)


def hw_out_fp16_for(s, p):
    return p["is_spatial"] or s["out_c"] >= 128 or p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE or _is_pointwise_wide(s)


class TileSession:
    """Reuse one DRM fd + BO set across many local-serial submits (depthwise/BY_K)."""

    def __init__(self, in_bytes=4 * 1024 * 1024, wt_bytes=4 * 1024 * 1024, out_bytes=4 * 1024 * 1024):
        self.fd = os.open("/dev/dri/card1", os.O_RDWR)
        self.task_map, self.task_mem = mem_allocate(self.fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
        self.regcmd_map, self.regcmd_mem = mem_allocate(self.fd, 8192, RKNPU_MEM_NON_CACHEABLE)
        self.input_map, self.input_mem = mem_allocate(self.fd, in_bytes, RKNPU_MEM_NON_CACHEABLE)
        self.weight_map, self.weight_mem = mem_allocate(self.fd, wt_bytes, RKNPU_MEM_NON_CACHEABLE)
        self.output_map, self.output_mem = mem_allocate(self.fd, out_bytes, RKNPU_MEM_NON_CACHEABLE)
        self._allocs = ((self.task_map, self.task_mem), (self.regcmd_map, self.regcmd_mem),
                        (self.input_map, self.input_mem), (self.weight_map, self.weight_mem),
                        (self.output_map, self.output_mem))

    def close(self):
        close_allocations(self.fd, self._allocs)
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _copy_u16(dst_map, src_u16):
    nbytes = src_u16.nbytes
    if nbytes > len(dst_map):
        raise ValueError(f"tile buffer too small: need {nbytes} have {len(dst_map)}")
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(dst_map)),
                   src_u16.ctypes.data, nbytes)


def run_hw_tile(tile_shape, tile_in, tile_wt, full_data_bank=False, session=None):
    """One local tile submit. Optional TileSession reuses fd/BOs across tiles."""
    p = _conv_params(tile_shape)
    out_fp16 = full_data_bank or hw_out_fp16_for(tile_shape, p)
    read_dtype = np.float16 if out_fp16 else np.float32
    c2 = UNPACK_C2 if out_fp16 else FP16_ATOM_ELEMENTS // FP32_BYTES
    input_flat = np.ascontiguousarray(pack_input(tile_in, p).view(np.uint16))
    weight_flat = np.ascontiguousarray(pack_weights(tile_wt, tile_shape, p).view(np.uint16))
    out_count = _ceil_div(p["align_out_c"], c2) * p["out_width_stride"] * c2
    output_bytes = out_count * np.dtype(read_dtype).itemsize
    own = session is None
    if own:
        session = TileSession(max(4096, input_flat.nbytes), max(4096, weight_flat.nbytes), max(4096, output_bytes))
    try:
        _copy_u16(session.input_map, input_flat)
        _copy_u16(session.weight_map, weight_flat)
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(session.output_map)), 0, min(len(session.output_map), output_bytes))
        write_tasks(session.task_map, session.regcmd_map, session.regcmd_mem,
                    [make_regs(tile_shape, p, session.input_mem.dma_addr, session.weight_mem.dma_addr,
                               session.output_mem.dma_addr, out_fp16, full_data_bank=full_data_bank)])
        if npu_submit(session.fd, session.task_mem.obj_addr, 1) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(session.output_map, dtype=read_dtype, count=out_count).copy()
        post_submit_reset(session.fd)
    finally:
        if own:
            session.close()
    return unpack_output(out_raw, tile_shape["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], c2)


def compute_expected(inp, wt, s):
    """Vectorized NCHW reference: batch, groups (incl. depthwise), stride, valid pad."""
    stride = s.get("stride", 1)
    out_h = (s["in_h"] - s["kh"]) // stride + 1
    out_w = (s["in_w"] - s["kw"]) // stride + 1
    expected = np.zeros((s["batch"], s["out_c"], out_h, out_w), dtype=np.float32)
    i32, w32 = inp.astype(np.float32), wt.astype(np.float32)
    in_per = s["in_c"] // s["groups"]
    out_per = s["out_c"] // s["groups"]
    for n in range(s["batch"]):
        for g in range(s["groups"]):
            oc0, oc1 = g * out_per, (g + 1) * out_per
            xin = i32[n, g * in_per:(g + 1) * in_per]
            w = w32[oc0:oc1]
            for kh in range(s["kh"]):
                for kw in range(s["kw"]):
                    expected[n, oc0:oc1] += np.einsum(
                        "ihw,oi->ohw",
                        xin[:, kh:kh + out_h * stride:stride, kw:kw + out_w * stride:stride],
                        w[:, :, kh, kw],
                    )
    return expected


def _padded_weight(wt, oc_start, oc_count, hw_oc, weight_in_c, kh, kw):
    if hw_oc == oc_count:
        return wt[oc_start:oc_start + oc_count]
    out = np.zeros((hw_oc, weight_in_c, kh, kw), dtype=np.float16)
    out[:oc_count] = wt[oc_start:oc_start + oc_count]
    return out


def _tile_full_data_bank(s, row, p):
    if not p["is_spatial"]:
        return True
    if row["family"] in ("y_tile", "yk_tile") or row["input_h"] < s["in_h"]:
        return True
    if row["family"] != "k_tile":
        return False
    oc = hw_oc_for(s, row["oc_count"])
    aligned_in = _align_up(s["in_c"], p["align_c"])
    wt_bytes = s["kh"] * s["kw"] * aligned_in * FP16_BYTES * oc
    return wt_bytes <= CBUF_BANK_SIZE


def run_planned(s, inp, wt):
    rows, p, _, _ = plan_local_serial_rows(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    tasks = 0
    with TileSession() as session:
        for n in range(s["batch"]):
            for row in rows:
                oc_count = row["oc_count"]
                hw_oc = hw_oc_for(s, oc_count)
                tile_shape = dict(s, name=s["name"] + "_local", batch=1, groups=1,
                                  in_h=row["input_h"], out_c=hw_oc, weight_in_c=s["in_c"])
                y0 = row["y_start"] * s.get("stride", 1)
                tile_in = inp[n, :, y0:y0 + row["input_h"], :]
                tile_wt = _padded_weight(wt, row["k_start"], oc_count, hw_oc, s["in_c"], s["kh"], s["kw"])
                tile = run_hw_tile(tile_shape, tile_in, tile_wt,
                                   full_data_bank=_tile_full_data_bank(s, row, p), session=session)[:oc_count]
                oh = min(tile.shape[1], row["output_h"], p["out_h"] - row["y_start"])
                got[n, row["k_start"]:row["k_start"] + oc_count, row["y_start"]:row["y_start"] + oh] = tile[:, :oh]
                tasks += 1
    return got, tasks, rows[0]["split_method"]


def run_depthwise_serial(s, inp, wt):
    """Depthwise spatial: per-channel group=1 tiles (proven). plan_depthwise_rows estimates batched cost."""
    p = _conv_params(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    tasks = 0
    base = dict(s, name=s["name"] + "_dw", batch=1, in_c=1, out_c=MIN_HW_OC, weight_in_c=1, groups=1)
    y_rows, _, _, _ = plan_local_serial_rows(base)
    stride = s.get("stride", 1)
    with TileSession() as session:
        for n in range(s["batch"]):
            for ch in range(s["out_c"]):
                for row in y_rows:
                    tile_shape = dict(base, name=s["name"] + f"_dw{ch}", in_h=row["input_h"])
                    tile_wt = np.zeros((MIN_HW_OC, 1, s["kh"], s["kw"]), dtype=np.float16)
                    tile_wt[0] = wt[ch, 0]
                    y0 = row["y_start"] * stride
                    tile_in = inp[n, ch:ch + 1, y0:y0 + row["input_h"], :]
                    y_semantics = row["family"] in ("y_tile", "yk_tile") or row["input_h"] < s["in_h"]
                    tile = run_hw_tile(tile_shape, tile_in, tile_wt, full_data_bank=y_semantics, session=session)[0]
                    oh = min(tile.shape[0], row["output_h"], p["out_h"] - row["y_start"])
                    got[n, ch, row["y_start"]:row["y_start"] + oh] = tile[:oh]
                    tasks += 1
    return got, tasks, "depthwise_serial"


def run_grouped_serial(s, inp, wt):
    if s["in_c"] % s["groups"] or s["out_c"] % s["groups"]:
        raise ValueError("grouped path requires divisible input/output channels")
    p = _conv_params(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    in_per, out_per = s["in_c"] // s["groups"], s["out_c"] // s["groups"]
    tasks = 0
    for n in range(s["batch"]):
        for g in range(s["groups"]):
            ic0, oc0 = g * in_per, g * out_per
            gshape = dict(s, name=s["name"] + f"_g{g}", batch=1, in_c=in_per, out_c=out_per,
                          weight_in_c=in_per, groups=1)
            g_got, g_tasks, _ = run_planned(gshape, inp[n:n + 1, ic0:ic0 + in_per], wt[oc0:oc0 + out_per])
            got[n, oc0:oc0 + out_per] = g_got[0]
            tasks += g_tasks
    return got, tasks, "grouped_serial"


def run_shape(s, dry_run=False):
    plan = plan_shape(s)
    rows, pp = plan["rows"], plan["p"]
    print(f"plan shape={s['name']} split={plan['split']} rows={len(rows)} "
          f"y_step={plan['y_step']} k_step={plan['k_step']} "
          f"out={pp['out_h']}x{pp['out_w']} groups={s['groups']}")
    if dry_run:
        if _is_depthwise(s["in_c"], s["out_c"], s["groups"]):
            for i, row in enumerate(rows[:16]):
                print(f"  [{i}] ch={row['ch_start']}+{row['ch_count']} "
                      f"y={row['y_start']}+{row['output_h']} in_h={row['input_h']}")
        else:
            for i, row in enumerate(rows[:16]):
                print(f"  [{i}] {row['family']} y={row['y_start']}+{row['output_h']} "
                      f"in_h={row['input_h']} k={row['k_start']}+{row['oc_count']}")
        if len(rows) > 16:
            print(f"  ... {len(rows) - 16} more rows")
        return True, 0.0, 0, plan["split"]

    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    expected = compute_expected(inp, wt, s)

    if _is_depthwise(s["in_c"], s["out_c"], s["groups"]):
        got, tasks, kind = run_depthwise_serial(s, inp, wt)
    elif s["groups"] != 1:
        got, tasks, kind = run_grouped_serial(s, inp, wt)
    else:
        got, tasks, kind = run_planned(s, inp, wt)

    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got.astype(np.float32), expected, atol=0.12, rtol=0.02))
    print(f"shape={s['name']} kind={kind} tasks={tasks} {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    return ok, max_diff, tasks, kind


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone first-principles FP16 CONV (local serial)")
    parser.add_argument("shape", nargs="?", help="encoded shape name")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print planner rows only, no submit")
    parser.add_argument("--sweep", action="store_true", help="delegate to conv_grok/sweep_217.py")
    parser.add_argument("--classify", action="store_true", help="delegate to sweep --classify (no NPU)")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--pattern", type=str, default="")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--extra-hard", action="store_true", help="run EXTRA_HARD_SHAPES stress set")
    parser.add_argument("--batch", type=int, help="make_shape: batch")
    parser.add_argument("--in-c", type=int, help="make_shape: input channels")
    parser.add_argument("--in-h", type=int, help="make_shape: input height")
    parser.add_argument("--in-w", type=int, help="make_shape: input width")
    parser.add_argument("--out-c", type=int, help="make_shape: output channels")
    parser.add_argument("--kh", type=int, help="make_shape: kernel height")
    parser.add_argument("--kw", type=int, help="make_shape: kernel width")
    parser.add_argument("--groups", type=int, help="make_shape: groups")
    parser.add_argument("--stride", type=int, default=1, help="make_shape: stride")
    parser.add_argument("--pvalid", action="store_true", help="make_shape: valid padding tag")
    args = parser.parse_args(argv)
    if args.sweep or args.classify:
        # Keep conv.py lean: forward to sibling sweep harness.
        sweep_argv = []
        if args.sweep:
            sweep_argv.append("--sweep")
        if args.classify:
            sweep_argv.append("--classify")
        sweep_argv += ["--timeout", str(args.timeout)]
        if args.limit:
            sweep_argv += ["--limit", str(args.limit)]
        if args.start:
            sweep_argv += ["--start", str(args.start)]
        if args.pattern:
            sweep_argv += ["--pattern", args.pattern]
        if args.skip_health:
            sweep_argv.append("--skip-health")
        if args.stop_on_error:
            sweep_argv.append("--stop-on-error")
        import importlib.util
        sweep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_217.py")
        spec = importlib.util.spec_from_file_location("conv_grok_sweep_217", sweep_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main(sweep_argv)
    if args.list:
        print("encoded: [conv2d_]bN_cN_hN_wN_ocN_wicN_kHxW_gN[_sN][_pvalid]")
        for status, name in LIST_SHAPES:
            print(f"{status:8s} {name}")
        print("extra-hard:")
        for name in EXTRA_HARD_SHAPES:
            print(f"{'extra':8s} {name}")
        return 0
    if args.extra_hard:
        names = EXTRA_HARD_SHAPES
        shapes = None
    elif args.in_c is not None:
        s = make_shape(batch=args.batch or 1, in_c=args.in_c, in_h=args.in_h or 14,
                       in_w=args.in_w, out_c=args.out_c or args.in_c, kh=args.kh or 1, kw=args.kw,
                       groups=args.groups or 1, stride=args.stride, pvalid=args.pvalid,
                       name=args.shape)
        names = (s["name"],)
        shapes = [s]
    else:
        names = (args.shape,) if args.shape else DEFAULT_SMOKE
        shapes = None
    failed = 0
    for i, name in enumerate(names):
        try:
            s = shapes[i] if shapes else shape_from_name(name)
            ok, _, _, _ = run_shape(s, dry_run=args.dry_run)
            if not args.dry_run and not ok:
                failed += 1
        except Exception as exc:
            print(f"shape={name} ERROR: {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
