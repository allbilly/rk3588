#!/usr/bin/env python3
import argparse
import re
import sys
from collections import Counter
from pathlib import Path


OPS = Path.home() / "npu" / "ops_rknn"
DUMP = OPS / "dump"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conv_expt import conv_tile_planner as planner  # noqa: E402
from examples.conv import shape_from_name  # noqa: E402

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def clean_line(line):
    return ANSI_RE.sub("", line)


TARGETS = {
    "h160_by_y": {
        "shape": "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid",
        "case": "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid 1 16 160 160 128 3 3 1 h160_by_y",
        "emit": "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid_emit.txt",
    },
    "h14_by_k": {
        "shape": "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid",
        "case": "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid 1 160 14 14 320 3 3 1 h14_by_k",
        "emit": "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid_emit.txt",
    },
    "cc_c256_h14_pw_by_k": {
        "shape": "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1",
        "case": "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1 1 256 14 14 512 1 1 1 cc_c256_h14_pw_by_k",
        "emit": None,
        "capture": "experimental/rknn/capture_rknpu_submit_dump_gems_pw_c256_h14_live_regcmd_832_20260524_131749.log",
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1_dump.txt",
        "note": "promoted after live regcmd-base comparison; pointwise exact-11 body rows use setup/k_half plus full-OC y_tile windows",
    },
    "c256_h14_pw_by_k": {
        "shape": "b1_c256_h14_w14_oc512_wic256_k1x1_g1",
        "case": "b1_c256_h14_w14_oc512_wic256_k1x1_g1 1 256 14 14 512 1 1 1 c256_h14_pw_by_k",
        "emit": None,
        "capture": "experimental/rknn/capture_rknpu_submit_dump_gems_pw_c256_h14_live_regcmd_832_20260524_131749.log",
        "note": "same RKNN capture covers the unprefixed alias; promoted with the pointwise exact-11 body rows proven on the prefixed shape",
    },
    "c128_h28_pw_by_k": {
        "shape": "b1_c128_h28_w28_oc256_wic128_k1x1_g1",
        "case": "b1_c128_h28_w28_oc256_wic128_k1x1_g1 1 128 28 28 256 1 1 1 c128_h28_pw_by_k",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1_dump.txt",
        "note": "unprefixed alias promoted with the pointwise exact-11 body rows proven by live regcmd-base capture on the prefixed shape",
    },
    "h7_by_k": {
        "shape": "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid",
        "case": "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid 1 160 7 7 320 3 3 1 h7_by_k",
        "emit": None,
    },
    "h40_by_yk": {
        "shape": "b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid",
        "case": "b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid 1 128 40 40 40 1 1 1 h40_by_yk",
        "emit": None,
    },
    "h40_spatial_by_yk": {
        "shape": "b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid",
        "case": "b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid 1 160 40 40 320 3 3 1 h40_spatial_by_yk",
        "emit": None,
    },
    "c144_h28_none": {
        "shape": "conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1",
        "case": "conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1 1 144 28 28 32 1 1 1 c144_h28_pointwise_none",
        "emit": "conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1_emit.txt",
    },
    "c192_h28_none": {
        "shape": "conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1",
        "case": "conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1 1 192 28 28 32 1 1 1 c192_h28_pointwise_none",
        "emit": None,
    },
    "h320_direct_by_y": {
        "shape": "b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid",
        "case": "b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid 1 3 320 320 32 3 3 1 h320_direct_by_y",
        "emit": None,
    },
    "c3_h224_spatial_by_y": {
        "shape": "conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1",
        "case": "conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1 1 3 224 224 32 3 3 1 c3_h224_spatial_by_y",
        "emit": None,
    },
    "c8_h160_spatial_by_y": {
        "shape": "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid",
        "case": "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid 1 8 160 160 16 3 3 1 c8_h160_spatial_by_y",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c8_h160_keep1_gem5/dump_gem5.txt",
        "note": "kept fenced: RKNN prefix output evidence showed 138/20 output windows, but local replay was numerically wrong; this artifact is an output GEM dump, not task/regcmd closure evidence",
    },
    "c96_h56_pw_by_y": {
        "shape": "conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1",
        "case": "conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1 1 96 56 56 24 1 1 1 c96_h56_pw_by_y",
        "emit": None,
    },
    "c96_h56_oc32_pw_by_y": {
        "shape": "conv2d_b1_c96_h56_w56_oc32_wic96_k1x1_g1",
        "case": "conv2d_b1_c96_h56_w56_oc32_wic96_k1x1_g1 1 96 56 56 32 1 1 1 c96_h56_oc32_pw_by_y",
        "emit": None,
        "note": "promoted after one-off local tile replay passed: 2 one-task submits, max_diff=0.0156, post simple_add PASS",
    },
    "c96_h56_oc16_pw_by_y": {
        "shape": "conv2d_b1_c96_h56_w56_oc16_wic96_k1x1_g1",
        "case": "conv2d_b1_c96_h56_w56_oc16_wic96_k1x1_g1 1 96 56 56 16 1 1 1 c96_h56_oc16_pw_by_y",
        "emit": None,
        "note": "promoted after one-off local tile replay passed: 2 one-task submits, max_diff=0.0156, post simple_add PASS",
    },
    "c64_h56_pw_by_y": {
        "shape": "conv2d_b1_c64_h56_w56_oc24_wic64_k1x1_g1",
        "case": "conv2d_b1_c64_h56_w56_oc24_wic64_k1x1_g1 1 64 56 56 24 1 1 1 c64_h56_pw_by_y",
        "emit": None,
        "note": "promoted after one-off local tile replay passed: 2 one-task submits, max_diff=0.0155, post simple_add PASS",
    },
    "c128_h56_pw_by_y": {
        "shape": "conv2d_b1_c128_h56_w56_oc24_wic128_k1x1_g1",
        "case": "conv2d_b1_c128_h56_w56_oc24_wic128_k1x1_g1 1 128 56 56 24 1 1 1 c128_h56_pw_by_y",
        "emit": None,
        "note": "promoted after one-off local tile replay passed: 3 one-task submits, max_diff=0.0233, post simple_add PASS",
    },
    "c256_h28_oc16_pw_by_y": {
        "shape": "conv2d_b1_c256_h28_w28_oc16_wic256_k1x1_g1",
        "case": "conv2d_b1_c256_h28_w28_oc16_wic256_k1x1_g1 1 256 28 28 16 1 1 1 c256_h28_oc16_pw_by_y",
        "emit": None,
        "note": "kept fenced: one-off local tile replay was numerically wrong (max_diff=119.6110), so narrow-OC pointwise-wide BY_Y still needs RKNN closure semantics",
    },
    "c144_h56_pw_by_y": {
        "shape": "conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1",
        "case": "conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1 1 144 56 56 24 1 1 1 c144_h56_pw_by_y",
        "emit": None,
    },
    "c144_h75_pw_by_y": {
        "shape": "b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid",
        "case": "b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid 1 144 75 75 24 1 1 1 c144_h75_pw_by_y",
        "emit": None,
    },
    "c144_h38_pw_by_y": {
        "shape": "b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid",
        "case": "b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid 1 144 38 38 32 1 1 1 c144_h38_pw_by_y",
        "emit": None,
    },
    "c192_h38_pw_by_y": {
        "shape": "b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid",
        "case": "b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid 1 192 38 38 32 1 1 1 c192_h38_pw_by_y",
        "emit": None,
    },
    "c576_h19_pw_by_y": {
        "shape": "b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid",
        "case": "b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid 1 576 19 19 12 1 1 1 c576_h19_pw_by_y",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_keep1_gem1/dump_gem1.txt",
        "note": "kept fenced: local replay timed out; RKNN task dump shows 12 descriptors with amounts 108,108,104,26,104,26,104,26,104,26,104,26 and masks 0x0d/0x60, so this needs RKNN closure semantics",
    },
    "c128_h80_pw_by_y": {
        "shape": "b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid",
        "case": "b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid 1 128 80 80 16 1 1 1 c128_h80_pw_by_y",
        "emit": None,
    },
    "c64_h80_pw_by_y": {
        "shape": "b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid",
        "case": "b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid 1 64 80 80 16 1 1 1 c64_h80_pw_by_y",
        "emit": None,
    },
    "c160_h40_pw_by_yk": {
        "shape": "b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid",
        "case": "b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid 1 160 40 40 40 1 1 1 c160_h40_pw_by_yk",
        "emit": None,
    },
    "cc_c32_h112_pw_by_yk": {
        "shape": "conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1",
        "case": "conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1 1 32 112 112 64 1 1 1 cc_c32_h112_pw_by_yk",
        "emit": None,
    },
    "cc_c64_h56_pw_by_yk": {
        "shape": "conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1",
        "case": "conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1 1 64 56 56 128 1 1 1 cc_c64_h56_pw_by_yk",
        "emit": None,
    },
    "cc_c128_h56_pw_by_yk": {
        "shape": "conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1",
        "case": "conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1 1 128 56 56 128 1 1 1 cc_c128_h56_pw_by_yk",
        "emit": None,
    },
    "c32_h112_pw_by_yk": {
        "shape": "b1_c32_h112_w112_oc64_wic32_k1x1_g1",
        "case": "b1_c32_h112_w112_oc64_wic32_k1x1_g1 1 32 112 112 64 1 1 1 c32_h112_pw_by_yk",
        "emit": None,
    },
    "c32_h112_oc16_pw_by_y": {
        "shape": "b1_c32_h112_w112_oc16_wic32_k1x1_g1",
        "case": "b1_c32_h112_w112_oc16_wic32_k1x1_g1 1 32 112 112 16 1 1 1 c32_h112_oc16_pw_by_y",
        "emit": None,
        "note": "handled by generic pointwise BY_Y/y_tile path: 4 generated rows, max_diff=0.0146, post simple_add PASS",
    },
    "c64_h56_pw_by_yk": {
        "shape": "b1_c64_h56_w56_oc128_wic64_k1x1_g1",
        "case": "b1_c64_h56_w56_oc128_wic64_k1x1_g1 1 64 56 56 128 1 1 1 c64_h56_pw_by_yk",
        "emit": None,
    },
    "c128_h56_pw_by_yk": {
        "shape": "b1_c128_h56_w56_oc128_wic128_k1x1_g1",
        "case": "b1_c128_h56_w56_oc128_wic128_k1x1_g1 1 128 56 56 128 1 1 1 c128_h56_pw_by_yk",
        "emit": None,
    },
    "c16_h150_pw_by_yk": {
        "shape": "b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid",
        "case": "b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid 1 16 150 150 96 1 1 1 c16_h150_pw_by_yk",
        "emit": None,
    },
    "g3_c3_h5_spatial_serial": {
        "shape": "conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3",
        "case": "conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3 1 3 5 7 6 3 3 3 g3_c3_h5_spatial_serial",
        "emit": None,
    },
    "g3_c3_h11_w28_spatial_serial": {
        "shape": "conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3",
        "case": "conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3 1 3 11 28 3 3 3 3 g3_c3_h11_w28_spatial_serial",
        "emit": None,
        "note": "promoted after one-off local grouped_serial replay passed (1-row depthwise setup dispatch): 3 one-task submits, max_diff=0.0038, post simple_add PASS; sibling shape conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3 also passes via the same dispatch with 6 one-task submits",
    },
    "g2_c4_h5_oc4_spatial_serial": {
        "shape": "conv2d_b1_c4_h5_w5_oc4_wic2_k3x3_g2",
        "case": "conv2d_b1_c4_h5_w5_oc4_wic2_k3x3_g2 1 4 5 5 4 3 3 2 g2_c4_h5_oc4_spatial_serial",
        "emit": None,
    },
    "g2_c4_h5_oc8_spatial_serial": {
        "shape": "conv2d_b1_c4_h5_w5_oc8_wic2_k3x3_g2",
        "case": "conv2d_b1_c4_h5_w5_oc8_wic2_k3x3_g2 1 4 5 5 8 3 3 2 g2_c4_h5_oc8_spatial_serial",
        "emit": None,
    },
    "g2_c4_h5_oc12_spatial_serial": {
        "shape": "conv2d_b1_c4_h5_w5_oc12_wic2_k3x3_g2",
        "case": "conv2d_b1_c4_h5_w5_oc12_wic2_k3x3_g2 1 4 5 5 12 3 3 2 g2_c4_h5_oc12_spatial_serial",
        "emit": None,
    },
    "g3_c6_h5_oc6_spatial_serial": {
        "shape": "conv2d_b1_c6_h5_w5_oc6_wic2_k3x3_g3",
        "case": "conv2d_b1_c6_h5_w5_oc6_wic2_k3x3_g3 1 6 5 5 6 3 3 3 g3_c6_h5_oc6_spatial_serial",
        "emit": None,
    },
    "g3_c6_h5_oc12_spatial_serial": {
        "shape": "conv2d_b1_c6_h5_w5_oc12_wic2_k3x3_g3",
        "case": "conv2d_b1_c6_h5_w5_oc12_wic2_k3x3_g3 1 6 5 5 12 3 3 3 g3_c6_h5_oc12_spatial_serial",
        "emit": None,
    },
    "g3_c6_h5_oc18_spatial_serial": {
        "shape": "conv2d_b1_c6_h5_w5_oc18_wic2_k3x3_g3",
        "case": "conv2d_b1_c6_h5_w5_oc18_wic2_k3x3_g3 1 6 5 5 18 3 3 3 g3_c6_h5_oc18_spatial_serial",
        "emit": None,
    },
    "g5_c15_h5_oc20_spatial_serial": {
        "shape": "conv2d_b1_c15_h5_w5_oc20_wic3_k3x3_g5",
        "case": "conv2d_b1_c15_h5_w5_oc20_wic3_k3x3_g5 1 15 5 5 20 3 3 5 g5_c15_h5_oc20_spatial_serial",
        "emit": None,
    },
    "g5_c15_h5_oc25_spatial_serial": {
        "shape": "conv2d_b1_c15_h5_w5_oc25_wic3_k3x3_g5",
        "case": "conv2d_b1_c15_h5_w5_oc25_wic3_k3x3_g5 1 15 5 5 25 3 3 5 g5_c15_h5_oc25_spatial_serial",
        "emit": None,
    },
    "g5_c15_h5_oc30_spatial_serial": {
        "shape": "conv2d_b1_c15_h5_w5_oc30_wic3_k3x3_g5",
        "case": "conv2d_b1_c15_h5_w5_oc30_wic3_k3x3_g5 1 15 5 5 30 3 3 5 g5_c15_h5_oc30_spatial_serial",
        "emit": None,
    },
    "g5_c15_h5_oc35_spatial_serial": {
        "shape": "conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5",
        "case": "conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5 1 15 5 5 35 3 3 5 g5_c15_h5_oc35_spatial_serial",
        "emit": None,
    },
    "g5_c15_h5_oc40_spatial_serial": {
        "shape": "conv2d_b1_c15_h5_w5_oc40_wic3_k3x3_g5",
        "case": "conv2d_b1_c15_h5_w5_oc40_wic3_k3x3_g5 1 15 5 5 40 3 3 5 g5_c15_h5_oc40_spatial_serial",
        "emit": None,
    },
    "g3_c3_h1w11_k1x5_serial": {
        "shape": "b1_c3_h1_w11_oc6_wic1_k1x5_g3",
        "case": "b1_c3_h1_w11_oc6_wic1_k1x5_g3 1 3 1 11 6 1 5 3 g3_c3_h1w11_k1x5_serial",
        "emit": None,
    },
    "b8_g3_c3_h1w11_k1x5_serial": {
        "shape": "b8_c3_h1_w11_oc6_wic1_k1x5_g3",
        "case": "b8_c3_h1_w11_oc6_wic1_k1x5_g3 8 3 1 11 6 1 5 3 b8_g3_c3_h1w11_k1x5_serial",
        "emit": None,
    },
    "b4_g5_c15_h5_oc35_spatial_serial": {
        "shape": "conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5",
        "case": "conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5 4 15 5 5 35 3 3 5 b4_g5_c15_h5_oc35_spatial_serial",
        "emit": None,
    },
    "c256_h3_oc24_pw_none": {
        "shape": "b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid",
        "case": "b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid 1 256 3 3 24 1 1 1 c256_h3_oc24_pw_none",
        "emit": None,
        "note": "pointwise-wide NONE remains fenced: one-off local setup-row replay was numerically wrong (max_diff=75.3440), so RKNN 108-row closure is still required",
    },
    "c256_h2_oc24_pw_none": {
        "shape": "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid",
        "case": "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid 1 256 2 2 24 1 1 1 c256_h2_oc24_pw_none",
        "emit": None,
        "note": "pointwise-wide NONE remains fenced: one-off local setup-row replay was numerically wrong (max_diff=69.0113), so RKNN 108-row closure is still required",
    },
    "c128_h1_oc24_pw_none": {
        "shape": "b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid",
        "case": "b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid 1 128 1 1 24 1 1 1 c128_h1_oc24_pw_none",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c128_h1_none_keep1_gem1/dump_gem1.txt",
        "note": "fresh RKNN prefix evidence: original submit is three one-task subcores; decoded task descriptors are 108/108/108, all mask 0x0d, all same regcmd_addr. Local three-lane same-row probe timed out and one-off local setup-row replay was numerically wrong (max_diff=64.3627), so regcmd/body value semantics remain unresolved.",
    },
    "dw32_h32_pw_serial": {
        "shape": "conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32",
        "case": "conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32 1 32 32 32 32 1 1 32 dw32_h32_pw_serial",
        "emit": None,
        "note": "promoted after one-off local grouped_serial replay passed: 32 one-task submits, max_diff=0.0010, post simple_add PASS; path is depthwise+1x1+setup/NONE, gated by run_grouped_serial_shape (no RKNN 108-row closure needed); sibling shapes conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3 and conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3 also pass via the same 1-row depthwise dispatch with grouped_serial submits=3 and 6 respectively; depthwise BY_K path (4 k_tile submits) on b1_c128_h3_w3_oc128_wic1_k3x3_g128_s1_pvalid confirmed as timing out (6s NPU submit timeout), so multi-row depthwise remains fenced pending RKNN 108/104/26-row closure derivation; conv2d_multi cross-check of fenced shapes (c32_h14_oc64, c32_h7_oc128, c16_h80_oc64, c16_h80_oc128, c128_h1_oc24, c256_h3_oc24, c256_h2_oc24, c40_h40_oc320, c8_h160, c256_h28_oc256, c576_h19_oc12, c96_h20_oc96) is VALID when the .rknn models are built with /tmp/build_matched.py which constructs weights from the same MT19937(seed=0) sequence that conv2d_multi consumes for its CPU reference (replacing the input RNG portion and then producing the weight RNG in the exact same order). All 12 representative fenced shapes PASS in rknn_runtime with max error 0.0070-0.0447. This proves the NPU works for all 12; the conv.py fences reflect missing register setup, not NPU bugs.",
    },
    "c32_h14_oc64_promoted_via_generic_fix": {
        "shape": "b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid",
        "case": "b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid 1 32 14 14 64 3 3 1 c32_h14_oc64",
        "emit": None,
        "note": "promoted via 3 generic make_regs fixes (cvt_con5=0, CORE_MISC_CFG=0x200, SURFACE_ADD=*2 instead of *max(2,align_out_c/16)). Un-fenced from KNOWN_BAD_SPATIAL_SETUP_SHAPES. Live regcmd from rknn_runtime shows generic produces correct values for oc=64. max_diff=0.0304 PASS. Sibling c32_h7_oc128 also passes via same generic fix. The 2 c16_h80_oc* shapes remain fenced because they are 11-task BY_K, not 1-task setup.",
    },
    "c32_h7_oc128_promoted_via_generic_fix": {
        "shape": "b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid",
        "case": "b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid 1 32 7 7 128 3 3 1 c32_h7_oc128",
        "emit": None,
        "note": "promoted via 3 generic make_regs fixes (cvt_con5=0, CORE_MISC_CFG=0x200, SURFACE_ADD=*2 instead of *max(2,align_out_c/16)). Live regcmd from rknn_runtime shows generic produces correct values for oc=128. max_diff=0.0265 PASS.",
    },
    "c8_h160_oc16_promoted_via_local_tile_replay": {
        "shape": "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid",
        "case": "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid 1 8 160 160 16 3 3 1 c8_h160_oc16",
        "note": "promoted via local_tile_replay: 3 planner y_tile rows (in_h=70/70/24, y_start=0/68/136, oc=16). The 70-2 stride is the standard 3x3 y_tile overlap pattern. Sibling c16_h160_oc128 uses the same y_tile pattern with 3 rows. Note that the live RKNN shows 138/20 output windows; the local replay uses 70/70/24 input which doesn't match the live, but the smaller-input passes the NPU's CBUF and produces numerically correct output. max_diff=0.0156 PASS.",
    },
    "c16_h80_oc64_promoted_via_exact11_byk": {
        "shape": "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid",
        "case": "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid 1 16 80 80 64 3 3 1 c16_h80_oc64",
        "note": "promoted via 11-task EXACT11 BY_K closure with per-shape overrides CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0. Un-fenced from KNOWN_BAD_SPATIAL_SETUP_SHAPES. Live regcmd captured at GEM 2 offset 0x5000 (98 qwords of setup body) confirms body field matches when overrides are applied. max_diff=0.0293 PASS. Sibling c16_h80_oc128 still FENCED because the hardcoded k_tile OC splits ((0,112),(112,112),(224,96)) are for oc=320, not oc=128; need parameterization of _exact11_task_regs k_tile splits. Sibling c16_h80_oc128_k5x5 still FENCED (also needs k5x5 body overrides).",
    },
    "cc_c128_h28_pw_by_k_dump": {
        "shape": "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1",
        "case": "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1 1 128 28 28 256 1 1 1 cc_c128_h28_pw_by_k_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1_dump.txt",
        "note": "promoted after live regcmd-base capture: generated task rows match RKNN GEM2 active regcmd stream and guarded submit passes",
    },
    "cc_c256_h28_pw_by_yk_dump": {
        "shape": "conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1",
        "case": "conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1 1 256 28 28 256 1 1 1 cc_c256_h28_pw_by_yk_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1_dump.txt",
        "note": "dump-only evidence for a remaining BY_YK fence; local setup-row replay was numerically wrong (max_diff=122.6959). RKNN rows show full-OC setup/y_tile rows with 128-channel k_half rows and PPU/PDP separators, so mixed Y/K task layout remains unresolved.",
    },
    "cc_c64_h112_dw_by_yk_dump": {
        "shape": "conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64",
        "case": "conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64 1 64 112 112 64 3 3 64 cc_c64_h112_dw_by_yk_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64_dump.txt",
        "note": "dump-only evidence for a remaining depthwise BY_YK fence; RKNN rows include irregular setup prologue rows, repeated 64-channel setup/y_tile pairs, and PPU/PDP separators, so grouped/depthwise replay is not promoted from this dump",
    },
    "cc_c32_h112_dw_by_y_dump": {
        "shape": "conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32",
        "case": "conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32 1 32 112 112 32 3 3 32 cc_c32_h112_dw_by_y_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32_dump.txt",
        "note": "dump-only evidence for a remaining depthwise BY_Y fence; RKNN rows include setup prologue rows, 32-channel setup/y_tile pairs, and PPU/PDP separators, so grouped/depthwise replay still needs task-safe closure",
    },
}


def emit_rows(path):
    if path is None or not path.exists():
        return []
    rows, cur = [], []
    for line in path.read_text().splitlines():
        line = clean_line(line)
        line = re.sub(r"\bEMIT\s*\(\s*", "EMIT(", line)
        start = line.find("EMIT(REG_")
        if start < 0:
            continue
        emit = line[start:]
        if emit.endswith(";"):
            emit = emit[:-1]
        reg, expr = emit[5:-1].split(",", 1)
        cur.append((reg, expr.strip()))
        if reg == "REG_PC_OPERATION_ENABLE":
            rows.append(cur)
            cur = []
    return rows


def family(row):
    regs = {reg: expr for reg, expr in row}
    if any(reg.startswith("REG_PPU") or reg.startswith("REG_PDP") for reg in regs):
        return "ppu_pdp"
    conv2 = regs.get("REG_CNA_CONV_CON2", "")
    if "RESERVED_0(64)" in conv2:
        return "k_half"
    if "RESERVED_0(32)" in conv2:
        return "y_tile"
    if "RESERVED_0(80)" in conv2:
        return "k_tile"
    return "setup" if conv2 or "REG_CNA_DATA_SIZE0" in regs else "tail"


def field(expr, name):
    m = re.search(name + r"\((\d+)\)", expr or "")
    return int(m.group(1)) if m else None


def summarize_row(row):
    regs = {reg: expr for reg, expr in row}
    row_family = family(row)
    is_compute = row_family in {"setup", "k_half", "k_tile", "y_tile"}
    out_ch = field(regs.get("REG_CORE_DATAOUT_SIZE_1"), "DATAOUT_CHANNEL")
    return {
        "family": row_family,
        "input_h": field(regs.get("REG_CNA_DATA_SIZE0"), "DATAIN_HEIGHT") if is_compute else None,
        "out_h": field(regs.get("REG_CORE_DATAOUT_SIZE_0"), "DATAOUT_HEIGHT") if is_compute else None,
        "out_c": out_ch + 1 if out_ch is not None and is_compute else None,
        "amount": field(regs.get("REG_PC_REGISTER_AMOUNTS"), "PC_DATA_AMOUNT"),
    }


def capture_task_summary(path):
    if path is None or not path.exists():
        return None
    current_gem = None
    tasks = []
    submit_task_number = None
    for line in path.read_text().splitlines():
        gem = re.match(r"GEM (\d+):", line)
        if gem:
            current_gem = int(gem.group(1))
            continue
        submit = re.search(r"submit #\d+: task_number=(\d+)", line)
        if submit:
            submit_task_number = int(submit.group(1))
            continue
        task = re.match(
            r"\s+task_like\[(\d+)\].*op_idx=(\d+) enable_mask=0x([0-9a-fA-F]+).*"
            r"regcfg_amount=(\d+).*regcmd_addr=0x([0-9a-fA-F]+)",
            line,
        )
        if current_gem == 1 and task:
            tasks.append({
                "idx": int(task.group(1)),
                "op_idx": int(task.group(2)),
                "mask": int(task.group(3), 16),
                "amount": int(task.group(4)),
                "regcmd_addr": int(task.group(5), 16),
            })
    if not tasks:
        return {"status": "missing_tasks"}
    base = tasks[0]["regcmd_addr"]
    offsets = tuple((task["regcmd_addr"] - base) // 8 for task in tasks)
    return {
        "status": "ok",
        "submit_task_number": submit_task_number,
        "amounts": tuple(task["amount"] for task in tasks),
        "masks": tuple(task["mask"] for task in tasks),
        "op_idx": tuple(task["op_idx"] for task in tasks),
        "offsets": offsets,
    }


def dump_task_summary(path, gem_index=1):
    if path is None or not path.exists():
        return None
    in_tasks = False
    current = None
    tasks = []
    for raw in path.read_text(errors="replace").splitlines():
        line = clean_line(raw)
        if line.startswith("Decoded rknpu_task entries for GEM "):
            in_tasks = int(line.rsplit(" ", 1)[1]) == gem_index
            current = None
            continue
        if not in_tasks:
            continue
        task = re.match(r"Task (\d+) @ offset 0x([0-9a-fA-F]+)", line)
        if task:
            if current:
                tasks.append(current)
            current = {"idx": int(task.group(1)), "task_offset": int(task.group(2), 16)}
            continue
        if current is None:
            continue
        m = re.match(r"\s+op_idx\s+:\s+(\d+)", line)
        if m:
            current["op_idx"] = int(m.group(1))
            continue
        m = re.match(r"\s+enable_mask\s+:\s+0x([0-9a-fA-F]+)", line)
        if m:
            current["mask"] = int(m.group(1), 16)
            continue
        m = re.match(r"\s+regcfg_amount:\s+(\d+) entries", line)
        if m:
            current["amount"] = int(m.group(1))
            continue
        m = re.match(r"\s+regcmd_addr\s+:\s+0x([0-9a-fA-F]+)", line)
        if m:
            current["regcmd_addr"] = int(m.group(1), 16)
            continue
        if line.startswith("Decoded ") or line.startswith("==="):
            break
    if current:
        tasks.append(current)
    if not tasks:
        return {"status": "missing_tasks"}
    base = tasks[0].get("regcmd_addr")
    offsets = None
    if base is not None and all("regcmd_addr" in task for task in tasks):
        offsets = tuple((task["regcmd_addr"] - base) // 8 for task in tasks)
    return {
        "status": "ok",
        "submit_task_number": None,
        "amounts": tuple(task.get("amount") for task in tasks),
        "masks": tuple(task.get("mask") for task in tasks),
        "op_idx": tuple(task.get("op_idx") for task in tasks),
        "offsets": offsets,
    }


def print_task_summary(prefix, summary):
    if summary is None:
        print(f"  {prefix}_status=missing_file")
        return
    if summary["status"] != "ok":
        print(f"  {prefix}_status={summary['status']}")
        return
    if summary["submit_task_number"] is not None:
        print(f"  {prefix}_submit_task_number={summary['submit_task_number']}")
    print(f"  {prefix}_amounts=" + ",".join(str(value) for value in summary["amounts"]))
    print(f"  {prefix}_masks=" + ",".join(hex(value) for value in summary["masks"]))
    if summary["offsets"] is not None:
        print(f"  {prefix}_offsets=" + ",".join(str(value) for value in summary["offsets"]))


def task_pattern(summary):
    if summary is None or summary["status"] != "ok":
        return None
    amounts = summary["amounts"]
    masks = summary["masks"]
    if len(amounts) == 11 and amounts == (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26):
        if masks == (0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60):
            return "by_k_exact11_108_104_26"
    return f"tasks{len(amounts)}_" + "_".join(str(value) for value in amounts[:6])


def row_pattern(rows):
    families = tuple(summarize_row(row)["family"] for row in rows)
    if len(families) == 12 and families[0] == "setup" and families[-1] == "tail":
        body = families[1:-1]
        if body == (
            "k_half", "ppu_pdp", "k_half", "ppu_pdp", "y_tile",
            "ppu_pdp", "y_tile", "ppu_pdp", "y_tile", "ppu_pdp",
        ):
            return "by_k_exact12_rows_setup_body_tail"
    return f"rows{len(families)}_" + ",".join(families[:8])


def print_alignment_summary(rows, task_summaries, meta=None):
    print(f"  row_pattern={row_pattern(rows)}")
    for name, summary in task_summaries:
        pattern = task_pattern(summary)
        if pattern:
            print(f"  {name}_pattern={pattern}")
    if replay_hint(rows, task_summaries, meta):
        print("  replay_hint=by_k_exact11_metadata_with_setup_tail_rows")


def target_evidence(meta):
    rows = emit_rows(DUMP / meta["emit"] if meta.get("emit") else None)
    if not rows and meta.get("dump"):
        rows = emit_rows(Path(meta["dump"]).expanduser())
    task_summaries = []
    if meta.get("capture"):
        task_summaries.append(("capture", capture_task_summary(REPO_ROOT / meta["capture"])))
    if meta.get("dump"):
        task_summaries.append(("dump_task", dump_task_summary(Path(meta["dump"]).expanduser())))
    return rows, task_summaries


def replay_hint(rows, task_summaries, meta=None):
    if meta and "numerically wrong" in meta.get("note", ""):
        return False
    return (row_pattern(rows) == "by_k_exact12_rows_setup_body_tail" and
            any(task_pattern(summary) == "by_k_exact11_108_104_26" for _, summary in task_summaries))


def fenced_shapes(summary_path):
    shapes = []
    for line in summary_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit() or parts[1] != "FENCED":
            continue
        run = next((part[4:] for part in parts if part.startswith("run=")), None)
        if run:
            shapes.append(run)
    return shapes


def print_coverage_summary(summary_path):
    if not summary_path.exists():
        print(f"summary={summary_path}")
        print("status=missing_summary")
        return
    by_shape = {}
    for key, meta in TARGETS.items():
        by_shape.setdefault(meta["shape"], []).append(key)
    fenced = fenced_shapes(summary_path)
    covered = [(shape, by_shape[shape]) for shape in fenced if shape in by_shape]
    captured = [(shape, keys) for shape, keys in covered
                if any(TARGETS[key].get("capture") for key in keys)]
    replay_ready = []
    for shape, keys in covered:
        for key in keys:
            rows, task_summaries = target_evidence(TARGETS[key])
            if rows and replay_hint(rows, task_summaries, TARGETS[key]):
                replay_ready.append((shape, key))
                break
    uncovered = [shape for shape in fenced if shape not in by_shape]
    buckets = Counter()
    bucket_examples = {}
    for shape in uncovered:
        rows = planner.descriptor_rows_for_shape(shape_from_name(shape))
        split = rows[0]["split_method"] if rows else "none"
        families = ",".join(sorted({row["family"] for row in rows})) if rows else "none"
        key = (split, families)
        buckets[key] += 1
        bucket_examples.setdefault(key, []).append(shape)
    print(f"summary={summary_path}")
    print(f"fenced_total={len(fenced)} manifest_covered={len(covered)} capture_covered={len(captured)} replay_ready={len(replay_ready)}")
    print(f"uncovered_total={len(uncovered)}")
    for (split, families), count in sorted(buckets.items(), key=lambda item: (-item[1], item[0])):
        print(f"  uncovered_bucket split={split} families={families} count={count}")
        for shape in bucket_examples[(split, families)][:5]:
            print(f"    sample={shape}")
    for shape, keys in covered:
        capture_keys = [key for key in keys if TARGETS[key].get("capture")]
        tag = "capture" if capture_keys else "manifest"
        print(f"  {tag} shape={shape} targets={','.join(keys)}")
    for shape, key in replay_ready:
        print(f"  replay_ready shape={shape} target={key} hint=by_k_exact11_metadata_with_setup_tail_rows")


def main():
    parser = argparse.ArgumentParser(description="Print RKNN prefix replay manifest without submitting hardware")
    parser.add_argument("target", nargs="?", choices=sorted(TARGETS), help="target to describe")
    parser.add_argument("--coverage-summary", type=Path, help="summarize manifest coverage for a conv.py sweep summary")
    args = parser.parse_args()
    if args.coverage_summary:
        print_coverage_summary(args.coverage_summary)
        return
    keys = [args.target] if args.target else sorted(TARGETS)
    for key in keys:
        meta = TARGETS[key]
        rows, task_summaries = target_evidence(meta)
        print(f"target={key} shape={meta['shape']} emit_rows={len(rows)}")
        if meta["emit"] is None:
            print("  status=no_emit_manifest use hardware submit decode for task topology")
        elif not rows:
            print("  status=missing_emit_dump generate RKNN model/dump before prefix replay")
        if meta.get("capture"):
            print(f"  capture={meta['capture']}")
            capture = next(summary for name, summary in task_summaries if name == "capture")
            print_task_summary("capture", capture)
        if meta.get("dump"):
            dump = Path(meta["dump"]).expanduser()
            print(f"  dump={dump}")
            print(f"  dump_status={'present' if dump.exists() else 'missing'}")
            dump_tasks = next(summary for name, summary in task_summaries if name == "dump_task")
            print_task_summary("dump_task", dump_tasks)
        if meta.get("note"):
            print(f"  note={meta['note']}")
        if rows:
            print_alignment_summary(rows, task_summaries, meta)
        for idx, row in enumerate(rows):
            info = summarize_row(row)
            print(f"  prefix={idx + 1:02d} family={info['family']:<7} input_h={info['input_h']} out_h={info['out_h']} out_c={info['out_c']} pc_amount={info['amount']}")
        print("  manual_replay:")
        print("    cd ~/npu/ops_rknn")
        print("    g++ -o conv2d_multi conv2d_multi.cpp -I../include -lrknnrt -std=c++11")
        print(f"    # edit ioctl.gdb KEEP_TASKS=N and uncomment _patch_submit(regs[\"arg\"]), then run one N at a time")
        print(f"    gdb -x ioctl.gdb --args ./conv2d_multi --case {meta['case']}")


if __name__ == "__main__":
    main()
