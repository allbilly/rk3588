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
    "c192_h28_oc96_pw_by_k": {
        "shape": "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid",
        "case": "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid 1 192 28 28 96 1 1 1 c192_h28_oc96_pw",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c192_h28_oc96_keep1_gem1/dump_gem1.txt",
        "note": "promoted via RKNN prefix replay and local exact11 OC-k-tile replay. RKNN KEEP_TASKS=1 PREFIX_MODE=linear passed max error 0.0311127 and post simple_add PASS. GEM1 task dump is exact11 amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800. GEM2 visible body fields showed CBUF0=0x2a, DATA_SIZE1=0x003f00c0, DMA_CON2=0x2a0, SURFACE_ADD=0x6200, and CONV2 sequence setup=0x90, two rows 0x40000090, then three rows 0x50000090. Existing h28 y-window reuse failed final tile, but OC k_tile splits 0:32/32:32/64:32 passed locally with max_diff=0.0306 and pre/post simple_add PASS.",
    },
    "c512_h14_oc32_pw_by_k": {
        "shape": "b1_c512_h14_w14_oc32_wic512_k1x1_g1",
        "case": "b1_c512_h14_w14_oc32_wic512_k1x1_g1 1 512 14 14 32 1 1 1 c512_h14_oc32_pw_by_k",
        "emit": None,
        "capture": "/tmp/c512_h14_oc32_capture.log",
        "note": "promoted via targeted RKNN live regcmd capture of matched model: 11 task descriptors amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 submit_task_number=3; body fields CBUF0=0x57 DATA_SIZE1=0x003f0200 DMA_CON2=0x8c CVT_CON0=0xb; y windows 0+5 conv2=0x60, 5+5 conv2=0x60, 10+4 conv2=0x50; NPU validation max_diff=0.0312 with pre/post simple_add PASS",
    },
    "c512_h14_oc112_pw_by_k": {
        "shape": "b1_c512_h14_w14_oc112_wic512_k1x1_g1",
        "case": "b1_c512_h14_w14_oc112_wic512_k1x1_g1 1 512 14 14 112 1 1 1 c512_h14_oc112_pw_by_k",
        "emit": None,
        "note": "promoted via the c512_h14 pointwise exact11 BY_K body proven for c512_h14_oc32: 11 task descriptors, CBUF0=0x57 DATA_SIZE1=0x003f0200 DMA_CON2=0x8c CVT_CON0=0xb, y windows 0+5/5+5/10+4 with conv2 0x60/0x60/0x50. One-off guarded submit PASS max_diff=0.0312, post simple_add PASS.",
    },
    "c512_h14_oc512_pw_by_k": {
        "shape": "b1_c512_h14_w14_oc512_wic512_k1x1_g1",
        "case": "b1_c512_h14_w14_oc512_wic512_k1x1_g1 1 512 14 14 512 1 1 1 c512_h14_oc512_pw_by_k",
        "emit": None,
        "note": "promoted via the c512_h14 pointwise exact11 BY_K body: 11 task descriptors, CBUF0=0x57 DATA_SIZE1=0x003f0200 DMA_CON2=0x8c CVT_CON0=0xb, y windows 0+5/5+5/10+4 with conv2 0x60/0x60/0x50. One-off guarded submit PASS max_diff=0.0312, post simple_add PASS.",
    },
    "cc_c512_h14_oc512_pw_by_k": {
        "shape": "conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1",
        "case": "conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1 1 512 14 14 512 1 1 1 cc_c512_h14_oc512_pw_by_k",
        "emit": None,
        "note": "promoted via the same c512_h14 pointwise exact11 BY_K body as the unprefixed alias: 11 task descriptors, CBUF0=0x57 DATA_SIZE1=0x003f0200 DMA_CON2=0x8c CVT_CON0=0xb, y windows 0+5/5+5/10+4 with conv2 0x60/0x60/0x50. One-off guarded submit PASS max_diff=0.0312, post simple_add PASS.",
    },
    "c384_h14_oc96_pw_by_k": {
        "shape": "b1_c384_h14_w14_oc96_wic384_k1x1_g1",
        "case": "b1_c384_h14_w14_oc96_wic384_k1x1_g1 1 384 14 14 96 1 1 1 c384_h14_oc96_pw_by_k",
        "emit": None,
        "note": "promoted via h14 pointwise exact11 BY_K body: 11 task descriptors, CBUF0=0x57 DATA_SIZE1=0x00bf0180 DMA_CON2=0x8c CVT_CON0=0xb, y windows 0+5/5+5/10+4 with conv2 0x60/0x60/0x50. One-off guarded submit PASS max_diff=0.0312, post simple_add PASS.",
    },
    "c480_h14_oc96_pw_by_k": {
        "shape": "b1_c480_h14_w14_oc96_wic480_k1x1_g1",
        "case": "b1_c480_h14_w14_oc96_wic480_k1x1_g1 1 480 14 14 96 1 1 1 c480_h14_oc96_pw_by_k",
        "emit": None,
        "note": "promoted via h14 pointwise exact11 BY_K body: 11 task descriptors, CBUF0=0x57 DATA_SIZE1=0x00ef01e0 DMA_CON2=0x8c CVT_CON0=0xb, y windows 0+5/5+5/10+4 with conv2 0x60/0x60/0x50. One-off guarded submit PASS max_diff=0.0312, post simple_add PASS.",
    },
    "c528_h14_pointwise_family_by_k": {
        "shape": "b1_c528_h14_w14_oc32_wic528_k1x1_g1",
        "case": "b1_c528_h14_w14_oc32_wic528_k1x1_g1 1 528 14 14 32 1 1 1 c528_h14_pointwise_family_by_k",
        "emit": None,
        "note": "promoted c528 h14 pointwise exact11 BY_K family with corrected natural DATA_SIZE1=0x01070210, CBUF0=0x57, DMA_CON2=0x8c, CVT_CON0=0xb, y windows 0+5/5+5/10+4 with conv2 0x60/0x60/0x50. Hardware PASS for oc32/128/160/256 and _s1_pvalid aliases; earlier oc32 negative used wrong c512 DATA_SIZE1=0x003f0200.",
    },
    "c528_h14_oc128_pw_by_k": {
        "shape": "b1_c528_h14_w14_oc128_wic528_k1x1_g1",
        "case": "b1_c528_h14_w14_oc128_wic528_k1x1_g1 1 528 14 14 128 1 1 1 c528_h14_oc128_pw_by_k",
        "emit": None,
        "note": "promoted via c528 h14 pointwise family body, DATA_SIZE1=0x01070210; guarded submit PASS max_diff=0.0311.",
    },
    "c528_h14_oc160_pw_by_k": {
        "shape": "b1_c528_h14_w14_oc160_wic528_k1x1_g1",
        "case": "b1_c528_h14_w14_oc160_wic528_k1x1_g1 1 528 14 14 160 1 1 1 c528_h14_oc160_pw_by_k",
        "emit": None,
        "note": "promoted via c528 h14 pointwise family body, DATA_SIZE1=0x01070210; guarded submit PASS max_diff=0.0311.",
    },
    "c528_h14_oc256_pw_by_k": {
        "shape": "b1_c528_h14_w14_oc256_wic528_k1x1_g1",
        "case": "b1_c528_h14_w14_oc256_wic528_k1x1_g1 1 528 14 14 256 1 1 1 c528_h14_oc256_pw_by_k",
        "emit": None,
        "note": "promoted via c528 h14 pointwise family body, DATA_SIZE1=0x01070210; guarded submit PASS max_diff=0.0312.",
    },
    "c528_h14_oc32_pvalid_pw_by_k": {
        "shape": "b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid",
        "case": "b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid 1 528 14 14 32 1 1 1 c528_h14_oc32_pvalid_pw_by_k",
        "emit": None,
        "note": "promoted via c528 h14 pointwise family body, DATA_SIZE1=0x01070210; guarded submit PASS max_diff=0.0311.",
    },
    "c528_h14_oc128_pvalid_pw_by_k": {
        "shape": "b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid",
        "case": "b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid 1 528 14 14 128 1 1 1 c528_h14_oc128_pvalid_pw_by_k",
        "emit": None,
        "note": "promoted via c528 h14 pointwise family body, DATA_SIZE1=0x01070210; guarded submit PASS max_diff=0.0311.",
    },
    "c528_h14_oc160_pvalid_pw_by_k": {
        "shape": "b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid",
        "case": "b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid 1 528 14 14 160 1 1 1 c528_h14_oc160_pvalid_pw_by_k",
        "emit": None,
        "note": "promoted via c528 h14 pointwise family body, DATA_SIZE1=0x01070210; guarded submit PASS max_diff=0.0311.",
    },
    "c528_h14_oc256_pvalid_pw_by_k": {
        "shape": "b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid",
        "case": "b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid 1 528 14 14 256 1 1 1 c528_h14_oc256_pvalid_pw_by_k",
        "emit": None,
        "note": "promoted via c528 h14 pointwise family body, DATA_SIZE1=0x01070210; guarded submit PASS max_diff=0.0312.",
    },
    "c576_h14_oc96_pw_by_k": {
        "shape": "b1_c576_h14_w14_oc96_wic576_k1x1_g1",
        "case": "b1_c576_h14_w14_oc96_wic576_k1x1_g1 1 576 14 14 96 1 1 1 c576_h14_oc96_pw_by_k",
        "emit": None,
        "note": "promoted via h14 pointwise exact11 BY_K body: 11 task descriptors, CBUF0=0x57 DATA_SIZE1=0x011f0240 DMA_CON2=0x8c CVT_CON0=0xb, y windows 0+5/5+5/10+4 with conv2 0x60/0x60/0x50. One-off guarded submit PASS max_diff=0.0312, post simple_add PASS.",
    },
    "c512_h14_oc24_pw_by_k_negative_probe": {
        "shape": "b1_c512_h14_w14_oc24_wic512_k1x1_g1",
        "case": "match_b1_c512_h14_w14_oc24_wic512_k1x1_g1 1 512 14 14 24 1 1 1 c512_h14_oc24_pw_by_k_negative_probe",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c512_h14_oc24_keep1_gem1/dump_gem1.txt",
        "note": "kept fenced: fresh RKNN prefix replay on matched model passed with max error 0.0539551 and NPU health PASS. GEM1 task dump shows exact11 descriptors amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800, original submit three one-task subcores. GEM2 body rows differ from wider c512 oc32: setup CONV2=0x90, then two full-OC input_h=7 body rows CONV2=0x10000080, then y rows 0x20000060/0x20000060/0x20000050, WEIGHT_SIZE0=0x6000 throughout. Replaying those visible row fields in conv.py still failed with max_diff=135.3407 and post simple_add PASS, so visible body/task fields alone are not the missing closure.",
    },
    "c832_h7_oc48_pw_by_k_negative_probe": {
        "shape": "b1_c832_h7_w7_oc48_wic832_k1x1_g1",
        "case": "match_b1_c832_h7_w7_oc48_wic832_k1x1_g1 1 832 7 7 48 1 1 1 c832_h7_oc48_pw_by_k",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c832_h7_oc48_keep1_gem1/dump_gem1.txt",
        "note": "kept fenced: safe RKNN KEEP_TASKS=1 PREFIX_MODE=linear passed max error 0.0584869 and post simple_add PASS. GEM1 is exact11 amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800. GEM2 shows a narrow h7 body setup,setup,ppu_pdp,setup,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp with full-OC setup rows then three 16-channel k_tile rows, constants CBUF0=0x93 DATA_SIZE1=0x003f0340 DMA_CON2=0x15 DST_SURF_STRIDE=0x34 SURFACE_ADD=0x68. No-submit materializer matches critical fields, but local 11-row replay and local task0 setup108 replay both produced inf output while health stayed PASS. Keep fenced pending buffer/layout parity.",
    },
    "c480_h14_oc16_pw_by_k_negative_probe": {
        "shape": "b1_c480_h14_w14_oc16_wic480_k1x1_g1",
        "case": "b1_c480_h14_w14_oc16_wic480_k1x1_g1 1 480 14 14 16 1 1 1 c480_h14_oc16_pw_by_k_negative_probe",
        "emit": None,
        "note": "kept fenced: same guessed h14 pointwise exact11 body as c480_h14_oc96 timed out on one-off guarded submit. Post simple_add PASS, but this shape needs a narrow-OC RKNN closure before promotion.",
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
        "note": "kept fenced: local replay timed out; RKNN task dump shows 12 descriptors with amounts 108,108,104,26,104,26,104,26,104,26,104,26 and masks 0x0d/0x60. examples/conv_c576_h19_oc12_no_submit.py --check all asserts task_structure=ok and generates 12 no-submit body rows (k_tile_splits=4+3+2+1+2=12 for oc=12) with cbuf0=0xb1, data_size1=0x8f0240, conv2_low=0x30 inherited from the c256_h2_oc64 + c256_h28_pw_1x1 closure pattern. The k_half row needs an explicit 4-qword prelude (108q vs 104q) which is inferred from amount=108. Body field register values (data_size1, dma2, cbuf0) are educated guesses from c256_h2_oc64, NOT captured from a fresh RKNN GEM2 dump, so promotion requires a fresh KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2 capture. Post simple_add PASS.",
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
    "c40_h40_oc320_pw_by_yk": {
        "shape": "b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid",
        "case": "match_b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid 1 40 40 40 320 1 1 1 c40_h40_oc320_pw_by_yk",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c40_h40_oc320_keep1_gem1/dump_gem1.txt",
        "note": "promoted via RKNN prefix replay and local one-task setup108 compact-weight replay. KEEP_TASKS=1 PREFIX_MODE=linear passed with max error 0.0178719 and post simple_add PASS. Original submit is three one-task subcores. GEM1 task dump is exact11 amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800. Fresh GEM2 dump /home/orangepi/npu/ops_rknn/dump/prefix_c40_h40_oc320_keep1_gem2/dump_gem2.txt shows setup row fields CBUF0=0x84, CONV2=0x160, DATA_SIZE1=0x00270028, CBUF1=0x32, WEIGHT_SIZE0=0x6400, WEIGHT_SIZE1=0x50, DMA_CON2=0x5a0, FC_DATA_SIZE1=0x28. Local 11-task replay failed numerically, matching the safe prefix result that task 0 alone computes the full output. Local examples/conv.py setup108 compact-weight replay PASS max_diff=0.0155 with pre/post simple_add PASS.",
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
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c256_h3_none_keep1_gem1/dump_gem1.txt",
        "note": "promoted via RKNN prefix replay and local exact11 compact-weight PC-chain replay. KEEP_TASKS=1 PREFIX_MODE=linear passed in RKNN with max error 0.0190849 and post simple_add PASS. GEM1 task dump is exact11 amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800. GEM2 at /home/orangepi/npu/ops_rknn/dump/prefix_c256_h3_none_keep1_gem2/dump_gem2.txt shows the active 11-row PC-chain: setup h3 conv2=0x40, h2 conv2=0x10000030, h1 conv2=0x10000020, then y_start 0/1/2 rows conv2=0x20000020, compact WEIGHT_SIZE0=0x3000, CBUF0=0xb1, DATA_SIZE1=0x003f0100, DMA_CON2=0x0ffffffd. Local examples/conv.py PASS max_diff=0.0149 and post simple_add PASS.",
    },
    "c256_h2_oc24_pw_none": {
        "shape": "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid",
        "case": "b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid 1 256 2 2 24 1 1 1 c256_h2_oc24_pw_none",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c256_h2_none_keep1_gem2/dump_gem2.txt",
        "note": "promoted via RKNN setup108 body plus compact pointwise weight packing. GEM2 active row at offset 0x3840 exactly matches _exact11_body_regs setup after overrides CBUF0=0xb1, CONV2_LOW=0x30, DATA_SIZE1=0x003f0100, CVT_CON0=0xb, DMA_CON2=0x0ffffffc. Local padded pointwise weight packing wrote 0x4000 bytes and made channels 16-23 wrong; compact 24-kernel packing writes RKNN's WEIGHT_SIZE0=0x3000 bytes and passes. Direct examples/conv.py PASS max_diff=0.0245 with pre/post simple_add PASS; one-shape sweep /tmp/opencode/conv_py_217_sweep_20260602_204347_summary.txt PASS pre_health_rc=0 post_health_rc=0.",
    },
    "c256_h2_oc64_crash_fence": {
        "shape": "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid",
        "case": "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid 1 256 2 2 64 1 1 1 c256_h2_oc64_crash_fence",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c256_h2_oc64_keep1_gem1/dump_gem1.txt",
        "note": "promoted via safe RKNN prefix replay and shape-scoped exact11 materializer. KEEP_TASKS=1 PREFIX_MODE=linear passed with max error 0.0294266 and post simple_add PASS. Existing GEM1 exact11 task metadata amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800. Fresh GEM2 dump /home/orangepi/npu/ops_rknn/dump/prefix_c256_h2_oc64_keep1_replay2_*/dump_gem2.txt decodes as setup,k_half,ppu_pdp,k_half,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp with row amounts 108,104,26,104,26,104,26,104,26,104,26 and out_c 64,32,none,32,none,32,none,16,none,16,none. Key fields: CBUF0=0xb1 DATA_SIZE1=0x003f0100 DMA_CON2=0x0ffffffc, setup WEIGHT_SIZE0=0x8000, k_half/k_tile 0x4000/0x2000, k splits 0:32/32:16/48:16, input_h=2, conv2_low=0x30. Weight layout uses _pack_pointwise_wide (0x8000 bytes for 64 OC). Local examples/conv.py exact11_byk submit PASS max_diff=0.0245 with pre/post simple_add PASS. Unblocked from CRASH_FENCED_SHAPES; c256_h2_oc546 stays fenced.",
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
    "dw96_h20_by_yk": {
        "shape": "b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid",
        "case": "match_b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid 1 96 20 20 96 3 3 96 dw_c96_h20_by_yk",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_dw_c96_h20_keep1_gem1/dump_gem1.txt",
        "note": "fresh RKNN prefix evidence for a depthwise BY_YK representative: KEEP_TASKS=1 PREFIX_MODE=linear passed with max error 0.00697899 and post simple_add PASS. Original submit is three one-task subcores. GEM1 task dump is exact11 amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800.",
    },
    "dw96_h112_by_yk_task_dump": {
        "shape": "b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid",
        "case": "match_b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid 1 96 112 112 96 3 3 96 c96_h112_dw_by_yk",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c96_h112_dw_by_yk_keep1_gem1/dump_gem1.txt",
        "note": "task-only prefix evidence for a remaining depthwise BY_YK fence. Existing matched model /home/orangepi/npu/ops_rknn/models/match_b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid.rknn was run with safe KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1. The one-task prefix exited cleanly and failed numerically with partial output, as expected; post simple_add PASS. examples/conv_depthwise_by_y_layout_no_submit.py c96_h112_by_yk_task asserts GEM1 task metadata: 31 tasks, amounts=108,108,12,12,12,12,16,16,12,104,104,104,104,26,104,104,104,104,26,104,104,104,26,104,104,104,26,104,104,104,26 masks=0x0d/0x60 offsets=0,112,224,240,256,272,288,312,336,352,464,576,688,800,832,944,1056,1168,1280,1312,1424,1536,1648,1680,1792,1904,2016,2048,2160,2272,2384. Local planner emits 90 equal BY_YK rows, so it is not RKNN-equivalent. This is not replay-ready because GEM2/body rows are not captured yet.",
    },
    "dw144_h56_by_yk_task_dump": {
        "shape": "b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid",
        "case": "match_b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid 1 144 56 56 144 3 3 144 c144_h56_dw_by_yk",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c144_h56_dw_by_yk_keep1_gem1/dump_gem1.txt",
        "note": "task-only prefix evidence for a remaining depthwise BY_YK fence. Built matched model /home/orangepi/npu/ops_rknn/models/match_b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid.rknn, then ran safe KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1. The one-task prefix exited cleanly and failed numerically with partial output, as expected. examples/conv_depthwise_by_y_layout_no_submit.py c144_h56_by_yk_task asserts GEM1 task metadata: 17 tasks, amounts=108,108,12,12,18,104,104,26,104,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,240,256,280,392,504,536,648,760,792,904,936,1048,1080,1192. Local planner emits 75 equal BY_YK rows, so it is not RKNN-equivalent. This is not replay-ready because GEM2/body rows are not captured yet.",
    },
    "cc_c512_h14_dw_by_k": {
        "shape": "conv2d_cc_b1_c512_h14_w14_oc512_wic1_k3x3_g512",
        "case": "match_b1_c512_h14_w14_oc512_wic1_k3x3_g512 1 512 14 14 512 3 3 512 c512_h14_dw_by_k",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c512_h14_dw_by_k_keep1_gem1/dump_gem1.txt",
        "note": "fresh difficult-shape depthwise BY_K prefix evidence after crash follow-up: built matched model /home/orangepi/npu/ops_rknn/models/match_b1_c512_h14_w14_oc512_wic1_k3x3_g512.rknn, then ran KEEP_TASKS=1 PREFIX_MODE=linear twice with DUMP_GEM=1 and DUMP_GEM=2. Both passed max error 0.0076313 and post simple_add PASS. Patched submit was task_number=1, subcore_task[0]=(0,1), other subcores zero. GEM1 exact11 task metadata amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800. GEM2 rows at /home/orangepi/npu/ops_rknn/dump/prefix_c512_h14_dw_by_k_keep1_gem2/dump_gem2.txt decode as setup,setup,ppu_pdp,setup,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp with row lengths 115,106,29,106,29,106,29,106,29,106,29; active rows use full out_c=512 with input_h/out_h 14/11, 8/5, 8/5, 6/3, 6/3, 6/3. This proves the local BY_K/k_tile fence is not RKNN-equivalent for this depthwise shape.",
    },
    "cc_c512_h7_oc1024_pw_by_k": {
        "shape": "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1",
        "case": "match_b1_c512_h7_w7_oc1024_wic512_k1x1_g1 1 512 7 7 1024 1 1 1 c512_h7_oc1024_pw_by_k",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c512_h7_oc1024_pw_by_k_keep1_gem1/dump_gem1.txt",
        "note": "promoted via safe RKNN prefix evidence plus local exact11 BY_K replay. Built matched model /home/orangepi/npu/ops_rknn/models/match_b1_c512_h7_w7_oc1024_wic512_k1x1_g1.rknn and ran KEEP_TASKS=1 PREFIX_MODE=linear with DUMP_GEM=1 and DUMP_GEM=2; both passed max error 0.0458069 and post simple_add PASS. GEM1 exact11 task metadata amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800. GEM2 rows decode as setup,k_half,ppu_pdp,k_half,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp with active out_c 1024,512,512,352,336,336 and constants CBUF0=0xa2 DATA_SIZE1=0x003f0200 DMA_CON2=0x15 WEIGHT_SIZE0=0x100000/0x80000/0x58000/0x54000/0x54000. Local examples/conv.py direct run PASS max_diff=0.0312 with pre/post simple_add PASS after adding the captured 352/336/336 OC splits and constants.",
    },
    "c512_h7_oc1024_pw_by_k": {
        "shape": "b1_c512_h7_w7_oc1024_wic512_k1x1_g1",
        "case": "match_b1_c512_h7_w7_oc1024_wic512_k1x1_g1 1 512 7 7 1024 1 1 1 c512_h7_oc1024_pw_by_k",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c512_h7_oc1024_pw_by_k_keep1_gem1/dump_gem1.txt",
        "note": "promoted as the unprefixed alias of cc_c512_h7_oc1024_pw_by_k using the same exact11 BY_K body: amounts=108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,256,368,400,512,544,656,688,800; GEM2 body is setup,k_half,ppu_pdp,k_half,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp with out_c 1024,512,512,352,336,336. Local examples/conv.py direct run PASS max_diff=0.0312 with post simple_add PASS.",
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
        "note": "dump-only evidence for a remaining BY_YK fence; local setup-row replay was numerically wrong (max_diff=122.6959). examples/conv_pointwise_by_yk_layout_no_submit.py cc_c256_h28 now asserts the RKNN 14-row closure: setup/setup, two 128-channel k_half pairs, PPU/PDP separators, and three full-OC y_tile rows. Constants are DATA_SIZE1=0x003f0100, DMA_CON2=0x2a0, DST_SURF_STRIDE=0x310, SURFACE_ADD=0x620, CBUF0=0x48/0x2048. Dynamic deltas are feature=0,0x1f80,0,0x1f80,...,0,0x1180,0x2140; weight second k_half block starts at 0x10000; output second k_half block starts at 0x31000/0x32f80. Local planner emits 48 32-OC setup/k_half/k_tile rows, so mixed Y/K task layout remains unresolved and fenced.",
    },
    "c256_h28_pw_by_yk_dump": {
        "shape": "b1_c256_h28_w28_oc256_wic256_k1x1_g1",
        "case": "b1_c256_h28_w28_oc256_wic256_k1x1_g1 1 256 28 28 256 1 1 1 c256_h28_pw_by_yk_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1_dump.txt",
        "note": "unprefixed alias of cc_c256_h28_pw_by_yk_dump. examples/conv_pointwise_by_yk_layout_no_submit.py asserts that the local planner descriptor sequence is identical to the prefixed RKNN-dumped shape, so the same 14-row RKNN BY_YK closure evidence applies; still fenced until runtime can emit that mixed full-OC/128-channel-k-half/PPU-PDP closure.",
    },
    "cc_c64_h112_dw_by_yk_dump": {
        "shape": "conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64",
        "case": "conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64 1 64 112 112 64 3 3 64 cc_c64_h112_dw_by_yk_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64_dump.txt",
        "note": "dump-only evidence for a remaining depthwise BY_YK fence; RKNN rows include irregular setup prologue rows, repeated 64-channel setup/y_tile pairs, and PPU/PDP separators. examples/conv_depthwise_by_y_layout_no_submit.py c64_by_yk asserts the 23-task closure: amounts=108,108,12,12,16,16,104,104,104,26,104,104,104,26,104,104,26,104,104,26,104,104,26 offsets=0,112,224,240,256,280,304,416,528,640,672,784,896,1008,1040,1152,1264,1296,1408,1520,1552,1664,1776 constants CBUF0=0x1b/0x201b DATA_SIZE1=0x003f0040 DMA_CON2=0x2f40 DST_SURF_STRIDE=0x2f44 SURFACE_ADD=0xbd10. Local planner emits 60 equal BY_YK rows and is not RKNN-equivalent, so grouped/depthwise replay is not promoted from this dump.",
    },
    "c64_h112_dw_by_yk_dump": {
        "shape": "b1_c64_h112_w112_oc64_wic1_k3x3_g64",
        "case": "b1_c64_h112_w112_oc64_wic1_k3x3_g64 1 64 112 112 64 3 3 64 c64_h112_dw_by_yk_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64_dump.txt",
        "note": "unprefixed alias of cc_c64_h112_dw_by_yk_dump. examples/conv_depthwise_by_y_layout_no_submit.py c64_by_yk asserts that the local planner descriptor sequence is identical to the prefixed RKNN-dumped shape, so the same 23-task RKNN BY_YK closure evidence applies; still fenced until grouped/depthwise runtime can emit that task-safe closure.",
    },
    "cc_c128_h56_dw_by_yk_task_dump": {
        "shape": "conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128",
        "case": "match_b1_c128_h56_w56_oc128_wic1_k3x3_g128 1 128 56 56 128 3 3 128 c128_h56_dw_by_yk",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c128_h56_dw_by_yk_keep1_gem1/dump_gem1.txt",
        "note": "task-only prefix evidence for a remaining c128 h56 depthwise BY_YK fence. Built matched model /home/orangepi/npu/ops_rknn/models/match_b1_c128_h56_w56_oc128_wic1_k3x3_g128.rknn with gen_conv2d_models.py, then ran safe KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1 prefix replay. The prefix output was intentionally partial and failed numerically, post simple_add PASS. examples/conv_depthwise_by_y_layout_no_submit.py c128_h56_by_yk_task asserts GEM1 task metadata amounts=108,108,12,12,104,104,26,104,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,240,256,368,480,512,624,736,768,880,912,1024,1056,1168. Local planner emits 60 equal BY_YK rows, so it is not RKNN-equivalent. This is not replay-ready because GEM2/body rows and constants are not captured yet.",
    },
    "c128_h56_dw_by_yk_task_dump": {
        "shape": "b1_c128_h56_w56_oc128_wic1_k3x3_g128",
        "case": "b1_c128_h56_w56_oc128_wic1_k3x3_g128 1 128 56 56 128 3 3 128 c128_h56_dw_by_yk_task",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c128_h56_dw_by_yk_keep1_gem1/dump_gem1.txt",
        "note": "unprefixed alias of cc_c128_h56_dw_by_yk_task_dump. examples/conv_depthwise_by_y_layout_no_submit.py c128_h56_by_yk_task asserts that the local planner descriptor sequence is identical to the prefixed shape, so the same 16-task RKNN BY_YK task-layout evidence applies; still fenced until GEM2/body rows are captured and runtime can emit that task-safe closure.",
    },
    "cc_c256_h28_dw_by_yk_task_dump": {
        "shape": "conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256",
        "case": "match_b1_c256_h28_w28_oc256_wic1_k3x3_g256 1 256 28 28 256 3 3 256 c256_h28_dw_by_yk",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c256_h28_dw_by_yk_keep1_gem1/dump_gem1.txt",
        "note": "task-only + body-field prefix evidence for a remaining c256 h28 depthwise BY_YK fence. examples/conv_depthwise_by_y_layout_no_submit.py c256_h28_by_yk_task asserts GEM1 task metadata amounts=108,108,104,26,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,336,368,480,512,624,656,768,800,912 (still task-only). examples/conv_c256_h28_dw_byyk_no_submit.py asserts the body field structure from the live GEM2 capture: cbuf0=0x1b/0x201b (per weight_reuse), data_size1=0x1f00e0, dma_con2=0x2a0, dst_surf_stride=0x2a4, surface_add=0xa90, conv_con1=0x123, conv_con2=0xc0 (feature_grains=12), weight_size0=0xfc0, weight_size1=0xfc0, weight_size2=0x03030001, cbuf_con1=0xc4, cvt_con0=0xb. Local planner emits 72 equal BY_YK rows, so a row materializer that translates planner rows into the RKNN 12-row closure (setup,setup,setup,ppu_pdp,setup,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp) is required before any submit.",
    },
    "c256_h28_dw_by_yk_task_dump": {
        "shape": "b1_c256_h28_w28_oc256_wic1_k3x3_g256",
        "case": "b1_c256_h28_w28_oc256_wic1_k3x3_g256 1 256 28 28 256 3 3 256 c256_h28_dw_by_yk_task",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c256_h28_dw_by_yk_keep1_gem1/dump_gem1.txt",
        "note": "unprefixed alias of cc_c256_h28_dw_by_yk_task_dump. examples/conv_depthwise_by_y_layout_no_submit.py c256_h28_by_yk_task asserts that the local planner descriptor sequence is identical to the prefixed shape, so the same 12-task RKNN BY_YK task-layout evidence applies; still fenced until GEM2/body rows are captured and runtime can emit that task-safe closure.",
    },
    "cc_c32_h112_dw_by_y_dump": {
        "shape": "conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32",
        "case": "conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32 1 32 112 112 32 3 3 32 cc_c32_h112_dw_by_y_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32_dump.txt",
        "note": "dump and fresh prefix evidence for a remaining depthwise BY_Y fence. RKNN rows include setup prologue rows, 32-channel setup/y_tile pairs, and PPU/PDP separators. Fresh KEEP_TASKS=1 PREFIX_MODE=linear showed original submit task_number=9 with subcore_task=(0,3),(0,3),(0,3),(0,0),(0,0); partial prefix exited normally and post simple_add PASS. GEM1 task layout is stable: amounts=108,108,17,104,104,26,104,104,26,104,26,104,26,104,26 masks=0x0d/0x60 offsets=0,112,224,248,360,472,504,616,728,760,872,904,1016,1048,1160. Visible body constants: CBUF0=0x1b/0x201b reuse, DATA_SIZE1=0x001f0020, DMA_CON2=0x2f40, DST_SURF_STRIDE=0x2f44, SURFACE_ADD=0xbd10. examples/conv_depthwise_by_y_layout_no_submit.py c32_by_y asserts this dump/task closure and shows the local planner emits a non-equivalent 10-row equal BY_Y layout; grouped/depthwise replay needs this 15-row/9-submit-lane closure.",
    },
    "c32_h112_dw_by_y_dump": {
        "shape": "b1_c32_h112_w112_oc32_wic1_k3x3_g32",
        "case": "b1_c32_h112_w112_oc32_wic1_k3x3_g32 1 32 112 112 32 3 3 32 c32_h112_dw_by_y_dump",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32_dump.txt",
        "note": "unprefixed alias of cc_c32_h112_dw_by_y_dump. examples/conv_depthwise_by_y_layout_no_submit.py c32_by_y asserts that the local planner descriptor sequence is identical to the prefixed RKNN-dumped shape, so the same 15-task RKNN BY_Y closure evidence applies; still fenced until grouped/depthwise runtime can emit that task-safe closure.",
    },
    "c32_h150_dw_by_y_task_dump": {
        "shape": "b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid",
        "case": "b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid 1 32 150 150 32 3 3 32 c32_h150_dw_by_y_task",
        "emit": None,
        "dump": "/home/orangepi/npu/ops_rknn/dump/prefix_c32_h150_dw_by_y_keep1_gem1/dump_gem1.txt",
        "note": "task-only prefix evidence for the remaining c32 h150 depthwise BY_Y fence. examples/conv_depthwise_by_y_layout_no_submit.py c32_h150_by_y_task asserts GEM1 task metadata amounts=108,108,13,12,17,104,104,104,26,104,104,104,26,104,104,26,104,104,26,104,104,26 masks=0x0d/0x60 offsets=0,112,224,248,264,288,400,512,624,656,768,880,992,1024,1136,1248,1280,1392,1504,1536,1648,1760. Local planner emits 13 equal BY_Y rows with input_h 14x12 then 6, so it is not RKNN-equivalent. This is not replay-ready because GEM2/body rows and constants are not captured yet.",
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
        "regcmd_addrs": tuple(task.get("regcmd_addr") for task in tasks),
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
    if summary.get("regcmd_addrs"):
        print(f"  {prefix}_regcmd_addrs=" + ",".join(f"0x{value:x}" for value in summary["regcmd_addrs"]))
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
    if len(amounts) == 3 and amounts == (108, 108, 108) and masks == (0x0d, 0x0d, 0x0d):
        addrs = summary.get("regcmd_addrs")
        if addrs and len(set(addrs)) == 1:
            return "same_regcmd_3subcore_setup108"
        return "triple_setup108"
    if len(amounts) == 15 and amounts == (108, 108, 17, 104, 104, 26, 104, 104, 26, 104, 26, 104, 26, 104, 26):
        if masks == (0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60):
            return "depthwise_by_y_15task_108_108_17_104_104_26"
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
        if len(parts) >= 2 and parts[0] == "FENCED":
            shapes.append(parts[1])
            continue
        if len(parts) >= 2 and parts[0].isdigit() and parts[1] == "FENCED":
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
        if meta.get("emit") is None:
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
        else:
            for name, summary in task_summaries:
                pattern = task_pattern(summary)
                if pattern:
                    print(f"  {name}_pattern={pattern}")
        for idx, row in enumerate(rows):
            info = summarize_row(row)
            print(f"  prefix={idx + 1:02d} family={info['family']:<7} input_h={info['input_h']} out_h={info['out_h']} out_c={info['out_c']} pc_amount={info['amount']}")
        print("  manual_replay:")
        print("    cd ~/npu/ops_rknn")
        print("    g++ -o conv2d_multi conv2d_multi.cpp -I../include -lrknnrt -std=c++11")
        print("    # safe prefix replay: one task, linearized onto subcore0")
        print(f"    KEEP_TASKS=1 PREFIX_MODE=linear gdb -q -x ioctl_prefix_replay.gdb --args ./conv2d_multi --case {meta['case']}")
        print("    # KEEP_TASKS>1 or PREFIX_MODE=subcores is unsafe on difficult shapes; ioctl_prefix_replay.gdb requires ALLOW_UNSAFE_PREFIX_REPLAY=1")


if __name__ == "__main__":
    main()
