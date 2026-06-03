#!/usr/bin/env python3
"""Apply c128_h5_oc256_3x3 promotion (sibling of just-promoted c128_h3_oc256_3x3)."""
from pathlib import Path
PATH = Path("/home/orangepi/rk3588/examples/conv.py")
text = PATH.read_text()

SHAPE = "b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid"

# 1. PREFIX_BY_K_SHAPES (after c128_h3_oc256_k3x3 line)
old1 = '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid",'''
new1 = '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid",
    "b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid",'''
assert old1 in text and text.count(old1) == 1
text = text.replace(old1, new1)

# 2-6: Add OVERRIDES entries (CBUF0, DATA_SIZE1, CVT_CON0, DMA_CON2, KT_TILE_SPLITS, CONV2_LOW)
# CBUF0
text = text.replace(
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x0b1,''',
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x0b1,
    "b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid": 0x0b1,'''
)
# DATA_SIZE1
text = text.replace(
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid":   0x003f0080,''',
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid":   0x003f0080,
    "b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid":   0x003f0080,'''
)
# CVT_CON0
text = text.replace(
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x000b,''',
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x000b,
    "b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid": 0x000b,'''
)
# DMA_CON2
text = text.replace(
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x0ffffffd,''',
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x0ffffffd,
    "b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid": 0x0ffffffd,'''
)
# KT_TILE_SPLITS (same as c128_h3_oc256: ((0,96),(96,96),(192,64)) = 256 total)
text = text.replace(
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": ((0, 96), (96, 96), (192, 64)),''',
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": ((0, 96), (96, 96), (192, 64)),
    "b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid": ((0, 96), (96, 96), (192, 64)),'''
)
# CONV2_LOW (0x080 for in_h=5, vs 0x060 for in_h=3; formula is in_h + kh)
text = text.replace(
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x060,''',
    '''    "b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid": 0x060,
    "b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid": 0x080,'''
)

PATH.write_text(text)
print("Applied 7-line edit to conv.py for c128_h5_oc256_3x3 promotion")
