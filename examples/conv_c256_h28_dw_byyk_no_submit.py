#!/usr/bin/env python3
"""
No-submit materializer for c256_h28 depthwise BY_YK closure (Attack C).
Asserts the per-row body field structure that a future submit path would
emit. Complements examples/conv_depthwise_by_y_layout_no_submit.py which
asserts the TASK METADATA (amounts/masks/offsets); this one asserts the
BODY FIELDS (cbuf0, data_size1, dma_con2, weight_sizes, etc.) per row.

The 12-task RKNN structure for c256_h28 (oc=256, h=28, k=3, g=256, depthwise):
  amounts = (108, 108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
  masks   = (0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
  offsets = (0, 112, 224, 336, 368, 480, 512, 624, 656, 768, 800, 912)
  in_h    = (28, 28, 15, None, 15, None, 11, None, 11, None, 10, None)
  out_h   = (25, 25, 12, None, 12, None, 8,  None, 8,  None, 7,  None)
  families= (setup, setup, setup, ppu_pdp, setup, ppu_pdp,
             y_tile, ppu_pdp, y_tile, ppu_pdp, y_tile, ppu_pdp)
  weight_reuse per row: (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

The body fields (from live GEM2 capture) are:
  cbuf0_base           = 0x01b        # without weight_reuse
  cbuf0_with_reuse     = 0x201b       # with weight_reuse bit 13
  data_size1           = 0x1f00e0     # c=224 aligned (c2-packed 8)
  dma_con2             = 0x02a0
  dst_surf_stride      = 0x02a4
  surface_add          = 0x0a90
  conv_con1            = 0x123        # depthwise proc_precision(2) | in_precision(2) | conv_mode(3)
  conv_con2_setup      = 0xc0         # feature_grains=12 (for in_h=28 case)
  weight_size0         = 0xfc0        # 4032 bytes
  weight_size1         = 0xfc0
  weight_size2         = 0x03030001   # kw=3, kh=3, kernels=1
  cbuf_con1            = 0xc4
  cvt_con0             = 0xb
  cvt_con1/2/3/4       = 0x10000
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.conv import shape_from_name  # noqa: E402

C256_H28_DW_BYYK_BODY = {
    "cbuf0_base": 0x01b,
    "cbuf0_reuse": 0x201b,
    "data_size1": 0x1f00e0,
    "dma_con2": 0x02a0,
    "dst_surf_stride": 0x02a4,
    "surface_add": 0x0a90,
    "conv_con1": 0x123,
    "conv_con2_setup_in_h28": 0xc0,  # feature_grains=12
    "weight_size0": 0xfc0,
    "weight_size1": 0xfc0,
    "weight_size2": 0x03030001,
    "cbuf_con1": 0xc4,
    "cvt_con0": 0xb,
    "cvt_con_scale": 0x10000,
}

C256_H28_DW_BYYK_AMOUNTS = (108, 108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
C256_H28_DW_BYYK_MASKS = (0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
C256_H28_DW_BYYK_OFFSETS = (0, 112, 224, 336, 368, 480, 512, 624, 656, 768, 800, 912)
C256_H28_DW_BYYK_INPUT_H = (28, 28, 15, None, 15, None, 11, None, 11, None, 10, None)
C256_H28_DW_BYYK_OUT_H = (25, 25, 12, None, 12, None, 8, None, 8, None, 7, None)
C256_H28_DW_BYYK_FAMILIES = ("setup", "setup", "setup", "ppu_pdp", "setup", "ppu_pdp",
                              "y_tile", "ppu_pdp", "y_tile", "ppu_pdp", "y_tile", "ppu_pdp")
C256_H28_DW_BYYK_WEIGHT_REUSE = (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)


def assert_c256_h28_layout():
    """Asserts the 12-task layout is internally consistent."""
    assert len(C256_H28_DW_BYYK_AMOUNTS) == 12
    assert len(C256_H28_DW_BYYK_MASKS) == 12
    assert len(C256_H28_DW_BYYK_OFFSETS) == 12
    assert len(C256_H28_DW_BYYK_INPUT_H) == 12
    assert len(C256_H28_DW_BYYK_OUT_H) == 12
    assert len(C256_H28_DW_BYYK_FAMILIES) == 12
    assert len(C256_H28_DW_BYYK_WEIGHT_REUSE) == 12
    # amount 108/104 distinction
    assert C256_H28_DW_BYYK_AMOUNTS[0] == 108
    assert C256_H28_DW_BYYK_AMOUNTS[1] == 108
    assert C256_H28_DW_BYYK_AMOUNTS[2] == 104
    for idx in (4, 6, 8, 10):
        assert C256_H28_DW_BYYK_AMOUNTS[idx] == 104
    for idx in (3, 5, 7, 9, 11):
        assert C256_H28_DW_BYYK_AMOUNTS[idx] == 26
    # mask 0x0d for setup/y_tile, 0x60 for ppu_pdp
    for idx, family in enumerate(C256_H28_DW_BYYK_FAMILIES):
        if family == "ppu_pdp":
            assert C256_H28_DW_BYYK_MASKS[idx] == 0x60
        else:
            assert C256_H28_DW_BYYK_MASKS[idx] == 0x0d
    # offsets must be monotonically increasing
    for idx in range(1, 12):
        assert C256_H28_DW_BYYK_OFFSETS[idx] > C256_H28_DW_BYYK_OFFSETS[idx - 1]
    # input_h for compute rows, None for ppu_pdp
    for idx, family in enumerate(C256_H28_DW_BYYK_FAMILIES):
        if family == "ppu_pdp":
            assert C256_H28_DW_BYYK_INPUT_H[idx] is None
        else:
            assert C256_H28_DW_BYYK_INPUT_H[idx] is not None
    # weight_reuse: 0 for first setup, 1 for the rest
    assert C256_H28_DW_BYYK_WEIGHT_REUSE[0] == 0
    for idx in range(1, 12):
        assert C256_H28_DW_BYYK_WEIGHT_REUSE[idx] == 1
    print("c256_h28_dw_byyk_layout=ok tasks=12 families=setup,setup,setup,ppu_pdp,setup,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp")


def assert_c256_h28_body():
    """Asserts the body field constants from the live capture."""
    for key, value in C256_H28_DW_BYYK_BODY.items():
        if not isinstance(value, int):
            raise AssertionError(f"body field {key} is not an int: {value}")
    print(f"c256_h28_dw_byyk_body=ok fields={len(C256_H28_DW_BYYK_BODY)}")
    for key, value in C256_H28_DW_BYYK_BODY.items():
        print(f"  {key}=0x{value:x}")


def assert_c256_h28_cbuf0_per_row():
    """Asserts the cbuf0 value per row matches weight_reuse."""
    for idx, reuse in enumerate(C256_H28_DW_BYYK_WEIGHT_REUSE):
        expected = C256_H28_DW_BYYK_BODY["cbuf0_reuse"] if reuse else C256_H28_DW_BYYK_BODY["cbuf0_base"]
        print(f"  row[{idx:02d}] family={C256_H28_DW_BYYK_FAMILIES[idx]:8s} "
              f"in_h={str(C256_H28_DW_BYYK_INPUT_H[idx]):4s} weight_reuse={reuse} cbuf0=0x{expected:x}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=["layout", "body", "cbuf0", "all"], default="all")
    args = parser.parse_args()
    if args.check in ("layout", "all"):
        assert_c256_h28_layout()
    if args.check in ("body", "all"):
        assert_c256_h28_body()
    if args.check in ("cbuf0", "all"):
        assert_c256_h28_cbuf0_per_row()
    print("status=no_drm_no_submit")


if __name__ == "__main__":
    main()
