#!/usr/bin/env python3
"""Offline RKNN direct-spatial schedule formula checks.

This script is intentionally experimental.  It compares formula-derived tile
descriptors from examples/conv_tiles.py against captured RKNN schedules kept in
this checker as validation data.  Captures should not be normal-path schedule
tables in the clean example.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RK3588_CONV_NO_DEVICE", "1")

import examples.conv_tiles as conv  # noqa: E402


@dataclass(frozen=True)
class Shape:
    name: str
    in_c: int
    in_h: int
    in_w: int
    out_c: int
    kh: int
    kw: int
    groups: int = 1
    stride: int = 1


def conv_params(shape: Shape) -> dict:
    return conv._conv_params(
        shape.in_c,
        shape.in_h,
        shape.in_w,
        shape.out_c,
        shape.kh,
        shape.kw,
        shape.groups,
        shape.stride,
    )


EXPECTED_SCHEDULES = {
    "H7 spatial 160->320": (
        ("setup", 0, 7, 5, 0, 320),
        ("k_half", 0, 7, 5, 0, 160),
        ("k_half", 0, 7, 5, 160, 160),
        ("k_tile", 0, 7, 5, 0, 112),
        ("k_tile", 0, 7, 5, 112, 112),
        ("k_tile", 0, 7, 5, 224, 96),
    ),
    "H14 spatial 160->320": (
        ("setup", 0, 14, 12, 0, 320),
        ("k_half", 0, 14, 12, 0, 160),
        ("k_half", 0, 14, 12, 160, 160),
        ("k_tile", 0, 14, 12, 0, 112),
        ("k_tile", 0, 14, 12, 112, 112),
        ("k_tile", 0, 14, 12, 224, 96),
    ),
    "H40 spatial 160->320": (
        ("setup", 0, 23, 21, 0, 320),
        ("setup", 21, 19, 17, 0, 320),
        ("k_half", 0, 23, 21, 0, 160),
        ("k_half", 21, 19, 17, 0, 160),
        ("k_half", 0, 23, 21, 160, 160),
        ("k_half", 21, 19, 17, 160, 160),
        ("k_tile", 0, 23, 21, 0, 112),
        ("k_tile", 21, 19, 17, 0, 112),
        ("k_tile", 0, 23, 21, 112, 112),
        ("k_tile", 21, 19, 17, 112, 112),
        ("k_tile", 0, 23, 21, 224, 96),
        ("k_tile", 21, 19, 17, 224, 96),
    ),
    "C32/H14 spatial": (
        ("setup", 0, 14, 12, 0, 128),
        ("k_half", 0, 14, 12, 0, 64),
        ("k_half", 0, 14, 12, 64, 64),
        ("y_tile", 0, 6, 4, 0, 128),
        ("y_tile", 4, 6, 4, 0, 128),
        ("y_tile", 8, 6, 4, 0, 128),
    ),
    "H20 spatial C32->320": (
        ("setup", 0, 20, 18, 0, 320),
        ("k_half", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 160, 160),
        ("y_tile", 0, 8, 6, 0, 320),
        ("y_tile", 6, 8, 6, 0, 320),
        ("y_tile", 12, 8, 6, 0, 320),
    ),
    "H20 spatial C40->320": (
        ("setup", 0, 20, 18, 0, 320),
        ("k_half", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 160, 160),
        ("y_tile", 0, 8, 6, 0, 320),
        ("y_tile", 6, 8, 6, 0, 320),
        ("y_tile", 12, 8, 6, 0, 320),
    ),
    "H20 spatial C64->320": (
        ("setup", 0, 20, 18, 0, 320),
        ("k_half", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 160, 160),
        ("k_tile", 0, 20, 18, 0, 112),
        ("k_tile", 0, 20, 18, 112, 112),
        ("k_tile", 0, 20, 18, 224, 96),
    ),
    "H20 spatial C72->320": (
        ("setup", 0, 20, 18, 0, 320),
        ("k_half", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 160, 160),
        ("k_tile", 0, 20, 18, 0, 112),
        ("k_tile", 0, 20, 18, 112, 112),
        ("k_tile", 0, 20, 18, 224, 96),
    ),
    "H20 spatial C96->320": (
        ("setup", 0, 20, 18, 0, 320),
        ("k_half", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 160, 160),
        ("k_tile", 0, 20, 18, 0, 112),
        ("k_tile", 0, 20, 18, 112, 112),
        ("k_tile", 0, 20, 18, 224, 96),
    ),
    "H20 spatial C128->320": (
        ("setup", 0, 20, 18, 0, 320),
        ("k_half", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 160, 160),
        ("k_tile", 0, 20, 18, 0, 112),
        ("k_tile", 0, 20, 18, 112, 112),
        ("k_tile", 0, 20, 18, 224, 96),
    ),
    "H20 spatial C192->320": (
        ("setup", 0, 20, 18, 0, 320),
        ("k_half", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 160, 160),
        ("k_tile", 0, 20, 18, 0, 112),
        ("k_tile", 0, 20, 18, 112, 112),
        ("k_tile", 0, 20, 18, 224, 96),
    ),
    "H20 spatial C256->320": (
        ("setup", 0, 20, 18, 0, 320),
        ("k_half", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 160, 160),
        ("k_tile", 0, 20, 18, 0, 112),
        ("k_tile", 0, 20, 18, 112, 112),
        ("k_tile", 0, 20, 18, 224, 96),
    ),
    "H20 spatial C64->288": (
        ("setup", 0, 20, 18, 0, 288),
        ("k_half", 0, 20, 18, 0, 144),
        ("k_half", 0, 20, 18, 144, 144),
        ("k_tile", 0, 20, 18, 0, 96),
        ("k_tile", 0, 20, 18, 96, 96),
        ("k_tile", 0, 20, 18, 192, 96),
    ),
    "H20 spatial C64->32": (
        ("setup", 0, 20, 18, 0, 32),
        ("k_half", 0, 20, 18, 0, 16),
        ("k_half", 0, 20, 18, 16, 16),
        ("y_tile", 0, 8, 6, 0, 32),
        ("y_tile", 6, 8, 6, 0, 32),
        ("y_tile", 12, 8, 6, 0, 32),
    ),
    "H20 spatial C64->48": (
        ("setup", 0, 20, 18, 0, 48),
        ("y_mid", 0, 11, 9, 0, 48),
        ("y_mid", 9, 11, 9, 0, 48),
        ("k_tile", 0, 20, 18, 0, 16),
        ("k_tile", 0, 20, 18, 16, 16),
        ("k_tile", 0, 20, 18, 32, 16),
    ),
    "H20 spatial C64->64": (
        ("setup", 0, 20, 18, 0, 64),
        ("k_half", 0, 20, 18, 0, 32),
        ("k_half", 0, 20, 18, 32, 32),
        ("y_tile", 0, 8, 6, 0, 64),
        ("y_tile", 6, 8, 6, 0, 64),
        ("y_tile", 12, 8, 6, 0, 64),
    ),
    "H20 spatial C64->96": (
        ("setup", 0, 20, 18, 0, 96),
        ("k_half", 0, 20, 18, 0, 48),
        ("k_half", 0, 20, 18, 48, 48),
        ("k_tile", 0, 20, 18, 0, 32),
        ("k_tile", 0, 20, 18, 32, 32),
        ("k_tile", 0, 20, 18, 64, 32),
    ),
    "H20 spatial C64->112": (
        ("setup", 0, 20, 18, 0, 112),
        ("y_mid", 0, 11, 9, 0, 112),
        ("y_mid", 9, 11, 9, 0, 112),
        ("y_tile", 0, 8, 6, 0, 112),
        ("y_tile", 6, 8, 6, 0, 112),
        ("y_tile", 12, 8, 6, 0, 112),
    ),
    "H20 spatial C64->128": (
        ("setup", 0, 20, 18, 0, 128),
        ("k_half", 0, 20, 18, 0, 64),
        ("k_half", 0, 20, 18, 64, 64),
        ("y_tile", 0, 8, 6, 0, 128),
        ("y_tile", 6, 8, 6, 0, 128),
        ("y_tile", 12, 8, 6, 0, 128),
    ),
    "H20 spatial C64->160": (
        ("setup", 0, 20, 18, 0, 160),
        ("k_half", 0, 20, 18, 0, 80),
        ("k_half", 0, 20, 18, 80, 80),
        ("y_tile", 0, 8, 6, 0, 160),
        ("y_tile", 6, 8, 6, 0, 160),
        ("y_tile", 12, 8, 6, 0, 160),
    ),
    "H20 spatial C64->192": (
        ("setup", 0, 20, 18, 0, 192),
        ("k_half", 0, 20, 18, 0, 96),
        ("k_half", 0, 20, 18, 96, 96),
        ("k_tile", 0, 20, 18, 0, 64),
        ("k_tile", 0, 20, 18, 64, 64),
        ("k_tile", 0, 20, 18, 128, 64),
    ),
    "H20 spatial C64->224": (
        ("setup", 0, 20, 18, 0, 224),
        ("k_half", 0, 20, 18, 0, 112),
        ("k_half", 0, 20, 18, 112, 112),
        ("y_tile", 0, 8, 6, 0, 224),
        ("y_tile", 6, 8, 6, 0, 224),
        ("y_tile", 12, 8, 6, 0, 224),
    ),
    "H20 spatial C64->256": (
        ("setup", 0, 20, 18, 0, 256),
        ("k_half", 0, 20, 18, 0, 128),
        ("k_half", 0, 20, 18, 128, 128),
        ("y_tile", 0, 8, 6, 0, 256),
        ("y_tile", 6, 8, 6, 0, 256),
        ("y_tile", 12, 8, 6, 0, 256),
    ),
    "H20 spatial C64->384": (
        ("setup", 0, 20, 18, 0, 384),
        ("k_half", 0, 20, 18, 0, 192),
        ("k_half", 0, 20, 18, 192, 192),
        ("k_tile", 0, 20, 18, 0, 128),
        ("k_tile", 0, 20, 18, 128, 128),
        ("k_tile", 0, 20, 18, 256, 128),
    ),
    "H20 spatial C64->512": (
        ("setup", 0, 20, 18, 0, 512),
        ("k_half", 0, 20, 18, 0, 256),
        ("k_half", 0, 20, 18, 256, 256),
        ("k_tile", 0, 20, 18, 0, 176),
        ("k_tile", 0, 20, 18, 176, 176),
        ("k_tile", 0, 20, 18, 352, 160),
    ),
    "H20 spatial C72->288": (
        ("setup", 0, 20, 18, 0, 288),
        ("k_half", 0, 20, 18, 0, 144),
        ("k_half", 0, 20, 18, 144, 144),
        ("k_tile", 0, 20, 18, 0, 96),
        ("k_tile", 0, 20, 18, 96, 96),
        ("k_tile", 0, 20, 18, 192, 96),
    ),
    "PW C256/H14": (
        ("setup", 0, 14, 14, 0, 512),
        ("k_half", 0, 14, 14, 0, 256),
        ("k_half", 0, 14, 14, 256, 256),
        ("y_tile", 0, 5, 5, 0, 512),
        ("y_tile", 5, 5, 5, 0, 512),
        ("y_tile", 10, 4, 4, 0, 512),
    ),
    "PW C256/H28": (
        ("setup", 0, 9, 9, 0, 512),
        ("setup", 9, 9, 9, 0, 512),
        ("k_half", 0, 9, 9, 0, 256),
        ("k_half", 9, 9, 9, 0, 256),
        ("k_half", 18, 9, 9, 0, 256),
        ("k_half", 27, 1, 1, 0, 256),
        ("k_half", 0, 9, 9, 256, 256),
        ("k_half", 9, 9, 9, 256, 256),
        ("k_half", 18, 9, 9, 256, 256),
        ("k_half", 27, 1, 1, 256, 256),
        ("y_tile", 0, 5, 5, 0, 512),
        ("y_tile", 15, 5, 5, 0, 512),
        ("y_tile", 5, 5, 5, 0, 512),
        ("y_tile", 20, 4, 4, 0, 512),
        ("y_tile", 10, 5, 5, 0, 512),
        ("y_tile", 24, 4, 4, 0, 512),
    ),
    "C64/H56 pointwise": (
        ("setup", 0, 50, 50, 0, 128),
        ("setup", 50, 6, 6, 0, 128),
        ("k_half", 0, 50, 50, 0, 64),
        ("k_half", 50, 6, 6, 0, 64),
        ("k_half", 0, 50, 50, 64, 64),
        ("k_half", 50, 6, 6, 64, 64),
        ("y_tile", 0, 19, 19, 0, 128),
        ("y_tile", 19, 19, 19, 0, 128),
        ("y_tile", 38, 18, 18, 0, 128),
    ),
    "PW C40/H20": (
        ("setup", 0, 20, 20, 0, 320),
        ("k_half", 0, 20, 20, 0, 160),
        ("k_half", 0, 20, 20, 160, 160),
        ("y_tile", 0, 7, 7, 0, 320),
        ("y_tile", 7, 7, 7, 0, 320),
        ("y_tile", 14, 6, 6, 0, 320),
    ),
    "PW C40/H28": (
        ("setup", 0, 28, 28, 0, 320),
        ("k_half", 0, 28, 28, 0, 160),
        ("k_half", 0, 28, 28, 160, 160),
        ("y_tile", 0, 10, 10, 0, 320),
        ("y_tile", 10, 9, 9, 0, 320),
        ("y_tile", 19, 9, 9, 0, 320),
    ),
    "PW C40/H40": (
        ("setup", 0, 40, 40, 0, 320),
        ("k_half", 0, 40, 40, 0, 160),
        ("k_half", 0, 40, 40, 160, 160),
        ("y_tile", 0, 14, 14, 0, 320),
        ("y_tile", 14, 13, 13, 0, 320),
        ("y_tile", 27, 13, 13, 0, 320),
    ),
    "PW C40/H56": (
        ("setup", 0, 56, 56, 0, 320),
        ("k_half", 0, 56, 56, 0, 160),
        ("k_half", 0, 56, 56, 160, 160),
        ("y_tile", 0, 19, 19, 0, 320),
        ("y_tile", 19, 19, 19, 0, 320),
        ("y_tile", 38, 18, 18, 0, 320),
    ),
    "PW C528/H14": (
        ("setup", 0, 14, 14, 0, 32),
        ("k_half", 0, 14, 14, 0, 16),
        ("k_half", 0, 14, 14, 16, 16),
        ("y_tile", 0, 5, 5, 0, 32),
        ("y_tile", 5, 5, 5, 0, 32),
        ("y_tile", 10, 4, 4, 0, 32),
    ),
    "PW C528/H20": (
        ("setup", 0, 20, 20, 0, 32),
        ("k_half", 0, 20, 20, 0, 16),
        ("k_half", 0, 20, 20, 16, 16),
        ("y_tile", 0, 7, 7, 0, 32),
        ("y_tile", 7, 7, 7, 0, 32),
        ("y_tile", 14, 6, 6, 0, 32),
    ),
    "PW C528/H40": (
        ("setup", 0, 40, 40, 0, 32),
        ("k_half", 0, 40, 40, 0, 16),
        ("k_half", 0, 40, 40, 16, 16),
        ("y_tile", 0, 14, 14, 0, 32),
        ("y_tile", 14, 13, 13, 0, 32),
        ("y_tile", 27, 13, 13, 0, 32),
    ),
}


def formula_schedule(shape: Shape) -> tuple[tuple[str, int, int, int, int, int], ...]:
    rows = conv.formula_direct_spatial_schedule(
        shape.in_c,
        shape.out_c,
        shape.kh,
        shape.kw,
        shape.in_h,
        shape.in_w,
        shape.groups,
        shape.stride,
    )
    if rows is None:
        raise ValueError(f"no formula case for {shape.name}")
    return rows


def captured_schedule(shape: Shape) -> tuple[tuple[str, int, int, int, int, int], ...]:
    return EXPECTED_SCHEDULES[shape.name]


def check_shape(shape: Shape) -> bool:
    actual = formula_schedule(shape)
    expected = captured_schedule(shape)
    if actual == expected:
        print(f"PASS {shape.name}: {len(actual)} descriptors")
        return True
    print(f"FAIL {shape.name}: formula does not match captured schedule")
    print(f"  formula : {actual}")
    print(f"  captured: {expected}")
    return False


def main() -> int:
    shapes = [
        Shape("H7 spatial 160->320", 160, 7, 7, 320, 3, 3),
        Shape("H14 spatial 160->320", 160, 14, 14, 320, 3, 3),
        Shape("H40 spatial 160->320", 160, 40, 40, 320, 3, 3),
        Shape("C32/H14 spatial", 32, 14, 14, 128, 3, 3),
        Shape("H20 spatial C32->320", 32, 20, 20, 320, 3, 3),
        Shape("H20 spatial C40->320", 40, 20, 20, 320, 3, 3),
        Shape("H20 spatial C64->320", 64, 20, 20, 320, 3, 3),
        Shape("H20 spatial C72->320", 72, 20, 20, 320, 3, 3),
        Shape("H20 spatial C96->320", 96, 20, 20, 320, 3, 3),
        Shape("H20 spatial C128->320", 128, 20, 20, 320, 3, 3),
        Shape("H20 spatial C192->320", 192, 20, 20, 320, 3, 3),
        Shape("H20 spatial C256->320", 256, 20, 20, 320, 3, 3),
        Shape("H20 spatial C64->288", 64, 20, 20, 288, 3, 3),
        Shape("H20 spatial C64->32", 64, 20, 20, 32, 3, 3),
        Shape("H20 spatial C64->48", 64, 20, 20, 48, 3, 3),
        Shape("H20 spatial C64->64", 64, 20, 20, 64, 3, 3),
        Shape("H20 spatial C64->96", 64, 20, 20, 96, 3, 3),
        Shape("H20 spatial C64->112", 64, 20, 20, 112, 3, 3),
        Shape("H20 spatial C64->128", 64, 20, 20, 128, 3, 3),
        Shape("H20 spatial C64->160", 64, 20, 20, 160, 3, 3),
        Shape("H20 spatial C64->192", 64, 20, 20, 192, 3, 3),
        Shape("H20 spatial C64->224", 64, 20, 20, 224, 3, 3),
        Shape("H20 spatial C64->256", 64, 20, 20, 256, 3, 3),
        Shape("H20 spatial C64->384", 64, 20, 20, 384, 3, 3),
        Shape("H20 spatial C64->512", 64, 20, 20, 512, 3, 3),
        Shape("H20 spatial C72->288", 72, 20, 20, 288, 3, 3),
        Shape("PW C256/H14", 256, 14, 14, 512, 1, 1),
        Shape("PW C256/H28", 256, 28, 28, 512, 1, 1),
        Shape("C64/H56 pointwise", 64, 56, 56, 128, 1, 1),
        Shape("PW C40/H20", 40, 20, 20, 320, 1, 1),
        Shape("PW C40/H28", 40, 28, 28, 320, 1, 1),
        Shape("PW C40/H40", 40, 40, 40, 320, 1, 1),
        Shape("PW C40/H56", 40, 56, 56, 320, 1, 1),
        Shape("PW C528/H14", 528, 14, 14, 32, 1, 1),
        Shape("PW C528/H20", 528, 20, 20, 32, 1, 1),
        Shape("PW C528/H40", 528, 40, 40, 32, 1, 1),
    ]
    ok = True
    for shape in shapes:
        ok = check_shape(shape) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
