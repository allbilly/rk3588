"""
conv_cpu.py — Pure-numpy tiled conv using the same tiling strategy as conv.py.
Proves the reversed RKNN bank-pressure-based Y/K tiling is mathematically correct.

No NPU hardware needed. Uses the same _plan_conv_tiles / _compute_k_step /
_compute_y_step functions, then computes each tile on CPU with numpy and
assembles the result.
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

# ---- constants (same as conv.py) ----
FP16_BYTES = 2
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK3588_CBUF_BANKS = 12
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_MAX_CONV_FLAT_STRIDE = 992
UNPACK_C2 = 8
OUTPUT_MEM_BYTES = 4 * 1024 * 1024

_CBUF_PROFILES = {
    "rk3588": {"banks": 12, "entry_bytes": 128, "entries_per_bank": 256},
    # nv_full has 16 CBUF banks. Keep the RK/NVDLA C-model entry geometry here
    # because this harness compares planner pressure, not generated NVDLA RTL.
    "nvdla_full": {"banks": 16, "entry_bytes": 128, "entries_per_bank": 256},
}

# ---- helpers (same as conv.py) ----

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

def _set_cbuf_profile(name):
    global CBUF_ENTRY_BYTES, CBUF_ENTRIES_PER_BANK, RK_CBUF_BANKS, CBUF_BANK_SIZE
    profile = _CBUF_PROFILES[name]
    CBUF_ENTRY_BYTES = profile["entry_bytes"]
    CBUF_ENTRIES_PER_BANK = profile["entries_per_bank"]
    RK_CBUF_BANKS = profile["banks"]
    CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES

def _with_cbuf_profile(name, fn, *args):
    global CBUF_ENTRY_BYTES, CBUF_ENTRIES_PER_BANK, RK_CBUF_BANKS, CBUF_BANK_SIZE
    old = (CBUF_ENTRY_BYTES, CBUF_ENTRIES_PER_BANK, RK_CBUF_BANKS, CBUF_BANK_SIZE)
    _set_cbuf_profile(name)
    try:
        return fn(*args)
    finally:
        CBUF_ENTRY_BYTES, CBUF_ENTRIES_PER_BANK, RK_CBUF_BANKS, CBUF_BANK_SIZE = old

def _mesa_entries_per_slice(input_width, input_channels):
    atomics_per_entry = CBUF_ENTRY_BYTES // 16
    total_c_atomics = _ceil_div(input_channels * FP16_BYTES, 16)
    last_c_atomics = total_c_atomics % atomics_per_entry
    int_c_entries = (total_c_atomics // atomics_per_entry) * input_width
    frac_c_entries = input_width if last_c_atomics == 3 else _ceil_div(last_c_atomics * input_width, atomics_per_entry)
    return int_c_entries + frac_c_entries

def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c

def _conv_align_c(in_c, groups, out_c):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    if not is_depthwise and (groups > 1 or in_c > 4):
        return 16
    min_align = 16 if is_depthwise else 8
    return max(min_align, min(1 << (max(1, in_c) - 1).bit_length(), 32 if is_depthwise else 16))

def _pointwise_weight_atom_groups(in_c):
    return _ceil_div(max(in_c, 16), 32)

def _uses_pointwise_weight_atom_layout(in_c, kh, kw, groups):
    return groups == 1 and kh == 1 and kw == 1 and _pointwise_weight_atom_groups(in_c) > 1

def _cdma_dc_feature_input_pack_c2(in_c, groups, out_c, align_c):
    if in_c == 1:
        return 2
    if not _is_depthwise(in_c, out_c, groups) and groups == 1 and 1 < in_c <= 4:
        return 8
    return 8

def _feature_grains(row_bytes, floor_grains, use_nhwc_pack=False, is_spatial=False, is_depthwise=False):
    if use_nhwc_pack and is_spatial:
        return floor_grains
    if is_depthwise and is_spatial:
        return min(13, floor_grains)
    even_rows_per_two_banks = (_ceil_div(2 * CBUF_BANK_SIZE, row_bytes) + 1) & ~1
    return min(floor_grains, even_rows_per_two_banks)

def _conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride=1):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    is_spatial = (kh != 1 or kw != 1)
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    align_c = 32 if _uses_pointwise_weight_atom_layout(in_c, kh, kw, groups) else _conv_align_c(in_c, groups, out_c)
    align_out_c = max(16, _align_up(out_c, 16))
    width_stride = _align_up(in_w, max(1, _ceil_div(16, align_c)))
    out_width_stride = out_h * out_w if not is_spatial else _align_up(out_h * out_w, 4)
    input_pack_c2 = _cdma_dc_feature_input_pack_c2(in_c, groups, out_c, align_c)
    use_nhwc = (not is_depthwise) and (not (groups > 1 and is_spatial)) and in_c < input_pack_c2
    return {"in_c": in_c, "in_h": in_h, "in_w": in_w, "out_c": out_c, "kh": kh, "kw": kw,
            "groups": groups, "stride": stride, "is_depthwise": is_depthwise, "is_spatial": is_spatial,
            "out_h": out_h, "out_w": out_w, "align_c": align_c, "align_out_c": align_out_c,
            "width_stride": width_stride, "out_width_stride": out_width_stride,
            "input_pack_c2": input_pack_c2, "use_nhwc": use_nhwc}

def _mesa_weight_banks(weights_width, weights_height, input_channels, output_channels, depthwise):
    weight_bytes = weights_width * weights_height * input_channels * FP16_BYTES
    if not depthwise:
        weight_bytes *= output_channels
    return _ceil_div(_ceil_div(weight_bytes, CBUF_ENTRY_BYTES), CBUF_ENTRIES_PER_BANK) + 1

def _pointwise_oc_tile_c(in_c):
    max_tile = CBUF_BANK_SIZE // (max(1, in_c) * FP16_BYTES)
    return 32 if max_tile >= 32 else 16 if max_tile >= 16 else 8 if max_tile >= 8 else 4

def _mesa_output_tile_h(input_width, out_h, input_channels, output_channels, kh, kw, stride, depthwise, input_banks=None):
    if input_banks is None:
        weight_banks = _mesa_weight_banks(kw, kh, input_channels, output_channels, depthwise)
        input_banks = RK_CBUF_BANKS - weight_banks if weight_banks + 1 < RK_CBUF_BANKS else 7
    entries_per_slice = max(1, _mesa_entries_per_slice(input_width, input_channels))
    input_slices = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // entries_per_slice)
    output_rows = max(1, (input_slices - kh) // stride + 1)
    return min(out_h, output_rows)

def _pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile, stride=1):
    return _mesa_output_tile_h(out_w, out_h, in_c, oc_tile, 1, 1, stride, False)

def _depthwise_tile_h(total_channels, out_h, in_w, kh, kw, stride=1):
    tile_h = _mesa_output_tile_h(in_w, out_h, total_channels, total_channels, kh, kw, stride, True, input_banks=7)
    if total_channels > 64:
        align_c = _conv_align_c(total_channels, total_channels, total_channels)
        row_bytes = in_w * align_c * FP16_BYTES
        max_feature_rows = _feature_grains(row_bytes, out_h + kh, is_spatial=True, is_depthwise=True) + 1
        tile_h = min(tile_h, max_feature_rows)
    tile_h = max(10, tile_h) if tile_h < out_h else tile_h
    return tile_h if tile_h == out_h or tile_h % 2 == 0 else tile_h - 1

def _needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
    if groups != 1 or kh != 1 or kw != 1:
        return False
    out_h = in_h
    out_w = in_w
    oc_tile = _pointwise_oc_tile_c(in_c)
    return out_c > oc_tile or (in_c >= 16 and out_c % oc_tile != 0) or _pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile) < out_h

# ---- tiling strategy (same as conv.py) ----

_SPLIT_NONE = 0
_SPLIT_BY_Y = 1
_SPLIT_BY_K = 2
_SPLIT_BY_YK = 3

_SPLIT_NAMES = {
    _SPLIT_NONE: "NONE",
    _SPLIT_BY_Y: "BY_Y",
    _SPLIT_BY_K: "BY_K",
    _SPLIT_BY_YK: "BY_YK",
}

_FAMILY_BITS = {
    "setup": 0x00000000,
    "y_tile": 0x20000000,
    "k_half": 0x40000000,
    "k_tile": 0x50000000,
}

_POINTWISE_Y_TILE_HARDCODED = {
    "conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1",
    "conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1",
}

def _compute_k_step(in_c, out_c, kh, kw, groups, p):
    is_depthwise, is_spatial = p["is_depthwise"], p["is_spatial"]
    data_in_channel_aligned = _align_up(in_c, p["align_c"])
    weight_kernel_bytes = kh * kw * data_in_channel_aligned * FP16_BYTES
    tweight = weight_kernel_bytes * (1 if is_depthwise else out_c)
    weight_banks = _ceil_div(tweight, CBUF_BANK_SIZE)

    k_step = out_c
    if is_depthwise and is_spatial:
        k_step = min(32, out_c)
    elif is_spatial and groups == 1 and not is_depthwise and weight_banks > 3:
        k_step = 32 if out_c >= 32 else out_c
    elif not is_spatial and groups == 1:
        pw_oc = _pointwise_oc_tile_c(in_c)
        if weight_banks > 3:
            k_step = max(pw_oc, 32)
        elif out_c > pw_oc:
            k_step = pw_oc
    return min(k_step, out_c)

def _compute_y_step(in_c, out_c, kh, kw, in_h, in_w, groups, stride, k_step, p):
    is_spatial, is_depthwise = p["is_spatial"], p["is_depthwise"]
    out_h = p["out_h"]

    if is_depthwise and is_spatial:
        data_in_channel_aligned = _align_up(in_c, p["align_c"])
        weight_kernel_bytes = kh * kw * data_in_channel_aligned * FP16_BYTES
        weight_banks = _ceil_div(_ceil_div(weight_kernel_bytes, CBUF_ENTRY_BYTES), CBUF_ENTRIES_PER_BANK) + 1
        input_banks = RK_CBUF_BANKS - weight_banks if weight_banks + 1 < RK_CBUF_BANKS else 7
        ae = CBUF_ENTRY_BYTES // 16
        tca = _ceil_div(in_c * FP16_BYTES, 16)
        lca = tca % ae
        ice = (tca // ae) * in_w
        fce = in_w if lca == 3 else _ceil_div(lca * in_w, ae)
        eps = max(1, ice + fce)
        s = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // eps)
        tile_h = min(out_h, max(1, (s - kh) // stride + 1))
        if in_c > 64:
            row_bytes = in_w * _conv_align_c(in_c, in_c, in_c) * FP16_BYTES
            max_rows = _feature_grains(row_bytes, out_h + kh, True, True) + 1
            tile_h = min(tile_h, max_rows)
        tile_h = max(10, tile_h) if tile_h < out_h else tile_h
        max_input_rows = min(15, out_h + kh - 1)
        tile_h = min(tile_h, max_input_rows - kh + 1)
        tile_h = tile_h if tile_h == out_h or tile_h % 2 == 0 else tile_h - 1
        return tile_h

    data_in_channel_aligned = _align_up(in_c, p["align_c"])
    tile_in_c = k_step if is_depthwise else in_c
    tile_data_in_channel_aligned = _align_up(tile_in_c, p["align_c"])
    row_bytes = p["width_stride"] * tile_data_in_channel_aligned * FP16_BYTES

    y_step = out_h
    tile_wb = _ceil_div(kh * kw * data_in_channel_aligned * FP16_BYTES * (k_step if not is_depthwise else 1), CBUF_BANK_SIZE)
    remaining = max(1, RK_CBUF_BANKS - tile_wb)
    tile_in_c = k_step if is_depthwise else in_c
    tile_data_in_channel_aligned = _align_up(tile_in_c, p["align_c"])
    row_bytes = p["width_stride"] * tile_data_in_channel_aligned * FP16_BYTES
    fg = _feature_grains(row_bytes, in_h + kh, False, is_spatial, is_depthwise)
    data_banks_needed = _ceil_div(row_bytes * fg, CBUF_BANK_SIZE)
    if data_banks_needed > remaining:
        y_step = max(1, out_h * remaining // max(1, data_banks_needed))

    if not is_spatial:
        out_w = p["out_w"]
        small_channel = in_c <= 4 and not is_depthwise
        if small_channel and p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE:
            y_step = min(y_step, max(1, RK_MAX_CONV_FLAT_STRIDE // out_w))
        elif out_h > 50:
            if in_c >= 128 and out_c >= 128:
                y_step = min(y_step, 25)
            elif p["out_width_stride"] > RK_MAX_CONV_FLAT_STRIDE:
                y_step = min(y_step, 32)
            else:
                y_step = min(y_step, 50)

    if not is_depthwise:
        eps = _mesa_entries_per_slice(p["width_stride"], data_in_channel_aligned)
        input_banks = RK_CBUF_BANKS - tile_wb if tile_wb + 1 < RK_CBUF_BANKS else 7
        max_input_h = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // eps)
        max_y = max(1, (max_input_h - kh) // stride + 1)
        y_step = min(y_step, max_y)

    if is_spatial and p["use_nhwc"]:
        nhwc_data_bank = RK_CBUF_BANKS - 1
        nhwc_row_bytes = p["width_stride"] * tile_data_in_channel_aligned * FP16_BYTES
        max_grains = (nhwc_data_bank * CBUF_BANK_SIZE) // nhwc_row_bytes
        max_y_for_cbuf = max(1, max_grains - 2 * kh + 1)
        y_step = min(y_step, max_y_for_cbuf)

    return y_step

def _plan_conv_tiles(in_c, out_c, kh, kw, in_h, in_w, groups, stride):
    p = _conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    out_h = p["out_h"]
    k_step = _compute_k_step(in_c, out_c, kh, kw, groups, p)
    y_step = _compute_y_step(in_c, out_c, kh, kw, in_h, in_w, groups, stride, k_step, p)

    split_method = _SPLIT_NONE
    if k_step < out_c and y_step < out_h:
        split_method = _SPLIT_BY_YK
    elif k_step < out_c:
        split_method = _SPLIT_BY_K
    elif y_step < out_h:
        split_method = _SPLIT_BY_Y

    y_boundary = [0]
    while y_boundary[-1] < out_h:
        y_boundary.append(int(min(y_boundary[-1] + y_step, out_h)))
    k_boundary = [0]
    while k_boundary[-1] < out_c:
        k_boundary.append(int(min(k_boundary[-1] + k_step, out_c)))

    tiles = []
    for yi in range(len(y_boundary) - 1):
        ys = y_boundary[yi]
        y_span = y_boundary[yi + 1] - ys
        for ki in range(len(k_boundary) - 1):
            ks = k_boundary[ki]
            k_span = k_boundary[ki + 1] - ks
            tiles.append({"y_start": ys, "y_step": y_span, "k_start": ks, "k_step": k_span})
    return p, split_method, tiles, y_step, k_step


def _split_name(split_method):
    return _SPLIT_NAMES[split_method]


def _conv_output_count(p):
    return _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2


def _conv_output_bytes(p):
    return _conv_output_count(p) * FP16_BYTES


def _tile_boundaries(tiles, key_start, key_step):
    boundaries = {0}
    for tile in tiles:
        boundaries.add(tile[key_start])
        boundaries.add(tile[key_start] + tile[key_step])
    return sorted(boundaries)


def _tile_windows(boundaries):
    return [(boundaries[i], boundaries[i + 1] - boundaries[i]) for i in range(len(boundaries) - 1)]


def _old_strategy_name(s):
    stride = s.get("stride", 1)
    p = _conv_params(s["in_c"], s["in_h"], s["in_w"], s["out_c"],
                     s["kh"], s["kw"], s["groups"], stride)
    if s["name"] in _POINTWISE_Y_TILE_HARDCODED:
        return "pointwise_y_tile_hardcoded"

    is_spatial = p["is_spatial"]
    is_depthwise = p["is_depthwise"]
    grouped_serial = is_spatial and s["groups"] > 1 and not is_depthwise
    spatial_weight_banks = _mesa_weight_banks(s["kw"], s["kh"], s["in_c"], s["out_c"], False) if is_spatial else 0
    spatial_im2col = (is_spatial and s["groups"] == 1 and not is_depthwise and
                      (spatial_weight_banks > RK3588_CBUF_BANKS // 3 or _conv_output_bytes(p) > OUTPUT_MEM_BYTES))
    spatial_oc_serial = (is_spatial and s["groups"] == 1 and not is_depthwise and
                         s["out_c"] > UNPACK_C2 and (s["in_c"] % 16 != 0 or s["in_c"] >= 16))
    depthwise_spatial_tiled = (is_depthwise and is_spatial and
                               (p["out_h"] > _depthwise_tile_h(s["out_c"], p["out_h"], s["in_w"], s["kh"], s["kw"], stride) or
                                s["out_c"] > p["align_c"]))
    if grouped_serial:
        return "grouped_serial"
    if spatial_im2col:
        return "spatial_im2col"
    if spatial_oc_serial:
        return "spatial_oc_serial"
    if depthwise_spatial_tiled:
        return "depthwise_spatial_tiled"
    if (not is_spatial and not is_depthwise and s["groups"] == 1 and
            _needs_pointwise_oc_tile_schedule(s["in_c"], s["out_c"], s["in_h"], s["in_w"], s["kh"], s["kw"], s["groups"])):
        return "pointwise_oc_tile"
    return "fallback/direct"


def _descriptor_families(split_method):
    if split_method == _SPLIT_NONE:
        return ["setup"]
    if split_method == _SPLIT_BY_Y:
        return ["y_tile"]
    if split_method == _SPLIT_BY_K:
        return ["k_tile"]
    return ["setup", "k_half", "k_tile"]


def _descriptor_unresolved_fields(split_method, groups, is_depthwise):
    fields = ["grain_bits", "cbuf0", "mc_treat_by_y_tile", "mc_treat_by_k_tile",
              "mc_treat_by_1c_y_tile", "mc_treat_by_1c_k_tile"]
    if split_method == _SPLIT_BY_YK:
        fields.append("family_k_window_assignment")
    if groups > 1 and not is_depthwise:
        fields.append("group_lowering_descriptor_contract")
    return fields


def _estimate_bank_fields(p, kh, kw, input_h, input_w, oc_count):
    tile_in_c = oc_count if p["is_depthwise"] else p["in_c"]
    aligned_in_c = _align_up(tile_in_c, p["align_c"])
    feature_bytes = input_h * input_w * aligned_in_c * FP16_BYTES
    weight_oc = 1 if p["is_depthwise"] else oc_count
    weight_bytes = kh * kw * _align_up(p["in_c"], p["align_c"]) * FP16_BYTES * weight_oc
    input_bank_num = max(1, _ceil_div(feature_bytes, CBUF_BANK_SIZE))
    weight_bank_num = max(1, _ceil_div(weight_bytes, CBUF_BANK_SIZE))
    return input_bank_num, weight_bank_num


def _descriptor_offsets(p, kh, kw, y_start, input_h, k_start, oc_count, stride):
    feature_off = y_start * stride * p["width_stride"] * p["align_c"] * FP16_BYTES
    if p["is_depthwise"]:
        weight_off = k_start * kh * kw * FP16_BYTES
    else:
        weight_off = k_start * kh * kw * _align_up(p["in_c"], p["align_c"]) * FP16_BYTES
    output_off = (k_start * p["out_h"] * p["out_w"] + y_start * p["out_w"]) * FP16_BYTES
    return feature_off, weight_off, output_off


def _descriptor_rows_for_shape(s):
    stride = s.get("stride", 1)
    p, split_method, tiles, _y_step, _k_step = _plan_conv_tiles(
        s["in_c"], s["out_c"], s["kh"], s["kw"], s["in_h"], s["in_w"], s["groups"], stride)
    y_boundaries = _tile_boundaries(tiles, "y_start", "y_step")
    k_boundaries = _tile_boundaries(tiles, "k_start", "k_step")
    y_windows = _tile_windows(y_boundaries)
    k_windows = _tile_windows(k_boundaries)
    families = _descriptor_families(split_method)
    rows = []
    for family in families:
        for k_index, (k_start, oc_count) in enumerate(k_windows):
            for y_index, (y_start, output_h) in enumerate(y_windows):
                input_h = min((output_h - 1) * stride + s["kh"], s["in_h"] - y_start * stride)
                feature_off, weight_off, output_off = _descriptor_offsets(
                    p, s["kh"], s["kw"], y_start, input_h, k_start, oc_count, stride)
                input_bank_num, weight_bank_num = _estimate_bank_fields(
                    p, s["kh"], s["kw"], input_h, s["in_w"], oc_count)
                rows.append({
                    "name": s["name"],
                    "old_strategy": _old_strategy_name(s),
                    "split_method": _split_name(split_method),
                    "family": family,
                    "family_bits": f"0x{_FAMILY_BITS[family]:08x}",
                    "grain_bits": None,
                    "y_start": y_start,
                    "input_h": input_h,
                    "output_h": output_h,
                    "output_w": p["out_w"],
                    "k_start": k_start,
                    "oc_count": oc_count,
                    "feature_off": feature_off,
                    "weight_off": weight_off,
                    "output_off": output_off,
                    "input_bank_num": input_bank_num,
                    "weight_bank_num": weight_bank_num,
                    "cbuf0": None,
                    "data_reuse": k_index > 0,
                    "weight_reuse": y_index > 0,
                    "mc_treat_by_y_tile": None,
                    "mc_treat_by_k_tile": None,
                    "mc_treat_by_1c_y_tile": None,
                    "mc_treat_by_1c_k_tile": None,
                })
    return p, split_method, y_boundaries, k_boundaries, rows


def _planner_report_row(s):
    p, split_method, y_boundaries, k_boundaries, desc_rows = _descriptor_rows_for_shape(s)
    families = sorted({row["family"] for row in desc_rows}, key=_descriptor_families(split_method).index)
    unresolved = _descriptor_unresolved_fields(split_method, s["groups"], p["is_depthwise"])
    return {
        "name": s["name"],
        "old_strategy": _old_strategy_name(s),
        "split_method": _split_name(split_method),
        "y_boundaries": y_boundaries,
        "k_boundaries": k_boundaries,
        "descriptor_count": len(desc_rows),
        "descriptor_families": families,
        "unresolved_fields": unresolved,
    }


def _planner_report_rows():
    rows = []
    for s in SHAPES:
        rows.append(_planner_report_row(s))
    return rows


def _format_cell(value):
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return str(value)


def _print_table(rows, columns):
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(_format_cell(row[col])))
    print("  ".join(f"{col:<{widths[col]}}" for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(f"{_format_cell(row[col]):<{widths[col]}}" for col in columns))


def print_planner_report():
    columns = ["name", "old_strategy", "split_method", "y_boundaries", "k_boundaries",
               "descriptor_count", "descriptor_families", "unresolved_fields"]
    _print_table(_planner_report_rows(), columns)


def print_descriptor_dump(shape_name=None):
    shapes = [s for s in SHAPES if shape_name is None or s["name"] == shape_name]
    if shape_name is not None and not shapes:
        raise SystemExit(f"unknown shape: {shape_name}")
    rows = []
    for s in shapes:
        rows.extend(_descriptor_rows_for_shape(s)[4])
    columns = ["name", "family", "family_bits", "grain_bits", "y_start", "input_h",
               "output_h", "output_w", "k_start", "oc_count", "feature_off", "weight_off",
               "output_off", "input_bank_num", "weight_bank_num", "cbuf0", "data_reuse", "weight_reuse"]
    _print_table(rows, columns)


def print_cross_tab():
    buckets = defaultdict(lambda: {"count": 0, "old": Counter()})
    for row in _planner_report_rows():
        key = (row["split_method"], tuple(row["descriptor_families"]))
        buckets[key]["count"] += 1
        buckets[key]["old"][row["old_strategy"]] += 1
    rows = []
    for (split_method, families), info in sorted(buckets.items()):
        old = ", ".join(f"{name}:{count}" for name, count in sorted(info["old"].items()))
        rows.append({
            "split_method": split_method,
            "descriptor_families": list(families),
            "count": info["count"],
            "old_branches_covered": old,
        })
    _print_table(rows, ["split_method", "descriptor_families", "count", "old_branches_covered"])


def _cbuf_compare_rows():
    rows = []
    for s in SHAPES:
        rk = _with_cbuf_profile("rk3588", _planner_report_row, s)
        nv = _with_cbuf_profile("nvdla_full", _planner_report_row, s)
        changed_fields = []
        for field in ("split_method", "y_boundaries", "k_boundaries", "descriptor_count", "descriptor_families"):
            if rk[field] != nv[field]:
                changed_fields.append(field)
        rows.append({
            "name": s["name"],
            "old_strategy": rk["old_strategy"],
            "rk_split": rk["split_method"],
            "nv_split": nv["split_method"],
            "rk_y_boundaries": rk["y_boundaries"],
            "nv_y_boundaries": nv["y_boundaries"],
            "rk_k_boundaries": rk["k_boundaries"],
            "nv_k_boundaries": nv["k_boundaries"],
            "rk_descriptor_count": rk["descriptor_count"],
            "nv_descriptor_count": nv["descriptor_count"],
            "rk_descriptor_families": rk["descriptor_families"],
            "nv_descriptor_families": nv["descriptor_families"],
            "changed_fields": changed_fields,
        })
    return rows


def print_cbuf_compare(all_rows=False):
    rows = _cbuf_compare_rows()
    changed = [row for row in rows if row["changed_fields"]]
    same = len(rows) - len(changed)
    print(f"profiles: rk3588=12x32KiB, nvdla_full=16x32KiB planner budget")
    print(f"same={same} changed={len(changed)} total={len(rows)}")
    print()
    split_counts = Counter((row["rk_split"], row["nv_split"]) for row in rows)
    summary_rows = []
    for (rk_split, nv_split), count in sorted(split_counts.items()):
        summary_rows.append({"rk_split": rk_split, "nv_split": nv_split, "count": count})
    _print_table(summary_rows, ["rk_split", "nv_split", "count"])
    print()
    detail_rows = rows if all_rows else changed
    columns = ["name", "old_strategy", "rk_split", "nv_split", "rk_y_boundaries", "nv_y_boundaries",
               "rk_k_boundaries", "nv_k_boundaries", "rk_descriptor_count", "nv_descriptor_count",
               "rk_descriptor_families", "nv_descriptor_families", "changed_fields"]
    _print_table(detail_rows, columns)


def _shape_by_name(name):
    for s in SHAPES:
        if s["name"] == name:
            return s
    raise KeyError(name)


def _unique_in_order(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _add_evidence_row(rows, target, check, ok, pass_detail, fail_detail, fail_status="FAIL"):
    rows.append({
        "target": target,
        "check": check,
        "status": "PASS" if ok else fail_status,
        "detail": pass_detail if ok else fail_detail,
    })


def _offsets_are_additive(rows, p):
    feature_by_y = {}
    weight_by_k = {}
    for row in rows:
        y_start = row["y_start"]
        k_start = row["k_start"]
        expected_output = (k_start * p["out_h"] * p["out_w"] + y_start * p["out_w"]) * FP16_BYTES
        if row["output_off"] != expected_output:
            return False
        if y_start in feature_by_y and feature_by_y[y_start] != row["feature_off"]:
            return False
        if k_start in weight_by_k and weight_by_k[k_start] != row["weight_off"]:
            return False
        feature_by_y[y_start] = row["feature_off"]
        weight_by_k[k_start] = row["weight_off"]
    return True


def _family_oc_counts(rows, family):
    counts = []
    for row in rows:
        if row["family"] == family and row["y_start"] == 0:
            counts.append(row["oc_count"])
    return counts


def _evidence_check_rows():
    rows = []

    mixed = dict(name="evidence_mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1",
                 batch=1, in_c=160, in_h=40, in_w=40, out_c=320,
                 weight_in_c=160, kh=3, kw=3, groups=1)
    p, split_method, y_boundaries, k_boundaries, desc = _descriptor_rows_for_shape(mixed)
    families = _unique_in_order([row["family"] for row in desc])
    target = "mixed 160->320 3x3 h40"
    _add_evidence_row(rows, target, "high-level split", _split_name(split_method) == "BY_YK",
                      f"split=BY_YK y={y_boundaries} k={k_boundaries}",
                      f"split={_split_name(split_method)} y={y_boundaries} k={k_boundaries}")
    _add_evidence_row(rows, target, "independent Y/K windows", len(y_boundaries) > 2 and len(k_boundaries) > 2,
                      f"y_windows={len(y_boundaries) - 1} k_windows={len(k_boundaries) - 1}",
                      f"y_windows={len(y_boundaries) - 1} k_windows={len(k_boundaries) - 1}")
    _add_evidence_row(rows, target, "family order", families == ["setup", "k_half", "k_tile"],
                      f"families={families}", f"families={families}")
    _add_evidence_row(rows, target, "additive offsets", _offsets_are_additive(desc, p),
                      "feature_off by Y, weight_off by K, output_off additive",
                      "offset dependency check failed")
    _add_evidence_row(rows, target, "exact RKNN descriptor count", len(desc) == 12,
                      "descriptor_count=12", f"descriptor_count={len(desc)} expected setup x2, k_half x4, k_tile x6", "GAP")
    _add_evidence_row(rows, target, "exact RKNN k_tile OC windows", _family_oc_counts(desc, "k_tile") == [112, 112, 96],
                      "k_tile OC windows match 112;112;96",
                      f"k_tile OC windows={_family_oc_counts(desc, 'k_tile')} expected 112;112;96", "GAP")

    h14 = _shape_by_name("b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid")
    _p, _split_method, _y_boundaries, _k_boundaries, h14_desc = _descriptor_rows_for_shape(h14)
    h14_target = "spatial 160->320 3x3 h14"
    _add_evidence_row(rows, h14_target, "spatial k_tile rows present",
                      any(row["family"] == "k_tile" for row in h14_desc),
                      f"k_tile_rows={sum(1 for row in h14_desc if row['family'] == 'k_tile')}",
                      "no k_tile rows")
    _add_evidence_row(rows, h14_target, "cbuf0/grain kept explicit",
                      all(row["grain_bits"] is None and row["cbuf0"] is None for row in h14_desc),
                      "grain_bits=unknown and cbuf0=unknown", "grain_bits or cbuf0 hidden behind a default")
    _add_evidence_row(rows, h14_target, "exact RKNN k_tile OC windows", _family_oc_counts(h14_desc, "k_tile") == [112, 112, 96],
                      "k_tile OC windows match 112;112;96",
                      f"k_tile OC windows={_family_oc_counts(h14_desc, 'k_tile')} expected 112;112;96", "GAP")

    pointwise = _shape_by_name("conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1")
    _p, pointwise_split, pointwise_y, _pointwise_k, pointwise_desc = _descriptor_rows_for_shape(pointwise)
    pointwise_target = "pointwise exported Y tile"
    _add_evidence_row(rows, pointwise_target, "y_tile rows present", _split_name(pointwise_split) == "BY_Y" and pointwise_y == [0, 25, 28],
                      f"split=BY_Y y={pointwise_y}", f"split={_split_name(pointwise_split)} y={pointwise_y}")
    _add_evidence_row(rows, pointwise_target, "grain_bits explicit unknown",
                      all(row["grain_bits"] is None for row in pointwise_desc),
                      "all descriptor grain_bits=unknown", "grain_bits hidden behind a default")
    _add_evidence_row(rows, pointwise_target, "cbuf0 separate from grain_bits",
                      all(row["cbuf0"] is None for row in pointwise_desc),
                      "all descriptor cbuf0=unknown", "cbuf0 hidden behind grain/default")

    hardcoded = []
    for name in sorted(_POINTWISE_Y_TILE_HARDCODED):
        _p, hard_split, hard_y, hard_k, hard_desc = _descriptor_rows_for_shape(_shape_by_name(name))
        hardcoded.append(f"{name}:{_split_name(hard_split)} y={hard_y} k={hard_k} desc={len(hard_desc)}")
    _add_evidence_row(rows, "pointwise_y_tile_hardcoded", "rare branch not explained as Y/K yet",
                      all(":NONE " not in item for item in hardcoded),
                      "; ".join(hardcoded), "; ".join(hardcoded), "GAP")

    all_desc = []
    for s in SHAPES:
        all_desc.extend(_descriptor_rows_for_shape(s)[4])
    _add_evidence_row(rows, "all descriptor rows", "unresolved fields visible",
                      all(row["grain_bits"] is None and row["cbuf0"] is None for row in all_desc),
                      f"checked {len(all_desc)} descriptor rows", "some unresolved fields are hidden")
    return rows


def print_evidence_check():
    rows = _evidence_check_rows()
    _print_table(rows, ["target", "check", "status", "detail"])
    counts = Counter(row["status"] for row in rows)
    print()
    _print_table([{"status": status, "count": count} for status, count in sorted(counts.items())],
                 ["status", "count"])


# ---- CPU convolution (numpy, fp16) ----

def _conv2d_tile_fp16(input_tile, weight_tile, kh, kw, stride, groups, is_depthwise):
    """
    Compute conv2d for a single tile in fp16.
    input_tile:  (C_in, H_in, W_in)   fp16
    weight_tile: (C_out, C_in/g, kh, kw) fp16   or (C_out, C_out, kh, kw) for depthwise
    Returns:     (C_out, H_out, W_out) fp16
    """
    c_out = weight_tile.shape[0]
    h_in, w_in = input_tile.shape[1], input_tile.shape[2]
    h_out = (h_in - kh) // stride + 1
    w_out = (w_in - kw) // stride + 1
    result = np.zeros((c_out, h_out, w_out), dtype=np.float16)

    c_in_per_group = input_tile.shape[0] // groups
    c_out_per_group = c_out // groups if not is_depthwise else c_out

    # Use float32 accumulator for correctness, cast back to fp16
    result_f32 = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    inp_f32 = input_tile.astype(np.float32)
    wt_f32 = weight_tile.astype(np.float32)

    for g in range(groups):
        oc_start = g * c_out_per_group
        oc_end = oc_start + c_out_per_group
        ic_start = g * c_in_per_group
        ic_end = ic_start + c_in_per_group
        for oc_local, oc in enumerate(range(oc_start, oc_end)):
            for ic in range(ic_start, ic_end):
                w_ic = ic - ic_start if not is_depthwise else oc_local
                for i in range(kh):
                    for j in range(kw):
                        patch = inp_f32[ic, i:i + stride * h_out:stride, j:j + stride * w_out:stride]
                        result_f32[oc] += patch * wt_f32[oc, w_ic, i, j]

    return result_f32.astype(np.float16)


def _conv2d_tile_fast(input_tile, weight_tile, kh, kw, stride, groups, is_depthwise):
    """
    Faster tiled conv2d using numpy vectorized operations.
    Same result as _conv2d_tile_fp16 but ~100x faster.
    """
    c_out = weight_tile.shape[0]
    h_in, w_in = input_tile.shape[1], input_tile.shape[2]
    c_in = input_tile.shape[0]
    h_out = (h_in - kh) // stride + 1
    w_out = (w_in - kw) // stride + 1

    inp_f32 = input_tile.astype(np.float32)
    wt_f32 = weight_tile.astype(np.float32)
    result_f32 = np.zeros((c_out, h_out, w_out), dtype=np.float32)

    c_in_per_group = c_in // groups

    if is_depthwise:
        # depthwise: each output channel uses its own input channel
        for i in range(kh):
            for j in range(kw):
                patch = inp_f32[:, i:i + stride * h_out:stride, j:j + stride * w_out:stride]
                # patch shape: (c_out, h_out, w_out), wt shape: (c_out, c_out, kh, kw)
                # depthwise weight has wt[c, c, i, j] for channel c
                w_slice = np.diagonal(wt_f32[:, :, i, j])  # shape (c_out,)
                result_f32 += patch * w_slice[:, None, None]
    else:
        c_out_per_group = c_out // groups
        for g in range(groups):
            oc_start = g * c_out_per_group
            oc_end = oc_start + c_out_per_group
            ic_start = g * c_in_per_group
            ic_end = ic_start + c_in_per_group
            w_group = wt_f32[oc_start:oc_end]  # (c_out_per_group, c_in_per_group, kh, kw)
            inp_group = inp_f32[ic_start:ic_end]  # (c_in_per_group, h_in, w_in)
            for i in range(kh):
                for j in range(kw):
                    # patch: (c_in_per_group, h_out, w_out)
                    patch = inp_group[:, i:i + stride * h_out:stride, j:j + stride * w_out:stride]
                    # w_slice: (c_out_per_group, c_in_per_group)
                    w_slice = w_group[:, :, i, j]
                    # result += einsum('oi,ihw->ohw', w_slice, patch)
                    result_f32[oc_start:oc_end] += np.einsum('oi,ihw->ohw', w_slice, patch)

    return result_f32.astype(np.float16)


def run_conv_tiled(batch, in_c, out_c, kh, kw, input_hw, groups=1, stride=1):
    """Run tiled conv on CPU using the same tiling strategy as conv.py."""
    in_h, in_w = input_hw
    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, in_c // groups, kh, kw)).astype(np.float16)

    p, split_method, tiles, y_step, k_step = _plan_conv_tiles(in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    out_h, out_w = p["out_h"], p["out_w"]
    is_spatial, is_depthwise = p["is_depthwise"], p["is_depthwise"]

    use_pointwise_oc_schedule = (not is_spatial and not is_depthwise and groups == 1 and
                                  _needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups))
    grouped_serial = is_spatial and groups > 1 and not is_depthwise
    spatial_oc_serial = is_spatial and groups == 1 and not is_depthwise and out_c > 8 and (in_c % 16 != 0 or in_c >= 16)
    depthwise_spatial_tiled = is_depthwise and is_spatial

    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)

    for n in range(batch):
        # ---- grouped_serial path ----
        if grouped_serial:
            input_per_group = in_c // groups
            out_per_group = out_c // groups
            for g in range(groups):
                ic_start = g * input_per_group
                oc_start = g * out_per_group
                inp_g = input_nchw[n, ic_start:ic_start + input_per_group]
                wt_g = weight_nchw[oc_start:oc_start + out_per_group]
                # run as groups=1 conv
                sub_result = _conv2d_tile_fast(inp_g, wt_g, kh, kw, stride, 1, False)
                result[n, oc_start:oc_start + out_per_group] = sub_result
            continue

        # ---- spatial_oc_serial path (8-oc tiles, 32-row tiles) ----
        if spatial_oc_serial:
            oc_tile = 8
            for out_row_start in range(0, out_h, 32):
                tile_out_h = min(32, out_h - out_row_start)
                tile_in_h = (tile_out_h - 1) * stride + kh
                inp_slice = input_nchw[n, :, out_row_start * stride:out_row_start * stride + tile_in_h, :]
                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    wt_slice = weight_nchw[oc_start:oc_end].reshape(tile_out_c, in_c, kh, kw)
                    tile_result = _conv2d_tile_fast(inp_slice, wt_slice, kh, kw, stride, 1, False)
                    result[n, oc_start:oc_end, out_row_start:out_row_start + tile_out_h] = tile_result[:tile_out_c]
            continue

        # ---- depthwise_spatial_tiled path ----
        if depthwise_spatial_tiled:
            channel_tile = min(32, out_c)
            row_tile_h = _depthwise_tile_h(out_c, out_h, in_w, kh, kw, stride) if is_spatial else out_h
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                tile_in_h = (tile_out_h - 1) * stride + kh
                for ch_start in range(0, out_c, channel_tile):
                    ch_end = min(ch_start + channel_tile, out_c)
                    tile_c = ch_end - ch_start
                    inp_slice = input_nchw[n, ch_start:ch_end,
                                           out_row_start * stride:out_row_start * stride + tile_in_h, :]
                    wt_slice = np.zeros((tile_c, tile_c, kh, kw), dtype=np.float16)
                    for local_c in range(tile_c):
                        wt_slice[local_c, local_c] = weight_nchw[ch_start + local_c, 0]
                    tile_result = _conv2d_tile_fast(inp_slice, wt_slice, kh, kw, stride, tile_c, True)
                    result[n, ch_start:ch_end, out_row_start:out_row_start + tile_out_h] = tile_result[:, :tile_out_h]
            continue

        # ---- pointwise_oc_schedule path ----
        if use_pointwise_oc_schedule:
            oc_tile = _pointwise_oc_tile_c(in_c)
            row_tile_h = _pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile, stride)
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                inp_slice = input_nchw[n, :, out_row_start * stride:out_row_start * stride + tile_out_h, :]
                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    wt_slice = weight_nchw[oc_start:oc_end].reshape(tile_out_c, in_c, kh, kw)
                    tile_result = _conv2d_tile_fast(inp_slice, wt_slice, kh, kw, stride, 1, False)
                    result[n, oc_start:oc_end, out_row_start:out_row_start + tile_out_h] = tile_result[:tile_out_c]
            continue

        # ---- generic tile loop (same as conv.py lines 785-833) ----
        for tile in tiles:
            ys, y_span = tile["y_start"], tile["y_step"]
            ks, k_span = tile["k_start"], tile["k_step"]

            # extract input tile (same logic as _get_input_tile in conv.py)
            if is_depthwise:
                t_in = (y_span - 1) * stride + kh
                tn = min(t_in, in_h - ys * stride)
                inp_slice = np.zeros((k_span, tn, in_w), dtype=np.float16)
                rih = min(tn, in_h - ys * stride)
                inp_slice[:, :rih] = input_nchw[n, ks:ks + k_span, ys * stride:ys * stride + rih, :]
            else:
                t_in = (y_span - 1) * stride + kh
                hw_th = max(t_in, y_span, 7)
                tn = min(t_in, in_h - ys * stride)
                inp_slice = np.zeros((in_c, hw_th, in_w), dtype=np.float16)
                rih = min(tn, in_h - ys * stride)
                inp_slice[:, :rih] = input_nchw[n, :, ys * stride:ys * stride + rih, :]

            # extract weight tile (same logic as _get_weight_tile in conv.py)
            if is_depthwise:
                wt_slice = np.zeros((k_span, k_span, kh, kw), dtype=np.float16)
                for i in range(k_span):
                    wt_slice[i, i] = weight_nchw[ks + i, 0]
            else:
                wt_slice = weight_nchw[ks:ks + k_span].reshape(k_span, in_c // groups, kh, kw)

            tile_groups = k_span if is_depthwise else groups
            tile_result = _conv2d_tile_fast(inp_slice, wt_slice, kh, kw, stride, tile_groups, is_depthwise)

            # place result at correct position
            if is_depthwise:
                result[n, ks:ks + k_span, ys:ys + y_span] = tile_result[:, :y_span]
            else:
                result[n, ks:ks + k_span, ys:ys + y_span] = tile_result[:k_span, :y_span]

    return result, input_nchw, weight_nchw


def _descriptor_input_tile(input_nchw, n, row, p, in_c, in_h, in_w, kh, stride):
    y_start = row["y_start"]
    input_y = y_start * stride
    input_h = row["input_h"]
    if p["is_depthwise"]:
        k_start, oc_count = row["k_start"], row["oc_count"]
        tile = np.zeros((oc_count, input_h, in_w), dtype=np.float16)
        real_h = min(input_h, in_h - input_y)
        tile[:, :real_h] = input_nchw[n, k_start:k_start + oc_count, input_y:input_y + real_h, :]
        return tile

    tile = np.zeros((in_c, input_h, in_w), dtype=np.float16)
    real_h = min(input_h, in_h - input_y)
    tile[:, :real_h] = input_nchw[n, :, input_y:input_y + real_h, :]
    return tile


def _descriptor_weight_tile(weight_nchw, row, p, in_c, kh, kw, groups):
    k_start, oc_count = row["k_start"], row["oc_count"]
    if p["is_depthwise"]:
        tile = np.zeros((oc_count, oc_count, kh, kw), dtype=np.float16)
        for local_c in range(oc_count):
            tile[local_c, local_c] = weight_nchw[k_start + local_c, 0]
        return tile, oc_count

    return weight_nchw[k_start:k_start + oc_count].reshape(oc_count, in_c // groups, kh, kw), groups


def run_conv_generic_only(s):
    """Run CPU conv by consuming descriptor rows instead of old strategy branches."""
    stride = s.get("stride", 1)
    batch = s["batch"]
    in_c, in_h, in_w = s["in_c"], s["in_h"], s["in_w"]
    out_c, kh, kw, groups = s["out_c"], s["kh"], s["kw"], s["groups"]

    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, in_c // groups, kh, kw)).astype(np.float16)

    p, _split_method, _y_boundaries, _k_boundaries, rows = _descriptor_rows_for_shape(s)
    result = np.zeros((batch, out_c, p["out_h"], p["out_w"]), dtype=np.float16)

    for n in range(batch):
        for row in rows:
            input_tile = _descriptor_input_tile(input_nchw, n, row, p, in_c, in_h, in_w, kh, stride)
            weight_tile, tile_groups = _descriptor_weight_tile(weight_nchw, row, p, in_c, kh, kw, groups)
            tile_result = _conv2d_tile_fast(input_tile, weight_tile, kh, kw, stride,
                                           tile_groups, p["is_depthwise"])
            y_start = row["y_start"]
            output_h = row["output_h"]
            output_w = row["output_w"]
            k_start = row["k_start"]
            oc_count = row["oc_count"]
            result[n, k_start:k_start + oc_count, y_start:y_start + output_h, :output_w] = \
                tile_result[:oc_count, :output_h, :output_w]

    return result, input_nchw, weight_nchw, rows


def compute_reference(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups=1, stride=1):
    """Brute-force reference using float64 (same as conv.py compute_expected_nchw)."""
    out_h, out_w = (in_h - kh) // stride + 1, (in_w - kw) // stride + 1
    i64, w64 = inp.astype(np.float64), wt.astype(np.float64)
    expected = np.zeros((batch, out_c, out_h, out_w))
    for n in range(batch):
        for g in range(groups):
            for oc in range(g * out_c // groups, (g + 1) * out_c // groups):
                for ic in range(g * in_c // groups, (g + 1) * in_c // groups):
                    for i in range(kh):
                        for j in range(kw):
                            expected[n, oc] += i64[n, ic, i:i+stride*out_h:stride, j:j+stride*out_w:stride] * w64[oc, ic - g * in_c // groups, i, j]
    return expected


# ---- test shapes (same as conv.py) ----

SHAPES = [
    dict(name="conv2d_1x6_1x1_4x4",                batch=1, in_c=1,  in_h=4,  in_w=4,  out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
    dict(name="conv2d_3x3_1x1_4x4",                batch=1, in_c=3,  in_h=4,  in_w=4,  out_c=3, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="conv2d_4x2_1x1_4x4",                batch=1, in_c=4,  in_h=4,  in_w=4,  out_c=2, weight_in_c=4, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c4_h9_w9_oc4_wic4_k1x1_g1",  batch=1, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c3_h52_w52_oc6_wic3_k1x1_g1", batch=1, in_c=3,  in_h=52, in_w=52, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1", batch=1, in_c=96, in_h=56, in_w=56, out_c=24, weight_in_c=96, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1", batch=1, in_c=144, in_h=56, in_w=56, out_c=24, weight_in_c=144, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1", batch=1, in_c=144, in_h=28, in_w=28, out_c=32, weight_in_c=144, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1", batch=1, in_c=192, in_h=28, in_w=28, out_c=32, weight_in_c=192, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1", batch=1, in_c=192, in_h=28, in_w=28, out_c=16, weight_in_c=192, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1", batch=1, in_c=256, in_h=28, in_w=28, out_c=32, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="conv2d_16x16_1x1_8x8",              batch=1, in_c=16, in_h=8,  in_w=8,  out_c=16, weight_in_c=16, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c16_h32_w32_oc16_wic16_k1x1_g1", batch=1, in_c=16, in_h=32, in_w=32, out_c=16, weight_in_c=16, kh=1, kw=1, groups=1),

    dict(name="conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1",  batch=1, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=3, kw=3, groups=1),
    dict(name="conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1", batch=1, in_c=16, in_h=18, in_w=18, out_c=16, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="conv2d_b2_c4_h9_w9_oc4_wic4_k3x3_g1",  batch=2, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=3, kw=3, groups=1),
    dict(name="conv2d_b1_c1_h5_w7_oc6_wic1_k3x3_g1",  batch=1, in_c=1,  in_h=5,  in_w=7,  out_c=6, weight_in_c=1, kh=3, kw=3, groups=1),

    dict(name="conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3", batch=1, in_c=3, in_h=11, in_w=28, out_c=3, weight_in_c=1, kh=3, kw=3, groups=3),
    dict(name="conv2d_3x6_1x3_5x5", batch=1, in_c=3, in_h=5, in_w=5, out_c=6, weight_in_c=3, kh=1, kw=3, groups=1),

    dict(name="conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=3, groups=3),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x1_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=1, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x3_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=3, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x5_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=5, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x1_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=1, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x3_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=3, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x5_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=5, groups=1),

    dict(name="conv2d_1x6_2x1_5x7", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=2, kw=1, groups=1),
    dict(name="conv2d_1x6_2x3_5x7", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=2, kw=3, groups=1),
    dict(name="conv2d_1x6_3x1_5x7_b", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=1, groups=1),
    dict(name="conv2d_1x6_3x5_5x7",   batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=5, groups=1),

    dict(name="conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2",     batch=1,  in_c=4,  in_h=1,  in_w=1,  out_c=2,  weight_in_c=2,  kh=1, kw=1, groups=2),
    dict(name="conv2d_4x4_1x1_1x1_g2",                   batch=1,  in_c=4,  in_h=1,  in_w=1,  out_c=4,  weight_in_c=2,  kh=1, kw=1, groups=2),
    dict(name="conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32", batch=1, in_c=32, in_h=32, in_w=32, out_c=32, weight_in_c=1,  kh=1, kw=1, groups=32),
    dict(name="conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5",   batch=1,  in_c=15, in_h=5,  in_w=5,  out_c=35, weight_in_c=3,  kh=3, kw=3, groups=5),

    dict(name="conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3", batch=2, in_c=3,  in_h=11, in_w=28, out_c=3,  weight_in_c=1, kh=3, kw=3, groups=3),
    dict(name="conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5", batch=4, in_c=15, in_h=5,  in_w=5,  out_c=35, weight_in_c=3, kh=3, kw=3, groups=5),

    dict(name="conv2d_b1_c4_h5_w5_oc4_wic2_k3x3_g2",   batch=1, in_c=4,  in_h=5, in_w=5, out_c=4,  weight_in_c=2, kh=3, kw=3, groups=2),
    dict(name="conv2d_b1_c4_h5_w5_oc8_wic2_k3x3_g2",   batch=1, in_c=4,  in_h=5, in_w=5, out_c=8,  weight_in_c=2, kh=3, kw=3, groups=2),
    dict(name="conv2d_b1_c4_h5_w5_oc12_wic2_k3x3_g2",  batch=1, in_c=4,  in_h=5, in_w=5, out_c=12, weight_in_c=2, kh=3, kw=3, groups=2),
    dict(name="conv2d_b1_c6_h5_w5_oc6_wic2_k3x3_g3",   batch=1, in_c=6,  in_h=5, in_w=5, out_c=6,  weight_in_c=2, kh=3, kw=3, groups=3),
    dict(name="conv2d_b1_c6_h5_w5_oc12_wic2_k3x3_g3",  batch=1, in_c=6,  in_h=5, in_w=5, out_c=12, weight_in_c=2, kh=3, kw=3, groups=3),
    dict(name="conv2d_b1_c6_h5_w5_oc18_wic2_k3x3_g3",  batch=1, in_c=6,  in_h=5, in_w=5, out_c=18, weight_in_c=2, kh=3, kw=3, groups=3),
    dict(name="conv2d_b1_c15_h5_w5_oc20_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=20, weight_in_c=3, kh=3, kw=3, groups=5),
    dict(name="conv2d_b1_c15_h5_w5_oc25_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=25, weight_in_c=3, kh=3, kw=3, groups=5),
    dict(name="conv2d_b1_c15_h5_w5_oc30_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=30, weight_in_c=3, kh=3, kw=3, groups=5),
    dict(name="conv2d_b1_c15_h5_w5_oc40_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=40, weight_in_c=3, kh=3, kw=3, groups=5),
    dict(name="conv2d_2x2_1x1_4x4",  batch=1, in_c=2, in_h=4, in_w=4, out_c=2, weight_in_c=2, kh=1, kw=1, groups=1),
    dict(name="conv2d_8x8_1x1_5x5",       batch=1, in_c=8,  in_h=5,  in_w=5,  out_c=8,  weight_in_c=8,  kh=1, kw=1, groups=1),
    dict(name="conv2d_10x20_3x3_9x9",     batch=1, in_c=10, in_h=9,  in_w=9,  out_c=20, weight_in_c=10, kh=3, kw=3, groups=1),
    dict(name="conv2d_16x16_3x3_9x9",     batch=1, in_c=16, in_h=9,  in_w=9,  out_c=16, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="conv2d_2x4_3x3_6x6",       batch=1, in_c=2,  in_h=6,  in_w=6,  out_c=4,  weight_in_c=2,  kh=3, kw=3, groups=1),
    dict(name="conv2d_2x4_2x2_5x5",       batch=1, in_c=2,  in_h=5,  in_w=5,  out_c=4,  weight_in_c=2,  kh=2, kw=2, groups=1),
    dict(name="conv2d_1x32_5x5_10x10",    batch=1, in_c=1,  in_h=10, in_w=10, out_c=32, weight_in_c=1,  kh=5, kw=5, groups=1),
    dict(name="conv2d_8x4_4x4_10x10",     batch=1, in_c=8,  in_h=10, in_w=10, out_c=4,  weight_in_c=8,  kh=4, kw=4, groups=1),

    # MobileNet layers
    dict(name="conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1", batch=1, in_c=3, in_h=224, in_w=224, out_c=32, weight_in_c=3, kh=3, kw=3, groups=1),
    dict(name="conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32", batch=1, in_c=32, in_h=112, in_w=112, out_c=32, weight_in_c=1, kh=3, kw=3, groups=32),
    dict(name="conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1", batch=1, in_c=32, in_h=112, in_w=112, out_c=64, weight_in_c=32, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64", batch=1, in_c=64, in_h=112, in_w=112, out_c=64, weight_in_c=1, kh=3, kw=3, groups=64),
    dict(name="conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1", batch=1, in_c=64, in_h=56, in_w=56, out_c=128, weight_in_c=64, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128", batch=1, in_c=128, in_h=56, in_w=56, out_c=128, weight_in_c=1, kh=3, kw=3, groups=128),
    dict(name="conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1", batch=1, in_c=128, in_h=56, in_w=56, out_c=128, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1", batch=1, in_c=128, in_h=28, in_w=28, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256", batch=1, in_c=256, in_h=28, in_w=28, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
    dict(name="conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1", batch=1, in_c=256, in_h=28, in_w=28, out_c=256, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1", batch=1, in_c=256, in_h=14, in_w=14, out_c=512, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c512_h14_w14_oc512_wic1_k3x3_g512", batch=1, in_c=512, in_h=14, in_w=14, out_c=512, weight_in_c=1, kh=3, kw=3, groups=512),
    dict(name="conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1", batch=1, in_c=512, in_h=14, in_w=14, out_c=512, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1", batch=1, in_c=512, in_h=7, in_w=7, out_c=1024, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1, kh=3, kw=3, groups=1024),
    dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1024, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1, kh=7, kw=7, groups=1024),
    dict(name="conv2d_cc_b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=1, in_w=1, out_c=1001, weight_in_c=1024, kh=1, kw=1, groups=1),

    # conv1d
    dict(name="conv1d_bs1_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs8_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs1_612_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs1_615_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=1),
    dict(name="conv1d_bs1_1311_631_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs1_1311_632_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs1_1311_635_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=5, groups=1),
    dict(name="conv1d_bs1_1311_615_g3_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=3),
    dict(name="conv1d_bs8_8111_611_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs8_8111_612_a_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs8_8111_612_b_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs8_8111_615_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=1),
    dict(name="conv1d_bs8_8311_631_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs8_8311_632_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs8_8311_635_a_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=5, groups=1),
    dict(name="conv1d_bs8_8311_635_g3_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=3),

    # large spatial 1x1
    dict(name="1x3_54x54_k1",  batch=1, in_c=3, in_h=54, in_w=54, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_56x56_k1",  batch=1, in_c=3, in_h=56, in_w=56, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_58x58_k1",  batch=1, in_c=3, in_h=58, in_w=58, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_60x60_k1",  batch=1, in_c=3, in_h=60, in_w=60, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_62x62_k1",  batch=1, in_c=3, in_h=62, in_w=62, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_64x64_k1",  batch=1, in_c=3, in_h=64, in_w=64, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_66x66_k1",  batch=1, in_c=3, in_h=66, in_w=66, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_68x68_k1",  batch=1, in_c=3, in_h=68, in_w=68, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_70x70_k1",  batch=1, in_c=3, in_h=70, in_w=70, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_72x72_k1",  batch=1, in_c=3, in_h=72, in_w=72, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),

    # large channel 1x1
    dict(name="b1_c256_h14_w14_oc512_wic256_k1x1_g1",  batch=1, in_c=256, in_h=14,  in_w=14,  out_c=512, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c384_h14_w14_oc96_wic384_k1x1_g1",   batch=1, in_c=384, in_h=14,  in_w=14,  out_c=96,  weight_in_c=384, kh=1, kw=1, groups=1),
    dict(name="b1_c480_h14_w14_oc96_wic480_k1x1_g1",   batch=1, in_c=480, in_h=14,  in_w=14,  out_c=96,  weight_in_c=480, kh=1, kw=1, groups=1),
    dict(name="b1_c480_h14_w14_oc16_wic480_k1x1_g1",   batch=1, in_c=480, in_h=14,  in_w=14,  out_c=16,  weight_in_c=480, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h14_w14_oc112_wic512_k1x1_g1",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=112, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h14_w14_oc24_wic512_k1x1_g1",   batch=1, in_c=512, in_h=14,  in_w=14,  out_c=24,  weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h14_w14_oc32_wic512_k1x1_g1",   batch=1, in_c=512, in_h=14,  in_w=14,  out_c=32,  weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h14_w14_oc512_wic512_k1x1_g1",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=512, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h7_w7_oc1024_wic512_k1x1_g1",   batch=1, in_c=512, in_h=7,   in_w=7,   out_c=1024, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc256_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=256, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc160_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=160, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc32_wic528_k1x1_g1",   batch=1, in_c=528, in_h=14,  in_w=14,  out_c=32,  weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc128_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=128, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h14_w14_oc96_wic576_k1x1_g1",   batch=1, in_c=576, in_h=14,  in_w=14,  out_c=96,  weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c832_h7_w7_oc48_wic832_k1x1_g1",     batch=1, in_c=832, in_h=7,   in_w=7,   out_c=48,  weight_in_c=832, kh=1, kw=1, groups=1),
    dict(name="b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1024, kh=1, kw=1, groups=1),
    dict(name="b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=1,  in_w=1,   out_c=1001, weight_in_c=1024, kh=1, kw=1, groups=1),
    dict(name="b1_c1280_h10_w10_oc24_wic1280_k1x1_g1", batch=1, in_c=1280, in_h=10, in_w=10,  out_c=24,  weight_in_c=1280, kh=1, kw=1, groups=1),
    dict(name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1", batch=1, in_c=1280, in_h=10, in_w=10, out_c=546, weight_in_c=1280, kh=1, kw=1, groups=1),

    # depthwise spatial
    dict(name="b1_c32_h112_w112_oc32_wic1_k3x3_g32",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=32,  weight_in_c=1,  kh=3, kw=3, groups=32),
    dict(name="b1_c64_h112_w112_oc64_wic1_k3x3_g64",   batch=1, in_c=64,  in_h=112, in_w=112, out_c=64,  weight_in_c=1,  kh=3, kw=3, groups=64),
    dict(name="b1_c128_h56_w56_oc128_wic1_k3x3_g128",  batch=1, in_c=128, in_h=56,  in_w=56,  out_c=128, weight_in_c=1,  kh=3, kw=3, groups=128),
    dict(name="b1_c256_h28_w28_oc256_wic1_k3x3_g256",  batch=1, in_c=256, in_h=28,  in_w=28,  out_c=256, weight_in_c=1,  kh=3, kw=3, groups=256),
    dict(name="b1_c512_h14_w14_oc512_wic1_k3x3_g512",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=512, weight_in_c=1,  kh=3, kw=3, groups=512),
    dict(name="b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1,  kh=3, kw=3, groups=1024),
    dict(name="b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1,  kh=7, kw=7, groups=1024),

    # pointwise
    dict(name="b1_c32_h112_w112_oc16_wic32_k1x1_g1",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=16,  weight_in_c=32,  kh=1, kw=1, groups=1),
    dict(name="b1_c32_h112_w112_oc64_wic32_k1x1_g1",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=64,  weight_in_c=32,  kh=1, kw=1, groups=1),
    dict(name="b1_c64_h56_w56_oc128_wic64_k1x1_g1",    batch=1, in_c=64,  in_h=56,  in_w=56,  out_c=128, weight_in_c=64,  kh=1, kw=1, groups=1),
    dict(name="b1_c128_h56_w56_oc128_wic128_k1x1_g1",  batch=1, in_c=128, in_h=56,  in_w=56,  out_c=128, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c128_h28_w28_oc256_wic128_k1x1_g1",  batch=1, in_c=128, in_h=28,  in_w=28,  out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h28_w28_oc256_wic256_k1x1_g1",  batch=1, in_c=256, in_h=28,  in_w=28,  out_c=256, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c3_h224_w224_oc32_wic3_k3x3_g1",     batch=1, in_c=3,   in_h=224, in_w=224, out_c=32,  weight_in_c=3,   kh=3, kw=3, groups=1),

    # mobilenet-like _s1_pvalid
    dict(name="b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=112, in_w=112, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
    dict(name="b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid", batch=1, in_c=144, in_h=56, in_w=56, out_c=144, weight_in_c=1, kh=3, kw=3, groups=144),
    dict(name="b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid", batch=1, in_c=192, in_h=28, in_w=28, out_c=96, weight_in_c=192, kh=1, kw=1, groups=1),
    dict(name="b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=14, in_w=14, out_c=64, weight_in_c=32, kh=3, kw=3, groups=1),
    dict(name="b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=256, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=160, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid", batch=1, in_c=160, in_h=14, in_w=14, out_c=320, weight_in_c=160, kh=3, kw=3, groups=1),
    dict(name="b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=32, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=14, in_w=14, out_c=128, weight_in_c=32, kh=3, kw=3, groups=1),
    dict(name="b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=128, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid", batch=1, in_c=160, in_h=7, in_w=7, out_c=320, weight_in_c=160, kh=3, kw=3, groups=1),
    dict(name="b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=7, in_w=7, out_c=128, weight_in_c=32, kh=3, kw=3, groups=1),
    dict(name="b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid", batch=1, in_c=192, in_h=7, in_w=7, out_c=384, weight_in_c=192, kh=3, kw=3, groups=1),
    dict(name="b1_c832_h7_w7_oc48_wic832_k1x1_g1_s1_pvalid", batch=1, in_c=832, in_h=7, in_w=7, out_c=48, weight_in_c=832, kh=1, kw=1, groups=1),
    dict(name="b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid", batch=1, in_c=32, in_h=150, in_w=150, out_c=32, weight_in_c=1, kh=3, kw=3, groups=32),
    dict(name="b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid", batch=1, in_c=32, in_h=150, in_w=150, out_c=16, weight_in_c=32, kh=1, kw=1, groups=1),
    dict(name="b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid", batch=1, in_c=16, in_h=150, in_w=150, out_c=96, weight_in_c=16, kh=1, kw=1, groups=1),
    dict(name="b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=150, in_w=150, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
    dict(name="b1_c96_h75_w75_oc24_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=75, in_w=75, out_c=24, weight_in_c=96, kh=1, kw=1, groups=1),
    dict(name="b1_c144_h75_w75_oc144_wic1_k3x3_g144_s1_pvalid", batch=1, in_c=144, in_h=75, in_w=75, out_c=144, weight_in_c=1, kh=3, kw=3, groups=144),
    dict(name="b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid", batch=1, in_c=144, in_h=75, in_w=75, out_c=24, weight_in_c=144, kh=1, kw=1, groups=1),
    dict(name="b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid", batch=1, in_c=144, in_h=38, in_w=38, out_c=32, weight_in_c=144, kh=1, kw=1, groups=1),
    dict(name="b1_c192_h38_w38_oc192_wic1_k3x3_g192_s1_pvalid", batch=1, in_c=192, in_h=38, in_w=38, out_c=192, weight_in_c=1, kh=3, kw=3, groups=192),
    dict(name="b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid", batch=1, in_c=192, in_h=38, in_w=38, out_c=32, weight_in_c=192, kh=1, kw=1, groups=1),
    dict(name="b1_c384_h19_w19_oc384_wic1_k3x3_g384_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=384, weight_in_c=1, kh=3, kw=3, groups=384),
    dict(name="b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=64, weight_in_c=384, kh=1, kw=1, groups=1),
    dict(name="b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=96, weight_in_c=384, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h19_w19_oc576_wic1_k3x3_g576_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=576, weight_in_c=1, kh=3, kw=3, groups=576),
    dict(name="b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=96, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=12, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=273, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c960_h10_w10_oc960_wic1_k3x3_g960_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=960, weight_in_c=1, kh=3, kw=3, groups=960),
    dict(name="b1_c1280_h10_w10_oc24_wic1280_k1x1_g1_s1_pvalid", batch=1, in_c=1280, in_h=10, in_w=10, out_c=24, weight_in_c=1280, kh=1, kw=1, groups=1),
    dict(name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid", batch=1, in_c=1280, in_h=10, in_w=10, out_c=546, weight_in_c=1280, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid", batch=1, in_c=256, in_h=10, in_w=10, out_c=512, weight_in_c=256, kh=3, kw=3, groups=1),
    dict(name="b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid", batch=1, in_c=128, in_h=5, in_w=5, out_c=256, weight_in_c=128, kh=3, kw=3, groups=1),
    dict(name="b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=24, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h3_w3_oc546_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=546, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=128, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=256, weight_in_c=128, kh=3, kw=3, groups=1),
    dict(name="b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=24, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=546, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=64, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=1, in_w=1, out_c=24, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid", batch=1, in_c=3, in_h=320, in_w=320, out_c=32, weight_in_c=3, kh=3, kw=3, groups=1),
    dict(name="b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid", batch=1, in_c=32, in_h=160, in_w=160, out_c=8, weight_in_c=32, kh=1, kw=1, groups=1),
    dict(name="b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid", batch=1, in_c=8, in_h=160, in_w=160, out_c=16, weight_in_c=8, kh=3, kw=3, groups=1),
    dict(name="b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=160, in_w=160, out_c=128, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=80, in_w=80, out_c=16, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=64, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid", batch=1, in_c=64, in_h=80, in_w=80, out_c=16, weight_in_c=64, kh=1, kw=1, groups=1),
    dict(name="b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=128, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=128, weight_in_c=16, kh=5, kw=5, groups=1),
    dict(name="b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=40, in_w=40, out_c=40, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid", batch=1, in_c=40, in_h=40, in_w=40, out_c=160, weight_in_c=40, kh=3, kw=3, groups=1),
    dict(name="b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid", batch=1, in_c=160, in_h=40, in_w=40, out_c=40, weight_in_c=160, kh=1, kw=1, groups=1),
    dict(name="b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid", batch=1, in_c=40, in_h=40, in_w=40, out_c=320, weight_in_c=40, kh=1, kw=1, groups=1),
    dict(name="b1_c320_h40_w40_oc320_wic1_k3x3_g320_s1_pvalid", batch=1, in_c=320, in_h=40, in_w=40, out_c=320, weight_in_c=1, kh=3, kw=3, groups=320),
    dict(name="b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid", batch=1, in_c=320, in_h=20, in_w=20, out_c=72, weight_in_c=320, kh=1, kw=1, groups=1),
    dict(name="b1_c72_h20_w20_oc576_wic72_k1x1_g1_s1_pvalid", batch=1, in_c=72, in_h=20, in_w=20, out_c=576, weight_in_c=72, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h20_w20_oc576_wic1_k3x3_g576_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=576, weight_in_c=1, kh=3, kw=3, groups=576),
    dict(name="b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=72, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid", batch=1, in_c=72, in_h=20, in_w=20, out_c=288, weight_in_c=72, kh=3, kw=3, groups=1),
    dict(name="b1_c288_h20_w20_oc72_wic288_k1x1_g1_s1_pvalid", batch=1, in_c=288, in_h=20, in_w=20, out_c=72, weight_in_c=288, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h20_w20_oc576_wic1_k5x5_g576_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=576, weight_in_c=1, kh=5, kw=5, groups=576),
    dict(name="b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=96, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=768, weight_in_c=1, kh=5, kw=5, groups=768),
    dict(name="b1_c768_h20_w20_oc96_wic768_k1x1_g1_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=96, weight_in_c=768, kh=1, kw=1, groups=1),
    dict(name="b1_c768_h20_w20_oc768_wic1_k3x3_g768_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=768, weight_in_c=1, kh=3, kw=3, groups=768),
    dict(name="b1_c768_h10_w10_oc120_wic768_k1x1_g1_s1_pvalid", batch=1, in_c=768, in_h=10, in_w=10, out_c=120, weight_in_c=768, kh=1, kw=1, groups=1),
    dict(name="b1_c960_h10_w10_oc120_wic960_k1x1_g1_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=120, weight_in_c=960, kh=1, kw=1, groups=1),
    dict(name="b1_c480_h10_w10_oc480_wic1_k5x5_g480_s1_pvalid", batch=1, in_c=480, in_h=10, in_w=10, out_c=480, weight_in_c=1, kh=5, kw=5, groups=480),
    dict(name="b1_c480_h10_w10_oc120_wic480_k1x1_g1_s1_pvalid", batch=1, in_c=480, in_h=10, in_w=10, out_c=120, weight_in_c=480, kh=1, kw=1, groups=1),
    dict(name="b1_c960_h10_w10_oc960_wic1_k5x5_g960_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=960, weight_in_c=1, kh=5, kw=5, groups=960),
    dict(name="b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid", batch=1, in_c=256, in_h=10, in_w=10, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
    dict(name="b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c128_h3_w3_oc128_wic1_k3x3_g128_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=128, weight_in_c=1, kh=3, kw=3, groups=128),
    dict(name="b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=2, in_w=2, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", batch=1, in_c=64, in_h=1, in_w=1, out_c=128, weight_in_c=64, kh=1, kw=1, groups=1),
    dict(name="b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
    dict(name="b1_c384_h10_w10_oc384_wic1_k3x3_g384_s1_pvalid", batch=1, in_c=384, in_h=10, in_w=10, out_c=384, weight_in_c=1, kh=3, kw=3, groups=384),
    dict(name="b1_c512_h5_w5_oc512_wic1_k3x3_g512_s1_pvalid", batch=1, in_c=512, in_h=5, in_w=5, out_c=512, weight_in_c=1, kh=3, kw=3, groups=512),
    dict(name="b1_c256_h3_w3_oc256_wic1_k3x3_g256_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
    dict(name="b1_c96_h20_w20_oc12_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=12, weight_in_c=96, kh=1, kw=1, groups=1),
    dict(name="b1_c96_h20_w20_oc273_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=273, weight_in_c=96, kh=1, kw=1, groups=1),
    dict(name="b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=10, in_w=10, out_c=546, weight_in_c=384, kh=1, kw=1, groups=1),
]


def run_all_shape_tests():
    nw = max(len(s["name"]) for s in SHAPES)
    iw = max(len(f"{s['in_c']}x{s['in_h']}x{s['in_w']}") for s in SHAPES)
    n_pass = 0
    n_fail = 0
    for s in SHAPES:
        stride = s.get("stride", 1)
        oh = (s["in_h"] - s["kh"]) // stride + 1
        ow = (s["in_w"] - s["kw"]) // stride + 1

        r, inp, wt = run_conv_tiled(s["batch"], s["in_c"], s["out_c"], s["kh"], s["kw"],
                                     (s["in_h"], s["in_w"]), groups=s["groups"], stride=stride)
        e = compute_reference(inp, wt, s["batch"], s["in_c"], s["in_h"], s["in_w"],
                              s["out_c"], s["kh"], s["kw"], groups=s["groups"], stride=stride)
        md = float(np.max(np.abs(r.astype(np.float64) - e)))
        ok = np.allclose(r, e, atol=0.2) and not np.any(np.isinf(r))
        in_str = f"{s['in_c']}x{s['in_h']}x{s['in_w']}"
        out_str = f"{s['out_c']}x{oh}x{ow}"
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  {s['name']:<{nw}s} {in_str:<{iw}s} -> {out_str}  {status}  (max_diff={md:.4f})")
    print(f"\n  {n_pass} PASS, {n_fail} FAIL out of {n_pass + n_fail} shapes")


def run_generic_only_tests():
    nw = max(len(s["name"]) for s in SHAPES)
    iw = max(len(f"{s['in_c']}x{s['in_h']}x{s['in_w']}") for s in SHAPES)
    n_pass = 0
    n_fail = 0
    failures = []
    for s in SHAPES:
        stride = s.get("stride", 1)
        oh = (s["in_h"] - s["kh"]) // stride + 1
        ow = (s["in_w"] - s["kw"]) // stride + 1

        r, inp, wt, rows = run_conv_generic_only(s)
        e = compute_reference(inp, wt, s["batch"], s["in_c"], s["in_h"], s["in_w"],
                              s["out_c"], s["kh"], s["kw"], groups=s["groups"], stride=stride)
        md = float(np.max(np.abs(r.astype(np.float64) - e)))
        ok = np.allclose(r, e, atol=0.2) and not np.any(np.isinf(r))
        in_str = f"{s['in_c']}x{s['in_h']}x{s['in_w']}"
        out_str = f"{s['out_c']}x{oh}x{ow}"
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            families = sorted({row["family"] for row in rows})
            failures.append({
                "name": s["name"],
                "old_strategy": _old_strategy_name(s),
                "split_method": rows[0]["split_method"] if rows else "unknown",
                "descriptor_families": families,
                "max_diff": f"{md:.4f}",
            })
        print(f"  {s['name']:<{nw}s} {in_str:<{iw}s} -> {out_str}  {status}  (max_diff={md:.4f})")

    print(f"\n  {n_pass} PASS, {n_fail} FAIL out of {n_pass + n_fail} shapes")
    if failures:
        print("\nFailure groups:")
        grouped = Counter((row["old_strategy"], row["split_method"], tuple(row["descriptor_families"]))
                          for row in failures)
        group_rows = []
        for (old_strategy, split_method, families), count in sorted(grouped.items()):
            group_rows.append({
                "old_strategy": old_strategy,
                "split_method": split_method,
                "descriptor_families": list(families),
                "count": count,
            })
        _print_table(group_rows, ["old_strategy", "split_method", "descriptor_families", "count"])
        print("\nFailures:")
        _print_table(failures, ["name", "old_strategy", "split_method", "descriptor_families", "max_diff"])


def main():
    parser = argparse.ArgumentParser(description="Offline RK3588 CONV tiling proof harness")
    parser.add_argument("--planner-report", action="store_true",
                        help="print one planner summary row per shape")
    parser.add_argument("--descriptor-dump", metavar="SHAPE", nargs="?", const="",
                        help="dump descriptor rows for one shape, or all shapes if no shape is supplied")
    parser.add_argument("--cross-tab", action="store_true",
                        help="summarize old branches by new split method and descriptor families")
    parser.add_argument("--cbuf-compare", action="store_true",
                        help="compare RK3588 and NVDLA nv_full CBUF budgets over all shapes")
    parser.add_argument("--cbuf-compare-all", action="store_true",
                        help="include unchanged rows in --cbuf-compare details")
    parser.add_argument("--generic-only", action="store_true",
                        help="execute all shapes from descriptor rows without old strategy branches")
    parser.add_argument("--evidence-check", action="store_true",
                        help="compare descriptor rows against known RKNN export observations")
    args = parser.parse_args()

    if args.planner_report:
        print_planner_report()
    if args.descriptor_dump is not None:
        print_descriptor_dump(args.descriptor_dump or None)
    if args.cross_tab:
        print_cross_tab()
    if args.cbuf_compare or args.cbuf_compare_all:
        print_cbuf_compare(all_rows=args.cbuf_compare_all)
    if args.generic_only:
        run_generic_only_tests()
    if args.evidence_check:
        print_evidence_check()
    if (not args.planner_report and args.descriptor_dump is None and not args.cross_tab and
            not args.cbuf_compare and not args.cbuf_compare_all and not args.generic_only and
            not args.evidence_check):
        run_all_shape_tests()


if __name__ == "__main__":
    main()
