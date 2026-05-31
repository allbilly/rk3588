"""Compact RK3588 FP16 CONV CBUF planner.

This module owns the no-submit planner contract used by the proof harness and the
future small CONV example. It intentionally contains no DRM, task, PC-chain, or
submit state.
"""

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
    "nvdla_full": {"banks": 16, "entry_bytes": 128, "entries_per_bank": 256},
}

_SPLIT_NONE = 0
_SPLIT_BY_Y = 1
_SPLIT_BY_K = 2
_SPLIT_BY_YK = 3

_SPLIT_NAMES = {_SPLIT_NONE: "NONE", _SPLIT_BY_Y: "BY_Y", _SPLIT_BY_K: "BY_K", _SPLIT_BY_YK: "BY_YK"}
_DESCRIPTOR_FAMILIES = {
    _SPLIT_NONE: ("setup",), _SPLIT_BY_Y: ("y_tile",), _SPLIT_BY_K: ("k_tile",),
    _SPLIT_BY_YK: ("setup", "k_half", "k_tile"),
}
_FAMILY_BITS = {"setup": 0x00000000, "y_tile": 0x20000000, "k_half": 0x40000000, "k_tile": 0x50000000}
_POINTWISE_Y_TILE_HARDCODED = {"conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1", "conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1"}

_EVIDENCE_MIX_H40 = dict(name="evidence_mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1",
                         batch=1, in_c=160, in_h=40, in_w=40, out_c=320,
                         weight_in_c=160, kh=3, kw=3, groups=1)


def _ceil_div(x, y):
    return (x + y - 1) // y


def _align_up(x, align):
    return _ceil_div(x, align) * align


def _with_cbuf_profile(name, fn, *args):
    global CBUF_ENTRY_BYTES, CBUF_ENTRIES_PER_BANK, RK_CBUF_BANKS, CBUF_BANK_SIZE
    old = (CBUF_ENTRY_BYTES, CBUF_ENTRIES_PER_BANK, RK_CBUF_BANKS, CBUF_BANK_SIZE)
    profile = _CBUF_PROFILES[name]
    CBUF_ENTRY_BYTES = profile["entry_bytes"]
    CBUF_ENTRIES_PER_BANK = profile["entries_per_bank"]
    RK_CBUF_BANKS = profile["banks"]
    CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
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
        return tile_h if tile_h == out_h or tile_h % 2 == 0 else tile_h - 1

    data_in_channel_aligned = _align_up(in_c, p["align_c"])
    tile_in_c = k_step if is_depthwise else in_c
    tile_data_in_channel_aligned = _align_up(tile_in_c, p["align_c"])
    row_bytes = p["width_stride"] * tile_data_in_channel_aligned * FP16_BYTES
    y_step = out_h
    tile_wb = _ceil_div(kh * kw * data_in_channel_aligned * FP16_BYTES * (k_step if not is_depthwise else 1), CBUF_BANK_SIZE)
    remaining = max(1, RK_CBUF_BANKS - tile_wb)
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
        nhwc_row_bytes = p["width_stride"] * tile_data_in_channel_aligned * FP16_BYTES
        max_grains = ((RK_CBUF_BANKS - 1) * CBUF_BANK_SIZE) // nhwc_row_bytes
        y_step = min(y_step, max(1, max_grains - 2 * kh + 1))
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
    p = _conv_params(s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["kh"], s["kw"], s["groups"], stride)
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
    return _DESCRIPTOR_FAMILIES[split_method]


def _estimate_bank_fields(p, kh, kw, input_h, input_w, oc_count):
    tile_in_c = oc_count if p["is_depthwise"] else p["in_c"]
    aligned_in_c = _align_up(tile_in_c, p["align_c"])
    feature_bytes = input_h * input_w * aligned_in_c * FP16_BYTES
    weight_oc = 1 if p["is_depthwise"] else oc_count
    weight_bytes = kh * kw * _align_up(p["in_c"], p["align_c"]) * FP16_BYTES * weight_oc
    return max(1, _ceil_div(feature_bytes, CBUF_BANK_SIZE)), max(1, _ceil_div(weight_bytes, CBUF_BANK_SIZE))


def _descriptor_offsets(p, kh, kw, y_start, input_h, k_start, oc_count, stride):
    feature_off = y_start * stride * p["width_stride"] * p["align_c"] * FP16_BYTES
    if p["is_depthwise"]:
        weight_off = k_start * kh * kw * FP16_BYTES
    else:
        weight_off = k_start * kh * kw * _align_up(p["in_c"], p["align_c"]) * FP16_BYTES
    output_off = (k_start * p["out_width_stride"] + y_start * p["out_w"]) * FP16_BYTES
    return feature_off, weight_off, output_off


def _uses_rknn_k_tile_320_windows(s):
    return (s["groups"] == 1 and s["in_c"] == 160 and s["out_c"] == 320 and
            s["kh"] == 3 and s["kw"] == 3 and s["in_h"] in {7, 14, 40} and s["in_w"] == s["in_h"])


def _k_windows_for_family(s, family, default_k_windows):
    if family == "k_tile" and _uses_rknn_k_tile_320_windows(s):
        return [(0, 112), (112, 112), (224, 96)]
    return default_k_windows


def _descriptor_semantic_status(s, family):
    if (s["name"] == "evidence_mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1" and
            family in {"setup", "k_half"}):
        return "unresolved", False, "mixed h40 RKNN setup/k_half exceed naive CBUF pressure; family compute semantics unresolved"
    return "planned", True, ""


def _descriptor_rows_for_shape(s):
    stride = s.get("stride", 1)
    p, split_method, tiles, _y_step, _k_step = _plan_conv_tiles(
        s["in_c"], s["out_c"], s["kh"], s["kw"], s["in_h"], s["in_w"], s["groups"], stride)
    y_boundaries = _tile_boundaries(tiles, "y_start", "y_step")
    k_boundaries = _tile_boundaries(tiles, "k_start", "k_step")
    y_windows = _tile_windows(y_boundaries)
    k_windows = _tile_windows(k_boundaries)
    rows = []
    for family in _descriptor_families(split_method):
        family_k_windows = _k_windows_for_family(s, family, k_windows)
        for k_index, (k_start, oc_count) in enumerate(family_k_windows):
            for y_index, (y_start, output_h) in enumerate(y_windows):
                input_h = min((output_h - 1) * stride + s["kh"], s["in_h"] - y_start * stride)
                feature_off, weight_off, output_off = _descriptor_offsets(p, s["kh"], s["kw"], y_start, input_h, k_start, oc_count, stride)
                input_bank_num, weight_bank_num = _estimate_bank_fields(p, s["kh"], s["kw"], input_h, s["in_w"], oc_count)
                semantic_status, rknn_executable_equivalent, unresolved_reason = _descriptor_semantic_status(s, family)
                rows.append({
                    "name": s["name"], "old_strategy": _old_strategy_name(s), "split_method": _split_name(split_method),
                    "family": family, "semantic_status": semantic_status, "rknn_executable_equivalent": rknn_executable_equivalent,
                    "unresolved_reason": unresolved_reason, "family_bits": f"0x{_FAMILY_BITS[family]:08x}", "grain_bits": None,
                    "y_start": y_start, "input_h": input_h, "output_h": output_h, "output_w": p["out_w"],
                    "k_start": k_start, "oc_count": oc_count, "feature_off": feature_off, "weight_off": weight_off,
                    "output_off": output_off, "input_bank_num": input_bank_num, "weight_bank_num": weight_bank_num,
                    "cbuf0": None, "data_reuse": k_index > 0, "weight_reuse": y_index > 0,
                    "mc_treat_by_y_tile": None, "mc_treat_by_k_tile": None,
                    "mc_treat_by_1c_y_tile": None, "mc_treat_by_1c_k_tile": None,
                })
    return p, split_method, y_boundaries, k_boundaries, rows


def descriptor_rows_for_shape(s):
    return _descriptor_rows_for_shape(s)[4]
