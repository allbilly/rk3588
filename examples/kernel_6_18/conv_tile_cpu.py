"""
conv_cpu.py — Pure-numpy tiled conv using the same tiling strategy as conv.py.
Proves the reversed RKNN bank-pressure-based Y/K tiling is mathematically correct.

No NPU hardware needed. Uses the same _plan_conv_tiles / _compute_k_step /
_compute_y_step functions, then computes each tile on CPU with numpy and
assembles the result.
"""
import numpy as np

# ---- constants (same as conv.py) ----
FP16_BYTES = 2
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
RK_MAX_CONV_FLAT_STRIDE = 992

# ---- helpers (same as conv.py) ----

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

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


if __name__ == "__main__":
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
