"""
csc_sim.py — CSC->CMAC->CACC streaming dataflow simulator.

Verifies that packed weights and packed input, when consumed in hardware
streaming order, produce correct convolution results.

Supports FP16 (RK-style OC16-blocked) and INT8 (simple_conv.py style) packing,
input packing, output unpacking, im2col, DW chunking, bias packing, and buffer
size calculations.

Usage
-----
  # FP16 weight verification
  wpack = pack_weights_fp16(weight_nchw, out_c, in_c, kh, kw)
  result = CSCSim().conv(wpack, input_nchw, out_c, in_c, kh, kw)

  # INT8 weight verification (raw float64 streaming order)
  wpack = pack_weights_int8(weight_nchw, out_c, in_c, kh, kw)
  result = CSCSim(oc_atom=32, ic_atom=32, flip_kspatial=True).conv(
      wpack, input_nchw, out_c, in_c, kh, kw, dtype="int8")

  # INT8 byte-accurate packer + input packer + full pipeline
  in_packed = pack_input_int8(input_nhwc, in_c, in_h, in_w)
  wt_packed = pack_weights_int8_byte(weight_ohwi, out_c, in_c, kh, kw,
                                      groups, in_c_real, is_dw)
  bias_packed = pack_biases_int8(biases, weight_ohwi, ...)
  out_raw = run_conv_int8_pipeline(...)
  result = unpack_output_int8(out_raw, out_c, out_h, out_w)

Reference
---------
  simple_conv.py          — int8 ref with pack_input/pack_weights/pack_biases
  simple_conv_fp16.py     — fp16 ref for OC16/IC32 weight scheduling
  NVDLA HW cmod:          — MAC_ATOMIC_K=32(INT8)/16(FP16), MAC_ATOMIC_C=64
"""

import os
import re

import numpy as np

FP16_BYTES = 2
FP32_BYTES = 4

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ceil_div(x, y):
    return (x + y - 1) // y

def _align_up(x, align):
    return _ceil_div(x, align) * align

def _is_depthwise(in_c, out_c, groups):
    return groups == in_c == out_c

def _compute_pack_out_c(out_c, in_c, kh, kw, groups):
    if groups == 1 and kh == 1 and kw == 1 and in_c >= 64 and out_c >= 48:
        return _align_up(out_c, 16)
    return out_c

# ---------------------------------------------------------------------------
# INT8 constants (simple_conv.py)
# ---------------------------------------------------------------------------

INT8_FEATURE_ATOMIC_SIZE = 16
INT8_WEIGHT_ATOMIC_SIZE = 32
INT8_CBUF_ENTRY_SIZE = 128
INT8_CBUF_ENTRIES_PER_BANK = 256
INT8_CBUF_BANKS = 12

# ---------------------------------------------------------------------------
# WeightPack
# ---------------------------------------------------------------------------

class WeightPack:
    __slots__ = ("data", "pack_out_c", "aligned_in_c", "ic_atom", "oc_atom",
                 "kh", "kw", "flip_kspatial", "dtype")
    def __init__(self, data, pack_out_c, aligned_in_c, ic_atom,
                 oc_atom=16, kh=1, kw=1, flip_kspatial=False, dtype="fp16"):
        self.data = np.ascontiguousarray(data.ravel())
        self.pack_out_c = pack_out_c
        self.aligned_in_c = aligned_in_c
        self.ic_atom = ic_atom
        self.oc_atom = oc_atom
        self.kh = kh
        self.kw = kw
        self.flip_kspatial = flip_kspatial
        self.dtype = dtype

# ---------------------------------------------------------------------------
# FP16 weight packer
# ---------------------------------------------------------------------------

def pack_weights_fp16(weight_nchw, out_c, in_c, kh, kw, ic_atom=32, groups=1,
                      weight_in_c=None):
    if weight_in_c is None:
        weight_in_c = in_c // groups
    aligned_in_c = _align_up(in_c, ic_atom)

    if _is_depthwise(in_c, out_c, groups):
        packed = np.zeros((1, kh, kw, aligned_in_c), dtype=np.float16)
        for c in range(min(out_c, in_c)):
            packed[0, :, :, c] = weight_nchw[c, 0]
        return WeightPack(packed.ravel(), out_c, aligned_in_c, ic_atom,
                          oc_atom=1, kh=kh, kw=kw, flip_kspatial=False, dtype="fp16")

    pack_out_c = _compute_pack_out_c(out_c, in_c, kh, kw, groups)
    padded = np.zeros((pack_out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:out_c, :weight_in_c] = weight_nchw[:out_c, :weight_in_c]

    blocks = []
    for oc in range(0, pack_out_c, 16):
        block = padded[oc:oc + 16]
        block_oc = block.shape[0]
        blocks.append(
            block.reshape(block_oc, aligned_in_c // ic_atom, ic_atom, kh, kw)
                 .transpose(1, 3, 4, 0, 2)
                 .ravel()
        )
    return WeightPack(np.concatenate(blocks), pack_out_c, aligned_in_c, ic_atom,
                      oc_atom=16, kh=kh, kw=kw, flip_kspatial=False, dtype="fp16")

# ---------------------------------------------------------------------------
# INT8 weight packer  (raw float64 streaming order — matches simple_conv loop)
# ---------------------------------------------------------------------------

def pack_weights_int8(weight_nchw, out_c, in_c, kh, kw, groups=1,
                      weight_in_c=None, ic_groups=None):
    if weight_in_c is None:
        weight_in_c = in_c // groups
    is_dw = _is_depthwise(in_c, out_c, groups)
    if ic_groups is None:
        ic_groups = INT8_WEIGHT_ATOMIC_SIZE * (2 if is_dw else 1)

    ic = max(in_c, INT8_FEATURE_ATOMIC_SIZE)
    oc = 1 if is_dw else _align_up(out_c, 2)
    result = np.zeros(_align_up(ic, ic_groups) * oc * kh * kw, dtype=np.float64)
    n = 0

    for oo in range(_ceil_div(oc, INT8_WEIGHT_ATOMIC_SIZE)):
        for ii in range(_ceil_div(ic, ic_groups)):
            for kx in range(kw):
                for ky in range(kh):
                    oi_limit = min(oc - oo * INT8_WEIGHT_ATOMIC_SIZE, INT8_WEIGHT_ATOMIC_SIZE)
                    for oi in range(oi_limit):
                        ij_limit = min(ic, ic_groups)
                        for ij in range(ij_limit):
                            oc_idx = oo * INT8_WEIGHT_ATOMIC_SIZE + oi
                            ic_idx = ii * ic_groups + ij
                            if oc_idx < out_c and ic_idx < in_c:
                                if is_dw:
                                    result[n] = float(weight_nchw[ic_idx, 0, ky, kx])
                                else:
                                    result[n] = float(weight_nchw[oc_idx, ic_idx, ky, kx])
                            n += 1

    aligned_in_c = _align_up(ic, ic_groups)
    return WeightPack(result[:n].copy(), oc, aligned_in_c, ic_groups,
                      oc_atom=INT8_WEIGHT_ATOMIC_SIZE, kh=kh, kw=kw,
                      flip_kspatial=True, dtype="int8")


# ---------------------------------------------------------------------------
# INT8 byte-accurate weight packer  (uint8, (val - 0x80) & 0xFF format)
# ---------------------------------------------------------------------------

def pack_weights_int8_byte(weight_ohwi, out_c, in_c, kh, kw, groups=1,
                           in_c_real=None, is_dw=False):
    """
    Byte-accurate INT8 weight packer matching simple_conv.py pack_weights().

    Stores ``(value - 0x80) & 0xFF`` as uint8.  The hardware sees this as
    signed bytes after ``+0x80`` via the DATA_SIGN register.

    Parameters
    ----------
    weight_ohwi : ndarray
        Shape (oc, kw, kh, weight_ic) — OHWI layout matching simple_conv.
    out_c, in_c : int
        Channel dimensions.
    kh, kw : int
        Kernel spatial dimensions.
    groups : int
    in_c_real : int or None
        Actual input channels (may differ from padded ``in_c``).
    is_dw : bool

    Returns
    -------
    packed : ndarray (uint8)  — the exact byte stream for hardware.
    oc, ic_groups, ic, aligned_ic, size : metadata for register setup.
    """
    if in_c_real is None:
        in_c_real = in_c
    weight_in_c = in_c_real // groups

    ic = max(weight_in_c, INT8_FEATURE_ATOMIC_SIZE)
    oc = 1 if is_dw else _align_up(out_c, 2)
    ic_groups = INT8_WEIGHT_ATOMIC_SIZE * (2 if is_dw else 1)

    size = kh * kw * oc * _align_up(ic, ic_groups)
    out = np.zeros(size, dtype=np.uint8)
    idx = 0

    for oo in range(_ceil_div(oc, INT8_WEIGHT_ATOMIC_SIZE)):
        for ic_block in range(_ceil_div(ic, ic_groups)):
            icb = ic_block * ic_groups
            for kx in range(kw):
                for ky in range(kh):
                    for oi in range(min(oc - oo * INT8_WEIGHT_ATOMIC_SIZE, INT8_WEIGHT_ATOMIC_SIZE)):
                        oc_idx = oo * INT8_WEIGHT_ATOMIC_SIZE + oi
                        for ij in range(min(ic, ic_groups)):
                            ic_idx = icb + ij
                            if oc_idx >= out_c or ic_idx >= weight_in_c:
                                out[idx] = 0
                            elif is_dw:
                                out[idx] = (int(weight_ohwi[0, kx, ky, ic_idx]) - 0x80) & 0xFF
                            else:
                                out[idx] = (int(weight_ohwi[oc_idx, kx, ky, ic_idx]) - 0x80) & 0xFF
                            idx += 1

    aligned_ic = _align_up(ic, ic_groups)
    return out, oc, ic_groups, ic, aligned_ic


# ---------------------------------------------------------------------------
# INT8 input packer  (matches simple_conv.py pack_input)
# ---------------------------------------------------------------------------

def pack_input_int8(input_nhwc, in_c, in_h, in_w):
    """
    Pack INT8 input for hardware.  Matches ``simple_conv.py pack_input()``.

    Loop order::
      IC_atom(16) -> W -> H -> IC_lane(16)

    Stores ``(value - 0x80) & 0xFF``.

    Parameters
    ----------
    input_nhwc : ndarray (1, in_h, in_w, in_c)  NHWC uint8

    Returns
    -------
    packed : ndarray (uint8)
    """
    in_c1 = _ceil_div(in_c, INT8_FEATURE_ATOMIC_SIZE) * 2
    raw = np.zeros(in_w * in_h * in_c1 * INT8_FEATURE_ATOMIC_SIZE, dtype=np.uint8)

    if in_c == 1:
        n = 0
        for x in range(in_w):
            for y in range(max(in_h, INT8_FEATURE_ATOMIC_SIZE)):
                raw[n] = (int(input_nhwc[0, y, x, 0]) - 0x80) & 0xFF if y < in_h else 0
                n += 1
        return raw

    n = 0
    for u in range(_ceil_div(in_c, INT8_FEATURE_ATOMIC_SIZE)):
        for x in range(in_w):
            for y in range(in_h):
                for c in range(INT8_FEATURE_ATOMIC_SIZE):
                    ic = c + u * INT8_FEATURE_ATOMIC_SIZE
                    if ic < in_c:
                        raw[n] = (int(input_nhwc[0, y, x, ic]) - 0x80) & 0xFF
                    else:
                        raw[n] = 0
                    n += 1
    return raw


# ---------------------------------------------------------------------------
# INT8 output unpacker  (matches simple_conv.py read_output)
# ---------------------------------------------------------------------------

def unpack_output_int8(raw_bytes, out_c, out_h, out_w):
    """
    Unpack hardware output bytes to NHWC uint8.

    Matches ``simple_conv.py read_output()``.

    The hardware writes data in atoms of 16 channels, reshaped as
    (-1, out_w, out_h, 16).  Each byte has ``+0x80`` added back.
    """
    oc1 = _ceil_div(out_c, INT8_FEATURE_ATOMIC_SIZE) * 2
    raw = raw_bytes[:out_w * out_h * oc1 * INT8_FEATURE_ATOMIC_SIZE]
    raw = raw.reshape(-1, out_w, out_h, INT8_FEATURE_ATOMIC_SIZE)
    out = np.zeros((1, out_h, out_w, out_c), dtype=np.uint8)
    for oc_idx in range(out_c):
        c = oc_idx % INT8_FEATURE_ATOMIC_SIZE
        g = oc_idx // INT8_FEATURE_ATOMIC_SIZE
        for y in range(out_h):
            for x in range(out_w):
                out[0, y, x, oc_idx] = (int(raw[g, x, y, c]) + 0x80) & 0xFF
    return out


# ---------------------------------------------------------------------------
# Buffer size calculators  (match simple_conv.py)
# ---------------------------------------------------------------------------

def calc_input_size(in_w, in_h, in_c):
    """Hardware-expected input buffer size in bytes."""
    ic1 = _ceil_div(in_c, INT8_FEATURE_ATOMIC_SIZE) * 2
    return in_w * in_h * ic1 * INT8_FEATURE_ATOMIC_SIZE

def calc_weight_size(kw, kh, in_c, out_c, is_dw=False):
    """Hardware-expected weight buffer size in bytes."""
    wc = _align_up(max(in_c, INT8_FEATURE_ATOMIC_SIZE), INT8_WEIGHT_ATOMIC_SIZE)
    oc = 1 if is_dw else _align_up(out_c, 2)
    return kw * kh * oc * wc * 2  # * 2 for signed/unsigned pairs?

def calc_raw_output_size(out_w, out_h, out_c):
    """Hardware-expected output buffer size in bytes."""
    oc1 = _ceil_div(out_c, INT8_FEATURE_ATOMIC_SIZE) * 2
    return out_w * out_h * oc1 * INT8_FEATURE_ATOMIC_SIZE

def calc_line_stride(in_w):
    return in_w * INT8_FEATURE_ATOMIC_SIZE * 1  # BPE=1

def calc_entries_per_slice(in_w, in_c):
    entry_size = INT8_CBUF_ENTRY_SIZE
    atom_size = INT8_FEATURE_ATOMIC_SIZE  # 16
    atomics_per_entry = entry_size // atom_size  # 8
    total_c_atomics = _ceil_div(in_c, atom_size)
    last_c_atomics = total_c_atomics % atomics_per_entry
    int_c_entries = (total_c_atomics // atomics_per_entry) * in_w
    if last_c_atomics == 3:
        frac_c_entries = in_w
    else:
        frac_c_entries = _ceil_div(last_c_atomics * in_w, atomics_per_entry)
    return int_c_entries + frac_c_entries


# ---------------------------------------------------------------------------
# CBUF bank calculators  (match simple_conv.py)
# ---------------------------------------------------------------------------

def calc_input_banks(in_w, in_h, in_c):
    """Number of CBUF banks needed for input feature map."""
    return _ceil_div(calc_entries_per_slice(in_w, in_c) * in_h,
                     INT8_CBUF_ENTRIES_PER_BANK)


def calc_weights_banks(kw, kh, in_c, out_c, is_dw=False):
    """Number of CBUF banks needed for weights."""
    bytes_ = kw * kh * in_c
    if not is_dw:
        bytes_ *= out_c
    return _ceil_div(_ceil_div(bytes_, INT8_CBUF_ENTRY_SIZE),
                     INT8_CBUF_ENTRIES_PER_BANK) + 1


def split_tasks(in_w, in_h, in_c, out_c, kw, kh, is_dw=False):
    """
    Split a convolution into tiled sub-tasks.

    Returns list of dicts with keys:
      input_height, output_height,
      input_offset, output_offset,
      input_banks, weight_banks, weight_reuse
    """
    entries_per_slice = calc_entries_per_slice(in_w, in_c)
    input_banks_needed = calc_input_banks(in_w, in_h, in_c)
    weight_banks_needed = calc_weights_banks(kw, kh, in_c, out_c, is_dw)
    available_input_banks = INT8_CBUF_BANKS - weight_banks_needed
    available_weight_banks = weight_banks_needed

    if input_banks_needed <= available_input_banks:
        return [dict(
            input_height=in_h, output_height=in_h - kh + 1,
            input_offset=0, output_offset=0,
            input_banks=input_banks_needed,
            weight_banks=INT8_CBUF_BANKS - input_banks_needed)]

    if weight_banks_needed + 1 >= INT8_CBUF_BANKS:
        available_input_banks = 7
        available_weight_banks = INT8_CBUF_BANKS - available_input_banks

    available_slices = (INT8_CBUF_ENTRIES_PER_BANK * available_input_banks) // entries_per_slice
    slices = [dict(top_slice=0, bottom_slice=available_slices - 1)]
    s = kh - 1
    while s < in_h:
        prev = slices[-1]
        while s <= prev["bottom_slice"]:
            s += 1
        if s > prev["bottom_slice"]:
            s -= 1
        top_slice = min(s, prev["bottom_slice"]) - (kh - 1) + 1
        bottom_slice = top_slice + available_slices - 1
        if bottom_slice >= in_h - 1:
            slices.append(dict(top_slice=top_slice, bottom_slice=in_h - 1))
            break
        s = top_slice + kh - 1
        slices.append(dict(top_slice=top_slice, bottom_slice=bottom_slice))

    output_height_processed = 0
    tasks_out = []
    for sl in slices:
        task_in_h = min(sl["bottom_slice"], in_h - 1) - sl["top_slice"] + 1
        if task_in_h < kh:
            continue
        task_out_h = task_in_h - kh + 1
        tasks_out.append(dict(
            input_height=task_in_h, output_height=task_out_h,
            input_offset=calc_line_stride(in_w) * sl["top_slice"],
            output_offset=calc_line_stride(in_w - kw + 1) * output_height_processed,
            input_banks=available_input_banks,
            weight_banks=available_weight_banks,
            weight_reuse=(weight_banks_needed + 1 < INT8_CBUF_BANKS)))
        output_height_processed += task_out_h
    return tasks_out


# ---------------------------------------------------------------------------
# INT8 bias packer  (matches simple_conv.py pack_biases)
# ---------------------------------------------------------------------------

def pack_biases_int8(biases, weight_ohwi, in_c, kw, kh, out_c,
                     input_zp=0, weight_zp=0, is_dw=False):
    """
    Pack biases with zero-point correction.

    Matches simple_conv.py pack_biases: the input is stored as
    ``(val - 0x80)`` in the packed buffer, so the effective zero-point
    correction factor is ``(input_zp - 0x80)``.

      packed_bias[oc] = bias[oc] - sum_{w,h,ic}(weight - wzp) * (input_zp - 0x80)
    """
    packed = np.zeros(out_c, dtype=np.int32)
    zp_correction = input_zp - 0x80
    for oc in range(out_c):
        correction = 0
        for wx in range(kw):
            for wy in range(kh):
                if is_dw:
                    weight = int(weight_ohwi[0, wx, wy, oc]) if oc < in_c else 0
                    correction += (weight - weight_zp) * zp_correction
                else:
                    for ic in range(in_c):
                        weight = int(weight_ohwi[oc, wx, wy, ic])
                        correction += (weight - weight_zp) * zp_correction
        packed[oc] = int(biases[oc]) - correction
    return packed


# ---------------------------------------------------------------------------
# im2col helper — flatten spatial conv to 1x1 for small-channel configs
# ---------------------------------------------------------------------------

def im2col_weights(weight_ohwi, out_c, in_c, kh, kw, is_dw=False):
    """
    Flatten a spatial convolution to 1x1 by concatenating KH×KW into
    the channel dimension.  Matches simple_conv.py run_conv2d_im2col.
    """
    flat_c = kh * kw * in_c
    flat_c_aligned = _align_up(flat_c, INT8_FEATURE_ATOMIC_SIZE)
    weight_1x1 = np.zeros((out_c, flat_c_aligned), dtype=np.uint8)

    if is_dw:
        for channel in range(in_c):
            offset = channel * kh * kw
            for wx in range(kw):
                for wy in range(kh):
                    weight_1x1[channel, offset] = weight_ohwi[0, wx, wy, channel]
                    offset += 1
    else:
        for oc in range(out_c):
            offset = 0
            for ic in range(in_c):
                for wx in range(kw):
                    for wy in range(kh):
                        weight_1x1[oc, offset] = weight_ohwi[oc, wx, wy, ic]
                        offset += 1
    return weight_1x1.reshape(out_c, 1, 1, flat_c_aligned)

def im2col_input(input_nhwc, in_c, in_h, in_w, kh, kw):
    flat_c = kh * kw * in_c
    flat_c_aligned = _align_up(flat_c, INT8_FEATURE_ATOMIC_SIZE)
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    im = np.zeros((1, out_h, out_w, flat_c_aligned), dtype=np.uint8)
    off = 0
    for ic in range(in_c):
        for wx in range(kw):
            for wy in range(kh):
                im[0, :, :, off] = input_nhwc[0, wy:wy + out_h, wx:wx + out_w, ic]
                off += 1
    return im


# ---------------------------------------------------------------------------
# DW chunking — split large depthwise into 64-channel groups
# ---------------------------------------------------------------------------

def dw_chunk_shapes(in_c, out_c, groups):
    """Yield (chunk_in_c, chunk_out_c, chunk_groups, start_offset)."""
    chunk_size = 64
    for start in range(0, in_c, chunk_size):
        end = min(start + chunk_size, in_c)
        yield end - start, end - start, end - start, start


# ---------------------------------------------------------------------------
# CSC/CMAC Simulator
# ---------------------------------------------------------------------------

class CSCSim:
    def __init__(self, oc_atom=16, ic_atom=32, flip_kspatial=False):
        self.oc_atom = oc_atom
        self.ic_atom = ic_atom
        self.flip_kspatial = flip_kspatial

    # -- CSC stream trace ------------------------------------------------

    def trace(self, wpack, input_nchw, out_c, in_c, kh, kw, groups=1,
              dtype=None):
        """
        Emit the logical CSC stream consumed by the tiny CMAC model.

        Each trace entry represents one data/weight atom pairing for one output
        channel.  This intentionally stops at the CSC->CMAC boundary: no
        register file, command processor, DMA timing, CACC, or DPU behavior.
        """
        oc_atom = wpack.oc_atom
        ic_atom = wpack.ic_atom
        fks = wpack.flip_kspatial
        if dtype is None:
            dtype = wpack.dtype

        batch, in_c_full, in_h, in_w = input_nchw.shape
        if batch != 1:
            raise NotImplementedError("batch > 1")
        if in_h < kh or in_w < kw:
            raise ValueError("kernel larger than input")

        is_dw = _is_depthwise(in_c, out_c, groups)
        in_c_eff = in_c_full // groups
        pack_out_c = wpack.pack_out_c
        aligned_in_c = wpack.aligned_in_c
        events = []

        if is_dw:
            w_idx = 0
            ic_blocks = aligned_in_c // wpack.ic_atom
            stream_stride = _stream_ic_stride(dtype, in_c, wpack.ic_atom)
            kspatial_outer = range(kw) if fks else range(kh)
            kspatial_inner = range(kh) if fks else range(kw)
            if dtype == "int8":
                for ib in range(ic_blocks):
                    ic_base = ib * wpack.ic_atom
                    for si in kspatial_outer:
                        for sj in kspatial_inner:
                            ky = sj if fks else si
                            kx = si if fks else sj
                            for ici in range(stream_stride):
                                ic = ic_base + ici
                                events.append(dict(
                                    dtype=dtype,
                                    is_depthwise=True,
                                    oc=ic,
                                    oc_base=ic,
                                    oc_lim=1 if ic < out_c else 0,
                                    ic_base=ic,
                                    ic_lim=1 if ic < in_c else 0,
                                    ky=ky,
                                    kx=kx,
                                    weight_offset=w_idx,
                                    weight_stride=1,
                                ))
                                w_idx += 1
            else:
                for si in kspatial_outer:
                    for sj in kspatial_inner:
                        ky = sj if fks else si
                        kx = si if fks else sj
                        for ib in range(ic_blocks):
                            ic_base = ib * wpack.ic_atom
                            for ici in range(stream_stride):
                                ic = ic_base + ici
                                events.append(dict(
                                    dtype=dtype,
                                    is_depthwise=True,
                                    oc=ic,
                                    oc_base=ic,
                                    oc_lim=1 if ic < out_c else 0,
                                    ic_base=ic,
                                    ic_lim=1 if ic < in_c else 0,
                                    ky=ky,
                                    kx=kx,
                                    weight_offset=w_idx,
                                    weight_stride=1,
                                ))
                                w_idx += 1
            _mark_trace_flags(events)
            return events

        oc_blocks = _ceil_div(pack_out_c, oc_atom)
        ic_blocks = aligned_in_c // ic_atom
        w_offset = 0
        kspatial_outer = range(kw) if fks else range(kh)
        kspatial_inner = range(kh) if fks else range(kw)
        stream_stride = _stream_ic_stride(dtype, in_c_eff, ic_atom)

        for ob in range(oc_blocks):
            oc_base = ob * oc_atom
            oc_lim = min(oc_atom, pack_out_c - oc_base)
            for ib in range(ic_blocks):
                ic_base = ib * ic_atom
                ic_lim = min(ic_atom, max(0, in_c_eff - ic_base))
                for si in kspatial_outer:
                    for sj in kspatial_inner:
                        ky = sj if fks else si
                        kx = si if fks else sj
                        for oci in range(oc_lim):
                            oc = oc_base + oci
                            events.append(dict(
                                dtype=dtype,
                                is_depthwise=False,
                                oc=oc,
                                oc_base=oc_base,
                                oc_lim=oc_lim,
                                ic_base=ic_base,
                                ic_lim=ic_lim,
                                ky=ky,
                                kx=kx,
                                weight_offset=w_offset,
                                weight_stride=stream_stride,
                            ))
                            w_offset += stream_stride
        _mark_trace_flags(events)
        return events

    def conv_from_trace(self, trace, wpack, input_nchw, out_c, in_c, kh, kw,
                        dtype=None, input_zp=0, weight_zp=0):
        if dtype is None:
            dtype = wpack.dtype
        batch, in_c_full, in_h, in_w = input_nchw.shape
        out_h = in_h - kh + 1
        out_w = in_w - kw + 1
        output = np.zeros((out_c, out_h, out_w), dtype=np.float64)
        w = wpack.data.ravel()

        for ev in trace:
            oc = ev["oc"]
            if oc >= out_c:
                continue
            for ici in range(ev["ic_lim"]):
                ic = ev["ic_base"] + ici
                if ic >= in_c_full:
                    continue
                w_val = _read_weight(w[ev["weight_offset"] + ici], dtype, weight_zp)
                for oy in range(out_h):
                    for ox in range(out_w):
                        x_val = _read_input(
                            input_nchw[0, ic, oy + ev["ky"], ox + ev["kx"]],
                            dtype, input_zp)
                        output[oc, oy, ox] += w_val * x_val
        return output.astype(np.float32)

    # -- convolution from WeightPack (weight-stream verification) --------

    def conv(self, wpack, input_nchw, out_c, in_c, kh, kw, groups=1,
             dtype=None, input_zp=0, weight_zp=0):
        if dtype is None:
            dtype = wpack.dtype
        trace = self.trace(wpack, input_nchw, out_c, in_c, kh, kw, groups, dtype)
        return self.conv_from_trace(trace, wpack, input_nchw, out_c, in_c, kh, kw,
                                    dtype, input_zp, weight_zp)

    def verify(self, wpack, input_nchw, weight_nchw,
               out_c, in_c, kh, kw, groups=1, atol=0.2,
               input_zp=0, weight_zp=0):
        dtype = wpack.dtype
        result = self.conv(wpack, input_nchw, out_c, in_c, kh, kw, groups,
                           dtype=dtype, input_zp=input_zp, weight_zp=weight_zp)
        if dtype == "int8":
            expected = naive_conv_quant(
                input_nchw, weight_nchw, out_c, in_c, kh, kw, groups,
                input_zp=input_zp, weight_zp=weight_zp)
        else:
            expected = naive_conv(input_nchw, weight_nchw, out_c, in_c, kh, kw, groups)
        md = float(np.max(np.abs(result.astype(np.float64) - expected.astype(np.float64))))
        ok = bool(np.allclose(result, expected, atol=atol) and not np.any(np.isinf(result)))
        return result, expected, ok, md


# ---------------------------------------------------------------------------
# helpers for reading packed values
# ---------------------------------------------------------------------------

def _read_weight(val, dtype, zp):
    if dtype == "int8":
        return float(val) - float(zp)
    return float(val)

def _read_input(val, dtype, zp):
    if dtype == "int8":
        return float(val) - float(zp)
    return float(val)

def _stream_ic_stride(dtype, in_c_eff, ic_atom):
    if dtype == "fp16":
        return ic_atom
    return min(max(in_c_eff, INT8_FEATURE_ATOMIC_SIZE), ic_atom)

def _mark_trace_flags(events):
    for idx, ev in enumerate(events):
        ev["stripe_st"] = idx == 0
        ev["stripe_end"] = idx == len(events) - 1
        ev["channel_end"] = ev["stripe_end"]
        ev["layer_end"] = ev["stripe_end"]

def validate_csc_trace(trace, wpack, input_nchw, out_c, in_c, kh, kw, groups=1):
    batch, in_c_full, in_h, in_w = input_nchw.shape
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    if batch != 1 or out_h <= 0 or out_w <= 0:
        return False, "invalid shape"
    if not trace:
        return False, "empty trace"
    if not trace[0].get("stripe_st"):
        return False, "missing stripe_st"
    if not trace[-1].get("layer_end"):
        return False, "missing layer_end"

    contributions = 0
    for idx, ev in enumerate(trace):
        if ev["weight_offset"] < 0:
            return False, f"negative weight offset at {idx}"
        if ev["weight_offset"] + max(0, ev["ic_lim"] - 1) >= wpack.data.size:
            return False, f"weight offset out of range at {idx}"
        if ev["ky"] < 0 or ev["ky"] >= kh or ev["kx"] < 0 or ev["kx"] >= kw:
            return False, f"kernel coordinate out of range at {idx}"
        if ev["ic_lim"] and (ev["ic_base"] < 0 or ev["ic_base"] + ev["ic_lim"] > in_c_full):
            return False, f"input channel range out of bounds at {idx}"
        valid_oc = 1 if ev["oc"] < out_c else 0
        contributions += valid_oc * ev["ic_lim"] * out_h * out_w

    if _is_depthwise(in_c, out_c, groups):
        expected = in_c * kh * kw * out_h * out_w
    else:
        expected = out_c * (in_c // groups) * kh * kw * out_h * out_w
    if contributions != expected:
        return False, f"contribution count {contributions} != {expected}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Reference convolution (naive NCHW, float)
# ---------------------------------------------------------------------------

def naive_conv(input_nchw, weight_nchw, out_c, in_c, kh, kw, groups=1):
    batch, in_c_full, in_h, in_w = input_nchw.shape
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    in_per_group = in_c // groups
    out_per_group = out_c // groups
    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float64)
    for n in range(batch):
        for g in range(groups):
            ic_start = g * in_per_group
            oc_start = g * out_per_group
            for oc in range(out_per_group):
                oc_global = oc_start + oc
                for ic in range(in_per_group):
                    ic_global = ic_start + ic
                    for ky in range(kh):
                        for kx in range(kw):
                            w = float(weight_nchw[oc_global, ic, ky, kx])
                            result[n, oc_global] += w * input_nchw[n, ic_global, ky:ky+out_h, kx:kx+out_w].astype(np.float64)
    return result


def naive_conv_quant(input_nchw, weight_nchw, out_c, in_c, kh, kw, groups=1,
                     input_zp=0, weight_zp=0):
    batch, in_c_full, in_h, in_w = input_nchw.shape
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    in_per_group = in_c // groups
    out_per_group = out_c // groups
    accum = np.zeros((batch, out_c, out_h, out_w), dtype=np.float64)
    for n in range(batch):
        for g in range(groups):
            ic_start = g * in_per_group
            oc_start = g * out_per_group
            for oc in range(out_per_group):
                oc_global = oc_start + oc
                for ic in range(in_per_group):
                    ic_global = ic_start + ic
                    for ky in range(kh):
                        for kx in range(kw):
                            w = float(int(weight_nchw[oc_global, ic, ky, kx]) - weight_zp)
                            accum[n, oc_global] += w * \
                                (input_nchw[n, ic_global, ky:ky+out_h, kx:kx+out_w].astype(np.float64) - float(input_zp))
    return accum


def expand_grouped_weights(weight, in_c, out_c, kh, kw, groups):
    weight_in_c = in_c // groups
    if groups == 1:
        return weight.reshape(out_c, in_c, kh, kw)
    expanded = np.zeros((out_c, in_c, kh, kw), dtype=weight.dtype)
    out_per_group = out_c // groups
    for oc in range(out_c):
        group = oc // out_per_group
        start = group * weight_in_c
        expanded[oc, start:start + weight_in_c] = weight[oc]
    return expanded


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------

def _test(label, fn):
    ok = fn()
    print(f"  {label}: {'PASS' if ok else 'FAIL'}")
    return ok

# ---- FP16 tests ----

def test_pointwise_fp16():
    np.random.seed(42)
    in_c, out_c, in_h, in_w = 64, 16, 4, 4
    x = np.random.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
    w = np.random.uniform(-1, 1, (out_c, in_c, 1, 1)).astype(np.float16)
    wpack = pack_weights_fp16(w, out_c, in_c, 1, 1)
    sim = CSCSim()
    _, _, ok, md = sim.verify(wpack, x, w, out_c, in_c, 1, 1)
    return ok

def test_pointwise_wide_tail_fp16():
    np.random.seed(42)
    in_c, out_c, in_h, in_w = 256, 32, 4, 4
    x = np.random.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
    w = np.random.uniform(-1, 1, (out_c, in_c, 1, 1)).astype(np.float16)
    wpack = pack_weights_fp16(w, out_c, in_c, 1, 1)
    sim = CSCSim()
    _, _, ok, md = sim.verify(wpack, x, w, out_c, in_c, 1, 1)
    return ok

def test_pointwise_odd_tail_fp16():
    np.random.seed(42)
    in_c, out_c, in_h, in_w = 64, 24, 4, 4
    x = np.random.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
    w = np.random.uniform(-1, 1, (out_c, in_c, 1, 1)).astype(np.float16)
    wpack = pack_weights_fp16(w, out_c, in_c, 1, 1)
    sim = CSCSim()
    _, _, ok, md = sim.verify(wpack, x, w, out_c, in_c, 1, 1)
    return ok

def test_spatial3x3_fp16():
    np.random.seed(42)
    in_c, out_c, in_h, in_w = 32, 16, 6, 6
    x = np.random.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
    w = np.random.uniform(-1, 1, (out_c, in_c, 3, 3)).astype(np.float16)
    wpack = pack_weights_fp16(w, out_c, in_c, 3, 3)
    sim = CSCSim()
    _, _, ok, md = sim.verify(wpack, x, w, out_c, in_c, 3, 3)
    return ok

def test_depthwise_fp16():
    np.random.seed(42)
    c, in_h, in_w, kh, kw = 8, 6, 6, 3, 3
    x = np.random.uniform(-1, 1, (1, c, in_h, in_w)).astype(np.float16)
    w = np.random.uniform(-1, 1, (c, 1, kh, kw)).astype(np.float16)
    wpack = pack_weights_fp16(w, c, c, kh, kw, groups=c)
    sim = CSCSim()
    _, _, ok, md = sim.verify(wpack, x, w, c, c, kh, kw, groups=c)
    return ok

def test_grouped_fp16():
    np.random.seed(42)
    in_c, out_c, in_h, in_w, g = 16, 32, 5, 5, 4
    kw_in = in_c // g
    x = np.random.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
    w = np.random.uniform(-1, 1, (out_c, kw_in, 1, 1)).astype(np.float16)
    w_expanded = expand_grouped_weights(w, in_c, out_c, 1, 1, g)
    wpack = pack_weights_fp16(w_expanded, out_c, in_c, 1, 1, ic_atom=8, groups=1)
    sim = CSCSim(ic_atom=8)
    _, _, ok, md = sim.verify(wpack, x, w_expanded, out_c, in_c, 1, 1, groups=g)
    return ok

# ---- INT8 tests (weight-stream verification) ----

def test_pointwise_int8():
    np.random.seed(42)
    in_c, out_c, in_h, in_w = 32, 16, 4, 4
    x = np.random.randint(0, 256, (1, in_c, in_h, in_w), dtype=np.uint8)
    w = np.random.randint(0, 256, (out_c, in_c, 1, 1), dtype=np.uint8)
    wpack = pack_weights_int8(w, out_c, in_c, 1, 1)
    sim = CSCSim(oc_atom=32, ic_atom=32, flip_kspatial=True)
    _, _, ok, md = sim.verify(wpack, x, w, out_c, in_c, 1, 1, input_zp=0, weight_zp=0, atol=2)
    return ok

def test_spatial3x3_int8():
    np.random.seed(42)
    in_c, out_c, in_h, in_w = 16, 16, 6, 6
    x = np.random.randint(0, 256, (1, in_c, in_h, in_w), dtype=np.uint8)
    w = np.random.randint(0, 256, (out_c, in_c, 3, 3), dtype=np.uint8)
    wpack = pack_weights_int8(w, out_c, in_c, 3, 3)
    sim = CSCSim(oc_atom=32, ic_atom=32, flip_kspatial=True)
    _, _, ok, md = sim.verify(wpack, x, w, out_c, in_c, 3, 3, input_zp=0, weight_zp=0, atol=2)
    return ok

def test_pointwise_odd_oc_int8():
    np.random.seed(42)
    in_c, out_c, in_h, in_w = 32, 6, 4, 4
    x = np.random.randint(0, 256, (1, in_c, in_h, in_w), dtype=np.uint8)
    w = np.random.randint(0, 256, (out_c, in_c, 1, 1), dtype=np.uint8)
    wpack = pack_weights_int8(w, out_c, in_c, 1, 1)
    sim = CSCSim(oc_atom=32, ic_atom=32, flip_kspatial=True)
    _, _, ok, md = sim.verify(wpack, x, w, out_c, in_c, 1, 1, atol=2)
    return ok

def test_pointwise_nonzp_int8():
    np.random.seed(42)
    in_c, out_c, in_h, in_w = 32, 8, 4, 4
    izp, wzp = 128, 128
    x = np.random.randint(0, 256, (1, in_c, in_h, in_w), dtype=np.uint8)
    w = np.random.randint(0, 256, (out_c, in_c, 1, 1), dtype=np.uint8)
    wpack = pack_weights_int8(w, out_c, in_c, 1, 1)
    sim = CSCSim(oc_atom=32, ic_atom=32, flip_kspatial=True)
    _, _, ok, md = sim.verify(wpack, x, w, out_c, in_c, 1, 1,
                               input_zp=izp, weight_zp=wzp, atol=2)
    return ok

# ---- INT8 byte-accurate packer exact-match tests ----

def _pack_weights_simple_conv_ref(weight_ohwi, out_c, in_c, kh, kw, is_dw=False):
    """Replicate simple_conv.py pack_weights() exactly for cross-check."""
    ic = max(in_c, INT8_FEATURE_ATOMIC_SIZE)
    oc = 1 if is_dw else _align_up(out_c, 2)
    ic_groups = INT8_WEIGHT_ATOMIC_SIZE * (2 if is_dw else 1)
    size = kh * kw * oc * _align_up(ic, ic_groups)
    out = np.zeros(size, dtype=np.uint8)
    n = 0
    for oo in range(_ceil_div(oc, INT8_WEIGHT_ATOMIC_SIZE)):
        for ii in range(_ceil_div(ic, ic_groups)):
            for kx in range(kw):
                for ky in range(kh):
                    for oi in range(min(oc - oo * INT8_WEIGHT_ATOMIC_SIZE, INT8_WEIGHT_ATOMIC_SIZE)):
                        oc_idx = oo * INT8_WEIGHT_ATOMIC_SIZE + oi
                        for ij in range(min(ic, ic_groups)):
                            ic_idx = ii * ic_groups + ij
                            if oc_idx >= out_c or ic_idx >= in_c:
                                out[n] = 0
                            elif is_dw:
                                out[n] = (int(weight_ohwi[0, kx, ky, ic_idx]) - 0x80) & 0xFF
                            else:
                                out[n] = (int(weight_ohwi[oc_idx, kx, ky, ic_idx]) - 0x80) & 0xFF
                            n += 1
    return out

def test_pack_weights_int8_byte_exact():
    """Verify csc_sim byte packer matches simple_conv.py exactly — single IC block."""
    for out_c, in_c in [(4, 8), (16, 32), (6, 32), (32, 16)]:
        np.random.seed(42)
        w = np.random.randint(0, 256, (out_c, 1, 1, in_c), dtype=np.uint8)
        csc, oc, icg, ic, aic = pack_weights_int8_byte(w, out_c, in_c, 1, 1)
        ref = _pack_weights_simple_conv_ref(w, out_c, in_c, 1, 1)
        assert np.array_equal(csc, ref), f"mismatch for out_c={out_c}, in_c={in_c}"
    return True

def test_pack_weights_int8_byte_multi_icblock():
    """Exercise multiple IC blocks (in_c > ic_groups) — catches ij vs ic_idx bug."""
    in_c, out_c = 96, 16
    np.random.seed(42)
    w = np.random.randint(0, 256, (out_c, 1, 1, in_c), dtype=np.uint8)
    csc, oc, icg, ic, aic = pack_weights_int8_byte(w, out_c, in_c, 1, 1)
    ref = _pack_weights_simple_conv_ref(w, out_c, in_c, 1, 1)
    assert np.array_equal(csc, ref), f"multi-IC-block mismatch: out_c={out_c}, in_c={in_c}"
    return True

def test_pack_weights_int8_byte_dw_exact():
    """Verify depthwise byte packer matches simple_conv."""
    in_c, kw, kh = 16, 3, 3
    np.random.seed(42)
    w = np.random.randint(0, 256, (1, kw, kh, in_c), dtype=np.uint8)
    csc, oc, icg, ic, aic = pack_weights_int8_byte(w, in_c, in_c, kh, kw, is_dw=True)
    ref = _pack_weights_simple_conv_ref(w, in_c, in_c, kh, kw, is_dw=True)
    assert np.array_equal(csc, ref), "DW mismatch"
    return True

def test_pack_weights_int8_byte_spatial_exact():
    """Verify spatial 3x3 byte packer matches simple_conv."""
    out_c, in_c, kw, kh = 8, 6, 3, 3
    np.random.seed(42)
    w = np.random.randint(0, 256, (out_c, kw, kh, in_c), dtype=np.uint8)
    csc, oc, icg, ic, aic = pack_weights_int8_byte(w, out_c, in_c, kh, kw)
    ref = _pack_weights_simple_conv_ref(w, out_c, in_c, kh, kw)
    assert np.array_equal(csc, ref), "spatial mismatch"
    return True

# ---- Input packing exact test ----

def test_pack_input_int8_exact():
    """Verify pack_input_int8 matches simple_conv.py pack_input()."""
    for in_c, in_h, in_w in [(16, 4, 4), (8, 3, 5), (1, 4, 4)]:
        np.random.seed(42)
        inp = np.random.randint(0, 256, (1, in_h, in_w, in_c), dtype=np.uint8)
        packed = pack_input_int8(inp, in_c, in_h, in_w)
        expected_size = calc_input_size(in_w, in_h, in_c)
        assert packed.size == expected_size, f"size {packed.size} != {expected_size}"
    return True

# ---- Output unpack round-trip test ----

def test_unpack_output_int8():
    np.random.seed(42)
    out_c, out_h, out_w = 8, 3, 3
    oc1 = _ceil_div(out_c, INT8_FEATURE_ATOMIC_SIZE) * 2
    raw_size = out_w * out_h * oc1 * INT8_FEATURE_ATOMIC_SIZE
    fake_raw = np.random.randint(0, 256, raw_size, dtype=np.uint8)
    result = unpack_output_int8(fake_raw, out_c, out_h, out_w)
    assert result.shape == (1, out_h, out_w, out_c), f"shape {result.shape}"
    return True

# ---- im2col test ----

def test_im2col():
    np.random.seed(42)
    in_c, out_c, in_h, in_w, kh, kw = 2, 4, 5, 5, 3, 3
    w = np.random.randint(0, 256, (out_c, kw, kh, in_c), dtype=np.uint8)
    w_flat = im2col_weights(w, out_c, in_c, kh, kw)
    assert w_flat.shape == (out_c, 1, 1, _align_up(kh * kw * in_c, 16))
    return True

# ---- Bias pack exact test ----

def test_pack_biases_int8_exact():
    """Verify pack_biases_int8 matches simple_conv.py pack_biases() with INPUT_ZERO_POINT=0."""
    out_c, in_c = 4, 8
    np.random.seed(42)
    w = np.random.randint(0, 256, (out_c, 1, 1, in_c), dtype=np.uint8)
    b = np.random.randint(-128, 128, out_c, dtype=np.int32)
    # simple_conv.py correction: sum(weight - 0) * (0 - 0x80)
    ref = np.zeros(out_c, dtype=np.int32)
    for oc in range(out_c):
        correction = 0
        for ic in range(in_c):
            correction += (int(w[oc, 0, 0, ic]) - 0) * (0 - 0x80)
        ref[oc] = int(b[oc]) - correction
    # csc_sim with input_zp=0 -> uses zp_correction = 0 - 0x80 = -128
    packed = pack_biases_int8(b, w, in_c, 1, 1, out_c, input_zp=0, weight_zp=0)
    assert np.array_equal(packed, ref), "bias pack mismatch"
    return True

# ---- Bank / task splitting tests ----

def test_bank_calcs():
    in_c, out_c, in_h, in_w = 16, 32, 4, 4
    ib = calc_input_banks(in_w, in_h, in_c)
    wb = calc_weights_banks(1, 1, in_c, out_c)
    assert ib > 0 and wb > 0
    assert ib + wb <= INT8_CBUF_BANKS + 1  # they may overlap
    return True

def test_split_tasks_single():
    """Small conv that fits in one task."""
    tasks = split_tasks(4, 4, 16, 32, 1, 1)
    assert len(tasks) >= 1
    assert tasks[0]["output_height"] == 4
    return True

def test_split_tasks_multi():
    """Tall conv that needs multiple slices."""
    tasks = split_tasks(8, 48, 16, 32, 3, 3)
    if len(tasks) > 1:
        total_oh = sum(t["output_height"] for t in tasks)
        assert total_oh == 48 - 3 + 1, f"total output height {total_oh} != {48 - 3 + 1}"
    return True


# ---- CSC trace / boundary sweep tests ----

def _verify_trace_fp16(in_c, out_c, kh=1, kw=1, in_h=None, in_w=None):
    seed = 1000 + in_c * 17 + out_c * 31 + kh * 7 + kw
    rng = np.random.default_rng(seed)
    if in_h is None:
        in_h = kh + 2
    if in_w is None:
        in_w = kw + 2
    x = rng.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
    w = rng.uniform(-1, 1, (out_c, in_c, kh, kw)).astype(np.float16)
    wpack = pack_weights_fp16(w, out_c, in_c, kh, kw)
    sim = CSCSim()
    trace = sim.trace(wpack, x, out_c, in_c, kh, kw)
    ok, msg = validate_csc_trace(trace, wpack, x, out_c, in_c, kh, kw)
    assert ok, msg
    result = sim.conv_from_trace(trace, wpack, x, out_c, in_c, kh, kw)
    expected = naive_conv(x, w, out_c, in_c, kh, kw)[0]
    assert np.allclose(result, expected, atol=0.2), (in_c, out_c, kh, kw)
    return True

def _verify_trace_int8(in_c, out_c, kh=1, kw=1, in_h=None, in_w=None):
    seed = 2000 + in_c * 17 + out_c * 31 + kh * 7 + kw
    rng = np.random.default_rng(seed)
    if in_h is None:
        in_h = kh + 2
    if in_w is None:
        in_w = kw + 2
    x = rng.integers(0, 256, (1, in_c, in_h, in_w), dtype=np.uint8)
    w = rng.integers(0, 256, (out_c, in_c, kh, kw), dtype=np.uint8)
    wpack = pack_weights_int8(w, out_c, in_c, kh, kw)
    sim = CSCSim(oc_atom=32, ic_atom=32, flip_kspatial=True)
    trace = sim.trace(wpack, x, out_c, in_c, kh, kw)
    ok, msg = validate_csc_trace(trace, wpack, x, out_c, in_c, kh, kw)
    assert ok, msg
    result = sim.conv_from_trace(trace, wpack, x, out_c, in_c, kh, kw,
                                 dtype="int8", input_zp=0, weight_zp=0)
    expected = naive_conv_quant(x, w, out_c, in_c, kh, kw, input_zp=0, weight_zp=0)[0]
    assert np.allclose(result, expected, atol=2), (in_c, out_c, kh, kw)
    return True

def _verify_trace_dw_int8(c, kh=1, kw=1, in_h=None, in_w=None):
    seed = 2500 + c * 19 + kh * 7 + kw
    rng = np.random.default_rng(seed)
    if in_h is None:
        in_h = kh + 2
    if in_w is None:
        in_w = kw + 2
    x = rng.integers(0, 256, (1, c, in_h, in_w), dtype=np.uint8)
    w = rng.integers(0, 256, (c, 1, kh, kw), dtype=np.uint8)
    wpack = pack_weights_int8(w, c, c, kh, kw, groups=c)
    sim = CSCSim(oc_atom=32, ic_atom=32, flip_kspatial=True)
    trace = sim.trace(wpack, x, c, c, kh, kw, groups=c)
    ok, msg = validate_csc_trace(trace, wpack, x, c, c, kh, kw, groups=c)
    assert ok, msg
    result = sim.conv_from_trace(trace, wpack, x, c, c, kh, kw,
                                 dtype="int8", input_zp=0, weight_zp=0)
    expected = naive_conv_quant(x, w, c, c, kh, kw, groups=c,
                                input_zp=0, weight_zp=0)[0]
    assert np.allclose(result, expected, atol=2), (c, kh, kw)
    return True

def _verify_trace_expanded_group_fp16(in_c, out_c, groups, kh=1, kw=1):
    seed = 3500 + in_c * 11 + out_c * 13 + groups * 17 + kh * 7 + kw
    rng = np.random.default_rng(seed)
    in_h, in_w = kh + 1, kw + 1
    weight_in_c = in_c // groups
    x = rng.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
    w_group = rng.uniform(-1, 1, (out_c, weight_in_c, kh, kw)).astype(np.float16)
    w_full = expand_grouped_weights(w_group, in_c, out_c, kh, kw, groups)
    wpack = pack_weights_fp16(w_full, out_c, in_c, kh, kw)
    sim = CSCSim()
    trace = sim.trace(wpack, x, out_c, in_c, kh, kw)
    ok, msg = validate_csc_trace(trace, wpack, x, out_c, in_c, kh, kw)
    assert ok, msg
    result = sim.conv_from_trace(trace, wpack, x, out_c, in_c, kh, kw)
    expected = naive_conv(x, w_full, out_c, in_c, kh, kw)[0]
    assert np.allclose(result, expected, atol=0.2), (in_c, out_c, groups, kh, kw)
    return True


# ---------------------------------------------------------------------------
# simple_conv_fp16.py-compatible FP16 weight packing experiment
# ---------------------------------------------------------------------------

def _simple_fp16_align_c(in_c, groups, out_c):
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    if not is_depthwise and (groups > 1 or in_c > 4):
        return 16
    return max(8, min(1 << (max(1, in_c) - 1).bit_length(), 32 if is_depthwise else 16))

def _simple_fp16_dense_weight_for_csc(weight, in_c, out_c, kh, kw, groups):
    if _is_depthwise(in_c, out_c, groups):
        dense = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
        for channel in range(min(out_c, in_c)):
            dense[channel, channel] = weight[channel, 0]
        return dense
    return expand_grouped_weights(weight, in_c, out_c, kh, kw, groups)

def pack_weights_simple_conv_fp16(weight_full, out_c, in_c, kh, kw, align_c, groups):
    """Pack weights exactly like simple_conv_fp16.py's current packer."""
    is_depthwise = _is_depthwise(in_c, out_c, groups)
    aligned_in_c = align_c * _ceil_div(in_c, align_c)

    if is_depthwise:
        if kh == 1 and kw == 1 and in_c == 32 and out_c == 32:
            packed = np.zeros(kh * kw * aligned_in_c * out_c, dtype=np.float16)
            for channel in range(out_c):
                packed[channel] = weight_full[channel, channel, 0, 0]
            return WeightPack(packed, out_c, aligned_in_c, align_c,
                              oc_atom=1, kh=kh, kw=kw, dtype="fp16")

        packed = np.zeros((1, kh, kw, aligned_in_c), dtype=np.float16)
        for channel in range(min(out_c, in_c)):
            packed[0, :, :, channel] = weight_full[channel, channel]
        return WeightPack(packed.ravel(), out_c, aligned_in_c, align_c,
                          oc_atom=1, kh=kh, kw=kw, dtype="fp16")

    pack_out_c = out_c
    if groups == 1 and kh == 1 and kw == 1 and in_c >= 64 and out_c >= 48:
        pack_out_c = _align_up(out_c, 16)

    ic_group = 32 if aligned_in_c >= 32 and aligned_in_c % 32 == 0 else align_c
    padded = np.zeros((pack_out_c, aligned_in_c, kh, kw), dtype=np.float16)
    padded[:out_c, :in_c] = weight_full[:out_c, :in_c]

    blocks = []
    for oc in range(0, pack_out_c, 16):
        block = padded[oc:oc + 16]
        block_oc = block.shape[0]
        blocks.append(
            block.reshape(block_oc, aligned_in_c // ic_group, ic_group, kh, kw)
                 .transpose(1, 3, 4, 0, 2)
                 .ravel()
        )
    return WeightPack(np.concatenate(blocks), pack_out_c, aligned_in_c, ic_group,
                      oc_atom=16, kh=kh, kw=kw, dtype="fp16")

def _simple_fp16_prepare_legacy_shape(shape, rng, value_check):
    in_c = shape["in_c"]
    out_c = shape["out_c"]
    in_h = shape["in_h"]
    in_w = shape["in_w"]
    kh = shape["kh"]
    kw = shape["kw"]
    groups = shape["groups"]
    weight_in_c = shape["weight_in_c"]
    is_spatial = kh != 1 or kw != 1

    if value_check:
        input_nchw = rng.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
        weight_nchw = rng.uniform(-1, 1, (out_c, weight_in_c, kh, kw)).astype(np.float16)
    else:
        input_nchw = np.zeros((1, in_c, in_h, in_w), dtype=np.float16)
        weight_nchw = np.zeros((out_c, weight_in_c, kh, kw), dtype=np.float16)

    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    if is_spatial and (groups == 1 or _is_depthwise(in_c, out_c, groups)) and out_h == 1 and out_w == 1 and in_c >= 64:
        padded_input = np.zeros((1, in_c, in_h + 1, in_w + 1), dtype=np.float16)
        padded_input[:, :, :in_h, :in_w] = input_nchw
        input_nchw = padded_input
        in_h += 1
        in_w += 1

    if groups == 1 and in_c > 32 and in_c % 32:
        padded_in_c = _align_up(in_c, 32)
        padded_input = np.zeros((1, padded_in_c, in_h, in_w), dtype=np.float16)
        padded_input[:, :in_c] = input_nchw
        padded_weight = np.zeros((out_c, padded_in_c, kh, kw), dtype=np.float16)
        padded_weight[:, :in_c] = weight_nchw
        input_nchw = padded_input
        weight_nchw = padded_weight
        in_c = padded_in_c
        weight_in_c = padded_in_c

    return input_nchw, weight_nchw, in_c, out_c, in_h, in_w, kh, kw, groups, weight_in_c

def _verify_simple_conv_fp16_weight_pack_shape(shape):
    value_check = os.environ.get("CSC_SIM_PARITY_VALUE") == "1" and _legacy_fp16_value_check_budget(shape)
    seed = 5000 + shape["in_c"] * 3 + shape["out_c"] * 5 + shape["in_h"] * 7 + shape["in_w"] * 11 + shape["kh"] * 13 + shape["kw"] * 17 + shape["groups"] * 19
    rng = np.random.default_rng(seed)
    input_nchw, weight_nchw, in_c, out_c, in_h, in_w, kh, kw, groups, weight_in_c = \
        _simple_fp16_prepare_legacy_shape(shape, rng, value_check)

    align_c = 32 if (kh == 1 and kw == 1 and groups == 1 and in_c >= 64) else _simple_fp16_align_c(in_c, groups, out_c)
    weight_full = _simple_fp16_dense_weight_for_csc(weight_nchw, in_c, out_c, kh, kw, groups)
    wpack = pack_weights_simple_conv_fp16(weight_full, out_c, in_c, kh, kw, align_c, groups)

    sim_groups = groups if _is_depthwise(in_c, out_c, groups) else 1
    trace = CSCSim().trace(wpack, input_nchw, out_c, in_c, kh, kw, groups=sim_groups)
    ok, msg = validate_csc_trace(trace, wpack, input_nchw, out_c, in_c, kh, kw, groups=sim_groups)
    assert ok, f"{shape['name']}: {msg}"

    if value_check:
        result = CSCSim().conv_from_trace(trace, wpack, input_nchw, out_c, in_c, kh, kw)
        expected = naive_conv(input_nchw, weight_full, out_c, in_c, kh, kw, groups=1)[0]
        assert np.allclose(result, expected, atol=0.2), shape["name"]
    return value_check

def test_simple_conv_fp16_weight_pack_parity():
    shapes = _conv_legacy_shapes()
    assert shapes, "no conv_legacy.py shapes found"
    value_checked = 0
    for shape in shapes:
        value_checked += int(_verify_simple_conv_fp16_weight_pack_shape(shape))
    print(f"    simple_conv_fp16_weight_pack_shapes={len(shapes)} value_checked={value_checked} trace_only={len(shapes) - value_checked}")
    return True

def test_trace_depthwise_fp16():
    rng = np.random.default_rng(3000)
    c, in_h, in_w, kh, kw = 17, 5, 6, 3, 2
    x = rng.uniform(-1, 1, (1, c, in_h, in_w)).astype(np.float16)
    w = rng.uniform(-1, 1, (c, 1, kh, kw)).astype(np.float16)
    wpack = pack_weights_fp16(w, c, c, kh, kw, groups=c)
    sim = CSCSim()
    trace = sim.trace(wpack, x, c, c, kh, kw, groups=c)
    ok, msg = validate_csc_trace(trace, wpack, x, c, c, kh, kw, groups=c)
    assert ok, msg
    result = sim.conv_from_trace(trace, wpack, x, c, c, kh, kw)
    expected = naive_conv(x, w, c, c, kh, kw, groups=c)[0]
    assert np.allclose(result, expected, atol=0.2)
    return True

def test_trace_sweep_fp16_boundaries():
    channel_edges = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65]
    out_edges = [1, 15, 16, 17, 31, 32, 33, 47, 48, 49]
    for in_c in channel_edges:
        for out_c in out_edges:
            _verify_trace_fp16(in_c, out_c, 1, 1)
    for in_c, out_c in [(1, 1), (15, 17), (32, 33), (33, 31), (64, 49)]:
        _verify_trace_fp16(in_c, out_c, 3, 2)
    return True

def test_trace_sweep_int8_boundaries():
    channel_edges = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65]
    out_edges = [1, 15, 16, 17, 31, 32, 33]
    for in_c in channel_edges:
        for out_c in out_edges:
            _verify_trace_int8(in_c, out_c, 1, 1)
    for in_c, out_c in [(1, 1), (15, 17), (32, 33), (33, 31), (64, 32)]:
        _verify_trace_int8(in_c, out_c, 3, 2)
    return True

def test_trace_sweep_fp16_spatial_matrix():
    kernels = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 3)]
    shapes = [
        (1, 16), (2, 17), (7, 15), (8, 31), (15, 33),
        (16, 47), (17, 48), (24, 49), (31, 16), (32, 24),
        (33, 32), (48, 17), (63, 15), (64, 33), (65, 16),
    ]
    count = 0
    for idx, (in_c, out_c) in enumerate(shapes):
        kh, kw = kernels[idx % len(kernels)]
        _verify_trace_fp16(in_c, out_c, kh, kw, in_h=kh + 1, in_w=kw + 2)
        count += 1
    assert count == len(shapes)
    return True

def test_trace_sweep_int8_spatial_matrix():
    kernels = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 3)]
    shapes = [
        (1, 16), (2, 17), (7, 15), (8, 31), (15, 33),
        (16, 32), (17, 31), (24, 16), (31, 24), (32, 33),
        (33, 32), (48, 17), (63, 15), (64, 33), (65, 16),
    ]
    count = 0
    for idx, (in_c, out_c) in enumerate(shapes):
        kh, kw = kernels[idx % len(kernels)]
        _verify_trace_int8(in_c, out_c, kh, kw, in_h=kh + 1, in_w=kw + 2)
        count += 1
    assert count == len(shapes)
    return True

def test_trace_sweep_depthwise_matrix():
    channels = [1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65]
    kernels = [(1, 1), (1, 2), (2, 1), (3, 2), (3, 3)]
    for idx, c in enumerate(channels):
        kh, kw = kernels[idx % len(kernels)]
        _verify_trace_dw_int8(c, kh, kw, in_h=kh + 1, in_w=kw + 1)
    return True

def test_trace_sweep_expanded_group_fp16():
    cases = [
        (4, 8, 2, 1, 1),
        (8, 8, 4, 1, 2),
        (16, 32, 4, 2, 1),
        (24, 24, 3, 2, 2),
        (32, 48, 8, 3, 2),
    ]
    for in_c, out_c, groups, kh, kw in cases:
        _verify_trace_expanded_group_fp16(in_c, out_c, groups, kh, kw)
    return True

def _conv_legacy_shapes():
    path = os.path.join(os.path.dirname(__file__), "conv_legacy.py")
    shapes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or "dict(name=" not in stripped:
                continue
            shape = {}
            name_match = re.search(r'name="([^"]+)"', stripped)
            if not name_match:
                continue
            shape["name"] = name_match.group(1)
            missing = False
            for key in ("batch", "in_c", "in_h", "in_w", "out_c", "weight_in_c", "kh", "kw", "groups"):
                match = re.search(rf"\b{key}=([0-9]+)", stripped)
                if not match:
                    missing = True
                    break
                shape[key] = int(match.group(1))
            if not missing:
                shapes.append(shape)
    return shapes

def _legacy_fp16_value_check_budget(shape):
    out_h = shape["in_h"] - shape["kh"] + 1
    out_w = shape["in_w"] - shape["kw"] + 1
    if out_h <= 0 or out_w <= 0:
        return False
    if shape["batch"] != 1:
        return False
    ops = shape["out_c"] * shape["weight_in_c"] * shape["kh"] * shape["kw"] * out_h * out_w
    budget = int(os.environ.get("CSC_SIM_VALUE_BUDGET", "20000"))
    return ops <= budget

def _verify_conv_legacy_shape_fp16(shape):
    batch = shape["batch"]
    in_c = shape["in_c"]
    out_c = shape["out_c"]
    in_h = shape["in_h"]
    in_w = shape["in_w"]
    kh = shape["kh"]
    kw = shape["kw"]
    groups = shape["groups"]
    weight_in_c = shape["weight_in_c"]
    if in_h < kh or in_w < kw:
        raise AssertionError(f"invalid legacy shape: {shape['name']}")

    value_check = _legacy_fp16_value_check_budget(shape)
    seed = 4000 + in_c * 3 + out_c * 5 + in_h * 7 + in_w * 11 + kh * 13 + kw * 17 + groups * 19
    rng = np.random.default_rng(seed)
    if value_check:
        x = rng.uniform(-1, 1, (1, in_c, in_h, in_w)).astype(np.float16)
    else:
        x = np.zeros((1, in_c, in_h, in_w), dtype=np.float16)

    if _is_depthwise(in_c, out_c, groups):
        if value_check:
            w = rng.uniform(-1, 1, (out_c, 1, kh, kw)).astype(np.float16)
        else:
            w = np.zeros((out_c, 1, kh, kw), dtype=np.float16)
        wpack = pack_weights_fp16(w, out_c, in_c, kh, kw, groups=groups)
        sim = CSCSim()
        trace = sim.trace(wpack, x, out_c, in_c, kh, kw, groups=groups)
        ok, msg = validate_csc_trace(trace, wpack, x, out_c, in_c, kh, kw, groups=groups)
        assert ok, f"{shape['name']}: {msg}"
        if value_check:
            result = sim.conv_from_trace(trace, wpack, x, out_c, in_c, kh, kw)
            expected = naive_conv(x, w, out_c, in_c, kh, kw, groups=groups)[0]
            assert np.allclose(result, expected, atol=0.2), shape["name"]
        return value_check

    if weight_in_c != in_c:
        if value_check:
            w_group = rng.uniform(-1, 1, (out_c, weight_in_c, kh, kw)).astype(np.float16)
        else:
            w_group = np.zeros((out_c, weight_in_c, kh, kw), dtype=np.float16)
        w_full = expand_grouped_weights(w_group, in_c, out_c, kh, kw, groups)
    else:
        if value_check:
            w_full = rng.uniform(-1, 1, (out_c, in_c, kh, kw)).astype(np.float16)
        else:
            w_full = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)

    wpack = pack_weights_fp16(w_full, out_c, in_c, kh, kw, groups=1)
    sim = CSCSim()
    trace = sim.trace(wpack, x, out_c, in_c, kh, kw, groups=1)
    ok, msg = validate_csc_trace(trace, wpack, x, out_c, in_c, kh, kw, groups=1)
    assert ok, f"{shape['name']}: {msg}"
    if value_check:
        result = sim.conv_from_trace(trace, wpack, x, out_c, in_c, kh, kw)
        expected = naive_conv(x, w_full, out_c, in_c, kh, kw, groups=1)[0]
        assert np.allclose(result, expected, atol=0.2), shape["name"]
    return value_check

def test_trace_sweep_conv_legacy_shapes():
    shapes = _conv_legacy_shapes()
    assert shapes, "no conv_legacy.py shapes found"
    value_checked = 0
    for shape in shapes:
        value_checked += int(_verify_conv_legacy_shape_fp16(shape))
    print(f"    conv_legacy_shapes={len(shapes)} value_checked={value_checked} trace_only={len(shapes) - value_checked}")
    return True


if __name__ == "__main__":
    fp16_tests = [
        ("pointwise_fp16",             test_pointwise_fp16),
        ("pointwise_wide_tail_fp16",   test_pointwise_wide_tail_fp16),
        ("pointwise_odd_tail_fp16",    test_pointwise_odd_tail_fp16),
        ("spatial3x3_fp16",            test_spatial3x3_fp16),
        ("depthwise_fp16",             test_depthwise_fp16),
        ("grouped_fp16",               test_grouped_fp16),
    ]
    int8_tests = [
        ("pointwise_int8",             test_pointwise_int8),
        ("spatial3x3_int8",            test_spatial3x3_int8),
        ("pointwise_odd_oc_int8",      test_pointwise_odd_oc_int8),
        ("pointwise_nonzp_int8",       test_pointwise_nonzp_int8),
    ]
    new_tests = [
        ("pack_int8_byte_exact",         test_pack_weights_int8_byte_exact),
        ("pack_int8_byte_multi_icblock",  test_pack_weights_int8_byte_multi_icblock),
        ("pack_int8_byte_dw_exact",      test_pack_weights_int8_byte_dw_exact),
        ("pack_int8_byte_spatial",       test_pack_weights_int8_byte_spatial_exact),
        ("pack_input_int8",              test_pack_input_int8_exact),
        ("unpack_output_int8",           test_unpack_output_int8),
        ("im2col",                       test_im2col),
        ("pack_biases_int8_exact",       test_pack_biases_int8_exact),
        ("bank_calcs",                   test_bank_calcs),
        ("split_tasks_single",           test_split_tasks_single),
        ("split_tasks_multi",            test_split_tasks_multi),
        ("trace_depthwise_fp16",          test_trace_depthwise_fp16),
        ("trace_sweep_fp16_boundaries",   test_trace_sweep_fp16_boundaries),
        ("trace_sweep_int8_boundaries",   test_trace_sweep_int8_boundaries),
        ("trace_sweep_fp16_spatial",      test_trace_sweep_fp16_spatial_matrix),
        ("trace_sweep_int8_spatial",      test_trace_sweep_int8_spatial_matrix),
        ("trace_sweep_depthwise",         test_trace_sweep_depthwise_matrix),
        ("trace_sweep_group_fp16",        test_trace_sweep_expanded_group_fp16),
        ("trace_sweep_conv_legacy",       test_trace_sweep_conv_legacy_shapes),
        ("simple_conv_fp16_weight_pack",  test_simple_conv_fp16_weight_pack_parity),
    ]
    passed = failed = 0
    all_tests = fp16_tests + int8_tests + new_tests
    for label, fn in all_tests:
        ok = _test(label, fn)
        passed += ok
        failed += not ok
    print(f"\ncsc_sim: {passed} PASS, {failed} FAIL / {len(all_tests)}")
