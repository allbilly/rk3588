# NC1HWC2 Data Format

## What It Is

NC1HWC2 is the native memory layout that NVDLA's CDMA engine expects for input feature maps. It stands for:

```
N  = Batch size
C1 = Outer channel group  (channels / c2)
H  = Height
W  = Width (stride-padded)
C2 = Inner channel width  (usually = align_c)
```

## Why It Exists

The NVDLA **CMAC array** processes multiple input channels in parallel. The MAC array has a fixed width (C dimension: configurable 8-64, RK3588 uses `align_c`). Data must arrive at the MAC array with all channels of one spatial position available simultaneously.

```
Memory layout comparison:

NCHW (standard):  [n][c][h][w] — channels are the outer loop
                  One (h,w) position loads one channel value at a time.
                  MAC array would stall waiting for channel data.

NC1HWC2:          [n][c1][h][w][c2] — inner C2 channels are contiguous
                  One load brings c2 channel values → feeds c2 MACs in parallel.
                  No stalling. Full MAC utilization.
```

## Implementation in conv.py

The function `pack_nc1hwc2_fp16()` at `conv.py:157` implements this transform:

```python
def pack_nc1hwc2_fp16(src, batch, channels, height, width, c2, width_stride, ...):
    c1 = (channels + c2 - 1) // c2
    plane_stride = height * width_stride * c2
    dst = np.zeros((batch * c1 * plane_stride,), dtype=np.float16)
    for n in range(batch):
        for c in range(channels):
            plane = c // c2            # → C1 index
            offset = c % c2            # → C2 index
            ...
            for w in range(width):
                dst_idx = dst_row_base + w * c2 + offset
```

This maps NCHW index `(n, c, h, w)` to NC1HWC2 index `(n, c//c2, h, w, c%c2)`.

## The Width Stride

`width_stride` is NOT the same as `width`. From `compute_conv2d_params()`:

```python
width_stride = align_up_int(in_w, align_c)  # aligned to align_c
```

This means each row is padded to `align_c` granularity. The MAC array processes full aligned rows, and the CDMA reads contiguous `c2`-wide chunks per row.

## The `use_nhwc` Shortcut

When `c_ratio = c2 // channels == 2` and `width_stride >= width`, conv.py uses **NHWC packing** instead of NC1HWC2:

```python
def should_use_nhwc_pack(...):
    c_ratio = c2 // channels if channels > 0 else 0
    use_nhwc = (c_ratio == 2) and (width_stride >= width)
```

This happens when input channels are very small (e.g., 3 channels with `align_c=8` gives `c_ratio=2.67`). NHWC packs as `[h][w][c]` — still channel-inside but without the C1 dimension, saving memory.

## Hardware Origin (from NVDLA CDMA C-model)

In `NV_NVDLA_cdma.cpp`, the CDMA's data loader (`cdma_dc`) reads from DRAM and writes to CBUF. It uses `entry_per_slice` (CBUF entries per slice) and converts the input format to the NC1HWC2 layout that CSC expects. The `PARALLEL_CHANNEL_NUM` constant in `NV_NVDLA_csc.h` matches `c2` and determines how many channels the MAC array processes per cycle.

## Why RK3588 Uses 8/16/32 Alignment

From `compute_conv2d_params()`:

```python
max_align = 32 if is_depthwise else 16
align_c = max(8, min(pow2, max_align))
```

- **Depthwise**: `align_c` up to 32 (each channel is its own group)
- **Non-depthwise**: `align_c` up to 16 (CMAC processes 16 channels in parallel)
- **Minimum**: 8 (smallest MAC array width)
