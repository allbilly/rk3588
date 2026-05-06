# Weight Packing — How CMAC Consumes Weights

## The Problem

The NVDLA CMAC array is a **systolic array** of MAC cells. Each cell has its own weight register. Weights must arrive at the correct cell at the correct cycle. The order in memory determines which MAC cell gets which weight.

## The 3 Weight Layouts in conv.py

conv.py supports 3 weight packing variants, selected by `pack_conv_weights_fp16()` at line 262:

### 1. Default Layout (`_pack_default`)

```python
kernel_stride = kh * kw * spatial_stride
total = out_c * kernel_stride
for oc in range(out_c):              # outer: output channels
    for kh_idx in range(kh):
        for kw_idx in range(kw):     # spatial loop
            for ic in range(in_c):    # inner: input channels
                # dst_idx = oc*kernel_stride + (kh*kw)*spatial_stride + (ic//c2)*c2 + (ic%c2)
```

**Memory order**: `OC → KH → KW → IC` grouped by `c2`

This is the standard layout: all weights for one output channel are contiguous. The CMAC processes one output channel at a time, streaming all input channels through.

### 2. KH-Major Layout (`_pack_kh_major`)

```python
spatial_stride = out_c * ((in_c + c2_out - 1) // c2_out)
for kh_idx in range(kh):
    for kw_idx in range(kw):
        for oc in range(out_c):       # output channels inside spatial
            ...
```

**Memory order**: `KH → KW → OC → IC` grouped by `c2`

Weights for different output channels at the *same spatial position* are adjacent. This is needed when the hardware's weight-reuse pattern processes multiple output channels simultaneously (seen in YOLO-style models).

**When used** (from `_KH_MAJOR_SHAPES`):
```python
(6, 3, 2, 1), (6, 3, 2, 3), (6, 3, 2, 5),  # YOLO conv heads
(6, 3, 3, 1), (6, 3, 3, 3), (6, 3, 3, 5),  # YOLO standard/depthwise
(16, 16, 3, 3), (4, 4, 3, 3), (6, 1, 3, 3) # various observed shapes
```

### 3. Depthwise Spatial-Major (`_pack_dw_spatial_major`)

```python
spatial_stride = c2_out
for kh_idx in range(kh):
    for kw_idx in range(kw):
        for oc in range(out_c):
            # src_idx = ((oc * in_c + oc) * kh + kh_idx) * kw + kw_idx
```

**Memory order**: `KH → KW → OC` (no IC dimension — depthwise has 1 IC per OC)

For depthwise conv (groups == in_c == out_c), each output channel has exactly 1 input channel. Weights are stored spatially with all output channels at each (kh, kw) position adjacent.

## CMAC Weight Loading (from NVDLA C-model)

### Weight Shadow Registers

The CMAC has double-buffered weight registers:

```cpp
// NV_NVDLA_cmac.cpp
nvdla_sc2mac_weight_if_t weight_operand_shadow_[MAC_CELL_NUM];
// At stripe start:
if (payload->pd.nvdla_stripe_info.stripe_st == 1)
    UpdateWeightFromShadowToActive();
```

- **Shadow registers** are written by CSC during current stripe
- **Active registers** feed the MAC array
- At stripe boundary, shadow → active atomically

### Weight Transport Protocol

```cpp
// sc2mac_wt_b_transport receives weight payload
struct nvdla_sc2mac_weight_if_t {
    uint8_t  data[128];    // 128 bytes = 64 fp16 values
    uint64_t mask[2];      // weight mask bits for sparsity
    uint16_t sel;          // which MAC cell (0..MAC_CELL_NUM-1)
};
```

The `sel` field routes each weight atom to the correct MAC cell. `MAC_CELL_NUM = 8`, each cell holds 8 results → 64 weights per cycle × 2 instances × 2 (int8 packing) = 2048 8-bit MACs.

### Why Weight Packing Matters

From the CMAC's perspective:
```
Default layout:    CMAC gets IC groups for one OC → accumulates across IC
KH-major layout:   CMAC gets multiple OCs at same spatial position → weight broadcast
Depthwise layout:  CMAC gets one weight per (kh,kw) position → pointwise multiply
```

The `c2_out` parameter (= `align_c`) controls how many input channels are grouped into one memory transaction. This must match `PARALLEL_CHANNEL_NUM` in the hardware.

## The 16KB Reserved Offset

In `_npu_submit()`:
```python
wt_ptr = ctypes.addressof(...) + REGCMD_RESERVED  # REGCMD_RESERVED = 16384
E(rk.REG_CNA_DCOMP_ADDR0, (weights_dma + REGCMD_RESERVED) & 0xFFFFFFFF)
```

The first 16KB of the weight buffer is reserved (likely for weight decompression metadata or WMB = Weight Mask Bits). Actual weight data starts at offset 16384. From NVDLA's CDMA, `D_WMB_BYTES` and `D_WMB_ADDR_LOW` configure the weight mask bits region.

## NVDLA Sources for Deeper Study

- **CMAC weight loading**: `NV_NVDLA_cmac.cpp` → `sc2mac_wt_b_transport`, `do_calc`, `calculation_int8/16`
- **CSC weight sending**: `NV_NVDLA_csc.cpp` → `SendWeightToMacSequencerDirectConvCommon`, `SendWeightToMacSequencerWinoConvCommon`
- **Weight compression**: `get_decompressed_weight()` in CSC for compressed format
- **CBUF weight layout**: CMOD weight packing logic in `NV_NVDLA_csc::readWeightFromCBUF()`
