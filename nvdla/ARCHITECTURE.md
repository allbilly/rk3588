# NVDLA Convolution Architecture

## Pipeline Overview

The NVDLA convolution core processes an input feature map through 5 sequential hardware blocks:

```
 DRAM ──► CDMA ──► CBUF ──► CSC ──► CMAC ──► CACC ──► SDP ──► DRAM
               (fetch)   (buffer)   (sequence)   (compute)   (accumulate)   (post-process)
```

| Block | NVDLA C-model | Function | conv.py equivalent |
|-------|---------------|----------|-------------------|
| **CDMA** | `cmod/cdma/` | Fetches input feature maps + weights from DRAM into CBUF. Supports pixel/feature/compressed formats. | `pack_nc1hwc2_fp16()` produces data in the format CDMA expects |
| **CBUF** | `cmod/cbuf/` | On-chip SRAM buffer (configurable 2-32 banks, 4-8 KiB each). Stores input slices + weight atoms. | `NPU_CBUF_BANK_SIZE = 32768`, `NPU_CBUF_BANKS = 12` |
| **CSC** | `cmod/csc/` | Convolution Stream Controller. Sequences data + weights from CBUF to CMAC. Manages convolution window sliding, handles direct/Winograd modes. | `REG_CNA_*` registers: data sizes, weight sizes, stride, padding |
| **CMAC** | `cmod/cmac/` | The MAC array: 2048 8-bit MACs (or 1024 16-bit). Systolic array. Two instances: cmac_a + cmac_b. | `REG_CORE_*` registers, weight packing order |
| **CACC** | `cmod/cacc/` | Accumulates partial sums from CMAC. 64 INT48/INT34/FP48 adders. Rounding + saturation. | `REG_CORE_DATAOUT_SIZE_*` |
| **SDP** | `cmod/sdp/` | Post-processing: ReLU, bias, scaling, LUT-based activation (sigmoid, tanh, etc.) | DPU registers handle this in RK3588 |
| **PDP** | `cmod/pdp/` | Pooling (max/avg/min) | Not directly in conv.py |

## RK3588 Differences from NVDLA

The RK3588 NPU is **not** a stock NVDLA — it's a derivative with these modifications:

1. **Register blocks renamed**: NVDLA's CDMA/CSC/CACC → RK3588's **CNA**. CMAC → **CORE**. SDP/PDP → **DPU**.
2. **DRM interface**: Uses `/dev/dri/card1` with DRM ioctls, not `opendla.ko`.
3. **Task-based submission**: Uses `struct_rknpu_task` with a command buffer (register pairs), not direct CSB writes.
4. **No Winograd support**: conv.py only does direct convolution.
5. **12 CBUF banks** (vs 2-32 configurable): fixed at `NPU_CBUF_BANK_SIZE = 32768`.
6. **Limitation**: 1x1 conv with >4 input channels requires manual channel slicing (see `_run_conv2d_channel_sliced`).

## Data Flow (One Convolution Layer)

1. **CPU** packs input + weights into NC1HWC2 format in pinned DMA buffers
2. **CDMA** reads from DRAM into CBUF (triggered by `REG_CNA_FEATURE_DATA_ADDR`)
3. **CSC** reads CBUF entries, sequences data to CMAC in atom-sized chunks
4. **CMAC** performs multiply-accumulate: each MAC cell does `sum += data * weight`
5. **CACC** accumulates partial sums across the convolution window
6. **DPU** (SDP equivalent) applies post-processing and writes output back to DRAM
7. **CPU** reads back output buffer and unpacks from NC1HWC2 to NCHW

## Detailed Block Descriptions (from NVDLA C-model)

### CDMA (`cmod/cdma/`)
- 4 internal sub-modules: CDMA_DC (data), CDMA_WG (weight), CDMA_IMG (image), CDMA_WT (weight read). Only one active at a time.
- Workflow: check CBUF space → issue read request → buffer in shared storage → reorder data → write to CBUF → update CBUF status.

### CSC (`cmod/csc/`)
- Reads CBUF entries scheduled by CDMA
- Sends data + weight atoms to CMAC via `sc2mac_dat_b_transport` / `sc2mac_wt_b_transport`
- Controls convolution window sliding
- Uses `nvdla_stripe_info_t` to mark stripe/channel/layer boundaries
- Supports direct mode (standard conv) and Winograd mode

### CMAC (`cmod/cmac/`)
- `MAC_CELL_NUM = 8`, `RESULT_NUM_PER_MACELL = 8` → 64 MACs per CMAC cell
- Two instances (cmac_a, cmac_b) → 128 MAC cells × 16 bit = 2048 8-bit MACs
- 7-stage pipeline
- Weight shadow registers: `weight_operand_shadow_` → `UpdateWeightFromShadowToActive()` at stripe start
- `do_calc()` → `calculation_int8()` or `calculation_int16()`

### CACC (`cmod/cacc/`)
- Adder array: 64 INT48 / 64 INT34 / 64 FP48 adders
- Partial sum SRAM: 4×96B×32 + 4×64B×32, configurable by mode
- Output: fixed 16 elements per cycle
- Round/saturate before sending to SDP
