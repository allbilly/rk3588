# Annotated Convolution Dataflow

This traces one convolution call through `conv.py`, mapping each step to NVDLA hardware blocks.

## Example: `in_c=2, out_c=2, kh=1, kw=1, input=4x4`

### Step 1: Parameter Computation

`compute_conv2d_params(2, 2, 1, 1, (4, 4))` at `conv.py:315`

```
is_depthwise=False (groups=1, in_c != out_c)
align_c = 8          (pow2 of 2, clamped to max_align=16, min 8)
align_out_c = 16     (aligned to 16)
width_stride = 8     (align_up_int(4, 8))
```

**NVDLA equivalent**: These parameters determine CBUF entry sizing (CDMA `D_ENTRY_PER_SLICE`), MAC array width (CMAC `PARALLEL_CHANNEL_NUM`), and accumulator sizing (CACC).

### Step 2: Data Packing (CDMA input format)

`pack_nc1hwc2_fp16()` at `conv.py:157`

```
Input shape:  (1, 2, 4, 4)  NCHW
c2 = align_c = 8
c1 = (2+8-1)//8 = 1
plane_stride = 4 * 8 * 8 = 256 (if width_stride=8)
```

This produces data in the format CDMA's `cdma_dc` sub-module expects. The CDMA reads this from DRAM at the address in `REG_CNA_FEATURE_DATA_ADDR` and writes it into CBUF.

### Step 3: Weight Packing (CDMA weight format)

`pack_conv_weights_fp16()` at `conv.py:262`

```
out_c=2, in_c=2, kh=1, kw=1, c2=8
Not depthwise (groups=1, in_c != out_c)
Not KH-major (not in _KH_MAJOR_SHAPES)
→ _pack_default

spatial_stride = 8 * ((2+7)//8) = 8 * 1 = 8
kernel_stride = 1*1*8 = 8
total = 2 * 8 = 16 fp16 values
```

Weights are packed OC-major. Each kernel (OC) has `in_c` channels padded to `align_c=8`. This matches what the CMAC's weight shadow registers expect per `sc2mac_wt_b_transport` in `NV_NVDLA_cmac.cpp`.

### Step 4: Register Programming (CSC + CMAC + CACC + SDP)

`build_conv2d_regs()` at `conv.py:386`

Key register writes (from dry run output):

```
CNA[0x100C] = 0x00000880   conv_con1: precision=int16, no depthwise, 1x1 hint
CNA[0x1010] = 0x00000014   feature_grains = 5 (in_h=4 + kh=1)
CNA[0x1020] = 0x00080004   width=8, height=4
CNA[0x1024] = 0x00010001   data_in_channel_real=1, data_in_channel_aligned=1
CNA[0x1028] = 0x00000004   out_w = 4
CNA[0x102C] = 0x00000010   out_atoms = 16 (4x4 output)
CNA[0x1030] = 0x00000020   weight_bytes_total = 32
CNA[0x1034] = 0x00000010   weight_bytes_per_kernel = 16
CNA[0x1038] = 0x00010001   kernel_w=1, kernel_h=1, weight_kernels=1
...
CORE[0x3010] = 0x00000200   depthwise=0, precision=int16
CORE[0x3014] = 0x00030003   out_h-1=3, out_w-1=3
CORE[0x3018] = 0x0000000F   out_channel_field=15 (16-1)
...
DPU[0x4004] = 0x0000000E   s_pointer: en=BS|BN|EW sub-blocks
DPU[0x400C] = 0x0000007E   feature_mode_cfg
DPU[0x4020] = <dma_addr>   output buffer address
...
PC[0x0008] = 0x0000000D   operation_enable: kick off
```

### Step 5: Hardware Execution

**NVDLA timeline** (from C-model behavior):

1. **CDMA phase**: `cdma_dc` reads input from DRAM → writes to CBUF. Uses `line_stride` and `surf_stride` to navigate NC1HWC2 rows. `fetch_grain` (= feature_grains) determines how many slices are fetched before CSC is notified.

2. **CSC phase**: Reads CBUF entries → sends atoms to CMAC. For each convolution window position, CSC sends one data atom (align_c channels × 1 spatial position) and one weight atom (1 kernel × align_c channels).

3. **CMAC phase**: Each MAC cell receives data + weight via `sc2mac_dat_b_transport` / `sc2mac_wt_b_transport`. `do_calc()` runs multiply-accumulate. Since this is 1x1, no kernel window sliding — just pointwise multiply.

4. **CACC phase**: Accumulates CMAC results. Since this is a simple 1x1 with 2 input channels, the accumulator sums the 2 partial products and saturates/rounds.

5. **SDP/DPU phase**: Applies post-processing (bypass since conv.py uses pure convolution). Writes output back to DRAM via BDMA at `REG_DPU_DST_BASE_ADDR`.

### Step 6: Output Unpacking

`unpack_nc1hwc2_fp16()` at `conv.py:270`

```
unpack_c2 = 8
result shape: (1, 2, 4, 4) reconstructed from NC1HWC2
```

## Pipeline Diagram (NVDLA C-model View)

```
Time →
CDMA: [fetch input stripe 0] [fetch input stripe 1] ...
        ↓ CBUF
CSC:  [read atom 0] [read atom 1] [read atom 2] ...
        ↓ data + weight atoms
CMAC: [calc stripe 0] [calc stripe 1] ...
        ↓ partial sums
CACC: [accum stripe 0] [accum stripe 1] ...
        ↓ accumulated results
SDP:  [post-process stripe 0] ...
        ↓ BDMA write
DRAM: [output cube ready]
```

Each stage pipelines: while CMAC processes stripe N, CSC reads stripe N+1, and CDMA fetches stripe N+2.

## Key NVDLA Files to Trace

| Stage | NVDLA file | Key functions |
|-------|-----------|---------------|
| CDMA fetch | `cmod/cdma/NV_NVDLA_cdma.cpp` | `cdma_dc::process()`, `cdma_wg::process()` |
| CBUF write | `cmod/cbuf/NV_NVDLA_cbuf.cpp` | `write_entry()`, `read_entry()` |
| CSC sequence | `cmod/csc/NV_NVDLA_csc.cpp` | `SendDataToMac()`, `SendWeightToMacSequencerDirectConvCommon()` |
| CMAC calc | `cmod/cmac/NV_NVDLA_cmac.cpp` | `do_calc()`, `calculation_int8/16()` |
| CACC accum | `cmod/cacc/NV_NVDLA_cacc.cpp` | `accumulate()`, `saturate_and_round()` |
| SDP process | `cmod/sdp/NV_NVDLA_sdp.cpp` | `process()` |
| Output write | `cmod/bdma/NV_NVDLA_bdma.cpp` | `write_to_dram()` |
