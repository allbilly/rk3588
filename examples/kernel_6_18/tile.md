# CDMA Input Data Packing — Three Approaches Compared

## 1. CDMA_DC vs CDMA_WG vs CDMA_IMG

```
                     SOFTWARE (CPU/Driver)                         HARDWARE (CDMA in cmod)
                     ──────────────────────                        ─────────────────────────

CDMA_DC:             translateCRSToFeatureData /                   DirectConvDataResponseSequencerCommon()
                     _pack_cdma_dc_feature_input_fp16              reads NC1HWC2 atoms from DRAM,
                     NCHW → NC1HWC2 in DRAM                        packs width-coord atoms into 128B CBUF

CDMA_WG:             (driver packs flat feature data)              WinoConvDataResponseSequencerCommon()
                                                                   reads atoms → composes 4×4×4 sub-cubes

CDMA_IMG:            (driver writes raw pixels)                    ImageConvDataResponseSequencerCommon()
                                                                   pixel format parse → mean/scale/truncate
                                                                   → padding → WriteOneEntryToCbuf()
```

### Mode Selection

```
D_MISC_CFG.CONV_MODE    D_DATAIN_FORMAT.DATAIN_FORMAT     →  Mode      C-model sequencer
─────────────────────    ─────────────────────────────────      ────    ─────────────────────────
DIRECT (0)              FEATURE (0)                            → CDMA_DC  DirectConvData*SequencerCommon
WINOGRAD (1)            FEATURE (0)                            → CDMA_WG  WinoConvData*SequencerCommon
DIRECT (0)              PIXEL (1)                              → CDMA_IMG  ImageConvData*SequencerCommon
```

### Comparison Table

| Aspect | CDMA_DC | CDMA_WG | CDMA_IMG |
|---|---|---|---|
| **Input from DRAM** | Pre-packed NC1HWC2 atoms | Feature atoms | Raw pixels (RGB/YUV/RAW) |
| **C-model sequencer** | `DirectConvData*SequencerCommon` | `WinoConvData*SequencerCommon` | `ImageConvData*SequencerCommon` |
| **Data transform** | NaN flush, precision convert | 4×4×4 sub-cube compose | Mean-sub, scale, truncate, pad |
| **Padding** | N/A (handled by CSC) | left/right | all 4 sides |
| **Width stride** | `width_stride` (aligned) | raw width | raw width |
| **C2 in DRAM** | Driver pre-packs C2 | NA | NA (per-element in HW) |
| **Extra FIFOs** | none | `wino_fetch_data_fifo_[4]`, `wino_req2resp_sync` (480) | `mean_data_read_rsp_fifo_` |

---

## 2. The Three Input Packing Functions

All three do **NC1HWC2 channel-interleave packing** — NCHW → C1×H×W×C2 — for CDMA_DC consumption. This runs on **CPU** before data reaches DRAM.

### 2a. `translateCRSToFeatureData` — `nvdla/sw/umd/` (C++)

```cpp
template <typename IT, typename RT>
static Weights translateCRSToFeatureData(WeightDims crsDims, Weights srcData, int channelsPerGroup = 0)
{
    int cf  = (channelsPerGroup != 0) ? channelsPerGroup :
              (sizeof(RT) == 1) ? 32 : 16;    // C2 = 32 (INT8) or 16 (FP16/INT16)
    int cfg = crsDims.numChannels / cf;         // C1 = num full groups
    int cp  = crsDims.numChannels % cf;         // partial channels
    int cpg = cp ? 1 : 0;                       // has partial group?

    for (ind_cfg = 0; ind_cfg < cfg; ind_cfg++)               // C1 loop
        for (ind_r = 0; ind_r < H; ind_r++)                    // H loop
            for (ind_s = 0; ind_s < W; ind_s++)                // W loop
                for (ind_c = ind_cfg*cf; ind_c < ind_cfg*cf+cf; ind_c++)  // C2 loop
                    *pDest++ = pSrc[ind_s + W*(ind_r + H*ind_c)];

    if (cpg)  // partial group: same loops, pad remainder with 0
        for (ind_r, ind_s, ind_c over cf)
            *pDest++ = (ind_c < cfg*cf+cp) ? pSrc[...] : 0;
}
```

- Template on `IT` (input type) and `RT` (return type) — arbitrary precision
- Loop order: **C1 → H → W → C2** (C1-outermost)
- C2 = 32 for INT8, 16 for FP16/INT16
- No width stride padding (raw `W`)
- Named "CRS" = Channel-Row-Column = NCHW without batch

### 2b. `_pack_cdma_dc_feature_input_fp16` — `conv_new.py` (Python)

```python
def _pack_cdma_dc_feature_input_fp16(input_nchw, p):
    if p["use_nhwc"]:
        return input_nchw.transpose(1, 2, 0).ravel()    # NHWC bypass

    c2 = p["input_pack_c2"]     # 2 (ic=1), 8 (1<ic≤4), else align_c
    c1 = ceil(max(ic, align_c) / c2)
    padded = np.zeros((c1*c2, H, width_stride))
    padded[:ic, :, :in_w] = input_nchw
    return padded.reshape(c1, c2, H, width_stride)     # C1×C2×H×W
           .transpose(0, 2, 3, 1).ravel()              # → C1×H×W×C2
```

- Hardcoded FP16
- Uses `width_stride = align(in_w, ceil(16/align_c))` — **width padded**
- C2 = 2, 8, or `align_c` (RK3588-specific)
- NHWC bypass mode for certain channel layouts
- Vectorized numpy (no explicit loops)

### 2c. `pack_input_like_mesa` — `conv_mesa.py` (Python)

```python
def pack_input_like_mesa(input_nhwc, ...):
    # single_channel bypass
    if ic == 1:
        for x in range(W):
            for y in range(max(H, 16)):
                raw[n] = pixel if y < H else zero_point

    # general path
    for u in range(ceil(ic/16)):              # C1 loop
        for x in range(W):                     # W loop (NOT H!)
            for y in range(H):                 # H loop
                for c in range(16):            # C2 = 16
                    raw[n] = (int(pixel) - 0x80) & 0xFF   # signed uint8
```

- uint8 quantized with zero-point subtraction (`-0x80`)
- C2 = **16** (FEATURE_ATOMIC_SIZE) fixed
- C1 = `ceil(ic/16) × 2` (×2 for sign byte)
- Loop order: **C1 → W → H → C2** (W **before** H — different layout!)
- No width stride padding
- Explicit nested loops

---

## 3. Side-by-Side Comparison

| Aspect | `translateCRSToFeatureData` | `_pack_cdma_dc_feature_input_fp16` | `pack_input_like_mesa` |
|---|---|---|---|
| **Language** | C++ template | Python numpy | Python loops |
| **Input shape** | CRS = (C, H, W) 3D | NCHW = (C, H, W) 4D | NHWC = (1, H, W, C) 4D |
| **Output layout** | C1×H×W×C2 | C1×H×W×C2 (or HWC) | C1×W×H×C2 (W-first!) |
| **C2** | 32 (INT8) / 16 (FP16) | 2, 8, or `align_c` | 16 |
| **C1** | `numChannels / cf` + partial | `ceil(max(ic, align_c) / c2)` | `ceil(ic/16) × 2` |
| **Width pad** | no | yes (`width_stride`) | no |
| **Value transform** | `RT(value)` cast | raw FP16 | `(uint8 - 0x80) & 0xFF` |
| **Zero pad** | explicit partial-group loop | implicit via `np.zeros` | `zero_point - 0x80` |
| **Special paths** | none | NHWC bypass mode | single_channel ravel, ic==1 width-major |
| **Loop structure** | `Cfg→R→S→C` nested loops | vectorized (no explicit loops) | `u→x→y→c` nested loops |
| **Purpose** | Generic (weights/bias/scale) | Feature (activation) only | Feature (activation) uint8 |

### Memory Layout Difference

```
translateCRSToFeatureData:     [C1_group][H][W][C2_chs]      C1 major, then H, W, C2
_pack_cdma_dc_feature_input:  [C1_group][H][W][C2_chs]      same layout via transpose
pack_input_like_mesa:          [C1_group][W][H][C2_chs]      W BEFORE H — DIFFERENT!
```

The Mesa loop order `u→x→y` = C1→W→H means the **same C1 group** has all width positions contiguous, then height. The other two pack height first. This is a **genuine difference** in memory layout — the data in DRAM is arranged differently.

---

## 4. Assessment: Which Is Cleanest?

| Rank | Function | Why |
|---|---|---|
| **1** | `translateCRSToFeatureData` | Most generic (templated, any precision), clean loop structure, no special cases. Produces standard C1×H×W×C2 layout with C2 matching HW atom size. |
| **2** | `_pack_cdma_dc_feature_input_fp16` | Concise (6 lines), vectorized, handles RK3588-specific width stride. But hardcoded FP16, has special NHWC bypass, and C2=2/8 is RK-specific not standard NVDLA. |
| **3** | `pack_input_like_mesa` | Most complex (multiple special cases), W-before-H layout differs from NVDLA standard, manual loops. Necessary for uint8 quantized path with sign-byte doubling. |

### Can `translateCRSToFeatureData` Replace the Other Two?

**Yes for `_pack_cdma_dc_feature_input_fp16`** — with minor adaptation:
- Add `width_stride` parameter (stride padding for alignment)
- Keep C2=2/8 RK-specific sizing as a `channelsPerGroup` override (the function already accepts this parameter!)
- Set `channelsPerGroup = input_pack_c2` → works directly

```cpp
// Equivalent to _pack_cdma_dc_feature_input_fp16 using translateCRSToFeatureData:
Weights result = translateCRSToFeatureData<uint16_t, uint16_t>(
    {in_h, in_w, in_c}, srcData,
    channelsPerGroup = input_pack_c2   // 2, 8, or align_c
);
// Missing: width_stride padding — would need a post-step or modified version
```

**No for `pack_input_like_mesa`** — because:
- `pack_input_like_mesa` produces **W-before-H** (`u→x→y→c` loop), while `translateCRSToFeatureData` produces **H-before-W** (`cfg→r→s→c` loop). These are incompatible memory layouts.
- Mesa's C1×2 (sign byte doubling) has no equivalent — `translateCRSToFeatureData` treats each element as one value, not two interleaved bytes.
- The `(pixel - 0x80)` signed conversion is quantization-specific, absent in `translateCRSToFeatureData`.

### Verdict

`translateCRSToFeatureData` can **replace `_pack_cdma_dc_feature_input_fp16`** (same conceptual layout, just need width stride parameter). It **cannot replace `pack_input_like_mesa`** without significant modification because Mesa uses a different loop order and uint8 sign-byte doubling.

`translateCRSToFeatureData` is the **cleanest** — generic, well-structured, with no special-case bypasses. It's the canonical NVDLA reference implementation for NC1HWC2 packing.
