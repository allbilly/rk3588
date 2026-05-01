# gemm.py Problems & Improvement Plan

## Summary

`gemm.py` works for a few hardcoded shapes (2×2×1, 64×64×64, 256×256×256) but is suboptimal for general GEMM on RK3588 NPU. The issues stem from using shape whitelists instead of deriving parameters from hardware config (`atomicKSize`), and from known bugs documented in AGENTS.md.

**Key reference repos to cross-check via deepwiki:**
- `nvdla/hw` — C-model (cmod/) for hardware behavior, register definitions
- `nvdla/sw` — UMD compiler for `make_matmul_params()`, `atomicKSize`, C2 derivation

---

## Problem 1: Shape Whitelists Instead of Hardware-Derived Parameters

### Current Behavior

`gemm.py` uses hardcoded shape lists for C2=8 input packing and C2=4 output decoding:

```python
PACK_INPUT = {
    (2, 2, 1): pack_input_2x2x1,
    (64, 64, 64): pack_input_c2_8,
    (256, 256, 256): pack_input_c2_8,
}
DECODE_OUTPUT = {
    (64, 64, 64): decode_output_c2_4,
    (256, 256, 256): decode_output_c2_4,
}
```

Similarly, `no_group_line_off` (which controls GROUP_LINE_OFF register bit) is determined by a 7-shape whitelist:

```python
no_group_line_off = is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or is_matmul_768 or is_matmul_768_2048 or is_matmul_2048
```

### What NVDLA/sw Does (verify via deepwiki)

Use deepwiki to check `nvdla/sw`:
```
deepwiki_ask_question("nvdla/sw",
    "How does make_matmul_params() determine align_in, align_out?
     What is atomicKSize for RK3588? How does compiler derive C2?")
```

Findings from deepwiki (see conversation above):
- `atomicKSize = 32` (from `nvdla_config_large.h`, `NVDLA_MAC_ATOMIC_K_SIZE`)
- For FP16: `C2 = atomicKSize / 2 = 16`
- Compiler sets `pixelFormat = FEATURE_X8` when `memAtomSize=8`, `FEATURE_X32` when `memAtomSize=32`
- `C2 = channelsPerGroup = atomicKSize / 2` (for FP16)

### What NVDLA/hw Confirms (verify via deepwiki)

Use deepwiki to check `nvdla/hw`:
```
deepwiki_ask_question("nvdla/hw",
    "What is PARALLEL_CHANNEL_NUM? How does CSC expect input in NC1HWC2 format?
     What is MAC_CELL_NUM and how does C2 relate to it?")
```

Findings from deepwiki (see conversation above):
- `PARALLEL_CHANNEL_NUM = 64` (MAC array width)
- `MAC_CELL_NUM = 8` cells
- C2=8 matches one MAC cell's width
- Input packing should match `memAtomSize` / `atomicKSize`

### The Issue

`gemm.py` hardcodes C2=8 for only 2 shapes. For RK3588 with `atomicKSize=32` (FP16 C2=16), many more shapes should use C2 packing when `align_in >= 64` (i.e., K ≥ 64 after alignment).

---

## Problem 2: `no_group_line_off` False Generalization

### From AGENTS.md

> `k_exact and m == k` rule (line 291-292): This false generalization disables GROUP_LINE_OFF for shapes like 32x32x32 and 64x99x64. gemm.c only disables GROUP_LINE_OFF for explicit shapes.

### The Real Rule

From `experimental/gemm.c` line 27-28:
```c
if (!is_KN_64 && !is_KN_256 && !is_KN_512 && !is_KN_lg_512 && !is_matmul_768 && !is_matmul_768_2048 && !is_matmul_2048)
    conv_con1 |= CNA_CONV_CON1_GROUP_LINE_OFF(1);
```

**Verify with deepwiki on `nvdla/hw`:**
```
deepwiki_ask_question("nvdla/hw",
    "What does GROUP_LINE_OFF do in CNA_CONV_CON1 register?
     When should it be set/cleared for GEMM operations?
     How does CSC handle line stride vs group mode?")
```

The correct generalization should be:
- **Disable GROUP_LINE_OFF when `align_in >= 64`** (C2 packing is active)
- The whitelist in both `gemm.c` and `gemm.py` is incomplete — verify with deepwiki what the hardware expects

---

## Problem 3: `data_bank` Boundary Bug

### From AGENTS.md

> `data_bank += 1` at exact 32768 boundary (gemm.py line 299): gemm.c does NOT increment data_bank when `input_bytes % 32768 == 0`. This caused 256x256x256 to have CBUF_CON0=0x75 (DATA_BANK=5) instead of 0x84 (DATA_BANK=4), producing md=54 instead of md=5e-5.

### The Bug

`gemm.py` line 302:
```python
data_bank = max(1, min(11, (input_bytes + 32767) // 32768))
```

**Verify CBUF bank calculation with deepwiki on `nvdla/hw`:**
```
deepwiki_ask_question("nvdla/hw",
    "How does CNA_CBUF_CON0 register work?
     What is NPU_CBUF_BANKS and NPU_CBUF_BANK_SIZE?
     How should DATA_BANK and WEIGHT_BANK be calculated?")
```

---

## Problem 4: C2=4 Output But C2=8/16 Input — Asymmetry

### Current State

- Input C2=8 for 64×64×64 and 256×256×256
- Output C2=4 for the same shapes
- All other shapes use row-major (C2=effective width)

### Hardware Constraint (verify with deepwiki)

**Check `nvdla/hw` C-model:**
```
deepwiki_ask_question("nvdla/hw",
    "How does DPU output formatter work?
     What is the C2 value for output in NC1HWC2 format?
     Why is output C2=4 while input can be C2=8 or C2=16?")
```

The output formatter uses C2=4 regardless of input C2 because:
- The output DMA writes 4 channels per cycle (`SURFACE_ADD` stride is based on 4-channel groups)
- This is a hardware constraint, not derived from `atomicKSize`

---

## Improvement Plan

### Step 1: Derive C2 from `atomicKSize` (like NVDLA/sw)

**Verify with deepwiki before implementing:**
```
deepwiki_ask_question(["nvdla/sw", "nvdla/hw"],
    "For RK3588 NVDLA, what is the correct C2 value for FP16 GEMM?
     Is C2=8, C2=16, or does it vary by shape?
     Check nvdla/sw compiler and nvdla/hw C-model for ground truth.")
```

```python
# RK3588 NVDLA config (verify via deepwiki)
ATOMIC_K_SIZE = 32  # from nvdla_config_large.h, verify with deepwiki
C2_INPUT = ATOMIC_K_SIZE // 2  # = 16 for FP16 (not hardcoded 8)
C2_OUTPUT = 4  # hardware constraint for output formatter (verify with nvdla/hw)
```

### Step 2: Generalize Input Packing

Replace the whitelist with a general rule:

```python
def get_input_packer(align_in):
    if align_in >= 64:
        # Use C2_INPUT (16 for RK3588 FP16, per nvdla/sw)
        return pack_input_c2(C2_INPUT)
    return pack_input_row_major

def pack_input_c2(c2):
    """Generalized C2 packing - works for any C2 value (8, 16, etc.)"""
    def pack(m, n, k, a_matrix, in_pack, align_in):
        k_planes = k // c2
        in_pack[:m * k] = a_matrix.reshape(m, k_planes, c2).transpose(1, 0, 2).ravel()
    return pack
```

### Step 3: Generalize `no_group_line_off`

```python
# Disable GROUP_LINE_OFF when C2 packing is active (align_in >= 64)
# VERIFY with deepwiki on nvdla/hw before finalizing
no_group_line_off = align_in >= 64
```

**WARNING**: This is a simplification. The `gemm.c` whitelist suggests certain shapes need GROUP_LINE_OFF disabled even when `align_in < 64`.
**Verify with deepwiki:**
```
deepwiki_ask_question("nvdla/hw",
    "In NV_NVDLA_csc.cpp, when is GROUP_LINE_OFF used?
     What shapes require GROUP_LINE_OFF=0?
     Check SendWeightToMacSequencer for GEMM-specific logic.")
```

### Step 4: Fix `data_bank` Calculation

```python
input_bytes = m * align_in * 2
data_bank = (input_bytes + 32767) // 32768
# Fix: don't increment at exact boundary (per gemm.c, AGENTS.md)
if input_bytes % 32768 == 0 and data_bank > 1:
    data_bank -= 1
data_bank = max(1, min(11, data_bank))
```

### Step 5: Generalize Output Decoding

```python
def get_output_decoder(align_out):
    if align_out >= 64:
        return decode_output_c2_4  # C2=4 for output (hardware constraint)
    return decode_output_linear
```

### Step 6: Verify with Register Comparison Table

For each new shape, build a register comparison table. Use deepwiki to understand unfamiliar registers:

```
deepwiki_ask_question("nvdla/hw",
    "What does CNA_DMA_CON1 LINE_STRIDE control?
     What does CNA_DMA_CON2 SURF_STRIDE do?
     How to verify register values against NVDLA C-model?")
```

| Register | gemm.py | gemm.c | ops_rknn | Verdict |
|----------|---------|--------|----------|---------|
| CBUF_CON0 | ? | ? | ? | check |
| CONV_CON1 | ? | ? | ? | check |
| DMA_CON1 (line_stride) | ? | ? | ? | check |
| DATA_CUBE_NOTCH | ? | ? | ? | check |

Use `gdb -x test.gdb ./build/ops_reg` at `submitTask` breakpoint to capture gemm.c registers.
Use `gdb -x matmul.gdb ./matmul_api` with `python3 dump.py 1` to capture ops_rknn registers.

---

## Deepwiki Verification Checklist

Before implementing, verify these with deepwiki:

- [ ] `nvdla/sw`: What is `atomicKSize` for RK3588? (Answer: 32)
- [ ] `nvdla/sw`: How does `make_matmul_params()` compute `align_in`?
- [ ] `nvdla/hw`: What is `PARALLEL_CHANNEL_NUM`? (Answer: 64)
- [ ] `nvdla/hw`: How does CSC handle C2=8 vs C2=16 input?
- [ ] `nvdla/hw`: What does GROUP_LINE_OFF actually do?
- [ ] `nvdla/hw`: How should CBUF_CON0 DATA_BANK be calculated?

Run these deepwiki queries:
```python
# Check implementation in nvdla/sw
deepwiki_ask_question("nvdla/sw",
    "Show me the make_matmul_params() function.
     How does the UMD compiler decide C2 value for GEMM?
     What is the relationship between atomicKSize and C2?")

# Check implementation in nvdla/hw
deepwiki_ask_question("nvdla/hw",
    "In NV_NVDLA_csc.cpp, how is input data formatted for GEMM?
     Show the NC1HWC2 packing logic. What C2 values are supported?
     How does the C-model handle different C2 values?")
```

---

## Summary of Changes

| Parameter | Current | Improved | Verify With |
|-----------|---------|----------|-------------|
| C2 input | hardcoded C2=8 for 2 shapes | `C2 = atomicKSize/2 = 16` for all `align_in >= 64` | deepwiki `nvdla/sw` + `nvdla/hw` |
| GROUP_LINE_OFF | whitelist of 7 shapes | `no_group_line_off = align_in >= 64` | deepwiki `nvdla/hw` CSC |
| Input packing | dispatch table with 2 entries | `pack_input_c2(16)` for all C2 shapes | deepwiki `nvdla/hw` CMAC |
| Output decode | dispatch table with 2 entries | `decode_output_c2_4` for all `align_out >= 64` | deepwiki `nvdla/hw` DPU |
| `data_bank` | may increment at exact boundary | fixed: no increment at exact boundary | deepwiki `nvdla/hw` CBUF |
| `align_in` | always `max(32, ...)` | derive from `atomicKSize` | deepwiki `nvdla/sw` |

---

## Next Steps

1. **Run deepwiki queries** to verify all assumptions against `nvdla/sw` and `nvdla/hw`
2. Implement the changes in `gemm.py`
3. Test new shapes: 32×32×32, 64×99×64, 128×128×128, 512×512×512
4. Build register comparison tables for failing shapes
5. Cross-reference with `experimental/gemm.c` and `ops_rknn` dumps
6. Update AGENTS.md with new findings
