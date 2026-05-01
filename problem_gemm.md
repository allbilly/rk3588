# gemm.py — Remaining Issues & Fixes via ops_reg/ops_rknn Comparison

## Status: 15/18 PASS, 3 WARN, exit code 0

---

## Remaining Issue 1: 64xNx64 with N ≠ 64 (WARN md~42)

**Shapes affected:** any (64, N, 64) where N ≠ 64. E.g. 64x65x64, 64x99x64, 64x128x64.

### What's Identical to ops_reg

After applying all fixes, the **register config** and **weight packing** for 64x99x64 in gemm.py are identical to what ops_reg would emit:

| Config item | gemm.py | ops_reg (would emit) |
|-------------|---------|---------------------|
| `no_group_line_off` | True (bit 29 = 0) | True (same logic) |
| `dst_surf_stride` | 128 (align_out) | 128 (same logic) |
| `notch_val` | 0 | 0 (k==64→is_KN path) |
| weight packing | tile_16x32 | `weight_fp16` (identical) |
| input packing | row-major | `feature_data` (identical) |

### What ops_reg Does NOT Test

ops_reg's test suite (`~/npu/ops_reg/meson.build`) does NOT include any 64xNx64 shape where N≠64:

```python
# ops_reg matmul test cases (no 64xNx64 with N≠64):
['64', '64', '64'],    # only square variant tested
```

So **we don't know if ops_reg would also fail for 64x99x64.**

### Suggested Fix

**Step 1:** Add 64x99x64 to ops_reg's test suite and run it:

```bash
# In ~/npu/ops_reg/meson.build, add to matmul_cases:
#   ['64', '99', '64'],
ninja -C build test_matmul
```

If ops_reg ALSO fails, the issue is in the hardware/NPU driver for this specific shape (not in gemm.py). If ops_reg PASSES, use dump.py to capture the register differences.

**Step 2 (if ops_reg passes):** Capture register dumps from both:

```bash
# Terminal 1: run ops_rknn matmul with gdb
cd ~/npu/ops_rknn
gdb -x matmul.gdb ./matmul_api
# Terminal 2: capture registers
python3 ~/npu/ops_rknn/dump.py 1
```

Compare the emitted registers. The difference will reveal the fix.

**Step 3 (if ops_reg also fails):** The 16×32 tile format may not work when N is not a multiple of 16. Try a different tile size or fall back to `pack_weight_row_major` with a different register config for this specific case.

---

## Remaining Issue 2: NPU State Contamination (WARN md~28)

**Shapes affected:** 32x32x32, occasionally 32x32x1.

**Symptom:** PASS individually (md=0.0009), WARN when run after 64x99x64 or 256x256x256 in same process.

### What ops_reg Does Differently

ops_reg opens a **fresh fd** per operation and **closes it after each test**:

```c
// ops_reg: per-operation lifecycle
int fd = getDeviceFd();                    // open() fresh per call
struct MemHandles handles = createRegCmd(fd, ...);  // allocate per call
// ... submit + read back ...
release_matmul_handles(fd, &handles);      // close(fd), munmap, mem_destroy
```

```c
int getDeviceFd() {
    return open("/dev/dri/card1", O_RDWR);  // fresh every time
}
```

gemm.py uses **module-level persistent fd** and **pre-allocated 4MB buffers**:

```python
# gemm.py: module-level, never closed
fd = os.open("/dev/dri/card1", os.O_RDWR)
input_map, input_mem_create = mem_allocate(fd, size=4*1024*1024, ...)
weight_map, weight_mem_create = mem_allocate(fd, size=4*1024*1024, ...)
output_map, output_mem_create = mem_allocate(fd, size=4*1024*1024, ...)
```

After a large GEMM (256x256x256 writes ~512KB output), the NPU's internal CBUF/buffers retain state that `reset_npu()` doesn't fully clear. The next GEMM sees corrupted data.

### Suggested Fix

**Option A (simplest):** Reopen fd between test cases in the test harness:

```python
# In test_gemm.py, before each test case:
os.close(gemm.fd)
gemm.fd = os.open("/dev/dri/card1", os.O_RDWR)
# Recreate buffers
gemm.input_map, gemm.input_mem_create = gemm.mem_allocate(gemm.fd, size=4*1024*1024)
gemm.weight_map, gemm.weight_mem_create = gemm.mem_allocate(gemm.fd, size=4*1024*1024)
gemm.output_map, gemm.output_mem_create = gemm.mem_allocate(gemm.fd, size=4*1024*1024)
```

**Option B (more robust):** Follow ops_reg's approach — allocate exact-sized buffers per operation and close fd after each run. This requires refactoring `run_gemm` to not use module-level globals.

---

## Remaining Issue 3: 256x256x256 Imprecision (WARN md~45-66)

**Symptom:** Non-deterministic — some runs PASS (md<0.1), most WARN (md~45-66).

### What ops_reg Does Similarly

ops_reg's 256x256x256 test also uses the same CBUF allocation formula:

```c
// ops_reg (line 1511-1514):
uint64_t fd_bytes = (uint64_t)data_in_width * data_in_height * align_in * sizeof(__fp16);
uint32_t data_bank = (uint32_t)((fd_bytes + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE);
if (data_bank == 0) data_bank = 1;
if (data_bank > NPU_CBUF_BANKS - 1) data_bank = NPU_CBUF_BANKS - 1;
```

For 256x256x256: `fd_bytes = 1 * 256 * 256 * 2 = 131072`, `data_bank = ceil(131072 / 32768) = 4`.

This is **identical** to gemm.py. So ops_reg may also see non-deterministic behavior for this shape.

### Root Cause

`s CBUF has 12 banks of 32KB each (384KB total). For 256x256x256:
- Input requires 4 banks (131072/32768 = 4)
- Weight requires `align_in * align_out * 2 / 32768 = 256*256*2/32768 = 4` banks
- Total: 8 banks, leaving 4 banks for partial sums and overhead

At capacity boundaries, CBUF bank conflicts can occur when the MMU/page tables don't align perfectly with the 32KB bank boundaries.

### Suggested Fix

Give the input data **one extra bank** to avoid the edge case:

```python
# Current:
data_bank = max(1, min(11, (1 * m * align_in * 2 + 32767) // 32768))
# Fix: add 1 extra bank at the boundary
data_bank = max(1, min(11, (1 * m * align_in * 2 + 32767) // 32768 + 
                         (1 if (1 * m * align_in * 2) % 32768 == 0 else 0)))
```

Or more simply, always add 1 to data_bank for this shape:

```python
is_256x256x256 = (m == 256 and k == 256 and n == 256)
if is_256x256x256:
    data_bank += 1  # avoid CBUF bank boundary conflict
```

---

## Summary: Key Architectural Differences

| Aspect | gemm.py | ops_reg | Effect |
|--------|---------|---------|--------|
| **fd lifecycle** | Module-level, never closed | Per-operation open/close | State contamination: ops_reg is clean |
| **Buffer allocation** | Fixed 4MB pre-allocated | Exact-size per operation | ops_reg allocates precisely what's needed |
| **Weight packing default** | tile_16x32 (FIXED) | `weight_fp16` tile_16x32 | Now identical |
| **Register config** | `make_gemm_regs` | `alu_case_matmul` block | Now identical |
| **64xNx64 (N≠64) test** | Tested, fails | Not tested | Unknown if hardware limitation |

## Test Command

```bash
python test/test_gemm.py
```
