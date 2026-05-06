# NVDLA Virtual Platform (VP) — Deep Dive

## How NVDLA VP Works

The NVDLA VP is a **SystemC + QEMU cosimulation** that simulates a complete ARMv8 SoC with NVDLA hardware. It is NOT a standalone NVDLA simulator — it is a full-system emulator that runs the actual Linux kernel and NVDLA drivers.

### Architecture Stack

```
┌─────────────────────────────────────────────────────────┐
│  User Space: nv_runtime + test applications              │
├─────────────────────────────────────────────────────────┤
│  Kernel Space: Linux 4.13.3 + NVDLA KMD (opendla.ko)    │
├─────────────────────────────────────────────────────────┤
│  QEMU ARMv8 Emulator (GreenSocs QBox TLM-2.0 wrapper)   │
├─────────────────────────────────────────────────────────┤
│  NVDLA C-model SystemC Module (from nvdla/hw cmod/)     │
├─────────────────────────────────────────────────────────┤
│  SystemC 2.3.0a Simulation Kernel                        │
└─────────────────────────────────────────────────────────┘
```

**How the layers connect:**

1. **QEMU** emulates an ARMv8 "virt" machine with CPU, GIC, UART, virtio devices. It is NOT a separate process — QEMU is compiled as a C++ library and linked into the SystemC simulation binary.

2. **GreenSocs QBox** wraps QEMU as a TLM-2.0 SystemC module. The QEMU CPU's CSB bus transactions (register reads/writes) are intercepted and forwarded to the NVDLA C-model over TLM-2.0 sockets.

3. The **NVDLA C-model** (`hw/cmod/*`) is compiled as a SystemC module (`cmod_top`). Each hardware block (CDMA, CSC, CMAC, CACC, SDP, PDP, BDMA, etc.) is a SystemC process. The register programming model in the C-model exactly matches the Verilog RTL register layout.

4. **SystemC 2.3.0a** is the simulation kernel that drives the event loop. Every register write triggers SystemC delta-cycles that propagate through the C-model's state machines.

### Key Files in the VP

| File | Purpose |
|------|---------|
| `src/aarch64_toplevel.cpp` | Top-level SystemC module: instantiates QBox + NVDLA C-model |
| `conf/aarch64_nvdla.lua` | Lua config: kernel image path, driver rootfs, memory map |
| `models/nvdla/CMakeLists.txt` | Compiles NVDLA C-model into the VP binary |
| `docker/Dockerfile` | Pre-built Docker image (Ubuntu 16.04 + all tools) |
| `fpga/aws-fpga/` | AWS F1 FPGA variant (not simulation) |

### What the VP Actually Simulates

- **Register-accurate NVDLA**: every CSB register write is handled by the C-model's register logic
- **Cycle-counted operations**: the C-model tracks pipeline cycles
- **DRAM read/write**: NVDLA's DBB memory accesses go through QEMU's memory model
- **Interrupts**: when NVDLA completes a layer, the C-model asserts an interrupt line to QEMU
- **The VP does NOT simulate**: timing of external DRAM, AXI bus contention, clock gates, power states

## What Can Be Learned from VP for `conv.py`

### 1. Register Programming Sequences

The VP test traces (`verif/traces/traceplayer/conv_8x8_fc_int16/input.txn`) contain the exact register write sequences for real convolution layers. Each line shows:

```
write_reg <misc:16><addr:16><data:32>   # write register
read_reg  <misc:16><addr:16><mask:32><expected:32>  # verify
```

These traces are the "ground truth" for:
- Register programming order (which block gets configured first)
- Which registers are read back for verification
- How `D_OP_ENABLE` is used to kick off operations
- Double-buffer group handling (PRODUCER/CONSUMER pointer)

### 2. CBUF Management Patterns

The VP's C-model code in `NV_NVDLA_cdma.cpp` and `NV_NVDLA_csc.cpp` shows exactly how data is tiled across CBUF banks. The `entry_per_slice`, `fetch_grain`, and `data_bank` calculations in `conv.py:compute_conv2d_params()` directly correspond to NVDLA's CBUF management logic.

### 3. DMA Addressing

The `line_stride` and `surf_stride` registers in conv.py (`REG_CNA_DMA_CON1`/`CON2`) directly match CDMA's `D_LINE_STRIDE`/`D_SURF_STRIDE`. The VP C-model shows how these address calculations work:

- `line_stride`: bytes between consecutive rows in the CBUF (NC1HWC2 aligned)
- `surf_stride`: bytes between consecutive "surfaces" (groups of rows)

### 4. Weight Compression Flow

The 16KB reserved offset in conv.py's weight buffer (`REGCMD_RESERVED = 16384`) corresponds to NVDLA's WMB (Weight Mask Bits) header. The VP C-model's `get_decompressed_weight()` function in CSC shows how compressed weights are unpacked.

### 5. MAC Array Behavior

The C-model's `NV_NVDLA_cmac.cpp` shows the exact MAC cell arrangement, weight shadow register timing, and stripe boundary handling. This validates conv.py's weight packing decisions for the 3 layout variants.

## How to Use VP for conv.py Debugging (On x86)

### Approach 1: Compare Register Sequences

```bash
# Run conv.py in dry-run mode to get register dump
cd /path/to/rk3588
python examples/conv.py  # produces register list

# Compare against NVDLA test trace
cat nvdla/hw/verif/traces/traceplayer/conv_8x8_fc_int16/input.txn

# Key comparisons:
# - Which registers are written vs. which NVDLA writes
# - Register value differences (align_c, stride, etc.)
# - Missing registers (NVDLA has per-block D_OP_ENABLE)
```

### Approach 2: Cross-Reference Register Bitfields

```bash
# Linux kernel rocket_registers.h (ground truth for bitfields)
grep -n "CNA_CONV_CON1" /path/to/linux/drivers/accel/rocket/rocket_registers.h

# NVDLA Verilog register file
grep -n "D_OP_ENABLE" nvdla/hw/vmod/nvdla/NV_NVDLA_CDMA_regfile.v
```

### Approach 3: Replay conv.py Registers Through VP

This would require modifying the VP to accept conv.py's register sequences instead of loading a Linux kernel — essentially creating a "register test mode." This is possible but requires C++ changes to `aarch64_toplevel.cpp`.

## How Hard Would It Be to Write an RK3588 VP?

**Critical context**: This repo already has **every RK3588 NPU register fully reversed** in `rockchip.py` (9614 lines, auto-generated from `registers.xml` via `clang2py`). The same `registers.xml` is the source that generates the upstream Linux kernel `rocket_registers.h` via Mesa's `rules-ng-ng`. The register map is **not a gap**.

The real difficulty is building the **behavioral C-models** — the pipeline logic that executes when register values are set.

### What We Already Have (the 80%)

```
┌──────────────────────────────────────────────────────────────┐
│  rk3588 repo — already an RK3588 VP at the functional level  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  rockchip.py (9614 lines)  — ALL register definitions        │
│  • Every CNA register with full bitfield masks/shifts        │
│  • Every CORE register                                       │
│  • Every DPU register                                        │
│  • Every PC register (interrupts, task control, etc.)        │
│  • Same source as upstream Linux rocket_registers.h           │
│                                                              │
│  conv.py — Register programming layer                        │
│  • build_conv2d_regs(): 40+ register writes per layer        │
│  • compute_conv2d_params(): align_c, width_stride, CBUF calc │
│  • pack_nc1hwc2_fp16(): NC1HWC2 data packing                 │
│  • pack_conv_weights_fp16(): 3 weight layout variants        │
│  • unpack_nc1hwc2_fp16(): output format conversion           │
│  • _npu_submit(): DRM ioctl submission + DMA sync            │
│                                                              │
│  test_conv.py — Validation against real hardware             │
│  • 8 fully passing shapes (1x1 kernels)                      │
│  • 12 partial shapes (non-1x1, sparse output known issue)    │
│  • VALIDATE mode: compares NPU output vs CPU golden          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### What's Missing for a Full Cycle-Accurate VP

| Component | Status | What's needed |
|-----------|--------|---------------|
| **CNA behavioral model** | Partially — NVDLA `cmod/cdma+csc+cacc` exists | Adapt register decode from CDMA/CSC/CACC layout → merged CNA layout. Register values are known; the state machine behavior (CBUF tiling, atom sequencing) is in the NVDLA C-model and largely reusable. |
| **CORE behavioral model** | Partially — NVDLA `cmod/cmac` exists | CMAC array is identical in concept (MAC cells, stripe handling, shadow registers). Need to verify MAC cell count and precision modes match RK3588. |
| **DPU behavioral model** | Partially — NVDLA `cmod/sdp+pdp` exists | Post-processing pipeline (BS/BN/EW sub-blocks) is register-compatible. SDP LUT behavior is defined in `rockchip.py` bitfields. |
| **PC + task scheduling** | Register definitions exist. Behavioral model missing. | 3-core scheduling, task queue management, interrupt arbitration. NVDLA has single-core. This is the biggest unknown. |
| **Weight decompression engine** | `REG_CNA_DCOMP_CTRL` + `DCOMP_ADDR0-15` + `DCOMP_AMOUNT0-15` defined in `rockchip.py` | The decompression algorithm itself is unknown. The NVDLA C-model handles compressed weights; whether RK3588 uses the same algorithm is unverified. |
| **Cycle-accurate timing** | Missing | conv.py is functional (correct results), not cycle-timed. A VP needs pipeline stage timing. |
| **DRM ioctl interface (vs CSB)** | conv.py already implements submission. VP would need a shim layer. | NVDLA VP uses TLM-2.0 CSB transactions. RK3588 uses DRM_IOCTL_RKNPU_SUBMIT with a DMA'd command buffer. The VP would need to decode this buffer format in the C-model. |

### Option 1: Fork NVDLA VP + Adapt C-Model (Recommended Path)

**Difficulty: 5/10** — Low for register work, moderate for behavioral modeling.

| Task | What's involved | Difficulty |
|------|----------------|-----------|
| **Map register addresses** | NVDLA CDMA(0x4000)+CSC(0x5000)+CACC(0x7000) → RK3588 CNA(0x1000). Done — `rockchip.py` has addresses. | **0/10** (already done) |
| **Map register bitfields** | Every NVDLA bitfield has an RK3588 counterpart. `rockchip.py` has all masks/shifts. | **0/10** (already done) |
| **Adapt C-model register decode** | The NVDLA C-model's register models (`cdma_reg_model.cpp`, `csc_reg_model.cpp`) use NVDLA address maps. Fork them and replace addresses/bitfields with RK3588 values from `rockchip.py`. The pipeline logic stays the same. | 4/10 |
| **Add DRM ioctl submission shim** | Replace CSB TLM-2.0 transport with a DRM ioctl decoder. The register pairs in conv.py's command buffer are the same format the kernel DMA's to hardware. | 5/10 |
| **Model 3-core task scheduling** | NVDLA has single core. RK3588 supports 3 cores + subcore tasks. Need to model core arbitration (registers exist in `rockchip.py`). | 6/10 |
| **Validate against real HW** | Run conv.py test cases through the VP and compare register values + output data. | 2/10 |
| **Add cycle timing** | The NVDLA C-model already tracks cycles. Verify they match RK3588 timing (likely different clock rates). | 4/10 |

### Option 2: conv.py-as-VP (the Pragmatic Approach)

**Difficulty: 1/10** — Already works, already validates against real hardware.

conv.py IS the closest thing to an RK3588 VP that exists today:

| VP feature | conv.py equivalent |
|-----------|-------------------|
| Register programming | `build_conv2d_regs()` + `emit()` |
| Data format conversion | `pack_nc1hwc2_fp16()` + `pack_conv_weights_fp16()` |
| Hardware execution | `_npu_submit()` via DRM ioctl |
| Output capture | `unpack_nc1hwc2_fp16()` |
| Result validation | CPU golden model comparison (`_validate_npu_result`) |
| Register dump (dry run) | `DRY_RUN` mode (`--submit` flag omitted) |

The NVDLA C-model from `nvdla/hw/cmod/` provides the **behavioral reference** — read it alongside conv.py to understand why each register has its value. This cross-reference is more valuable than building a SystemC simulation because conv.py runs on real hardware with real results.

### Option 3: SystemC Model from Scratch

**Difficulty: 7/10** — Register definitions are done; pipeline behavior needs modeling.

| Task | Difficulty | Reason |
|------|-----------|--------|
| Model CNA pipeline (CDMA+CSC+CACC merged) | 5/10 | Much easier now: register bitfields are fully known. Can lift pipeline logic from NVDLA C-model and adapt. |
| Model CORE (CMAC) | 4/10 | MAC array is conceptually identical. Register decode is in `rockchip.py`. |
| Model DPU (SDP+PDP merged) | 5/10 | Post-processing logic similar to NVDLA. BS/BN/EW sub-block configs are fully documented in `rockchip.py`. |
| Model PC + task scheduling | 7/10 | Scheduling logic is custom. Registers are known (`TASK_CON`, `TASK_STATUS`, `INTERRUPT_MASK/CLEAR/STATUS` in `rockchip.py`). |
| Full-system cosimulation | 6/10 | Could skip QEMU entirely: write a lightweight harness that reads conv.py's register pairs and drives the C-model directly. |

A functional (not cycle-accurate) VP is feasible in **weeks, not years** — the register work is 100% done, and the pipeline logic can be adapted from the NVDLA C-model.

### Option 4: FPGA Emulation

**Difficulty: 9/10** — Still requires proprietary RTL.

NVDLA's FPGA platform maps Verilog to AWS F1 instances. Without the RK3588 NPU RTL (proprietary), FPGA emulation is impossible. A reverse-engineered RTL implementation would take years.

## Summary: Usefulness Matrix

| Method | Difficulty | Useful for conv.py? | What you learn |
|--------|-----------|-------------------|----------------|
| **Read `rockchip.py`** | 0/10 | ★★★★★ | Every RK3588 register and bitfield |
| **Read `conv.py`** | 1/10 | ★★★★★ | Complete register programming for convolution |
| **Read `test_conv.py`** | 1/10 | ★★★★★ | Hardware validation patterns, known issues |
| **Cross-ref conv.py vs nvdla/hw C-model** | 2/10 | ★★★★★ | Why each register value works |
| **Build NVDLA VP on x86** | 4/10 | ★★★☆☆ | Generic NVDLA register sequencing |
| **Write functional RK3588 VP** | 5/10 | ★★★★☆ | Would validate register values offline |
| **Write cycle-accurate RK3588 VP** | 8/10 | ★★★★☆ | Would enable performance modeling |
| **FPGA emulation** | 9/10 | ★★★☆☆ | Requires proprietary RTL |

## Bottom Line

**The RK3588 register definitions are fully done.** `rockchip.py` (9614 lines from `registers.xml`) has every register address, every bitfield mask, every bitfield shift, for all 4 blocks (CNA/CORE/DPU/PC) and all sub-blocks (incl. decompression engine, clock gating, task scheduling). This is the same source used by the upstream Linux kernel driver.

What remains for a VP:
- A **4/10 difficulty** fork of NVDLA C-model adapting register decode to RK3588's merged block layout
- A **5/10 difficulty** DRM ioctl shim replacing CSB transport
- The NVDLA C-model pipeline logic (CBUF, atom sequencing, MAC array, accumulator) is **largely reusable** because RK3588 hardware behavior mirrors NVDLA

The most practical approach: keep cross-referencing conv.py against nvdla/hw C-model code. When register configuration questions arise, the upstream `rocket_registers.h` (identical to `rockchip.py`) is the canonical bitfield reference.
