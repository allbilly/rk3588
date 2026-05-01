# NVDLA Blog & Article Summaries

## Learning NVDLA Notes — Junning Wu

**URL**: https://github.com/JunningWu/Learning-NVDLA-Notes
**Summary**: The most comprehensive third-party NVDLA study notes available. Covers:

1. **Test patterns**: `sanity0` (register R/W), `sanity1` (BDMA memcpy), `sanity3` (convolution test), `conv_8x8_fc_int16` (full conv trace)
2. **Bus interfaces**: CSB (Configuration Space Bus) request/read/write channels with timing diagrams. AXI4-like DBB (Data Backbone) simplified protocol.
3. **Register maps**: Full register tables for CDMA (`0x5000-0x50E8`), CSC (`0x6000-0x6064`), CACC (`0x9000-0x9034`), CMAC (`0x7000-0x700C`), SDP (`0xB000+`). Each register with address, bitfield function, and description.
4. **CMAC architecture**: 16 MAC Cells × (64×16bit MACs) = 1024 16-bit MACs (2048 int8). 7-stage pipeline. Winograd POA support. Two instances (cmac_a, cmac_b).
5. **CACC architecture**: 64 INT48/INT34/FP48 adders. Partial sum SRAM configurable by mode. Round + saturate before SDP output.
6. **Software stack**: KMD/UMD/compiler flow. Loadable format description.
7. **FPGA on ZCU104**: Physical FPGA implementation guide.

**Key insight for conv.py**: The register tables in these notes directly map to conv.py's `REG_CNA_*`, `REG_CORE_*`, `REG_DPU_*` registers. The input.txn example shows the exact programming sequence NVDLA uses, which conv.py replicates through emit/target.

---

## NVDLA Primer — Official Docs

**URL**: https://nvdla.org/primer.html
**Summary**: Official NVIDIA documentation covering:

1. **Small vs Large model**: Small = headless (CPU manages NVDLA), Large = headed (dedicated microcontroller + SRAM)
2. **Fused vs Independent mode**: Independent = memory-to-memory per block. Fused = blocks pipeline through FIFOs.
3. **Component descriptions**: Convolution Core (Winograd, weight compression, batch conv), SDP (LUT-based activation), PDP (pooling), CDP (LRN), Reshape Engine, BDMA.
4. **Configurability parameters**: 17 tunable parameters (MAC array size, CBUF banks, precision support, memory interface width, etc.)
5. **Software design**: Compiler (Caffe → loadable), UMD (loadable parsing + ioctl), KMD (engine scheduler + interrupt handler)
6. **Performance table**: ResNet-50 at 1GHz: 2048 MACs → 269 fps (28nm: 5.5mm², 16nm: 3.3mm²)
7. **System integration guidance**: How to size MAC count, CBUF, and SRAM for target workloads.

**Key insight for conv.py**: The Primer confirms conv.py is implementing a "headless" NVDLA-like system — the CPU (RK3588's Cortex-A) directly programs registers without a dedicated microcontroller, matching the Small model description.

---

## NVDLA Runtime VP Build — CSDN (Chinese)

**URL**: https://blog.csdn.net/q2519008/article/details/115721678
**Summary**: Step-by-step VP build guide with bug fixes:

1. Cloning vp + submodules
2. Installing dependencies (SystemC 2.3.0a, YAML perl module, IO-Tee perl module)
3. Building NVDLA CMOD (`tools/bin/tmake -build cmod_top`)
4. CMake with correct paths for hw and SystemC
5. **Critical fix**: Remove `-Werror` from VP's CMakeLists.txt for modern gcc compatibility
6. **memfd.c fix**: Patch for newer glibc versions

---

## NVDLA Virtual Platform Build — Nerox (Chinese)

**URL**: https://blog.csdn.net/smiler_sun/article/details/89608320
**Summary**: Another VP build guide covering:
- Compiling Linux kernel with buildroot for aarch64
- QEMU configuration for virt machine type
- NVDLA driver loading (`opendla_1.ko` / `opendla_2.ko`)
- Running tests on the simulated platform

---

## NVDLA Study Notes — rmheng (Chinese)

**URL**: https://rmheng.github.io/2019/12/29/2020-04-19-NVDLA学习/
**Summary**: Personal study blog covering:
1. Full VP build walkthrough with multiple tool version issues documented
2. Running inference on VP: `nvdla_compiler` with Caffe models (ResNet-50)
3. Compiler flags: `--profile fast-math`, `--configtarget nv_small/nv_large/nv_full`, `--cprecision int8`
4. Runtime: `nv_runtime --loadable --image --rawdump --normalize`
5. Performance note: One inference on VP takes ~2 hours (nv_full config)

---

## Linux Kernel "rocket" Driver — Upstream Register Definitions

**URL**: https://github.com/torvalds/linux/blob/master/drivers/accel/rocket/rocket_registers.h
**Summary**: The official upstream Linux kernel register definitions for the Rockchip NPU (kernel codename "rocket"). Auto-generated from `registers.xml` (via Mesa's rules-ng-ng), maintained by Tomeu Vizoso. Every register in `rockchip.py` and `conv.py` has an exact counterpart with full bitfield masks/shifts. Key bitfields:
- `REG_CNA_CONV_CON1` at `0x100C`: `NONALIGN_DMA`, `GROUP_LINE_OFF`, `DECONV`, `ARGB_IN`, `PROC_PRECISION`, `IN_PRECISION`, `CONV_MODE`
- `REG_CNA_CONV_CON2` at `0x1010`: `KERNEL_GROUP`, `FEATURE_GRAINS`, `CSC_WO_EN`, `CSC_DO_EN`
- `REG_CNA_CONV_CON3` at `0x1014`: `NN_MODE`, `ATROUS_Y/X_DILATION`, `CONV_Y/X_STRIDE`
- `REG_CNA_DCOMP_CTRL` at `0x1100`: `WT_DEC_BYPASS`, `DECOMP_CONTROL` (weight decompression)
**Relevance to conv.py**: **Very High** — Ground truth for register bitfield layout. When conv.py writes `CNA_CONV_CON1 = (2 << 7) | (2 << 4)`, the kernel header shows `PROC_PRECISION=2=FP16`, `IN_PRECISION=2=FP16`.

---

## NC1HWC2 in RKNN Ecosystem (grep.app search)

**URLs**: https://github.com/airockchip/rknn_model_zoo, https://github.com/deepinsight/insightface
**Summary**: Multiple Rockchip NPU open-source projects use `RKNN_TENSOR_NC1HWC2` as the native tensor format. `rknn_api.h` defines `RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR`/`OUTPUT_ATTR`. Projects implement `NC1HWC2_int8_to_NCHW_float()` — the exact inverse of conv.py's `unpack_nc1hwc2_fp16()`.
**Relevance to conv.py**: **High** — Confirms NC1HWC2 is Rockchip's native NPU format (inherited from NVDLA), not a conv.py invention.

---

## CSDN NVDLA专题 Series — fangfanglovezhou (Chinese)

**URLs**:
- [专题1: 框架介绍](https://blog.csdn.net/fangfanglovezhou/article/details/141110684)
- [专题13: Software + Compiler Library](https://blog.csdn.net/fangfanglovezhou/article/details/142096805)
**Summary**: Comprehensive Chinese translation of official NVDLA docs. Covers register address spaces (CDMA 0x9000, CACC 0xD000 for NV_LARGE), compiler decisions (Winograd vs direct conv, CBUF-based operation splitting), concrete hardware params (Small: Atomic-C 8, Atomic-K 8, 32×4KB CBUF; Large: Atomic-C 64, Atomic-K 16, 16×32KB CBUF).
**Relevance to conv.py**: **High** — Register address tables map directly to conv.py register writes. Compiler splitting logic explains channel slicing for >4 input channels.

---

## CSDN NVDLA学习笔记（2）— 硬件架构规范 — 夏风喃喃 (Chinese)

**URL**: https://blog.csdn.net/qq_41019681/article/details/118604450
**Summary**: Deep dive into all five convolution modes (direct, image-input, Winograd, batching), MAC efficiency (Atomic-C/K impact), CBUF 4-port architecture, ping-pong register programming (PRODUCER/CONSUMER pointer mechanism), full address space table (GLB 0x0000 → RUBIK 0xF000).
**Relevance to conv.py**: **High** — Ping-pong mechanism explains why conv.py doesn't need dual register group management. CBUF bank details explain `data_bank`/`cbuf_entries` calculations.

---

## Official NVDLA Software Documentation

**URLs**:
- [SW Manual TOC](http://nvdla.org/sw/contents.html)
- [Runtime + Loadable Format](http://nvdla.org/sw/runtime_environment.html)
- [Compiler Library](http://nvdla.org/sw/compiler_library.html)
**Summary**: Full software stack docs: compiler (Caffe → IR → hardware layers), runtime (UMD loadable loading + KMD engine scheduler + interrupt handling), NVDLA Loadable format (flatbuffer-based, cross-platform). The compiler chooses Winograd vs direct conv, splits ops by CBUF size, quantizes to int8.
**Relevance to conv.py**: **High** — Compiler partitioning logic is what conv.py must replicate manually via `compute_conv2d_params()`.
