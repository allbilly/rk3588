# NVDLA Reference for RK3588 NPU (`conv.py`)

The RK3588 NPU is **NVDLA-like** — it inherits the same 4-block architecture (CNA/CORE/DPU/PC) and register programming model from NVIDIA's open-source NVDLA design. This folder documents everything from the upstream NVDLA repos that helps demystify `examples/conv.py`.

## Upstream Repositories

| Repo | Purpose | Key contents |
|------|---------|-------------|
| [nvdla/hw](https://github.com/nvdla/hw) | RTL + C-model | `cmod/` — cycle-accurate C++ models of CSC, CMAC, CACC, CDMA, CBUF, SDP, PDP. Verilog RTL. |
| [nvdla/vp](https://github.com/nvdla/vp) | Virtual Platform | SystemC simulator + QEMU aarch64 for running NVDLA software stack. **Build on x86, not RK3588.** |
| [nvdla/sw](https://github.com/nvdla/sw) | Software stack | Compiler, runtime (UMD/KMD), loadable format |
| [JunningWu/Learning-NVDLA-Notes](https://github.com/JunningWu/Learning-NVDLA-Notes) | Study notes | Detailed register docs, test examples, diagrams of CMAC pipeline |
| [torvalds/linux](https://github.com/torvalds/linux) `drivers/accel/rocket/` | Linux "rocket" driver | Canonical RK3588 NPU register definitions in `rocket_registers.h` |

## How to Use This Folder

1. Start with `ARCHITECTURE.md` — understand the convolution pipeline
2. Read `REGISTER_MAP.md` — map NVDLA registers to `conv.py` register names
3. Study `NC1HWC2_FORMAT.md` — understand why data is packed this way
4. Read `WEIGHT_PACKING.md` — understand the 3 weight layout variants
5. Trace through `CONV_DATAFLOW.md` — follow one convolution end-to-end
6. Read `VP_EXPLAINED.md` — understand NVDLA VP, why building it on RK3588 is futile, and how to use conv.py + nvdla/hw C-model as your "poor man's VP"

## Key Insight

Every register `conv.py` writes (`REG_CNA_*`, `REG_CORE_*`, `REG_DPU_*`) is a direct descendent of the corresponding NVDLA register. The `_target()` function in `conv.py:307-312` maps address ranges to the same block IDs:
- `0x1000-0x1FFF` → CNA (= 512)
- `0x3000-0x3FFF` → CORE (= 2048)
- `0x4000-0x4FFF` → DPU (= 4096)
- `< 0x1000` → PC (= 256)

The upstream Linux kernel's `rocket_registers.h` (`drivers/accel/rocket/`) is the **canonical bitfield reference** — always check it when `rockchip.py` or `conv.py` register values seem wrong.
