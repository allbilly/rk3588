review info in experimental/* and nvdla/* when face and problem, and ask deepwiki

On this Orange Pi RK3588 machine, the NPU card is present. Do not assume hardware is
unavailable just because an earlier probe failed; `python examples/elementwise.py` is
a known quick check that proves the card can be used.

Use `deepwiki_ask_question` on these repos:
- `nvdla/hw` — For C-model behavior (cmod/), register definitions, convolution pipeline details, weight/activation data flow. This is the **canonical hardware reference**. Key files:
  - `cmod/csc/NV_NVDLA_csc.cpp` — Convolution stream controller (data sequencing, Winograd)
  - `cmod/cmac/NV_NVDLA_cmac.cpp` — MAC array (weight loading, calculation)
  - `cmod/cacc/NV_NVDLA_cacc.cpp` — Accumulator
  - `cmod/cdma/*` — DMA data fetching and CBUF management
  - `cmod/*/*_reg_model.cpp` — Register field definitions
  - `vmod/nvdla/NV_NVDLA_*_regfile.v` — Verilog register implementation
  - `verif/traces/traceplayer/conv_8x8_fc_int16/input.txn` — Real register write sequences
- `soDLA-publishment/soDLA`, Chisel implementation of the NVIDIA hw

- `allbilly/rknpu_driver`, official rknpu driver  

- `torvalds/linux` (drivers/accel/rocket/) — The **upstream Linux kernel** "rocket" driver for Rockchip NPU. Contains the canonical register definitions in `rocket_registers.h` (auto-generated from Mesa). When `rockchip.py` or `conv.py` register values seem wrong, check this file for ground-truth bitfield masks/shifts. Tomu got multicore NPU working on the merged linux mainline rocket NPU driver.

- `nvdla/sw` — For compiler loadable format, UMD/KMD driver logic, how the software stack partitions and programs layers. Key files:
  - `prebuilt/` — Prebuilt kernel images and drivers for VP
  - `umd/` — User mode driver (loadable parsing, inference submission)
  - `kmd/` — Kernel mode driver (register programming, interrupt handling)

- `allbilly/mesa` — For the Mesa Gallium driver (`src/gallium/drivers/rocket/`), which includes the `registers.xml` that generates `rocket_registers.h`. Useful for understanding how convolution is compiled for RK3588. Tomu got multicore working with mesa.
- `ONNC/onnc` — Open Neural Network Compiler, includes NVDLA backend support. Useful for understanding compiler-level convolution partitioning.
- `allbilly/npu` — my another NPU reverse-engineering repo, contains how to run model in ops_rknn and gdb dump scripts

folder structure
- *rawbuf*.py is keeping its hexblob intentionally
- examples/*.py , each op example py file shd be standalone and contain only decoded registers, no hex blob. coding style shd reference to elementwise.py and each op.py shd have min line diff comparing to elementwise.py as the golden reference, compare with command diff ops.py elementwise.py
- Dont edit files in examples/mainline6-18/ unless specified

