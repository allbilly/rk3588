When analyzing RK3588 NPU behavior — especially when `conv.py`, `gemm.py`, or `test_conv.py` produce unexpected results or when register configuration logic seems incorrect or guessy — check the upstream NVDLA source code for ground truth.

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

- `allbilly/rknpu_driver`, official rknpu driver  

- `torvalds/linux` (drivers/accel/rocket/) — The **upstream Linux kernel** "rocket" driver for Rockchip NPU. Contains the canonical register definitions in `rocket_registers.h` (auto-generated from Mesa). When `rockchip.py` or `conv.py` register values seem wrong, check this file for ground-truth bitfield masks/shifts. Tomu got multicore NPU working on the merged linux mainline rocket NPU driver.

- `nvdla/sw` — For compiler loadable format, UMD/KMD driver logic, how the software stack partitions and programs layers. Key files:
  - `prebuilt/` — Prebuilt kernel images and drivers for VP
  - `umd/` — User mode driver (loadable parsing, inference submission)
  - `kmd/` — Kernel mode driver (register programming, interrupt handling)

- `allbilly/mesa` — For the Mesa Gallium driver (`src/gallium/drivers/rocket/`), which includes the `registers.xml` that generates `rocket_registers.h`. Useful for understanding how convolution is compiled for RK3588. Tomu got multicore working with mesa.
- `ONNC/onnc` — Open Neural Network Compiler, includes NVDLA backend support. Useful for understanding compiler-level convolution partitioning.
- `allbilly/npu` — Community NPU reverse-engineering efforts, may contain RK3588-specific insights.


## Gemm Debug Method

When `gemm.py` produces wrong results for a shape, follow this process:

1. **Check `experimental/gemm.c`** — This is the canonical register reference for matmul. It contains `alu_case_matmul` (174 lines, identical to `experimental/rknnops.h:1440-1611`). Build a register comparison table.

2. **Check `experimental/rknnops.h`** supporting functions:
   - `make_matmul_params` (line 164) — alignment, stride calculation
   - `pack_matmul_weights_fp16` (line 834) — weight packing (column-major for 32x32, tile_16x32 for others)
   - `feature_data` (line 765) — input packing formula
   - `unpack_matmul_output_fp32` (in `experimental/main.c:129`) — output decode (C2=4 for 64x64/256x256, else C2=align_out)

3. **Check ops_rknn** — Vendor library at `~/npu/ops_rknn/matmul_api.cpp`. Runs on real hardware, passes all shapes. Use `gdb -x matmul.gdb ./matmul_api` with `python3 dump.py 1` to capture register dumps.

4. **Build a register comparison table** for the failing shape. Compare:
   - `gemm.py` `make_gemm_regs()` — what you're emitting (use `--dry` flag)
   - `gemm.c` / `ops_reg` — capture via `gdb -x test.gdb ./build/ops_reg` at `submitTask` breakpoint
   - `ops_rknn` dump — capture via `gdb -x matmul.gdb ./matmul_api` at `rknn_matmul_run` breakpoint (use `$count` to skip earlier calls)
   
   **Example: 256x256x256 register comparison table** (the only register that differed):
   ```
   Register         gemm.py      ops_reg       ops_rknn      Verdict
   ──────────────── ──────────── ──────────── ──────────── ──────────────
   CBUF_CON0        0x75 (D=5)   0x84 (D=4)   0x84 (D=4)   ← BUG (gemm.py only)
   CONV_CON1        0x120        0x120        0x120        ✓ all match
   DATA_SIZE1       real=255     real=255     real=255     ✓ all match
   WEIGHT_SIZE1     512          512          512          ✓ all match
   CBUF_CON1        8            8            8            ✓ all match
   DMA_CON1         4            4            32           ops_rknn differs (not bug)
   NOTCH            0            0            63           ops_rknn differs (not bug)
   DST_SURF_STRIDE  256          256          1            ops_rknn differs (not bug)
   SURFACE_ADD      1024         1024         4            ops_rknn differs (not bug)
   ```
   Key: focus on registers where gemm.py differs from **both** ops_reg AND ops_rknn.
   Registers where ops_rknn disagrees with ops_reg (but both produce correct results)
   are implementation differences, not bugs.

5. **Common bugs found so far:**
   - `k_exact and m == k` rule (line 291-292): This false generalization disables GROUP_LINE_OFF for shapes like 32x32x32 and 64x99x64. gemm.c only disables GROUP_LINE_OFF for explicit `is_KN_64`/`is_KN_256`/`is_KN_512`/`is_KN_lg_512`/`is_matmul_768`/`is_matmul_768_2048`/`is_matmul_2048`.
   - `no_group_line_off` in notch zeroing: gemm.c only zeros notch for `is_KN_64 || is_KN_256 || is_KN_512 || K > 7872`, not for `no_group_line_off`.
   - Output decode format: gemm.c uses C2=4 only for 64x64x64 and 256x256x256. All other shapes use C2=align_out (which is row-major when C2 == align_out).
   - Input packing: gemm.c uses `feature_data(align_in, M, 1, align_in, k, m, 1)` for shapes where align_in < 64 (C2=align_in → row-major), and `pack_matmul_input_64x64_fp16` (C2=8) for 64x64. ops_rknn always uses NC1HWC2 C2=8 for >=64 channel shapes.
   - `data_bank += 1` at exact 32768 boundary (gemm.py line 299): gemm.c does NOT increment data_bank when `input_bytes % 32768 == 0`. This caused 256x256x256 to have CBUF_CON0=0x75 (DATA_BANK=5) instead of 0x84 (DATA_BANK=4), producing md=54 instead of md=5e-5.

When investigating:
1. First try deepwiki on `nvdla/hw` for hardware-level answers (data format, register bitfields, pipeline timing)
2. Then try deepwiki on `nvdla/sw` for software-level answers (compiler decisions, driver submission flow)
3. Cross-reference with the local `nvdla/` docs (ARCHITECTURE.md, REGISTER_MAP.md, NC1HWC2_FORMAT.md, WEIGHT_PACKING.md)

folder structure
- examples/*.py , each op example py file shd be standalone and contain only decoded registers, no hex blob. coding style shd reference to elementwise.py and each op.py shd have min line diff comparing to elementwise.py as the golden reference.