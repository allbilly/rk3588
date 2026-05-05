nvdla/REGISTER_MAP.md         local RK/NVDLA register map notes
nvdla/NC1HWC2_FORMAT.md       feature layout notes
nvdla/WEIGHT_PACKING.md       convolution/GEMM weight layout notes
nvdla/CONV_DATAFLOW.md        conv pipeline walk-through
```

1. Check `experimental/gemm.c`, especially `alu_case_matmul`.
2. Check `experimental/rknnops.h` helpers: `make_matmul_params`, `feature_data`, `pack_matmul_weights_fp16`.
3. Capture vendor RKNN or `ops_reg` dumps if needed.
4. Build a register comparison table:

Focus on registers where Python differs from both known-good sources. If RKNN differs from `gemm.c` but both produce correct output, treat that as an implementation difference first, not a bug.

Known GEMM bug classes already fixed:

- Exact 32768-byte CBUF bank boundary must not allocate an extra data bank.
- `GROUP_LINE_OFF` is not disabled for every `m == k` shape.
- Notch zeroing is narrower than `no_group_line_off`.
- C2=4 output decode is only for the proven square C2-input schedules.
- Input packing changes between row-major and C2=8 NC1HWC2 depending on shape.

## 4. Debug conv

For convolution, use upstream NVDLA as ground truth before changing register logic:
- `nvdla/hw/cmod/csc/NV_NVDLA_csc.cpp` for CSC data sequencing.
- `nvdla/hw/cmod/cmac/NV_NVDLA_cmac.cpp` for MAC/weight behavior.
- `nvdla/hw/cmod/cacc/NV_NVDLA_cacc.cpp` for accumulation.
- `nvdla/hw/cmod/cdma/*` for CBUF and DMA fetch behavior.
- `nvdla/hw/vmod/nvdla/*_regfile.v` for register implementation.
- `torvalds/linux drivers/accel/rocket/rocket_registers.h` for canonical RK/Rocket bitfield masks and shifts.
- Mesa `src/gallium/drivers/rocket/` and `registers.xml` for userspace register generation.

