# NVDLA → RK3588 Register Cross-Reference

## Block Mapping

| Architecture | NVDLA Name | RK3588 Name | Address Range | Target ID |
|-------------|------------|-------------|---------------|-----------|
| Program Control | GLB | PC | `0x0000-0x0FFF` | `0x81` (256) |
| Convolution | CDMA + CSC + CACC | **CNA** | `0x1000-0x1FFF` | `0x201` (512) |
| Matrix Compute | CMAC_A + CMAC_B | **CORE** | `0x3000-0x3FFF` | `0x801` (2048) |
| Post-Processing | SDP + PDP + BDMA | **DPU** | `0x4000-0x4FFF` | `0x1001` (4096) |

## Register-by-Register Mapping

### CNA Registers (conv.py → NVDLA CDMA/CSC/CACC)

| conv.py name | Address | NVDLA origin | Purpose |
|-------------|---------|-------------|---------|
| `REG_CNA_CONV_CON1` | `0x100C` | CDMA misc + CSC misc | Conv mode: precision, depthwise flag, 1x1 hint |
| `REG_CNA_CONV_CON2` | `0x1010` | CDMA fetch grain | `feature_grains` = how many CBUF slices |
| `REG_CNA_CONV_CON3` | `0x1014` | CDMA op enable | Start trigger bits |
| `REG_CNA_DATA_SIZE0` | `0x1020` | CDMA `D_DATAIN_SIZE_0` | Input width × height packed |
| `REG_CNA_DATA_SIZE1` | `0x1024` | CDMA `D_DATAIN_SIZE_1` | Input channels (real + aligned) |
| `REG_CNA_DATA_SIZE2` | `0x1028` | CSC `D_DATAOUT_SIZE_0` | Output width |
| `REG_CNA_DATA_SIZE3` | `0x102C` | CSC `D_ATOMICS` | Output atoms = out_w × out_h |
| `REG_CNA_WEIGHT_SIZE0` | `0x1030` | CDMA `D_WEIGHT_BYTES` | Total weight bytes |
| `REG_CNA_WEIGHT_SIZE1` | `0x1034` | CDMA `D_WEIGHT_SIZE_0` | Bytes per kernel |
| `REG_CNA_WEIGHT_SIZE2` | `0x1038` | CSC `D_WEIGHT_SIZE_EXT_0/1` | Kernel W/H + weight kernels |
| `REG_CNA_CBUF_CON0` | `0x1040` | CDMA `D_BANK` | Data bank allocation |
| `REG_CNA_CBUF_CON1` | `0x1044` | CDMA `D_ENTRY_PER_SLICE` | CBUF entry count per slice |
| `REG_CNA_CVT_CON0-4` | `0x104C-0x105C` | CDMA `D_CVT_CFG` | Data converter: format, truncation |
| `REG_CNA_CVT_CON5` | `0x1180` | SDP LUT | CVT mask for lane selection |
| `REG_CNA_FEATURE_DATA_ADDR` | `0x1070` | CDMA `D_DAIN_ADDR_LOW_0` | Input DRAM address |
| `REG_CNA_DMA_CON0` | `0x1078` | CDMA line stride misc | DMA burst config |
| `REG_CNA_DMA_CON1` | `0x107C` | CDMA `D_LINE_STRIDE` | Line stride (bytes) |
| `REG_CNA_DMA_CON2` | `0x1080` | CDMA `D_SURF_STRIDE` | Surface stride (bytes) |
| `REG_CNA_FC_DATA_SIZE0` | `0x1084` | CDMA `D_DATAIN_SIZE_EXT_0` | Extended input size |
| `REG_CNA_FC_DATA_SIZE1` | `0x1088` | CDMA format | Aligned channel count |
| `REG_CNA_DCOMP_ADDR0` | `0x1110` | Weight address | Weight DRAM address (with 16KB reserved offset) |

### CORE Registers

| conv.py name | Address | NVDLA origin | Purpose |
|-------------|---------|-------------|---------|
| `REG_CORE_MISC_CFG` | `0x3010` | CMAC `D_MISC_CFG` | Conv mode, depthwise flag |
| `REG_CORE_DATAOUT_SIZE_0` | `0x3014` | CACC `D_DATAOUT_SIZE_0` | Output height × width |
| `REG_CORE_DATAOUT_SIZE_1` | `0x3018` | CACC `D_DATAOUT_SIZE_1` | Output channels (aligned) |
| `REG_CORE_CLIP_TRUNCATE` | `0x301C` | CACC `D_CLIP_CFG` | Round/truncate config |

### DPU Registers

| conv.py name | Address | NVDLA origin | Purpose |
|-------------|---------|-------------|---------|
| `REG_DPU_S_POINTER` | `0x4004` | GLB pointer | Enable DPU sub-blocks (BS/BN/EW) |
| `REG_DPU_FEATURE_MODE_CFG` | `0x400C` | SDP mode | Feature mode: bypass/normal |
| `REG_DPU_DATA_FORMAT` | `0x4010` | SDP format | Output data format |
| `REG_DPU_DST_BASE_ADDR` | `0x4020` | BDMA/SDP | Output DRAM address |
| `REG_DPU_DST_SURF_STRIDE` | `0x4024` | SDP surface stride | Output surface stride |
| `REG_DPU_DATA_CUBE_WIDTH` | `0x4030` | SDP width | Output width - 1 |
| `REG_DPU_DATA_CUBE_HEIGHT` | `0x4034` | SDP height | Output height - 1 |
| `REG_DPU_DATA_CUBE_CHANNEL` | `0x403C` | SDP channel | Output channels (packed) |
| `REG_DPU_BS_CFG` | `0x4040` | SDP BS cfg | Bypass/single-point config |
| `REG_DPU_BS_OW_CFG` | `0x4050` | SDP OW cfg | Element-wise operation config |
| `REG_DPU_WDMA_SIZE_0` | `0x4058` | BDMA stride | Write DMA config |
| `REG_DPU_WDMA_SIZE_1` | `0x405C` | BDMA size | Write DMA output dimensions |
| `REG_DPU_BN_CFG` | `0x4060` | SDP BN cfg | Batch norm config |
| `REG_DPU_EW_CFG` | `0x4070` | SDP EW cfg | Element-wise config |
| `REG_DPU_EW_CVT_SCALE_VALUE` | `0x4078` | SDP scale | Scale value |
| `REG_DPU_OUT_CVT_SCALE` | `0x4090` | SDP output scale | Output scale |
| `REG_DPU_SURFACE_ADD` | `0x4094` | SDP surf add | Surface addition offset |

### PC Registers

| conv.py name | Address | NVDLA origin | Purpose |
|-------------|---------|-------------|---------|
| `REG_PC_OPERATION_ENABLE` | `0x0008` | GLB op enable | Kick off all blocks |

## Address Range Check Logic (from conv.py)

```python
def _target(addr):
    if 0x1000 <= addr < 0x2000: return CNA | 0x1    # 0x201
    if 0x3000 <= addr < 0x4000: return CORE | 0x1   # 0x801
    if 0x4000 <= addr < 0x5000: return DPU | 0x1     # 0x1001
    if addr < 0x1000: return addr | 0x1               # PC (0x81 for 0x40c4, etc.)
```

## Key NVDLA Sources for Register Details

- **C-model register models**: `cmod/cdma/cdma_reg_model.cpp`, `cmod/csc/csc_reg_model.cpp`, `cmod/cacc/cacc_reg_model.cpp`
- **Verilog register files**: `vmod/nvdla/NV_NVDLA_CSC_regfile.v`, `NV_NVDLA_CDMA_regfile.v`
- **Test traces**: `verif/traces/traceplayer/conv_8x8_fc_int16/input.txn` — real register write sequences
- **Junning Wu notes**: Full register tables for CDMA (`0x5000-0x50E8`), CSC (`0x6000-0x6064`), CACC (`0x9000-0x9034`), CMAC (`0x7000-0x700C`)
