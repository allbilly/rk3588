# NVDLA Hardware Architecture Reference

Sources:
- http://nvdla.org/hwarch.html
- https://nvdla.org/primer.html
- DeepWiki: https://deepwiki.com/nvdla/hw

## Architecture Diagram (Headless Core)

```
                    ┌──────────────┐
                    │  CSB/IRQ IF  │◄── CPU control
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │  Memory I/F  │◄── DBB (AXI-like) to DRAM
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐
        │ Conv Core │ │CBUF  │ │ BDMA     │
        │ (CSC→CMAC │ │(SRAM)│ │ Reshape  │
        │  →CACC)   │ └──────┘ │ Rubik    │
        └─────┬─────┘          └───────────┘
              │
        ┌─────┴─────┐
        │ SDP (act) │
        └─────┬─────┘
              │
        ┌─────┴─────┐
        │ PDP (pool)│
        └─────┬─────┘
              │
        ┌─────┴─────┐
        │ CDP (LRN) │
        └─────┬─────┘
              │
        ┌─────┴─────┐
        │ Memory I/F│──► DBB to DRAM
        └───────────┘
```

## Key Parameters (nvdlav1, nv_full config)

| Parameter | Value |
|-----------|-------|
| MAC count | 2048 int8 / 1024 fp16 |
| CBUF banks | 2-32 × 4-8 KiB |
| MAC array C dim | 8-64 |
| MAC array K dim | 4-64 |
| SDP output/cycle | 1-16 |
| PDP output/cycle | 1-4 |
| CDP output/cycle | 1-4 |
| Data types | int4/8/16/32, fp16/32/64 |
| Memory I/F | 1 primary + 1 optional SRAM |
| CSB address | 16-bit (4KB per block) |

## Block Address Ranges (NV_SMALL)

| Block | Start | End | Size |
|-------|-------|-----|------|
| CFGROM | 0x0000 | 0x0FFF | 4KB |
| GLB | 0x1000 | 0x1FFF | 4KB |
| MCIF | 0x2000 | 0x2FFF | 4KB |
| CDMA | 0x3000 | 0x3FFF | 4KB |
| CSC | 0x4000 | 0x4FFF | 4KB |
| CMAC_A | 0x5000 | 0x5FFF | 4KB |
| CMAC_B | 0x6000 | 0x6FFF | 4KB |
| CACC | 0x7000 | 0x7FFF | 4KB |
| SDP_RDMA | 0x8000 | 0x8FFF | 4KB |
| SDP | 0x9000 | 0x9FFF | 4KB |
| PDP_RDMA | 0xA000 | 0xAFFF | 4KB |
| PDP | 0xB000 | 0xBFFF | 4KB |
| CDP_RDMA | 0xC000 | 0xCFFF | 4KB |
| CDP | 0xD000 | 0xDFFF | 4KB |
| BDMA | 0xE000 | 0xEFFF | 4KB |
| RUBIK | 0xF000 | 0xFFFF | 4KB |

*Note: RK3588 maps these differently — CDMA+CSC+CACC → CNA (0x1000), CMAC → CORE (0x3000), SDP+PDP → DPU (0x4000)*

## Convolution Modes

1. **Direct Convolution** — Standard im2col-like. CSC sends (kernel_h × kernel_w) weight atoms per output position.
2. **Winograd Convolution** — F(2×2, 3×3) transform. Reduces MAC count for 3×3 kernels. Requires POA (post-operation add) in CMAC.
3. **Depthwise Convolution** — Each output channel uses exactly 1 input channel. CMAC bypasses channel accumulation.

## Memory Subsystem

- CBUF: On-chip SRAM buffer. Organized as banks. Stores input slices + weight atoms.
- DRAM: System memory via DBB (AXI-like). CDMA reads/writes in burst transfers.
- CVSRAM: Optional dedicated SRAM port for higher bandwidth.

## Control Flow

1. CPU writes registers via CSB (Configuration Space Bus)
2. CPU sets `D_OP_ENABLE` register in the last block to configure
3. Blocks auto-start when all dependencies met (CSC waits for CDMA, CMAC waits for CSC, etc.)
4. Interrupt fires when layer completes
5. CPU reads status, writes next layer's registers
6. Double-buffering: group0 executes while CPU configures group1

## Software Stack Data Flow

```
Caffe Model → Parser → IR → Compiler → Loadable (.nvdla)
                                              ↓
                                         UMD (user-mode)
                                              ↓ ioctl
                                         KMD (kernel-mode)
                                              ↓ register writes
                                         NVDLA Hardware
                                              ↓ interrupt
                                         KMD → UMD → Application
```
