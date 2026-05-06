# NVDLA KMD vs RKNPU Driver: Detailed Architectural Comparison

## 1. Overview

| Aspect | NVDLA KMD (`opendla.ko`) | RKNPU Driver (`rknpu.ko`) |
|--------|--------------------------|---------------------------|
| **Origin** | NVIDIA (open-source, nvdla/sw) | Rockchip (Felix Zeng, GPL v2) |
| **Version** | 0.0.0 (NVDLA reference) | 0.9.8 (2024-08-28) |
| **Module size** | ~12 C files, 4 headers (firmware) + 2 C files, 2 headers (Linux port) | 10 C files, 11 headers (flat) |
| **Total code** | ~2,800 lines (Linux port) + ~130K lines (firmware + register maps) | ~160 KB (10 .c + 11 .h) |
| **Kernel interface** | DRM subsystem only | DRM GEM **or** DMA Heap (miscdevice) |
| **License** | Dual BSD/GPL (port is GPLv2/BSD) | GPL v2 |
| **HW target** | NVDLA (NVIDIA DLA v1) | Rockchip NPU (RK356x, RK3588, RK3576, RV1106, etc.) |

## 2. Directory Structure Comparison

### NVDLA KMD (`nvdla/sw/kmd/`)
```
kmd/
├── port/linux/           # Linux OS port layer (thin, ~2 files)
│   ├── nvdla_core_callbacks.c  # Probe, ISR, register R/W, DMA callbacks
│   ├── nvdla_gem.c             # DRM GEM buffer management + IOCTL dispatch
│   └── include/
│       ├── nvdla_ioctl.h       # UAPI: submit_task, gem_create/mmap/destroy
│       └── nvdla_linux.h       # Linux-specific header types
├── firmware/              # The "engine" — bulk of HW programming logic
│   ├── conv.c, sdp.c, pdp.c, cdp.c, bdma.c, rubik.c  # Per-op programming
│   ├── scheduler.c             # Dependency-graph task scheduler (~1160 lines)
│   ├── engine.c, engine_data.c # Engine lifecycle, processor registration
│   ├── engine_isr.c            # Interrupt: dispatches per-unit done events
│   ├── engine_debug.c          # Debug/trace support
│   ├── common.c                # LUT loading, op cache
│   └── include/
│       ├── dla_interface.h     # ALL op/surface/stat descriptor structs (~886 lines)
│       ├── opendla_initial.h   # Auto-generated register map (1 MB!)
│       └── opendla_small.h     # Small-config register map (436 KB)
└── include/
    └── nvdla_interface.h       # Cross-port KMD interface (task, engine handles)
```

### RKNPU Driver (`rknpu_driver/`)
```
rknpu_driver/
├── rknpu_drv.c          # Probe, remove, power mgmt, IOCTL dispatch (1664 lines)
├── rknpu_job.c          # Job lifecycle, commit, IRQ handler, scheduling (1056 lines)
├── rknpu_gem.c          # DRM GEM buffer management (45 KB)
├── rknpu_mem.c          # DMA heap buffer management (8.4 KB)
├── rknpu_iommu.c        # IOMMU domain management (16 KB)
├── rknpu_devfreq.c      # DVFS / devfreq integration (23 KB)
├── rknpu_debugger.c     # debugfs/procfs observability (15 KB)
├── rknpu_mm.c           # SRAM internal memory manager (5.5 KB)
├── rknpu_reset.c        # NPU soft reset sequencing (3.2 KB)
├── rknpu_fence.c        # DMA-fence synchronization (1.7 KB)
├── include/
│   ├── rknpu_ioctl.h    # UAPI: submit, mem_create/map/destroy/sync, action
│   ├── rknpu_drv.h      # Core device struct + SoC config struct
│   ├── rknpu_job.h      # Job struct, IRQ handler decls
│   └── ... (debugger, fence, gem, mem, iommu, mm, devfreq, reset headers)
└── Kconfig, Makefile    # Build system
```

**Key structural difference**: NVDLA has a **two-layer architecture** (thin Linux port + HW-agnostic firmware engine). RKNPU is a **flat single-layer driver** with no separate firmware layer — all hardware programming is inline in `rknpu_job.c`.

## 3. Architecture: Firmware vs. No Firmware

### NVDLA: Firmware-Based Architecture
```
UMD → KMD (port/linux/) → Firmware (kmd/firmware/) → HW registers
```

- The `firmware/` directory is **not actual firmware** but a C-based "emulation" of what a real NVDLA firmware would do in hardware
- It reads **operation descriptors** from DRAM, decodes them, and programs HW registers
- Key flow in `scheduler.c`:
  1. `dla_execute_task()` — entry point from KMD
  2. `dla_read_network_config()` — reads network desc, op desc, surface desc from DRAM
  3. `dla_initiate_processors()` — submits first ops from each op_head
  4. Event loop: `dla_process_events()` → `dla_handle_events()` → `dla_op_completion()` → `dla_dequeue_operation()`
- Each processor (CONV, SDP, etc.) has **2 register groups** for pipelining
- Dependency graph managed via `dependency_count` on each `dla_common_op_desc`

### RKNPU: Direct Register Programming
```
UMD → KMD (rknpu_drv.c) → HW registers directly
```

- No firmware layer — UMD prepares all register commands in a task buffer
- KMD writes task buffer address directly to HW registers:
  - `RKNPU_OFFSET_PC_DATA_ADDR` = base address of register command sequence
  - `RKNPU_OFFSET_PC_DATA_AMOUNT` = size of command sequence
  - `RKNPU_OFFSET_PC_OP_EN` = "go" bit
- NPU has an internal **Program Counter (PC)** that autonomously executes the register command sequence
- Two modes: **SLAVE** (KMD writes registers one-by-one) and **PC** (NPU fetches commands from memory)
- The `rknpu_task` struct carries `regcmd_addr`, `regcfg_amount`, `enable_mask`, `int_mask` etc.

## 4. Task / Job Submission

### NVDLA Submission Flow

```
UMD → DRM_IOCTL_NVDLA_SUBMIT
  → nvdla_submit() [nvdla_gem.c:79]
    → copies task from userspace
    → nvdla_task_submit() [nvdla_core_callbacks.c:310]
      → dla_execute_task() [scheduler.c:1067]
        → reads network/op/surface configs from DRAM
        → initiates processors programmatically
      → wait_for_completion() on interrupt
      → dla_process_events() handles completions
```

**IOCTL**: `DRM_IOCTL_NVDLA_SUBMIT`
- `nvdla_submit_args` contains pointer to `nvdla_ioctl_submit_task` array
- Each task contains `num_addresses` + `address_list` (memory handles)
- UMD pre-compiles network descriptor → op descriptor → surface descriptor chain in DRAM
- KMD reads these descriptors from DRAM and programs HW

### RKNPU Submission Flow

```
UMD → DRM_IOCTL_RKNPU_SUBMIT (or IOCTL_RKNPU_SUBMIT via miscdev)
  → rknpu_submit_ioctl() [rknpu_job.c:849]
    → rknpu_submit() [rknpu_job.c:746]
      → rknpu_job_alloc()
      → rknpu_job_schedule()
        → rknpu_job_next() → rknpu_job_commit() → rknpu_job_subcore_commit_pc()
          → writes task's regcmd_addr, int_mask, task_control etc. to registers
          → sets RKNPU_OFFSET_PC_OP_EN = 1
      → rknpu_job_wait() (blocking) or submit + cleanup_work (non-blocking)
```

**IOCTL**: `DRM_IOCTL_RKNPU_SUBMIT` or `IOCTL_RKNPU_SUBMIT`
- `rknpu_submit` contains `task_start`, `task_number`, `task_obj_addr` (GEM object pointer), `core_mask`, `flags`
- UMD pre-compiles **all register configurations into a task array** in a GEM/DMA buffer
- KMD just writes pointers — the NPU PC autonomously executes the commands

## 5. Hardware Register Programming

### NVDLA Register Access Model

```c
// nvdla_core_callbacks.c
void dla_reg_write(void *driver_context, uint32_t addr, uint32_t reg) {
    writel(reg, nvdla_dev->base + addr);
}
```

- Uses `dla_engine_internal.h` macros:
  ```c
  #define glb_reg_write(reg, val)  reg_write(GLB_REG(reg), val)
  #define cdma_reg_write(reg, val) reg_write(CDMA_REG(reg), val)
  #define csc_reg_write(reg, val)  reg_write(CSC_REG(reg), val)
  ```
- Register addresses defined in auto-generated `opendla_initial.h` (1 MB) as macros like `GLB_S_INTR_MASK_0`, `CSC_D_CFG_0`, `CMAC_A_D_CFG_0`
- Firmware code programs each sub-module individually (conv.c programs CDMA, CSC, CMAC, CACC)

### RKNPU Register Access Model

```c
// rknpu_job.c
#define _REG_READ(base, offset)  readl(base + (offset))
#define _REG_WRITE(base, value, offset)  writel(value, base + (offset))
#define REG_WRITE(value, offset)  _REG_WRITE(rknpu_core_base, value, offset)
```

- Register offsets defined in `rknpu_ioctl.h`:
  ```c
  #define RKNPU_OFFSET_PC_DATA_ADDR     0x10
  #define RKNPU_OFFSET_PC_DATA_AMOUNT   0x14
  #define RKNPU_OFFSET_INT_MASK         0x20
  #define RKNPU_OFFSET_INT_CLEAR        0x24
  #define RKNPU_OFFSET_PC_OP_EN         0x08
  ```
- **Much simpler register interface**: ~7 registers for the PC interface, vs. hundreds in NVDLA
- The actual per-op registers are programmed by the UMD into the task buffer, not by the KMD
- RKNPU uses a **register command buffer** approach: the task buffer contains `{address, value}` pairs that the NPU PC writes sequentially

## 6. Interrupt Handling

### NVDLA Interrupt (`engine_isr.c`)

```c
int32_t dla_isr_handler(void *engine_data) {
    mask = glb_reg_read(S_INTR_MASK);
    reg  = glb_reg_read(S_INTR_STATUS);

    if (reg & MASK(GLB_S_INTR_STATUS_0, CACC_DONE_STATUS0))
        processor->groups[0].events |= (1 << DLA_EVENT_OP_COMPLETED);
    if (reg & MASK(GLB_S_INTR_STATUS_0, CACC_DONE_STATUS1))
        processor->groups[1].events |= (1 << DLA_EVENT_OP_COMPLETED);
    // ... same for SDP, CDP, PDP, RUBIK, BDMA, CDMA_DAT, CDMA_WT ...
    glb_reg_write(S_INTR_STATUS, reg);  // write-1-to-clear
}
```

- **Per-processing-unit done status**: Each unit (CACC, SDP, CDP, etc.) has two group status bits
- Also tracks CDMA data/weight load done events for dependency resolution
- ISR sets group events, then `dla_process_events()` handles dependency chain updates

### RKNPU Interrupt (`rknpu_job.c:640`)

```c
irqreturn_t rknpu_irq_handler(int irq, void *data, int core_index) {
    status = REG_READ(RKNPU_OFFSET_INT_STATUS);
    job->int_status[core_index] = status;
    REG_WRITE(RKNPU_INT_CLEAR, RKNPU_OFFSET_INT_CLEAR);
    rknpu_job_done(job, 0, core_index);
    return IRQ_HANDLED;
}
```

- **Single interrupt per core** — just "job done" status
- Validates status against `job->int_mask[core_index]` (fuzz matching)
- Per-core IRQ handlers: `nvdla_core0_irq_handler`, `core1`, `core2`
- On multi-core: each core fires independently, job is done when all cores interrupt

## 7. Memory Management

### NVDLA (`nvdla_gem.c`)

| Feature | Implementation |
|---------|---------------|
| Backend | DRM GEM only |
| Allocation | `dma_alloc_attrs()` with `DMA_ATTR_WRITE_COMBINE` |
| Memory type | DMA-coherent, write-combine |
| Addressing | IOVA via `dma_addr`, 1G coherent region at `0xC0000000` (hardcoded) |
| Address translation | GEM handle → `nvdla_gem_dma_addr()` → DMA address |
| Buffer sharing | DRM PRIME (handle_to_fd, fd_to_handle) |
| IOCTLs | CREATE, MMAP (map_offset), DESTROY (via GEM dumb_destroy) |
| CPU access | `dma_buf_vmap()` / `dma_buf_vunmap()` via `dla_data_write/read` callbacks |
| Sync | Manual via `dma_buf_begin/end_cpu_access()` |

### RKNPU (`rknpu_gem.c` + `rknpu_mem.c`)

| Feature | Implementation |
|---------|---------------|
| Backend | DRM GEM **OR** DMA Heap (compile-time choice) |
| Allocation | DRM: `dma_alloc_attrs()`; DMA Heap: `rk_dma_heap` |
| Memory types | 11 flags: CONTIGUOUS, CACHEABLE, WRITE_COMBINE, KERNEL_MAPPING, IOMMU, SECURE, SRAM, NBUF, DMA32, ZEROING, IOMMU_LIMIT_IOVA_ALIGNMENT |
| Addressing | IOVA (via IOMMU) or physical; `dma_mask` up to 40-bit on RK3588 |
| IOMMU | Full multi-domain support (up to 16 domains) with domain switching |
| Buffer sharing | DRM PRIME + DMA-BUF |
| IOCTLs | CREATE, MAP, DESTROY, SYNC |
| CPU access | `dma_buf_vmap()` + per-object `kv_addr` tracking |
| Sync | Explicit SYNC ioctl (TO_DEVICE / FROM_DEVICE) |
| SRAM/NBUF | Optional high-speed internal memory manager |

**Key difference**: RKNPU has far richer memory management (IOMMU domains, SRAM, NBUF, multiple allocation backends, secure buffers). NVDLA is simpler: single DRM GEM path with hardcoded coherent memory region.

## 8. Multi-Core Support

### NVDLA
- Single-core only in the reference KMD (no multi-core infrastructure)

### RKNPU
- Up to **3 cores** (RK3588: core0, core1, core2)
- `core_mask` in submit: `RKNPU_CORE0_MASK` (0x01), `CORE1` (0x02), `CORE2` (0x04)
- Each core has its own:
  - MMIO base address (`rknpu_dev->base[i]`)
  - IRQ handler (`rknpu_core{i}_irq_handler`)
  - Todo list and running job (`subcore_datas[i]`)
- **Load balancing**: `rknpu_schedule_core_index()` picks core with fewest tasks
- **Synchronization**: Job done only when all cores have fired their interrupts + dma-fence signaling
- **Configurable**: `rknpu_config.core_mask` defines available cores per SoC

## 9. Power Management

### NVDLA
- Minimal: `dma_declare_coherent_memory()` at probe, no explicit power management

### RKNPU
- **Sophisticated multi-layer PM**:
  - Regulators: `vdd` + `mem` voltage control
  - Clocks: `clk_bulk_get_all()` with gate/enable
  - **Multiple power domains** (npu0, npu1, npu2 per core)
  - **DVFS**: `rknpu_devfreq` with devfreq framework
  - **Reference counting**: `rknpu_power_get/put/put_delay`
  - **Auto power-off**: Delayed work queue (3s default), deferred power-off
  - **Runtime PM**: suspend/resume callbacks
  - **Thermal**: IPA power model + devfreq cooling
  - **IOMMU-safe shutdown**: poll until IOMMU fully disabled before cutting power

## 10. SoC Configuration System

### NVDLA
- 3 static configs selected by DT compatible string:
  ```c
  { "nvidia,nvdla_os_initial" → atom=32, BDMA=on, Rubik=on, weight_compress=on }
  { "nvidia,nv_small"         → atom=8,  BDMA=off, Rubik=off }
  { "nvidia,nv_large"         → atom=32, BDMA=off, Rubik=off }
  ```

### RKNPU
- 6+ configs with **detailed per-SoC parameters**:
  ```c
  rk356x_rknpu_config   → 32-bit DMA, 1 core, old top amount
  rk3588_rknpu_config   → 40-bit DMA, 3 cores, pc_data_amount_scale=2
  rk3583_rknpu_config   → 40-bit DMA, 2 cores (rk3588 variant)
  rv1106_rknpu_config   → 32-bit DMA, 1 core, 16-bit task number
  rk3562_rknpu_config   → 40-bit DMA, 1 core, pc_dma_ctrl=1, NBUF
  rk3576_rknpu_config   → 40-bit DMA, 2 cores, state_init, cache_sgt, NBUF
  ```
- Config struct fields: `bw_priority_addr`, `pc_data_amount_scale`, `pc_task_number_bits/mask`, `pc_task_status_offset`, `nbuf_phyaddr/size`, `irqs`, `amount_top/core`, `state_init`, `cache_sgt_init`
- **Dynamic detection**: RK3588 probe checks nvmem for disabled cores (→ RK3583)
- **Per-core IRQ naming**: "npu0_irq", "npu1_irq", "npu2_irq"

## 11. Key Functional Differences

| Feature | NVDLA KMD | RKNPU Driver |
|---------|-----------|--------------|
| **Programming model** | Firmware reads op descriptors from DRAM | UMD pre-computes register commands |
| **Register complexity** | Hundreds of per-submodule regs | ~7 PC interface regs + command buffer |
| **Dependency resolution** | In-kernel firmware (scheduler.c) | Pre-computed by UMD compiler |
| **Pipeline depth** | 2 register groups per processor | Single execution context per core |
| **Synchronization** | `completion` + `spinlock` | dma-fence, waitqueue, completion |
| **Debug interface** | `engine_debug.c` (print traces) | `rknpu_debugger.c` (debugfs/procfs) |
| **Bandwidth monitoring** | None | `rknpu_get_rw_amount()` (DT_WR, DT_RD, WT_RD counters) |
| **Reset handling** | None | `rknpu_reset.c` — soft reset on timeout |
| **Task timeout** | None (infinite wait) | Configurable per-submit, timeout recovery |
| **Buffer sync** | Manual via dma_buf CPU access | Explicit SYNC ioctl (TO/FROM device) |
| **Register map source** | Auto-generated from HW spec (~1 MB header) | Manually defined offsets in ioctl.h |
| **Multi-process** | Single DRM device | Sessions per open fd |

## 12. Code Size Summary

| Metric | NVDLA KMD (Linux port) | NVDLA KMD (firmware) | RKNPU Driver |
|--------|------------------------|---------------------|--------------|
| .c files | 2 | 12 | 10 |
| .h files | 2 (+2 interface) | 8 | 11 |
| Lines (code) | ~900 | ~12,000 (excl. reg maps) | ~11,000 |
| Register maps | — | `opendla_initial.h` 1 MB | Offsets only (~40 lines in ioctl.h) |
| License | Dual BSD/GPL | BSD 3-Clause | GPL v2 |

## 13. Architectural Insights

### Why NVDLA needs a firmware layer
NVDLA's hardware has **fine-grained control** over individual sub-modules (CDMA, CSC, CMAC, CACC for convolution; RDMA, ALU, etc. for SDP). The firmware reads network descriptors from DRAM and orchestrates the pipeline. This makes the KMD simple but requires a complex programming model from the compiler (UMD must emit structured descriptors).

### Why RKNPU is simpler
RKNPU's NPU has a **program counter that executes register write commands from a buffer**. This shifts complexity to the UMD/compiler — the KMD just writes the buffer address and says "go." The hardware handles the rest internally. This is similar to a GPU command buffer model.

### Shared Patterns
Both drivers:
- Use `platform_driver` with `of_match_table` for DT-based probing
- Use DRM GEM for buffer management (RKNPU optionally uses DMA heap)
- Map MMIO registers with `devm_ioremap_resource()`
- Handle interrupts via `devm_request_irq()`
- Use `writel`/`readl` for register access
- Use `copy_from_user`/`copy_to_user` for UAPI data transfer

## 14. Tengine OpenDLA Comparison

Tengine's `source/device/opendla/` backend is not a kernel driver. It is a user-space NVDLA integration layer that splits graphs, builds a loadable, binds input/output tensors, and submits work through the NVDLA runtime.

| Aspect | NVDLA KMD (`opendla.ko`) | Tengine OpenDLA backend | RKNPU Driver (`rknpu.ko`) |
|--------|--------------------------|--------------------------|---------------------------|
| **Layer** | Kernel driver + firmware | User-space backend | Kernel driver + UMD command buffer |
| **Primary role** | Program NVDLA HW from op descriptors | Translate Tengine graphs into NVDLA runtime objects | Program RK NPU PC / job engine |
| **Execution model** | `dla_execute_task()` + scheduler + ISR | `odla_split_graph()` -> `odla_describe()` -> `runtime->load()` -> `bindInputTensor()` / `bindOutputTensor()` -> `runtime->submit()` | UMD-prepared register command buffer |
| **Operator scope** | Full NVDLA op set handled by firmware | Limited backend support; `odla_describe()` publishes allowed ops and blocks the rest | Whatever the RK compiler emits |
| **Precision** | Hardware-dependent | `odla_describe()` advertises int8 for the small config | Compiler-defined |
| **Control surface** | Kernel IOCTLs, MMIO, ISR | C++ runtime objects, no direct register access | Kernel IOCTLs + PC interface registers |
| **Best mental model** | "Kernel-side NVDLA firmware" | "Tengine's NVDLA frontend/backend bridge" | "Command-buffer NPU driver" |

The practical distinction is that Tengine OpenDLA sits above the NVDLA driver stack. It does not replace `opendla.ko`; it prepares and submits graphs through the NVDLA runtime API, while the kernel driver still owns HW programming and interrupt handling.
