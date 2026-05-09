# RK3588 PC Chain Notes

PC-chain submission lets one task's register stream point the PC block at the next
register stream. This is how RKNN split tasks can run as one submit while each
task still has a small `regcfg_amount`.

The current working GEMM PC-chain reference is `examples/gemm.py`. It builds
decoded register streams for tiled GEMM tasks, writes PC tails between segments,
and submits them through the normal GEMM path. The older
`experimental/gemm_pcchain.py` script is only a historical decoded reproduction of
one captured RKNN `394x394x394` stream.

## Register Word Format

Each regcmd qword is:

```text
bits 63:48  target
bits 47:16  value
bits 15:0   register offset
```

Helper:

```python
def E(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr
```

Important PC targets and registers:

```text
TARGET_PC      = 0x0081
TARGET_PC_REG  = 0x0101
TARGET_VERSION = 0x0041

REG_PC_OPERATION_ENABLE  = 0x0008
REG_PC_BASE_ADDRESS      = 0x0010
REG_PC_REGISTER_AMOUNTS  = 0x0014
```

## PC Tail Layout

Put this tail after the body registers for every chained segment:

```python
tail = [
    E(TARGET_PC_REG, REG_PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
    E(TARGET_PC_REG, REG_PC_REGISTER_AMOUNTS, next_amount),
    E(TARGET_VERSION, 0, 0),
    E(TARGET_PC, REG_PC_OPERATION_ENABLE, enable_mask),
]
```

For the last segment:

```python
tail = [
    0,
    E(TARGET_PC_REG, REG_PC_REGISTER_AMOUNTS, 0),
    E(TARGET_VERSION, 0, 0),
    E(TARGET_PC, REG_PC_OPERATION_ENABLE, enable_mask),
]
```

Use the same `enable_mask` that starts the pipeline for that op. GEMM uses
`0x0D` in the captured RKNN chain. Elementwise DPU/RDMA uses `0x18`.

## Address Encoding

`REG_PC_BASE_ADDRESS` stores the aligned low 32 bits of the next regcmd DMA
address:

```python
next_addr_value = next_regcmd_dma & 0xFFFFFFF0
```

This matches the generated register field:

```c
PC_BASE_ADDRESS_PC_SOURCE_ADDR((uint32_t)(next_addr >> 4))
```

Do not store a Python pointer, an object address, or a task descriptor address
there. It is the DMA address of the next register command stream.

## Amount Encoding

`REG_PC_REGISTER_AMOUNTS` is the tricky field. It is not bytes.

Observed rules:

```text
RKNN/rknnops tracked task path:
  PC_REGISTER_AMOUNTS = reg_task_lengths[i]

Verified GEMM 394x394x394 capture:
  task body amounts: 108,108,13,12,12,12,12,12,12,12,17
  PC next amounts:   55,8,7,7,7,7,7,7,7,10,0
```

For the verified GEMM chain, the next amount is not equal to the next task's
descriptor `regcfg_amount`. It is a PC-fetch amount for the next segment, and it
matches the raw RKNN capture exactly.

The verified GEMM capture fits:

```text
PC next amount = ceil(next_body_reg_count / 2) + 1
108 -> 55
13  -> 8
12  -> 7
17  -> 10
```

This formula is not proven universal. In current elementwise tests it can submit
a uniform ADD chain but still does not make later tasks correct.

For elementwise, do not guess between Mesa, RKNN, and matmul amount formulas
without hardware testing. In the current `experimental/multicore_elementwise.py`
tests:

```text
no --pc-chain, offset regcmd mode:
  ioctl returned EINVAL

--pc-chain --pc-chain-style rknnops:
  submit returned 0, task[0] PASS, task[1]/task[2] FAIL

--pc-chain --pc-chain-style mesa:
  ioctl returned EINVAL

--pc-chain --pc-chain-style rknn-matmul:
  ioctl returned EINVAL
```

That means the elementwise PC tail is close enough to submit with the `rknnops`
style, but the chain is not yet executing later tasks correctly.

## Task Descriptor Rules

There are two valid ways to point a task descriptor at a regcmd segment:

```python
# Absolute mode: each task points directly at its segment.
task.regcfg_offset = 0
task.regcmd_addr = regcmd_dma + segment_offset

# Offset mode: each task shares base address and uses regcfg_offset.
task.regcfg_offset = segment_offset
task.regcmd_addr = regcmd_dma
```

For the verified GEMM PC-chain script, absolute mode is used:

```python
task.regcfg_amount = body_amount
task.regcfg_offset = 0
task.regcmd_addr = regcmd_dma + segment_offset
```

The descriptor `regcfg_amount` is the number of body registers the task starts
with. The PC tail lives immediately after those body registers in the regcmd
buffer, but it is not counted in the descriptor amount for the GEMM RKNN capture.

## Building a Chain

Use explicit segment offsets, not implicit list lengths, when trying to reproduce
an RKNN capture:

```python
TASK_OFFSETS = [0x000, 0x380, 0x700]
TASK_AMOUNTS = [108, 108, 13]
PC_NEXT_AMOUNTS = [55, 8, 0]

for task_idx, regs in enumerate(task_regs):
    base = TASK_OFFSETS[task_idx] // 8

    for i in range(segment_stride_qwords):
        regcmd[base + i] = 0

    for i, qword in enumerate(regs):
        regcmd[base + i] = qword

    for i, qword in enumerate(pc_tail(task_idx, regcmd_dma)):
        regcmd[base + TASK_AMOUNTS[task_idx] + i] = qword
```

Then write task descriptors:

```python
for idx, (offset, amount) in enumerate(zip(TASK_OFFSETS, TASK_AMOUNTS)):
    tasks[idx].flags = 0
    tasks[idx].op_idx = op_idx
    tasks[idx].enable_mask = enable_mask
    tasks[idx].int_mask = 0x300
    tasks[idx].int_clear = 0x1FFFF
    tasks[idx].int_status = 0
    tasks[idx].regcfg_amount = amount
    tasks[idx].regcfg_offset = 0
    tasks[idx].regcmd_addr = regcmd_dma + offset
```

## Submit Shape

Elementwise must be made correct as a single-core chain before attempting
nonzero-core or multicore submits. The safe bring-up target is:

```bash
python3 experimental/multicore_elementwise.py --core-ranges core0 --ops ADD,ADD,ADD --n 4096 --pc-chain --pc-chain-style rknnops
```

Do not treat multicore as a way to debug this. If `core0` cannot run all chained
elementwise tasks correctly, split-core execution only adds scheduler/core
mapping variables on top of an unresolved PC/DPU/RDMA re-arm problem.

For a single-core PC chain, use one submit over all tasks:

```python
flags = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG
core_mask = 1
task_number = TASKS
subcore_task[0] = (0, TASKS)
```

For official RKNN-style three-core submit, the verified GEMM script uses:

```python
task_number = TASKS * 3
core_mask = 0
subcore_task[0] = (0, TASKS)
subcore_task[1] = (0, TASKS)
subcore_task[2] = (0, TASKS)
```

Raw nonzero-core or multicore elementwise submits have locked up this kernel in
earlier experiments. Keep the guard in place unless testing with physical reset
access.

## Conv PC-Chain

Conv rawbuf bring-up now follows the same staged pattern as GEMM:

```bash
python3 experimental/conv_rawbuf_small.py --mode core0 --alloc-mode official --flags 0x7
python3 experimental/conv_rawbuf_big.py --mode core0 --alloc-mode official --flags 0x7
python3 experimental/conv_rawbuf_pcchain.py --mode core0 --alloc-mode official --flags 0x7
python3 experimental/conv_pcchain.py --mode core0 --alloc-mode official --flags 0x7 --decoded
```

These scripts capture the RKNN conv task/regcmd buffers from
`~/npu/ops_rknn/conv2d_dump_regs`, patch the DMA addresses, and replay them
through the raw `/dev/dri/card1` submit path. `conv_pcchain.py --decoded`
rebuilds each register command from decoded `(target, register, value)` fields
before submit instead of patching an opaque qword blob.

The conv multi-task replay needs:

```text
flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG = 0x5
mode  = core0
task descriptors from the captured RKNN task buffer
```

Without `RKNPU_JOB_BLOCK`, the captured 11-task conv stream timed out even when
the same buffers and PC tails were used.

## Debug Checklist

1. Generate a dry regcmd buffer and decode every qword as target/register/value.
2. Compare against a known-good raw capture qword-for-qword when one exists.
3. Check that each `PC_BASE_ADDRESS` equals the next segment DMA address.
4. Check that each final segment has `PC_REGISTER_AMOUNTS = 0`.
5. For GEMM captures, keep descriptor `regcfg_amount` equal to the body register
   count. For the minimal ADD/DPU chain, descriptor `regcfg_amount` must include
   the body plus the four-qword PC tail.
6. Test constant data first; then test random data.
7. For elementwise, check each output buffer independently. If task 0 passes and
   task 1/task 2 fail, the first stream ran but the PC chain did not correctly
   advance or re-arm the later task streams.

## ADD Reference Captures

The `~/npu/ops_rknn/add` binary is the current ground-truth reference for
elementwise ADD. Captures taken with `dump_ops_rknn_add.gdb` show:

- width 2: `task_number=3`, `core_mask=0`, `subcore_task[0..2]=(0,1)`
- width 8: `task_number=9`, `core_mask=0`, `subcore_task[0..2]=(0,3)`
- width 16: `task_number=9`, `core_mask=0`, `subcore_task[0..2]=(0,3)`

The ADD regcmd body is much larger than the toy 13-qword stream used in the
current rawbuf probes. The first task descriptor for width 2 reports
`regcfg_amount=69`, and the later descriptors in the same submit also use 69 or
104/26 qword bodies depending on the internal subgraph stage.

The key takeaway is that the working RKNN ADD path does not rely on
`PC_BASE_ADDRESS` chaining in the regcmd stream the way the GEMM PC-chain
capture does. The body/descriptor layout has to match the captured `ops_rknn`
stream first, then any PC-chain experiment can be layered on top.

## Known Good Commands

Verified GEMM:

```bash
python3 examples/gemm.py
```

Historical RKNN `394x394x394` capture reproducer, useful only when comparing
against the old raw captured PC-chain layout:

```bash
python3 experimental/gemm_pcchain.py --dry
python3 experimental/gemm_pcchain.py --constant-data
```

Verified minimal ADD/DPU PC-chain:

```bash
python3 experimental/min_add_pcchain.py
python3 experimental/add_pcchain.py
python3 experimental/multicore_elementwise.py --core-ranges core0 --ops ADD,MUL,SUB --n 4096 --pc-chain --pc-chain-style add --regcmd-mode absolute --submit-block
```

The ADD/DPU fix was to re-arm both ping-pong pointers in every chained segment:

```text
DPU.S_POINTER      = 0x0e
DPU_RDMA.S_POINTER = 0x0e
```

Without these writes, later output buffers are overwritten but contain task 0's
result. With them, use:

```text
PC_REGISTER_AMOUNTS = next body qword count
descriptor amount   = body qwords + four PC-tail qwords
regcmd_addr         = absolute segment DMA address
submit flags        = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG
```

Elementwise multicore testing can start from this single-core chain, but keep
raw nonzero-core submits behind the explicit unsafe-submit guard.
