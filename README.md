⚠️ Documentations still WIP. 

# RK3588 

This repo run ops on RK3588 NPU (NVDLA based) with pure python and NPU register programming. No RKNN, no compiler, nothing.

Thanks to prior effort by [mtx512](https://github.com/mtx512/rk3588-npu), [Tomeu Vizoso](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/gallium/drivers/rocket), [liej6799](https://github.com/liej6799/rk3588)

The NPU can be described as a 5-stage pipeline: CNA → CORE → DPU → PPU, with PC controlling operation enable and RDMA feeding data in parallel. The output of each stage passes to the next. With this in mind, it will be much easier to understand the registers.

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│   CNA   │───▶│  CORE   │───▶│   DPU   │───▶│   PPU   │───▶│   PC    │
│(Conv    │    │(Compute │    │(Data    │    │(Pooling │    │(Process │
│ Neural  │    │ Engine) │    │Process) │    │Process) │    │ Control)│
│ Accel)  │    │         │    │         │    │         │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
Weight/Feature   MAC Array      Post-Proc      Pooling        Operation
   Data          Accumulation   (BS/BN/EW)     (AVG/MAX)      Enable

┌─────────┐
│  RDMA   │  (0x2001, range 0x5000) — separate DMA for
│(Read    │   elementwise input/weight data, not in the
│  DMA)   │   CNA→CORE pipeline. Selected by enable_mask
└─────────┘   in struct_rknpu_task (0xd for conv, 0x18 for gemm).
```

TODO
- fork and add more regsiters to [mesa rocket](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/gallium/drivers/rocket)
- check [NVDLA/vp](https://github.com/nvdla/vp)  
- op fusing examples

# For Normal user

Tested on
- ✅ Hardware: OrangePi 5 16GB
- ✅ [OrangePi Ubuntu 1.2.2 Jammy](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-pi-5.html)  Linux 6.1.99-rockchip-rk3588 OrangePi 5 
- ✅ [Armbian Ubuntu 24.04](https://www.armbian.com/boards/orangepi5) 6.18.24-current-rockchip64

## 1. Running simple examples 

The python files in examples are self-contained. Zero dependency.
Simply run with python.

🍊 OrangePi Ubuntu with RKNPU driver preinstalled,
```bash
pip install numpy
python examples/elementwise.py
python examples/gemm.py
python examples/conv.py
python examples/conv_gemm.py    # matmul is just 1x1 conv, so lets fuse into one file

python experimental/gemm_int8.py
python experimental/gemm_int16.py
python experimental/conv.py
python experimental/pool.py
```

🐧 Mainline Armbian Ubuntu with ROCKET driver preinstalled,
```bash
python experimental/mainline6_18/elementwise.py
python experimental/mainline6_18/gemm.py
python experimental/mainline6_18/conv.py
python experimental/mainline6_18/pool.py
```

⚙️ 
If you are on Mainline, you can test out [mesa teflon](https://docs.mesa3d.org/teflon.html) as well. More instrusctions on [mesa_example_run.md](./mesa_example_run.md)
```bash
sudo apt-get -y build-dep mesa
sudo apt-get -y install git cmake
git clone https://gitlab.freedesktop.org/mesa/mesa.git
cd mesa
meson setup build -Dgallium-drivers=rocket -Dvulkan-drivers= -Dteflon=true
meson compile -C build
uv venv python=3.10 && source .venv/bin/activate
uv pip install tflite-runtime==2.13.0 numpy<2 pillow
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp

TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs python3.10 src/gallium/frontends/teflon/tests/classification.py \
       -i ./grace_hopper.bmp \
       -m src/gallium/targets/teflon/tests/models/mobilenetv1/mobilenet_v1_1_224_quant.tflite \
       -l src/gallium/frontends/teflon/tests/labels_mobilenet_quant_v1_224.txt \
       -e build/src/gallium/targets/teflon/libteflon.so
```

## 2. PC chaining 

Some example like elementwise.py above just contain one op per submit, simple but overhead will be large for many ops.
Program counter chaining is needed to run multiple ops in one submit.

```bash
python experimental/pool_pcchain.py
```

What if the mamtul size is too large
- You can just split and pc chain it to one submit
- When mamtul size > 1x8192x8192, it splited by N, such that C[:, j] = A × B[:, j]
- C[:, :8144] = A × B[:, :8144]
- C[:, 8144:8144+48] = A × B[:, 8144:8144+48]

## 3. Multicore ⚠️ WIP

The NPU comes with 3 core. You can run different op on different core at the same time.
Or split GEMM at the N dimension and run at 3 cores at the same time.

I will add examples of ADD/SUB/MUL running at the same time, GEMM split and mainline.
For now, test run pool_multicore.py first.
```bash
python experimental/pool_multicore.py
```

## 4. How to do matmul in RK3588

RK3588 NPU is NVDLA based NPU designed for Convulution workload.
So how can we run matmul on it?

Please view the below carefully crafted ascii diagram on a wide screen.
```
In matmul, we have C(M,N) = A(M,K) * B(K,N)
c[m][n] = Σₖ A[m][k] · B[k][n]
                                                <----------- N columns ----------->
                                          ^   ┌────────────────────────────────────────┐
                                          |   │ b[0][0] b[0][1] ... b[0][N-1]          │
                                          |   │ b[1][0] b[1][1] ... b[1][N-1]          │
                                   K rows |   │ b[2][0] b[2][1] ... b[2][N-1]          │
                                          |   │     .        .        .                │
                                          |   │ b[K-1][0] b[K-1][1] ... b[K-1][N-1]    │
                                          V   └────────────────────────────────────────┘
        <----------- K columns ----------->
^      ┌────────────────────────────────────┐ ┌────────────────────────────────────────┐      
|      │ a[0][0] a[0][1] ... a[0][K-1]      │ │ c[0][0] c[0][1] ... c[0][N-1]          │
|      │ a[1][0] a[1][1] ... a[1][K-1]      │ │ c[1][0] c[1][1] ... c[1][N-1]          │
M rows │ a[2][0] a[2][1] ... a[2][K-1]      │ │ c[2][0] c[2][1] ... c[2][N-1]          │
|      │     .        .        .            │ │     .        .        .                │
|      │ a[M-1][0] a[M-1][1] ... a[M-1][K-1]│ | c[M-1][0] c[M-1][1] ... c[M-1][N-1]    │
V      └────────────────────────────────────┘ └────────────────────────────────────────┘

As RK3588 is a NVDLA-based NPU which designed for convulution, we can use a special convulution config to do matmul.
For normal 2d conv, 

        Feature Data (H,W=1)   Kernel(R=1,S=1)
        ┌────────────┐         ┌─────┐
        │ a[0][0]    │         │  k  │
        │ a[2][0]    │         └─────┘
        │   ...      │
        │ a[H-1][0]  │
        └────────────┘

We need 3d conv to do matmul, and N kernels
                                                                       Kernel (N=N, R=1, S=1, C=K)
        Feature Map A  (H=M, W=1, C=K)                                  kernel 0
                  +───────── front-face channels C ─────────────+.         +───────── front-face channels C ──────+
        W=1      /                                             /|     S=1 /                                      /|                    
                +─────────────────────────────────────────────+ |        +──────────────────────────────────────+ |
        ^       | a[0][0][0]   a[0][0][1]   ... a[0][0][N-1]  | |   R=1  | b[0][0][0]   b[0][1]   ... b[0][K-1] | |
        |       | a[1][0][0]   a[1][0][1]   ... a[1][0][N-1]  | |        +──────────────────────────────────────+/
        |       | a[2][0][0]   a[2][0][1]   ... a[2][0][N-1]  | |       kernel 1
        H rows  |        .        .           .               | |          +───────── front-face channels C ──────+
        |       | a[H-1][0][0] a[H-1][1] ... a[H-1][0][N-1]   | /     S=1 /                                      /|
        V       +─────────────────────────────────────────────+/        +──────────────────────────────────────+ |
                                                                    R=1 | b[0][0][0]   b[0][1]   ... b[0][K-1] | |
                                                                        +──────────────────────────────────────+/
                                                                        .
                                                                        .

                                                                        kernel N-1
                                                                          +───────── front-face channels C ──────+
                                                                     S=1 /                                      /|
                                                                        +──────────────────────────────────────+ |
                                                                    R=1 | b[0][0][0]   b[0][1]   ... b[0][K-1] | |
                                                                        +──────────────────────────────────────+/

Input vector:  A[h, :]
Kernel n:      B[:, n]
out[h][n] = Σ_k A[h][k] * B[k][n]

        Output Feature Map C  (H=M, W=1, C=N)                                  
                  +───────── front-face channels C=N ───────────+.  
        W=1      /                                             /|                
                +─────────────────────────────────────────────+ |    
        ^       | a[0][0][0]   a[0][0][1]   ... a[0][0][N-1]  | |  
        |       | a[1][0][0]   a[1][0][1]   ... a[1][0][N-1]  | |   
        |       | a[2][0][0]   a[2][0][1]   ... a[2][0][N-1]  | |     
        M rows  |        .        .           .               | |   
        |       | a[M-1][0][0] a[M-1][1] ... a[M-1][0][N-1]   | /    
        V       +─────────────────────────────────────────────+/     
```

# For Developer

## 0. Known issue and FAQ
- Do not launch two NPU target commands in parallel. It crashed the kerenel and need replug power to reboot. For parallel workload, checkout exampels/pool_multicore.py

How to convert onnx to rknn
- python3 -m rknn.api.rknn_convert -t rk3588 -i /home/orangepi/npu/models/8_add.onnx -o /home/orangepi/npu/models/

failed to allocate handle, ret: -1, errno: 14, errstr: Bad address and need reboot after 32 times mem_create and destroy for 8165x8165
- 100 times on rknn no problem
- mem_destroy was called with input_dma instead of the required input_obj, so the input buffer never got destroyed.
- munmap was given non‑page‑aligned sizes (e.g., 133,350,848 bytes), which typically fails silently and leaves VMAs mapped; over many loops this leaks address space/resources. mem_create/mmap also used unaligned sizes.


## 1. Registers

checkout experimental/rockchip.py

Common target IDs:
```text
0x0201  CNA / convolution front-end
0x0801  CORE / MAC array
0x1001  DPU / output and elementwise
0x2001  RDMA
0x0081  PC / operation enable
```

Here's my registers note, still need cleanup

## CNA registers - diemsion

Feature Map A  (H=M, W=1, C=K) 
```
datain_height = M
datain_width = 1
datain_channel = K
```

Kernel (N=N, R=1, S=1, C=K)
```
weight_width = 1
weight_height = 1
weight_kernels = N
```

Stride value in x and y direction
```
conv_x_stride = 1
conv_y_stride = 1
```

Output Feature Map C  (H=M, W=1, C=N)
```
dataout_height = M
dataout_width = 1
```

for CONV should use formula but our MATMUL case just need H=M, W=1, C=N
H_out = floor((H + pad_top + pad_bottom - k_h) / stride_y) + 1
W_out = floor((W + pad_left + pad_right - k_w) / stride_x) + 1

dma_width = datain_width
dma_height = cna_desc.datain_height
dma_channel = cna_desc.datain_channel
RKNN: dma_channel = align_in

## CNA registers - non-diemsion

Dataout Atomics 
TRM: Data atomics after convolution which is data out total pixels number.
I think its like CUDA atomicAdd for each output pixel 
```
dataout_atomics = dataout_width * dataout_height
```

Feature Grains
TRM: Feature data rows needs to be buffered before convolution start. Its suggested to set this field as y_stride+weight_height+1.
In matmul mode this over-buffers rows so the whole MxK block fits, which is why M+1 is used instead of the TRM minimum.
```
feature_grains = M + 1
```

weight_bytes_per_kernel
Jasbir: weight_bytes_per_kernel = weight_width * weight_height * datain_channel * sizeof(__fp16);
RKNN:   weight_bytes_per_kernel = align_in * sizeof(__fp16);

weight_bytes_total
Jasbir: weight_bytes = weight_bytes_per_kernel * cna_desc.weight_kernels;
RKNN:   weight_bytes_total = weight_bytes_per_kernel * align_out;

CBUF Weight Bank and Data Bank
CBUF is Multi-bank SRAM, shared for feature and weight
```
fd_bytes = M × K × sizeof(type)
fd_banks = ceil(fd_bytes / CBUF_BANK_SIZE)
weight_bytes_total = K x N x sizeof(type)
weight_bank = ceil(weight_bytes_total / NPU_CBUF_BANK_SIZE)
```

Data Entries
TRM: How many banks space needed to store one feature map row.

in matmul, datain_width=dataout_width=1
```
data_entries = ceil((datain_width * datain_channel) / 32);
RKNN: 
int cbuf_entries = ((dataout_width * align_in) + 31) / 32;
if (cbuf_entries <= 0) cbuf_entries = 1;
```

weight_burst_len and data_burst_len
AXI burst length for weight/feature data DMA.
```
weight_burst_len = 15
data_burst_len = 15
```

line_stride
line_stride = datain_width * 4

// TODO fully fix RKNN hardcode
surf_stride

Maths:
lane_span_bytes = rows_per_lane * line_stride
surf_stride = lane_span_bytes - line_stride
           = (rows_per_lane - 1) * line_stride
           = (H/4 - 1) * line_stride

```
surf_stride = (line_stride * ((datain_height / 4) - 1));
```

RKNN: hardcoded 
surf_stride = 268435453 if is_matmul_768 || is_matmul_768_2048 || is_matmul_2048


NVDLA SW CONV: treats a surface as the full plane
lineStride = W * channelsPerGroup * bytesPerElement
surf_stride = lineStride * H

ONNC CONV: 
lineStride = align(W * FEATURE_ATOM_CUBE_SIZE, 32)
stride_surface = lineStride * H

GROUP_LINE_OFF
TRM: Group line fetch, 0: enable, 1:disable. This setting only influence line fetch efficiency.
But it does affect result correctness

RKNN: CNA_CONV_CON1_GROUP_LINE_OFF(1) if (!is_matmul_64 && !is_matmul_256 && !is_matmul_768 && !is_matmul_768_2048 && !is_matmul_2048)

DATA_CUBE_NOTCH_ADDR
notch_val
TRM: notch_addr_1, How many pixels from the end of width to the end of the shape line end.
TRM: notch_addr_0, How many pixels from the end of width to the end of the shape line end.

surface_add
TRM: How many surfaces in a row.
```
surface_add = dst_surf_stride * (align_out / 8u);
surface_add = dst_surf_stride * 4u if (is_matmul_64 || is_matmul_256 || is_matmul_768 || is_matmul_768_2048 || is_matmul_2048) 
```


# other config registers

qd_en
TRM: Quantify feature data calculate enable
```
qd_en=1
```

data_sign
Feature data is signed or not.  0:unsigned
```
data_sign = 1
```

cvt_type
Cal type of the input convert. 0: Multiply first then add,  1: revesr
```
cvt_type = 1
```

cvt_bypass
Bypass input convert.
```
cvt_bypass = 1 
```

cvt_scale0123
Multiplier operand for 1st/2nd/3rd/4th channel.
```
cvt_scale0=1
```

feature_base_addr
```
feature_base_addr = input_dma
```

decompress_addr0
```
decompress_addr0 = params->weights_dma


```
we have input in fp16 and process in fp16
```
cna_desc.in_precision = precision_float16;
cna_desc.proc_precision = precision_float16;
```

```
EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) | DPU_S_POINTER_POINTER_PP_EN(1));
```

## 2. RKNPU vs rocket mainline driver

| Aspect | rknpu (vendor) | rocket (upstream) | Effort |
|--------|----------------|-------------------|--------|
| IOCTL prefix | `DRM_IOCTL_RKNPU_*` | `DRM_IOCTL_ROCKET_*` | High |
| Memory alloc | `RKNPU_MEM_CREATE` → DMA addr + GEM handle | `ROCKET_CREATE_BO` → GEM handle + offset | Medium |
| Submit model | 64-bit packed register entries in DMA task buffer (user builds `task_obj_addr`) | `drm_rocket_submit` with job arrays, structured job descriptors | High |
| Cache sync | `RKNPU_MEM_SYNC` ioctl | `ROCKET_PREP_BO` / `ROCKET_FINI_BO` | Medium |
| Fence | `fence_fd = -1` (none) | `dma_fence` + `sync_file` | Low (ignore) |
| Multi-core | Core mask in submit struct | `drm_sched` per-core entities | Low (same capability) |

What stays the same
- The **register-level hardware programming** — NC1HWC2 format, weight packing, conv/gemm register setup, ALU ops, DPU config — is purely hardware-defined and **identical** across both drivers. All the reverse-engineering work in `rknnops.h`, `rockchip.py`, `conv.c`, `gemm.c` transfers unchanged.

Recommended approach
- **Option A** (pragmatic): Stick with the vendor rknpu driver (already in your kernel). Focus on math/format correctness. The driver works fine for research.
- **Option B** (shim): Add an abstraction layer in `rknnops.h` over the IOCTL calls. Implement two backends — `DRM_IOCTL_RKNPU_*` and `DRM_IOCTL_ROCKET_*` — with compile-time or runtime selection.
- **Option C** (full port): Rewrite the IOCTL layer for the `rocket` API. Hardware math stays; kernel communication changes. Estimated ~500-800 lines in `rknnops.h` need updating (all `rknpu_mem_*` and `rknpu_submit` calls).

### Driver Comparison: rknpu (0.9.8) vs rocket (Linux 6.18)

| Feature | rknpu 0.9.8 (stock OrangePi 6.1.99) | rocket (upstream Linux 6.18) |
|---|---|---|
| Kernel location | `drivers/rknpu/` (vendored) | `drivers/accel/rocket/` (upstream) |
| Build mode | **Builtin** (compiled into kernel) | **Module** (modprobe/rmmod) |
| Timeout mechanism | Userspace `wait_event_timeout()` with 3-retry loop (best-effort) | DRM scheduler hard timeout, **500ms**, always fires |
| NPU reset on hang | `rknpu_soft_reset()` — assert/deassert reset lines + IOMMU re-sync | `rocket_core_reset()` — assert/deassert 2 reset lines via reset controller framework |
| Scheduler recovery | None — soft reset inline in abort path | `drm_sched_stop()` → reset → `drm_sched_start()` |
| IOMMU cleanup | Manually detach/reattach in soft reset path | Detaches IOMMU group, full cleanup before reset |
| Module unload on hang | **Impossible** (builtin) | **Possible** (`rmmod rocket`) |
| Recovery on full hang | **Reboot required** — NPU bus hang + no escape | Likely recoverable via reset + scheduler restart; module reload as last resort |
| Source | [allbilly/rknpu_driver](https://github.com/allbilly/rknpu_driver) | [torvalds/linux: drivers/accel/rocket/](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/accel/rocket) |

**Key takeaway:** rknpu 0.9.8 is builtin with no escape hatch — when the NPU hangs hard enough to survive soft reset, reboot is the only option. The rocket driver uses the DRM scheduler's proper timeout framework (kernel-enforced, not userspace), has a cleaner reset path with scheduler restart, and supports module unload as a last resort.

## 2. How to submit task to NPU with IOCTL

To submit a compute job to the NPU using ioctl
```C
int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
if (ret < 0) {
    perror("DRM_IOCTL_RKNPU_SUBMIT");
}
```

DRM_IOCTL_RKNPU_SUBMIT is defined in rknpu_ioctl.h
which is provided in the official rknpu-driver

https://github.com/allbilly/rknpu_driver/blob/0e23a914e5322d6b7cdaf3de6e91fdf76b2a9055/include/rknpu_ioctl.h#L310-L311

```C
#define DRM_IOCTL_RKNPU_SUBMIT \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_SUBMIT, struct rknpu_submit)
```

The DRM_IOWR macro wraps the standard Linux _IOWR ioctl macro, using the DRM ioctl base character 'd'. It's used for ioctl commands that both pass data to the kernel and receive data back.
And it is defined at linux/tools/include/uapi/drm/drm.h

https://github.com/torvalds/linux/blob/9e1e9d660255d7216067193d774f338d08d8528d/tools/include/uapi/drm/drm.h#L1098

```C
#define DRM_IOCTL_BASE			'd'  
#define DRM_IOWR(nr,type)		_IOWR(DRM_IOCTL_BASE,nr,type)
```

Now we have DRM_IOCTL_RKNPU_SUBMIT figured out,
what kind of struct should we put into &submit  
It is also defined in the same rknpu_ioctl.h linked above
```C
/**
 * struct rknpu_submit structure for job submit
 *
 * @flags: flags for job submit
 * @timeout: submit timeout
 * @task_start: task start index
 * @task_number: task number
 * @task_counter: task counter
 * @priority: submit priority
 * @task_obj_addr: address of task object
 * @iommu_domain_id: iommu domain id
 * @reserved: just padding to be 64-bit aligned.
 * @task_base_addr: task base address
 * @hw_elapse_time: hardware elapse time
 * @core_mask: core mask of rknpu
 * @fence_fd: dma fence fd
 * @subcore_task: subcore task
 *
 */
struct rknpu_submit {
	__u32 flags;
	__u32 timeout;
	__u32 task_start;
	__u32 task_number;
	__u32 task_counter;
	__s32 priority;
	__u64 task_obj_addr;
	__u32 iommu_domain_id;
	__u32 reserved;
	__u64 task_base_addr;
	__s64 hw_elapse_time;
	__u32 core_mask;
	__s32 fence_fd;
	struct rknpu_subcore_task subcore_task[5];
};
```

note that the rknpu_submit struct has been changed across version
the old old one in rknpu driver 0.9.6 is 
```C
struct rknpu_submit {
        __u32 flags;
        __u32 timeout;
        __u32 task_start;
        __u32 task_number;
        __u32 task_counter;
        __s32 priority;
        __u64 task_obj_addr;
        __u64 regcfg_obj_addr;
        __u64 task_base_addr;
        __u64 user_data;
        __u32 core_mask;
        __s32 fence_fd;
        struct rknpu_subcore_task subcore_task[5];
};
```

```diff
1c1
< 0.9.6
---
> 0.9.8
10c10,11
<         __u64 regcfg_obj_addr;
---
>       __u32 iommu_domain_id;
>       __u32 reserved;
12c13
<         __u64 user_data;
---
>       __s64 hw_elapse_time;
```



From gdb,


```C
int submitTask(int fd, uint64_t tasks_obj, size_t task_count){
   if (task_count == 0) task_count = 1;
   uint32_t submit_flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK;
   if (current_alu_algorithm != 13) {
      submit_flags |= RKNPU_JOB_PINGPONG;
   }
   printf("submitTask flags %u\n", submit_flags);
   struct rknpu_submit submit = {
      .flags = submit_flags,
      .timeout = 10000,
      .task_start = 0,
      .task_number = (uint32_t)task_count,
      .task_counter = 0,
      .priority = 0,
      .task_obj_addr = tasks_obj,
      .iommu_domain_id = 0,
      .reserved = 0,
      .task_base_addr = 0,
      .hw_elapse_time = 0, 
      .core_mask = 0,
      .fence_fd = -1,
      .subcore_task = {
         {.task_start = 0, .task_number = (uint32_t)task_count},
         {.task_start = 0, .task_number = 0},
         {.task_start = 0, .task_number = 0},
      }, // Only use core 0
   };
   printf("DRM_IOCTL_RKNPU_SUBMIT\n");
   int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
   if (ret < 0) {
      perror("DRM_IOCTL_RKNPU_SUBMIT");
   }
   return ret;
}
```

# Reference
- https://github.com/liej6799/rk3588
- https://github.com/nvdla/hw
- https://github.com/nvdla/sw
- https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/accel/rocket
- https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/gallium/drivers/rocket>
- https://github.com/nvdla/vp
- https://clehaxze.tw/gemlog/2023/08-26-benchmarking-rk3588-npu-matrix-multiplcation-performance.gmi
- https://clehaxze.tw/gemlog/2023/09-02-benchmarking-rk3588-npu-matrix-multiplcation-performance-ep2.gmi
- https://clehaxze.tw/gemlog/2024/02-14-benchmarking-rk3588-npu-matrix-multiplcation-performance-ep2.gmi
- https://clehaxze.tw/gemlog/2023/12-24-accelerating-piper-text-to-speech-on-the-rk3588-npu.gmi