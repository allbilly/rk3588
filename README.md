⚠️ Documentations still WIP. 
- For now u can read the code at examples/elementwise.py or [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/allbilly/rk3588) (it has up to 1 week delay)


# RK3588 

This repo run ops on RK3588 NPU with pure python and NPU register programming. No RKNN, no ONNX, no compiler, nothing.

TODO
- Aim to write details documentation as my [ane](https://github.com/allbilly/ane) repo
- test openrockchip kernel driver merged in kernel 6.18 thanks to [Tomeu Vizoso](https://blog.tomeuvizoso.net/2025/07/rockchip-npu-update-6-we-are-in-mainline.html)
- PR my rk3588 npu knowledge to [mesa](https://gitlab.freedesktop.org/mesa/mesa)
- NVDLA docker simulation
- multi op in one submit
- check nvdla/hw source code or ghidra rknn to find solution to too many magic in conv matmul input packing and too much magic if


# For Normal user

✅ Tested on Official Ubuntu image Orange Pi 1.2.2 Jammy with Linux 6.1.99-rockchip-rk3588 OrangePi 5
http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-pi-5.html

## What is rk3588

rk3588 is a NVDLA like NPU designed for convulution.
You can send op to npu with pure python.

```bash
python examples/elementwise.py
python examples/gemm.py
python examples/conv.py
```

# For Developer

## Adapting This Codebase: rknpu (vendor) → rocket (upstream mainline)

The upstream [`rocket` driver](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/accel/rocket) (`drivers/accel/rocket/`, merged in **v6.10-rc1**) already supports RK3588 via `compatible = "rockchip,rk3588-rknn-core"`. It is authored by Tomeu Vizoso (Collabora) and uses the proper DRM accel framework.

### What changes

| Aspect | rknpu (vendor) | rocket (upstream) | Effort |
|--------|----------------|-------------------|--------|
| IOCTL prefix | `DRM_IOCTL_RKNPU_*` | `DRM_IOCTL_ROCKET_*` | High |
| Memory alloc | `RKNPU_MEM_CREATE` → DMA addr + GEM handle | `ROCKET_CREATE_BO` → GEM handle + offset | Medium |
| Submit model | 64-bit packed register entries in DMA task buffer (user builds `task_obj_addr`) | `drm_rocket_submit` with job arrays, structured job descriptors | High |
| Cache sync | `RKNPU_MEM_SYNC` ioctl | `ROCKET_PREP_BO` / `ROCKET_FINI_BO` | Medium |
| Fence | `fence_fd = -1` (none) | `dma_fence` + `sync_file` | Low (ignore) |
| Multi-core | Core mask in submit struct | `drm_sched` per-core entities | Low (same capability) |

### What stays the same

The **register-level hardware programming** — NC1HWC2 format, weight packing, conv/gemm register setup, ALU ops, DPU config — is purely hardware-defined and **identical** across both drivers. All the reverse-engineering work in `rknnops.h`, `rockchip.py`, `conv.c`, `gemm.c` transfers unchanged.

### Recommended approach

**Option A** (pragmatic): Stick with the vendor rknpu driver (already in your kernel). Focus on math/format correctness. The driver works fine for research.

**Option B** (shim): Add an abstraction layer in `rknnops.h` over the IOCTL calls. Implement two backends — `DRM_IOCTL_RKNPU_*` and `DRM_IOCTL_ROCKET_*` — with compile-time or runtime selection.

**Option C** (full port): Rewrite the IOCTL layer for the `rocket` API. Hardware math stays; kernel communication changes. Estimated ~500-800 lines in `rknnops.h` need updating (all `rknpu_mem_*` and `rknpu_submit` calls).



## Driver Comparison: rknpu (0.9.8) vs rocket (Linux 6.18)

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

## How to submit task to NPU with IOCTL

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
