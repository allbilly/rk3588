# rk3588 Documentation

## TODO

- test openrockchip kernel driver merged in kernel 6.18
- merge my rk3588 npu knowledge to mesa

## What is rk3588

rk3588 is a NVDLA like NPU designed for convulution.

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