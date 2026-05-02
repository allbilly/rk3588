# conv.py — Remaining Problems

## Known Limitations

- **Cross-process isolation (P7)**: one bad NPU submission can corrupt state for later processes, even after `reset_npu()`. Recovery still requires reloading the kernel module.
