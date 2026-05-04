from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock, ioctl
from pathlib import Path

import numpy as np


OPS_RKNN = Path("/home/orangepi/npu/ops_rknn")
CONV_DUMP_REGS = OPS_RKNN / "conv2d_dump_regs"
RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_OFFICIAL_TASK = 0x40B
RKNPU_MEM_OFFICIAL_TENSOR = 0x403
RKNPU_JOB_PC = 0x1
RKNPU_JOB_BLOCK = 0x2
RKNPU_JOB_PINGPONG = 0x4
RKNPU_ACT_RESET = 1
LOCK_PATH = "/tmp/rk3588_npu_submit.lock"


class rknpu_mem_create(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("obj_addr", ctypes.c_uint64),
        ("dma_addr", ctypes.c_uint64),
        ("sram_size", ctypes.c_uint64),
    ]


class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64)]


class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]


class rknpu_submit(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("timeout", ctypes.c_uint32),
        ("task_start", ctypes.c_uint32),
        ("task_number", ctypes.c_uint32),
        ("task_counter", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
        ("task_obj_addr", ctypes.c_uint64),
        ("regcfg_obj_addr", ctypes.c_uint64),
        ("task_base_addr", ctypes.c_uint64),
        ("user_data", ctypes.c_uint64),
        ("core_mask", ctypes.c_uint32),
        ("fence_fd", ctypes.c_int32),
        ("subcore_task", rknpu_subcore_task * 5),
    ]


class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]


class rknpu_task(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("op_idx", ctypes.c_uint32),
        ("enable_mask", ctypes.c_uint32),
        ("int_mask", ctypes.c_uint32),
        ("int_clear", ctypes.c_uint32),
        ("int_status", ctypes.c_uint32),
        ("regcfg_amount", ctypes.c_uint32),
        ("regcfg_offset", ctypes.c_uint32),
        ("regcmd_addr", ctypes.c_uint64),
    ]


def _iowr(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | (nr << 0) | (size << 16)


DRM_IOCTL_RKNPU_ACTION = _iowr("d", 0x40, ctypes.sizeof(rknpu_action))
DRM_IOCTL_RKNPU_MEM_CREATE = _iowr("d", 0x42, ctypes.sizeof(rknpu_mem_create))
DRM_IOCTL_RKNPU_MEM_MAP = _iowr("d", 0x43, ctypes.sizeof(rknpu_mem_map))
DRM_IOCTL_RKNPU_SUBMIT = _iowr("d", 0x41, ctypes.sizeof(rknpu_submit))


@dataclass
class Allocation:
    name: str
    dma: int
    size: int
    aligned_size: int
    gem: int


@dataclass
class Capture:
    case_idx: int
    path: Path
    allocations: dict[str, Allocation]
    output_offset: int
    output_size: int


def align_up(value, align):
    return ((value + align - 1) // align) * align


class NpuSubmitLock:
    def __init__(self, path: str = LOCK_PATH):
        self.path = path
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            flock(self.fd, LOCK_EX | LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(f"another NPU experiment holds {self.path}; refusing parallel conv capture/replay") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            flock(self.fd, LOCK_UN)
            os.close(self.fd)
            self.fd = None


def mem_allocate(fd, size, flags):
    mc = rknpu_mem_create(flags=flags, size=size)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, mc)
    mm = rknpu_mem_map(handle=mc.handle)
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mm)
    buf = mmap.mmap(fd, mc.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
    return buf, mc


def parse_capture_log(text: str, case_idx: int, path: Path) -> Capture:
    allocations = {}
    alloc_re = re.compile(
        r"allocated memory, name: (\w+), .* dma addr: 0x([0-9a-f]+), .* size: (\d+), aligned size: (\d+), .* gem name: (\d+)"
    )
    for match in alloc_re.finditer(text):
        name, dma, size, aligned_size, gem = match.groups()
        allocations[name] = Allocation(name, int(dma, 16), int(size), int(aligned_size), int(gem))

    out_re = re.compile(r"OutputOperator output\s+FLOAT16\s+\S+\s+\([^)]+\)\s+\([^)]+\)\s+\|\s+0x([0-9a-f]+)\*?0x([0-9a-f]+)\s+0x([0-9a-f]+)")
    out_matches = out_re.findall(text)
    if not out_matches:
        raise RuntimeError("could not parse native OutputOperator allocation from RKNN log")
    out_start, _out_end, out_size = (int(x, 16) for x in out_matches[-1])
    internal = allocations["internal"]
    return Capture(case_idx, path, allocations, out_start - internal.dma, out_size)


def capture_case(case_idx: int, cache_root: Path | None = None) -> Capture:
    if not CONV_DUMP_REGS.exists():
        raise RuntimeError(f"missing {CONV_DUMP_REGS}")
    cache_root = cache_root or Path(tempfile.mkdtemp(prefix=f"convraw-case{case_idx}-"))
    cache_root.mkdir(parents=True, exist_ok=True)
    script = cache_root / "capture.gdb"
    ignore = max(0, case_idx - 1)
    script.write_text(
        "\n".join(
            [
                "set pagination off",
                "set confirm off",
                "break breakpoint",
                f"ignore 1 {ignore}",
                "commands",
                "  silent",
                f"  shell rm -rf {cache_root}/dump && mkdir -p {cache_root}/dump",
                "  shell cd /home/orangepi/npu/ops_rknn && python3 dump.py 1 2 3 4 5 6 7 8 >/tmp/convraw_dump.log 2>&1",
                f"  shell cp -r /home/orangepi/npu/ops_rknn/dump {cache_root}/gemdump",
                "  quit",
                "end",
                "run",
                "",
            ]
        )
    )
    proc = subprocess.run(
        ["gdb", "-q", "-x", str(script), "--args", str(CONV_DUMP_REGS)],
        cwd=OPS_RKNN,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (cache_root / "capture.log").write_text(proc.stdout)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"gdb capture failed with {proc.returncode}\n{proc.stdout[-4000:]}")
    return parse_capture_log(proc.stdout, case_idx, cache_root)


def translate_addr(value: int, old: Allocation, new_dma: int) -> int | None:
    if old.dma <= value < old.dma + old.aligned_size:
        return new_dma + (value - old.dma)
    return None


def patch_address_value(value: int, cap: Capture, new_dmas: dict[str, int]) -> int:
    for name, alloc in cap.allocations.items():
        patched = translate_addr(value, alloc, new_dmas[name])
        if patched is not None:
            return patched & 0xFFFFFFFF
    return value


def patch_regcmd_buffer(words, cap: Capture, new_dmas: dict[str, int]):
    for i, qword in enumerate(words):
        value = (int(qword) >> 16) & 0xFFFFFFFF
        new_value = patch_address_value(value, cap, new_dmas)
        if new_value != value:
            words[i] = (int(qword) & 0xFFFF00000000FFFF) | (new_value << 16)


def emit(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | (addr & 0xFFFF)


def decoded_regcmd_segments(weight_dump: bytes, task_dump: bytes, cap: Capture, new_dmas: dict[str, int], task_count: int):
    old_weight = cap.allocations["weight"]
    old_tasks = (rknpu_task * task_count).from_buffer_copy(task_dump)
    segments = []
    for task in old_tasks:
        offset = int(task.regcmd_addr) - old_weight.dma
        # Include the four PC-tail qwords immediately after the descriptor body.
        count = int(task.regcfg_amount) + 4
        regs = []
        for i in range(count):
            qword = int.from_bytes(weight_dump[offset + i * 8:offset + i * 8 + 8], "little")
            target = (qword >> 48) & 0xFFFF
            value = (qword >> 16) & 0xFFFFFFFF
            addr = qword & 0xFFFF
            value = patch_address_value(value, cap, new_dmas)
            regs.append((target, addr, value))
        segments.append((offset, regs))
    return segments


def task_segment_ranges(task_dump: bytes, cap: Capture, task_count: int):
    old_weight = cap.allocations["weight"]
    old_tasks = (rknpu_task * task_count).from_buffer_copy(task_dump)
    ranges = []
    for idx, task in enumerate(old_tasks):
        offset = int(task.regcmd_addr) - old_weight.dma
        qwords = int(task.regcfg_amount) + 4
        ranges.append((idx, offset, offset + qwords * 8, qwords))
    return ranges


def in_known_segment(byte_offset: int, ranges) -> bool:
    return any(start <= byte_offset < end for _, start, end, _ in ranges)


def preflight_capture(cap: Capture, verbose=False) -> bool:
    gemdir = cap.path / "gemdump"
    task_alloc = cap.allocations["task"]
    weight_alloc = cap.allocations["weight"]
    task_dump = (gemdir / f"gem{task_alloc.gem}-dump").read_bytes()
    weight_dump = (gemdir / f"gem{weight_alloc.gem}-dump").read_bytes()
    task_count = task_count_from_dump(task_dump)
    old_tasks = (rknpu_task * task_count).from_buffer_copy(task_dump)
    ranges = task_segment_ranges(task_dump, cap, task_count)

    print(f"capture={cap.path} case={cap.case_idx} tasks={task_count}")
    for name in sorted(cap.allocations):
        alloc = cap.allocations[name]
        print(
            f"alloc {name}: gem={alloc.gem} dma=0x{alloc.dma:08x} "
            f"size=0x{alloc.size:x} aligned=0x{alloc.aligned_size:x}"
        )
    print(f"output internal_offset=0x{cap.output_offset:x} size=0x{cap.output_size:x}")

    ok = True
    for idx, task in enumerate(old_tasks):
        _, start, end, qwords = ranges[idx]
        print(
            f"task[{idx}]: op_idx={task.op_idx} enable=0x{task.enable_mask:x} "
            f"amount={task.regcfg_amount} offset=0x{task.regcfg_offset:x} "
            f"regcmd=0x{int(task.regcmd_addr):08x} weight_off=0x{start:x} qwords={qwords}"
        )
        if start < 0 or end > len(weight_dump):
            print(f"  ERROR segment outside weight dump: 0x{start:x}..0x{end:x}")
            ok = False
        else:
            tail = []
            for tail_idx in range(max(0, qwords - 4), qwords):
                off = start + tail_idx * 8
                qword = int.from_bytes(weight_dump[off:off + 8], "little")
                tail.append(f"0x{qword:016x}")
            print(f"  pc_tail={' '.join(tail)}")

    decoded = decoded_regcmd_segments(weight_dump, task_dump, cap, {name: alloc.dma for name, alloc in cap.allocations.items()}, task_count)
    decoded_mismatches = 0
    for offset, regs in decoded:
        for i, (target, addr, value) in enumerate(regs):
            old_qword = int.from_bytes(weight_dump[offset + i * 8:offset + i * 8 + 8], "little")
            new_qword = emit(target, addr, value)
            if old_qword != new_qword:
                decoded_mismatches += 1
                if verbose:
                    print(f"decoded mismatch off=0x{offset + i * 8:x}: old=0x{old_qword:016x} new=0x{new_qword:016x}")
    print(f"decoded identity mismatches={decoded_mismatches}")
    ok = ok and decoded_mismatches == 0

    blind_inside = 0
    blind_outside = 0
    words = np.frombuffer(weight_dump, dtype=np.uint64)
    for i, qword in enumerate(words):
        value = (int(qword) >> 16) & 0xFFFFFFFF
        matches = [name for name, alloc in cap.allocations.items() if translate_addr(value, alloc, alloc.dma) is not None]
        if not matches:
            continue
        byte_offset = i * 8
        if in_known_segment(byte_offset, ranges):
            blind_inside += 1
        else:
            blind_outside += 1
            if verbose:
                print(f"unsafe blind-patch candidate off=0x{byte_offset:x} value=0x{value:08x} alloc={matches[0]}")
    print(f"blind address-like qwords: inside_segments={blind_inside} outside_segments={blind_outside}")
    if blind_outside:
        print("ERROR blind whole-weight patch would touch bytes outside task regcmd segments")
        ok = False
    print("CONV PREFLIGHT PASS" if ok else "CONV PREFLIGHT FAIL")
    return ok


def write_decoded_segments(weight_map, segments):
    words = np.frombuffer(weight_map, dtype=np.uint64)
    for offset, regs in segments:
        base = offset // 8
        for i, (target, addr, value) in enumerate(regs):
            words[base + i] = emit(target, addr, value)


def patch_task_buffer(tasks, cap: Capture, new_dmas: dict[str, int], task_count: int):
    for i in range(task_count):
        old_addr = int(tasks[i].regcmd_addr)
        new_addr = patch_address_value(old_addr & 0xFFFFFFFF, cap, new_dmas)
        tasks[i].regcmd_addr = new_addr
        tasks[i].int_status = 0


def task_count_from_dump(task_dump: bytes) -> int:
    tasks = (rknpu_task * (len(task_dump) // ctypes.sizeof(rknpu_task))).from_buffer_copy(task_dump)
    count = 0
    for task in tasks:
        if task.regcfg_amount == 0 and task.regcmd_addr == 0:
            break
        count += 1
    return count


def submit(fd, task_obj_addr, task_count, mode, timeout, flags):
    official = mode == "official"
    req = rknpu_submit(
        flags=flags,
        timeout=timeout,
        task_start=0,
        task_number=task_count * 3 if official else task_count,
        task_counter=0,
        priority=0,
        task_obj_addr=task_obj_addr,
        regcfg_obj_addr=0,
        task_base_addr=0,
        user_data=0,
        core_mask=0 if official else 1,
        fence_fd=-1,
    )
    req.subcore_task[0] = rknpu_subcore_task(0, task_count)
    req.subcore_task[1] = rknpu_subcore_task(0, task_count if official else 0)
    req.subcore_task[2] = rknpu_subcore_task(0, task_count if official else 0)
    req.subcore_task[3] = rknpu_subcore_task(0, 0)
    req.subcore_task[4] = rknpu_subcore_task(0, 0)
    return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, req)


def reset_npu(fd):
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))


def replay_capture(cap: Capture, mode="core0", timeout=6000, verbose=False, alloc_mode="official", flags=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG, decoded=True) -> bool:
    gemdir = cap.path / "gemdump"
    task_alloc = cap.allocations["task"]
    weight_alloc = cap.allocations["weight"]
    internal_alloc = cap.allocations["internal"]
    input_alloc = cap.allocations["input"]
    output_alloc = cap.allocations["output"]

    task_dump = (gemdir / f"gem{task_alloc.gem}-dump").read_bytes()
    weight_dump = (gemdir / f"gem{weight_alloc.gem}-dump").read_bytes()
    internal_dump = bytearray((gemdir / f"gem{internal_alloc.gem}-dump").read_bytes())
    input_dump = (gemdir / f"gem{input_alloc.gem}-dump").read_bytes()
    output_dump = bytearray((gemdir / f"gem{output_alloc.gem}-dump").read_bytes())

    for i in range(len(internal_dump)):
        internal_dump[i] = 0
    for i in range(len(output_dump)):
        output_dump[i] = 0

    task_count = task_count_from_dump(task_dump)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        task_flags = RKNPU_MEM_OFFICIAL_TASK if alloc_mode == "official" else RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE
        tensor_flags = RKNPU_MEM_OFFICIAL_TENSOR if alloc_mode == "official" else RKNPU_MEM_NON_CACHEABLE
        task_map, task_mc = mem_allocate(fd, len(task_dump), task_flags)
        weight_map, weight_mc = mem_allocate(fd, len(weight_dump), tensor_flags)
        internal_map, internal_mc = mem_allocate(fd, len(internal_dump), tensor_flags)
        input_map, input_mc = mem_allocate(fd, len(input_dump), tensor_flags)
        output_map, output_mc = mem_allocate(fd, len(output_dump), tensor_flags)
        new_dmas = {
            "task": task_mc.dma_addr,
            "weight": weight_mc.dma_addr,
            "internal": internal_mc.dma_addr,
            "input": input_mc.dma_addr,
            "output": output_mc.dma_addr,
        }

        ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), task_dump, len(task_dump))
        ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(weight_map)), weight_dump, len(weight_dump))
        ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(internal_map)), bytes(internal_dump), len(internal_dump))
        ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(input_map)), input_dump, len(input_dump))
        ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), bytes(output_dump), len(output_dump))

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(rknpu_task))
        patch_task_buffer(tasks, cap, new_dmas, task_count)
        if decoded:
            write_decoded_segments(weight_map, decoded_regcmd_segments(weight_dump, task_dump, cap, new_dmas, task_count))
        else:
            regcmd_words = np.frombuffer(weight_map, dtype=np.uint64)
            patch_regcmd_buffer(regcmd_words, cap, new_dmas)

        reset_npu(fd)
        ret = submit(fd, task_mc.obj_addr, task_count, mode, timeout, flags)
        out = np.frombuffer(internal_map, dtype=np.float16, count=cap.output_size // 2, offset=cap.output_offset).copy()
        ok = ret == 0 and np.all(np.isfinite(out)) and float(np.max(np.abs(out))) == 0.0
        if verbose:
            print(f"case={cap.case_idx} mode={mode} tasks={task_count} submit ret={ret}")
            print(f"output offset=0x{cap.output_offset:x} size=0x{cap.output_size:x} max_abs={float(np.max(np.abs(out))):.6f}")
        return ok
    finally:
        os.close(fd)


def run_replay_main(case_idx: int, label: str):
    parser = argparse.ArgumentParser(description=f"Replay RKNN conv raw buffers for {label}.")
    parser.add_argument("--mode", choices=("core0", "official"), default="core0")
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--alloc-mode", choices=("raw", "official"), default="official")
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG)
    parser.add_argument("--raw-blind-patch", action="store_true", help="Unsafe: patch address-looking qwords across the whole weight GEM.")
    parser.add_argument("--preflight-only", action="store_true", help="Capture and inspect RKNN buffers without raw NPU submit.")
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    cap = capture_case(case_idx, args.capture_dir)
    if args.preflight_only:
        return 0 if preflight_capture(cap, args.verbose) else 1
    ok = replay_capture(cap, args.mode, args.timeout, args.verbose, args.alloc_mode, args.flags, not args.raw_blind_patch)
    print(f"{label} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1
