#!/usr/bin/env python3
import argparse
import ctypes
import fcntl
import mmap
import os
import struct


DRM_COMMAND_BASE = 0x40
TASK_STRUCT = struct.Struct("<8IQ")
TASK_FIELDS = (
    "flags",
    "op_idx",
    "enable_mask",
    "int_mask",
    "int_clear",
    "int_status",
    "regcfg_amount",
    "regcfg_offset",
    "regcmd_addr",
)
QWORD_STRUCT = struct.Struct("<Q")


class drm_gem_open(ctypes.Structure):
    _fields_ = [("name", ctypes.c_uint32), ("handle", ctypes.c_uint32), ("size", ctypes.c_uint64)]


class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("offset", ctypes.c_uint64)]


def _IOWR(type_, nr, size):
    return (3 << 30) | (ord(type_) << 8) | nr | (size << 16)


DRM_IOCTL_GEM_OPEN = _IOWR("d", 0x0B, ctypes.sizeof(drm_gem_open))
DRM_IOCTL_RKNPU_MEM_MAP = _IOWR("d", DRM_COMMAND_BASE + 0x03, ctypes.sizeof(rknpu_mem_map))


def decode_tasks(buf):
    rows = []
    for offset in range(0, len(buf) - TASK_STRUCT.size + 1, TASK_STRUCT.size):
        values = TASK_STRUCT.unpack(buf[offset:offset + TASK_STRUCT.size])
        row = dict(zip(TASK_FIELDS, values))
        if row["regcfg_amount"] == 0 and row["regcmd_addr"] == 0:
            continue
        if row["regcfg_amount"] > 0x10000:
            continue
        if row["regcmd_addr"] & 0xf:
            continue
        rows.append((offset, row))
    return rows


def parse_qword_window(spec):
    fields = spec.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("qword window must be GEM:OFFSET:COUNT")
    try:
        return tuple(int(field, 0) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def dump_gem(fd, flink, qword_windows):
    gem = drm_gem_open(name=flink)
    fcntl.ioctl(fd, DRM_IOCTL_GEM_OPEN, gem)
    mapper = rknpu_mem_map(handle=gem.handle)
    fcntl.ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, mapper)
    mm = mmap.mmap(fd, gem.size, mmap.MAP_SHARED, mmap.PROT_READ, offset=mapper.offset)
    try:
        data = mm[:]
    finally:
        mm.close()
    print(f"GEM {flink}: handle={gem.handle} size={gem.size} map_offset=0x{mapper.offset:x}")
    tasks = decode_tasks(data)
    print(f"  task_like_entries={len(tasks)}")
    for idx, (offset, row) in enumerate(tasks):
        print(
            f"  task_like[{idx}] off=0x{offset:04x} flags=0x{row['flags']:08x} "
            f"op_idx={row['op_idx']} enable_mask=0x{row['enable_mask']:08x} "
            f"int_mask=0x{row['int_mask']:08x} int_clear=0x{row['int_clear']:08x} "
            f"int_status=0x{row['int_status']:08x} regcfg_amount={row['regcfg_amount']} "
            f"regcfg_offset={row['regcfg_offset']} regcmd_addr=0x{row['regcmd_addr']:016x}"
        )
    for window_gem, offset, count in qword_windows:
        if window_gem != flink:
            continue
        end = offset + count * QWORD_STRUCT.size
        if end > len(data):
            print(f"  QWORD_WINDOW off=0x{offset:x} count={count} unavailable size={len(data)}")
            continue
        print(f"  QWORD_WINDOW off=0x{offset:x} count={count}")
        for idx in range(count):
            qword_off = offset + idx * QWORD_STRUCT.size
            value = QWORD_STRUCT.unpack_from(data, qword_off)[0]
            print(f"  regcmd_qword[{idx}] off=0x{qword_off:x} value=0x{value:016x}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gems", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--device", default="/dev/dri/card1")
    parser.add_argument("--qword-window", action="append", type=parse_qword_window, default=[])
    args = parser.parse_args()
    fd = os.open(args.device, os.O_RDWR)
    try:
        for flink in args.gems:
            try:
                dump_gem(fd, flink, args.qword_window)
            except OSError as exc:
                print(f"GEM {flink}: unavailable: {exc}")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
