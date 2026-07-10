#!/usr/bin/env python3
"""Read selected RK3588 RKNPU core registers without modifying device state."""

import argparse
import mmap
import os
import struct


CORE_BASES = (0xFDAB0000, 0xFDAC0000, 0xFDAD0000)
CORE_WINDOW_SIZE = 0x10000
REGISTERS = (
    ("VERSION", 0x0000),
    ("VERSION_NUM", 0x0004),
    ("PC_OP_EN", 0x0008),
    ("PC_DATA_ADDR", 0x0010),
    ("PC_DATA_AMOUNT", 0x0014),
    ("INT_MASK", 0x0020),
    ("INT_CLEAR", 0x0024),
    ("INT_STATUS", 0x0028),
    ("INT_RAW_STATUS", 0x002C),
    ("PC_TASK_CONTROL", 0x0030),
    ("PC_DMA_BASE_ADDR", 0x0034),
    ("PC_TASK_STATUS", 0x003C),
    ("CNA_S_POINTER", 0x1004),
    ("CORE_S_POINTER", 0x3004),
    ("ENABLE_MASK", 0xF008),
)


def read_core(fd, base):
    mm = mmap.mmap(
        fd,
        CORE_WINDOW_SIZE,
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ,
        offset=base,
    )
    try:
        return [(name, offset, struct.unpack_from("<I", mm, offset)[0])
                for name, offset in REGISTERS]
    finally:
        mm.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/mem")
    parser.add_argument("--label", default="snapshot")
    args = parser.parse_args()

    print(f"MMIO_SNAPSHOT {args.label}")
    try:
        fd = os.open(args.device, os.O_RDONLY | os.O_SYNC)
    except OSError as exc:
        print(f"  MMIO_UNAVAILABLE device={args.device} errno={exc.errno} error={exc.strerror}")
        return 0
    try:
        for core, base in enumerate(CORE_BASES):
            print(f"  core={core} base=0x{base:08x}")
            for name, offset, value in read_core(fd, base):
                print(f"    {name:20s} off=0x{offset:04x} value=0x{value:08x}")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
