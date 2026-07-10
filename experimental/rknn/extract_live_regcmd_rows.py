#!/usr/bin/env python3
import argparse
import re
from collections import Counter


QWORD_RE = re.compile(r"regcmd_qword\[(?P<idx>\d+)\] off=0x(?P<off>[0-9a-f]+) value=0x(?P<value>[0-9a-f]{16})")
TASK_RE = re.compile(
    r"task_like\[(?P<idx>\d+)\] off=0x(?P<off>[0-9a-f]+).*?"
    r"enable_mask=0x(?P<mask>[0-9a-f]+).*?"
    r"regcfg_amount=(?P<amount>\d+).*?"
    r"regcmd_addr=0x(?P<addr>[0-9a-f]+)"
)

TARGET_NAMES = {
    0x0201: "CNA",
    0x0801: "CORE",
    0x1001: "DPU",
    0x0081: "PC",
    0x0101: "PC_REG",
    0x0041: "VERSION",
    0x4001: "SDP",
    0x8001: "PDP",
}

REG_NAMES = {
    (0x1001, 0x4004): "S_POINTER",
    (0x0201, 0x100c): "CNA_CONV_CON1",
    (0x0201, 0x1010): "CNA_CONV_CON2",
    (0x0201, 0x1020): "CNA_DATA_SIZE0",
    (0x0201, 0x1024): "CNA_DATA_SIZE1",
    (0x0201, 0x1030): "CNA_WEIGHT_SIZE0",
    (0x0201, 0x1034): "CNA_WEIGHT_SIZE1",
    (0x0201, 0x1038): "CNA_WEIGHT_SIZE2",
    (0x0201, 0x1040): "CNA_CBUF_CON0",
    (0x0201, 0x1044): "CNA_CBUF_CON1",
    (0x0201, 0x104c): "CNA_CVT_CON0",
    (0x0201, 0x1070): "CNA_FEATURE_DATA_ADDR",
    (0x0201, 0x1078): "CNA_DMA_CON0",
    (0x0201, 0x107c): "CNA_DMA_CON1",
    (0x0201, 0x1080): "CNA_DMA_CON2",
    (0x0201, 0x1084): "CNA_FC_DATA_SIZE0",
    (0x0201, 0x1088): "CNA_FC_DATA_SIZE1",
    (0x0201, 0x1110): "CNA_DCOMP_ADDR0",
    (0x0801, 0x3010): "CORE_MISC_CFG",
    (0x0801, 0x3014): "CORE_DATAOUT_SIZE_0",
    (0x0801, 0x3018): "CORE_DATAOUT_SIZE_1",
    (0x1001, 0x4020): "DST_BASE_ADDR",
    (0x1001, 0x4024): "DST_SURF_STRIDE",
    (0x1001, 0x4030): "DATA_CUBE_WIDTH",
    (0x1001, 0x4034): "DATA_CUBE_HEIGHT",
    (0x1001, 0x403c): "DATA_CUBE_CHANNEL",
    (0x1001, 0x4058): "WDMA_SIZE_0",
    (0x1001, 0x405c): "WDMA_SIZE_1",
    (0x1001, 0x40c0): "SURFACE_ADD",
    (0x0101, 0x0010): "PC_BASE_ADDRESS",
    (0x0101, 0x0014): "PC_REGISTER_AMOUNTS",
    (0x0081, 0x0008): "PC_OPERATION_ENABLE",
}


def decode_qword(qword):
    return (qword >> 48) & 0xffff, qword & 0xffff, (qword >> 16) & 0xffffffff


def parse_log(path):
    tasks = []
    qwords = []
    for line in open(path, errors="ignore"):
        task = TASK_RE.search(line)
        if task:
            tasks.append({key: int(value, 16 if key in {"off", "mask", "addr"} else 10)
                          for key, value in task.groupdict().items()})
            continue
        qword = QWORD_RE.search(line)
        if qword:
            qwords.append(tuple(int(qword.group(key), 16 if key in {"off", "value"} else 10)
                                for key in ("idx", "off", "value")))
    qwords.sort(key=lambda row: row[1])
    return tasks, qwords


def split_rows(qwords):
    rows = []
    current = []
    for item in qwords:
        _idx, _off, value = item
        target, addr, _reg_value = decode_qword(value)
        starts_body = target == 0x1001 and addr == 0x4004
        starts_setup_prelude = target == 0x0201 and addr == 0x1040
        starts_aux = target in (0x4001, 0x8001) and addr in (0x6004, 0x7004)
        if current and (starts_body or starts_setup_prelude or starts_aux):
            rows.append(current)
            current = []
        current.append(item)
    if current:
        rows.append(current)
    return rows


def describe_row(row, index):
    decoded = [decode_qword(value) for _idx, _off, value in row]
    targets = Counter(TARGET_NAMES.get(target, f"0x{target:04x}") for target, _addr, _value in decoded)
    first_off = row[0][1]
    print(f"row={index:02d} off=0x{first_off:x} qwords={len(row)} targets={dict(targets)}")
    for target, addr, value in decoded:
        name = REG_NAMES.get((target, addr))
        if name:
            target_name = TARGET_NAMES.get(target, f"0x{target:04x}")
            print(f"  {target_name}.{name}=0x{value:08x}")


def main():
    parser = argparse.ArgumentParser(description="Decode qword windows from RKNN live-regcmd capture logs")
    parser.add_argument("log")
    parser.add_argument("--row", type=int, action="append", help="only print selected decoded row index")
    parser.add_argument("--tasks", action="store_true", help="also print task-like rows from the capture")
    args = parser.parse_args()
    tasks, qwords = parse_log(args.log)
    if args.tasks:
        for task in tasks:
            print(
                f"task_like[{task['idx']}] off=0x{task['off']:x} mask=0x{task['mask']:x} "
                f"amount={task['amount']} regcmd_addr=0x{task['addr']:x}"
            )
    rows = split_rows(qwords)
    selected = set(args.row or range(len(rows)))
    print(f"log={args.log} qwords={len(qwords)} decoded_rows={len(rows)}")
    for idx, row in enumerate(rows):
        if idx in selected:
            describe_row(row, idx)


if __name__ == "__main__":
    main()
