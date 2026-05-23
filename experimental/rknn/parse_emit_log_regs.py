#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


EMIT_RE = re.compile(
    r"^\[(0x[0-9a-fA-F]+)\]\s+lsb\s+([0-9a-fA-F]{16}).*?EMIT\(([^,]+),\s*(.*?)\);"
)


REG_NAMES = {
    (0x0201, 0x1010): "CONV_CON2",
    (0x0201, 0x1020): "DATA_SIZE0",
    (0x0201, 0x1024): "DATA_SIZE1",
    (0x0201, 0x102c): "DATA_SIZE3",
    (0x0201, 0x1030): "WEIGHT_SIZE0",
    (0x0201, 0x1034): "WEIGHT_SIZE1",
    (0x0201, 0x1038): "WEIGHT_SIZE2",
    (0x0201, 0x1040): "CBUF_CON0",
    (0x0201, 0x1044): "CBUF_CON1",
    (0x0201, 0x1070): "FEATURE_ADDR",
    (0x0201, 0x1084): "FC_SIZE0",
    (0x0201, 0x1110): "DCOMP_ADDR0",
    (0x0801, 0x3014): "CORE_SIZE0",
    (0x0801, 0x3018): "CORE_SIZE1",
    (0x1001, 0x4020): "DST_BASE",
    (0x1001, 0x4024): "DST_SURF_STRIDE",
    (0x1001, 0x403c): "DPU_DST_C",
    (0x1001, 0x405c): "DPU_SIZE",
    (0x1001, 0x40c0): "SURFACE_ADD",
    (0x0101, 0x0010): "PC_BASE_ADDRESS",
    (0x0101, 0x0014): "PC_REGISTER_AMOUNTS",
}


FIELDS = [
    "run", "CONV_CON2", "CBUF_CON0", "CBUF_CON1", "DATA_SIZE0", "DATA_SIZE1",
    "DATA_SIZE3", "WEIGHT_SIZE0", "WEIGHT_SIZE1", "WEIGHT_SIZE2", "FEATURE_ADDR",
    "FC_SIZE0", "DCOMP_ADDR0", "CORE_SIZE0", "CORE_SIZE1", "DST_BASE",
    "DST_SURF_STRIDE", "DPU_DST_C", "DPU_SIZE", "SURFACE_ADD",
    "PC_BASE_ADDRESS", "PC_REGISTER_AMOUNTS",
]


def decode_qword(hex_text):
    qword = int(hex_text, 16)
    target = (qword >> 48) & 0xffff
    value = (qword >> 16) & 0xffffffff
    reg = qword & 0xffff
    return target, reg, value


def iter_commands(path):
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        match = EMIT_RE.search(line)
        if not match:
            continue
        log_addr = int(match.group(1), 16)
        qword = int(match.group(2), 16)
        emit_name = match.group(3)
        expr = match.group(4)
        target, reg, value = decode_qword(match.group(2))
        name = REG_NAMES.get((target, reg))
        yield {
            "line": line_no,
            "log_addr": log_addr,
            "qword": qword,
            "target": target,
            "reg": reg,
            "value": value,
            "emit_name": emit_name,
            "expr": expr,
            "name": name,
        }


def rows(path):
    out = []
    current = None
    for command in iter_commands(path):
        name = command["name"]
        if not name:
            continue
        value = command["value"]
        if name == "CONV_CON2":
            if current is not None:
                out.append(current)
            current = {"run": len(out) + 1}
        if current is not None:
            current[name] = value
    if current is not None:
        out.append(current)
    return out


def fmt(value):
    if value is None:
        return ""
    sign = "-" if value < 0 else ""
    return f"{sign}0x{abs(value):x}"


def print_slots(path):
    print("slot,line,log_addr,target,reg,name,value,emit_name,expr")
    slot = 0
    for command in iter_commands(path):
        print(",".join([
            str(slot),
            str(command["line"]),
            fmt(command["log_addr"]),
            fmt(command["target"]),
            fmt(command["reg"]),
            command["name"] or "",
            fmt(command["value"]),
            command["emit_name"],
            command["expr"],
        ]))
        slot += 1


def print_core_correlation(path):
    observed_core = {0: 0, 64: 1, 32: 2}
    print("run,conv_reserved,pc_core,pc_amount,observed_core,match")
    for row in rows(path):
        conv = row.get("CONV_CON2")
        pc = row.get("PC_REGISTER_AMOUNTS")
        if conv is None or pc is None:
            continue
        conv_reserved = conv >> 24
        pc_core = (pc >> 16) & 0xffff
        pc_amount = pc & 0xffff
        mapped_core = observed_core.get(conv_reserved)
        match = "unproven" if pc_amount == 0 else ("yes" if mapped_core == pc_core else "no")
        print(",".join([
            str(row["run"]),
            str(conv_reserved),
            str(pc_core),
            str(pc_amount),
            "" if mapped_core is None else str(mapped_core),
            match,
        ]))


def _bank_bits(cbuf_con0):
    if cbuf_con0 is None:
        return "", "", ""
    return str(1 if cbuf_con0 & (1 << 13) else 0), str((cbuf_con0 >> 4) & 0xf), str(cbuf_con0 & 0xf)


def print_address_report(path):
    print("run,feature_addr,feature_delta,feature_reused,dcomp_addr,dcomp_reused,dst_base,dst_delta,pc_base,pc_amount,pc_core,weight_reuse,weight_bank,data_bank")
    prev_feature = None
    prev_dcomp = None
    prev_dst = None
    seen_feature = set()
    seen_dcomp = set()
    for row in rows(path):
        feature = row.get("FEATURE_ADDR")
        dcomp = row.get("DCOMP_ADDR0")
        dst = row.get("DST_BASE")
        pc = row.get("PC_REGISTER_AMOUNTS")
        pc_amount = "" if pc is None else str(pc & 0xffff)
        pc_core = "" if pc is None else str((pc >> 16) & 0xffff)
        weight_reuse, weight_bank, data_bank = _bank_bits(row.get("CBUF_CON0"))
        feature_delta = "" if prev_feature is None or feature is None else fmt(feature - prev_feature)
        dst_delta = "" if prev_dst is None or dst is None else fmt(dst - prev_dst)
        feature_reused = "" if feature is None else ("yes" if feature in seen_feature else "no")
        dcomp_reused = "" if dcomp is None else ("yes" if dcomp in seen_dcomp else "no")
        print(",".join([
            str(row["run"]),
            fmt(feature),
            feature_delta,
            feature_reused,
            fmt(dcomp),
            dcomp_reused,
            fmt(dst),
            dst_delta,
            fmt(row.get("PC_BASE_ADDRESS")),
            pc_amount,
            pc_core,
            weight_reuse,
            weight_bank,
            data_bank,
        ]))
        if feature is not None:
            seen_feature.add(feature)
            prev_feature = feature
        if dcomp is not None:
            seen_dcomp.add(dcomp)
            prev_dcomp = dcomp
        if dst is not None:
            prev_dst = dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("emit_log", type=Path)
    parser.add_argument("--slots", action="store_true", help="dump every EMIT command with decoded slot metadata")
    parser.add_argument("--core-correlation", action="store_true", help="compare CONV_CON2 high bits with PC_REGISTER_AMOUNTS core bits")
    parser.add_argument("--address-report", action="store_true", help="summarize run-level DMA address reuse and CBUF bank fields")
    args = parser.parse_args()

    if args.slots:
        print_slots(args.emit_log)
        return
    if args.core_correlation:
        print_core_correlation(args.emit_log)
        return
    if args.address_report:
        print_address_report(args.emit_log)
        return

    print(",".join(FIELDS))
    for row in rows(args.emit_log):
        print(",".join(str(row["run"]) if field == "run" else fmt(row.get(field)) for field in FIELDS))


if __name__ == "__main__":
    main()
