#!/usr/bin/env python3
"""Decode GEM1 task descriptors and GEM2 body fields for each captured shape.

For each prefix_<slug>_keep1_gem{1,2}/ directory under
/home/orangepi/npu/ops_rknn/dump/, parse:
- GEM1: task descriptors (amounts, masks, offsets, regcmd_addr)
- GEM2: visible body row fields (cbuf0, data_size1, dma_con2, dst_base_addr,
  weight_size0/1/2, conv2, etc.)

Output: writes a per-slug JSON evidence file to
conv_expt/capture_harness/decoded/<slug>.json with the body field evidence.

Stores in worktree (NOT /tmp) so it's safe across reboots.
"""
import json
import re
from pathlib import Path

DUMP_ROOT = Path("/home/orangepi/npu/ops_rknn/dump")
OUT_DIR = Path("/home/orangepi/rk3588/conv_expt/capture_harness/decoded")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def decode_task_descriptors(gem1_text):
    """Parse 'Task N @ offset ...' sections to get amounts, masks, offsets, regcmd_addr."""
    gem1_text = strip_ansi(gem1_text)
    tasks = []
    for m in re.finditer(r"Task (\d+) @ offset .*?regcfg_amount: (\d+) entries.*?regcmd_addr\s*:\s*0x([0-9a-f]+)", gem1_text, re.DOTALL):
        idx = int(m.group(1))
        amount = int(m.group(2))
        regcmd_addr = int(m.group(3), 16)
        tasks.append(dict(idx=idx, amount=amount, regcmd_addr=regcmd_addr))
    # Extract masks
    masks = re.findall(r"enable_mask\s*:\s*0x([0-9a-f]+)", gem1_text)
    masks = [int(m, 16) for m in masks]
    for i, t in enumerate(tasks):
        if i < len(masks):
            t["enable_mask"] = masks[i]
    return tasks


def decode_body_rows(gem2_text, tasks=None):
    """Parse body field EMIT statements to get per-row values."""
    gem2_text = strip_ansi(gem2_text)
    dma_base_match = re.search(r"base selection for GEM 2: dma_base=(\d+)", gem2_text)
    dma_base = int(dma_base_match.group(1)) if dma_base_match else None
    # Match patterns like:
    # [0x...] lsb 0201000000481040 - CNA EMIT(REG_CNA_CBUF_CON0, ...);
    # The packed command word already contains the literal register value.
    emit_re = re.compile(
        r"\[\s*0x([0-9a-fA-F]+)\s*\].*?lsb\s+([0-9a-fA-F]{16}).*?-\s+(\w+)\s+EMIT\((\w+),\s*(.*)\);"
    )
    field_re = re.compile(r"\w+\((\d+|\w+)\)")

    entries = []
    for line in gem2_text.splitlines():
        m = emit_re.search(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        packed = int(m.group(2), 16)
        target = m.group(3)
        regname = m.group(4)
        args = m.group(5)
        values = {}
        target_code = (packed >> 48) & 0xffff
        reg_addr = packed & 0xffff
        values["__literal__"] = (packed >> 16) & 0xffffffff
        for fmatch in field_re.finditer(args):
            v = fmatch.group(1)
            try:
                values[fmatch.group(0)] = int(v)
            except ValueError:
                pass
        entries.append(dict(addr=addr, offset=addr & 0xfffff, target=target, target_code=target_code,
                            reg=regname, reg_addr=reg_addr, args=args, values=values))

    if tasks and dma_base is not None:
        task_offsets = []
        for t in tasks:
            off = t["regcmd_addr"] - dma_base
            if off >= 0:
                task_offsets.append(off)
        if not task_offsets:
            first_cbuf = next((e["offset"] for e in entries if e["reg"] == "REG_CNA_CBUF_CON0"), None)
            if first_cbuf is not None:
                command_base = tasks[0]["regcmd_addr"] - max(0, first_cbuf - 0x30)
                task_offsets = []
                for t in tasks:
                    off = t["regcmd_addr"] - command_base
                    if off >= 0:
                        task_offsets.append(off)
        if task_offsets:
            rows = []
            for idx, start in enumerate(task_offsets):
                end = task_offsets[idx + 1] if idx + 1 < len(task_offsets) else start + tasks[idx]["amount"] * 8
                row = [e for e in entries if start <= e["offset"] < end]
                if row:
                    rows.append(row)
            return rows

    rows = []
    current = []
    for entry in entries:
        if entry["reg"] == "REG_CNA_CBUF_CON0" and current and any(
            e["reg"] in {"REG_DPU_DST_BASE_ADDR", "REG_PC_BASE_ADDRESS"} for e in current
        ):
            rows.append(current)
            current = []
        current.append(entry)
    if current:
        rows.append(current)
    return rows


def summarize_body_rows(rows):
    """Group rows by their CNA CBUF_CON0, DATA_SIZE1, DMA_CON2 values."""
    if not rows:
        return {}
    body_rows = [r for r in rows if any(e["reg"] == "REG_CNA_CBUF_CON0" for e in r)]
    out = []
    for r in body_rows:
        e_map = {e["reg"]: e for e in r}
        out.append(dict(
            cbuf0=e_map.get("REG_CNA_CBUF_CON0", {}).get("values", {}).get("__literal__"),
            data_size0=e_map.get("REG_CNA_DATA_SIZE0", {}).get("values", {}).get("__literal__"),
            data_size1=e_map.get("REG_CNA_DATA_SIZE1", {}).get("values", {}).get("__literal__"),
            data_size3=e_map.get("REG_CNA_DATA_SIZE3", {}).get("values", {}).get("__literal__"),
            fc_data_size0=e_map.get("REG_CNA_FC_DATA_SIZE0", {}).get("values", {}).get("__literal__"),
            fc_data_size1=e_map.get("REG_CNA_FC_DATA_SIZE1", {}).get("values", {}).get("__literal__"),
            dma_con2=e_map.get("REG_CNA_DMA_CON2", {}).get("values", {}).get("__literal__"),
            weight_size0=e_map.get("REG_CNA_WEIGHT_SIZE0", {}).get("values", {}).get("__literal__"),
            weight_size1=e_map.get("REG_CNA_WEIGHT_SIZE1", {}).get("values", {}).get("__literal__"),
            weight_size2=e_map.get("REG_CNA_WEIGHT_SIZE2", {}).get("values", {}).get("__literal__"),
            conv_con1=e_map.get("REG_CNA_CONV_CON1", {}).get("values", {}).get("__literal__"),
            conv_con2=e_map.get("REG_CNA_CONV_CON2", {}).get("values", {}).get("__literal__"),
            core_dataout_size0=e_map.get("REG_CORE_DATAOUT_SIZE_0", {}).get("values", {}).get("__literal__"),
            core_dataout_size1=e_map.get("REG_CORE_DATAOUT_SIZE_1", {}).get("values", {}).get("__literal__"),
            dst_base=e_map.get("REG_DPU_DST_BASE_ADDR", {}).get("values", {}).get("__literal__"),
            feature_data_addr=e_map.get("REG_CNA_FEATURE_DATA_ADDR", {}).get("values", {}).get("__literal__"),
            dcomp_addr=e_map.get("REG_CNA_DCOMP_ADDR0", {}).get("values", {}).get("__literal__"),
            dst_surf_stride=e_map.get("REG_DPU_DST_SURF_STRIDE", {}).get("values", {}).get("__literal__"),
            data_cube_height=e_map.get("REG_DPU_DATA_CUBE_HEIGHT", {}).get("values", {}).get("__literal__"),
            data_cube_channel=e_map.get("REG_DPU_DATA_CUBE_CHANNEL", {}).get("values", {}).get("__literal__"),
            wdma_size0=e_map.get("REG_DPU_WDMA_SIZE_0", {}).get("values", {}).get("__literal__"),
            wdma_size1=e_map.get("REG_DPU_WDMA_SIZE_1", {}).get("values", {}).get("__literal__"),
            surface_add=e_map.get("REG_DPU_SURFACE_ADD", {}).get("values", {}).get("__literal__"),
        ))
    return out


def main():
    n_decoded = 0
    for d in sorted(DUMP_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("prefix_"):
            continue
        m = re.match(r"prefix_(.+?)_keep\d+_gem(\d+)$", d.name)
        if not m:
            continue
        slug, gem_n = m.group(1), int(m.group(2))
        if gem_n != 2:
            continue  # only decode body (GEM2) for now
        gem2 = d / f"dump_gem{gem_n}.txt"
        if not gem2.exists() or gem2.stat().st_size < 100:
            continue
        # also need GEM1
        gem1 = d.parent / f"prefix_{slug}_keep1_gem1" / "dump_gem1.txt"
        if not gem1.exists():
            continue
        gem1_text = gem1.read_text(errors="ignore")
        gem2_text = gem2.read_text(errors="ignore")
        tasks = decode_task_descriptors(gem1_text)
        rows = decode_body_rows(gem2_text, tasks)
        body_summary = summarize_body_rows(rows)
        out = dict(
            slug=slug,
            dump_dir=str(d),
            task_count=len(tasks),
            amounts=tuple(t["amount"] for t in tasks),
            masks=tuple(t.get("enable_mask", 0) for t in tasks),
            regcmd_addrs=tuple(hex(t["regcmd_addr"]) for t in tasks),
            body_row_count=len(body_summary),
            body_rows=body_summary,
        )
        out_path = OUT_DIR / f"{slug}.json"
        out_path.write_text(json.dumps(out, indent=2))
        n_decoded += 1
    print(f"Decoded {n_decoded} shapes -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
