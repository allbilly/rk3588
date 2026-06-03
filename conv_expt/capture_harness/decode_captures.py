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


def decode_task_descriptors(gem1_text):
    """Parse 'Task N @ offset ...' sections to get amounts, masks, offsets, regcmd_addr."""
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


def decode_body_rows(gem2_text):
    """Parse body field EMIT statements to get per-row values."""
    rows = []
    # Match patterns like: 'CNA    EMIT(REG_CNA_CBUF_CON0, CNA_CBUF_CON0_WEIGHT_BANK(4) | CNA_CBUF_CON0_DATA_BANK(8));'
    emit_re = re.compile(r"(\w+)\s+EMIT\((\w+),\s*([^)]+)\);")
    field_re = re.compile(r"\w+\((\d+|\w+)\)")

    # Group by absolute file offset; each "row" is a sequence of EMITs at increasing addresses
    last_addr = -1
    current = []
    for m in re.finditer(r"\[\s*0x([0-9a-f]+)\s*\][^EMIT]*(\w+)\s+EMIT\((\w+),\s*([^)]+)\);", gem2_text):
        addr = int(m.group(1), 16)
        target = m.group(2)
        regname = m.group(3)
        args = m.group(4)
        if last_addr >= 0 and addr < last_addr:
            # new row
            if current:
                rows.append(current)
            current = []
        # Extract field values
        values = {}
        for fmatch in field_re.finditer(args):
            v = fmatch.group(1)
            try:
                values[fmatch.group(0)] = int(v)
            except ValueError:
                pass
        # Also try to get literal value
        lit = re.search(r"0x[0-9a-f]+|\d+", args)
        if lit:
            try:
                values["__literal__"] = int(lit.group(0), 16) if lit.group(0).startswith("0x") else int(lit.group(0))
            except ValueError:
                pass
        current.append(dict(addr=addr, target=target, reg=regname, args=args, values=values))
        last_addr = addr
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
            data_size1=e_map.get("REG_CNA_DATA_SIZE1", {}).get("values", {}).get("__literal__"),
            dma_con2=e_map.get("REG_CNA_DMA_CON2", {}).get("values", {}).get("__literal__"),
            weight_size0=e_map.get("REG_CNA_WEIGHT_SIZE0", {}).get("values", {}).get("__literal__"),
            weight_size1=e_map.get("REG_CNA_WEIGHT_SIZE1", {}).get("values", {}).get("__literal__"),
            weight_size2=e_map.get("REG_CNA_WEIGHT_SIZE2", {}).get("values", {}).get("__literal__"),
            conv_con1=e_map.get("REG_CNA_CONV_CON1", {}).get("values", {}).get("__literal__"),
            conv_con2=e_map.get("REG_CNA_CONV_CON2", {}).get("values", {}).get("__literal__"),
            dst_base=e_map.get("REG_DPU_DST_BASE_ADDR", {}).get("values", {}).get("__literal__"),
            feature_data_addr=e_map.get("REG_CNA_FEATURE_DATA_ADDR", {}).get("values", {}).get("__literal__"),
            dcomp_addr=e_map.get("REG_CNA_DCOMP_ADDR0", {}).get("values", {}).get("__literal__"),
            dst_surf_stride=e_map.get("REG_DPU_DST_SURF_STRIDE", {}).get("values", {}).get("__literal__"),
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
        rows = decode_body_rows(gem2_text)
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
