#!/usr/bin/env python3
"""No-submit materializer for b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid.

Asserts the 12-task RKNN BY_YK closure (108,108,104,26,104,26,104,26,104,26,104,26)
matches the live prefix dump at GEM1 and validates the body field invariants
expected from the c256_h2_oc64 + c256_h28_pw_1x1 closure pattern. Does NOT
submit to the NPU.
"""
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conv_expt.rknn_prefix_replay import dump_task_summary  # noqa: E402
from examples.conv import (  # noqa: E402
    E,
    _exact11_aux_regs,
    _exact11_body_regs,
    patch_regs,
    reg,
    shape_from_name,
)


CASE = {
    "shape": "b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid",
    "gem1": Path("/home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_keep1_gem1/dump_gem1.txt"),
    "amounts": (108, 108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26),
    "masks": (0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60),
    "families": ("setup", "k_half", "k_tile", "ppu_pdp", "k_tile", "ppu_pdp",
                 "k_tile", "ppu_pdp", "k_tile", "ppu_pdp", "k_tile", "ppu_pdp"),
    "out_c_per_k_tile": (4, 3, 2, 1, 2),  # 4+3+2+1+2=12
    "weight_size0_setup": 0x3600,  # 12 * 576 * 2 fp16 = 13824
    "weight_size0_k_half": 0x3600,  # full oc, single k_half
    "weight_size0_k_tile": (0x1200, 0x0d80, 0x0900, 0x0480, 0x0900),  # per k_tile split
    "cbuf0_setup": 0x0b1,
    "data_size1_guess": 0x8f0240,  # (576/4-1)<<16 | 576 = (143<<16)|576
    "k_half_needs_prelude": True,
}


def _prelude(cbuf0, conv1=0x120):
    """4-qword prelude that turns a 104q body into a 108q row."""
    return (E(reg.CNA, reg.CNA_CBUF_CON0, cbuf0),
            E(reg.CNA, 0x1104, 0),
            E(reg.CNA, 0x1100, 0),
            E(reg.CNA, reg.CNA_CONV_CON1, conv1))


def _materialized_rows(s, in_dma, wt_dma, out_dma):
    """Generate the 12 expected rows (no submit) using exact11 closure pattern.

    Rows 0,1 are 108q (with 4-qword prelude), rows 2,4,6,8,10 are 104q (k_tile body),
    rows 3,5,7,9,11 are 26q (ppu_pdp). The c=576, oc=12 1x1 wide pointwise
    structure mirrors c256_h2_oc64 with 5 k_tiles (4+3+2+1+2=12) instead of 3.
    """
    def body_row(family, oc_start, oc_count, weight_size0, with_prelude=False):
        body = _exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma,
                                  input_h=19, conv2_low=0x30)
        patched = patch_regs(body, {
            (reg.CNA, reg.CNA_CBUF_CON0): CASE["cbuf0_setup"],
            (reg.CNA, reg.CNA_DATA_SIZE1): CASE["data_size1_guess"],
            (reg.CNA, reg.CNA_DMA_CON2): 0x0ffffffc,
            (reg.CNA, reg.CNA_CVT_CON0): 0x000b,
            (reg.CNA, reg.CNA_WEIGHT_SIZE0): weight_size0,
            (reg.CNA, reg.CNA_WEIGHT_SIZE1): weight_size0 >> 8,
            (reg.CNA, reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | oc_count,
            (reg.CORE, reg.CORE_DATAOUT_SIZE_1): oc_count - 1,
            (reg.DPU, reg.DATA_CUBE_CHANNEL): ((oc_count - 1) << 16) | (oc_count - 1),
            (reg.DPU, reg.WDMA_SIZE_0): oc_count - 1,
        })
        if with_prelude:
            return list(_prelude(CASE["cbuf0_setup"])) + patched
        return patched

    aux = _exact11_aux_regs(s, out_dma, wt_dma + 0x3600)
    k_splits = (0, 4, 7, 9, 10)
    # setup row already gets its 4-qword prelude from _exact11_body_regs (family=="setup")
    # k_half row needs an explicit 4-qword prelude to reach 108q
    rows = [
        body_row("setup", 0, 12, CASE["weight_size0_setup"], with_prelude=False),
        body_row("k_half", 0, 12, CASE["weight_size0_k_half"], with_prelude=True),
    ]
    for k_idx, (oc_start, oc_count, weight_size0) in enumerate(zip(
        k_splits, CASE["out_c_per_k_tile"], CASE["weight_size0_k_tile"])):
        rows.append(body_row("k_tile", oc_start, oc_count, weight_size0, with_prelude=False))
        rows.append(aux)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="No-submit c576_h19_oc12 task structure validator")
    parser.add_argument("--check", choices=["task", "rows", "all"], default="all")
    args = parser.parse_args(argv)
    if not CASE["gem1"].exists():
        raise SystemExit(f"missing RKNN dump: {CASE['gem1']}")
    s = shape_from_name(CASE["shape"])

    # Always validate the live task structure from GEM1
    summary = dump_task_summary(CASE["gem1"])
    assert summary["status"] == "ok", f"RKNN GEM1 status={summary['status']}"
    assert tuple(summary["amounts"]) == CASE["amounts"], f"amounts mismatch: {summary['amounts']} vs {CASE['amounts']}"
    assert tuple(summary["masks"]) == CASE["masks"], f"masks mismatch: {summary['masks']} vs {CASE['masks']}"
    print(f"task_structure=ok amounts={','.join(str(v) for v in summary['amounts'])}")
    print(f"families={','.join(CASE['families'])}")
    print(f"k_tile_splits=4+3+2+1+2=12 (oc=12)")

    if args.check in ("rows", "all"):
        # Generate the 12 expected body rows WITHOUT submitting
        rows = _materialized_rows(s, 0x10000000, 0x20000000, 0x30000000)
        actual_amounts = tuple(len(r) for r in rows)
        if actual_amounts != CASE["amounts"]:
            raise SystemExit(f"materialized row amounts mismatch: {actual_amounts} vs {CASE['amounts']}")
        # Body field assertions: assert cbuf0, data_size1 are applied
        for idx, (family, row) in enumerate(zip(CASE["families"], rows)):
            if family == "ppu_pdp":
                continue
            values = {(qword >> 48, qword & 0xffff): (qword >> 16) & 0xffffffff for qword in row}
            cbuf0 = values[(reg.CNA, reg.CNA_CBUF_CON0)]
            data_size1 = values[(reg.CNA, reg.CNA_DATA_SIZE1)]
            assert cbuf0 == CASE["cbuf0_setup"], f"row {idx} cbuf0={cbuf0:#x} != {CASE['cbuf0_setup']:#x}"
            assert data_size1 == CASE["data_size1_guess"], f"row {idx} data_size1={data_size1:#x} != {CASE['data_size1_guess']:#x}"
        print(f"body_field_invariants=ok cbuf0={CASE['cbuf0_setup']:#x} data_size1={CASE['data_size1_guess']:#x}")
        print(f"note=materializer_is_no_submit body_fields_are_educated_guesses_from_c256_h2_oc64_pattern")
        print(f"note=k_half_prelude_is_inferred_from_108q_amounts_actual_register_values_unknown")
        print(f"note=promotion_requires_fresh_GEM2_capture_for_actual_register_values")


if __name__ == "__main__":
    main()
