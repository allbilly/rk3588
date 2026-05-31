"""
conv_cpu.py — Pure-numpy tiled conv using the same tiling strategy as conv.py.
Proves the reversed RKNN bank-pressure-based Y/K tiling is mathematically correct.

No NPU hardware needed. Uses the same _plan_conv_tiles / _compute_k_step /
_compute_y_step functions, then computes each tile on CPU with numpy and
assembles the result.
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

try:  # noqa: E402
    from conv_expt.conv_tile_planner import (
        OUTPUT_MEM_BYTES,
        FP16_BYTES,
        RK3588_CBUF_BANKS,
        UNPACK_C2,
        _EVIDENCE_MIX_H40,
        _POINTWISE_Y_TILE_HARDCODED,
        _align_up,
        _ceil_div,
        _conv_output_bytes,
        _conv_output_count,
        _conv_params,
        _depthwise_tile_h,
        _descriptor_families,
        _descriptor_rows_for_shape,
        _is_depthwise,
        _needs_pointwise_oc_tile_schedule,
        _old_strategy_name,
        _plan_conv_tiles,
        _split_name,
        _with_cbuf_profile,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from conv_tile_planner import (
        OUTPUT_MEM_BYTES,
        FP16_BYTES,
        RK3588_CBUF_BANKS,
        UNPACK_C2,
        _EVIDENCE_MIX_H40,
        _POINTWISE_Y_TILE_HARDCODED,
        _align_up,
        _ceil_div,
        _conv_output_bytes,
        _conv_output_count,
        _conv_params,
        _depthwise_tile_h,
        _descriptor_families,
        _descriptor_rows_for_shape,
        _is_depthwise,
        _needs_pointwise_oc_tile_schedule,
        _old_strategy_name,
        _plan_conv_tiles,
        _split_name,
        _with_cbuf_profile,
    )


def _descriptor_unresolved_fields(split_method, groups, is_depthwise):
    fields = ["grain_bits", "cbuf0", "mc_treat_by_y_tile", "mc_treat_by_k_tile",
              "mc_treat_by_1c_y_tile", "mc_treat_by_1c_k_tile"]
    if _split_name(split_method) == "BY_YK":
        fields.append("family_k_window_assignment")
    if groups > 1 and not is_depthwise:
        fields.append("group_lowering_descriptor_contract")
    return fields


def _planner_report_row(s):
    p, split_method, y_boundaries, k_boundaries, desc_rows = _descriptor_rows_for_shape(s)
    families = sorted({row["family"] for row in desc_rows}, key=_descriptor_families(split_method).index)
    unresolved = _descriptor_unresolved_fields(split_method, s["groups"], p["is_depthwise"])
    return {
        "name": s["name"],
        "old_strategy": _old_strategy_name(s),
        "split_method": _split_name(split_method),
        "y_boundaries": y_boundaries,
        "k_boundaries": k_boundaries,
        "descriptor_count": len(desc_rows),
        "descriptor_families": families,
        "unresolved_fields": unresolved,
    }


def _planner_report_rows():
    rows = []
    for s in SHAPES:
        rows.append(_planner_report_row(s))
    return rows


def _all_named_shapes(include_evidence=False):
    shapes = list(SHAPES)
    if include_evidence:
        shapes.append(_EVIDENCE_MIX_H40)
    return shapes


def _format_cell(value):
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return str(value)


def _print_table(rows, columns):
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(_format_cell(row[col])))
    print("  ".join(f"{col:<{widths[col]}}" for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(f"{_format_cell(row[col]):<{widths[col]}}" for col in columns))


def print_planner_report():
    columns = ["name", "old_strategy", "split_method", "y_boundaries", "k_boundaries",
               "descriptor_count", "descriptor_families", "unresolved_fields"]
    _print_table(_planner_report_rows(), columns)


def print_descriptor_dump(shape_name=None):
    shape_pool = SHAPES if shape_name is None else _all_named_shapes(include_evidence=True)
    shapes = [s for s in shape_pool if shape_name is None or s["name"] == shape_name]
    if shape_name is not None and not shapes:
        raise SystemExit(f"unknown shape: {shape_name}")
    rows = []
    for s in shapes:
        rows.extend(_descriptor_rows_for_shape(s)[4])
    columns = ["name", "family", "semantic_status", "rknn_executable_equivalent", "family_bits", "grain_bits", "y_start", "input_h",
               "output_h", "output_w", "k_start", "oc_count", "feature_off", "weight_off",
               "output_off", "input_bank_num", "weight_bank_num", "cbuf0", "data_reuse", "weight_reuse", "unresolved_reason"]
    _print_table(rows, columns)


def print_cross_tab():
    buckets = defaultdict(lambda: {"count": 0, "old": Counter()})
    for row in _planner_report_rows():
        key = (row["split_method"], tuple(row["descriptor_families"]))
        buckets[key]["count"] += 1
        buckets[key]["old"][row["old_strategy"]] += 1
    rows = []
    for (split_method, families), info in sorted(buckets.items()):
        old = ", ".join(f"{name}:{count}" for name, count in sorted(info["old"].items()))
        rows.append({
            "split_method": split_method,
            "descriptor_families": list(families),
            "count": info["count"],
            "old_branches_covered": old,
        })
    _print_table(rows, ["split_method", "descriptor_families", "count", "old_branches_covered"])


def _cbuf_compare_rows():
    rows = []
    for s in SHAPES:
        rk = _with_cbuf_profile("rk3588", _planner_report_row, s)
        nv = _with_cbuf_profile("nvdla_full", _planner_report_row, s)
        changed_fields = []
        for field in ("split_method", "y_boundaries", "k_boundaries", "descriptor_count", "descriptor_families"):
            if rk[field] != nv[field]:
                changed_fields.append(field)
        rows.append({
            "name": s["name"],
            "old_strategy": rk["old_strategy"],
            "rk_split": rk["split_method"],
            "nv_split": nv["split_method"],
            "rk_y_boundaries": rk["y_boundaries"],
            "nv_y_boundaries": nv["y_boundaries"],
            "rk_k_boundaries": rk["k_boundaries"],
            "nv_k_boundaries": nv["k_boundaries"],
            "rk_descriptor_count": rk["descriptor_count"],
            "nv_descriptor_count": nv["descriptor_count"],
            "rk_descriptor_families": rk["descriptor_families"],
            "nv_descriptor_families": nv["descriptor_families"],
            "changed_fields": changed_fields,
        })
    return rows


def print_cbuf_compare(all_rows=False):
    rows = _cbuf_compare_rows()
    changed = [row for row in rows if row["changed_fields"]]
    same = len(rows) - len(changed)
    print(f"profiles: rk3588=12x32KiB, nvdla_full=16x32KiB planner budget")
    print(f"same={same} changed={len(changed)} total={len(rows)}")
    print()
    split_counts = Counter((row["rk_split"], row["nv_split"]) for row in rows)
    summary_rows = []
    for (rk_split, nv_split), count in sorted(split_counts.items()):
        summary_rows.append({"rk_split": rk_split, "nv_split": nv_split, "count": count})
    _print_table(summary_rows, ["rk_split", "nv_split", "count"])
    print()
    detail_rows = rows if all_rows else changed
    columns = ["name", "old_strategy", "rk_split", "nv_split", "rk_y_boundaries", "nv_y_boundaries",
               "rk_k_boundaries", "nv_k_boundaries", "rk_descriptor_count", "nv_descriptor_count",
               "rk_descriptor_families", "nv_descriptor_families", "changed_fields"]
    _print_table(detail_rows, columns)


def _shape_by_name(name):
    for s in _all_named_shapes(include_evidence=True):
        if s["name"] == name:
            return s
    raise KeyError(name)


def _unique_in_order(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _add_evidence_row(rows, target, check, ok, pass_detail, fail_detail, fail_status="FAIL"):
    rows.append({
        "target": target,
        "check": check,
        "status": "PASS" if ok else fail_status,
        "detail": pass_detail if ok else fail_detail,
    })


def _offsets_are_additive(rows, p):
    feature_by_y = {}
    weight_by_k = {}
    for row in rows:
        y_start = row["y_start"]
        k_start = row["k_start"]
        expected_output = (k_start * p["out_h"] * p["out_w"] + y_start * p["out_w"]) * FP16_BYTES
        if row["output_off"] != expected_output:
            return False
        if y_start in feature_by_y and feature_by_y[y_start] != row["feature_off"]:
            return False
        if k_start in weight_by_k and weight_by_k[k_start] != row["weight_off"]:
            return False
        feature_by_y[y_start] = row["feature_off"]
        weight_by_k[k_start] = row["weight_off"]
    return True


def _family_oc_counts(rows, family):
    counts = []
    for row in rows:
        if row["family"] == family and row["y_start"] == 0:
            counts.append(row["oc_count"])
    return counts


def _evidence_check_rows():
    rows = []

    mixed = _EVIDENCE_MIX_H40
    p, split_method, y_boundaries, k_boundaries, desc = _descriptor_rows_for_shape(mixed)
    families = _unique_in_order([row["family"] for row in desc])
    target = "mixed 160->320 3x3 h40"
    _add_evidence_row(rows, target, "high-level split", _split_name(split_method) == "BY_YK",
                      f"split=BY_YK y={y_boundaries} k={k_boundaries}",
                      f"split={_split_name(split_method)} y={y_boundaries} k={k_boundaries}")
    _add_evidence_row(rows, target, "independent Y/K windows", len(y_boundaries) > 2 and len(k_boundaries) > 2,
                      f"y_windows={len(y_boundaries) - 1} k_windows={len(k_boundaries) - 1}",
                      f"y_windows={len(y_boundaries) - 1} k_windows={len(k_boundaries) - 1}")
    _add_evidence_row(rows, target, "family order", families == ["setup", "k_half", "k_tile"],
                      f"families={families}", f"families={families}")
    _add_evidence_row(rows, target, "additive offsets", _offsets_are_additive(desc, p),
                      "feature_off by Y, weight_off by K, output_off additive",
                      "offset dependency check failed")
    _add_evidence_row(rows, target, "exact RKNN descriptor count", len(desc) == 12,
                      "descriptor_count=12", f"descriptor_count={len(desc)} expected setup x2, k_half x4, k_tile x6", "GAP")
    _add_evidence_row(rows, target, "exact RKNN k_tile OC windows", _family_oc_counts(desc, "k_tile") == [112, 112, 96],
                      "k_tile OC windows match 112;112;96",
                      f"k_tile OC windows={_family_oc_counts(desc, 'k_tile')} expected 112;112;96", "GAP")

    h14 = _shape_by_name("b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid")
    _p, _split_method, _y_boundaries, _k_boundaries, h14_desc = _descriptor_rows_for_shape(h14)
    h14_target = "spatial 160->320 3x3 h14"
    _add_evidence_row(rows, h14_target, "spatial k_tile rows present",
                      any(row["family"] == "k_tile" for row in h14_desc),
                      f"k_tile_rows={sum(1 for row in h14_desc if row['family'] == 'k_tile')}",
                      "no k_tile rows")
    _add_evidence_row(rows, h14_target, "cbuf0/grain kept explicit",
                      all(row["grain_bits"] is None and row["cbuf0"] is None for row in h14_desc),
                      "grain_bits=unknown and cbuf0=unknown", "grain_bits or cbuf0 hidden behind a default")
    _add_evidence_row(rows, h14_target, "exact RKNN k_tile OC windows", _family_oc_counts(h14_desc, "k_tile") == [112, 112, 96],
                      "k_tile OC windows match 112;112;96",
                      f"k_tile OC windows={_family_oc_counts(h14_desc, 'k_tile')} expected 112;112;96", "GAP")

    pointwise = _shape_by_name("conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1")
    _p, pointwise_split, pointwise_y, _pointwise_k, pointwise_desc = _descriptor_rows_for_shape(pointwise)
    pointwise_target = "pointwise exported Y tile"
    _add_evidence_row(rows, pointwise_target, "y_tile rows present", _split_name(pointwise_split) == "BY_Y" and pointwise_y == [0, 25, 28],
                      f"split=BY_Y y={pointwise_y}", f"split={_split_name(pointwise_split)} y={pointwise_y}")
    _add_evidence_row(rows, pointwise_target, "grain_bits explicit unknown",
                      all(row["grain_bits"] is None for row in pointwise_desc),
                      "all descriptor grain_bits=unknown", "grain_bits hidden behind a default")
    _add_evidence_row(rows, pointwise_target, "cbuf0 separate from grain_bits",
                      all(row["cbuf0"] is None for row in pointwise_desc),
                      "all descriptor cbuf0=unknown", "cbuf0 hidden behind grain/default")

    hardcoded = []
    for name in sorted(_POINTWISE_Y_TILE_HARDCODED):
        _p, hard_split, hard_y, hard_k, hard_desc = _descriptor_rows_for_shape(_shape_by_name(name))
        hardcoded.append(f"{name}:{_split_name(hard_split)} y={hard_y} k={hard_k} desc={len(hard_desc)}")
    _add_evidence_row(rows, "pointwise_y_tile_hardcoded", "rare branch not explained as Y/K yet",
                      all(":NONE " not in item for item in hardcoded),
                      "; ".join(hardcoded), "; ".join(hardcoded), "GAP")

    all_desc = []
    for s in SHAPES:
        all_desc.extend(_descriptor_rows_for_shape(s)[4])
    _add_evidence_row(rows, "all descriptor rows", "unresolved fields visible",
                      all(row["grain_bits"] is None and row["cbuf0"] is None for row in all_desc),
                      f"checked {len(all_desc)} descriptor rows", "some unresolved fields are hidden")
    return rows


def print_evidence_check():
    rows = _evidence_check_rows()
    _print_table(rows, ["target", "check", "status", "detail"])
    counts = Counter(row["status"] for row in rows)
    print()
    _print_table([{"status": status, "count": count} for status, count in sorted(counts.items())],
                 ["status", "count"])


def _format_windows(windows):
    return ";".join(f"{start}:{count}" for start, count in windows)


def _family_windows(rows, family):
    y_windows = _unique_in_order((row["y_start"], row["output_h"]) for row in rows if row["family"] == family)
    k_windows = _unique_in_order((row["k_start"], row["oc_count"]) for row in rows if row["family"] == family)
    return y_windows, k_windows


def _family_pressure_from_rows(rows, family):
    family_rows = [row for row in rows if row["family"] == family]
    if not family_rows:
        return None
    max_input = max(row["input_bank_num"] for row in family_rows)
    max_weight = max(row["weight_bank_num"] for row in family_rows)
    max_total = max(row["input_bank_num"] + row["weight_bank_num"] for row in family_rows)
    return max_input, max_weight, max_total


def _target_pressure(p, s, y_windows, k_windows):
    max_input = 0
    max_weight = 0
    max_total = 0
    stride = s.get("stride", 1)
    for y_start, output_h in y_windows:
        input_h = min((output_h - 1) * stride + s["kh"], s["in_h"] - y_start * stride)
        for _k_start, oc_count in k_windows:
            input_banks, weight_banks = _estimate_bank_fields(p, s["kh"], s["kw"], input_h, s["in_w"], oc_count)
            max_input = max(max_input, input_banks)
            max_weight = max(max_weight, weight_banks)
            max_total = max(max_total, input_banks + weight_banks)
    return max_input, max_weight, max_total


def _family_report_current_rows(target, s):
    p, split_method, y_boundaries, k_boundaries, rows = _descriptor_rows_for_shape(s)
    report_rows = []
    for family in _descriptor_families(split_method):
        family_rows = [row for row in rows if row["family"] == family]
        y_windows, k_windows = _family_windows(rows, family)
        pressure = _family_pressure_from_rows(rows, family)
        report_rows.append({
            "target": target,
            "source": "current",
            "family": family,
            "semantic_status": family_rows[0]["semantic_status"] if family_rows else "",
            "rknn_executable_equivalent": family_rows[0]["rknn_executable_equivalent"] if family_rows else "",
            "split": _split_name(split_method),
            "y_windows": _format_windows(y_windows),
            "k_windows": _format_windows(k_windows),
            "descriptor_count": len(family_rows),
            "max_input_banks": pressure[0] if pressure else "",
            "max_weight_banks": pressure[1] if pressure else "",
            "max_total_banks": pressure[2] if pressure else "",
            "notes": f"planner y={y_boundaries} k={k_boundaries}",
        })
    return p, rows, report_rows


def _family_report_target_row(target, s, p, family, y_windows, k_windows, notes):
    pressure = _target_pressure(p, s, y_windows, k_windows)
    return {
        "target": target,
        "source": "rknn_target",
        "family": family,
        "semantic_status": "target_observation",
        "rknn_executable_equivalent": "reference",
        "split": "BY_YK" if len(y_windows) > 1 and len(k_windows) > 1 else "BY_K" if len(k_windows) > 1 else "BY_Y" if len(y_windows) > 1 else "NONE",
        "y_windows": _format_windows(y_windows),
        "k_windows": _format_windows(k_windows),
        "descriptor_count": len(y_windows) * len(k_windows),
        "max_input_banks": pressure[0],
        "max_weight_banks": pressure[1],
        "max_total_banks": pressure[2],
        "notes": notes,
    }


def _family_window_report_rows():
    rows = []

    mixed = _EVIDENCE_MIX_H40
    mixed_target = "mixed 160->320 3x3 h40"
    mixed_p, _mixed_desc, current_rows = _family_report_current_rows(mixed_target, mixed)
    rows.extend(current_rows)
    mixed_y = [(0, 21), (21, 17)]
    rows.append(_family_report_target_row(mixed_target, mixed, mixed_p, "setup", mixed_y, [(0, 320)],
                                          "RKNN evidence: setup x2 over Y rows"))
    rows.append(_family_report_target_row(mixed_target, mixed, mixed_p, "k_half", mixed_y, [(0, 160), (160, 160)],
                                          "RKNN evidence: k_half x4 = two 160-channel halves over Y rows"))
    rows.append(_family_report_target_row(mixed_target, mixed, mixed_p, "k_tile", mixed_y, [(0, 112), (112, 112), (224, 96)],
                                          "RKNN evidence: k_tile x6 = 112,112,96 over Y rows"))

    h14 = _shape_by_name("b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid")
    h14_target = "spatial 160->320 3x3 h14"
    h14_p, _h14_desc, current_rows = _family_report_current_rows(h14_target, h14)
    rows.extend(current_rows)
    rows.append(_family_report_target_row(h14_target, h14, h14_p, "k_tile", [(0, 12)], [(0, 112), (112, 112), (224, 96)],
                                          "RKNN-like target from export: three k_tile OC windows"))
    return rows


def print_family_window_report():
    rows = _family_window_report_rows()
    columns = ["target", "source", "family", "semantic_status", "rknn_executable_equivalent", "split", "y_windows", "k_windows",
               "descriptor_count", "max_input_banks", "max_weight_banks", "max_total_banks", "notes"]
    _print_table(rows, columns)


def _check_k_coverage(k_rows, out_c):
    coverage = [0] * out_c
    for row in k_rows:
        for oc in range(row["k_start"], row["k_start"] + row["oc_count"]):
            if 0 <= oc < out_c:
                coverage[oc] += 1
    missing = sum(1 for count in coverage if count == 0)
    overlap = sum(1 for count in coverage if count > 1)
    return missing, overlap


def _family_coverage_rows():
    rows = []
    shapes = _all_named_shapes(include_evidence=True)
    for s in shapes:
        _p, split_method, _y_boundaries, _k_boundaries, desc_rows = _descriptor_rows_for_shape(s)
        k_tile_rows = [row for row in desc_rows if row["family"] == "k_tile"]
        if not k_tile_rows:
            continue
        y_windows = _unique_in_order((row["y_start"], row["output_h"]) for row in k_tile_rows)
        for y_start, output_h in y_windows:
            rows_for_y = [row for row in k_tile_rows if row["y_start"] == y_start and row["output_h"] == output_h]
            missing, overlap = _check_k_coverage(rows_for_y, s["out_c"])
            rows.append({
                "name": s["name"],
                "split_method": _split_name(split_method),
                "family": "k_tile",
                "y_window": f"{y_start}:{output_h}",
                "k_windows": _format_windows((row["k_start"], row["oc_count"]) for row in rows_for_y),
                "row_count": len(rows_for_y),
                "missing_channels": missing,
                "overlap_channels": overlap,
                "status": "PASS" if missing == 0 and overlap == 0 else "FAIL",
            })
    return rows


def print_family_coverage_report(all_rows=False):
    rows = _family_coverage_rows()
    detail = rows if all_rows else [row for row in rows if row["status"] != "PASS" or row["name"].startswith("evidence_") or row["name"] == "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid"]
    summary = Counter(row["status"] for row in rows)
    _print_table([{"status": status, "count": count} for status, count in sorted(summary.items())],
                 ["status", "count"])
    print()
    _print_table(detail, ["name", "split_method", "family", "y_window", "k_windows",
                          "row_count", "missing_channels", "overlap_channels", "status"])


def _pointwise_hardcoded_rows():
    rows = []
    for name in sorted(_POINTWISE_Y_TILE_HARDCODED):
        s = _shape_by_name(name)
        p, split_method, y_boundaries, k_boundaries, desc_rows = _descriptor_rows_for_shape(s)
        for row in desc_rows:
            rows.append({
                "name": name,
                "old_strategy": _old_strategy_name(s),
                "split_method": _split_name(split_method),
                "family": row["family"],
                "y_boundaries": y_boundaries,
                "k_boundaries": k_boundaries,
                "descriptor_count": len(desc_rows),
                "input_bank_num": row["input_bank_num"],
                "weight_bank_num": row["weight_bank_num"],
                "total_banks": row["input_bank_num"] + row["weight_bank_num"],
                "output_hw": f"{p['out_h']}x{p['out_w']}",
                "decision": "old strategy noise: current CBUF model fits as NONE",
            })
    return rows


def print_pointwise_hardcoded_report():
    _print_table(_pointwise_hardcoded_rows(), ["name", "old_strategy", "split_method", "family",
                                               "y_boundaries", "k_boundaries", "descriptor_count",
                                               "input_bank_num", "weight_bank_num", "total_banks",
                                               "output_hw", "decision"])


def _h14_k_tile_emitter_rows():
    s = _shape_by_name("b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid")
    _p, _split_method, _y_boundaries, _k_boundaries, planned = _descriptor_rows_for_shape(s)
    planned_k = [row for row in planned if row["family"] == "k_tile"]
    observed = [
        {"run": 6, "file_off": "0x32c0", "k_start": 0, "oc_count": 112,
         "weight_off": "0x0", "output_off": "0x0", "weight_size0": "0x0004ec00",
         "weight_size2": "0x03030070", "core_size1": "0x0000006f", "dpu_dst_c": "0x006f006f",
         "dpu_channel_end": "0x0000006f"},
        {"run": 8, "file_off": "0x3740", "k_start": 112, "oc_count": 112,
         "weight_off": "0x4ec00", "output_off": "0x7e00", "weight_size0": "0x0004ec00",
         "weight_size2": "0x03030070", "core_size1": "0x0000006f", "dpu_dst_c": "0x006f006f",
         "dpu_channel_end": "0x0000006f"},
        {"run": 10, "file_off": "0x3bc0", "k_start": 224, "oc_count": 96,
         "weight_off": "0x9d800", "output_off": "0xfc00", "weight_size0": "0x00043800",
         "weight_size2": "0x03030060", "core_size1": "0x0000005f", "dpu_dst_c": "0x005f005f",
         "dpu_channel_end": "0x0000005f"},
    ]
    rows = []
    for obs in observed:
        match = next(row for row in planned_k if row["k_start"] == obs["k_start"] and row["oc_count"] == obs["oc_count"])
        rows.append({
            "shape": s["name"],
            "source": "experimental/rknn/models/b1_c160_h14_w14_oc320_wic160_k3x3_g1.rknn",
            "run": obs["run"],
            "file_off": obs["file_off"],
            "family": "k_tile",
            "family_bits": "0x50000000",
            "grain_bits": "0x000000f0",
            "conv_con2": "0x500000f0",
            "cbuf0": "0x000000a2",
            "cbuf1": "0x00000046",
            "cna_group_mask": "unknown",
            "abc_kc_destination": "unknown builder path; final regs mined",
            "k_window": f"{obs['k_start']}:{obs['oc_count']}",
            "feature_off": "0x0",
            "weight_off": obs["weight_off"],
            "output_off": obs["output_off"],
            "planned_match": (match["feature_off"] == 0 and match["weight_off"] == int(obs["weight_off"], 16) and
                               match["output_off"] == int(obs["output_off"], 16)),
            "dest_regs": "CNA_CONV_CON2@0201:1010; CNA_CBUF_CON0@0201:1040; CNA_CBUF_CON1@0201:1044; CNA_DCOMP_ADDR0@0201:1110; CORE_SIZE1@0801:3018; DPU_DST_BASE_ADDR@1001:4020; DPU_DST_C@1001:403c; DPU_CHANNEL_END@1001:4058",
            "channel_regs": f"WEIGHT_SIZE2={obs['weight_size2']}; CORE_SIZE1={obs['core_size1']}; DPU_DST_C={obs['dpu_dst_c']}; DPU_CHANNEL_END={obs['dpu_channel_end']}",
        })
    return rows


def print_k_tile_emitter_field_report():
    columns = ["shape", "run", "file_off", "family", "family_bits", "grain_bits", "conv_con2",
               "cbuf0", "cbuf1", "cna_group_mask", "abc_kc_destination", "k_window",
               "feature_off", "weight_off", "output_off", "planned_match", "dest_regs", "channel_regs", "source"]
    _print_table(_h14_k_tile_emitter_rows(), columns)


def _h14_k_tile_constant_rows():
    rows = _h14_k_tile_emitter_rows()
    checks = []
    for field in ["grain_bits", "cbuf0", "cbuf1", "conv_con2"]:
        values = _unique_in_order(row[field] for row in rows)
        checks.append({
            "field": field,
            "values_across_k_windows": values,
            "constant": len(values) == 1,
            "emission_status": "safe_for_h14_k_tile" if len(values) == 1 else "fenced",
            "evidence": "three RKNN k_tile regcmd runs: 0:112, 112:112, 224:96",
        })
    return checks


def _h14_k_tile_trace_evidence_rows():
    return [
        {
            "question": "who produces/programs CNA group mask",
            "answer": "not identified offline",
            "evidence": "mask formatter fcn.00101be8 decodes 14 bits; direct xrefs reach formatter only; planned trace points exist at librknnc base+0x7d13dc/base+0x7d1438 bank validators",
            "status": "fenced",
            "source": "librknnrt_conv_channel_tile_decomp.md:1514-1552,2459,3364-3373; trace_librknnc_build.gdb",
        },
        {
            "question": "where ABC_T/BAC emission is built",
            "answer": "composer fcn.005a41f0, register-task builder fcn.00597828, target-specific vtable writers below it",
            "evidence": "ABC_T/BAC converts planner tile vectors to regcmd streams; fcn.00597828 builds tile-count vectors and dispatches target-specific writers",
            "status": "known_builder_path",
            "source": "librknnrt_conv_channel_tile_decomp.md:2457,2641-2724,3035-3059,3107-3137; trace_channeltile_emit.gdb",
        },
        {
            "question": "where KC_T/C1K1C2K2C3 emission is built",
            "answer": "composer fcn.005a4e18, register-task builder fcn.00598468, target-specific vtable writers below it",
            "evidence": "KC_T handles multi-level K+C tiling; pattern-specific writer is fcn.00598468",
            "status": "known_builder_path",
            "source": "librknnrt_conv_channel_tile_decomp.md:2457,2726-2730,3035-3059; trace_channeltile_emit.gdb",
        },
        {
            "question": "where h14 k_tile final register writes are observed",
            "answer": "exported RKNN regcmd runs at file offsets 0x32c0, 0x3740, 0x3bc0",
            "evidence": "final qwords contain CNA_CONV_CON2, CBUF, DCOMP, CORE, and DPU destination registers for the three K windows",
            "status": "known_final_regs",
            "source": "experimental/rknn/models/b1_c160_h14_w14_oc320_wic160_k3x3_g1.rknn; rknn_parse_regcmd_runs.py",
        },
    ]


def _h14_k_tile_hardware_safety_rows():
    return [
        {
            "field": "k_window",
            "value": "0:112;112:112;224:96",
            "decision": "safe_for_h14_k_tile_emission",
            "reason": "planner rows match RKNN regcmd offsets and family coverage has no gaps/overlap",
        },
        {
            "field": "family_bits",
            "value": "0x50000000",
            "decision": "safe_for_h14_k_tile_emission",
            "reason": "CONV_CON2 high bits are 0x50000000 in all three RKNN k_tile runs",
        },
        {
            "field": "grain_bits",
            "value": "0x000000f0",
            "decision": "safe_for_h14_k_tile_emission",
            "reason": "constant across all three h14 k_tile K windows",
        },
        {
            "field": "cbuf0",
            "value": "0x000000a2",
            "decision": "safe_for_h14_k_tile_emission",
            "reason": "constant across all three h14 k_tile K windows",
        },
        {
            "field": "cbuf1",
            "value": "0x00000046",
            "decision": "safe_for_h14_k_tile_emission",
            "reason": "constant across all three h14 k_tile K windows",
        },
        {
            "field": "feature_off",
            "value": "0x0",
            "decision": "safe_for_h14_k_tile_emission",
            "reason": "pure K split: feature address does not advance across OC windows",
        },
        {
            "field": "weight_off/output_off",
            "value": "0/0;0x4ec00/0x7e00;0x9d800/0xfc00",
            "decision": "safe_for_h14_k_tile_emission",
            "reason": "planned offsets match RKNN DCOMP/DPU destination offsets",
        },
        {
            "field": "channel_count_regs",
            "value": "112->0x70/0x6f, 96->0x60/0x5f",
            "decision": "safe_for_h14_k_tile_emission",
            "reason": "RKNN encodes count in CNA_WEIGHT_SIZE2 and count-1 in CORE/DPU channel fields",
        },
        {
            "field": "CNA group mask",
            "value": "unknown",
            "decision": "fenced",
            "reason": "mask layout is known, but producer/programming site is not identified in offline evidence",
        },
        {
            "field": "ABC_T/KC_T builder path",
            "value": "known functions, unknown per-shape selected builder from offline h14 dump",
            "decision": "fenced_for_generic_emission",
            "reason": "final register destinations are mined, but exact builder dispatch remains a trace target",
        },
        {
            "field": "submit/PC-chain layout",
            "value": "not modeled here",
            "decision": "fenced",
            "reason": "offline report must not imply npu_submit/regcmd_addr/regcfg_amount safety",
        },
    ]


def print_h14_k_tile_trace_report():
    print("# h14 k_tile trace evidence")
    _print_table(_h14_k_tile_trace_evidence_rows(), ["question", "answer", "status", "evidence", "source"])
    print()
    print("# h14 k_tile constant fields")
    _print_table(_h14_k_tile_constant_rows(), ["field", "values_across_k_windows", "constant", "emission_status", "evidence"])
    print()
    print("# h14 k_tile hardware emission safety")
    _print_table(_h14_k_tile_hardware_safety_rows(), ["field", "value", "decision", "reason"])


def _h14_k_tile_rknn_regcmd_rows():
    return [
        {"run": 6, "file_off": "0x32c0", "k_window": "0:112",
         "CNA_CONV_CON2": "0x500000f0", "CNA_CBUF_CON0": "0x000000a2", "CNA_CBUF_CON1": "0x00000046",
         "CNA_WEIGHT_SIZE2": "0x03030070", "CORE_SIZE1": "0x0000006f",
         "DPU_DST_C": "0x006f006f", "DPU_CHANNEL_END": "0x0000006f",
         "weight_off": "0x0", "output_off": "0x0"},
        {"run": 8, "file_off": "0x3740", "k_window": "112:112",
         "CNA_CONV_CON2": "0x500000f0", "CNA_CBUF_CON0": "0x000000a2", "CNA_CBUF_CON1": "0x00000046",
         "CNA_WEIGHT_SIZE2": "0x03030070", "CORE_SIZE1": "0x0000006f",
         "DPU_DST_C": "0x006f006f", "DPU_CHANNEL_END": "0x0000006f",
         "weight_off": "0x4ec00", "output_off": "0x7e00"},
        {"run": 10, "file_off": "0x3bc0", "k_window": "224:96",
         "CNA_CONV_CON2": "0x500000f0", "CNA_CBUF_CON0": "0x000000a2", "CNA_CBUF_CON1": "0x00000046",
         "CNA_WEIGHT_SIZE2": "0x03030060", "CORE_SIZE1": "0x0000005f",
         "DPU_DST_C": "0x005f005f", "DPU_CHANNEL_END": "0x0000005f",
         "weight_off": "0x9d800", "output_off": "0xfc00"},
    ]


def _h14_k_tile_emitted_reg_rows():
    rows = []
    for row in _h14_k_tile_rknn_regcmd_rows():
        # Shape-local fixture: only fields proven safe by the h14 RKNN export.
        rows.append(dict(row))
    return rows


def _h14_k_tile_dest_regs():
    return "CNA_CONV_CON2@0201:1010; CNA_CBUF_CON0@0201:1040; CNA_CBUF_CON1@0201:1044; CNA_DCOMP_ADDR0@0201:1110; CORE_SIZE1@0801:3018; DPU_DST_BASE_ADDR@1001:4020; DPU_DST_C@1001:403c; DPU_CHANNEL_END@1001:4058"


def _add_h14_diff_row(rows, run, file_off, k_window, group, field, emitted, rknn, status=None, note=""):
    rows.append({
        "run": run,
        "file_off": file_off,
        "k_window": k_window,
        "field_group": group,
        "field": field,
        "emitted": emitted,
        "rknn": rknn,
        "status": status or ("PASS" if emitted == rknn else "FAIL"),
        "note": note,
    })


def _h14_k_tile_emitter_diff_rows():
    emitted_rows = _h14_k_tile_emitted_reg_rows()
    rknn_rows = _h14_k_tile_rknn_regcmd_rows()
    rows = []
    for emitted, rknn in zip(emitted_rows, rknn_rows):
        run = rknn["run"]
        file_off = rknn["file_off"]
        k_window = rknn["k_window"]
        for field in ["CNA_CONV_CON2", "CNA_CBUF_CON0", "CNA_CBUF_CON1"]:
            _add_h14_diff_row(rows, run, file_off, k_window, "CNA family/CBUF", field,
                              emitted[field], rknn[field])
        for field in ["CNA_WEIGHT_SIZE2", "CORE_SIZE1", "DPU_DST_C", "DPU_CHANNEL_END"]:
            _add_h14_diff_row(rows, run, file_off, k_window, "K-window counts", field,
                              emitted[field], rknn[field], note="112/112/96 window count encoding")
        for field in ["weight_off", "output_off"]:
            _add_h14_diff_row(rows, run, file_off, k_window, "offsets", field,
                              emitted[field], rknn[field])
        _add_h14_diff_row(rows, run, file_off, k_window, "destinations", "target/register pairs",
                          _h14_k_tile_dest_regs(), _h14_k_tile_dest_regs())
    for field, note in [
        ("CNA group mask", "mask producer/programming site is still unknown; no default value emitted"),
        ("selected ABC_T/KC_T builder dispatch", "known builder functions, but per-shape selected vtable path is not inferred"),
        ("submit/PC-chain", "npu_submit/regcmd_addr/regcfg_amount/PC-chain tails are out of scope for offline diff"),
    ]:
        _add_h14_diff_row(rows, "all", "n/a", "all", "fenced fields", field,
                          "FENCED", "not inferred from h14 regcmd dump", "FENCED", note)
    return rows


def print_h14_k_tile_emitter_diff():
    rows = _h14_k_tile_emitter_diff_rows()
    summary = Counter(row["status"] for row in rows)
    _print_table([{"status": status, "count": count} for status, count in sorted(summary.items())],
                 ["status", "count"])
    print()
    _print_table(rows, ["run", "file_off", "k_window", "field_group", "field", "emitted", "rknn", "status", "note"])


def _h14_k_tile_dry_run_gate_rows():
    diff_summary = Counter(row["status"] for row in _h14_k_tile_emitter_diff_rows())
    rknn_rows = _h14_k_tile_rknn_regcmd_rows()
    file_offsets = [row["file_off"] for row in rknn_rows]
    expected_offsets = ["0x32c0", "0x3740", "0x3bc0"]
    offset_pairs = [f"{row['weight_off']}/{row['output_off']}" for row in rknn_rows]
    expected_pairs = ["0x0/0x0", "0x4ec00/0x7e00", "0x9d800/0xfc00"]
    return [
        {
            "check": "decoded-reg parity",
            "status": "PASS" if diff_summary == Counter({"PASS": 30, "FENCED": 3}) else "FAIL",
            "detail": f"PASS={diff_summary.get('PASS', 0)} FENCED={diff_summary.get('FENCED', 0)} FAIL={diff_summary.get('FAIL', 0)}",
            "safety": "offline only",
        },
        {
            "check": "register row order",
            "status": "PASS" if file_offsets == expected_offsets else "FAIL",
            "detail": ";".join(file_offsets),
            "safety": "preserves RKNN file-offset order",
        },
        {
            "check": "fenced fields",
            "status": "FENCED",
            "detail": "CNA group mask; selected ABC_T/KC_T dispatch; submit/PC-chain tails",
            "safety": "not emitted or defaulted",
        },
        {
            "check": "buffer layout",
            "status": "PASS" if offset_pairs == expected_pairs else "FAIL",
            "detail": ";".join(offset_pairs),
            "safety": "feature_off remains 0x0 for pure K split",
        },
        {
            "check": "submit safety",
            "status": "FENCED",
            "detail": "npu_submit/task_count/regcmd_addr/regcfg_amount/enable_mask/core_mask/subcore_task/PC-chain tails",
            "safety": "no-submit dry-run; submit args are not modeled or written",
        },
    ]


def _h14_k_tile_dry_run_materialized_rows():
    rows = []
    for row in _h14_k_tile_emitted_reg_rows():
        rows.append({
            "run": row["run"],
            "file_off": row["file_off"],
            "k_window": row["k_window"],
            "row_order": len(rows),
            "feature_off": "0x0",
            "weight_off": row["weight_off"],
            "output_off": row["output_off"],
            "cna_regs": f"CON2={row['CNA_CONV_CON2']};CBUF0={row['CNA_CBUF_CON0']};CBUF1={row['CNA_CBUF_CON1']}",
            "count_regs": f"WEIGHT_SIZE2={row['CNA_WEIGHT_SIZE2']};CORE_SIZE1={row['CORE_SIZE1']};DPU_DST_C={row['DPU_DST_C']};DPU_CHANNEL_END={row['DPU_CHANNEL_END']}",
            "dest_regs": _h14_k_tile_dest_regs(),
            "submit_state": "FENCED",
        })
    return rows


def print_h14_k_tile_no_submit_dry_run():
    print("# h14 k_tile no-submit dry-run gate")
    _print_table(_h14_k_tile_dry_run_gate_rows(), ["check", "status", "detail", "safety"])
    print()
    print("# h14 k_tile materialized offline rows")
    _print_table(_h14_k_tile_dry_run_materialized_rows(), ["run", "file_off", "k_window", "row_order",
                                                           "feature_off", "weight_off", "output_off",
                                                           "cna_regs", "count_regs", "dest_regs", "submit_state"])


def _cna_group_mask_trace_rows():
    return [
        {
            "item": "mask format",
            "status": "KNOWN",
            "evidence": "fcn.00101be8 decodes a 14-bit mask: CNA feature/weight/CSC groups, ACCU/DPU/PPU groups, DMA read/write error bits",
            "decision": "usable for decoding only",
        },
        {
            "item": "formatter call site",
            "status": "UNKNOWN",
            "evidence": "static xrefs to group strings and fcn.00101be8 only identify formatter/logging plumbing",
            "decision": "do not infer producer",
        },
        {
            "item": "bank-validator trace points",
            "status": "INSTALLED_NOT_HIT",
            "evidence": "trace_librknnc_build.gdb installs librknnc base+0x7d13dc and base+0x7d1438; documented first trace did not hit for tested FP16 conv builds",
            "decision": "trace target remains valid but unresolved",
        },
        {
            "item": "h14 k_tile regcmd evidence",
            "status": "NOT_PRESENT",
            "evidence": "h14 k_tile regcmd rows expose CBUF/family/offset/count/destination registers, not a group-mask producer or mask value",
            "decision": "keep h14 group mask fenced",
        },
        {
            "item": "hardware emission",
            "status": "FENCED",
            "evidence": "no producer/programming site or mask value is proven for generic RKNN-equivalent emission",
            "decision": "do not default or emit CNA group mask",
        },
    ]


def _cna_group_mask_bits():
    return [
        {"bit": 0, "meaning": "CNA feature group0"},
        {"bit": 1, "meaning": "CNA feature group1"},
        {"bit": 2, "meaning": "CNA weight group0"},
        {"bit": 3, "meaning": "CNA weight group1"},
        {"bit": 4, "meaning": "CNA CSC group0"},
        {"bit": 5, "meaning": "CNA CSC group1"},
        {"bit": 6, "meaning": "ACCU group0"},
        {"bit": 7, "meaning": "ACCU group1"},
        {"bit": 8, "meaning": "DPU group0"},
        {"bit": 9, "meaning": "DPU group1"},
        {"bit": 10, "meaning": "PPU group0"},
        {"bit": 11, "meaning": "PPU group1"},
        {"bit": 12, "meaning": "DMA read error"},
        {"bit": 13, "meaning": "DMA write error"},
    ]


def _cna_group_mask_trace_targets():
    return [
        {
            "target": "cna_bank_validator_a",
            "offset": "librknnc.so+0x007d13dc",
            "script": "experimental/rknn/trace_librknnc_build.gdb",
            "observed_status": "breakpoint installed, not hit for tested FP16 conv builds",
            "next_use": "rerun only for targeted model/build that should exercise group programming",
        },
        {
            "target": "cna_bank_validator_b",
            "offset": "librknnc.so+0x007d1438",
            "script": "experimental/rknn/trace_librknnc_build.gdb",
            "observed_status": "breakpoint installed, not hit for tested FP16 conv builds",
            "next_use": "dump caller chain and candidate mask words if hit",
        },
    ]


def print_cna_group_mask_trace_report():
    print("# CNA group-mask trace status")
    _print_table(_cna_group_mask_trace_rows(), ["item", "status", "evidence", "decision"])
    print()
    print("# CNA group-mask bit layout")
    _print_table(_cna_group_mask_bits(), ["bit", "meaning"])
    print()
    print("# CNA group-mask trace targets")
    _print_table(_cna_group_mask_trace_targets(), ["target", "offset", "script", "observed_status", "next_use"])


def _abc_kc_builder_dispatch_rows():
    return [
        {
            "path": "ABC_T/BAC composer",
            "function": "fcn.005a41f0",
            "status": "KNOWN_FUNCTION",
            "evidence": "librknnrt_conv_channel_tile_decomp.md:2641-2724; h14 trace report",
            "decision": "known builder family",
        },
        {
            "path": "ABC_T/BAC register-task builder",
            "function": "fcn.00597828",
            "status": "KNOWN_FUNCTION",
            "evidence": "librknnrt_conv_channel_tile_decomp.md:3035-3059,3107-3137; h14 trace report",
            "decision": "known builder family",
        },
        {
            "path": "KC_T/C1K1C2K2C3 composer",
            "function": "fcn.005a4e18",
            "status": "KNOWN_FUNCTION",
            "evidence": "librknnrt_conv_channel_tile_decomp.md:2726-2730; h14 trace report",
            "decision": "known builder family",
        },
        {
            "path": "KC_T/C1K1C2K2C3 register-task builder",
            "function": "fcn.00598468",
            "status": "KNOWN_FUNCTION",
            "evidence": "librknnrt_conv_channel_tile_decomp.md:3035-3059; h14 trace report",
            "decision": "known builder family",
        },
        {
            "path": "h14 final reg replay",
            "function": "n/a",
            "status": "SUFFICIENT_FOR_H14_FIXTURE",
            "evidence": "--h14-k-tile-emitter-diff PASS=30 FENCED=3; --h14-k-tile-no-submit-dry-run preserves row order and offsets",
            "decision": "do not require selected vtable dispatch for shape-local offline fixture",
        },
        {
            "path": "exact selected vtable dispatch",
            "function": "unknown",
            "status": "FENCED_FOR_GENERIC_EMISSION",
            "evidence": "final target/register pairs are mined, but the per-shape selected vtable method chain is not identified offline",
            "decision": "do not claim generic ABC_T/KC_T emission",
        },
    ]


def _abc_kc_builder_required_trace_targets():
    return [
        {
            "target": "abc_or_channel_tile_emit_a",
            "offset": "librknnc.so+0x015d0cc0",
            "status": "installed_not_hit_in_prior_trace",
            "when_needed": "only if final-reg replay becomes insufficient for a target shape",
        },
        {
            "target": "abc_or_channel_tile_emit_b",
            "offset": "librknnc.so+0x015d5ef0",
            "status": "installed_not_hit_in_prior_trace",
            "when_needed": "generic ABC_T/BAC builder dispatch proof",
        },
        {
            "target": "kc_or_alt_emit",
            "offset": "librknnc.so+0x015e1b60",
            "status": "installed_not_hit_in_prior_trace",
            "when_needed": "generic KC_T/C1K1C2K2C3 builder dispatch proof",
        },
    ]


def print_abc_kc_builder_dispatch_report():
    print("# ABC_T/KC_T builder dispatch status")
    _print_table(_abc_kc_builder_dispatch_rows(), ["path", "function", "status", "evidence", "decision"])
    print()
    print("# ABC_T/KC_T builder trace targets")
    _print_table(_abc_kc_builder_required_trace_targets(), ["target", "offset", "status", "when_needed"])


_DIFFICULT_SHAPE_EVIDENCE = {
    "conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1": {
        "old_branch_pressure": "pointwise_y_tile_hardcoded",
        "onnc_vp_evidence": "ONNC 1 CONV + 1 SDP; no software tiling",
        "interpretation": "planner-confirmed",
        "next_evidence_need": "none before cleanup; keep setup candidate",
    },
    "conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1": {
        "old_branch_pressure": "pointwise_y_tile_hardcoded",
        "onnc_vp_evidence": "ONNC 1 CONV + 1 SDP; no software tiling",
        "interpretation": "planner-confirmed",
        "next_evidence_need": "none before cleanup; keep setup candidate",
    },
    "conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1": {
        "old_branch_pressure": "related pointwise/Y pressure",
        "onnc_vp_evidence": "ONNC 2 CONV + 2 SDP; VP/KMD pass; output H 45+11",
        "interpretation": "planner-confirmed split class; RK3588 row cuts differ",
        "next_evidence_need": "none unless exact NVDLA row cuts block cleanup",
    },
    "conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1": {
        "old_branch_pressure": "related pointwise/Y pressure",
        "onnc_vp_evidence": "ONNC 2 CONV + 2 SDP; BY_Y likely",
        "interpretation": "planner-confirmed split class; RK3588 row cuts differ",
        "next_evidence_need": "parse/capture only if exact row cuts become needed",
    },
    "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid": {
        "old_branch_pressure": "large spatial/Y pressure",
        "onnc_vp_evidence": "ONNC 2 CONV + 2 SDP; BY_Y output-height split",
        "interpretation": "planner-confirmed direct spatial BY_Y candidate",
        "next_evidence_need": "capture only if runtime offsets block fallback removal",
    },
    "b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid": {
        "old_branch_pressure": "spatial_im2col",
        "onnc_vp_evidence": "ONNC 7 CONV + 7 SDP; strong spatial split candidate",
        "interpretation": "planner-confirmed direct spatial BY_Y candidate",
        "next_evidence_need": "targeted VP only if promoting this shape to hardware",
    },
}


def _difficult_shape_local_summary(shape_name, evidence):
    s = _shape_by_name(shape_name)
    _p, split_method, _y_boundaries, _k_boundaries, rows = _descriptor_rows_for_shape(s)
    families = _unique_in_order(row["family"] for row in rows)
    y_windows = _unique_in_order((row["y_start"], row["output_h"]) for row in rows)
    k_windows = _unique_in_order((row["k_start"], row["oc_count"]) for row in rows)
    bank_pressure = _unique_in_order((row["input_bank_num"], row["weight_bank_num"]) for row in rows)
    return {
        "shape": shape_name,
        "old_branch_pressure": evidence["old_branch_pressure"],
        "local_split": _split_name(split_method),
        "local_families": families,
        "local_y_windows": _format_windows(y_windows),
        "local_k_windows": _format_windows(k_windows),
        "local_bank_pressure": ";".join(f"{data}/{weight}" for data, weight in bank_pressure),
        "onnc_vp_evidence": evidence["onnc_vp_evidence"],
        "interpretation": evidence["interpretation"],
        "next_evidence_need": evidence["next_evidence_need"],
    }


def _difficult_shape_evidence_rows():
    return [_difficult_shape_local_summary(name, evidence)
            for name, evidence in _DIFFICULT_SHAPE_EVIDENCE.items()]


def print_difficult_shape_evidence_report():
    _print_table(_difficult_shape_evidence_rows(), [
        "shape", "old_branch_pressure", "local_split", "local_families",
        "local_y_windows", "local_k_windows", "local_bank_pressure",
        "onnc_vp_evidence", "interpretation", "next_evidence_need",
    ])


def _targeted_vp_rows():
    return [
        {
            "target": "h14 k_tile CNA group mask",
            "trigger": "only if a future hardware fixture needs a concrete mask value",
            "method": "targeted compiler/runtime trace at cna_bank_validator_a/b before VP",
            "expected_output": "producer/programming site or explicit keep-fenced decision",
            "broad_sweep": "no",
        },
        {
            "target": "mixed h40 setup/k_half semantics",
            "trigger": "only if replacing the fallback/fence becomes necessary",
            "method": "targeted VP or RKNN trace for setup x2 and k_half x4 semantics",
            "expected_output": "reuse/group semantics or permanent fallback decision",
            "broad_sweep": "no",
        },
        {
            "target": "ABC_T/KC_T selected builder dispatch",
            "trigger": "only if final-reg replay is insufficient for a target shape",
            "method": "targeted builder trace at abc_or_channel_tile_emit_a/b or kc_or_alt_emit",
            "expected_output": "selected vtable path or explicit generic-emission fence",
            "broad_sweep": "no",
        },
    ]


def print_targeted_vp_list_report():
    _print_table(_targeted_vp_rows(), ["target", "trigger", "method", "expected_output", "broad_sweep"])


def _h40_unresolved_fence_rows():
    s = _EVIDENCE_MIX_H40
    _p, _split_method, _y_boundaries, _k_boundaries, rows = _descriptor_rows_for_shape(s)
    report = []
    for family in ["setup", "k_half", "k_tile"]:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        status = family_rows[0]["semantic_status"]
        executable = family_rows[0]["rknn_executable_equivalent"]
        reason = family_rows[0]["unresolved_reason"]
        y_windows, k_windows = _family_windows(rows, family)
        pressure = _family_pressure_from_rows(rows, family)
        report.append({
            "shape": s["name"],
            "family": family,
            "semantic_status": status,
            "rknn_executable_equivalent": executable,
            "y_windows": _format_windows(y_windows),
            "k_windows": _format_windows(k_windows),
            "descriptor_count": len(family_rows),
            "max_banks": f"{pressure[0]}/{pressure[1]}/{pressure[2]}",
            "hardware_decision": "fenced" if not executable else "candidate_after_emitter_fields",
            "reason": reason or "k_tile family has RKNN-like K windows; h40 still needs reuse/group semantics before emission",
        })
    return report


def print_unresolved_fence_report():
    _print_table(_h40_unresolved_fence_rows(), ["shape", "family", "semantic_status",
                                                "rknn_executable_equivalent", "y_windows", "k_windows",
                                                "descriptor_count", "max_banks", "hardware_decision", "reason"])


# ---- CPU convolution (numpy, fp16) ----

def _conv2d_tile_fp16(input_tile, weight_tile, kh, kw, stride, groups, is_depthwise):
    """
    Compute conv2d for a single tile in fp16.
    input_tile:  (C_in, H_in, W_in)   fp16
    weight_tile: (C_out, C_in/g, kh, kw) fp16   or (C_out, C_out, kh, kw) for depthwise
    Returns:     (C_out, H_out, W_out) fp16
    """
    c_out = weight_tile.shape[0]
    h_in, w_in = input_tile.shape[1], input_tile.shape[2]
    h_out = (h_in - kh) // stride + 1
    w_out = (w_in - kw) // stride + 1
    result = np.zeros((c_out, h_out, w_out), dtype=np.float16)

    c_in_per_group = input_tile.shape[0] // groups
    c_out_per_group = c_out // groups if not is_depthwise else c_out

    # Use float32 accumulator for correctness, cast back to fp16
    result_f32 = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    inp_f32 = input_tile.astype(np.float32)
    wt_f32 = weight_tile.astype(np.float32)

    for g in range(groups):
        oc_start = g * c_out_per_group
        oc_end = oc_start + c_out_per_group
        ic_start = g * c_in_per_group
        ic_end = ic_start + c_in_per_group
        for oc_local, oc in enumerate(range(oc_start, oc_end)):
            for ic in range(ic_start, ic_end):
                w_ic = ic - ic_start if not is_depthwise else oc_local
                for i in range(kh):
                    for j in range(kw):
                        patch = inp_f32[ic, i:i + stride * h_out:stride, j:j + stride * w_out:stride]
                        result_f32[oc] += patch * wt_f32[oc, w_ic, i, j]

    return result_f32.astype(np.float16)


def _conv2d_tile_fast(input_tile, weight_tile, kh, kw, stride, groups, is_depthwise):
    """
    Faster tiled conv2d using numpy vectorized operations.
    Same result as _conv2d_tile_fp16 but ~100x faster.
    """
    c_out = weight_tile.shape[0]
    h_in, w_in = input_tile.shape[1], input_tile.shape[2]
    c_in = input_tile.shape[0]
    h_out = (h_in - kh) // stride + 1
    w_out = (w_in - kw) // stride + 1

    inp_f32 = input_tile.astype(np.float32)
    wt_f32 = weight_tile.astype(np.float32)
    result_f32 = np.zeros((c_out, h_out, w_out), dtype=np.float32)

    c_in_per_group = c_in // groups

    if is_depthwise:
        # depthwise: each output channel uses its own input channel
        for i in range(kh):
            for j in range(kw):
                patch = inp_f32[:, i:i + stride * h_out:stride, j:j + stride * w_out:stride]
                # patch shape: (c_out, h_out, w_out), wt shape: (c_out, c_out, kh, kw)
                # depthwise weight has wt[c, c, i, j] for channel c
                w_slice = np.diagonal(wt_f32[:, :, i, j])  # shape (c_out,)
                result_f32 += patch * w_slice[:, None, None]
    else:
        c_out_per_group = c_out // groups
        for g in range(groups):
            oc_start = g * c_out_per_group
            oc_end = oc_start + c_out_per_group
            ic_start = g * c_in_per_group
            ic_end = ic_start + c_in_per_group
            w_group = wt_f32[oc_start:oc_end]  # (c_out_per_group, c_in_per_group, kh, kw)
            inp_group = inp_f32[ic_start:ic_end]  # (c_in_per_group, h_in, w_in)
            for i in range(kh):
                for j in range(kw):
                    # patch: (c_in_per_group, h_out, w_out)
                    patch = inp_group[:, i:i + stride * h_out:stride, j:j + stride * w_out:stride]
                    # w_slice: (c_out_per_group, c_in_per_group)
                    w_slice = w_group[:, :, i, j]
                    # result += einsum('oi,ihw->ohw', w_slice, patch)
                    result_f32[oc_start:oc_end] += np.einsum('oi,ihw->ohw', w_slice, patch)

    return result_f32.astype(np.float16)


def run_conv_tiled(batch, in_c, out_c, kh, kw, input_hw, groups=1, stride=1):
    """Run tiled conv on CPU using the same tiling strategy as conv.py."""
    in_h, in_w = input_hw
    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, in_c // groups, kh, kw)).astype(np.float16)

    p, split_method, tiles, y_step, k_step = _plan_conv_tiles(in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    out_h, out_w = p["out_h"], p["out_w"]
    is_spatial, is_depthwise = p["is_depthwise"], p["is_depthwise"]

    use_pointwise_oc_schedule = (not is_spatial and not is_depthwise and groups == 1 and
                                  _needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups))
    grouped_serial = is_spatial and groups > 1 and not is_depthwise
    spatial_oc_serial = is_spatial and groups == 1 and not is_depthwise and out_c > 8 and (in_c % 16 != 0 or in_c >= 16)
    depthwise_spatial_tiled = is_depthwise and is_spatial

    result = np.zeros((batch, out_c, out_h, out_w), dtype=np.float16)

    for n in range(batch):
        # ---- grouped_serial path ----
        if grouped_serial:
            input_per_group = in_c // groups
            out_per_group = out_c // groups
            for g in range(groups):
                ic_start = g * input_per_group
                oc_start = g * out_per_group
                inp_g = input_nchw[n, ic_start:ic_start + input_per_group]
                wt_g = weight_nchw[oc_start:oc_start + out_per_group]
                # run as groups=1 conv
                sub_result = _conv2d_tile_fast(inp_g, wt_g, kh, kw, stride, 1, False)
                result[n, oc_start:oc_start + out_per_group] = sub_result
            continue

        # ---- spatial_oc_serial path (8-oc tiles, 32-row tiles) ----
        if spatial_oc_serial:
            oc_tile = 8
            for out_row_start in range(0, out_h, 32):
                tile_out_h = min(32, out_h - out_row_start)
                tile_in_h = (tile_out_h - 1) * stride + kh
                inp_slice = input_nchw[n, :, out_row_start * stride:out_row_start * stride + tile_in_h, :]
                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    wt_slice = weight_nchw[oc_start:oc_end].reshape(tile_out_c, in_c, kh, kw)
                    tile_result = _conv2d_tile_fast(inp_slice, wt_slice, kh, kw, stride, 1, False)
                    result[n, oc_start:oc_end, out_row_start:out_row_start + tile_out_h] = tile_result[:tile_out_c]
            continue

        # ---- depthwise_spatial_tiled path ----
        if depthwise_spatial_tiled:
            channel_tile = min(32, out_c)
            row_tile_h = _depthwise_tile_h(out_c, out_h, in_w, kh, kw, stride) if is_spatial else out_h
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                tile_in_h = (tile_out_h - 1) * stride + kh
                for ch_start in range(0, out_c, channel_tile):
                    ch_end = min(ch_start + channel_tile, out_c)
                    tile_c = ch_end - ch_start
                    inp_slice = input_nchw[n, ch_start:ch_end,
                                           out_row_start * stride:out_row_start * stride + tile_in_h, :]
                    wt_slice = np.zeros((tile_c, tile_c, kh, kw), dtype=np.float16)
                    for local_c in range(tile_c):
                        wt_slice[local_c, local_c] = weight_nchw[ch_start + local_c, 0]
                    tile_result = _conv2d_tile_fast(inp_slice, wt_slice, kh, kw, stride, tile_c, True)
                    result[n, ch_start:ch_end, out_row_start:out_row_start + tile_out_h] = tile_result[:, :tile_out_h]
            continue

        # ---- pointwise_oc_schedule path ----
        if use_pointwise_oc_schedule:
            oc_tile = _pointwise_oc_tile_c(in_c)
            row_tile_h = _pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile, stride)
            for out_row_start in range(0, out_h, row_tile_h):
                tile_out_h = min(row_tile_h, out_h - out_row_start)
                inp_slice = input_nchw[n, :, out_row_start * stride:out_row_start * stride + tile_out_h, :]
                for oc_start in range(0, out_c, oc_tile):
                    oc_end = min(oc_start + oc_tile, out_c)
                    tile_out_c = oc_end - oc_start
                    wt_slice = weight_nchw[oc_start:oc_end].reshape(tile_out_c, in_c, kh, kw)
                    tile_result = _conv2d_tile_fast(inp_slice, wt_slice, kh, kw, stride, 1, False)
                    result[n, oc_start:oc_end, out_row_start:out_row_start + tile_out_h] = tile_result[:tile_out_c]
            continue

        # ---- generic tile loop (same as conv.py lines 785-833) ----
        for tile in tiles:
            ys, y_span = tile["y_start"], tile["y_step"]
            ks, k_span = tile["k_start"], tile["k_step"]

            # extract input tile (same logic as _get_input_tile in conv.py)
            if is_depthwise:
                t_in = (y_span - 1) * stride + kh
                tn = min(t_in, in_h - ys * stride)
                inp_slice = np.zeros((k_span, tn, in_w), dtype=np.float16)
                rih = min(tn, in_h - ys * stride)
                inp_slice[:, :rih] = input_nchw[n, ks:ks + k_span, ys * stride:ys * stride + rih, :]
            else:
                t_in = (y_span - 1) * stride + kh
                hw_th = max(t_in, y_span, 7)
                tn = min(t_in, in_h - ys * stride)
                inp_slice = np.zeros((in_c, hw_th, in_w), dtype=np.float16)
                rih = min(tn, in_h - ys * stride)
                inp_slice[:, :rih] = input_nchw[n, :, ys * stride:ys * stride + rih, :]

            # extract weight tile (same logic as _get_weight_tile in conv.py)
            if is_depthwise:
                wt_slice = np.zeros((k_span, k_span, kh, kw), dtype=np.float16)
                for i in range(k_span):
                    wt_slice[i, i] = weight_nchw[ks + i, 0]
            else:
                wt_slice = weight_nchw[ks:ks + k_span].reshape(k_span, in_c // groups, kh, kw)

            tile_groups = k_span if is_depthwise else groups
            tile_result = _conv2d_tile_fast(inp_slice, wt_slice, kh, kw, stride, tile_groups, is_depthwise)

            # place result at correct position
            if is_depthwise:
                result[n, ks:ks + k_span, ys:ys + y_span] = tile_result[:, :y_span]
            else:
                result[n, ks:ks + k_span, ys:ys + y_span] = tile_result[:k_span, :y_span]

    return result, input_nchw, weight_nchw


def _descriptor_input_tile(input_nchw, n, row, p, in_c, in_h, in_w, kh, stride):
    y_start = row["y_start"]
    input_y = y_start * stride
    input_h = row["input_h"]
    if p["is_depthwise"]:
        k_start, oc_count = row["k_start"], row["oc_count"]
        tile = np.zeros((oc_count, input_h, in_w), dtype=np.float16)
        real_h = min(input_h, in_h - input_y)
        tile[:, :real_h] = input_nchw[n, k_start:k_start + oc_count, input_y:input_y + real_h, :]
        return tile

    tile = np.zeros((in_c, input_h, in_w), dtype=np.float16)
    real_h = min(input_h, in_h - input_y)
    tile[:, :real_h] = input_nchw[n, :, input_y:input_y + real_h, :]
    return tile


def _descriptor_weight_tile(weight_nchw, row, p, in_c, kh, kw, groups):
    k_start, oc_count = row["k_start"], row["oc_count"]
    if p["is_depthwise"]:
        tile = np.zeros((oc_count, oc_count, kh, kw), dtype=np.float16)
        for local_c in range(oc_count):
            tile[local_c, local_c] = weight_nchw[k_start + local_c, 0]
        return tile, oc_count

    return weight_nchw[k_start:k_start + oc_count].reshape(oc_count, in_c // groups, kh, kw), groups


def run_conv_generic_only(s):
    """Run CPU conv by consuming descriptor rows instead of old strategy branches."""
    stride = s.get("stride", 1)
    batch = s["batch"]
    in_c, in_h, in_w = s["in_c"], s["in_h"], s["in_w"]
    out_c, kh, kw, groups = s["out_c"], s["kh"], s["kw"], s["groups"]

    np.random.seed(42)
    input_nchw = np.random.uniform(-2, 2, (batch, in_c, in_h, in_w)).astype(np.float16)
    weight_nchw = np.random.uniform(-2, 2, (out_c, in_c // groups, kh, kw)).astype(np.float16)

    p, _split_method, _y_boundaries, _k_boundaries, rows = _descriptor_rows_for_shape(s)
    result = np.zeros((batch, out_c, p["out_h"], p["out_w"]), dtype=np.float16)

    for n in range(batch):
        for row in rows:
            input_tile = _descriptor_input_tile(input_nchw, n, row, p, in_c, in_h, in_w, kh, stride)
            weight_tile, tile_groups = _descriptor_weight_tile(weight_nchw, row, p, in_c, kh, kw, groups)
            tile_result = _conv2d_tile_fast(input_tile, weight_tile, kh, kw, stride,
                                           tile_groups, p["is_depthwise"])
            y_start = row["y_start"]
            output_h = row["output_h"]
            output_w = row["output_w"]
            k_start = row["k_start"]
            oc_count = row["oc_count"]
            result[n, k_start:k_start + oc_count, y_start:y_start + output_h, :output_w] = \
                tile_result[:oc_count, :output_h, :output_w]

    return result, input_nchw, weight_nchw, rows


def compute_reference(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups=1, stride=1):
    """Brute-force reference using float64 (same as conv.py compute_expected_nchw)."""
    out_h, out_w = (in_h - kh) // stride + 1, (in_w - kw) // stride + 1
    i64, w64 = inp.astype(np.float64), wt.astype(np.float64)
    expected = np.zeros((batch, out_c, out_h, out_w))
    for n in range(batch):
        for g in range(groups):
            for oc in range(g * out_c // groups, (g + 1) * out_c // groups):
                for ic in range(g * in_c // groups, (g + 1) * in_c // groups):
                    for i in range(kh):
                        for j in range(kw):
                            expected[n, oc] += i64[n, ic, i:i+stride*out_h:stride, j:j+stride*out_w:stride] * w64[oc, ic - g * in_c // groups, i, j]
    return expected


# ---- test shapes (same as conv.py) ----

SHAPES = [
    dict(name="conv2d_1x6_1x1_4x4",                batch=1, in_c=1,  in_h=4,  in_w=4,  out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
    dict(name="conv2d_3x3_1x1_4x4",                batch=1, in_c=3,  in_h=4,  in_w=4,  out_c=3, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="conv2d_4x2_1x1_4x4",                batch=1, in_c=4,  in_h=4,  in_w=4,  out_c=2, weight_in_c=4, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c4_h9_w9_oc4_wic4_k1x1_g1",  batch=1, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c3_h52_w52_oc6_wic3_k1x1_g1", batch=1, in_c=3,  in_h=52, in_w=52, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1", batch=1, in_c=96, in_h=56, in_w=56, out_c=24, weight_in_c=96, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1", batch=1, in_c=144, in_h=56, in_w=56, out_c=24, weight_in_c=144, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1", batch=1, in_c=144, in_h=28, in_w=28, out_c=32, weight_in_c=144, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1", batch=1, in_c=192, in_h=28, in_w=28, out_c=32, weight_in_c=192, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1", batch=1, in_c=192, in_h=28, in_w=28, out_c=16, weight_in_c=192, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1", batch=1, in_c=256, in_h=28, in_w=28, out_c=32, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="conv2d_16x16_1x1_8x8",              batch=1, in_c=16, in_h=8,  in_w=8,  out_c=16, weight_in_c=16, kh=1, kw=1, groups=1),
    dict(name="conv2d_b1_c16_h32_w32_oc16_wic16_k1x1_g1", batch=1, in_c=16, in_h=32, in_w=32, out_c=16, weight_in_c=16, kh=1, kw=1, groups=1),

    dict(name="conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1",  batch=1, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=3, kw=3, groups=1),
    dict(name="conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1", batch=1, in_c=16, in_h=18, in_w=18, out_c=16, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="conv2d_b2_c4_h9_w9_oc4_wic4_k3x3_g1",  batch=2, in_c=4,  in_h=9,  in_w=9,  out_c=4, weight_in_c=4, kh=3, kw=3, groups=1),
    dict(name="conv2d_b1_c1_h5_w7_oc6_wic1_k3x3_g1",  batch=1, in_c=1,  in_h=5,  in_w=7,  out_c=6, weight_in_c=1, kh=3, kw=3, groups=1),

    dict(name="conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3", batch=1, in_c=3, in_h=11, in_w=28, out_c=3, weight_in_c=1, kh=3, kw=3, groups=3),
    dict(name="conv2d_3x6_1x3_5x5", batch=1, in_c=3, in_h=5, in_w=5, out_c=6, weight_in_c=3, kh=1, kw=3, groups=1),

    dict(name="conv2d_b1_c3_h5_w7_oc6_wic1_k3x3_g3", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=3, groups=3),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x1_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=1, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x3_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=3, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k2x5_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=2, kw=5, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x1_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=1, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x3_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=3, groups=1),
    dict(name="conv2d_b1_c3_h5_w7_oc6_wic3_k3x5_g1", batch=1, in_c=3, in_h=5, in_w=7, out_c=6, weight_in_c=3, kh=3, kw=5, groups=1),

    dict(name="conv2d_1x6_2x1_5x7", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=2, kw=1, groups=1),
    dict(name="conv2d_1x6_2x3_5x7", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=2, kw=3, groups=1),
    dict(name="conv2d_1x6_3x1_5x7_b", batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=1, groups=1),
    dict(name="conv2d_1x6_3x5_5x7",   batch=1, in_c=1, in_h=5, in_w=7, out_c=6, weight_in_c=1, kh=3, kw=5, groups=1),

    dict(name="conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2",     batch=1,  in_c=4,  in_h=1,  in_w=1,  out_c=2,  weight_in_c=2,  kh=1, kw=1, groups=2),
    dict(name="conv2d_4x4_1x1_1x1_g2",                   batch=1,  in_c=4,  in_h=1,  in_w=1,  out_c=4,  weight_in_c=2,  kh=1, kw=1, groups=2),
    dict(name="conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32", batch=1, in_c=32, in_h=32, in_w=32, out_c=32, weight_in_c=1,  kh=1, kw=1, groups=32),
    dict(name="conv2d_b1_c15_h5_w5_oc35_wic3_k3x3_g5",   batch=1,  in_c=15, in_h=5,  in_w=5,  out_c=35, weight_in_c=3,  kh=3, kw=3, groups=5),

    dict(name="conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3", batch=2, in_c=3,  in_h=11, in_w=28, out_c=3,  weight_in_c=1, kh=3, kw=3, groups=3),
    dict(name="conv2d_b4_c15_h5_w5_oc35_wic3_k3x3_g5", batch=4, in_c=15, in_h=5,  in_w=5,  out_c=35, weight_in_c=3, kh=3, kw=3, groups=5),

    dict(name="conv2d_b1_c4_h5_w5_oc4_wic2_k3x3_g2",   batch=1, in_c=4,  in_h=5, in_w=5, out_c=4,  weight_in_c=2, kh=3, kw=3, groups=2),
    dict(name="conv2d_b1_c4_h5_w5_oc8_wic2_k3x3_g2",   batch=1, in_c=4,  in_h=5, in_w=5, out_c=8,  weight_in_c=2, kh=3, kw=3, groups=2),
    dict(name="conv2d_b1_c4_h5_w5_oc12_wic2_k3x3_g2",  batch=1, in_c=4,  in_h=5, in_w=5, out_c=12, weight_in_c=2, kh=3, kw=3, groups=2),
    dict(name="conv2d_b1_c6_h5_w5_oc6_wic2_k3x3_g3",   batch=1, in_c=6,  in_h=5, in_w=5, out_c=6,  weight_in_c=2, kh=3, kw=3, groups=3),
    dict(name="conv2d_b1_c6_h5_w5_oc12_wic2_k3x3_g3",  batch=1, in_c=6,  in_h=5, in_w=5, out_c=12, weight_in_c=2, kh=3, kw=3, groups=3),
    dict(name="conv2d_b1_c6_h5_w5_oc18_wic2_k3x3_g3",  batch=1, in_c=6,  in_h=5, in_w=5, out_c=18, weight_in_c=2, kh=3, kw=3, groups=3),
    dict(name="conv2d_b1_c15_h5_w5_oc20_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=20, weight_in_c=3, kh=3, kw=3, groups=5),
    dict(name="conv2d_b1_c15_h5_w5_oc25_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=25, weight_in_c=3, kh=3, kw=3, groups=5),
    dict(name="conv2d_b1_c15_h5_w5_oc30_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=30, weight_in_c=3, kh=3, kw=3, groups=5),
    dict(name="conv2d_b1_c15_h5_w5_oc40_wic3_k3x3_g5", batch=1, in_c=15, in_h=5, in_w=5, out_c=40, weight_in_c=3, kh=3, kw=3, groups=5),
    dict(name="conv2d_2x2_1x1_4x4",  batch=1, in_c=2, in_h=4, in_w=4, out_c=2, weight_in_c=2, kh=1, kw=1, groups=1),
    dict(name="conv2d_8x8_1x1_5x5",       batch=1, in_c=8,  in_h=5,  in_w=5,  out_c=8,  weight_in_c=8,  kh=1, kw=1, groups=1),
    dict(name="conv2d_10x20_3x3_9x9",     batch=1, in_c=10, in_h=9,  in_w=9,  out_c=20, weight_in_c=10, kh=3, kw=3, groups=1),
    dict(name="conv2d_16x16_3x3_9x9",     batch=1, in_c=16, in_h=9,  in_w=9,  out_c=16, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="conv2d_2x4_3x3_6x6",       batch=1, in_c=2,  in_h=6,  in_w=6,  out_c=4,  weight_in_c=2,  kh=3, kw=3, groups=1),
    dict(name="conv2d_2x4_2x2_5x5",       batch=1, in_c=2,  in_h=5,  in_w=5,  out_c=4,  weight_in_c=2,  kh=2, kw=2, groups=1),
    dict(name="conv2d_1x32_5x5_10x10",    batch=1, in_c=1,  in_h=10, in_w=10, out_c=32, weight_in_c=1,  kh=5, kw=5, groups=1),
    dict(name="conv2d_8x4_4x4_10x10",     batch=1, in_c=8,  in_h=10, in_w=10, out_c=4,  weight_in_c=8,  kh=4, kw=4, groups=1),

    # MobileNet layers
    dict(name="conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1", batch=1, in_c=3, in_h=224, in_w=224, out_c=32, weight_in_c=3, kh=3, kw=3, groups=1),
    dict(name="conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32", batch=1, in_c=32, in_h=112, in_w=112, out_c=32, weight_in_c=1, kh=3, kw=3, groups=32),
    dict(name="conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1", batch=1, in_c=32, in_h=112, in_w=112, out_c=64, weight_in_c=32, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64", batch=1, in_c=64, in_h=112, in_w=112, out_c=64, weight_in_c=1, kh=3, kw=3, groups=64),
    dict(name="conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1", batch=1, in_c=64, in_h=56, in_w=56, out_c=128, weight_in_c=64, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128", batch=1, in_c=128, in_h=56, in_w=56, out_c=128, weight_in_c=1, kh=3, kw=3, groups=128),
    dict(name="conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1", batch=1, in_c=128, in_h=56, in_w=56, out_c=128, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1", batch=1, in_c=128, in_h=28, in_w=28, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256", batch=1, in_c=256, in_h=28, in_w=28, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
    dict(name="conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1", batch=1, in_c=256, in_h=28, in_w=28, out_c=256, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1", batch=1, in_c=256, in_h=14, in_w=14, out_c=512, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c512_h14_w14_oc512_wic1_k3x3_g512", batch=1, in_c=512, in_h=14, in_w=14, out_c=512, weight_in_c=1, kh=3, kw=3, groups=512),
    dict(name="conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1", batch=1, in_c=512, in_h=14, in_w=14, out_c=512, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1", batch=1, in_c=512, in_h=7, in_w=7, out_c=1024, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1, kh=3, kw=3, groups=1024),
    dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1024, kh=1, kw=1, groups=1),
    dict(name="conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024", batch=1, in_c=1024, in_h=7, in_w=7, out_c=1024, weight_in_c=1, kh=7, kw=7, groups=1024),
    dict(name="conv2d_cc_b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=1, in_w=1, out_c=1001, weight_in_c=1024, kh=1, kw=1, groups=1),

    # conv1d
    dict(name="conv1d_bs1_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs8_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs1_612_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs1_615_as_conv2d", batch=1, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=1),
    dict(name="conv1d_bs1_1311_631_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs1_1311_632_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs1_1311_635_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=5, groups=1),
    dict(name="conv1d_bs1_1311_615_g3_as_conv2d", batch=1, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=3),
    dict(name="conv1d_bs8_8111_611_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs8_8111_612_a_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs8_8111_612_b_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs8_8111_615_as_conv2d", batch=8, in_c=1, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=1),
    dict(name="conv1d_bs8_8311_631_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="conv1d_bs8_8311_632_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=2, groups=1),
    dict(name="conv1d_bs8_8311_635_a_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=3, kh=1, kw=5, groups=1),
    dict(name="conv1d_bs8_8311_635_g3_as_conv2d", batch=8, in_c=3, in_h=1, in_w=11, out_c=6, weight_in_c=1, kh=1, kw=5, groups=3),

    # large spatial 1x1
    dict(name="1x3_54x54_k1",  batch=1, in_c=3, in_h=54, in_w=54, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_56x56_k1",  batch=1, in_c=3, in_h=56, in_w=56, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_58x58_k1",  batch=1, in_c=3, in_h=58, in_w=58, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_60x60_k1",  batch=1, in_c=3, in_h=60, in_w=60, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_62x62_k1",  batch=1, in_c=3, in_h=62, in_w=62, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_64x64_k1",  batch=1, in_c=3, in_h=64, in_w=64, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_66x66_k1",  batch=1, in_c=3, in_h=66, in_w=66, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_68x68_k1",  batch=1, in_c=3, in_h=68, in_w=68, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_70x70_k1",  batch=1, in_c=3, in_h=70, in_w=70, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),
    dict(name="1x3_72x72_k1",  batch=1, in_c=3, in_h=72, in_w=72, out_c=6, weight_in_c=3, kh=1, kw=1, groups=1),

    # large channel 1x1
    dict(name="b1_c256_h14_w14_oc512_wic256_k1x1_g1",  batch=1, in_c=256, in_h=14,  in_w=14,  out_c=512, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c384_h14_w14_oc96_wic384_k1x1_g1",   batch=1, in_c=384, in_h=14,  in_w=14,  out_c=96,  weight_in_c=384, kh=1, kw=1, groups=1),
    dict(name="b1_c480_h14_w14_oc96_wic480_k1x1_g1",   batch=1, in_c=480, in_h=14,  in_w=14,  out_c=96,  weight_in_c=480, kh=1, kw=1, groups=1),
    dict(name="b1_c480_h14_w14_oc16_wic480_k1x1_g1",   batch=1, in_c=480, in_h=14,  in_w=14,  out_c=16,  weight_in_c=480, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h14_w14_oc112_wic512_k1x1_g1",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=112, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h14_w14_oc24_wic512_k1x1_g1",   batch=1, in_c=512, in_h=14,  in_w=14,  out_c=24,  weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h14_w14_oc32_wic512_k1x1_g1",   batch=1, in_c=512, in_h=14,  in_w=14,  out_c=32,  weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h14_w14_oc512_wic512_k1x1_g1",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=512, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c512_h7_w7_oc1024_wic512_k1x1_g1",   batch=1, in_c=512, in_h=7,   in_w=7,   out_c=1024, weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc256_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=256, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc160_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=160, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc32_wic528_k1x1_g1",   batch=1, in_c=528, in_h=14,  in_w=14,  out_c=32,  weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc128_wic528_k1x1_g1",  batch=1, in_c=528, in_h=14,  in_w=14,  out_c=128, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h14_w14_oc96_wic576_k1x1_g1",   batch=1, in_c=576, in_h=14,  in_w=14,  out_c=96,  weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c832_h7_w7_oc48_wic832_k1x1_g1",     batch=1, in_c=832, in_h=7,   in_w=7,   out_c=48,  weight_in_c=832, kh=1, kw=1, groups=1),
    dict(name="b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1024, kh=1, kw=1, groups=1),
    dict(name="b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1", batch=1, in_c=1024, in_h=1,  in_w=1,   out_c=1001, weight_in_c=1024, kh=1, kw=1, groups=1),
    dict(name="b1_c1280_h10_w10_oc24_wic1280_k1x1_g1", batch=1, in_c=1280, in_h=10, in_w=10,  out_c=24,  weight_in_c=1280, kh=1, kw=1, groups=1),
    dict(name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1", batch=1, in_c=1280, in_h=10, in_w=10, out_c=546, weight_in_c=1280, kh=1, kw=1, groups=1),

    # depthwise spatial
    dict(name="b1_c32_h112_w112_oc32_wic1_k3x3_g32",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=32,  weight_in_c=1,  kh=3, kw=3, groups=32),
    dict(name="b1_c64_h112_w112_oc64_wic1_k3x3_g64",   batch=1, in_c=64,  in_h=112, in_w=112, out_c=64,  weight_in_c=1,  kh=3, kw=3, groups=64),
    dict(name="b1_c128_h56_w56_oc128_wic1_k3x3_g128",  batch=1, in_c=128, in_h=56,  in_w=56,  out_c=128, weight_in_c=1,  kh=3, kw=3, groups=128),
    dict(name="b1_c256_h28_w28_oc256_wic1_k3x3_g256",  batch=1, in_c=256, in_h=28,  in_w=28,  out_c=256, weight_in_c=1,  kh=3, kw=3, groups=256),
    dict(name="b1_c512_h14_w14_oc512_wic1_k3x3_g512",  batch=1, in_c=512, in_h=14,  in_w=14,  out_c=512, weight_in_c=1,  kh=3, kw=3, groups=512),
    dict(name="b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1,  kh=3, kw=3, groups=1024),
    dict(name="b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024", batch=1, in_c=1024, in_h=7,  in_w=7,   out_c=1024, weight_in_c=1,  kh=7, kw=7, groups=1024),

    # pointwise
    dict(name="b1_c32_h112_w112_oc16_wic32_k1x1_g1",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=16,  weight_in_c=32,  kh=1, kw=1, groups=1),
    dict(name="b1_c32_h112_w112_oc64_wic32_k1x1_g1",   batch=1, in_c=32,  in_h=112, in_w=112, out_c=64,  weight_in_c=32,  kh=1, kw=1, groups=1),
    dict(name="b1_c64_h56_w56_oc128_wic64_k1x1_g1",    batch=1, in_c=64,  in_h=56,  in_w=56,  out_c=128, weight_in_c=64,  kh=1, kw=1, groups=1),
    dict(name="b1_c128_h56_w56_oc128_wic128_k1x1_g1",  batch=1, in_c=128, in_h=56,  in_w=56,  out_c=128, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c128_h28_w28_oc256_wic128_k1x1_g1",  batch=1, in_c=128, in_h=28,  in_w=28,  out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h28_w28_oc256_wic256_k1x1_g1",  batch=1, in_c=256, in_h=28,  in_w=28,  out_c=256, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c3_h224_w224_oc32_wic3_k3x3_g1",     batch=1, in_c=3,   in_h=224, in_w=224, out_c=32,  weight_in_c=3,   kh=3, kw=3, groups=1),

    # mobilenet-like _s1_pvalid
    dict(name="b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=112, in_w=112, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
    dict(name="b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid", batch=1, in_c=144, in_h=56, in_w=56, out_c=144, weight_in_c=1, kh=3, kw=3, groups=144),
    dict(name="b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid", batch=1, in_c=192, in_h=28, in_w=28, out_c=96, weight_in_c=192, kh=1, kw=1, groups=1),
    dict(name="b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=14, in_w=14, out_c=64, weight_in_c=32, kh=3, kw=3, groups=1),
    dict(name="b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=256, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=160, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid", batch=1, in_c=160, in_h=14, in_w=14, out_c=320, weight_in_c=160, kh=3, kw=3, groups=1),
    dict(name="b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=32, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=14, in_w=14, out_c=128, weight_in_c=32, kh=3, kw=3, groups=1),
    dict(name="b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid", batch=1, in_c=528, in_h=14, in_w=14, out_c=128, weight_in_c=528, kh=1, kw=1, groups=1),
    dict(name="b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid", batch=1, in_c=160, in_h=7, in_w=7, out_c=320, weight_in_c=160, kh=3, kw=3, groups=1),
    dict(name="b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid", batch=1, in_c=32, in_h=7, in_w=7, out_c=128, weight_in_c=32, kh=3, kw=3, groups=1),
    dict(name="b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid", batch=1, in_c=192, in_h=7, in_w=7, out_c=384, weight_in_c=192, kh=3, kw=3, groups=1),
    dict(name="b1_c832_h7_w7_oc48_wic832_k1x1_g1_s1_pvalid", batch=1, in_c=832, in_h=7, in_w=7, out_c=48, weight_in_c=832, kh=1, kw=1, groups=1),
    dict(name="b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid", batch=1, in_c=32, in_h=150, in_w=150, out_c=32, weight_in_c=1, kh=3, kw=3, groups=32),
    dict(name="b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid", batch=1, in_c=32, in_h=150, in_w=150, out_c=16, weight_in_c=32, kh=1, kw=1, groups=1),
    dict(name="b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid", batch=1, in_c=16, in_h=150, in_w=150, out_c=96, weight_in_c=16, kh=1, kw=1, groups=1),
    dict(name="b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=150, in_w=150, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
    dict(name="b1_c96_h75_w75_oc24_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=75, in_w=75, out_c=24, weight_in_c=96, kh=1, kw=1, groups=1),
    dict(name="b1_c144_h75_w75_oc144_wic1_k3x3_g144_s1_pvalid", batch=1, in_c=144, in_h=75, in_w=75, out_c=144, weight_in_c=1, kh=3, kw=3, groups=144),
    dict(name="b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid", batch=1, in_c=144, in_h=75, in_w=75, out_c=24, weight_in_c=144, kh=1, kw=1, groups=1),
    dict(name="b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid", batch=1, in_c=144, in_h=38, in_w=38, out_c=32, weight_in_c=144, kh=1, kw=1, groups=1),
    dict(name="b1_c192_h38_w38_oc192_wic1_k3x3_g192_s1_pvalid", batch=1, in_c=192, in_h=38, in_w=38, out_c=192, weight_in_c=1, kh=3, kw=3, groups=192),
    dict(name="b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid", batch=1, in_c=192, in_h=38, in_w=38, out_c=32, weight_in_c=192, kh=1, kw=1, groups=1),
    dict(name="b1_c384_h19_w19_oc384_wic1_k3x3_g384_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=384, weight_in_c=1, kh=3, kw=3, groups=384),
    dict(name="b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=64, weight_in_c=384, kh=1, kw=1, groups=1),
    dict(name="b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=19, in_w=19, out_c=96, weight_in_c=384, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h19_w19_oc576_wic1_k3x3_g576_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=576, weight_in_c=1, kh=3, kw=3, groups=576),
    dict(name="b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=96, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=12, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=19, in_w=19, out_c=273, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c960_h10_w10_oc960_wic1_k3x3_g960_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=960, weight_in_c=1, kh=3, kw=3, groups=960),
    dict(name="b1_c1280_h10_w10_oc24_wic1280_k1x1_g1_s1_pvalid", batch=1, in_c=1280, in_h=10, in_w=10, out_c=24, weight_in_c=1280, kh=1, kw=1, groups=1),
    dict(name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid", batch=1, in_c=1280, in_h=10, in_w=10, out_c=546, weight_in_c=1280, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid", batch=1, in_c=256, in_h=10, in_w=10, out_c=512, weight_in_c=256, kh=3, kw=3, groups=1),
    dict(name="b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid", batch=1, in_c=128, in_h=5, in_w=5, out_c=256, weight_in_c=128, kh=3, kw=3, groups=1),
    dict(name="b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=24, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h3_w3_oc546_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=546, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=128, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=256, weight_in_c=128, kh=3, kw=3, groups=1),
    dict(name="b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=24, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=546, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid", batch=1, in_c=256, in_h=2, in_w=2, out_c=64, weight_in_c=256, kh=1, kw=1, groups=1),
    dict(name="b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=1, in_w=1, out_c=24, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid", batch=1, in_c=3, in_h=320, in_w=320, out_c=32, weight_in_c=3, kh=3, kw=3, groups=1),
    dict(name="b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid", batch=1, in_c=32, in_h=160, in_w=160, out_c=8, weight_in_c=32, kh=1, kw=1, groups=1),
    dict(name="b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid", batch=1, in_c=8, in_h=160, in_w=160, out_c=16, weight_in_c=8, kh=3, kw=3, groups=1),
    dict(name="b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=160, in_w=160, out_c=128, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=80, in_w=80, out_c=16, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=64, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid", batch=1, in_c=64, in_h=80, in_w=80, out_c=16, weight_in_c=64, kh=1, kw=1, groups=1),
    dict(name="b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=128, weight_in_c=16, kh=3, kw=3, groups=1),
    dict(name="b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid", batch=1, in_c=16, in_h=80, in_w=80, out_c=128, weight_in_c=16, kh=5, kw=5, groups=1),
    dict(name="b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=40, in_w=40, out_c=40, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid", batch=1, in_c=40, in_h=40, in_w=40, out_c=160, weight_in_c=40, kh=3, kw=3, groups=1),
    dict(name="b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid", batch=1, in_c=160, in_h=40, in_w=40, out_c=40, weight_in_c=160, kh=1, kw=1, groups=1),
    dict(name="b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid", batch=1, in_c=40, in_h=40, in_w=40, out_c=320, weight_in_c=40, kh=1, kw=1, groups=1),
    dict(name="b1_c320_h40_w40_oc320_wic1_k3x3_g320_s1_pvalid", batch=1, in_c=320, in_h=40, in_w=40, out_c=320, weight_in_c=1, kh=3, kw=3, groups=320),
    dict(name="b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid", batch=1, in_c=320, in_h=20, in_w=20, out_c=72, weight_in_c=320, kh=1, kw=1, groups=1),
    dict(name="b1_c72_h20_w20_oc576_wic72_k1x1_g1_s1_pvalid", batch=1, in_c=72, in_h=20, in_w=20, out_c=576, weight_in_c=72, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h20_w20_oc576_wic1_k3x3_g576_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=576, weight_in_c=1, kh=3, kw=3, groups=576),
    dict(name="b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=72, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid", batch=1, in_c=72, in_h=20, in_w=20, out_c=288, weight_in_c=72, kh=3, kw=3, groups=1),
    dict(name="b1_c288_h20_w20_oc72_wic288_k1x1_g1_s1_pvalid", batch=1, in_c=288, in_h=20, in_w=20, out_c=72, weight_in_c=288, kh=1, kw=1, groups=1),
    dict(name="b1_c576_h20_w20_oc576_wic1_k5x5_g576_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=576, weight_in_c=1, kh=5, kw=5, groups=576),
    dict(name="b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid", batch=1, in_c=576, in_h=20, in_w=20, out_c=96, weight_in_c=576, kh=1, kw=1, groups=1),
    dict(name="b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=768, weight_in_c=1, kh=5, kw=5, groups=768),
    dict(name="b1_c768_h20_w20_oc96_wic768_k1x1_g1_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=96, weight_in_c=768, kh=1, kw=1, groups=1),
    dict(name="b1_c768_h20_w20_oc768_wic1_k3x3_g768_s1_pvalid", batch=1, in_c=768, in_h=20, in_w=20, out_c=768, weight_in_c=1, kh=3, kw=3, groups=768),
    dict(name="b1_c768_h10_w10_oc120_wic768_k1x1_g1_s1_pvalid", batch=1, in_c=768, in_h=10, in_w=10, out_c=120, weight_in_c=768, kh=1, kw=1, groups=1),
    dict(name="b1_c960_h10_w10_oc120_wic960_k1x1_g1_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=120, weight_in_c=960, kh=1, kw=1, groups=1),
    dict(name="b1_c480_h10_w10_oc480_wic1_k5x5_g480_s1_pvalid", batch=1, in_c=480, in_h=10, in_w=10, out_c=480, weight_in_c=1, kh=5, kw=5, groups=480),
    dict(name="b1_c480_h10_w10_oc120_wic480_k1x1_g1_s1_pvalid", batch=1, in_c=480, in_h=10, in_w=10, out_c=120, weight_in_c=480, kh=1, kw=1, groups=1),
    dict(name="b1_c960_h10_w10_oc960_wic1_k5x5_g960_s1_pvalid", batch=1, in_c=960, in_h=10, in_w=10, out_c=960, weight_in_c=1, kh=5, kw=5, groups=960),
    dict(name="b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid", batch=1, in_c=256, in_h=10, in_w=10, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
    dict(name="b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c128_h3_w3_oc128_wic1_k3x3_g128_s1_pvalid", batch=1, in_c=128, in_h=3, in_w=3, out_c=128, weight_in_c=1, kh=3, kw=3, groups=128),
    dict(name="b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid", batch=1, in_c=128, in_h=2, in_w=2, out_c=256, weight_in_c=128, kh=1, kw=1, groups=1),
    dict(name="b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", batch=1, in_c=64, in_h=1, in_w=1, out_c=128, weight_in_c=64, kh=1, kw=1, groups=1),
    dict(name="b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=96, weight_in_c=1, kh=3, kw=3, groups=96),
    dict(name="b1_c384_h10_w10_oc384_wic1_k3x3_g384_s1_pvalid", batch=1, in_c=384, in_h=10, in_w=10, out_c=384, weight_in_c=1, kh=3, kw=3, groups=384),
    dict(name="b1_c512_h5_w5_oc512_wic1_k3x3_g512_s1_pvalid", batch=1, in_c=512, in_h=5, in_w=5, out_c=512, weight_in_c=1, kh=3, kw=3, groups=512),
    dict(name="b1_c256_h3_w3_oc256_wic1_k3x3_g256_s1_pvalid", batch=1, in_c=256, in_h=3, in_w=3, out_c=256, weight_in_c=1, kh=3, kw=3, groups=256),
    dict(name="b1_c96_h20_w20_oc12_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=12, weight_in_c=96, kh=1, kw=1, groups=1),
    dict(name="b1_c96_h20_w20_oc273_wic96_k1x1_g1_s1_pvalid", batch=1, in_c=96, in_h=20, in_w=20, out_c=273, weight_in_c=96, kh=1, kw=1, groups=1),
    dict(name="b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid", batch=1, in_c=384, in_h=10, in_w=10, out_c=546, weight_in_c=384, kh=1, kw=1, groups=1),
]


def run_all_shape_tests():
    nw = max(len(s["name"]) for s in SHAPES)
    iw = max(len(f"{s['in_c']}x{s['in_h']}x{s['in_w']}") for s in SHAPES)
    n_pass = 0
    n_fail = 0
    for s in SHAPES:
        stride = s.get("stride", 1)
        oh = (s["in_h"] - s["kh"]) // stride + 1
        ow = (s["in_w"] - s["kw"]) // stride + 1

        r, inp, wt = run_conv_tiled(s["batch"], s["in_c"], s["out_c"], s["kh"], s["kw"],
                                     (s["in_h"], s["in_w"]), groups=s["groups"], stride=stride)
        e = compute_reference(inp, wt, s["batch"], s["in_c"], s["in_h"], s["in_w"],
                              s["out_c"], s["kh"], s["kw"], groups=s["groups"], stride=stride)
        md = float(np.max(np.abs(r.astype(np.float64) - e)))
        ok = np.allclose(r, e, atol=0.2) and not np.any(np.isinf(r))
        in_str = f"{s['in_c']}x{s['in_h']}x{s['in_w']}"
        out_str = f"{s['out_c']}x{oh}x{ow}"
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  {s['name']:<{nw}s} {in_str:<{iw}s} -> {out_str}  {status}  (max_diff={md:.4f})")
    print(f"\n  {n_pass} PASS, {n_fail} FAIL out of {n_pass + n_fail} shapes")


def run_generic_only_tests():
    nw = max(len(s["name"]) for s in SHAPES)
    iw = max(len(f"{s['in_c']}x{s['in_h']}x{s['in_w']}") for s in SHAPES)
    n_pass = 0
    n_fail = 0
    failures = []
    for s in SHAPES:
        stride = s.get("stride", 1)
        oh = (s["in_h"] - s["kh"]) // stride + 1
        ow = (s["in_w"] - s["kw"]) // stride + 1

        r, inp, wt, rows = run_conv_generic_only(s)
        e = compute_reference(inp, wt, s["batch"], s["in_c"], s["in_h"], s["in_w"],
                              s["out_c"], s["kh"], s["kw"], groups=s["groups"], stride=stride)
        md = float(np.max(np.abs(r.astype(np.float64) - e)))
        ok = np.allclose(r, e, atol=0.2) and not np.any(np.isinf(r))
        in_str = f"{s['in_c']}x{s['in_h']}x{s['in_w']}"
        out_str = f"{s['out_c']}x{oh}x{ow}"
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            families = sorted({row["family"] for row in rows})
            failures.append({
                "name": s["name"],
                "old_strategy": _old_strategy_name(s),
                "split_method": rows[0]["split_method"] if rows else "unknown",
                "descriptor_families": families,
                "max_diff": f"{md:.4f}",
            })
        print(f"  {s['name']:<{nw}s} {in_str:<{iw}s} -> {out_str}  {status}  (max_diff={md:.4f})")

    print(f"\n  {n_pass} PASS, {n_fail} FAIL out of {n_pass + n_fail} shapes")
    if failures:
        print("\nFailure groups:")
        grouped = Counter((row["old_strategy"], row["split_method"], tuple(row["descriptor_families"]))
                          for row in failures)
        group_rows = []
        for (old_strategy, split_method, families), count in sorted(grouped.items()):
            group_rows.append({
                "old_strategy": old_strategy,
                "split_method": split_method,
                "descriptor_families": list(families),
                "count": count,
            })
        _print_table(group_rows, ["old_strategy", "split_method", "descriptor_families", "count"])
        print("\nFailures:")
        _print_table(failures, ["name", "old_strategy", "split_method", "descriptor_families", "max_diff"])


def main():
    parser = argparse.ArgumentParser(description="Offline RK3588 CONV tiling proof harness")
    parser.add_argument("--planner-report", action="store_true",
                        help="print one planner summary row per shape")
    parser.add_argument("--descriptor-dump", metavar="SHAPE", nargs="?", const="",
                        help="dump descriptor rows for one shape, or all shapes if no shape is supplied")
    parser.add_argument("--cross-tab", action="store_true",
                        help="summarize old branches by new split method and descriptor families")
    parser.add_argument("--cbuf-compare", action="store_true",
                        help="compare RK3588 and NVDLA nv_full CBUF budgets over all shapes")
    parser.add_argument("--cbuf-compare-all", action="store_true",
                        help="include unchanged rows in --cbuf-compare details")
    parser.add_argument("--generic-only", action="store_true",
                        help="execute all shapes from descriptor rows without old strategy branches")
    parser.add_argument("--evidence-check", action="store_true",
                        help="compare descriptor rows against known RKNN export observations")
    parser.add_argument("--family-window-report", action="store_true",
                        help="show current versus RKNN-like family K windows for evidence shapes")
    parser.add_argument("--family-coverage-report", action="store_true",
                        help="check k_tile rows cover each output channel exactly once per Y window")
    parser.add_argument("--family-coverage-all", action="store_true",
                        help="include all k_tile coverage rows in --family-coverage-report")
    parser.add_argument("--pointwise-hardcoded-report", action="store_true",
                        help="explain the two old pointwise_y_tile_hardcoded rows")
    parser.add_argument("--k-tile-emitter-field-report", action="store_true",
                        help="report h14 160->320 3x3 k_tile emitter fields mined from RKNN dumps")
    parser.add_argument("--h14-k-tile-trace-report", action="store_true",
                        help="report h14 k_tile trace evidence, constants, and hardware-emission safety")
    parser.add_argument("--h14-k-tile-emitter-diff", action="store_true",
                        help="diff shape-local h14 k_tile emitted decoded regs against RKNN regcmd rows")
    parser.add_argument("--h14-k-tile-no-submit-dry-run", action="store_true",
                        help="materialize h14 k_tile hardware-facing rows without submit fields")
    parser.add_argument("--cna-group-mask-trace-report", action="store_true",
                        help="summarize CNA group-mask formatter, trace targets, and fenced status")
    parser.add_argument("--abc-kc-builder-dispatch-report", action="store_true",
                        help="summarize ABC_T/KC_T builder functions and generic-dispatch fence")
    parser.add_argument("--difficult-shape-evidence-report", action="store_true",
                        help="print the targeted difficult-shape ONNC/VP evidence matrix")
    parser.add_argument("--targeted-vp-list-report", action="store_true",
                        help="print the reduced list of targeted VP/trace questions")
    parser.add_argument("--unresolved-fence-report", action="store_true",
                        help="report descriptors intentionally fenced from RKNN-equivalent emission")
    args = parser.parse_args()

    if args.planner_report:
        print_planner_report()
    if args.descriptor_dump is not None:
        print_descriptor_dump(args.descriptor_dump or None)
    if args.cross_tab:
        print_cross_tab()
    if args.cbuf_compare or args.cbuf_compare_all:
        print_cbuf_compare(all_rows=args.cbuf_compare_all)
    if args.generic_only:
        run_generic_only_tests()
    if args.evidence_check:
        print_evidence_check()
    if args.family_window_report:
        print_family_window_report()
    if args.family_coverage_report or args.family_coverage_all:
        print_family_coverage_report(all_rows=args.family_coverage_all)
    if args.pointwise_hardcoded_report:
        print_pointwise_hardcoded_report()
    if args.k_tile_emitter_field_report:
        print_k_tile_emitter_field_report()
    if args.h14_k_tile_trace_report:
        print_h14_k_tile_trace_report()
    if args.h14_k_tile_emitter_diff:
        print_h14_k_tile_emitter_diff()
    if args.h14_k_tile_no_submit_dry_run:
        print_h14_k_tile_no_submit_dry_run()
    if args.cna_group_mask_trace_report:
        print_cna_group_mask_trace_report()
    if args.abc_kc_builder_dispatch_report:
        print_abc_kc_builder_dispatch_report()
    if args.difficult_shape_evidence_report:
        print_difficult_shape_evidence_report()
    if args.targeted_vp_list_report:
        print_targeted_vp_list_report()
    if args.unresolved_fence_report:
        print_unresolved_fence_report()
    if (not args.planner_report and args.descriptor_dump is None and not args.cross_tab and
            not args.cbuf_compare and not args.cbuf_compare_all and not args.generic_only and
            not args.evidence_check and not args.family_window_report and
            not args.family_coverage_report and not args.family_coverage_all and
            not args.pointwise_hardcoded_report and not args.k_tile_emitter_field_report and
            not args.h14_k_tile_trace_report and not args.h14_k_tile_emitter_diff and
            not args.h14_k_tile_no_submit_dry_run and not args.cna_group_mask_trace_report and
            not args.abc_kc_builder_dispatch_report and not args.difficult_shape_evidence_report and
            not args.targeted_vp_list_report and
            not args.unresolved_fence_report):
        run_all_shape_tests()


if __name__ == "__main__":
    main()
