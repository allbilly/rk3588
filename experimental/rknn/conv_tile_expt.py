#!/usr/bin/env python3
"""Analytical convolution tiling experiment for conv_new_clean.py.

This deliberately does not import conv_new_clean.py because that module opens the
NPU and allocates buffers at import time.  The goal is to classify every shape in
its __main__ test list, compare tiling strategies, and write a cleanup report.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


FP16_BYTES = 2
FP16_ATOM_ELEMENTS = 16
WEIGHT_ATOMIC_ELEMENTS = 32
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES
UNPACK_C2 = FP16_ATOM_ELEMENTS // FP16_BYTES
RK_MAX_CONV_FLAT_STRIDE = 992
OUTPUT_BYTES = 4 * 1024 * 1024

SOURCE = Path("examples/kernel_6_18/conv_new_clean.py")
REPORT = Path("conv_tile_result_and_cleanup_plan.md")


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align_up(x: int, align: int) -> int:
    return ceil_div(x, align) * align


def is_depthwise(in_c: int, out_c: int, groups: int) -> bool:
    return groups == in_c == out_c


def is_grouped_spatial(in_c: int, out_c: int, kh: int, kw: int, groups: int) -> bool:
    return (kh != 1 or kw != 1) and groups > 1 and in_c != groups and not is_depthwise(in_c, out_c, groups)


def pointwise_weight_atom_groups(in_c: int) -> int:
    return ceil_div(max(in_c, FP16_ATOM_ELEMENTS), WEIGHT_ATOMIC_ELEMENTS)


def uses_pointwise_weight_atom_layout(in_c: int, kh: int, kw: int, groups: int) -> bool:
    return groups == 1 and kh == 1 and kw == 1 and pointwise_weight_atom_groups(in_c) > 1


def conv_align_c(in_c: int, groups: int, out_c: int) -> int:
    if not is_depthwise(in_c, out_c, groups) and groups == 1 and in_c > 1:
        return 16
    if not is_depthwise(in_c, out_c, groups) and in_c > 4:
        return 16
    return min(max(1 << (max(1, in_c) - 1).bit_length(), 8), 32)


def cdma_dc_feature_input_pack_c2(in_c: int, groups: int, out_c: int, align_c: int) -> int:
    if in_c == 1:
        return 2
    if not is_depthwise(in_c, out_c, groups) and groups == 1 and 1 < in_c <= 4:
        return 8
    if is_depthwise(in_c, out_c, groups) or groups > 1 or in_c > 4:
        return 8
    return align_c


def mesa_entries_per_slice(input_width: int, input_channels: int) -> int:
    atomics_per_entry = CBUF_ENTRY_BYTES // FP16_ATOM_ELEMENTS
    total_c_atomics = ceil_div(input_channels * FP16_BYTES, FP16_ATOM_ELEMENTS)
    last_c_atomics = total_c_atomics % atomics_per_entry
    int_c_entries = (total_c_atomics // atomics_per_entry) * input_width
    frac_c_entries = input_width if last_c_atomics == 3 else ceil_div(last_c_atomics * input_width, atomics_per_entry)
    return int_c_entries + frac_c_entries


def mesa_weight_banks(weights_width: int, weights_height: int, input_channels: int, output_channels: int, depthwise: bool) -> int:
    weight_bytes = weights_width * weights_height * input_channels * FP16_BYTES
    if not depthwise:
        weight_bytes *= output_channels
    return ceil_div(ceil_div(weight_bytes, CBUF_ENTRY_BYTES), CBUF_ENTRIES_PER_BANK) + 1


def mesa_output_tile_h(input_width: int, out_h: int, input_channels: int, output_channels: int, kh: int, kw: int, stride: int, depthwise: bool, input_banks: int | None = None) -> int:
    if input_banks is None:
        weight_banks = mesa_weight_banks(kw, kh, input_channels, output_channels, depthwise)
        input_banks = RK_CBUF_BANKS - weight_banks if weight_banks + 1 < RK_CBUF_BANKS else 7
    entries_per_slice = max(1, mesa_entries_per_slice(input_width, input_channels))
    input_slices = max(1, (CBUF_ENTRIES_PER_BANK * input_banks) // entries_per_slice)
    output_rows = max(1, (input_slices - kh) // stride + 1)
    return min(out_h, output_rows)


def pointwise_oc_tile_c(in_c: int) -> int:
    max_tile = CBUF_BANK_SIZE // (max(1, in_c) * FP16_BYTES)
    return 32 if max_tile >= 32 else 16 if max_tile >= 16 else 8 if max_tile >= 8 else 4


def pointwise_oc_tile_h(in_c: int, out_h: int, out_w: int, oc_tile: int, stride: int = 1) -> int:
    return mesa_output_tile_h(out_w, out_h, in_c, oc_tile, 1, 1, stride, False)


def feature_grains(row_bytes: int, floor_grains: int, use_nhwc_pack: bool = False, is_spatial: bool = False, depthwise: bool = False) -> int:
    if use_nhwc_pack and is_spatial:
        return floor_grains
    if depthwise and is_spatial:
        return min(13, floor_grains)
    even_rows_per_two_banks = (ceil_div(2 * CBUF_BANK_SIZE, row_bytes) + 1) & ~1
    return min(floor_grains, even_rows_per_two_banks)


def depthwise_tile_h(total_channels: int, out_h: int, in_w: int, kh: int, kw: int, stride: int = 1) -> int:
    tile_h = mesa_output_tile_h(in_w, out_h, total_channels, total_channels, kh, kw, stride, True, input_banks=7)
    if total_channels > 64:
        align_c = conv_align_c(total_channels, total_channels, total_channels)
        row_bytes = in_w * align_c * FP16_BYTES
        max_feature_rows = feature_grains(row_bytes, out_h + kh, is_spatial=True, depthwise=True) + 1
        tile_h = min(tile_h, max_feature_rows)
    tile_h = max(10, tile_h) if tile_h < out_h else tile_h
    return tile_h if tile_h == out_h or tile_h % 2 == 0 else tile_h - 1


def pointwise_tile_h(in_c: int, out_c: int, out_h: int, out_w: int) -> int:
    if out_h > 50:
        return min(out_h, 20)
    if (in_c, out_c, out_h, out_w) == (256, 32, 28, 28):
        return 20
    return out_h


def needs_pointwise_tile_schedule(in_c: int, out_c: int, in_h: int, in_w: int, kh: int, kw: int, groups: int) -> bool:
    if groups != 1 or kh != 1 or kw != 1:
        return False
    return ((in_c, out_c, in_h, in_w) == (96, 24, 56, 56) or
            (in_c, out_c, in_h, in_w) == (144, 24, 56, 56) or
            (in_c, out_c, in_h, in_w) == (144, 32, 28, 28) or
            (in_c, out_c, in_h, in_w) == (192, 32, 28, 28) or
            (in_c, out_c, in_h, in_w) == (192, 16, 28, 28) or
            (in_c, out_c, in_h, in_w) == (256, 32, 28, 28))


def needs_pointwise_oc_tile_schedule(in_c: int, out_c: int, in_h: int, in_w: int, kh: int, kw: int, groups: int) -> bool:
    if groups != 1 or kh != 1 or kw != 1:
        return False
    out_h = in_h
    out_w = in_w
    oc_tile = pointwise_oc_tile_c(in_c)
    return out_c > oc_tile or (in_c >= 16 and out_c % oc_tile != 0) or pointwise_oc_tile_h(in_c, out_h, out_w, oc_tile) < out_h


@dataclass(frozen=True)
class Params:
    batch: int
    in_c: int
    in_h: int
    in_w: int
    out_c: int
    kh: int
    kw: int
    groups: int
    stride: int
    depthwise: bool
    out_h: int
    out_w: int
    align_c: int
    align_out_c: int
    width_stride: int
    out_width_stride: int
    input_pack_c2: int
    use_nhwc: bool
    mesa_aligned_small: bool


def conv_params(batch: int, in_c: int, in_h: int, in_w: int, out_c: int, kh: int, kw: int, groups: int, stride: int = 1) -> Params:
    depthwise = is_depthwise(in_c, out_c, groups)
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    align_c = WEIGHT_ATOMIC_ELEMENTS if uses_pointwise_weight_atom_layout(in_c, kh, kw, groups) else conv_align_c(in_c, groups, out_c)
    align_out_c = max(32, align_up(out_c, 16))
    width_stride = align_up(in_w, max(1, ceil_div(16, align_c)))
    out_atoms = max(1, out_h * out_w)
    out_width_stride = out_atoms if kh == 1 and kw == 1 else align_up(out_atoms, 4)
    mesa_aligned_small = (not depthwise) and groups == 1 and 1 < in_c <= 4
    input_pack_c2 = cdma_dc_feature_input_pack_c2(in_c, groups, out_c, align_c)
    use_nhwc = (not mesa_aligned_small) and (not depthwise) and (in_c > 0 and input_pack_c2 // in_c == 2)
    return Params(batch, in_c, in_h, in_w, out_c, kh, kw, groups, stride, depthwise,
                  out_h, out_w, align_c, align_out_c, width_stride, out_width_stride,
                  input_pack_c2, use_nhwc, mesa_aligned_small)


@dataclass
class Plan:
    strategy: str
    output_format: str
    tasks: int
    detail: str


def output_bytes(p: Params, out_c: int | None = None, width_stride: int | None = None) -> int:
    out_c = p.align_out_c if out_c is None else max(32, align_up(out_c, 16))
    width_stride = p.out_width_stride if width_stride is None else width_stride
    return ceil_div(out_c, UNPACK_C2) * width_stride * UNPACK_C2 * FP16_BYTES


def current_plan(shape: dict) -> Plan:
    batch = shape["batch"]
    in_c = shape["in_c"]
    in_h = shape["in_h"]
    in_w = shape["in_w"]
    out_c = shape["out_c"]
    kh = shape["kh"]
    kw = shape["kw"]
    groups = shape["groups"]
    stride = shape.get("stride", 1)
    p = conv_params(1, in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    spatial = kh != 1 or kw != 1
    grouped_spatial = is_grouped_spatial(in_c, out_c, kh, kw, groups)
    grouped_serial = spatial and groups > 1 and not p.depthwise
    spatial_output_bytes = output_bytes(p)
    spatial_weight_banks = mesa_weight_banks(kw, kh, in_c, out_c, False) if spatial else 0
    spatial_im2col = spatial and groups == 1 and not p.depthwise and (spatial_weight_banks > RK_CBUF_BANKS // 3 or spatial_output_bytes > OUTPUT_BYTES)
    spatial_oc_serial = spatial and groups == 1 and not p.depthwise and out_c > UNPACK_C2 and (in_c % 16 != 0 or in_c >= 16)
    dw_tile_h = depthwise_tile_h(out_c, p.out_h, in_w, kh, kw, stride) if p.depthwise and spatial else p.out_h
    depthwise_spatial_tiled = p.depthwise and spatial and (p.out_h > dw_tile_h or out_c > p.align_c)

    if spatial_im2col:
        flat_c = in_c * kh * kw
        oc_tile = pointwise_oc_tile_c(flat_c)
        row_tile = pointwise_oc_tile_h(flat_c, p.out_h, p.out_w, oc_tile, stride)
        tasks = batch * ceil_div(p.out_h, row_tile) * ceil_div(out_c, oc_tile)
        return Plan("current_spatial_im2col_oc_h", "flat_1x1", tasks, f"flat_c={flat_c}, oc_tile={oc_tile}, row_tile={row_tile}, weight_banks={spatial_weight_banks}")
    if grouped_serial:
        input_per_group = in_c // groups
        out_per_group = out_c // groups
        gp = conv_params(1, input_per_group, in_h, in_w, out_per_group, kh, kw, 1, stride)
        row_tile = mesa_output_tile_h(in_w, gp.out_h, input_per_group, out_per_group, kh, kw, stride, False)
        tasks = batch * groups * ceil_div(gp.out_h, row_tile)
        return Plan("current_grouped_serial", "nc1hwc2", tasks, f"groups={groups}, per_group={input_per_group}->{out_per_group}, row_tile={row_tile}")
    if spatial_oc_serial:
        tasks = batch * ceil_div(p.out_h, 32) * ceil_div(out_c, UNPACK_C2)
        return Plan("current_spatial_oc_serial", "nc1hwc2", tasks, f"oc_tile={UNPACK_C2}, row_tile=32, weight_banks={spatial_weight_banks}")
    if depthwise_spatial_tiled:
        channel_tile = min(32, out_c)
        row_tile = dw_tile_h
        tasks = batch * ceil_div(p.out_h, row_tile) * ceil_div(out_c, channel_tile)
        return Plan("current_depthwise_channel_h", "nc1hwc2", tasks, f"channel_tile={channel_tile}, row_tile={row_tile}")
    if needs_pointwise_oc_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
        oc_tile = pointwise_oc_tile_c(in_c)
        row_tile = pointwise_oc_tile_h(in_c, p.out_h, p.out_w, oc_tile, stride)
        tasks = batch * ceil_div(p.out_h, row_tile) * ceil_div(out_c, oc_tile)
        return Plan("current_pointwise_oc_h", "flat_1x1", tasks, f"oc_tile={oc_tile}, row_tile={row_tile}")
    if needs_pointwise_tile_schedule(in_c, out_c, in_h, in_w, kh, kw, groups):
        row_tile = pointwise_tile_h(in_c, out_c, p.out_h, p.out_w)
        return Plan("current_pointwise_weight_reuse_h", "flat_1x1", batch * ceil_div(p.out_h, row_tile), f"row_tile={row_tile}")
    if not spatial:
        out_c1 = ceil_div(p.align_out_c, UNPACK_C2)
        output_atoms_fit = OUTPUT_BYTES // (FP16_BYTES * out_c1 * UNPACK_C2)
        needs_output_tile = p.out_width_stride > output_atoms_fit
        tile_h = max(1, RK_MAX_CONV_FLAT_STRIDE // p.out_w) if ((p.use_nhwc or p.mesa_aligned_small) and align_up(p.out_h * p.out_w, 4) > RK_MAX_CONV_FLAT_STRIDE) else p.out_h
        tile_h = min(tile_h, max(1, output_atoms_fit // p.out_w)) if needs_output_tile else tile_h
        return Plan("current_default_flat_h" if tile_h < p.out_h else "current_default_flat", "flat_1x1", batch * ceil_div(p.out_h, tile_h), f"row_tile={tile_h}")
    fmt = "grouped_spatial" if grouped_spatial else "nc1hwc2"
    return Plan("current_default_spatial", fmt, batch, f"weight_banks={spatial_weight_banks}")


def precision_risk(shape: dict) -> bool:
    if "known_reason" in shape and "precision" in shape["known_reason"]:
        return True
    return shape["groups"] == 1 and shape["kh"] == 1 and shape["kw"] == 1 and shape["in_c"] >= 96 and shape["out_c"] <= max(128, shape["in_c"] // 2)


def direct_full_weight_fits(shape: dict) -> bool:
    p = conv_params(1, shape["in_c"], shape["in_h"], shape["in_w"], shape["out_c"], shape["kh"], shape["kw"], shape["groups"], shape.get("stride", 1))
    if shape["groups"] > 1 and not p.depthwise:
        weight_banks = mesa_weight_banks(shape["kw"], shape["kh"], shape["in_c"] // shape["groups"], shape["out_c"] // shape["groups"], False)
    else:
        weight_banks = mesa_weight_banks(shape["kw"], shape["kh"], shape["in_c"], shape["out_c"], p.depthwise)
    return weight_banks + 1 < RK_CBUF_BANKS and output_bytes(p) <= OUTPUT_BYTES


def h_split_full_weight_fits(shape: dict) -> bool:
    p = conv_params(1, shape["in_c"], shape["in_h"], shape["in_w"], shape["out_c"], shape["kh"], shape["kw"], shape["groups"], shape.get("stride", 1))
    if shape["groups"] > 1 and not p.depthwise:
        input_c = shape["in_c"] // shape["groups"]
        output_c = shape["out_c"] // shape["groups"]
    else:
        input_c = shape["in_c"]
        output_c = shape["out_c"]
    weight_banks = mesa_weight_banks(shape["kw"], shape["kh"], input_c, output_c, p.depthwise)
    if weight_banks >= RK_CBUF_BANKS:
        return False
    row_tile = mesa_output_tile_h(shape["in_w"], p.out_h, input_c, output_c, shape["kh"], shape["kw"], shape.get("stride", 1), p.depthwise)
    return row_tile >= 1


def onnc_weight_tile_fits(shape: dict) -> bool:
    p = conv_params(1, shape["in_c"], shape["in_h"], shape["in_w"], shape["out_c"], shape["kh"], shape["kw"], shape["groups"], shape.get("stride", 1))
    if p.depthwise or shape["groups"] > 1:
        return h_split_full_weight_fits(shape)
    bytes_per_output_kernel = shape["kh"] * shape["kw"] * shape["in_c"] * FP16_BYTES
    return bytes_per_output_kernel <= CBUF_BANK_SIZE * (RK_CBUF_BANKS - 1)


def ic_split_sum_fits(shape: dict) -> bool:
    if shape["groups"] != 1 or shape["kh"] != 1 or shape["kw"] != 1:
        return False
    ic_tile = min(128, shape["in_c"])
    sub = dict(shape)
    sub["in_c"] = ic_tile
    sub["weight_in_c"] = ic_tile
    return h_split_full_weight_fits(sub)


def safe_mechanism(shape: dict, cur: Plan) -> str:
    p = conv_params(1, shape["in_c"], shape["in_h"], shape["in_w"], shape["out_c"], shape["kh"], shape["kw"], shape["groups"], shape.get("stride", 1))
    spatial = shape["kh"] != 1 or shape["kw"] != 1
    if shape["groups"] > 1 and not p.depthwise:
        return "group_serial_h"
    if p.depthwise and spatial:
        return "depthwise_channel_h"
    if spatial and shape["groups"] == 1 and not p.depthwise:
        if cur.strategy == "current_spatial_im2col_oc_h":
            return "dense_spatial_im2col_oc_h"
        return "direct_h_split"
    if precision_risk(shape):
        return "pointwise_oc_h_precision_fallback"
    if cur.strategy.startswith("current_pointwise"):
        return "pointwise_oc_h_layout_fallback"
    return "direct_h_split"


def future_mechanism(shape: dict, cur: Plan) -> str:
    p = conv_params(1, shape["in_c"], shape["in_h"], shape["in_w"], shape["out_c"], shape["kh"], shape["kw"], shape["groups"], shape.get("stride", 1))
    spatial = shape["kh"] != 1 or shape["kw"] != 1
    if shape["groups"] > 1 and not p.depthwise:
        return "group_serial_h"
    if p.depthwise and spatial:
        return "depthwise_channel_h"
    if precision_risk(shape) and ic_split_sum_fits(shape):
        return "ic_split_sum_h"
    if spatial and shape["groups"] == 1 and not p.depthwise and not h_split_full_weight_fits(shape):
        return "hw_weight_tile_h"
    return "direct_h_split"


def safe_primitives(shape: dict, cur: Plan) -> tuple[str, ...]:
    # The executor primitive is direct convolution.  Some shapes first build an
    # im2col feature tile, then feed that tile to the same direct 1x1 executor.
    if cur.strategy == "current_spatial_im2col_oc_h":
        return ("im2col_materialize", "direct_conv_tile")
    return ("direct_conv_tile",)


def future_primitives(shape: dict, cur: Plan) -> tuple[str, ...]:
    future = future_mechanism(shape, cur)
    if future == "hw_weight_tile_h":
        return ("hw_weight_stream_tile",)
    if future == "ic_split_sum_h":
        return ("direct_conv_tile", "npu_sum_reduce")
    return ("direct_conv_tile",)


def strategy_coverage(shape: dict) -> dict[str, bool]:
    p = conv_params(1, shape["in_c"], shape["in_h"], shape["in_w"], shape["out_c"], shape["kh"], shape["kw"], shape["groups"], shape.get("stride", 1))
    spatial = shape["kh"] != 1 or shape["kw"] != 1
    dense_spatial = spatial and shape["groups"] == 1 and not p.depthwise
    return {
        "direct_full": direct_full_weight_fits(shape),
        "h_split_full_weight": h_split_full_weight_fits(shape),
        "current_fallbacks": True,
        "im2col_oc_h": dense_spatial,
        "oc_h_tile": shape["groups"] == 1 and not p.depthwise,
        "depthwise_channel_h": p.depthwise and spatial,
        "group_serial_h": shape["groups"] > 1 and not p.depthwise,
        "onnc_hw_weight_tile": onnc_weight_tile_fits(shape),
        "onnc_ic_split_sum": ic_split_sum_fits(shape),
        "nvdla_sw_no_ic_split": (h_split_full_weight_fits(shape) or onnc_weight_tile_fits(shape)) and not precision_risk(shape),
    }


def dict_from_call(node: ast.Call) -> dict:
    if not isinstance(node.func, ast.Name) or node.func.id != "dict":
        raise ValueError("shape entry is not dict(...)")
    out = {}
    for keyword in node.keywords:
        out[keyword.arg] = ast.literal_eval(keyword.value)
    return out


def load_shapes(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "shapes" for t in stmt.targets):
                return [dict_from_call(elt) for elt in stmt.value.elts]
    raise RuntimeError("could not find shapes assignment")


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def write_report(shapes: list[dict], rows: list[dict]) -> None:
    strategy_counts = Counter(row["current"].strategy for row in rows)
    format_counts = Counter(row["current"].output_format for row in rows)
    safe_counts = Counter(row["safe_mechanism"] for row in rows)
    future_counts = Counter(row["future_mechanism"] for row in rows)
    safe_primitive_counts = Counter()
    future_primitive_counts = Counter()
    for row in rows:
        safe_primitive_counts.update(row["safe_primitives"])
        future_primitive_counts.update(row["future_primitives"])
    coverage_counts = Counter()
    for row in rows:
        for name, ok in row["coverage"].items():
            if ok:
                coverage_counts[name] += 1

    precision_rows = [row for row in rows if row["precision_risk"]]
    im2col_rows = [row for row in rows if row["current"].strategy == "current_spatial_im2col_oc_h"]
    dense_weight_pressure_rows = [row for row in rows if row["spatial_weight_banks"] > RK_CBUF_BANKS // 3]
    grouped_spatial_rows = [row for row in rows if is_grouped_spatial(row["shape"]["in_c"], row["shape"]["out_c"], row["shape"]["kh"], row["shape"]["kw"], row["shape"]["groups"])]

    lines = []
    lines.append("# Conv Tiling Experiment Result And Cleanup Plan")
    lines.append("")
    lines.append("Generated by `python conv_tile_expt.py`. This is an analytical tiling experiment over every active shape in `examples/kernel_6_18/conv_new_clean.py`; it does not import that file or submit NPU jobs.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Shapes analyzed: {len(shapes)}")
    lines.append(f"- Current `run_conv2d` strategies: {', '.join(f'{k}={v}' for k, v in sorted(strategy_counts.items()))}")
    lines.append(f"- Current live output formats by shape: {', '.join(f'{k}={v}' for k, v in sorted(format_counts.items()))}")
    lines.append(f"- Precision-risk pointwise shapes: {len(precision_rows)}")
    lines.append(f"- Current dense spatial im2col fallback shapes: {len(im2col_rows)}")
    lines.append(f"- Spatial shapes with dense full-weight pressure over 4 CBUF banks: {len(dense_weight_pressure_rows)}")
    lines.append(f"- Grouped spatial shapes: {len(grouped_spatial_rows)}")
    lines.append("")
    lines.append("## `run_conv2d` Problem")
    lines.append("")
    lines.append("`run_conv2d` is hard to clean up because it mixes three concerns in every branch: tiling decision, input/weight materialization, and output unpack/stitching. The active test list exercises six separate branch families: default flat/H, default spatial, dense spatial im2col+OC/H, spatial OC-serial, depthwise channel/H, grouped serial, plus the pointwise weight-reuse special case.")
    lines.append("")
    lines.append("The root cause is RK3588's 12-bank CBUF. NVDLA/OpenDLA reference software can often keep FP16 weights resident with 32 banks, but RK3588 needs extra fallback strategies for dense spatial weight pressure and FP16 `IC >> OC` pointwise precision. The current script solves this pragmatically with OC tiling and im2col, but the duplicated loop bodies make output layout and offset bugs likely.")
    lines.append("")
    lines.append("## Strategy Coverage")
    lines.append("")
    lines.append("| Strategy | Shapes covered | Notes |")
    lines.append("| --- | ---: | --- |")
    notes = {
        "direct_full": "No tiling; full input and full weights fit RK3588 CBUF/output buffer.",
        "h_split_full_weight": "Mesa/NVDLA-style row tiling with full weights resident.",
        "current_fallbacks": "The current Python script's special cases.",
        "im2col_oc_h": "Spatial dense fallback lowered to 1x1 plus OC/H tiles.",
        "oc_h_tile": "Output-channel plus row tiling; avoids CACC cross-submit accumulation.",
        "depthwise_channel_h": "Depthwise channel plus row tiling.",
        "group_serial_h": "Grouped conv split into per-group direct convs.",
        "onnc_hw_weight_tile": "ONNC/NVDLA-style hardware weight/KG tiling candidate; requires RK proof.",
        "onnc_ic_split_sum": "Input-channel split plus NPU-side sum candidate for FP16 precision-risk pointwise convs.",
        "nvdla_sw_no_ic_split": "NVDLA SW-like coverage excluding shapes that need IC split for FP16 precision.",
    }
    for name in notes:
        lines.append(f"| `{name}` | {coverage_counts[name]} | {notes[name]} |")
    lines.append("")
    lines.append("## Smallest Strategy Sets")
    lines.append("")
    lines.append("This is the cleanup-relevant result: do not pick a different best strategy for each shape. Pick the smallest set of mechanism families that covers the active tests, then implement those with shared tile/executor code.")
    lines.append("")
    lines.append("### Safe Now")
    lines.append("")
    lines.append("This set keeps only mechanisms that `conv_new_clean.py` already exercises, so it is the safest cleanup target before adding new hardware behavior.")
    lines.append("")
    lines.append("| Mechanism family | Shapes | Keep/merge |")
    lines.append("| --- | ---: | --- |")
    safe_notes = {
        "direct_h_split": "One direct-conv H tiler for default flat and spatial cases; no separate pointwise weight-reuse chain.",
        "pointwise_oc_h_precision_fallback": "Keep OC/H tiling only for FP16 IC>>OC numerical-risk pointwise cases until IC-split+DPU-sum is proven.",
        "pointwise_oc_h_layout_fallback": "Temporary pointwise OC/H path for current layout quirks; should collapse into direct_h_split or precision fallback.",
        "dense_spatial_im2col_oc_h": "Keep im2col+OC/H only for dense spatial cases that current direct scheduling cannot safely handle.",
        "depthwise_channel_h": "One depthwise channel/H tiler.",
        "group_serial_h": "One per-group direct/H tiler.",
    }
    for name, note in safe_notes.items():
        if safe_counts[name]:
            lines.append(f"| `{name}` | {safe_counts[name]} | {note} |")
    lines.append("")
    lines.append("Safe-now minimal set size: 6 mechanism families. This is already smaller than the current code because `current_default_flat`, `current_default_flat_h`, `current_default_spatial`, `current_pointwise_weight_reuse_h`, and non-overflow `current_spatial_oc_serial` can all be represented as `direct_h_split` tiles executed by one shared loop.")
    lines.append("")
    lines.append("### Cleaner Future")
    lines.append("")
    lines.append("This set is smaller, but it depends on proving two new hardware paths on RK3588: hardware partial-weight/KG tiling and NPU-side sum for IC-split.")
    lines.append("")
    lines.append("| Mechanism family | Shapes | Requirement |")
    lines.append("| --- | ---: | --- |")
    future_notes = {
        "direct_h_split": "Shared direct row tiler for pointwise, normal spatial, small grouped/depthwise-independent cases.",
        "hw_weight_tile_h": "Prove CBUF weight reuse / split-K behavior and CACC persistence for dense spatial weight overflow.",
        "ic_split_sum_h": "Prove DPU/NPU sum path for partial input-channel convolution outputs; no CPU/GPU accumulation.",
        "depthwise_channel_h": "Still useful because depthwise has channel layout constraints and tiny per-channel weights.",
        "group_serial_h": "Still useful because grouped conv is naturally multiple independent convs.",
    }
    for name, note in future_notes.items():
        if future_counts[name]:
            lines.append(f"| `{name}` | {future_counts[name]} | {note} |")
    lines.append("")
    lines.append("Cleaner-future minimal set size: 5 mechanism families. If depthwise channel tiling can be expressed as the same generic tile executor with only a channel-slice dimension, the executor code can be shared even though the strategy family remains separate.")
    lines.append("")
    lines.append("## Can It Be Fewer Than 4?")
    lines.append("")
    lines.append("Yes, if we count implementation primitives instead of semantic strategy families. The six safe-now families above collapse to two executor primitives because group, depthwise, OC, channel, and H tiling are all just different slices of a direct conv tile.")
    lines.append("")
    lines.append("### Safe-Now Primitive Set")
    lines.append("")
    lines.append("| Primitive | Shapes using it | What it covers |")
    lines.append("| --- | ---: | --- |")
    lines.append(f"| `direct_conv_tile` | {safe_primitive_counts['direct_conv_tile']} | Direct conv executor with optional H, OC, channel, or group slicing. Covers pointwise, direct spatial, grouped serial, depthwise channel tiles, and pointwise OC fallbacks. |")
    lines.append(f"| `im2col_materialize` | {safe_primitive_counts['im2col_materialize']} | Prepares dense spatial overflow tiles as 1x1 inputs before `direct_conv_tile`. |")
    lines.append("")
    lines.append("Safe-now lower-level set size: 2 primitives. This is possible without new hardware behavior, but it is honest only if `direct_conv_tile` accepts tile descriptors with independent `n`, `group`, `channel`, `oc`, and `h` slices. Then depthwise/grouped/OC/H are descriptor dimensions, not separate loop bodies.")
    lines.append("")
    lines.append("### Cleaner-Future Primitive Set")
    lines.append("")
    lines.append("| Primitive | Shapes using it | Requirement |")
    lines.append("| --- | ---: | --- |")
    lines.append(f"| `direct_conv_tile` | {future_primitive_counts['direct_conv_tile']} | Same shared executor for all full-weight direct tiles. |")
    lines.append(f"| `hw_weight_stream_tile` | {future_primitive_counts['hw_weight_stream_tile']} | Prove RK3588 partial weight/KG streaming and accumulator behavior. |")
    lines.append(f"| `npu_sum_reduce` | {future_primitive_counts['npu_sum_reduce']} | Prove NPU/DPU sum for IC-split partial outputs. |")
    lines.append("")
    lines.append("Cleaner-future lower-level set size: 3 primitives. It removes im2col materialization, but adds two real hardware primitives. This is still fewer than four, but it is less safe until those hardware experiments pass.")
    lines.append("")
    lines.append("### Lower Bound")
    lines.append("")
    lines.append("One primitive is not realistic for a clean implementation. Either safe-now needs `im2col_materialize` for dense spatial overflow, or cleaner-future needs `hw_weight_stream_tile` plus `npu_sum_reduce` for dense spatial overflow and FP16 IC-split precision. Calling all of those `direct_conv_tile` would hide real dataflow differences and make the code less clear, not cleaner.")
    lines.append("")
    lines.append("## Output Format Finding")
    lines.append("")
    lines.append("`conv_new_clean.py` has three unpackers, but for the active shape list only two are live in the current dispatch: `flat_1x1` and `nc1hwc2`. `grouped_spatial` is still present in the default path, but grouped spatial shapes are intercepted by `current_grouped_serial` before that path, so it is dead for these tests.")
    lines.append("")
    lines.append("A single output format is technically possible only by lowering every conv to 1x1/im2col-style flat output, including grouped and depthwise convs. That would simplify unpacking but is not the best cleanup: it explodes tasks and input materialization for depthwise/grouped/spatial layers and gives up direct-conv hardware scheduling. The practical cleanup target is two output formats: keep `flat_1x1` for pointwise/im2col and `nc1hwc2` for direct spatial/depthwise/group-serial. Delete or quarantine `grouped_spatial` only after confirming no caller reaches it.")
    lines.append("")
    lines.append("## Strategies To Use")
    lines.append("")
    lines.append("- Safe cleanup set: `direct_h_split`, `pointwise_oc_h_precision_fallback`, `dense_spatial_im2col_oc_h`, `depthwise_channel_h`, `group_serial_h`, plus a temporary `pointwise_oc_h_layout_fallback` bucket until those layout cases are proven direct.")
    lines.append("- Future cleanup set: `direct_h_split`, `hw_weight_tile_h`, `ic_split_sum_h`, `depthwise_channel_h`, and `group_serial_h`.")
    lines.append("- If counting implementation primitives, less than four is possible: safe-now is 2 primitives (`direct_conv_tile`, `im2col_materialize`), cleaner-future is 3 primitives (`direct_conv_tile`, `hw_weight_stream_tile`, `npu_sum_reduce`).")
    lines.append("- If counting semantic strategy families, keep the six safe-now names because they explain why each tile exists. The implementation should still share one direct tile executor.")
    lines.append("- Do not use CPU/GPU fallback for IC-split accumulation. The sum path must be NPU/DPU if it replaces OC tiling.")
    lines.append("")
    lines.append("## Cleanup Plan For `run_conv2d`")
    lines.append("")
    lines.append("1. Extract a `ConvTile` record with `input_slice`, `weight_slice`, `output_slice`, `params`, `output_format`, and `full_data_bank/weight_reuse` flags.")
    lines.append("2. Replace the repeated loop nests with one executor: pack input, pack/select weights, build regs, submit, unpack into the destination slice.")
    lines.append("3. Implement only two safe-now primitives: `direct_conv_tile` and `im2col_materialize`. Keep the six safe-now family names as small planner cases that only emit descriptors for those primitives.")
    lines.append("4. Make group, depthwise, OC, channel, and H tiling descriptor dimensions. They should not have separate submit/unpack loops.")
    lines.append("5. Collapse output handling to `flat_1x1` and `nc1hwc2` for now. Move `grouped_spatial` behind an assertion or delete it after a direct test confirms grouped_serial covers all grouped spatial cases.")
    lines.append("6. Remove the pointwise weight-reuse chain at lines 941-967 after the generic H-tiler can place tile outputs with the same offsets; it is special-case control flow for two active shapes in this analysis.")
    lines.append("7. Add separate hardware experiments for `CNA_CBUF_CON0_WEIGHT_REUSE`/partial-weight tiling and for NPU-side sum. If they pass, replace dense spatial im2col with `hw_weight_tile_h` and replace pointwise OC precision fallback with `ic_split_sum_h`, reducing the strategy set from six to five families.")
    lines.append("")
    lines.append("## Per-Shape Classification")
    lines.append("")
    lines.append("| Shape | In | Out | k/g/s | Current | Tasks | Format | Safe family | Future family |")
    lines.append("| --- | --- | --- | --- | --- | ---: | --- | --- | --- |")
    for row in rows:
        shape = row["shape"]
        p = row["params"]
        cur = row["current"]
        in_shape = f"{shape['in_c']}x{shape['in_h']}x{shape['in_w']}"
        out_shape = f"{shape['out_c']}x{p.out_h}x{p.out_w}"
        kgs = f"{shape['kh']}x{shape['kw']}/g{shape['groups']}/s{shape.get('stride', 1)}"
        lines.append(f"| `{shape['name']}` | {in_shape} | {out_shape} | {kgs} | `{cur.strategy}` | {cur.tasks} | `{cur.output_format}` | `{row['safe_mechanism']}` | `{row['future_mechanism']}` |")
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    shapes = load_shapes(SOURCE)
    rows = []
    for shape in shapes:
        p = conv_params(1, shape["in_c"], shape["in_h"], shape["in_w"], shape["out_c"], shape["kh"], shape["kw"], shape["groups"], shape.get("stride", 1))
        cur = current_plan(shape)
        coverage = strategy_coverage(shape)
        rows.append({
            "shape": shape,
            "params": p,
            "current": cur,
            "coverage": coverage,
            "safe_mechanism": safe_mechanism(shape, cur),
            "future_mechanism": future_mechanism(shape, cur),
            "safe_primitives": safe_primitives(shape, cur),
            "future_primitives": future_primitives(shape, cur),
            "precision_risk": precision_risk(shape),
            "spatial_weight_banks": mesa_weight_banks(shape["kw"], shape["kh"], shape["in_c"], shape["out_c"], False) if (shape["kh"] != 1 or shape["kw"] != 1) else 0,
        })
    write_report(shapes, rows)
    print(f"wrote {REPORT} for {len(shapes)} shapes")


if __name__ == "__main__":
    main()
