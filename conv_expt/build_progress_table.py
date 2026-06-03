#!/usr/bin/env python3
"""Build per-family progress table for the 80 fenced shapes."""
import re
from pathlib import Path

DUMP_DIR = Path("/home/orangepi/npu/ops_rknn/dump")
SWEEP = Path("/tmp/opencode/conv_py_217_sweep_20260603_121116_summary.txt")
fenced = []
for line in SWEEP.read_text().splitlines():
    if line.startswith("FENCED "):
        fenced.append(line.split(" ", 1)[1])

# Build slug-to-capture map
all_prefixes = {}
for d in DUMP_DIR.iterdir():
    if not d.is_dir() or not d.name.startswith("prefix_"):
        continue
    name = d.name[len("prefix_"):]
    m = re.match(r"(.+?)_keep\d+_gem\d+", name)
    if not m:
        m = re.match(r"(.+?)_gem\d+", name)
        if not m:
            continue
    slug = m.group(1)
    is_gem1 = "_gem1" in d.name and "_gem2" not in d.name
    is_gem2 = "_gem2" in d.name
    all_prefixes.setdefault(slug, ([], []))[0 if is_gem1 else 1].append(d.name)

# All slugs in the dump for reference
print(f"Distinct prefix slugs in dump: {len(all_prefixes)}")
print("Sample slugs:")
for s in sorted(all_prefixes)[:30]:
    print(f"  {s}")
print()

def has_capture(shape):
    s_norm = shape.replace("conv2d_cc_b1_", "cc_b1_")
    candidates = set()
    # Try matching progressively shorter patterns
    m = re.match(r"cc_b1_c(\d+)_h(\d+)_w\d+_oc(\d+)_wic(\d+)_k(\d+)x(\d+)_g(\d+)", s_norm)
    if m:
        c, h, oc, wic = m.group(1), m.group(2), m.group(3), m.group(4)
        candidates.add(f"cc_c{c}_h{h}_oc{oc}")
        candidates.add(f"cc_c{c}_h{h}")
    m = re.match(r"b1_c(\d+)_h(\d+)_w\d+_oc(\d+)_wic(\d+)_k(\d+)x(\d+)_g(\d+)(_s1_pvalid)?", s_norm)
    if m:
        c, h, oc, wic = m.group(1), m.group(2), m.group(3), m.group(4)
        candidates.add(f"c{c}_h{h}_oc{oc}")
        candidates.add(f"c{c}_h{h}")
        # also try the long form like c512_h14_oc24_baseline
        candidates.add(f"c{c}_h{h}_oc{oc}_baseline")
    # Search slugs that contain the cX_hY part
    for slug in all_prefixes:
        c_match = re.match(r"c(\d+)_h(\d+)", slug)
        if c_match:
            c, h = c_match.group(1), c_match.group(2)
            if f"c{c}_h{h}" in s_norm:
                candidates.add(slug)
    g1, g2 = [], []
    for cand in candidates:
        if cand in all_prefixes:
            g1l, g2l = all_prefixes[cand]
            g1.extend(g1l)
            g2.extend(g2l)
    return sorted(set(g1)), sorted(set(g2))

def categorize(shape):
    s_norm = shape.replace("conv2d_cc_b1_", "b1_")
    if re.match(r"b1_c(\d+)_h(\d+)_w\2_oc(\d+)_wic1_k3x3_g(\d+)(_s1_pvalid)?$", s_norm):
        return "depthwise 3x3 (k3_g=in_c)"
    if re.match(r"b1_c(\d+)_h(\d+)_w\2_oc(\d+)_wic(\d+)_k1x1_g1(_s1_pvalid)?$", s_norm):
        return "pointwise 1x1 (k1_g1)"
    if re.match(r"b1_c(\d+)_h(\d+)_w\2_oc(\d+)_wic(\d+)_k3x3_g1_s1_pvalid$", s_norm):
        return "spatial 3x3 (k3_g1)"
    if re.match(r"b1_c(\d+)_h(\d+)_w\2_oc(\d+)_wic1_k5x5_g(\d+)_s1_pvalid$", s_norm):
        return "depthwise 5x5 (k5_g=in_c)"
    if re.match(r"b1_c(\d+)_h(\d+)_w\2_oc(\d+)_wic(\d+)_k5x5_g1_s1_pvalid$", s_norm):
        return "spatial 5x5 (k5_g1)"
    if re.match(r"b1_c(\d+)_h(\d+)_w\2_oc(\d+)_wic1_k7x7_g(\d+)(_s1_pvalid)?$", s_norm):
        return "depthwise 7x7 (k7_g=in_c)"
    return "OTHER"

families = {}
for shape in fenced:
    fam = categorize(shape)
    g1, g2 = has_capture(shape)
    families.setdefault(fam, []).append((shape, g1, g2))

total = len(fenced)
total_g1 = sum(1 for s in fenced if has_capture(s)[0])
total_g2 = sum(1 for s in fenced if has_capture(s)[1])
no_capture = total - total_g1
print(f"Total fenced: {total}")
print(f"With any capture (GEM1 or GEM2): {total_g1}/{total} ({total_g1*100/total:.0f}%)")
print(f"With GEM2 body capture: {total_g2}/{total} ({total_g2*100/total:.0f}%)")
print(f"With NO capture at all: {no_capture}/{total} ({no_capture*100/total:.0f}%)")
print()
print("=" * 100)
print(f"{'Family':<40} {'#':>3} {'GEM1':>5} {'GEM2':>5} {'%G1':>5}  {'Distinct capture slugs'}")
print("=" * 100)
for fam in sorted(families.keys()):
    items = families[fam]
    g1_count = sum(1 for _, g, _ in items if g)
    g2_count = sum(1 for _, _, g in items if g)
    pct = f"{g1_count*100/len(items):.0f}%"
    slugs = set()
    for s, g1l, g2l in items:
        for d in g1l + g2l:
            m = re.match(r"prefix_(.+?)_keep\d+_gem\d+", d)
            if m:
                slugs.add(m.group(1))
    print(f"{fam:<40} {len(items):>3} {g1_count:>5} {g2_count:>5} {pct:>5}  {', '.join(sorted(slugs)[:4])}{' ...' if len(slugs) > 4 else ''}")
print("=" * 100)
print()
print("DETAILED — per shape (sorted by family)")
print("=" * 100)
print(f"{'Shape':<62} {'G1':<3} {'G2':<3}  {'Capture slugs'}")
print("-" * 100)
for fam in sorted(families.keys()):
    print(f"--- {fam} ---")
    for shape, g1, g2 in sorted(families[fam]):
        g1m = "Y" if g1 else "-"
        g2m = "Y" if g2 else "-"
        slugs = set()
        for d in g1 + g2:
            m = re.match(r"prefix_(.+?)_keep\d+_gem\d+", d)
            if m:
                slugs.add(m.group(1))
        slug_str = ", ".join(sorted(slugs)) if slugs else ""
        print(f"  {shape:<60} {g1m:<3} {g2m:<3}  {slug_str}")
