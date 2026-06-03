
C576_H19_OC12_EXACT12_SHAPE = "b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid"
C576_H19_OC12_EXACT12_AMOUNTS = (108, 108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
C576_H19_OC12_EXACT12_MASKS = (0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
C576_H19_OC12_EXACT12_PC_AMOUNTS = (0, 0x1000e, 0, 0x2000e, 0, 0x2000e, 0, 0x2000e, 0, 0x2000e, 0)
C576_H19_OC12_EXACT12_ROLES = ("setup_body", "k_half_body0", "k_tile_body0", "aux0", "k_tile_body1", "aux1", "k_tile_body2", "aux2", "k_tile_body3", "aux3", "k_tile_body4", "aux4")
C576_H19_OC12_K_TILE_SPLITS = ((0, 4), (4, 3), (7, 2), (9, 1), (10, 2))  # 4+3+2+1+2=12
C576_H19_OC12_K_TILE_WEIGHT_SIZES = (0x1200, 0x0d80, 0x0900, 0x0480, 0x0900)  # 4*576*2, 3*576*2, 2*576*2, 1*576*2, 2*576*2
C576_H19_OC12_SETUP_WEIGHT_SIZE = 0x3600  # 12 * 576 * 2 = 13824
C576_H19_OC12_K_HALF_WEIGHT_SIZE = 0x3600  # full weight (single k_half covers all 12 OC)
C576_H19_OC12_AUX_DMA_OFFSET = 0x3600
C576_H19_OC12_CBUF0 = 0x0b1
C576_H19_OC12_DATA_SIZE1 = 0x8f0240  # (576/4-1)<<16 | 576, educated guess from c256_h2_oc64 pattern
C576_H19_OC12_DMA_CON2 = 0x0ffffffc
C576_H19_OC12_CVT_CON0 = 0x000b
C576_H19_OC12_CONV2_LOW = 0x30


def _c576_h19_oc12_exact12_task_regs(s, in_dma, wt_dma, out_dma):
    """Emit the 12-task 108/108/104/26... closure for c576_h19_oc12 (Attack H).

    Modeled on c256_h2_oc64 EXACT11 path. k_half row uses 108q (with explicit
    4-qword prelude); k_tile rows use 104q; aux rows use 26q. Body field
    constants (CBUF0, DATA_SIZE1, DMA_CON2, CVT_CON0, CONV2_LOW) are educated
    guesses from the c256_h2_oc64 + c256_h28_pw_1x1 closure pattern; a fresh
    KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2 capture is needed to validate
    the actual register values for promotion.
    """
    if s["name"] != C576_H19_OC12_EXACT12_SHAPE:
        raise ValueError("c576_h19_oc12 exact12 rows are scoped only to the prefix-proven c576_h19_oc12 shape")

    def body_row(family, oc_start, oc_count, weight_size0, with_prelude=False):
        regs = _exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma, input_h=19, conv2_low=C576_H19_OC12_CONV2_LOW)
        patched = patch_regs(regs, {
            (reg.CNA, reg.CNA_DATA_SIZE1): C576_H19_OC12_DATA_SIZE1,
            (reg.CNA, reg.CNA_CBUF_CON0): C576_H19_OC12_CBUF0,
            (reg.CNA, reg.CNA_CVT_CON0): C576_H19_OC12_CVT_CON0,
            (reg.CNA, reg.CNA_DMA_CON2): C576_H19_OC12_DMA_CON2,
            (reg.CNA, reg.CNA_WEIGHT_SIZE0): weight_size0,
            (reg.CNA, reg.CNA_WEIGHT_SIZE1): weight_size0 >> 8,
            (reg.CNA, reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | oc_count,
            (reg.CORE, reg.CORE_DATAOUT_SIZE_1): oc_count - 1,
            (reg.DPU, reg.DATA_CUBE_CHANNEL): ((oc_count - 1) << 16) | (oc_count - 1),
            (reg.DPU, reg.WDMA_SIZE_0): oc_count - 1,
        })
        if with_prelude:
            return (E(reg.CNA, reg.CNA_CBUF_CON0, C576_H19_OC12_CBUF0),
                    E(reg.CNA, 0x1104, 0),
                    E(reg.CNA, 0x1100, 0),
                    E(reg.CNA, reg.CNA_CONV_CON1, 0x120)) + tuple(patched)
        return patched

    aux_dma = wt_dma + C576_H19_OC12_AUX_DMA_OFFSET
    rows = [
        body_row("setup", 0, 12, C576_H19_OC12_SETUP_WEIGHT_SIZE, with_prelude=False),
        body_row("k_half", 0, 12, C576_H19_OC12_K_HALF_WEIGHT_SIZE, with_prelude=True),
    ]
    for (oc_start, oc_count), weight_size0 in zip(C576_H19_OC12_K_TILE_SPLITS, C576_H19_OC12_K_TILE_WEIGHT_SIZES):
        rows.append(body_row("k_tile", oc_start, oc_count, weight_size0, with_prelude=False))
        rows.append(_exact11_aux_regs(s, out_dma, aux_dma))
    if tuple(len(row_regs) for row_regs in rows) != C576_H19_OC12_EXACT12_AMOUNTS:
        raise RuntimeError("c576_h19_oc12 exact12 row amounts changed")
    return rows


def _c576_h19_oc12_exact12_layout_check(s):
    """Return the layout (amounts, masks, pc_amounts, roles, offsets) for c576_h19_oc12."""
    if s["name"] != C576_H19_OC12_EXACT12_SHAPE:
        raise ValueError("c576_h19_oc12 exact12 dry-run is scoped only to the prefix-proven c576_h19_oc12 shape")
    amounts = C576_H19_OC12_EXACT12_AMOUNTS
    masks = C576_H19_OC12_EXACT12_MASKS
    pc_amounts = C576_H19_OC12_EXACT12_PC_AMOUNTS
    roles = C576_H19_OC12_EXACT12_ROLES
    offsets = []
    qoff = 0
    for idx, amount in enumerate(amounts):
        offsets.append(qoff)
        if idx + 1 == len(amounts):
            break
        if idx == 0:
            tail_qwords = 4
        elif roles[idx].startswith("aux"):
            tail_qwords = 6
        else:
            tail_qwords = 8
        qoff += _align_up(amount + tail_qwords, 8)
    return {
        "amounts": amounts,
        "masks": masks,
        "pc_amounts": pc_amounts,
        "roles": roles,
        "offsets": tuple(offsets),
    }


def write_c576_h19_oc12_exact12_tasks(task_map, regcmd_map, regcmd_mem, task_regs, layout):
    offsets = layout["offsets"]
    amounts = layout["amounts"]
    masks = layout["masks"]
    pc_amounts = layout["pc_amounts"]
    roles = layout["roles"]
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            role = roles[idx]
            if role.startswith("aux"):
                tail = (0, E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, 0), E(reg.VERSION, 0, 0), E(reg.PC, reg.OPERATION_ENABLE, 0x60), 0, 0)
            else:
                next_addr = (regcmd_mem.dma_addr + offsets[idx + 1] * 8) & 0xfffffff0
                tail = (E(reg.PC_REG, reg.PC_BASE_ADDRESS, next_addr),
                        E(reg.PC_REG, reg.PC_REGISTER_AMOUNTS, pc_amounts[idx]),
                        E(reg.VERSION, 0, 0),
                        E(reg.PC, reg.OPERATION_ENABLE, masks[idx]),
                        0, 0, 0, 0)
            for i, qword in enumerate(tail):
                regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 1
        tasks[idx].enable_mask = masks[idx]
        tasks[idx].int_mask = 0xc00 if tasks[idx].enable_mask == 0x60 else 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = len(regs)
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * 8


def dry_run_c576_h19_oc12_exact12(s):
    """Print c576_h19_oc12 exact12 metadata without DRM or submit."""
    if s["name"] != C576_H19_OC12_EXACT12_SHAPE:
        raise ValueError("c576_h19_oc12 exact12 dry-run is scoped only to the prefix-proven c576_h19_oc12 shape")
    layout = _c576_h19_oc12_exact12_layout_check(s)
    in_dma, wt_dma, out_dma = 0x10000000, 0x20000000, 0x30000000
    rows = _c576_h19_oc12_exact12_task_regs(s, in_dma, wt_dma, out_dma)
    out_c = []
    weight_size0 = []
    for idx, (role, regs_row) in enumerate(zip(layout["roles"], rows)):
        if role.startswith("aux"):
            out_c.append(None)
            weight_size0.append(None)
            continue
        values = {(qword >> 48, qword & 0xffff): (qword >> 16) & 0xffffffff for qword in regs_row}
        out_c.append((values[(reg.CORE, reg.CORE_DATAOUT_SIZE_1)] & 0xffff) + 1)
        weight_size0.append(values[(reg.CNA, reg.CNA_WEIGHT_SIZE0)])
    regcmd_bytes = (layout["offsets"][-1] + layout["amounts"][-1]) * 8
    print(f"dry_run=c576_h19_oc12_exact12 shape={s['name']} status=no_drm_no_submit")
    print("amounts=" + ",".join(str(v) for v in layout["amounts"]))
    print("masks=" + ",".join(hex(v) for v in layout["masks"]))
    print("offsets=" + ",".join(str(v) for v in layout["offsets"]))
    print("roles=" + ",".join(layout["roles"]))
    print("out_c=" + ",".join("none" if v is None else str(v) for v in out_c))
    print("weight_size0=" + ",".join("none" if v is None else hex(v) for v in weight_size0))
    print(f"regcmd_bytes={regcmd_bytes}")
    print("consts=cbuf0:0xb1,data_size1:0x8f0240,dma2:0x0ffffffc,conv2_low:0x30,weight_setup:0x3600,weight_k_half:0x3600")
    print("note=body_fields_are_educated_guesses_from_c256_h2_oc64_pattern; promotion_requires_fresh_GEM2_capture")


def run_c576_h19_oc12_exact12_shape(s):
    """Run the c576_h19_oc12 12-task materializer. Body fields are educated guesses."""
    if s["name"] != C576_H19_OC12_EXACT12_SHAPE:
        raise ValueError("c576_h19_oc12 exact12 run is scoped only to the prefix-proven c576_h19_oc12 shape")
    layout = _c576_h19_oc12_exact12_layout_check(s)
    p = _conv_params(s)
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    input_flat = pack_input(inp[0], p).view(np.uint16)
    weight_flat = _pack_pointwise_wide(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = _ceil_div(p["align_out_c"], UNPACK_C2) * p["out_width_stride"] * UNPACK_C2
    output_bytes = out_count * np.dtype(np.float16).itemsize
    regcmd_bytes = (layout["offsets"][-1] + layout["amounts"][-1]) * ctypes.sizeof(ctypes.c_uint64)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = mem_allocate(fd, regcmd_alloc_bytes(regcmd_bytes), RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = mem_allocate(fd, 4 * 1024 * 1024, RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = mem_allocate(fd, max(4 * 1024 * 1024, output_bytes), RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        task_regs = _c576_h19_oc12_exact12_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        write_c576_h19_oc12_exact12_tasks(task_map, regcmd_map, regcmd_mem, task_regs, layout)
        subcores = ((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))
        print(f"c576_h19_oc12_exact12_submit tasks={len(layout['amounts'])} submit_task_number=3 amounts=" + ";".join(str(v) for v in layout["amounts"]) +
              " masks=" + ";".join(hex(v) for v in layout["masks"]) + " subcores=(0,1),(0,1),(0,1),(0,0),(0,0)")
        if npu_submit(fd, task_mem.obj_addr, 3, core_mask=0, subcores=subcores) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        post_submit_reset(fd)
    finally:
        close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], UNPACK_C2).reshape(1, s["out_c"], p["out_h"], p["out_w"])
    expected = compute_expected_vectorized(inp, wt, s)
    max_diff = float(np.max(np.abs(got.astype(np.float32) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} guarded=c576_h19_oc12_exact12 tasks={len(layout['amounts'])} submit_tasks=3 {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        print("debug_c576_h19_oc12_oc=" + ";".join(
            f"{start}:{float(np.max(np.abs(got[:, start:start + 2].astype(np.float32) - expected[:, start:start + 2]))):.4f}"
            for start in range(0, s["out_c"], 2)))
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")
