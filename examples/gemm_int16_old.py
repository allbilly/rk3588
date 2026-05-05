import ctypes
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples"))
import gemm as raw_gemm  # noqa: E402


INT16_BYTES = 2
INT32_BYTES = 4
MIN_CHANNEL_TILE = 32
PRECISION_INT16 = 1
PRECISION_INT32 = 4


def _pack_input_row_major(m, n, k, a_matrix, in_pack, align_in):
    in_pack.reshape(m, align_in)[:, :k] = a_matrix[:, :k]


def _pack_input_c2_8(m, n, k, a_matrix, in_pack, align_in):
    in_pack[:] = a_matrix[:, :k].reshape(m, -1, 8).transpose(1, 0, 2).ravel()


def _pack_weight_tile_16x32(m, n, k, b_matrix, wt_pack, align_in, align_out):
    wt = np.zeros((align_out, align_in), dtype=np.int16)
    wt[:n, :k] = b_matrix.T[:n, :k]
    wt_pack[:] = wt.reshape(align_out // 16, 16, align_in // 32, 32).transpose(0, 2, 1, 3).ravel()


def _decode_output_linear(m, n, k, raw, align_out):
    row_start = np.arange(m) * align_out
    return raw[row_start[:, None] + np.arange(n)]


def _decode_output_c2_4(m, n, k, raw, align_out):
    return raw[: n // 4 * m * 4].reshape(n // 4, m, 4).transpose(1, 0, 2).reshape(m, n).copy()


def _get_input_packer(m, n, k, align_in):
    if raw_gemm._uses_c2_input(m, n, k):
        return _pack_input_c2_8
    return _pack_input_row_major


def _get_output_decoder(m, n, k, align_out):
    if raw_gemm._uses_c2_input(m, n, k):
        return _decode_output_c2_4
    return _decode_output_linear


def _emit(target, reg_addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr


def make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma):
    align_in, align_out, eff_k, _ = raw_gemm._gemm_layout(m, n, k)

    no_group_line_off = raw_gemm._rk_no_group_line_off(m, n, k)
    line_stride = raw_gemm._rk_line_stride(m, n, k, eff_k)
    surf_groups = m // 4
    surf_stride = 0
    if align_in >= 64 and raw_gemm._uses_c2_input(m, n, k):
        surf_stride = line_stride * (surf_groups - 1) + int(surf_groups == 0)

    input_bytes = m * align_in * INT16_BYTES
    data_bank = raw_gemm._rk_data_bank_count(input_bytes)
    data_entries = raw_gemm._rk_cbuf_data_entries(align_in)
    dst_surf_stride = align_out if no_group_line_off else 1
    feature_grains = raw_gemm._rk_feature_grains(m, align_in, eff_k)
    notch_val = raw_gemm._rk_notch_value(m, k, n, align_out, eff_k)

    conv_con1 = (PRECISION_INT16 << 7) | (PRECISION_INT16 << 4)
    if not no_group_line_off:
        conv_con1 |= 1 << 29

    r = raw_gemm.reg
    return [
        _emit(r.TARGET_DPU, r.S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        _emit(r.TARGET_CNA, r.CNA_CONV_CON1, conv_con1),
        _emit(r.TARGET_CNA, r.CNA_CONV_CON2, feature_grains << 4),
        _emit(r.TARGET_CNA, r.CNA_CONV_CON3, (1 << 3) | 1),
        _emit(r.TARGET_CNA, r.CNA_DATA_SIZE0, (1 << 16) | m),
        _emit(r.TARGET_CNA, r.CNA_DATA_SIZE1, ((align_in - 1) << 16) | align_in),
        _emit(r.TARGET_CNA, r.CNA_DATA_SIZE2, 1),
        _emit(r.TARGET_CNA, r.CNA_DATA_SIZE3, m),
        _emit(r.TARGET_CNA, r.CNA_WEIGHT_SIZE0, align_in * INT16_BYTES * align_out),
        _emit(r.TARGET_CNA, r.CNA_WEIGHT_SIZE1, align_in * INT16_BYTES),
        _emit(r.TARGET_CNA, r.CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | align_out),
        _emit(r.TARGET_CNA, r.CNA_CBUF_CON0, ((raw_gemm.RK_CBUF_BANKS - data_bank) << 4) | data_bank),
        _emit(r.TARGET_CNA, r.CNA_CBUF_CON1, data_entries),
        _emit(r.TARGET_CNA, r.CNA_CVT_CON0, 1),  # bypass CNA input conversion
        _emit(r.TARGET_CNA, r.CNA_CVT_CON1, 1 << 16),
        _emit(r.TARGET_CNA, r.CNA_CVT_CON2, 1 << 16),
        _emit(r.TARGET_CNA, r.CNA_CVT_CON3, 1 << 16),
        _emit(r.TARGET_CNA, r.CNA_CVT_CON4, 1 << 16),
        _emit(r.TARGET_CNA, r.CNA_FEATURE_DATA_ADDR, in_dma & 0xFFFFFFFF),
        _emit(r.TARGET_CNA, r.CNA_DMA_CON0, (15 << 16) | 15),
        _emit(r.TARGET_CNA, r.CNA_DMA_CON1, line_stride),
        _emit(r.TARGET_CNA, r.CNA_DMA_CON2, surf_stride),
        _emit(r.TARGET_CNA, r.CNA_FC_DATA_SIZE0, (1 << 16) | m),
        _emit(r.TARGET_CNA, r.CNA_FC_DATA_SIZE1, align_in),
        _emit(r.TARGET_CNA, r.CNA_DCOMP_ADDR0, wt_dma & 0xFFFFFFFF),
        _emit(r.TARGET_CORE, r.CORE_MISC_CFG, (PRECISION_INT16 << 8) | 1),
        _emit(r.TARGET_CORE, r.CORE_DATAOUT_SIZE_0, ((m - 1) << 16) | 0),
        _emit(r.TARGET_CORE, r.CORE_DATAOUT_SIZE_1, align_out - 1),
        _emit(r.TARGET_CORE, r.CORE_RESERVED_3030, 0),
        _emit(r.TARGET_DPU, r.FEATURE_MODE_CFG, (15 << 5) | (2 << 1)),
        _emit(
            r.TARGET_DPU,
            r.DATA_FORMAT,
            (PRECISION_INT32 << 29) | (PRECISION_INT16 << 26) | PRECISION_INT16,
        ),
        _emit(r.TARGET_DPU, r.DST_BASE_ADDR, out_dma & 0xFFFFFFFF),
        _emit(r.TARGET_DPU, r.DST_SURF_STRIDE, dst_surf_stride << 4),
        _emit(r.TARGET_DPU, r.DATA_CUBE_WIDTH, 0),
        _emit(r.TARGET_DPU, r.DATA_CUBE_HEIGHT, m - 1),
        _emit(r.TARGET_DPU, r.DATA_CUBE_NOTCH, (notch_val << 16) | notch_val),
        _emit(r.TARGET_DPU, r.DATA_CUBE_CHANNEL, ((align_out - 1) << 16) | (align_out - 1)),
        _emit(r.TARGET_DPU, r.BS_CFG, 0x00000053),
        _emit(r.TARGET_DPU, r.BS_OW_CFG, 0x0000036E),
        _emit(r.TARGET_DPU, r.WDMA_SIZE_0, align_out - 1),
        _emit(r.TARGET_DPU, r.WDMA_SIZE_1, ((m - 1) << 16) | 0),
        _emit(r.TARGET_DPU, r.BN_CFG, 0x00000053),
        _emit(r.TARGET_DPU, r.EW_CFG, 0x00000383),
        _emit(r.TARGET_DPU, r.OUT_CVT_SCALE, 1),
        _emit(r.TARGET_DPU, r.SURFACE_ADD, (dst_surf_stride * INT32_BYTES) << 4),
        _emit(r.TARGET_PC, r.OPERATION_ENABLE, 0x0000000D),
    ]


def run_gemm(m, n, k, a_matrix, b_matrix):
    align_in, align_out, _, pad_k = raw_gemm._gemm_layout(m, n, k)

    a_matrix = np.asarray(a_matrix, dtype=np.int16, order="C").reshape(m, k)
    b_matrix = np.asarray(b_matrix, dtype=np.int16, order="C").reshape(k, n)
    in_pack = np.zeros(align_in * m, dtype=np.int16)
    wt_pack = np.zeros(align_in * align_out, dtype=np.int16)

    pack_input = _get_input_packer(m, n, k, align_in)
    if pad_k:
        pack_input = _pack_input_row_major
    pack_input(m, n, k, a_matrix, in_pack, align_in)
    _pack_weight_tile_16x32(m, n, k, b_matrix, wt_pack, align_in, align_out)

    ct_inputs = (ctypes.c_int16 * len(in_pack)).from_buffer(raw_gemm.input_map)
    ct_weights = (ctypes.c_int16 * len(wt_pack)).from_buffer(raw_gemm.weight_map)
    ct_inputs[:] = in_pack.tolist()
    ct_weights[:] = wt_pack.tolist()

    out_nbytes = max(256, ((m - 1) * align_out + n) * INT32_BYTES)
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(raw_gemm.output_map)), 0, out_nbytes)

    npu_regs = make_gemm_regs(
        m,
        n,
        k,
        raw_gemm.input_mem_create.dma_addr,
        raw_gemm.weight_mem_create.dma_addr,
        raw_gemm.output_mem_create.dma_addr,
    )

    if "--dry" in sys.argv:
        print(f"\n=== GEMM INT16 {m}x{n}x{k} DRY RUN ===")
        target_names = {0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU", 0x2001: "RDMA", 0x0081: "PC"}
        for i, v in enumerate(npu_regs):
            tgt = (v >> 48) & 0xFFFF
            ra = v & 0xFFFF
            val = (v >> 16) & 0xFFFFFFFF
            print(f"  [{i:3d}] {target_names.get(tgt, f'0x{tgt:04x}')}[0x{ra:04x}] = 0x{val:08x}")
        return None

    for i, reg_value in enumerate(npu_regs):
        raw_gemm.regcmd[i] = reg_value
    for i in range(len(npu_regs), 64):
        raw_gemm.regcmd[i] = 0

    raw_gemm._write_task_descriptor(len(npu_regs))
    raw_gemm.reset_npu(raw_gemm.fd)
    raw_gemm.submit(raw_gemm.tasks_mem_create.obj_addr)

    raw = np.frombuffer(raw_gemm.output_map, dtype=np.int32, count=out_nbytes // INT32_BYTES).copy()
    return _get_output_decoder(m, n, k, align_out)(m, n, k, raw, align_out).astype(np.int32, copy=False)


if __name__ == "__main__":
    np.random.seed(42)
    cases = [
        (2, 2, 1),
        (32, 32, 32),
        (64, 64, 64),
        (64, 99, 64),
        (128, 128, 128),
    ]

    ok_all = True
    for m, n, k in cases:
        a = np.random.randint(-16, 16, size=(m, k), dtype=np.int16)
        b = np.random.randint(-16, 16, size=(k, n), dtype=np.int16)
        print(f"\n{m}x{n}x{k}:")
        result = run_gemm(m, n, k, a, b)
        if result is None:
            continue
        expected = a.astype(np.int32) @ b.astype(np.int32)
        ok = np.array_equal(result, expected)
        md = int(np.max(np.abs(result - expected))) if result.size else 0
        print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md})")
        ok_all &= ok

    if not ok_all:
        raise SystemExit(1)
