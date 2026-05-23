#!/usr/bin/env python3
import argparse
from pathlib import Path


TARGETS = {
    0x0201: "CNA",
    0x0801: "CORE",
    0x1001: "DPU",
    0x0081: "PC",
    0x0101: "PCREG",
    0x0401: "PPU",
    0x4001: "PPU",
    0x8001: "PPU_RDMA",
    0x2001: "RDMA",
}

REG_NAMES = {
    (0x0041, 0x0000): "VERSION",
    (0x0081, 0x0008): "PC_OPERATION_ENABLE",
    (0x0101, 0x0010): "PC_BASE_ADDRESS",
    (0x0101, 0x0014): "PC_REGISTER_AMOUNTS",
    (0x0201, 0x100c): "CNA_CONV_CON1",
    (0x0201, 0x1010): "CNA_CONV_CON2",
    (0x0201, 0x1014): "CNA_CONV_CON3",
    (0x0201, 0x1020): "CNA_DATA_SIZE0",
    (0x0201, 0x1024): "CNA_DATA_SIZE1",
    (0x0201, 0x1028): "CNA_DATA_SIZE2",
    (0x0201, 0x102c): "CNA_DATA_SIZE3",
    (0x0201, 0x1030): "CNA_WEIGHT_SIZE0",
    (0x0201, 0x1034): "CNA_WEIGHT_SIZE1",
    (0x0201, 0x1038): "CNA_WEIGHT_SIZE2",
    (0x0201, 0x1040): "CNA_CBUF_CON0",
    (0x0201, 0x1044): "CNA_CBUF_CON1",
    (0x0201, 0x104c): "CNA_CVT_CON0",
    (0x0201, 0x1050): "CNA_CVT_CON1",
    (0x0201, 0x1054): "CNA_CVT_CON2",
    (0x0201, 0x1058): "CNA_CVT_CON3",
    (0x0201, 0x105c): "CNA_CVT_CON4",
    (0x0201, 0x1070): "CNA_FEATURE_DATA_ADDR",
    (0x0201, 0x1078): "CNA_DMA_CON0",
    (0x0201, 0x107c): "CNA_DMA_CON1",
    (0x0201, 0x1080): "CNA_DMA_CON2",
    (0x0201, 0x1084): "CNA_FC_DATA_SIZE0",
    (0x0201, 0x1088): "CNA_FC_DATA_SIZE1",
    (0x0201, 0x1100): "CNA_DCOMP_CTRL",
    (0x0201, 0x1104): "CNA_DCOMP_REGNUM",
    (0x0201, 0x1110): "CNA_DCOMP_ADDR0",
    (0x0201, 0x1180): "CNA_CVT_CON5",
    (0x0801, 0x3010): "CORE_MISC_CFG",
    (0x0801, 0x3014): "CORE_SIZE0",
    (0x0801, 0x3018): "CORE_SIZE1",
    (0x0801, 0x3030): "CORE_RESERVED_3030",
    (0x1001, 0x4004): "DPU_S_POINTER",
    (0x1001, 0x400c): "DPU_FEATURE_MODE_CFG",
    (0x1001, 0x4010): "DPU_DATA_FORMAT",
    (0x1001, 0x4020): "DPU_DST_BASE_ADDR",
    (0x1001, 0x4024): "DPU_DST_SURF_STRIDE",
    (0x1001, 0x4030): "DPU_DST_W",
    (0x1001, 0x4034): "DPU_DST_H",
    (0x1001, 0x4038): "DPU_DST_NOTCH",
    (0x1001, 0x403c): "DPU_DST_C",
    (0x1001, 0x4040): "DPU_BS_CFG",
    (0x1001, 0x4050): "DPU_BS_OW_CFG",
    (0x1001, 0x4058): "DPU_CHANNEL_END",
    (0x1001, 0x405c): "DPU_SIZE",
    (0x1001, 0x4060): "DPU_BN_CFG",
    (0x1001, 0x4070): "DPU_MISC",
    (0x1001, 0x4078): "DPU_EW_CVT_SCALE_VALUE",
    (0x1001, 0x4084): "DPU_OUT_CVT_SCALE",
    (0x1001, 0x40a8): "DPU_EW_OP_VALUE_6",
    (0x1001, 0x40ac): "DPU_EW_OP_VALUE_7",
    (0x1001, 0x40c0): "DPU_SURFACE_ADD",
    (0x1001, 0x40c4): "DPU_SURFACE_ADD_EXTRA",
    (0x2001, 0x5004): "RDMA_S_POINTER",
    (0x2001, 0x500c): "RDMA_DATA_CUBE_WIDTH",
    (0x2001, 0x5010): "RDMA_DATA_CUBE_HEIGHT",
    (0x2001, 0x5014): "RDMA_DATA_CUBE_CHANNEL",
    (0x2001, 0x5018): "RDMA_SRC_BASE_ADDR",
    (0x2001, 0x5034): "RDMA_ERDMA_CFG",
    (0x2001, 0x5038): "RDMA_EW_BASE_ADDR",
    (0x2001, 0x5044): "RDMA_FEATURE_MODE_CFG",
}

for reg_addr in range(0x1140, 0x1180, 4):
    REG_NAMES[(0x0201, reg_addr)] = f"CNA_DCOMP_AMOUNT{(reg_addr - 0x1140) // 4}"
for reg_addr in range(0x4100, 0x4130, 4):
    REG_NAMES[(0x1001, reg_addr)] = f"DPU_CLEAR_{reg_addr:04x}"

KEYS = [
    (0x0201, 0x100c, "CNA_CONV_CON1"),
    (0x0201, 0x1010, "CNA_CONV_CON2"),
    (0x0201, 0x1020, "CNA_DATA_SIZE0"),
    (0x0201, 0x1024, "CNA_DATA_SIZE1"),
    (0x0201, 0x1028, "CNA_DATA_SIZE2"),
    (0x0201, 0x102c, "CNA_DATA_SIZE3"),
    (0x0201, 0x1030, "CNA_WEIGHT_SIZE0"),
    (0x0201, 0x1034, "CNA_WEIGHT_SIZE1"),
    (0x0201, 0x1038, "CNA_WEIGHT_SIZE2"),
    (0x0201, 0x1040, "CNA_CBUF_CON0"),
    (0x0201, 0x1044, "CNA_CBUF_CON1"),
    (0x0201, 0x1070, "CNA_FEATURE_DATA_ADDR"),
    (0x0201, 0x107c, "CNA_DMA_CON1"),
    (0x0201, 0x1080, "CNA_DMA_CON2"),
    (0x0201, 0x1088, "CNA_FC_DATA_SIZE1"),
    (0x0201, 0x1100, "CNA_DCOMP_CTRL"),
    (0x0201, 0x1104, "CNA_DCOMP_REGNUM"),
    (0x0201, 0x1110, "CNA_DCOMP_ADDR0"),
    (0x0801, 0x3010, "CORE_MISC_CFG"),
    (0x0801, 0x3014, "CORE_SIZE0"),
    (0x0801, 0x3018, "CORE_SIZE1"),
    (0x1001, 0x4004, "DPU_S_POINTER"),
    (0x1001, 0x400c, "DPU_FEATURE_MODE_CFG"),
    (0x1001, 0x4010, "DPU_DATA_FORMAT"),
    (0x1001, 0x4020, "DPU_DST_BASE_ADDR"),
    (0x1001, 0x4024, "DPU_DST_SURF_STRIDE"),
    (0x1001, 0x4030, "DPU_DST_W"),
    (0x1001, 0x4034, "DPU_DST_H"),
    (0x1001, 0x403c, "DPU_DST_C"),
    (0x1001, 0x4058, "DPU_CHANNEL_END"),
    (0x1001, 0x405c, "DPU_SIZE"),
    (0x1001, 0x4070, "DPU_MISC"),
    (0x1001, 0x40c0, "DPU_SURFACE_ADD"),
]


KEY_NAMES = {(target, reg): name for target, reg, name in KEYS}


def decode(buf, off):
    qword = int.from_bytes(buf[off:off + 8], "little")
    target = (qword >> 48) & 0xffff
    reg = qword & 0xffff
    value = (qword >> 16) & 0xffffffff
    return target, reg, value, qword


def valid(buf, off):
    target, reg, value, qword = decode(buf, off)
    return target in TARGETS and reg % 4 == 0 and reg < 0x8000


def find_runs(buf, min_qwords):
    runs = []
    for mod in range(8):
        current = []
        for off in range(mod, len(buf) - 7, 8):
            if valid(buf, off):
                current.append(off)
            else:
                if len(current) >= min_qwords:
                    runs.append(current)
                current = []
        if len(current) >= min_qwords:
            runs.append(current)
    return sorted(runs, key=lambda run: run[0])


def run_values(buf, run):
    values = {}
    for off in run:
        target, reg, value, _qword = decode(buf, off)
        if (target, reg) in KEY_NAMES:
            values[(target, reg)] = value
    return values


def value(values, target, reg, default=0):
    return values.get((target, reg), default)


def split_hw(value32):
    return (value32 >> 16) & 0xffff, value32 & 0xffff


def classify_family(conv_con2):
    prefix = conv_con2 & 0xf0000000
    low = conv_con2 & 0x0fffffff
    names = {
        0x00000000: "setup",
        0x10000000: "y_mid",
        0x20000000: "y_tile",
        0x40000000: "k_half",
        0x50000000: "k_tile",
    }
    return names.get(prefix, f"family_{prefix >> 28:x}"), prefix, low


def infer_descriptor(values):
    conv_con2 = value(values, 0x0201, 0x1010)
    family, family_bits, grain_bits = classify_family(conv_con2)
    data_w, input_h = split_hw(value(values, 0x0201, 0x1020))
    _core_h, core_w = split_hw(value(values, 0x0801, 0x3014))
    dpu_h, dpu_w = split_hw(value(values, 0x1001, 0x405c))
    weight_size2 = value(values, 0x0201, 0x1038)
    kernel_w = (weight_size2 >> 24) & 0xff
    kernel_h = (weight_size2 >> 16) & 0xff
    output_c_count = weight_size2 & 0xffff
    output_c_end = value(values, 0x1001, 0x4058) + 1
    return {
        "family": family,
        "family_bits": family_bits,
        "grain_bits": grain_bits,
        "input_h": input_h,
        "input_w": data_w,
        "output_h": dpu_h + 1,
        "output_w": dpu_w + 1,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "output_c_count": output_c_count,
        "output_c_end": output_c_end,
        "feature_offset": value(values, 0x0201, 0x1070),
        "weight_offset": value(values, 0x0201, 0x1110),
        "output_offset": value(values, 0x1001, 0x4020),
        "weight_bytes": value(values, 0x0201, 0x1030),
        "weight_per_kernel": value(values, 0x0201, 0x1034),
        "weight_size2": weight_size2,
        "cbuf0": value(values, 0x0201, 0x1040),
        "cbuf1": value(values, 0x0201, 0x1044),
        "data_size0": value(values, 0x0201, 0x1020),
        "data_size3": value(values, 0x0201, 0x102c),
        "dst_surf_stride": value(values, 0x1001, 0x4024),
        "surface_add": value(values, 0x1001, 0x40c0),
    }


def summarize(path, min_qwords):
    buf = path.read_bytes()
    print(f"# {path.name}")
    print(f"size={len(buf)} magic={buf[:4]!r}")
    runs = find_runs(buf, min_qwords)
    print(f"runs={len(runs)}")
    key_ids = [(target, reg) for target, reg, _name in KEYS]
    for idx, run in enumerate(runs, 1):
        values = run_values(buf, run)
        print(f"run {idx}: off=0x{run[0]:x} qwords={len(run)} end=0x{run[-1] + 8:x}")
        for target, reg, name in KEYS:
            if (target, reg) in values:
                print(f"  {TARGETS[target]:5s} {reg:04x} {name:22s} 0x{values[(target, reg)]:08x}")
    print()


def summarize_descriptors(path, min_qwords):
    buf = path.read_bytes()
    print(f"# {path.name}")
    print("idx,off,n,family,family_bits,grain_bits,input_h,output_h,output_w,oc_count,oc_end,feature_off,weight_off,output_off,weight_bytes,cbuf0")
    for idx, run in enumerate(find_runs(buf, min_qwords), 1):
        desc = infer_descriptor(run_values(buf, run))
        print(
            f"{idx},0x{run[0]:x},{len(run)},{desc['family']},"
            f"0x{desc['family_bits']:08x},0x{desc['grain_bits']:08x},"
            f"{desc['input_h']},{desc['output_h']},{desc['output_w']},"
            f"{desc['output_c_count']},{desc['output_c_end']},"
            f"0x{desc['feature_offset']:x},0x{desc['weight_offset']:x},"
            f"0x{desc['output_offset']:x},0x{desc['weight_bytes']:x},"
            f"0x{desc['cbuf0']:x}"
        )
    print()


def summarize_descriptor_csv(paths, min_qwords):
    print("file,idx,off,n,family,family_bits,grain_bits,input_h,output_h,output_w,oc_count,oc_end,feature_off,weight_off,output_off,weight_bytes,weight_per_kernel,weight_size2,cbuf0,cbuf1,data_size0,data_size3,dst_surf_stride,surface_add")
    for path in paths:
        buf = path.read_bytes()
        for idx, run in enumerate(find_runs(buf, min_qwords), 1):
            desc = infer_descriptor(run_values(buf, run))
            print(
                f"{path.name},{idx},0x{run[0]:x},{len(run)},{desc['family']},"
                f"0x{desc['family_bits']:08x},0x{desc['grain_bits']:08x},"
                f"{desc['input_h']},{desc['output_h']},{desc['output_w']},"
                f"{desc['output_c_count']},{desc['output_c_end']},"
                f"0x{desc['feature_offset']:x},0x{desc['weight_offset']:x},"
                f"0x{desc['output_offset']:x},0x{desc['weight_bytes']:x},"
                f"0x{desc['weight_per_kernel']:x},0x{desc['weight_size2']:x},"
                f"0x{desc['cbuf0']:x},0x{desc['cbuf1']:x},"
                f"0x{desc['data_size0']:x},0x{desc['data_size3']:x},"
                f"0x{desc['dst_surf_stride']:x},0x{desc['surface_add']:x}"
            )

def summarize_run_tails(paths, min_qwords, tail_qwords):
    print("file,idx,off,n,tail_qwords")
    for path in paths:
        buf = path.read_bytes()
        for idx, run in enumerate(find_runs(buf, min_qwords), 1):
            tail = []
            for off in run[-tail_qwords:]:
                _target, _reg, _value, qword = decode(buf, off)
                tail.append(f"0x{qword:016x}")
            print(f"{path.name},{idx},0x{run[0]:x},{len(run)},{' '.join(tail)}")


def summarize_run_dump(paths, min_qwords, run_index=None):
    print("file,idx,pos,off,target,reg,name,value,pc_core,pc_amount,qword")
    for path in paths:
        buf = path.read_bytes()
        for idx, run in enumerate(find_runs(buf, min_qwords), 1):
            if run_index is not None and idx != run_index:
                continue
            for pos, off in enumerate(run):
                target, reg, value, qword = decode(buf, off)
                target_name = TARGETS.get(target, f"target_{target:04x}")
                name = REG_NAMES.get((target, reg), f"{target_name}_{reg:04x}")
                pc_core = ""
                pc_amount = ""
                if (target, reg) == (0x0101, 0x0014):
                    pc_core = str((value >> 16) & 0xffff)
                    pc_amount = str(value & 0xffff)
                print(
                    f"{path.name},{idx},{pos},0x{off:x},{target_name},0x{reg:04x},"
                    f"{name},0x{value:08x},{pc_core},{pc_amount},0x{qword:016x}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--min-qwords", type=int, default=24)
    parser.add_argument("--descriptors", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--tails", action="store_true")
    parser.add_argument("--dump-runs", action="store_true")
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--tail-qwords", type=int, default=8)
    args = parser.parse_args()
    if args.csv:
        summarize_descriptor_csv(args.files, args.min_qwords)
        return
    if args.tails:
        summarize_run_tails(args.files, args.min_qwords, args.tail_qwords)
        return
    if args.dump_runs:
        summarize_run_dump(args.files, args.min_qwords, args.run_index)
        return
    for path in args.files:
        if args.descriptors:
            summarize_descriptors(path, args.min_qwords)
        else:
            summarize(path, args.min_qwords)


if __name__ == "__main__":
    main()
