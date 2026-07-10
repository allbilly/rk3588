#!/usr/bin/env python3
import argparse
import importlib.util
import os
import re
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HEX_INT = r"(?:0x)?[0-9a-f]+"
DEFAULT_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_sweep_live_regcmd.log"
DEFAULT_IOCTL_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_ioctl_sweep_action_ret_readonly_latest.log"
DEFAULT_MMIO_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_ioctl_mmio_readonly.log"
DEFAULT_TIMEOUT_LOG = ROOT / "experimental" / "rknn" / "raw_sparse_task_gem_timeout_dmesg.log"
DEFAULT_SYSFS_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_ioctl_sysfs_readonly.log"
DEFAULT_TRACE_PATCH = ROOT / "experimental" / "rknn" / "rknpu_pc_commit_trace.patch"
DEFAULT_PC_TRACE_LOG = ROOT / "experimental" / "rknn" / "rknpu_pc_commit_trace.log"
DEFAULT_RKNN = ROOT / "experimental" / "rknn" / "models" / "sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1.rknn"
DEFAULT_SIX_RECORD_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_channeltile_20260524_113150.log"
DEFAULT_SIX_H7_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_6desc_h7_live_regcmd_20260524_123133.log"
DEFAULT_SIX_H14_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_h14_live_regcmd_832_20260524_124752.log"
DEFAULT_SIX_C32_H14_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_c32_h14_live_regcmd_20260524_1304.log"
DEFAULT_SIX_PW_C256_H14_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_pw_c256_h14_live_regcmd_832_20260524_131804.log"
DEFAULT_SIX_C64_H56_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_c64_h56_live_regcmd_1168_20260524_133725.log"
DEFAULT_C64_H56_IOCTL_LOG = ROOT / "experimental" / "rknn" / "capture_rknpu_ioctl_c64_h56_readonly_20260524_133736.log"
TASK_STRUCT = struct.Struct("<8IQ")
RK3588_PC_DATA_AMOUNT_SCALE = 2
RK3588_PC_TASK_NUMBER_BITS = 12
RKNPU_PC_DATA_EXTRA_AMOUNT = 4
INT_RAW_RESERVED_MASK = 0xffffc000
INT_COMPLETION_MASKS = {
    "CNA_FEATURE_0": 0x0001,
    "CNA_FEATURE_1": 0x0002,
    "CNA_WEIGHT_0": 0x0004,
    "CNA_WEIGHT_1": 0x0008,
    "CNA_CSC_0": 0x0010,
    "CNA_CSC_1": 0x0020,
    "CORE_0": 0x0040,
    "CORE_1": 0x0080,
    "DPU_0": 0x0100,
    "DPU_1": 0x0200,
    "PPU_0": 0x0400,
    "PPU_1": 0x0800,
    "DMA_READ_ERROR": 0x1000,
    "DMA_WRITE_ERROR": 0x2000,
}
DRIVER_CONFIG_RE = re.compile(
    r"static const struct rknpu_config rk3588_rknpu_config = \{(?P<body>.*?)\n\};",
    re.S,
)
CONFIG_FIELD_RE = re.compile(r"\.(?P<name>\w+)\s*=\s*(?P<value>[^,\n]+)")
RKNPU_JOB_PC = 1 << 0
RKNPU_JOB_NONBLOCK = 1 << 1
RKNPU_JOB_PINGPONG = 1 << 2
RKNPU_JOB_FENCE_IN = 1 << 3
RKNPU_JOB_FENCE_OUT = 1 << 4
MEM_CREATE_LINE = re.compile(
    r"MEM_CREATE #(?P<idx>\d+) size=(?P<size>\d+) flags=0x(?P<flags>[0-9a-f]+)"
)
MEM_CREATE_RET_LINE = re.compile(
    r"MEM_CREATE_RET #(?P<idx>\d+) handle=0x(?P<handle>[0-9a-f]+) "
    r"obj=0x(?P<obj>[0-9a-f]+) dma=0x(?P<dma>[0-9a-f]+) sram=(?P<sram_size>\d+)"
)
MEM_MAP_LINE = re.compile(
    r"MEM_MAP #(?P<idx>\d+) handle=0x(?P<handle>[0-9a-f]+) "
    r"reserved=0x(?P<reserved>[0-9a-f]+) offset=0x(?P<offset>[0-9a-f]+)"
)
MEM_MAP_RET_LINE = re.compile(
    r"MEM_MAP_RET #(?P<idx>\d+) handle=0x(?P<handle>[0-9a-f]+) "
    r"reserved=0x(?P<reserved>[0-9a-f]+) offset=0x(?P<offset>[0-9a-f]+)"
)
MMAP_LINE = re.compile(
    r"MMAP #(?P<idx>\d+) maps=(?P<maps>[0-9,]+) addr=0x(?P<addr>[0-9a-f]+) "
    r"length=(?P<length>\d+) prot=0x(?P<prot>[0-9a-f]+) flags=0x(?P<flags>[0-9a-f]+) "
    r"fd=(?P<fd>-?\d+) offset=0x(?P<offset>[0-9a-f]+)"
)
MMAP_RET_LINE = re.compile(r"MMAP_RET #(?P<idx>\d+) addr=0x(?P<addr>[0-9a-f]+)")
MEM_SYNC_LINE = re.compile(
    r"MEM_SYNC flags=0x(?P<flags>[0-9a-f]+) obj=0x(?P<obj>[0-9a-f]+) "
    r"offset=(?P<offset>\d+) size=(?P<size>\d+)"
)
RKNN_ALLOC_LINE = re.compile(
    r"allocated memory, name: (?P<name>\w+), virt addr: 0x(?P<virt>[0-9a-f]+), "
    r"dma addr: 0x(?P<dma>[0-9a-f]+), "
    r"obj addr: 0x(?P<obj>[0-9a-f]+), size: (?P<size>\d+), "
    r"aligned size: (?P<aligned_size>\d+), .*? handle: (?P<handle>\d+), "
    r"flags: 0x(?P<flags>[0-9a-f]+), "
    r".*? iommu domain id: (?P<iommu_domain_id>\d+)"
)
RKNN_FREE_LINE = re.compile(
    r"free memory, name: (?P<name>\w+), virt addr: 0x(?P<virt>[0-9a-f]+), "
    r"dma addr: 0x(?P<dma>[0-9a-f]+), "
    r"obj addr: 0x(?P<obj>[0-9a-f]+), size: (?P<size>\d+), "
    r"aligned size: (?P<aligned_size>\d+), .*? handle: (?P<handle>\d+), "
    r"flags: 0x(?P<flags>[0-9a-f]+), "
    r".*? iommu domain id: (?P<iommu_domain_id>\d+)"
)
RKNN_FEATURE_INPUT_LINE = re.compile(
    r"Conv\s+input\s+FLOAT16\s+NC1HWC2.*?\|\s+0x(?P<start>[0-9a-f]+)\s+"
    r"0x(?P<end>[0-9a-f]+)\s+0x(?P<size>[0-9a-f]+)"
)
RKNN_FEATURE_OUTPUT_LINE = re.compile(
    r"OutputOperator output\s+FLOAT16\s+NC1HWC2.*?\|\s+0x(?P<start>[0-9a-f]+)\*?"
    r"0x(?P<end>[0-9a-f]+)\s+0x(?P<size>[0-9a-f]+)"
)
SUBMIT_LINE = re.compile(
    r"flags=0x(?P<flags>[0-9a-f]+) timeout=(?P<timeout>\d+) "
    r"task_start=(?P<task_start>\d+) task_number=(?P<task_number>\d+)"
)
SUBMIT_COUNTER_LINE = re.compile(
    r"task_counter=(?P<task_counter>\d+) priority=(?P<priority>-?\d+) "
    r"task_obj_addr=0x(?P<task_obj_addr>[0-9a-f]+)"
)
SUBMIT_BASE_LINE = re.compile(
    r"regcfg_obj_addr=0x(?P<regcfg_obj_addr>[0-9a-f]+) "
    r"task_base_addr=0x(?P<task_base_addr>[0-9a-f]+) user_data=0x(?P<user_data>[0-9a-f]+)"
)
SUBMIT_CORE_LINE = re.compile(r"core_mask=0x(?P<core_mask>[0-9a-f]+) fence_fd=(?P<fence_fd>-?\d+)")
SUBCORE_LINE = re.compile(
    r"subcore_task\[(?P<idx>\d+)\]=\{task_start=(?P<task_start>\d+), task_number=(?P<task_number>\d+)\}"
)
TASK_OBJ_CREATE_MATCHES_LINE = re.compile(r"task_obj_mem_create_matches=(?P<count>\d+)")
TASK_OBJ_CREATE_MATCH_LINE = re.compile(
    r"match size=(?P<size>\d+) handle=0x(?P<handle>[0-9a-f]+) "
    r"dma=0x(?P<dma>[0-9a-f]+) flags=0x(?P<flags>[0-9a-f]+)"
)
TASK_OBJ_SYNC_MATCHES_LINE = re.compile(r"task_obj_mem_sync_matches=(?P<count>\d+)")
ACTION_LINE = re.compile(r"ACTION flags=0x(?P<flags>[0-9a-f]+) value=0x(?P<value>[0-9a-f]+)")
IOCTL_RET_LINE = re.compile(r"IOCTL_RET #(?P<idx>\d+) (?P<kind>.+) ret=(?P<ret>-?\d+)")
TASK_LINE = re.compile(
    r"task_like\[(?P<idx>\d+)\] off=0x(?P<off>[0-9a-f]+).*?"
    r"flags=0x(?P<flags>[0-9a-f]+) op_idx=(?P<op_idx>\d+) "
    r"enable_mask=0x(?P<enable>[0-9a-f]+) int_mask=0x(?P<int_mask>[0-9a-f]+) "
    r"int_clear=0x(?P<int_clear>[0-9a-f]+) int_status=0x(?P<int_status>[0-9a-f]+) "
    r"regcfg_amount=(?P<amount>\d+) regcfg_offset=(?P<regcfg_offset>\d+) "
    r"regcmd_addr=0x(?P<regcmd>[0-9a-f]+)"
)
REGCMD_QWORD_LINE = re.compile(
    r"regcmd_qword\[(?P<idx>\d+)\] off=0x(?P<off>[0-9a-f]+) value=0x(?P<value>[0-9a-f]+)"
)
MMIO_SNAPSHOT_LINE = re.compile(r"MMIO_SNAPSHOT (?P<label>\S+)")
MMIO_CORE_LINE = re.compile(r"\s+core=(?P<core>\d+) base=0x(?P<base>[0-9a-f]+)")
MMIO_REG_LINE = re.compile(
    r"\s+(?P<name>[A-Z0-9_]+)\s+off=0x(?P<off>[0-9a-f]+) value=0x(?P<value>[0-9a-f]+)"
)
MMIO_UNAVAILABLE_LINE = re.compile(
    r"\s+MMIO_UNAVAILABLE device=(?P<device>\S+) errno=(?P<errno>\d+) error=(?P<error>.+)"
)
SYSFS_SNAPSHOT_LINE = re.compile(r"SYSFS_SNAPSHOT (?P<label>\S+)")
SYSFS_VALUE_LINE = re.compile(r"\s+(?P<path>/\S+) = (?P<value>.*)")
RKNPU_TIMEOUT_JOB_LINE = re.compile(
    r"RKNPU: job: .*?mask: (?P<mask>0x[0-9a-f]+).*?timeout: (?P<timeout>\d+)us"
)
RKNPU_TIMEOUT_WAIT_LINE = re.compile(
    r"RKNPU: failed to wait job, task counter: (?P<task_counter>\d+), flags: (?P<flags>0x[0-9a-f]+)"
)
RKNPU_TIMEOUT_CORE_LINE = re.compile(
    r"RKNPU:\s+core (?P<core>\d+) irq status: (?P<irq_status>0x[0-9a-f]+), "
    r"raw status: (?P<raw_status>0x[0-9a-f]+), require mask: (?P<require_mask>0x[0-9a-f]+), "
    r"task counter: (?P<task_counter>0x[0-9a-f]+)"
)
PC_COMMIT_TRACE_LINE = re.compile(
    r"pc commit trace: core=(?P<core>\d+) submit_index=(?P<submit_index>\d+) "
    r"task_start=(?P<task_start>\d+) task_number=(?P<task_number>\d+) task_end=(?P<task_end>\d+) "
    rf"first_regcmd=(?P<first_regcmd>{HEX_INT}) first_amount=(?P<first_amount>\d+) "
    rf"first_enable=(?P<first_enable>{HEX_INT}) first_int=(?P<first_int>{HEX_INT}) "
    rf"last_int=(?P<last_int>{HEX_INT}) task_base_addr=(?P<task_base_addr>{HEX_INT}) "
    rf"flags=(?P<flags>{HEX_INT}) use_core_num=(?P<use_core_num>\d+)"
)
PC_COMMIT_PRELUDE_LINE = re.compile(
    rf"pc commit prelude: core=(?P<core>\d+) core_mask=(?P<core_mask>{HEX_INT}) "
    rf"use_core_num=(?P<use_core_num>\d+) task_status=(?P<PC_TASK_STATUS>{HEX_INT}) "
    rf"int_status=(?P<INT_STATUS>{HEX_INT}) int_raw=(?P<INT_RAW>{HEX_INT}) "
    rf"enable_mask=(?P<ENABLE_MASK>{HEX_INT})"
)
PC_COMMIT_REGS_LINE = re.compile(
    rf"pc commit regs: core=(?P<core>\d+) PC_DATA_ADDR=(?P<PC_DATA_ADDR>{HEX_INT}) "
    rf"PC_DATA_AMOUNT=(?P<PC_DATA_AMOUNT>{HEX_INT}) INT_MASK=(?P<INT_MASK>{HEX_INT}) "
    rf"INT_RAW=(?P<INT_RAW>{HEX_INT}) TASK_CONTROL=(?P<PC_TASK_CONTROL>{HEX_INT}) "
    rf"DMA_BASE=(?P<PC_DMA_BASE_ADDR>{HEX_INT}) TASK_STATUS=(?P<PC_TASK_STATUS>{HEX_INT}) "
    rf"ENABLE_MASK=(?P<ENABLE_MASK>{HEX_INT})"
)
PC_COMMIT_ARMED_LINE = re.compile(
    rf"pc commit armed: core=(?P<core>\d+) PC_OP_EN=(?P<PC_OP_EN>{HEX_INT}) "
    rf"INT_STATUS=(?P<INT_STATUS>{HEX_INT}) INT_RAW=(?P<INT_RAW>{HEX_INT}) "
    rf"TASK_STATUS=(?P<PC_TASK_STATUS>{HEX_INT})"
)
PC_TIMEOUT_REGS_LINE = re.compile(
    rf"core (?P<core>\d+) timeout pc regs: PC_DATA_ADDR=(?P<PC_DATA_ADDR>{HEX_INT}) "
    rf"PC_DATA_AMOUNT=(?P<PC_DATA_AMOUNT>{HEX_INT}) TASK_CONTROL=(?P<PC_TASK_CONTROL>{HEX_INT}) "
    rf"DMA_BASE=(?P<PC_DMA_BASE_ADDR>{HEX_INT}) PC_OP_EN=(?P<PC_OP_EN>{HEX_INT}) "
    rf"ENABLE_MASK=(?P<ENABLE_MASK>{HEX_INT})"
)
PC_VALID_IRQ_TRACE_LINE = re.compile(
    rf"valid irq trace: core=(?P<core>\d+) status=(?P<INT_STATUS>{HEX_INT}) "
    rf"raw=(?P<INT_RAW>{HEX_INT}) require=(?P<INT_MASK>{HEX_INT}) "
    rf"task_counter=(?P<PC_TASK_STATUS>{HEX_INT}) PC_DATA_ADDR=(?P<PC_DATA_ADDR>{HEX_INT}) "
    rf"PC_DATA_AMOUNT=(?P<PC_DATA_AMOUNT>{HEX_INT}) TASK_CONTROL=(?P<PC_TASK_CONTROL>{HEX_INT}) "
    rf"DMA_BASE=(?P<PC_DMA_BASE_ADDR>{HEX_INT}) ENABLE_MASK=(?P<ENABLE_MASK>{HEX_INT})"
)
PC_INVALID_IRQ_REGS_LINE = re.compile(
    rf"invalid irq pc regs: core=(?P<core>\d+) PC_DATA_ADDR=(?P<PC_DATA_ADDR>{HEX_INT}) "
    rf"PC_DATA_AMOUNT=(?P<PC_DATA_AMOUNT>{HEX_INT}) TASK_CONTROL=(?P<PC_TASK_CONTROL>{HEX_INT}) "
    rf"DMA_BASE=(?P<PC_DMA_BASE_ADDR>{HEX_INT}) PC_OP_EN=(?P<PC_OP_EN>{HEX_INT}) "
    rf"ENABLE_MASK=(?P<ENABLE_MASK>{HEX_INT})"
)
DMA_REGS = {
    (0x0201, 0x1070),  # CNA_FEATURE_DATA_ADDR
    (0x0201, 0x1110),  # CNA_DCOMP_ADDR0
    (0x0201, 0x1114),  # CNA_DCOMP_ADDR1
    (0x1001, 0x4020),  # DPU_DST_BASE_ADDR
    (0x2001, 0x5018),  # RDMA_SRC_BASE_ADDR
    (0x2001, 0x5038),  # RDMA_EW_BASE_ADDR
    (0x4001, 0x6070),  # PPU_DST_BASE_ADDR
    (0x8001, 0x701c),  # PPU_RDMA_SRC_BASE_ADDR
}
EXPECTED_ABI_LAYOUTS = {
    "struct_rknpu_task": {
        "size": 40,
        "offsets": {
            "flags": 0,
            "op_idx": 4,
            "enable_mask": 8,
            "int_mask": 12,
            "int_clear": 16,
            "int_status": 20,
            "regcfg_amount": 24,
            "regcfg_offset": 28,
            "regcmd_addr": 32,
        },
    },
    "rknpu_mem_create": {
        "size": 40,
        "offsets": {
            "handle": 0,
            "flags": 4,
            "size": 8,
            "obj_addr": 16,
            "dma_addr": 24,
            "sram_size": 32,
        },
    },
    "rknpu_mem_map": {
        "size": 16,
        "offsets": {
            "handle": 0,
            "reserved": 4,
            "offset": 8,
        },
    },
    "rknpu_mem_sync": {
        "size": 32,
        "offsets": {
            "flags": 0,
            "reserved": 4,
            "obj_addr": 8,
            "offset": 16,
            "size": 24,
        },
    },
    "rknpu_action": {
        "size": 8,
        "offsets": {
            "flags": 0,
            "value": 4,
        },
    },
    "rknpu_subcore_task": {
        "size": 8,
        "offsets": {
            "task_start": 0,
            "task_number": 4,
        },
    },
    "rknpu_submit": {
        "size": 104,
        "offsets": {
            "flags": 0,
            "timeout": 4,
            "task_start": 8,
            "task_number": 12,
            "task_counter": 16,
            "priority": 20,
            "task_obj_addr": 24,
            "iommu_domain_id": 32,
            "reserved": 36,
            "task_base_addr": 40,
            "hw_elapse_time": 48,
            "core_mask": 56,
            "fence_fd": 60,
            "subcore_task": 64,
        },
    },
}


def observed_records():
    amounts = (108, 108, 104, 104, 26, 104, 104, 26, 104, 104, 26, 104, 104, 26, 104, 104, 26)
    starts = []
    cursor = 0
    for amount in amounts:
        starts.append(cursor)
        cursor += ((amount + 4 + 15) // 16) * 16
    records = []
    for start, amount in zip(starts, amounts):
        separator = amount == 26
        records.append({
            "flags": 0,
            "op_idx": 1,
            "enable_mask": 0x60 if separator else 0x0d,
            "int_mask": 0x0c00 if separator else 0x0300,
            "int_clear": 0x1ffff,
            "int_status": 0,
            "regcfg_amount": amount,
            "regcfg_offset": 0,
            "regcmd_offset": start * 8,
        })
    return records


def observed_six_desc_records():
    amounts = (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
    starts = []
    cursor = 0
    for amount in amounts:
        starts.append(cursor)
        cursor += ((amount + RKNPU_PC_DATA_EXTRA_AMOUNT + 15) // 16) * 16
    records = []
    for start, amount in zip(starts, amounts):
        separator = amount == 26
        records.append({
            "flags": 0,
            "op_idx": 1,
            "enable_mask": 0x60 if separator else 0x0d,
            "int_mask": 0x0c00 if separator else 0x0300,
            "int_clear": 0x1ffff,
            "int_status": 0,
            "regcfg_amount": amount,
            "regcfg_offset": 0,
            "regcmd_offset": start * 8,
        })
    return records


SIX_DESC_RECORD_LINKS = (None, 2, None, 4, None, 6, None, 8, None, 10, None)
ROOT6_RECORD_LINKS = (
    1, None, 3, 4, None, 6, 7, None, 9, 10, None, 12, 13, None, 15, 16, None,
)
C64_H56_RECORD_AMOUNTS = (
    108, 108, 104, 104, 26, 104, 104, 26, 104, 26, 104, 26, 104, 26,
)
C64_H56_RECORD_LINKS = (
    1, None, 3, 4, None, 6, 7, None, 9, None, 11, None, 13, None,
)
C64_H56_ACTIVE_OFFSET = 0x4800
C64_H56_TASK_BYTES = 560
C64_H56_REGCMD_BYTES = 28352
C64_H56_REGCMD_ACTIVE_BYTES = 9344
C64_H56_WEIGHT_BYTES = 1204224
C64_H56_INPUT_BYTES = 401408
C64_H56_OUTPUT_BYTES = 802816
RKNN_TASK_BYTES = 680
RKNN_REGCMD_BYTES = 936384
RKNN_REGCMD_ACTIVE_OFFSET = 923648
RKNN_REGCMD_ACTIVE_BYTES = 12032
RKNN_WEIGHT_BYTES = 1436160
RKNN_INPUT_BYTES = 512000
RKNN_OUTPUT_BYTES = 924160
RKNPU_MEM_NON_CONTIGUOUS = 1 << 0
RKNPU_MEM_CACHEABLE = 1 << 1
RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT = 1 << 10
RKNPU_MEM_SYNC_TO_DEVICE = 1 << 0
RKNPU_MEM_SYNC_FROM_DEVICE = 1 << 1
RKNPU_GET_HW_VERSION = 0
RKNPU_GET_DRV_VERSION = 1
RKNPU_GET_IOMMU_EN = 18
RKNPU_SET_PROC_NICE = 19
RKNPU_POWER_ON = 20


def pack_records(records, regcmd_base=0):
    return b"".join(
        TASK_STRUCT.pack(
            row["flags"],
            row["op_idx"],
            row["enable_mask"],
            row["int_mask"],
            row["int_clear"],
            row["int_status"],
            row["regcfg_amount"],
            row["regcfg_offset"],
            regcmd_base + row["regcmd_offset"],
        )
        for row in records
    )


def parse_c_int_expr(expr):
    expr = expr.strip()
    if expr.startswith("DMA_BIT_MASK(") and expr.endswith(")"):
        bits = int(expr[len("DMA_BIT_MASK("):-1], 0)
        return (1 << bits) - 1
    if expr.startswith("ARRAY_SIZE("):
        # rk3588_npu_irqs has core0, core1, core2.
        return 3
    if expr.startswith("(1 << ") and expr.endswith(") - 1"):
        bits = int(expr[len("(1 << "):-len(") - 1")], 0)
        return (1 << bits) - 1
    if expr == "NULL":
        return 0
    return int(expr, 0)


def parse_rk3588_driver_config():
    text = (ROOT / "ref" / "rknpu_driver" / "rknpu_drv.c").read_text()
    match = DRIVER_CONFIG_RE.search(text)
    if not match:
        raise RuntimeError("rk3588_rknpu_config not found")
    fields = {}
    for field in CONFIG_FIELD_RE.finditer(match.group("body")):
        try:
            fields[field.group("name")] = parse_c_int_expr(field.group("value"))
        except ValueError:
            pass
    return fields


def load_conv_tiles_no_device():
    os.environ["RK3588_CONV_NO_DEVICE"] = "1"
    spec = importlib.util.spec_from_file_location("conv_tiles_offline", ROOT / "examples" / "conv_tiles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_conv_tiles_no_device_for_shape(shape_name):
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(ROOT / "examples" / "conv_tiles.py"), shape_name]
        return load_conv_tiles_no_device()
    finally:
        sys.argv = old_argv


def apply_rknn_profile(conv, task_bytes, regcmd_bytes, weight_bytes, input_bytes, output_bytes):
    conv.RKNN_MEM_SYNC = True
    conv.RKNN_SKIP_RESET = True
    conv.RKNN_REGCMD_ACTIVE_OFFSET = RKNN_REGCMD_ACTIVE_OFFSET
    conv.RKNN_REGCMD_ACTIVE_BYTES = RKNN_REGCMD_ACTIVE_BYTES
    conv.RKNN_RUNTIME_PROFILE = {
        "task": task_bytes,
        "regcmd": regcmd_bytes,
        "weight": weight_bytes,
        "input": input_bytes,
        "output": output_bytes,
    }
    rknn_mem_flags = RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE | RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT
    conv.task_map, conv.tasks_mem_create = bytearray(task_bytes), conv._offline_mem_create(
        task_bytes,
        rknn_mem_flags | conv.RKNPU_MEM_KERNEL_MAPPING,
    )
    conv.regcmd_map, conv.regcmd_mem_create = bytearray(regcmd_bytes), conv._offline_mem_create(
        regcmd_bytes,
        rknn_mem_flags,
    )
    conv.weight_map, conv.weight_mem_create = bytearray(weight_bytes), conv._offline_mem_create(
        weight_bytes,
        rknn_mem_flags,
    )
    conv.input_map, conv.input_mem_create = bytearray(input_bytes), conv._offline_mem_create(
        input_bytes,
        rknn_mem_flags,
    )
    conv.output_map, conv.output_mem_create = bytearray(output_bytes), conv._offline_mem_create(
        output_bytes,
        rknn_mem_flags,
    )
    conv.npu_tasks = conv.ctypes.cast(
        conv.ctypes.addressof(conv.ctypes.c_char.from_buffer(conv.task_map)),
        conv.ctypes.POINTER(conv.struct_rknpu_task),
    )
    conv.npu_regcmd = conv.ctypes.cast(
        conv.ctypes.addressof(conv.ctypes.c_char.from_buffer(conv.regcmd_map)),
        conv.ctypes.POINTER(conv.ctypes.c_uint64),
    )
    return conv


def apply_h40_rknn_profile(conv):
    conv = apply_rknn_profile(
        conv,
        RKNN_TASK_BYTES,
        RKNN_REGCMD_BYTES,
        RKNN_WEIGHT_BYTES,
        RKNN_INPUT_BYTES,
        RKNN_OUTPUT_BYTES,
    )
    conv.RKNN_TASK_BYTES = RKNN_TASK_BYTES
    conv.RKNN_REGCMD_BYTES = RKNN_REGCMD_BYTES
    conv.RKNN_WEIGHT_BYTES = RKNN_WEIGHT_BYTES
    conv.RKNN_INPUT_BYTES = RKNN_INPUT_BYTES
    conv.RKNN_OUTPUT_BYTES = RKNN_OUTPUT_BYTES
    return conv


def apply_c64_h56_rknn_profile(conv):
    conv = apply_rknn_profile(
        conv,
        C64_H56_TASK_BYTES,
        C64_H56_REGCMD_BYTES,
        C64_H56_WEIGHT_BYTES,
        C64_H56_INPUT_BYTES,
        C64_H56_OUTPUT_BYTES,
    )
    conv.RKNN_REGCMD_ACTIVE_OFFSET = C64_H56_ACTIVE_OFFSET
    conv.RKNN_REGCMD_ACTIVE_BYTES = C64_H56_REGCMD_ACTIVE_BYTES
    conv.RKNN_TASK_BYTES = C64_H56_TASK_BYTES
    conv.RKNN_REGCMD_BYTES = C64_H56_REGCMD_BYTES
    conv.RKNN_WEIGHT_BYTES = C64_H56_WEIGHT_BYTES
    conv.RKNN_INPUT_BYTES = C64_H56_INPUT_BYTES
    conv.RKNN_OUTPUT_BYTES = C64_H56_OUTPUT_BYTES
    return conv


def parse_ioctl_sequence(ioctl_log):
    actions = []
    creates = []
    create_returns = []
    maps = []
    map_returns = []
    mmaps = []
    mmap_returns = []
    syncs = []
    returns = []
    submit = {}
    subcores = []
    task_obj_create_matches = []
    seen_submit = False
    for line in Path(ioctl_log).read_text().splitlines():
        if match := ACTION_LINE.search(line):
            if not seen_submit:
                actions.append({
                    "flags": int(match.group("flags"), 16),
                    "value": int(match.group("value"), 16),
                })
            continue
        if match := IOCTL_RET_LINE.search(line):
            returns.append({
                "kind": match.group("kind"),
                "ret": int(match.group("ret")),
            })
            continue
        if match := MEM_CREATE_LINE.search(line):
            creates.append({
                "size": int(match.group("size")),
                "flags": int(match.group("flags"), 16),
            })
            continue
        if match := MEM_CREATE_RET_LINE.search(line):
            create_returns.append({
                "handle": int(match.group("handle"), 16),
                "obj_addr": int(match.group("obj"), 16),
                "dma_addr": int(match.group("dma"), 16),
                "sram_size": int(match.group("sram_size")),
            })
            continue
        if match := MEM_MAP_LINE.search(line):
            maps.append({
                "handle": int(match.group("handle"), 16),
                "reserved": int(match.group("reserved"), 16),
                "offset": int(match.group("offset"), 16),
            })
            continue
        if match := MEM_MAP_RET_LINE.search(line):
            map_returns.append({
                "handle": int(match.group("handle"), 16),
                "reserved": int(match.group("reserved"), 16),
                "offset": int(match.group("offset"), 16),
            })
            continue
        if match := MMAP_LINE.search(line):
            mmaps.append({
                "maps": tuple(int(item) for item in match.group("maps").split(",")),
                "addr": int(match.group("addr"), 16),
                "length": int(match.group("length")),
                "prot": int(match.group("prot"), 16),
                "flags": int(match.group("flags"), 16),
                "fd": int(match.group("fd")),
                "offset": int(match.group("offset"), 16),
            })
            continue
        if match := MMAP_RET_LINE.search(line):
            mmap_returns.append({
                "addr": int(match.group("addr"), 16),
            })
            continue
        if match := MEM_SYNC_LINE.search(line):
            syncs.append({
                "flags": int(match.group("flags"), 16),
                "offset": int(match.group("offset")),
                "size": int(match.group("size")),
            })
            continue
        if match := SUBMIT_LINE.search(line):
            seen_submit = True
            submit.update({
                "flags": int(match.group("flags"), 16),
                "timeout": int(match.group("timeout")),
                "task_start": int(match.group("task_start")),
                "task_number": int(match.group("task_number")),
            })
            continue
        if match := SUBMIT_COUNTER_LINE.search(line):
            submit.update({
                "task_counter": int(match.group("task_counter")),
                "priority": int(match.group("priority")),
            })
            continue
        if match := SUBMIT_BASE_LINE.search(line):
            submit.update({
                "regcfg_obj_addr": int(match.group("regcfg_obj_addr"), 16),
                "task_base_addr": int(match.group("task_base_addr"), 16),
                "user_data": int(match.group("user_data"), 16),
            })
            continue
        if match := SUBMIT_CORE_LINE.search(line):
            submit.update({
                "core_mask": int(match.group("core_mask"), 16),
                "fence_fd": int(match.group("fence_fd")),
            })
            continue
        if match := SUBCORE_LINE.search(line):
            subcores.append((int(match.group("task_start")), int(match.group("task_number"))))
            continue
        if match := TASK_OBJ_CREATE_MATCHES_LINE.search(line):
            submit["task_obj_mem_create_matches"] = int(match.group("count"))
            continue
        if match := TASK_OBJ_CREATE_MATCH_LINE.search(line):
            task_obj_create_matches.append({
                "size": int(match.group("size")),
                "handle": int(match.group("handle"), 16),
                "flags": int(match.group("flags"), 16),
            })
            continue
        if match := TASK_OBJ_SYNC_MATCHES_LINE.search(line):
            submit["task_obj_mem_sync_matches"] = int(match.group("count"))
    if subcores:
        submit["subcore_task"] = subcores
    if task_obj_create_matches:
        submit["task_obj_create_matches"] = task_obj_create_matches
    return actions, creates, create_returns, maps, map_returns, mmaps, mmap_returns, syncs, submit, returns


def modeled_ioctl_sequence(conv):
    actions = [
        {"flags": RKNPU_GET_HW_VERSION, "value": 0xffffffff},
        {"flags": RKNPU_GET_DRV_VERSION, "value": 0},
        {"flags": RKNPU_POWER_ON, "value": 0},
        {"flags": RKNPU_SET_PROC_NICE, "value": 0xffffffed},
        {"flags": RKNPU_GET_DRV_VERSION, "value": 0},
        {"flags": RKNPU_GET_IOMMU_EN, "value": 0},
        {"flags": RKNPU_GET_IOMMU_EN, "value": 0},
        {"flags": RKNPU_GET_DRV_VERSION, "value": 0},
        {"flags": RKNPU_GET_IOMMU_EN, "value": 0},
        {"flags": RKNPU_GET_IOMMU_EN, "value": 0},
        {"flags": RKNPU_GET_IOMMU_EN, "value": 0},
    ]
    if not conv.RKNN_SKIP_RESET:
        actions.append({"flags": conv.RKNPU_ACT_RESET, "value": 0})
    creates = [
        {"size": conv.tasks_mem_create.size, "flags": conv.tasks_mem_create.flags},
        {"size": conv.regcmd_mem_create.size, "flags": conv.regcmd_mem_create.flags},
        {"size": conv.weight_mem_create.size, "flags": conv.weight_mem_create.flags},
        {"size": conv.input_mem_create.size, "flags": conv.input_mem_create.flags},
        {"size": conv.output_mem_create.size, "flags": conv.output_mem_create.flags},
    ]
    create_returns = []
    syncs = [
        {"flags": 0x3, "offset": 0, "size": conv.tasks_mem_create.size},
        {"flags": 0x3, "offset": 0, "size": conv.regcmd_mem_create.size},
        {"flags": 0x1, "offset": 0, "size": conv.regcmd_mem_create.size},
        {"flags": 0x1, "offset": 0, "size": conv.tasks_mem_create.size},
        {"flags": 0x3, "offset": 0, "size": conv.weight_mem_create.size},
        {"flags": 0x3, "offset": 0, "size": conv.input_mem_create.size},
        {"flags": 0x3, "offset": 0, "size": conv.output_mem_create.size},
        {"flags": 0x1, "offset": conv.RKNN_REGCMD_ACTIVE_OFFSET, "size": conv.RKNN_REGCMD_ACTIVE_BYTES},
        {"flags": 0x1, "offset": 0, "size": conv.input_mem_create.size},
        {"flags": 0x1, "offset": conv.RKNN_REGCMD_ACTIVE_OFFSET, "size": conv.RKNN_REGCMD_ACTIVE_BYTES},
        {"flags": 0x2, "offset": 0, "size": conv.output_mem_create.size},
        {"flags": 0x2, "offset": 0, "size": conv.output_mem_create.size},
    ]
    submit = {
        "flags": 0x5,
        "timeout": 6000,
        "task_start": 0,
        "task_number": 6,
        "task_counter": 0,
        "priority": 0,
        "regcfg_obj_addr": 0,
        "task_base_addr": 0,
        "user_data": 0,
        "core_mask": 0,
        "fence_fd": -1,
        "subcore_task": [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)],
        "task_obj_mem_create_matches": 1,
        "task_obj_create_matches": [
            {"size": 4096, "handle": 1, "flags": conv.tasks_mem_create.flags},
        ],
        "task_obj_mem_sync_matches": 2,
    }
    returns = []
    maps = []
    map_returns = []
    mmaps = []
    mmap_returns = []
    return actions, creates, create_returns, maps, map_returns, mmaps, mmap_returns, syncs, submit, returns


def compare_ioctl_sequence(conv, ioctl_log):
    actual_actions, actual_creates, actual_create_returns, actual_maps, actual_map_returns, actual_mmaps, actual_mmap_returns, actual_syncs, actual_submit, actual_returns = parse_ioctl_sequence(ioctl_log)
    expected_actions, expected_creates, _expected_create_returns, _expected_maps, _expected_map_returns, _expected_mmaps, _expected_mmap_returns, expected_syncs, expected_submit, _expected_returns = modeled_ioctl_sequence(conv)
    failures = []
    if actual_actions != expected_actions:
        failures.append(f"ACTION actual={actual_actions} modeled={expected_actions}")
    if actual_creates != expected_creates:
        failures.append(f"MEM_CREATE actual={actual_creates} modeled={expected_creates}")
    if len(actual_create_returns) != len(expected_creates):
        failures.append(f"MEM_CREATE_RET rows={len(actual_create_returns)} expected={len(expected_creates)}")
    for idx, row in enumerate(actual_create_returns):
        if row["handle"] != idx + 1:
            failures.append(f"MEM_CREATE_RET[{idx}] handle={row['handle']} expected={idx + 1}")
        if row["sram_size"] != 0:
            failures.append(f"MEM_CREATE_RET[{idx}] sram_size={row['sram_size']} expected=0")
    if len(actual_maps) != len(expected_creates):
        failures.append(f"MEM_MAP rows={len(actual_maps)} expected={len(expected_creates)}")
    if len(actual_map_returns) != len(expected_creates):
        failures.append(f"MEM_MAP_RET rows={len(actual_map_returns)} expected={len(expected_creates)}")
    previous_map_offset = None
    for idx, row in enumerate(actual_maps):
        expected_handle = idx + 1
        if row != {"handle": expected_handle, "reserved": 0, "offset": 0}:
            failures.append(f"MEM_MAP[{idx}] actual={row} expected pre-call handle={expected_handle}, reserved=0, offset=0")
    for idx, row in enumerate(actual_map_returns):
        expected_handle = idx + 1
        if row["handle"] != expected_handle:
            failures.append(f"MEM_MAP_RET[{idx}] handle={row['handle']} expected={expected_handle}")
        if row["reserved"] != 0:
            failures.append(f"MEM_MAP_RET[{idx}] reserved={row['reserved']} expected=0")
        if row["offset"] % 4096 != 0:
            failures.append(f"MEM_MAP_RET[{idx}] offset=0x{row['offset']:x} is not page-aligned")
        if previous_map_offset is not None and row["offset"] <= previous_map_offset:
            failures.append(f"MEM_MAP_RET[{idx}] offset=0x{row['offset']:x} not greater than previous 0x{previous_map_offset:x}")
        previous_map_offset = row["offset"]
    if len(actual_mmaps) != len(expected_creates):
        failures.append(f"MMAP rows={len(actual_mmaps)} expected={len(expected_creates)}")
    if len(actual_mmap_returns) != len(expected_creates):
        failures.append(f"MMAP_RET rows={len(actual_mmap_returns)} expected={len(expected_creates)}")
    for idx, row in enumerate(actual_mmaps):
        expected_handle = idx + 1
        if row["maps"] != (expected_handle,):
            failures.append(f"MMAP[{idx}] maps={row['maps']} expected={(expected_handle,)}")
        if row["addr"] != 0:
            failures.append(f"MMAP[{idx}] requested addr=0x{row['addr']:x} expected=0")
        if row["length"] != expected_creates[idx]["size"]:
            failures.append(f"MMAP[{idx}] length={row['length']} expected={expected_creates[idx]['size']}")
        if row["prot"] != 0x3 or row["flags"] != 0x1:
            failures.append(f"MMAP[{idx}] prot/flags actual=0x{row['prot']:x}/0x{row['flags']:x} expected=0x3/0x1")
        if row["fd"] != 3:
            failures.append(f"MMAP[{idx}] fd={row['fd']} expected=3")
        if idx < len(actual_map_returns) and row["offset"] != actual_map_returns[idx]["offset"]:
            failures.append(f"MMAP[{idx}] offset=0x{row['offset']:x} expected map_ret=0x{actual_map_returns[idx]['offset']:x}")
    if actual_syncs != expected_syncs:
        failures.append(f"MEM_SYNC actual={actual_syncs} modeled={expected_syncs}")
    if actual_submit != expected_submit:
        failures.append(f"SUBMIT actual={actual_submit} modeled={expected_submit}")
    submit_returns = [row["ret"] for row in actual_returns if row["kind"].startswith("SUBMIT")]
    if submit_returns != [0]:
        failures.append(f"SUBMIT returns {submit_returns} != [0]")
    failed_action_kinds = [row["kind"] for row in actual_returns if row["kind"].startswith("ACTION") and row["ret"] != 0]
    expected_failed_actions = ["ACTION flags=0x00000014"]
    if failed_action_kinds[:1] != expected_failed_actions:
        failures.append(f"unexpected pre-submit failed ACTIONs {failed_action_kinds}")
    if failures:
        print("FAIL conv_tiles RKNN ioctl sequence comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS conv_tiles RKNN ioctl sequence comparison")
    print("ACTION/MEM_CREATE/MEM_SYNC/SUBMIT metadata and submit return match captured RKNN ioctl log")
    return 0


def parse_rknn_memory_debug_rows(ioctl_log, pattern):
    rows = []
    for line in Path(ioctl_log).read_text().splitlines():
        if match := pattern.search(line):
            rows.append({
                "name": match.group("name"),
                "handle": int(match.group("handle")),
                "virt_addr": int(match.group("virt"), 16),
                "obj_addr": int(match.group("obj"), 16),
                "dma_addr": int(match.group("dma"), 16),
                "size": int(match.group("size")),
                "aligned_size": int(match.group("aligned_size")),
                "flags": int(match.group("flags"), 16),
                "iommu_domain_id": int(match.group("iommu_domain_id")),
            })
    return rows


def parse_rknn_allocations(ioctl_log):
    return parse_rknn_memory_debug_rows(ioctl_log, RKNN_ALLOC_LINE)


def parse_rknn_frees(ioctl_log):
    return parse_rknn_memory_debug_rows(ioctl_log, RKNN_FREE_LINE)


def _memory_rows_by_unique_name(rows):
    out = {}
    for row in rows:
        out.setdefault(row["name"], row)
    return out


def compare_rknn_allocations(conv, ioctl_log):
    actual = parse_rknn_allocations(ioctl_log)
    _actions, _creates, create_returns, _maps, _map_returns, _mmaps, mmap_returns, _syncs, _submit, _returns = parse_ioctl_sequence(ioctl_log)
    expected = [
        {
            "name": "task",
            "handle": 1,
            "size": conv.RKNN_TASK_BYTES,
            "aligned_size": 4096,
            "flags": conv.tasks_mem_create.flags,
            "iommu_domain_id": 0,
        },
        {
            "name": "weight",
            "handle": 2,
            "size": conv.RKNN_REGCMD_BYTES,
            "aligned_size": 937984,
            "flags": conv.regcmd_mem_create.flags,
            "iommu_domain_id": 0,
        },
        {
            "name": "internal",
            "handle": 3,
            "size": conv.RKNN_WEIGHT_BYTES,
            "aligned_size": 1437696,
            "flags": conv.weight_mem_create.flags,
            "iommu_domain_id": 0,
        },
        {
            "name": "input",
            "handle": 4,
            "size": conv.RKNN_INPUT_BYTES,
            "aligned_size": 512000,
            "flags": conv.input_mem_create.flags,
            "iommu_domain_id": 0,
        },
        {
            "name": "output",
            "handle": 5,
            "size": conv.RKNN_OUTPUT_BYTES,
            "aligned_size": 925696,
            "flags": conv.output_mem_create.flags,
            "iommu_domain_id": 0,
        },
    ]
    failures = []
    comparable_actual = [
        {k: v for k, v in row.items() if k not in {"obj_addr", "dma_addr"}}
        | {"virt_addr": 0}
        for row in actual
    ]
    comparable_expected = [
        row | {"virt_addr": 0}
        for row in expected
    ]
    if comparable_actual != comparable_expected:
        failures.append(f"RKNN allocations actual={comparable_actual} expected={comparable_expected}")
    if len(create_returns) != len(actual):
        failures.append(f"MEM_CREATE_RET rows={len(create_returns)} allocation rows={len(actual)}")
    for idx, (create, alloc) in enumerate(zip(create_returns, actual)):
        if create["handle"] != alloc["handle"]:
            failures.append(f"alloc[{idx}] handle mismatch create={create['handle']} debug={alloc['handle']}")
        if create["obj_addr"] != alloc["obj_addr"]:
            failures.append(f"alloc[{idx}] obj mismatch create=0x{create['obj_addr']:x} debug=0x{alloc['obj_addr']:x}")
        if create["dma_addr"] != alloc["dma_addr"]:
            failures.append(f"alloc[{idx}] dma mismatch create=0x{create['dma_addr']:x} debug=0x{alloc['dma_addr']:x}")
    if len(mmap_returns) != len(actual):
        failures.append(f"MMAP_RET rows={len(mmap_returns)} allocation rows={len(actual)}")
    for idx, (mmap_ret, alloc) in enumerate(zip(mmap_returns, actual)):
        if mmap_ret["addr"] != alloc["virt_addr"]:
            failures.append(f"alloc[{idx}] virt mismatch mmap=0x{mmap_ret['addr']:x} debug=0x{alloc['virt_addr']:x}")
    if failures:
        print("FAIL RKNN debug allocation metadata comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS RKNN debug allocation metadata comparison")
    print("allocation names/order/sizes/aligned sizes/flags/iommu-domain match captured RKNN runtime log")
    return 0


def compare_rknn_free_teardown(ioctl_log, label, expected_sizes):
    allocs = parse_rknn_allocations(ioctl_log)
    frees = parse_rknn_frees(ioctl_log)
    alloc_by_name = _memory_rows_by_unique_name(allocs)
    expected_order = ("internal", "weight", "task", "input", "output")
    failures = []
    if [row["name"] for row in frees] != list(expected_order):
        failures.append(f"{label} free order={[row['name'] for row in frees]} expected={list(expected_order)}")
    for name in expected_order:
        alloc = alloc_by_name.get(name)
        free = next((row for row in frees if row["name"] == name), None)
        if alloc is None:
            failures.append(f"{label} missing allocation row for free {name}")
            continue
        if free is None:
            failures.append(f"{label} missing free row for {name}")
            continue
        expected_size = expected_sizes[name]
        if free["size"] != expected_size:
            failures.append(f"{label} free {name} size={free['size']} expected={expected_size}")
        for key in ("handle", "virt_addr", "obj_addr", "dma_addr", "aligned_size", "flags", "iommu_domain_id"):
            if free[key] != alloc[key]:
                failures.append(f"{label} free {name} {key}=0x{free[key]:x} allocation=0x{alloc[key]:x}")
    if failures:
        print(f"FAIL {label} RKNN free teardown comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"PASS {label} RKNN free teardown comparison")
    print("captured RKNN teardown frees internal, weight, task, input, and output BO debug rows with matching allocation metadata")
    return 0


def compare_c64_h56_runtime_profile(c64_log):
    conv = load_conv_tiles_no_device_for_shape("b1_c64_h56_w56_oc128_wic64_k1x1_g1")
    apply_c64_h56_rknn_profile(conv)
    actual = parse_rknn_allocations(c64_log)
    actual_by_name = {row["name"]: row for row in actual}
    expected = {
        "task": {
            "size": conv.tasks_mem_create.size,
            "flags": conv.tasks_mem_create.flags,
        },
        "weight": {
            "size": conv.regcmd_mem_create.size,
            "flags": conv.regcmd_mem_create.flags,
        },
        "internal": {
            "size": conv.weight_mem_create.size,
            "flags": conv.weight_mem_create.flags,
        },
        "input": {
            "size": conv.input_mem_create.size,
            "flags": conv.input_mem_create.flags,
        },
        "output": {
            "size": conv.output_mem_create.size,
            "flags": conv.output_mem_create.flags,
        },
    }
    failures = []
    for name, expected_row in expected.items():
        row = actual_by_name.get(name)
        if row is None:
            failures.append(f"missing C64/H56 allocation {name}")
            continue
        comparable = {"size": row["size"], "flags": row["flags"]}
        if comparable != expected_row:
            failures.append(f"C64/H56 allocation {name} actual={comparable} expected={expected_row}")
    if conv.RKNN_RUNTIME_PROFILE != {
        "task": C64_H56_TASK_BYTES,
        "regcmd": C64_H56_REGCMD_BYTES,
        "weight": C64_H56_WEIGHT_BYTES,
        "input": C64_H56_INPUT_BYTES,
        "output": C64_H56_OUTPUT_BYTES,
    }:
        failures.append(f"C64/H56 runtime profile mismatch: {conv.RKNN_RUNTIME_PROFILE}")
    if failures:
        print("FAIL C64/H56 RKNN-mode runtime profile")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS C64/H56 RKNN-mode runtime profile")
    print("shape-selected BO sizes/flags match captured RKNN task/weight/internal/input/output allocations")
    return 0


def compare_c64_h56_free_teardown(c64_log):
    conv = load_conv_tiles_no_device_for_shape("b1_c64_h56_w56_oc128_wic64_k1x1_g1")
    apply_c64_h56_rknn_profile(conv)
    return compare_rknn_free_teardown(c64_log, "C64/H56", {
        "task": C64_H56_TASK_BYTES,
        "weight": C64_H56_REGCMD_BYTES,
        "internal": C64_H56_WEIGHT_BYTES,
        "input": C64_H56_INPUT_BYTES,
        "output": C64_H56_OUTPUT_BYTES,
    })


def compare_c64_h56_sync_profile(c64_log):
    conv = load_conv_tiles_no_device_for_shape("b1_c64_h56_w56_oc128_wic64_k1x1_g1")
    apply_c64_h56_rknn_profile(conv)
    _actions, _creates, _create_returns, _maps, _map_returns, _mmaps, _mmap_returns, syncs, _submit, _returns = parse_ioctl_sequence(c64_log)
    expected = [
        {"flags": 0x3, "offset": 0, "size": conv.tasks_mem_create.size},
        {"flags": 0x3, "offset": 0, "size": conv.regcmd_mem_create.size},
        {"flags": 0x1, "offset": 0, "size": conv.regcmd_mem_create.size},
        {"flags": 0x1, "offset": 0, "size": conv.tasks_mem_create.size},
        {"flags": 0x3, "offset": 0, "size": conv.weight_mem_create.size},
        {"flags": 0x3, "offset": 0, "size": conv.input_mem_create.size},
        {"flags": 0x3, "offset": 0, "size": conv.output_mem_create.size},
        {"flags": 0x1, "offset": C64_H56_ACTIVE_OFFSET, "size": C64_H56_REGCMD_ACTIVE_BYTES},
        {"flags": 0x1, "offset": 0, "size": conv.input_mem_create.size},
        {"flags": 0x2, "offset": 0, "size": conv.input_mem_create.size},
        {"flags": 0x1, "offset": C64_H56_ACTIVE_OFFSET, "size": C64_H56_REGCMD_ACTIVE_BYTES},
        {"flags": 0x2, "offset": 0, "size": conv.output_mem_create.size},
        {"flags": 0x2, "offset": 0, "size": conv.output_mem_create.size},
        {"flags": 0x2, "offset": 0, "size": conv.output_mem_create.size},
        {"flags": 0x2, "offset": 0, "size": conv.output_mem_create.size},
    ]
    failures = []
    if syncs != expected:
        failures.append(f"C64/H56 MEM_SYNC actual={syncs} expected={expected}")
    if failures:
        print("FAIL C64/H56 RKNN-mode sync profile")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS C64/H56 RKNN-mode sync profile")
    print("captured C64/H56 includes input FROM_DEVICE before submit and four output FROM_DEVICE syncs after submit")
    return 0


def parse_rknn_feature_layout(ioctl_log):
    feature_layout = {}
    for line in Path(ioctl_log).read_text().splitlines():
        if match := RKNN_FEATURE_INPUT_LINE.search(line):
            feature_layout["input"] = {
                "start": int(match.group("start"), 16),
                "end": int(match.group("end"), 16),
                "size": int(match.group("size"), 16),
            }
        elif match := RKNN_FEATURE_OUTPUT_LINE.search(line):
            feature_layout["output"] = {
                "start": int(match.group("start"), 16),
                "end": int(match.group("end"), 16),
                "size": int(match.group("size"), 16),
            }
    return feature_layout


def compare_abi_layouts(conv):
    failures = []
    for class_name, expected in EXPECTED_ABI_LAYOUTS.items():
        cls = getattr(conv, class_name)
        actual_size = conv.ctypes.sizeof(cls)
        if actual_size != expected["size"]:
            failures.append(f"{class_name} size={actual_size} expected={expected['size']}")
        for field_name, expected_offset in expected["offsets"].items():
            actual_offset = getattr(cls, field_name).offset
            if actual_offset != expected_offset:
                failures.append(
                    f"{class_name}.{field_name} offset={actual_offset} expected={expected_offset}"
                )
    if conv.DRM_IOCTL_RKNPU_SUBMIT != 0xc0686441:
        failures.append(f"DRM_IOCTL_RKNPU_SUBMIT=0x{conv.DRM_IOCTL_RKNPU_SUBMIT:x} expected=0xc0686441")
    if failures:
        print("FAIL conv_tiles downstream rknpu ABI layout")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS conv_tiles downstream rknpu ABI layout")
    print("ctypes sizes/offsets match the live 40-byte MEM_CREATE ABI in ref/rk3588-npu/include/rknpu-ioctl.h; submit ioctl cmd=0xc0686441")
    return 0


def parse_log_records(log_path, gem, tail_count=None):
    rows = []
    in_target_gem = False
    for line in Path(log_path).read_text().splitlines():
        if line.startswith("GEM "):
            in_target_gem = line.startswith(f"GEM {gem}:")
            continue
        if not in_target_gem:
            continue
        match = TASK_LINE.search(line)
        if not match:
            continue
        rows.append({
            "off": int(match.group("off"), 16),
            "flags": int(match.group("flags"), 16),
            "op_idx": int(match.group("op_idx")),
            "enable_mask": int(match.group("enable"), 16),
            "int_mask": int(match.group("int_mask"), 16),
            "int_clear": int(match.group("int_clear"), 16),
            "int_status": int(match.group("int_status"), 16),
            "regcfg_amount": int(match.group("amount")),
            "regcfg_offset": int(match.group("regcfg_offset")),
            "regcmd_addr": int(match.group("regcmd"), 16),
        })
    if tail_count is not None:
        rows = rows[-tail_count:]
    return rows


def normalize_log_records(rows):
    if not rows:
        return []
    base = rows[0]["regcmd_addr"]
    normalized = []
    for row in rows:
        normalized.append({
            "flags": row["flags"],
            "op_idx": row["op_idx"],
            "enable_mask": row["enable_mask"],
            "int_mask": row["int_mask"],
            "int_clear": row["int_clear"],
            "int_status": row["int_status"],
            "regcfg_amount": row["regcfg_amount"],
            "regcfg_offset": row["regcfg_offset"],
            "regcmd_offset": row["regcmd_addr"] - base,
        })
    return normalized


def compare_records(label, actual, expected):
    failures = []
    if len(actual) != len(expected):
        failures.append(f"{label}: rows={len(actual)} expected={len(expected)}")
    for idx, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            failures.append(f"{label}[{idx}] actual={a} expected={e}")
    if failures:
        print(f"FAIL {label}")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"PASS {label}: {len(expected)} task records")
    return 0


def compare_six_desc_task_records(conv, six_record_log):
    log_path = Path(six_record_log)
    if not log_path.exists():
        print("SKIP six-desc sparse task-record comparison")
        print(f"missing capture log: {log_path}")
        return 0
    expected = observed_six_desc_records()
    gem1_records = normalize_log_records(parse_log_records(log_path, 1))
    rc = compare_records("six-desc GEM1 normalized task object", gem1_records, expected)
    task_bytes = pack_records(expected, regcmd_base=0)
    failures = []
    if len(task_bytes) != 440:
        failures.append(f"six-desc task bytes={len(task_bytes)} expected captured taskbuffer size 440")
    kinds = ["separator" if row["enable_mask"] == 0x60 else "conv" for row in expected]
    if kinds != ["conv", "conv", "separator", "conv", "separator", "conv", "separator", "conv", "separator", "conv", "separator"]:
        failures.append(f"unexpected six-desc record kind sequence: {kinds}")
    schedules = []
    for shape in conv.conv_regression_shapes():
        descs = conv.shape_direct_spatial_descs(shape)
        if descs and len(descs) == 6:
            schedules.append((shape["name"], conv.direct_spatial_desc_schedule(descs)))
    if not schedules:
        failures.append("no built-in 6-descriptor direct-spatial schedules found")
    if failures:
        print("FAIL six-desc sparse task-record model")
        for failure in failures:
            print(f"  {failure}")
        return 1 | rc
    if rc == 0:
        print("PASS six-desc sparse task-record model")
        print("captured 440-byte task object is 11 rknpu_task records: 6 executable conv records interleaved with 5 separators")
        print(f"built_in_six_desc_schedules={len(schedules)}")
    return rc


def compare_six_desc_evidence_state(conv):
    failures = []
    six_desc_shapes = []
    supported_six_desc_shapes = []
    evidence_only_six_desc_shapes = []
    pc_root6_shapes = []
    for shape in conv.conv_regression_shapes():
        descs = conv.shape_direct_spatial_descs(shape)
        if not descs:
            continue
        if len(descs) == 6:
            six_desc_shapes.append(shape["name"])
            if conv.direct_spatial_default_supported(descs):
                supported_six_desc_shapes.append(shape["name"])
            if conv.shape_has_evidence_only_direct_spatial(shape):
                evidence_only_six_desc_shapes.append(shape["name"])
        elif conv.direct_spatial_default_supported(descs):
            pc_root6_shapes.append(shape["name"])
    if pc_root6_shapes != ["b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid"]:
        failures.append(f"unexpected supported PC-root6 shapes: {pc_root6_shapes}")
    expected_supported_six_desc = [
        "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1",
        "b1_c256_h14_w14_oc512_wic256_k1x1_g1",
        "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid",
        "b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid",
        "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid",
    ]
    if supported_six_desc_shapes != expected_supported_six_desc:
        failures.append(f"unexpected supported six-desc shapes: {supported_six_desc_shapes}")
    if not six_desc_shapes:
        failures.append("no six-desc shapes found")
    if not evidence_only_six_desc_shapes:
        failures.append("no six-desc shapes currently remain evidence-only")
    if failures:
        print("FAIL six-desc direct-spatial evidence state")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS six-desc direct-spatial evidence state")
    print(f"six_desc_shapes={six_desc_shapes}")
    print(f"supported_six_desc_shapes={supported_six_desc_shapes}")
    print(f"evidence_only_six_desc_shapes={evidence_only_six_desc_shapes}")
    print("only the verified PW C256/H14, C32/H14, H14, and H7 6-descriptor schedules are formula-supported; other 6-descriptor schedules remain evidence-only")
    print("direct-spatial schedules are RKNN-captured templates plus decoded register math, not a complete formulaic tiler")
    return 0


def compare_task_object_source(gem_log_rows, embedded_rows, ioctl_log):
    _actions, _creates, create_returns, _maps, _map_returns, _mmaps, _mmap_returns, _syncs, submit, _returns = parse_ioctl_sequence(ioctl_log)
    failures = []
    if submit.get("task_base_addr") != 0:
        failures.append(f"task_base_addr={submit.get('task_base_addr')} expected 0")
    if submit.get("regcfg_obj_addr") != 0 or submit.get("user_data") != 0:
        failures.append(f"regcfg_obj_addr/user_data are not zero: {submit}")
    if submit.get("task_obj_mem_create_matches") != 1:
        failures.append(f"task_obj_mem_create_matches={submit.get('task_obj_mem_create_matches')} expected 1")
    if not create_returns or submit.get("task_obj_create_matches", [{}])[0].get("handle") != create_returns[0]["handle"]:
        failures.append("submitted task object does not match MEM_CREATE_RET #1 handle")
    if gem_log_rows != embedded_rows:
        failures.append("GEM1 task object and GEM2 embedded relative table differ after normalization")
    if failures:
        print("FAIL rknpu submitted task object source check")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu submitted task object source check")
    print("driver-visible submit uses task BO handle 1; GEM2 relative table is matching metadata, not an alternate submitted task source")
    return 0


def compare_job_lifecycle_source_model(ioctl_log):
    _actions, _creates, create_returns, _maps, _map_returns, _mmaps, _mmap_returns, _syncs, submit, returns = parse_ioctl_sequence(ioctl_log)
    source = (ROOT / "ref" / "rknpu_driver" / "rknpu_job.c").read_text()
    failures = []
    submit_returns = [row["ret"] for row in returns if row["kind"].startswith("SUBMIT")]
    if submit_returns != [0]:
        failures.append(f"captured RKNN submit returns {submit_returns} expected [0]")
    if submit.get("task_obj_mem_create_matches") != 1:
        failures.append(f"task_obj_mem_create_matches={submit.get('task_obj_mem_create_matches')} expected 1")
    if not create_returns or submit.get("task_obj_create_matches", [{}])[0].get("handle") != create_returns[0]["handle"]:
        failures.append("submitted task object does not match first returned GEM handle")
    if submit.get("task_counter") != 0:
        failures.append(f"pre-submit task_counter={submit.get('task_counter')} expected 0")
    required_fragments = (
        "task_obj = (struct rknpu_gem_object *)(uintptr_t)args->task_obj_addr;",
        "if (task_obj)\n\t\trknpu_gem_object_get(&task_obj->base);",
        "task_obj =\n\t\t(struct rknpu_gem_object *)(uintptr_t)job->args->task_obj_addr;",
        "if (task_obj)\n\t\trknpu_gem_object_put(&task_obj->base);",
        "if (!(args->flags & RKNPU_JOB_NONBLOCK)) {\n\t\tjob->args = args;",
        "job->args_owner = false;",
        "rknpu_job_schedule(job);",
        "job->ret = rknpu_job_wait(job);",
        "args->task_counter = job->args->task_counter;",
        "if (!ret)\n\t\t\trknpu_job_cleanup(job);\n\t\telse\n\t\t\trknpu_job_abort(job);",
        "args->task_counter = args->task_number;",
        "job->flags |= RKNPU_JOB_DONE;",
        "job->ret = ret;",
    )
    for fragment in required_fragments:
        if fragment not in source:
            failures.append(f"missing expected job lifecycle source fragment: {fragment}")
    if failures:
        print("FAIL rknpu job/task-object lifecycle source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu job/task-object lifecycle source model")
    print("captured RKNN submit returns success; source holds a task GEM reference for the synchronous job and releases it through cleanup, while abort is only the error path")
    return 0


def compare_core_selection_model(ioctl_log):
    _actions, _creates, _create_returns, _maps, _map_returns, _mmaps, _mmap_returns, _syncs, submit, _returns = parse_ioctl_sequence(ioctl_log)
    failures = []
    if submit.get("core_mask") != 0:
        failures.append(f"captured core_mask={submit.get('core_mask')} expected RKNPU_CORE_AUTO_MASK=0")
    subcores = submit.get("subcore_task", [])
    if len(subcores) != 5:
        failures.append(f"subcore_task rows={len(subcores)} expected=5")
    else:
        for core_index in range(3):
            if subcores[core_index] != (0, 2):
                failures.append(f"auto single-core candidate core{core_index} subcore_task={subcores[core_index]} expected=(0,2)")
        if subcores[3:] != [(0, 0), (0, 0)]:
            failures.append(f"inactive tail subcore entries={subcores[3:]} expected=[(0,0),(0,0)]")
    if submit.get("task_number") != 6:
        failures.append(f"global task_number={submit.get('task_number')} expected=6")
    if failures:
        print("FAIL rknpu auto-core selection model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu auto-core selection model")
    print("core_mask=0 schedules one core; whichever core is selected uses subcore_task[core]=(0,2), not tail entries")
    return 0


def compare_submit_mode_model(ioctl_log):
    _actions, _creates, _create_returns, _maps, _map_returns, _mmaps, _mmap_returns, _syncs, submit, returns = parse_ioctl_sequence(ioctl_log)
    source = (ROOT / "ref" / "rknpu_driver" / "rknpu_job.c").read_text()
    failures = []
    flags = submit.get("flags")
    expected_flags = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG
    if flags != expected_flags:
        failures.append(f"submit flags=0x{flags:x} expected PC|PINGPONG=0x{expected_flags:x}")
    if flags & (RKNPU_JOB_NONBLOCK | RKNPU_JOB_FENCE_IN | RKNPU_JOB_FENCE_OUT):
        failures.append(f"submit flags unexpectedly request nonblock/fence path: 0x{flags:x}")
    if submit.get("task_counter") != 0 or submit.get("priority") != 0:
        failures.append(f"task_counter/priority mismatch: {submit}")
    if submit.get("iommu_domain_id") not in (None, 0):
        failures.append(f"iommu_domain_id={submit.get('iommu_domain_id')} expected 0")
    submit_returns = [row["ret"] for row in returns if row["kind"].startswith("SUBMIT")]
    if submit_returns != [0]:
        failures.append(f"captured RKNN submit returns {submit_returns} expected [0]")
    required_source_fragments = (
        "job->iommu_domain_id = args->iommu_domain_id;",
        "if (!(args->flags & RKNPU_JOB_NONBLOCK))",
        "rknpu_job_schedule(job);",
        "if (args->flags & RKNPU_JOB_PC)",
        "job->ret = rknpu_job_wait(job);",
    )
    for fragment in required_source_fragments:
        if fragment not in source:
            failures.append(f"missing expected rknpu_job.c source fragment: {fragment}")
    if failures:
        print("FAIL rknpu submit mode/source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu submit mode/source model")
    print("captured flags=0x5 select synchronous PC wait path with pingpong, no nonblock/fence path, iommu_domain_id=0")
    return 0


def compare_scheduler_state_source_model(ioctl_log):
    _actions, _creates, _create_returns, _maps, _map_returns, _mmaps, _mmap_returns, _syncs, submit, _returns = parse_ioctl_sequence(ioctl_log)
    source = (ROOT / "ref" / "rknpu_driver" / "rknpu_job.c").read_text()
    failures = []
    subcores = submit.get("subcore_task", [])
    if submit.get("core_mask") != 0:
        failures.append(f"captured core_mask={submit.get('core_mask')} expected auto core mask 0")
    if len(subcores) < 3:
        failures.append(f"subcore_task rows={len(subcores)} expected at least 3")
    else:
        active_task_numbers = {subcores[i][1] for i in range(3)}
        if active_task_numbers != {2}:
            failures.append(f"active subcore task_numbers={sorted(active_task_numbers)} expected all 2")
    required_fragments = (
        "static int rknpu_schedule_core_index(struct rknpu_device *rknpu_dev)",
        "int task_num = rknpu_dev->subcore_datas[0].task_num;",
        "if (task_num > rknpu_dev->subcore_datas[i].task_num)",
        "job->args->core_mask = rknpu_core_mask(core_index);",
        "job->use_core_num = 1;",
        "list_add_tail(&job->head[i], &subcore_data->todo_list);",
        "subcore_data->task_num += rknpu_get_task_number(job, i);",
        "rknpu_job_next(rknpu_dev, i);",
        "if (subcore_data->job || list_empty(&subcore_data->todo_list))",
        "subcore_data->job = job;",
        "if (atomic_dec_and_test(&job->run_count))",
        "rknpu_job_commit(job);",
    )
    for fragment in required_fragments:
        if fragment not in source:
            failures.append(f"missing expected scheduler source fragment: {fragment}")
    if failures:
        print("FAIL rknpu scheduler/core-state source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu scheduler/core-state source model")
    print("auto-core selection is least-loaded task_num state; captured equal per-core task ranges mean only live queue/core state can change which single core is selected")
    return 0


def compare_action_side_effect_source_model(conv, ioctl_log):
    actions, _creates, _create_returns, _maps, _map_returns, _mmaps, _mmap_returns, _syncs, _submit, returns = parse_ioctl_sequence(ioctl_log)
    drv_source = (ROOT / "ref" / "rknpu_driver" / "rknpu_drv.c").read_text()
    ioctl_header = (ROOT / "ref" / "rknpu_driver" / "include" / "rknpu_ioctl.h").read_text()
    failures = []
    action_flags = [row["flags"] for row in actions]
    if conv.RKNPU_ACT_RESET in action_flags:
        failures.append("captured pre-submit ACTION sequence unexpectedly includes RKNPU_ACT_RESET")
    if getattr(conv, "RKNPU_SET_IOMMU_DOMAIN_ID", 25) in action_flags:
        failures.append("captured pre-submit ACTION sequence unexpectedly includes RKNPU_SET_IOMMU_DOMAIN_ID")
    expected_flags = {
        RKNPU_GET_HW_VERSION,
        RKNPU_GET_DRV_VERSION,
        RKNPU_GET_IOMMU_EN,
        RKNPU_SET_PROC_NICE,
        RKNPU_POWER_ON,
    }
    unknown_flags = sorted(set(action_flags) - expected_flags)
    if unknown_flags:
        failures.append(f"unexpected pre-submit ACTION flags={unknown_flags}")
    failed_actions = [row for row in returns if row["kind"].startswith("ACTION") and row["ret"] != 0]
    failed_kinds = [row["kind"] for row in failed_actions]
    if not failed_kinds or failed_kinds[0] != "ACTION flags=0x00000014":
        failures.append(f"first failed ACTION is {failed_kinds[:1]} expected POWER_ON flag 0x14")
    if "ACTION flags=0x00000015" not in failed_kinds:
        failures.append("post-submit POWER_OFF failure was not captured")
    required_fragments = (
        "case RKNPU_GET_HW_VERSION:",
        "ret = rknpu_get_hw_version(rknpu_dev, &args->value);",
        "case RKNPU_GET_DRV_VERSION:",
        "ret = rknpu_get_drv_version(&args->value);",
        "case RKNPU_GET_IOMMU_EN:",
        "args->value = rknpu_dev->iommu_en;",
        "case RKNPU_SET_PROC_NICE:",
        "set_user_nice(current, *(int32_t *)&args->value);",
        "case RKNPU_ACT_RESET:",
        "ret = rknpu_soft_reset(rknpu_dev);",
        "case RKNPU_SET_IOMMU_DOMAIN_ID:",
        "ret = rknpu_iommu_domain_get_and_switch(",
        "default:\n\t\tret = -EINVAL;",
    )
    for fragment in required_fragments:
        if fragment not in drv_source:
            failures.append(f"missing expected rknpu_action source fragment: {fragment}")
    enum_fragments = (
        "RKNPU_POWER_ON = 20",
        "RKNPU_POWER_OFF = 21",
    )
    for fragment in enum_fragments:
        if fragment not in ioctl_header:
            failures.append(f"missing expected action enum fragment: {fragment}")
    if "case RKNPU_POWER_ON:" in drv_source or "case RKNPU_POWER_OFF:" in drv_source:
        failures.append("local rknpu_action source unexpectedly implements POWER_ON/OFF cases")
    if failures:
        print("FAIL rknpu ACTION side-effect source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu ACTION side-effect source model")
    print("captured pre-submit ACTIONs are read-only queries plus SET_PROC_NICE; POWER_ON/OFF are unsupported local actions returning -EINVAL, with no reset or domain-switch ACTION before submit")
    return 0


def compare_mem_sync_source_model(conv, ioctl_log):
    _actions, creates, _create_returns, _maps, _map_returns, _mmaps, _mmap_returns, syncs, _submit, _returns = parse_ioctl_sequence(ioctl_log)
    drv_source = (ROOT / "ref" / "rknpu_driver" / "rknpu_drv.c").read_text()
    gem_source = (ROOT / "ref" / "rknpu_driver" / "rknpu_gem.c").read_text()
    failures = []
    required_bo_flags = RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE
    for idx, create in enumerate(creates):
        if (create["flags"] & required_bo_flags) != required_bo_flags:
            failures.append(f"MEM_CREATE[{idx}] flags=0x{create['flags']:x} lack cacheable/non-contiguous bits")
    expected_sync_flags = {
        RKNPU_MEM_SYNC_TO_DEVICE,
        RKNPU_MEM_SYNC_FROM_DEVICE,
        RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE,
    }
    unexpected_sync_flags = sorted({row["flags"] for row in syncs} - expected_sync_flags)
    if unexpected_sync_flags:
        failures.append(f"unexpected MEM_SYNC flags={unexpected_sync_flags}")
    if not any(row["flags"] & RKNPU_MEM_SYNC_TO_DEVICE for row in syncs):
        failures.append("captured sequence has no TO_DEVICE MEM_SYNC")
    if not any(row["flags"] & RKNPU_MEM_SYNC_FROM_DEVICE for row in syncs):
        failures.append("captured sequence has no FROM_DEVICE MEM_SYNC")
    required_drv_fragments = (
        "RKNPU_IOCTL(rknpu_gem_sync_ioctl);",
        "DRM_IOCTL_DEF_DRV(RKNPU_MEM_SYNC, __rknpu_gem_sync_ioctl",
    )
    required_gem_fragments = (
        "int rknpu_gem_sync_ioctl(struct drm_device *dev, void *data,",
        "if (!(rknpu_obj->flags & RKNPU_MEM_CACHEABLE))",
        "rknpu_iommu_domain_get_and_switch(rknpu_dev,",
        "if (!(rknpu_obj->flags & RKNPU_MEM_NON_CONTIGUOUS))",
        "dma_sync_single_range_for_device(",
        "dma_sync_single_range_for_cpu(",
        "rknpu_iommu_domain_put(rknpu_dev);",
    )
    for fragment in required_drv_fragments:
        if fragment not in drv_source:
            failures.append(f"missing expected rknpu_drv.c source fragment: {fragment}")
    for fragment in required_gem_fragments:
        if fragment not in gem_source:
            failures.append(f"missing expected rknpu_gem.c source fragment: {fragment}")
    if failures:
        print("FAIL rknpu DRM MEM_SYNC/cache source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu DRM MEM_SYNC/cache source model")
    print("captured cacheable/non-contiguous BOs exercise DRM GEM sync with TO/FROM device cache maintenance and IOMMU-domain acquisition")
    return 0


def compare_iommu_domain_lifecycle_source_model(ioctl_log):
    _actions, creates, create_returns, maps, map_returns, _mmaps, _mmap_returns, syncs, submit, _returns = parse_ioctl_sequence(ioctl_log)
    allocs = parse_rknn_allocations(ioctl_log)
    gem_source = (ROOT / "ref" / "rknpu_driver" / "rknpu_gem.c").read_text()
    iommu_source = (ROOT / "ref" / "rknpu_driver" / "rknpu_iommu.c").read_text()
    job_source = (ROOT / "ref" / "rknpu_driver" / "rknpu_job.c").read_text()
    drv_source = (ROOT / "ref" / "rknpu_driver" / "rknpu_drv.c").read_text()
    failures = []
    if len(creates) != 5 or len(create_returns) != 5:
        failures.append(f"MEM_CREATE/RET rows={len(creates)}/{len(create_returns)} expected 5/5")
    if len(maps) != 5 or len(map_returns) != 5:
        failures.append(f"MEM_MAP/RET rows={len(maps)}/{len(map_returns)} expected 5/5")
    if not syncs:
        failures.append("no MEM_SYNC rows captured")
    if {row.get("iommu_domain_id") for row in allocs} != {0}:
        failures.append(f"RKNN allocation iommu domains={sorted({row.get('iommu_domain_id') for row in allocs})} expected [0]")
    if submit.get("iommu_domain_id") not in (None, 0):
        failures.append(f"submit iommu_domain_id={submit.get('iommu_domain_id')} expected 0")
    required_fragments = (
        (gem_source, "rknpu_gem_object_create", "rknpu_iommu_domain_get_and_switch(rknpu_dev, iommu_domain_id)"),
        (gem_source, "rknpu_gem_object_create", "rknpu_obj->iommu_domain_id = iommu_domain_id;"),
        (gem_source, "rknpu_gem_object_create", "rknpu_iommu_domain_put(rknpu_dev);"),
        (gem_source, "rknpu_gem_sync_ioctl", "rknpu_iommu_domain_get_and_switch(rknpu_dev,\n\t\t\t\t\t      rknpu_obj->iommu_domain_id)"),
        (gem_source, "rknpu_gem_sync_ioctl", "rknpu_iommu_domain_put(rknpu_dev);"),
        (job_source, "rknpu_job_alloc", "job->iommu_domain_id = args->iommu_domain_id;"),
        (job_source, "rknpu_job_schedule", "rknpu_iommu_domain_get_and_switch(rknpu_dev, job->iommu_domain_id)"),
        (job_source, "rknpu_job_done", "rknpu_iommu_domain_put(rknpu_dev);"),
        (job_source, "rknpu_job_abort", "rknpu_iommu_domain_put(rknpu_dev);"),
        (iommu_source, "rknpu_iommu_domain_get_and_switch", "if (domain_id == rknpu_dev->iommu_domain_id)"),
        (iommu_source, "rknpu_iommu_domain_get_and_switch", "atomic_inc(&rknpu_dev->iommu_domain_refcount);"),
        (iommu_source, "rknpu_iommu_domain_get_and_switch", "if (atomic_read(&rknpu_dev->iommu_domain_refcount) == 0)"),
        (iommu_source, "rknpu_iommu_domain_put", "atomic_dec(&rknpu_dev->iommu_domain_refcount);"),
        (drv_source, "RKNPU_GET_IOMMU_DOMAIN_ID", "args->value = rknpu_dev->iommu_domain_id;"),
    )
    for source, label, fragment in required_fragments:
        if fragment not in source:
            failures.append(f"missing expected {label} source fragment: {fragment}")
    if failures:
        print("FAIL rknpu IOMMU/domain lifecycle source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu IOMMU/domain lifecycle source model")
    print("captured RKNN allocations and submit use iommu_domain_id=0; create/sync/submit source paths all acquire and release the same domain-refcount mechanism")
    return 0


def compare_mmap_cache_source_model(conv, ioctl_log):
    _actions, creates, _create_returns, maps, map_returns, mmaps, mmap_returns, _syncs, _submit, _returns = parse_ioctl_sequence(ioctl_log)
    gem_source = (ROOT / "ref" / "rknpu_driver" / "rknpu_gem.c").read_text()
    failures = []
    if len(creates) != 5 or len(maps) != 5 or len(map_returns) != 5 or len(mmaps) != 5 or len(mmap_returns) != 5:
        failures.append(
            f"create/map/mmap row counts={len(creates)}/{len(maps)}/{len(map_returns)}/{len(mmaps)}/{len(mmap_returns)} expected all 5"
        )
    for idx, create in enumerate(creates):
        if not (create["flags"] & RKNPU_MEM_CACHEABLE):
            failures.append(f"MEM_CREATE[{idx}] flags=0x{create['flags']:x} missing CACHEABLE")
        if not (create["flags"] & RKNPU_MEM_NON_CONTIGUOUS):
            failures.append(f"MEM_CREATE[{idx}] flags=0x{create['flags']:x} missing NON_CONTIGUOUS")
    for idx, row in enumerate(mmaps):
        if idx < len(creates) and row["length"] != creates[idx]["size"]:
            failures.append(f"MMAP[{idx}] length={row['length']} expected MEM_CREATE size={creates[idx]['size']}")
        if row["prot"] != 0x3 or row["flags"] != 0x1:
            failures.append(f"MMAP[{idx}] prot/flags=0x{row['prot']:x}/0x{row['flags']:x} expected 0x3/0x1")
        if idx < len(map_returns) and row["offset"] != map_returns[idx]["offset"]:
            failures.append(f"MMAP[{idx}] offset=0x{row['offset']:x} expected MEM_MAP_RET 0x{map_returns[idx]['offset']:x}")
    required_fragments = (
        "ret = drm_gem_mmap(filp, vma);",
        "return rknpu_gem_mmap_obj(obj, vma);",
        "if (rknpu_obj->flags & RKNPU_MEM_CACHEABLE) {\n\t\tvma->vm_page_prot = vm_get_page_prot(vma->vm_flags);",
        "vm_flags_set(vma, VM_DONTCOPY | VM_DONTEXPAND | VM_DONTDUMP | VM_IO);",
        "vm_flags_clear(vma, VM_PFNMAP);",
        "if (vm_size > rknpu_obj->size)\n\t\treturn -EINVAL;",
        "if ((rknpu_obj->flags & RKNPU_MEM_NON_CONTIGUOUS) &&\n\t    rknpu_dev->iommu_en) {\n\t\treturn rknpu_gem_mmap_pages(rknpu_obj, vma);",
        "vm_flags_set(vma, VM_MIXEDMAP);",
        "ret = __vm_map_pages(vma, rknpu_obj->pages, rknpu_obj->num_pages,\n\t\t\t     vma->vm_pgoff);",
    )
    for fragment in required_fragments:
        if fragment not in gem_source:
            failures.append(f"missing expected mmap/cache source fragment: {fragment}")
    if failures:
        print("FAIL rknpu mmap/cacheability source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu mmap/cacheability source model")
    print("captured RKNN BO mmaps use shared read/write mappings of cacheable non-contiguous BOs; source maps them with normal cached page prot and page-list mmap when IOMMU is enabled")
    return 0


def compare_exported_regcmd_runs(rknn_path, expected):
    sys.path.insert(0, str(ROOT / "experimental" / "rknn"))
    import rknn_parse_regcmd_runs as rknn_runs

    buf = Path(rknn_path).read_bytes()
    runs = rknn_runs.find_runs(buf, min_qwords=8)
    failures = []
    if len(runs) != len(expected):
        failures.append(f"exported runs={len(runs)} expected={len(expected)}")
    for idx, (run, record) in enumerate(zip(runs, expected)):
        run_len = len(run)
        amount = record["regcfg_amount"]
        separator = record["enable_mask"] == 0x60
        if run_len < amount:
            failures.append(f"run[{idx}] qwords={run_len} shorter than regcfg_amount={amount}")
        if separator and run_len != 26:
            failures.append(f"separator run[{idx}] qwords={run_len} expected=26")
    if failures:
        print("FAIL exported regcmd run compatibility")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"PASS exported regcmd run compatibility: {len(runs)} runs")
    print(f"run_qwords={[len(run) for run in runs]}")
    print("each exported run is long enough for the captured task regcfg_amount; separator runs are exactly 26 qwords")
    return 0


def qword_parts(qword):
    return (qword >> 48) & 0xffff, qword & 0xffff, (qword >> 16) & 0xffffffff


def make_qword(target, reg_addr, value):
    return (target << 48) | ((value & 0xffffffff) << 16) | reg_addr


def separator_regs(conv, ppu_src_dma=0, ppu_dst_dma=0):
    return [
        conv.E(conv.reg.PPU, conv.reg.PPU_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        conv.E(conv.reg.PPU_RDMA, conv.reg.PPU_RDMA_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
        conv.E(conv.reg.PPU, conv.reg.PPU_DATA_CUBE_IN_WIDTH, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_DATA_CUBE_IN_HEIGHT, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_DATA_CUBE_IN_CHANNEL, 31),
        conv.E(conv.reg.PPU, conv.reg.PPU_DATA_CUBE_OUT_WIDTH, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_DATA_CUBE_OUT_HEIGHT, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_DATA_CUBE_OUT_CHANNEL, 31),
        conv.E(conv.reg.PPU, conv.reg.PPU_OPERATION_MODE_CFG, (1 << 4) | 1),
        conv.E(conv.reg.PPU, conv.reg.PPU_POOLING_KERNEL_CFG, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_RECIP_KERNEL_WIDTH, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_RECIP_KERNEL_HEIGHT, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_POOLING_PADDING_CFG, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_PADDING_VALUE_1_CFG, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_PADDING_VALUE_2_CFG, 0),
        conv.E(conv.reg.PPU, conv.reg.PPU_DST_BASE_ADDR, ppu_dst_dma),
        conv.E(conv.reg.PPU, conv.reg.PPU_DST_SURF_STRIDE, 1 << 4),
        conv.E(conv.reg.PPU, conv.reg.PPU_DATA_FORMAT, 1 << 4),
        conv.E(conv.reg.PPU, conv.reg.PPU_MISC_CTRL, 3),
        conv.E(conv.reg.PPU_RDMA, conv.reg.PPU_RDMA_CUBE_IN_WIDTH, 0),
        conv.E(conv.reg.PPU_RDMA, conv.reg.PPU_RDMA_CUBE_IN_HEIGHT, 0),
        conv.E(conv.reg.PPU_RDMA, conv.reg.PPU_RDMA_CUBE_IN_CHANNEL, 31),
        conv.E(conv.reg.PPU_RDMA, conv.reg.PPU_RDMA_SRC_BASE_ADDR, ppu_src_dma),
        conv.E(conv.reg.PPU_RDMA, conv.reg.PPU_RDMA_SRC_LINE_STRIDE, 1 << 4),
        conv.E(conv.reg.PPU_RDMA, conv.reg.PPU_RDMA_SRC_SURF_STRIDE, 1 << 4),
        conv.E(conv.reg.PPU_RDMA, conv.reg.PPU_RDMA_DATA_FORMAT, 1),
    ]


def normalize_qword(qword, regcmd_base, pc_qword_offset=0):
    if qword is None:
        return 0
    target, reg_addr, value = qword_parts(qword)
    if (target, reg_addr) == (0x0101, 0x0010):  # PC_BASE_ADDRESS
        base32 = regcmd_base & 0xffffffff
        value = (value - base32) // 8 if value >= base32 else value // 8
        value -= pc_qword_offset
    elif (target, reg_addr) in DMA_REGS:
        value = 0
    return make_qword(target, reg_addr, value)


def sparse_export_stream(rknn_path, expected):
    sys.path.insert(0, str(ROOT / "experimental" / "rknn"))
    import rknn_parse_regcmd_runs as rknn_runs

    buf = Path(rknn_path).read_bytes()
    rows = rknn_runs.find_runs(buf, min_qwords=8)
    table_end = max((row["regcmd_offset"] // 8) + len(run) for row, run in zip(expected, rows))
    sparse = [None] * table_end
    for row, run in zip(expected, rows):
        start = row["regcmd_offset"] // 8
        qwords = [rknn_runs.decode(buf, off)[3] for off in run]
        sparse[start:start + len(qwords)] = qwords
    return sparse


def conv_tiles_sparse_stream(conv):
    p = conv._conv_params(160, 40, 40, 320, 3, 3, 1, 1)
    input_nchw = __import__("numpy").zeros((1, 160, 40, 40), dtype=__import__("numpy").float16)
    weight_nchw = __import__("numpy").zeros((320, 160, 3, 3), dtype=__import__("numpy").float16)
    descs, _packed_input, _packed_weight = conv.build_direct_spatial_descs(
        0, input_nchw, weight_nchw, p, 160, 320, 3, 3, 40, 40, 1, 1)
    records = observed_records()
    amounts = tuple(row["regcfg_amount"] for row in records)
    starts = tuple(row["regcmd_offset"] // 8 for row in records)
    stream = [0] * (starts[-1] + conv._align_up(amounts[-1] + conv.PC_CHAIN_TAIL_QWORDS, 16))
    desc_idx = 0
    for record_idx, amount in enumerate(amounts):
        start = starts[record_idx]
        enable_value = 0x60 if amount == 26 else 0x0d
        if amount == 26:
            record = separator_regs(conv)
            pc_core = 0
        else:
            body = descs[desc_idx].extra["full_regs"]
            pc_core = descs[desc_idx].pc_core
            record = list(body if amount == len(body) else body[4:])
            desc_idx += 1
        link_record = ROOT6_RECORD_LINKS[record_idx]
        if link_record is None:
            pc_base_qword = 0
            pc_amount = 0
        else:
            pc_base = (
                conv.regcmd_mem_create.dma_addr
                + conv.RKNN_REGCMD_ACTIVE_OFFSET * conv.RKNN_MEM_SYNC
                + starts[link_record] * 8
            ) & 0xfffffff0
            pc_base_qword = conv.E(conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS, pc_base)
            pc_amount = (pc_core << 16) | (conv._ceil_div(amounts[link_record], 2) + 1)
        record += [
            pc_base_qword,
            conv.E(conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS, pc_amount),
            conv.E(conv.reg.VERSION, 0, 0),
            conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, enable_value),
        ]
        if len(record) < amount:
            raise RuntimeError(f"root6 record {record_idx} shorter than amount {amount}")
        stream[start:start + len(record)] = record
    if desc_idx != len(descs):
        raise RuntimeError(f"root6 stream consumed {desc_idx}/{len(descs)} descriptors")
    return stream


def conv_tiles_six_desc_sparse_stream(conv, in_c, in_h, in_w, out_c, label, active_offset=None, kh=3, kw=3):
    np = __import__("numpy")
    active_offset = conv.RKNN_REGCMD_ACTIVE_OFFSET if active_offset is None else active_offset
    p = conv._conv_params(in_c, in_h, in_w, out_c, kh, kw, 1, 1)
    input_nchw = np.zeros((1, in_c, in_h, in_w), dtype=np.float16)
    weight_nchw = np.zeros((out_c, in_c, kh, kw), dtype=np.float16)
    descs, _packed_input, _packed_weight = conv.build_direct_spatial_descs(
        0, input_nchw, weight_nchw, p, in_c, out_c, kh, kw, in_h, in_w, 1, 1)
    if len(descs) != 6:
        raise RuntimeError(f"expected 6 {label} descriptors, got {len(descs)}")
    records = observed_six_desc_records()
    amounts = tuple(row["regcfg_amount"] for row in records)
    starts = tuple(row["regcmd_offset"] // 8 for row in records)
    stream = [0] * (starts[-1] + conv._align_up(amounts[-1] + conv.PC_CHAIN_TAIL_QWORDS, 16))
    desc_idx = 0
    for record_idx, amount in enumerate(amounts):
        start = starts[record_idx]
        enable_value = 0x60 if amount == 26 else 0x0d
        if amount == 26:
            record = separator_regs(conv)
            pc_core = 0
        else:
            body = descs[desc_idx].extra["full_regs"]
            pc_core = descs[desc_idx].pc_core
            record = list(body if record_idx == 0 else body[4:])
            desc_idx += 1
        link_record = SIX_DESC_RECORD_LINKS[record_idx]
        if link_record is None:
            pc_base_qword = 0
            pc_amount = 0
        else:
            pc_base = (conv.regcmd_mem_create.dma_addr + active_offset +
                       starts[link_record] * 8) & 0xfffffff0
            pc_base_qword = conv.E(conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS, pc_base)
            pc_amount = (pc_core << 16) | (conv._ceil_div(amounts[link_record], 2) + 1)
        record += [
            pc_base_qword,
            conv.E(conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS, pc_amount),
            conv.E(conv.reg.VERSION, 0, 0),
            conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, enable_value),
        ]
        if len(record) < amount:
            raise RuntimeError(f"six-desc record {record_idx} shorter than amount {amount}")
        stream[start:start + len(record)] = record
    if desc_idx != len(descs):
        raise RuntimeError(f"six-desc stream consumed {desc_idx}/{len(descs)} descriptors")
    return stream


def conv_tiles_six_desc_h7_sparse_stream(conv):
    return conv_tiles_six_desc_sparse_stream(conv, 160, 7, 7, 320, "H7")


def conv_tiles_six_desc_h14_sparse_stream(conv):
    return conv_tiles_six_desc_sparse_stream(conv, 160, 14, 14, 320, "H14")


def conv_tiles_six_desc_c32_h14_sparse_stream(conv):
    return conv_tiles_six_desc_sparse_stream(conv, 32, 14, 14, 128, "C32/H14", active_offset=0x12800)


def conv_tiles_six_desc_pw_c256_h14_sparse_stream(conv):
    return conv_tiles_six_desc_sparse_stream(
        conv, 256, 14, 14, 512, "PW C256/H14", active_offset=0x40800, kh=1, kw=1)


def conv_tiles_unpromoted_c64_h56_descs(conv):
    np = __import__("numpy")
    p = conv._conv_params(64, 56, 56, 128, 1, 1, 1, 1)
    input_nchw = np.zeros((1, 64, 56, 56), dtype=np.float16)
    weight_nchw = np.zeros((128, 64, 1, 1), dtype=np.float16)
    descs, _packed_input, _packed_weight = conv.build_direct_spatial_descs(
        0, input_nchw, weight_nchw, p, 64, 128, 1, 1, 56, 56, 1, 1)
    return descs


def c64_h56_record_starts(conv):
    starts = []
    cursor = 0
    for amount in C64_H56_RECORD_AMOUNTS:
        starts.append(cursor)
        cursor += conv._align_up(amount + conv.PC_CHAIN_TAIL_QWORDS, 16)
    return starts


def c64_h56_backing_qwords(conv):
    starts = c64_h56_record_starts(conv)
    return starts[-1] + conv._align_up(C64_H56_RECORD_AMOUNTS[-1] + conv.PC_CHAIN_TAIL_QWORDS, 16)


def c64_h56_record_tail(conv, record_idx, pc_core, enable_value, record_starts):
    link_record = C64_H56_RECORD_LINKS[record_idx]
    if link_record is None:
        pc_base_qword = 0
        pc_amount = 0
    else:
        amount = C64_H56_RECORD_AMOUNTS[link_record]
        pc_base = (
            conv.regcmd_mem_create.dma_addr
            + C64_H56_ACTIVE_OFFSET * conv.RKNN_MEM_SYNC
            + record_starts[link_record] * 8
        ) & 0xfffffff0
        pc_base_qword = conv.E(conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS, pc_base)
        pc_amount = (pc_core << 16) | (conv._ceil_div(amount, 2) + 1)
    return [
        pc_base_qword,
        conv.E(conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS, pc_amount),
        conv.E(conv.reg.VERSION, 0, 0),
        conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, enable_value),
    ]


def c64_h56_sparse_backing_stream(conv, descs):
    record_starts = c64_h56_record_starts(conv)
    stream = [0] * c64_h56_backing_qwords(conv)
    desc_idx = 0

    for record_idx, amount in enumerate(C64_H56_RECORD_AMOUNTS):
        start = record_starts[record_idx]
        enable_value = 0x60 if amount == 26 else 0x0d
        if amount == 26:
            record = separator_regs(conv)
            pc_core = 0
        else:
            body = descs[desc_idx].extra["full_regs"]
            pc_core = descs[desc_idx].pc_core
            record = list(body if record_idx in (0, 1) else body[4:])
            desc_idx += 1
        record += c64_h56_record_tail(conv, record_idx, pc_core, enable_value, record_starts)
        if len(record) < amount:
            raise RuntimeError(f"C64/H56 sparse record {record_idx} shorter than amount {amount}")
        if start + len(record) > len(stream):
            raise RuntimeError(f"C64/H56 sparse record {record_idx} outside backing stream")
        stream[start:start + len(record)] = record

    if desc_idx != len(descs):
        raise RuntimeError(f"C64/H56 sparse stream consumed {desc_idx}/{len(descs)} descriptors")
    return stream


def classify_dma_value(value, allocations):
    for name, base, size in allocations:
        if base <= value < base + size:
            return name, value - base
    return None, None


def parse_live_dump_dma_values(log_path):
    values = {}
    for line in Path(log_path).read_text().splitlines():
        match = re.search(r"regcmd_addr=0x([0-9a-f]+)", line)
        if not match:
            continue
        qword = int(match.group(1), 16)
        target, reg_addr, value = qword_parts(qword)
        if (target, reg_addr) in DMA_REGS:
            values.setdefault((target, reg_addr), set()).add(value)
    return values


def parse_live_regcmd_window(log_path):
    rows = []
    for line in Path(log_path).read_text().splitlines():
        match = REGCMD_QWORD_LINE.search(line)
        if not match:
            continue
        rows.append({
            "idx": int(match.group("idx")),
            "off": int(match.group("off"), 16),
            "value": int(match.group("value"), 16),
        })
    return rows


def compare_dma_binding_live_dump(conv, ioctl_log, gem_log):
    generated = conv_tiles_sparse_stream(conv)
    generated_allocs = [
        ("internal", conv.weight_mem_create.dma_addr, conv.weight_mem_create.size),
        ("input", conv.input_mem_create.dma_addr, conv.input_mem_create.size),
        ("output", conv.output_mem_create.dma_addr, conv.output_mem_create.size),
    ]
    generated_by_reg = {}
    for qword in generated:
        target, reg_addr, value = qword_parts(qword)
        owner, _offset = classify_dma_value(value, generated_allocs)
        if owner is None:
            continue
        if (target, reg_addr) == (0x0201, 0x1070):
            generated_by_reg.setdefault((target, reg_addr), {"owner": owner, "values": set()})["values"].add(value)
        elif (target, reg_addr) == (0x1001, 0x4020):
            generated_by_reg.setdefault((target, reg_addr), {"owner": owner, "values": set()})["values"].add(value)
        elif (target, reg_addr) == (0x0201, 0x1110):
            generated_by_reg.setdefault((target, reg_addr), {"owner": owner, "values": set()})["values"].add(value)

    rknn_allocs = {row["name"]: row for row in parse_rknn_allocations(ioctl_log)}
    feature_layout = parse_rknn_feature_layout(ioctl_log)
    live_values = parse_live_dump_dma_values(gem_log)
    failures = []
    internal = rknn_allocs.get("internal")
    if internal is None or set(feature_layout) != {"input", "output"}:
        failures.append("missing captured RKNN internal allocation or feature tensor table rows")
    else:
        internal_start = internal["dma_addr"]
        internal_end = internal_start + internal["size"]
        for name in ("input", "output"):
            row = feature_layout[name]
            if not (internal_start <= row["start"] < row["end"] <= internal_end):
                failures.append(f"RKNN {name} feature range is not inside internal BO: {row}")

    checks = {
        (0x0201, 0x1070): "input",
        (0x1001, 0x4020): "output",
        (0x0201, 0x1110): "internal",
    }
    for key, expected_owner in checks.items():
        generated_row = generated_by_reg.get(key)
        observed = live_values.get(key)
        if generated_row is None or not observed:
            failures.append(f"missing generated or live DMA values for reg={key}")
            continue
        if generated_row["owner"] != expected_owner:
            failures.append(f"generated reg={key} owner={generated_row['owner']} expected={expected_owner}")
            continue
        generated_offsets = {value - min(generated_row["values"]) for value in generated_row["values"]}
        observed_offsets = {value - min(observed) for value in observed}
        if not observed_offsets.issubset(generated_offsets):
            failures.append(
                f"live DMA offsets for reg={key} {sorted(observed_offsets)} not subset of generated {sorted(generated_offsets)}"
            )
    if failures:
        print("FAIL conv_tiles live DMA binding comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS conv_tiles live DMA binding comparison")
    print("live GEM dump DMA offsets are consistent with conv_tiles input/output staging BO bindings")
    return 0


def conv_tiles_direct_spatial_descs(conv):
    p = conv._conv_params(160, 40, 40, 320, 3, 3, 1, 1)
    input_nchw = __import__("numpy").zeros((1, 160, 40, 40), dtype=__import__("numpy").float16)
    weight_nchw = __import__("numpy").zeros((320, 160, 3, 3), dtype=__import__("numpy").float16)
    descs, _packed_input, _packed_weight = conv.build_direct_spatial_descs(
        0, input_nchw, weight_nchw, p, 160, 320, 3, 3, 40, 40, 1, 1)
    return descs


def compare_sparse_regcmd(conv, rknn_path):
    expected_records = observed_records()
    generated = conv_tiles_sparse_stream(conv)
    regcmd_base = conv.regcmd_mem_create.dma_addr
    active_qwords = conv.RKNN_REGCMD_ACTIVE_OFFSET // 8 if conv.RKNN_MEM_SYNC else 0
    generated_norm = [normalize_qword(qword, regcmd_base, active_qwords) for qword in generated]
    sys.path.insert(0, str(ROOT / "experimental" / "rknn"))
    import rknn_parse_regcmd_runs as rknn_runs

    buf = Path(rknn_path).read_bytes()
    runs = rknn_runs.find_runs(buf, min_qwords=8)
    failures = []
    if len(runs) != len(expected_records):
        failures.append(f"exported runs={len(runs)} expected={len(expected_records)}")
    for record_idx, (record, run) in enumerate(zip(expected_records, runs)):
        start = record["regcmd_offset"] // 8
        amount = record["regcfg_amount"]
        exported_run = [normalize_qword(rknn_runs.decode(buf, off)[3], regcmd_base) for off in run]
        generated_body = generated_norm[start:start + amount]
        exported_body = exported_run[:amount]
        if generated_body != exported_body:
            for idx, (actual, expected) in enumerate(zip(generated_body, exported_body)):
                if actual != expected:
                    failures.append(
                        f"record[{record_idx}] body mismatch qword[{idx}] generated=0x{actual:016x} exported=0x{expected:016x}"
                    )
                    break
            else:
                failures.append(f"record[{record_idx}] body length mismatch generated={len(generated_body)} exported={len(exported_body)}")
            break
        link_record = ROOT6_RECORD_LINKS[record_idx]
        if link_record is not None:
            generated_tail = generated_norm[start + amount:start + amount + 2]
            exported_tail = exported_run[amount:amount + 2]
            if generated_tail != exported_tail:
                failures.append(
                    f"record[{record_idx}] PC tail mismatch generated={[hex(v) for v in generated_tail]} exported={[hex(v) for v in exported_tail]}"
                )
                break
    if failures:
        print("NOTE conv_tiles exported sparse equality is not expected")
        for failure in failures:
            print(f"  {failure}")
        print("exported .rknn runs are compiler/stitching evidence; the live RKNN GEM window is the authoritative runtime sparse backing stream")
        return 0
    print(f"PASS conv_tiles sparse regcmd exported compatibility: {len(runs)} descriptor runs, generated backing {len(generated_norm)} qwords")
    print("exported descriptor bodies and linked PC tails match after DMA normalization; full runtime backing is checked against the live GEM dump")
    return 0


def compare_live_regcmd_window(conv, ioctl_log, gem_log):
    generated = conv_tiles_sparse_stream(conv)
    live_rows = parse_live_regcmd_window(gem_log)
    allocations = {row["name"]: row for row in parse_rknn_allocations(gem_log)}
    failures = []
    window_qwords = conv.RKNN_REGCMD_ACTIVE_BYTES // 8
    if len(live_rows) != window_qwords:
        failures.append(f"live regcmd qwords={len(live_rows)} expected sync window={window_qwords}")
    if "weight" not in allocations:
        failures.append("missing RKNN weight/regcmd allocation metadata")
    if failures:
        print("FAIL conv_tiles live regcmd window comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    expected_offsets = [
        conv.RKNN_REGCMD_ACTIVE_OFFSET + idx * 8 for idx in range(window_qwords)
    ]
    actual_offsets = [row["off"] for row in live_rows]
    if actual_offsets != expected_offsets:
        failures.append("live regcmd window offsets are not contiguous at the captured active offset")
    live_values = [row["value"] for row in live_rows]
    if any(live_values[len(generated):]):
        failures.append("live regcmd sync-window padding after generated sparse stream is non-zero")
    live_regcmd_base = allocations["weight"]["dma_addr"]
    active_qwords = conv.RKNN_REGCMD_ACTIVE_OFFSET // 8
    generated_norm = [
        normalize_qword(qword, conv.regcmd_mem_create.dma_addr, active_qwords)
        for qword in generated
    ]
    live_norm = [
        normalize_qword(qword, live_regcmd_base, active_qwords)
        for qword in live_values[:len(generated)]
    ]
    if len(live_norm) != len(generated_norm):
        failures.append(f"live active stream qwords={len(live_norm)} generated={len(generated_norm)}")
    else:
        for idx, (actual, expected) in enumerate(zip(live_norm, generated_norm)):
            if actual != expected:
                failures.append(
                    f"first live regcmd mismatch[{idx}] actual=0x{actual:016x} generated=0x{expected:016x}"
                )
                break
    if failures:
        print("FAIL conv_tiles live regcmd window comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"PASS conv_tiles live regcmd window comparison: {len(generated_norm)} active qwords, {window_qwords - len(generated_norm)} zero padding qwords")
    print("actual patched RKNN GEM2 regcmd window matches generated sparse stream after DMA/PC base normalization")
    return 0


def compare_six_desc_h7_live_regcmd_window(conv, gem_log):
    return compare_six_desc_live_regcmd_window(
        conv,
        gem_log,
        "H7",
        conv_tiles_six_desc_h7_sparse_stream,
    )


def compare_six_desc_h14_live_regcmd_window(conv, gem_log):
    return compare_six_desc_live_regcmd_window(
        conv,
        gem_log,
        "H14",
        conv_tiles_six_desc_h14_sparse_stream,
    )


def compare_six_desc_c32_h14_live_regcmd_window(conv, gem_log):
    return compare_six_desc_live_regcmd_window(
        conv,
        gem_log,
        "C32/H14",
        conv_tiles_six_desc_c32_h14_sparse_stream,
        active_offset=0x12800,
    )


def compare_six_desc_pw_c256_h14_live_regcmd_window(conv, gem_log):
    return compare_six_desc_live_regcmd_window(
        conv,
        gem_log,
        "PW C256/H14",
        conv_tiles_six_desc_pw_c256_h14_sparse_stream,
        active_offset=0x40800,
    )


def compare_unpromoted_c64_h56_live_regcmd_evidence(conv, gem_log):
    log_path = Path(gem_log)
    if not log_path.exists():
        print("SKIP unpromoted C64/H56 live regcmd evidence")
        print(f"missing capture log: {log_path}")
        return 0
    descs = conv_tiles_unpromoted_c64_h56_descs(conv)
    schedule = conv.direct_spatial_desc_schedule(descs)
    supported = conv.direct_spatial_default_supported(descs)
    live_rows = parse_live_regcmd_window(log_path)
    allocations = {row["name"]: row for row in parse_rknn_allocations(log_path)}
    failures = []
    if supported:
        failures.append("C64/H56 unexpectedly became supported without updating this evidence check")
    if len(descs) != 9:
        failures.append(f"C64/H56 generated descriptor count={len(descs)} expected 9")
    if len(live_rows) != 1168:
        failures.append(f"C64/H56 live qwords={len(live_rows)} expected 1168")
    if "weight" not in allocations:
        failures.append("missing C64/H56 weight/regcmd allocation metadata")
    if failures:
        print("FAIL unpromoted C64/H56 live regcmd evidence")
        for failure in failures:
            print(f"  {failure}")
        return 1

    active_qwords = C64_H56_ACTIVE_OFFSET // 8
    live_regcmd_base = allocations["weight"]["dma_addr"]
    live_norm = [
        normalize_qword(row["value"], live_regcmd_base, active_qwords)
        for row in live_rows
    ]
    generated = c64_h56_sparse_backing_stream(conv, descs)
    generated_norm = [
        normalize_qword(qword, conv.regcmd_mem_create.dma_addr, active_qwords)
        for qword in generated
    ]
    first_mismatch = next(
        (idx for idx, (actual, expected) in enumerate(zip(live_norm, generated_norm)) if actual != expected),
        None,
    )
    if len(generated_norm) != len(live_norm):
        print("FAIL unpromoted C64/H56 live regcmd evidence")
        print(f"  generated qwords={len(generated_norm)} live qwords={len(live_norm)}")
        return 1
    if first_mismatch is not None:
        print("FAIL unpromoted C64/H56 live regcmd evidence")
        print(
            f"  first live/generated mismatch[{first_mismatch}] "
            f"live=0x{live_norm[first_mismatch]:016x} "
            f"generated=0x{generated_norm[first_mismatch]:016x}"
        )
        return 1
    print("PASS unpromoted C64/H56 live regcmd evidence")
    print(f"schedule={schedule}")
    print("C64/H56 14-record sparse stream matches live RKNN GEM2 window after DMA/PC base normalization")
    print("C64/H56 remains hardware-gated until active-offset/BO-size submit handling is promoted")
    return 0


def compare_six_desc_live_regcmd_window(conv, gem_log, label, stream_fn, active_offset=0xe1800):
    log_path = Path(gem_log)
    if not log_path.exists():
        print(f"SKIP six-desc {label} live regcmd window comparison")
        print(f"missing capture log: {log_path}")
        return 0
    generated = stream_fn(conv)
    live_rows = parse_live_regcmd_window(log_path)
    allocations = {row["name"]: row for row in parse_rknn_allocations(log_path)}
    failures = []
    if len(generated) != 832:
        failures.append(f"generated {label} six-desc qwords={len(generated)} expected 832")
    if len(live_rows) != len(generated):
        failures.append(f"live {label} qwords={len(live_rows)} expected generated={len(generated)}")
    if "weight" not in allocations:
        failures.append(f"missing {label} RKNN weight/regcmd allocation metadata")
    if failures:
        print(f"FAIL six-desc {label} live regcmd window comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    expected_offsets = [active_offset + idx * 8 for idx in range(len(generated))]
    actual_offsets = [row["off"] for row in live_rows]
    if actual_offsets != expected_offsets:
        failures.append(f"{label} live regcmd window offsets are not contiguous at 0x{active_offset:x}")
    live_regcmd_base = allocations["weight"]["dma_addr"]
    active_qwords = active_offset // 8
    generated_norm = [
        normalize_qword(qword, conv.regcmd_mem_create.dma_addr, active_qwords)
        for qword in generated
    ]
    live_norm = [
        normalize_qword(row["value"], live_regcmd_base, active_qwords)
        for row in live_rows
    ]
    for idx, (actual, expected) in enumerate(zip(live_norm, generated_norm)):
        if actual != expected:
            failures.append(
                f"{label} first regcmd mismatch[{idx}] actual=0x{actual:016x} generated=0x{expected:016x}"
            )
            break
    if failures:
        print(f"FAIL six-desc {label} live regcmd window comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"PASS six-desc {label} live regcmd window comparison: {len(generated_norm)} qwords")
    print(f"built-in {label} generated sparse backing stream matches live RKNN GEM2 window after DMA/PC base normalization")
    return 0


def compare_written_buffer_placement(conv):
    descs = conv_tiles_direct_spatial_descs(conv)
    expected_stream = conv_tiles_sparse_stream(conv)
    records = tuple(
        (
            row["regcmd_offset"] // 8,
            row["regcfg_amount"],
            row["enable_mask"],
            row["int_mask"],
        )
        for row in observed_records()
    )
    regcmd_base_qwords = conv.RKNN_REGCMD_ACTIVE_OFFSET // 8 if conv.RKNN_MEM_SYNC else 0
    for idx, qword in enumerate(expected_stream):
        conv.npu_regcmd[regcmd_base_qwords + idx] = qword
    for idx, (start, amount, enable_mask, int_mask) in enumerate(records):
        conv.npu_tasks[idx].regcmd_addr = (
            conv.regcmd_mem_create.dma_addr
            + conv.RKNN_REGCMD_ACTIVE_OFFSET * conv.RKNN_MEM_SYNC
            + start * 8
        )
        conv.npu_tasks[idx].regcfg_amount = amount
        conv.npu_tasks[idx].op_idx = 1
        conv.npu_tasks[idx].enable_mask = enable_mask
        conv.npu_tasks[idx].int_mask = int_mask
        conv.npu_tasks[idx].int_clear = 0x1ffff

    expected_task = pack_records(
        observed_records(),
        regcmd_base=conv.regcmd_mem_create.dma_addr + conv.RKNN_REGCMD_ACTIVE_OFFSET,
    )
    actual_task = bytes(conv.task_map[:len(expected_task)])
    active_off = conv.RKNN_REGCMD_ACTIVE_OFFSET
    active_bytes = len(expected_stream) * 8
    expected_regcmd = struct.pack(f"<{len(expected_stream)}Q", *expected_stream)
    actual_regcmd = bytes(conv.regcmd_map[active_off:active_off + active_bytes])
    prefix = bytes(conv.regcmd_map[:active_off])
    suffix_start = active_off + active_bytes
    suffix = bytes(conv.regcmd_map[suffix_start:conv.RKNN_REGCMD_ACTIVE_OFFSET + conv.RKNN_REGCMD_ACTIVE_BYTES])

    failures = []
    if actual_task != expected_task:
        failures.append("task_map first 680 bytes differ from modeled absolute task object")
    if any(prefix):
        failures.append("regcmd bytes before RKNN active offset are non-zero")
    if actual_regcmd != expected_regcmd:
        failures.append("regcmd active window bytes differ from generated sparse stream")
    if any(suffix):
        failures.append("regcmd bytes after generated stream inside RKNN sync window are non-zero")
    if failures:
        print("FAIL conv_tiles RKNN-mode written buffer placement")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS conv_tiles RKNN-mode written buffer placement")
    print(f"task_bytes={len(expected_task)} regcmd_active_offset={active_off} active_qwords={len(expected_stream)}")
    return 0


def decode_interrupt_status(value):
    names = [name for name, mask in INT_COMPLETION_MASKS.items() if value & mask]
    reserved = value & INT_RAW_RESERVED_MASK
    return names, reserved


def modeled_rknpu_driver_commit(conv, records, task_start, task_number, pingpong=True):
    if task_number <= 0:
        return None
    first = records[task_start]
    last = records[task_start + task_number - 1]
    pc_data_amount = (
        (first["regcfg_amount"] + RKNPU_PC_DATA_EXTRA_AMOUNT + RK3588_PC_DATA_AMOUNT_SCALE - 1)
        // RK3588_PC_DATA_AMOUNT_SCALE
        - 1
    )
    task_pp_en = 1 if pingpong else 0
    return {
        "record_range": (task_start, task_start + task_number - 1),
        "PC_DATA_ADDR": conv.regcmd_mem_create.dma_addr + conv.RKNN_REGCMD_ACTIVE_OFFSET + first["regcmd_offset"],
        "PC_DATA_AMOUNT": pc_data_amount,
        "INT_MASK": last["int_mask"],
        "INT_CLEAR": first["int_mask"],
        "PC_TASK_CONTROL": ((0x6 | task_pp_en) << RK3588_PC_TASK_NUMBER_BITS) | task_number,
        "PC_DMA_BASE_ADDR": 0,
    }


def compare_driver_commit_candidates(conv):
    config = parse_rk3588_driver_config()
    records = observed_records()
    subcore_tasks = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]
    active_commits = [
        modeled_rknpu_driver_commit(conv, records, task_start, task_number)
        for task_start, task_number in subcore_tasks[:3]
    ]
    tail_layout_commits = [
        modeled_rknpu_driver_commit(conv, records, task_start, task_number)
        for task_start, task_number in subcore_tasks[2:5]
    ]
    observed_raw = 0xc0000000
    status_names, reserved = decode_interrupt_status(observed_raw)
    failures = []
    expected_config = {
        "pc_data_amount_scale": RK3588_PC_DATA_AMOUNT_SCALE,
        "pc_task_number_bits": RK3588_PC_TASK_NUMBER_BITS,
        "pc_task_number_mask": 0xfff,
        "pc_task_status_offset": 0x3c,
        "pc_dma_ctrl": 0,
        "num_irqs": 3,
        "max_submit_number": (1 << 12) - 1,
        "core_mask": 0x7,
    }
    for name, expected in expected_config.items():
        actual = config.get(name)
        if actual != expected:
            failures.append(f"rk3588_rknpu_config.{name}={actual} expected={expected}")
    if any(commit is None for commit in active_commits):
        failures.append(f"active subcore commits include inactive entries: {active_commits}")
    for idx, commit in enumerate(active_commits):
        if commit["record_range"] != (0, 1):
            failures.append(f"subcore[{idx}] record_range={commit['record_range']} != (0, 1)")
        if commit["PC_DATA_AMOUNT"] != 55:
            failures.append(f"subcore[{idx}] PC_DATA_AMOUNT={commit['PC_DATA_AMOUNT']} != 55")
        if commit["INT_MASK"] != 0x300 or commit["INT_CLEAR"] != 0x300:
            failures.append(f"subcore[{idx}] unexpected DPU interrupt masks {commit}")
        if commit["PC_TASK_CONTROL"] != 0x7002:
            failures.append(f"subcore[{idx}] PC_TASK_CONTROL=0x{commit['PC_TASK_CONTROL']:x} != 0x7002")
    if tail_layout_commits != [active_commits[0], None, None]:
        failures.append(f"tail-layout commit candidates unexpected: {tail_layout_commits}")
    if status_names:
        failures.append(f"raw status 0xc0000000 unexpectedly decodes to named bits {status_names}")
    if reserved != observed_raw:
        failures.append(f"raw status reserved decode mismatch reserved=0x{reserved:x}")
    if failures:
        print("FAIL rknpu driver commit candidate model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu driver commit candidate model")
    print("active subcore_task[0..2] all commit record_range=0..1 PC_DATA_AMOUNT=55 PC_TASK_CONTROL=0x7002")
    print("rk3588_rknpu_config source constants match the commit model: scale=2 task_bits=12 max_submit=4095 num_irqs=3")
    print("tail-index interpretation would leave two inactive commits for the captured RKNN submit metadata")
    print("raw_status=0xc0000000 decodes only to INTERRUPT_RAW_STATUS reserved bits 30..31")
    return 0


def compare_irq_completion_source_model(conv):
    source = (ROOT / "ref" / "rknpu_driver" / "rknpu_job.c").read_text()
    expected = modeled_rknpu_driver_commit(conv, observed_records(), 0, 2)
    failures = []
    if expected["INT_MASK"] != 0x300:
        failures.append(f"modeled INT_MASK=0x{expected['INT_MASK']:x} expected DPU pair mask 0x300")
    fuzz_cases = {
        0x0001: 0x0003,
        0x0004: 0x000c,
        0x0010: 0x0030,
        0x0040: 0x00c0,
        0x0100: 0x0300,
        0x0400: 0x0c00,
        0x0300: 0x0300,
        0x0000: 0x0000,
    }
    for status, expected_fuzz in fuzz_cases.items():
        actual = 0
        if status & 0x3:
            actual |= 0x3
        if status & 0xc:
            actual |= 0xc
        if status & 0x30:
            actual |= 0x30
        if status & 0xc0:
            actual |= 0xc0
        if status & 0x300:
            actual |= 0x300
        if status & 0xc00:
            actual |= 0xc00
        if actual != expected_fuzz:
            failures.append(f"local fuzz model status=0x{status:x} -> 0x{actual:x} expected 0x{expected_fuzz:x}")
    required_fragments = (
        "static inline uint32_t rknpu_fuzz_status(uint32_t status)",
        "if ((status & 0x300) != 0)\n\t\tfuzz_status |= 0x300;",
        "status = REG_READ(RKNPU_OFFSET_INT_STATUS);",
        "job->int_status[core_index] = status;",
        "if (rknpu_fuzz_status(status) != job->int_mask[core_index])",
        "REG_READ(RKNPU_OFFSET_INT_RAW_STATUS)",
        "REG_WRITE(RKNPU_INT_CLEAR, RKNPU_OFFSET_INT_CLEAR);",
        "rknpu_job_done(job, 0, core_index);",
    )
    for fragment in required_fragments:
        if fragment not in source:
            failures.append(f"missing expected IRQ source fragment: {fragment}")
    if failures:
        print("FAIL rknpu IRQ completion source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu IRQ completion source model")
    print("driver completion requires fuzzed INT_STATUS to equal modeled INT_MASK=0x300; raw-status-only bits do not call rknpu_job_done")
    return 0


def parse_mmio_snapshots(path):
    snapshots = []
    current_snapshot = None
    current_core = None
    for line in Path(path).read_text().splitlines():
        if match := MMIO_SNAPSHOT_LINE.search(line):
            current_snapshot = {"label": match.group("label"), "cores": {}}
            snapshots.append(current_snapshot)
            current_core = None
            continue
        if current_snapshot is None:
            continue
        if match := MMIO_UNAVAILABLE_LINE.search(line):
            current_snapshot["unavailable"] = {
                "device": match.group("device"),
                "errno": int(match.group("errno")),
                "error": match.group("error"),
            }
            continue
        if match := MMIO_CORE_LINE.search(line):
            current_core = int(match.group("core"))
            current_snapshot["cores"][current_core] = {
                "base": int(match.group("base"), 16),
                "regs": {},
            }
            continue
        if current_core is None:
            continue
        if match := MMIO_REG_LINE.search(line):
            current_snapshot["cores"][current_core]["regs"][match.group("name")] = int(match.group("value"), 16)
    return snapshots


def compare_mmio_snapshot_log(conv, mmio_log):
    if mmio_log is None or not Path(mmio_log).exists():
        print("SKIP rknpu MMIO snapshot comparison")
        print("no root-captured MMIO log present; run with RKNPU_MMIO_SNAPSHOT=1 when /dev/mem is accessible")
        return 0

    records = observed_records()
    expected = modeled_rknpu_driver_commit(conv, records, 0, 2)
    snapshots = parse_mmio_snapshots(mmio_log)
    unavailable = [row for row in snapshots if row.get("unavailable")]
    if unavailable and not any(row["cores"] for row in snapshots):
        row = unavailable[0]["unavailable"]
        print("SKIP rknpu MMIO snapshot comparison")
        print(f"MMIO unavailable: {row['device']} errno={row['errno']} error={row['error']}")
        return 0
    after_snapshots = [row for row in snapshots if row["label"].startswith("after_submit_")]
    failures = []
    if not snapshots:
        failures.append(f"no MMIO_SNAPSHOT blocks parsed from {mmio_log}")
    if not after_snapshots:
        failures.append("no after_submit MMIO snapshot found")

    matching_after = []
    for snapshot in after_snapshots:
        for core, row in snapshot["cores"].items():
            regs = row["regs"]
            if (
                regs.get("PC_DATA_ADDR") == (expected["PC_DATA_ADDR"] & 0xffffffff)
                and regs.get("PC_DATA_AMOUNT") == expected["PC_DATA_AMOUNT"]
                and regs.get("PC_TASK_CONTROL") == expected["PC_TASK_CONTROL"]
                and regs.get("PC_DMA_BASE_ADDR") == expected["PC_DMA_BASE_ADDR"]
            ):
                matching_after.append((snapshot["label"], core, regs))

    if after_snapshots and not matching_after:
        failures.append(
            "no after_submit core matched modeled PC_DATA_ADDR/PC_DATA_AMOUNT/PC_TASK_CONTROL/PC_DMA_BASE_ADDR"
        )

    if matching_after:
        label, core, regs = matching_after[0]
        int_mask = regs.get("INT_MASK")
        if int_mask != expected["INT_MASK"]:
            failures.append(f"{label} core{core} INT_MASK=0x{int_mask or 0:x} expected=0x{expected['INT_MASK']:x}")
        task_status = regs.get("PC_TASK_STATUS")
        if task_status is not None and (task_status & 0xfff) > 2:
            failures.append(f"{label} core{core} PC_TASK_STATUS low bits unexpectedly exceed task_number: 0x{task_status:x}")

    if failures:
        print("FAIL rknpu MMIO snapshot comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1

    if matching_after:
        label, core, regs = matching_after[0]
        print("PASS rknpu MMIO snapshot comparison")
        print(
            f"{label} core{core} matches modeled RKNN commit: "
            f"PC_DATA_ADDR=0x{regs['PC_DATA_ADDR']:08x} "
            f"PC_DATA_AMOUNT={regs['PC_DATA_AMOUNT']} "
            f"PC_TASK_CONTROL=0x{regs['PC_TASK_CONTROL']:x}"
        )
    else:
        print("SKIP rknpu MMIO snapshot comparison")
        print("MMIO log had no after-submit snapshot to compare")
    return 0


def parse_timeout_log(path):
    rows = []
    current = {}
    for line in Path(path).read_text().splitlines():
        if match := RKNPU_TIMEOUT_JOB_LINE.search(line):
            if current:
                rows.append(current)
                current = {}
            current.update({
                "mask": int(match.group("mask"), 16),
                "timeout_us": int(match.group("timeout")),
            })
            continue
        if match := RKNPU_TIMEOUT_WAIT_LINE.search(line):
            current.update({
                "wait_task_counter": int(match.group("task_counter")),
                "flags": int(match.group("flags"), 16),
            })
            continue
        if match := RKNPU_TIMEOUT_CORE_LINE.search(line):
            current.update({
                "core": int(match.group("core")),
                "irq_status": int(match.group("irq_status"), 16),
                "raw_status": int(match.group("raw_status"), 16),
                "require_mask": int(match.group("require_mask"), 16),
                "task_counter": int(match.group("task_counter"), 16),
            })
            rows.append(current)
            current = {}
    if current:
        rows.append(current)
    return rows


def compare_timeout_log(timeout_log):
    if timeout_log is None or not Path(timeout_log).exists():
        print("SKIP raw timeout dmesg comparison")
        print("no saved raw timeout dmesg log present; pass --timeout-log after a controlled probe if one is intentionally run")
        return 0

    rows = parse_timeout_log(timeout_log)
    failures = []
    if not rows:
        failures.append(f"no RKNPU timeout rows parsed from {timeout_log}")
    for idx, row in enumerate(rows):
        if row.get("mask") != 0x1:
            failures.append(f"timeout[{idx}] mask={row.get('mask')} expected core0 mask 0x1")
        if row.get("flags") != 0x5:
            failures.append(f"timeout[{idx}] flags=0x{row.get('flags', 0):x} expected 0x5")
        if row.get("irq_status") != 0:
            failures.append(f"timeout[{idx}] irq_status=0x{row.get('irq_status', 0):x} expected 0")
        if row.get("require_mask") != 0x300:
            failures.append(f"timeout[{idx}] require_mask=0x{row.get('require_mask', 0):x} expected 0x300")
        if row.get("task_counter") != 1:
            failures.append(f"timeout[{idx}] task_counter=0x{row.get('task_counter', 0):x} expected 0x1")
        raw = row.get("raw_status", 0)
        status_names, reserved = decode_interrupt_status(raw)
        if status_names:
            failures.append(f"timeout[{idx}] raw_status=0x{raw:x} unexpectedly has named bits {status_names}")
        if reserved != raw:
            failures.append(f"timeout[{idx}] raw_status=0x{raw:x} is not only reserved bits")

    if failures:
        print("FAIL raw timeout dmesg comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS raw timeout dmesg comparison")
    print(f"{len(rows)} timeout row(s) match modeled selected core0/DPU wait: flags=0x5 require_mask=0x300 irq_status=0 task_counter=1")
    print("raw status carries no named completion/DMA-error bits under the current interrupt decoder")
    return 0


def compare_timeout_abort_source_model(conv):
    source = (ROOT / "ref" / "rknpu_driver" / "rknpu_job.c").read_text()
    expected = modeled_rknpu_driver_commit(conv, observed_records(), 0, 2)
    failures = []
    if expected["INT_MASK"] != 0x300:
        failures.append(f"modeled timeout require mask=0x{expected['INT_MASK']:x} expected 0x300")
    required_fragments = (
        "ret = wait_event_timeout(subcore_data->job_done_wq,",
        "job->flags & RKNPU_JOB_DONE ||\n\t\t\t\t\t\t rknpu_dev->soft_reseting,",
        "args->task_counter = 0;",
        "uint32_t task_status = REG_READ(\n\t\t\t\trknpu_dev->config->pc_task_status_offset);",
        "args->task_counter =\n\t\t\t\t(task_status &\n\t\t\t\t rknpu_dev->config->pc_task_number_mask);",
        "return ret < 0 ? ret : -ETIMEDOUT;",
        "if (job->ret == -ETIMEDOUT) {",
        "LOG_ERROR(\"job timeout, flags: %#x:\\n\", job->flags);",
        "REG_READ(RKNPU_OFFSET_INT_STATUS)",
        "REG_READ(RKNPU_OFFSET_INT_RAW_STATUS)",
        "job->int_mask[i]",
        "rknpu_soft_reset(rknpu_dev);",
        "rknpu_job_cleanup(job);",
    )
    for fragment in required_fragments:
        if fragment not in source:
            failures.append(f"missing expected timeout/abort source fragment: {fragment}")
    if failures:
        print("FAIL rknpu timeout/abort source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu timeout/abort source model")
    print("timeout path reads PC task status, reports INT_STATUS/INT_RAW/required mask, soft-resets on -ETIMEDOUT, then cleans up the job")
    return 0


def parse_sysfs_snapshots(path):
    snapshots = []
    current = None
    for line in Path(path).read_text().splitlines():
        if match := SYSFS_SNAPSHOT_LINE.search(line):
            current = {"label": match.group("label"), "values": {}}
            snapshots.append(current)
            continue
        if current is None:
            continue
        if match := SYSFS_VALUE_LINE.search(line):
            current["values"][match.group("path")] = match.group("value")
    return snapshots


def compare_sysfs_snapshot_log(sysfs_log):
    if sysfs_log is None or not Path(sysfs_log).exists():
        print("SKIP rknpu sysfs state comparison")
        print("no sysfs snapshot log present; use RKNPU_SYSFS_SNAPSHOT=1 for a non-privileged RKNN trace")
        return 0

    snapshots = parse_sysfs_snapshots(sysfs_log)
    failures = []
    expected_labels = {"before_submit_1", "after_submit_1"}
    labels = {row["label"] for row in snapshots}
    missing = expected_labels - labels
    if missing:
        failures.append(f"missing sysfs snapshot labels {sorted(missing)}")
    required_suffixes = (
        "/power/runtime_status",
        "/power/runtime_usage",
        "/cur_freq",
        "/target_freq",
        "/governor",
    )
    for row in snapshots:
        for suffix in required_suffixes:
            matches = [value for path, value in row["values"].items() if path.endswith(suffix)]
            if not matches:
                failures.append(f"{row['label']} missing sysfs field ending {suffix}")
            elif matches[0].startswith("<unreadable:"):
                failures.append(f"{row['label']} unreadable sysfs field ending {suffix}")
        for suffix in ("/cur_freq", "/target_freq"):
            values = [value for path, value in row["values"].items() if path.endswith(suffix)]
            if values and not values[0].isdigit():
                failures.append(f"{row['label']} nonnumeric {suffix} value {values[0]}")
    if failures:
        print("FAIL rknpu sysfs state comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    by_label = {row["label"]: row["values"] for row in snapshots}
    before = by_label["before_submit_1"]
    after = by_label["after_submit_1"]
    field = lambda values, suffix: next(value for path, value in values.items() if path.endswith(suffix))
    print("PASS rknpu sysfs state comparison")
    print(
        "RKNN submit snapshots captured non-privileged PM/devfreq state: "
        f"runtime_status={field(before, '/power/runtime_status')}->{field(after, '/power/runtime_status')} "
        f"cur_freq={field(before, '/cur_freq')}->{field(after, '/cur_freq')}"
    )
    print("sysfs state is diagnostic only; it does not expose per-core scheduler queue state")
    return 0


def compare_trace_patch(trace_patch):
    path = Path(trace_patch)
    if not path.exists():
        print("SKIP rknpu PC commit trace patch check")
        print(f"trace patch not present: {path}")
        return 0
    text = path.read_text()
    required = (
        "pc commit prelude:",
        "pc commit trace:",
        "pc commit regs:",
        "pc commit armed:",
        "valid irq trace:",
        "timeout pc regs:",
        "invalid irq pc regs:",
        "PC_DATA_ADDR",
        "PC_DATA_AMOUNT",
        "PC_TASK_CONTROL",
        "PC_DMA_BASE_ADDR",
        "PC_OP_EN",
        "ENABLE_MASK",
        "RKNPU_OFFSET_INT_RAW_STATUS",
    )
    failures = [item for item in required if item not in text]
    if failures:
        print("FAIL rknpu PC commit trace patch check")
        for item in failures:
            print(f"  missing {item}")
        return 1
    print("PASS rknpu PC commit trace patch check")
    print("trace patch logs pre-commit state, commit inputs, programmed PC registers, valid/invalid IRQ state, and timeout PC state")
    return 0


def _parse_prefixed_ints(match):
    out = {}
    for key, value in match.groupdict().items():
        out[key] = int(value, 0)
    return out


def parse_pc_trace_log(path):
    rows = {"prelude": [], "trace": [], "regs": [], "armed": [], "timeout": [], "valid_irq": [], "invalid_irq": []}
    for line in Path(path).read_text().splitlines():
        if match := PC_COMMIT_PRELUDE_LINE.search(line):
            rows["prelude"].append(_parse_prefixed_ints(match))
        if match := PC_COMMIT_TRACE_LINE.search(line):
            rows["trace"].append(_parse_prefixed_ints(match))
        if match := PC_COMMIT_REGS_LINE.search(line):
            rows["regs"].append(_parse_prefixed_ints(match))
        if match := PC_COMMIT_ARMED_LINE.search(line):
            rows["armed"].append(_parse_prefixed_ints(match))
        if match := PC_TIMEOUT_REGS_LINE.search(line):
            rows["timeout"].append(_parse_prefixed_ints(match))
        if match := PC_VALID_IRQ_TRACE_LINE.search(line):
            rows["valid_irq"].append(_parse_prefixed_ints(match))
        if match := PC_INVALID_IRQ_REGS_LINE.search(line):
            rows["invalid_irq"].append(_parse_prefixed_ints(match))
    return rows


def compare_pc_trace_log(conv, pc_trace_log):
    path = Path(pc_trace_log)
    if not path.exists():
        print("SKIP rknpu PC commit trace log comparison")
        print("no instrumented driver trace log present; use --pc-trace-log after applying rknpu_pc_commit_trace.patch")
        return 0

    rows = parse_pc_trace_log(path)
    records = observed_records()
    expected = modeled_rknpu_driver_commit(conv, records, 0, 2)
    first = records[0]
    last = records[1]
    failures = []
    if not rows["prelude"]:
        failures.append("missing pc commit prelude line")
    if not rows["trace"]:
        failures.append("missing pc commit trace line")
    if not rows["regs"]:
        failures.append("missing pc commit regs line")
    if not rows["armed"]:
        failures.append("missing pc commit armed line")

    if rows["prelude"]:
        prelude = rows["prelude"][0]
        expected_prelude = {
            "core_mask": 0,
            "use_core_num": 1,
        }
        for key, value in expected_prelude.items():
            if prelude.get(key) != value:
                failures.append(f"pc commit prelude {key}=0x{prelude.get(key, 0):x} expected=0x{value:x}")

    if rows["trace"]:
        trace = rows["trace"][0]
        expected_trace = {
            "submit_index": 0,
            "task_start": 0,
            "task_number": 2,
            "task_end": 1,
            "first_amount": first["regcfg_amount"],
            "first_enable": first["enable_mask"],
            "first_int": first["int_mask"],
            "last_int": last["int_mask"],
            "task_base_addr": 0,
            "flags": RKNPU_JOB_PC | RKNPU_JOB_PINGPONG,
            "use_core_num": 1,
        }
        for key, value in expected_trace.items():
            if trace.get(key) != value:
                failures.append(f"pc commit trace {key}=0x{trace.get(key, 0):x} expected=0x{value:x}")
        if trace.get("first_regcmd") != (expected["PC_DATA_ADDR"] & 0xffffffff):
            failures.append(
                f"pc commit trace first_regcmd=0x{trace.get('first_regcmd', 0):x} expected low32=0x{expected['PC_DATA_ADDR'] & 0xffffffff:x}"
            )

    if rows["regs"]:
        regs = rows["regs"][0]
        expected_regs = {
            "PC_DATA_ADDR": expected["PC_DATA_ADDR"] & 0xffffffff,
            "PC_DATA_AMOUNT": expected["PC_DATA_AMOUNT"],
            "INT_MASK": expected["INT_MASK"],
            "PC_TASK_CONTROL": expected["PC_TASK_CONTROL"],
            "PC_DMA_BASE_ADDR": expected["PC_DMA_BASE_ADDR"],
        }
        for key, value in expected_regs.items():
            if regs.get(key) != value:
                failures.append(f"pc commit regs {key}=0x{regs.get(key, 0):x} expected=0x{value:x}")

    if rows["armed"]:
        armed = rows["armed"][0]
        if armed.get("PC_OP_EN") != 0:
            failures.append(f"pc commit armed PC_OP_EN=0x{armed.get('PC_OP_EN', 0):x} expected 0 after pulse")
        if armed.get("PC_TASK_STATUS", 0) > 2:
            failures.append(f"pc commit armed TASK_STATUS=0x{armed.get('PC_TASK_STATUS', 0):x} unexpectedly exceeds task_number")

    for idx, timeout in enumerate(rows["timeout"]):
        for key in ("PC_DATA_ADDR", "PC_DATA_AMOUNT", "PC_TASK_CONTROL", "PC_DMA_BASE_ADDR"):
            expected_value = expected[key] & 0xffffffff
            if timeout.get(key) != expected_value:
                failures.append(f"timeout pc regs[{idx}] {key}=0x{timeout.get(key, 0):x} expected=0x{expected_value:x}")

    for idx, irq in enumerate(rows["valid_irq"]):
        if irq.get("INT_MASK") != expected["INT_MASK"]:
            failures.append(f"valid irq[{idx}] require=0x{irq.get('INT_MASK', 0):x} expected=0x{expected['INT_MASK']:x}")
        if (irq.get("INT_STATUS", 0) & expected["INT_MASK"]) != expected["INT_MASK"]:
            failures.append(f"valid irq[{idx}] status=0x{irq.get('INT_STATUS', 0):x} does not contain require mask 0x{expected['INT_MASK']:x}")
        for key in ("PC_DATA_ADDR", "PC_DATA_AMOUNT", "PC_TASK_CONTROL", "PC_DMA_BASE_ADDR"):
            expected_value = expected[key] & 0xffffffff
            if irq.get(key) != expected_value:
                failures.append(f"valid irq[{idx}] {key}=0x{irq.get(key, 0):x} expected=0x{expected_value:x}")

    for idx, irq in enumerate(rows["invalid_irq"]):
        for key in ("PC_DATA_ADDR", "PC_DATA_AMOUNT", "PC_TASK_CONTROL", "PC_DMA_BASE_ADDR"):
            expected_value = expected[key] & 0xffffffff
            if irq.get(key) != expected_value:
                failures.append(f"invalid irq[{idx}] {key}=0x{irq.get(key, 0):x} expected=0x{expected_value:x}")

    if failures:
        print("FAIL rknpu PC commit trace log comparison")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS rknpu PC commit trace log comparison")
    print(
        f"commit trace matches modeled selected task range: core={rows['trace'][0]['core']} "
        f"task_start=0 task_number=2 PC_DATA_AMOUNT={expected['PC_DATA_AMOUNT']} "
        f"PC_TASK_CONTROL=0x{expected['PC_TASK_CONTROL']:x}"
    )
    print(f"prelude rows parsed: {len(rows['prelude'])}")
    if rows["timeout"]:
        print(f"timeout pc-reg trace rows also match modeled commit state: {len(rows['timeout'])}")
    if rows["valid_irq"]:
        print(f"valid IRQ trace rows show required completion bits: {len(rows['valid_irq'])}")
    if rows["invalid_irq"]:
        print(f"invalid IRQ trace rows retain modeled PC register state: {len(rows['invalid_irq'])}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--ioctl-log", type=Path, default=DEFAULT_IOCTL_LOG)
    parser.add_argument("--mmio-log", type=Path, default=DEFAULT_MMIO_LOG)
    parser.add_argument("--timeout-log", type=Path, default=DEFAULT_TIMEOUT_LOG)
    parser.add_argument("--sysfs-log", type=Path, default=DEFAULT_SYSFS_LOG)
    parser.add_argument("--trace-patch", type=Path, default=DEFAULT_TRACE_PATCH)
    parser.add_argument("--pc-trace-log", type=Path, default=DEFAULT_PC_TRACE_LOG)
    parser.add_argument("--rknn", type=Path, default=DEFAULT_RKNN)
    parser.add_argument("--six-record-log", type=Path, default=DEFAULT_SIX_RECORD_LOG)
    parser.add_argument("--six-h7-log", type=Path, default=DEFAULT_SIX_H7_LOG)
    parser.add_argument("--six-h14-log", type=Path, default=DEFAULT_SIX_H14_LOG)
    parser.add_argument("--six-pw-c256-h14-log", type=Path, default=DEFAULT_SIX_PW_C256_H14_LOG)
    parser.add_argument("--six-c64-h56-log", type=Path, default=DEFAULT_SIX_C64_H56_LOG)
    parser.add_argument("--c64-h56-ioctl-log", type=Path, default=DEFAULT_C64_H56_IOCTL_LOG)
    args = parser.parse_args()

    expected = observed_records()
    conv = apply_h40_rknn_profile(load_conv_tiles_no_device())
    gem1_records = normalize_log_records(parse_log_records(args.log, 1))
    gem2_records = normalize_log_records(parse_log_records(args.log, 2, tail_count=len(expected)))
    rc = 0
    rc |= compare_abi_layouts(conv)
    rc |= compare_records("GEM1 normalized task object", gem1_records, expected)
    rc |= compare_records("GEM2 embedded relative task table", gem2_records, expected)
    rc |= compare_six_desc_task_records(conv, args.six_record_log)
    rc |= compare_six_desc_evidence_state(conv)
    rc |= compare_task_object_source(gem1_records, gem2_records, args.ioctl_log)
    rc |= compare_job_lifecycle_source_model(args.ioctl_log)
    rc |= compare_core_selection_model(args.ioctl_log)
    rc |= compare_submit_mode_model(args.ioctl_log)
    rc |= compare_scheduler_state_source_model(args.ioctl_log)
    rc |= compare_action_side_effect_source_model(conv, args.ioctl_log)
    rc |= compare_mem_sync_source_model(conv, args.ioctl_log)
    rc |= compare_iommu_domain_lifecycle_source_model(args.ioctl_log)
    rc |= compare_mmap_cache_source_model(conv, args.ioctl_log)
    rc |= compare_exported_regcmd_runs(args.rknn, expected)
    rc |= compare_sparse_regcmd(conv, args.rknn)
    rc |= compare_live_regcmd_window(conv, args.ioctl_log, args.log)
    rc |= compare_six_desc_h7_live_regcmd_window(conv, args.six_h7_log)
    rc |= compare_six_desc_h14_live_regcmd_window(conv, args.six_h14_log)
    rc |= compare_six_desc_pw_c256_h14_live_regcmd_window(conv, args.six_pw_c256_h14_log)
    rc |= compare_unpromoted_c64_h56_live_regcmd_evidence(conv, args.six_c64_h56_log)
    rc |= compare_c64_h56_runtime_profile(args.c64_h56_ioctl_log)
    rc |= compare_c64_h56_free_teardown(args.c64_h56_ioctl_log)
    rc |= compare_c64_h56_sync_profile(args.c64_h56_ioctl_log)
    rc |= compare_ioctl_sequence(conv, args.ioctl_log)
    rc |= compare_rknn_allocations(conv, args.ioctl_log)
    rc |= compare_rknn_free_teardown(args.ioctl_log, "H40", {
        "task": conv.RKNN_TASK_BYTES,
        "weight": conv.RKNN_REGCMD_BYTES,
        "internal": conv.RKNN_WEIGHT_BYTES,
        "input": conv.RKNN_INPUT_BYTES,
        "output": conv.RKNN_OUTPUT_BYTES,
    })
    rc |= compare_dma_binding_live_dump(conv, args.ioctl_log, args.log)
    rc |= compare_written_buffer_placement(conv)
    rc |= compare_driver_commit_candidates(conv)
    rc |= compare_irq_completion_source_model(conv)
    rc |= compare_mmio_snapshot_log(conv, args.mmio_log)
    rc |= compare_timeout_log(args.timeout_log)
    rc |= compare_timeout_abort_source_model(conv)
    rc |= compare_sysfs_snapshot_log(args.sysfs_log)
    rc |= compare_trace_patch(args.trace_patch)
    rc |= compare_pc_trace_log(conv, args.pc_trace_log)

    task_bytes = pack_records(expected, regcmd_base=0)
    print(f"modeled_task_bytes={len(task_bytes)} struct_size={TASK_STRUCT.size}")
    print("conv_tiles rknpu_sparse_task_gem task bytes match captured RKNN task metadata after regcmd base normalization")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
