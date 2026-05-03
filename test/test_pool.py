import os
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, "examples")
import pool  # noqa: E402


def decode_reg(word):
  return ((word >> 48) & 0xffff, word & 0xffff, (word >> 16) & 0xffffffff)


def reg_values(regs, addr):
  return [value for _, reg_addr, value in map(decode_reg, regs) if reg_addr == addr]


class TestPoolHelpers(unittest.TestCase):
  def test_reference_outputs(self):
    x = (np.arange(16, dtype=np.float16).reshape(4, 4) / np.float16(8.0)).astype(np.float16)
    np.testing.assert_array_equal(pool.pool2d_reference(x, "max"),
      np.asarray([[0.625, 0.75, 0.875], [1.125, 1.25, 1.375], [1.625, 1.75, 1.875]], dtype=np.float16))
    np.testing.assert_array_equal(pool.pool2d_reference(x, "min"),
      np.asarray([[0.0, 0.125, 0.25], [0.5, 0.625, 0.75], [1.0, 1.125, 1.25]], dtype=np.float16))
    np.testing.assert_array_equal(pool.pool2d_reference(x, "avg"),
      np.asarray([[0.3125, 0.4375, 0.5625], [0.8125, 0.9375, 1.0625], [1.3125, 1.4375, 1.5625]], dtype=np.float16))
    np.testing.assert_array_equal(pool.pool2d_reference(x, "globalmax"), np.asarray([[1.875]], dtype=np.float16))
    np.testing.assert_array_equal(pool.pool2d_reference(x, "globalmin"), np.asarray([[0.0]], dtype=np.float16))
    np.testing.assert_array_equal(pool.pool2d_reference(x, "globalavg"), np.asarray([[0.9375]], dtype=np.float16))

  def test_strided_pack_and_read(self):
    buf = bytearray(16 * 8 * 2)
    x = np.arange(16, dtype=np.float16)
    pool.pack_f16_strided(buf, x)
    raw = np.frombuffer(buf, dtype=np.float16, count=16 * 8)
    np.testing.assert_array_equal(raw[::8], x)
    self.assertEqual(np.count_nonzero(raw.reshape(16, 8)[:, 1:]), 0)
    np.testing.assert_array_equal(pool.read_f16_strided(buf, 16), x)

  def test_all_pool_modes_present(self):
    self.assertEqual(pool.POOL_OPS, ("min", "max", "avg", "globalmin", "globalmax", "globalavg"))


class TestPoolRegs(unittest.TestCase):
  def test_reg_counts_match_rknnops_cases(self):
    self.assertEqual(len(pool.pooling_regs("min")), 23)
    self.assertEqual(len(pool.pooling_regs("max")), 23)
    self.assertEqual(len(pool.pooling_regs("avg")), 25)
    self.assertEqual(len(pool.pooling_regs("globalmin")), 23)
    self.assertEqual(len(pool.pooling_regs("globalmax")), 21)
    self.assertEqual(len(pool.pooling_regs("globalavg")), 25)

  def test_min_uses_maxpool_on_negated_input_registers(self):
    self.assertEqual(pool.pooling_regs("min"), pool.pooling_regs("max"))
    self.assertEqual(pool.pooling_regs("globalmin"), pool.pooling_regs("max"))

  def test_globalmax_register_shape(self):
    regs = pool.pooling_regs("globalmax")
    addrs = [addr for _, addr, _ in map(decode_reg, regs)]
    self.assertNotIn(pool.rk.REG_PPU_DATA_CUBE_OUT_WIDTH, addrs)
    self.assertNotIn(pool.rk.REG_PPU_DATA_CUBE_OUT_HEIGHT, addrs)
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_DATA_CUBE_OUT_CHANNEL), [7])
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_DST_SURF_STRIDE), [16])
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_DATA_FORMAT), [18])
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_POOLING_KERNEL_CFG), [0x00330303])

  def test_globalavg_register_shape(self):
    regs = pool.pooling_regs("globalavg")
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_DATA_CUBE_OUT_WIDTH), [0])
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_DATA_CUBE_OUT_HEIGHT), [0])
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_RECIP_KERNEL_WIDTH), [30720])
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_RECIP_KERNEL_HEIGHT), [30720])
    self.assertEqual(reg_values(regs, pool.rk.REG_PPU_POOLING_KERNEL_CFG), [0x00330303])

  def test_ppu_task_descriptor_constants_match_ops_reg(self):
    self.assertEqual(pool.POOL_TASK_OP_IDX, 1)
    self.assertEqual(pool.POOL_ENABLE_MASK, 0x60)
    self.assertEqual(pool.POOL_INT_MASK, 0xc00)
    self.assertEqual(pool.POOL_PC_ENABLE, 48)


@unittest.skipUnless(os.path.exists("/dev/dri/card1"), "requires RK3588 NPU at /dev/dri/card1")
class TestPoolSubmit(unittest.TestCase):
  def test_submit_all_pool_ops(self):
    for op in pool.POOL_OPS:
      with self.subTest(op=op):
        proc = subprocess.run([sys.executable, "examples/pool.py", "--submit", "--op", op],
                              check=False, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
  unittest.main(verbosity=2)
