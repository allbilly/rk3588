"""Rocket ABI, ownership and shipped schedules; no neural arithmetic or NPU IO."""
import ctypes as C
import errno
import hashlib
import json
import mmap
from pathlib import Path
from types import SimpleNamespace
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from cache import expand
from driver import Device, Task, discover
from infer import Model
import rocket


class SparseMapping:
    def __init__(self):
        self.writes = {}

    def __getitem__(self, span):
        raw = self.writes[span.start]
        return raw[:span.stop - span.start]


class SparseBuffer:
    def __init__(self, device, size, flags):
        self.device, self.size = device, size
        self.info = SimpleNamespace(handle=len(device.buffers) + 1)
        self.dma = sum(buffer.size for buffer in device.buffers)
        self.mapping = SparseMapping()

    def write(self, raw, offset=0):
        if offset + len(raw) > self.size:
            raise ValueError("Test write outside BO")
        self.mapping.writes[offset] = raw

    def sync(self, direction, timeout_ms=6000):
        pass


class RocketTests(unittest.TestCase):
    def test_uapi_layout_and_ioctl_directions(self):
        # Expected sizes/offsets and ioctl values from the Linux v6.18 UAPI.
        for cls, size, offsets in [
            (rocket.CreateBO, 24, [0, 4, 8, 16]),
            (rocket.PrepBO, 16, [0, 4, 8]),
            (rocket.FiniBO, 8, [0, 4]),
            (rocket.CloseBO, 8, [0, 4]),
            (rocket.Task, 8, [0, 4]),
            (rocket.Job, 40, [0, 8, 16, 24, 28, 32, 36]),
            (rocket.Submit, 24, [0, 8, 12, 16]),
        ]:
            self.assertEqual(C.sizeof(cls), size)
            self.assertEqual([getattr(cls, field).offset for field, _ in cls._fields_], offsets)
        self.assertEqual([rocket.CREATE_BO, rocket.SUBMIT, rocket.PREP_BO,
                          rocket.FINI_BO, rocket.GEM_CLOSE],
                         [0xc0186440, 0x40186441, 0x40106442, 0x40086443, 0x40086409])

    def test_discovery_skips_gpu_and_honors_selection(self):
        nodes = {"/dev/dri/card0": "rockchip-drm", "/dev/dri/card1": "rknpu",
                 "/dev/accel/accel0": "rocket"}
        with patch("driver.glob.glob", side_effect=lambda pattern: [p for p in nodes if
                   ("accel" in pattern) == ("accel" in p)]), \
                patch("driver.driver_name", side_effect=nodes.get), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(discover(), ("/dev/accel/accel0", "rocket"))
            self.assertEqual(discover(driver="rknpu"), ("/dev/dri/card1", "rknpu"))
            self.assertEqual(discover("/dev/dri/card1"), ("/dev/dri/card1", "rknpu"))
            with self.assertRaises(RuntimeError):
                discover("/dev/dri/card0", "rocket")
            with self.assertRaises(RuntimeError):
                discover("/dev/dri/card1", "rocket")
            with patch.dict("os.environ", {"ROCKET_DEVICE": "/dev/accel/accel0"}):
                self.assertEqual(discover(), ("/dev/accel/accel0", "rocket"))
        with patch("driver.glob.glob", return_value=[]), patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                discover(driver="rocket")

    def test_ownership_deadline_retry_and_gem_close(self):
        events = []
        timeout_once = [False]

        def ioctl(fd, number, args):
            events.append((number, bytes(args)))
            if number == rocket.CREATE_BO:
                args.handle, args.dma_address, args.offset = 7, 0, 0x1000
            if number == rocket.PREP_BO:
                self.assertEqual(args.timeout_ns, 9_000_000_000)
                if timeout_once[0]:
                    timeout_once[0] = False
                    raise OSError(errno.ETIMEDOUT, "still running")

        real_mmap = mmap.mmap
        with patch("rocket.ioctl", side_effect=ioctl), \
                patch("rocket.time.monotonic_ns", return_value=3_000_000_000), \
                patch("rocket.mmap.mmap", side_effect=lambda *args, **kwargs: real_mmap(-1, args[1])):
            buffer = rocket.Buffer(SimpleNamespace(fd=42), 100, flags=1035)
            self.assertEqual(buffer.size, 4096)
            self.assertEqual(rocket.CreateBO.from_buffer_copy(events[0][1]).size, 4096)
            buffer.write(b"one")
            buffer.sync(1)
            timeout_once[0] = True
            buffer.write(b"two")
            self.assertEqual(buffer.mapping[:3], b"two")
            buffer.sync(1)
            buffer.close()
        self.assertEqual([number for number, _ in events],
                         [rocket.CREATE_BO, rocket.PREP_BO, rocket.FINI_BO,
                          rocket.PREP_BO, rocket.PREP_BO, rocket.FINI_BO,
                          rocket.PREP_BO, rocket.GEM_CLOSE])

    def test_trailer_retains_dpu_only_enable_and_rejects_corruption(self):
        task = {"regcfg_amount": 1, "enable_mask": 24}
        raw = struct.pack("<5Q", 0x100100000000400c, 0x0101000010000010,
                          0x0101000000240014, 0x0041000000000000, 0x0081000000180008)
        changed = rocket.prepare_commands(raw, task)
        self.assertEqual(struct.unpack("<5Q", changed),
                         (0x100100000000400c, 0x0101000000000010,
                          0x0101000000000014, 0x0041000000000000, 0x0081000000180008))
        with self.assertRaises(ValueError):
            rocket.prepare_commands(raw[:-8], task)
        with self.assertRaises(ValueError):
            rocket.prepare_commands(raw[:-1] + b"\xff", task)

    def test_all_shipped_model_schedules_and_command_bodies(self):
        root = Path(__file__).resolve().parents[1]
        templates = sorted((root / "openpilot/templates").glob("*.json")) + \
                    sorted((root / "gpt2/templates").glob("*.json"))
        self.assertEqual(len(templates), 7)
        totals = {}
        for path in templates:
            with self.subTest(model=path.stem), tempfile.TemporaryDirectory() as directory:
                program = expand(json.loads(path.read_text()))
                # No coefficients are needed to verify command adaptation. Use
                # empty, hashed test payloads and sparse mappings of the real BO
                # geometry; the actual Model initializer still runs unchanged.
                for descriptor in program["buffers"]:
                    descriptor["payload_size"] = 0
                    descriptor["sha256"] = hashlib.sha256(b"").hexdigest()
                    (Path(directory) / descriptor["file"]).write_bytes(b"")
                (Path(directory) / "program.json").write_text(json.dumps(program))
                device = Device.__new__(Device)
                device.fd, device.driver, device.rocket = 42, "rocket", rocket
                device.buffers, device.busy = [], False
                with patch("rocket.Buffer", SparseBuffer):
                    model = Model(directory, device)
                for task in program["tasks"]:
                    location = task["regcmd"]
                    adapted = model.buffers[location["buffer"]].mapping.writes[location["offset"]]
                    original = b"".join(struct.pack("<Q", model.encode(c)) for c in task["commands"])
                    self.assertEqual(adapted[:-32], original[:-32])
                    self.assertEqual(adapted[-16:], original[-16:])
                    self.assertEqual(struct.unpack("<2Q", adapted[-32:-16]),
                                     (0x0101000000000010, 0x0101000000000014))
                requests = []

                def ioctl(fd, number, request):
                    self.assertEqual(number, rocket.SUBMIT)
                    self.assertEqual(request.job_count, 1)
                    self.assertEqual(request.job_struct_size, 40)
                    self.assertEqual(request.reserved, 0)
                    job = C.cast(request.jobs, C.POINTER(rocket.Job))[0]
                    self.assertEqual(job.task_struct_size, 8)
                    self.assertEqual(job.in_bo_handle_count, 0)
                    self.assertEqual(job.out_bo_handle_count, len(program["buffers"]))
                    handles = C.cast(job.out_bo_handles, C.POINTER(C.c_uint32))
                    self.assertEqual([handles[i] for i in range(job.out_bo_handle_count)],
                                     [b.info.handle for b in model.buffers.values()])
                    tasks = C.cast(job.tasks, C.POINTER(rocket.Task))
                    requests.append([(tasks[i].regcmd, tasks[i].regcmd_count) for i in range(job.task_count)])

                with patch("rocket.ioctl", side_effect=ioctl):
                    for settings in program.get("submits", [program["submit"]]):
                        device.submit(model.task_buffer, settings, model.buffers.values())
                        first, count = settings["subcores"][0]
                        expected = []
                        for index in range(first, first + count):
                            raw = model.task_buffer.mapping.writes[index * 40]
                            descriptor = Task.from_buffer_copy(raw)
                            expected.append((descriptor.regcmd_addr, descriptor.regcfg_amount + 4))
                        self.assertEqual(requests[-1], expected)
                        self.assertFalse(device.busy)
                totals[path.stem] = sum(map(len, requests))
        # Navigation contains 273 descriptor copies but only core 0's 91 run.
        self.assertEqual(totals, {"navigation": 91, "dmonitoring": 419, "supercombo": 620,
                                "embedding": 3, "head": 23, "layer-00-qkv": 17,
                                "layer-00-attention-ffn": 170})

    def test_rejected_submit_allows_context_cleanup(self):
        device = Device.__new__(Device)
        device.fd, device.driver, device.rocket = 42, "rocket", rocket
        device.buffers, device.busy = [], False
        buffer = SparseBuffer(device, 4096, 0)
        buffer.write(bytes(Task(enable_mask=24, regcmd_addr=0, regcfg_amount=1)))
        buffer.close = Mock()
        buffer.sync = Mock()
        device.buffers.append(buffer)
        settings = {"task_start": 0, "task_number": 1, "flags": 1, "core_mask": 1,
                    "subcores": [[0, 1]] + [[0, 0]] * 4, "timeout": 6000}
        failure = OSError(errno.ENOMEM, "submit rejected before queueing")
        with tempfile.TemporaryFile() as lock, patch("rocket.ioctl", side_effect=failure) as ioctl, \
                patch("driver.os.close") as close, patch("driver.flock") as flock:
            device.lock = lock
            with self.assertRaises(OSError) as caught:
                with device:
                    device.submit(buffer, settings, [buffer])
            self.assertIs(caught.exception, failure)
            self.assertFalse(device.busy)
            self.assertEqual(device.buffers, [])
            self.assertTrue(lock.closed)
            buffer.close.assert_called_once_with()
            close.assert_called_once_with(42)
            flock.assert_called_once_with(lock, 8)  # LOCK_UN
            ioctl.assert_called_once()
            self.assertEqual(ioctl.call_args.args[1], rocket.SUBMIT)
            self.assertEqual([call.args for call in buffer.sync.call_args_list], [(2,), (1,)])

    def test_interrupted_submit_retains_busy_guard(self):
        device = SimpleNamespace(fd=42, busy=False, buffers=[])
        buffer = SparseBuffer(device, 4096, 0)
        buffer.write(bytes(Task(enable_mask=24, regcmd_addr=0, regcfg_amount=1)))
        # A Python interruption does not establish whether the kernel queued work.
        with patch("rocket.ioctl", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                rocket.submit(device, buffer, 0, 1, {"timeout": 6000}, [buffer])
        self.assertTrue(device.busy)
        with self.assertRaises(RuntimeError):
            Device.close(device)

    def test_failed_fence_wait_prevents_buffer_cleanup(self):
        device = SimpleNamespace(fd=42, busy=False, buffers=[])
        buffer = SparseBuffer(device, 4096, 0)
        buffer.write(bytes(Task(enable_mask=24, regcmd_addr=0, regcfg_amount=1)))

        def sync(direction, **kwargs):
            if direction == 2 and device.busy:
                raise OSError(errno.EIO, "fence wait failed")

        buffer.sync = sync
        with patch("rocket.ioctl"):
            with self.assertRaises(OSError):
                rocket.submit(device, buffer, 0, 1, {"timeout": 6000}, [buffer])
        self.assertTrue(device.busy)
        with self.assertRaises(RuntimeError):
            Device.close(device)

    def test_invalid_task_range_is_rejected_before_submit(self):
        device = SimpleNamespace(fd=42, busy=False, buffers=[])
        buffer = SparseBuffer(device, 4096, 0)
        buffer.write(bytes(Task(enable_mask=24, regcmd_addr=4080, regcfg_amount=1)))
        with patch("rocket.ioctl") as ioctl:
            with self.assertRaises(ValueError):
                rocket.submit(device, buffer, 0, 1, {"timeout": 6000}, [buffer])
            ioctl.assert_not_called()

    def test_bo_address_and_size_limits_and_failed_mapping_cleanup(self):
        device = SimpleNamespace(fd=42)
        with patch("rocket.ioctl") as ioctl:
            for size in [1 << 32, 0xffffffff]:
                with self.assertRaises(ValueError):
                    rocket.Buffer(device, size)
            ioctl.assert_not_called()

        def create(fd, number, info):
            if number == rocket.CREATE_BO:
                info.handle, info.dma_address = 9, 0xfffff800

        with patch("rocket.ioctl", side_effect=create) as ioctl:
            with self.assertRaises(ValueError):
                rocket.Buffer(device, 4096)
            self.assertEqual([call.args[1] for call in ioctl.call_args_list],
                             [rocket.CREATE_BO, rocket.GEM_CLOSE])
        with patch("rocket.ioctl") as ioctl, patch("rocket.mmap.mmap", side_effect=OSError("mmap failed")):
            with self.assertRaises(OSError):
                rocket.Buffer(device, 4096)
            self.assertEqual([call.args[1] for call in ioctl.call_args_list],
                             [rocket.CREATE_BO, rocket.GEM_CLOSE])

    def test_initial_prep_failure_releases_unregistered_buffer(self):
        device = Device.__new__(Device)
        device.fd, device.rocket, device.buffers = 42, rocket, []
        failure = OSError(errno.EIO, "initial PREP failed")
        events = []
        mapping = mmap.mmap(-1, 4096)

        def ioctl(fd, number, info):
            events.append(number)
            if number == rocket.CREATE_BO:
                info.handle, info.dma_address, info.offset = 11, 0, 0
            elif number == rocket.PREP_BO:
                raise failure
            elif number == rocket.GEM_CLOSE:
                self.assertEqual(info.handle, 11)
                self.assertTrue(mapping.closed)

        try:
            with patch("rocket.ioctl", side_effect=ioctl), \
                    patch("rocket.mmap.mmap", return_value=mapping):
                with self.assertRaises(OSError) as caught:
                    device.allocate(4096)
            self.assertIs(caught.exception, failure)
            self.assertTrue(mapping.closed)
            self.assertEqual(events, [rocket.CREATE_BO, rocket.PREP_BO, rocket.GEM_CLOSE])
            self.assertEqual(device.buffers, [])
        finally:
            mapping.close()

    def test_submission_requires_complete_unique_model_bo_list(self):
        device = SimpleNamespace(fd=42, busy=False, buffers=[])
        buffer = SparseBuffer(device, 4096, 0)
        buffer.write(bytes(Task(enable_mask=24, regcmd_addr=0, regcfg_amount=1)))
        foreign = SparseBuffer(SimpleNamespace(buffers=[]), 4096, 0)
        with patch("rocket.ioctl") as ioctl:
            for buffers in [[], [buffer, foreign], [buffer, buffer]]:
                with self.assertRaises(ValueError):
                    rocket.submit(device, buffer, 0, 1, {"timeout": 6000}, buffers)
            for first, count in [(-1, 1), (0, 0), (0, 103)]:
                with self.assertRaises(ValueError):
                    rocket.submit(device, buffer, first, count, {"timeout": 6000}, [buffer])
            ioctl.assert_not_called()

    def test_only_active_dpu_tasks_can_use_stock_completion(self):
        device = SimpleNamespace(fd=42, busy=False, buffers=[])
        buffer = SparseBuffer(device, 4096, 0)
        # The second descriptor resembles navigation's unused PPU copies.
        buffer.write(bytes(Task(enable_mask=24, regcmd_addr=0, regcfg_amount=1)), 0)
        buffer.write(bytes(Task(enable_mask=96, regcmd_addr=0, regcfg_amount=1)), 40)
        with patch("rocket.ioctl") as ioctl:
            rocket.submit(device, buffer, 0, 1, {"timeout": 6000}, [buffer])
            ioctl.assert_called_once()
            ioctl.reset_mock()
            for mask in [0, 96, 0x68]:
                buffer.write(bytes(Task(enable_mask=mask, regcmd_addr=0, regcfg_amount=1)), 40)
                with self.assertRaisesRegex(ValueError, "DPU completion"):
                    rocket.submit(device, buffer, 1, 1, {"timeout": 6000}, [buffer])
            ioctl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
