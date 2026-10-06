# openpilot model inference on RK3588

Offline navigation, driver monitoring, and supercombo inference using decoded
NPU registers. The runtime uses Python's standard library: no NumPy, RKNN
runtime library, compiler extension, or GPU inference.

## Run

Prepare the cache from a fresh checkout with Python's standard library:

```sh
cd ~/rk3588
python3 -S openpilot/prepare_models.py
```

This downloads about 120 MB of pinned public coefficient sources and creates
about 245 MB of model buffers in `~/.cache/rk3588-models/openpilot`.
Use `prepare_models.py navigation` to prepare one model, or `--cache DIRECTORY`
to select a different cache root. Preparation checks source and payload hashes,
and writes each complete model atomically. Subsequent runs verify existing
buffers. Internet access is needed for the initial download; inference is offline.

The vendor RKNPU DRM driver is required (this board uses kernel 6.1.99 and
RKNPU 0.9.8). The mainline rocket driver has a different ABI.

```sh
cd ~/rk3588
python3 -S openpilot/infer.py navigation --verify
python3 -S openpilot/infer.py dmonitoring --verify
python3 -S openpilot/infer.py supercombo --verify
```

`--verify` checks the supplied fixture against saved RKNN output, including
output length and exact values. Each invocation allocates new buffers, relocates
DMA pointers, reconstructs decoded register commands, poisons the output, and
runs blocking NPU jobs. The examples execute inference; they do not print a
recorded answer.

A newly prepared cache uses zero-image reference fixtures. Supercombo's traffic
convention is `[1,0]`; its other fixture inputs are zero. Existing board caches
retain their original image fixtures. `--verify` checks the fixture belonging to
the selected cache.

```sh
python3 -S openpilot/infer.py navigation --case ramp --output /tmp/navigation.f32
python3 -S openpilot/infer.py dmonitoring --case zeros
python3 -S openpilot/infer.py supercombo --dry-run
```

For real inputs, repeat `--input NAME=FILE` for **every** model input. Files are
little-endian FP16 in the native layout shown in `program.json`. Alternatively
use `--canonical` for contiguous FP16 in its `io[].logical` layout. The CLI
writes a contiguous FP32 output with `--output`.

| Model | Inputs | Output elements |
| --- | --- | ---: |
| navigation | `input_img` | 420 |
| dmonitoring | `input_img`, `calib` | 84 |
| supercombo | `input_imgs`, `big_input_imgs`, `desire`, `traffic_convention`, `lat_planner_state`, `nav_features`, `nav_instructions`, `features_buffer` | 6768 |

Consult the cache's logical dimensions rather than assuming that RKNN kept the
source ONNX dimensions: the reference compiler removes reshapes. Driver
monitoring's flattened image becomes `[1,960,720,2]`; its calibration is three
values. The decoded schedule includes a byte transpose `[0,3,2,1]` between its
two NPU jobs. `layout.py` and `tensor_io.py` copy representations; they perform
no neural network arithmetic.

## Model preparation and provenance

Readable preparation templates ship in `templates/`: named register commands,
task schedules, byte-copy recipes, sparse coefficient constants and fixture
references. `prepare_models.py` downloads the RKNN files from
[AndrewJNg/openpilot-rk3588](https://github.com/AndrewJNg/openpilot-rk3588/tree/4f3a4486e14baf27628a561dc38cbc04b62987c9/selfdrive/modeld/models),
pinned to revision `4f3a4486e14baf27628a561dc38cbc04b62987c9`. Each source's
size and SHA256 are recorded in its template. The downloaded files supply
coefficient bytes; execution uses the decoded schedules verified here. Driver
monitoring and supercombo retain their reference-compiled 2.3.2 schedules.
Every reconstructed coefficient buffer must match its verified checksum.
Expanded programs, packed buffers and fixture files live in the external cache.
Both preparation and inference work with `python3 -S`, without an RKNN SDK.

`prepare_capture.py` is an optional tool for adding newly captured kernels.
It uses only the standard library.
It converts a complete reference capture, checks every address relocation,
checks changes between submissions, and proves the output destination agrees
with the reference. It refuses unexplained graph boundaries. Command words are
rebuilt from named registers and bitfields, not submitted as a captured blob.

The original reference-only tools used for validation are at `~/pilot/evaluation`:
`compile_pilot_models.py`, `model_reference.py`, `capture_memory.c`, and
`onnx_pilot_reference.py`. They use RKNN Toolkit 2.3.2 and ONNX/NumPy to prepare
and compare models. The source ONNX files are in
`~/pilot/openpilot-rk3588/selfdrive/modeld/models`. These tools and dependencies
are not imported by inference.

They are unnecessary for preparing or running the shipped models. For an
existing capture of a different model:

```sh
python3 -S openpilot/prepare_capture.py CAPTURE CACHE --xml ~/npu/ops_rknn/registers.xml
# Driver monitoring also needs the verified layout description:
python3 -S openpilot/prepare_capture.py CAPTURE CACHE --layout LAYOUT.json
```

The capture observer records the complete allocations and the four PC trailer
words per task. Cache synchronization is necessary when reading mapped NPU
outputs. The driver preserves the captured task ranges, core mask and ping-pong
mode; it supports verified blocking core-0 submissions.

NHWC canonical inputs support padding on every row, including multiple batches
and channels. Run the CPU byte-layout regression checks with
`python3 -S openpilot/test_tensor_io.py`.

See `validation.json` for measured results. Navigation was compared with five
changing inputs in one process. Its register output matches RKNN on all five;
normalized inputs agree closely with ONNX. The supplied older RKNN model has
large ONNX errors on the extra 0–255 stress inputs. Those cases are not semantic
accuracy passes.

Supercombo's register outputs also match RKNN exactly on all three tested
profiles. Its ONNX comparisons failed: normalized RMS error was about 29.1%
for random 0–255 images, 43.2% for normalized images, and 35.2% for zero images.
Driver monitoring agrees closely on the supplied image (about 0.072% normalized
RMS error); the ramp/calibration stress case reaches about 2.15%. These accuracy
limitations remain in the reference-compiled models.

These are offline model examples. The full openpilot application, camera stack
and vehicle controls are outside this example's scope.
