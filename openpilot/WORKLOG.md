# GPT-2 and openpilot inference work

Requested outcome: runnable model inference examples through the RK3588 NPU
register interface. Runtime code uses Python's standard library only. RKNN and
ONNX are reference tools in `/home/orangepi/pilot/evaluation`, outside this
runtime. CPU reference computation is used only for numerical verification.

INTENT: code currently provides operator examples and register captures; the task expects runnable GPT-2 and openpilot inference; README specifies direct NPU register programming without RKNN at inference time.

- [x] Reproduce openpilot RKNN outputs and compare with original ONNX models.
- [x] Capture navigation memory, decoded registers, and task scheduling for relocation.
- [x] Implement Python allocation, address validation, and decoded submissions.
- [x] Run navigation through the standalone driver with five changing inputs; exact RKNN parity.
- [x] Run driver monitoring and supercombo through the standalone driver.
- [x] Implement GPT-2 loading, NPU operators, KV cache, and text generation.
- [x] Compare full-model outputs and GPT-2 generation with references.
- [x] Document setup, inputs, commands, evidence, and remaining limitations.

Existing user changes in the root README and operator examples predate this work.
No full openpilot GUI, camera, or vehicle integration is required by the selected
offline inference scope.

## Observed outcome

- Navigation: five changing input cases, exact register/RKNN output matches.
- Driver monitoring: image, zeros, and ramp/calibration cases, including repeats
  in a persistent model; exact register/RKNN matches.
- Supercombo: random pixels, normalized pixels, and zeros, including repeated
  persistent-model runs; exact register/RKNN matches on all 6768 outputs.
- GPT-2: full 12-layer inference and KV caching; five continuation tokens match
  Hugging Face on three prompts, with a reset/repeat check.

Accuracy comparisons are recorded in each directory's `validation.json`.
Supercombo's ONNX checks and navigation's pixel-range stress checks failed.
Driver monitoring's stress input has about 2.15% normalized RMS error. GPT-2
uses FP16 and its logits are not identical to FP32. These are recorded limits,
not claims that every accuracy check passed.

The reference capture tool's object-address matching and task extent logic were
also corrected. The older register-only observer had the same error and was
fixed and exercised on both driver-monitoring submissions. Superseded capture
scratch files were removed; the validated source captures and numerical
outputs remain in `~/pilot/evaluation`.

TWINS: searched object-address range resolution - found 1 other site: evaluation/capture_ioctl.c (fixed).

TWINS: searched object-address range resolution and direct tokenizer isspace calls - found 4 other sites: evaluation/capture_ioctl.c and gpt2/tokenizer.py (fixed).

## Commit review fixes

- [x] Correct NHWC row padding in both packing and unpacking.
- [x] Provide preparation from public, pinned model sources in a fresh cache.
- [x] Run fresh-cache openpilot and GPT-2 inference and regression checks.
- [x] Update preparation instructions and record the verification results.

INTENT: setup currently depends on board-local templates and NHWC copies ignore row padding; the fixes must support a fresh checkout and preserve tensor rows; the documentation calls for standard-library preparation and byte-only layout copies.

`prepare_models.py` reconstructs the three openpilot models using pinned public
RKNN coefficient sources, shipped decoded command tables, and sparse constants.
All reconstructed coefficient payloads match their original capture checksums.
New caches have zero-image fixtures, with supercombo traffic convention `[1,0]`.
The shipped cache recipes need no compiler, capture observer, or register XML.

An isolated copy containing only `openpilot/` and `gpt2/` passed preparation and
inference with `/usr/bin/python3 -S -B`, empty `PYTHONPATH`, and empty
`LD_PRELOAD`. All three openpilot fixtures matched RKNN exactly. GPT-2 matched
five continuation tokens for each of its three reference prompts. Repeat
preparation verified all caches. Five byte-layout regression tests and eight
independent NumPy layout/transpose cases passed. The NumPy reference check ran
in the external toolkit environment; the repository tests use only the stdlib.

TWINS: searched contiguous tensor-copy shortcuts and required board-local templates - found 4 other sites: NHWC unpacking, NC1HWC2 packing, NC1HWC2 unpacking, and openpilot preparation (fixed).

Evidence is in `~/pilot/evaluation/review-fixes/verification.json` and each
example's `validation.json`. The temporary source copy and fresh test cache were
removed after verification; numerical evidence was retained. The original
board caches also pass repeat preparation. Existing ONNX/FP32 accuracy limits
are still recorded; cache portability and byte padding fixes do not resolve
those numerical differences.
