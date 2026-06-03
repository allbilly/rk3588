# Current Task: Prefix-Replay Debug of All FENCED Shapes in `examples/conv.py`

**Last updated:** 2026-06-03 01:16 (Asia/Shanghai) — Rechecked quick-success candidates from the latest sweep: the obvious low-risk pointwise BY_Y/BY_YK shapes are already counted as PASS in `/tmp/opencode/conv_py_217_sweep_20260603_000137_summary.txt` (`PASS=136`, `FENCED=81`, pre/post health PASS). The remaining quick-looking shapes are still in unsafe buckets (`BY_K/k_tile`, depthwise `BY_YK`, or `NONE/setup`) and were not promoted without fresh closure evidence. Continued safe prefix replay on depthwise BY_YK: captured GEM2/body rows for `b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid` and `b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid` with `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2`. Both exited cleanly with expected partial-output mismatch, and post `python3 examples/simple_add.py` PASS. Updated `examples/conv_depthwise_by_y_layout_no_submit.py` so those cases now assert both GEM1 task metadata and GEM2 body topology/constants.
**Current blocker table (2026-06-03 01:16):**
| Family | Remaining in latest coverage | Biggest blocker | Best evidence | Next safe action |
|---|---:|---|---|---|
| `BY_K/k_tile` | 44 uncovered, plus covered fenced evidence | Multiple incompatible subfamilies share the planner bucket. h7 c512 wide pointwise passes, but h7 c832 narrow pointwise shows RKNN task0 PASS while local visible-row replay gives `inf`, so hidden buffer/layout parity is still missing for narrow OC. | Promoted c512_h7 pointwise; c832_h7 oc48 RKNN prefix PASS but local negative; c512_h14 depthwise GEM1/GEM2 prefix evidence; c256_h2 oc64 crash-fence no-submit materialization. | For narrow pointwise, compare RKNN input/weight/output GEM layouts against local buffers before any more submits. For other BY_K candidates, continue safe `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1/2` plus no-submit materializers. |
| `BY_YK/k_half,k_tile,setup` | 15 uncovered | RKNN emits extra setup/prologue rows, PPU/PDP separators, uneven y windows, and mixed full-OC/partial-K rows; local planner emits equal BY_YK rows. | c64_h112, c96_h112, c128_h56, c144_h56, c256_h28 depthwise evidence; c96_h112 and c144_h56 now have GEM2 body assertions; c256_h28 pointwise has 14-row no-submit closure. | Finish GEM2/body captures for the remaining task-only depthwise targets, then implement row materializers before any submit. |
| `NONE/setup` | 1 uncovered | `b1_c16_h80_oc128` generic exact11 reuse produced `inf`; likely custom multi-task/y-tile despite planner saying setup/NONE. | Negative probe only. | Safe matched-model prefix replay with GEM1/GEM2 capture. |
| Crash-fenced narrow pointwise | 2 known | Prior speculative local submit rebooted board even though it appeared to print PASS. | c256_h2_oc64 GEM1/GEM2 no-submit evidence proves exact11 multi-row, not oc24 setup108. | Keep fenced until exact local runtime path is proven with no-submit parity and a fresh safe prefix numeric PASS. |
**Latest depthwise BY_YK body evidence (2026-06-03 01:16):** `c96_h112` GEM2 decodes to 31 rows: `setup,setup,tail,tail,tail,tail,setup,setup,tail,setup,setup,setup,setup,ppu_pdp,setup,setup,setup,setup,ppu_pdp,y_tile,y_tile,y_tile,ppu_pdp,y_tile,y_tile,y_tile,ppu_pdp,y_tile,y_tile,y_tile,ppu_pdp`, with compute constants `DATA_SIZE1=0x001f0060 DMA_CON2=0x2f40 DST_SURF_STRIDE=0x2f44 SURFACE_ADD=0xbd10 CBUF0=0x1b/0x201b`; local planner is still 90 equal BY_YK rows. `c144_h56` GEM2 decodes to 17 rows: `setup,setup,tail,tail,setup,setup,setup,ppu_pdp,setup,setup,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp`, with compute constants `DATA_SIZE1=0x000f0090 DMA_CON2=0x0b60 DST_SURF_STRIDE=0x0b64 SURFACE_ADD=0x2d90 CBUF0=0x1b/0x201b`; local planner is still 75 equal BY_YK rows. These are no-submit assertions only, not replay-ready promotions.
**Latest validation (2026-06-03 01:16):** `python3 examples/conv_depthwise_by_y_layout_no_submit.py c96_h112_by_yk_task` PASS; `python3 examples/conv_depthwise_by_y_layout_no_submit.py c144_h56_by_yk_task` PASS; `python3 -m py_compile examples/conv.py conv_expt/rknn_prefix_replay.py examples/conv_crash_fence_no_submit.py examples/conv_pointwise_by_yk_layout_no_submit.py examples/conv_depthwise_by_y_layout_no_submit.py` PASS; `python3 conv_expt/rknn_prefix_replay.py --coverage-summary /tmp/opencode/conv_py_217_sweep_20260603_000137_summary.txt` reports `fenced_total=81 manifest_covered=21 uncovered_total=60`; post `python3 examples/simple_add.py` PASS.
**Previous depthwise BY_YK task evidence (2026-06-03 01:12):** Continued safe prefix replay on depthwise BY_YK. Added task-only evidence for `b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid` and `b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid`. Both used `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1`, exited cleanly with expected partial-output mismatch, and post `python3 examples/simple_add.py` PASS. Added no-submit assertions in `examples/conv_depthwise_by_y_layout_no_submit.py` and manifest entries `dw96_h112_by_yk_task_dump`, `dw144_h56_by_yk_task_dump`.
**Latest depthwise BY_YK task evidence (2026-06-03 01:12):** `c96_h112` RKNN GEM1 is 31 tasks with amounts `108,108,12,12,12,12,16,16,12,104,104,104,104,26,104,104,104,104,26,104,104,104,26,104,104,104,26,104,104,104,26`, while local planner emits 90 equal BY_YK rows. `c144_h56` RKNN GEM1 is 17 tasks with amounts `108,108,12,12,18,104,104,26,104,104,26,104,26,104,26,104,26`, while local planner emits 75 equal BY_YK rows. These are task-only and not replay-ready until GEM2/body rows are captured.
**Latest c832_h7 prefix/debug (2026-06-03 01:06):** `KEEP_TASKS=1 PREFIX_MODE=linear MAX_ERRORS=8 DUMP_GEM=1/2 ... match_b1_c832_h7_w7_oc48_wic832_k1x1_g1 ...` passed max error `0.0584869`. GEM1 task metadata is exact11 `108,104,26,104,26,104,26,104,26,104,26`, masks `0x0d/0x60`, offsets `0,112,224,256,368,400,512,544,656,688,800`. GEM2 rows decode as `setup,setup,ppu_pdp,setup,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp`; full setup rows use `out_c=48`, and k rows use three `16`-channel tiles at weight offsets `0,0x6800,0xd000` and output offsets `0,0x680,0xd00`. Local 11-row and local task0 setup108 probes failed with `inf`, then normal direct run returned the fence before submit and `python3 examples/simple_add.py` PASS.
**Latest coverage (2026-06-03 00:01):** `python3 conv_expt/rknn_prefix_replay.py --coverage-summary /tmp/opencode/conv_py_217_sweep_20260603_000137_summary.txt` reports `fenced_total=81 manifest_covered=18 uncovered_total=63`; uncovered buckets are 45 `BY_K/k_tile`, 17 `BY_YK/k_half,k_tile,setup`, and 1 `NONE/setup`.
**Latest validation (2026-06-03 00:01):** `python3 -m py_compile examples/conv.py conv_expt/rknn_prefix_replay.py` PASS; dry-runs for both c512_h7 aliases PASS; direct c512_h7 prefixed and unprefixed aliases PASS with `max_diff=0.0312`; full sweep PASS=136/FENCED=81 with pre/post `simple_add.py` PASS.
**Latest coverage (2026-06-02 23:17):** `python3 conv_expt/rknn_prefix_replay.py --coverage-summary /tmp/opencode/conv_py_217_sweep_20260602_223030_summary.txt` reports `fenced_total=83 manifest_covered=13 uncovered_total=70`. The previous uncovered `BY_Y/y_tile` bucket is now covered by task-only evidence; remaining uncovered buckets are 48 `BY_K/k_tile`, 21 `BY_YK/k_half,k_tile,setup`, and 1 `NONE/setup` (`b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid`).
**Latest validation (2026-06-02 23:17):** `python3 examples/conv_depthwise_by_y_layout_no_submit.py c32_h150_by_y_task`, `c32_by_y`, and `c64_by_yk` all PASS; `python3 -m py_compile conv_expt/rknn_prefix_replay.py examples/conv.py examples/conv_crash_fence_no_submit.py examples/conv_pointwise_by_yk_layout_no_submit.py examples/conv_depthwise_by_y_layout_no_submit.py` PASS; post `python3 examples/simple_add.py` PASS.
**Latest prefix replay (2026-06-02 23:25):** built `/home/orangepi/npu/ops_rknn/models/match_b1_c128_h56_w56_oc128_wic1_k3x3_g128.rknn` with `python3 gen_conv2d_models.py --custom --batch 1 --in-ch 128 --height 56 --width 56 --out-ch 128 --k-h 3 --k-w 3 --groups 128 --name match_b1_c128_h56_w56_oc128_wic1_k3x3_g128 --out-dir models`, then ran safe RKNN prefix replay `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1 DUMP_DIR=/home/orangepi/npu/ops_rknn/dump/prefix_c128_h56_dw_by_yk_keep1_gem1 gdb -q -x ioctl_prefix_replay.gdb --args ./conv2d_multi --case match_b1_c128_h56_w56_oc128_wic1_k3x3_g128 1 128 56 56 128 3 3 128 c128_h56_dw_by_yk`. The one-task prefix exited cleanly but failed numerically with partial output, as expected; post `python3 examples/simple_add.py` PASS.
**Latest no-submit depthwise BY_YK evidence (2026-06-02 23:26):** added `examples/conv_depthwise_by_y_layout_no_submit.py c128_h56_by_yk_task`, covering both `conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128` and `b1_c128_h56_w56_oc128_wic1_k3x3_g128`. It asserts GEM1 task metadata `amounts=108,108,12,12,104,104,26,104,104,26,104,26,104,26,104,26`, masks `0x0d/0x60`, offsets `0,112,224,240,256,368,480,512,624,736,768,880,912,1024,1056,1168`, and local planner mismatch (`60` equal BY_YK rows). This is task-only and not replay-ready until GEM2/body rows are captured.
**Latest coverage (2026-06-02 23:26):** coverage now reports `fenced_total=83 manifest_covered=15 uncovered_total=68`; remaining uncovered buckets are 48 `BY_K/k_tile`, 19 `BY_YK/k_half,k_tile,setup`, and 1 `NONE/setup`.
**Latest validation (2026-06-02 23:26):** `python3 examples/conv_depthwise_by_y_layout_no_submit.py c128_h56_by_yk_task` PASS; `python3 examples/conv_depthwise_by_y_layout_no_submit.py c32_h150_by_y_task` PASS; `python3 -m py_compile conv_expt/rknn_prefix_replay.py examples/conv.py examples/conv_crash_fence_no_submit.py examples/conv_pointwise_by_yk_layout_no_submit.py examples/conv_depthwise_by_y_layout_no_submit.py` PASS; post `python3 examples/simple_add.py` PASS.
**Latest prefix replay (2026-06-02 23:29):** built `/home/orangepi/npu/ops_rknn/models/match_b1_c256_h28_w28_oc256_wic1_k3x3_g256.rknn` with `python3 gen_conv2d_models.py --custom --batch 1 --in-ch 256 --height 28 --width 28 --out-ch 256 --k-h 3 --k-w 3 --groups 256 --name match_b1_c256_h28_w28_oc256_wic1_k3x3_g256 --out-dir models`, then ran safe prefix replay `MAX_ERRORS=8 KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1 DUMP_DIR=/home/orangepi/npu/ops_rknn/dump/prefix_c256_h28_dw_by_yk_keep1_gem1 gdb -q -x ioctl_prefix_replay.gdb --args ./conv2d_multi --case match_b1_c256_h28_w28_oc256_wic1_k3x3_g256 1 256 28 28 256 3 3 256 c256_h28_dw_by_yk`. The one-task prefix exited cleanly and failed numerically with partial output, as expected; post `python3 examples/simple_add.py` PASS.
**Latest no-submit depthwise BY_YK evidence (2026-06-02 23:30):** added `examples/conv_depthwise_by_y_layout_no_submit.py c256_h28_by_yk_task`, covering both `conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256` and `b1_c256_h28_w28_oc256_wic1_k3x3_g256`. It asserts GEM1 task metadata `amounts=108,108,104,26,104,26,104,26,104,26,104,26`, masks `0x0d/0x60`, offsets `0,112,224,336,368,480,512,624,656,768,800,912`, and local planner mismatch (`72` equal BY_YK rows). This is task-only and not replay-ready until GEM2/body rows are captured.
**Latest coverage (2026-06-02 23:30):** coverage now reports `fenced_total=83 manifest_covered=17 uncovered_total=66`; remaining uncovered buckets are 48 `BY_K/k_tile`, 17 `BY_YK/k_half,k_tile,setup`, and 1 `NONE/setup`.
**Latest validation (2026-06-02 23:30):** `python3 examples/conv_depthwise_by_y_layout_no_submit.py c256_h28_by_yk_task` PASS; `python3 examples/conv_depthwise_by_y_layout_no_submit.py c128_h56_by_yk_task` PASS; `python3 -m py_compile conv_expt/rknn_prefix_replay.py examples/conv.py examples/conv_crash_fence_no_submit.py examples/conv_pointwise_by_yk_layout_no_submit.py examples/conv_depthwise_by_y_layout_no_submit.py` PASS; post `python3 examples/simple_add.py` PASS.
**Latest difficult-shape prefix replay (2026-06-02 23:35):** After the crash report, `python3 examples/simple_add.py` PASS. Generated `/home/orangepi/npu/ops_rknn/models/match_b1_c512_h14_w14_oc512_wic1_k3x3_g512.rknn` and ran only safe linear one-task prefix replay: `KEEP_TASKS=1 PREFIX_MODE=linear MAX_ERRORS=8 DUMP_GEM=1 ... c512_h14_dw_by_k` and then `DUMP_GEM=2`. Both runs patched submit to `task_number=1`, `subcore_task[0]=(0,1)`, other lanes zero, and both passed (`max error: 0.0076313`). Post `python3 examples/simple_add.py` PASS. GEM1 task metadata is exact11 `amounts=108,104,26,104,26,104,26,104,26,104,26`, masks `0x0d/0x60`, offsets `0,112,224,256,368,400,512,544,656,688,800`. GEM2 body rows decode as `setup,setup,ppu_pdp,setup,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp`, row lengths `115,106,29,106,29,106,29,106,29,106,29`, full `out_c=512`, input/output windows `14/11`, `8/5`, `8/5`, then three `6/3` y tiles. Added manifest target `cc_c512_h14_dw_by_k`.
**Current blocker table (2026-06-02 23:38):**
| Family | Remaining in latest coverage | Biggest blocker | Best evidence | Next safe action |
|---|---:|---|---|---|
| `BY_K/k_tile` | 47 before h7 pointwise manifest update | Mixed subfamilies. Local k_tile planner is not a universal RKNN closure: depthwise c512_h14 uses setup/y_tile rows, while pointwise h7 uses exact11 setup/k_half/k_tile with large channel windows. | c512_h14 depthwise GEM2 and c512_h7 pointwise GEM2 are both safe one-task prefix captures with post health PASS. | Add no-submit materializers for each subfamily, then promote only after exact body/weight/output parity. |
| `BY_YK/k_half,k_tile,setup` | 17 | RKNN task topology has extra prologue/setup rows and uneven windows; local planner emits equal BY_YK rows. | c64_h112, c128_h56, c256_h28 depthwise task layouts; c256_h28 pointwise detailed closure target. | Capture missing GEM2 bodies and implement row materializers. |
| `NONE/setup` | 1 | `b1_c16_h80_oc128` generic exact11 reuse gives `inf`; likely custom multi-task/y-tile. | Negative probe only. | Safe matched-model `KEEP_TASKS=1 PREFIX_MODE=linear` GEM1/GEM2 capture. |
| Crash-fenced narrow pointwise | 2 known | Prior speculative local submit rebooted board. | c256_h2_oc64 GEM1/GEM2 no-submit evidence proves exact11 multi-row, not oc24 setup108. | Keep fenced until full local regcmd parity exists; do not submit. |
**Latest pointwise BY_K prefix replay (2026-06-02 23:39):** Generated `/home/orangepi/npu/ops_rknn/models/match_b1_c512_h7_w7_oc1024_wic512_k1x1_g1.rknn` and ran only safe linear one-task prefix replay: `KEEP_TASKS=1 PREFIX_MODE=linear MAX_ERRORS=8 DUMP_GEM=1 ... c512_h7_oc1024_pw_by_k` and then `DUMP_GEM=2`. Both runs patched submit to `task_number=1`, `subcore_task[0]=(0,1)`, other lanes zero, and both passed (`max error: 0.0458069`). Post `python3 examples/simple_add.py` PASS. GEM1 task metadata is exact11 `amounts=108,104,26,104,26,104,26,104,26,104,26`, masks `0x0d/0x60`, offsets `0,112,224,256,368,400,512,544,656,688,800`. GEM2 body rows decode as `setup,k_half,ppu_pdp,k_half,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp`, row lengths `155,106,29,106,29,106,29,106,29,106,29`, with active out_c sequence `1024,512,512,352,336,336`, `CBUF0=0xa2`, `DATA_SIZE1=0x003f0200`, `DMA_CON2=0x15`, `DST_SURF_STRIDE=0x34`, `SURFACE_ADD=0x68`, and `WEIGHT_SIZE0=0x100000/0x80000/0x58000/0x54000/0x54000`. Added manifest targets `cc_c512_h7_oc1024_pw_by_k` and alias `c512_h7_oc1024_pw_by_k`. This is evidence only; local `examples/conv.py` remains fenced until the exact body and offsets are materialized.
**Crash follow-up (current session):** User reported another crash after difficult-shape prefix replay. `python3 examples/simple_add.py` PASS after recovery. Verified `/home/orangepi/npu/ops_rknn/ioctl_prefix_replay.gdb` now refuses `PREFIX_MODE=subcores` unless `ALLOW_UNSAFE_PREFIX_REPLAY=1`, even with `KEEP_TASKS=1`; subcore mode expands the patched submit to `task_number=3` and is not safe-prefix replay. Safe command remains only `KEEP_TASKS=1 PREFIX_MODE=linear ...`. Revalidated no-submit/safe guards: `python3 -m py_compile conv_expt/rknn_prefix_replay.py examples/conv.py examples/conv_crash_fence_no_submit.py examples/conv_pointwise_by_yk_layout_no_submit.py examples/conv_depthwise_by_y_layout_no_submit.py` PASS; `python3 conv_expt/rknn_prefix_replay.py --coverage-summary /tmp/opencode/conv_py_217_sweep_20260602_223030_summary.txt` reports `manifest_covered=12 uncovered_total=71`; `python3 examples/conv_crash_fence_no_submit.py c256_h2_oc64` PASS and confirms oc64 is exact11, not oc24 setup108; direct `python3 examples/conv.py b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid` exits before submit with the crash-fence error.
**Latest no-submit pointwise BY_YK evidence (current session):** Added `examples/conv_pointwise_by_yk_layout_no_submit.py`. `python3 examples/conv_pointwise_by_yk_layout_no_submit.py cc_c256_h28` asserts the RKNN 14-row closure for `conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1`: `setup,setup,k_half,k_half,ppu_pdp,k_half,k_half,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp,y_tile,ppu_pdp`, input windows `18,10,18,10,...,10,9,9`, full-OC setup/y_tile rows, 128-channel k_half rows, constants `DATA_SIZE1=0x003f0100 DMA_CON2=0x2a0 DST_SURF_STRIDE=0x310 SURFACE_ADD=0x620 CBUF0=0x48/0x2048`. It now also asserts dynamic deltas: feature offsets `0,0x1f80,0,0x1f80,...,0,0x1180,0x2140`, weight offsets `0/0x10000` for the two k-half channel blocks, and output offsets `0,0x1f80,0,0x1f80,0x31000,0x32f80,0,0x1180,0x2140`. The checker also asserts that unprefixed `b1_c256_h28_w28_oc256_wic256_k1x1_g1` has an identical local planner descriptor sequence, so the same RKNN closure evidence covers both remaining fenced aliases. Local planner currently emits 48 rows of 32-OC tiles, so these shapes remain fenced but now have a precise no-DRM closure target.
**Owner:** Codex session (continuing from a prior model that handed off at 13:20).
**CWD:** `/home/orangepi/rk3588`
**Latest sweep snapshot:** `/tmp/opencode/conv_py_217_sweep_20260602_205149_summary.txt`
**Commit:** d464ea1 (Fix 4 regressions: remove buggy DMA_CON2 patches)
**NPU health (re-verified 2026-06-02 17:49):** `python3 examples/simple_add.py` PASS before and after the full sweep. `examples/kernel_6_18/simple_add.py` is not the right path on this 6.1.99 kernel; it failed at BO mmap with `OSError: [Errno 22] Invalid argument`.
**NPU health (re-verified 2026-06-02 17:58):** `python3 examples/simple_add.py` PASS after a controlled one-task probe of `b1_c512_h14_w14_oc32_wic512_k1x1_g1`.
**NPU health (re-verified 2026-06-02 18:09):** full `python3 sweep_217.py --output-dir /tmp/opencode` ran pre/post `examples/simple_add.py`, both rc=0.
**NPU health (re-verified 2026-06-02 18:15):** full `python3 sweep_217.py --output-dir /tmp/opencode` ran pre/post `examples/simple_add.py`, both rc=0.
**NPU health (re-verified 2026-06-02 18:19):** full `python3 sweep_217.py --output-dir /tmp/opencode` ran pre/post `examples/simple_add.py`, both rc=0.
**NPU health (re-verified 2026-06-02 18:31):** `python3 examples/simple_add.py` PASS after RKNN prefix replay on `b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid`.
**Latest prefix replay (2026-06-02 18:30-18:31):** added `/home/orangepi/npu/ops_rknn/ioctl_prefix_replay.gdb` as an env-driven copy of `ioctl.gdb` with `_patch_submit` enabled. For `c128_h1_oc24_pw_none`, `KEEP_TASKS=1` completed but failed numerically (24/24 mismatches; first 8 errors up to 48.9215), and `KEEP_TASKS=2` hit RKNN submit failure after the 6000 ms task timeout. Do not run `KEEP_TASKS=3` on this shape until same-regcmd 3-subcore semantics are handled without collapsing tasks onto subcore0.
**NPU health (re-verified 2026-06-02 current session):** `python3 examples/simple_add.py` PASS after targeted c528 h14 exact11 checks and after the rejected `c16_h80_oc128` exact11 reuse probe.
**NPU health (re-verified 2026-06-02 19:51):** `python3 examples/simple_add.py` PASS before and after c512_h14_oc24 RKNN prefix replay and after the rejected local visible-row replay probe.
**Latest prefix replay (2026-06-02 19:51):** `match_b1_c512_h14_w14_oc24_wic512_k1x1_g1` RKNN prefix replay with `KEEP_TASKS=1 PREFIX_MODE=subcores` passed (`max error: 0.0539551`). GEM1 task dump: exact11 descriptors `108,104,26,104,26,104,26,104,26,104,26`, masks `0x0d/0x60`, offsets `0,112,224,256,368,400,512,544,656,688,800`; original submit is three one-task subcores. GEM2 visible body rows: setup `CONV2=0x90`, two full-OC `input_h=7` rows `CONV2=0x10000080`, then y rows `0x20000060/0x20000060/0x20000050`, with `WEIGHT_SIZE0=0x6000`. A scoped conv.py replay of those visible fields still failed (`max_diff=135.3407`), post-health PASS, and the runtime promotion was reverted. This shape stays fenced; next hypothesis is hidden RKNN buffer binding/data parity/aux semantics, not the visible exact11 task metadata.
**Latest prefix replay (2026-06-02 19:59):** `conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32` depthwise BY_Y `KEEP_TASKS=1 PREFIX_MODE=linear` completed with expected partial-output failure and post `simple_add.py` PASS. It revealed original submit topology: `task_number=9`, `subcore_task=(0,3),(0,3),(0,3),(0,0),(0,0)`. Fresh GEM1 task dump matches the old manifest: amounts `108,108,17,104,104,26,104,104,26,104,26,104,26,104,26`, masks `0x0d/0x60`, offsets `0,112,224,248,360,472,504,616,728,760,872,904,1016,1048,1160`. RKNN row constants are now summarized in `conv_expt/rknn_prefix_replay.py`: `CBUF0=0x1b/0x201b`, `DATA_SIZE1=0x001f0020`, `DMA_CON2=0x2f40`, `DST_SURF_STRIDE=0x2f44`, `SURFACE_ADD=0xbd10`. Local planner's 10-row equal BY_Y plan is not RKNN-equivalent; closure needs this 15-row/9-submit-lane structure.
**Latest prefix replay (2026-06-02 20:08):** three difficult matched RKNN shapes passed with `KEEP_TASKS=1 PREFIX_MODE=linear` and post `python3 examples/simple_add.py` PASS: `b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid` max error `0.0190849`, `b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid` max error `0.0178719`, and `b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid` max error `0.00697899`. All three original submits were three one-task subcores, and all GEM1 dumps decode to exact11 `amounts=108,104,26,104,26,104,26,104,26,104,26`, masks `0x0d/0x60`, offsets `0,112,224,256,368,400,512,544,656,688,800`. Artifacts are recorded in `conv_expt/rknn_prefix_replay.py` as `c256_h3_oc24_pw_none`, `c40_h40_oc320_pw_by_yk`, and `dw96_h20_by_yk`.
**Latest prefix replay (2026-06-02 20:23-20:26):** ran three more `KEEP_TASKS=1 PREFIX_MODE=linear` GDB prefix replays, bracketed by `python3 examples/simple_add.py` health checks. `b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid` exited cleanly but failed numerically; transcript `/tmp/opencode/prefix_replay_c576_h19_keep1_20260602_202343.log`. `match_b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid` passed with max error `0.0201073`; transcript `/tmp/opencode/prefix_replay_c256_h2_keep1_20260602_202437.log`. `match_b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid` re-confirmed PASS with max error `0.0178719`; transcript `/tmp/opencode/prefix_replay_c40_h40_keep1_20260602_202628.log`. Decoded patched submits used `task_number=1`, `subcore_task[0]=(0,1)`, all other subcore lanes zero. Post-run `simple_add.py` PASS.
**Latest promotion (2026-06-02 20:44):** `b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid` now passes local `examples/conv.py` via a one-task setup108 body and compact pointwise weight packing. GEM2 dump `/home/orangepi/npu/ops_rknn/dump/prefix_c256_h2_none_keep1_gem2/dump_gem2.txt` has the active row at offset `0x3840`; generated `_exact11_body_regs(..., "setup")` matches all 108 qwords with captured DMA bases after overrides `CBUF0=0xb1`, `CONV2_LOW=0x30`, `DATA_SIZE1=0x003f0100`, `CVT_CON0=0xb`, `DMA_CON2=0x0ffffffc`. The missing piece was weight layout: standard `_pack_pointwise_wide` padded to `0x4000` bytes and made channels 16-23 wrong, while compact 24-kernel packing matches RKNN `WEIGHT_SIZE0=0x3000` and passes. Validation: direct run PASS `max_diff=0.0245` with pre/post `simple_add.py` PASS, and one-shape sweep `/tmp/opencode/conv_py_217_sweep_20260602_204347_summary.txt` PASS with pre/post health rc=0.
**Latest negative probe (2026-06-02 20:43):** `b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid` GEM2 capture completed at `/home/orangepi/npu/ops_rknn/dump/prefix_c256_h3_none_keep1_gem2/dump_gem2.txt`, but it shows a different multi-row active body/tail structure. Do not reuse the `c256_h2` single setup108 compact-weight path blindly.
**Latest depthwise probe (2026-06-02 20:54-20:55):** Fresh `DUMP_GEM=2` RKNN prefix replay for `match_b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid` passed (`KEEP_TASKS=1 PREFIX_MODE=linear`, max error `0.00697899`) and saved `/home/orangepi/npu/ops_rknn/dump/prefix_fresh_dw_c96_h20_keep1_gem2/dump_gem2.txt`. The captured task-0 setup row is 108 qwords and the local `make_depthwise_setup108_regs` now matches that row exactly except dynamic DMA addresses. Local one-task replay no longer times out but still fails numerically (`max_diff=23.1198` with OC-major compact weights, `25.4591` with C1/H/W/C2 compact weights), with post `simple_add.py` PASS. Keep `DEPTHWISE_SETUP108_SHAPES` empty; the remaining blocker is local input/weight/buffer layout parity, not visible register fields.
**Latest promotion (2026-06-02 21:10):** `b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid` now passes local `examples/conv.py` with a shape-scoped exact11 compact-weight PC-chain replay. RKNN prefix evidence already showed `KEEP_TASKS=1 PREFIX_MODE=linear` PASS max error `0.0190849`; the useful GEM2 row body is `/home/orangepi/npu/ops_rknn/dump/prefix_c256_h3_none_keep1_gem2/dump_gem2.txt`. Captured row schedule: setup h3 `CONV2=0x40`, h2 `0x10000030`, h1 `0x10000020`, then y_start 0/1/2 rows `0x20000020`, all with compact `WEIGHT_SIZE0=0x3000`, `CBUF0=0xb1`, `DATA_SIZE1=0x003f0100`, `DMA_CON2=0x0ffffffd`. Local direct validation PASS `max_diff=0.0149`; one-shape sweep `/tmp/opencode/conv_py_217_sweep_20260602_211034_summary.txt` PASS with pre/post health rc=0. Fresh full sweep `/tmp/opencode/conv_py_217_sweep_20260602_211043_summary.txt` has `total=217 counts={'PASS': 133, 'FENCED': 84}` with `pre_health_rc=0 post_health_rc=0`.
**Latest crash/reboot (2026-06-02 22:11):** A speculative local monkey-patch/promotion attempt reused the c256_h2 oc24 setup108 compact-weight constants for `b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid`. The command printed PASS and post `simple_add.py` PASS, but the user reported a crash and the next `uptime` showed only 4 minutes, confirming a board reboot. The persisted oc64 entries were removed from `examples/conv.py`. Do not promote or probe c256_h2 oc64/oc546 by generalization; first capture their own RKNN prefix GEM1/GEM2 task/body rows.
**Latest promotion (2026-06-02 22:24):** `b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid` now passes local `examples/conv.py` through one-task setup108 compact-weight replay, max_diff `0.0155`, with pre/post `python3 examples/simple_add.py` PASS. Safe RKNN prefix replay command was `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2 DUMP_DIR=/home/orangepi/npu/ops_rknn/dump/prefix_c40_h40_oc320_keep1_gem2 gdb -q -x ioctl_prefix_replay.gdb --args ./conv2d_multi --case match_b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid 1 40 40 40 320 1 1 1 c40_h40_oc320_pw_by_yk`, which passed max error `0.0178719` and produced GEM2 body evidence. Setup row fields: `CBUF0=0x84`, `CONV2=0x160`, `DATA_SIZE1=0x00270028`, `CBUF1=0x32`, `WEIGHT_SIZE0=0x6400`, `WEIGHT_SIZE1=0x50`, `DMA_CON2=0x5a0`, `FC_DATA_SIZE1=0x28`. A local 11-task pointwise exact11 replay using the later y rows failed numerically (`max_diff=92.5402`) while health stayed PASS, so this shape is intentionally promoted via task-0 setup108 only.
**Latest full sweep (2026-06-02 22:26):** `/tmp/opencode/conv_py_217_sweep_20260602_222614_summary.txt` reports `total=217 counts={'PASS': 134, 'FENCED': 83}`, `pre_health_rc=0 post_health_rc=0`.
**Latest crash-fence recovery (2026-06-02 22:30):** NPU health rechecked with `python3 examples/simple_add.py` PASS after reboot. Added `CRASH_FENCED_SHAPES` in `examples/conv.py` for c256_h2 oc64/oc546 so direct runs stop with `shape is crash-fenced...` before DRM allocation or submit. Verified oc64 and oc546 direct runs return rc=1 before submit, proven c256_h2 oc24 still PASS, and full sweep `/tmp/opencode/conv_py_217_sweep_20260602_223030_summary.txt` reports `total=217 counts={'PASS': 134, 'FENCED': 83}`, `pre_health_rc=0 post_health_rc=0`.
**Latest promotion (2026-06-02 20:18):** `b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid` now passes local `examples/conv.py` with exact11 OC-k-tile replay, max_diff `0.0306`, and pre/post `simple_add.py` PASS. RKNN prefix replay first showed exact11 task topology and GEM2 body fields `CBUF0=0x2a`, `DATA_SIZE1=0x003f00c0`, `DMA_CON2=0x2a0`, `SURFACE_ADD=0x6200`, `CONV2` sequence `0x90`, two rows `0x40000090`, then three rows `0x50000090`. A previous y-window reuse failed final tile; the passing local path uses k_tile splits `0:32/32:32/64:32`.
**Latest no-submit checker (2026-06-02 22:38):** extended `examples/conv_depthwise_by_y_layout_no_submit.py` with selectable cases. `c32_by_y` asserts the `conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32` 15-task RKNN closure (`rknn_input_h=50,50,16,30,29,...,39,39,38` vs local `14x9,4`). `c64_by_yk` asserts the `conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64` 23-task RKNN closure with amounts `108,108,12,12,16,16,104,104,104,26,104,104,104,26,104,104,26,104,104,26,104,104,26`, constants `CBUF0=0x1b/0x201b`, `DATA_SIZE1=0x003f0040`, `DMA_CON2=0x2f40`, `DST_SURF_STRIDE=0x2f44`, `SURFACE_ADD=0xbd10`, and local planner mismatch (`60` equal BY_YK rows). These are no-DRM/no-submit preflight artifacts for implementing depthwise closures.
**Depthwise alias coverage (current session):** `examples/conv_depthwise_by_y_layout_no_submit.py` now asserts that unprefixed `b1_c32_h112_w112_oc32_wic1_k3x3_g32` and `b1_c64_h112_w112_oc64_wic1_k3x3_g64` have identical local planner descriptor sequences to the corresponding `conv2d_cc_...` RKNN dump shapes. Added manifest targets `c32_h112_dw_by_y_dump` and `c64_h112_dw_by_yk_dump`, so the same RKNN closure evidence covers four fenced depthwise aliases instead of only the two prefixed names.
**Validation (current session):** no-submit/syntax/coverage/health checks passed: `python3 -m py_compile examples/conv_pointwise_by_yk_layout_no_submit.py examples/conv_depthwise_by_y_layout_no_submit.py conv_expt/rknn_prefix_replay.py`; `python3 examples/conv_pointwise_by_yk_layout_no_submit.py cc_c256_h28`; `python3 examples/conv_depthwise_by_y_layout_no_submit.py c32_by_y`; `python3 examples/conv_depthwise_by_y_layout_no_submit.py c64_by_yk`; `python3 conv_expt/rknn_prefix_replay.py --coverage-summary /tmp/opencode/conv_py_217_sweep_20260602_223030_summary.txt` => `manifest_covered=11 uncovered_total=72`; `python3 examples/simple_add.py` PASS. No difficult-shape submit was run.
**Crash-fence no-submit evidence (current session):** Added `examples/conv_crash_fence_no_submit.py`. `python3 examples/conv_crash_fence_no_submit.py c256_h2_oc64` asserts existing RKNN dumps for crash-fenced `b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid`: GEM1 exact11 task metadata `108,104,26,104,26,104,26,104,26,104,26` with masks `0x0d/0x60` and offsets `0,112,224,256,368,400,512,544,656,688,800`; GEM2 body rows `setup,k_half,ppu_pdp,k_half,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp`, row lengths `130,106,29,106,29,106,29,106,29,106,29`, `CBUF0=0xb1 DATA_SIZE1=0x003f0100 DMA_CON2=0x0ffffffc`, setup `WEIGHT_SIZE0=0x8000`, and k split `32,16,16`. This proves the crashed generalization was wrong: oc64 is exact11 multi-row, not the oc24 one-task setup108 closure. Added manifest target `c256_h2_oc64_crash_fence`; still no hardware submit.
**Validation update (current session):** `python3 -m py_compile examples/conv_crash_fence_no_submit.py conv_expt/rknn_prefix_replay.py` PASS; `python3 examples/conv_crash_fence_no_submit.py c256_h2_oc64` PASS; coverage now reports `manifest_covered=12 uncovered_total=71`; `python3 examples/simple_add.py` PASS.
**Latest no-submit materialization (current session):** Extended `examples/conv_crash_fence_no_submit.py c256_h2_oc64` to generate a local exact11 candidate from `examples.conv._exact11_body_regs` and compare critical fields against RKNN GEM1/GEM2 evidence, still with no DRM/no submit. The materialized row family/order now matches RKNN: `setup,k_half,ppu_pdp,k_half,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp,k_tile,ppu_pdp`; out channel sequence `64,32,none,32,none,32,none,16,none,16,none`; `WEIGHT_SIZE0=0x8000,0x4000,none,0x4000,none,0x4000,none,0x2000,none,0x2000,none`; DST offsets `0,0,none,0x100,none,0,none,0x100,none,0x180,none`; constants `CBUF0=0xb1 DATA_SIZE1=0x003f0100 DMA_CON2=0x0ffffffc`. The important local emitter correction is that the 16-channel k_tile rows need explicit CORE/DPU channel-end patches; the default emitter rounds them to 32 channels. Validation: `python3 examples/conv_crash_fence_no_submit.py c256_h2_oc64` PASS; `python3 -m py_compile examples/conv_crash_fence_no_submit.py` PASS; direct `python3 examples/conv.py b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid` still exits before submit with crash-fence error; coverage remains `manifest_covered=12 uncovered_total=71`; post `python3 examples/simple_add.py` PASS.
**Latest conv.py preflight (current session):** Added a no-DRM dry-run path in `examples/conv.py` for crash-fenced `b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid`: `python3 examples/conv.py b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid --dry-run-c256-h2-oc64-exact11`. It materializes the same exact11 candidate inside the runtime file and asserts the RKNN-derived row order, amounts, masks, offsets, out-channel sequence, `WEIGHT_SIZE0`, destination offsets, `CBUF0=0xb1`, `DATA_SIZE1=0x003f0100`, and `DMA_CON2=0x0ffffffc`. Normal execution remains crash-fenced and still exits before DRM allocation/submit. Validation: `python3 -m py_compile examples/conv.py examples/conv_crash_fence_no_submit.py` PASS; the new dry-run command PASS; normal direct command returns the crash-fence error; `python3 examples/conv_crash_fence_no_submit.py c256_h2_oc64` PASS; coverage remains `manifest_covered=12 uncovered_total=71`; post `python3 examples/simple_add.py` PASS. This is not a promotion: no local oc64 submit was run because the available evidence does not yet show a safe numeric PASS for local replay after the previous reboot.
**Latest targeted checks (2026-06-02 current session):** all eight c528 h14 pointwise exact11 BY_K shapes that were stale-FENCED in `/tmp/opencode/conv_py_217_sweep_20260602_181917_summary.txt` passed direct `examples/conv.py` runs: oc32/128/160/256 and `_s1_pvalid` aliases, max_diff 0.0311-0.0312. Negative probe: temporarily enabling `b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid` on the generic exact11 BY_K path completed but failed with `max_diff=inf` across all OC blocks. The promotion was reverted; keep this shape in `KNOWN_BAD_SPATIAL_SETUP_SHAPES` pending the custom 4-task/y-tile structure.
**Last full sweep outcome (2026-06-02 20:51):**
```
total=217 counts={'PASS': 132, 'FENCED': 85, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}
pre_health_rc=-1  post_health_rc=-1  # sweep was run with --skip-health; separate simple_add checks passed before/after probes
snapshot: /tmp/opencode/conv_py_217_sweep_20260602_205149_summary.txt
```

**About the "crash"** the user just reported: NPU is healthy right now (`simple_add.py` PASS at 13:25). What the user observed is most likely the **regression from the c64_h1_oc128 promotion turn**: 4 previously-PASS shapes now FAIL with `max_diff=42-117`, plus 1 ERROR (depthwise c256_h10 timeout). See §0.4 and §6 Step 1 for the recovery plan.

---

## 0. TL;DR (read this first)

1. **Goal:** Use prefix replay to debug and fix every failed/fenced shape in `examples/conv.py`. End state: **PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0**, with pre/post `python3 examples/simple_add.py` both PASS.
2. **Current sweep state (last full sweep 20:51):** PASS=132, FENCED=85, FAIL=0, ERROR=0, TIMEOUT=0. This sweep used `--skip-health`; separate `simple_add.py` checks passed around the latest prefix/layout probes.
3. **Latest verified progress:** the c528 h14 pointwise exact11 BY_K family is fully counted as PASS in the fresh sweep: oc32/128/160/256 and `_s1_pvalid` aliases.
4. **Negative probes to avoid repeating:** the same guessed h14 body fails for `b1_c512_h14_w14_oc24_wic512_k1x1_g1` (max_diff=135.3407), times out for `b1_c480_h14_w14_oc16_wic480_k1x1_g1`, and generic exact11 reuse for `b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid` returns `inf` output across all OC blocks. These stay fenced pending RKNN-specific closure.
5. **Biggest blocker right now:** deriving correct prefix-replay bodies for the remaining fenced families, especially Attack A BY_K 1x1 and depthwise BY_YK/BY_K.
6. **Current fenced coverage (fresh sweep):** `rknn_prefix_replay.py --coverage-summary /tmp/opencode/conv_py_217_sweep_20260602_205149_summary.txt` reports `fenced_total=85`, `manifest_covered=10`, `uncovered_total=75`: 49 `BY_K/k_tile`, 23 `BY_YK`, 2 depthwise `BY_Y`, and 1 `NONE` setup fence.

---

## 1. Goal (End State)

| Metric | Target |
|---|---:|
| PASS | 217 |
| FENCED | 0 |
| FAIL | 0 |
| ERROR | 0 |
| TIMEOUT | 0 |
| Pre-sweep `simple_add.py` | PASS |
| Post-sweep `simple_add.py` | PASS |
| `git diff` of `examples/conv.py` | reviewable, no debug print, no dead code |
| Manifest entries in `conv_expt/rknn_prefix_replay.py` | one per promoted shape, with `note: "promoted via ..."` |

The fences are **NOT NPU-driver bugs**. The 12-shape rknn_runtime cross-check has proven the NPU can compute every representative fenced shape correctly (max err 0.0070-0.0447). The bug is in the register-submit setup that `conv.py` produces. The job is to derive the correct register setup for each fenced family, replay it in `make_regs` / `run_*` paths, and add the shape to the manifest.

### 1.1 New evidence / tooling (2026-06-02 17:58)

Added `experimental/rknn/extract_live_regcmd_rows.py`, a no-submit parser for old `capture_rknpu_submit_dump_gems*_live_regcmd*.log` files. It decodes `regcmd_qword[...]` windows into row summaries and prints the body fields needed for prefix replay (`CONV_CON2`, `DATA_SIZE*`, `WEIGHT_SIZE*`, `CBUF_CON*`, `DMA_CON*`, `CORE_DATAOUT_SIZE_*`, DPU output fields, and PC tails).

Validation:

```bash
python3 experimental/rknn/extract_live_regcmd_rows.py experimental/rknn/capture_rknpu_submit_dump_gems_h14_live_regcmd_832_20260524_124752.log --row 0 --row 1 --row 2 --row 3 --row 4
python3 experimental/rknn/extract_live_regcmd_rows.py experimental/rknn/capture_rknpu_submit_dump_gems_sweep_live_regcmd.log --row 0 --row 1 --row 2 --row 3 --row 4
```

Observed h14 fields match the promoted exact-11 BY_K body: setup `CBUF_CON0=0xa2`, `CONV_CON2=0x000000f0`, `DMA_CON2=0x8c`, `WEIGHT_SIZE0=0xe1000`; k_half `CONV_CON2=0x400000f0`, `WEIGHT_SIZE0=0x70800`; DPU `SURFACE_ADD=0x1200`, `DST_SURF_STRIDE=0x900`.

The same extractor showed `experimental/rknn/capture_rknpu_submit_dump_gems_pw_c256_h14_live_regcmd_832_20260524_131749.log` is not a useful regcmd-body window: decoded targets look like fp16/data payload, not register targets. Do not use that file as body-field evidence.

Controlled probe:

```bash
timeout 30 python3 - <<'PY'
import numpy as np
from examples.conv import shape_from_name, _run_single_tile_shape, compute_expected_vectorized
name='b1_c512_h14_w14_oc32_wic512_k1x1_g1'
s=shape_from_name(name)
np.random.seed(42)
inp=np.random.uniform(-2,2,(s['batch'],s['in_c'],s['in_h'],s['in_w'])).astype(np.float16)
wt=np.random.uniform(-2,2,(s['out_c'],s['weight_in_c'],s['kh'],s['kw'])).astype(np.float16)
got=_run_single_tile_shape(s, inp[0], wt).reshape(1,s['out_c'],s['in_h'],s['in_w'])
exp=compute_expected_vectorized(inp, wt, s)
print(float(np.max(np.abs(got.astype(np.float32)-exp))), bool(np.allclose(got,exp,atol=0.12)))
PY
python3 examples/simple_add.py
```

Result: generic one-task replay for `b1_c512_h14_w14_oc32_wic512_k1x1_g1` is numerically wrong (`max_diff=135.6888`, `allclose=False`), and post-probe `simple_add.py` PASS. Keep pointwise-wide NONE fenced until the RKNN 108-row body is captured.

---

## 2. Current Sweep State (re-verified 2026-06-02 13:25)

```
total=217 counts={'PASS': 122, 'FENCED': 95, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}
pre_health_rc=0  post_health_rc=0
snapshot: /tmp/opencode/conv_py_217_sweep_20260602_181917_summary.txt
```

### 2.0 Latest h14 pointwise BY_K promotions (2026-06-02 18:19)

Added these shapes to `POINTWISE_EXACT11_BYK_SHAPES` with the h14 three-window body `(0+5:0x60, 5+5:0x60, 10+4:0x50)`, `CBUF0=0x57`, `DMA_CON2=0x8c`, `CVT_CON0=0xb`, and per-input-channel `DATA_SIZE1`:
- `b1_c512_h14_w14_oc112_wic512_k1x1_g1` — `DATA_SIZE1=0x003f0200`, PASS max_diff=0.0312
- `b1_c384_h14_w14_oc96_wic384_k1x1_g1` — `DATA_SIZE1=0x00bf0180`, PASS max_diff=0.0312
- `b1_c480_h14_w14_oc96_wic480_k1x1_g1` — `DATA_SIZE1=0x00ef01e0`, PASS max_diff=0.0312
- `b1_c512_h14_w14_oc512_wic512_k1x1_g1` — `DATA_SIZE1=0x003f0200`, PASS max_diff=0.0312
- `conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1` — `DATA_SIZE1=0x003f0200`, PASS max_diff=0.0312
- `b1_c576_h14_w14_oc96_wic576_k1x1_g1` — `DATA_SIZE1=0x011f0240`, PASS max_diff=0.0312

Full sweep after latest promotion: `/tmp/opencode/conv_py_217_sweep_20260602_181917_summary.txt`, PASS=122/FENCED=95, no FAIL/ERROR/TIMEOUT, pre/post health rc=0.

### 2.1 The 4 regressions (FIXED this turn)

**Root cause:** the `DMA_CON2` patches added to `make_y_tile_regs`, `make_k_tile_regs`, and `make_yk_pointwise_regs` used `s["in_h"]` and `p["width_stride"]` (the **full** shape) but the `make_regs` default they "default" to uses the **tile's** `in_h`. For shapes NOT in `DMA_CON2_OVERRIDES` (99% of shapes), the patch silently overrode the correct tile value with the wrong full-shape value, corrupting the DMA stride by `(in_h_full - in_h_tile) * width_stride` bytes.

**Fix:** removed the 3 buggy `DMA_CON2` patches from these helper functions. The `make_regs` default already writes the correct tile-based `DMA_CON2`. All 4 regressions now pass:
- `conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1` — max_diff=0.0307 (was 116.88)
- `b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid` — max_diff=0.0131 (was 43.12)
- `b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid` — max_diff=0.0156 (was 45.90)
- `b1_c32_h112_w112_oc16_wic32_k1x1_g1` — max_diff=0.0146 (was 42.98)

The c64_h1_oc128 1x1 promotion is preserved (max_diff=0.0078 unchanged).


| Shape | Path | max_diff | Was at 12:04 |
|---|---|---:|---|
| `conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1` | `run_pointwise_chained_y_shape` L1531 | **116.88** | PASS |
| `b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid` | `run_shape` L1772 | **43.12** | PASS |
| `b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid` | `run_shape` L1772 | **45.90** | PASS |
| `b1_c32_h112_w112_oc16_wic32_k1x1_g1` | `run_shape` L1772 | **42.98** | PASS |

All 4 share: **1x1, c=32, small oc (8/16/32), large h*w (112-160)**. They all run via `local_tile_replay` (multiple y-tiles, `hw_out=fp16`). The error is at `run_shape` L1772 (`AssertionError: output mismatch max_diff=...`).

The 1x1 promotion in this turn added c64_h1_oc128 to `PREFIX_BY_K_SHAPES`. That should NOT have affected these 4 shapes (they don't share the path). Most likely: a generic `make_regs` change (e.g., one of the 3 generic fixes in L432-509) interacts badly with the BY_Y/y-tile body for these specific shapes, OR a change in `run_local_tile_replay_shape` or `_tile_replay_specs` lookup. **Hypothesis to test:** `git stash`, re-sweep to confirm 114 baseline, then re-apply changes one by one to bisect. (See §6 Step 1.)

### 2.2 Depthwise c256_h10 ERROR (DEFERRED)

| Shape | Status | Notes |
|---|---|---|
| `b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid` | FENCED (back to fence this turn) | DEPTHWISE_BODY_SHAPES emptied; single-task rewrite needs more capture work |

Tried the single-task rewrite this turn (emit `make_depthwise_setup_regs` instead of 8 k_tile tasks). The NPU still timed out. The live capture body fields are partially decoded but the structure (single task vs multi-task with per-tile in_h=6) is unclear. Deferred until more capture work is done.

### 2.3 Fence histogram (101 total, from 13:20 detail log)

| Count | Fence error | Attack ID | Strategy |
|---:|---|---|---|
| 51 | `BY_K/k_tile is fenced pending RKNN 108/104/26-row closure` | A | 11-task BY_K with per-shape overrides; **1x1 sub-family** needs a NEW task family (path now proven via c64_h1_oc128) |
| 19 | `depthwise BY_YK closure is unfenced without DEPTHWISE_BODY_SHAPES membership` | C | RKNN row derivation, inherits from B body |
| 13 | `depthwise BY_K closure is unfenced without DEPTHWISE_BODY_SHAPES membership` | B | depthwise-specific body (scaffold added; needs single-task rewrite) |
| 7  | `BY_YK is disabled before allocation; mixed Y/K setup and k_half semantics are unresolved` | E | k_half derivation |
| 6  | `pointwise-wide NONE is fenced pending RKNN 108-row closure` | D (DEFERRED) | 3-subcore setup body for c=24, needs 3KB aux weight |
| 3  | `depthwise BY_Y/y_tile closure is unfenced without DEPTHWISE_BODY_SHAPES membership` | G | depthwise body patches (scaffold added) |
| 1  | `spatial setup/NONE path is fenced as known numerically wrong for this shape` | F | c16_h80_oc128 (3x3) — needs custom 4-task writer |
| 1  | `pointwise-wide BY_Y is fenced pending proven row closure` | H | c576_h19_oc12 12-task |
| **101** | | | |

(Note: count for "BY_K/k_tile" went 52→51 because c64_h1_oc128 was promoted. Count for "depthwise BY_K" went 14→13 because c256_h10_oc256 became ERROR (no longer FENCED).)

---

## 3. Working Tree (uncommitted, all safe to keep)

### 3.1 `examples/conv.py` (1831 lines, +314/-29 since HEAD)

#### 3.1.1 Override dicts (verified live in `examples/conv.py` L155-241)

```python
CBUF0_OVERRIDES = {
    "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid": 0x0a2,
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid":   0x0a2,
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid":   0x057,
    # 1x1 pointwise family: cbuf0=0xb1 (per live rknn_runtime c64_h1_oc128 capture)
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid":    0x0b1,   # NEW this turn
}
DATA_SIZE1_OVERRIDES = {
    "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid": 0x1f00a0,
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid":   0x1f00a0,
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid":   0x000F0010,
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid":    0x003f0040,  # NEW this turn
}
CONV2_LOW_OVERRIDES = {
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid":   0x1a0,
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid":    0x020,   # NEW this turn
}
CVT_CON0_OVERRIDES = {
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid":    0x000b,  # NEW this turn
}
DMA_CON2_OVERRIDES = {
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid":    0x0ffffffd,  # NEW this turn
}
WEIGHT_SIZE0_OVERRIDES = {
    # 1x1 pointwise c64_h1_oc128: per-family weight sizes from live capture
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "setup"):  0x4000,  # NEW this turn
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_half"): 0x2000,  # NEW this turn
    # k_tiles use planner default = oc_count * in_c * 2
}
KT_TILE_SPLITS = {
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid": ((0, 48), (48, 48), (96, 32)),
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid":   ((0, 64), (48, 48), (96, 32)),  # NEW this turn
}
DST_OFFSETS_OVERRIDES = {
    # 1x1 with h*w=1 needs output_off = oc_start * 2 (c2-packed), not oc_start * out_width_stride * 2
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "setup"):    0x00,  # NEW
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_half", 0):  0x00,  # NEW
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_half", 64): 0x80,  # NEW
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_tile", 0):  0x00,  # NEW
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_tile", 48): 0x60,  # NEW
    ("b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid", "k_tile", 96): 0xc0,  # NEW
}

DEPTHWISE_OVERRIDES = {     # scaffold, body still needs wiring
    "conv_con1": 0x123,
    "conv2_low": 0x90,
    "data_size1": 0x003f0100,
    "cbuf0": 0xa2,
    "cbuf_con1": 0x50,
    "weight_size0": 0x1200,
    "weight_size1": 0x1200,
    "weight_size2": 0x03030001,
    "dma_con2": 0x3c,
    "feature_addr_padding": 0x0,
}
```

#### 3.1.2 Promoted shape sets
```python
PREFIX_BY_K_SHAPES = {
    EXACT11_BYK_SHAPE,                                  # b1_c160_h14_oc320
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid",     # c160_h7_oc320
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid",      # c16_h80_oc64
    "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid",       # c64_h1_oc128 (NEW this turn, 1x1)
}

POINTWISE_EXACT11_BYK_SHAPES = {                         # 4 shapes
    "conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1",
    "conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1",
    "b1_c128_h28_w28_oc256_wic128_k1x1_g1",
    "b1_c256_h14_w14_oc512_wic256_k1x1_g1",
}

LOCAL_TILE_REPLAY_SHAPES = POINTWISE_YK_SHAPES | LOCAL_POINTWISE_YK_SHAPES | {
    "conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1",
    "b1_c3_h224_w224_oc32_wic3_k3x3_g1",
    "b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid",
    "b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid",
}

KNOWN_BAD_SPATIAL_SETUP_SHAPES = {
    "b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid",     # 3x3 — needs 4-task writer
    "b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid",     # 5x5 — newly added (now fenced in 13:20)
}

DEPTHWISE_BODY_SHAPES = {                                # NEW this turn
    "b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid",   # scaffolded; body still needs wiring
}
```

#### 3.1.3 Generic `make_regs` fixes (universally safer, do NOT revert)
- `cvt_con5 = 0` when not NHWC
- `CORE_MISC_CFG = 0x200` (not `(2<<8)|is_spatial`)
- `SURFACE_ADD = (out_width_stride * 2) << 4`

#### 3.1.4 New functions added (across this turn and prior turn)
- `run_depthwise_shape(s)` (L1320-1407) — handles single-row + multi-row depthwise (BY_Y, BY_K)
- `full_data_bank=False` kwarg on `make_regs` (L435)
- Short descriptive form parser in `shape_from_name` (L276-296)
- Depthwise dispatch in `main()` (L1731-1736)
- New `_tile_replay_specs` row for `b1_c8_h160_w160_oc16` (L1546-1547)
- `post_submit_reset` + hw_oc padding in `run_grouped_serial_shape` (L1287-1318)
- `(name, family)`-keyed overrides in `_exact11_body_regs` (L630-640)
- New `(name, family, oc_start)`-keyed `DST_OFFSETS_OVERRIDES` lookup in `_exact11_body_regs` (L694-695) — fixes 1x1 h*w=1 dst_base

### 3.2 `conv_expt/rknn_prefix_replay.py` (manifest, 5+1=6 entries this turn)

```python
"g3_c3_h11_w28_spatial_serial": { ... max_diff=0.0038 },
"c32_h14_oc64_promoted_via_generic_fix": { ... max_diff=0.0304 },
"c32_h7_oc128_promoted_via_generic_fix": { ... max_diff=0.0265 },
"c8_h160_oc16_promoted_via_local_tile_replay": { ... max_diff=0.0156 },
"c16_h80_oc64_promoted_via_exact11_byk": { ... max_diff=0.0293 },
"c64_h1_oc128_promoted_via_1x1_byk": { ... max_diff=0.0078 },   # NEW this turn
```

### 3.3 Other modified files
- `examples/conv_tiles.py` (-1298 lines refactor)
- `experimental/rknn/*.gdb, dump_rknpu_task_gems.py` (gdb capture infrastructure)
- `README.md` (notes)
- `current_task.md` (this file)

### 3.4 Untracked scratch files (NOT to be committed per AGENTS.md)
- `examples/conv_h14_k_tile_no_submit.py`
- `examples/conv_h14_task_layout_no_submit.py`
- `examples/conv_h160_setup3_task_layout_no_submit.py`
- `examples/conv_legacy.py`
- `examples/conv_no_submit_*.py` (6 files)
- `examples/conv_spatial_by_y_layout_no_submit.py`
- `examples/conv_tiles_no_submit.py`
- `examples/conv_small_no_submit.py`
- `examples/conv_output_bo_map_no_submit.py`
- `experimental/rknn/capture_c64_state.py`
- `experimental/rknn/capture_rknpu_ioctl_*.log` (40+ logs)
- `/tmp/c64_h1_oc128_capture.log`, `/tmp/c64_h1_oc128_v2.log`, `/tmp/c64_h1_oc128_v3.log`

DO NOT commit these. They are work-in-progress no-submit fixtures for understanding register layouts.

---

## 4. Methodology: Prefix Replay

Prefix replay is the methodology that has driven every promotion. The pattern: capture the live `regcmd` that the rknn_runtime driver submits for a known-correct run, decode each qword into `(target_id, address, value)`, and replay those exact values in `conv.py` for fenced shapes. This is possible because `conv.py`'s regcmd layout is structurally identical to the rknn_runtime's; the fenced shapes fail because `make_regs` uses different body field values (cbuf0, conv2_low, weight_size0, dma_con2, etc.) than the live capture.

### 4.1 Per-shape override derivation workflow (8 steps)

1. **Build matched-weight .rknn** (so rknn_runtime computes the same output as conv.py's expected output):
   ```bash
   python3 /tmp/build_matched.py <shape_name>   # writes /tmp/match_<shape>.rknn
   ln -sf /tmp/match_<shape>.rknn /home/orangepi/npu/ops_rknn/models/<shape>.rknn
   ```
2. **Capture live regcmd via gdb** (regcmd lives in GEM 2 at offset 0x5000-0x7000 typically):
   ```bash
   cat > /tmp/capture_X.gdb << 'EOF'
   set pagination off
   set breakpoint pending on
   cd /home/orangepi/npu/ops_rknn
   python
   import gdb
   class Capture:
       @classmethod
       def handle(cls):
           cmd = int(gdb.parse_and_eval("$x1"))
           nr = (cmd >> 0) & 0xff
           if nr != 0x41: return
           for off in [0x4000, 0x5000, 0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000]:
               gdb.execute(f"shell python3 /home/orangepi/rk3588/experimental/rknn/dump_rknpu_task_gems.py --qword-window 2:{off}:128", to_string=False)
   end
   break ioctl
   commands
     silent
     python Capture.handle()
     continue
   end
   run
   EOF
   gdb -batch -x /tmp/capture_X.gdb --args /home/orangepi/npu/ops_rknn/conv2d_multi_dbg \
       --case <shape> <batch> <c> <h> <w> <oc> <kh> <kw> <g> "match_<name>"
   ```
3. **Decode each qword:** `target = (qword >> 48) & 0xFFFF`, `value = (qword >> 16) & 0xFFFFFFFF`, `addr = qword & 0xFFFF`. Diff against `_exact11_body_regs` / `make_regs` to find which fields differ.
4. **Add per-shape overrides** to the relevant `*_OVERRIDES` dict(s) in `examples/conv.py`. Per-family keys: `KEY_OVERRIDES[(name, "k_half")]` etc. for fields that differ per task family within the same shape.
5. **Promote:** add to `PREFIX_BY_K_SHAPES` (or `POINTWISE_EXACT11_BYK_SHAPES`, etc.). The submit path calls `_exact11_task_regs` (L669-700), which loops over `[setup, k_half+aux, k_half+aux, k_tile*3]` with `EXACT11_BYK_AMOUNTS = (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)`.
6. **Test single shape:**
   ```bash
   timeout 30 python3 examples/conv.py <shape_name>
   ```
7. **Add manifest entry** to `conv_expt/rknn_prefix_replay.py` (with `note: "promoted via ..."`).
8. **Run full sweep:**
   ```bash
   timeout 200 python3 sweep_217.py --skip-health --output-dir /tmp/opencode
   ```

### 4.2 Live regcmd captures (cached in /tmp/)

| Capture | Shape | Region | qwords | Tasks | Decoded? |
|---|---|---|---:|---:|---|
| `/tmp/c16_h80_oc64_capture_v10.log` | c16_h80_oc64 (3x3) | GEM 2 @ 0x5000 | 1024 | 6 | YES — works as PREFIX_BY_K |
| `/tmp/c16_h80_oc128_full3.log` | c16_h80_oc128 (3x3) | GEM 2 @ 0xA000 | 1792 | **4** | YES — needs custom 4-task writer (Attack F) |
| `/tmp/c64_h1_oc128_v3.log` | c64_h1_oc128 (1x1) | GEM 2 @ 0x4000-0x7000 | 384 | **6** | YES — works as PREFIX_BY_K (PROMOTED this turn) |
| `/tmp/c128_h1_regcmd.txt` | c128_h1_oc24 (1x1, c=24) | GEM 2 | 112 | 1 | YES — single task, body decoded |
| `/tmp/dw_c256_h10_oc256_v2.log` | b1_c256_h10_oc256 depthwise 3x3 | GEM 2 @ 0x2000 | 768 | 1 | YES — body fields decoded (Attack B) |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_keep1_gem1/dump_gem1.txt` | c576_h19_oc12 | GEM 1 | varies | **12** | partial — needs more decoding (Attack H) |

### 4.3 c64_h1_oc128 task structure (6 tasks, 1x1 family — PROMOTED this turn)

The capture has 6 weight_size0 entries with weight_size2 saying oc=128, 64, 64, 48, 48, 32 respectively. Tasks:
- **Task 0 (setup, full OC)**: oc=128, weight_size0=0x4000, cbuf0=0xb1, conv2_low=0x20, data_size1=0x003f0040 (c=64), dma_con2=0x0ffffffd.
- **Task 1 (k_half, oc=0-64)**: oc_count=64, weight_size0=0x2000, weight_size2=0x01010040.
- **Task 2 (k_half, oc=64-128)**: oc_count=64, weight_size0=0x2000, weight_size2=0x01010040.
- **Tasks 3-5 (k_tile, oc=0-48, 48-96, 96-128)**: oc_count=48/48/32, weight_size0 from planner default = oc_count * in_c * 2.

**Key 1x1 family body overrides:**
- `cbuf0 = 0xb1` (vs 0xa2 for 3x3 EXACT11 or 0x57 for c16_h80_oc64)
- `conv2_low = 0x20` (vs 0xf0/0xa0 for 3x3 or 0x1a0 for c16_h80_oc64)
- `weight_size0_setup = 0x4000` (oc * in_c * 2 = 128 * 64 * 2 = 16384)
- `weight_size0_k_half = 0x2000` (64 * 64 * 2 = 8192)
- `dma_con2 = 0x0ffffffd` (huge aux buffer, like c=24 family)
- `data_size1 = 0x003f0040` (c=64, c2=8, in_c-1=63, c2-1=7)
- `cvt_con0 = 0xb` (live shows 0xb even though kh=kw=1 and is_spatial=False in make_regs)

**Critical 1x1 DST_BASE formula:** For c2-packed 1x1 with h*w=1, `output_off = oc_start * 2` bytes. The default `oc_start * out_width_stride * 2` is WRONG for h*w=1 (off by `out_atoms` factor). Fix lives in `_exact11_body_regs` via `DST_OFFSETS_OVERRIDES` lookup (L694-695).

### 4.4 c16_h80_oc128 task structure (4 tasks, NOT 11) — Attack F

The 3x3 c16_h80_oc128 shape uses a 4-task structure: 1 preamble (half-oc weight load) + 3 y-tile computes (full oc=128). Each compute has `weight_size0=0x9000` (full oc=128). The 5x5 variant likely has the same 4-task structure but with `weight_size2=0x05050080`.

**Cannot fit the 11-task EXACT11 framework.** Need a separate `_c16_h80_oc128_task_regs` with new amount tuple (108, 104, 104, 104) and offsets (0, 112, 220, 328).

### 4.5 Depthwise body fields (from b1_c256_h10_oc256 capture)

| Field | Value | Comment |
|---|---:|---|
| conv_con1 | 0x123 | depthwise-specific (vs 0x120 pointwise) |
| conv_con2 | 0x10000090 | family_bits=0x10000000 (depthwise), conv2_low=0x90 |
| conv_con3 | 0x9 | standard |
| data_size0 | 0x000a0006 | in_h=0x6=6 (per-tile), w_stride=0xa=10 |
| data_size1 | 0x003f0100 | c2-packed form (c=256, c2=8) |
| data_size2 | 0x8 | out_w=8 |
| data_size3 | 0x20 | out_w*out_h=8*4=32 (per-tile) |
| weight_size0 | 0x1200 | 4608 = oc*kh*kw*c2*FP16 = 256*9*2 |
| weight_size1 | 0x1200 | per-kernel |
| weight_size2 | 0x03030001 | kw=3, kh=3, oc=1 (depthwise) |
| cbuf0 | 0xa2 | same as c160 default |
| cbuf_con1 | 0x50 | |
| cvt_con0 | 0xb | |
| dma_con2 | 0x3c | |
| fc_data_size1 | 0x100 | |
| dcomp_addr0 | 0xffad9000 | weight DMA |

The capture shows a **SINGLE task body**, but `run_depthwise_shape` (current scaffold) emits 8 BY_K rows of oc=32 each. This mismatch causes the NPU to hang (6s timeout). **Fix:** rewrite `run_depthwise_shape` to submit a single setup task for multi-row BY_K depthwise (matches the live capture).

### 4.6 Generic 3 make_regs fixes (ALREADY APPLIED, do NOT revert)

The 3 generic `make_regs` changes (in L432-509) brought c32_h14_oc64, c32_h7_oc128, c8_h160_oc16, and many "small" shapes to PASS without per-shape overrides:
- `cvt_con5 = ((1 << in_c) - 1) if p["use_nhwc"] else 0`
- `CORE_MISC_CFG = 0x200` (was: `(2 << 8) | is_spatial`)
- `SURFACE_ADD = (p["out_width_stride"] * 2) << 4`

The cross-check confirms that for non-NHWC, non-wide shapes, the rknn_runtime output matches `make_regs` with these generic fixes. **But** they may be the root cause of the 4 regressions (see §6 Step 1) — needs bisection.

---

## 5. The "Crash" — Diagnosis & Status

**User's most recent report:** "it crashed" — timestamp ~13:25.

**Most likely interpretation:** the user ran `python3 examples/conv.py b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid` (or another previously-fenced shape) and saw the 6s `TimeoutError`. Or they ran the sweep and saw 4 FAILs + 1 ERROR (compared to the previous 0/0 baseline). This is the **regression from this turn's promotion**, not an NPU hardware crash.

**NPU hardware status (re-verified 13:25):** `python3 examples/simple_add.py` PASS. NPU is healthy and usable.

**Recovery check (canonical):** `python3 examples/simple_add.py` PASS. If this fails, the NPU card is genuinely stuck and may need a board reboot. From the AGENTS.md safety rule: "Do NOT kill long-running NPU processes, it will crash and reboot." Always let processes time out, never `kill -9` an NPU process.

**CMA pressure soft-resets** can also cause transient errors. The NPU kernel naturally recovers after the test process exits; no manual intervention needed.

**The 4 FAILs in the 13:20 sweep are reproducible on every sweep run** — they are not flaky. They are caused by code changes in `examples/conv.py` that need to be fixed.

---

## 6. Next Steps (super detail, in priority order)

The 101 FENCED shapes split into 8 attack categories (A-H). The plan below attacks them in priority order, but **Step 1 (fix regressions) is BLOCKING** — no new promotions until the 4 regressions are fixed.

### Step 1 (URGENT): Fix the 4 regressions from the c64_h1_oc128 promotion turn

**Status: BLOCKING all future work**

The 4 newly-failing shapes all run via `local_tile_replay` with `hw_out=fp16`:
- `conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1` (max_diff=116.88)
- `b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid` (max_diff=43.12)
- `b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid` (max_diff=45.90)
- `b1_c32_h112_w112_oc16_wic32_k1x1_g1` (max_diff=42.98)

All share: 1x1, c=32, small oc, large h*w (≥112). `out_width_stride=25600` for h=w=160, `hw_out_fp16=True` (because stride > RK_MAX_CONV_FLAT_STRIDE=992).

**Bisection plan:**
1. `git stash` all current `examples/conv.py` changes.
2. Re-run sweep: `timeout 200 python3 sweep_217.py --skip-health --output-dir /tmp/opencode`
3. Verify PASS=114 baseline restored.
4. `git stash pop` to restore.
5. Re-apply changes ONE BY ONE in this order, re-sweeping after each:
   - (a) Generic 3 `make_regs` fixes (most likely culprit for fp16 hw_out):
     - `cvt_con5 = ((1 << in_c) - 1) if p["use_nhwc"] else 0`
     - `CORE_MISC_CFG = 0x200`
     - `SURFACE_ADD = (p["out_width_stride"] * 2) << 4`
   - (b) `post_submit_reset` + hw_oc padding in `run_grouped_serial_shape` (L1287-1318)
   - (c) `(name, family)`-keyed overrides in `_exact11_body_regs` (L630-640)
   - (d) `(name, family, oc_start)`-keyed `DST_OFFSETS_OVERRIDES` lookup (L694-695)
   - (e) `full_data_bank=False` kwarg on `make_regs` (L435)
   - (f) New `_tile_replay_specs` row for `b1_c8_h160_w160_oc16` (L1546-1547)
   - (g) New override dicts: CBUF0, DATA_SIZE1, CONV2_LOW, CVT_CON0, DMA_CON2, WEIGHT_SIZE0, KT_TILE_SPLITS, DST_OFFSETS_OVERRIDES (these only affect c64_h1_oc128, so should be safe)
   - (h) Adding c64_h1_oc128 to `PREFIX_BY_K_SHAPES` (should be safe — only c64_h1_oc128 uses this path)
   - (i) `DEPTHWISE_BODY_SHAPES` membership for c256_h10_oc256 (this caused the c256_h10 ERROR, not the 4 FAILs)
6. After bisection: identify the offending change, fix it (e.g., add a guard like `if in_c == 32 and s["out_c"] <= 32 and s["in_h"] >= 100: ...` for SURFACE_ADD or similar), re-sweep to confirm 114+1=115 PASS.

**Alternative quick diagnosis:**
- Run `python3 examples/conv.py b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid` (single shape) with extra print statements in `_tile_replay_specs` and `run_local_tile_replay_shape` to see which y-tile produces the wrong output.
- The shape has out_h = out_w = 160, local tiles are 3 y-tiles. Print the per-tile output sum, the per-tile `out_width_stride`, and the per-tile `SURFACE_ADD` value.
- Compare to the value computed in the 12:04 working version (need to git checkout the old version temporarily).

### Step 1.5 (CURRENT): Make EXACT11 BY_K structure flexible for variable k_tile counts

**Status: PARTIAL — dynamic layout scaffold added and hardware-validated on existing promoted shapes**

The EXACT11 BY_K structure is hardcoded for exactly 3 k_tiles (11 tasks total):
- EXACT11_BYK_AMOUNTS = (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26) — 11 tasks
- EXACT11_BYK_TAIL_CLASSES/VALUES hardcoded for 11 tasks
- EXACT11_BYK_MASKS hardcoded for 11 tasks

For c832_oc48 (1 k_tile): would need 7 tasks. For c192_oc96 (3 k_tiles but different sizes): might work with current structure but body fields differ.

**Tried this turn:** add c832_oc48 and c192_oc96 to PREFIX_BY_K_SHAPES with 3 k_tiles (16 each, 32 each) overlapping with k_halves. The structure worked (no RuntimeError) but output was wrong (max_diff=221, 113). The body fields (cbuf0, conv2_low, dma_con2, etc.) are shape-specific and need per-shape capture.

**Progress 2026-06-02 16:49:**
- Added `exact_byk_layout(s)` to compute roles, amounts, masks, non-final PC amounts, and regcmd offsets from `KT_TILE_SPLITS`.
- Added `exact_byk_tail_classes(...)` / `exact_byk_tail_value(...)` and wired `write_exact11_byk_tasks(...)` to use the generated layout.
- Kept legacy guardrails: any 3-k-tile shape asserts exact equality with the old hardcoded `EXACT11_BYK_AMOUNTS`, `EXACT11_BYK_MASKS`, `EXACT11_BYK_PC_AMOUNTS`, roles, and offsets `(0,112,224,256,368,400,512,544,656,688,800)`.
- Widened `--dry-run-exact11-byk` to cover all `PREFIX_BY_K_SHAPES`, including `b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid`.

**Validated without DRM submit:**
```
python3 -m py_compile examples/conv.py
python3 examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid --dry-run-exact11-byk
python3 examples/conv.py b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid --dry-run-exact11-byk
```
Both dry-runs report the legacy 11-task layout unchanged: amounts `108,104,26,104,26,104,26,104,26,104,26`, masks `0xd,0xd,0x60,...`, offsets `0,112,224,256,368,400,512,544,656,688,800`, and PC amounts `0,0x1000e,0,0x1000e,0,0x2000e,0,0x2000e,0,0x2000e`.

Additional 2026-06-02 16:54 in-memory writer check passed: `write_exact11_byk_tasks(...)` generated tail qwords and task `enable_mask`/`regcfg_amount` fields matching the old `EXACT11_BYK_TAIL_CLASSES`, `EXACT11_BYK_TAIL_VALUES`, `EXACT11_BYK_MASKS`, and `EXACT11_BYK_AMOUNTS` for the legacy 11-task layout. No DRM allocation or submit was used.

**Hardware validation 2026-06-02 17:49:**
- `python3 examples/simple_add.py` PASS before exact BY_K validation.
- `timeout 30 python3 examples/conv.py b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid` PASS, max_diff=0.0078.
- `timeout 30 python3 examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` PASS, max_diff=0.0624.
- `python3 examples/simple_add.py` PASS after exact BY_K validation.
- Full `timeout 240 python3 sweep_217.py --skip-health --output-dir /tmp/opencode` completed: PASS=115, FENCED=102, FAIL=0, ERROR=0, TIMEOUT=0.

**Plan:**
1. Capture live regcmd for each new 1x1 shape to derive body fields (proper prefix-replay), or add variable `KT_TILE_SPLITS` one shape at a time with dry-run + hardware submit validation.
2. For any promoted shape, run single-shape validation plus pre/post `simple_add.py`.
3. Run a full sweep after each family promotion.

**Quick wins with option 1:** make EXACT11 dynamic first (1 hour of work), then add shapes with computed body fields (fast).

### Step 2 (after Step 1.5): Promote more 1x1 shapes (30+ remaining in Attack A)

**Status: BLOCKED on Step 1.5**

Once regressions are fixed, the 1x1 family promotion workflow is now mechanical. Best candidates (smallest oc first for fast iteration):

| Shape | oc | in_c | h | w | Notes |
|---|---:|---:|---:|---:|---|
| `b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid` | 128 | 64 | 1 | 1 | **DONE** (PROMOTED this turn, max_diff=0.0078) |
| `b1_c32_h112_w112_oc16_wic32_k1x1_g1` | 16 | 32 | 112 | 112 | candidate (h*w>1, may use different dst_offset) |
| `b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid` | 16 | 32 | 150 | 150 | candidate |
| `b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid` | 8 | 32 | 160 | 160 | candidate |
| `b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid` | 96 | 192 | 28 | 28 | large oc |
| `b1_c528_h14_w14_oc32_wic528_k1x1_g1` | 32 | 528 | 14 | 14 | large in_c |
| `b1_c528_h14_w14_oc128_wic528_k1x1_g1` | 128 | 528 | 14 | 14 | large in_c |
| `b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid` | 64 | 384 | 19 | 19 | |
| `b1_c576_h14_w14_oc96_wic576_k1x1_g1` | 96 | 576 | 14 | 14 | large in_c |
| `b1_c832_h7_w7_oc48_wic832_k1x1_g1` | 48 | 832 | 7 | 7 | |
| `b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid` | 384 | 192 | 7 | 7 | **3x3** (needs separate body overrides, not 1x1) |

For each (1x1) shape: capture, decode, derive body fields (likely same family: cbuf0=0xb1, conv2_low=0x20, dma2=0xffffffd, cvt0=0xb), promote. Most should share the c64_h1_oc128 family.

**Note:** the 3x3 shapes in Attack A (e.g., b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid, b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid, b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid, etc.) use a **different** body — cbuf0=0xa2, conv2_low=0xf0, weight_size0=0x9000, etc. They are 3x3 EXACT11 family and need the c160-style overrides (already in CBUF0_OVERRIDES, DATA_SIZE1_OVERRIDES).

### Step 3: Wire depthwise body for Attack B (13 shapes) + Attack C (19) + Attack G (3)

**Status: SCAFFOLDED, body not yet wired**

The `DEPTHWISE_OVERRIDES` dict is defined and `DEPTHWISE_BODY_SHAPES = {c256_h10_oc256}` is populated, but the depthwise body field overrides are not actually applied to the generated register set. The shape went from FENCED to a 6s timeout, confirming the path is reachable but the body is wrong.

**Fix plan:**
1. **Rewrite `run_depthwise_shape(s)`** (L1320-1407) to:
   - For multi-row BY_K depthwise: emit a SINGLE setup task (matching live capture) instead of N tasks.
   - Apply `DEPTHWISE_OVERRIDES` fields when generating the body.
2. Add the depthwise body field overrides to `_exact11_body_regs` (or a new `_depthwise_body_regs`) via a new dispatch in `make_regs`:
   ```python
   if s["name"] in DEPTHWISE_BODY_SHAPES:
       body = make_depthwise_body_regs(s, family, oc_start, oc_count)
   elif s["name"] in PREFIX_BY_K_SHAPES:
       body = _exact11_body_regs(s, family, oc_start, oc_count)
   else:
       body = _default_body_regs(s, family, oc_start, oc_count)
   ```
3. Add to PREFIX_BY_K_SHAPES (or new `DEPTHWISE_K_SHAPES`) so the submit path picks it up.
4. Test c256_h10_oc256 (should go from 6s timeout → PASS).
5. Then iterate for the other 12 Attack B shapes (BY_K), 19 Attack C shapes (BY_YK), and 3 Attack G shapes (BY_Y).

**Risk:** depthwise body is structurally different from pointwise (data_size0 uses per-tile in_h, not full in_h; weight_size0 = oc*kh*kw*c2*FP16, not oc*in_c*kh*kw*FP16). The `make_depthwise_body_regs` function will be a parallel implementation of `_exact11_body_regs`, not a copy.

### Step 4: Wire depthwise body for Attack C/G (19+3 shapes)

**Status: NOT STARTED, depends on Step 3**

Same as Step 3 but for BY_YK and BY_Y. After the single-task body works, add the y-tile body for multi-row Y splits. Inherits the `DEPTHWISE_OVERRIDES` cbuf0=0xa2, conv_con1=0x123, etc.

### Step 5: Attack F — c16_h80_oc128 (3x3 and 5x5, 2 shapes) — custom 4-task writer

**Status: NOT STARTED — structure decoded**

The c16_h80_oc128 (3x3) and (5x5) shapes need a 4-task structure (1 preamble + 3 y-tile computes), NOT the 11-task EXACT11 pattern. Live body overrides:
- `CBUF0_OVERRIDES[shape] = 0x57`
- `WEIGHT_SIZE0_OVERRIDES[shape] = 0x9000` (full oc, not 0x4800)
- `WEIGHT_SIZE2_OVERRIDES[shape] = 0x03030080` (3x3) or `0x05050080` (5x5)
- `CONV2_LOW_OVERRIDES[shape] = 0x1a0` (same as oc64)

**Plan:**
1. Create `EXACT4_BYK_AMOUNTS = (108, 104, 104, 104)` and `EXACT4_BYK_OFFSETS = (0, 112, 220, 328)`.
2. Build `_c16_h80_oc128_task_regs(s, fam)` that emits 4 tasks (1 preamble + 3 computes).
3. Add to a new `C16_H80_OC128_SHAPES` set or extend PREFIX_BY_K_SHAPES.
4. Test c16_h80_oc128 (3x3) first, then (5x5).

### Step 6: Attack E — BY_YK k_half derivation (7 shapes)

**Status: NOT STARTED**

The 7 BY_YK mixed shapes have `mixed Y/K setup and k_half semantics are unresolved`. The body has a k_half task (108 qwords, same as setup) that needs specific overrides.

**Plan:**
1. Start with `b1_c256_h28_w28_oc256_wic256_k1x1_g1` (oc=256, 1x1, the simplest)
2. Build matched .rknn, capture regcmd, decode k_half body
3. Derive overrides for the k_half body
4. Un-fence the BY_YK path, add overrides, test
5. Promote

### Step 7: Attack D — pointwise-wide NONE c=24 (6 shapes, DEFERRED)

**Status: DEFERRED (topology decoded; 108-row body still missing)**

The 6 c=24 pointwise-wide NONE shapes (`b1_c480_h14_w14_oc16_wic480_k1x1_g1`, `b1_c512_h14_w14_oc24_wic512_k1x1_g1`, etc.) need a 3-subcore setup body that the rknn_runtime computes for c=24.

**Progress 2026-06-02 17:47:**
`python3 conv_expt/rknn_prefix_replay.py c128_h1_oc24_pw_none` now labels the existing RKNN dump topology as `same_regcmd_3subcore_setup108`: three task descriptors, each `regcfg_amount=108`, each `enable_mask=0x0d`, all pointing to the same `regcmd_addr=0xffc3a000`. This is a concrete prefix-replay topology for Attack D. The missing piece is still the full 108-row regcmd body; the current dump has task descriptors plus output/aux GEM data, not the body qwords.

**Plan:**
- Look at `rknn_runtime`'s actual weight allocation logic in `~/npu/ops_rknn/` (the `rknn_runtime_infer.py` script). Find the exact size of the aux buffer for c=24 and replicate it in `conv.py`.
- Need 3KB of aux weight (per `weight_bytes_per_kernel * out_c + 3072 if in_c == 24 else weight_bytes_per_kernel * out_c` formula).
- Try `(CNA_WEIGHT_SIZE0 = 0x2000)` for c=24 first; if that doesn't work, capture and decode the actual aux weight bytes.

### Step 8: Attack H — c576_h19_oc12 12-task structure (1 shape)

**Status: NOT STARTED — high risk**

The 12-task structure for `b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid`:
- Task 0: setup (108 qwords)
- Task 1: k_half (108 qwords) — extra setup-like task
- Tasks 2-11: 5×(k_tile=104, aux=26) pairs

**Plan:**
- Capture live regcmd for c576_h19_oc12 (use `/home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_keep1_gem1/dump_gem1.txt` as starting point).
- Extract the live cbuf0 and PC_REGISTER_AMOUNTS.
- Use them as overrides in `_exact11_task_regs` or a new `run_c576_h19_oc12_12task_shape(s)`.

### Step 9: Final sweep + commit

**Status: NOT STARTED**

Once all 101 shapes are promoted:
1. **Final sweep:** `timeout 200 python3 sweep_217.py --skip-health --output-dir /tmp/opencode`
2. **Verify NPU health:** `python3 examples/simple_add.py` (pre and post)
3. **Verify FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0**
4. **Commit staged files** (per AGENTS.md, do NOT commit scratch files):
   ```bash
   git add examples/conv.py conv_expt/rknn_prefix_replay.py \
           experimental/rknn/dump_rknpu_task_gems.py \
           experimental/rknn/capture_rknpu_ioctl_readonly.gdb \
           experimental/rknn/capture_rknpu_submit_dump_gems.gdb \
           examples/conv_tiles.py \
           current_task.md README.md
   git commit -m "Promote all 217 conv shapes via prefix-replay"
   ```

---

## 7. Fence inventory: 101 shapes by category

### 7.1 51 BY_K/k_tile (pointwise 1x1) — Attack A
Mostly 1x1 conv. Key subset (full list in `examples/conv.py`):
```
conv2d_cc_b1_c512_h14_w14_oc512_wic512_k1x1_g1
conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1
conv2d_cc_b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1
conv2d_cc_b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1
b1_c384_h14_w14_oc96_wic384_k1x1_g1
b1_c480_h14_w14_oc96_wic480_k1x1_g1
b1_c512_h14_w14_oc112_wic512_k1x1_g1
b1_c512_h14_w14_oc512_wic512_k1x1_g1
b1_c512_h7_w7_oc1024_wic512_k1x1_g1
b1_c528_h14_w14_oc256_wic528_k1x1_g1
b1_c528_h14_w14_oc160_wic528_k1x1_g1
b1_c528_h14_w14_oc32_wic528_k1x1_g1
b1_c528_h14_w14_oc128_wic528_k1x1_g1
b1_c576_h14_w14_oc96_wic576_k1x1_g1
b1_c832_h7_w7_oc48_wic832_k1x1_g1
b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1
b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1
b1_c1280_h10_w10_oc24_wic1280_k1x1_g1
b1_c1280_h10_w10_oc546_wic1280_k1x1_g1
b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid
b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid
b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid
b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid
b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid
b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid
b1_c832_h7_w7_oc48_wic832_k1x1_g1_s1_pvalid
b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid
b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid
b1_c1280_h10_w10_oc24_wic1280_k1x1_g1_s1_pvalid
b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid
b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid
b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid
b1_c256_h3_w3_oc546_wic256_k1x1_g1_s1_pvalid
b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid
b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid
b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid
b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid
b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid
b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid
b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid
b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid
b1_c72_h20_w20_oc576_wic72_k1x1_g1_s1_pvalid
b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid
b1_c288_h20_w20_oc72_wic288_k1x1_g1_s1_pvalid
b1_c768_h10_w10_oc120_wic768_k1x1_g1_s1_pvalid
b1_c960_h10_w10_oc120_wic960_k1x1_g1_s1_pvalid
b1_c480_h10_w10_oc120_wic480_k1x1_g1_s1_pvalid
b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid
b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid
b1_c96_h20_w20_oc273_wic96_k1x1_g1_s1_pvalid
b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid
```

### 7.2 19 depthwise BY_YK — Attack C
```
conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64
conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128
conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256
b1_c64_h112_w112_oc64_wic1_k3x3_g64
b1_c128_h56_w56_oc128_wic1_k3x3_g128
b1_c256_h28_w28_oc256_wic1_k3x3_g256
b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid
b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid
b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid
b1_c144_h75_w75_oc144_wic1_k3x3_g144_s1_pvalid
b1_c192_h38_w38_oc192_wic1_k3x3_g192_s1_pvalid
b1_c384_h19_w19_oc384_wic1_k3x3_g384_s1_pvalid
b1_c576_h19_w19_oc576_wic1_k3x3_g576_s1_pvalid
b1_c320_h40_w40_oc320_wic1_k3x3_g320_s1_pvalid
b1_c576_h20_w20_oc576_wic1_k3x3_g576_s1_pvalid
b1_c576_h20_w20_oc576_wic1_k5x5_g576_s1_pvalid
b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid
b1_c768_h20_w20_oc768_wic1_k3x3_g768_s1_pvalid
b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid
```

### 7.3 13 depthwise BY_K — Attack B
```
conv2d_cc_b1_c512_h14_w14_oc512_wic1_k3x3_g512
conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024
conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024
b1_c512_h14_w14_oc512_wic1_k3x3_g512
b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024
b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024
b1_c960_h10_w10_oc960_wic1_k3x3_g960_s1_pvalid
b1_c480_h10_w10_oc480_wic1_k5x5_g480_s1_pvalid
b1_c960_h10_w10_oc960_wic1_k5x5_g960_s1_pvalid
b1_c128_h3_w3_oc128_wic1_k3x3_g128_s1_pvalid
b1_c384_h10_w10_oc384_wic1_k3x3_g384_s1_pvalid
b1_c512_h5_w5_oc512_wic1_k3x3_g512_s1_pvalid
b1_c256_h3_w3_oc256_wic1_k3x3_g256_s1_pvalid
```

### 7.4 7 BY_YK mixed (k_half unresolved) — Attack E
```
conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1
b1_c256_h28_w28_oc256_wic256_k1x1_g1
b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid
b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid
b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid
b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid
b1_c768_h20_w20_oc96_wic768_k1x1_g1_s1_pvalid
```

### 7.5 6 pointwise-wide NONE (c=24 family, 3-subcore setup) — Attack D (DEFERRED)
```
b1_c480_h14_w14_oc16_wic480_k1x1_g1
b1_c512_h14_w14_oc24_wic512_k1x1_g1
b1_c512_h14_w14_oc32_wic512_k1x1_g1
b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid
b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid
b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid
```

### 7.6 3 depthwise BY_Y — Attack G
```
conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32
b1_c32_h112_w112_oc32_wic1_k3x3_g32
b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid
```

### 7.7 2 spatial setup/NONE (c16_h80_oc128 3x3 + 5x5) — Attack F
```
b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid
b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid
```

### 7.8 1 pointwise-wide BY_Y (c576_h19_oc12) — Attack H
```
b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid
```

---

## 8. Critical files (reference)

| File | Purpose |
|---|---|
| `examples/conv.py` (1831 lines) | main file, override dicts L155-241 |
| `conv_expt/rknn_prefix_replay.py` (~700 lines, 67 entries) | manifest of promoted shapes |
| `current_task.md` (this file) | comprehensive tracker (READ THIS) |
| `/tmp/c16_h80_oc64_capture_v10.log` | c16_h80_oc64 live regcmd at GEM 2:0x5000:1024 |
| `/tmp/c16_h80_oc128_full3.log` | c16_h80_oc128 live regcmd at GEM 2:0xA000:1792 (**4 tasks** decoded) |
| `/tmp/c64_h1_oc128_v3.log` | c64_h1_oc128 1x1 live regcmd at GEM 2:0x4000-0x7000:384 (**6 tasks** decoded) |
| `/tmp/c128_h1_regcmd.txt` | c128_h1_oc24 live regcmd (1-task structure for c=24 1x1) |
| `/tmp/dw_c256_h10_oc256_v2.log` | depthwise c256_h10 live regcmd (body fields decoded) |
| `/tmp/cross_check2.py` + `/tmp/build_matched.py` | 12-shape NPU correctness proof (max err 0.0070-0.0447) |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_keep1_gem1/dump_gem1.txt` | c576 12-task structure |
| `~/npu/ops_rknn/models/match_*.rknn` | 17+ matched-weight .rknn models |
| `/home/orangepi/npu/ops_rknn/conv2d_multi_dbg` | un-stripped binary for gdb |
| `experimental/rknn/dump_rknpu_task_gems.py` | supports `--qword-window GEM:OFFSET:COUNT` |
| `sweep_217.py` | sweep driver with `--skip-health --output-dir` |

---

## 9. Critical Constants

```python
# Target IDs
CNA = 0x0201
CORE = 0x0801
DPU = 0x1001
PC = 0x0081
PC_REG = 0x0101
VERSION = 0x0041

# Hardware
RK_CBUF_BANKS = 12
CBUF_BANK_SIZE = 32768
FP16_ATOM_ELEMENTS = 16
UNPACK_C2 = 8
FP16_BYTES = 2
FP32_BYTES = 4
RK_MAX_CONV_FLAT_STRIDE = 992
PC_CHAIN_TAIL_QWORDS = 4

# EXACT11 BY_K family
EXACT11_BYK_SHAPE = "b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid"
EXACT11_BYK_AMOUNTS = (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)  # 11 tasks
EXACT11_BYK_OFFSETS = (0, 112, 224, 256, 368, 400, 512, 544, 656, 688, 800)

# KT_FAMILY_BITS defaults
family_bits_default = {
    "setup": 0,
    "y_tile": 0x20000000,
    "k_half": 0x40000000,
    "k_tile": 0x50000000,
}

# Spatial BY_Y shapes (existed before, do not change)
H40_SPATIAL_BY_Y_SHAPE = "b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid"
H160_SPATIAL_BY_Y_SHAPE = "b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid"

# 1x1 family body field defaults (c64_h1_oc128 baseline)
ONEX1_CBUF0 = 0x0b1
ONEX1_CONV2_LOW = 0x020
ONEX1_DMA_CON2 = 0x0ffffffd
ONEX1_CVT_CON0 = 0x000b
ONEX1_DATA_SIZE1_C64 = 0x003f0040   # c=64
# Weight sizes for 1x1: setup = oc * in_c * 2, k_half = (oc/2) * in_c * 2
ONEX1_WS0_FACTOR = lambda oc, in_c: oc * in_c * 2
```

---

## 10. Test Runner Commands

```bash
# Single shape
timeout 30 python3 examples/conv.py <shape_name>

# Full sweep
timeout 200 python3 sweep_217.py --skip-health --output-dir /tmp/opencode

# NPU health check
python3 examples/simple_add.py

# Dry run (no DRM allocation)
python3 examples/conv.py <shape> --dry-run-exact11-byk

# NPU reset (only if simple_add fails)
# (reboot the board — never kill -9 an NPU process)
```

---

## 11. Pattern for New Shape Promotion (one-line summary)

```
build matched .rknn → symlink to models/ → capture regcmd via gdb → decode body →
add per-shape overrides → test single shape → add to promoted set + manifest → re-sweep
```

For 1x1 family specifically: `cbuf0=0xb1, conv2_low=0x20, dma_con2=0x0ffffffd, cvt_con0=0xb, data_size1=0x003f0040 (c=64), KT_TILE_SPLITS=((0,64),(48,48),(96,32)), DST_OFFSETS_OVERRIDES per (name, family, oc_start)`.

---

## 12. Pass-progress quick reference

| Date / time | PASS | FENCED | FAIL | ERROR | TIMEOUT | Notes |
|---|---:|---:|---:|---:|---:|---|
| 2026-06-02 09:54 | 114 | 103 | 0 | 0 | 0 | baseline |
| 2026-06-02 10:25 | 114 | 103 | 0 | 0 | 0 | KT_TILE_SPLITS scaffold added |
| 2026-06-02 10:37 | 114 | 103 | 0 | 0 | 0 | c16_h80_oc128 attempt reverted |
| 2026-06-02 10:45 | 114 | 103 | 0 | 0 | 0 | c128_h2_oc256 attempt reverted |
| 2026-06-02 11:20 | 114 | 103 | 0 | 0 | 0 | c64_h1_oc128 capture done, attempt reverted |
| 2026-06-02 11:31 | 114 | 103 | 0 | 0 | 0 | depthwise scaffold added |
| 2026-06-02 12:04 | 114 | 103 | 0 | 0 | 0 | baseline (re-verified) |
| 2026-06-02 12:10 | 114 | 103 | 0 | 0 | 0 | post-crash health check PASS |
| 2026-06-02 13:20 | 111 | 101 | 4 | 1 | 0 | c64_h1_oc128 promoted (1x1 family), but 4 regressions + 1 depthwise ERROR introduced |
| 2026-06-02 13:25 | 111 | 101 | 4 | 1 | 0 | post-13:20 NPU health check: simple_add.py PASS |
| 2026-06-02 13:51 | **115** | **101** | **0** | **1** | **0** | **4 regressions fixed** (DMA_CON2 patch bug). 1 depthwise ERROR still present. |
| 2026-06-02 14:10 | 115 | 102 | 0 | 0 | 0 | depthwise ERROR reverted to FENCED (DEPTHWISE_BODY_SHAPES emptied). Clean state. |
| 2026-06-02 14:16 | 115 | 102 | 0 | 0 | 0 | verified clean state, 2 attempted 1x1 promotions reverted (body fields shape-specific), 1 attempted c16_h80_oc128 reverted (needs 4-task writer) |
| 2026-06-02 14:25 | 115 | 102 | 0 | 0 | 0 | progress committed (d464ea1). NPU healthy. Decoded c128_h1_oc24 capture (single task, 1x1 family body fields: cbuf0=0xb1, conv2_low=0x20, dma_con2=0x0ffffffd, data_size1=0x3f0080, weight_size0=0x1800). |
| 2026-06-02 16:49 | 115 | 102 | 0 | 0 | 0 | dynamic EXACT11 BY_K layout scaffold added and dry-run-equivalent for existing promoted shapes. Current `simple_add.py` health check fails before submit at BO mmap, so hardware validation is pending. |
| 2026-06-02 16:54 | **115** | **102** | **0** | **0** | **0** | **CURRENT** — offline writer-tail equivalence passed for legacy 11-task exact BY_K layout. `simple_add.py` still fails before submit at BO mmap; sysfs shows runtime suspended/usage 0. |
| **Target** | **217** | **0** | **0** | **0** | **0** | end state |

**Distance to goal:** 102 fenced → 0 fenced. Largest block is Attack A (51 BY_K 1x1 shapes) where the path forward is now proven via c64_h1_oc128, but further hardware validation is currently blocked by BO mmap health failure.

---

## 13. End state target (unchanged)

PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0, with pre/post `python3 examples/simple_add.py` both PASS.

---

## 14. This turn's diff (delta from prior current_task.md)

- **Re-verified post-13:20 NPU health** (simple_add.py PASS at 13:25).
- **Documented the c64_h1_oc128 promotion** as the 1x1 family breakthrough: 6-task structure, cbuf0=0xb1, conv2_low=0x20, dma_con2=0x0ffffffd, cvt_con0=0xb, data_size1=0x003f0040, KT_TILE_SPLITS=((0,64),(48,48),(96,32)), DST_OFFSETS_OVERRIDES per (name, family, oc_start).
- **Documented the 4 regressions** (c256_h28_oc32 max_diff=116.88, c32_h160_oc8 max_diff=43.12, c32_h150_oc16 max_diff=45.90, c32_h112_oc16 max_diff=42.98) and the 1 new ERROR (c256_h10_oc256 6s timeout).
- **Updated the bisection plan** for the 4 regressions in §6 Step 1.
- **Updated the fence histogram** to 101 total (52→51 BY_K, 14→13 depthwise BY_K, 1 depthwise BY_K → ERROR).
- **Promoted the 1x1 family path** from "needs new task generator" to "proven via c64_h1_oc128, ready to scale to 30+ more shapes after Step 1".
- **Added depthwise single-task rewrite** to §6 Step 3 (the live capture shows single task but planner emits 8 BY_K rows).
- **Documented the new DST_OFFSETS_OVERRIDES** mechanism (L694-695) that fixes 1x1 h*w=1 dst_base formula.
- **Added 1x1 family constants** to §9 (ONEX1_CBUF0, ONEX1_CONV2_LOW, etc.).
- **Updated sweep history** with the 13:20 row and 13:25 health check.
