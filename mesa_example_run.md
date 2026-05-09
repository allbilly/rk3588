# Teflon Delegate Model Testing

## Prerequisites

```bash
# activate virtual environment
source .venv/bin/activate

# install test image (if missing)
mkdir -p ~/tensorflow/assets
wget -O ~/tensorflow/assets/grace_hopper.bmp \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp

# tflite delegate path
DELEGATE=build/src/gallium/targets/teflon/libteflon.so
```

## Models Available

| Model | Script | Teflon | Notes |
|-------|--------|--------|-------|
| mobilenetv1 | `classification.py` | yes | 224x224 uint8, 1001 classes |
| mobilenetv2 | `classification.py` | yes | 224x224 uint8, 1001 classes |
| inception | `classification.py` | yes | 224x224 uint8, 1001 classes |
| ssdmobilenetv2 | `detection.py` | yes | 300x300 uint8, COCO 80 classes |
| mobiledet | `detection.py` | yes | 320x320 uint8, COCO 80 classes |
| efficientdet | `detection.py` | no | 320x320 uint8, COCO — driver assertion bug |
| movenetlightning | `posenet.py` | no | 192x192 uint8, 17 keypoints — driver bug |
| movenetthunder | `posenet.py` | no | 256x256 uint8, 17 keypoints — driver bug |
| yolox | `yolox.py` | no | 416x416 int8, COCO — driver bug + needs letterbox |
| micronetlarge | — | — | 32x32 int8 grayscale, 8 outputs — audio model, not image |

## Test Scripts

All scripts accept `-e <delegate>` to enable the Teflon delegate (omit for CPU-only).

### Classification

Uses the existing `classification.py` and `labels_mobilenet_quant_v1_224.txt`.

```bash
# mobilenetv1
python3.10 classification.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/mobilenetv1/mobilenet_v1_1_224_quant.tflite \
  -l labels_mobilenet_quant_v1_224.txt \
  -e $DELEGATE

# mobilenetv2
python3.10 classification.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/mobilenetv2/mobilenet_v2_tflite_1_0_224_quantized_v1.tflite \
  -l labels_mobilenet_quant_v1_224.txt \
  -e $DELEGATE

# inception
python3.10 classification.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/inception/inception_v1_224_quant.tflite \
  -l labels_mobilenet_quant_v1_224.txt \
  -e $DELEGATE
```

### Detection (SSD models with post-processing ops)

Uses `detection.py` (COCO labels built-in). Supports `--output` for annotated image and `--score_threshold` (default 0.5).

```bash
# ssdmobilenetv2 (works with teflon) — 300x300
python3.10 detection.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/ssdmobilenetv2/ssd_mobilenet_v2_coco_quant_postprocess.tflite \
  -e $DELEGATE \
  --output /tmp/ssd_out.bmp

# mobiledet (works with teflon) — 320x320
python3.10 detection.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/mobiledet/ssdlite_mobiledet_coco_qat_postprocess.tflite \
  -e $DELEGATE \
  --output /tmp/mobiledet_out.bmp

# efficientdet (CPU only — driver assertion bug)
python3.10 detection.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/efficientdet/efficientdet_tflite_lite0_int8_v1.tflite \
  --output /tmp/efficientdet_out.bmp
```

### Pose Estimation

Uses `posenet.py` (17 keypoints, built-in skeleton edges). Supports `--output` and `--score_threshold`.

```bash
# movenetlightning (CPU only — driver assertion bug) — 192x192
python3.10 posenet.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/movenetlightning/movenet_single_pose_lightning_ptq.tflite \
  --output /tmp/movenet_lightning_out.bmp

# movenetthunder (CPU only — driver assertion bug) — 256x256
python3.10 posenet.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/movenetthunder/movenet_single_pose_thunder_ptq.tflite \
  --output /tmp/movenet_thunder_out.bmp
```

### YOLOX

Uses `yolox.py` (COCO labels, built-in NMS). Input is normalized to [0,1] then quantized per model parameters.

```bash
# CPU only — 416x416 int8
python3.10 yolox.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/yolox/yolox_nano_full_integer_quant.tflite \
  --score_threshold 0.3 \
  --output /tmp/yolox_out.bmp
```

## Performance Results

| Model | Teflon | CPU-only | Quality |
|-------|--------|----------|---------|
| mobilenetv1 | ~11ms | — | military uniform 0.867 |
| mobilenetv2 | ~15ms | — | military uniform |
| inception | ~34ms | — | military uniform |
| ssdmobilenetv2 | ~27ms | — | 20 detections (low scores) |
| mobiledet | ~38ms | — | person 0.715, snowboard 0.676 |
| efficientdet | assertion | ~135ms | person 0.965 |
| movenetlightning | assertion | ~42ms | face/shoulders detected |
| movenetthunder | assertion | — | — |
| yolox | assertion | ~175ms | 0 detections @ 0.3 threshold |

## Known Driver Bug

Models failing with the assertion `input_op_1' failed` in `rkt_ml_subgraph_create` (`src/gallium/drivers/rocket/rkt_ml.c:343`):

- **efficientdet** — 320x320 uint8, SSD post-processing ops
- **movenetlightning** / **movenetthunder** — single-subgraph, float32 keypoint output
- **yolox** — 416x416 int8, single output tensor 3549x85

The root cause appears to be the rocket driver not handling certain graph structures — likely models where the first operation in a subgraph is not directly fed by the subgraph input tensor. All these models run correctly on CPU (XNNPACK delegate).
