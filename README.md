# Shahoshi

Multimodal edge-ML threat detection for a wrist-worn personal-safety wearable.
Inference runs entirely on device; a confirmed event sounds a local alarm and
sends a WhatsApp alert with a GPS fix.

Target hardware: **ESP32-S3** (vector unit + ESP-NN for int8 acceleration),
MPU-6050 IMU, MAX30100 PPG, INMP441 I2S microphone, NEO-6M GPS.

---

## Layout

```
src/shahoshi/
  datasets/     one module per corpus, all returning the same WindowSet
  signal.py     resampling, gravity separation, ADC conversion, SVM
  windows.py    sliding windows, frozen normalization, event labelling
  splits.py     subject-disjoint / LOSO / leave-one-dataset-out
  augment.py    mount-invariance augmentation
  scoring.py    entropy + Mahalanobis novelty, false-alarm calibration
  models/       Keras definitions            (needs TensorFlow)
  quantize.py   int8 conversion + inference  (needs TensorFlow)
  export.py     C array, op resolver, runtime config for ESP-IDF
  config.py     YAML experiment config
  manifest.py   per-run provenance
configs/        experiment configs, one per run
notebooks/      thin drivers -- logic lives in src/
tests/          258 tests, none requiring TensorFlow
artifacts/      generated .tflite / .h / .cc / config  (gitignored)
reports/        per-run manifests                       (gitignored)
```

TensorFlow is not a hard dependency. Everything except `models/` and
`quantize.py` is pure NumPy/SciPy and testable without it — which is why the
harmonization code that SisFall depends on is under test rather than under hope.

## Running

**Colab** (training). Set `REPO_URL` in the first cell of
`notebooks/01_movement.ipynb`, then run top to bottom. Do not `pip install
tensorflow` — Colab's TF is pinned against a matching NumPy and pip cannot
hot-swap a C extension underneath a live kernel.

**Locally** (tests and the numeric code). TensorFlow has no wheels for Python
3.14, so the local venv covers everything except training:

```bash
py -3 -m venv .venv && ./.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

```bash
./.venv/Scripts/python.exe -m pytest tests/
```

## Status

**Stage 0 — done.** UCI HAR + MotionSense, 6 classes, subject-disjoint, int8
export, generated op resolver, false-alarm calibration in alarms per hour.

Four defects from the pre-refactor notebook (`movement_model_esp32_3.ipynb`,
kept at the repo root for reference) are fixed here:

| Defect | Fix |
|---|---|
| `class_weight=` reweights train loss but not val loss, so `val_loss` was incomparable and `EarlyStopping` restored the **epoch-5** weights | class weighting inside the loss; early stopping on val macro-F1 |
| `lay` scored 0.25 recall float / 0.09 int8 — gravity is removed, so posture is unrecoverable | class dropped, documented |
| dilated depthwise convs lower to `SPACE_TO_BATCH_ND`/`BATCH_TO_SPACE_ND`, which ESP-NN cannot accelerate, and the firmware sketch registered 8 ops for a 13-op model — `AllocateTensors()` would have failed | dilation replaced with a stride-2 stage; resolver generated from the converted model's own op list |
| two `.tflite` files held a byte-identical copy of the same trunk (~45 KB wasted flash, two arenas) | one model, two outputs |

**Stage 1 — next.** SisFall. Phase A scores the frozen Stage 0 model against
real labelled falls, giving the project its first honest threat-detection
number. Phase B adds a fall head and retrains multi-task.

**Stage 2.** Acoustic branch (INMP441, log-mel, AudioSet positives with
SNR-controlled urban-noise mixing). Note that ESC-50 and UrbanSound8K contain
no scream class, contrary to the original proposal.

**Stage 3.** Fusion. No public dataset records movement, HR and audio
simultaneously during distress, so the 2-of-3 vote stays hand-designed with each
branch calibrated to its own alarms-per-hour budget. The heart-rate branch does
not exist yet — the vote currently has one voter.

## Two things to keep in view

**Every corpus is waist- or pocket-mounted; the device is worn on the wrist.**
Rotation augmentation reduces the gap and does not close it: a wrist moves
differently, not merely in a rotated frame. Leave-one-dataset-out is reported
alongside every merged number so the gap is visible rather than assumed away.

**Thresholds are stated in alarms per hour, not percentiles.** A
99th-percentile threshold reads as a strict 1% budget; at the 1.28 s inference
hop it is 28 false alarms per hour — one every two minutes. A wearer switches
that off, and a device that is off has zero recall whatever the confusion matrix
says.
