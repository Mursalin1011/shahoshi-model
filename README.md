# Shahoshi

Multimodal edge-ML threat detection for a wrist-worn personal-safety wearable.
Inference runs entirely on device; a confirmed event sounds a local alarm and
sends a WhatsApp alert with a GPS fix.

Target hardware: **ESP32** DevKit (Xtensa LX6), MPU-6050 IMU, MAX30102 PPG,
3-pin sound detection module, NEO-M8N GPS. This part has no SIMD int8 path --
ESP-NN's optimized kernels are written for the S3 -- so the model is budgeted
for reference-C++ speed rather than for a vectorized one. Every other document
in this project states a different device list; `MULTIMODAL_PLAN.md` section 2
tabulates the disagreement and records the bill of materials above as ground
truth.

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
  fusion.py     2-of-3 consensus engine, its FAR arithmetic, its generated C
  models/       Keras definitions            (needs TensorFlow)
  quantize.py   int8 conversion + inference  (needs TensorFlow)
  export.py     C array, op resolver, runtime config for ESP-IDF
  config.py     YAML experiment config
  manifest.py   per-run provenance
configs/        experiment configs, one per run
notebooks/      thin drivers -- logic lives in src/
tests/          361 tests, none requiring TensorFlow
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
| dilated depthwise convs lower to `SPACE_TO_BATCH_ND`/`BATCH_TO_SPACE_ND`, which compute nothing and only shuffle memory, and the firmware sketch registered 8 ops for a 13-op model — `AllocateTensors()` would have failed | dilation replaced with a stride-2 stage; resolver generated from the converted model's own op list |
| two `.tflite` files held a byte-identical copy of the same trunk (~45 KB wasted flash, two arenas) | one model, two outputs |

**Stage 1 — next.** SisFall. Phase A scores the frozen Stage 0 model against
real labelled falls, giving the project its first honest threat-detection
number. Phase B adds a fall head and retrains multi-task.

**Stage 2.** Acoustic branch. The confirmed BOM carries a 3-pin sound
detection module, not the INMP441 previously assumed here, and whether that
module outputs an analog envelope or a comparator bit decides the whole branch:
either an ~8 kHz ADC feed with band-energy features and a tiny dense
classifier, or a 1-bit amplitude trip carrying a weight well below 1.0. A
log-mel CNN was never the right shape for this part; `esp-dsp` supplies the
LX6-optimized FFT if the analog path is confirmed. `MULTIMODAL_PLAN.md`
section 7 states the fork and the bench test that settles it. Note that ESC-50
and UrbanSound8K contain no scream class, contrary to the original proposal.

**Stage 3 — the vote is built, ahead of two of its voters.** No public dataset
records movement, HR and audio simultaneously during distress, so the 2-of-3
vote cannot be learned; it is specified in `fusion.py`, pinned by tests, and
emitted to the firmware as generated C (`fusion_source`) rather than
transcribed by hand into a sketch. Each branch is calibrated to its own
alarms-per-hour budget by `calibrate_branches`; consensus is what makes the
fused rate small.

The engine latches each fire for a hold window (the branches are asynchronous:
a 1.28 s movement hop cannot coincide with a millisecond-long acoustic event
otherwise), requires consensus to persist, and enforces a cooldown so one
confirmed event sends one alert rather than thirty. Three things it makes
explicit:

| | |
|---|---|
| **The vote has one voter today** | Under the default `strict` degradation policy a 2-of-3 rule with only the movement branch live *cannot fire*. The engine reports `can_alarm = False`, the config prints `fusion UNREACHABLE`, and the generated header carries a `CANNOT RAISE AN ALARM` warning — so a silent field test is not mistaken for a quiet device. |
| **The plan's 4 s hold and 4 s sustain cancel** | A fire latches its branch for the whole hold, so a single coincident pair already satisfies a 4 s sustain: `sustain_margin` reports 0.00 s. Raising the sustain to 8 s (`configs/movement_fusion.yaml`) is what makes the clause demand repeated agreement. |
| **The fused false-alarm rate is arithmetic, not a hope** | At 6 alarms/hour per branch the fused bound is 0.12/h and the measured rate (the real engine over Poisson fires) is ~0/h; even at 60/h per branch — one interruption a minute, individually unusable — the fused rate is 0.29/h. Both assume the branches fail independently, and they do not: a struggle drives movement and corrupts the PPG from one physical cause. That correlation is the largest unmeasurable risk in the design. |

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
