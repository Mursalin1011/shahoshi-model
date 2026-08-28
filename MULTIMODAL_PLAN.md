# Plan: the heart-rate and audio branches

Working document. Written 2026-08-28, before any code was changed, so that the
reasoning behind the next stages survives without the conversation that produced
it. Nothing in this file has been implemented yet.

The short version: Stage 3 built the 2-of-3 vote ahead of two of its three
voters. This plan builds those two voters. Along the way it corrects the target
hardware, which every document in this project currently states differently.

---

## 1. Where the project stands

Branch `stage3-fusion-engine` @ `8629f27`. 361 tests pass in ~10 s, none
requiring TensorFlow.

**`stage3-fusion-engine` has never been merged into `main`.** `main` sits at
`ba0bb90` and contains no fusion engine at all. All Stage 3 work lives only on
this branch, and the branches below depend on `fusion.py`, so they should build
on top of it rather than on `main`.

| Stage | State |
|---|---|
| **0 — movement** | Done. UCI HAR + MotionSense, 6 classes, subject-disjoint, int8 export, generated op resolver, FAR in alarms/hour, int8 gap diagnosed and attributed |
| **1 — SisFall / fall head** | Not started. Scaffolding exists and is unused: `y_fall` with `-1` masking, `masked_binary_ce`, `with_fall_head`, `heads.assign_roles(with_fall_head=True)` |
| **2 — audio** | Not started |
| **3 — fusion** | Engine built, tested and generating C, ahead of two of its voters. Config prints `fusion UNREACHABLE`; the generated header carries `CANNOT RAISE AN ALARM` |

### What is already modality-agnostic

An audit for this plan found that `splits.py`, `windows.py`, `scoring.py`,
`quantize.py`, `export.py`, `manifest.py` and `fusion.py` all take arrays,
op-name lists and score streams rather than IMU-specific objects.
`fusion_source` is generic in branch count. Only three things are IMU-shaped:

- `datasets/base.WindowSet` — hard-asserts `(n, win, 6)` in `CHANNELS` order
- `models/movement.py`
- `augment.py` — the rotation augmentations

So adding two modalities is mostly new code, not a refactor of working code.

---

## 2. Hardware ground truth

The confirmed bill of materials, which supersedes every other list in this
project:

| | Part |
|---|---|
| MCU | **ESP32** (DevKit, Xtensa LX6 — *not* an S3) |
| IMU | GY-521 / MPU-6050 |
| PPG | **MAX30102** |
| Sound | **3-pin voice sound detection module** |
| GPS | **NEO-M8N** |
| Alarm | Buzzer |
| Power | Type-C charger + 3.7 V **350 mAh** LiPo |
| Build | Dot vero board |

### Device alignment across the project's documents

Checked against `Shahoshi_Implementation_Plan.pdf`, this repo's README, and the
companion firmware repository
[`AsifZaman777/Shahoshi_Wearable`](https://github.com/AsifZaman777/Shahoshi_Wearable).
**Only the MPU-6050 agrees across all sources.**

| Device | BOM (above) | Proposal PDF | Firmware `WIRING.txt` | Firmware **code** | This README |
|---|---|---|---|---|---|
| MCU | ESP32 | ESP32 DevKit V1 | ESP32 DevKit V1 | `board = esp32dev` ✓ | **ESP32-S3** ✗ |
| IMU | MPU-6050 | MPU-6050 | MPU6050 @0x68, GPIO21/22 | Adafruit MPU6050, real ✓ | MPU-6050 ✓ |
| PPG | MAX30102 | MAX30100 | MAX30100 | **potentiometer** ✗ | MAX30100 ✗ |
| Sound | 3-pin module | LM393 analog A0 | LM393 AO → GPIO34 | **potentiometer** ✗ | **INMP441 I2S** ✗ |
| GPS | NEO-M8N | NEO-6M | "ATGM336H / NEO-6M" | absent from `diagram.json` | NEO-6M ✗ |
| Buzzer | unspecified | Passive | "**Active** Siren" | `tone()` ⇒ passive | — |
| Power | Type-C, 350 mAh | TP4056, 500 mAh | TP4056 → 5V/VIN | — | — |

### Consequences

**No ESP-NN.** ESP-NN's int8 acceleration targets the S3's vector unit. On an
LX6 nothing is SIMD-accelerated, so `export.py`'s ESP-NN warnings and its
`CONFIG_NN_OPTIMIZATIONS=y` advice are inapplicable. Stage 0's decision to drop
dilated convolutions remains correct — fewer registered ops, and the
space-to-batch pair is slow reference C++ on any MCU — but the *stated reason*
needs restating. **The trained model itself does not need retraining.**

**esp-dsp replaces ESP-NN as the acceleration story.** It has hand-optimized
FFT and dot-product routines for the LX6, which is what the audio branch needs.

---

## 3. Findings from the companion firmware repository

`Shahoshi_Wearable` contains a working PlatformIO firmware (GPS, buzzer, Brevo
email alerts, LittleFS logging) **and a second training pipeline**,
`SheShield_Multimodal_Fusion_Training_updated.ipynb`, which has already produced
`imu_model.tflite`, `hr_model.tflite`, `audio_model.tflite` and
`fusion_config.h`.

That notebook's prose is careful — it argues for late fusion over a joint model,
and explicitly flags that EmoWear ships no discrete stress labels and that an
LM393 cannot feed an audio model. **The defects are in the gap between what its
markdown says and what its code does.** They are recorded here so this repo does
not reintroduce them.

### F1 — Two of three branches have no data path on real hardware

```c
#define HR_PIN 35   // ADC1_CH7 (MAX30100 Stand-in)
int hrRaw = analogRead(HR_PIN);
float bpm = map(hrRaw, 0, 4095, 40, 180);
```

Heart rate is a linear map of a potentiometer. `platformio.ini` lists no
MAX3010x library; the only `Wire.begin()` in the codebase serves the MPU-6050.
`diagram.json` contains five parts: ESP32, MPU6050, buzzer, `pot_sound`,
`pot_hr`. There is no microphone and no PPG sensor. The firmware is still the
Wokwi simulation with ML bolted onto it.

`WIRING.txt` additionally wires the MAX30100 to both the I2C bus *and* to
`GPIO 35 — Purple (Stand-in Mode)`. A MAX30100 has no analog output; the
simulator's stand-in has leaked into a document that presents itself as a
real-hardware wiring guide.

### F2 — The audio model is fed synthetic data

`ml_engine.cpp`, `feedAudioRaw()`:

```c
float norm = (float)soundRaw / AUDIO_RAW_MAX;
for (int k = 0; k < 13; k++) mfcc[k] = norm * cosf((float)k * 3.14159265f * norm);
```

One scalar ADC reading generates 13 numbers by a closed-form function of that
single value, so every "MFCC frame" carries exactly one degree of freedom. The
model was trained on librosa MFCCs from 16 kHz audio. `soundRaw` is also
sampled at 50 Hz in the main loop, which aliases away everything above 25 Hz —
there is no audio information in the input regardless. Given F1, this was
written to digitize a potentiometer knob, not a microphone.

### F3 — There is no scream in the scream class

The notebook's section header names MIVIA; the code uses ESC-50 with

```python
threat_mapping = {'scream': ['crying_baby'], 'gunshot': ['gunshot'], ...}
```

`scream` is 40 clips of a crying baby. ESC-50 has no `gunshot` category, so that
class receives zero samples — the class-weight workaround in cell 42
("calculate weights only for existing classes to avoid ValueError") is the
symptom. The exported model presents a 4-way softmax with one never-trained
output. Re-running the notebook's own `value_counts()` printout confirms this.

### F4 — The HR labels are constant within each subject

Cell 26: *"For this simplified pipeline, we take the mode label for the
participant. In a full run, you'd sync timestamps between survey entries and
signal."* Every window from a participant carries that participant's median
arousal. Under a subject-wise split there is no within-subject label variation
to learn from, so the model can only fit "this person's physiology → this
person's label", which does not transfer to held-out subjects by construction.
Cell 28 also carries a fallback that prints `RESULTS WILL BE INVALID` and then
trains and exports anyway.

### F5 — All three models receive the wrong time base

| Model | Trained on | Device feeds |
|---|---|---|
| HR | 30 s @ 4 Hz | 120 samples @ 50 Hz = **2.4 s** |
| Audio | 1 s @ 16 kHz | 40 frames @ 12.5 Hz = **3.2 s** |
| IMU | 2 s @ 50 Hz | 2 s @ 50 Hz ✓ |

### F6 — All three models receive the wrong normalization

Training used `(X − mean)/std` from the train split. The firmware uses fixed
max-scaling: `constrain(accel, ±20)/20`, `(bpm−30)/190`, raw cosine values.
**The train-split statistics are never exported to any header.** This is the
same class of defect this repo already documents costing a full training run
(positional output indexing → 0.0196 int8 accuracy).

Related: `parse_sisfall_file` slices `df.iloc[:, :6]` — raw ADXL345/ITG3200 ADC
counts, never converted to m/s² or rad/s — while the device feeds MPU-6050 SI
units.

### F7 — The fusion weights encode a rule nobody chose

`FUSION_W_IMU=0.40, W_HR=0.30, W_AUDIO=0.30`, noisy-OR, alert at 0.55:

- HR + audio, both at **100 % confidence**: `1 − (1−0.3)(1−0.3) = 0.51` → **no alert**
- IMU + either other branch at 1.0: `1 − (1−0.4)(1−0.3) = 0.58` → alert

The soft rule is therefore not 2-of-3 but **"IMU plus one other"**. A scream and
a heart-rate spike together, both certain, cannot raise an alert. Separately,
`FUSION_HARD_THRESHOLD = 0.90` lets any single modality fire alone, which
contradicts the proposal's central claim that a single sensor firing alone does
not raise an alert. The warm-up path in `heuristics.cpp` is a plain
single-sensor OR, so `bpm > 130` alone sends an emergency email.

(`alert_email.cpp` does implement a 30 s email cooldown, so alerts are not
spammed.)

### Why this repo is the right place to fix F6 and F7

`shahoshi-model` already contains the machinery whose absence caused them:

- `export.normalization_source()` emits frozen train-split mean/std as C — prevents F6
- `fusion.fusion_source()` generates the consensus state machine as C, with latching, sustain and cooldown — prevents F7
- `heads.assign_roles()` resolves outputs by width rather than position — prevents the class of bug that already cost one training run
- thresholds in alarms per hour rather than confidence percentiles — how `bpm > 130` would have been caught

The firmware repo hand-wrote all three and got all three wrong. The intended end
state: **`Shahoshi_Wearable` stays the firmware; its training notebook is
retired; `shahoshi-model` emits `imu_model.h`, `hr_model.h`, `audio_model.h`,
`normalization.h` and `fusion.h` as build artifacts.**

### Worth keeping from that repo

- A working SisFall download path: Kaggle `nvnikhil0001/sis-fall-original-dataset`, with parsing code to adapt. Makes Stage 1 cheaper.
- **EmoWear** (Zenodo record 10407279) — ECG/BVP/EDA/ACC/GYRO with SAM valence/arousal ratings. Not a replacement for WESAD or PPG-DaLiA, but a third corpus for leave-one-dataset-out.
- The firmware skeleton itself is real and is not the part that needs replacing.

---

## 4. The framing that keeps this honest

This is not a multimodal model. No corpus records movement, PPG and audio
simultaneously during distress, which is why `fusion.py` is hand-specified. Each
new branch is **an independent scorer calibrated to its own alarms-per-hour
budget**, handed to the existing engine.

The deliverable is `implemented: true` on all three branches in
`configs/movement_fusion.yaml` and the `CANNOT RAISE AN ALARM` warning
disappearing from the generated header.

---

## 5. Datasets

### Heart rate

| Corpus | Role |
|---|---|
| **WESAD** (Uni Siegen, ~18 GB) | 15 subjects, Empatica E4 **on the wrist** — BVP 64 Hz, ACC 32 Hz, EDA, TEMP — with baseline/stress/amusement/meditation labels. The only public corpus with wrist PPG *and* wrist accelerometer *and* a stress label on one clock. Also the first wrist-mounted data this project has seen. |
| **PPG-DaLiA** (UCI repo, ~1.6 GB) | 15 subjects, E4 wrist PPG + ACC across 8 everyday activities, with **ECG ground-truth HR**. The negatives corpus: where alarms-per-hour is measured and where motion artefacts become measurable. |
| **EmoWear** (Zenodo 10407279) | Optional third corpus for leave-one-dataset-out. Note it ships no discrete stress labels — any are derived, and that must be stated as a modelling assumption. |

Caveats for the module docstrings, in the style of `motionsense.py`: WESAD's
TSST stress is public-speaking stress, not assault; and an Empatica E4 is not a
MAX30102 — different wavelength, adaptive gain, a properly tensioned strap.

### Audio

| Corpus | Role |
|---|---|
| **FSD50K** (Zenodo, CC-licensed) | Scream/shout/yell/crying positives. Preferred over AudioSet, which ships only YouTube IDs and VGGish embeddings — many clips are dead, scraping is fragile, and embeddings give no waveform. Supersedes the README's current "AudioSet positives" note. |
| **UrbanSound8K** / **ESC-50** | SNR-controlled noise mixing and **hard negatives** — car horn, siren, jackhammer, dog bark are exactly the loud transients an amplitude trip cannot separate from a scream. |

Total download across all corpora ≈ 50 GB. Colab's session disk can hold it but
does not persist it; a Drive cache plus a class-filtered FSD50K subset is worth
building before the first full run.

**Stage A below opens with an availability gate that counts actual positives
before any modelling** — the same discipline as the harmonization gate in cell 6
of `notebooks/01_movement.ipynb`, and the check that F3 shows is not optional.

---

## 6. Code

### New modules

```
datasets/wesad.py       download() / load() -> HRWindowSet
datasets/dalia.py       same contract, the negatives corpus
datasets/fsd50k.py      download() / load() -> AudioClipSet
datasets/urbansound.py  noise bed and hard negatives
hr.py       bandpass 0.5-4 Hz, peak/spectral HR, ACC artefact gate,
            rolling personal baseline, deviation + rate-of-rise score
audio.py    framing, pre-emphasis, band energies, SNR mixing, and the
            sensor simulation described below
models/acoustic.py   only if the analog path is confirmed
```

`hr.py` and `audio.py` stay pure NumPy/SciPy, holding the no-TensorFlow rule,
and are where most of the new tests live.

### Data contract

Add sibling dataclasses (`HRWindowSet`, `AudioClipSet`) sharing a small
provenance base with `WindowSet`, rather than widening `WindowSet`. Its
`__post_init__` assertion that `X` is `(n, win, 6)` in `CHANNELS` order has been
catching real bugs; forcing a `(n, frames, bands)` array through it means either
weakening that check or lying about the channel vocabulary. Keep the strict
class strict.

### Changed modules — all additive

- `augment.py` — `specaugment` (time/frequency masking) and `mix_at_snr`; rotations stay IMU-only
- `config.py` — `AudioConfig`, `HRConfig`; `configs/audio.yaml`, `configs/hr.yaml`, `configs/fusion_three_branch.yaml`
- `export.py` — retarget the ESP-NN text to ESP32/esp-dsp; verify `RESOLVER_METHODS` covers any new op set
- notebooks `02_audio.ipynb`, `03_hr.ipynb` as thin drivers

---

## 7. The audio branch depends on one unanswered question

**Does the 3-pin sound module output analog or digital?** Three pins means VCC,
GND and one output. `WIRING.txt` says LM393 with analog AO, but it is describing
a potentiometer (F1), so it is not evidence.

Answer it before writing `audio.py`:
- product link or a photo of the board usually settles it
- a blue trimpot on the board almost always means a comparator with digital output — the pot *is* the threshold
- definitive: `analogRead(34)` in a loop, then clap. Continuous values tracking loudness ⇒ analog. Slamming between ~0 and ~4095 ⇒ digital.

### Plan A — analog output

Not a log-mel CNN. On an LX6 with no ESP-NN, ~320 KB usable DRAM and a 350 mAh
cell, a CNN over a 63×40 spectrogram is the wrong shape.

- Sample AO via **ESP32 ADC1 in continuous/DMA mode at ~8 kHz**, 12-bit (≈9–10 effective bits after ADC noise). **This is a firmware change and it is mandatory** — sampling at 50 Hz in the main loop, as the current firmware does, aliases away all audio content.
- 256-sample frames (32 ms), 50 % overlap, 256-point real FFT via **esp-dsp**.
- ~20–30 features per 1 s window: log energy in 8–10 bands, zero-crossing rate, spectral centroid, spectral flux, crest factor, fraction of frames above an adaptive noise floor.
- Tiny dense classifier (32-16-1), int8; a few KB of flash and a trivial arena.

This is close to what the Implementation Plan §2 already specifies ("sound
amplitude statistics"), and unlike a CNN it fits the part.

**Training discipline: simulate the sensor before training.** Downsample corpora
to 8 kHz, band-limit, requantize to ~10 bits, add the module's measured noise
floor, *then* compute features. Otherwise the model is trained on studio
recordings and deployed on a cheap electret through a noisy ADC. This is the
audio equivalent of the wrist/waist gap, except that it is simulatable.

### Plan B — digital output

The branch becomes a scalar derived from the bit: fraction of the last 2 s
asserted, or rising-edge count per window. Still a valid `BranchSpec`, still
calibratable in alarms/hour, but with **weight well below 1.0** —
`BranchSpec.weight`'s docstring already anticipates this case.

No FSD50K, no training, no `models/acoustic.py`. One piece survives and is worth
doing: **simulate the comparator offline.** Run FSD50K and UrbanSound through a
modelled threshold trip across a sweep of levels, and measure recall-at-FAR.
That produces an honest number for a 1-bit branch *and* tells you where to set
the physical trimpot, which otherwise is set by ear and never written down.

Under Plan B the README's Stage 2 claim must change from "acoustic branch" to
"amplitude trip". A pot-tuned comparator firing on a slammed door is
indistinguishable from one firing on a scream.

---

## 8. The HR branch

MAX30102 is the part that makes this work: it has a real FIFO, so raw red/IR at
50–100 sps is available rather than a library-computed BPM. Sample IR at 100 sps;
the HR band is 0.5–4 Hz, so that is ample.

`signal.resample` already handles the 64 Hz → device-rate harmonization.

**Named risk, equal in weight to the wrist/waist gap.** WESAD and DaLiA both use
an Empatica E4: green LED, adaptive gain, a properly tensioned strap. The device
is a MAX30102 breakout on hand-cut veroboard with a hand-made wrist mount. That
gap gets measured and reported, not augmented away.

This is also the strongest argument for keeping the HR branch an **untrained
deviation statistic** rather than a learned model: a z-score against a rolling
personal baseline transfers across sensors far better than anything fitted to E4
amplitudes. Stage C tests that before any model is built.

---

## 9. Staged sequence

Each stage lands independently.

| # | Work | What it settles |
|---|---|---|
| **A0** | Bench diagnostic: sound module analog or digital; MAX30102 raw FIFO read at 100 sps on a wrist | Half a day, and it decides Plan A vs Plan B before code is written |
| **A1** | Correct the hardware claims in `README.md:7-8,76`, `export.py:63-64,124,226,235`, `models/movement.py:8-13`, and the 4 stale assertions in `tests/test_export.py`; rewrite `firmware_notes` for LX6 + esp-dsp | The repo currently prints firmware advice that does nothing on this part. Unblocked by everything else |
| **B** | `datasets/wesad.py`, `datasets/dalia.py`, loaders and tests, behind an availability gate | The HR corpora load and contain what is assumed |
| **C** | `hr.py` plus a **frozen, untrained** deviation rule scored against WESAD stress and DaLiA negatives, as recall-at-FAR | Whether a statistic already captures most of the separation. If it does, do not train a model — the same lesson as the dropped `lay` class |
| **D** | Audio, per the A0 fork | The first honest audio number, or an honest admission there is not one |
| **E** | **Correlation measurement.** Compute the movement score and the HR score on the *same* WESAD windows and measure their dependence | The headline. `fused_far_upper_bound` assumes branch independence, which the README names as the largest unmeasurable risk in the design. WESAD makes it measurable |
| **F** | Calibrate three branches, set `implemented: true`, regenerate `fusion_source` | `CANNOT RAISE AN ALARM` disappears |
| **G** | Export budget: flash, arenas and RAM against ~320 KB DRAM | Whether it fits |

Stage 1 (SisFall / fall head) is currently out of scope by request. Folding it in
during Stage B is roughly 20 % extra work versus a second pass through the same
files later, and the scaffolding is already written.

---

## 10. Risks

- **Independence gets worse, not better.** A struggle drives movement *and* corrupts the PPG from one physical cause; exercise raises HR *and* movement together. Stage E measures the movement×HR pair. Nothing public measures the audio pair.
- **The 2-of-3 vote still cannot be validated end to end** after all this work. Three calibrated branches is not a validated system. This belongs in the README rather than being left for a reader to infer.
- **Power management and the vote are coupled.** Duty-cycling the MAX30102 so it samples only after the movement branch fires silently converts 2-of-3 into "movement AND (hr OR audio)" — the same accidental rule that F7 found in the firmware's weights. Any power manager must be checked against the vote's semantics. Note the fused-FAR arithmetic in `fusion.py` no longer describes the device if this happens.
- **Every audio positive is acted.** FSD50K screams are performed, on unknown microphones, not a MEMS element under a sleeve.
- **Privacy.** The proposal promises raw audio is never stored or transmitted. An on-device feature path keeps that promise; band energies at a 32 ms frame are not speech-reconstructible, which is worth stating explicitly.

### Hardware items that will bite during the build

1. **No 3.3 V regulator or boost converter in the BOM.** The Implementation Plan §3 wires TP4056 OUT+ to the DevKit's **5V** pin, but a 3.7 V cell into an AMS1117 (≈1.1 V dropout) browns out as soon as the cell sags. Needs either a boost to 5 V or a clean 3.3 V feed.
2. **TP4056 charge current.** Stock boards ship ~1 A via a 1.2 kΩ Rprog. Into a 350 mAh cell that is ≈2.9 C. Verify the resistor.
3. **Buzzer type is contradictory** across sources (§2). `tone()` implies passive.
4. Battery life at continuous duty is roughly 3–4 h against the plan's >6 h minimum. Explicitly deferred — not a blocker for the modelling work.

---

## 11. Open decisions

| # | Question | Default if unanswered |
|---|---|---|
| 1 | Sound module: analog or digital? | **Blocking for Stage D.** No default |
| 2 | Is the hardware in hand? Decides whether the sensor-simulation step can use a measured noise floor | Assume not; defer that calibration |
| 3 | Colab tier and Drive space — decides whether a caching layer is worth writing | Assume free tier; write the subset downloader |
| 4 | Kaggle credentials for SisFall | Not needed until Stage 1 is scoped in |
| 5 | Own data collection (Implementation Plan §2: 30–50 volunteers, ethics approval)? It is the only path to a synchronized triad and therefore to validating the vote | Assume no; state the limitation in the README |
| 6 | Merge `stage3-fusion-engine` into `main` before building on it? | Branch from `stage3-fusion-engine`; leave the merge as a separate decision |
| 7 | Deliverable and deadline — thesis, paper or demo? Changes whether to optimize for measured numbers or a running device | Assume measured numbers |

---

## 12. Immediate next actions

1. **A1** — the hardware corrections in §9. Self-contained, unblocked by every open question above, one commit.
2. **A0** — the bench diagnostic, which unblocks Stage D.
3. **B** — the WESAD and DaLiA loaders, gated only on decision 3.
