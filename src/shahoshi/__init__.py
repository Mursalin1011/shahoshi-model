"""Shahoshi: multimodal edge-ML threat detection for a wrist-worn safety wearable.

Layout
------
datasets/   one module per corpus, all returning the same `WindowSet` contract
signal.py   resampling, gravity separation, magnitude -- pure NumPy/SciPy
windows.py  sliding-window extraction and event-based window labelling
splits.py   subject-disjoint, leave-one-subject-out, leave-one-dataset-out
augment.py  mount-invariance augmentation (rotation, scale, jitter, warp)
scoring.py  entropy / Mahalanobis novelty scoring and false-alarm calibration
models/     Keras model definitions            (requires TensorFlow)
quantize.py int8 conversion and int8 inference (requires TensorFlow)
export.py   C array + op-resolver + config emission for ESP-IDF
manifest.py per-run provenance capture
"""

__version__ = "0.1.0"
