"""R-peak detection, used only to read ground truth out of a corpus.

WESAD ships a 700 Hz chest ECG but no heart-rate track, so the reference HR this
project scores against has to be derived. PPG-DaLiA, by contrast, ships its
ECG-derived HR directly and never comes through here.

**This is not part of the device pipeline and must never be.** It runs once, at
load time, on a chest ECG that the wearable does not have, to produce the truth
that the wrist PPG estimate in `hr.py` will be measured against. Using it as a
feature would be scoring a model against its own input.

The detector is deliberately the classical Pan-Tompkins chain -- bandpass,
derivative, square, moving-window integrate, threshold -- rather than anything
adaptive. On a clean chest ECG that is more than sufficient, it has no fitted
parameters that could quietly encode something about WESAD, and every stage is
inspectable when a subject's trace turns out to be unusual.

Beats that survive detection are still filtered on physiology: a rate outside
30-220 bpm, or an interval that jumps more than `MAX_RR_CHANGE` against its
neighbours, is dropped rather than averaged in. Chest electrodes come loose, and
an artefact burst otherwise contributes a 400 bpm "beat" to the ground truth.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps

# Pan-Tompkins passband: isolates the QRS complex from P/T waves and baseline
# wander below, and from muscle noise above.
BAND_HZ = (5.0, 15.0)

# Moving-window integration length. Roughly one QRS width, so the integrator
# produces one broad hump per complex instead of several sharp edges.
INTEGRATION_S = 0.150

# Refractory period. 0.25 s caps detection at 240 bpm, safely above any rate a
# resting or mildly stressed subject reaches, and rejects the T wave that would
# otherwise be counted as a second beat.
REFRACTORY_S = 0.25

# Peak height as a fraction of the integrated signal's 99th percentile. A
# percentile rather than the max: one motion spike can be orders of magnitude
# above every real QRS and would push a max-relative threshold above all of them.
PEAK_HEIGHT_FRAC = 0.25

# Physiological bounds on an accepted instantaneous rate, in bpm.
HR_MIN_BPM = 30.0
HR_MAX_BPM = 220.0

# Maximum fractional change between an RR interval and the local median before
# the beat is treated as a detection error rather than a real rate change.
MAX_RR_CHANGE = 0.33


def rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Sample indices of detected R peaks.

    Parameters
    ----------
    ecg : (n,) array
    fs : float
        Sample rate in Hz. WESAD's chest ECG is 700 Hz.
    """
    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if fs <= 0:
        raise ValueError(f"fs must be positive; got {fs}")
    if ecg.size < int(round(fs)):
        return np.zeros(0, dtype=np.int64)

    nyq = 0.5 * fs
    if BAND_HZ[1] >= nyq:
        raise ValueError(
            f"fs={fs} Hz is too low for a {BAND_HZ[0]}-{BAND_HZ[1]} Hz passband"
        )

    sos = sps.butter(2, [BAND_HZ[0] / nyq, BAND_HZ[1] / nyq], btype="band", output="sos")
    padlen = int(min(3 * 5, max(0, ecg.size - 1)))
    filtered = sps.sosfiltfilt(sos, ecg, padlen=padlen)

    squared = np.diff(filtered, prepend=filtered[0]) ** 2
    width = max(1, int(round(INTEGRATION_S * fs)))
    integrated = np.convolve(squared, np.ones(width) / width, mode="same")

    scale = np.percentile(integrated, 99)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(0, dtype=np.int64)

    peaks, _ = sps.find_peaks(
        integrated / scale,
        height=PEAK_HEIGHT_FRAC,
        distance=max(1, int(round(REFRACTORY_S * fs))),
    )
    return peaks.astype(np.int64)


def instantaneous_hr(ecg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-beat heart rate from an ECG.

    Returns
    -------
    t : (k,) float64
        Time in seconds of each accepted interval, stamped at its midpoint --
        an RR interval describes the span between two beats, not either end.
    bpm : (k,) float64

    Both arrays are empty when nothing survives detection and filtering, which
    is a real outcome for a subject whose electrodes failed; callers treat that
    as "no reference for this subject" rather than as an error.
    """
    peaks = rpeaks(ecg, fs)
    if len(peaks) < 3:
        return np.zeros(0), np.zeros(0)

    rr = np.diff(peaks) / float(fs)
    t = (peaks[:-1] + peaks[1:]) / 2.0 / float(fs)
    with np.errstate(divide="ignore", invalid="ignore"):
        bpm = 60.0 / rr

    ok = np.isfinite(bpm) & (bpm >= HR_MIN_BPM) & (bpm <= HR_MAX_BPM)
    if ok.sum() < 3:
        return np.zeros(0), np.zeros(0)

    # Reject intervals that disagree with their neighbourhood. A 5-beat median
    # tracks a genuine rate change within a couple of beats but is unmoved by an
    # isolated missed or doubled detection.
    rr_ok = rr[ok]
    local = _running_median(rr_ok, k=5)
    keep = np.abs(rr_ok - local) <= (MAX_RR_CHANGE * local)
    return t[ok][keep], bpm[ok][keep]


def _running_median(x: np.ndarray, k: int) -> np.ndarray:
    """Centred running median with edge replication, k odd."""
    if k % 2 == 0:
        raise ValueError("k must be odd")
    if len(x) < k:
        return np.full(len(x), float(np.median(x)))
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    idx = np.arange(len(x))[:, None] + np.arange(k)[None, :]
    return np.median(padded[idx], axis=1)
