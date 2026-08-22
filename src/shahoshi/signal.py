"""Signal conditioning shared by every corpus. Pure NumPy/SciPy, no TensorFlow.

Every loader must put its data through the same conditioning, or the merged
dataset is incoherent. Two specific hazards this module exists to prevent:

1. **Aliasing on downsample.** SisFall samples at 200 Hz; our windows are 50 Hz.
   A fall impact is a broadband transient, so plain subsampling folds its
   high-frequency energy straight into the passband and *fabricates* a signal
   that was never there. `resample` always low-pass filters first.

2. **Inconsistent gravity handling.** UCI HAR ships acceleration with gravity
   already removed by a specific documented filter chain. MotionSense's
   `userAcceleration` is likewise gravity-free (Core Motion does it on-device).
   SisFall ships *raw* accelerometer counts, gravity included. Removing gravity
   from SisFall with a generic detrend rather than UCI's procedure would put it
   in a different feature space, and every merged-dataset number downstream
   would be measuring the mismatch instead of the activity.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal as sps
from scipy.ndimage import median_filter

# UCI HAR's documented filter chain (Anguita et al. 2013), replicated so that
# SisFall lands in the same feature space as the corpora that ship pre-filtered.
UCI_MEDIAN_KERNEL = 3        # samples
UCI_DENOISE_HZ = 20.0        # low-pass, removes sensor noise
UCI_GRAVITY_HZ = 0.3         # low-pass, isolates the gravity component
UCI_FILTER_ORDER = 3


def adc_to_units(raw: np.ndarray, full_scale: float, bits: int) -> np.ndarray:
    """Convert signed ADC counts to physical units.

    Implements the conversion given in the SisFall dataset readme::

        value = (2 * Range) / (2 ** Resolution) * AD

    Parameters
    ----------
    raw : array
        Signed integer ADC counts.
    full_scale : float
        Sensor range as the +/- figure (16 for an ADXL345 at +/-16 g,
        2000 for an ITG3200 at +/-2000 deg/s).
    bits : int
        ADC resolution in bits (13 for ADXL345, 16 for ITG3200, 14 for MMA8451Q).
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    return np.asarray(raw, dtype=np.float64) * (2.0 * full_scale) / (2**bits)


def resample(sig: np.ndarray, fs_in: float, fs_out: float, axis: int = 0) -> np.ndarray:
    """Rate-convert along `axis` with proper anti-alias filtering.

    Uses polyphase resampling, which designs an FIR anti-alias filter for the
    ratio and applies it before decimating. Never subsample by slicing.
    """
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("sample rates must be positive")
    if float(fs_in) == float(fs_out):
        return np.asarray(sig, dtype=np.float64)

    # Integer up/down factors from the exact ratio, so fractional reported rates
    # (which do turn up in the wild) do not silently round to something else.
    scale = 1000
    up, down = int(round(fs_out * scale)), int(round(fs_in * scale))
    g = gcd(up, down)
    return sps.resample_poly(np.asarray(sig, dtype=np.float64), up // g, down // g, axis=axis)


def _butter_lowpass(sig: np.ndarray, cutoff: float, fs: float, order: int, axis: int) -> np.ndarray:
    """Zero-phase Butterworth low-pass.

    `sosfiltfilt` rather than `lfilter`: we filter offline, and a causal filter's
    group delay would shift the acceleration channels relative to the gyroscope
    channels by a frequency-dependent amount. That misalignment is invisible in
    summary statistics and quietly degrades a temporal model.
    """
    sos = sps.butter(order, cutoff / (0.5 * fs), btype="low", output="sos")
    # filtfilt requires padlen < signal length along the filtered axis; back off
    # for short trials rather than raising.
    default_pad = 3 * (2 * order + 1)
    padlen = int(min(default_pad, max(0, sig.shape[axis] - 1)))
    return sps.sosfiltfilt(sos, sig, axis=axis, padlen=padlen)


def denoise(sig: np.ndarray, fs: float, axis: int = 0) -> np.ndarray:
    """UCI's noise-removal stage: median filter, then a 20 Hz low-pass."""
    sig = np.asarray(sig, dtype=np.float64)
    k = UCI_MEDIAN_KERNEL
    if sig.shape[axis] > k:
        size = [1] * sig.ndim
        size[axis] = k
        sig = median_filter(sig, size=tuple(size), mode="nearest")
    if fs > 2 * UCI_DENOISE_HZ:
        sig = _butter_lowpass(sig, UCI_DENOISE_HZ, fs, UCI_FILTER_ORDER, axis)
    return sig


def gravity_split(
    acc: np.ndarray, fs: float, cutoff: float = UCI_GRAVITY_HZ, axis: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split total acceleration into (body, gravity) using UCI HAR's procedure.

    A `cutoff`-Hz low-pass isolates the slowly-varying gravity component; the
    remainder is body acceleration. Returns both, because `gravity` is not
    waste: it is the only cue that distinguishes lying from sitting, and if the
    posture classes are ever revisited it is the channel to add back.

    Returns
    -------
    body, gravity : arrays with the same shape as `acc`
    """
    acc = np.asarray(acc, dtype=np.float64)
    if fs <= 2 * cutoff:
        raise ValueError(f"fs={fs} too low to isolate a {cutoff} Hz component")
    gravity = _butter_lowpass(acc, cutoff, fs, UCI_FILTER_ORDER, axis)
    return acc - gravity, gravity


def svm(acc: np.ndarray, axis: int = -1) -> np.ndarray:
    """Signal vector magnitude, sqrt(ax^2 + ay^2 + az^2).

    Used to localize fall impacts: the impact is the global maximum of SVM
    within a trial. Feed it *total* acceleration (gravity included) when
    locating impacts -- the peak is what matters, not the DC offset.
    """
    acc = np.asarray(acc, dtype=np.float64)
    return np.sqrt((acc**2).sum(axis=axis))
