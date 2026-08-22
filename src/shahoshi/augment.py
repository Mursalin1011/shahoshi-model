"""Augmentation, aimed squarely at the mounting-position gap.

Every corpus we can train on is mounted at the waist (UCI HAR, SisFall) or in a
trouser pocket (MotionSense). The device is worn on the wrist. That mismatch is
the largest single source of error in the system and no additional public
dataset fixes it, because the problem is the sensor's orientation and lever arm
relative to the body, not the amount of data.

Rotation augmentation is the cheapest available lever: training the model on
randomly reoriented copies of each window forces it toward features that do not
depend on which way the device happens to be facing. It does not close the gap
-- a wrist also experiences genuinely different motion, not merely rotated
motion -- but it is measurable, and the leave-one-dataset-out delta with and
without it is a result worth reporting rather than a guess.

Physics note on rotating the gyroscope
--------------------------------------
Acceleration is a vector, so it transforms as ``a' = R a``. Angular velocity is
a pseudovector, so it transforms as ``omega' = det(R) * R omega``. For a proper
rotation (``det(R) == +1``) the two are identical and the same matrix applies to
both triads. This module therefore only ever generates proper rotations -- a
reflection would flip the gyroscope's sign relative to the accelerometer and
teach the model a body that does not exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .datasets.base import ACC, GYR


def random_rotations(
    n: int, max_deg: float = 180.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """`n` proper rotation matrices, shape (n, 3, 3).

    Built by axis-angle (Rodrigues) with a uniformly-distributed axis and an
    angle drawn uniformly from [0, max_deg]. This is not Haar-uniform on SO(3),
    which is deliberate: parameterizing by angle is what lets `max_deg` bound
    the severity of the augmentation, so the ablation can sweep it.

    `max_deg=0` yields exact identities, so the augmentation can be switched off
    without a separate code path.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= max_deg <= 360.0:
        raise ValueError(f"max_deg must be in [0, 360]; got {max_deg}")
    rng = np.random.default_rng() if rng is None else rng

    if max_deg == 0.0:
        return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()

    # Uniform axis on the sphere: normalize a Gaussian, which is isotropic.
    axis = rng.normal(size=(n, 3))
    norms = np.linalg.norm(axis, axis=1, keepdims=True)
    # A zero draw is astronomically unlikely but would divide by zero.
    degenerate = norms[:, 0] < 1e-12
    axis[degenerate] = np.array([1.0, 0.0, 0.0])
    norms[degenerate] = 1.0
    axis = axis / norms

    angle = rng.uniform(0.0, np.deg2rad(max_deg), size=n)

    # Rodrigues: R = I + sin(t) K + (1 - cos(t)) K^2, K the cross-product matrix.
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = np.zeros(n)
    K = np.stack(
        [
            np.stack([zero, -kz, ky], axis=-1),
            np.stack([kz, zero, -kx], axis=-1),
            np.stack([-ky, kx, zero], axis=-1),
        ],
        axis=-2,
    )
    s = np.sin(angle)[:, None, None]
    c = np.cos(angle)[:, None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def rotate(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply per-window rotations to the accelerometer and gyroscope triads.

    Parameters
    ----------
    X : (n, win, 6)
    R : (n, 3, 3) or (3, 3)
        Proper rotations. The *same* matrix is applied to both triads; see the
        pseudovector note in the module docstring.
    """
    X = np.asarray(X, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    if R.ndim == 2:
        R = np.broadcast_to(R, (len(X), 3, 3))
    if R.shape != (len(X), 3, 3):
        raise ValueError(f"R must be (n, 3, 3) or (3, 3); got {R.shape}")

    out = X.copy()
    # (n, win, 3) @ (n, 3, 3)^T  ->  rotate each 3-vector in each window
    out[:, :, ACC] = np.einsum("nij,nkj->nik", X[:, :, ACC], R)
    out[:, :, GYR] = np.einsum("nij,nkj->nik", X[:, :, GYR], R)
    return out


def scale(
    X: np.ndarray, sigma: float = 0.1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Per-window, per-channel multiplicative gain jitter.

    Models sensor gain differences and body-size differences. Applied per
    channel rather than globally so it does not degenerate into a pure
    amplitude change the model can normalize away.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, dtype=np.float32)
    gain = rng.normal(1.0, sigma, size=(len(X), 1, X.shape[2])).astype(np.float32)
    return X * gain


def jitter(
    X: np.ndarray, sigma: float = 0.02, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Additive white noise, modelling sensor noise floor differences."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, dtype=np.float32)
    return X + rng.normal(0.0, sigma, size=X.shape).astype(np.float32)


def time_warp(
    X: np.ndarray, max_ratio: float = 0.15, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Resample each window on a slightly stretched or squeezed time axis.

    Models pace variation between people. Implemented as linear interpolation
    onto a warped grid, keeping the window length fixed so the output still
    feeds a fixed-shape model.
    """
    if not 0.0 <= max_ratio < 1.0:
        raise ValueError(f"max_ratio must be in [0, 1); got {max_ratio}")
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, dtype=np.float32)
    if max_ratio == 0.0:
        return X.copy()

    n, win, ch = X.shape
    ratios = rng.uniform(1.0 - max_ratio, 1.0 + max_ratio, size=n)
    base = np.arange(win, dtype=np.float64)
    out = np.empty_like(X)
    for i in range(n):
        # Warped source positions, clipped so we never extrapolate past the edge.
        src = np.clip((base - (win - 1) / 2) * ratios[i] + (win - 1) / 2, 0, win - 1)
        for c in range(ch):
            out[i, :, c] = np.interp(src, base, X[i, :, c])
    return out


def channel_dropout(
    X: np.ndarray, p: float = 0.1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Zero whole channels at random, per window.

    Models a partially failed or saturated sensor axis and discourages the model
    from betting everything on one channel. Never drops all six.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must be in [0, 1); got {p}")
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, dtype=np.float32).copy()
    if p == 0.0:
        return X

    drop = rng.random((len(X), X.shape[2])) < p
    # Keep at least one channel alive in every window.
    all_dropped = drop.all(axis=1)
    if all_dropped.any():
        keep = rng.integers(0, X.shape[2], size=int(all_dropped.sum()))
        drop[np.where(all_dropped)[0], keep] = False
    X[drop[:, None, :].repeat(X.shape[1], axis=1)] = 0.0
    return X


def make_augmenter(
    rotation_deg: float = 180.0,
    scale_sigma: float = 0.1,
    jitter_sigma: float = 0.02,
    warp_ratio: float = 0.15,
    dropout_p: float = 0.0,
    seed: int = 42,
) -> Callable[[np.ndarray], np.ndarray]:
    """Compose the augmentations into one callable over a batch of windows.

    Every parameter defaults to something that can be set to 0 to disable that
    stage, so the ablation table is a matter of config rather than code. Order
    matters: rotation first, while the triads still mean what they say, then
    gain, then time warp, then additive noise last so the noise floor is not
    itself rescaled.
    """
    rng = np.random.default_rng(seed)

    def apply(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if rotation_deg:
            X = rotate(X, random_rotations(len(X), rotation_deg, rng))
        if scale_sigma:
            X = scale(X, scale_sigma, rng)
        if warp_ratio:
            X = time_warp(X, warp_ratio, rng)
        if dropout_p:
            X = channel_dropout(X, dropout_p, rng)
        if jitter_sigma:
            X = jitter(X, jitter_sigma, rng)
        return X

    return apply


def expand(
    X: np.ndarray,
    y: Sequence[np.ndarray],
    times: int = 1,
    augmenter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the original data plus `times` augmented copies, labels tiled to match.

    Offline expansion rather than on-the-fly augmentation: it keeps the training
    loop identical between the augmented and un-augmented arms of the ablation,
    at the cost of memory. At our dataset size that trade is comfortable.
    """
    if times < 0:
        raise ValueError("times must be non-negative")
    augmenter = make_augmenter() if augmenter is None else augmenter

    Xs = [np.asarray(X, dtype=np.float32)]
    for _ in range(times):
        Xs.append(augmenter(X))
    reps = times + 1
    return np.concatenate(Xs), [np.tile(np.asarray(a), reps) for a in y]
