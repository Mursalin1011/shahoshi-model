"""Tests for augmentation.

The rotation tests carry the weight. Rotation augmentation is the project's only
lever against the waist/pocket-to-wrist mounting gap, and two ways of getting it
subtly wrong -- generating reflections instead of rotations, or rotating the
accelerometer and gyroscope by different matrices -- both produce output that
looks entirely plausible while teaching the model a body that cannot exist.
"""

import numpy as np
import pytest

from shahoshi.augment import (
    channel_dropout,
    expand,
    jitter,
    make_augmenter,
    random_rotations,
    rotate,
    scale,
    time_warp,
)
from shahoshi.datasets.base import ACC, GYR
from shahoshi.signal import svm


def windows(n=8, win=128, ch=6, seed=0):
    return np.random.default_rng(seed).normal(0, 1, (n, win, ch)).astype(np.float32)


class TestRandomRotations:
    def test_shape(self):
        assert random_rotations(5, rng=np.random.default_rng(0)).shape == (5, 3, 3)

    def test_are_orthogonal(self):
        R = random_rotations(50, rng=np.random.default_rng(0))
        eye = np.einsum("nij,nkj->nik", R, R)
        assert np.allclose(eye, np.eye(3), atol=1e-10)

    def test_are_proper_rotations_not_reflections(self):
        """det must be +1. A reflection flips the gyroscope pseudovector's sign
        relative to the accelerometer, which is physically impossible."""
        R = random_rotations(200, rng=np.random.default_rng(1))
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_zero_degrees_is_exactly_identity(self):
        R = random_rotations(4, max_deg=0.0, rng=np.random.default_rng(0))
        assert np.array_equal(R, np.broadcast_to(np.eye(3), (4, 3, 3)))

    def test_max_deg_bounds_the_rotation_angle(self):
        R = random_rotations(200, max_deg=30.0, rng=np.random.default_rng(2))
        # angle from the trace: tr(R) = 1 + 2 cos(theta)
        cos_theta = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        assert np.rad2deg(np.arccos(cos_theta)).max() <= 30.0 + 1e-6

    def test_axes_are_isotropic(self):
        """No preferred axis: the mean rotation axis should be near zero."""
        R = random_rotations(4000, max_deg=180.0, rng=np.random.default_rng(3))
        # Extract the axis via the antisymmetric part.
        A = (R - np.transpose(R, (0, 2, 1))) / 2
        axes = np.stack([A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]], axis=1)
        norm = np.linalg.norm(axes, axis=1, keepdims=True)
        axes = axes / np.maximum(norm, 1e-12)
        assert np.abs(axes.mean(axis=0)).max() < 0.1

    def test_deterministic_for_a_seed(self):
        a = random_rotations(10, rng=np.random.default_rng(5))
        b = random_rotations(10, rng=np.random.default_rng(5))
        assert np.array_equal(a, b)

    def test_rejects_bad_max_deg(self):
        with pytest.raises(ValueError):
            random_rotations(3, max_deg=400.0)


class TestRotate:
    def test_preserves_per_triad_magnitude(self):
        """Rotation cannot change the length of a vector, so SVM of each triad
        must be identical before and after. If this fails, the accelerometer and
        gyroscope are not being rotated consistently."""
        X = windows(16)
        Xr = rotate(X, random_rotations(16, rng=np.random.default_rng(0)))
        assert np.allclose(svm(X[:, :, ACC]), svm(Xr[:, :, ACC]), atol=1e-4)
        assert np.allclose(svm(X[:, :, GYR]), svm(Xr[:, :, GYR]), atol=1e-4)

    def test_identity_rotation_is_a_no_op(self):
        X = windows(4)
        assert np.allclose(rotate(X, np.eye(3)), X, atol=1e-6)

    def test_accepts_a_single_shared_matrix(self):
        X = windows(4)
        R = random_rotations(1, rng=np.random.default_rng(0))[0]
        out = rotate(X, R)
        assert out.shape == X.shape
        # every window got the same rotation
        assert np.allclose(out, rotate(X, np.broadcast_to(R, (4, 3, 3))), atol=1e-6)

    def test_ninety_degrees_about_z_maps_x_to_y(self):
        X = np.zeros((1, 4, 6), dtype=np.float32)
        X[0, :, 0] = 1.0   # acc_x
        X[0, :, 3] = 1.0   # gyr_x
        Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        out = rotate(X, Rz)
        assert np.allclose(out[0, :, 0], 0.0, atol=1e-6)
        assert np.allclose(out[0, :, 1], 1.0, atol=1e-6)
        assert np.allclose(out[0, :, 4], 1.0, atol=1e-6)

    def test_does_not_mix_accelerometer_into_gyroscope(self):
        X = np.zeros((2, 8, 6), dtype=np.float32)
        X[:, :, ACC] = 1.0  # gyro channels left at zero
        out = rotate(X, random_rotations(2, rng=np.random.default_rng(0)))
        assert np.abs(out[:, :, GYR]).max() < 1e-6

    def test_does_not_mutate_its_input(self):
        X = windows(4)
        before = X.copy()
        rotate(X, random_rotations(4, rng=np.random.default_rng(0)))
        assert np.array_equal(X, before)

    def test_rejects_mismatched_batch(self):
        with pytest.raises(ValueError):
            rotate(windows(4), random_rotations(3, rng=np.random.default_rng(0)))


class TestScaleJitterWarpDropout:
    def test_scale_changes_amplitude_but_not_shape(self):
        X = windows(8)
        out = scale(X, sigma=0.2, rng=np.random.default_rng(0))
        assert out.shape == X.shape
        assert not np.allclose(out, X)
        # correlation per channel is preserved (pure gain)
        for c in range(6):
            r = np.corrcoef(X[0, :, c], out[0, :, c])[0, 1]
            assert abs(r) > 0.999

    def test_scale_sigma_zero_is_identity(self):
        X = windows(4)
        assert np.allclose(scale(X, sigma=0.0, rng=np.random.default_rng(0)), X)

    def test_jitter_adds_noise_of_about_the_requested_size(self):
        X = np.zeros((32, 128, 6), dtype=np.float32)
        out = jitter(X, sigma=0.05, rng=np.random.default_rng(0))
        assert out.std() == pytest.approx(0.05, rel=0.1)

    def test_time_warp_preserves_shape_and_stays_in_range(self):
        X = windows(8)
        out = time_warp(X, max_ratio=0.2, rng=np.random.default_rng(0))
        assert out.shape == X.shape
        # interpolation never extrapolates, so values stay within the original span
        assert out.max() <= X.max() + 1e-5
        assert out.min() >= X.min() - 1e-5

    def test_time_warp_zero_ratio_is_identity(self):
        X = windows(4)
        assert np.allclose(time_warp(X, max_ratio=0.0, rng=np.random.default_rng(0)), X)

    def test_time_warp_actually_warps(self):
        X = windows(8, seed=3)
        assert not np.allclose(time_warp(X, 0.2, np.random.default_rng(0)), X)

    def test_channel_dropout_zeroes_whole_channels(self):
        X = np.ones((64, 16, 6), dtype=np.float32)
        out = channel_dropout(X, p=0.5, rng=np.random.default_rng(0))
        # every channel is either fully intact or fully zero, per window
        for w in out:
            for c in range(6):
                col = w[:, c]
                assert np.all(col == 0) or np.all(col == 1)

    def test_channel_dropout_never_drops_every_channel(self):
        X = np.ones((256, 8, 6), dtype=np.float32)
        out = channel_dropout(X, p=0.95, rng=np.random.default_rng(0))
        assert (np.abs(out).sum(axis=(1, 2)) > 0).all()

    def test_channel_dropout_p_zero_is_identity(self):
        X = windows(4)
        assert np.allclose(channel_dropout(X, p=0.0, rng=np.random.default_rng(0)), X)

    @pytest.mark.parametrize(
        "fn,kwargs",
        [
            (scale, {"sigma": -1.0}),
            (jitter, {"sigma": -1.0}),
            (time_warp, {"max_ratio": 1.5}),
            (channel_dropout, {"p": 1.0}),
        ],
    )
    def test_reject_out_of_range_parameters(self, fn, kwargs):
        with pytest.raises(ValueError):
            fn(windows(2), **kwargs)


class TestMakeAugmenter:
    def test_output_shape_and_dtype(self):
        X = windows(16)
        out = make_augmenter(seed=0)(X)
        assert out.shape == X.shape
        assert out.dtype == np.float32

    def test_all_stages_disabled_is_identity(self):
        X = windows(8)
        aug = make_augmenter(
            rotation_deg=0, scale_sigma=0, jitter_sigma=0, warp_ratio=0, dropout_p=0
        )
        assert np.allclose(aug(X), X)

    def test_changes_the_data_when_enabled(self):
        X = windows(8)
        assert not np.allclose(make_augmenter(seed=0)(X), X)

    def test_deterministic_for_a_seed(self):
        X = windows(8)
        assert np.allclose(make_augmenter(seed=1)(X), make_augmenter(seed=1)(X))

    def test_successive_calls_differ(self):
        """A stateful generator, so two batches are not augmented identically."""
        X = windows(8)
        aug = make_augmenter(seed=0)
        assert not np.allclose(aug(X), aug(X))


class TestExpand:
    def test_returns_original_plus_copies(self):
        X = windows(10)
        y = np.arange(10)
        Xe, (ye,) = expand(X, [y], times=2, augmenter=make_augmenter(seed=0))
        assert len(Xe) == 30
        assert len(ye) == 30

    def test_original_data_comes_first_unmodified(self):
        X = windows(10)
        Xe, _ = expand(X, [np.arange(10)], times=1, augmenter=make_augmenter(seed=0))
        assert np.array_equal(Xe[:10], X)

    def test_labels_stay_aligned_with_windows(self):
        X = windows(6)
        y = np.arange(6)
        Xe, (ye,) = expand(X, [y], times=2, augmenter=make_augmenter(seed=0))
        # copy k of window i sits at index k*6 + i and must keep label i
        for k in range(3):
            assert np.array_equal(ye[k * 6 : (k + 1) * 6], y)

    def test_multiple_label_arrays_are_tiled_together(self):
        X = windows(5)
        y_act, y_fall = np.arange(5), np.zeros(5, dtype=np.int8)
        _, (a, f) = expand(X, [y_act, y_fall], times=1, augmenter=make_augmenter(seed=0))
        assert len(a) == len(f) == 10

    def test_times_zero_returns_the_original_only(self):
        X = windows(5)
        Xe, (ye,) = expand(X, [np.arange(5)], times=0)
        assert np.array_equal(Xe, X)
        assert len(ye) == 5

    def test_rejects_negative_times(self):
        with pytest.raises(ValueError):
            expand(windows(2), [np.arange(2)], times=-1)
