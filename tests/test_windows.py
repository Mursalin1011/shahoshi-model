"""Tests for windowing and event-overlap labelling."""

import numpy as np
import pytest

from shahoshi.windows import (
    channel_stats,
    label_by_overlap,
    n_windows,
    normalize,
    overlap_fraction,
    sliding_windows,
    window_bounds,
)


class TestNWindows:
    @pytest.mark.parametrize(
        "n,win,stride,expected",
        [
            (128, 128, 64, 1),
            (192, 128, 64, 2),
            (256, 128, 64, 3),
            (127, 128, 64, 0),   # shorter than one window
            (0, 128, 64, 0),
            (100, 10, 10, 10),   # exact tiling, no overlap
        ],
    )
    def test_counts(self, n, win, stride, expected):
        assert n_windows(n, win, stride) == expected

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            n_windows(100, 0, 10)
        with pytest.raises(ValueError):
            n_windows(100, 10, 0)


class TestSlidingWindows:
    def test_shape_and_stride(self):
        sig = np.arange(300 * 6, dtype=np.float32).reshape(300, 6)
        w = sliding_windows(sig, win=128, stride=64)
        assert w.shape == (3, 128, 6)
        # window k starts at sample k*stride
        assert np.array_equal(w[0], sig[0:128])
        assert np.array_equal(w[1], sig[64:192])
        assert np.array_equal(w[2], sig[128:256])

    def test_fifty_percent_overlap_is_real(self):
        """Consecutive windows at stride=win/2 share exactly half their samples.

        This is why splits must be subject-disjoint: these two windows are
        near-duplicates, and a random split would scatter them across train
        and test.
        """
        sig = np.arange(200 * 2, dtype=np.float32).reshape(200, 2)
        w = sliding_windows(sig, win=100, stride=50)
        assert np.array_equal(w[0][50:], w[1][:50])

    def test_short_signal_returns_empty_with_correct_dims(self):
        w = sliding_windows(np.zeros((10, 6)), win=128, stride=64)
        assert w.shape == (0, 128, 6)
        # concatenating an empty result must not blow up
        assert np.concatenate([w, np.zeros((2, 128, 6))]).shape == (2, 128, 6)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError):
            sliding_windows(np.zeros(100), win=10, stride=5)

    def test_result_is_contiguous_and_owns_its_data(self):
        w = sliding_windows(np.zeros((300, 6)), win=128, stride=64)
        assert w.flags["C_CONTIGUOUS"]
        assert w.base is None


class TestWindowBounds:
    def test_bounds_match_sliding_windows(self):
        b = window_bounds(300, win=128, stride=64)
        assert b.shape == (3, 2)
        assert np.array_equal(b, [[0, 128], [64, 192], [128, 256]])

    def test_empty_when_too_short(self):
        assert window_bounds(10, 128, 64).shape == (0, 2)


class TestOverlapFraction:
    def test_full_containment_either_normalizer(self):
        b = np.array([[0, 100]])
        assert overlap_fraction(b, 0, 100, "window")[0] == pytest.approx(1.0)
        assert overlap_fraction(b, 0, 100, "event")[0] == pytest.approx(1.0)

    def test_no_overlap(self):
        b = np.array([[0, 100]])
        assert overlap_fraction(b, 200, 300, "window")[0] == 0.0
        assert overlap_fraction(b, 200, 300, "event")[0] == 0.0

    def test_half_overlap_window_relative(self):
        b = np.array([[0, 100]])
        assert overlap_fraction(b, 50, 150, "window")[0] == pytest.approx(0.5)

    def test_short_event_inside_window(self):
        """The two normalizers answer different questions and must differ here."""
        b = np.array([[0, 100]])
        # 20 samples of event inside a 100-sample window:
        assert overlap_fraction(b, 40, 60, "window")[0] == pytest.approx(0.2)
        # ... but the window captures the entire event.
        assert overlap_fraction(b, 40, 60, "event")[0] == pytest.approx(1.0)

    def test_window_relative_ceiling_for_short_events(self):
        """Why the default for label_by_overlap is event-relative.

        A 1 s impact in a 2.56 s window can never exceed 39% window coverage, so
        a window-relative positive_min above that is unreachable.
        """
        b = np.array([[0, 128]])
        best = overlap_fraction(b, 40, 90, "window")[0]
        assert best == pytest.approx(50 / 128)
        assert best < 0.5

    def test_touching_boundary_is_not_overlap(self):
        """Half-open spans: an event starting exactly at window end does not overlap."""
        b = np.array([[0, 100]])
        assert overlap_fraction(b, 100, 200, "window")[0] == 0.0

    def test_rejects_inverted_span(self):
        with pytest.raises(ValueError):
            overlap_fraction(np.array([[0, 100]]), 200, 100)

    def test_rejects_bad_bounds_shape(self):
        with pytest.raises(ValueError):
            overlap_fraction(np.array([0, 100]), 0, 50)

    def test_rejects_unknown_normalizer(self):
        with pytest.raises(ValueError, match="relative_to"):
            overlap_fraction(np.array([[0, 100]]), 0, 50, relative_to="trial")

    def test_zero_length_event_does_not_divide_by_zero(self):
        assert overlap_fraction(np.array([[0, 100]]), 50, 50, "event")[0] == 0.0


class TestLabelByOverlap:
    def test_the_sisfall_shape_of_problem(self):
        """A 15 s trial at 50 Hz with a 1 s fall two thirds of the way in.

        Most windows must come out negative, at least one positive, and the
        transition windows must land in the ambiguous band rather than being
        forced into either class.
        """
        fs, win, stride = 50, 128, 64
        n = 15 * fs
        impact_start, impact_end = 10 * fs, 11 * fs

        bounds = window_bounds(n, win, stride)
        labels = label_by_overlap(bounds, impact_start, impact_end)

        assert (labels == 1).sum() >= 1, "the fall itself must produce a positive"
        assert (labels == 0).sum() > (labels == 1).sum(), (
            "most of a fall trial is not the fall -- if this inverts, we are "
            "labelling whole trials and inflating results"
        )
        assert (labels == -1).sum() >= 1, "partial-overlap windows must be ambiguous"

    def test_positive_window_actually_contains_the_impact(self):
        fs, win, stride = 50, 128, 64
        bounds = window_bounds(15 * fs, win, stride)
        labels = label_by_overlap(bounds, 10 * fs, 11 * fs)
        for lo, hi in bounds[labels == 1]:
            assert lo <= 10 * fs and hi >= 11 * fs

    def test_negative_windows_do_not_touch_the_impact(self):
        fs, win, stride = 50, 128, 64
        bounds = window_bounds(15 * fs, win, stride)
        labels = label_by_overlap(bounds, 10 * fs, 11 * fs)
        for lo, hi in bounds[labels == 0]:
            assert hi <= 10 * fs or lo >= 11 * fs

    def test_negative_max_default_excludes_any_touch(self):
        b = np.array([[0, 100]])
        # window captures 10 of the event's 20 samples: neither clean class
        assert label_by_overlap(b, 90, 110)[0] == -1

    def test_negative_max_can_be_loosened(self):
        b = np.array([[0, 100]])
        assert label_by_overlap(b, 90, 110, positive_min=0.8, negative_max=0.5)[0] == 0

    def test_positive_min_boundary_is_inclusive(self):
        b = np.array([[0, 100]])
        # window captures exactly half the event
        assert label_by_overlap(b, 50, 150, positive_min=0.5)[0] == 1

    def test_window_relative_mode_still_available(self):
        b = np.array([[0, 100]])
        assert label_by_overlap(b, 0, 100, positive_min=0.9, relative_to="window")[0] == 1

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            label_by_overlap(np.array([[0, 100]]), 0, 50, positive_min=0.2, negative_max=0.8)


class TestNormalize:
    def test_standardizes_to_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        X = rng.normal(3.0, 2.0, (200, 128, 6)).astype(np.float32)
        mean, std = channel_stats(X)
        Xn = normalize(X, mean, std)
        assert np.abs(Xn.reshape(-1, 6).mean(0)).max() < 1e-4
        assert np.abs(Xn.reshape(-1, 6).std(0) - 1.0).max() < 1e-3

    def test_output_is_float32(self):
        X = np.zeros((4, 128, 6), dtype=np.float64)
        assert normalize(X, np.zeros(6), np.ones(6)).dtype == np.float32

    def test_rejects_wrong_length_stats(self):
        with pytest.raises(ValueError):
            normalize(np.zeros((4, 128, 6)), np.zeros(3), np.ones(3))

    def test_rejects_zero_std(self):
        std = np.ones(6)
        std[2] = 0.0
        with pytest.raises(ValueError):
            normalize(np.zeros((4, 128, 6)), np.zeros(6), std)

    def test_stats_eps_keeps_constant_channel_usable(self):
        """A constant channel must not produce a zero std that then fails validation."""
        X = np.zeros((10, 128, 6), dtype=np.float32)
        mean, std = channel_stats(X)
        assert (std > 0).all()
        normalize(X, mean, std)  # must not raise
