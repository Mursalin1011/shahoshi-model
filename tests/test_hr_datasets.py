"""HR-branch loader tests against synthetic corpus trees.

Same discipline as `test_datasets.py`: build the on-disk layout each corpus
actually ships -- including the Python 2 pickles -- and assert the loader
produces a correct `HRWindowSet`. The parsing is where the silent mistakes live,
and for this branch there are three specific ones worth pinning:

  * a categorical label track resampled by filtering instead of by index,
    producing windows labelled with conditions that never occurred;
  * an accelerometer left in the E4's 1/64 g units, which loads and trains and
    is wrong by a factor of 64;
  * a window straddling a condition boundary being given a majority label,
    which inflates apparent separability in the flattering direction.

Each has a test below that fails if the guard is removed.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from shahoshi import datasets
from shahoshi.datasets import availability, cache, dalia, e4, ecg, wesad
from shahoshi.datasets.base import (
    HR_FS,
    N_HR_CHANNELS,
    UNKNOWN,
    HRWindowSet,
)

FS_CHEST = 700


# --------------------------------------------------------------------------
# Synthetic signal builders
# --------------------------------------------------------------------------

def synth_ecg(seconds: float, bpm: float, fs: int = FS_CHEST, noise: float = 0.01):
    """An ECG-like trace: narrow biphasic QRS complexes at an exact rate.

    Each complex is written only into its own +-6 sigma neighbourhood. Adding a
    full-length Gaussian per beat instead is the obvious way to write this and
    costs ~500 passes over a 336k-sample array per subject, which is what turned
    this file into five minutes of the suite's runtime. Beat times stay exact --
    the window is placed around the unrounded time, not snapped to a sample.
    """
    n = int(seconds * fs)
    sig = np.zeros(n)
    period = 60.0 / bpm
    sigma = 0.008                                  # ~8 ms, a plausible QRS width
    half = int(round(6 * sigma * fs))
    for beat in np.arange(period / 2, seconds, period):
        centre = int(round(beat * fs))
        lo, hi = max(centre - half, 0), min(centre + half + 1, n)
        if lo >= hi:
            continue
        tau = np.arange(lo, hi) / fs - beat
        sig[lo:hi] += np.exp(-0.5 * (tau / sigma) ** 2)
        sig[lo:hi] -= 0.3 * np.exp(-0.5 * ((tau - 2 * sigma) / sigma) ** 2)
    rng = np.random.default_rng(0)
    return sig + rng.normal(0, noise, n)


def synth_ppg(seconds: float, bpm: float, fs: int = 64):
    n = int(seconds * fs)
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * (bpm / 60.0) * t) + 0.1 * np.sin(2 * np.pi * 0.25 * t)


def build_wesad(root, subjects=("S2", "S3", "S4"), minutes=8, bpm=66.0):
    """A WESAD tree: one pickle per subject, four equal condition blocks."""
    seconds = minutes * 60
    for k, sid in enumerate(subjects):
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)

        n700 = int(seconds * FS_CHEST)
        label = np.zeros(n700, dtype=np.int64)
        q = n700 // 5
        # 0 = transient, then baseline / stress / amusement / meditation.
        for i, code in enumerate((1, 2, 3, 4), start=1):
            label[i * q:(i + 1) * q] = code

        acc = np.zeros((int(seconds * 32), 3))
        acc[:, 2] = 64.0                            # exactly 1 g in E4 units
        payload = {
            "subject": sid,
            "signal": {
                "wrist": {
                    "BVP": synth_ppg(seconds, bpm + k).reshape(-1, 1),
                    "ACC": acc,
                },
                "chest": {"ECG": synth_ecg(seconds, bpm + k).reshape(-1, 1)},
            },
            "label": label,
        }
        with open(d / f"{sid}.pkl", "wb") as fh:
            pickle.dump(payload, fh)
    return root


def build_dalia(root, subjects=("S1", "S2"), minutes=8, bpm=72.0):
    """A PPG-DaLiA tree, including its 2 s-stamped HR ground truth."""
    seconds = minutes * 60
    for k, sid in enumerate(subjects):
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)

        n4 = int(seconds * 4)
        activity = np.zeros(n4, dtype=np.int64)
        q = n4 // 4
        for i, code in enumerate((1, 4, 7), start=1):
            activity[i * q:(i + 1) * q] = code

        acc = np.zeros((int(seconds * 32), 3))
        acc[:, 1] = 64.0
        payload = {
            "subject": sid,
            "signal": {
                "wrist": {
                    "BVP": synth_ppg(seconds, bpm + k).reshape(-1, 1),
                    "ACC": acc,
                },
            },
            "activity": activity,
            "label": np.full(int(seconds / 2), bpm + k),
        }
        with open(d / f"{sid}.pkl", "wb") as fh:
            pickle.dump(payload, fh)
    return root


def make_hrws(n=200, stress_frac=0.4, subjects=("a", "b", "c", "d"), win=800):
    """A synthetic HRWindowSet that passes the gate, for gate-failure tests."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (n, win, N_HR_CHANNELS)).astype(np.float32)
    X[:, :, 1] = 0.0
    X[:, :, 2] = 0.0
    X[:, :, 3] = 1.0                                 # |acc| = 1 g
    n_pos = int(n * stress_frac)
    y = np.zeros(n, dtype=np.int8)
    y[:n_pos] = 1
    cond = np.where(y == 1, "stress", "baseline").astype(object)
    return HRWindowSet(
        X=X,
        y_stress=y,
        condition=cond,
        hr_ref=np.full(n, 70.0, dtype=np.float32),
        subject=np.array([subjects[i % len(subjects)] for i in range(n)], dtype=object),
        dataset=np.full(n, "synth", dtype=object),
    )


# --------------------------------------------------------------------------
# HRWindowSet contract
# --------------------------------------------------------------------------

class TestHRWindowSet:
    def test_rejects_wrong_channel_count(self):
        with pytest.raises(ValueError, match="HR_CHANNELS"):
            HRWindowSet(
                X=np.zeros((4, 10, 6), dtype=np.float32),
                y_stress=np.zeros(4),
                condition=np.array(["x"] * 4, dtype=object),
                hr_ref=np.zeros(4),
                subject=np.array(["s"] * 4, dtype=object),
                dataset=np.array(["d"] * 4, dtype=object),
            )

    def test_rejects_out_of_range_stress_label(self):
        with pytest.raises(ValueError, match="y_stress"):
            HRWindowSet(
                X=np.zeros((2, 10, 4), dtype=np.float32),
                y_stress=np.array([1, 7]),
                condition=np.array(["x"] * 2, dtype=object),
                hr_ref=np.zeros(2),
                subject=np.array(["s"] * 2, dtype=object),
                dataset=np.array(["d"] * 2, dtype=object),
            )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="condition has length"):
            HRWindowSet(
                X=np.zeros((3, 10, 4), dtype=np.float32),
                y_stress=np.zeros(3),
                condition=np.array(["x"], dtype=object),
                hr_ref=np.zeros(3),
                subject=np.array(["s"] * 3, dtype=object),
                dataset=np.array(["d"] * 3, dtype=object),
            )

    def test_unknown_is_a_legal_stress_label(self):
        ws = make_hrws(n=8)
        ws.y_stress[:2] = UNKNOWN
        HRWindowSet(
            X=ws.X, y_stress=ws.y_stress, condition=ws.condition,
            hr_ref=ws.hr_ref, subject=ws.subject, dataset=ws.dataset,
        )

    def test_hours_uses_window_duration(self):
        ws = make_hrws(n=450, win=800)            # 450 windows x 8 s
        assert ws.win_seconds == pytest.approx(8.0)
        assert ws.hours() == pytest.approx(1.0)

    def test_has_reference_tracks_nan(self):
        ws = make_hrws(n=10)
        ws.hr_ref[:4] = np.nan
        assert ws.has_reference.sum() == 6

    def test_subset_and_concat_round_trip(self):
        ws = make_hrws(n=20)
        a, b = ws.subset(np.arange(0, 10)), ws.subset(np.arange(10, 20))
        merged = HRWindowSet.concat([a, b])
        assert len(merged) == 20
        assert np.array_equal(merged.y_stress, ws.y_stress)

    def test_concat_refuses_mismatched_sample_rates(self):
        a = make_hrws(n=4)
        b = make_hrws(n=4)
        b.fs = 64
        with pytest.raises(ValueError, match="sample rates disagree"):
            HRWindowSet.concat([a, b])

    def test_summary_reports_conditions_and_reference(self):
        text = make_hrws(n=50).summary()
        assert "stress" in text and "baseline" in text
        assert "hr_ref" in text


# --------------------------------------------------------------------------
# ECG -> ground-truth HR
# --------------------------------------------------------------------------

class TestEcg:
    @pytest.mark.parametrize("bpm", [48.0, 66.0, 95.0, 130.0])
    def test_recovers_a_known_rate(self, bpm):
        t, got = ecg.instantaneous_hr(synth_ecg(60, bpm), FS_CHEST)
        assert len(got) > 30
        assert np.median(got) == pytest.approx(bpm, abs=1.0)
        assert np.all(np.diff(t) > 0)

    def test_returns_empty_for_a_flat_trace(self):
        t, bpm = ecg.instantaneous_hr(np.zeros(FS_CHEST * 10), FS_CHEST)
        assert len(t) == 0 and len(bpm) == 0

    def test_returns_empty_for_a_trace_shorter_than_a_second(self):
        assert len(ecg.rpeaks(np.zeros(10), FS_CHEST)) == 0

    def test_rejects_a_sample_rate_below_the_passband(self):
        with pytest.raises(ValueError, match="too low"):
            ecg.rpeaks(np.zeros(1000), 20)

    def test_physiologically_impossible_beats_are_dropped(self):
        """An artefact burst must not contribute a 400 bpm beat to ground truth."""
        sig = synth_ecg(60, 60.0)
        sig[20 * FS_CHEST: 20 * FS_CHEST + 40] += 50.0     # electrode pop
        _, bpm = ecg.instantaneous_hr(sig, FS_CHEST)
        assert bpm.max() <= ecg.HR_MAX_BPM
        assert np.median(bpm) == pytest.approx(60.0, abs=1.5)


# --------------------------------------------------------------------------
# Shared E4 parsing
# --------------------------------------------------------------------------

class TestE4:
    def test_window_defaults_match_dalia_ground_truth(self):
        """8 s / 2 s is DaLiA's own convention; drifting off it silently breaks
        the exact alignment between a window and its reference HR."""
        assert e4.WIN_SECONDS == 8.0 and e4.STRIDE_SECONDS == 2.0
        assert e4.win_samples(HR_FS) == 800
        assert e4.stride_samples(HR_FS) == 200

    def test_acc_is_converted_to_g(self, tmp_path):
        build_wesad(tmp_path, subjects=("S2",), minutes=2)
        data = e4.read_pickle(tmp_path / "S2" / "S2.pkl")
        _, acc = e4.wrist_channels(data, "S2")
        assert np.linalg.norm(acc, axis=1).mean() == pytest.approx(1.0, abs=1e-6)

    def test_to_device_rate_truncates_to_the_shorter_stream(self):
        bvp = np.zeros(64 * 10)                     # 10 s
        acc = np.zeros((32 * 6, 3))                 # 6 s
        out = e4.to_device_rate(bvp, acc, fs=HR_FS)
        assert out.shape[1] == N_HR_CHANNELS
        assert len(out) == pytest.approx(6 * HR_FS, abs=2)

    def test_labels_resample_by_index_not_by_filtering(self):
        """Filtering a categorical track invents class ids that never occurred."""
        labels = np.array([1] * 4 + [7] * 4)
        out = e4.resample_labels(labels, 4, 200, 100)
        assert set(np.unique(out).tolist()) <= {1, 7}

    def test_labels_are_never_extrapolated_past_the_track(self):
        out = e4.resample_labels(np.array([3, 3, 5]), 1, 3, 1)
        assert out.tolist() == [3, 3, 5]

    def test_homogeneous_windows_drops_boundary_spanners(self):
        codes = np.array([1] * 10 + [2] * 10)
        starts, code = e4.homogeneous_windows(codes, win=4, stride=2)
        # No accepted window may contain both conditions.
        for s, c in zip(starts, code):
            assert set(codes[s:s + 4].tolist()) == {c}

    def test_homogeneous_windows_handles_a_too_short_track(self):
        starts, code = e4.homogeneous_windows(np.array([1, 1]), win=8, stride=2)
        assert len(starts) == 0 and len(code) == 0

    def test_reference_beat_mode_averages_within_the_window(self):
        ref_t = np.array([0.5, 1.5, 2.5, 3.5])
        ref_bpm = np.array([60.0, 62.0, 64.0, 66.0])
        got = e4.reference_for_windows(
            np.array([0]), win=400, fs=100, ref_t=ref_t, ref_bpm=ref_bpm, kind="beat"
        )
        assert got[0] == pytest.approx(63.0)

    def test_reference_beat_mode_needs_two_beats(self):
        got = e4.reference_for_windows(
            np.array([0]), win=100, fs=100,
            ref_t=np.array([0.5]), ref_bpm=np.array([60.0]), kind="beat",
        )
        assert np.isnan(got[0])

    def test_reference_window_mode_takes_the_nearest_centre(self):
        ref_t = np.array([4.0, 6.0, 8.0])
        ref_bpm = np.array([70.0, 80.0, 90.0])
        got = e4.reference_for_windows(
            np.array([0, 200]), win=800, fs=100,
            ref_t=ref_t, ref_bpm=ref_bpm, kind="window",
        )
        assert got.tolist() == [70.0, 80.0]

    def test_reference_window_mode_rejects_a_distant_stamp(self):
        got = e4.reference_for_windows(
            np.array([0]), win=800, fs=100,
            ref_t=np.array([99.0]), ref_bpm=np.array([70.0]), kind="window",
        )
        assert np.isnan(got[0])

    def test_reference_rejects_an_unknown_kind(self):
        with pytest.raises(ValueError, match="kind must be"):
            e4.reference_for_windows(
                np.array([0]), 800, 100, np.array([1.0]), np.array([60.0]), "mean"
            )

    def test_read_pickle_rejects_a_non_corpus_file(self, tmp_path):
        p = tmp_path / "junk.pkl"
        with open(p, "wb") as fh:
            pickle.dump([1, 2, 3], fh)
        with pytest.raises(RuntimeError, match="signal"):
            e4.read_pickle(p)


# --------------------------------------------------------------------------
# WESAD
# --------------------------------------------------------------------------

class TestWesad:
    def test_subject_list_omits_the_two_absent_subjects(self):
        assert "S1" not in wesad.SUBJECTS and "S12" not in wesad.SUBJECTS
        assert len(wesad.SUBJECTS) == 15

    def test_loads_and_labels(self, tmp_path):
        build_wesad(tmp_path)
        ws = wesad.load(tmp_path, cache_dir=None)
        assert ws.X.shape[1:] == (800, N_HR_CHANNELS)
        assert ws.fs == HR_FS
        assert set(ws.condition.tolist()) == {
            "baseline", "stress", "amusement", "meditation"
        }

    def test_only_stress_is_positive(self, tmp_path):
        build_wesad(tmp_path)
        ws = wesad.load(tmp_path, cache_dir=None)
        assert set(ws.condition[ws.y_stress == 1].tolist()) == {"stress"}
        # Amusement is a negative on purpose -- elevated HR that is not distress.
        assert "amusement" in set(ws.condition[ws.y_stress == 0].tolist())

    def test_transient_and_ignored_labels_are_dropped(self, tmp_path):
        build_wesad(tmp_path)
        ws = wesad.load(tmp_path, cache_dir=None)
        assert not np.isin(ws.condition, ["0", "transient"]).any()
        assert len(ws) > 0

    def test_subject_ids_are_corpus_prefixed(self, tmp_path):
        build_wesad(tmp_path)
        ws = wesad.load(tmp_path, cache_dir=None)
        assert all(s.startswith("wesad_S") for s in ws.subjects)

    def test_derives_reference_hr_from_chest_ecg(self, tmp_path):
        build_wesad(tmp_path, subjects=("S2",), bpm=66.0)
        ws = wesad.load(tmp_path, cache_dir=None)
        assert ws.has_reference.mean() > 0.9
        assert np.median(ws.hr_ref[ws.has_reference]) == pytest.approx(66.0, abs=1.5)

    def test_reference_is_all_nan_when_disabled(self, tmp_path):
        build_wesad(tmp_path, subjects=("S2",))
        ws = wesad.load(tmp_path, cache_dir=None, with_reference=False)
        assert not ws.has_reference.any()

    def test_missing_subject_is_skipped_not_fatal(self, tmp_path):
        build_wesad(tmp_path, subjects=("S2", "S3"))
        ws = wesad.load(tmp_path, cache_dir=None, subjects=["S2", "S3", "S9"])
        assert len(ws.subjects) == 2

    def test_empty_tree_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no usable WESAD"):
            wesad.load(tmp_path, cache_dir=None)


# --------------------------------------------------------------------------
# PPG-DaLiA
# --------------------------------------------------------------------------

class TestDalia:
    def test_loads_with_activity_conditions(self, tmp_path):
        build_dalia(tmp_path)
        ws = dalia.load(tmp_path, cache_dir=None)
        assert set(ws.condition.tolist()) == {"sitting", "cycling", "walking"}
        assert ws.X.shape[1:] == (800, N_HR_CHANNELS)

    def test_every_window_is_a_negative(self, tmp_path):
        """The assumption that makes DaLiA a false-alarm denominator."""
        build_dalia(tmp_path)
        ws = dalia.load(tmp_path, cache_dir=None)
        assert (ws.y_stress == 0).all()

    def test_reference_hr_comes_from_the_shipped_labels(self, tmp_path):
        build_dalia(tmp_path, subjects=("S1",), bpm=72.0)
        ws = dalia.load(tmp_path, cache_dir=None)
        assert ws.has_reference.mean() > 0.9
        assert np.allclose(ws.hr_ref[ws.has_reference], 72.0)

    def test_subject_ids_are_corpus_prefixed(self, tmp_path):
        build_dalia(tmp_path)
        ws = dalia.load(tmp_path, cache_dir=None)
        assert all(s.startswith("dalia_S") for s in ws.subjects)

    def test_empty_tree_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no usable PPG-DaLiA"):
            dalia.load(tmp_path, cache_dir=None)


# --------------------------------------------------------------------------
# The two corpora together
# --------------------------------------------------------------------------

class TestMerged:
    def test_subject_ids_never_collide_across_corpora(self, tmp_path):
        w, d = tmp_path / "w", tmp_path / "d"
        build_wesad(w, subjects=("S2", "S3"))
        build_dalia(d, subjects=("S2", "S3"))       # deliberately the same ids
        merged = HRWindowSet.concat([
            wesad.load(w, cache_dir=None), dalia.load(d, cache_dir=None)
        ])
        for s in merged.subjects:
            tags = set(merged.dataset[merged.subject == s].tolist())
            assert len(tags) == 1, f"{s} appears in {tags}"


# --------------------------------------------------------------------------
# Distilled cache
# --------------------------------------------------------------------------

class TestCache:
    def test_round_trip(self, tmp_path):
        p = cache.subject_path(tmp_path, "wesad", "S2")
        cache.write(p, {"sig": np.arange(6, dtype=np.float32).reshape(3, 2)})
        got = cache.read(p)
        assert got is not None
        assert np.array_equal(got["sig"], np.arange(6, dtype=np.float32).reshape(3, 2))
        assert "cache_version" not in got

    def test_missing_entry_is_a_miss_not_an_error(self, tmp_path):
        assert cache.read(tmp_path / "nope.npz") is None

    def test_stale_version_is_a_miss(self, tmp_path, monkeypatch):
        p = cache.subject_path(tmp_path, "wesad", "S2")
        cache.write(p, {"sig": np.zeros(3, dtype=np.float32)})
        monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
        assert cache.read(p) is None

    def test_corrupt_entry_is_a_miss_not_an_error(self, tmp_path):
        p = tmp_path / "broken.npz"
        p.write_bytes(b"not an npz at all")
        assert cache.read(p) is None

    def test_write_leaves_no_temp_file(self, tmp_path):
        p = cache.subject_path(tmp_path, "wesad", "S2")
        cache.write(p, {"sig": np.zeros(3, dtype=np.float32)})
        assert not list(p.parent.glob("*.tmp"))

    def test_reserved_key_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="reserved"):
            cache.write(tmp_path / "x.npz", {"cache_version": np.zeros(1)})

    def test_default_root_prefers_the_explicit_argument(self, tmp_path):
        assert cache.default_root(tmp_path) == tmp_path

    def test_default_root_reads_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHAHOSHI_CACHE", str(tmp_path))
        assert cache.default_root() == tmp_path

    def test_default_root_is_none_without_a_durable_location(self, monkeypatch):
        monkeypatch.delenv("SHAHOSHI_CACHE", raising=False)
        monkeypatch.setattr(cache, "COLAB_DRIVE", type(cache.COLAB_DRIVE)("/nope/nope"))
        assert cache.default_root() is None

    def test_a_cached_run_matches_an_uncached_one(self, tmp_path):
        """The whole point: the cache must be an optimization, not a variant."""
        tree = tmp_path / "tree"
        build_wesad(tree, subjects=("S2",))
        cold = wesad.load(tree, cache_dir=None)
        wesad.load(tree, cache_dir=tmp_path / "cache")       # populates
        warm = wesad.load(tree, cache_dir=tmp_path / "cache")
        assert np.array_equal(cold.X, warm.X)
        assert np.array_equal(cold.condition, warm.condition)
        assert np.allclose(cold.hr_ref, warm.hr_ref, equal_nan=True)


# --------------------------------------------------------------------------
# Availability gate
# --------------------------------------------------------------------------

class TestAvailability:
    def test_clean_data_passes(self):
        ws = make_hrws(n=400)
        assert [c for c in availability.gate(ws) if not c.ok and c.fatal] == []
        availability.require(ws)

    def test_zero_positives_is_fatal(self):
        """Defect F3: a class with no samples that trained and exported anyway."""
        ws = make_hrws(n=400, stress_frac=0.0)
        with pytest.raises(RuntimeError, match="positives exist"):
            availability.require(ws)

    def test_positives_confined_to_one_subject_is_fatal(self):
        ws = make_hrws(n=400, subjects=("only",))
        with pytest.raises(RuntimeError, match="positives span subjects"):
            availability.require(ws)

    def test_unscaled_accelerometer_is_fatal(self):
        """The 1/64 g error: loads, windows, trains, and is wrong by 64x."""
        ws = make_hrws(n=400)
        ws.X[:, :, 3] = 64.0
        with pytest.raises(RuntimeError, match="acc scaling"):
            availability.require(ws)

    def test_flat_bvp_is_fatal(self):
        ws = make_hrws(n=400)
        ws.X[:, :, 0] = 0.0
        with pytest.raises(RuntimeError, match="bvp varies"):
            availability.require(ws)

    def test_thin_reference_coverage_warns_but_does_not_raise(self):
        ws = make_hrws(n=400)
        ws.hr_ref[:300] = np.nan
        availability.require(ws)
        names = [c.name for c in availability.gate(ws) if not c.ok]
        assert "hr_ref coverage" in names

    def test_report_names_every_condition_and_ends_in_a_verdict(self):
        text = availability.report(make_hrws(n=400))
        assert "stress" in text and "baseline" in text
        assert "VERDICT" in text

    def test_report_says_so_when_a_fatal_check_fails(self):
        text = availability.report(make_hrws(n=400, stress_frac=0.0))
        assert "fatal" in text and "do not model" in text

    def test_require_returns_the_input_for_chaining(self):
        ws = make_hrws(n=400)
        assert availability.require(ws) is ws


# --------------------------------------------------------------------------
# Corpus resolution
# --------------------------------------------------------------------------

class TestCorpusResolution:
    def test_each_loader_declares_its_extraction_subdir(self):
        for mod in (wesad, dalia):
            assert isinstance(mod.SUBDIR, str) and mod.SUBDIR

    def test_wesad_and_dalia_share_an_ambiguous_marker(self):
        """The premise of the test below. If these ever stop colliding the
        scoping is still correct, but this test is no longer testing it."""
        assert wesad.MARKER.split("/")[-1].startswith("S")
        assert dalia.MARKER.split("/")[-1].startswith("S")

    def test_side_by_side_corpora_resolve_to_their_own_trees(self, tmp_path):
        """Both corpora lay out subjects as SN/SN.pkl and normally extract side
        by side under one data root. An unscoped search for WESAD's marker finds
        the DaLiA tree, loads it without error, and returns the wrong corpus."""
        from shahoshi.datasets import _expect_extracted

        build_dalia(tmp_path / dalia.SUBDIR, subjects=("S1", "S2", "S3"))
        build_wesad(tmp_path / wesad.SUBDIR, subjects=("S2", "S3", "S4"))

        assert _expect_extracted(tmp_path, dalia).samefile(tmp_path / dalia.SUBDIR)
        assert _expect_extracted(tmp_path, wesad).samefile(tmp_path / wesad.SUBDIR)

    def test_load_hr_keeps_the_two_corpora_apart(self, tmp_path):
        build_dalia(tmp_path / dalia.SUBDIR, subjects=("S1", "S2", "S3"))
        build_wesad(tmp_path / wesad.SUBDIR, subjects=("S2", "S3", "S4"))

        hr = datasets.load_hr(
            tmp_path, tags=["dalia", "wesad"], cache_dir=None, download=False
        )
        by_tag = {
            t: set(hr.condition[hr.dataset == t].tolist()) for t in hr.tags
        }
        assert by_tag["wesad"] == {"baseline", "stress", "amusement", "meditation"}
        assert by_tag["dalia"] == {"sitting", "cycling", "walking"}
        assert (hr.y_stress[hr.dataset == "dalia"] == 0).all()
