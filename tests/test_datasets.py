"""Loader tests against synthetic corpus trees.

These build the on-disk layout each corpus actually ships and assert the loader
produces a correct WindowSet. That keeps the parsing logic under test without a
multi-GB download, which matters because the parsing is where the silent
mistakes live: a transposed channel order or an off-by-one subject id produces
plausible-looking arrays and a quietly wrong model.
"""

import numpy as np
import pandas as pd
import pytest

from shahoshi.datasets import motionsense, uci
from shahoshi.datasets.base import CLS, N_CHANNELS, UNKNOWN
from shahoshi.datasets.download import find_root


# --------------------------------------------------------------------------
# UCI HAR
# --------------------------------------------------------------------------

def build_uci(root, n_train=6, n_test=4, win=128):
    """Write a minimal UCI HAR tree with distinguishable per-channel values."""
    rng = np.random.default_rng(0)
    layout = {"train": n_train, "test": n_test}
    # Include label 6 (LAYING) so we can assert it gets dropped -- but give
    # every subject at least one surviving window, so a dropped class does not
    # silently remove a whole person from the fixture.
    labels = {"train": [1, 2, 3, 4, 5, 6], "test": [1, 4, 6, 5]}
    subjects = {"train": [1, 1, 2, 2, 3, 3], "test": [11, 11, 12, 12]}

    for split, n in layout.items():
        sig_dir = root / split / "Inertial Signals"
        sig_dir.mkdir(parents=True, exist_ok=True)
        # Channel c gets a constant offset of c so channel order is checkable.
        for ci, (sig, ax) in enumerate(
            [(s, a) for s in ("body_acc", "body_gyro") for a in ("x", "y", "z")]
        ):
            arr = np.full((n, win), float(ci)) + rng.normal(0, 1e-3, (n, win))
            np.savetxt(sig_dir / f"{sig}_{ax}_{split}.txt", arr)
        np.savetxt(root / split / f"y_{split}.txt", labels[split][:n], fmt="%d")
        np.savetxt(root / split / f"subject_{split}.txt", subjects[split][:n], fmt="%d")
    return root


class TestUciLoader:
    def test_loads_and_drops_laying(self, tmp_path):
        ws = uci.load(build_uci(tmp_path / "uci"))
        # 6 train + 4 test = 10 rows, of which 2 are LAYING (one per split)
        assert len(ws) == 8
        assert not np.isin(ws.y_act, [-1]).any()
        assert set(np.unique(ws.y_act).tolist()) <= set(CLS.values())

    def test_channel_order_is_preserved(self, tmp_path):
        """Channel c was written as a constant c; assert it lands at index c."""
        ws = uci.load(build_uci(tmp_path / "uci"))
        means = ws.X.reshape(-1, N_CHANNELS).mean(0)
        assert np.allclose(means, np.arange(N_CHANNELS), atol=1e-2)

    def test_shape_and_dtype(self, tmp_path):
        ws = uci.load(build_uci(tmp_path / "uci"))
        assert ws.X.shape[1:] == (128, N_CHANNELS)
        assert ws.X.dtype == np.float32
        assert ws.win == 128

    def test_subject_ids_are_prefixed_and_zero_padded(self, tmp_path):
        ws = uci.load(build_uci(tmp_path / "uci"))
        assert all(s.startswith("uci_") for s in ws.subject)
        assert "uci_01" in set(ws.subject.tolist())
        assert "uci_11" in set(ws.subject.tolist())

    def test_train_and_test_subjects_both_present(self, tmp_path):
        """The loader must merge both official splits; we re-split by subject
        ourselves, so discarding UCI's split would throw away data."""
        subs = set(uci.load(build_uci(tmp_path / "uci")).subject.tolist())
        assert {"uci_01", "uci_02", "uci_03"} <= subs   # from train
        assert {"uci_11", "uci_12"} <= subs             # from test

    def test_fall_labels_are_unknown_not_zero(self, tmp_path):
        """UCI has no falls, but also no evidence a window is a confirmed
        non-fall. Marking them 0 would teach the fall head that ordinary
        walking is verified fall-free."""
        ws = uci.load(build_uci(tmp_path / "uci"))
        assert (ws.y_fall == UNKNOWN).all()

    def test_dataset_tag(self, tmp_path):
        assert set(uci.load(build_uci(tmp_path / "uci")).dataset.tolist()) == {"uci"}

    def test_rejects_a_directory_that_is_not_uci(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="does not look like UCI"):
            uci.load(tmp_path / "empty")

    def test_reports_missing_channel_file(self, tmp_path):
        root = build_uci(tmp_path / "uci")
        (root / "train" / "Inertial Signals" / "body_gyro_z_train.txt").unlink()
        with pytest.raises(FileNotFoundError, match="missing channel file"):
            uci.load(root)

    def test_detects_label_count_mismatch(self, tmp_path):
        root = build_uci(tmp_path / "uci")
        np.savetxt(root / "train" / "y_train.txt", [1, 2], fmt="%d")
        with pytest.raises(ValueError, match="labels"):
            uci.load(root)


# --------------------------------------------------------------------------
# MotionSense
# --------------------------------------------------------------------------

def build_motionsense(root, n_samples=400, subjects=(1, 2)):
    """Write a minimal MotionSense tree, including the decoy columns the real
    CSVs carry so that reading by name rather than position is exercised."""
    rng = np.random.default_rng(1)
    for activity in ("wlk", "jog", "sit"):
        folder = root / f"{activity}_15"
        folder.mkdir(parents=True, exist_ok=True)
        for sub in subjects:
            df = pd.DataFrame(
                {
                    # decoys first, deliberately, so positional reads break
                    "Unnamed: 0": np.arange(n_samples),
                    "attitude.roll": rng.normal(0, 1, n_samples),
                    "gravity.x": rng.normal(0, 1, n_samples),
                    "userAcceleration.x": np.full(n_samples, 0.0),
                    "userAcceleration.y": np.full(n_samples, 1.0),
                    "userAcceleration.z": np.full(n_samples, 2.0),
                    "rotationRate.x": np.full(n_samples, 3.0),
                    "rotationRate.y": np.full(n_samples, 4.0),
                    "rotationRate.z": np.full(n_samples, 5.0),
                }
            )
            df.to_csv(folder / f"sub_{sub}.csv", index=False)
    return root


class TestMotionSenseLoader:
    def test_windows_continuous_trials(self, tmp_path):
        ws = motionsense.load(build_motionsense(tmp_path / "ms"), win=128, stride=64)
        # (400 - 128)//64 + 1 = 5 windows per csv, 3 activities x 2 subjects
        assert len(ws) == 5 * 3 * 2
        assert ws.X.shape[1:] == (128, N_CHANNELS)

    def test_reads_channels_by_name_not_position(self, tmp_path):
        """The synthetic CSVs put decoy columns first; channel c holds value c."""
        ws = motionsense.load(build_motionsense(tmp_path / "ms"))
        means = ws.X.reshape(-1, N_CHANNELS).mean(0)
        assert np.allclose(means, np.arange(N_CHANNELS), atol=1e-5)

    def test_label_comes_from_folder_name(self, tmp_path):
        ws = motionsense.load(build_motionsense(tmp_path / "ms"))
        assert set(np.unique(ws.y_act).tolist()) == {CLS["walk"], CLS["jog"], CLS["sit"]}

    def test_subject_ids_are_prefixed(self, tmp_path):
        ws = motionsense.load(build_motionsense(tmp_path / "ms"))
        assert set(ws.subject.tolist()) == {"ms_sub_1", "ms_sub_2"}

    def test_windows_of_one_subject_share_one_id(self, tmp_path):
        """Subject ids must be per-person, not per-trial, or a subject-disjoint
        split silently leaks the same person across train and test."""
        ws = motionsense.load(build_motionsense(tmp_path / "ms"))
        assert len(set(ws.subject.tolist())) == 2

    def test_fall_labels_are_unknown(self, tmp_path):
        ws = motionsense.load(build_motionsense(tmp_path / "ms"))
        assert (ws.y_fall == UNKNOWN).all()

    def test_skips_trials_shorter_than_a_window(self, tmp_path):
        root = build_motionsense(tmp_path / "ms", n_samples=400)
        short = root / "wlk_15" / "sub_9.csv"
        pd.read_csv(root / "wlk_15" / "sub_1.csv").head(10).to_csv(short, index=False)
        ws = motionsense.load(root, win=128, stride=64)
        assert "ms_sub_9" not in set(ws.subject.tolist())

    def test_skips_csv_missing_required_columns(self, tmp_path):
        root = build_motionsense(tmp_path / "ms")
        bad = root / "wlk_15" / "sub_8.csv"
        pd.DataFrame({"nope": np.arange(400)}).to_csv(bad, index=False)
        ws = motionsense.load(root)  # must not raise
        assert "ms_sub_8" not in set(ws.subject.tolist())

    def test_ignores_unknown_activity_folders(self, tmp_path):
        root = build_motionsense(tmp_path / "ms")
        (root / "xyz_99").mkdir()
        ws = motionsense.load(root)
        assert len(ws) == 5 * 3 * 2

    def test_raises_when_no_trials_found(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no trial folders"):
            motionsense.load(tmp_path / "empty")

    def test_stride_controls_overlap(self, tmp_path):
        root = build_motionsense(tmp_path / "ms", subjects=(1,))
        few = motionsense.load(root, win=128, stride=128)
        many = motionsense.load(root, win=128, stride=32)
        assert len(many) > len(few)


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------

class TestMerge:
    def test_corpora_concatenate_and_stay_distinguishable(self, tmp_path):
        from shahoshi.datasets.base import WindowSet

        a = uci.load(build_uci(tmp_path / "uci"))
        b = motionsense.load(build_motionsense(tmp_path / "ms"))
        merged = WindowSet.concat([a, b])

        assert len(merged) == len(a) + len(b)
        assert set(merged.dataset.tolist()) == {"uci", "ms"}

    def test_subject_ids_cannot_collide_across_corpora(self, tmp_path):
        """Both corpora number subjects from 1. Without the corpus prefix,
        'uci_1' and MotionSense's subject 1 would merge into one person and a
        subject-disjoint split would leak across train and test."""
        a = uci.load(build_uci(tmp_path / "uci"))
        b = motionsense.load(build_motionsense(tmp_path / "ms"))
        assert not (set(a.subject.tolist()) & set(b.subject.tolist()))

    def test_summary_mentions_both_corpora(self, tmp_path):
        from shahoshi.datasets.base import WindowSet

        a = uci.load(build_uci(tmp_path / "uci"))
        b = motionsense.load(build_motionsense(tmp_path / "ms"))
        text = WindowSet.concat([a, b]).summary()
        assert "uci" in text and "ms" in text


# --------------------------------------------------------------------------
# download helpers
# --------------------------------------------------------------------------

class TestFindRoot:
    def test_finds_marker_at_the_top(self, tmp_path):
        (tmp_path / "train").mkdir()
        (tmp_path / "train" / "y_train.txt").touch()
        assert find_root(tmp_path, "train/y_train.txt") == tmp_path

    def test_finds_marker_nested_one_level(self, tmp_path):
        """Mirrors of the same corpus disagree about whether the zip has a
        top-level folder, so the root must be found rather than hardcoded."""
        inner = tmp_path / "UCI HAR Dataset" / "train"
        inner.mkdir(parents=True)
        (inner / "y_train.txt").touch()
        assert find_root(tmp_path, "train/y_train.txt") == tmp_path / "UCI HAR Dataset"

    def test_finds_marker_nested_deeply(self, tmp_path):
        inner = tmp_path / "a" / "b" / "c" / "train"
        inner.mkdir(parents=True)
        (inner / "y_train.txt").touch()
        assert find_root(tmp_path, "train/y_train.txt") == tmp_path / "a" / "b" / "c"

    def test_raises_with_a_useful_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="could not find"):
            find_root(tmp_path, "train/y_train.txt")
