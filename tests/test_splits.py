"""Tests for evaluation splits.

The leak tests are the point. A subject appearing in both train and test is the
single most common way HAR results get quietly inflated, and it is invisible in
the metrics -- accuracy just looks better than it is.
"""

import numpy as np
import pytest

from shahoshi.splits import (
    assert_disjoint,
    describe,
    leave_dataset_out,
    loso_folds,
    subject_split,
)


def make_subjects(n_subjects=20, per_subject=10, prefix="uci"):
    return np.array([f"{prefix}_{i}" for i in range(n_subjects) for _ in range(per_subject)])


class TestSubjectSplit:
    def test_splits_are_subject_disjoint(self):
        subjects = make_subjects(20, 10)
        masks, groups = subject_split(subjects, seed=0)
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not (set(groups[a].tolist()) & set(groups[b].tolist()))

    def test_no_window_level_leak(self):
        """Stronger check: no subject id appears in two window-mask selections."""
        subjects = make_subjects(20, 10)
        masks, _ = subject_split(subjects, seed=0)
        tr = set(subjects[masks["train"]].tolist())
        te = set(subjects[masks["test"]].tolist())
        va = set(subjects[masks["val"]].tolist())
        assert not (tr & te) and not (tr & va) and not (va & te)

    def test_masks_partition_every_window(self):
        subjects = make_subjects(20, 10)
        masks, _ = subject_split(subjects, seed=0)
        total = masks["train"].astype(int) + masks["val"].astype(int) + masks["test"].astype(int)
        assert (total == 1).all()

    def test_all_splits_non_empty(self):
        for seed in range(10):
            masks, _ = subject_split(make_subjects(20, 10), seed=seed)
            for k in ("train", "val", "test"):
                assert masks[k].sum() > 0, f"seed {seed} produced an empty {k} split"

    def test_deterministic_for_a_seed(self):
        s = make_subjects(20, 10)
        a, _ = subject_split(s, seed=7)
        b, _ = subject_split(s, seed=7)
        assert all(np.array_equal(a[k], b[k]) for k in a)

    def test_different_seeds_give_different_splits(self):
        s = make_subjects(20, 10)
        a, _ = subject_split(s, seed=1)
        b, _ = subject_split(s, seed=2)
        assert not np.array_equal(a["test"], b["test"])

    def test_survives_minimum_subject_count(self):
        masks, groups = subject_split(make_subjects(3, 5), seed=0)
        for k in ("train", "val", "test"):
            assert masks[k].sum() > 0

    def test_rejects_too_few_subjects(self):
        with pytest.raises(ValueError):
            subject_split(make_subjects(2, 5))

    def test_rejects_fracs_that_do_not_sum_to_one(self):
        with pytest.raises(ValueError):
            subject_split(make_subjects(20, 10), fracs=(0.5, 0.2, 0.2))

    def test_respects_requested_proportions(self):
        subjects = make_subjects(100, 2)
        _, groups = subject_split(subjects, seed=3, fracs=(0.70, 0.15, 0.15))
        assert len(groups["train"]) == pytest.approx(70, abs=2)
        assert len(groups["val"]) == pytest.approx(15, abs=2)
        assert len(groups["test"]) == pytest.approx(15, abs=2)


class TestLosoFolds:
    def test_one_fold_per_subject(self):
        subjects = make_subjects(6, 4)
        folds = list(loso_folds(subjects))
        assert len(folds) == 6

    def test_each_fold_holds_out_exactly_one_subject(self):
        subjects = make_subjects(6, 4)
        for held, train, test in loso_folds(subjects):
            assert set(subjects[test].tolist()) == {held}
            assert held not in set(subjects[train].tolist())
            assert (train | test).all()
            assert not (train & test).any()

    def test_every_subject_is_held_out_once(self):
        subjects = make_subjects(6, 4)
        assert sorted(h for h, _, _ in loso_folds(subjects)) == sorted(set(subjects.tolist()))


class TestLeaveDatasetOut:
    def setup_method(self):
        self.datasets = np.array(["uci"] * 30 + ["ms"] * 30 + ["sisfall"] * 30)
        self.subjects = np.array(
            [f"uci_{i // 3}" for i in range(30)]
            + [f"ms_{i // 3}" for i in range(30)]
            + [f"sisfall_{i // 3}" for i in range(30)]
        )

    def test_holds_out_the_named_corpus_entirely(self):
        masks = leave_dataset_out(self.datasets, "sisfall")
        assert set(self.datasets[masks["test"]].tolist()) == {"sisfall"}
        assert "sisfall" not in set(self.datasets[masks["train"]].tolist())

    def test_accepts_a_list_of_corpora(self):
        masks = leave_dataset_out(self.datasets, ["ms", "sisfall"])
        assert set(self.datasets[masks["test"]].tolist()) == {"ms", "sisfall"}
        assert set(self.datasets[masks["train"]].tolist()) == {"uci"}

    def test_val_split_is_subject_disjoint_from_train(self):
        masks = leave_dataset_out(self.datasets, "sisfall", subjects=self.subjects)
        tr = set(self.subjects[masks["train"]].tolist())
        va = set(self.subjects[masks["val"]].tolist())
        assert va and not (tr & va)

    def test_val_split_never_contains_the_held_out_corpus(self):
        masks = leave_dataset_out(self.datasets, "sisfall", subjects=self.subjects)
        assert "sisfall" not in set(self.datasets[masks["val"]].tolist())

    def test_masks_partition_when_val_requested(self):
        masks = leave_dataset_out(self.datasets, "sisfall", subjects=self.subjects)
        total = sum(masks[k].astype(int) for k in masks)
        assert (total == 1).all()

    def test_no_val_split_without_subjects(self):
        masks = leave_dataset_out(self.datasets, "sisfall")
        assert set(masks) == {"train", "test"}

    def test_rejects_unknown_corpus(self):
        with pytest.raises(ValueError, match="not present"):
            leave_dataset_out(self.datasets, "wesad")

    def test_rejects_holding_out_everything(self):
        with pytest.raises(ValueError, match="no training data"):
            leave_dataset_out(self.datasets, ["uci", "ms", "sisfall"])


class TestAssertDisjoint:
    def test_passes_on_clean_groups(self):
        assert_disjoint({"a": np.array(["s1", "s2"]), "b": np.array(["s3"])})

    def test_catches_a_leak_and_names_the_subject(self):
        with pytest.raises(AssertionError, match="s2"):
            assert_disjoint({"a": np.array(["s1", "s2"]), "b": np.array(["s2", "s3"])})


class TestDescribe:
    def test_reports_subject_and_window_counts(self):
        subjects = make_subjects(20, 10)
        masks, _ = subject_split(subjects, seed=0)
        text = describe(masks, subjects)
        assert "train" in text and "val" in text and "test" in text
        assert "subjects" in text and "windows" in text

    def test_handles_a_two_way_split(self):
        datasets = np.array(["uci"] * 10 + ["sisfall"] * 10)
        masks = leave_dataset_out(datasets, "sisfall")
        text = describe(masks, np.array([f"s{i}" for i in range(20)]))
        assert "train" in text and "test" in text and "val" not in text
