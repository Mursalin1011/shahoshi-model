"""Evaluation splits. Subject-disjoint always; never window-level.

Windows overlap by 50%, so a random window-level split puts near-duplicate
windows in both train and test, inflating accuracy by 5-15 points and producing
a model that collapses on a new wearer. Every function here splits on *people*.

Three protocols, and Stage 1 reports all three because they answer different
questions:

`subject_split`
    Merged, subject-disjoint. "How well does this work on a new person, given
    training data that looks like theirs?"

`loso_folds`
    Leave-one-subject-out. The low-variance version of the above, for the small
    per-corpus subject counts where a single random split is noisy.

`leave_dataset_out`
    Train on some corpora, test on another entirely. "How much of the accuracy
    is the model learning activity, and how much is it learning one corpus's
    sensor placement?" This is the honest measurement of the domain gap between
    the waist- and pocket-mounted training data and our wrist-worn device, and
    it is the number that will be lowest and most worth reporting.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

SplitMasks = dict[str, np.ndarray]


def subject_split(
    subjects: np.ndarray,
    seed: int = 42,
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[SplitMasks, dict[str, np.ndarray]]:
    """Partition subjects into train/val/test, then map back to window masks.

    Returns
    -------
    masks : {"train"|"val"|"test" -> (n,) bool}
    groups : {"train"|"val"|"test" -> array of subject ids}
    """
    if len(fracs) != 3 or not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"fracs must be three values summing to 1; got {fracs}")

    uniq = np.array(sorted(set(np.asarray(subjects).tolist())))
    if len(uniq) < 3:
        raise ValueError(f"need at least 3 subjects to make 3 splits; got {len(uniq)}")

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_tr = max(1, round(fracs[0] * n))
    n_va = max(1, round(fracs[1] * n))
    # Guarantee a non-empty test split even when rounding is unkind.
    n_tr = min(n_tr, n - 2)
    n_va = min(n_va, n - n_tr - 1)

    groups = {
        "train": uniq[:n_tr],
        "val": uniq[n_tr : n_tr + n_va],
        "test": uniq[n_tr + n_va :],
    }
    masks = {k: np.isin(subjects, v) for k, v in groups.items()}
    assert_disjoint(groups)
    _assert_partitions(masks, len(subjects))
    return masks, groups


def loso_folds(subjects: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_subject, train_mask, test_mask) for each subject in turn."""
    subjects = np.asarray(subjects)
    for s in sorted(set(subjects.tolist())):
        test = subjects == s
        yield s, ~test, test


def leave_dataset_out(
    datasets: np.ndarray,
    test_dataset: str | list[str],
    subjects: np.ndarray | None = None,
    val_frac: float = 0.15,
    seed: int = 42,
) -> SplitMasks:
    """Hold out entire corpora for test; optionally carve a val split from the rest.

    Parameters
    ----------
    datasets : (n,) array of corpus tags
    test_dataset : str or list of str
        Corpus tag(s) to hold out entirely.
    subjects : (n,) array, optional
        If given, the validation split is carved out subject-disjointly from the
        remaining corpora. If omitted, no val split is produced.

    Returns
    -------
    {"train", "test"} or {"train", "val", "test"} -> (n,) bool masks
    """
    datasets = np.asarray(datasets)
    held = [test_dataset] if isinstance(test_dataset, str) else list(test_dataset)

    known = set(datasets.tolist())
    missing = [d for d in held if d not in known]
    if missing:
        raise ValueError(f"dataset(s) {missing} not present; have {sorted(known)}")

    test = np.isin(datasets, held)
    rest = ~test
    if not rest.any():
        raise ValueError("holding out those corpora leaves no training data")

    if subjects is None:
        return {"train": rest, "test": test}

    subjects = np.asarray(subjects)
    pool = np.array(sorted(set(subjects[rest].tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    n_va = max(1, round(val_frac * len(pool)))
    n_va = min(n_va, len(pool) - 1)  # never consume the whole training pool
    val_subjects = pool[:n_va]

    val = rest & np.isin(subjects, val_subjects)
    train = rest & ~val
    masks = {"train": train, "val": val, "test": test}
    _assert_partitions(masks, len(datasets))
    return masks


def assert_disjoint(groups: dict[str, np.ndarray]) -> None:
    """Fail loudly if any subject appears in more than one split."""
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(groups[a].tolist()) & set(groups[b].tolist())
            if shared:
                raise AssertionError(
                    f"subject leak between {a!r} and {b!r}: {sorted(shared)}"
                )


def _assert_partitions(masks: SplitMasks, n: int) -> None:
    """Every window belongs to exactly one split."""
    stacked = np.stack([masks[k] for k in masks])
    counts = stacked.sum(0)
    if not (counts == 1).all():
        raise AssertionError(
            f"splits do not partition the data: {int((counts == 0).sum())} windows "
            f"in no split, {int((counts > 1).sum())} in more than one"
        )
    if stacked.sum() != n:
        raise AssertionError(f"splits cover {int(stacked.sum())} of {n} windows")


def describe(masks: SplitMasks, subjects: np.ndarray) -> str:
    """Human-readable split sizes, for the notebook and the run manifest."""
    subjects = np.asarray(subjects)
    rows = []
    for k in ("train", "val", "test"):
        if k not in masks:
            continue
        m = masks[k]
        rows.append(
            f"{k:<5s} {len(set(subjects[m].tolist())):>4d} subjects  "
            f"{int(m.sum()):>7,} windows"
        )
    return "\n".join(rows)
