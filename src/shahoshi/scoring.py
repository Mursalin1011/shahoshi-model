"""Novelty scoring and false-alarm calibration.

Two scores, both computable on device from tensors the model already produces:

**Predictive entropy** -- a window unlike anything in training produces a flat
softmax. Free: it is a function of the output the classifier already computed.

**Mahalanobis distance** -- distance of the embedding from the training
embedding distribution. Sharper than entropy, and costs one 64x64 matrix. Export
the Cholesky factor rather than the inverse covariance so the device computes a
triangular solve instead of a full quadratic form.

Why the calibration functions matter more than the scores
---------------------------------------------------------
Thresholding at "the 99th percentile of normal data" sounds like a 1% false
alarm budget, and reads as strict. At a 1.28 s window hop that is 2812 windows
per hour, so 1% is **28 false alarms per hour** -- roughly one every two
minutes. A wearer switches that device off inside a day, and a device that is
switched off has zero recall no matter what the confusion matrix says.

So the useful operating point is expressed in alarms per hour, not in
percentiles, and `recall_at_far` is the function that should generate the
headline number for this project. `far_per_hour` exists to make the translation
impossible to forget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECONDS_PER_HOUR = 3600.0


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalized predictive entropy in [0, 1], one value per window.

    Normalized by log(n_classes) so the number is comparable across models with
    different class counts -- which matters here, because dropping `lay` changed
    the class count and any absolute entropy threshold from before is stale.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"expected (n, n_classes); got {probs.shape}")
    if probs.shape[1] < 2:
        raise ValueError("entropy needs at least 2 classes")

    p = np.clip(probs, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1) / np.log(probs.shape[1])


@dataclass
class MahalanobisScorer:
    """Mahalanobis distance from a fitted training embedding distribution."""

    mean: np.ndarray
    precision: np.ndarray
    cholesky: np.ndarray

    @classmethod
    def fit(cls, embeddings: np.ndarray, shrinkage: float = 1e-3) -> MahalanobisScorer:
        """Fit on *training* embeddings only.

        `shrinkage` adds a ridge to the covariance diagonal. It is not
        cosmetic: a 64-dimensional embedding from a ReLU layer routinely has
        near-dead dimensions, whose near-zero variance makes the covariance
        ill-conditioned and lets the inverse amplify numerical noise into
        enormous distances for perfectly ordinary windows.
        """
        E = np.asarray(embeddings, dtype=np.float64)
        if E.ndim != 2:
            raise ValueError(f"expected (n, dim); got {E.shape}")
        if len(E) <= E.shape[1]:
            raise ValueError(
                f"{len(E)} samples for a {E.shape[1]}-dimensional embedding: the "
                f"covariance would be singular. Fit on the full training split."
            )

        mean = E.mean(axis=0)
        cov = np.cov(E, rowvar=False) + np.eye(E.shape[1]) * shrinkage
        precision = np.linalg.inv(cov)
        # Lower-triangular L with cov = L L^T, for the device-side solve.
        cholesky = np.linalg.cholesky(cov)
        return cls(mean=mean, precision=precision, cholesky=cholesky)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Mahalanobis distance (not squared) for each embedding."""
        E = np.asarray(embeddings, dtype=np.float64)
        if E.ndim != 2 or E.shape[1] != len(self.mean):
            raise ValueError(
                f"expected (n, {len(self.mean)}); got {E.shape}"
            )
        d = E - self.mean
        # Clip at zero: the quadratic form is mathematically non-negative but
        # can go very slightly negative through floating-point cancellation.
        sq = np.einsum("ij,jk,ik->i", d, self.precision, d)
        return np.sqrt(np.clip(sq, 0.0, None))


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def windows_per_hour(hop_seconds: float) -> float:
    """How many inference windows the device evaluates per hour."""
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")
    return SECONDS_PER_HOUR / hop_seconds


def far_per_hour(flag_rate: float, hop_seconds: float) -> float:
    """Convert a per-window false-positive *rate* into alarms per hour.

    The translation everyone forgets. At a 1.28 s hop, a 1% per-window false
    positive rate is 28 alarms per hour.
    """
    if not 0.0 <= flag_rate <= 1.0:
        raise ValueError(f"flag_rate must be a fraction in [0, 1]; got {flag_rate}")
    return flag_rate * windows_per_hour(hop_seconds)


def flag_rate_for_far(target_far: float, hop_seconds: float) -> float:
    """Inverse of `far_per_hour`: the per-window rate a FAR budget allows."""
    if target_far < 0:
        raise ValueError("target_far must be non-negative")
    return min(1.0, target_far / windows_per_hour(hop_seconds))


def threshold_at_percentile(scores: np.ndarray, percentile: float) -> float:
    """Threshold placed at a percentile of *normal* scores.

    Kept for continuity with the pre-refactor baseline, but prefer
    `threshold_for_far`: a percentile hides the alarm rate it implies.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100]; got {percentile}")
    return float(np.percentile(np.asarray(scores, dtype=np.float64), percentile))


def threshold_for_far(
    normal_scores: np.ndarray, target_far: float, hop_seconds: float
) -> float:
    """Highest threshold whose false-alarm rate on normal data meets the budget.

    Returns
    -------
    float
        Score threshold; flag a window when its score is strictly greater.
    """
    rate = flag_rate_for_far(target_far, hop_seconds)
    return threshold_at_percentile(normal_scores, 100.0 * (1.0 - rate))


def recall_at_far(
    positive_scores: np.ndarray,
    normal_scores: np.ndarray,
    target_far: float,
    hop_seconds: float,
) -> dict[str, float]:
    """Recall on real events at a fixed false-alarm-per-hour budget.

    This is the metric that should carry the project's headline claim, in place
    of accuracy. Accuracy on a merged dataset says nothing useful when real
    events are a fraction of a percent of windows, and a percentile threshold
    says nothing about how often the wearer is interrupted.

    Returns
    -------
    dict with `threshold`, `recall`, `achieved_far` (alarms/hour actually
    produced on the normal data), and `target_far`.
    """
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(normal_scores, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("need both positive and normal scores")

    thr = threshold_for_far(neg, target_far, hop_seconds)
    return {
        "threshold": float(thr),
        "recall": float((pos > thr).mean()),
        "achieved_far": far_per_hour(float((neg > thr).mean()), hop_seconds),
        "target_far": float(target_far),
    }


# ---------------------------------------------------------------------------
# threshold-free summaries
# ---------------------------------------------------------------------------

def auroc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney rank statistic.

    Handles ties correctly (a tie contributes 0.5), which matters for int8
    outputs where the score is heavily quantized and ties are common.
    """
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("need both positive and negative scores")

    alls = np.concatenate([pos, neg])
    order = alls.argsort(kind="mergesort")
    ranks = np.empty(len(alls), dtype=np.float64)
    ranks[order] = np.arange(1, len(alls) + 1)

    # Average ranks within tied groups so ties score 0.5.
    srt = alls[order]
    start = 0
    for i in range(1, len(srt) + 1):
        if i == len(srt) or srt[i] != srt[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auprc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Average precision -- area under the precision-recall curve.

    Reported alongside AUROC because AUROC is optimistic under the extreme class
    imbalance of real deployment: falls and assaults are a fraction of a percent
    of windows, and AUROC's false-positive axis is normalized by the enormous
    negative count, which flatters a model that would still alarm constantly.
    """
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("need both positive and negative scores")

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    order = scores.argsort(kind="mergesort")[::-1]
    labels = labels[order]

    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / len(pos)

    # Step-wise integral: sum precision * delta-recall (average precision).
    d_recall = np.diff(recall, prepend=0.0)
    return float((precision * d_recall).sum())


def score_report(
    positive_scores: np.ndarray,
    normal_scores: np.ndarray,
    hop_seconds: float,
    fars: tuple[float, ...] = (0.5, 1.0, 2.0, 6.0),
    label: str = "score",
) -> str:
    """A text block summarizing one score's separation of events from normal data."""
    lines = [
        f"{label}:",
        f"  AUROC {auroc(positive_scores, normal_scores):.4f}"
        f"   AUPRC {auprc(positive_scores, normal_scores):.4f}"
        f"   ({len(positive_scores):,} events vs {len(normal_scores):,} normal)",
    ]
    for far in fars:
        r = recall_at_far(positive_scores, normal_scores, far, hop_seconds)
        lines.append(
            f"  recall {r['recall']:.3f} at {far:>4.1f} false alarms/hour"
            f"   (threshold {r['threshold']:.4f}, achieved {r['achieved_far']:.2f}/h)"
        )
    return "\n".join(lines)
