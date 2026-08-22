"""Tests for novelty scoring and false-alarm calibration.

The calibration tests exist to keep one number honest. "99th percentile of
normal data" reads as a strict 1% budget but at a 1.28 s hop it is 28 alarms per
hour, and a wearer switches that off. `test_the_one_percent_trap` pins the
arithmetic down so nobody has to rediscover it.
"""

import numpy as np
import pytest

from shahoshi.scoring import (
    MahalanobisScorer,
    auprc,
    auroc,
    entropy,
    far_per_hour,
    flag_rate_for_far,
    recall_at_far,
    score_report,
    threshold_at_percentile,
    threshold_for_far,
    windows_per_hour,
)

HOP = 64 / 50  # 1.28 s: our 128-step window at 50% overlap, 50 Hz


class TestEntropy:
    def test_uniform_is_one(self):
        p = np.full((3, 6), 1 / 6)
        assert np.allclose(entropy(p), 1.0)

    def test_one_hot_is_zero(self):
        p = np.zeros((2, 6))
        p[:, 0] = 1.0
        assert np.allclose(entropy(p), 0.0, atol=1e-6)

    def test_is_in_unit_range(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(6), size=200)
        e = entropy(p)
        assert e.min() >= 0.0 and e.max() <= 1.0

    def test_normalization_makes_class_counts_comparable(self):
        """Uniform over 6 classes and uniform over 7 must both score 1.0.

        Without normalization, dropping the `lay` class would silently shift
        every entropy threshold carried over from the old 7-class model.
        """
        assert entropy(np.full((1, 6), 1 / 6))[0] == pytest.approx(1.0)
        assert entropy(np.full((1, 7), 1 / 7))[0] == pytest.approx(1.0)

    def test_flatter_distribution_scores_higher(self):
        peaked = np.array([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]])
        flat = np.array([[0.3, 0.2, 0.15, 0.15, 0.1, 0.1]])
        assert entropy(flat)[0] > entropy(peaked)[0]

    def test_renormalizes_unnormalized_input(self):
        """int8 dequantized softmax output does not sum to exactly 1."""
        p = np.full((1, 6), 1 / 6) * 1.05
        assert entropy(p)[0] == pytest.approx(1.0, abs=1e-6)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            entropy(np.ones(6))

    def test_rejects_single_class(self):
        with pytest.raises(ValueError):
            entropy(np.ones((3, 1)))


class TestMahalanobisScorer:
    def make(self, n=500, dim=8, seed=0):
        return np.random.default_rng(seed).normal(0, 1, (n, dim))

    def test_scores_near_zero_at_the_distribution_mean(self):
        E = self.make()
        s = MahalanobisScorer.fit(E)
        assert s.score(s.mean[None, :])[0] < 1e-6

    def test_outliers_score_higher_than_inliers(self):
        E = self.make()
        s = MahalanobisScorer.fit(E)
        inlier = s.score(E).mean()
        outlier = s.score(E[:50] + 20.0).mean()
        assert outlier > 5 * inlier

    def test_scores_are_non_negative(self):
        E = self.make()
        s = MahalanobisScorer.fit(E)
        assert (s.score(E) >= 0).all()

    def test_cholesky_reconstructs_the_covariance(self):
        """The device exports L, not the inverse, and does a triangular solve."""
        E = self.make()
        s = MahalanobisScorer.fit(E, shrinkage=1e-3)
        cov = s.cholesky @ s.cholesky.T
        assert np.allclose(cov @ s.precision, np.eye(len(s.mean)), atol=1e-6)

    def test_is_exactly_affine_invariant_without_shrinkage(self):
        """Mahalanobis distance is scale-free by construction: rescaling an
        embedding dimension must not change any distance."""
        E = self.make(dim=6)
        A = np.diag([1.0, 10.0, 0.1, 5.0, 2.0, 0.5])
        a = MahalanobisScorer.fit(E, shrinkage=0.0).score(E)
        b = MahalanobisScorer.fit(E @ A, shrinkage=0.0).score(E @ A)
        assert np.allclose(a, b, rtol=1e-6)

    def test_shrinkage_trades_exact_invariance_for_stability(self):
        """The ridge is added in the embedding's own units, so it is not
        rescaled along with the data and exact affine invariance is lost.

        That is an acceptable trade -- see the dead-dimension test below -- but
        it means the shrinkage value is part of the model and must be recorded
        in the run manifest alongside the thresholds it influences.
        """
        E = self.make(dim=6)
        A = np.diag([1.0, 10.0, 0.1, 5.0, 2.0, 0.5])
        a = MahalanobisScorer.fit(E, shrinkage=1e-3).score(E)
        b = MahalanobisScorer.fit(E @ A, shrinkage=1e-3).score(E @ A)
        assert not np.allclose(a, b, rtol=1e-6)   # not exact ...
        assert np.allclose(a, b, rtol=0.05)       # ... but close

    def test_shrinkage_keeps_a_dead_dimension_from_exploding(self):
        """A ReLU embedding routinely has near-dead dimensions. Without the
        ridge, the inverse covariance amplifies their numerical noise into huge
        distances for perfectly ordinary windows."""
        E = self.make(dim=6)
        E[:, 3] = 1e-9 * np.random.default_rng(1).normal(size=len(E))
        scores = MahalanobisScorer.fit(E, shrinkage=1e-3).score(E)
        assert np.isfinite(scores).all()
        assert scores.max() < 100

    def test_rejects_too_few_samples_for_the_dimension(self):
        with pytest.raises(ValueError, match="singular"):
            MahalanobisScorer.fit(self.make(n=8, dim=16))

    def test_rejects_wrong_embedding_width(self):
        s = MahalanobisScorer.fit(self.make(dim=8))
        with pytest.raises(ValueError):
            s.score(np.zeros((4, 5)))


class TestFalseAlarmArithmetic:
    def test_windows_per_hour_at_our_hop(self):
        assert windows_per_hour(HOP) == pytest.approx(2812.5)

    def test_the_one_percent_trap(self):
        """A 99th-percentile threshold is 28 false alarms per hour, not 'strict'.

        This is the arithmetic behind choosing the operating point in alarms per
        hour rather than in percentiles.
        """
        assert far_per_hour(0.01, HOP) == pytest.approx(28.125, rel=1e-3)

    def test_far_and_flag_rate_are_inverses(self):
        for far in (0.5, 1.0, 6.0, 30.0):
            assert far_per_hour(flag_rate_for_far(far, HOP), HOP) == pytest.approx(far)

    def test_one_alarm_per_hour_is_a_tiny_flag_rate(self):
        assert flag_rate_for_far(1.0, HOP) == pytest.approx(1 / 2812.5, rel=1e-6)

    def test_flag_rate_is_capped_at_one(self):
        assert flag_rate_for_far(1e9, HOP) == 1.0

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            windows_per_hour(0)
        with pytest.raises(ValueError):
            far_per_hour(1.5, HOP)
        with pytest.raises(ValueError):
            flag_rate_for_far(-1, HOP)


class TestThresholds:
    def test_percentile_threshold_matches_numpy(self):
        s = np.arange(101, dtype=float)
        assert threshold_at_percentile(s, 99) == pytest.approx(99.0)

    def test_threshold_for_far_meets_the_budget(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(0, 1, 200_000)
        thr = threshold_for_far(normal, target_far=1.0, hop_seconds=HOP)
        achieved = far_per_hour(float((normal > thr).mean()), HOP)
        assert achieved <= 1.3   # sampling noise, but the right order

    def test_tighter_budget_gives_a_higher_threshold(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(0, 1, 100_000)
        loose = threshold_for_far(normal, 30.0, HOP)
        tight = threshold_for_far(normal, 1.0, HOP)
        assert tight > loose

    def test_rejects_bad_percentile(self):
        with pytest.raises(ValueError):
            threshold_at_percentile(np.arange(10), 101)


class TestRecallAtFar:
    def test_perfectly_separated_scores_give_full_recall(self):
        normal = np.random.default_rng(0).normal(0, 1, 10_000)
        events = np.full(200, 50.0)
        r = recall_at_far(events, normal, target_far=1.0, hop_seconds=HOP)
        assert r["recall"] == pytest.approx(1.0)

    def test_identical_distributions_give_near_zero_recall(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(0, 1, 50_000)
        events = rng.normal(0, 1, 500)
        r = recall_at_far(events, normal, target_far=1.0, hop_seconds=HOP)
        assert r["recall"] < 0.05

    def test_recall_is_monotone_in_the_far_budget(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(0, 1, 50_000)
        events = rng.normal(2.5, 1, 1000)
        recalls = [
            recall_at_far(events, normal, far, HOP)["recall"]
            for far in (0.5, 1.0, 6.0, 30.0)
        ]
        assert recalls == sorted(recalls)

    def test_reports_the_far_it_actually_achieved(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(0, 1, 100_000)
        r = recall_at_far(np.full(50, 9.0), normal, 2.0, HOP)
        assert r["achieved_far"] <= 2.6
        assert r["target_far"] == 2.0

    def test_rejects_empty_inputs(self):
        with pytest.raises(ValueError):
            recall_at_far(np.array([]), np.arange(10.0), 1.0, HOP)


class TestAurocAuprc:
    def test_auroc_perfect_separation(self):
        assert auroc(np.arange(10.0) + 100, np.arange(10.0)) == pytest.approx(1.0)

    def test_auroc_inverted_separation(self):
        assert auroc(np.arange(10.0), np.arange(10.0) + 100) == pytest.approx(0.0)

    def test_auroc_identical_distributions_is_a_half(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=5000), rng.normal(size=5000)
        assert auroc(a, b) == pytest.approx(0.5, abs=0.03)

    def test_auroc_handles_ties_as_half_credit(self):
        """int8 output is heavily quantized, so ties are common and must not
        silently count as wins."""
        assert auroc(np.ones(10), np.ones(10)) == pytest.approx(0.5)

    def test_auroc_matches_a_brute_force_pair_count(self):
        rng = np.random.default_rng(1)
        pos, neg = rng.integers(0, 5, 40).astype(float), rng.integers(0, 5, 60).astype(float)
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        assert auroc(pos, neg) == pytest.approx(wins / (len(pos) * len(neg)))

    def test_auprc_perfect_separation(self):
        assert auprc(np.arange(10.0) + 100, np.arange(10.0)) == pytest.approx(1.0)

    def test_auprc_approaches_prevalence_for_random_scores(self):
        rng = np.random.default_rng(0)
        pos, neg = rng.normal(size=100), rng.normal(size=9900)
        assert auprc(pos, neg) == pytest.approx(0.01, abs=0.02)

    def test_auprc_is_harsher_than_auroc_under_imbalance(self):
        """Why both get reported: AUROC's false-positive axis is normalized by
        the enormous negative count, which flatters a model that would still
        alarm constantly in deployment."""
        rng = np.random.default_rng(2)
        pos = rng.normal(2.0, 1.0, 100)
        neg = rng.normal(0.0, 1.0, 100_000)
        assert auprc(pos, neg) < auroc(pos, neg)

    def test_reject_empty_inputs(self):
        with pytest.raises(ValueError):
            auroc(np.array([]), np.arange(3.0))
        with pytest.raises(ValueError):
            auprc(np.arange(3.0), np.array([]))


class TestScoreReport:
    def test_mentions_the_metrics_and_every_requested_far(self):
        rng = np.random.default_rng(0)
        text = score_report(
            rng.normal(3, 1, 300), rng.normal(0, 1, 20_000), HOP, label="mahalanobis"
        )
        assert "mahalanobis" in text
        assert "AUROC" in text and "AUPRC" in text
        for far in ("0.5", "1.0", "2.0", "6.0"):
            assert far in text
