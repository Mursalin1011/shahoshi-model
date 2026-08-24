"""Tests for the two-of-three consensus engine.

These tests are the specification. The vote cannot be validated against data --
no public corpus records movement, PPG and audio simultaneously during real
distress -- so what stops it being wrong is that every clause of it is pinned
here: a lone branch never alarms, a coincidence shorter than the sustain never
alarms, one confirmed event sends one alert, and a device with a dead sensor
reports that it cannot alarm instead of silently never alarming.

`test_a_single_branch_never_alarms` is the product claim itself. If it ever
fails, the device is a motion-triggered panic button with three sensors on it.
"""

import json
import math
import re

import numpy as np
import pytest

from shahoshi.config import BranchConfig, Config, FusionConfig
from shahoshi.export import write_config
from shahoshi.fusion import (
    BranchSpec,
    FusionEngine,
    FusionRule,
    alarm_times,
    alarms_per_hour,
    budget_per_branch,
    calibrate_branches,
    consensus_probability,
    evaluate,
    fused_far_upper_bound,
    fusion_report,
    fusion_source,
    latch_probability,
    measured_far,
    planned_specs,
    rule_from_config,
    runtime_config,
    simulate,
    specs_from_config,
    sustain_margin,
    write_runtime_config,
)

HOP = 64 / 50  # 1.28 s: the 128-step window at 50% overlap, 50 Hz

MOVEMENT = BranchSpec("movement", threshold=0.5, hold_seconds=4.0)
HR = BranchSpec("hr", threshold=0.5, hold_seconds=4.0)
AUDIO = BranchSpec("audio", threshold=0.5, hold_seconds=4.0)
THREE = [MOVEMENT, HR, AUDIO]

RULE = FusionRule(votes_required=2.0, sustain_seconds=4.0, cooldown_seconds=30.0)


def ticks(n: int, hop: float = HOP) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * hop


def stream(engine: FusionEngine, table: dict[str, list[float | None]], hop: float = HOP):
    """Drive the engine tick by tick from lists, letting None mean 'not reporting'."""
    n = {len(v) for v in table.values()}.pop()
    return [
        engine.update(i * hop, {k: v[i] for k, v in table.items()})
        for i in range(n)
    ]


class TestSpecValidation:
    def test_zero_hold_is_rejected(self):
        """A zero coincidence window means asynchronous branches can never agree."""
        with pytest.raises(ValueError, match="hold_seconds"):
            BranchSpec("movement", threshold=0.5, hold_seconds=0.0)

    def test_non_positive_weight_is_rejected(self):
        with pytest.raises(ValueError, match="weight"):
            BranchSpec("movement", threshold=0.5, weight=0.0)

    def test_unnamed_branch_is_rejected(self):
        with pytest.raises(ValueError, match="name"):
            BranchSpec("", threshold=0.5)

    def test_bad_degraded_policy_is_rejected(self):
        with pytest.raises(ValueError, match="degraded_policy"):
            FusionRule(degraded_policy="whatever")

    def test_duplicate_branch_names_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            FusionEngine([MOVEMENT, MOVEMENT], RULE)

    def test_unreachable_vote_is_rejected(self):
        """2 votes out of a single branch is not a strict rule, it is a dead one."""
        with pytest.raises(ValueError, match="unreachable"):
            FusionEngine([MOVEMENT], RULE)

    def test_empty_branch_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one branch"):
            FusionEngine([], RULE)


class TestConsensus:
    def test_a_single_branch_never_alarms(self):
        """The product claim: one sensor firing alone does not raise an alert.

        Movement pinned high for a minute, the other two silent throughout.
        """
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0] * 50,
            "hr": [0.0] * 50,
            "audio": [0.0] * 50,
        })
        assert not any(s.alarm for s in states)
        assert not any(s.in_consensus for s in states)

    def test_two_branches_reach_consensus(self):
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0] * 10,
            "hr": [1.0] * 10,
            "audio": [0.0] * 10,
        })
        assert states[0].in_consensus
        assert states[0].vote_weight == pytest.approx(2.0)

    def test_a_fire_latches_across_the_hold_window(self):
        """Branches are asynchronous: audio fires once, movement three ticks later."""
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "audio": [1.0, 0.0, 0.0, 0.0, 0.0],
            "movement": [0.0, 0.0, 0.0, 1.0, 0.0],
            "hr": [0.0] * 5,
        })
        # audio fired at t=0, hold 4 s covers t=3.84 where movement fires.
        assert states[3].in_consensus
        assert set(states[3].latched) == {"audio", "movement"}

    def test_a_fire_expires_after_the_hold_window(self):
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "audio": [1.0] + [0.0] * 6,
            "movement": [0.0] * 6 + [1.0],   # t=7.68, well past the 4 s hold
            "hr": [0.0] * 7,
        })
        assert not any(s.in_consensus for s in states)

    def test_weights_let_a_weak_branch_count_for_less(self):
        """A 190-taka amplitude trip should not carry a trained model's vote."""
        weak = BranchSpec("audio", threshold=0.5, weight=0.5)
        engine = FusionEngine([MOVEMENT, weak], FusionRule(votes_required=1.5))
        both = engine.update(0.0, {"movement": 1.0, "audio": 1.0})
        assert both.in_consensus and both.vote_weight == pytest.approx(1.5)
        engine.reset()
        audio_only = engine.update(0.0, {"movement": 0.0, "audio": 1.0})
        assert not audio_only.in_consensus


class TestSustain:
    def test_no_alarm_before_the_sustain_elapses(self):
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0] * 4,
            "hr": [1.0] * 4,
            "audio": [0.0] * 4,
        })
        # ticks at 0, 1.28, 2.56, 3.84 -- consensus held under 4 s throughout.
        assert all(s.in_consensus for s in states)
        assert not any(s.alarm for s in states)

    def test_alarm_on_the_first_tick_past_the_sustain(self):
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0] * 6,
            "hr": [1.0] * 6,
            "audio": [0.0] * 6,
        })
        fired = [s.t for s in states if s.alarm]
        assert fired == [pytest.approx(4 * HOP)]     # 5.12 s
        assert states[4].consensus_seconds == pytest.approx(4 * HOP)

    def test_a_break_in_consensus_restarts_the_clock(self):
        """Two short bursts of agreement, separated by a gap: no alarm.

        This is the clause that makes chance coincidences survivable. It needs a
        hold shorter than the sustain to bite at all -- see
        `test_the_sustain_only_bites_beyond_the_hold`.
        """
        short = [BranchSpec(b.name, 0.5, hold_seconds=2.0) for b in THREE]
        rule = FusionRule(votes_required=2.0, sustain_seconds=6.0)
        engine = FusionEngine(short, rule)
        fire = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        states = stream(engine, {
            "movement": fire, "hr": fire, "audio": [0.0] * 10,
        })
        assert not any(s.alarm for s in states)

    def test_the_sustain_only_bites_beyond_the_hold(self):
        """A latch carries consensus for `hold` seconds after the last fire.

        So a burst of firing lasting `d` seconds holds consensus for about
        `d + hold`, and the sustain clause only demands `sustain - hold` seconds
        of repeated agreement. With the implementation plan's 4 s hold and 4 s
        sustain that margin is zero, and one coincident pair of fires alarms --
        which reads as far stricter than it is.
        """
        assert sustain_margin(THREE, RULE) == 0.0
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0, 1.0] + [0.0] * 8,      # fires twice, then silent
            "hr": [1.0, 1.0] + [0.0] * 8,
            "audio": [0.0] * 10,
        })
        assert any(s.alarm for s in states)

        strict = FusionRule(votes_required=2.0, sustain_seconds=8.0)
        assert sustain_margin(THREE, strict) == 4.0
        engine = FusionEngine(THREE, strict)
        states = stream(engine, {
            "movement": [1.0, 1.0] + [0.0] * 8,
            "hr": [1.0, 1.0] + [0.0] * 8,
            "audio": [0.0] * 10,
        })
        assert not any(s.alarm for s in states)

    def test_a_sustain_of_zero_alarms_on_the_first_consenting_tick(self):
        engine = FusionEngine(THREE, FusionRule(sustain_seconds=0.0, cooldown_seconds=30.0))
        first = engine.update(0.0, {"movement": 1.0, "hr": 1.0, "audio": 0.0})
        assert first.alarm


class TestCooldown:
    def test_one_event_sends_one_alert(self):
        """An alert is a phone call. Ninety seconds of consensus, 30 s cooldown."""
        n = 70                                     # 89.6 s of held consensus
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0] * n,
            "hr": [1.0] * n,
            "audio": [0.0] * n,
        })
        fired = alarm_times(states)
        assert len(fired) == 3                      # first alert, then two repeats
        gaps = np.diff(fired)
        assert (gaps >= RULE.cooldown_seconds).all()

    def test_repeat_alarm_needs_cooldown_and_a_fresh_sustain(self):
        n = 40
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0] * n,
            "hr": [1.0] * n,
            "audio": [0.0] * n,
        })
        fired = alarm_times(states)
        assert fired[0] == pytest.approx(4 * HOP)
        # 5.12 + 30 = 35.12; the first tick at or past it is 28 * 1.28 = 35.84.
        assert fired[1] == pytest.approx(28 * HOP)

    def test_cooldown_remaining_counts_down(self):
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0] * 8,
            "hr": [1.0] * 8,
            "audio": [0.0] * 8,
        })
        assert states[4].cooldown_remaining == pytest.approx(30.0)
        assert states[5].cooldown_remaining == pytest.approx(30.0 - HOP)


class TestAvailability:
    def test_nan_is_not_reporting_rather_than_a_low_score(self):
        engine = FusionEngine(THREE, RULE)
        s = engine.update(0.0, {"movement": 1.0, "hr": float("nan"), "audio": 1.0})
        assert set(s.available) == {"movement", "audio"}
        assert "hr" not in s.latched

    def test_none_is_also_not_reporting(self):
        engine = FusionEngine(THREE, RULE)
        s = engine.update(0.0, {"movement": 1.0, "hr": None, "audio": 0.0})
        assert set(s.available) == {"movement", "audio"}

    def test_strict_policy_reports_that_it_cannot_alarm(self):
        """One live branch, 2-of-3 strict: the wearable is jewellery, and says so.

        This is the current state of the project -- movement exists, HR and audio
        do not -- and the point of the flag is that a zero false-alarm rate from
        this configuration cannot be mistaken for good news.
        """
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "movement": [1.0] * 20,
            "hr": [None] * 20,
            "audio": [None] * 20,
        })
        assert all(s.degraded for s in states)
        assert not any(s.can_alarm for s in states)
        assert not any(s.alarm for s in states)

    def test_proportional_policy_scales_the_requirement(self):
        engine = FusionEngine(
            THREE, FusionRule(degraded_policy="proportional", sustain_seconds=4.0)
        )
        states = stream(engine, {
            "movement": [1.0] * 6,
            "hr": [None] * 6,
            "audio": [None] * 6,
        })
        # 2.0 * (1/3) = 0.667, floored at min_votes_required = 1.0.
        assert states[0].votes_required == pytest.approx(1.0)
        assert states[0].can_alarm
        assert any(s.alarm for s in states)

    def test_proportional_with_two_of_three_live_still_needs_both(self):
        engine = FusionEngine(THREE, FusionRule(degraded_policy="proportional"))
        s = engine.update(0.0, {"movement": 1.0, "hr": 0.0, "audio": None})
        assert s.votes_required == pytest.approx(2.0 * 2 / 3)
        assert not s.in_consensus          # one vote of 1.0 against 1.33 required

    def test_dropout_clears_the_latch(self):
        """A fire from a sensor that has since gone quiet is not evidence about now."""
        engine = FusionEngine(THREE, RULE)
        states = stream(engine, {
            "audio": [1.0, None, 1.0],       # fires, drops out, comes back
            "movement": [0.0, 0.0, 1.0],
            "hr": [0.0, 0.0, 0.0],
        })
        assert states[1].latched == ()
        assert set(states[2].latched) == {"audio", "movement"}

    def test_unknown_branch_name_raises(self):
        engine = FusionEngine(THREE, RULE)
        with pytest.raises(KeyError, match="unknown branch"):
            engine.update(0.0, {"movment": 1.0})

    def test_time_going_backwards_raises(self):
        """On device this is the millis() rollover at 49.7 days."""
        engine = FusionEngine(THREE, RULE)
        engine.update(10.0, {"movement": 0.0, "hr": 0.0, "audio": 0.0})
        with pytest.raises(ValueError, match="backwards"):
            engine.update(9.0, {"movement": 0.0, "hr": 0.0, "audio": 0.0})


class TestCalibration:
    def test_thresholds_hit_their_own_per_branch_budget(self):
        rng = np.random.default_rng(0)
        normal = {
            "movement": rng.normal(size=20_000),
            "hr": rng.normal(loc=5.0, scale=2.0, size=20_000),
        }
        specs = calibrate_branches(normal, {"movement": 6.0, "hr": 6.0}, HOP)
        by_name = {s.name: s for s in specs}
        for name, spec in by_name.items():
            rate = (normal[name] > spec.threshold).mean()
            assert rate * 3600.0 / HOP == pytest.approx(6.0, rel=0.25)

    def test_branches_are_calibrated_on_their_own_scales(self):
        """A shared threshold across differently scaled branches would be nonsense."""
        rng = np.random.default_rng(1)
        normal = {
            "movement": rng.normal(size=5000),
            "audio": rng.normal(loc=500.0, scale=100.0, size=5000),
        }
        specs = calibrate_branches(normal, {"movement": 6.0, "audio": 6.0}, HOP)
        thresholds = {s.name: s.threshold for s in specs}
        assert thresholds["audio"] > 100 * thresholds["movement"]

    def test_per_branch_hold_and_weight_are_carried_through(self):
        rng = np.random.default_rng(2)
        normal = {"movement": rng.normal(size=5000), "hr": rng.normal(size=5000)}
        specs = calibrate_branches(
            normal,
            {"movement": 6.0, "hr": 12.0},
            HOP,
            hold_seconds={"movement": 4.0, "hr": 8.0},
            weights={"movement": 1.0, "hr": 0.5},
        )
        by_name = {s.name: s for s in specs}
        assert by_name["hr"].hold_seconds == 8.0
        assert by_name["hr"].weight == 0.5

    def test_budget_without_scores_raises(self):
        with pytest.raises(KeyError, match="no normal scores"):
            calibrate_branches({"movement": np.zeros(10)}, {"hr": 6.0}, HOP)


class TestSimulateAndEvaluate:
    def test_mismatched_stream_lengths_raise(self):
        engine = FusionEngine(THREE, RULE)
        with pytest.raises(ValueError, match="different lengths"):
            simulate(engine, {"movement": np.zeros(5), "hr": np.zeros(4),
                              "audio": np.zeros(5)}, HOP)

    def test_alarms_per_hour_arithmetic(self):
        """One alarm in a simulated hour is one alarm per hour."""
        n = int(round(3600 / HOP))
        scores = {k: np.zeros(n) for k in ("movement", "hr", "audio")}
        for k in ("movement", "hr"):
            scores[k][:10] = 1.0
        states = simulate(FusionEngine(THREE, RULE), scores, HOP)
        assert len(alarm_times(states)) == 1
        assert alarms_per_hour(states, HOP) == pytest.approx(1.0, rel=1e-3)

    def test_event_recall_and_latency(self):
        n = 400
        scores = {k: np.zeros(n) for k in ("movement", "hr", "audio")}
        onset_tick = 100
        for k in ("movement", "hr"):
            scores[k][onset_tick:onset_tick + 10] = 1.0
        states = simulate(FusionEngine(THREE, RULE), scores, HOP)

        result = evaluate(states, [onset_tick * HOP], HOP, latency_budget_seconds=10.0)
        assert result["recall"] == 1.0
        assert result["false_alarms"] == 0
        # Four seconds of sustain, resolved on the tick grid.
        assert result["median_latency"] == pytest.approx(4 * HOP)

    def test_an_alarm_outside_the_latency_budget_is_a_miss_and_a_false_alarm(self):
        """Detection twenty seconds late is not a rescue, and still interrupts."""
        n = 400
        scores = {k: np.zeros(n) for k in ("movement", "hr", "audio")}
        for k in ("movement", "hr"):
            scores[k][100:110] = 1.0
        states = simulate(FusionEngine(THREE, RULE), scores, HOP)
        # Claim the event started 30 s before the branches ever fired.
        result = evaluate(states, [100 * HOP - 30.0], HOP, latency_budget_seconds=10.0)
        assert result["recall"] == 0.0
        assert result["false_alarms"] == 1
        assert math.isnan(result["median_latency"])

    def test_false_alarms_per_hour_is_the_reported_unit(self):
        n = int(round(1800 / HOP))               # half an hour
        scores = {k: np.zeros(n) for k in ("movement", "hr", "audio")}
        for k in ("movement", "hr"):
            scores[k][:10] = 1.0
        states = simulate(FusionEngine(THREE, RULE), scores, HOP)
        result = evaluate(states, [], HOP)
        assert result["false_alarms"] == 1
        assert result["false_alarms_per_hour"] == pytest.approx(2.0, rel=1e-3)

    def test_report_mentions_the_rule_and_the_cost(self):
        n = 400
        scores = {k: np.zeros(n) for k in ("movement", "hr", "audio")}
        for k in ("movement", "hr"):
            scores[k][100:110] = 1.0
        states = simulate(FusionEngine(THREE, RULE), scores, HOP)
        text = fusion_report(evaluate(states, [100 * HOP], HOP), THREE, RULE)
        assert "sustained 4s" in text
        assert "false alarms" in text
        assert "movement" in text


class TestFalseAlarmArithmetic:
    def test_latch_probability_matches_the_small_rate_approximation(self):
        assert latch_probability(1.0, 4.0) == pytest.approx(4.0 / 3600.0, rel=1e-3)

    def test_latch_probability_grows_with_the_hold_window(self):
        assert latch_probability(6.0, 8.0) > latch_probability(6.0, 4.0)

    def test_consensus_probability_matches_the_binomial(self):
        """Two-of-three with equal weights is the binomial tail, exactly."""
        p = 0.1
        expected = 3 * p**2 * (1 - p) + p**3
        got = consensus_probability([p] * 3, [1.0] * 3, votes_required=2.0)
        assert got == pytest.approx(expected)

    def test_consensus_probability_endpoints(self):
        assert consensus_probability([1.0] * 3, [1.0] * 3, 2.0) == pytest.approx(1.0)
        assert consensus_probability([0.0] * 3, [1.0] * 3, 2.0) == pytest.approx(0.0)

    def test_the_vote_is_far_quieter_than_any_branch_in_it(self):
        """The whole justification for fusion, as a number.

        Three branches at 60 alarms/hour each -- individually unusable, one
        interruption a minute -- make a fused alarm rate two orders of magnitude
        lower once consensus must be held for four seconds.
        """
        fars = {b.name: 60.0 for b in THREE}
        fused = measured_far(fars, THREE, RULE, HOP, hours=48.0, seed=0)
        assert fused < 60.0 / 20

    def test_the_bound_really_bounds_the_measurement(self):
        fars = {b.name: 60.0 for b in THREE}
        bound = fused_far_upper_bound(fars, THREE, RULE)
        measured = measured_far(fars, THREE, RULE, HOP, hours=48.0, seed=1)
        assert measured <= bound

    def test_a_longer_hold_window_costs_false_alarms(self):
        loose = [BranchSpec(b.name, b.threshold, hold_seconds=12.0) for b in THREE]
        fars = {b.name: 60.0 for b in THREE}
        assert fused_far_upper_bound(fars, loose, RULE) > fused_far_upper_bound(
            fars, THREE, RULE
        )

    def test_budget_per_branch_inverts_the_bound(self):
        target = 1.0
        budget = budget_per_branch(target, THREE, RULE)
        got = fused_far_upper_bound({b.name: budget for b in THREE}, THREE, RULE)
        assert got == pytest.approx(target, rel=1e-3)

    def test_a_bound_with_no_sustain_is_refused(self):
        with pytest.raises(ValueError, match="sustain_seconds"):
            fused_far_upper_bound(
                {b.name: 6.0 for b in THREE}, THREE, FusionRule(sustain_seconds=0.0)
            )

    def test_measured_far_is_deterministic_for_a_seed(self):
        fars = {b.name: 60.0 for b in THREE}
        a = measured_far(fars, THREE, RULE, HOP, hours=12.0, seed=7)
        b = measured_far(fars, THREE, RULE, HOP, hours=12.0, seed=7)
        assert a == b


class TestExport:
    def test_runtime_config_carries_every_threshold(self):
        cfg = runtime_config(THREE, RULE)["fusion"]
        assert cfg["votes_required"] == 2.0
        assert cfg["degraded_policy"] == "strict"
        assert [b["name"] for b in cfg["branches"]] == ["movement", "hr", "audio"]
        assert all("threshold" in b for b in cfg["branches"])

    def test_runtime_config_merges_into_the_firmware_config(self, tmp_path):
        path = write_config(
            tmp_path / "runtime_config.json",
            win=128, channels=6, fs=50, stride=64,
            classes=["walk", "sit"], mean=[0.0] * 6, std=[1.0] * 6,
            thresholds={"entropy": 0.9},
            extra=runtime_config(THREE, RULE),
        )
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["hop_seconds"] == pytest.approx(HOP)
        assert loaded["fusion"]["sustain_seconds"] == 4.0

    def test_write_runtime_config_round_trips(self, tmp_path):
        path = write_runtime_config(tmp_path / "fusion.json", THREE, RULE)
        assert json.loads(path.read_text(encoding="utf-8")) == runtime_config(THREE, RULE)

    def test_generated_c_holds_the_calibrated_constants(self):
        specs = [
            BranchSpec("movement", threshold=3.25, hold_seconds=4.0, weight=1.0),
            BranchSpec("hr", threshold=12.5, hold_seconds=8.0, weight=0.5),
        ]
        src = fusion_source(specs, FusionRule(votes_required=1.5))
        assert "#define SHAHOSHI_FUSION_N_BRANCHES 2" in src
        thresholds = re.search(r"_thresholds\[\w+\] = \{([^}]*)\}", src).group(1)
        assert thresholds == "3.250000f, 12.500000f"
        holds = re.search(r"_holds_s\[\w+\] = \{([^}]*)\}", src).group(1)
        assert holds == "4.000000f, 8.000000f"
        assert '"movement", "hr"' in src
        assert "shahoshi_fusion_votes_required = 1.500000f" in src

    def test_generated_c_encodes_the_degradation_policy(self):
        strict = fusion_source(THREE, RULE)
        loose = fusion_source(THREE, FusionRule(degraded_policy="proportional"))
        assert "_proportional = false" in strict
        assert "_proportional = true" in loose

    def test_generated_c_warns_about_the_millis_rollover(self):
        """The one device-side trap the Python engine cannot catch for firmware."""
        assert "millis()" in fusion_source(THREE, RULE)

    def test_generated_c_warns_when_the_vote_cannot_be_reached(self):
        """One branch, two votes required: it compiles, runs, and never alarms."""
        src = fusion_source([MOVEMENT], RULE)
        assert "CANNOT RAISE AN ALARM" in src

    def test_a_reachable_vote_carries_no_warning(self):
        assert "CANNOT RAISE AN ALARM" not in fusion_source(THREE, RULE)

    def test_planned_thresholds_cannot_be_exported(self):
        """A placeholder must not reach the device, in JSON or in C."""
        planned = planned_specs(Config().fusion)
        with pytest.raises(ValueError, match="planned rather than calibrated"):
            runtime_config(planned, RULE)
        with pytest.raises(ValueError, match="no calibrated threshold"):
            fusion_source(planned, RULE)

    def test_generated_c_is_brace_balanced(self):
        src = fusion_source(THREE, RULE)
        assert src.count("{") == src.count("}")
        assert src.count("(") == src.count(")")


class TestConfigIntegration:
    def test_default_config_carries_the_two_of_three_rule(self):
        cfg = Config()
        assert cfg.fusion.votes_required == 2.0
        assert [b.name for b in cfg.fusion.branches] == ["movement", "hr", "audio"]

    def test_default_config_reports_the_vote_as_unreachable_today(self):
        """Only the movement branch is implemented, and the config says so."""
        note = Config().fusion.reachability_note()
        assert "UNREACHABLE" in note
        assert "hr" in note and "audio" in note

    def test_reachability_note_flips_when_the_branches_exist(self):
        cfg = FusionConfig(branches=[
            BranchConfig(name="movement", implemented=True),
            BranchConfig(name="hr", implemented=True),
            BranchConfig(name="audio"),
        ])
        assert cfg.reachability_note().startswith("fusion reachable")

    def test_yaml_round_trip_through_from_dict(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "name: fusion-test\n"
            "fusion:\n"
            "  votes_required: 1.5\n"
            "  sustain_seconds: 5.12\n"
            "  branches:\n"
            "    - {name: movement, far_budget: 6.0, weight: 1.0, implemented: true}\n"
            "    - {name: audio, far_budget: 12.0, weight: 0.5, threshold: 800.0}\n",
            encoding="utf-8",
        )
        cfg = Config.load(path)
        assert cfg.fusion.sustain_seconds == 5.12
        assert cfg.fusion.total_weight == 1.5
        assert cfg.fusion.implemented_weight == 1.0

    def test_unknown_fusion_key_raises(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("fusion:\n  votes_requried: 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown key"):
            Config.load(path)

    def test_unknown_branch_key_raises(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "fusion:\n  branches:\n    - {name: movement, hold: 4.0}\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="unknown key"):
            Config.load(path)

    def test_sustain_shorter_than_a_hop_raises(self, tmp_path):
        """A sustain the tick grid cannot express is a sustain that does nothing."""
        path = tmp_path / "c.yaml"
        path.write_text("fusion:\n  sustain_seconds: 0.5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="shorter than one inference hop"):
            Config.load(path)

    def test_votes_exceeding_total_weight_raises(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "fusion:\n"
            "  votes_required: 2.0\n"
            "  branches:\n"
            "    - {name: movement, weight: 1.0}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unreachable"):
            Config.load(path)

    def test_config_survives_the_manifest_round_trip(self):
        """`to_dict` feeds the manifest, so the vote is recorded with the run."""
        d = Config().to_dict()
        assert d["fusion"]["branches"][0]["name"] == "movement"

    def test_specs_from_config_calibrates_on_normal_scores(self):
        rng = np.random.default_rng(3)
        cfg = Config().fusion
        normal = {"movement": rng.normal(size=5000)}
        specs = specs_from_config(cfg, normal, HOP, only_implemented=True)
        assert [s.name for s in specs] == ["movement"]
        rate = (normal["movement"] > specs[0].threshold).mean()
        assert rate * 3600.0 / HOP == pytest.approx(6.0, rel=0.4)

    def test_specs_from_config_uses_a_pinned_threshold_when_given(self):
        cfg = FusionConfig(branches=[
            BranchConfig(name="audio", threshold=800.0, weight=1.0),
            BranchConfig(name="movement", threshold=3.0, weight=1.0),
        ])
        specs = specs_from_config(cfg)
        assert {s.name: s.threshold for s in specs} == {"audio": 800.0, "movement": 3.0}

    def test_a_branch_with_neither_scores_nor_a_threshold_raises(self):
        """No default threshold gets to decide when to call for help."""
        with pytest.raises(ValueError, match="neither normal scores"):
            specs_from_config(Config().fusion)

    def test_planned_specs_size_the_vote_before_the_sensors_exist(self):
        """The budget arithmetic runs on branches nobody has built yet."""
        cfg = Config().fusion
        planned = planned_specs(cfg)
        assert [s.name for s in planned] == ["movement", "hr", "audio"]
        assert all(math.isinf(s.threshold) for s in planned)

        budgets = {b.name: b.far_budget for b in cfg.branches}
        bound = fused_far_upper_bound(budgets, planned, rule_from_config(cfg))
        assert bound > 0

    def test_a_planned_branch_never_votes_if_it_reaches_an_engine(self):
        """The +inf placeholder is chosen so that misuse fails safe."""
        engine = FusionEngine(planned_specs(Config().fusion), RULE)
        states = stream(engine, {
            "movement": [1e9] * 10, "hr": [1e9] * 10, "audio": [1e9] * 10,
        })
        assert not any(s.alarm for s in states)

    def test_rule_from_config_matches_the_configured_rule(self):
        cfg = Config().fusion
        rule = rule_from_config(cfg)
        assert rule.votes_required == cfg.votes_required
        assert rule.sustain_seconds == cfg.sustain_seconds
        assert rule.degraded_policy == cfg.degraded_policy

    def test_the_configured_rule_drives_a_real_engine(self):
        """End to end: config -> rule + specs -> engine -> alarm."""
        cfg = FusionConfig(branches=[
            BranchConfig(name="movement", threshold=0.5, implemented=True),
            BranchConfig(name="audio", threshold=0.5, implemented=True),
        ])
        engine = FusionEngine(specs_from_config(cfg), rule_from_config(cfg))
        states = stream(engine, {"movement": [1.0] * 6, "audio": [1.0] * 6})
        assert any(s.alarm for s in states)
