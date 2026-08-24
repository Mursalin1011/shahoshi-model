"""The two-of-three consensus, as one specified, testable engine.

The product claim is a fusion claim: *a single sensor firing alone does not
raise an alert*. Elevated heart rate is exercise, a loud transient is a slammed
door, a sharp jerk is setting a bag down. At least two of movement, heart rate
and sound must agree, and agree for several seconds, before the buzzer sounds
and a WhatsApp alert leaves the device.

Why this is hand-designed rather than learned
---------------------------------------------
No public corpus records movement, PPG and audio simultaneously during real
distress. There is nothing to fit a fusion layer to, so the vote is specified by
hand -- which means the only defence against it being wrong is that it is
written down in one place, exercised by tests, and exported to the firmware as
generated code rather than transcribed by a human. That is what this module is.

The three things the vote actually needs
----------------------------------------
**Latching**, because the branches are asynchronous. Movement infers every
1.28 s, a PPG spike resolves over several seconds, an acoustic event lasts
milliseconds. "Two branches fired on the same tick" is a condition that would
almost never be true. Each fire therefore latches for `hold_seconds`, and
consensus means two latches overlap.

**Sustain**, because coincidence is cheap. Two independent branches at one alarm
per hour each, latched 4 s, collide by chance roughly twice a day; requiring the
coincidence to *persist* for four seconds is what turns that into a rare event.
`fused_far_upper_bound` puts a number on it before any hardware exists.

**Cooldown**, because an alert is a phone call to another human. A confirmed
event lasting thirty seconds must not send thirty alerts.

Latching and sustain interact, and the interaction is not obvious
-----------------------------------------------------------------
A latch keeps a single fire counting for `hold_seconds`, so a burst of firing
lasting `d` seconds holds consensus for roughly `d + hold`. The sustain
requirement therefore only demands `sustain - hold` seconds of *repeated*
firing: with the implementation plan's 4 s hold and 4 s sustain, one coincident
pair of fires is very nearly enough on its own, and the sustain clause is doing
far less work than it reads as doing. `sustain_margin` computes the difference
and `fusion_report` prints it, so nobody has to rediscover this from a field
false-alarm log. Widening the hold to catch asynchronous branches and requiring
a long sustain to reject coincidences are the same knob pulling in opposite
directions.

What this module deliberately does not do
-----------------------------------------
It does not decide what a branch score *is*. The movement branch may hand it a
Mahalanobis novelty score, a fall-head probability, or the maximum of the two;
the acoustic branch a log-mel classifier output; the HR branch a deviation from
a personal resting baseline. Each is calibrated separately, in alarms per hour,
by `calibrate_branches`, and the engine only ever compares a score against that
branch's own threshold.

The honest state of the vote today
----------------------------------
Only the movement branch exists. Under the default `strict` degradation policy a
2-of-3 rule with one live branch **cannot fire**, and the engine says so --
`FusionState.can_alarm` is False and `degraded` is True -- rather than quietly
never alarming and being mistaken for a device with an excellent false-alarm
rate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .scoring import flag_rate_for_far, threshold_for_far, windows_per_hour

# Float slack for time and weight comparisons. Ticks land on multiples of a hop
# that is not representable in binary (1.28 s), so `t - t0 >= sustain` is a
# comparison that fails on the exact tick without a tolerance -- and a sustain
# that needs one extra tick is a quarter-second of extra latency on every real
# event, arrived at by accident.
TOL = 1e-6

DEGRADED_POLICIES = ("strict", "proportional")


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BranchSpec:
    """One voter: what it is called, when it fires, and how long that counts.

    Parameters
    ----------
    threshold : float
        The branch fires on a tick when its score is strictly greater. Set it
        with `calibrate_branches`, in alarms per hour, not by eye.
    hold_seconds : float
        How long a fire keeps counting toward consensus. This is the coincidence
        window, and it is the most expensive knob in the design: false consensus
        grows roughly linearly with it (see `latch_probability`).
    weight : float
        Vote weight. Not decoration: the movement branch is a trained,
        subject-disjoint-validated model, while the LM393 acoustic branch is an
        amplitude threshold on a 190-taka microphone. One vote each asserts an
        equality the evidence does not support.
    """

    name: str
    threshold: float
    hold_seconds: float = 4.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch needs a name; it is the key firmware indexes by")
        if self.hold_seconds <= 0:
            raise ValueError(
                f"{self.name}: hold_seconds must be positive, else no two "
                f"asynchronous branches can ever overlap; got {self.hold_seconds}"
            )
        if self.weight <= 0:
            raise ValueError(f"{self.name}: weight must be positive; got {self.weight}")
        if math.isnan(self.threshold) or self.threshold == -math.inf:
            raise ValueError(
                f"{self.name}: threshold must be a real number or +inf (a branch that "
                f"never fires); got {self.threshold}"
            )


@dataclass(frozen=True)
class FusionRule:
    """The consensus rule: how many votes, held how long, how often.

    Parameters
    ----------
    votes_required : float
        Summed weight needed for consensus. 2.0 with three unit-weight branches
        is the two-of-three rule from the implementation plan.
    sustain_seconds : float
        Consensus must hold continuously for this long before an alarm. The plan
        specifies 4 s.
    cooldown_seconds : float
        Refractory period after an alarm, during which consensus is still
        tracked but no second alert is raised.
    degraded_policy : {"strict", "proportional"}
        What to do when a sensor is not reporting. `strict` leaves the vote
        requirement unchanged, so a device with one live branch cannot alarm and
        reports that. `proportional` scales the requirement by the fraction of
        weight still available -- which, taken to its conclusion, is a
        single-sensor trigger, i.e. exactly the false-alarm mode the consensus
        rule exists to prevent. It is offered because "the PPG lost contact so
        nothing can ever alarm" is also a failure; it is not the default because
        that failure is at least visible.
    min_votes_required : float
        Floor under the proportional policy. Left at 1.0 it permits a
        single-branch alarm on a badly degraded device.
    """

    votes_required: float = 2.0
    sustain_seconds: float = 4.0
    cooldown_seconds: float = 30.0
    degraded_policy: str = "strict"
    min_votes_required: float = 1.0

    def __post_init__(self) -> None:
        if self.votes_required <= 0:
            raise ValueError("votes_required must be positive")
        if self.sustain_seconds < 0:
            raise ValueError("sustain_seconds must be non-negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if self.degraded_policy not in DEGRADED_POLICIES:
            raise ValueError(
                f"degraded_policy must be one of {DEGRADED_POLICIES}; "
                f"got {self.degraded_policy!r}"
            )
        if self.min_votes_required <= 0:
            raise ValueError("min_votes_required must be positive")


@dataclass(frozen=True)
class FusionState:
    """Everything the engine knows after one tick. Firmware logs this verbatim.

    `can_alarm` is the field worth watching. False means the alarm is
    unreachable given which sensors are currently reporting -- a wearable in
    that state is jewellery, and the wearer should be told.
    """

    t: float
    available: tuple[str, ...]
    latched: tuple[str, ...]
    vote_weight: float
    votes_required: float
    in_consensus: bool
    consensus_seconds: float
    alarm: bool
    cooldown_remaining: float
    degraded: bool
    can_alarm: bool


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

class FusionEngine:
    """Streaming consensus state machine. One tick in, one `FusionState` out.

    This is the reference implementation: `fusion_source` generates the C the
    firmware runs from the same specification, so the two cannot drift the way a
    hand-maintained op resolver did (see `shahoshi.export`).

    A score of `None` or NaN means *that sensor is not reporting on this tick* --
    no skin contact on the PPG, a mic buffer not yet filled. That is different
    from a low score, and conflating the two is how a dead sensor becomes a
    permanent silent no-vote.
    """

    def __init__(self, branches: Sequence[BranchSpec], rule: FusionRule | None = None):
        if not branches:
            raise ValueError("a vote needs at least one branch")
        names = [b.name for b in branches]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate branch name(s) {dupes}: scores are keyed by name")

        self.branches = tuple(branches)
        self.rule = rule or FusionRule()
        self.total_weight = float(sum(b.weight for b in self.branches))
        if self.rule.votes_required > self.total_weight + TOL:
            raise ValueError(
                f"votes_required {self.rule.votes_required} exceeds the total weight "
                f"of the configured branches ({self.total_weight}): the alarm is "
                f"unreachable by construction. Add the missing branch, or lower the "
                f"requirement deliberately."
            )
        self._by_name = {b.name: b for b in self.branches}
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        self._last_fire: dict[str, float] = {b.name: -math.inf for b in self.branches}
        self._consensus_since: float | None = None
        self._cooldown_until: float = -math.inf
        self._last_t: float | None = None

    # -- one tick ------------------------------------------------------------

    def update(self, t: float, scores: Mapping[str, float | None]) -> FusionState:
        """Advance to time `t` (seconds) with this tick's branch scores.

        Unknown branch names raise: a typo would otherwise remove a voter
        silently, and a vote missing a voter still returns confident states.
        """
        t = float(t)
        if self._last_t is not None and t < self._last_t - TOL:
            raise ValueError(
                f"time went backwards: {t} after {self._last_t}. On device this is "
                f"the millis() rollover at 49.7 days -- keep a 64-bit tick counter "
                f"rather than letting the state machine see it."
            )
        unknown = sorted(set(scores) - set(self._by_name))
        if unknown:
            raise KeyError(
                f"unknown branch(es) {unknown}; configured branches are "
                f"{sorted(self._by_name)}"
            )
        self._last_t = t

        available: list[str] = []
        latched: list[str] = []
        for b in self.branches:
            raw = scores.get(b.name)
            score = None if raw is None else float(raw)
            if score is None or math.isnan(score):
                # Not reporting. Drop the latch too: a fire from a sensor that
                # has since gone quiet is not evidence about now.
                self._last_fire[b.name] = -math.inf
                continue
            available.append(b.name)
            if score > b.threshold:
                self._last_fire[b.name] = t
            if t - self._last_fire[b.name] <= b.hold_seconds + TOL:
                latched.append(b.name)

        available_weight = sum(self._by_name[n].weight for n in available)
        vote_weight = sum(self._by_name[n].weight for n in latched)
        required = self._required(available_weight)
        degraded = available_weight < self.total_weight - TOL
        can_alarm = available_weight >= required - TOL

        in_consensus = vote_weight >= required - TOL
        if in_consensus:
            if self._consensus_since is None:
                self._consensus_since = t
        else:
            self._consensus_since = None
        held = 0.0 if self._consensus_since is None else t - self._consensus_since

        alarm = (
            in_consensus
            and held >= self.rule.sustain_seconds - TOL
            and t >= self._cooldown_until - TOL
        )
        if alarm:
            self._cooldown_until = t + self.rule.cooldown_seconds
            # Re-earn the sustain after an alert. A genuinely ongoing event
            # re-alarms no sooner than cooldown plus sustain, which is the point.
            self._consensus_since = None

        return FusionState(
            t=t,
            available=tuple(available),
            latched=tuple(latched),
            vote_weight=float(vote_weight),
            votes_required=float(required),
            in_consensus=bool(in_consensus),
            consensus_seconds=float(held),
            alarm=bool(alarm),
            cooldown_remaining=float(max(0.0, self._cooldown_until - t)),
            degraded=bool(degraded),
            can_alarm=bool(can_alarm),
        )

    def _required(self, available_weight: float) -> float:
        if self.rule.degraded_policy == "strict" or self.total_weight <= 0:
            return self.rule.votes_required
        scaled = self.rule.votes_required * (available_weight / self.total_weight)
        return max(self.rule.min_votes_required, scaled)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def calibrate_branches(
    normal_scores: Mapping[str, np.ndarray],
    budgets: Mapping[str, float],
    hop_seconds: float,
    hold_seconds: Mapping[str, float] | float = 4.0,
    weights: Mapping[str, float] | None = None,
) -> list[BranchSpec]:
    """Turn per-branch normal-data scores into calibrated `BranchSpec`s.

    Each branch gets its own alarms-per-hour budget, set on *its own* normal
    data, before anything is fused. Calibrating the branches jointly against the
    fused outcome would fit the coincidence structure of whatever recording
    happened to be on hand, and there is no corpus of real simultaneous
    multimodal distress against which to check that.

    Per-branch budgets are deliberately loose compared with the fused target:
    the whole point of consensus is that two 6-alarm-per-hour branches make a
    far quieter alarm together. `fused_far_upper_bound` says how much quieter,
    and `budget_per_branch` inverts it.
    """
    missing = sorted(set(budgets) - set(normal_scores))
    if missing:
        raise KeyError(f"budget given for {missing} but no normal scores to calibrate on")
    if not budgets:
        raise ValueError("no branch budgets given")

    specs: list[BranchSpec] = []
    for name, budget in budgets.items():
        scores = np.asarray(normal_scores[name], dtype=np.float64)
        if scores.ndim != 1 or not len(scores):
            raise ValueError(f"{name}: expected a non-empty 1-D array of normal scores")
        hold = (
            float(hold_seconds)
            if isinstance(hold_seconds, (int, float))
            else float(hold_seconds[name])
        )
        specs.append(
            BranchSpec(
                name=name,
                threshold=float(threshold_for_far(scores, budget, hop_seconds)),
                hold_seconds=hold,
                weight=float((weights or {}).get(name, 1.0)),
            )
        )
    return specs


def rule_from_config(fusion_cfg: Any) -> FusionRule:
    """Build the rule from a `config.FusionConfig`, without importing it.

    Duck-typed on purpose: `shahoshi.config` stays a leaf module that pulls in
    nothing but YAML, so a config can be loaded and inspected in a context where
    the numeric stack is not wanted.
    """
    return FusionRule(
        votes_required=float(fusion_cfg.votes_required),
        sustain_seconds=float(fusion_cfg.sustain_seconds),
        cooldown_seconds=float(fusion_cfg.cooldown_seconds),
        degraded_policy=str(fusion_cfg.degraded_policy),
        min_votes_required=float(fusion_cfg.min_votes_required),
    )


def planned_specs(fusion_cfg: Any) -> list[BranchSpec]:
    """Specs for the coincidence *arithmetic*, including branches not yet built.

    Real holds and weights, and a threshold of +inf -- a branch that never fires.
    The false-alarm arithmetic (`fused_far_upper_bound`, `measured_far`) is
    driven by each branch's FAR *budget*, not by its threshold, so a sensor that
    does not exist yet can still be sized: that is the whole point of costing the
    vote before buying the hardware.

    The +inf placeholder is chosen so that misuse fails safe. Hand these to a
    real `FusionEngine` and the unbuilt branches simply never vote; hand it a
    zero placeholder instead and they would vote on every tick.
    `specs_from_config` is the path for an engine that will see real scores.
    """
    return [
        BranchSpec(
            name=b.name,
            threshold=math.inf,
            hold_seconds=float(b.hold_seconds),
            weight=float(b.weight),
        )
        for b in fusion_cfg.branches
    ]


def specs_from_config(
    fusion_cfg: Any,
    normal_scores: Mapping[str, np.ndarray] | None = None,
    hop_seconds: float | None = None,
    only_implemented: bool = False,
) -> list[BranchSpec]:
    """Build calibrated `BranchSpec`s from a `config.FusionConfig`.

    A branch is calibrated from `normal_scores[name]` against its own
    `far_budget` when scores are supplied, and otherwise must carry an explicit
    `threshold` in the config -- a branch with neither raises, because the
    alternative is a default threshold nobody chose deciding when to call for
    help.
    """
    specs: list[BranchSpec] = []
    for b in fusion_cfg.branches:
        if only_implemented and not b.implemented:
            continue
        scores = None if normal_scores is None else normal_scores.get(b.name)
        if scores is not None:
            if hop_seconds is None:
                raise ValueError("hop_seconds is required to calibrate from scores")
            threshold = float(threshold_for_far(
                np.asarray(scores, dtype=np.float64), b.far_budget, hop_seconds
            ))
        elif b.threshold is not None:
            threshold = float(b.threshold)
        else:
            raise ValueError(
                f"branch {b.name!r} has neither normal scores to calibrate on nor an "
                f"explicit threshold. Pin one in the config only for a branch with no "
                f"scores to calibrate against, such as a hardware amplitude trip."
            )
        specs.append(
            BranchSpec(
                name=b.name,
                threshold=threshold,
                hold_seconds=float(b.hold_seconds),
                weight=float(b.weight),
            )
        )
    if not specs:
        raise ValueError("no branches selected; a vote needs at least one")
    return specs


# ---------------------------------------------------------------------------
# offline evaluation
# ---------------------------------------------------------------------------

def simulate(
    engine: FusionEngine,
    scores: Mapping[str, Sequence[float] | np.ndarray],
    hop_seconds: float,
    timestamps: Sequence[float] | np.ndarray | None = None,
    reset: bool = True,
) -> list[FusionState]:
    """Run the engine over aligned per-branch score streams. NaN = not reporting.

    The caller resamples the branches onto one tick grid. That is a real
    constraint rather than an oversight: aligning a 1.28 s movement hop with a
    1 Hz PPG reading is a decision with consequences for detection latency, and
    burying it inside the evaluator would hide it.
    """
    if not scores:
        raise ValueError("no branch scores given")
    lengths = {len(np.asarray(v)) for v in scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"branch score streams have different lengths: {sorted(lengths)}")
    n = lengths.pop()
    if timestamps is None:
        ts = np.arange(n, dtype=np.float64) * hop_seconds
    else:
        ts = np.asarray(timestamps, dtype=np.float64)
        if len(ts) != n:
            raise ValueError(f"{len(ts)} timestamps for {n} score ticks")

    cols = {k: np.asarray(v, dtype=np.float64) for k, v in scores.items()}
    if reset:
        engine.reset()
    return [
        engine.update(float(ts[i]), {k: float(v[i]) for k, v in cols.items()})
        for i in range(n)
    ]


def alarm_times(states: Iterable[FusionState]) -> np.ndarray:
    """Timestamps of the ticks on which an alert would have been sent."""
    return np.array([s.t for s in states if s.alarm], dtype=np.float64)


def alarms_per_hour(states: Sequence[FusionState], hop_seconds: float) -> float:
    """Alarm rate over the simulated stretch, in this project's only FAR unit."""
    if not len(states):
        raise ValueError("no states to measure")
    hours = len(states) * hop_seconds / 3600.0
    return float(len(alarm_times(states)) / hours)


def evaluate(
    states: Sequence[FusionState],
    event_onsets: Sequence[float] | np.ndarray,
    hop_seconds: float,
    latency_budget_seconds: float = 10.0,
) -> dict[str, Any]:
    """Event-level recall, detection latency, and the false alarms it cost.

    Event-level, not window-level, because sustain and cooldown collapse many
    consenting windows into one alert: window recall would score a single alarm
    during a 20 s assault as fifteen misses, and a per-window FAR would charge
    one alert once per window it spans. What the wearer experiences is alerts,
    so alerts are what is counted.

    An event counts as detected when an alarm falls within
    `latency_budget_seconds` of its onset. Later than that is not a rescue. An
    alarm matching no event is a false alarm, charged at the rate the wearer
    feels it: per hour.
    """
    if not len(states):
        raise ValueError("no states to evaluate")
    if latency_budget_seconds <= 0:
        raise ValueError("latency_budget_seconds must be positive")

    fired = alarm_times(states)
    onsets = np.asarray(event_onsets, dtype=np.float64).ravel()
    hours = len(states) * hop_seconds / 3600.0

    latencies: list[float] = []
    detected = np.zeros(len(onsets), dtype=bool)
    matched = np.zeros(len(fired), dtype=bool)
    for i, onset in enumerate(onsets):
        window = (fired >= onset - TOL) & (fired <= onset + latency_budget_seconds + TOL)
        matched |= window
        if window.any():
            detected[i] = True
            latencies.append(float(fired[window].min() - onset))

    false_alarms = int((~matched).sum())
    return {
        "n_events": int(len(onsets)),
        "n_alarms": int(len(fired)),
        "recall": float(detected.mean()) if len(onsets) else float("nan"),
        "missed": [float(o) for o, d in zip(onsets, detected) if not d],
        "false_alarms": false_alarms,
        "false_alarms_per_hour": float(false_alarms / hours) if hours else float("nan"),
        "median_latency": float(np.median(latencies)) if latencies else float("nan"),
        "p90_latency": float(np.percentile(latencies, 90)) if latencies else float("nan"),
        "latency_budget_seconds": float(latency_budget_seconds),
        "hours": float(hours),
    }


def sustain_margin(branches: Sequence[BranchSpec], rule: FusionRule) -> float:
    """Seconds of *repeated* firing the sustain clause actually demands.

    `sustain_seconds - max(hold_seconds)`. A single fire holds its branch's vote
    for the whole hold window, so a burst of firing lasting `d` seconds keeps
    consensus for about `d + hold`. At or below zero, one coincident pair of
    fires satisfies the sustain on its own and the clause buys nothing beyond
    the coincidence itself -- which is the case for the implementation plan's
    4 s hold with 4 s sustain.

    Raising the sustain is the cheap fix; shortening the hold is not, because the
    hold is what lets a 1.28 s movement hop agree with a millisecond-long
    acoustic event at all.
    """
    return float(rule.sustain_seconds - max(b.hold_seconds for b in branches))


def fusion_report(
    result: Mapping[str, Any], branches: Sequence[BranchSpec], rule: FusionRule
) -> str:
    """A text block for the manifest: what the vote was, and what it bought."""
    total = sum(b.weight for b in branches)
    lines = [
        f"fusion: {rule.votes_required:g} of {total:g} weight, sustained "
        f"{rule.sustain_seconds:g}s, cooldown {rule.cooldown_seconds:g}s "
        f"({rule.degraded_policy} when a sensor is down)",
    ]
    for b in branches:
        lines.append(
            f"  {b.name:<10} threshold {b.threshold:>9.4f}"
            f"   hold {b.hold_seconds:>4.1f}s   weight {b.weight:g}"
        )
    lines.append(
        f"  events {result['n_events']}   recall {result['recall']:.3f}"
        f"   median latency {result['median_latency']:.2f}s"
        f"   p90 {result['p90_latency']:.2f}s"
    )
    lines.append(
        f"  false alarms {result['false_alarms']} over {result['hours']:.2f} h"
        f"  = {result['false_alarms_per_hour']:.2f}/h"
    )
    margin = sustain_margin(branches, rule)
    if margin > 0:
        lines.append(
            f"  sustain margin {margin:.2f}s: consensus must be re-fired, not just latched"
        )
    else:
        lines.append(
            f"  sustain margin {margin:.2f}s: the {rule.sustain_seconds:g}s sustain is "
            f"inside the {max(b.hold_seconds for b in branches):g}s hold, so a single "
            f"coincident pair of fires can alarm on its own"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# what the vote costs, before any hardware exists
# ---------------------------------------------------------------------------

def latch_probability(branch_far: float, hold_seconds: float) -> float:
    """Fraction of time a branch alarming at `branch_far` per hour sits latched.

    Poisson arrivals: P(at least one fire within the last `hold` seconds).
    """
    if branch_far < 0:
        raise ValueError("branch_far must be non-negative")
    if hold_seconds <= 0:
        raise ValueError("hold_seconds must be positive")
    return float(1.0 - math.exp(-(branch_far / 3600.0) * hold_seconds))


def consensus_probability(
    latch_probs: Sequence[float], weights: Sequence[float], votes_required: float
) -> float:
    """P(latched weight >= votes_required) at a random instant, under independence.

    Exact by enumeration -- there are three branches, not thirty.
    """
    p = [float(x) for x in latch_probs]
    w = [float(x) for x in weights]
    if len(p) != len(w):
        raise ValueError(f"{len(p)} latch probabilities for {len(w)} weights")
    if any(not 0.0 <= x <= 1.0 for x in p):
        raise ValueError("latch probabilities must be in [0, 1]")

    n = len(p)
    total = 0.0
    for k in range(n + 1):
        for subset in combinations(range(n), k):
            if sum(w[i] for i in subset) < votes_required - TOL:
                continue
            prob = 1.0
            for i in range(n):
                prob *= p[i] if i in subset else (1.0 - p[i])
            total += prob
    return float(total)


def fused_far_upper_bound(
    branch_fars: Mapping[str, float],
    branches: Sequence[BranchSpec],
    rule: FusionRule,
) -> float:
    """An upper bound on fused alarms per hour, under independence.

    The derivation, so it can be argued with: consensus occupies a fraction `p`
    of the time, i.e. 3600p seconds in an hour. Every alarm consumes at least
    `sustain_seconds` of *continuous* consensus, so there can be at most
    3600p/sustain alarms per hour. Cooldown only lowers that further and is
    ignored, which is why this is a bound and not an estimate.

    Two warnings, both load-bearing:

    **It is loose.** Chance coincidences are short, and requiring one to persist
    for four seconds kills nearly all of them, so the true rate is far below the
    bound. `measured_far` runs the real engine over Poisson branch fires and is
    the number to quote.

    **Independence is false.** Motion artifact corrupts the PPG signal, so a
    struggle drives the movement branch and the heart-rate branch from one
    physical cause; a scream co-occurs with the flailing that produces it. That
    correlation is exactly what makes the vote *sensitive* to real distress and
    exactly what makes it *noisier* than this arithmetic during a jog. No public
    dataset lets us measure it, which is the largest open risk in the design and
    belongs next to any fused FAR number this function produces.
    """
    if rule.sustain_seconds <= 0:
        raise ValueError(
            "the bound divides by sustain_seconds; with no sustain requirement "
            "every momentary coincidence is an alarm and the bound is meaningless"
        )
    missing = sorted({b.name for b in branches} - set(branch_fars))
    if missing:
        raise KeyError(f"no per-branch FAR given for {missing}")

    probs = [latch_probability(branch_fars[b.name], b.hold_seconds) for b in branches]
    p = consensus_probability(probs, [b.weight for b in branches], rule.votes_required)
    return float(3600.0 * p / rule.sustain_seconds)


def budget_per_branch(
    target_fused_far: float,
    branches: Sequence[BranchSpec],
    rule: FusionRule,
    hi: float = 3600.0,
) -> float:
    """The equal per-branch alarms-per-hour budget whose *bound* meets a fused target.

    The design question asked in the direction it actually arises: "I will accept
    one fused alarm per hour -- how noisy is each branch allowed to be?" Because
    the bound is conservative, the answer is safe under independence and
    optimistic under correlation.
    """
    if target_fused_far <= 0:
        raise ValueError("target_fused_far must be positive")

    def bound(f: float) -> float:
        return fused_far_upper_bound({b.name: f for b in branches}, branches, rule)

    if bound(hi) <= target_fused_far:
        return float(hi)
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bound(mid) > target_fused_far:
            hi = mid
        else:
            lo = mid
    return float(lo)


def measured_far(
    branch_fars: Mapping[str, float],
    branches: Sequence[BranchSpec],
    rule: FusionRule,
    hop_seconds: float,
    hours: float = 24.0,
    seed: int = 0,
) -> float:
    """Fused alarms per hour, by running the real engine over synthetic branch fires.

    Each branch fires as an independent Bernoulli process at the per-tick rate
    its FAR budget implies, and the *actual* `FusionEngine` decides. The reuse is
    the point: an analytic model of the vote can be wrong about the vote, while
    this cannot -- it is the vote.

    Still assumes independence between branches, which reality does not. Read it
    as the floor set by chance, not as a field number.
    """
    if hours <= 0:
        raise ValueError("hours must be positive")
    missing = sorted({b.name for b in branches} - set(branch_fars))
    if missing:
        raise KeyError(f"no per-branch FAR given for {missing}")

    n = int(round(hours * windows_per_hour(hop_seconds)))
    if n < 1:
        raise ValueError("the simulated stretch is shorter than one tick")
    rng = np.random.default_rng(seed)

    # A fire is a score of 1.0 against a 0.5 threshold, so the engine under test
    # keeps its own weights, holds and rule while only the scores are synthetic.
    synthetic = [replace(b, threshold=0.5) for b in branches]
    scores = {
        b.name: (rng.random(n) < flag_rate_for_far(branch_fars[b.name], hop_seconds))
        .astype(np.float64)
        for b in branches
    }
    states = simulate(FusionEngine(synthetic, rule), scores, hop_seconds)
    return alarms_per_hour(states, hop_seconds)


# ---------------------------------------------------------------------------
# export: the firmware runs generated code, not a transcription
# ---------------------------------------------------------------------------

def runtime_config(branches: Sequence[BranchSpec], rule: FusionRule) -> dict[str, Any]:
    """The fusion block for the runtime JSON; pass as `export.write_config(extra=...)`.

    Refuses a non-finite threshold. `json.dumps` would happily write `Infinity`,
    which is not JSON, and the placeholder from `planned_specs` has no business
    reaching a device: export the branches that exist, and let the vote read as
    unreachable rather than as calibrated.
    """
    unfinished = [b.name for b in branches if not math.isfinite(b.threshold)]
    if unfinished:
        raise ValueError(
            f"branch(es) {unfinished} carry a non-finite threshold, so they are "
            f"planned rather than calibrated. Export the implemented branches "
            f"(specs_from_config(..., only_implemented=True)); planned_specs is for "
            f"the false-alarm arithmetic, not for the device."
        )
    return {
        "fusion": {
            "votes_required": float(rule.votes_required),
            "sustain_seconds": float(rule.sustain_seconds),
            "cooldown_seconds": float(rule.cooldown_seconds),
            "degraded_policy": rule.degraded_policy,
            "min_votes_required": float(rule.min_votes_required),
            "branches": [
                {
                    "name": b.name,
                    "threshold": float(b.threshold),
                    "hold_seconds": float(b.hold_seconds),
                    "weight": float(b.weight),
                }
                for b in branches
            ],
        }
    }


def write_runtime_config(
    path: str | Path, branches: Sequence[BranchSpec], rule: FusionRule
) -> Path:
    """Write the fusion block alone, for firmware that reads it as its own file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_config(branches, rule), indent=2), encoding="utf-8")
    print(f"  wrote {path.name}")
    return path


def fusion_source(
    branches: Sequence[BranchSpec], rule: FusionRule, prefix: str = "shahoshi_fusion"
) -> str:
    """Generate the C state machine the firmware runs.

    Generated rather than documented, for the same reason the op resolver is:
    the one hand-maintained correspondence in the pre-refactor build (eight
    registered operators against a thirteen-operator model) was also the one
    that would have failed on the device. A consensus rule transcribed by hand
    into firmware is the same bet, with a quieter failure mode -- a vote that is
    subtly wrong still runs.
    """
    n = len(branches)
    if n == 0:
        raise ValueError("no branches to emit")

    def fmt(v: float) -> str:
        return f"{float(v):.6f}f"

    unfinished = [b.name for b in branches if not math.isfinite(b.threshold)]
    if unfinished:
        raise ValueError(
            f"branch(es) {unfinished} have no calibrated threshold; a placeholder "
            f"must not be compiled into firmware"
        )

    names = ", ".join(f'"{b.name}"' for b in branches)
    thresholds = ", ".join(fmt(b.threshold) for b in branches)
    holds = ", ".join(fmt(b.hold_seconds) for b in branches)
    weights = ", ".join(fmt(b.weight) for b in branches)
    total = sum(b.weight for b in branches)
    upper = prefix.upper()
    proportional = "true" if rule.degraded_policy == "proportional" else "false"

    # The firmware equivalent of export.py's un-accelerated-operator warning: the
    # code compiles and runs, and never alarms. Said in the file, not in a report
    # nobody rereads at flash time.
    warning = ""
    if total < rule.votes_required - TOL:
        warning = (
            f"//\n// WARNING: the branches emitted here carry {total:g} vote weight "
            f"against\n// {rule.votes_required:g} required, so this build CANNOT RAISE "
            f"AN ALARM. That is the\n// expected state while the remaining branches are "
            f"unbuilt -- but a field test\n// of this firmware measures nothing except "
            f"that it stayed silent.\n"
        )

    return f"""#pragma once
// Generated by shahoshi.fusion.fusion_source. Do not hand-edit: regenerate when
// thresholds are recalibrated. This is a transcription of FusionEngine.update()
// in src/shahoshi/fusion.py, which is the tested reference implementation.
//
// Rule: {rule.votes_required:g} of {total:g} weight, sustained {rule.sustain_seconds:g}s, cooldown {rule.cooldown_seconds:g}s, {rule.degraded_policy} when a sensor is not reporting.
{warning}#include <stdbool.h>
#include <math.h>

#define {upper}_N_BRANCHES {n}
#define {upper}_TOL 1e-6f

static const char *const {prefix}_names[{upper}_N_BRANCHES] = {{{names}}};
static const float {prefix}_thresholds[{upper}_N_BRANCHES] = {{{thresholds}}};
static const float {prefix}_holds_s[{upper}_N_BRANCHES] = {{{holds}}};
static const float {prefix}_weights[{upper}_N_BRANCHES] = {{{weights}}};

static const float {prefix}_votes_required = {fmt(rule.votes_required)};
static const float {prefix}_total_weight = {fmt(total)};
static const float {prefix}_sustain_s = {fmt(rule.sustain_seconds)};
static const float {prefix}_cooldown_s = {fmt(rule.cooldown_seconds)};
static const float {prefix}_min_votes = {fmt(rule.min_votes_required)};
static const bool {prefix}_proportional = {proportional};

typedef struct {{
  float last_fire_s[{upper}_N_BRANCHES];
  float consensus_since_s;
  bool  in_consensus;
  float cooldown_until_s;
  // Outputs of the last update, for logging and for the "sensor down" UI.
  float vote_weight;
  float votes_required;
  bool  degraded;
  bool  can_alarm;
}} {prefix}_state;

static inline void {prefix}_reset({prefix}_state *s) {{
  for (int i = 0; i < {upper}_N_BRANCHES; i++) s->last_fire_s[i] = -INFINITY;
  s->consensus_since_s = 0.0f;
  s->in_consensus = false;
  s->cooldown_until_s = -INFINITY;
  s->vote_weight = 0.0f;
  s->votes_required = {prefix}_votes_required;
  s->degraded = false;
  s->can_alarm = false;
}}

// One tick. `scores[i]` is branch i's score; `available[i]` is false when that
// sensor produced no reading (no PPG contact, mic buffer not filled) -- which is
// NOT the same as a low score and must not be faked with a zero.
// `now_s` must be monotonic: derive it from a 64-bit tick counter, not from
// millis(), which wraps at 49.7 days and would rewind this state machine.
// Returns true on the tick an alert should be raised.
static inline bool {prefix}_update({prefix}_state *s, float now_s,
                                   const float *scores, const bool *available) {{
  float available_weight = 0.0f, vote_weight = 0.0f;
  for (int i = 0; i < {upper}_N_BRANCHES; i++) {{
    if (!available[i]) {{
      s->last_fire_s[i] = -INFINITY;  // a fire from a sensor now silent is not evidence
      continue;
    }}
    available_weight += {prefix}_weights[i];
    if (scores[i] > {prefix}_thresholds[i]) s->last_fire_s[i] = now_s;
    if (now_s - s->last_fire_s[i] <= {prefix}_holds_s[i] + {upper}_TOL)
      vote_weight += {prefix}_weights[i];
  }}

  float required = {prefix}_votes_required;
  if ({prefix}_proportional && {prefix}_total_weight > 0.0f) {{
    float scaled = {prefix}_votes_required * (available_weight / {prefix}_total_weight);
    required = scaled > {prefix}_min_votes ? scaled : {prefix}_min_votes;
  }}

  s->vote_weight = vote_weight;
  s->votes_required = required;
  s->degraded = available_weight < {prefix}_total_weight - {upper}_TOL;
  s->can_alarm = available_weight >= required - {upper}_TOL;

  bool consensus = vote_weight >= required - {upper}_TOL;
  if (consensus) {{
    if (!s->in_consensus) {{
      s->in_consensus = true;
      s->consensus_since_s = now_s;
    }}
  }} else {{
    s->in_consensus = false;
  }}

  bool alarm = consensus &&
               (now_s - s->consensus_since_s) >= {prefix}_sustain_s - {upper}_TOL &&
               now_s >= s->cooldown_until_s - {upper}_TOL;
  if (alarm) {{
    s->cooldown_until_s = now_s + {prefix}_cooldown_s;
    s->in_consensus = false;  // re-earn the sustain before alerting again
  }}
  return alarm;
}}
"""
