"""Count what is actually in a corpus before anything is modelled on it.

The companion repository trained and exported a four-class model whose `gunshot`
class contained zero samples, and shipped it: the softmax had an output that no
gradient had ever touched. The defect was not the missing data, it was that
nothing counted the data before training consumed it. This module is that count,
and it is deliberately a gate rather than a printout -- `require` raises.

The checks split into two kinds, and the distinction is the point:

**Fatal** checks are ones where continuing produces a number that looks fine and
means nothing -- no positives to detect, one subject holding all the positives
so no subject-disjoint split exists, or an accelerometer scaled wrongly by a
factor of 64. These stop the run.

**Warnings** are ones where the work is still valid but narrower than it looks,
such as thin ground-truth coverage or too few hours of negatives to quote a
false-alarm rate against. These print and continue.

The accelerometer check earns its place. `e4.E4_ACC_LSB_PER_G` encodes an
assumption about how the corpus stores acceleration, and nothing in the archive
states it. If it is wrong the arrays still load, still window, and still train;
the only visible symptom is that a wrist at rest reads 64 g. So the gate asserts
the physics instead of trusting the constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import HRWindowSet

# A wrist at rest reads ~1 g. These bounds are wide enough for a corpus that
# stores acceleration in some other but still sane unit, and narrow enough to
# catch the 1/64 scaling error by two orders of magnitude.
ACC_G_RANGE = (0.5, 2.0)

# Plausible resting-to-stressed human heart rate, for sanity-checking a
# reference track rather than for filtering it.
HR_REF_RANGE_BPM = (40.0, 120.0)

# Below this, a stress/non-stress separation measured on the corpus is a
# statement about a handful of people.
MIN_POSITIVE_WINDOWS = 100
MIN_POSITIVE_SUBJECTS = 3

# A 6 alarms/hour budget cannot be measured against less than this.
MIN_NEGATIVE_HOURS = 1.0

# Ground-truth HR coverage below this means most windows cannot be scored
# against truth, which is the whole reason WESAD's ECG is read.
MIN_REFERENCE_FRACTION = 0.5


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool

    @property
    def mark(self) -> str:
        if self.ok:
            return "ok"
        return "FAIL" if self.fatal else "warn"


def gate(hrws: HRWindowSet) -> list[Check]:
    """Run every availability check and return the results unraised."""
    checks: list[Check] = []

    def add(name, ok, detail, fatal=False):
        checks.append(Check(name, bool(ok), detail, fatal))

    n = len(hrws)
    pos = hrws.y_stress == 1
    neg = hrws.y_stress == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())

    add(
        "positives exist",
        n_pos >= MIN_POSITIVE_WINDOWS,
        f"{n_pos:,} stress windows (need >= {MIN_POSITIVE_WINDOWS:,})",
        fatal=True,
    )
    add(
        "negatives exist",
        n_neg > 0,
        f"{n_neg:,} non-stress windows",
        fatal=True,
    )

    pos_subjects = sorted(set(hrws.subject[pos].tolist()))
    add(
        "positives span subjects",
        len(pos_subjects) >= MIN_POSITIVE_SUBJECTS,
        f"{len(pos_subjects)} subjects carry a positive "
        f"(need >= {MIN_POSITIVE_SUBJECTS}) -- a subject-disjoint split needs "
        f"positives on both sides",
        fatal=True,
    )

    neg_hours = hrws.subset(neg).hours() if n_neg else 0.0
    add(
        "negative hours",
        neg_hours >= MIN_NEGATIVE_HOURS,
        f"{neg_hours:.2f} h of negatives (need >= {MIN_NEGATIVE_HOURS} h to "
        f"quote alarms/hour)",
    )

    # -- physics, not bookkeeping ---------------------------------------------
    acc_mag = np.linalg.norm(hrws.X[:, :, 1:], axis=2)
    med_g = float(np.median(acc_mag))
    add(
        "acc scaling",
        ACC_G_RANGE[0] <= med_g <= ACC_G_RANGE[1],
        f"median |acc| = {med_g:.3f} g (expect ~1.0; a value near 64 means "
        f"e4.E4_ACC_LSB_PER_G is wrong for this release)",
        fatal=True,
    )

    bvp_std = float(np.median(hrws.X[:, :, 0].std(axis=1)))
    add(
        "bvp varies",
        bvp_std > 0.0,
        f"median per-window BVP std = {bvp_std:.4g}",
        fatal=True,
    )

    # -- ground truth ---------------------------------------------------------
    has_ref = hrws.has_reference
    frac = float(has_ref.mean()) if n else 0.0
    add(
        "hr_ref coverage",
        frac >= MIN_REFERENCE_FRACTION,
        f"{int(has_ref.sum()):,}/{n:,} windows carry a reference HR ({frac:.1%})",
    )
    if has_ref.any():
        med_bpm = float(np.median(hrws.hr_ref[has_ref]))
        add(
            "hr_ref plausible",
            HR_REF_RANGE_BPM[0] <= med_bpm <= HR_REF_RANGE_BPM[1],
            f"median reference HR = {med_bpm:.1f} bpm",
        )

    # -- provenance -----------------------------------------------------------
    collisions = [
        s for s in hrws.subjects if len(set(hrws.dataset[hrws.subject == s].tolist())) > 1
    ]
    add(
        "subject ids unique",
        not collisions,
        f"{len(collisions)} subject id(s) appear in more than one corpus: "
        f"{collisions[:5]}",
        fatal=True,
    )

    return checks


def report(hrws: HRWindowSet) -> str:
    """The gate's findings plus the condition breakdown, as one block of text."""
    checks = gate(hrws)
    width = max(len(c.name) for c in checks)
    lines = ["Availability gate", "-----------------"]
    lines += [f"  [{c.mark:>4s}] {c.name:<{width}s}  {c.detail}" for c in checks]

    lines += ["", "Windows by condition and corpus"]
    conds = sorted(set(hrws.condition.tolist()))
    tags = hrws.tags
    lines.append(
        f"  {'condition':<14s}" + "".join(f"{t:>12s}" for t in tags) + f"{'stress':>9s}"
    )
    for cond in conds:
        m = hrws.condition == cond
        row = f"  {cond:<14s}"
        for t in tags:
            row += f"{int((m & (hrws.dataset == t)).sum()):>12,}"
        flag = "yes" if hrws.y_stress[m].max(initial=-1) == 1 else "no"
        lines.append(row + f"{flag:>9s}")

    lines.append("")
    lines.append(f"  {'total':<14s}" + "".join(
        f"{int((hrws.dataset == t).sum()):>12,}" for t in tags
    ))

    failures = [c for c in checks if not c.ok and c.fatal]
    warnings = [c for c in checks if not c.ok and not c.fatal]
    lines.append("")
    if failures:
        lines.append(f"VERDICT: {len(failures)} fatal check(s) failed -- do not model on this.")
    elif warnings:
        lines.append(f"VERDICT: usable, with {len(warnings)} warning(s) above.")
    else:
        lines.append("VERDICT: every check passed.")
    return "\n".join(lines)


def require(hrws: HRWindowSet) -> HRWindowSet:
    """Raise unless every fatal check passes. Returns the input for chaining."""
    failures = [c for c in gate(hrws) if not c.ok and c.fatal]
    if failures:
        detail = "\n".join(f"  - {c.name}: {c.detail}" for c in failures)
        raise RuntimeError(
            f"availability gate failed {len(failures)} fatal check(s):\n{detail}\n"
            f"Run availability.report() for the full breakdown."
        )
    return hrws
