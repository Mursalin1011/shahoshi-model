"""The one data contract every dataset loader in this package returns.

Design notes
------------
`lay` is deliberately absent from CLASSES. Gravity is removed from the
acceleration channels of every corpus we use (see `shahoshi.signal`), and the
gravity vector is the only thing that distinguishes lying down from sitting.
Keeping the class meant carrying a label that is unidentifiable by construction:
in the pre-refactor baseline it scored 0.25 recall in float32 and 0.09 in int8,
and its class weight distorted the loss for every other class.

Both label fields use -1 as "unknown", not as a class. This matters more than it
looks: SisFall's ADLs include transition motions ("sit down, then stand up")
that map onto no single activity label but are unambiguously *not falls*. With
-1 masking they can train the fall head without corrupting the activity head,
which is the whole reason the Stage 1 model is multi-task rather than one
softmax with a `fall` class bolted on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Unified activity vocabulary. Index == label.
CLASSES: tuple[str, ...] = ("walk", "upstairs", "downstairs", "sit", "stand", "jog")
CLS: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES: int = len(CLASSES)

# Unified channel vocabulary. Index == channel.
#   acc_* : linear acceleration, gravity removed, in g
#   gyr_* : angular velocity, in rad/s
CHANNELS: tuple[str, ...] = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
N_CHANNELS: int = len(CHANNELS)

ACC = slice(0, 3)
GYR = slice(3, 6)

UNKNOWN = -1

# Unified channel vocabulary for the heart-rate branch. Deliberately a separate
# vocabulary from CHANNELS, and the difference is not cosmetic:
#
#   * `bvp` is a photoplethysmogram, not an inertial channel.
#   * `acc_*` here is RAW acceleration WITH GRAVITY, in g. CHANNELS' acc_* has
#     gravity removed. The HR branch wants gravity in: the accelerometer is
#     there only as a motion-artefact gate, and |acc| ~ 1 g at rest is the
#     cheapest available check that the corpus's ACC scaling was read correctly.
#
# So an HRWindowSet and a WindowSet must never be concatenated, and the two
# distinct names are what stops that from happening quietly.
HR_CHANNELS: tuple[str, ...] = ("bvp", "acc_x", "acc_y", "acc_z")
N_HR_CHANNELS: int = len(HR_CHANNELS)

# The device rate, not either corpus's native rate. WESAD and PPG-DaLiA are both
# Empatica E4 (BVP 64 Hz, ACC 32 Hz); the target is a MAX30102 read at 100 sps
# (MULTIMODAL_PLAN.md section 8). Loaders resample to this so that `hr.py` is
# developed at exactly the rate it will run at on the device, rather than
# leaving a 64 -> 100 Hz conversion as an untested step at deployment.
HR_FS: int = 100

# Heart rate lives in 0.5-4 Hz (30-240 bpm). Recorded here so the loaders, the
# availability gate and hr.py all quote one number.
HR_BAND_HZ: tuple[float, float] = (0.5, 4.0)


class ProvenanceBase:
    """Fields every corpus-derived stack carries, and the checks they need.

    `WindowSet` and `HRWindowSet` are separate dataclasses sharing this base
    rather than one widened class. `WindowSet.__post_init__`'s assertion that X
    is ``(n, win, 6)`` in `CHANNELS` order has caught real bugs; forcing a
    ``(n, win, 4)`` PPG stack through it would mean either weakening that check
    or lying about the channel vocabulary. What the two genuinely share is
    provenance -- which subject, which corpus -- and it is provenance that the
    subject-disjoint splitting in `splits.py` depends on, so that is what is
    factored out and nothing more.
    """

    subject: np.ndarray
    dataset: np.ndarray

    def _coerce_provenance(self, n: int) -> None:
        """Validate lengths and force both provenance fields to string arrays."""
        for name in ("subject", "dataset"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")
            setattr(self, name, np.asarray(arr, dtype=object).astype(str))

    @property
    def subjects(self) -> list[str]:
        """Sorted unique subject ids. Corpus-prefixed, so never ambiguous."""
        return sorted(set(self.subject.tolist()))

    @property
    def tags(self) -> list[str]:
        """Sorted unique corpus tags present in this stack."""
        return sorted(set(self.dataset.tolist()))


@dataclass
class WindowSet(ProvenanceBase):
    """A stack of fixed-length multichannel windows with per-window metadata.

    Attributes
    ----------
    X : (n, win, 6) float32
        Windows in the unified channel order `CHANNELS`.
    y_act : (n,) int64
        Activity label in [0, N_CLASSES), or UNKNOWN (-1) if this window has no
        meaningful activity label. Masked out of the activity loss.
    y_fall : (n,) int8
        1 = window contains a fall impact, 0 = confirmed non-fall,
        UNKNOWN (-1) = not known (e.g. every window from UCI/MotionSense, which
        contain no falls *and* no evidence either way about violent events).
        Masked out of the fall loss.
    subject : (n,) str
        Globally unique subject id, prefixed by corpus (``uci_12``, ``ms_sub_3``,
        ``sisfall_SA07``) so that ids from different corpora can never collide
        and silently leak across a subject-disjoint split.
    dataset : (n,) str
        Corpus tag, for leave-one-dataset-out evaluation.
    """

    X: np.ndarray
    y_act: np.ndarray
    y_fall: np.ndarray
    subject: np.ndarray
    dataset: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.X)
        if self.X.ndim != 3 or self.X.shape[2] != N_CHANNELS:
            raise ValueError(
                f"X must be (n, win, {N_CHANNELS}); got {self.X.shape}. "
                "Channel order must be shahoshi.datasets.base.CHANNELS."
            )
        for name in ("y_act", "y_fall"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")
        self._coerce_provenance(n)

        self.X = np.ascontiguousarray(self.X, dtype=np.float32)
        self.y_act = np.asarray(self.y_act, dtype=np.int64)
        self.y_fall = np.asarray(self.y_fall, dtype=np.int8)

        bad = self.y_act[(self.y_act != UNKNOWN) & ~np.isin(self.y_act, range(N_CLASSES))]
        if bad.size:
            raise ValueError(f"y_act has out-of-range labels: {np.unique(bad)}")
        bad = self.y_fall[~np.isin(self.y_fall, (UNKNOWN, 0, 1))]
        if bad.size:
            raise ValueError(f"y_fall must be in {{-1,0,1}}; got {np.unique(bad)}")

    # -- basics ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.X)

    @property
    def win(self) -> int:
        return self.X.shape[1]

    def subset(self, mask: np.ndarray) -> WindowSet:
        """Index every field with the same boolean mask or integer index array."""
        return WindowSet(
            X=self.X[mask],
            y_act=self.y_act[mask],
            y_fall=self.y_fall[mask],
            subject=self.subject[mask],
            dataset=self.dataset[mask],
        )

    @staticmethod
    def concat(parts: list[WindowSet]) -> WindowSet:
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("nothing to concatenate")
        wins = {p.win for p in parts}
        if len(wins) != 1:
            raise ValueError(f"window lengths disagree: {sorted(wins)}")
        return WindowSet(
            X=np.concatenate([p.X for p in parts]),
            y_act=np.concatenate([p.y_act for p in parts]),
            y_fall=np.concatenate([p.y_fall for p in parts]),
            subject=np.concatenate([p.subject for p in parts]),
            dataset=np.concatenate([p.dataset for p in parts]),
        )

    # -- reporting -------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"{len(self):,} windows of {self.win} steps x {N_CHANNELS} ch"
            f" | {len(self.subjects)} subjects"
            f" | datasets: {self.tags}"
        ]
        lines.append("  activity:")
        for i, c in enumerate(CLASSES):
            n = int((self.y_act == i).sum())
            if n:
                lines.append(f"    {c:<11s} {n:>7,}")
        n_unk = int((self.y_act == UNKNOWN).sum())
        if n_unk:
            lines.append(f"    {'(unlabelled)':<11s} {n_unk:>7,}")
        lines.append("  fall:")
        for val, name in ((1, "fall"), (0, "non-fall"), (UNKNOWN, "(unknown)")):
            n = int((self.y_fall == val).sum())
            if n:
                lines.append(f"    {name:<12s} {n:>7,}")
        return "\n".join(lines)


# Condition name used when a corpus records no protocol condition at all.
UNLABELLED = "unlabelled"


@dataclass
class HRWindowSet(ProvenanceBase):
    """A stack of wrist-PPG windows with per-window stress labels and HR truth.

    Attributes
    ----------
    X : (n, win, 4) float32
        Windows in `HR_CHANNELS` order at `fs` Hz. `acc_*` retains gravity --
        see the HR_CHANNELS comment; this is not the WindowSet convention.
    y_stress : (n,) int8
        1 = stress, 0 = non-stress, UNKNOWN (-1) = no basis for either.
        **This is a derived field.** It is a binarization of `condition` under a
        mapping each loader states explicitly, and the binarization is a
        modelling choice, not a fact about the corpus. `condition` is kept
        alongside it precisely so that choice can be revisited without
        reloading 18 GB.
    condition : (n,) str
        The corpus's own condition name for this window -- 'stress',
        'amusement', 'meditation', 'baseline' for WESAD; the activity name for
        PPG-DaLiA. Windows spanning two conditions are dropped by the loaders
        rather than assigned a majority label, so every entry here describes the
        whole window.
    hr_ref : (n,) float32
        Ground-truth heart rate in bpm, or NaN where none is available. From
        chest ECG for WESAD and from the shipped ECG-derived labels for DaLiA.
        This is truth to score against, never a model input.
    subject : (n,) str
        Corpus-prefixed subject id (``wesad_S2``, ``dalia_S7``).
    dataset : (n,) str
        Corpus tag.
    fs : int
        Sample rate of X. Carried explicitly rather than assumed, because the
        one thing that must not drift here is the rate hr.py filters at.
    """

    X: np.ndarray
    y_stress: np.ndarray
    condition: np.ndarray
    hr_ref: np.ndarray
    subject: np.ndarray
    dataset: np.ndarray
    fs: int = HR_FS

    def __post_init__(self) -> None:
        n = len(self.X)
        if self.X.ndim != 3 or self.X.shape[2] != N_HR_CHANNELS:
            raise ValueError(
                f"X must be (n, win, {N_HR_CHANNELS}); got {self.X.shape}. "
                "Channel order must be shahoshi.datasets.base.HR_CHANNELS."
            )
        for name in ("y_stress", "condition", "hr_ref"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")
        self._coerce_provenance(n)

        self.X = np.ascontiguousarray(self.X, dtype=np.float32)
        self.y_stress = np.asarray(self.y_stress, dtype=np.int8)
        self.condition = np.asarray(self.condition, dtype=object).astype(str)
        self.hr_ref = np.asarray(self.hr_ref, dtype=np.float32)
        self.fs = int(self.fs)

        bad = self.y_stress[~np.isin(self.y_stress, (UNKNOWN, 0, 1))]
        if bad.size:
            raise ValueError(f"y_stress must be in {{-1,0,1}}; got {np.unique(bad)}")
        if self.fs <= 0:
            raise ValueError(f"fs must be positive; got {self.fs}")

    # -- basics ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.X)

    @property
    def win(self) -> int:
        return self.X.shape[1]

    @property
    def win_seconds(self) -> float:
        return self.win / self.fs

    @property
    def has_reference(self) -> np.ndarray:
        """Boolean mask of windows carrying a ground-truth HR."""
        return ~np.isnan(self.hr_ref)

    def hours(self) -> float:
        """Total windowed duration in hours, counting each window once.

        Overlapping windows mean this is not wall-clock recording time. It is
        the denominator that alarms-per-hour is quoted against, which is the
        number that actually matters downstream, so it is what gets reported.
        """
        return len(self) * self.win_seconds / 3600.0

    def subset(self, mask: np.ndarray) -> HRWindowSet:
        """Index every per-window field with the same mask or index array."""
        return HRWindowSet(
            X=self.X[mask],
            y_stress=self.y_stress[mask],
            condition=self.condition[mask],
            hr_ref=self.hr_ref[mask],
            subject=self.subject[mask],
            dataset=self.dataset[mask],
            fs=self.fs,
        )

    @staticmethod
    def concat(parts: list[HRWindowSet]) -> HRWindowSet:
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("nothing to concatenate")
        wins = {p.win for p in parts}
        if len(wins) != 1:
            raise ValueError(f"window lengths disagree: {sorted(wins)}")
        rates = {p.fs for p in parts}
        if len(rates) != 1:
            raise ValueError(
                f"sample rates disagree: {sorted(rates)}. Every loader must "
                f"resample to base.HR_FS before returning."
            )
        return HRWindowSet(
            X=np.concatenate([p.X for p in parts]),
            y_stress=np.concatenate([p.y_stress for p in parts]),
            condition=np.concatenate([p.condition for p in parts]),
            hr_ref=np.concatenate([p.hr_ref for p in parts]),
            subject=np.concatenate([p.subject for p in parts]),
            dataset=np.concatenate([p.dataset for p in parts]),
            fs=parts[0].fs,
        )

    # -- reporting -------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"{len(self):,} windows of {self.win} steps x {N_HR_CHANNELS} ch"
            f" @ {self.fs} Hz ({self.win_seconds:.1f} s, {self.hours():.2f} h)"
            f" | {len(self.subjects)} subjects"
            f" | datasets: {self.tags}"
        ]
        lines.append("  stress:")
        for val, name in ((1, "stress"), (0, "non-stress"), (UNKNOWN, "(unknown)")):
            n = int((self.y_stress == val).sum())
            if n:
                lines.append(f"    {name:<14s} {n:>7,}")
        lines.append("  condition:")
        for cond in sorted(set(self.condition.tolist())):
            n = int((self.condition == cond).sum())
            lines.append(f"    {cond:<14s} {n:>7,}")
        n_ref = int(self.has_reference.sum())
        lines.append(
            f"  hr_ref: {n_ref:,}/{len(self):,} windows "
            f"({100.0 * n_ref / max(len(self), 1):.1f}%)"
        )
        return "\n".join(lines)
