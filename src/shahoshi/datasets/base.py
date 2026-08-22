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


@dataclass
class WindowSet:
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
        for name in ("y_act", "y_fall", "subject", "dataset"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")

        self.X = np.ascontiguousarray(self.X, dtype=np.float32)
        self.y_act = np.asarray(self.y_act, dtype=np.int64)
        self.y_fall = np.asarray(self.y_fall, dtype=np.int8)
        self.subject = np.asarray(self.subject, dtype=object).astype(str)
        self.dataset = np.asarray(self.dataset, dtype=object).astype(str)

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
            f" | {len(set(self.subject.tolist()))} subjects"
            f" | datasets: {sorted(set(self.dataset.tolist()))}"
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
