"""Experiment configuration, loaded from YAML.

Every knob that changes a result lives here rather than in a notebook cell, for
one reason: the ablations this project needs -- augmentation on/off, rotation
severity, leave-one-dataset-out versus merged, fall head on/off -- have to be
comparable across runs, and a knob edited in a cell is a knob nobody can
reconstruct three weeks later when the number goes into a paper.

`Config.to_dict()` feeds `shahoshi.manifest`, so the config that produced a
result is stored next to the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data"
    datasets: list[str] = field(default_factory=lambda: ["uci", "ms"])
    win: int = 128
    stride: int = 64
    fs: int = 50
    download: bool = True

    @property
    def hop_seconds(self) -> float:
        """Seconds between consecutive inferences. Drives the alarms-per-hour
        arithmetic in `shahoshi.scoring`, so it must match the firmware."""
        return self.stride / self.fs


@dataclass
class SplitConfig:
    protocol: str = "subject"          # "subject" | "leave_dataset_out"
    seed: int = 42
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15)
    test_dataset: str | list[str] | None = None   # required for leave_dataset_out


@dataclass
class AugmentConfig:
    times: int = 0                    # 0 disables augmentation entirely
    rotation_deg: float = 180.0
    scale_sigma: float = 0.1
    jitter_sigma: float = 0.02
    warp_ratio: float = 0.15
    dropout_p: float = 0.0
    seed: int = 42


@dataclass
class ModelConfig:
    embed_dim: int = 64
    width: float = 1.0
    dropout: float = 0.3
    with_fall_head: bool = False
    # Cap activations at 6. Unbounded ReLU is free in float and expensive in
    # int8: TFLite sizes activation scales from observed min/max, so a long
    # tail spends the range on outliers. See quantize.activation_ranges.
    bounded_relu: bool = False


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 20
    monitor: str = "val_macro_f1"
    class_weighted: bool = True
    # tf.random.set_seed alone leaves cuDNN free to pick nondeterministic
    # convolution kernels; two runs of identical code gave 0.9024 and 0.8758.
    deterministic: bool = True
    fall_positive_weight: float = 1.0
    seed: int = 42


@dataclass
class QuantizeConfig:
    n_representative: int = 512
    # int8 outputs as well as int8 kernels. Turning this off leaves a trailing
    # DEQUANTIZE and float32 outputs while every kernel stays integer -- see
    # quantize.to_int8. An int8 softmax has ~1/256 resolution, so the argmax
    # can flip whenever the top two classes are within one step.
    output_int8: bool = True
    seed: int = 42


@dataclass
class ScoringConfig:
    shrinkage: float = 1e-3
    # Operating points in alarms per hour, not percentiles -- see scoring.py.
    target_fars: tuple[float, ...] = (0.5, 1.0, 2.0, 6.0)
    # Which FAR budget the exported thresholds are set to.
    export_far: float = 1.0


@dataclass
class BranchConfig:
    """One voter in the consensus rule, as configured rather than as calibrated.

    `far_budget` is the knob; `threshold` is what calibration derives from it on
    the run's own normal data, and is left None here so a stale number copied
    between configs cannot masquerade as a calibrated one. Set it only to pin a
    branch that has no scores to calibrate against (a hardware amplitude trip).
    """

    name: str
    far_budget: float = 6.0        # alarms per hour for this branch *alone*
    hold_seconds: float = 4.0      # coincidence window
    weight: float = 1.0
    threshold: float | None = None
    # False for a branch that is specified but not yet built, which is the
    # normal state of this project: it keeps the rule's arithmetic honest
    # (a 2-of-3 vote with one implemented branch cannot fire) instead of
    # letting the config imply three working sensors.
    implemented: bool = False


@dataclass
class FusionConfig:
    """The hand-designed 2-of-3 vote. See `shahoshi.fusion` for the semantics."""

    branches: list[BranchConfig] = field(
        default_factory=lambda: [
            BranchConfig(name="movement", far_budget=6.0, weight=1.0, implemented=True),
            BranchConfig(name="hr", far_budget=6.0, weight=1.0),
            BranchConfig(name="audio", far_budget=6.0, weight=1.0),
        ]
    )
    votes_required: float = 2.0
    sustain_seconds: float = 4.0
    cooldown_seconds: float = 30.0
    degraded_policy: str = "strict"
    min_votes_required: float = 1.0
    # An alarm later than this after an event onset is not a rescue; it is the
    # window `fusion.evaluate` scores event-level recall over.
    latency_budget_seconds: float = 10.0
    # Fused budget the per-branch budgets are meant to buy, for reporting
    # against `fusion.fused_far_upper_bound`.
    target_fused_far: float = 1.0

    @classmethod
    def from_raw(cls, raw: dict) -> FusionConfig:
        raw = dict(raw or {})
        branches = raw.pop("branches", None)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) {sorted(unknown)} under 'fusion'; "
                f"expected {sorted(known)}"
            )
        cfg = cls(**raw)
        if branches is not None:
            branch_known = {f.name for f in fields(BranchConfig)}
            parsed = []
            for entry in branches:
                bad = set(entry) - branch_known
                if bad:
                    raise ValueError(
                        f"unknown key(s) {sorted(bad)} under a fusion branch; "
                        f"expected {sorted(branch_known)}"
                    )
                parsed.append(BranchConfig(**entry))
            cfg.branches = parsed
        return cfg

    @property
    def total_weight(self) -> float:
        return float(sum(b.weight for b in self.branches))

    @property
    def implemented_weight(self) -> float:
        """Vote weight actually reachable today. Below `votes_required`, the
        alarm cannot fire on hardware -- which is the current state of Stage 3
        and should be visible in the config, not discovered on a wrist."""
        return float(sum(b.weight for b in self.branches if b.implemented))

    def reachability_note(self) -> str:
        """One line saying whether the configured vote can fire on today's hardware.

        Not a validation error -- a 2-of-3 rule specified before the second and
        third sensors exist is the correct order to build in. It is a line for
        the run report, so the gap is read off a manifest rather than inferred
        from a device that never alarms.
        """
        built = [b.name for b in self.branches if b.implemented]
        pending = [b.name for b in self.branches if not b.implemented]
        if self.implemented_weight >= self.votes_required - 1e-9:
            return (
                f"fusion reachable: {self.implemented_weight:g} of "
                f"{self.votes_required:g} required vote weight is implemented "
                f"({', '.join(built)})"
            )
        return (
            f"fusion UNREACHABLE on hardware: implemented branches "
            f"{built or ['none']} carry {self.implemented_weight:g} weight against "
            f"{self.votes_required:g} required. Pending: {pending or ['none']}. "
            f"Under the 'strict' policy the alarm cannot fire, and a zero "
            f"false-alarm rate from this configuration means nothing."
        )

    def validate(self, hop_seconds: float) -> None:
        if not self.branches:
            raise ValueError("fusion needs at least one branch")
        names = [b.name for b in self.branches]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate fusion branch name(s) {dupes}")
        if self.degraded_policy not in ("strict", "proportional"):
            raise ValueError(
                f"fusion.degraded_policy must be 'strict' or 'proportional'; "
                f"got {self.degraded_policy!r}"
            )
        if self.votes_required <= 0:
            raise ValueError("fusion.votes_required must be positive")
        if self.votes_required > self.total_weight + 1e-9:
            raise ValueError(
                f"fusion.votes_required ({self.votes_required}) exceeds the total "
                f"branch weight ({self.total_weight}): no combination of votes can "
                f"reach it, so the alarm is unreachable by construction"
            )
        for b in self.branches:
            if b.weight <= 0:
                raise ValueError(f"fusion branch {b.name!r}: weight must be positive")
            if b.hold_seconds <= 0:
                raise ValueError(
                    f"fusion branch {b.name!r}: hold_seconds must be positive, else "
                    f"two asynchronous branches can never overlap"
                )
            if b.far_budget < 0:
                raise ValueError(f"fusion branch {b.name!r}: far_budget must be >= 0")
        if self.sustain_seconds < 0:
            raise ValueError("fusion.sustain_seconds must be non-negative")
        if 0 < self.sustain_seconds < hop_seconds:
            raise ValueError(
                f"fusion.sustain_seconds ({self.sustain_seconds}) is shorter than one "
                f"inference hop ({hop_seconds:.2f} s), so a single consenting tick "
                f"alarms and the sustain requirement does nothing. Set it to 0 to "
                f"disable sustain deliberately, or to a multiple of the hop."
            )
        if self.cooldown_seconds < 0:
            raise ValueError("fusion.cooldown_seconds must be non-negative")
        if self.min_votes_required <= 0:
            raise ValueError("fusion.min_votes_required must be positive")
        if self.latency_budget_seconds <= 0:
            raise ValueError("fusion.latency_budget_seconds must be positive")
        if self.target_fused_far <= 0:
            raise ValueError("fusion.target_fused_far must be positive")


@dataclass
class Config:
    name: str = "movement-baseline"
    notes: str = ""
    out_dir: str = "artifacts"
    report_dir: str = "reports"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # -- io -------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> Config:
        """Load YAML, then apply dotted-path keyword overrides.

        Overrides use dots for nesting, so a notebook can vary one knob without
        writing a new file::

            Config.load("configs/movement.yaml", **{"augment.times": 2})
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = cls.from_dict(raw)
        for dotted, value in overrides.items():
            cfg.set(dotted, value)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        kwargs: dict[str, Any] = {}
        sub_types = {
            "data": DataConfig,
            "split": SplitConfig,
            "augment": AugmentConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "quantize": QuantizeConfig,
            "scoring": ScoringConfig,
        }
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown config key(s) {sorted(unknown)}; expected {sorted(known)}"
            )
        for key, value in raw.items():
            if key == "fusion":
                # Its `branches` entry is a list of dataclasses, so it parses
                # itself rather than through the flat **value path below.
                kwargs[key] = FusionConfig.from_raw(value or {})
            elif key in sub_types:
                sub_known = {f.name for f in fields(sub_types[key])}
                sub_unknown = set(value or {}) - sub_known
                if sub_unknown:
                    raise ValueError(
                        f"unknown key(s) {sorted(sub_unknown)} under {key!r}; "
                        f"expected {sorted(sub_known)}"
                    )
                kwargs[key] = sub_types[key](**(value or {}))
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def set(self, dotted: str, value: Any) -> None:
        """Set a nested field by dotted path, failing loudly on a typo."""
        parts = dotted.split(".")
        target: Any = self
        for p in parts[:-1]:
            if not hasattr(target, p):
                raise KeyError(f"no config section {p!r} in {dotted!r}")
            target = getattr(target, p)
        if not hasattr(target, parts[-1]):
            raise KeyError(f"no config field {dotted!r}")
        setattr(target, parts[-1], value)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    # -- validation -----------------------------------------------------------

    def validate(self) -> None:
        if self.split.protocol not in ("subject", "leave_dataset_out"):
            raise ValueError(
                f"split.protocol must be 'subject' or 'leave_dataset_out'; "
                f"got {self.split.protocol!r}"
            )
        if self.split.protocol == "leave_dataset_out" and not self.split.test_dataset:
            raise ValueError(
                "split.protocol='leave_dataset_out' needs split.test_dataset -- "
                "which corpus is being held out?"
            )
        if self.data.stride > self.data.win:
            raise ValueError(
                f"stride ({self.data.stride}) exceeds win ({self.data.win}): "
                "windows would skip samples entirely"
            )
        if self.model.with_fall_head and "sisfall" not in self.data.datasets:
            raise ValueError(
                "model.with_fall_head is on but no corpus with fall labels is "
                "loaded, so the fall head would train on nothing but masked rows"
            )
        if self.scoring.export_far <= 0:
            raise ValueError("scoring.export_far must be positive")
        self.fusion.validate(self.data.hop_seconds)
