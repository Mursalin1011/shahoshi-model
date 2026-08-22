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


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 20
    monitor: str = "val_macro_f1"
    class_weighted: bool = True
    fall_positive_weight: float = 1.0
    seed: int = 42


@dataclass
class QuantizeConfig:
    n_representative: int = 512
    seed: int = 42


@dataclass
class ScoringConfig:
    shrinkage: float = 1e-3
    # Operating points in alarms per hour, not percentiles -- see scoring.py.
    target_fars: tuple[float, ...] = (0.5, 1.0, 2.0, 6.0)
    # Which FAR budget the exported thresholds are set to.
    export_far: float = 1.0


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
            if key in sub_types:
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
