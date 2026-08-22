"""Dataset loaders. Every one returns a `WindowSet` in the same channel order.

Adding a corpus means adding one module here with `download(data_dir) -> Path`
and `load(root, ...) -> WindowSet`, then registering it in `LOADERS`. Nothing
downstream -- splits, augmentation, training, scoring, export -- needs to know
which corpora exist.
"""

from __future__ import annotations

from pathlib import Path

from . import motionsense, uci
from .base import (
    ACC,
    CHANNELS,
    CLASSES,
    CLS,
    GYR,
    N_CHANNELS,
    N_CLASSES,
    UNKNOWN,
    WindowSet,
)

__all__ = [
    "ACC",
    "CHANNELS",
    "CLASSES",
    "CLS",
    "GYR",
    "LOADERS",
    "N_CHANNELS",
    "N_CLASSES",
    "UNKNOWN",
    "WindowSet",
    "load_all",
    "motionsense",
    "uci",
]

# tag -> module. Each module exposes download() and load().
LOADERS = {
    uci.TAG: uci,
    motionsense.TAG: motionsense,
}


def load_all(
    data_dir: str | Path,
    tags: list[str] | None = None,
    win: int = 128,
    stride: int = 64,
    download: bool = True,
) -> WindowSet:
    """Download (if needed) and load the named corpora into one WindowSet.

    Parameters
    ----------
    data_dir : path
        Where archives are downloaded and extracted.
    tags : list of str, optional
        Corpus tags to load. Defaults to everything registered.
    win, stride : int
        Windowing for corpora that ship continuous recordings. Ignored by
        corpora that are natively windowed (UCI), which will raise if `win`
        disagrees with their native length.
    download : bool
        If False, expect the data to already be extracted and fail otherwise.
    """
    data_dir = Path(data_dir)
    tags = list(LOADERS) if tags is None else tags

    unknown = [t for t in tags if t not in LOADERS]
    if unknown:
        raise ValueError(f"unknown corpus tag(s) {unknown}; have {sorted(LOADERS)}")

    parts = []
    for tag in tags:
        mod = LOADERS[tag]
        print(f"[{tag}]")
        root = mod.download(data_dir) if download else _expect_extracted(data_dir, mod)
        # Natively-windowed corpora take no windowing arguments.
        kwargs = {} if getattr(mod, "WIN", None) else {"win": win, "stride": stride}
        parts.append(mod.load(root, **kwargs))

    out = WindowSet.concat(parts)
    print(f"\nmerged:\n{out.summary()}")
    return out


def _expect_extracted(data_dir: Path, mod) -> Path:
    from .download import find_root

    return find_root(data_dir, mod.MARKER)
