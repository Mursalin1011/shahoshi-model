"""Dataset loaders, in two families that deliberately do not mix.

The movement corpora return a `WindowSet` in `CHANNELS` order; the heart-rate
corpora return an `HRWindowSet` in `HR_CHANNELS` order. They are registered
separately in `LOADERS` and `HR_LOADERS` and merged by `load_all` and `load_hr`
respectively, because the two stacks carry different channels, different labels
and -- see `base.HR_CHANNELS` -- different conventions about gravity.

Adding a corpus means adding one module here with `download(data_dir) -> Path`
and `load(root, ...)`, then registering it in the matching table. Nothing
downstream -- splits, augmentation, training, scoring, export -- needs to know
which corpora exist.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from . import availability, cache, dalia, e4, ecg, motionsense, uci, wesad
from .base import (
    ACC,
    CHANNELS,
    CLASSES,
    CLS,
    GYR,
    HR_BAND_HZ,
    HR_CHANNELS,
    HR_FS,
    N_CHANNELS,
    N_CLASSES,
    N_HR_CHANNELS,
    UNKNOWN,
    HRWindowSet,
    WindowSet,
)

__all__ = [
    "ACC",
    "CHANNELS",
    "CLASSES",
    "CLS",
    "GYR",
    "HR_BAND_HZ",
    "HR_CHANNELS",
    "HR_FS",
    "HR_LOADERS",
    "HRWindowSet",
    "LOADERS",
    "N_CHANNELS",
    "N_CLASSES",
    "N_HR_CHANNELS",
    "UNKNOWN",
    "WindowSet",
    "availability",
    "cache",
    "dalia",
    "e4",
    "ecg",
    "load_all",
    "load_hr",
    "motionsense",
    "uci",
    "wesad",
]

# tag -> module. Each module exposes download() and load().
LOADERS = {
    uci.TAG: uci,
    motionsense.TAG: motionsense,
}

# tag -> module, for the heart-rate branch. Same contract, different return type.
HR_LOADERS = {
    wesad.TAG: wesad,
    dalia.TAG: dalia,
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
    """Locate an already-extracted corpus, scoped to its own subdirectory.

    The scoping is load-bearing, not tidiness. WESAD and PPG-DaLiA both lay out
    their subjects as ``SN/SN.pkl``, so a bare search of `data_dir` for WESAD's
    marker finds whichever of the two extracted first -- and since both corpora
    normally extract side by side under one data root, that is the common case,
    not an edge case. The result loads without error and is the wrong corpus.
    `download()` already scopes its own search this way; this is the same rule
    for the path that skips the download.
    """
    from .download import find_root

    sub = getattr(mod, "SUBDIR", None)
    if sub and (data_dir / sub).is_dir():
        return find_root(data_dir / sub, mod.MARKER)
    return find_root(data_dir, mod.MARKER)


def load_hr(
    data_dir: str | Path,
    tags: list[str] | None = None,
    win: int | None = None,
    stride: int | None = None,
    cache_dir: str | Path | None = None,
    subjects: dict[str, list[str]] | None = None,
    download: bool = True,
    with_reference: bool = True,
) -> HRWindowSet:
    """Download (if needed) and load the named HR corpora into one HRWindowSet.

    Parameters
    ----------
    data_dir : path
        Where archives are downloaded and extracted. On Colab this is session
        disk and does not persist; `cache_dir` is what does.
    tags : list of str, optional
        Corpus tags. Defaults to everything in `HR_LOADERS`.
    win, stride : int, optional
        Window and hop in samples at `HR_FS`. Defaults to `e4`'s 8 s / 2 s.
    cache_dir : path, optional
        Distilled-cache location. Resolved through `cache.default_root`, so
        leaving it None still finds `$SHAHOSHI_CACHE` or a mounted Drive.
    subjects : dict, optional
        ``{tag: [subject ids]}`` to load a subset, for a smoke test.
    download : bool
        If False, expect the data to already be extracted and fail otherwise.
    with_reference : bool
        Derive ground-truth HR where a corpus ships only ECG. Ignored by corpora
        that ship a heart-rate track directly.
    """
    data_dir = Path(data_dir)
    tags = list(HR_LOADERS) if tags is None else tags
    subjects = subjects or {}

    unknown = [t for t in tags if t not in HR_LOADERS]
    if unknown:
        raise ValueError(f"unknown HR corpus tag(s) {unknown}; have {sorted(HR_LOADERS)}")

    resolved = cache.default_root(cache_dir)
    print(cache.describe(resolved))

    parts = []
    for tag in tags:
        mod = HR_LOADERS[tag]
        print(f"[{tag}]")
        root = mod.download(data_dir) if download else _expect_extracted(data_dir, mod)
        kwargs = {"win": win, "stride": stride, "cache_dir": resolved}
        if tag in subjects:
            kwargs["subjects"] = subjects[tag]
        # Only WESAD has an ECG to reduce; DaLiA ships HR directly.
        if "with_reference" in inspect.signature(mod.load).parameters:
            kwargs["with_reference"] = with_reference
        parts.append(mod.load(root, **kwargs))

    out = HRWindowSet.concat(parts)
    print(f"\nmerged:\n{out.summary()}")
    return out
