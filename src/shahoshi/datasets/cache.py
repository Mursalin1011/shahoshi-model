"""Distilled per-subject cache, so a multi-GB corpus is parsed at most once.

WESAD ships as a single ~18 GB archive and PPG-DaLiA as ~1.6 GB, but almost all
of that is signals this project does not use: chest EMG, respiration, EDA, and
in WESAD's case a 700 Hz ECG we consume once and reduce to a beat series. What
the HR branch actually needs per subject -- BVP and ACC resampled to `HR_FS`, a
condition code, and a ground-truth HR series -- is a few MB.

So the loaders parse the archive once, write that distilled form here, and read
it on every subsequent run. On Colab, where the session disk is wiped but a
mounted Drive is not, pointing `cache_dir` at Drive turns an 18 GB download into
a few hundred MB of reads.

Two failure modes this guards against, both of which are silent:

**Stale distillation.** If the resampling or the ECG reduction changes, a cache
written by the old code is still perfectly readable and completely wrong. Every
entry carries `CACHE_VERSION`; a mismatch is treated as a miss, not as an error,
so bumping the constant is all it takes to invalidate every cache everywhere.

**Partial writes.** A Colab runtime dying mid-write leaves a truncated .npz that
reads as a corrupt archive on the next run. Writes go to a temporary file in the
same directory and are renamed into place, so an entry is either complete or
absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Bump when the distilled contents change meaning -- a different resample target,
# a different condition encoding, a different ECG reduction. Any existing cache
# entry then reads as a miss and is rebuilt.
CACHE_VERSION = 1

COLAB_DRIVE = Path("/content/drive/MyDrive")


def default_root(explicit: str | Path | None = None) -> Path | None:
    """Where to cache, preferring the most durable location available.

    Order: an explicit argument, then ``$SHAHOSHI_CACHE``, then a mounted Colab
    Drive, then nothing. Returning None rather than falling back to the session
    disk is deliberate: a cache on ephemeral storage costs a write on every run
    and pays back on none of them, and silently doing that would look like
    caching while behaving like no caching at all.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get("SHAHOSHI_CACHE")
    if env:
        return Path(env).expanduser()
    if COLAB_DRIVE.is_dir():
        return COLAB_DRIVE / "shahoshi_cache"
    return None


def subject_path(root: str | Path, tag: str, subject: str) -> Path:
    """Cache file for one subject of one corpus."""
    return Path(root) / tag / f"{subject}.npz"


def read(path: str | Path) -> dict[str, np.ndarray] | None:
    """Load a cache entry, or None if it is missing, stale or unreadable.

    Never raises. A cache is an optimization, and a damaged one must degrade to
    a slow run rather than to a failed one.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if int(z["cache_version"][0]) != CACHE_VERSION:
                return None
            return {k: z[k] for k in z.files if k != "cache_version"}
    except Exception as exc:                      # corrupt, truncated, or partial
        print(f"  cache: ignoring unreadable {path.name} ({type(exc).__name__}: {exc})")
        return None


def write(path: str | Path, payload: dict[str, np.ndarray]) -> Path:
    """Write a cache entry atomically, stamping it with CACHE_VERSION."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "cache_version" in payload:
        raise ValueError("'cache_version' is reserved and set by this function")

    tmp = path.with_suffix(".npz.tmp")
    stamped = dict(payload)
    stamped["cache_version"] = np.array([CACHE_VERSION], dtype=np.int32)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **stamped)
    os.replace(tmp, path)                          # atomic within one filesystem
    return path


def describe(root: str | Path | None) -> str:
    """One line for the notebook, so the caching state is never a guess."""
    if root is None:
        return (
            "cache: OFF (no --cache-dir, no $SHAHOSHI_CACHE, no mounted Drive). "
            "Every run re-parses the archive."
        )
    root = Path(root)
    if not root.exists():
        return f"cache: {root} (empty -- this run will populate it)"
    files = sorted(root.rglob("*.npz"))
    mb = sum(f.stat().st_size for f in files) / 1e6
    return f"cache: {root} ({len(files)} subjects, {mb:.1f} MB)"
