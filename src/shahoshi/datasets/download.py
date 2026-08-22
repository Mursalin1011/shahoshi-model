"""Mirror-aware download and extraction.

Every corpus we use is hosted somewhere that has gone down at least once. UCI's
own host has intermittent TLS and rate-limit trouble; SisFall's university host
(sistemic.udea.edu.co) is frequently unreachable for days. So every fetch takes
a *list* of mirrors, tries them in order, verifies the result, and -- crucially
-- reuses an already-present file so a failed mirror never costs you a
re-download of the ones that worked.

If every mirror fails the error message tells you exactly what to upload and
where, because on Colab that is the realistic fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path


def fetch(
    urls: str | list[str],
    dest: str | Path,
    min_mb: float = 1.0,
    force: bool = False,
) -> Path:
    """Download `urls` (tried in order) to `dest`, reusing an existing valid file.

    Parameters
    ----------
    urls : str or list of str
        Mirrors, most-preferred first.
    dest : path
        Destination file.
    min_mb : float
        Sanity floor. A mirror that returns an HTML error page will produce a
        few KB and be rejected rather than silently extracted as a corrupt zip.
    force : bool
        Re-download even if a valid-looking file is already present.

    Raises
    ------
    RuntimeError
        If every mirror fails, with instructions for manual placement.
    """
    dest = Path(dest)
    if isinstance(urls, str):
        urls = [urls]

    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1e6
        if size_mb >= min_mb:
            print(f"  reusing {dest.name} ({size_mb:.1f} MB)")
            return dest
        print(f"  {dest.name} is only {size_mb:.1f} MB (< {min_mb} MB), re-downloading")

    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        print(f"  -> {url}")
        rc = subprocess.call(
            ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "2",
             "-#", "-o", str(dest), url]
        )
        if rc == 0 and dest.exists() and dest.stat().st_size >= min_mb * 1e6:
            print(f"     ok, {dest.stat().st_size / 1e6:.1f} MB")
            return dest
        got = f"{dest.stat().st_size / 1e6:.1f} MB" if dest.exists() else "nothing"
        print(f"     failed (curl rc={rc}, got {got}), trying next mirror")

    raise RuntimeError(
        f"Every mirror failed for {dest.name}.\n"
        f"Download it by hand from one of:\n"
        + "".join(f"  {u}\n" for u in urls)
        + f"then place it at {dest.resolve()} and re-run this cell -- it will "
        f"reuse the file rather than re-downloading."
    )


def extract(archive: str | Path, into: str | Path, expect: str | None = None) -> Path:
    """Unzip `archive` into `into`, skipping the work if `expect` already exists.

    Parameters
    ----------
    expect : str, optional
        A path relative to `into` whose presence means extraction already
        happened. Extraction of a multi-GB corpus takes minutes on Colab and is
        pure waste on a re-run.

    Returns
    -------
    The `into` directory.
    """
    archive, into = Path(archive), Path(into)
    if expect and (into / expect).exists():
        print(f"  already extracted ({expect} present), skipping")
        return into

    into.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(
            f"{archive} is not a valid zip. It is most likely an HTML error page "
            f"saved under a .zip name -- delete it and re-run to try the next mirror."
        )
    with zipfile.ZipFile(archive) as z:
        z.extractall(into)
    print(f"  extracted {archive.name} -> {into}")
    return into


def find_root(search_in: str | Path, marker: str, max_depth: int = 4) -> Path:
    """Locate the directory containing `marker`, wherever the zip nested it.

    Archives disagree about whether they contain a top-level folder, and some
    mirrors of the same corpus differ from each other. Rather than hardcoding a
    path that works for one mirror, find the marker.

    Parameters
    ----------
    search_in : path
        Directory to search under.
    marker : str
        A relative path that must exist inside the corpus root, e.g.
        ``"train/y_train.txt"`` for UCI HAR.
    """
    search_in = Path(search_in)
    if (search_in / marker).exists():
        return search_in

    candidates = [search_in]
    for depth in range(max_depth):
        next_level = []
        for d in candidates:
            if not d.is_dir():
                continue
            for child in sorted(d.iterdir()):
                if child.is_dir():
                    if (child / marker).exists():
                        return child
                    next_level.append(child)
        if not next_level:
            break
        candidates = next_level

    raise FileNotFoundError(
        f"could not find a directory containing {marker!r} under {search_in.resolve()} "
        f"(searched {max_depth} levels). Check that extraction succeeded."
    )


def free_space_mb(path: str | Path = ".") -> float:
    """Free disk space in MB, for pre-flight checks before a multi-GB download."""
    return shutil.disk_usage(Path(path)).free / 1e6


def require_space(need_mb: float, path: str | Path = ".") -> None:
    """Fail before downloading rather than halfway through.

    Colab gives a limited disk and SisFall is large enough to matter, especially
    alongside an already-extracted UCI HAR and MotionSense.
    """
    free = free_space_mb(path)
    if free < need_mb:
        raise RuntimeError(
            f"need ~{need_mb:.0f} MB free at {Path(path).resolve()} but only "
            f"{free:.0f} MB available. Delete the intermediate .zip files after "
            f"extraction, or mount Drive and point the data root there."
        )
    print(f"  disk ok: {free:.0f} MB free, need ~{need_mb:.0f} MB")


def env_root(var: str, default: str) -> Path:
    """Data root from an environment variable, so Colab and local runs share code."""
    return Path(os.environ.get(var, default)).expanduser()
