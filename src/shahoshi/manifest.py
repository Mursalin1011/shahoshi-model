"""Per-run provenance capture.

Every training run writes one JSON file recording the config, the metrics, the
git commit, and the library versions. It exists so that a number in the paper can
be traced to the exact code and settings that produced it -- and so that when two
runs disagree, the diff is a file comparison rather than an archaeology exercise.

Deliberately cheap and dependency-free. A provenance system that is any effort
to use does not get used.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_info(repo: str | Path = ".") -> dict[str, Any]:
    """Current commit, branch, and whether the tree is dirty.

    A dirty tree means the recorded commit does not fully describe the code that
    ran, so it is recorded explicitly rather than left to be assumed clean.
    """
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": sha,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def environment() -> dict[str, Any]:
    """Library versions that can change a numeric result."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "scipy", "pandas", "tensorflow", "sklearn"):
        try:
            mod = __import__(name)
            env[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[name] = None
    return env


def write(
    report_dir: str | Path,
    name: str,
    config: dict,
    metrics: dict,
    extra: dict | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write one run manifest and return its path.

    Parameters
    ----------
    name : str
        Run name; becomes part of the filename alongside a UTC timestamp, so
        repeated runs of the same config accumulate rather than overwrite.
    timestamp : str, optional
        Override for the UTC timestamp, for reproducible tests.
    """
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{stamp}_{_slug(name)}.json"

    payload = {
        "name": name,
        "timestamp_utc": stamp,
        "git": git_info(),
        "environment": environment(),
        "config": config,
        "metrics": metrics,
    }
    if extra:
        payload["extra"] = extra

    path.write_text(json.dumps(payload, indent=2, default=_fallback), encoding="utf-8")
    print(f"  wrote manifest {path.name}")
    if payload["git"]["dirty"]:
        print(
            "  NOTE: the working tree was dirty, so the recorded commit does not "
            "fully describe this run. Commit before a run whose numbers you intend "
            "to report."
        )
    return path


def collect(report_dir: str | Path) -> list[dict]:
    """Load every manifest in `report_dir`, newest first, for comparison tables."""
    report_dir = Path(report_dir)
    if not report_dir.exists():
        return []
    out = []
    for p in sorted(report_dir.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"  skipping unreadable manifest {p.name}")
    return out


def compare(report_dir: str | Path, keys: list[str] | None = None) -> str:
    """A flat text table of metrics across runs, for the ablation writeup."""
    runs = collect(report_dir)
    if not runs:
        return "(no manifests yet)"

    if keys is None:
        seen: list[str] = []
        for r in runs:
            for k in r.get("metrics", {}):
                if k not in seen and isinstance(r["metrics"][k], (int, float)):
                    seen.append(k)
        keys = seen

    width = max((len(r["name"]) for r in runs), default=4)
    lines = ["  ".join([f"{'run':<{width}}", *(f"{k:>18s}" for k in keys)])]
    for r in runs:
        cells = []
        for k in keys:
            v = r.get("metrics", {}).get(k)
            cells.append(f"{v:>18.4f}" if isinstance(v, (int, float)) else f"{'-':>18s}")
        lines.append("  ".join([f"{r['name']:<{width}}", *cells]))
    return "\n".join(lines)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-")


def _fallback(obj: Any) -> Any:
    """Coerce numpy scalars and arrays, which json cannot serialize."""
    if hasattr(obj, "item") and getattr(obj, "size", 2) == 1:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
