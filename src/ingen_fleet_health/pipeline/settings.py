"""Pipeline settings (``configs/pipeline.yaml``) and the partition layout.

Layout under ``data_root`` (all Parquet unless noted)::

    landing/<kind>/platform=<p>/date=<YYYY-MM-DD>/<file>   kind: telemetry (CSV), maintenance, labels
    validated/platform=<p>/date=<d>/telemetry.parquet
    quarantine/platform=<p>/date=<d>/rejected.parquet
    features/platform=<p>/date=<d>/features.parquet
    predictions/platform=<p>/date=<d>/predictions.parquet
    alerts/platform=<p>/date=<d>/alerts.parquet
    monitoring/drift/platform=<p>/date=<d>/psi.parquet
    monitoring/performance/date=<d>/performance.json
    runs/platform=<p>/date=<d>/<step>.json                  run manifests (input/output fingerprints)

A partition is written to a temporary folder and swapped in whole, so a
reader never sees a half-written partition and a re-run replaces it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from ingen_fleet_health.config import REPO_ROOT, load_config

DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"
TABLES = ("validated", "quarantine", "features", "predictions", "alerts")


class PartitionNotReady(RuntimeError):
    """The upstream landing partition for this date does not exist (yet)."""


def _resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


@dataclass
class PlatformSettings:
    name: str
    config_path: Path
    source_dir: Path
    config: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class Settings:
    data_root: Path
    platforms: dict[str, PlatformSettings]
    models: dict[str, dict[str, Any]]
    release: dict[str, Any]
    alerting: dict[str, Any]
    landing: dict[str, Any]
    monitoring: dict[str, Any]
    source: Path

    # ---- partition paths -------------------------------------------------
    def landing_dir(self, kind: str, platform: str, day: date) -> Path:
        return self.data_root / "landing" / kind / f"platform={platform}" / f"date={day.isoformat()}"

    def table_dir(self, table: str, platform: str, day: date) -> Path:
        return self.data_root / table / f"platform={platform}" / f"date={day.isoformat()}"

    def drift_dir(self, platform: str, day: date) -> Path:
        return self.data_root / "monitoring" / "drift" / f"platform={platform}" / f"date={day.isoformat()}"

    def performance_dir(self, day: date) -> Path:
        return self.data_root / "monitoring" / "performance" / f"date={day.isoformat()}"

    def run_manifest(self, platform: str, day: date, step: str) -> Path:
        return self.data_root / "runs" / f"platform={platform}" / f"date={day.isoformat()}" / f"{step}.json"

    def partition_dates(self, table: str, platform: str) -> list[date]:
        root = self.data_root / table / f"platform={platform}"
        if not root.exists():
            return []
        return sorted(date.fromisoformat(p.name.removeprefix("date=")) for p in root.glob("date=*") if p.is_dir())

    def landing_dates(self, kind: str, platform: str) -> list[date]:
        root = self.data_root / "landing" / kind / f"platform={platform}"
        if not root.exists():
            return []
        return sorted(date.fromisoformat(p.name.removeprefix("date=")) for p in root.glob("date=*") if p.is_dir())


def load_settings(path: str | Path | None = None, data_root: str | Path | None = None) -> Settings:
    """``data_root`` (or env ``FLEET_DATA_ROOT``) overrides the YAML value, e.g. for tests."""
    source = _resolve(path) if path else Path(os.environ.get("FLEET_PIPELINE_CONFIG", DEFAULT_PIPELINE_CONFIG))
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = data_root or os.environ.get("FLEET_DATA_ROOT") or raw["data_root"]
    platforms = {}
    for name, spec in raw["platforms"].items():
        cfg_path = _resolve(spec["config"])
        platforms[name] = PlatformSettings(name, cfg_path, _resolve(spec["source_dir"]), load_config(cfg_path))
    return Settings(_resolve(root), platforms, raw["models"], raw["release_training"], raw["alerting"],
                    raw["landing"], raw["monitoring"], source)


def daterange(start: date, end: date) -> Iterator[date]:
    """Inclusive date range."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(paths: list[Path]) -> str:
    """Order-independent hash of a set of files (name + content)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.as_posix()):
        h.update(p.as_posix().encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


@contextmanager
def atomic_partition(target: Path) -> Iterator[Path]:
    """Yield a fresh temporary folder; on success it replaces ``target`` whole.

    On failure the temporary folder is removed and ``target`` is untouched,
    so a failed run never leaves a partial partition behind.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp-{target.name}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    old = None
    if target.exists():
        old = target.parent / f".old-{target.name}-{uuid.uuid4().hex[:8]}"
        os.replace(target, old)
    os.replace(tmp, target)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
