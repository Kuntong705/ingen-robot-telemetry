"""Run the daily pipeline for one date, a backfill range, or report its status.

    python -m ingen_fleet_health.pipeline run --date 2026-12-01               # all platforms, all steps
    python -m ingen_fleet_health.pipeline run --date 2026-12-01 --step ingest --platform fari
    python -m ingen_fleet_health.pipeline backfill --start 2026-11-25 --end 2026-12-20
    python -m ingen_fleet_health.pipeline status                              # stale / missing partitions

Idempotency: each step reads only its inputs (partitions up to the run date)
and replaces its output partition whole, and nothing in an output depends on
the wall clock, so re-running a date gives identical files (tested by hash).

Backfill runs dates in order, oldest first, because a date's features need
the validated history before it and its alerts need the earlier alerts
(refractory period). After a late partition arrives, backfill from that date
to the latest scored date: ``status`` lists the dates whose inputs changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from ingen_fleet_health.monitoring.drift import psi_table
from ingen_fleet_health.monitoring.performance import weekly_check
from ingen_fleet_health.pipeline.alerting import alerts_for_day
from ingen_fleet_health.pipeline.featurize import features_for_day
from ingen_fleet_health.pipeline.ingest import ingest_day
from ingen_fleet_health.pipeline.scoring import models, score_day, scoring_start
from ingen_fleet_health.pipeline.settings import (
    PartitionNotReady,
    Settings,
    atomic_partition,
    daterange,
    fingerprint,
    load_settings,
    read_manifest,
    write_manifest,
)

STEPS = ("ingest", "features", "score", "alerts", "drift")


def drift_for_day(settings: Settings, platform: str, day: date) -> dict[str, Any]:
    feat_path = settings.table_dir("features", platform, day) / "features.parquet"
    pred_path = settings.table_dir("predictions", platform, day) / "predictions.parquet"
    if not pred_path.exists():
        raise PartitionNotReady(f"{platform} {day}: no predictions; run score first")
    ew, _, _ = models(settings)
    frame = pd.read_parquet(feat_path).merge(pd.read_parquet(pred_path)[["unit_id", "t", "risk"]], on=["unit_id", "t"])
    ref = ew.psi_reference[platform]
    table = psi_table(frame, ref, float(settings.monitoring["psi_watch"]), float(settings.monitoring["psi_shift"]))
    table.insert(0, "date", day.isoformat())
    table.insert(0, "platform", platform)
    target = settings.drift_dir(platform, day)
    with atomic_partition(target) as tmp:
        table.to_parquet(tmp / "psi.parquet", index=False)
    summary = {"platform": platform, "date": day.isoformat(),
               "bands": table["band"].value_counts().to_dict() if len(table) else {},
               "max_psi_feature": (table.sort_values("psi", ascending=False).iloc[0]["feature"]
                                   if len(table) and table["psi"].notna().any() else None)}
    write_manifest(settings.run_manifest(platform, day, "drift"), {
        "step": "drift", "inputs": fingerprint([feat_path, pred_path]),
        "output": fingerprint([target / "psi.parquet"]), "summary": summary})
    return summary


STEP_FUNCS: dict[str, Callable[[Settings, str, date], dict[str, Any]]] = {
    "ingest": ingest_day, "features": features_for_day, "score": score_day, "alerts": alerts_for_day,
    "drift": drift_for_day,
}


MODEL_STEPS = ("score", "alerts", "drift")


def run_step(settings: Settings, step: str, platform: str, day: date) -> dict[str, Any]:
    """Run one step. Model steps on dates inside the training window are skipped
    (the history load only needs ingest and features; scoring those dates would
    be in-sample), so one DAG backfill from the fleet start covers both."""
    if step in MODEL_STEPS and day < scoring_start(settings, platform):
        return {"platform": platform, "date": day.isoformat(), "skipped": "training window (before scoring start "
                f"{scoring_start(settings, platform)}); only ingest and features run"}
    return STEP_FUNCS[step](settings, platform, day)


def is_performance_day(settings: Settings, day: date) -> bool:
    return day.weekday() == int(settings.monitoring["performance_weekday"])


def performance_for_day(settings: Settings, day: date, force: bool = False) -> dict[str, Any] | None:
    if not force and not is_performance_day(settings, day):
        return None
    ew, _, _ = models(settings)
    return weekly_check(settings, day, ew.training_units)


def run_day(settings: Settings, day: date, platforms: list[str] | None = None,
            steps: tuple[str, ...] = STEPS) -> dict[str, Any]:
    """All steps for one date. A platform whose landing partition is missing is
    reported as ``missing`` and gets no outputs for that date (ADR-0006);
    other platforms still run."""
    result: dict[str, Any] = {"date": day.isoformat(), "platforms": {}}
    for platform in platforms or list(settings.platforms):
        out: dict[str, Any] = {}
        try:
            for step in steps:
                out[step] = run_step(settings, step, platform, day)
            out["status"] = "ok"
        except PartitionNotReady as exc:
            out["status"] = "missing"
            out["reason"] = str(exc)
        result["platforms"][platform] = out
    if "drift" in steps:
        perf = performance_for_day(settings, day)
        if perf is not None:
            result["performance"] = "written"
    return result


def backfill(settings: Settings, start: date, end: date, platforms: list[str] | None = None,
             echo: Callable[[str], None] = print, steps: tuple[str, ...] = STEPS) -> list[dict[str, Any]]:
    out = []
    for d in daterange(start, end):
        res = run_day(settings, d, platforms, steps)
        statuses = {p: r["status"] for p, r in res["platforms"].items()}
        echo(f"{d} {json.dumps(statuses)}")
        out.append(res)
    return out


def status(settings: Settings) -> dict[str, Any]:
    """Per platform: landed vs scored dates, and scored dates whose inputs changed
    since they were scored (stale: re-run them, e.g. after a late partition)."""
    from ingen_fleet_health.pipeline.ingest import landing_file

    report: dict[str, Any] = {}
    for platform in settings.platforms:
        landed = settings.landing_dates("telemetry", platform)
        scored = settings.partition_dates("predictions", platform)
        stale = []
        for d in settings.partition_dates("validated", platform):
            m = read_manifest(settings.run_manifest(platform, d, "ingest"))
            try:
                src = landing_file(settings, platform, d)
            except PartitionNotReady:
                continue
            prev = [x for x in settings.partition_dates("validated", platform) if x < d]
            inputs = [src] + ([settings.table_dir("validated", platform, prev[-1]) / "telemetry.parquet"] if prev else [])
            if m is None or m["inputs"] != fingerprint(inputs):
                stale.append(d.isoformat())
        first = min(landed) if landed else None
        gaps = []
        if first and scored:
            # dates before the scoring start (training window) are never scored by design
            begin = max(first, scoring_start(settings, platform))
            gaps = [d.isoformat() for d in daterange(begin, max(scored)) if d not in set(scored)]
        report[platform] = {"landed": len(landed), "first_landed": first.isoformat() if first else None,
                            "last_landed": max(landed).isoformat() if landed else None,
                            "scored": len(scored), "last_scored": max(scored).isoformat() if scored else None,
                            "unscored_dates_before_last_scored": gaps, "stale_ingest_dates": stale,
                            "landed_not_scored": [d.isoformat() for d in landed if d not in set(scored)
                                                  and d >= scoring_start(settings, platform)]}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingen_fleet_health.pipeline")
    parser.add_argument("--config", help="pipeline YAML (default configs/pipeline.yaml)")
    parser.add_argument("--data-root")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one date")
    r.add_argument("--date", type=date.fromisoformat, required=True)
    r.add_argument("--platform", action="append")
    r.add_argument("--step", choices=[*STEPS, "performance", "performance-if-due"],
                   help="one step only (default: all); performance-if-due runs the weekly check only on its weekday")
    b = sub.add_parser("backfill", help="run a date range in order")
    b.add_argument("--start", type=date.fromisoformat, required=True)
    b.add_argument("--end", type=date.fromisoformat, required=True)
    b.add_argument("--platform", action="append")
    b.add_argument("--steps", default=",".join(STEPS),
                   help="comma list; the history load before the scoring start uses 'ingest,features'")
    sub.add_parser("status", help="landed / scored / stale partitions")
    args = parser.parse_args(argv)
    settings = load_settings(args.config, args.data_root)
    if args.cmd == "run":
        if args.step in ("performance", "performance-if-due"):
            perf = performance_for_day(settings, args.date, force=args.step == "performance")
            print(json.dumps(perf, indent=2, default=str) if perf is not None
                  else f"{args.date}: not the weekly check day; nothing to do")
            return 0
        if args.step:
            rc = 0
            for p in args.platform or list(settings.platforms):
                try:
                    print(json.dumps(run_step(settings, args.step, p, args.date), default=str))
                except PartitionNotReady as exc:
                    print(f"NOT READY: {exc}", file=sys.stderr)
                    rc = 3
            return rc
        res = run_day(settings, args.date, args.platform)
        print(json.dumps(res, indent=2, default=str))
        return 3 if any(p["status"] != "ok" for p in res["platforms"].values()) else 0
    if args.cmd == "backfill":
        steps = tuple(x.strip() for x in args.steps.split(",") if x.strip())
        unknown = set(steps) - set(STEPS)
        if unknown:
            parser.error(f"unknown steps {sorted(unknown)}")
        res_all = backfill(settings, args.start, args.end, args.platform, steps=steps)
        missing = [(r["date"], p) for r in res_all for p, v in r["platforms"].items() if v["status"] != "ok"]
        if missing:
            print(f"missing partitions (nothing written for these): {missing}", file=sys.stderr)
        return 3 if missing else 0
    print(json.dumps(status(settings), indent=2))
    return 0
