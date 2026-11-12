"""Synthetic maintenance-log and operator-note corpus tied to simulated events.

    python -m ingen_fleet_health.text.corpus --data-dir data/w02

The generator plays the role of the people around the robots, so it reads the
simulator's *ground truth* (fault timelines, sensor drift, outages). Its
output is split the same way as the telemetry:

    text/log_corpus.csv                 observable: what was written, when, and its label
    text/log_corpus_manifest.json       seed, config hash, counts, held-out phrase ids
    ground_truth/log_corpus_truth.csv   hidden: why each ticket exists (analysis only)

As-of rules a consumer must respect (enforced in ``text.features`` and the
experiment splits, tested in ``tests/w03``):

* a ticket is visible to a feature at time t only if ``created_at_utc <= t``;
* its ``label`` may be used for training only if ``label_available_utc`` is
  not after the training cut-off (the category is confirmed at triage or at
  the closing repair, not when the note is written).

ALL TEXT IS SYNTHETIC (templates + rule-based paraphrase + seeded noise).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.config import REPO_ROOT, load_config
from ingen_fleet_health.labels import load_fault_events
from ingen_fleet_health.simulate.faults import ramp
from ingen_fleet_health.text import TAXONOMY
from ingen_fleet_health.text.noise import apply_noise
from ingen_fleet_health.text.templates import (
    DRIFT_SERVICE,
    MINOR,
    MISATTRIBUTED,
    NFF_CLOSE,
    NFF_HEDGE,
    NFF_TOPIC,
    PREFIX,
    SERVICE_CLOSE,
    SERVICE_OPEN,
    SLOT_VALUES,
    SUFFIX,
    SYMPTOM,
    phrase_ids,
)

DEFAULT_CORPUS_CONFIG = REPO_ROOT / "configs" / "maintenance_logs.yaml"
GENERATOR_VERSION = "0.1"
FAULT_CATEGORY = {
    "bearing_wear": "drivetrain",
    "cooling_degradation": "thermal",
    "battery_cell_fault": "battery",
}
OBSERVABLE_COLUMNS = [
    "ticket_id", "unit_id", "created_at_utc", "doc_type", "text", "label", "label_available_utc", "is_synthetic",
]
TRUTH_COLUMNS = [
    "ticket_id", "source", "style", "phase", "fault_id", "phrase_id", "severity_at_creation",
    "n_abbrev", "n_typos", "unseen_phrase",
]


@dataclass
class Draft:
    unit_id: str
    created: pd.Timestamp
    doc_type: str                 # operator_note | service_ticket
    label: str
    label_available: pd.Timestamp
    core: str
    phrase_id: str
    source: str
    style: str = "specific"       # specific | vague | misattributed | hedged
    phase: str = "none"           # pre_onset | post_onset | none
    fault_id: str = ""
    severity: float = float("nan")
    decorate: bool = True         # add prefix/suffix (operator notes only)


def _hours(lo_hi: list[float], rng: np.random.Generator) -> pd.Timedelta:
    return pd.Timedelta(hours=float(rng.uniform(float(lo_hi[0]), float(lo_hi[1]))))


def _pick(group: str, rng: np.random.Generator, bank: dict[str, list[str]] = SYMPTOM,
          id_prefix: str = "") -> tuple[str, str]:
    items = bank[group]
    i = int(rng.integers(0, len(items)))
    return items[i], f"{id_prefix}{group}:{i}"


def _in_operator_hours(t: pd.Timestamp, hours: list[int]) -> bool:
    return int(hours[0]) <= t.hour < int(hours[1])


def _slot_time(slot_start: pd.Timestamp, slot_hours: float, hours: list[int],
               rng: np.random.Generator) -> pd.Timestamp | None:
    """A staffed minute inside the slot. Always draws 8 candidates so the
    random stream stays aligned whatever the outcome."""
    offsets = rng.uniform(0.0, slot_hours * 60.0, 8)
    for off in offsets:
        t = (slot_start + pd.Timedelta(minutes=float(off))).floor("min")
        if _in_operator_hours(t, hours):
            return t
    return None


def severity_at(event: Any, t: pd.Timestamp) -> float:
    """Fault severity in [0, 1] at t (same ramp as the simulator)."""
    span = (event.failure_utc - event.initiation_utc) / pd.Timedelta(minutes=1)
    x = (t - event.initiation_utc) / pd.Timedelta(minutes=1) / span
    return float(ramp(event.ramp_shape, np.array([x]))[0])


def fault_drafts(events: pd.DataFrame, end: pd.Timestamp, cfg: dict[str, Any],
                 rng: np.random.Generator, notice_rate: float) -> list[Draft]:
    fc = cfg["fault_notes"]
    slot_h = float(fc["slot_hours"])
    hours = cfg["operator_hours_utc"]
    drafts: list[Draft] = []
    for ev in events.sort_values("fault_id").itertuples(index=False):
        category = FAULT_CATEGORY[ev.fault_type]
        n_slots = int(np.ceil((ev.repair_start_utc - ev.initiation_utc) / pd.Timedelta(hours=slot_h)))
        for k in range(n_slots):
            slot_start = ev.initiation_utc + pd.Timedelta(hours=k * slot_h)
            t = _slot_time(slot_start, slot_h, hours, rng)
            u_note, u_style = rng.random(2)
            cores = {
                "specific": _pick(category, rng),
                "vague": _pick("vague", rng),
                "misattributed": _pick(category, rng, MISATTRIBUTED, "misattributed_"),
            }
            if t is None or t >= ev.repair_start_utc or t >= end:
                continue
            sev = severity_at(ev, t)
            p = min(float(fc["max_slot_probability"]), notice_rate * sev ** float(fc["notice_exponent"]))
            if u_note >= p:
                continue
            vague, mis = float(fc["vague_fraction"]), float(fc["misattribution_fraction"])
            style = "vague" if u_style < vague else ("misattributed" if u_style < vague + mis else "specific")
            core, pid = cores[style]
            drafts.append(Draft(ev.unit_id, t, "operator_note", category, ev.repair_end_utc, core, pid,
                                "fault_symptom", style, "pre_onset" if t < ev.onset_utc else "post_onset",
                                ev.fault_id, sev))
        opened = _pick(category, rng, SERVICE_OPEN, "service_open_")
        closed = _pick(category, rng, SERVICE_CLOSE, "service_close_")
        for when, (core, pid), source in ((ev.repair_start_utc, opened, "fault_service_open"),
                                          (ev.repair_end_utc, closed, "fault_service_close")):
            if when < end:
                drafts.append(Draft(ev.unit_id, when.floor("min"), "service_ticket", category, ev.repair_end_utc,
                                    core, pid, source, "specific", "post_onset", ev.fault_id,
                                    decorate=False))
    return drafts


def drift_drafts(impairments: pd.DataFrame, end: pd.Timestamp, cfg: dict[str, Any],
                 rng: np.random.Generator) -> list[Draft]:
    dc = cfg["sensor_drift_notes"]
    hours = cfg["operator_hours_utc"]
    drafts: list[Draft] = []
    drift = impairments[impairments["event_type"] == "sensor_drift"].sort_values(["unit_id", "start_utc"])
    for ev in drift.itertuples(index=False):
        service_at = (ev.end_utc + pd.Timedelta(days=float(rng.uniform(*dc["recalibration_delay_days"])))).floor("min")
        last = min(service_at, end)
        n_slots = int(np.ceil((last - ev.start_utc) / pd.Timedelta(hours=12)))
        for k in range(n_slots):
            t = _slot_time(ev.start_utc + pd.Timedelta(hours=12 * k), 12.0, hours, rng)
            u_note, u_word = rng.random(2)
            plain = _pick("sensor_drift", rng)
            as_thermal = _pick("sensor_drift_as_thermal", rng)
            if t is None or t >= last:
                continue
            level = min(1.0, (t - ev.start_utc) / max(ev.end_utc - ev.start_utc, pd.Timedelta(minutes=1)))
            if u_note >= float(dc["slot_probability_at_peak"]) * level:
                continue
            thermal_wording = u_word < float(dc["thermal_wording_fraction"])
            core, pid = as_thermal if thermal_wording else plain
            drafts.append(Draft(ev.unit_id, t, "operator_note", "sensor", service_at, core, pid, "drift_symptom",
                                "misattributed" if thermal_wording else "specific", severity=float(level)))
        i = int(rng.integers(0, len(DRIFT_SERVICE)))
        if service_at < end:
            drafts.append(Draft(ev.unit_id, service_at, "service_ticket", "sensor", service_at, DRIFT_SERVICE[i],
                                f"drift_service:{i}", "drift_service", decorate=False))
    return drafts


def outage_drafts(impairments: pd.DataFrame, end: pd.Timestamp, cfg: dict[str, Any],
                  rng: np.random.Generator) -> list[Draft]:
    oc = cfg["outage_tickets"]
    drafts: list[Draft] = []
    outages = impairments[impairments["event_type"] == "network_outage"].sort_values(["unit_id", "start_utc"])
    for ev in outages.itertuples(index=False):
        minutes = float(ev.magnitude)
        u = rng.random()
        created = (ev.end_utc + _hours(oc["report_delay_hours"], rng)).floor("min")
        core, pid = _pick("connectivity", rng)
        available = created + _hours(cfg["triage_delay_hours"]["connectivity"], rng)
        if u >= min(1.0, float(oc["probability_per_hour_of_outage"]) * minutes / 60.0) or created >= end:
            continue
        core = core.replace("{min}", str(int(max(5, round(minutes / 5) * 5))))
        drafts.append(Draft(ev.unit_id, created, "operator_note", "connectivity", available, core, pid, "outage"))
    return drafts


def _busy_windows(events: pd.DataFrame, impairments: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Per unit, periods with a developing fault or active sensor drift."""
    busy: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ev in events.itertuples(index=False):
        busy.setdefault(ev.unit_id, []).append((ev.initiation_utc, ev.repair_end_utc))
    for ev in impairments[impairments["event_type"] == "sensor_drift"].itertuples(index=False):
        busy.setdefault(ev.unit_id, []).append((ev.start_utc, pd.Timestamp.max.tz_localize("UTC")))
    return busy


def background_drafts(units: list[str], start: pd.Timestamp, end: pd.Timestamp, events: pd.DataFrame,
                      impairments: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator) -> list[Draft]:
    rates = cfg["background_rates_per_unit_day"]
    hours = cfg["operator_hours_utc"]
    days = (end - start) / pd.Timedelta(days=1)
    triage = cfg["triage_delay_hours"]
    busy = _busy_windows(events, impairments)
    drafts: list[Draft] = []

    def when() -> tuple[str, pd.Timestamp]:
        unit = units[int(rng.integers(0, len(units)))]
        day = int(rng.integers(0, int(days)))
        minute = float(rng.uniform(int(hours[0]) * 60, int(hours[1]) * 60))
        return unit, (start + pd.Timedelta(days=day, minutes=minute)).floor("min")

    for kind in sorted(rates):
        n = int(rng.poisson(float(rates[kind]) * len(units) * days))
        for _ in range(n):
            unit, t = when()
            if kind.startswith("minor_") and kind != "minor_sensor":
                cat = kind.removeprefix("minor_")
                core, pid = _pick(cat, rng, MINOR, "minor_")
                drafts.append(Draft(unit, t, "operator_note", cat, t + _hours(triage["minor"], rng), core, pid, kind))
            elif kind == "minor_sensor":
                core, pid = _pick("sensor_minor", rng)
                drafts.append(Draft(unit, t, "operator_note", "sensor", t + _hours(triage["minor"], rng), core, pid, kind))
            elif kind == "connectivity_misc":
                core, pid = _pick("connectivity", rng)
                core = core.replace("{min}", str(int(rng.integers(1, 12)) * 5))
                drafts.append(Draft(unit, t, "operator_note", "connectivity",
                                    t + _hours(triage["connectivity"], rng), core, pid, kind))
            elif kind == "user_error":
                core, pid = _pick("user_or_operational_error", rng)
                drafts.append(Draft(unit, t, "operator_note", "user_or_operational_error",
                                    t + _hours(triage["user_error"], rng), core, pid, kind))
            elif kind == "routine_note":
                core, pid = _pick("routine", rng)
                drafts.append(Draft(unit, t, "operator_note", "no_fault_found",
                                    t + _hours(triage["routine"], rng), core, pid, kind))
            elif kind == "nff_episode":
                topic = str(rng.choice(sorted(NFF_TOPIC)))
                core, pid = _pick(topic, rng)
                hedge = NFF_HEDGE[int(rng.integers(0, len(NFF_HEDGE)))]
                hedged = rng.random() < float(cfg["nff"]["hedged_fraction"])
                closure = (t + _hours(cfg["nff"]["closure_delay_hours"], rng)).floor("min")
                close_text = NFF_CLOSE[int(rng.integers(0, len(NFF_CLOSE)))].format(topic=NFF_TOPIC[topic])
                if any(a <= t < b for a, b in busy.get(unit, [])):
                    continue        # a complaint on a degrading unit would be a real symptom, not NFF
                drafts.append(Draft(unit, t, "operator_note", "no_fault_found", closure,
                                    hedge.format(core=core) if hedged else core, pid, "nff_complaint",
                                    "hedged" if hedged else "specific"))
                if closure < end:
                    drafts.append(Draft(unit, closure, "service_ticket", "no_fault_found", closure, close_text,
                                        f"nff_close_{topic}", "nff_closure", decorate=False))
            else:
                raise ValueError(f"Unknown background kind: {kind}")
    return drafts


def _fill_slots(core: str, unit_id: str, rng: np.random.Generator) -> str:
    pct = int(rng.integers(8, 36))
    hh = int(rng.integers(6, 18))
    values = {
        "pct": pct,
        "pct2": pct + int(rng.integers(15, 41)),
        "temp": int(rng.integers(70, 96)),
        "d": int(rng.integers(2, 6)),
        "min": int(rng.integers(1, 12)) * 5,
        "hh": hh,
        "hh2": hh + int(rng.integers(1, 4)),
        "side": str(rng.choice(SLOT_VALUES["side"])),
        "place": str(rng.choice(SLOT_VALUES["place"])),
    }
    for key, value in values.items():
        core = core.replace("{" + key + "}", str(value))
    return core


def _unit_mention(unit_id: str, rng: np.random.Generator) -> str:
    number = int(unit_id.rsplit("_", 1)[-1]) if "_" in unit_id else 0
    name = unit_id.split("_", 1)[0].lower()
    return str(rng.choice([f"{name} {number}", f"{name[:1].upper()}{number:03d}", unit_id, f"unit {number}"]))


def render(draft: Draft, cfg: dict[str, Any], seed: int) -> tuple[str, dict[str, int]]:
    """Deterministic text for one draft: the generator is keyed on the
    ticket's own identity, so unrelated corpus changes do not reword it."""
    key = f"{draft.source}|{draft.unit_id}|{draft.created.isoformat()}|{draft.phrase_id}"
    rng = np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])
    noise = cfg["noise"]
    text = _fill_slots(draft.core, draft.unit_id, rng)
    if draft.decorate:
        u_prefix, u_suffix = rng.random(2)
        if u_prefix < float(noise["unit_mention_fraction"]):
            template = str(rng.choice(PREFIX[:2]))
            text = template.format(unit=_unit_mention(draft.unit_id, rng)) + text
        elif u_prefix < float(noise["unit_mention_fraction"]) + 0.25:
            template = str(rng.choice(PREFIX[2:]))
            text = template.format(shift=str(rng.choice(SLOT_VALUES["shift"]))) + text
        if u_suffix < float(noise["suffix_fraction"]):
            text = text + str(rng.choice([", ", ". ", " - "])) + str(rng.choice(SUFFIX))
    return apply_noise(text, noise, rng)


def unseen_phrase_ids(fraction: float, seed: int) -> list[str]:
    """Phrases reserved for the lexical-shift test: a seeded share of every group."""
    rng = np.random.default_rng([seed, 77])
    by_group: dict[str, list[str]] = {}
    for pid in phrase_ids():
        by_group.setdefault(pid.split(":")[0], []).append(pid)
    held: list[str] = []
    for group in sorted(by_group):
        ids = by_group[group]
        k = max(1, round(fraction * len(ids))) if len(ids) >= 3 else 0
        held += [str(x) for x in rng.choice(ids, k, replace=False)]
    return sorted(held)


def generate_corpus(data_dir: str | Path, config_path: str | Path | None = None, seed: int | None = None,
                    notice_rate: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data_dir = Path(data_dir)
    config_path = Path(config_path) if config_path else DEFAULT_CORPUS_CONFIG
    cfg = load_config(config_path)
    if list(cfg["taxonomy"]) != list(TAXONOMY):
        raise ValueError("Corpus config taxonomy differs from ingen_fleet_health.text.TAXONOMY")
    seed = int(cfg["seed"]) if seed is None else int(seed)
    notice = float(cfg["fault_notes"]["notice_rate"]) if notice_rate is None else float(notice_rate)

    manifest_sim = json.loads((data_dir / "simulation_manifest.json").read_text(encoding="utf-8"))
    start, end = pd.Timestamp(manifest_sim["start_utc"]), pd.Timestamp(manifest_sim["end_utc"])
    events = load_fault_events(data_dir / "ground_truth" / "fault_events.csv")
    impairments = pd.read_csv(data_dir / "ground_truth" / "impairment_events.csv")
    for col in ("start_utc", "end_utc"):
        impairments[col] = pd.to_datetime(impairments[col], utc=True)
    units = sorted(pd.read_csv(data_dir / "events" / "units.csv")["unit_id"])

    # independent streams: changing one process (e.g. notice_rate) leaves the others untouched
    s_fault, s_drift, s_outage, s_back = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))
    drafts = (
        fault_drafts(events, end, cfg, s_fault, notice)
        + drift_drafts(impairments, end, cfg, s_drift)
        + outage_drafts(impairments, end, cfg, s_outage)
        + background_drafts(units, start, end, events, impairments, cfg, s_back)
    )
    drafts = [d for d in drafts if start <= d.created < end]
    drafts.sort(key=lambda d: (d.created, d.unit_id, d.source, d.phrase_id))
    held = set(unseen_phrase_ids(float(cfg["unseen_phrase_holdout_fraction"]), seed))

    rows, truth = [], []
    for i, d in enumerate(drafts):
        text, stats = render(d, cfg, seed)
        tid = f"T{i + 1:05d}"
        rows.append({
            "ticket_id": tid,
            "unit_id": d.unit_id,
            "created_at_utc": d.created.isoformat(),
            "doc_type": d.doc_type,
            "text": text,
            "label": d.label,
            "label_available_utc": d.label_available.floor("min").isoformat(),
            "is_synthetic": True,
        })
        truth.append({
            "ticket_id": tid, "source": d.source, "style": d.style, "phase": d.phase, "fault_id": d.fault_id,
            "phrase_id": d.phrase_id, "severity_at_creation": d.severity, **stats,
            "unseen_phrase": d.phrase_id in held,
        })
    corpus = pd.DataFrame(rows, columns=OBSERVABLE_COLUMNS)
    truth_frame = pd.DataFrame(truth, columns=TRUTH_COLUMNS)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "is_synthetic": True,
        "note": "Synthetic text generated from templates, rule-based paraphrase and seeded noise. "
                "No real InGen ticket, operator or customer text. No LLM paraphrase was applied.",
        "seed": seed,
        "notice_rate": notice,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "simulator_config_sha256": manifest_sim["config_sha256"],
        "tickets": len(corpus),
        "by_label": {k: int(v) for k, v in corpus["label"].value_counts().sort_index().items()},
        "by_source": {k: int(v) for k, v in truth_frame["source"].value_counts().sort_index().items()},
        "by_doc_type": {k: int(v) for k, v in corpus["doc_type"].value_counts().sort_index().items()},
        "fault_notes_pre_onset": int(((truth_frame["source"] == "fault_symptom")
                                      & (truth_frame["phase"] == "pre_onset")).sum()),
        "unseen_phrase_ids": sorted(held),
    }
    return corpus, truth_frame, manifest


def write_corpus(data_dir: str | Path, corpus: pd.DataFrame, truth: pd.DataFrame, manifest: dict[str, Any],
                 subdir: str = "text") -> Path:
    data_dir = Path(data_dir)
    out = data_dir / subdir
    out.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(out / "log_corpus.csv", index=False)
    (out / "log_corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    truth_name = "log_corpus_truth.csv" if subdir == "text" else f"log_corpus_truth_{subdir}.csv"
    truth.to_csv(data_dir / "ground_truth" / truth_name, index=False)
    return out / "log_corpus.csv"


def load_corpus(data_dir: str | Path, subdir: str = "text") -> pd.DataFrame:
    """Observable corpus only (the hidden truth table is loaded separately)."""
    frame = pd.read_csv(Path(data_dir) / subdir / "log_corpus.csv")
    for col in ("created_at_utc", "label_available_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def load_corpus_truth(data_dir: str | Path, subdir: str = "text") -> pd.DataFrame:
    name = "log_corpus_truth.csv" if subdir == "text" else f"log_corpus_truth_{subdir}.csv"
    return pd.read_csv(Path(data_dir) / "ground_truth" / name, keep_default_na=False,
                       na_values={"severity_at_creation": [""]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic maintenance-log corpus")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--notice-rate", type=float, help="override fault_notes.notice_rate (sensitivity runs)")
    parser.add_argument("--subdir", default="text", help="output folder under the data dir")
    args = parser.parse_args()
    corpus, truth, manifest = generate_corpus(args.data_dir, args.config, args.seed, args.notice_rate)
    path = write_corpus(args.data_dir, corpus, truth, manifest, args.subdir)
    print(json.dumps({k: v for k, v in manifest.items() if k != "unseen_phrase_ids"}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
