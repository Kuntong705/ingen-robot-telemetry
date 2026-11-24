"""Cost-benefit of acting on the alerts, under stated assumptions (Week 4 model card).

Per alerting strategy and evaluation:

    savings          = caught failures x repair success x (unplanned failure cost - planned repair cost)
    false-alarm cost = false alarms x (inspection cost + swap probability x unnecessary-swap cost)
    net benefit      = savings - false-alarm cost       (also per unit-week)
    break-even ratio = false alarms / (caught failures x repair success)

Acting on the alerts pays off when (failure cost - planned repair cost) /
(false-alarm cost) exceeds the break-even ratio. That ratio uses only measured
counts, so it holds whatever the real InGen costs turn out to be; every money
value comes from ``configs/cost_assumptions.yaml`` (illustrative placeholders).

Counts come from leakage-safe evaluations already reported (Week 2 walk-forward
on unseen robots, the same alert budget for every strategy) and from the
production v1 held-out robots in the Week 4 operational period. Nothing is
re-tuned here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ingen_fleet_health.config import REPO_ROOT

DEFAULT_ASSUMPTIONS = REPO_ROOT / "configs" / "cost_assumptions.yaml"


@dataclass(frozen=True)
class Assumptions:
    cost_unplanned_failure: float
    cost_planned_repair: float
    cost_false_alarm_inspection: float
    cost_unnecessary_swap: float
    swap_probability_on_false_alarm: float
    repair_success_if_caught: float

    @property
    def false_alarm_cost(self) -> float:
        return self.cost_false_alarm_inspection + self.swap_probability_on_false_alarm * self.cost_unnecessary_swap

    @property
    def saving_per_catch(self) -> float:
        return self.repair_success_if_caught * (self.cost_unplanned_failure - self.cost_planned_repair)


def load_assumptions(path: str | Path = DEFAULT_ASSUMPTIONS) -> tuple[Assumptions, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fields = {k: float(raw[k]) for k in Assumptions.__dataclass_fields__}
    return Assumptions(**fields), raw


def evaluate(alerts: float, true_alerts: float, events: float, events_detected: float, unit_weeks: float,
             assumptions: Assumptions) -> dict[str, float]:
    false_alarms = alerts - true_alerts
    savings = events_detected * assumptions.saving_per_catch
    fa_cost = false_alarms * assumptions.false_alarm_cost
    caught_eff = events_detected * assumptions.repair_success_if_caught
    return {
        "alerts": alerts, "true_alerts": true_alerts, "false_alarms": false_alarms, "events": events,
        "events_detected": events_detected, "unit_weeks": unit_weeks,
        "savings_from_caught_failures": savings, "false_alarm_cost": fa_cost, "net_benefit": savings - fa_cost,
        "net_benefit_per_unit_week": (savings - fa_cost) / unit_weeks if unit_weeks else float("nan"),
        "run_to_failure_cost": events * assumptions.cost_unplanned_failure,
        "break_even_ratio": false_alarms / caught_eff if caught_eff else float("inf"),
        "assumed_ratio": (assumptions.cost_unplanned_failure - assumptions.cost_planned_repair)
        / assumptions.false_alarm_cost,
    }


def sensitivity(counts: dict[str, float], base: Assumptions, grid: dict[str, list[float]]) -> list[dict[str, Any]]:
    """One-at-a-time sensitivity: cost values are multipliers, probabilities are absolute values."""
    rows = []
    for name, values in grid.items():
        for v in values:
            if name.startswith("cost_"):
                a = replace(base, **{name: getattr(base, name) * v})
                label = f"{name} x{v:g}"
            else:
                a = replace(base, **{name: v})
                label = f"{name} = {v:g}"
            r = evaluate(**counts, assumptions=a)
            rows.append({"parameter": name, "setting": label, "net_benefit": r["net_benefit"],
                         "net_benefit_per_unit_week": r["net_benefit_per_unit_week"],
                         "assumed_ratio": r["assumed_ratio"], "break_even_ratio": r["break_even_ratio"]})
    return rows


def counts_from_w02(results: dict[str, Any], model: str) -> dict[str, float]:
    m = results["models"][model]
    alerts = float(m["alerts"])
    return {"alerts": alerts, "true_alerts": round(alerts * float(m["precision"])), "events": float(m["events"]),
            "events_detected": float(m["events_detected"]), "unit_weeks": float(m["unit_weeks"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-benefit table for the model card")
    parser.add_argument("--assumptions", type=Path, default=DEFAULT_ASSUMPTIONS)
    parser.add_argument("--production", type=Path, default=REPO_ROOT / "reports" / "w04" / "operational_results.json",
                        help="held-out production counts written by scripts/build_w04_evidence.py (optional)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "w04" / "cost_benefit.json")
    args = parser.parse_args()
    base, raw = load_assumptions(args.assumptions)
    w02 = json.loads((REPO_ROOT / "reports" / "w02" / "early_warning_results.json").read_text(encoding="utf-8"))
    strategies: dict[str, dict[str, float]] = {
        f"w02_walk_forward:{m}": counts_from_w02(w02, m) for m in ("lgbm_selected", "iforest", "physics_rule")}
    if args.production.exists():
        prod = json.loads(args.production.read_text(encoding="utf-8"))
        for key, c in prod.get("cost_counts", {}).items():
            strategies[f"w04_operational:{key}"] = {k: float(v) for k, v in c.items()}
    out: dict[str, Any] = {"assumptions": {**asdict(base), "currency": raw.get("currency"),
                           "false_alarm_cost": base.false_alarm_cost, "saving_per_catch": base.saving_per_catch},
           "notice": "Money values are illustrative placeholders, not InGen figures. The break-even ratio uses "
                     "measured counts only.",
           "strategies": {k: evaluate(**c, assumptions=base) for k, c in strategies.items()},
           "sensitivity": {k: sensitivity(c, base, raw["sensitivity"]) for k, c in strategies.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, r in out["strategies"].items():
        print(f"{k:40s} net/unit-week {r['net_benefit_per_unit_week']:8.1f}  break-even ratio "
              f"{r['break_even_ratio']:.2f} (assumed {r['assumed_ratio']:.2f})")


if __name__ == "__main__":
    main()
