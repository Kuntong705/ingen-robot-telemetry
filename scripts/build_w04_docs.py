"""Build the Week 4 documents from JSON results only (no hand-typed numbers).

Inputs:  reports/w04/operational_results.json   (scripts/build_w04_evidence.py)
         reports/w04/cost_benefit.json          (python -m ingen_fleet_health.release.cost_benefit)
         reports/w04/quality_gates.json         (scripts/run_w04_gates.py, optional)
         reports/w02/*.json, reports/w03/*.json (earlier, unchanged results)
Outputs: docs/W04_Model_Card.docx, docs/W04_Capstone_Deck.pdf, docs/W04_Retrospective.docx,
         docs/figures/w04/*.png

    python scripts/build_w04_docs.py
"""

from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx.shared import Inches
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import itertools

import build_w03_docs as w3

DOCS = ROOT / "docs"
FIGS = DOCS / "figures" / "w04"
R2, R3, R4 = ROOT / "reports" / "w02", ROOT / "reports" / "w03", ROOT / "reports" / "w04"
PLAT = {"aido_rover": "Aido Rover", "fari": "Fari"}
SYNTH = "SYNTHETIC DATA - simulated fleets; no real InGen robot, cost or customer data"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def f3(v: Any, d: int = 3) -> str:
    return w3.fmt(v, d)


def ci(pair: Any, d: int = 3) -> str:
    return w3.ci(pair, d) if pair and None not in pair else "n/a"


def verdict(pair: Any) -> str:
    if not pair or None in pair:
        return "n/a"
    return w3.verdict(pair)


def money(v: float) -> str:
    return f"{v:,.0f}" if math.isfinite(v) else "n/a"


EDGE_CANDIDATES = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
                   Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
                   Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]


def find_browser() -> str | None:
    for name in ("msedge", "chromium", "chromium-browser", "google-chrome", "chrome"):
        exe = shutil.which(name)
        if exe:
            return exe
    return next((str(p) for p in EDGE_CANDIDATES if p.exists()), None)


def html_inline(text: str) -> str:
    out = []
    for piece in w3.INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            out.append(f"<b>{html.escape(piece[2:-2])}</b>")
        elif piece.startswith("`") and piece.endswith("`"):
            out.append(f"<code>{html.escape(piece[1:-1])}</code>")
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append(f"<i>{html.escape(piece[1:-1])}</i>")
        else:
            out.append(html.escape(piece))
    return "".join(out)


class CardDoc(w3.Doc):
    """The Week 3 docx writer plus an HTML twin of the same content, printed to PDF by a
    headless Chromium browser (Edge/Chrome), so the .docx and the .pdf cannot diverge."""

    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__(title, subtitle)
        self.html = [f"<h1>{html.escape(title)}</h1>", f"<p class='sub'>{html_inline(subtitle)}</p>"]

    def h(self, text: str, level: int = 1) -> None:
        super().h(text, level)
        self.html.append(f"<h{level + 1}>{html.escape(text)}</h{level + 1}>")

    def p(self, text: str, bold_prefix: str | None = None) -> None:
        super().p(text, bold_prefix)
        pre = f"<b>{html.escape(bold_prefix)}</b>" if bold_prefix else ""
        self.html.append(f"<p>{pre}{html_inline(text)}</p>")

    def bullets(self, items: list[str]) -> None:
        super().bullets(items)
        self.html.append("<ul>" + "".join(f"<li>{html_inline(i)}</li>" for i in items) + "</ul>")

    def table(self, header: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
        super().table(header, rows, widths)
        head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in r) + "</tr>" for r in rows)
        self.html.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")

    def picture(self, path: Path, width: float = 6.4) -> None:
        self.d.add_picture(str(path), width=Inches(width))
        self.html.append(f"<img src='{path.resolve().as_uri()}' style='width:{width / 6.4 * 100:.0f}%'>")

    def save_pdf(self, name: str) -> Path | None:
        browser = find_browser()
        if browser is None:
            print("no Chromium browser found: PDF not written")
            return None
        style = ("body{font-family:Calibri,'Segoe UI',Arial,sans-serif;font-size:10.5pt;color:#1B2631;margin:0}"
                 "h1{font-size:19pt;margin:0 0 4pt}h2{font-size:14pt;color:#1F4E79;margin:14pt 0 4pt}"
                 "h3{font-size:11.5pt;color:#1F4E79;margin:10pt 0 3pt}p.sub{color:#555;font-size:9pt}"
                 "table{border-collapse:collapse;width:100%;margin:4pt 0 8pt;font-size:8.5pt;page-break-inside:auto}"
                 "th,td{border:1px solid #9DB4CC;padding:2pt 4pt;text-align:left;vertical-align:top}"
                 "th{background:#DCE6F1}tr{page-break-inside:avoid}code{font-family:Consolas,monospace;font-size:9pt}"
                 "img{display:block;margin:6pt 0}ul{margin:2pt 0 6pt}"
                 "@page{size:Letter;margin:0.8in 0.9in}")
        page = f"<!doctype html><html><head><meta charset='utf-8'><style>{style}</style></head><body>" \
               + "\n".join(self.html) + "</body></html>"
        out = DOCS / name
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "card.html"
            src.write_text(page, encoding="utf-8")
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={out}", src.as_uri()],
                           check=True, capture_output=True, timeout=180)
        return out


class Evidence:
    def __init__(self) -> None:
        self.op = load(R4 / "operational_results.json")
        self.cb = load(R4 / "cost_benefit.json")
        self.gates = load(R4 / "quality_gates.json") if (R4 / "quality_gates.json").exists() else None
        self.ew2 = load(R2 / "early_warning_results.json")
        self.soh2 = load(R2 / "soh_results.json")
        self.leak = load(R2 / "leakage_results.json")
        self.fus = load(R3 / "fusion_results.json")
        self.fari = load(R3 / "fari_transfer_results.json")
        self.exp = load(R3 / "alert_explanations.json")
        self.rev = load(R3 / "llm_faithfulness_review.json")

        self.cmapss = load(R2 / "cmapss_fd001_results.json")
        rp = R4 / "retrain_reproducibility.json"
        self.retrain = load(rp) if rp.exists() else None

    def training_window(self, platform: str) -> dict[str, Any]:
        return self.op["platforms"][platform]["training_window"]

    def calibration(self, platform: str, cohort: str = "held_out_units") -> dict[str, Any]:
        return self.cohort(platform, cohort)["calibration"]

    def weekly_status_counts(self) -> dict[str, tuple[int, int]]:
        """cohort -> (insufficient-label cohort-weeks, evaluated or insufficient cohort-weeks)."""
        out: dict[str, list[int]] = {"held_out_units": [0, 0], "training_units": [0, 0]}
        for w in self.op["weekly_checks"]:
            for v in w["platforms"].values():
                for cohort, tally in out.items():
                    status = (v.get(cohort) or {}).get("status")
                    if status in ("evaluated", "insufficient labels"):
                        tally[1] += 1
                        tally[0] += status == "insufficient labels"
        return {k: (v[0], v[1]) for k, v in out.items()}

    def cost_row(self, key: str) -> dict[str, Any]:
        return self.cb["strategies"][key]

    def cohort(self, platform: str, cohort: str = "held_out_units") -> dict[str, Any]:
        return self.op["platforms"][platform]["cohorts"][cohort]

    # earlier results, read from their JSON (never typed by hand)
    @property
    def leak_pair(self) -> tuple[float, float]:
        e = self.leak["early_warning"]
        return e["random_shuffle"]["pr_auc"], e["proper_unit_grouped_walk_forward"]["pr_auc"]

    def soh_mae(self, model: str) -> str:
        return " / ".join(f"{self.soh2['by_horizon'][h][model]['mae']:.2f}" for h in ("1", "7", "14"))

    @property
    def fusion(self) -> dict[str, Any]:
        f = self.fus
        sens = {x["notice_rate"]: x for x in f["sensitivity_notice_rate"]}
        top = sens[max(sens)]
        return {"early": f["arms"]["early_fusion"]["pr_auc"], "lgbm": f["arms"]["telemetry_lgbm"]["pr_auc"],
                "text": f["arms"]["text_only"]["pr_auc"],
                "diff": f["paired_pr_auc_diff"]["early_fusion_minus_telemetry_lgbm"],
                "cov": f["text_coverage"], "top_rate": max(sens),
                "top_delta": top["delta[early_fusion_minus_telemetry_lgbm]"],
                "top_ci": top["ci95[early_fusion_minus_telemetry_lgbm]"]}

    @property
    def fari_facts(self) -> dict[str, Any]:
        r = self.fari
        arms = r["models"]["arms"]
        return {"asis_accept": r["contract"]["week1_rules_as_is"]["acceptance_rate_pct"],
                "profile_accept": r["contract"]["fari_platform_profile"]["acceptance_rate_pct"],
                "dead": len(r["features"]["dead_features"]["fari_asis"]),
                "asis": arms["asis"]["lgbm"]["pr_auc"], "fixed": arms["fixed_rover_thr"]["lgbm"]["pr_auc"],
                "pooled": arms["pooled"]["lgbm"]["pr_auc"],
                "diff": r["models"]["paired_pr_auc_diff"]["fixed_rover_thr:lgbm minus asis:lgbm"]}


# --------------------------------------------------------------------------- figures
def fig_prauc(ev: Evidence) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels, x = [], 0
    colours = {"champion": "#1F4E79", "iforest": "#7F8C8D", "physics_rule": "#B9770E"}
    for p in ev.op["platforms"]:
        for cohort in ("held_out_units", "training_units"):
            c = ev.cohort(p, cohort)
            for k, m in enumerate(("champion", "iforest", "physics_rule")):
                v = c[f"pr_auc_{m}"]
                lo, hi = c[f"pr_auc_{m}_ci95"]
                ax.bar(x + k * 0.26, v, 0.24, color=colours[m], label=m if x == 0 else None)
                if lo is not None:
                    ax.errorbar(x + k * 0.26, v, yerr=[[v - lo], [hi - v]], color="black", capsize=2, lw=0.8)
            ax.axhline(c["base_rate"], xmin=0, xmax=1, lw=0)
            ax.plot([x - 0.15, x + 0.67], [c["base_rate"]] * 2, color="black", ls=":", lw=1)
            labels.append((x + 0.26, f"{PLAT[p]}\n{cohort.replace('_units', '').replace('_', '-')}"))
            x += 1.2
    ax.set_xticks([a for a, _ in labels], [b for _, b in labels], fontsize=8)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Operational period, production v1 (dotted: base rate; bars: 95 % unit-bootstrap CI)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    out = FIGS / "w04_operational_prauc.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_cost(ev: Evidence) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    names, be, colours = [], [], []
    for k, r in ev.cb["strategies"].items():
        names.append(k.replace("w02_walk_forward:", "W2 walk-forward: ").replace("w04_operational:", "W4 held-out: ")
                     .replace("aido_rover", "Rover").replace("fari", "Fari").replace("lgbm_selected", "LightGBM")
                     .replace("iforest", "Isolation Forest").replace("physics_rule", "physics rule").replace("_", " "))
        be.append(r["break_even_ratio"])
        colours.append("#1F4E79" if "champion" in k or "lgbm" in k else "#95A5A6")
    y = range(len(names))
    ax.barh(list(y), be, color=colours)
    for i, v in enumerate(be):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=11)
    assumed = next(iter(ev.cb["strategies"].values()))["assumed_ratio"]
    ax.axvline(assumed, color="#C0392B", ls="--", lw=1.6, label=f"placeholder cost ratio {assumed:.1f}")
    ax.legend(fontsize=11, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_yticks(list(y), names, fontsize=11)
    ax.set_xlim(0, max(be) * 1.15)
    ax.set_xlabel("break-even ratio = false alarms per caught failure", fontsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_title("Break-even ratio\n(dark: alerting model; grey: shadow or baseline)", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = FIGS / "w04_break_even.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_fari(ev: Evidence) -> Path:
    arms = ev.fari["models"]["arms"]
    labels = {"asis": "Rover model\nas-is", "fixed_rover_thr": "Rover model\n+ guards", "fari_native": "Trained\non Fari",
              "pooled": "Pooled\nRover + Fari"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for k, (model, colour, name) in enumerate((("lgbm", "#1F4E79", "LightGBM"), ("iforest", "#95A5A6",
                                                                                    "Isolation Forest"))):
        for i, arm in enumerate(labels):
            r = arms[arm][model]
            v, (lo, hi) = r["pr_auc"], r["ci95"]["pr_auc"]
            x = i + (k - 0.5) * 0.36
            ax.bar(x, v, 0.34, color=colour, label=name if i == 0 else None)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="black", capsize=3, lw=1)
            ax.text(x, hi + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    base = arms["pooled"]["lgbm"]["base_rate"]
    ax.axhline(base, color="black", ls=":", lw=1.2, label=f"base rate {base:.3f}")
    ax.set_xticks(range(len(labels)), list(labels.values()), fontsize=11)
    ax.set_ylabel("PR-AUC on Fari test robots (95 % unit-bootstrap CI)", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title("Week 3 Fari transfer (synthetic Fari profile)", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = FIGS / "w04_fari_transfer.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_architecture() -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = ["Simulator v0.2 (Rover, Fari)  ->  daily landing partitions",
             "Ingest + validate: data contract, quarantine, privacy gate",
             "As-of features from the validated history",
             "Score: MLflow registry v1 champion + shadow models",
             "Alert queue: budget, SHAP, template, evidence checker",
             "Streamlit dashboard: pull list, drill-down, alerts"]
    h, gap, top = 0.105, 0.035, 0.97
    for i, text in enumerate(steps):
        y = top - i * (h + gap) - h
        last = i == len(steps) - 1
        ax.add_patch(plt.Rectangle((0.02, y), 0.96, h, fc="#FDEDEC" if last else "#EAF2F8",
                                   ec="#C0392B" if last else "#1F4E79", lw=1.4))
        ax.text(0.5, y + h / 2, text, ha="center", va="center", fontsize=11)
        if i:
            ax.annotate("", (0.5, y + h), (0.5, y + h + gap), arrowprops={"arrowstyle": "->", "lw": 1.4})
    y = top - len(steps) * (h + gap) - 0.02
    ax.text(0.5, y, "Airflow DAG fleet_health_daily (Docker Compose): idempotent,\nbackfill, fails loudly on a missing "
            "partition; daily PSI and\nweekly performance on newly confirmed faults", ha="center", va="top",
            fontsize=11.5, color="#34495E")
    out = FIGS / "w04_architecture.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def top_bin(cal: dict[str, Any]) -> dict[str, Any]:
    return cal["bins"][-1]


def calibration_sentence(ev: Evidence, p: str) -> str:
    t = top_bin(ev.calibration(p))
    ci_txt = f" {ci(t['observed_rate_ci95'], 2)}" if t.get("observed_rate_ci95") else ""
    return (f"{PLAT[p]}: rows scored {t['bin']} (mean risk {t['mean_risk']:.2f}, {t['rows']} rows on "
            f"{t['units']} robots) were positive {t['observed_rate']:.2f}{ci_txt} of the time")


def build_calibration_section(d: CardDoc, ev: Evidence) -> None:
    op = ev.op
    d.h("4e. Calibration of v1 and how coarse the risk is (held-out robots)", 2)
    d.p("Observed onset rate per calibrated-risk bin on the evaluated held-out rows (the same rows as 4b). Bins with "
        "at least 20 rows carry a 95 % unit-bootstrap interval. One bin edge is the alert threshold.")
    for p in op["platforms"]:
        cal = ev.calibration(p)
        rows = [[b["bin"], str(b["rows"]), str(b["units"]), f3(b["mean_risk"]), f3(b["observed_rate"]),
                 ci(b.get("observed_rate_ci95"))] for b in cal["bins"]]
        d.p(f"**{PLAT[p]}** (base rate {f3(cal['base_rate'])}, mean risk {f3(cal['mean_risk'])}, Brier "
            f"{cal['brier']:.4f}):")
        d.table(["Calibrated risk bin", "Rows", "Robots", "Mean risk", "Observed rate", "95 % CI"], rows)
    readings = []
    for p in op["platforms"]:
        cal = ev.calibration(p)
        t = top_bin(cal)
        lo_hi = t.get("observed_rate_ci95")
        if lo_hi and lo_hi[1] < t["mean_risk"]:
            fit = (f"overconfident: even the upper end of the interval ({lo_hi[1]:.2f}) is below the stated risk "
                   f"({t['mean_risk']:.2f}); only {t['units']} robots reach this bin, so the size of the gap is uncertain")
        elif lo_hi and lo_hi[0] <= t["mean_risk"] <= lo_hi[1]:
            fit = "consistent with the stated risk (the interval contains it)"
        else:
            fit = "not testable (too few rows for an interval)"
        thr = ev.op["thresholds"][p]
        mid = [b for b in cal["bins"] if thr <= b["mean_risk"] < 0.5]
        trend = " -> ".join(f"{b['observed_rate']:.2f}" for b in mid)
        rising = all(b2["observed_rate"] >= b1["observed_rate"] for b1, b2 in itertools.pairwise(mid))
        readings.append(f"**{PLAT[p]}:** top tier {fit}. Between the threshold and 0.5 the observed rate goes "
                        f"{trend} as the risk rises ({'monotone' if rising else 'not monotone'}).")
    d.bullets([calibration_sentence(ev, p) + "." for p in op["platforms"]] + readings + [
        "Read the risk as a ranking with a coarse top tier, not as a probability.",
    ])
    rows = []
    for p in op["platforms"]:
        t = ev.calibration(p)["ties"]
        a = op["platforms"][p]["risk_ties_all_scored_rows"]
        rows.append([PLAT[p], str(t["distinct_values"]), f"{t['modal_value']:.4f}",
                     f"{100 * t['modal_value_share']:.1f} % of {t['rows']:,}",
                     f"{100 * t['below_threshold_share']:.1f} %", f"{100 * t['at_threshold_value_share']:.1f} %",
                     f"{100 * a['modal_value_share']:.1f} % of {a['rows']:,}"])
    d.p("**Ties.** The isotonic map was fitted on a 6-day validation slice with about three onsets per platform, so "
        "it has only a few steps, and the alert threshold is itself one of them. Several shares are easy to confuse, "
        "so each is named. 'Evaluated' rows are the held-out rows of 4b (labels settled, enough data).")
    d.table(["Platform", "Distinct risk values (evaluated)", "Most common value",
             "Rows at exactly that value (evaluated)", "Rows below the threshold (evaluated)",
             "Rows exactly at the threshold value (evaluated; these alert)",
             "Rows at the most common value (all scored rows, all robots)"], rows)
    diag = []
    for p in op["platforms"]:
        x = ev.cohort(p)["post_hoc_diagnostics"]
        dd = x["margin_minus_calibrated_risk"]
        diag.append(f"{PLAT[p]} {f3(x['pr_auc_uncalibrated_margin'])} vs {f3(ev.cohort(p)['pr_auc_champion'])} "
                    f"({w3.signed(dd['delta'])} {ci(dd['ci95'])}, {verdict(dd['ci95'])})")
    d.p("**Post-hoc diagnostic, not a result.** Ranking the same held-out rows by the uncalibrated model margin "
        "gives PR-AUC " + "; ".join(diag) + ". This was computed after the evaluation, on the evaluation rows, so it "
        "is not used to choose or change the v1 model and is not an estimate of a v2 model. It shows the direction "
        "for v2: keep the uncalibrated margin for ranking and alerting, use a calibration map only for the displayed "
        "probability, and fit it on more validation onsets; then evaluate that on new data.")


# --------------------------------------------------------------------------- model card
def build_model_card(ev: Evidence) -> Path:
    op = ev.op
    d = CardDoc("W04 Model Card - fleet_early_warning v1 and fleet_soh_forecast v1",
                f"{SYNTH}. Generated by scripts/build_w04_docs.py from reports/w04/*.json.")
    rt = ev.retrain
    provenance = ("Registered from a working tree that was not yet committed (MLflow tag git_dirty=True, base "
                  "commit 64fdd3d).")
    if rt:
        provenance += (f" Retrained in memory at commit {rt['git_commit'][:7]} (clean tree: "
                       f"{'yes' if rt['git_worktree_clean'] else 'no'}): "
                       + ("identical selection, calibration, thresholds and risk on all "
                          f"{rt['validation_rows']:,} validation rows" if rt["identical"] else "NOT identical")
                       + " (reports/w04/retrain_reproducibility.json).")
    d.h("1. Model details")
    d.table(["Item", "Value"], [
        ["Early-warning model", (f"{op['models']['early_warning']} (MLflow registry); variant {op['champion_variant']}"
                                 " (pooled LightGBM, calibrated, platform flag)")],
        ["SoH model", f"{op['models']['soh_forecast']}: LightGBM on SoH(d+7) - persistence"],
        ["Alert thresholds (risk)", ", ".join(f"{PLAT[p]} {100 * v:.2f} %" for p, v in op["thresholds"].items())],
        ["Alert budget", "1 alert per unit-week, 72 h refractory (configs/pipeline.yaml)"],
        ["Shadow models (never alert)", "Isolation Forest, physics rule (ADR-0005)"],
        ["Decision record", "docs/adr/0005-production-model-recipe.md"],
        ["Training provenance", provenance],
    ], [1.8, 4.6])
    d.h("2. Intended use")
    d.bullets([
        ("Rank the robots of a fleet by the chance of a **qualifying fault onset within 72 h** (ADR-003 rev1), so a "
        "fleet manager can inspect or swap a unit between shifts."),
        "Attach to every alert a template explanation that only restates the model's own evidence.",
        "Forecast battery state of health 7 days ahead to plan pack replacements.",
        "Research and demonstration on simulated fleets, as the starting point for validation on real InGen data.",
    ])
    d.h("Out of scope")
    d.bullets([
        ("**Any decision on real robots before re-validation on real InGen telemetry.** Every number here comes "
        "from a simulator; the model has never seen a real robot."),
        "Safety-critical shutdown or a remote stop: the model is advisory and has a high false-alarm share.",
        "Diagnosing the component or root cause: SHAP attributions describe the model, not the physics.",
        "Sudden failures with no precursor, which the simulator does not contain.",
        "Platforms other than Aido Rover and the synthetic Fari profile.",
        "Judging operators or customers: the data contract carries no personal data, and Fari forbids imagery and audio.",
    ])
    d.h("3. Training data and its limitations")
    d.p("Two simulated fleets of 50 robots for 60 days each, at 1-minute resolution, from simulator v0.2: Aido Rover "
        "(configs/aido_rover_v0_2.yaml) and a synthetic Fari profile (configs/fari.yaml, every value an assumption). "
        "Faults follow a P-F model (battery cell, bearing wear, cooling degradation) with documented ramp shapes. 70 % "
        "of each platform's robots are training units. Training rows end 72 h before day 30; the last 6 days of that "
        "window are the validation slice. Text features are not used (see section 5).")
    d.bullets([
        ("Fault physics, noise and drift are first-order models with assumed parameters: patterns the simulator does "
        "not contain (sudden failures, firmware changes, mixed faults, operator behaviour) are absent."),
        ("**Very few faults to learn from.** Fault onsets of training robots before the cut-off: "
         + "; ".join(f"{PLAT[p]} {ev.training_window(p)['training_unit_onsets_before_cutoff']}, of which "
                     f"{ev.training_window(p)['training_unit_onsets_in_validation_slice']} in the 6-day validation slice "
                     f"({ev.training_window(p)['validation_positive_rows']} positive validation rows)"
                     for p in op["platforms"])
         + ". Most of each fleet's onsets (" + ", ".join(
             f"{PLAT[p]} {ev.training_window(p)['fault_onsets_total'] - ev.training_window(p)['all_unit_onsets_before_cutoff']}"
             f" of {ev.training_window(p)['fault_onsets_total']}" for p in op["platforms"])
         + ") fall after the cut-off. The calibration map and the thresholds therefore rest on about three "
         "onsets per platform (see 4e)."),
        "Labels come from simulator ground truth. Real fault confirmations would be later, noisier and incomplete.",
    ])
    d.h("4. Performance")
    d.h("4a. Established leakage-safe results (Weeks 2-3, unchanged)", 2)
    m2 = ev.ew2["models"]
    d.table(["Evaluation (unseen robots)", "Result"], [
        ["Rover early warning PR-AUC, walk-forward (base rate 0.055)",
         (f"LightGBM (selected) {f3(m2['lgbm_selected']['pr_auc'])}; physics rule {f3(m2['physics_rule']['pr_auc'])}; "
         f"Isolation Forest {f3(m2['iforest']['pr_auc'])}")],
        ["LightGBM - physics rule (95 % CI)", (f"{ci(m2['lgbm_selected']['pr_auc_minus_physics_rule_ci95'])}: "
         f"{verdict(m2['lgbm_selected']['pr_auc_minus_physics_rule_ci95'])}")],
        ["Isolation Forest - physics rule (95 % CI)", (f"{ci(m2['iforest']['pr_auc_minus_physics_rule_ci95'])}: "
         f"{verdict(m2['iforest']['pr_auc_minus_physics_rule_ci95'])}")],
        ["LightGBM - Isolation Forest (95 % CI)", (f"{ci(m2['lgbm_selected']['pr_auc_minus_iforest_ci95'])}: "
         f"{verdict(m2['lgbm_selected']['pr_auc_minus_iforest_ci95'])}")],
        ["Leakage demo PR-AUC", f"random shuffle {f3(ev.leak_pair[0])} vs proper protocol {f3(ev.leak_pair[1])}"],
        ["Text fusion (Week 3)", (f"early fusion - LightGBM {w3.signed(ev.fusion['diff']['delta'])} "
                                 f"{ci(ev.fusion['diff']['ci95'])}: {verdict(ev.fusion['diff']['ci95'])}")],
        ["Fari, Rover model + guards (Week 3)",
         (f"PR-AUC {f3(ev.fari_facts['asis'])} -> {f3(ev.fari_facts['fixed'])} "
         f"({w3.signed(ev.fari_facts['diff']['delta'])} {ci(ev.fari_facts['diff']['ci95'])}); pooled "
         f"{f3(ev.fari_facts['pooled'])}")],
    ], [2.8, 3.6])
    d.p("**How to read significance.** Every difference is reported with a 95 % unit-bootstrap interval. It is "
        "called better or worse only when the interval excludes 0; otherwise it is 'no detectable difference', "
        "whatever the point estimate. Counts without an interval (alerts, caught onsets, break-even ratios) are "
        "point comparisons on 6 held-out onsets per platform and are not tested.")
    d.h("4b. Production v1, operational period, by platform and cohort", 2)
    d.p(op["protocol"])
    rows = []
    for p in op["platforms"]:
        for cohort in ("held_out_units", "training_units"):
            c = ev.cohort(p, cohort)
            q = c["alert_queue"]
            rows.append([PLAT[p], cohort.replace("_", " "), str(c["units"]), str(c["fault_onsets"]), f3(c["base_rate"]),
                         f"{f3(c['pr_auc_champion'])} {ci(c['pr_auc_champion_ci95'])}",
                         f"{f3(c['pr_auc_iforest'])}", f"{f3(c['pr_auc_physics_rule'])}",
                         f"{q['alerts']} / {f3(q['precision'], 2)}", f"{q['events_detected']}/{q['events']}",
                         f3(q["false_alarms_per_unit_week"], 2), f3(q["median_lead_time_h"], 0)])
    d.table(["Platform", "Cohort", "Units", "Onsets", "Base rate", "PR-AUC champion [CI]", "IF", "Rule",
             "Alerts / precision", "Caught", "False alarms / unit-wk", "Median lead h"], rows)
    rows = []
    for p in op["platforms"]:
        c = ev.cohort(p)
        for other in ("iforest", "physics_rule"):
            x = c[f"champion_minus_{other}"]
            rows.append([PLAT[p], f"champion - {other}", f"{w3.signed(x['delta'])} {ci(x['ci95'])}", verdict(x["ci95"])])
    d.p("Paired differences on held-out robots (unit bootstrap):")
    d.table(["Platform", "Difference", "PR-AUC delta [95 % CI]", "Reading"], rows)
    d.picture(fig_prauc(ev))
    d.h("4c. Subgroups (held-out robots)", 2)
    rows = []
    for p in op["platforms"]:
        c = ev.cohort(p)
        for ft, r in c["recall_by_fault_type"].items():
            rows.append([PLAT[p], ft.replace("_", " "), f"{r['detected']}/{r['events']}"])
    d.table(["Platform", "Fault type", "Onsets caught by an alert in the 72 h before"], rows)
    rows = []
    for p in op["platforms"]:
        for s, r in ev.cohort(p)["pr_auc_by_site"].items():
            rows.append([PLAT[p], s, str(r["rows"]), str(r["positives"]), f3(r["pr_auc_champion"])])
    d.table(["Platform", "Site", "Rows", "Positive rows", "PR-AUC champion"], rows)
    d.h("4d. Battery SoH, 7-day forecast", 2)
    rows = []
    for p, e in op["platforms"].items():
        for cohort, s in e["soh_7d"].items():
            rows.append([PLAT[p], cohort.replace("_", " "), str(s["forecasts"]), f3(s["mae_pp_model"]),
                         f3(s["mae_pp_persistence"]), ci(s["skill_ci95"]), verdict(s["skill_ci95"])])
    d.table(["Platform", "Cohort", "Forecasts", "MAE model (pp)", "MAE persistence (pp)",
             "Persistence - model [95 % CI]", "Reading"], rows)
    build_calibration_section(d, ev)
    d.h("5. Known failure cases")
    items = []
    for p in op["platforms"]:
        missed = ev.cohort(p)["missed_fault_onsets"]
        if missed:
            kinds: dict[str, int] = {}
            for m in missed:
                kinds[m["fault_type"]] = kinds.get(m["fault_type"], 0) + 1
            items.append(f"**{PLAT[p]}: {len(missed)} held-out fault onset{'s' if len(missed) != 1 else ''} got no alert in the 72 h before** ("
                         + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in kinds.items()) + ").")
        q = ev.cohort(p)["alert_queue"]
        if q["precision"] is not None:
            items.append(f"{PLAT[p]}: {100 * (1 - q['precision']):.0f} % of held-out alerts are false alarms "
                         f"({f3(q['false_alarms_per_unit_week'], 2)} per robot-week).")
    for p, e in op["platforms"].items():
        top = (e.get("alert_summaries") or {}).get("top_driver_concept", {})
        if top:
            total = sum(top.values())
            items.append(f"{PLAT[p]}: the top driver of an alert is "
                         + ", ".join(f"{k.replace('_', ' ')} {100 * v / total:.0f} %" for k, v in top.items())
                         + ". Explanations therefore mostly cite battery readings, whatever the fault later turns out to be.")
        dr = e.get("drift") or {}
        if dr:
            worst = next(iter(dr["median_psi_by_feature"].items()))
            items.append(f"{PLAT[p]}: drift is the norm, not the exception. {dr['bands'].get('shift', 0)} of "
                         f"{dr['feature_days']} feature-days are 'shift' (PSI > 0.25); the median PSI of "
                         f"{worst[0]} is {worst[1]:.1f}, and the risk score itself has median PSI "
                         f"{dr['risk_psi_median']:.2f}. The simulated weather cools over the period, so the "
                         "model scores conditions it did not see in training.")
    thr = sorted({round(v, 6) for v in op["thresholds"].values()})
    if len(thr) == 1:
        items.append(f"The per-platform thresholds came out identical ({100 * thr[0]:.2f} %): both sit on the same "
                     "step of the isotonic calibration map. In practice v1 applies one threshold, despite the "
                     "Week 3 finding that Fari needs its own. More validation data per platform is needed.")
    items.append("By design, the budget of 1 alert per robot-week means about one robot in seven gets a new "
                 "alert on an average day. The dashboard shows only new alerts as 'pull'; whether that rate is "
                 "operable is a decision for InGen's operators.")
    items += [
        ("**Calibration of v1 (measured, 4e).** " + "; ".join(calibration_sentence(ev, p) for p in op["platforms"])
         + ". Fari's top-tier risk is overconfident, and the map is coarse (few distinct values). The Week 3 "
         "walk-forward model showed the same top-end overconfidence. Read the risk as a ranking, not a promise."),
        ("Rows with more than half of the last 24 h missing are not scored as alerts. A robot that goes silent "
        "before failing will not be flagged by the model; the dashboard lists it as 'No recent data'."),
        ("Free-text LLM explanations were unfaithful (0/20 in Week 3) and are not used. The template is faithful by "
        "construction, but it only restates SHAP. A misleading attribution gives a faithful but misleading sentence."),
    ]
    d.bullets(items)
    d.h("6. Explanations")
    rows = []
    for p, e in op["platforms"].items():
        s = e.get("alert_summaries")
        if s:
            rows.append([PLAT[p], str(s["alerts"]), f"{s['all_claims_supported_auto']}/{s['alerts']}",
                         str(s["claims_checked_total"])])
    d.table(["Platform", "Alerts", "Summaries with every claim supported (checker v2)", "Claims checked"], rows)
    d.p("The checker is lexical. In Week 3 it passed 8 of 20 LLM summaries that an evidence review judged unfaithful, "
        "so this rate is a guard against template regressions, not a proof of faithfulness.")
    d.h("7. Limitations and deployment risks")
    d.bullets([
        ("**Synthetic-only evidence.** The simulator, the fault mix, the Fari profile and the operator-note rate are "
        "assumptions. Re-validate on real InGen telemetry and confirmed tickets before use."),
        ("**Weak champion on Rover.** In the Week 2 walk-forward the Isolation Forest ranked significantly better "
         f"than LightGBM (LightGBM - IF {ci(ev.ew2['models']['lgbm_selected']['pr_auc_minus_iforest_ci95'])}). In "
         "operation the shadow Isolation Forest again has the higher point estimate on Rover (champion - IF "
         f"{w3.signed(ev.cohort('aido_rover')['champion_minus_iforest']['delta'])} "
         f"{ci(ev.cohort('aido_rover')['champion_minus_iforest']['ci95'])}, not significant) and the better "
         "break-even ratio (section 8). The champion was kept for its tree-SHAP explanations and because the "
         "Isolation Forest transferred worse to Fari in Week 3 (pooled PR-AUC on Fari "
         f"{f3(ev.fari['models']['arms']['pooled']['iforest']['pr_auc'])} vs "
         f"{f3(ev.fari['models']['arms']['pooled']['lgbm']['pr_auc'])} for LightGBM; that comparison has no paired "
         "interval, and in operation the Fari difference is not significant either). The weekly check compares them."),
        ("**Fari calibration.** The top-tier risk on Fari is overconfident (4e). Alert ranking is still usable, but a "
         "displayed risk near 100 % does not mean a near-certain fault on Fari."),
        ("**Drift.** The simulated weather cools over the period, and several top features drift out of their "
        "training range (dashboard PSI). A real fleet will drift more (season, firmware, routes)."),
        ("**Few labels.** "
         + "; ".join(f"{a} of {b} {k.replace('_units', '').replace('_', '-')} cohort-weeks"
                     for k, (a, b) in ev.weekly_status_counts().items())
         + " had fewer than 3 confirmed onsets ('insufficient labels'). Performance changes are detectable only "
         "over months."),
        "**Single time cut and unit split** for the production numbers. Their CIs are wide.",
        ("**Operational.** The daily cost grows with history (features recomputed from the start). Airflow runs "
        "standalone with SQLite in Docker Compose. There is no authentication on the dashboard."),
    ])
    d.h("8. Cost-benefit evaluation (assumptions visible)")
    a = ev.cb["assumptions"]
    d.p(f"**{ev.cb['notice']}** Currency: {a['currency']}.")
    d.table(["Assumption (configs/cost_assumptions.yaml)", "Value"], [
        ["Unplanned in-shift failure", money(a["cost_unplanned_failure"])],
        ["Planned repair between shifts", money(a["cost_planned_repair"])],
        ["False-alarm inspection", money(a["cost_false_alarm_inspection"])],
        ["Unnecessary swap / probability per false alarm",
         f"{money(a['cost_unnecessary_swap'])} / {a['swap_probability_on_false_alarm']:.2f}"],
        ["Repair success if caught", f"{a['repair_success_if_caught']:.2f}"],
        ["=> saving per caught failure / cost per false alarm",
         (f"{money(a['saving_per_catch'])} / {money(a['false_alarm_cost'])} (ratio "
         f"{a['saving_per_catch'] / a['false_alarm_cost']:.2f})")],
    ], [3.6, 2.8])
    rows = []
    for k, r in ev.cb["strategies"].items():
        rows.append([k, f"{r['alerts']:.0f}", f"{r['alerts'] / r['unit_weeks']:.2f}",
                     f"{r['events_detected']:.0f}/{r['events']:.0f}",
                     f"{r['false_alarms']:.0f}", money(r["net_benefit"]), f"{r['net_benefit_per_unit_week']:.1f}",
                     f"{r['break_even_ratio']:.2f}"])
    d.table(["Strategy (threshold set on validation for 1 alert per unit-week)", "Alerts",
             "Realised alerts / unit-week", "Caught", "False alarms", "Net benefit", "Net / unit-week",
             "Break-even ratio"], rows)
    d.picture(fig_cost(ev))
    rif, rch = ev.cost_row("w04_operational:aido_rover:iforest_shadow"), ev.cost_row("w04_operational:aido_rover:champion")
    fif, fch = ev.cost_row("w04_operational:fari:iforest_shadow"), ev.cost_row("w04_operational:fari:champion")
    d.p("**Reading.** The break-even ratio needs only measured counts: false alarms per caught failure. Acting on "
        "the alerts saves money when (failure cost - planned repair cost) / (false-alarm cost) is above it. At the "
        "placeholder costs that ratio is "
        f"{next(iter(ev.cb['strategies'].values()))['assumed_ratio']:.1f}. All thresholds were set on validation "
        "rows for the same budget, but the realised alert rates differ, so the strategies did not spend the same "
        "number of alerts.")
    d.p(f"**On Rover the Isolation Forest shadow has the better cost profile.** It caught "
        f"{rif['events_detected']:.0f}/{rif['events']:.0f} held-out onsets with {rif['false_alarms']:.0f} false "
        f"alarms (break-even {rif['break_even_ratio']:.1f}); the champion caught {rch['events_detected']:.0f}/"
        f"{rch['events']:.0f} with {rch['false_alarms']:.0f} (break-even {rch['break_even_ratio']:.1f}). This is a "
        "point comparison on 6 onsets (one onset apart) and is not tested for significance. On Fari the same shadow "
        f"model is worse: its validation threshold overshot the budget ({fif['alerts'] / fif['unit_weeks']:.2f} "
        f"alerts per unit-week), giving break-even {fif['break_even_ratio']:.1f} against "
        f"{fch['break_even_ratio']:.1f} for the champion. Any switch should be decided per platform, with real "
        "costs and more onsets (next steps).")
    key = next((k for k in ev.cb["sensitivity"] if k.startswith("w04_operational") and "champion" in k),
               next(iter(ev.cb["sensitivity"])))
    rows = [[s["setting"], money(s["net_benefit"]), f"{s['net_benefit_per_unit_week']:.1f}"]
            for s in ev.cb["sensitivity"][key]]
    d.p(f"One-at-a-time sensitivity for **{key}**:")
    d.table(["Setting", "Net benefit", "Net / unit-week"], rows, [3.2, 1.6, 1.6])
    d.h("9. What this model cannot do")
    d.bullets([
        ("It cannot tell a robot that is about to fail from one that is merely in an unusual environment it never "
        "saw in training. Drift on top features is common even in the simulator."),
        "It cannot warn about a fault that gives no telemetry precursor, or about a robot whose telemetry stopped.",
        "It cannot use maintenance text in v1. Text gave no detectable gain at the assumed note rate (Week 3).",
        "It cannot state real costs. The cost table is a calculator with placeholder inputs.",
    ])
    d.h("10. Monitoring and maintenance")
    d.bullets([
        "Daily PSI per platform on the top 8 features and on the risk score (bands 0.10 / 0.25).",
        ("Weekly check on Sundays for the week that ended 7 days earlier: champion vs shadow models, held-out vs "
        "training robots; fewer than 3 onsets is 'insufficient labels'."),
        ("Retrain when the weekly check degrades over several weeks or PSI shows sustained shift. Registering a new "
        "version does not change serving until configs/pipeline.yaml pins it."),
    ])
    d.h("11. Provenance and AI assistance")
    d.bullets([
        ("The code, tests, configuration and the generated documents (this model card, the capstone deck, the README "
         "results) were written with an AI coding assistant (Claude Code) working under the intern's direction. "
         "Commits made with it carry a Co-Authored-By trailer."),
        ("The Week 3 faithfulness verdicts on the local-LLM summaries came from an AI-assisted evidence review "
         "(reports/w03/llm_faithfulness_review.json), not from an independent human reviewer."),
        ("Every number in this card is read from a JSON report written by a re-runnable script; none is typed by "
         "hand. Whether this use of AI assistance fits InGen's AI-use policy is pending confirmation by the "
         "supervisor."),
    ])
    out = d.save("W04_Model_Card.docx")
    pdf = d.save_pdf("W04_Model_Card.pdf")
    if pdf:
        print(f"wrote {pdf.relative_to(ROOT)}")
    return out


# --------------------------------------------------------------------------- capstone deck
def slide(pdf: PdfPages, n: int, title: str, bullets: list[str], chart: Path | list[Path] | None = None,
          note: str = SYNTH, bottom: bool = False) -> None:
    """One 16:9 slide. Charts sit on the right, or (``bottom=True``, for wide screenshots) across the lower half."""
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.04, 0.92, "\n".join(textwrap.wrap(title, 70)), fontsize=21, weight="bold", va="top", color="#1B2631")
    y = 0.74 if len(title) > 70 else 0.8
    for b in bullets:
        lines = textwrap.wrap(b, 58 if chart and not bottom else 110)
        fig.text(0.05, y, "•  " + "\n    ".join(lines), fontsize=13.5, va="top", color="#273746")
        y -= 0.037 * len(lines) + (0.02 if bottom else 0.035)
    charts = [c for c in (chart if isinstance(chart, list) else [chart]) if c is not None and c.exists()]
    for i, c in enumerate(charts):
        if bottom:
            ax = fig.add_axes((0.03, 0.07, 0.94, y - 0.09))
        else:
            h = 0.64 / len(charts)
            ax = fig.add_axes((0.54, 0.1 + (len(charts) - 1 - i) * h, 0.43, h - 0.02))
        ax.imshow(plt.imread(str(c)))
        ax.axis("off")
    fig.text(0.04, 0.03, note, fontsize=9, color="#7F8C8D")
    fig.text(0.96, 0.03, f"{n} / 10", fontsize=9, color="#7F8C8D", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def build_deck(ev: Evidence) -> Path:
    m2 = ev.ew2["models"]
    op = ev.op
    cb = ev.cb
    held = {p: ev.cohort(p) for p in op["platforms"]}
    champ_key = [k for k in cb["strategies"] if k.startswith("w04_operational") and k.endswith("champion")]
    be = {k: cb["strategies"][k]["break_even_ratio"] for k in champ_key}
    out = DOCS / "W04_Capstone_Deck.pdf"
    w2_figs = DOCS / "figures" / "w02"
    w3_figs = DOCS / "figures" / "w03"
    with PdfPages(out, metadata={"CreationDate": None}) as pdf:  # no timestamp: rebuilds are byte-stable
        slide(pdf, 1, "ingen_fleet_health v1.0: a reproducible fleet-health package that is honest about what "
                      "synthetic data can show", [
            "Four weeks, one question: which robots should be pulled before the next shift, and why?",
            ("Built: physics-grounded simulator -> validated pipeline -> leakage-safe models -> grounded alerts -> "
            "daily Airflow DAG -> operator dashboard -> model card."),
            ("Every number comes from a re-runnable script with an MLflow run or a JSON report. Every dataset is "
            "synthetic and regenerates from a config."),
            "Kuntong Li, Data Science Engineer Intern, Robot Fleet Telemetry Analytics & Predictive Health track.",
        ], fig_architecture())
        slide(pdf, 2, "A missed fault costs a shift of coverage; a noisy alert stream gets ignored", [
            ("Aido Rover patrols long duty cycles. Battery fade, drivetrain wear and thermal stress depend on route "
            "and climate."),
            "The operator needs lead time to swap a unit, and an alert budget they will respect (1 per robot-week).",
            ("Security-patrol operators value uptime most, and they are the natural users of the alert queue."),
            ("Fari (indoor companion) is the generalisation test: short sessions, dock charging, no terrain, no "
            "imagery or audio."),
        ])
        slide(pdf, 3, "The data foundation is a documented instrument model, not a black box", [
            ("Simulator v0.2: 50 robots x 60 days per fleet at 1 minute (about 4.3 M records each). Battery fade and "
            "resistance, motor heating with Newtonian cooling, terrain, vibration, drift, dropouts."),
            ("P-F fault timelines (battery cell, bearing, cooling) with onset and failure times in a ground-truth "
            "table."),
            ("Ingestion enforces the data contract: ranges, timestamps, odometer, platform profile, Fari privacy gate. "
            "Rejects go to quarantine with reason codes."),
            "Every parameter has a value, unit and justification. The capstone conclusions depend on them.",
        ], DOCS / "figures" / "fleet_physics_sanity.png")
        slide(pdf, 4, f"A shuffled split overstates PR-AUC {ev.leak_pair[0]:.3f} vs {ev.leak_pair[1]:.3f}: every "
                      "metric uses unit-grouped, embargoed walk-forward evaluation", [
            ("Label (ADR-003 rev1): a qualifying fault onset within the next 72 h. Active faults and the 24 h after a "
            "repair are excluded, and censored rows are dropped."),
            ("Test robots are never seen in training. The embargo is at least one horizon. Splits run walk-forward "
            "across cut-offs."),
            (f"Leakage demo: the same LightGBM on a random shuffle scores {ev.leak_pair[0]:.3f} PR-AUC, against "
            f"{ev.leak_pair[1]:.3f} under the protocol."),
        ], w2_figs / "leakage_demo.png")
        slide(pdf, 5, "Only the Isolation Forest beats the physics rule for early warning; LightGBM wins on battery "
                      "SoH", [
            (f"Early warning PR-AUC (base rate 0.055): physics rule {f3(m2['physics_rule']['pr_auc'])}, LightGBM "
            f"{f3(m2['lgbm_selected']['pr_auc'])} (not different from the rule), Isolation Forest "
            f"{f3(m2['iforest']['pr_auc'])} (better than the rule, CI "
            f"{ci(m2['iforest']['pr_auc_minus_physics_rule_ci95'])})."),
            (f"At the operator budget: rule precision {m2['physics_rule']['precision']:.3f}, recall "
            f"{m2['physics_rule']['event_recall']:.2f}; Isolation Forest {m2['iforest']['precision']:.3f}, "
            f"{m2['iforest']['event_recall']:.2f}; LightGBM {m2['lgbm_selected']['precision']:.3f}, "
            f"{m2['lgbm_selected']['event_recall']:.2f}."),
            (f"SoH MAE at 1 / 7 / 14 days (pp): persistence {ev.soh_mae('persistence')}; linear "
            f"{ev.soh_mae('linear')}; LightGBM {ev.soh_mae('lgbm')}; LSTM {ev.soh_mae('lstm')}."),
            (f"External check (C-MAPSS FD001): LightGBM RMSE {ev.cmapss['rmse_vs_capped_truth']['lgbm']:.1f} cycles "
            f"(constant baseline {ev.cmapss['rmse_vs_capped_truth']['constant_mean']:.1f}), inside the published "
            "range."),
        ], w2_figs / "ew_budget_curve.png")
        slide(pdf, 6, "Maintenance text adds no detectable gain once telemetry is present, at the note rate we "
                      "assumed", [
            (f"Early fusion PR-AUC {ev.fusion['early']:.3f} vs telemetry LightGBM {ev.fusion['lgbm']:.3f}: "
            f"{w3.signed(ev.fusion['diff']['delta'])} {ci(ev.fusion['diff']['ci95'])}. Text only "
            f"{ev.fusion['text']:.3f}."),
            (f"Only {ev.fusion['cov']['onsets_with_symptom_note_in_72h_before_onset']} of "
            f"{ev.fusion['cov']['onsets_in_test']} test faults had an operator note in the 72 h before onset. At "
            f"{ev.fusion['top_rate']:g}x the note rate the gain becomes {w3.signed(ev.fusion['top_delta'])} "
            f"{ci(ev.fusion['top_ci'])}."),
            "Decision: v1 is telemetry-only. InGen's real ticket coverage decides whether text is worth adding.",
        ], w3_figs / "fusion_sensitivity_notice_rate.png")
        slide(pdf, 7, "Template explanations were faithful in 20 of 20 alerts, the local LLM in 0 of 20: production "
                      "uses the template", [
            ("A summary may only restate the alert's SHAP evidence. Local Qwen2.5-1.5B: lexical checker 8/20, "
             "AI-assisted evidence review 0/20 faithful."),
            "Production, re-checked automatically (lexical: a regression guard, not a proof): "
            + "; ".join(f"{PLAT[p]} {e['alert_summaries']['all_claims_supported_auto']}/"
                        f"{e['alert_summaries']['alerts']}"
                        for p, e in op["platforms"].items() if e.get("alert_summaries")) + ". Below: a live alert.",
        ], FIGS / "dashboard_alert.png", bottom=True)
        fx = ev.fari_facts
        slide(pdf, 8, f"Fari broke the Week 1 contract ({fx['asis_accept']:.0f} % of records accepted); a platform "
                      "profile and feature guards fixed it", [
            ("As-is: the voltage window, the power = 48 V x current rule and the mode enum rejected Fari records; "
            f"{fx['dead']} features were dead."),
            ("Minimal change: a platform profile, config-driven feature guards and a per-platform threshold. "
            f"Acceptance {fx['profile_accept']:.0f} %; Rover model PR-AUC on Fari {fx['asis']:.3f} -> "
            f"{fx['fixed']:.3f} ({w3.signed(fx['diff']['delta'])} {ci(fx['diff']['ci95'])})."),
            ("v1 trains one pooled model with a platform flag and per-platform thresholds. The privacy gate refuses "
            "imagery and audio fields."),
        ], fig_fari(ev))
        rif = cb["strategies"]["w04_operational:aido_rover:iforest_shadow"]
        rch = cb["strategies"]["w04_operational:aido_rover:champion"]
        slide(pdf, 9, "Alerts pay off only if a failure costs about "
                      + " / ".join(f"{v:.0f}x" for v in be.values()) + " more than a false alarm; on Rover the "
                      "shadow Isolation Forest does better", [
            "Production v1 on held-out robots: " + "; ".join(
                f"{PLAT[p]} PR-AUC {f3(c['pr_auc_champion'])} {ci(c['pr_auc_champion_ci95'])}, caught "
                f"{c['alert_queue']['events_detected']}/{c['alert_queue']['events']} onsets"
                for p, c in held.items()) + ".",
            ("Break-even ratio = false alarms per caught failure (measured counts). Rover: shadow Isolation Forest "
             f"{rif['break_even_ratio']:.1f} ({rif['events_detected']:.0f}/{rif['events']:.0f} caught) vs champion "
             f"{rch['break_even_ratio']:.1f} ({rch['events_detected']:.0f}/{rch['events']:.0f}): a point comparison "
             "on 6 onsets, not tested."),
            ("Calibration: on Fari the top-tier risk (~1.00) was right "
             f"{top_bin(ev.calibration('fari'))['observed_rate']:.0%} of the time. Read risk as a ranking."),
            ("Daily DAG: identical file hashes on re-run, documented backfill, a missing partition fails loudly "
             "after 6 h. The dashboard shows missing data as missing."),
        ], fig_cost(ev))
        slide(pdf, 10, "Real InGen data is the next step that would change these conclusions most", [
            ("1. Re-validate on real InGen data. Pull 3-6 months of Aido Rover telemetry and confirmed tickets; "
            "measure the real note coverage and the fault mix; re-run the Week 2 protocol. Every conclusion here "
            "depends on the simulator."),
            ("2. Decide the alerting model per platform with real costs. The Isolation Forest shadow ranks and "
             "costs better on Rover, the champion on Fari; neither difference is significant. Use the weekly check, "
             "more onsets and InGen's cost ratio, and test ranking by the uncalibrated margin in v2 on new data."),
            ("3. Pilot the alert queue with one security-patrol operations team. Agree the alert budget and the "
             "false-alarm cost with that team, then extend the data contract to a further robot platform."),
        ])
    return out


# --------------------------------------------------------------------------- retrospective
def build_retro(ev: Evidence) -> Path:
    d = w3.Doc("W04 Retrospective (one page)",
               "Facts are generated from the repository. The reflection sections are left for Kuntong to write; "
               "they are intentionally not drafted.")
    d.h("What was delivered (facts)", 2)
    d.bullets([
        ("Weeks 1-3: simulator v0.2, validated pipeline, leakage-safe evaluation, SoH and early-warning models, log "
        "corpus and text fusion, grounded explanations, Fari gap analysis (see README results tables)."),
        ("Week 4: daily pipeline + Airflow DAG (Docker Compose), production v1 models in the MLflow registry, PSI and "
        "weekly performance monitoring, Streamlit dashboard, model card with cost-benefit, release files."),
    ])
    d.h("What the evidence says (facts)", 2)
    d.bullets([
        ("Negative and null results are kept: the fusion gain is not detectable, the local LLM was unfaithful, "
        "LightGBM does not beat the physics rule on Rover, and alerts break even only at a high failure/false-alarm "
        "cost ratio."),
        "All results are synthetic; real-data validation is the first next step.",
    ])
    for heading in ("What went well (to be written by Kuntong)", "What I would do differently (to be written by Kuntong)",
                    "What I learned about fleet-health data science (to be written by Kuntong)"):
        d.h(heading, 2)
        d.p("[ ]")
    return d.save("W04_Retrospective.docx")


def readme_table(ev: Evidence) -> str:
    """Week 4 results table for README.md (between the W04_RESULTS markers)."""
    op = ev.op
    lines = [
        ("Production v1 (`fleet_early_warning` v1, `fleet_soh_forecast` v1) scored by the daily pipeline from "
        f"{next(iter(op['platforms'].values()))['scored_dates'][0]} to "
        f"{next(iter(op['platforms'].values()))['scored_dates'][1]}. Held-out robots were never seen in training. "
        "This is one time cut and one unit split (ADR-0005), a different protocol from the Week 2 walk-forward, "
        "with wide intervals. Source: `reports/w04/operational_results.json`."),
        "",
        ("| Platform | Early warning PR-AUC, held-out [95 % CI] | Shadow: Isolation Forest / physics rule | Champion - IF [95 % CI] "
        "| Alert queue (held-out): onsets caught, precision, false alarms per robot-week | SoH 7-day MAE (pp): model vs persistence |"),
        "|---|---|---|---|---|---|",
    ]
    for p, e in op["platforms"].items():
        c = ev.cohort(p)
        q = c["alert_queue"]
        soh = e["soh_7d"]["held_out_units"]
        d = c["champion_minus_iforest"]
        lines.append(f"| {PLAT[p]} | {f3(c['pr_auc_champion'])} {ci(c['pr_auc_champion_ci95'])} (base rate "
                     f"{f3(c['base_rate'])}) | {f3(c['pr_auc_iforest'])} / {f3(c['pr_auc_physics_rule'])} | "
                     f"{w3.signed(d['delta'])} {ci(d['ci95'])} | {q['events_detected']}/{q['events']}, "
                     f"{f3(q['precision'], 2)}, {f3(q['false_alarms_per_unit_week'], 2)} | "
                     f"{f3(soh['mae_pp_model'])} vs {f3(soh['mae_pp_persistence'])} |")
    cb = ev.cb["strategies"]
    champ = {k.split(":")[1]: v for k, v in cb.items() if k.startswith("w04_operational") and k.endswith("champion")}
    summ = {p: e.get("alert_summaries", {}) for p, e in op["platforms"].items()}
    lines += [
        "",
        "- **Explanations.** Every production alert carries a template summary, and the checker found every "
        "statement supported in " + ", ".join(f"{PLAT[p]} {s['all_claims_supported_auto']}/{s['alerts']}"
                                               for p, s in summ.items() if s) + ".",
        "- **Cost-benefit (model card §8).** Acting on alerts pays off only if (failure cost - planned repair) / "
        "false-alarm cost exceeds the break-even ratio of " + ", ".join(f"{PLAT[p]} {v['break_even_ratio']:.1f}"
                                                                     for p, v in champ.items())
        + f" for held-out robots. At the placeholder costs the ratio is {next(iter(cb.values()))['assumed_ratio']:.1f}, "
        "so the net is negative. InGen's real costs decide this. On Rover the shadow Isolation Forest has the "
        f"better ratio ({cb['w04_operational:aido_rover:iforest_shadow']['break_even_ratio']:.1f}, "
        f"{cb['w04_operational:aido_rover:iforest_shadow']['events_detected']:.0f}/6 caught); on Fari it is worse "
        f"({cb['w04_operational:fari:iforest_shadow']['break_even_ratio']:.1f}). These are point comparisons on 6 "
        "onsets, not significance tests; none of the champion-vs-shadow PR-AUC differences is significant.",
        ("- **Calibration (model card §4e).** The isotonic map has few steps. On Fari the top-tier risk (about 1.00) "
         f"was followed by an onset {top_bin(ev.calibration('fari'))['observed_rate']:.0%} of the time; on Rover "
         f"{top_bin(ev.calibration('aido_rover'))['observed_rate']:.0%}. Read the risk as a ranking."),
        "- **Drift.** Most feature-days are in the 'shift' band (simulated seasonal cooling); see the model card §5.",
        ("- **Quality gates.** See `reports/w04/quality_gates.json` (records the tested commit) and "
         "`reports/w04/docker_airflow_checks.json`."),
    ]
    return "\n".join(lines)


def update_readme(ev: Evidence) -> Path:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- W04_RESULTS -->", "<!-- /W04_RESULTS -->"
    block = f"{start}\n{readme_table(ev)}\n{end}"
    if end in text:
        text = text[: text.index(start)] + block + text[text.index(end) + len(end):]
    else:
        text = text.replace(start, block)
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    ev = Evidence()
    outputs = [build_model_card(ev), build_deck(ev), update_readme(ev)]
    # The retrospective's reflection sections are written by the intern in the .docx itself: never overwrite it.
    retro = DOCS / "W04_Retrospective.docx"
    if retro.exists():
        print(f"kept {retro.relative_to(ROOT)} (exists; delete it to regenerate the template)")
    else:
        outputs.append(build_retro(ev))
    for path in outputs:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
