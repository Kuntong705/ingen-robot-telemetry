"""Live Week 2 demo dashboard: a real run shown in a dedicated window for screen recording.

A separate worker process runs the project code on one walk-forward fold,
robots never seen in training. It builds and audits the fold, trains and
calibrates the early-warning models, scores the held-out robots, trains the
7-day SoH forecaster, and trains the same LightGBM under the proper and the
random-split protocol for the leakage contrast. The window, in this process,
draws each result as soon as the worker reports it. The worker uses its own
process so that model training never freezes the window.

Panels also show the saved 20-fold results from ``reports/w02/*.json``, marked
as such, so the one-fold run and the full evaluation are never confused.

    python scripts/w02_live_dashboard.py            # opens the window
    python scripts/record_live_dashboard.py         # opens it and screen-records it

Two presentation profiles change only pacing and window placement, never the computation:
``window`` (default; recorded by record_live_dashboard.py) and ``desktop`` (centred, DPI-aware, faster
replay; used automatically when started from PyCharm, which sets PYCHARM_HOSTED, for the desktop demo).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TITLE = "W02 live dashboard"
DEMO_UNIT = "Rover_023"
CUTOFF_DAY = 40
REFRACTORY_H = 72
PROFILES = {
    # replay: scored hours per second (excluded active-fault hours are skipped)
    "window": {"replay": 8, "takeaway": 4.5, "hold": 6.0, "dpi_aware": False, "position": "+8+8"},
    "desktop": {"replay": 20, "takeaway": 2.5, "hold": 3.0, "dpi_aware": True, "position": "+160+70"},
}
# "lgbm_selected" is the LightGBM of the formal report: per fold, the variant (all features vs physics
# features) with the higher PR-AUC on the inner validation slice.
MODELS = ("physics_rule", "lgbm_selected", "iforest")


# ----------------------------------------------------------------------------- worker process
def worker_main(q: Any, go: Any, data_dir: str) -> None:
    """Imports and computes everything; reports results to the window through ``q``."""
    import pandas as pd

    from ingen_fleet_health.evaluation.metrics import (
        evaluate_alerts,
        events_in_scored_period,
        mae,
        pr_auc,
        raise_alerts,
        threshold_for_budget,
    )
    from ingen_fleet_health.evaluation.splits import (
        check_fold,
        random_row_split,
        walk_forward_folds,
    )
    from ingen_fleet_health.experiments.common import load_w02
    from ingen_fleet_health.experiments.early_warning_run import TEST_SPAN, VAL_SPAN
    from ingen_fleet_health.models.early_warning import (
        fit_lgbm,
        model_feature_columns,
        positive_proba,
        score_fold,
    )
    from ingen_fleet_health.models.soh_forecast import (
        daily_table,
        fit_predict_lgbm,
        linear_extrapolation,
        persistence,
    )

    q.put(("ready", None))
    go.wait()  # the window is open: the computation (and the clock) starts now
    t0 = time.perf_counter()

    def log(text: str) -> None:
        q.put(("log", (time.perf_counter() - t0, text)))

    log("$ python scripts/w02_live_dashboard.py")
    data = load_w02(Path(data_dir))
    eligible = data.eligible()
    cols = model_feature_columns()
    log(f"loaded {len(data.features):,} rows · {data.features['unit_id'].nunique()} robots · {len(cols)} features")

    horizon = pd.Timedelta(hours=72)
    cutoff = data.cutoffs([CUTOFF_DAY])[0]
    group = data.groups[DEMO_UNIT]
    fold = next(f for f in walk_forward_folds(eligible, data.groups, [cutoff], TEST_SPAN, horizon, VAL_SPAN)
                if f.group == group)
    audit = check_fold(eligible, fold, horizon)
    log(f"fold {fold.fold_id}: {len(fold.test_units)} held-out robots")
    log(f"audit: shared robots {audit['shared_units']} · dup rows {audit['duplicate_keys']} · "
        f"embargo {audit['embargo_hours']:.0f} h  PASS")
    q.put(("audit", {"fold": fold.fold_id, **audit}))

    scores = score_fold(eligible, fold, cols, seed=0, models=MODELS)
    chosen = scores.info["calibration"]["selected"]
    log(f"LightGBM fit on {scores.info['fit_rows']:,} rows; validation picks "
        f"{chosen.removeprefix('lgbm_')} features, {scores.info['calibration'][chosen]}")
    log("Isolation Forest fit (no labels)")
    test = eligible.iloc[fold.test_idx][["unit_id", "t", "y"]].reset_index(drop=True)
    metrics: dict[str, Any] = {}
    short = {"physics_rule": "rule", "lgbm_selected": "LightGBM", "iforest": "IsoForest"}
    iforest_scored = None
    for model in MODELS:
        val = eligible.iloc[fold.val_idx][["unit_id", "t"]].assign(score=scores.val_scores[model])
        thr = threshold_for_budget(val, 1.0, REFRACTORY_H)
        scored = test.assign(score=scores.scores[model], thr=thr)
        m = evaluate_alerts(scored, events_in_scored_period(scored, data.events), None, REFRACTORY_H, 72)
        m["pr_auc"] = pr_auc(scored["y"].to_numpy(), scored["score"].to_numpy())
        metrics[model] = {k: m[k] for k in ("pr_auc", "events", "events_detected", "false_alarms_per_unit_week")}
        if model == "iforest":
            iforest_scored = scored
        log(f"{short[model]:9s} PR-AUC {m['pr_auc']:.3f} · warned {int(m['events_detected'])}/{int(m['events'])}")
    q.put(("metrics", metrics))

    assert iforest_scored is not None
    unit = iforest_scored.query("unit_id == @DEMO_UNIT").sort_values("t")
    onset = data.events.query("unit_id == @DEMO_UNIT and onset_utc >= @cutoff").iloc[0]
    q.put(("risk", {"t": list(unit["t"]), "score": unit["score"].to_numpy(), "thr": float(unit["thr"].iloc[0]),
                    "onset": onset.onset_utc, "fault": onset.fault_type,
                    "reference_alerts": list(raise_alerts(unit, None, REFRACTORY_H)["t"])}))
    log(f"{DEMO_UNIT}: online replay started")

    h = 7
    table = daily_table(data.features, data.battery_truth, [h]).drop(columns="t").rename(columns={"day": "t"})
    table = table[table[f"target_h{h}"].notna()].reset_index(drop=True)
    sfold = next(f for f in walk_forward_folds(table, data.groups, [cutoff], TEST_SPAN, pd.Timedelta(days=h),
                                               pd.Timedelta(0)) if f.group == group)
    check_fold(table, sfold, pd.Timedelta(days=h))
    s_train, s_test = table.iloc[sfold.fit_idx], table.iloc[sfold.test_idx]
    preds = {"persistence": persistence(s_test, h), "linear": linear_extrapolation(s_test, h),
             "lgbm": fit_predict_lgbm(s_train, s_test, h)}
    soh_mae = {k: mae(s_test[f"target_h{h}"], v) for k, v in preds.items()}
    u = (s_test["unit_id"] == DEMO_UNIT).to_numpy()
    log(f"SoH 7 d MAE: LightGBM {soh_mae['lgbm']:.3f} vs persistence {soh_mae['persistence']:.3f} pp")
    q.put(("soh", {"t": list(s_test.loc[u, "t"] + pd.Timedelta(days=h)),
                   "truth": s_test.loc[u, f"target_h{h}"].to_numpy(), "lgbm": preds["lgbm"][u],
                   "persistence": preds["persistence"][u], "mae": soh_mae}))

    # leakage contrast: the same LightGBM under the proper protocol, then under a random row split
    q.put(("leak_started", "leak_proper"))
    proper = fit_lgbm(eligible.iloc[fold.train_idx][cols], eligible.iloc[fold.train_idx]["y"].to_numpy())
    value = pr_auc(test["y"].to_numpy(), positive_proba(proper, eligible.iloc[fold.test_idx][cols]))
    log(f"held-out robots (proper): PR-AUC {value:.3f}")
    q.put(("leak_proper", value))
    q.put(("leak_started", "leak_random"))
    tr, te = random_row_split(len(eligible), 0.2, seed=0)
    rand = fit_lgbm(eligible.iloc[tr][cols], eligible.iloc[tr]["y"].to_numpy())
    value = pr_auc(eligible.iloc[te]["y"].to_numpy(), positive_proba(rand, eligible.iloc[te][cols]))
    log(f"random split (wrong): PR-AUC {value:.3f}")
    q.put(("leak_random", value))

    log("done: every live number computed now")
    q.put(("done", time.perf_counter() - t0))


# ----------------------------------------------------------------------------- window
def run_window(args: argparse.Namespace) -> None:
    prof = PROFILES[args.profile]
    replay_rate, takeaway_s, hold_s = prof["replay"], prof["takeaway"], prof["hold"]
    if sys.platform == "win32" and not prof["dpi_aware"]:
        # Render at 96 DPI (1280x720) and let Windows scale on screen, so FFmpeg's gdigrab -- also
        # DPI-unaware -- captures the whole window instead of a cropped corner on a 125 % display.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-1))
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    bg, card_bg, edge = "#0b1220", "#111a2e", "#26354f"
    text_c, muted = "#e8eef9", "#a3b3cf"
    blue, green, amber, red, grey, violet = "#5aa9ff", "#3ddc97", "#ffc145", "#ff6b6b", "#8a97ad", "#b48cff"
    label = {"physics_rule": "Physics rule", "lgbm_selected": "LightGBM", "iforest": "Isolation Forest"}
    color = {"physics_rule": grey, "lgbm_selected": blue, "iforest": green}
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "toolbar": "None"})
    saved = load_saved()

    q: Any = mp.Queue()
    go = mp.Event()
    proc = mp.Process(target=worker_main, args=(q, go, str(args.data_dir)), daemon=True)
    proc.start()
    if q.get(timeout=300)[0] != "ready":
        raise RuntimeError("worker did not start")

    state: dict[str, Any] = {"lines": []}
    marks: dict[str, float] = {}
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=bg)
    manager = fig.canvas.manager
    manager.set_window_title(TITLE)
    manager.window.geometry(prof["position"])
    manager.window.resizable(False, False)
    t_start = [time.perf_counter()]

    def elapsed() -> float:
        return time.perf_counter() - t_start[0]

    def card(rect: tuple[float, float, float, float], title: str, tag: str) -> Any:
        ax = fig.add_axes(rect)
        ax.set_facecolor(card_bg)
        for sp in ax.spines.values():
            sp.set_color(edge)
        ax.tick_params(colors=muted, labelsize=10)
        fig.text(rect[0], rect[1] + rect[3] + 0.013, title, color=text_c, fontsize=13.5, weight="bold")
        fig.text(rect[0] + rect[2], rect[1] + rect[3] + 0.015, tag, color=muted, fontsize=10, ha="right")
        return ax

    def waiting(ax: Any, text: str, c: str = muted) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, text, color=c, fontsize=13, ha="center", va="center", transform=ax.transAxes)

    def text_box(ax: Any, x: float, y: float, s: str, c: str, size: float = 12.0) -> None:
        ax.text(x, y, s, color=c, fontsize=size, weight="bold", transform=ax.transAxes, va="top", zorder=5,
                bbox={"facecolor": card_bg, "edgecolor": "none", "alpha": 0.92, "pad": 2})

    def draw_console() -> None:
        ax = fig.add_axes((0.015, 0.155, 0.33, 0.69))
        ax.set_facecolor("#060b16")
        for sp in ax.spines.values():
            sp.set_color(edge)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.text(0.015, 0.858, "Program output", color=text_c, fontsize=13.5, weight="bold")
        y = 0.965
        for t, text in state["lines"]:
            c = amber if text.startswith("$") else green if ("PASS" in text or text.startswith("done")) else text_c
            for k, chunk in enumerate(textwrap.wrap(text, 42) or [""]):
                ax.text(0.03, y, (f"{t:4.1f}s " if k == 0 else "      ") + chunk, color=c, fontsize=10.2,
                        family="Consolas", va="top", transform=ax.transAxes)
                y -= 0.045
            y -= 0.007

    def draw_warning(ax: Any) -> None:
        m = state.get("metrics")
        if not m:
            waiting(ax, f"training & calibrating…  {elapsed():.0f} s")
            return
        x = np.arange(len(MODELS))
        vals = [m[k]["pr_auc"] for k in MODELS]
        ax.bar(x, vals, 0.56, color=[color[k] for k in MODELS])
        ax.scatter(x, [saved["pr_auc"][k] for k in MODELS], marker="D", s=60, color=text_c, edgecolor=bg, zorder=3,
                   label="20-fold (saved)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.035, f"{v:.2f}", ha="center", color=text_c, fontsize=14, weight="bold")
        ax.set_xticks(x, [f"{label[k]}\nwarned {int(m[k]['events_detected'])}/{int(m[k]['events'])}" for k in MODELS],
                      color=text_c, fontsize=10.5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        ax.axhline(0.055, color=muted, ls=":", lw=1.4, label="no-skill 0.055")
        ax.legend(loc="upper left", fontsize=10, frameon=False, labelcolor=text_c)

    def draw_risk(ax: Any) -> None:
        r = state.get("risk")
        if not r:
            waiting(ax, "waiting for model scores…")
            return
        s = pd.Series(r["score"], index=pd.to_datetime(r["t"]))
        full = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="h"))  # excluded hours stay blank
        valid_times = full.dropna().index
        k = min(len(valid_times), int((elapsed() - marks["risk"]) * replay_rate) + 1)
        n = len(full) if k == len(valid_times) else int(full.index.get_loc(valid_times[k - 1])) + 1
        shown = full.iloc[:n]
        alerts, muted_until = [], None
        for t, v in shown.dropna().items():  # online alert logic, identical to raise_alerts
            if v >= r["thr"] and (muted_until is None or t >= muted_until):
                alerts.append(t)
                muted_until = t + pd.Timedelta(hours=REFRACTORY_H)
        if n == len(full):
            if alerts != list(pd.to_datetime(r["reference_alerts"])):
                raise RuntimeError("online alerts differ from raise_alerts")
            marks.setdefault("replay_done", elapsed())
        onset = pd.Timestamp(r["onset"])
        now = shown.index[-1]
        if now > onset - pd.Timedelta(hours=72):
            ax.axvspan(onset - pd.Timedelta(hours=72), min(onset, now), color=red, alpha=0.16)
        ax.plot(shown.index, shown.values, color=green, lw=2)
        ax.axhline(r["thr"], color=grey, ls="--", lw=1.2)
        for a in alerts:
            ax.axvline(a, color=amber, lw=2.4)
        if now >= onset:
            ax.axvline(onset, color=red, lw=2.4)
            warned = [a for a in alerts if onset - pd.Timedelta(hours=72) <= a < onset]
            state["lead_h"] = (onset - warned[0]) / pd.Timedelta(hours=1) if warned else float("nan")
            text_box(ax, 0.03, 0.93, f"{r['fault'].replace('_', ' ')} onset: warned {state['lead_h']:.0f} h ahead",
                     red, 12.5)
        else:
            text_box(ax, 0.03, 0.93, f"{now:%b %d %H:00} · alerts so far {len(alerts)}", text_c)
        handles = [Line2D([], [], color=green, lw=2, label="anomaly score"),
                   Line2D([], [], color=grey, ls="--", lw=1.2, label="threshold"),
                   Line2D([], [], color=amber, lw=2.4, label="alert"), Line2D([], [], color=red, lw=2.4, label="onset")]
        leg = ax.legend(handles=handles, loc="lower left", ncol=4, fontsize=9, frameon=True, labelcolor=text_c,
                        handlelength=1.4, columnspacing=1.0, facecolor=card_bg, edgecolor="none", framealpha=0.92)
        leg.set_zorder(5)
        ax.set_xlim(full.index[0], full.index[-1])
        lo, hi = np.nanmin(full.values), np.nanmax(full.values)
        ax.set_ylim(lo - (hi - lo) * 0.45, hi + (hi - lo) * 0.45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))

    def draw_leak(ax: Any) -> None:
        x = np.array([0.0, 1.0])
        ax.bar(x - 0.2, [saved["leak_random"], saved["leak_proper"]], 0.36, color="none", edgecolor=muted, hatch="///")
        for xi, v in zip(x, (saved["leak_random"], saved["leak_proper"]), strict=True):
            ax.text(xi - 0.2, v + 0.035, f"{v:.3f}", ha="center", color=muted, fontsize=10.5)
        for xi, key, c in ((0.0, "leak_random", red), (1.0, "leak_proper", blue)):
            v = state.get(key)
            if v is None:
                status, sc = ((f"training…\n{elapsed() - marks[key]:.0f} s", amber) if key in marks
                              else ("queued", muted))
                ax.text(xi + 0.2, 0.1, status, ha="center", color=sc, fontsize=11)
                continue
            ax.bar(xi + 0.2, v, 0.36, color=c)
            ax.text(xi + 0.2, v + 0.035, f"{v:.3f}", ha="center", color=text_c, fontsize=14, weight="bold")
        ax.set_xticks(x, ["random row split\n(wrong)", "held-out robots\n+ 72 h embargo"], color=text_c,
                      fontsize=10.5)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0, 0.5, 1.0])
        handles = [Patch(facecolor="none", edgecolor=muted, hatch="///", label="20-fold (saved)"),
                   Patch(facecolor=muted, label="this fold (live)")]
        ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=False, labelcolor=text_c)

    def draw_soh(ax: Any) -> None:
        s = state.get("soh")
        if not s:
            waiting(ax, "training SoH model…" if "metrics" in state else "queued")
            return
        t = pd.to_datetime(s["t"])
        ax.plot(t, s["truth"], color=text_c, lw=2.6, label="true SoH")
        ax.plot(t, s["persistence"], color=grey, ls=":", marker="o", ms=4, lw=1.6, label="persistence")
        ax.plot(t, s["lgbm"], color=violet, ls="--", marker="o", ms=4, lw=1.8, label="LightGBM")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        lo = min(np.min(s["truth"]), np.min(s["lgbm"]), np.min(s["persistence"]))
        hi = max(np.max(s["truth"]), np.max(s["lgbm"]), np.max(s["persistence"]))
        ax.set_ylim(lo - (hi - lo) * 0.75, hi + (hi - lo) * 0.2)
        ax.legend(loc="upper right", fontsize=9.5, frameon=False, labelcolor=text_c, ncol=3)
        m, sv = s["mae"], saved["soh_mae"]
        ax.text(0.03, 0.2, f"MAE this fold: LightGBM {m['lgbm']:.2f} vs persistence {m['persistence']:.2f} pp",
                color=text_c, fontsize=10, transform=ax.transAxes, weight="bold")
        ax.text(0.03, 0.07, f"MAE 20-fold:   LightGBM {sv['lgbm']:.2f} vs persistence {sv['persistence']:.2f} pp",
                color=muted, fontsize=10, transform=ax.transAxes)

    def draw_footer() -> None:
        fig.patches.append(plt.Rectangle((0.0, 0.0), 1.0, 0.118, transform=fig.transFigure, color="#0f2238",
                                         zorder=0))
        if "done" not in state or "replay_done" not in marks:
            fig.text(0.015, 0.072, "How to read: solid bars and lines are computed live in this run "
                     "(one fold, cut-off day 40).", color=text_c, fontsize=11.5)
            fig.text(0.015, 0.035, "Diamonds and hatched bars are the saved 20-fold results (reports/w02), "
                     "for comparison.", color=muted, fontsize=11.5)
            return
        m = state["metrics"]
        takeaways = [
            (f"Leakage: a random split scores {state['leak_random']:.3f}; held-out robots score "
             f"{state['leak_proper']:.3f} here and {saved['leak_proper']:.3f} across 20 folds."),
            (f"Early warning: Isolation Forest {m['iforest']['pr_auc']:.2f} on this fold "
             f"({saved['pr_auc']['iforest']:.2f} over 20 folds); {DEMO_UNIT} was warned {state['lead_h']:.0f} h "
             "before onset."),
            (f"SoH: LightGBM {state['soh']['mae']['lgbm']:.2f} pp vs persistence "
             f"{state['soh']['mae']['persistence']:.2f} pp (20 folds: {saved['soh_mae']['lgbm']:.2f} vs "
             f"{saved['soh_mae']['persistence']:.2f})."),
        ]
        shown = min(len(takeaways), int((elapsed() - marks["replay_done"]) / takeaway_s) + 1)
        for i, text in enumerate(takeaways[:shown]):
            fig.text(0.015, 0.086 - i * 0.033, f"{i + 1}. {text}", color=text_c if i == shown - 1 else muted,
                     fontsize=11.5, weight="bold" if i == shown - 1 else "normal")

    def draw() -> None:
        fig.clf()
        fig.set_facecolor(bg)
        fig.text(0.015, 0.948, "InGen Fleet Health · Week 2 live demo", color=text_c, fontsize=20, weight="bold")
        fig.text(0.015, 0.912, "Leakage-safe evaluation · 72 h fault early warning · battery SoH forecast",
                 color=muted, fontsize=12)
        done = "done" in state
        clock = f"RUN COMPLETE · computed in {state['done']:.1f} s" if done else f"LIVE RUN  {elapsed():4.1f} s"
        fig.text(0.985, 0.950, clock, color=green if done else amber, fontsize=15, weight="bold", ha="right",
                 family="Consolas")
        fig.text(0.985, 0.913, "synthetic fleet · 1 fold computed now", color=muted, fontsize=11, ha="right")
        draw_console()
        draw_warning(card((0.375, 0.575, 0.29, 0.27), "Early warning on held-out robots", "PR-AUC"))
        draw_risk(card((0.695, 0.575, 0.29, 0.27), f"{DEMO_UNIT}: risk score & alerts", "online replay"))
        draw_leak(card((0.375, 0.205, 0.29, 0.27), "Leakage check: same LightGBM", "PR-AUC"))
        draw_soh(card((0.695, 0.205, 0.29, 0.27), f"SoH 7 days ahead: {DEMO_UNIT}", "% capacity"))
        draw_footer()

    draw()
    plt.show(block=False)
    plt.pause(0.5)
    t_start[0] = time.perf_counter()
    go.set()
    while plt.fignum_exists(fig.number):
        while not q.empty():
            kind, payload = q.get()
            if kind == "log":
                state["lines"].append(payload)
            elif kind == "leak_started":
                marks[payload] = elapsed()
            else:
                state[kind] = payload
                marks[kind] = elapsed()
        draw()
        fig.canvas.draw_idle()
        plt.pause(0.1)
        if "done" in marks and "replay_done" in marks and elapsed() >= (
                marks["replay_done"] + 2 * takeaway_s + hold_s):
            break
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps({"finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                                    "compute_seconds": state.get("done"), "lead_h": state.get("lead_h"),
                                    "lines": state["lines"]}, indent=1, default=str), encoding="utf-8")
    plt.close("all")
    proc.join(timeout=10)


def load_saved() -> dict[str, Any]:
    """Saved 20-fold results (read, not recomputed)."""
    rep = ROOT / "reports" / "w02"
    ew = json.loads((rep / "early_warning_results.json").read_text(encoding="utf-8"))["models"]
    leak = json.loads((rep / "leakage_results.json").read_text(encoding="utf-8"))["early_warning"]
    soh = json.loads((rep / "soh_results.json").read_text(encoding="utf-8"))["by_horizon"]["7"]
    return {
        "pr_auc": {m: ew[m]["pr_auc"] for m in MODELS},
        "leak_random": leak["random_shuffle"]["pr_auc"],
        "leak_proper": leak["proper_unit_grouped_walk_forward"]["pr_auc"],
        "soh_mae": {k: soh[k]["mae"] for k in ("persistence", "linear", "lgbm")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "w02")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--profile", choices=sorted(PROFILES),
                        default="desktop" if os.environ.get("PYCHARM_HOSTED") else "window")
    args = parser.parse_args()
    if args.log is None:
        suffix = "" if args.profile == "window" else f"_{args.profile}"
        args.log = ROOT / "media" / f"w02_live_dashboard{suffix}_log.json"
    run_window(args)


if __name__ == "__main__":
    main()
