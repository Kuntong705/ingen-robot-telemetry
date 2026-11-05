"""Render the Week 2 demo MP4 from the log written by ``scripts/w02_demo.py``.

This is a *generated* video, not a screen recording. Every line of terminal
text and every plotted value comes from the saved log of one real run. The
renderer only lays them out over time and adds titles.

    python scripts/render_demo_video.py --log media/w02_demo_log.json --out media/W02_demo.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FFMpegWriter

FPS = 15
BG, PANEL, TEXT, MUTED = "#0f172a", "#111827", "#e5e7eb", "#94a3b8"
BLUE, GREEN, AMBER, RED, GREY = "#60a5fa", "#34d399", "#fbbf24", "#f87171", "#9ca3af"
# video timeline (seconds): which log sections are revealed by when, and which chart is shown
SCENES = [
    (0.0, 4.0, "title", []),
    (4.0, 10.0, "split", ["load", "split"]),
    (10.0, 17.0, "scores", ["train", "score"]),
    (17.0, 29.0, "risk", ["alert"]),
    (29.0, 36.0, "leak", ["leak"]),
    (36.0, 45.0, "soh", ["soh"]),
    (45.0, 50.0, "summary", ["done"]),
]
DURATION = SCENES[-1][1]


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for root in (Path.home() / "AppData/Local/Microsoft/WinGet/Packages").glob("*FFmpeg*"):
        for cand in root.rglob("ffmpeg.exe"):
            return str(cand)
    raise FileNotFoundError("ffmpeg not found")


def visible_lines(log: dict[str, Any], t: float) -> list[dict[str, Any]]:
    """Lines revealed by video time t: each scene types its sections' lines in order."""
    out = []
    for start, end, _, sections in SCENES:
        lines = [ln for ln in log["lines"] if ln["section"] in sections]
        if not lines:
            continue
        span = max(end - start - 1.0, 0.5)
        for i, ln in enumerate(lines):
            if t >= start + span * i / len(lines):
                out.append(ln)
    return out


def scene_at(t: float) -> tuple[str, float]:
    for start, end, name, _ in SCENES:
        if t < end:
            return name, (t - start) / (end - start)
    return SCENES[-1][2], 1.0


def style_axis(ax: Any) -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#334155")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def draw(fig: Any, log: dict[str, Any], t: float) -> None:
    fig.clf()
    fig.patch.set_facecolor(BG)
    name, frac = scene_at(t)
    footer = (f"Rendered from the live output of scripts/w02_demo.py (run {log['started_utc']}). "
              "Generated video, not a screen recording. Synthetic Aido Rover fleet (simulator v0.2).")
    fig.text(0.5, 0.018, footer, ha="center", color=MUTED, fontsize=8)

    if name == "title":
        fig.text(0.5, 0.62, "InGen Fleet Health: Week 2 demo", ha="center", color=TEXT, fontsize=26, weight="bold")
        fig.text(0.5, 0.50, "Leakage-safe evaluation · 72 h fault early warning · battery SoH forecast",
                 ha="center", color=BLUE, fontsize=14)
        fig.text(0.5, 0.40, "One walk-forward fold, run end to end on robots never seen in training",
                 ha="center", color=MUTED, fontsize=11)
        return

    term = fig.add_axes((0.02, 0.07, 0.50, 0.86))
    term.set_facecolor("#020617")
    term.set_xticks([])
    term.set_yticks([])
    for s in term.spines.values():
        s.set_color("#334155")
    term.text(0.02, 0.975, "terminal: live run output", color=MUTED, fontsize=8, va="top", transform=term.transAxes)
    shown = visible_lines(log, t)
    y = 0.93
    for ln in shown:
        color = GREEN if ln["section"] in ("split", "done") and ("PASS" in ln["text"] or "done" in ln["text"]) else TEXT
        if ln["text"].startswith("$"):
            color = AMBER
        text = f"[{ln['elapsed_s']:5.1f}s] {ln['text']}"
        wrapped = textwrap.wrap(text, width=78, subsequent_indent="         ")
        for piece in wrapped:
            term.text(0.02, y, piece, color=color, fontsize=7.4, family="monospace", va="top",
                      transform=term.transAxes)
            y -= 0.034
        y -= 0.008

    ax = fig.add_axes((0.58, 0.16, 0.39, 0.70))
    style_axis(ax)
    if name == "split":
        fold = log["fold_id"]
        ax.set_title(f"Fold {fold}: test robots never seen in training", fontsize=10)
        cutoff = pd.Timestamp(log["risk"]["t"][0]).normalize()
        start = cutoff - pd.Timedelta(days=40)
        ax.barh(1, (cutoff - pd.Timedelta(days=3) - start).days, left=0, color=BLUE, height=0.5,
                label="training robots: history")
        ax.barh(1, 3, left=(cutoff - pd.Timedelta(days=3) - start).days, color=GREY, height=0.5,
                label="72 h embargo (purged)")
        ax.barh(0, 10, left=40, color=GREEN, height=0.5, label="test robots: next 10 days")
        ax.set_yticks([0, 1], ["test robots", "training robots"])
        ax.set_xlabel(f"days since fleet start (cut-off day 40 = {cutoff:%Y-%m-%d})")
        ax.set_xlim(0, 52)
        ax.legend(fontsize=7, loc="lower left", facecolor=PANEL, labelcolor=TEXT)
    elif name == "scores":
        mm = log["model_metrics"]
        labels = {"physics_rule": "physics rule", "lgbm_physics": "LightGBM", "iforest": "Isolation Forest"}
        n = min(3, max(0, int(frac * 4)))
        keys = list(labels)[:n]
        ax.bar([labels[k] for k in keys], [mm[k]["pr_auc"] for k in keys], color=[GREY, BLUE, GREEN][:n])
        for i, k in enumerate(keys):
            ax.text(i, mm[k]["pr_auc"] + 0.02, f"{mm[k]['pr_auc']:.2f}", ha="center", color=TEXT, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel("PR-AUC on this fold's unseen robots")
        ax.set_title("72 h onset warning: models scored live", fontsize=10)
    elif name == "risk":
        r = log["risk"]
        # hours excluded from scoring (active fault, post-repair) stay blank instead of being interpolated
        series = pd.Series(r["score"], index=pd.to_datetime(r["t"]))
        series = series.reindex(pd.date_range(series.index[0], series.index[-1], freq="h"))
        times = series.index
        r = {**r, "score": series.tolist()}
        n = max(2, int(len(times) * min(1.0, frac * 1.15)))
        ax.plot(times[:n], r["score"][:n], color=GREEN, lw=1.2, label="Isolation Forest anomaly score")
        ax.axhline(r["thr"], color=GREY, ls="--", lw=1, label="alert threshold (set on validation)")
        onset = pd.Timestamp(log["onset_utc"])
        if times[n - 1] >= onset - pd.Timedelta(hours=72):
            ax.axvspan(onset - pd.Timedelta(hours=72), min(onset, times[n - 1]), color=RED, alpha=0.15)
        for a in pd.to_datetime(log["alerts"]):
            if a <= times[n - 1]:
                ax.axvline(a, color=AMBER, lw=1.5)
        if times[n - 1] >= onset:
            ax.axvline(onset, color=RED, lw=1.5)
            ax.text(onset, series.max() * 0.98, f" onset ({log['fault_type']})\n lead {log['lead_time_h']:.0f} h",
                    color=RED, fontsize=8, va="top")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(series.min() * 0.98, series.max() * 1.02)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.set_title(f"{log['demo_unit']} (never in training): score, alerts (amber), onset (red)", fontsize=10)
        ax.legend(fontsize=7, loc="lower left", facecolor=PANEL, labelcolor=TEXT)
    elif name == "leak":
        lk = log["leakage"]
        vals = [lk["random_split_pr_auc"], lk["fold_pr_auc"]]
        bars = ax.bar(["random row split\n(wrong)", "unit-grouped fold\n+ 72 h embargo"], vals, color=[RED, BLUE])
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", color=TEXT, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("PR-AUC, same LightGBM")
        ax.set_title("Leakage check: a random split overstates performance", fontsize=10)
    elif name == "soh":
        s = log["soh"]
        times = pd.to_datetime(s["t"]) + pd.Timedelta(days=7)
        ax.plot(times, s["truth"], color=TEXT, lw=1.8, label="true SoH (ground truth)")
        n = max(1, int(len(times) * min(1.0, frac * 1.3)))
        ax.plot(times[:n], s["persistence"][:n], color=GREY, ls=":", marker="o", ms=3, label="persistence")
        ax.plot(times[:n], s["lgbm"][:n], color=BLUE, ls="--", marker="o", ms=3, label="LightGBM, 7 days ahead")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.set_ylabel("SoH (%)")
        m = s["mae"]
        ax.set_title(f"SoH 7-day forecast, {log['demo_unit']}  |  fold MAE: persistence {m['persistence']:.2f}, "
                     f"linear {m['linear']:.2f}, LightGBM {m['lgbm']:.2f} pp", fontsize=8.5)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
    else:
        ax.axis("off")
        mm, lk, s = log["model_metrics"], log["leakage"], log["soh"]["mae"]
        lines = [
            "Results of this run (one fold; full results are in W02_Model_Results.docx)",
            "",
            f"Leakage: random split PR-AUC {lk['random_split_pr_auc']:.3f} vs proper fold {lk['fold_pr_auc']:.3f}",
            (f"Early warning PR-AUC: physics rule {mm['physics_rule']['pr_auc']:.2f}, LightGBM "
             f"{mm['lgbm_physics']['pr_auc']:.2f}, Isolation Forest {mm['iforest']['pr_auc']:.2f}"),
            f"{log['demo_unit']}: warned {log['lead_time_h']:.0f} h before a {log['fault_type'].replace('_', ' ')} onset",
            f"SoH 7 d MAE: persistence {s['persistence']:.2f}, linear {s['linear']:.2f}, LightGBM {s['lgbm']:.2f} pp",
        ]
        for i, line in enumerate(lines):
            ax.text(0.0, 0.95 - i * 0.12, line, color=TEXT if i else BLUE, fontsize=9 if i else 10,
                    transform=ax.transAxes, wrap=True, va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("media/w02_demo_log.json"))
    parser.add_argument("--out", type=Path, default=Path("media/W02_demo.mp4"))
    args = parser.parse_args()
    log = json.loads(args.log.read_text(encoding="utf-8"))
    plt.rcParams["animation.ffmpeg_path"] = find_ffmpeg()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=2500,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                          metadata={"title": "InGen Fleet Health Week 2 demo",
                                    "comment": "Generated from live program output; not a screen recording"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(args.out), dpi=100):
        for i in range(int(DURATION * FPS)):
            draw(fig, log, i / FPS)
            writer.grab_frame(facecolor=BG)
    print(args.out)


if __name__ == "__main__":
    main()
