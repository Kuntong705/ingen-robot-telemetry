# W03 Demo — 60-second recording script

> **Superseded for the plan's weekly recording.** The Week 3 demo video (`media/W03_demo.mp4`, 55 s, git-ignored) is a real screen recording of `scripts/w03_demo.py` (fault alert -> template and local-LLM explanation -> faithfulness check), made by `python scripts/record_w03_screen_demo.py`. The steps below are a figure walk-through for a live talk only.

All data is synthetic. Every number below comes from `reports/w03/*.json` (the same numbers as `docs/W03_Report.docx`). Nothing has to run live: the recording only shows figures and documents that already exist.

## Recording steps (about 5 minutes, Windows)

1. Open these four files in an image viewer or browser tab, in this order:
   1. `docs/figures/w03/fusion_ablation_pr_auc.png`
   2. `docs/figures/w03/fusion_sensitivity_notice_rate.png`
   3. `docs/W03_Alert_Explanations.md`, scrolled to section 6, alert A01 (template and LLM summary next to each other)
   4. `docs/figures/w03/fari_transfer_models.png`
2. Start a screen recording: Snipping Tool, Win + Shift + R, then select the window. Xbox Game Bar (Win + Alt + R) also works.
3. Read the script below while you switch between the four windows (about 15 s each).
4. Stop the recording and save it as `media/W03_demo.mp4`. The `media/` folder is git-ignored, so the video is not committed; share it directly.

## Script (about 60 s)

**0:00–0:15 — Fusion (figure 1).**
"Week 3 added maintenance text to the early-warning model. Telemetry alone reaches a PR-AUC of 0.21 with LightGBM and 0.33 with the Isolation Forest, against a base rate of 0.055. Adding text in early fusion gives 0.226. The paired difference of +0.016 has a confidence interval from −0.04 to +0.08, so there is no detectable gain."

**0:15–0:30 — Why (figure 2).**
"Only 9 of the 31 test-period faults had an operator note in the 72 hours before onset. If operators wrote two to four times as often, the gain becomes significant: +0.13 at four times. InGen's real ticket history decides this."

**0:30–0:45 — Explanations (A01).**
"Every alert gets a summary written only from its SHAP evidence. The template summaries were faithful in 20 of 20 alerts. The local 1.5-billion-parameter LLM was faithful in none of 20 in the evidence review. Here it lists a risk-lowering note as a main cause. A lexical checker passed 8 of them, so we ship the template."

**0:45–1:00 — Fari (figure 4).**
"On a synthetic Fari fleet, the Week 1 code rejected every record. With a platform profile and config-driven feature guards, acceptance is 100 % and the Rover model's PR-AUC on Fari rises from 0.046 to 0.157. Next week: telemetry-first scoring, recalibration and the dashboard."
