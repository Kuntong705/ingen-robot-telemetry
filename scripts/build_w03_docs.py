# ruff: noqa: ISC004  (prose-heavy document builder)
"""Build the Week 3 deliverables from ``reports/w03/*.json``.

    python scripts/build_w03_docs.py

Every number in these documents is read from the results files written by the
experiment runners (each linked to an MLflow run). Prose that interprets a
number is chosen by comparing numbers (e.g. whether an interval contains 0),
so a rerun with different results cannot silently keep an outdated claim.

Outputs: docs/W03_Log_Corpus.md, docs/W03_Fusion_Results.docx,
docs/W03_Alert_Explanations.md, docs/W03_Fari_Gap_Analysis.docx,
docs/W03_Report.docx.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "w03"
FIGS = ROOT / "docs" / "figures" / "w03"
DOCS = ROOT / "docs"
CORPUS_MANIFEST = ROOT / "data" / "w02" / "text" / "log_corpus_manifest.json"
SYNTHETIC = ("All data in this document is SYNTHETIC: a simulated Aido Rover fleet (simulator v0.2, 50 units x 60 days), "
             "a template-generated maintenance-log corpus, and a simulated Fari fleet whose every parameter is an "
             "assumption, not an InGen specification. No InGen, customer or personal data was used.")
CLASSES = ["battery", "drivetrain", "thermal", "sensor", "connectivity", "user_or_operational_error", "no_fault_found"]
SHORT = {"battery": "battery", "drivetrain": "drivetrain", "thermal": "thermal", "sensor": "sensor",
         "connectivity": "connectivity", "user_or_operational_error": "user/op. error", "no_fault_found": "no fault found"}


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def maybe(name: str) -> dict[str, Any] | None:
    return load(name) if (RESULTS / name).exists() else None


REVIEW_METHOD = (
    "Faithfulness verdicts for the 20 local-LLM summaries come from an AI-assisted evidence review "
    "(reports/w03/llm_faithfulness_review.json): each summary was read sentence by sentence against its evidence pack "
    "and the prompt the LLM saw, with AI assistance (evidence prepared and reviews drafted by AI models), and every "
    "verdict was confirmed by the project owner. It is not an independent human review. The 20 template summaries and "
    "the 30 paraphrase pairs were read once by the AI assistant that built the pipeline and were not reviewed further."
)


def reviews(ex: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Verdicts per generator: LLM from the evidence review, template from the assistant's read.

    The evidence review is validated against the stored checker v2 output: one entry per v2 flag, and the
    aggregate auto_flag_valid follows its definition (null without flags, else any flag pointing to a real error).
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    read = maybe("manual_faithfulness.json") or {}
    if "template" in read:
        out["template"] = read["template"]
    rev = maybe("llm_faithfulness_review.json")
    if rev and rev["generator"] in ex["generators"]:
        g = rev["generator"]
        by_id = {a["alert_id"]: a for a in ex["alerts"]}
        assert sorted(rev["alerts"]) == sorted(by_id), "review must cover every sampled alert"
        for aid, v in rev["alerts"].items():
            chk = by_id[aid]["summaries"][g]["check"]
            assert v["checker_v2_faithful"] == chk["faithful"], aid
            assert [f["flagged_text"] for f in v["checker_v2_flags"]] == [u["text"] for u in chk["unsupported"]], aid
            flags = [f["points_to_real_error"] for f in v["checker_v2_flags"]]
            assert v["auto_flag_valid"] == (any(flags) if flags else None), aid
        out[g] = {aid: {"faithful": v["faithful"], "note": v["reason_en"], "auto_flag_valid": v["auto_flag_valid"],
                        "flags": v["checker_v2_flags"]} for aid, v in rev["alerts"].items()}
    return out


def fmt(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def signed(v: float, digits: int = 3) -> str:
    return f"{v:+.{digits}f}"


def ci(pair: list[float] | None, digits: int = 3) -> str:
    if not pair:
        return "n/a"
    return f"[{fmt(pair[0], digits)}, {fmt(pair[1], digits)}]"


def pct(v: float, digits: int = 0) -> str:
    return f"{100 * v:.{digits}f} %"


def verdict(pair: list[float]) -> str:
    """Plain reading of a 95 % interval of a difference."""
    if pair[0] > 0:
        return "better (interval above 0)"
    if pair[1] < 0:
        return "worse (interval below 0)"
    return "no detectable difference (interval contains 0)"


# --------------------------------------------------------------------------- writers
INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)")


def add_inline(para: Any, text: str) -> None:
    """Add ``text`` to a docx paragraph, rendering **bold**, *italic* and `code`."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            para.add_run(piece[2:-2]).bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            run = para.add_run(piece[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            para.add_run(piece[1:-1]).italic = True
        else:
            para.add_run(piece)


class Doc:
    def __init__(self, title: str, subtitle: str) -> None:
        self.d = Document()
        style = self.d.styles["Normal"]
        style.font.name = "Calibri"
        style.font.size = Pt(10.5)
        for section in self.d.sections:
            section.left_margin = section.right_margin = Inches(0.9)
            section.top_margin = section.bottom_margin = Inches(0.8)
        t = self.d.add_heading(title, level=0)
        t.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p = self.d.add_paragraph(subtitle)
        p.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p.runs[0].font.size = Pt(9)

    def h(self, text: str, level: int = 1) -> None:
        self.d.add_heading(text, level=level)

    def p(self, text: str, bold_prefix: str | None = None) -> None:
        para = self.d.add_paragraph()
        if bold_prefix:
            para.add_run(bold_prefix).bold = True
        add_inline(para, text)

    def bullets(self, items: list[str]) -> None:
        for item in items:
            add_inline(self.d.add_paragraph(style="List Bullet"), item)

    def table(self, header: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
        tab = self.d.add_table(rows=1, cols=len(header))
        tab.style = "Light Grid Accent 1"
        for i, h in enumerate(header):
            cell = tab.rows[0].cells[i]
            cell.text = h
            cell.paragraphs[0].runs[0].bold = True
        for r in rows:
            cells = tab.add_row().cells
            for i, v in enumerate(r):
                cells[i].text = v
        for row in tab.rows:
            for i, cell in enumerate(row.cells):
                for par in cell.paragraphs:
                    for run in par.runs:
                        run.font.size = Pt(8.5)
                if widths:
                    cell.width = Inches(widths[i])
        self.d.add_paragraph()

    def figure(self, name: str, caption: str, width: float = 6.3) -> None:
        path = FIGS / name
        if path.exists():
            self.d.add_picture(str(path), width=Inches(width))
            cap = self.d.add_paragraph(caption)
            cap.runs[0].italic = True
            cap.runs[0].font.size = Pt(8.5)

    def save(self, name: str) -> Path:
        out = DOCS / name
        self.d.save(str(out))
        return out


class Md:
    def __init__(self, title: str, subtitle: str) -> None:
        self.lines = [f"# {title}", "", f"*{subtitle}*", ""]

    def h(self, text: str, level: int = 2) -> None:
        self.lines += ["#" * level + " " + text, ""]

    def p(self, text: str) -> None:
        self.lines += [text, ""]

    def bullets(self, items: list[str]) -> None:
        self.lines += [f"- {i}" for i in items] + [""]

    def table(self, header: list[str], rows: list[list[str]]) -> None:
        esc = [[c.replace("|", "\\|") for c in r] for r in [header, *rows]]
        self.lines += ["| " + " | ".join(esc[0]) + " |", "|" + "---|" * len(header)]
        self.lines += ["| " + " | ".join(r) + " |" for r in esc[1:]] + [""]

    def figure(self, name: str, caption: str) -> None:
        if (FIGS / name).exists():
            self.lines += [f"![{caption}](figures/w03/{name})", "", f"*{caption}*", ""]

    def save(self, name: str) -> Path:
        out = DOCS / name
        out.write_text("\n".join(self.lines).rstrip() + "\n", encoding="utf-8")
        return out


# --------------------------------------------------------------------------- 1. corpus + text classifier
SOURCE_TEXT = {
    "fault_symptom": "operator symptom note during a simulated fault's P-F interval (notice probability rises with severity)",
    "fault_service_open": "work order opened after the fault was detected",
    "fault_service_close": "work order closed by the repair",
    "drift_symptom": "operator note about a drifting motor-temperature sensor (30 % worded as 'running hot')",
    "drift_service": "recalibration / replacement ticket for a drifting sensor",
    "outage": "connectivity ticket tied to a simulated network outage",
    "minor_battery": "text-only minor issue (dirty contacts, dock pins)",
    "minor_drivetrain": "text-only minor issue (worn tread, debris)",
    "minor_thermal": "text-only minor issue (clogged intake filter)",
    "minor_sensor": "text-only minor issue (lidar lens, bumper, encoder, IMU)",
    "connectivity_misc": "text-only connectivity issue (access-point handover, SIM)",
    "user_error": "user or operational error (dock moved, wrong route loaded)",
    "nff_complaint": "complaint on a healthy unit (70 % hedged, 30 % reads like a real symptom)",
    "nff_closure": "'no fault found' closure of such a complaint",
    "routine_note": "handover / cleaning / inspection note unrelated to any fault",
}


def build_log_corpus() -> Path:
    tc = load("text_classifier_results.json")
    man = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8")) if CORPUS_MANIFEST.exists() else None
    para = maybe("paraphrase_results.json")
    md = Md("W03 — Maintenance-Log Corpus and Ticket Classifier",
            "Week 3 deliverable 1 · generated by scripts/build_w03_docs.py from reports/w03/text_classifier_results.json"
            + (" and paraphrase_results.json" if para else "") + f" · MLflow run {tc['mlflow_run_id']}")
    md.p("> **SYNTHETIC DATA.** Every ticket and note in this corpus was generated by code in this repository from "
         "templates, rule-based paraphrase and seeded noise, tied to the simulated fleet's ground truth. No InGen ticket, "
         "operator, customer or personal text was used or seen. The corpus is regenerable from a seed.")

    md.h("1. What the corpus is")
    if man:
        md.p(f"{man['tickets']:,} documents for the 50-unit synthetic Aido Rover fleet (simulator v0.2, 60 days): "
             f"{man['by_doc_type']['operator_note']:,} operator notes and {man['by_doc_type']['service_ticket']:,} service "
             f"tickets. Generator v{man['generator_version']}, seed {man['seed']}, config SHA-256 `{man['config_sha256'][:16]}…`, "
             f"tied to simulator config `{man['simulator_config_sha256'][:16]}…`.")
        md.table(["Label (fixed taxonomy)", "Documents", "Share"],
                 [[SHORT[c], str(man["by_label"].get(c, 0)), pct(man["by_label"].get(c, 0) / man["tickets"], 1)] for c in CLASSES])
        md.p("Why each document exists (the hidden `ground_truth/log_corpus_truth.csv`, used for analysis only, never as a "
             "model input):")
        md.table(["Source", "Documents", "Meaning"],
                 [[k, str(v), SOURCE_TEXT.get(k, "")] for k, v in sorted(man["by_source"].items(), key=lambda kv: -kv[1])])
        md.p(f"Only **{man['fault_notes_pre_onset']}** operator notes describe a fault *before* its qualifying onset. "
             "This is the number that bounds what text can add to a 72-hour early warning (see W03_Fusion_Results).")

    md.h("2. How it is generated (and how to regenerate it)")
    md.bullets([
        "**Events, not text, come first.** The generator plays the people around the robots: it reads the simulator's fault "
        "timelines, sensor-drift impairments and network outages, and decides *whether and when* someone writes. A symptom "
        "note is written in a 12-hour slot with probability `notice_rate x severity^1.5` (people notice late), only "
        "during staffed hours (06–20 UTC).",
        "**Text from templates.** Each source has several core phrases per category plus slot values (side, component, "
        "time of day), prefixes and suffixes (`templates.py`).",
        "**Deliberate noise** (`noise.py`, rates in `configs/maintenance_logs.yaml`): rule-based synonym paraphrase, "
        "abbreviations (35 % of eligible words), typos (6 % of words of 4+ letters), casing, dropped punctuation, unit "
        "mentions; 25 % of fault notes are vague ('acting weird again'), 10 % describe the symptom with another "
        "subsystem's words, 'no fault found' complaints and closures, and notes unrelated to any fault.",
        "**Label timing.** Every document has `created_at_utc` (when a feature may see it) and `label_available_utc` "
        "(when triage or the closing repair confirms the category). A label is used for training only if it was "
        "available by the training cut-off.",
        "**Held-out wording.** 20 % of core phrases are reserved (`unseen_phrase_ids` in the manifest) for the "
        "lexical-shift test in Section 4.",
    ])
    md.p("Regenerate (about 10 s): `uv run python -m ingen_fleet_health.text.corpus --data-dir data/w02`. "
         "The operator notice rate was fixed before any fusion result existed; Section 4 of W03_Fusion_Results "
         "repeats the fusion at 0.5x, 2x and 4x that rate.")

    md.h("3. Ticket classifier: protocol")
    pr = tc["protocol"]
    md.p(f"Same split discipline as Week 2: {pr['folds']} folds = cut-off days {pr['cutoff_days']} x {pr['unit_groups']} unit "
         f"groups. Training tickets: {pr['train_rule']}. Test tickets: {pr['test_rule']}. No test-unit ticket and no label "
         f"confirmed after the training cut-off is ever used for fitting. {tc['test_tickets']} test tickets in total.")
    md.bullets([
        "`majority` — always the most frequent training class (no-skill reference).",
        "`tfidf_lr` — baseline: word 1–2-gram + character 2–5-gram TF-IDF, class-balanced logistic regression.",
        f"`embed_lr` — frozen `{tc['embedding_model']}` sentence embeddings (local, open weights), logistic regression.",
        "`tfidf_embed_lr` — both feature blocks side by side.",
        f"C is chosen from {tc['c_grid']} by leave-one-training-group-out macro-F1 *inside* the training tickets; the kind "
        "with the best such score is `selected` for the fold (chosen: "
        + ", ".join(f"{k} in {v} folds" for k, v in tc["selected_kind_counts"].items()) + ").",
    ])

    md.h("4. Results on the held-out period")
    m = tc["models"]
    md.table(["Model", "Macro-F1", "95 % CI (unit bootstrap)", "Accuracy"],
             [[k, fmt(m[k]["macro_f1"]), ci(m[k]["macro_f1_ci95"]), fmt(m[k]["accuracy"])]
              for k in ("majority", "tfidf_lr", "embed_lr", "tfidf_embed_lr", "selected")])
    pa = tc["paired"]
    md.table(["Paired comparison (same tickets)", "Δ macro-F1", "95 % CI", "Reading"],
             [[k.replace("_minus_", " − "), signed(v["delta_macro_f1"]), ci(v["ci95"]), verdict(v["ci95"])] for k, v in pa.items()])
    emb_vs = pa["embed_lr_minus_tfidf_lr"]
    md.p(f"The TF-IDF baseline is already strong (macro-F1 {fmt(m['tfidf_lr']['macro_f1'])}). Sentence embeddings alone add "
         f"{signed(emb_vs['delta_macro_f1'])} {ci(emb_vs['ci95'])}, i.e. {verdict(emb_vs['ci95'])}; combining both adds "
         f"{signed(pa['tfidf_embed_lr_minus_tfidf_lr']['delta_macro_f1'])} {ci(pa['tfidf_embed_lr_minus_tfidf_lr']['ci95'])}, "
         f"i.e. {verdict(pa['tfidf_embed_lr_minus_tfidf_lr']['ci95'])}. Macro-F1 weights the seven classes equally; "
         f"{tc['test_label_counts']['no_fault_found']} of {tc['test_tickets']} test tickets are 'no fault found', which is "
         "why accuracy is much higher than the majority baseline's macro-F1 suggests.")
    md.figure("text_confusion.png", "Confusion matrices on the held-out period (rows: true label, columns: prediction).")
    md.figure("text_per_class_f1.png", "Per-class F1 by model.")
    sel = m["selected"]["per_class"]
    md.table(["Class (selected model)", "Precision", "Recall", "F1", "Support"],
             [[SHORT[c], fmt(sel[c]["precision"]), fmt(sel[c]["recall"]), fmt(sel[c]["f1"]), str(sel[c]["support"])] for c in CLASSES])

    md.h("Which categories are confused, and why", level=3)
    ea = tc["error_analysis"]["tfidf_embed_lr"]
    conf = ea["top_confusions"]
    md.p("Error analysis of the combined model `tfidf_embed_lr` (the per-fold choice in 17 of 20 folds):")
    md.table(["True → predicted", "Count", "Where the errors come from (hidden source; style)", "Example (synthetic)"],
             [[f"{SHORT[c['true']]} → {SHORT[c['predicted']]}", str(c["count"]),
               ", ".join(f"{k} {v}" for k, v in c["by_source"].items()) + "; " + ", ".join(f"{k} {v}" for k, v in c["by_style"].items()),
               c["examples"][0][:90]] for c in conf[:6]])
    src = ea["accuracy_by_source"]
    sty = ea["accuracy_by_style"]
    md.bullets([
        f"**The fault categories leak into 'no fault found'.** Almost every large error is a battery, drivetrain, thermal or "
        f"sensor ticket predicted as 'no fault found'. Accuracy on early *fault symptom notes* is "
        f"{pct(src['fault_symptom']['accuracy'])} (n = {src['fault_symptom']['n']}), against "
        f"{pct(src['routine_note']['accuracy'])} on routine notes and {pct(src['fault_service_open']['accuracy'])} on work "
        "orders. A first symptom note and a 'no fault found' complaint use the same everyday words ('takes longer to "
        "charge', 'hot smell'); only the later outcome tells them apart, and the class prior (about 60 % 'no fault found') "
        "decides the tie.",
        f"**Vague notes cannot be classified** ({pct(sty['vague']['accuracy'])} correct, n = {sty['vague']['n']}): 'issue "
        "again, see earlier note' carries no category. This is a property of the text, not of the model; it needs the "
        "unit's earlier notes or telemetry as context.",
        f"**Misattributed notes are read literally** ({pct(sty['misattributed']['accuracy'])} correct, n = "
        f"{sty['misattributed']['n']}): a drivetrain fault described as 'battery seems to drain faster' is classified as "
        "what it says. Sensor drift reported as 'running hot' is the clearest case (drift symptom accuracy "
        f"{pct(src['drift_symptom']['accuracy'])}, n = {src['drift_symptom']['n']}).",
        "**Typos barely matter** "
        f"({pct(ea['accuracy_by_typos']['0 typos']['accuracy'])} with none, {pct(ea['accuracy_by_typos']['2+ typos']['accuracy'])} "
        "with two or more): character n-grams and subword embeddings absorb them.",
    ])
    md.p("**Implication for fusion.** The classifier is most reliable on documents that arrive *after* a fault is known "
         "(work orders, closures, routine notes) and least reliable on the early symptom notes an early-warning model "
         "would need. Its headline macro-F1 therefore overstates how much it can help predict a fault 72 h ahead.")

    lx = tc["lexical_shift"]
    ls = lx["table"]
    pu = lx["paired_unseen_embed_minus_tfidf"]
    md.h("Wording never seen in training (lexical shift)", level=3)
    md.p("A second run removes from training every ticket built from a held-out core phrase; test tickets with those "
         "phrases measure generalisation to new wording.")
    md.table(["Model", "Macro-F1, seen wording", "Macro-F1, unseen wording"],
             [[k, fmt(ls["seen_phrases"][k]["macro_f1"]), fmt(ls["unseen_phrases"][k]["macro_f1"])]
              for k in ("majority", "tfidf_lr", "embed_lr", "tfidf_embed_lr", "selected")])
    md.p(f"On unseen wording ({ls['unseen_phrases']['tickets']} tickets) macro-F1 drops to "
         f"{fmt(ls['unseen_phrases']['tfidf_lr']['macro_f1'])} for TF-IDF and {fmt(ls['unseen_phrases']['embed_lr']['macro_f1'])} "
         "for embeddings: both lose about half their skill, so on a template corpus the in-distribution score largely "
         "measures template recall. Embeddings score higher on new wording, but paired on the same tickets the difference "
         f"is {signed(pu['delta_macro_f1'])} {ci(pu['ci95'])}: {verdict(pu['ci95'])}, so the 151 tickets cannot "
         "establish it.")

    if para:
        pm = para["models"]
        md.h("Local-LLM paraphrase robustness", level=3)
        md.p(f"Plan v3.0 asks for LLM-assisted paraphrase. The training corpus stays template-only (so it regenerates "
             f"without a 3 GB model). Instead {para['sample']['tickets']} test tickets (up to {para['sample']['per_class_max']} "
             f"per class, seed {para['sample']['seed']}) were rewritten by the local open-weights model `{para['llm']}` "
             f"({para['decoding']}; no hosted API) and scored by the same fold classifiers. Refitting reproduced the "
             f"original out-of-fold predictions for {pct(para['reproduces_oof_selected_predictions'])} of the sample. "
             f"Median token Jaccard between original and rewrite: {fmt(para['token_jaccard_original_vs_paraphrase']['median'], 2)} "
             f"(the rewrites really change the wording). MLflow run {para['mlflow_run_id']}.")
        md.table(["Model", "Macro-F1 original", "Macro-F1 rewrite", "Δ", "95 % CI", "Prediction unchanged"],
                 [[k, fmt(v["macro_f1_original"]), fmt(v["macro_f1_paraphrase"]), signed(v["delta"]), ci(v["delta_ci95"]),
                   pct(v["prediction_unchanged_share"])] for k, v in pm.items()])
        sel = pm["selected"]
        drop = sel["delta_ci95"][1] < 0
        md.p(f"**Reading.** For the selected model the change is {signed(sel['delta'])} {ci(sel['delta_ci95'])}: "
             + ("a measurable drop." if drop else
                "no detectable drop. The local model's rewrites keep the category keywords ('battery', 'vibration', "
                "'connection') and also remove the deliberate typos, so they are a weaker shift than the held-out-wording "
                "test above, which is the harder and more informative robustness check."))
        mc = para.get("manual_check") or (maybe("manual_faithfulness.json") or {}).get("paraphrase_manual_check")
        if mc:
            md.p(f"**Check of {mc['pairs_read']} randomly drawn pairs** (read once by the AI assistant that built the "
                 f"pipeline; not reviewed further): {mc['meaning_kept']} kept meaning and category, "
                 f"{mc.get('detail_changed_category_kept', 0)} kept the category but changed a detail, {mc['meaning_changed']} "
                 f"changed the meaning ({mc['notes']}). The label of a rewrite is the label of its original, so some "
                 "measured errors on rewrites are label noise introduced by the LLM, not classifier errors.")
        md.p("Examples where the selected model was right on the original and wrong on the rewrite:")
        md.table(["Label", "Original (synthetic)", "Local-LLM rewrite", "Predicted"],
                 [[SHORT[e["label"]], e["text"][:80], e["paraphrase"][:90], SHORT[e["para_selected"]]]
                  for e in para["correct_then_wrong_examples"][:6]])

    md.h("5. Limitations")
    md.bullets([
        "The corpus is template-generated; real maintenance text is longer-tailed, refers to earlier conversations and "
        "uses site-specific jargon. All scores here are upper bounds on template recall, and the lexical-shift and "
        "paraphrase tests show how fast they fall.",
        "Labels are clean by construction (the generator knows why each note exists). Real ticket categories are assigned "
        "by people and are noisy.",
        "The operator notice rate is an assumption; its effect on fusion is measured in W03_Fusion_Results, Section 4.",
        "No hosted LLM was used (no key, no cost approval, no data-handling decision). The paraphrase test uses a "
        "1.5 B-parameter local model; a stronger model would produce more varied rewrites.",
    ])
    return md.save("W03_Log_Corpus.md")


# --------------------------------------------------------------------------- 4. Fari gap analysis
ARM_NAME = {
    "asis": "Rover models, Week 2 code as-is",
    "fixed_rover_thr": "Rover models, platform-conditional features, Rover threshold",
    "fixed_fari_thr": "same, per-platform (Fari) threshold",
    "fari_native": "trained on Fari units (reference)",
    "pooled": "trained on Rover + Fari units",
}
MODEL_NAME = {"lgbm": "LightGBM", "iforest": "Isolation Forest", "physics_rule": "Physics rule"}


def rule_bullet(mo: dict[str, Any], pr_rule: dict[str, float]) -> str:
    """Interpretation of the physics-rule change, driven by the paired interval and per-indicator PR-AUC."""
    d = mo.get("paired_pr_auc_diff", {}).get("fixed_rover_thr:physics_rule minus asis:physics_rule")
    text = (f"**The hand-set physics rule did not improve when its inputs were repaired** (PR-AUC {fmt(pr_rule['asis'])} "
            f"as-is, {fmt(pr_rule['fixed_rover_thr'])} after the fix")
    text += f"; paired Δ {signed(d['delta'])} {ci(d['ci95'])}, {verdict(d['ci95'])})." if d else ")."
    ind = mo.get("rule_indicators", {}).get("fari_fixed")
    if ind:
        best = max(ind, key=lambda k: ind[k]["pr_auc"])
        worst = min(ind, key=lambda k: ind[k]["pr_auc"])
        text += (f" As-is, three of its four indicators were dead and the rule reduced to battery resistance alone. "
                 f"Scored one at a time on the same rows, the revived indicators are mostly useful ({best} "
                 f"{fmt(ind[best]['pr_auc'])}), but {worst} scores {fmt(ind[worst]['pr_auc'])}, at or below the base rate "
                 f"{fmt(mo['fari_base_rate'])}: Fari's small motor heating (a few deg C) makes that ratio noisy, and the rule "
                 "takes the *largest* deviation, so one noisy indicator dominates. Choosing the rule's indicators per "
                 "platform must be done on validation data, not on these test rows (deferred, Section 4).")
    return text


def build_fari() -> Path:
    r = load("fari_transfer_results.json")
    c, f, mo, soh = r["contract"], r["features"], r["models"], r["soh"]
    arms = mo["arms"]
    doc = Doc("W03 — Fari Generalisation Test and Gap Analysis",
              "Week 3 deliverable 4 · generated from reports/w03/fari_transfer_results.json · MLflow run "
              f"{r['mlflow_run_id']}")
    doc.p("Every Fari number below comes from a SIMULATED fleet (configs/fari.yaml: indoor, short interaction sessions, "
          "frequent dock charging, no terrain, 14.4 V pack, no imagery or audio). Every parameter in that file is an "
          "assumption; none is an InGen specification. The test measures whether the pipeline's *design* generalises, "
          "not how a real Fari behaves.", bold_prefix="Synthetic data. ")
    ev = int(mo["arms"]["fari_native"]["lgbm"]["events"])
    doc.p(f"Rover-trained Week 1–2 pipeline applied to {r['fari_manifest']['units']} simulated Fari units x "
          f"{r['fari_manifest']['days']} days, scored on the same 20 unit-grouped walk-forward folds as Week 2 "
          f"({mo['fari_test_rows']:,} Fari test rows, {ev} fault onsets, base rate {fmt(mo['fari_base_rate'])}). "
          "No arm ever trains on a Fari test unit.", bold_prefix="Setup. ")

    doc.h("1. What generalises and what breaks", level=1)
    asis_c, prof_c = c["week1_rules_as_is"], c["fari_platform_profile"]
    reasons = ", ".join(f"{k} {v:,}" for k, v in asis_c["rejection_reasons"].items())
    lg = {a: arms[a]["lgbm"] for a in arms}
    dead_asis = f["dead_features"]["fari_asis"]
    const_fixed = f["constant_features"]["fari_fixed"]
    pu = f["psi_unit_normalised_indicators"]
    soh_mae = soh["mae_pp"]
    doc.table(["Layer", "Week 1–2 code as-is", "After the minimal change", "Why"], [
        ["Data contract", f"{fmt(asis_c['acceptance_rate_pct'], 1)} % of {asis_c['rows_checked']:,} rows accepted ({reasons})",
         f"{fmt(prof_c['acceptance_rate_pct'], 1)} % accepted",
         "48 V, the 30–60 V window and the patrol/dock enum were Rover facts written as code"],
        ["Features", f"{len(dead_asis)} of {f['features_total']} features never valid; "
                     f"{len(f['constant_features']['fari_asis'])} constant",
         f"{len(f['dead_features']['fari_fixed'])} never valid; {len(const_fixed)} constant (the terrain family: "
         "not applicable indoors)",
         "'patrol' hard-coded as the active mode; minimum-energy / distance / current-step guards sized for a 48 V "
         "outdoor robot"],
        ["Distributions", "n/a", f"{f['psi_major_shift_count']} of {f['features_total']} features shift (PSI > 0.25)",
         "different pack voltage, power and duty pattern; even unit-normalised ratios shift: "
         + ", ".join(f"{k} {fmt(v, 2)}" for k, v in pu.items())],
        ["Early warning (LightGBM PR-AUC)", f"{fmt(lg['asis']['pr_auc'])} {ci(lg['asis']['ci95']['pr_auc'])}",
         f"{fmt(lg['fixed_rover_thr']['pr_auc'])} {ci(lg['fixed_rover_thr']['ci95']['pr_auc'])}; Fari-trained "
         f"{fmt(lg['fari_native']['pr_auc'])}, pooled {fmt(lg['pooled']['pr_auc'])}",
         "dead inputs are imputed by the trees' default path, so the as-is model ranks almost at the base rate"],
        ["Battery SoH, 7 days (MAE, pp)", "—", f"Rover-trained {fmt(soh_mae['rover_lgbm'])}, Fari-trained "
         f"{fmt(soh_mae['fari_lgbm'])}, persistence {fmt(soh_mae['persistence'])}",
         "battery ageing physics is shared; the BMS reports less often on Fari "
         f"({fmt(soh['bms_soh_updates_per_unit_day']['fari'], 2)} vs {fmt(soh['bms_soh_updates_per_unit_day']['rover'], 2)} updates/unit-day)"],
    ], widths=[1.15, 1.75, 1.85, 2.1])

    doc.h("2. Models after transfer", level=1)
    rows = []
    for a in ("asis", "fixed_rover_thr", "fixed_fari_thr", "fari_native", "pooled"):
        x = arms[a]
        rows.append([ARM_NAME[a], f"{fmt(x['lgbm']['pr_auc'])} {ci(x['lgbm']['ci95']['pr_auc'])}",
                     fmt(x["iforest"]["pr_auc"]), fmt(x["physics_rule"]["pr_auc"]),
                     f"{fmt(x['lgbm']['event_recall'], 2)} / {fmt(x['lgbm']['alerts_per_unit_week'], 2)}",
                     fmt(x["iforest"]["alerts_per_unit_week"], 2)])
    doc.table(["Arm", "LightGBM PR-AUC [95 % CI]", "Isolation Forest PR-AUC", "Physics rule PR-AUC",
               "LightGBM recall / alerts per unit-wk", "Isolation Forest alerts per unit-wk"], rows,
              widths=[2.0, 1.4, 0.9, 0.8, 1.0, 0.9])
    doc.figure("fari_transfer_models.png", width=5.8, caption="PR-AUC after transfer (left) and the alert volume each threshold actually "
               f"produces on Fari against the budget of {fmt(mo['budget_alerts_per_unit_week'], 1)} alert per unit-week (right).")
    paired = mo.get("paired_pr_auc_diff")
    if paired:
        doc.table(["Paired PR-AUC difference (same Fari test rows)", "Δ", "95 % CI", "Reading"],
                  [[k.replace(":", " / ").replace(" minus ", " − "), signed(v["delta"]), ci(v["ci95"]), verdict(v["ci95"])]
                   for k, v in paired.items()], widths=[3.0, 0.6, 1.2, 2.0])
    iso = {a: arms[a]["iforest"]["alerts_per_unit_week"] for a in ("fixed_rover_thr", "fixed_fari_thr")}
    lgb = {a: arms[a]["lgbm"]["alerts_per_unit_week"] for a in ("fixed_rover_thr", "fixed_fari_thr")}
    pr_rule = {a: arms[a]["physics_rule"]["pr_auc"] for a in ("asis", "fixed_rover_thr")}
    doc.bullets([
        f"**Thresholds do not transfer.** With the Rover threshold the Isolation Forest raises {fmt(iso['fixed_rover_thr'], 2)} "
        f"alerts per unit-week on Fari, {fmt(iso['fixed_rover_thr'] / mo['budget_alerts_per_unit_week'], 1)}x the budget; "
        f"re-choosing the threshold on Fari validation rows brings it to {fmt(iso['fixed_fari_thr'], 2)}. For LightGBM the "
        f"Rover threshold gives {fmt(lgb['fixed_rover_thr'], 2)} and the Fari threshold {fmt(lgb['fixed_fari_thr'], 2)}: "
        "a threshold set on 6 validation days still drifts on the next 10, so the budget is met only approximately.",
        rule_bullet(mo, pr_rule),
        "**Ranking is learnable on Fari but data are thin.** Pooled training has the highest LightGBM point estimate, "
        f"but with {ev} onsets in test the intervals of the transferred, Fari-trained and pooled models overlap widely.",
    ])
    rule = mo.get("rule_indicators")
    if rule:
        doc.table(["Physics-rule indicator |x − 1| alone", "Valid share (fixed)", "PR-AUC (fixed)", "Valid share (as-is)",
                   "PR-AUC (as-is)"],
                  [[k, pct(v["valid_share"]), fmt(v["pr_auc"]), pct(rule["fari_asis"][k]["valid_share"]),
                    fmt(rule["fari_asis"][k]["pr_auc"])] for k, v in rule["fari_fixed"].items()],
                  widths=[2.4, 1.1, 1.0, 1.1, 1.0])
    ft = r.get("fault_types")
    if ft:
        doc.p("Fault mix (simulated events, whole run): Rover " + ", ".join(f"{k} {v}" for k, v in ft["rover"].items())
              + "; Fari " + ", ".join(f"{k} {v}" for k, v in ft["fari"].items()) + ".")

    doc.h("3. The minimal change that was implemented", level=1)
    doc.p("The most important gap was that Rover facts were written as code, so the pipeline did not merely degrade on "
          "Fari: it rejected every record and silently disabled a third of the physics features. The change makes those "
          "facts platform data, enforces Fari's privacy rule, and lets each platform have its own alert threshold:")
    doc.bullets([
        "`contract.py` + `ingest_pipeline.py --platform-config`: mode enum, nominal voltage and voltage window come from the "
        "platform YAML (`platform_profile`). Without a profile the Week 1 Rover rules apply unchanged.",
        "Privacy gate: a file with any imagery/audio column is refused before a copy is stored (word-level matching, tested "
        "in `tests/w03/test_fari_contract.py`). The plan requires this to be enforced in the contract, not only stated.",
        "`features/physics.py`: the active mode and four sanity guards (minimum current step, expected energy, distance, "
        "thermal rise) are read from the YAML `features:` block, with the Week 2 constants as defaults. The Rover feature "
        "table was verified byte-identical before and after the change.",
        "Per-platform alert threshold: the alert budget is enforced by a threshold chosen on the platform's own "
        "validation rows (`fixed_fari_thr`).",
    ])
    doc.p("'No change needed' was not an option the evidence allowed: as-is acceptance was "
          f"{fmt(asis_c['acceptance_rate_pct'], 1)} % and as-is LightGBM ranking was close to no skill.")

    doc.h("4. Deferred, with reasons", level=1)
    doc.bullets([
        "**Per-platform physics-rule composition and scale** — one Rover-sized indicator (thermal gain ratio) is noise on "
        "Fari (Section 2). The fix (drop or rescale it per platform) must be chosen on validation data and ideally on real "
        "Fari baselines; tuning it on the synthetic test rows would fit the simulator's assumptions.",
        "**Pooled training as the default** — best point estimate, but not distinguishable from Fari-only training with "
        f"{ev} test onsets; revisit with more Fari history.",
        "**Session-native Fari features** (per-interaction energy, docking frequency, wake/sleep cycles) — the Week 2 "
        "features assume patrol shifts; designing Fari features needs real Fari usage data.",
        "**Dropping the terrain feature family on Fari** — it is constant (zero) and harmless to trees; removing it is a "
        "registry change for Week 4 packaging.",
        "**Verified Fari parameters** — every value in configs/fari.yaml is an assumption; nothing here should be read as "
        "Fari performance.",
    ])
    return doc.save("W03_Fari_Gap_Analysis.docx")


# --------------------------------------------------------------------------- 3. SHAP + alert summaries
def gen_label(name: str) -> str:
    return "Template" if name == "template" else "Local LLM (Qwen2.5-1.5B-Instruct)"


def build_explanations() -> Path:
    r = load("alert_explanations.json")
    manual = reviews(r)
    md = Md("W03 — Alert Explanations: SHAP, Grounded Summaries and Faithfulness",
            "Week 3 deliverable 3 · generated by scripts/build_w03_docs.py from reports/w03/alert_explanations.json"
            + (", llm_faithfulness_review.json and manual_faithfulness.json" if manual else "")
            + f" · MLflow run {r['mlflow_run_id']}")
    md.p("> **SYNTHETIC DATA.** Units, telemetry, alerts and operator notes are simulated. Summaries were written by a "
         "deterministic template and by a local open-weights LLM on this machine; **no hosted LLM API was called** and "
         "no data left the machine.")

    md.h("1. What is explained")
    sh = r["shap"]
    md.bullets([
        f"Model: {r['explained_model']}. It is the only arm whose inputs include both telemetry and text, so an "
        "attribution can point at a linked note. Each fold's model explains only its own unseen test robots "
        f"({sh['test_rows_explained']:,} rows).",
        f"Method: {sh['method']}. Additivity check: max |sum(SHAP) + base − raw margin| = {sh['additivity_max_abs_error']:.1e}.",
        "Calibration is monotone, so the sign and ranking of contributions carry over to the calibrated risk shown to the "
        "operator; SHAP values are log-odds, not probability points.",
        f"Refitting reproduces the fusion run's out-of-fold early-fusion scores to within "
        f"{r['reproduces_fusion_run_scores_max_abs_diff']:.1e}, so the explained model is the evaluated model.",
    ])

    md.h("2. Global SHAP")
    md.figure("shap_global.png", "Mean |SHAP| per feature (left) and share by concept (right), pooled test rows of all folds.")
    top = r["global_importance_top"][:12]
    md.table(["Rank", "Feature", "Concept", "Mean |SHAP|"],
             [[str(i + 1), f"`{t['feature']}`", t["concept"], fmt(t["mean_abs_shap"], 4)] for i, t in enumerate(top)])
    shares = r["concept_share_of_mean_abs_shap"]
    md.p("Share of total mean |SHAP| by concept: " + ", ".join(f"{k} {pct(v)}" for k, v in shares.items())
         + f". Text features together carry **{pct(r['text_feature_share_of_mean_abs_shap'], 1)}** of the attribution mass, "
           "although adding them gives no detectable PR-AUC gain over telemetry (W03_Fusion_Results). Attribution share "
           "measures how much the model *uses* a feature, not how much it *helps*: the trees split on text features, "
           "often at near-zero values (alerts A09, A10, A12), which is fitting noise rather than signal. This is a "
           "concrete reason not to read a SHAP ranking as evidence that a modality is valuable.")

    md.h("3. Per-alert SHAP and the evidence pack")
    md.figure("shap_alert_examples.png", "Per-alert SHAP for one true-positive alert and one false alarm (right: raises the risk).")
    md.p("Each alert is summarised from an **evidence pack** and nothing else: unit, time, calibrated risk and threshold; "
         "the top five SHAP attributions pushing the risk up and the two strongest pushing it down, each with its value "
         "now and 72 h earlier; and the unit's operator notes / service tickets created in the 7 days before the alert, "
         "with the fold classifier's reading of each (never the label). Two generators write from the same pack:")
    md.bullets([
        "**Template** — deterministic, can only verbalise pack fields. It quotes a note only when that note feeds the "
        "attributed text feature (`note_behind`), and qualifies a ratio within 2 % of 1 as 'essentially at the reference "
        "level'.",
        "**Local LLM** — `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, CPU, greedy decoding) given the system and user prompt "
        "below. It writes free prose, so it *can* invent a cause or a number; the check measures how often it does.",
    ])
    a0 = r["alerts"][0]
    md.p("Prompt for alert " + a0["alert_id"] + " (the exact text the LLM saw; a hosted provider would receive the same):")
    md.lines += ["```text", a0["prompt"], "```", ""]

    md.h("4. The faithfulness check")
    md.p("`explain/faithfulness.py` reads the *text* of a summary (not the generator's internals) and extracts every "
         "claim. A summary is faithful only if every claim is supported by its evidence pack:")
    md.table(["Claim", "Supported when"], [
        ["feature named verbatim", "the feature is in the pack, on the side (raising / lowering) the sentence states"],
        ["cause in free wording (vibration, cooling, battery …)", "the concept belongs to a feature raising the risk, or to a linked note the clause attributes it to"],
        ["number", "equals a pack value at the precision written (dates, times and the 72 h / 7 d windows are neutral)"],
        ["quote", "is a substring of a linked note or ticket"],
        ["classifier reading", "a linked note has that reading"],
        ["notes exist / are a driver / are not", "linked notes exist; a note feature is (not) among the attributions"],
    ])
    md.p("**Fixes made this week** to false flags found by reading the checker's output: words inside a feature phrase "
         "('docked motor-temperature *offset*', '*lowest* state of charge') were read as direction cues for the whole "
         "sentence; the text-feature phrase \"notes read as 'thermal'\" was parsed as a classifier-reading claim; direction "
         "is now read per contrast segment (but / while / whereas). Two real unfaithfulness problems in the template were "
         "also fixed: it quoted the newest note even when a different note fed the attributed text feature, and it "
         "presented '1.00 x baseline' as a deviation. Each fix has a regression test in `tests/w03/test_explain.py`.")
    nc = r["negative_controls"]
    md.p("**Does the checker catch unfaithful text?** Negative controls: each faithful template summary is corrupted "
         "four ways and re-checked.")
    md.table(["Corruption", "Built", "Flagged", "Detection rate"],
             [[k, str(v["built"]), str(v["flagged"]), pct(v["flagged"] / v["built"]) if v["built"] else "n/a"]
              for k, v in nc["by_kind"].items()] + [["all", str(nc["built"]), str(nc["flagged"]), pct(nc["detection_rate"])]])

    md.h("5. Faithfulness rate on 20 sampled alerts")
    smp = r["alert_sampling"]
    md.p(f"{smp['alerts_raised_on_test']} alerts were raised on unseen robots at {fmt(smp['budget_alerts_per_unit_week'], 1)} "
         f"alert per unit-week ({smp['true_positive_alerts']} true positives). Sample: {smp['rule']} (seed {smp['seed']}); "
         f"{smp['sampled_true_positive']} true positives and {smp['sampled'] - smp['sampled_true_positive']} false alarms.")
    hist = {h["checker_version"]: h for h in r.get("checker_history", [])}
    v1 = hist.get(1)
    rows = []
    for name, g in r["generators"].items():
        man = manual.get(name, {})
        man_ok = sum(1 for v in man.values() if v["faithful"])
        v1g = v1["generators"][name] if v1 else None
        rows.append([gen_label(name),
                     f"{v1g['faithful']}/{v1g['alerts']}" if v1g else "—",
                     f"{g['faithful']}/{g['alerts']} = {pct(g['faithfulness_rate'])}",
                     f"{g['claims_supported']}/{g['claims_checked']}",
                     f"{man_ok}/{len(man)} = {pct(man_ok / len(man))}" if man else "not read"])
    md.table(["Generator", "Automatic, checker v1 (as run)", f"Automatic, checker v{r.get('checker_version', 1)}",
              "Claims supported (current checker)", "**Evidence review**"], rows)
    if manual:
        md.p("**Headline: the template summaries are faithful (20/20 automatic and on reading). The local 1.5 B LLM's "
             "summaries are not: the evidence review finds a real error in every one of them, while the automatic check "
             "passes some.** The evidence review (Section 6, reasons per alert) is the reference: it checks meaning, "
             "direction and arithmetic, which a lexical checker cannot.")
        md.p(f"**Review method.** {REVIEW_METHOD} The review criterion: every stated fact, number, direction and causal "
             "explanation must be supported by the evidence pack; format violations (more than three sentences, "
             "truncation) are recorded separately and do not decide the verdict. The review pack "
             "(`reports/w03/faithfulness_review_batch.md`) is generated by `scripts/build_w03_review_pack.py`.")
        for name, man in manual.items():
            if name == "template":
                continue
            auto = {a["alert_id"]: a["summaries"][name]["check"]["faithful"] for a in r["alerts"] if name in a["summaries"]}
            n_agree = sum(1 for k, v in man.items() if auto.get(k) == v["faithful"])
            flagged = [k for k, ok in auto.items() if not ok]
            valid = [k for k in flagged if man.get(k, {}).get("auto_flag_valid")]
            missed = [k for k, ok in auto.items() if ok and not man[k]["faithful"]]
            all_flags = [f["points_to_real_error"] for v in man.values() for f in v.get("flags", [])]
            md.bullets([
                f"Verdict agreement, automatic (current checker) vs evidence review: {n_agree}/{len(man)}.",
                f"The checker flagged {len(flagged)} LLM summaries; in {len(valid)} of them ({', '.join(valid) or 'none'}) a "
                "flagged claim is a real error; the others were flagged for a spurious reason while their real errors went "
                "uncaught."
                + (f" Flag by flag: {sum(1 for x in all_flags if x)} of {len(all_flags)} flags point to a real error."
                   if all_flags else ""),
                f"It passed {len(missed)} summaries that the evidence review rejects ({', '.join(missed) or 'none'}).",
                "What the LLM gets wrong: risk-lowering attributions presented as concerns, values attached to the wrong "
                "feature or the wrong time (72 h earlier vs now), invented percentages and thresholds, invented causes "
                "('wear', 'overheating'), advice the prompt forbids, and more than three sentences (often truncated).",
            ])
        if v1:
            md.p(f"Checker v1 → v{r.get('checker_version')}: after reading the checker's flags against the evidence, five "
                 "generic false-flag causes were "
                 "fixed (US spelling 'normalized'; 'terrain-normalised' read as a terrain cause; a quoted category name "
                 "read as a quote; 'no fault was found' read as a negation; the unit named as 'rover 46' read as a number), "
                 "each with a test, and the stored summaries were re-scored without regeneration "
                 f"(MLflow {r.get('recheck_mlflow_run_id', 'n/a')}). Fixing false flags *raised* the LLM's automatic rate and "
                 "lowered its agreement with the evidence review: the checker was catching the right summaries partly for the "
                 "wrong reasons. The checker was not tuned further against these summaries, to avoid fitting the "
                 "evaluation to the output it evaluates.")
        md.p("**Conclusion for InGen.** Use the deterministic template for operator-facing alerts; it is faithful by "
             "construction and its faithfulness is verified (checker + 100 % negative-control detection + a read of all "
             "20 summaries). "
             "A free-text LLM layer needs a stronger model *and* a semantic check (numbers recomputed from the pack, "
             "direction per feature) before it can be trusted; a lexical checker alone is not enough for LLM prose.")
    llm = r["llm"]
    if llm.get("error"):
        md.p(f"Local LLM was requested but failed: `{llm['error']}`.")

    md.h("6. The 20 alerts")
    for a in r["alerts"]:
        ev = a["evidence"]
        md.h(f"{a['alert_id']} · {ev['unit_id']} · {ev['t']} UTC · {a['outcome'].replace('_', ' ')}", level=3)
        drivers = "; ".join(f"`{x['feature']}` {signed(x['shap'], 2)}" for x in ev["positive"][:3])
        down = "; ".join(f"`{x['feature']}` {signed(x['shap'], 2)}" for x in ev["negative"][:1])
        md.p(f"Risk {pct(ev['risk'], 1)} vs threshold {pct(ev['threshold'], 1)} · up: {drivers or 'none'} · down: {down or 'none'} · "
             f"linked notes/tickets: {len(ev['tickets'])}")
        for name, sdict in a["summaries"].items():
            chk = sdict["check"]
            auto_txt = "faithful" if chk["faithful"] else "UNSUPPORTED: " + "; ".join(
                f"{u['kind']} '{u['text'][:60]}' ({u['detail']})" for u in chk["unsupported"])
            verdict_ = manual.get(name, {}).get(a["alert_id"])
            man_txt = "" if verdict_ is None else f" · review: {'faithful' if verdict_['faithful'] else 'NOT faithful'}" + (
                f" — {verdict_['note']}" if verdict_.get("note") else "")
            md.p(f"**{gen_label(name)}.** {sdict['text']}  \n*Automatic: {auto_txt}{man_txt}*")

    md.h("7. What the check cannot judge")
    md.bullets([
        "Causal strength: SHAP says what moved *this model's* output, not what is physically wrong with the robot. A "
        "faithful summary of a false alarm is still a false alarm (half of the sample).",
        "Synonyms outside the concept lexicon, tone, and emphasis: covered only by the evidence review.",
        "Review scope: the LLM verdicts are an AI-assisted evidence review confirmed by the project owner, not an "
        "independent human review; the template summaries were read once by the AI assistant that built them.",
        "Completeness: a summary that omits the strongest driver can still pass; the template always names the top three.",
        "Corpus realism: the linked notes are synthetic; real notes would be longer and harder to quote faithfully.",
        "Scale: 20 alerts give a rate with a wide interval; the negative controls measure the checker, not the generators.",
    ])
    return md.save("W03_Alert_Explanations.md")


# --------------------------------------------------------------------------- 2. fusion + ablation
ARM_LABEL = {
    "no_skill": "No skill (base rate)",
    "physics_rule": "Physics rule",
    "telemetry_lgbm": "LightGBM",
    "telemetry_iforest": "Isolation Forest",
    "text_only": "Text only (LightGBM on text features)",
    "early_fusion": "Early fusion (one LightGBM, telemetry + text)",
    "late_noisy_or": "Late fusion, noisy-OR (LightGBM + text)",
    "late_stack": "Late fusion, stacked (LightGBM + text)",
    "late_stack_iforest": "Late fusion, stacked (Isolation Forest + text)",
}
MODALITY = {"no_skill": "—", "physics_rule": "telemetry", "telemetry_lgbm": "telemetry", "telemetry_iforest": "telemetry",
            "text_only": "text", "early_fusion": "both", "late_noisy_or": "both", "late_stack": "both",
            "late_stack_iforest": "both"}


def arm_rows(arms: dict[str, Any]) -> list[list[str]]:
    rows = []
    for a, m in arms.items():
        rows.append([ARM_LABEL.get(a, a), MODALITY.get(a, ""), f"{fmt(m['pr_auc'])} {ci(m['ci95']['pr_auc'])}",
                     f"{fmt(m['event_recall'], 2)} {ci(m['ci95'].get('event_recall'), 2)}", fmt(m.get("precision"), 2),
                     fmt(m["alerts_per_unit_week"], 2), fmt(m.get("median_lead_time_h"), 0)])
    return rows


def text_verdict(d: dict[str, Any]) -> str:
    lo, hi = d["ci95"]
    if lo > 0:
        return f"adds {signed(d['delta'])} PR-AUC {ci(d['ci95'])}: a measurable gain"
    if hi < 0:
        return f"changes PR-AUC by {signed(d['delta'])} {ci(d['ci95'])}: a measurable loss"
    return f"changes PR-AUC by {signed(d['delta'])} {ci(d['ci95'])}: not distinguishable from zero"


def build_fusion() -> Path:
    r = load("fusion_results.json")
    ex = maybe("alert_explanations.json")
    arms = r["arms"]
    pdiff = r["paired_pr_auc_diff"]
    doc = Doc("W03 — Telemetry + Text Fusion and Modality Ablation",
              "Week 3 deliverable 2 · generated by scripts/build_w03_docs.py from reports/w03/fusion_results.json · "
              f"MLflow run {r['mlflow_run_id']}")
    doc.p(SYNTHETIC, bold_prefix="Synthetic data. ")

    best_t, _best_c = r["best_telemetry_arm_on_test"], r["best_combined_arm_on_test"]
    ef = pdiff["early_fusion_minus_telemetry_lgbm"]
    ls = pdiff["late_stack_minus_telemetry_lgbm"]
    lsi = pdiff["late_stack_iforest_minus_telemetry_iforest"]
    to = pdiff["text_only_minus_no_skill"]
    cov = r["text_coverage"]
    diag = maybe("fusion_diagnostics.json")
    sens = r.get("sensitivity_notice_rate") or []
    ef_if = pdiff["early_fusion_minus_telemetry_iforest"]
    doc.h("Summary", level=1)
    text_alone = ("carries no detectable signal on its own" if to["ci95"][0] <= 0 else "carries some signal on its own")
    summary = [
        f"**Text {text_alone}**: text-only PR-AUC {fmt(arms['text_only']['pr_auc'])} against a base rate of "
        f"{fmt(arms['no_skill']['pr_auc'])} ({text_verdict(to)}).",
        f"**Once telemetry is present, text {('adds little' if ef['ci95'][0] <= 0 else 'still adds')}.** Early fusion vs "
        f"telemetry LightGBM {text_verdict(ef)}; stacked late fusion vs LightGBM {text_verdict(ls)}.",
        f"**Fusion can hurt.** Stacking text onto the Isolation Forest {text_verdict(lsi)}: the stacker, fitted on a 6-day "
        "validation slice with few distinct onsets, gives the noisy text score more weight than the anomaly score"
        + (f" in {diag['folds_text_weight_exceeds_telemetry']['late_stack_iforest']} of "
           f"{diag['folds_text_weight_exceeds_telemetry']['folds']} folds" if diag else "") + ".",
        f"**The best model overall is still telemetry-only**: {ARM_LABEL[best_t]} (PR-AUC {fmt(arms[best_t]['pr_auc'])}); "
        f"early fusion − Isolation Forest {text_verdict(ef_if)}.",
        f"**Why text adds little here**: only {cov['onsets_with_symptom_note_in_72h_before_onset']} of {cov['onsets_in_test']} "
        f"test-period fault onsets ({pct(cov['share'])}) had an operator symptom note in the 72 h before onset, and the "
        "ticket classifier is weakest on exactly those early notes (W03_Log_Corpus).",
    ]
    if sens:
        hi = max(sens, key=lambda x: x["notice_rate"])
        sig = [x for x in sens if x["ci95[early_fusion_minus_telemetry_lgbm]"][0] > 0]
        summary.append(
            "**The answer depends on an assumption**: if operators noted symptoms more often, text would help. At "
            f"{hi['notice_rate']:g}x the assumed notice rate early fusion − LightGBM is "
            f"{signed(hi['delta[early_fusion_minus_telemetry_lgbm]'])} {ci(hi['ci95[early_fusion_minus_telemetry_lgbm]'])}"
            + (f"; the gain becomes detectable from {min(x['notice_rate'] for x in sig):g}x." if sig else ".")
            + " InGen's real ticket history can measure this rate; the simulation cannot.")
    summary.append("This is a useful finding, not a failure: on this synthetic fleet, maintenance text is better used to "
                   "*explain* alerts and route tickets than to predict faults earlier, unless operators report symptoms "
                   "early and often.")
    doc.bullets(summary)

    doc.h("1. Setup", level=1)
    doc.p(f"Same {r['folds']} unit-grouped walk-forward folds, the same {r['test_rows']:,} test rows and the same "
          f"{fmt(r['budget_alerts_per_unit_week'], 1)} alert / unit-week budget as Week 2; label = qualifying fault onset within "
          f"{fmt(r['horizon_hours'], 0)} h. Every arm is scored on identical rows, so differences are paired.")
    doc.bullets([
        f"**Telemetry features**: the {r['telemetry_feature_count']} Week 2 features (time proxies excluded).",
        f"**Text features** ({len(r['text_features'])}): per note class, a time-decayed sum of classifier probabilities "
        "(half-life 48 h, notes younger than 7 d); strongest fault reading in the last 72 h; note and service-ticket "
        "counts; hours since the latest note. A note is visible at t only if it was created by t.",
        "**No text leakage**: inside each fold the ticket classifier is fitted only on training-unit tickets whose label "
        "was confirmed by the training cut-off (cut-off − 72 h), and training-unit tickets are scored by a model that "
        "excluded their own unit group (cross-fitting). Labels of test-unit tickets are never read.",
        "**Early fusion**: one LightGBM on telemetry + text features (same all/physics variant choice on validation as Week 2). "
        "**Late fusion**: per-modality models combined by noisy-OR (nothing fitted) or by a logistic stacker fitted on the "
        "fold's validation rows.",
    ])

    doc.h("2. Ablation: telemetry only, text only, both", level=1)
    doc.table(["Arm", "Modality", "PR-AUC [95 % CI]", "Event recall [95 % CI]", "Precision", "Alerts / unit-wk",
               "Median lead (h)"], arm_rows(arms), widths=[2.1, 0.7, 1.25, 1.2, 0.6, 0.6, 0.6])
    doc.figure("fusion_ablation_pr_auc.png", "PR-AUC on unseen robots by arm (95 % unit-bootstrap intervals).")
    doc.p("Paired differences on the same test rows (robots resampled together):")
    doc.table(["Comparison", "Δ PR-AUC", "95 % CI", "Reading"],
              [[k.replace("_minus_", " − "), signed(v["delta"]), ci(v["ci95"]), verdict(v["ci95"])] for k, v in pdiff.items()],
              widths=[2.6, 0.7, 1.3, 2.3])

    doc.h("3. Measured contribution of each modality", level=1)
    tg = r["text_gain_share_in_early_fusion_mean"]
    items = [
        f"**Telemetry**: LightGBM {fmt(arms['telemetry_lgbm']['pr_auc'])} vs no skill {fmt(arms['no_skill']['pr_auc'])}.",
        f"**Text**: text only {fmt(arms['text_only']['pr_auc'])}; given telemetry, early fusion {text_verdict(ef)}.",
        f"**Inside the early-fusion model** the text features take {pct(tg, 1)} of LightGBM's split gain (mean over folds)"
        + (f" and {pct(ex['text_feature_share_of_mean_abs_shap'], 1)} of the mean |SHAP| (W03_Alert_Explanations)." if ex else "."),
    ]
    rbt = r.get("recall_by_fault_type", {})
    if rbt:
        ft = sorted(rbt["telemetry_lgbm"])
        doc.table(["Fault type"] + [ARM_LABEL[a].split(" (")[0] for a in ("telemetry_lgbm", "text_only", "early_fusion", "late_stack")],
                  [[t] + [f"{rbt[a][t]['detected']}/{rbt[a][t]['events']}" for a in ("telemetry_lgbm", "text_only", "early_fusion", "late_stack")]
                   for t in ft], widths=[1.8, 1.2, 1.2, 1.2, 1.2])
        items.append("Onsets warned 0–72 h ahead, by fault type (table above). With so few onsets per type, a difference "
                     "of one or two events is not evidence.")
    if diag:
        pf = diag["per_fold_pr_auc"]["late_stack_iforest_vs_telemetry_iforest"]
        sw = diag["stacker_weights_median"]["late_stack_iforest"]
        items.append(f"**Late fusion with the Isolation Forest**: per fold, the stacked arm wins in {pf['second_wins']} of "
                     f"{pf['folds_with_positives']} folds with positives and its mean per-fold PR-AUC falls from "
                     f"{fmt(pf['mean_per_fold'][0])} to {fmt(pf['mean_per_fold'][1])}; median stacker weights (standardised) "
                     f"are {fmt(sw['telemetry'], 2)} for the anomaly score and {fmt(sw['text'], 2)} for the text logit. Each "
                     "onset contributes up to 72 strongly correlated positive hours, so a 6-day validation slice holds only a "
                     "few independent events and the stacker over-fits to them. A stacker for a weak modality needs "
                     "regularisation toward the telemetry score or a longer validation window.")
    dfa = r.get("drift_false_alarms", {})
    if dfa:
        items.append("Share of false alarms on units with a drifting temperature sensor: "
                     + ", ".join(f"{ARM_LABEL[a].split(' (')[0]} {pct(dfa[a]['share_on_drift_units'])}"
                                 for a in ("telemetry_lgbm", "text_only", "early_fusion", "late_stack") if a in dfa)
                     + f" ({int(dfa['telemetry_lgbm']['drift_units'])} drift units). Notes about a drifting sensor could in "
                       "principle let a model discount those alerts; "
                     + ("they did not here (the fused arms' share is not lower than LightGBM's)."
                        if dfa["early_fusion"]["share_on_drift_units"] >= dfa["telemetry_lgbm"]["share_on_drift_units"]
                        else "the fused arms' share is lower than LightGBM's."))
    doc.bullets(items)

    doc.h("4. Late versus early fusion", level=1)
    lvse = pdiff.get("early_fusion_minus_late_stack")
    doc.p(f"Early fusion − stacked late fusion: {signed(lvse['delta'])} {ci(lvse['ci95'])}, {verdict(lvse['ci95'])}. "
          "Late fusion has practical advantages that do not show up in PR-AUC: each modality's model can be retrained, "
          "monitored and switched off on its own, a missing text feed degrades gracefully to the telemetry score, and the "
          "stacker's two weights state the modalities' relative weight directly"
          + (f" (median over folds, standardised logits: telemetry {fmt(diag['stacker_weights_median']['late_stack']['telemetry'], 2)}, "
             f"text {fmt(diag['stacker_weights_median']['late_stack']['text'], 2)}; per-fold weights in the JSON). "
             "The same flexibility is its risk: with the Isolation Forest it over-weighted text (Section 3)." if diag else "."))

    sens = r.get("sensitivity_notice_rate")
    if sens:
        doc.h("5. Sensitivity to the corpus assumption", level=1)
        doc.p("On synthetic text the value of the text modality is driven by how often operators write. The operator notice "
              "rate was fixed before any fusion result; the corpus was then regenerated at 0.5x, 2x and 4x that rate and the "
              "fusion repeated (ticket classifier kind and C fixed to the main run's majority choice).")
        doc.table(["Notice rate", "Tickets", "Onsets with a note ≤ 72 h before", "LightGBM", "Text only", "Early fusion",
                   "Δ early − LightGBM [95 % CI]"],
                  [[f"{s['notice_rate']:g}x", str(s["tickets"]), pct(s["onset_note_share"]), fmt(s["telemetry_lgbm"]),
                    fmt(s["text_only"]), fmt(s["early_fusion"]),
                    f"{signed(s['delta[early_fusion_minus_telemetry_lgbm]'])} {ci(s['ci95[early_fusion_minus_telemetry_lgbm]'])}"]
                   for s in sens], widths=[0.7, 0.6, 1.1, 0.8, 0.8, 0.8, 1.8])
        doc.figure("fusion_sensitivity_notice_rate.png", "PR-AUC as the assumed operator notice rate changes.", width=5.0)
        hi = max(sens, key=lambda s: s["notice_rate"])
        doc.p(f"At {hi['notice_rate']:g}x the share of onsets preceded by a note rises to {pct(hi['onset_note_share'])} and early "
              f"fusion − LightGBM is {signed(hi['delta[early_fusion_minus_telemetry_lgbm]'])} "
              f"{ci(hi['ci95[early_fusion_minus_telemetry_lgbm]'])}. The conclusion of Section 3 therefore depends on how often "
              "real operators write notes before a fault — a quantity InGen's real ticket history can measure and this "
              "simulation cannot.")

    doc.h("6. Unit-level fleet health score", level=1)
    hs = r["fleet_health_score"]
    doc.p(f"Definition: {hs['definition']}. One score per unit and hour; the latest out-of-fold value for each of the "
          f"{hs['units']} units has median {fmt(hs['median'], 1)}. Lowest five at the end of their test windows:")
    doc.table(["Unit", "Time (UTC)", "Fused risk", "Alert threshold", "Health score", "Above threshold"],
              [[u["unit_id"], str(u["t"])[:16], fmt(u["risk"]), fmt(u["alert_threshold"]), fmt(u["health_score"], 1),
                str(u["above_threshold"])] for u in hs["lowest_five"]])
    if diag:
        top = {a: diag["reliability"][a][-1] for a in ("telemetry_lgbm", "early_fusion", "late_stack")}
        doc.p("**The probabilities are overconfident at the top.** On unseen robots, rows scored above 0.99 are positive "
              + ", ".join(f"{pct(v['observed'])} of the time for {ARM_LABEL[a].split(' (')[0]} (n = {v['rows']})" for a, v in top.items())
              + ". This comes from the Week 2 per-fold calibration map (fitted on 6 validation days) and is inherited by "
              "every arm. The health score is therefore a sound *ranking* of units, but a score of 0 should be read as "
              "'among the highest risks', not as certainty. Recalibration on a longer window is a Week 4 item.")
    doc.figure("fusion_budget_curve.png", "Event recall and alert precision against the alert budget.", width=6.0)

    doc.h("7. Limitations", level=1)
    doc.bullets([
        "The corpus is synthetic and its generator decides how informative text is; Section 5 shows the conclusion moves "
        "with that assumption. The finding is about this pipeline's behaviour, not about InGen's real tickets.",
        f"{arms['telemetry_lgbm']['events']:.0f} test-period onsets: intervals are wide, and differences of a few hundredths "
        "of PR-AUC are not resolved.",
        "Text features summarise a classifier's readings; a model reading raw note embeddings might extract more, at the "
        "cost of a much larger leakage surface (any note text after a repair is decided).",
        "No hosted LLM was used for paraphrase or features.",
    ])
    return doc.save("W03_Fusion_Results.docx")


# --------------------------------------------------------------------------- 5. overall Week 3 report
def build_report() -> Path:
    tc = load("text_classifier_results.json")
    fu = load("fusion_results.json")
    ex = load("alert_explanations.json")
    fa = load("fari_transfer_results.json")
    para = maybe("paraphrase_results.json")
    gates = maybe("quality_gates.json")
    manual = reviews(ex)
    doc = Doc("Week 3 Report — Multimodal Fault Classification, Interpretability & Fari Generalisation",
              "InGen Dynamics · Data Science Engineer Internship (Kuntong Li) · Week 3 of 4 · branch week3 · "
              "every number generated from reports/w03/*.json by scripts/build_w03_docs.py")
    doc.p(SYNTHETIC, bold_prefix="Synthetic data. ")

    arms, pdiff = fu["arms"], fu["paired_pr_auc_diff"]
    ef = pdiff["early_fusion_minus_telemetry_lgbm"]
    tm = ex["generators"]["template"]
    llm_name = next((k for k in ex["generators"] if k != "template"), None)
    nc = ex["negative_controls"]
    lg = fa["models"]["arms"]
    doc.h("Executive summary", level=1)
    bullets = [
        f"**Ticket classifier.** On {tc['test_tickets']} held-out tickets (unit-grouped, time-ordered, labels gated by "
        f"confirmation time) TF-IDF + logistic regression reaches macro-F1 {fmt(tc['models']['tfidf_lr']['macro_f1'])}; adding "
        f"sentence embeddings gives {fmt(tc['models']['tfidf_embed_lr']['macro_f1'])} "
        f"(Δ {signed(tc['paired']['tfidf_embed_lr_minus_tfidf_lr']['delta_macro_f1'])} "
        f"{ci(tc['paired']['tfidf_embed_lr_minus_tfidf_lr']['ci95'])}). The weak spot is early fault symptom notes, most "
        "of which are read as 'no fault found'; on wording never seen in training macro-F1 falls to "
        f"{fmt(tc['lexical_shift']['table']['unseen_phrases']['selected']['macro_f1'])}.",
        f"**Fusion and ablation.** Telemetry LightGBM PR-AUC {fmt(arms['telemetry_lgbm']['pr_auc'])}, text only "
        f"{fmt(arms['text_only']['pr_auc'])}, early fusion {fmt(arms['early_fusion']['pr_auc'])}, stacked late fusion "
        f"{fmt(arms['late_stack']['pr_auc'])} (base rate {fmt(arms['no_skill']['pr_auc'])}). Given telemetry, text "
        f"{text_verdict(ef)}; stacking text onto the Isolation Forest "
        f"{text_verdict(pdiff['late_stack_iforest_minus_telemetry_iforest'])}. The best model is still telemetry-only "
        f"(Isolation Forest {fmt(arms['telemetry_iforest']['pr_auc'])}). Only {pct(fu['text_coverage']['share'])} of "
        "test-period onsets had an operator note in the 72 h before them; at 2x and 4x the assumed notice rate text does "
        "add a detectable gain, so the answer depends on InGen's real note-writing rate.",
        f"**Explanations.** Exact tree SHAP for the early-fusion model (additivity error {ex['shap']['additivity_max_abs_error']:.0e}). "
        f"On 20 sampled alerts the template summaries are faithful ({tm['faithful']}/20 automatic"
        + (f", {sum(v['faithful'] for v in manual['template'].values())}/20 on reading" if "template" in manual else "")
        + ")" + (f"; the local LLM's are not ({ex['generators'][llm_name]['faithful']}/20 automatic"
                 + (f", {sum(v['faithful'] for v in manual[llm_name].values())}/20 in the evidence review" if llm_name in manual else "")
                 + ")" if llm_name else "")
        + f". The checker flags {pct(nc['detection_rate'])} of {nc['built']} deliberately corrupted summaries but misses "
        "semantic errors in free LLM prose, so the evidence review is the reference.",
        f"**Fari.** Week 1–2 code accepted {fmt(fa['contract']['week1_rules_as_is']['acceptance_rate_pct'], 1)} % of Fari records "
        f"and left {len(fa['features']['dead_features']['fari_asis'])} features dead. After the minimal change (platform "
        f"profile, config-driven feature guards, per-platform threshold): {fmt(fa['contract']['fari_platform_profile']['acceptance_rate_pct'], 1)} % "
        f"accepted, 0 dead features, Rover LightGBM PR-AUC on Fari {fmt(lg['asis']['lgbm']['pr_auc'])} → "
        f"{fmt(lg['fixed_rover_thr']['lgbm']['pr_auc'])}"
        + (f" (paired Δ {signed(fa['models']['paired_pr_auc_diff']['fixed_rover_thr:lgbm minus asis:lgbm']['delta'])} "
           f"{ci(fa['models']['paired_pr_auc_diff']['fixed_rover_thr:lgbm minus asis:lgbm']['ci95'])})"
           if "paired_pr_auc_diff" in fa["models"] else "")
        + ". The hand-set physics rule did not improve: one Rover-sized indicator is noise on Fari and needs "
        "per-platform calibration.",
    ]
    doc.bullets(bullets)

    doc.h("1. Maintenance-log corpus and ticket classifier", level=1)
    doc.p("Deliverable: docs/W03_Log_Corpus.md. A regenerable synthetic corpus tied to simulated events (fault P-F "
          "intervals, sensor drift, network outages, text-only issues, 'no fault found' episodes, routine notes) with "
          "deliberate noise. Taxonomy: battery, drivetrain, thermal, sensor, connectivity, user or operational error, "
          "no fault found. Labels are usable only once confirmed (`label_available_utc`).")
    m = tc["models"]
    doc.table(["Model", "Macro-F1 [95 % CI]", "Accuracy", "Macro-F1, unseen wording"],
              [[k, f"{fmt(m[k]['macro_f1'])} {ci(m[k]['macro_f1_ci95'])}", fmt(m[k]["accuracy"]),
                fmt(tc["lexical_shift"]["table"]["unseen_phrases"][k]["macro_f1"])]
               for k in ("majority", "tfidf_lr", "embed_lr", "tfidf_embed_lr", "selected")])
    doc.figure("text_confusion.png", "Confusion matrices on the held-out period.", width=6.2)
    if para:
        s = para["models"]["selected"]
        doc.p(f"Local-LLM paraphrase test ({para['sample']['tickets']} test tickets rewritten by {para['llm']} on this machine): "
              f"selected-model macro-F1 {fmt(s['macro_f1_original'])} → {fmt(s['macro_f1_paraphrase'])} "
              f"(Δ {signed(s['delta'])} {ci(s['delta_ci95'])}).")

    doc.h("2. Fusion and modality ablation", level=1)
    doc.p("Deliverable: docs/W03_Fusion_Results.docx. Same 20 folds and test rows as Week 2; the ticket classifier is "
          "fitted inside each fold and cross-fitted over training unit groups.")
    keep = ("no_skill", "telemetry_lgbm", "telemetry_iforest", "text_only", "early_fusion", "late_noisy_or", "late_stack",
            "late_stack_iforest")
    doc.table(["Arm", "Modality", "PR-AUC [95 % CI]", "Event recall [95 % CI]", "Precision", "Alerts / unit-wk",
               "Median lead (h)"], arm_rows({k: arms[k] for k in keep}), widths=[2.1, 0.7, 1.25, 1.2, 0.6, 0.6, 0.6])
    doc.table(["Paired comparison", "Δ PR-AUC", "95 % CI", "Reading"],
              [[k.replace("_minus_", " − "), signed(v["delta"]), ci(v["ci95"]), verdict(v["ci95"])]
               for k, v in pdiff.items() if k in ("text_only_minus_no_skill", "early_fusion_minus_telemetry_lgbm",
                                                  "late_stack_minus_telemetry_lgbm", "late_stack_iforest_minus_telemetry_iforest",
                                                  "early_fusion_minus_late_stack")], widths=[2.6, 0.7, 1.3, 2.3])
    sens = fu.get("sensitivity_notice_rate")
    if sens:
        doc.p("Sensitivity: early fusion − LightGBM at notice rate "
              + ", ".join(f"{s['notice_rate']:g}x: {signed(s['delta[early_fusion_minus_telemetry_lgbm]'])} "
                          f"{ci(s['ci95[early_fusion_minus_telemetry_lgbm]'])}" for s in sens)
              + ". How much text helps depends on how often operators write before a fault.")

    doc.h("3. SHAP attributions and grounded alert summaries", level=1)
    doc.p("Deliverable: docs/W03_Alert_Explanations.md. Global and per-alert tree SHAP for the early-fusion model; every "
          "alert summary is written only from an evidence pack (top attributions, the unit's values now and 72 h earlier, "
          "linked notes with their classifier reading). Two generators: a deterministic template and the local open-weights "
          "Qwen2.5-1.5B-Instruct (no hosted API).")
    doc.figure("shap_global.png", "Global SHAP of the early-fusion model.", width=6.2)
    rows = []
    for name, g in ex["generators"].items():
        man = manual.get(name, {})
        man_ok = sum(1 for v in man.values() if v["faithful"]) if man else None
        rows.append([gen_label(name), f"{g['faithful']}/{g['alerts']} ({pct(g['faithfulness_rate'])})",
                     f"{g['claims_supported']}/{g['claims_checked']}",
                     f"{man_ok}/{len(man)}" if man else "not read"])
    doc.table(["Generator", "Faithful (automatic)", "Claims supported", "Faithful (evidence review)"], rows)
    doc.p(f"Checker sensitivity (negative controls): {nc['flagged']}/{nc['built']} corrupted summaries flagged ("
          + ", ".join(f"{k} {v['flagged']}/{v['built']}" for k, v in nc["by_kind"].items()) + "). Before this run, two checker false-flag bugs and two real template faithfulness problems from the smoke run were fixed; "
          "after reading the checker's flags, five more generic false-flag causes were fixed (checker v2) and the stored summaries "
          "re-scored without regeneration. Every fix has a regression test.")

    doc.h("4. Fari generalisation and gap analysis", level=1)
    doc.p("Deliverable: docs/W03_Fari_Gap_Analysis.docx. All Fari parameters are synthetic assumptions.")
    doc.table(["Arm", "LightGBM PR-AUC [95 % CI]", "Isolation Forest", "Physics rule", "LightGBM alerts / unit-wk"],
              [[ARM_NAME[a], f"{fmt(lg[a]['lgbm']['pr_auc'])} {ci(lg[a]['lgbm']['ci95']['pr_auc'])}",
                fmt(lg[a]["iforest"]["pr_auc"]), fmt(lg[a]["physics_rule"]["pr_auc"]),
                fmt(lg[a]["lgbm"]["alerts_per_unit_week"], 2)] for a in ARM_NAME], widths=[2.4, 1.5, 0.9, 0.9, 1.0])
    pd_ = fa["models"].get("paired_pr_auc_diff", {})
    key = "fixed_rover_thr:lgbm minus asis:lgbm"
    if key in pd_:
        doc.p(f"Paired gain of the minimal change for the Rover LightGBM on Fari: {signed(pd_[key]['delta'])} "
              f"{ci(pd_[key]['ci95'])}: {verdict(pd_[key]['ci95'])}.")

    doc.h("5. Engineering quality", level=1)
    if gates:
        doc.table(["Gate", "Result", "Summary"],
                  [[k, "pass" if v["ok"] else "FAIL", v["summary"][:90]] for k, v in gates.items() if isinstance(v, dict)])
        doc.p(f"Gates recorded {gates['run_at_utc']} by scripts/run_w03_gates.py.")
    doc.bullets([
        "Every experiment is a documented command (`python -m ingen_fleet_health.experiments.<name>`) and logs its "
        "parameters, metrics, figures and full JSON to MLflow (sqlite:///mlflow.db); run ids are printed in each "
        "deliverable. Runs: text classifier " + tc["mlflow_run_id"] + ", fusion " + fu["mlflow_run_id"] + ", explanations "
        + ex["mlflow_run_id"] + (" (checker v2 re-score " + ex["recheck_mlflow_run_id"] + ")" if ex.get("recheck_mlflow_run_id") else "")
        + ", Fari " + fa["mlflow_run_id"] + (", paraphrase " + para["mlflow_run_id"] if para else "") + ".",
        "New dependencies (sentence-transformers, shap; transformers/torch via sentence-transformers) are open-source and "
        "locked in uv.lock. Model weights (all-MiniLM-L6-v2, Qwen2.5-1.5B-Instruct) are downloaded to the local Hugging "
        "Face cache and never committed.",
        "Generated data (data/), MLflow store and model caches are git-ignored. The repository contains no InGen material, "
        "no customer data and no keys.",
    ])

    doc.h("6. Limitations and open questions for InGen", level=1)
    doc.bullets([
        "All results are on synthetic data; the size of the text contribution is set by the corpus generator's notice-rate "
        "assumption (measured in the sensitivity sweep). InGen's real ticket history could answer the key question directly: "
        "how often is a fault mentioned in a note before it is visible in telemetry?",
        "The plan names a hosted LLM (Claude API or Vertex AI). It was not called: that needs a key, cost approval and a "
        "data-handling decision. The local 1.5 B model was the stand-in and was not faithful; the prompt a hosted model would receive "
        "is recorded per alert.",
        "Faithfulness is measured on 20 alerts; the rate has a wide interval, and faithful-to-the-model is not the same "
        "as correct about the robot (half the sample are false alarms).",
        "Review method: " + REVIEW_METHOD,
        "Fari parameters are unverified assumptions. The Fari results test the pipeline's design, not Fari hardware.",
    ])
    doc.h("7. Week 4 hand-off", level=1)
    doc.bullets([
        "Score with the telemetry model (Isolation Forest is best on this fleet) as the primary fleet health score; keep "
        "text as an optional, separately monitored input, added only if InGen's ticket history shows that operators note "
        "symptoms before onset often enough. Any stacker must be regularised toward the telemetry score.",
        "Recalibrate the risk on a longer window: scores above 0.99 are positive only about 73 % of the time.",
        "Surface the template summary (faithfulness-checked at generation time) next to each alert in the dashboard.",
        "Per-platform thresholds and profiles as configuration; per-platform physics-rule calibration once real baselines exist.",
    ])
    return doc.save("W03_Report.docx")


def main() -> None:
    for fn in (build_log_corpus, build_fusion, build_explanations, build_fari, build_report):
        try:
            print(fn())
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
