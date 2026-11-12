"""Train and evaluate the ticket classifiers under the Week 2 split discipline.

    python -m ingen_fleet_health.experiments.text_classifier_run --data-dir data/w02

For every cut-off c and test unit group g (the same 20 folds as Week 2):

* training tickets: units not in g whose label was confirmed by c - 72 h;
* test tickets: units in g, created in [c, c + 10 days).

Two variants are run. ``main`` uses all training tickets. ``lexical_shift``
removes every training ticket built from a held-out core phrase, so test
tickets with those phrases measure generalisation to wording never seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ingen_fleet_health import plotstyle
from ingen_fleet_health.experiments.common import load_w02
from ingen_fleet_health.experiments.w03_common import (
    CUTOFF_DAYS,
    TEST_SPAN,
    check_text_fold,
    figure_path,
    load_text,
    text_test_mask,
    text_train_mask,
    write_json,
)
from ingen_fleet_health.models.text_classifier import (
    C_GRID,
    EMBEDDING_MODEL,
    KINDS,
    crossfit_fold,
    macro_f1,
)
from ingen_fleet_health.text import TAXONOMY
from ingen_fleet_health.text.corpus import load_corpus_truth
from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

EMBARGO = pd.Timedelta(hours=72)
MODELS = ("majority", *KINDS, "selected")
SHORT = {"battery": "battery", "drivetrain": "drivetrain", "thermal": "thermal", "sensor": "sensor",
         "connectivity": "connect.", "user_or_operational_error": "user/op", "no_fault_found": "NFF"}


def out_of_fold(data_dir: Path, variant: str, seed: int, reuse: bool = False,
                ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Out-of-fold predictions for one variant. They are written to disk as
    soon as they exist; ``reuse`` reloads them instead of refitting (~25 min)."""
    oof_path = data_dir / "predictions" / f"text_classifier_oof_{variant}.parquet"
    info_path = data_dir / "predictions" / f"text_classifier_folds_{variant}.json"
    if reuse and oof_path.exists() and info_path.exists():
        return pd.read_parquet(oof_path), json.loads(info_path.read_text(encoding="utf-8"))
    data = load_w02(data_dir)
    text = load_text(data_dir, data.groups)
    truth = load_corpus_truth(data_dir).set_index("ticket_id").reindex(text.corpus["ticket_id"])
    unseen = truth["unseen_phrase"].astype(str).str.lower().eq("true").to_numpy()
    parts, info = [], []
    for day, cutoff in zip(CUTOFF_DAYS, data.cutoffs(CUTOFF_DAYS), strict=True):
        for g in sorted(set(data.groups.values())):
            train = text_train_mask(text, g, cutoff - EMBARGO)
            if variant == "lexical_shift":
                train &= ~unseen
            test = text_test_mask(text, g, cutoff, TEST_SPAN)
            check_text_fold(text, train, test, EMBARGO)
            fs = crossfit_fold(text.inputs, text.labels, text.group, train, g, seed=seed)
            frame = text.corpus.loc[test, ["ticket_id", "unit_id", "created_at_utc", "doc_type", "text", "label"]].copy()
            frame["fold_id"] = f"c{cutoff:%m%d}_g{g}"
            frame["cutoff_day"] = day
            for kind in (*KINDS, "majority"):
                frame[f"pred_{kind}"] = np.array(TAXONOMY)[np.argmax(fs.proba[kind][test], axis=1)]
            frame["pred_selected"] = frame[f"pred_{fs.selected_kind}"]
            parts.append(frame)
            info.append({"fold_id": frame["fold_id"].iloc[0] if len(frame) else f"c{cutoff:%m%d}_g{g}",
                         "cutoff_day": day, "group": g, "train_tickets": fs.n_train, "test_tickets": int(test.sum()),
                         "train_classes": len(set(text.labels[train])), "chosen_c": fs.chosen_c,
                         "cv_macro_f1": fs.cv_macro_f1, "selected_kind": fs.selected_kind})
    oof = pd.concat(parts, ignore_index=True)
    oof = oof.merge(truth.reset_index(), on="ticket_id", how="left")
    oof["unseen_phrase"] = oof["unseen_phrase"].astype(str).str.lower().eq("true")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_path, index=False)
    info_path.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return oof, info


def _unit_bootstrap(frame: pd.DataFrame, stat: Any, n_boot: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx_by_unit = [np.asarray(g) for g in frame.groupby("unit_id").indices.values()]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_unit), len(idx_by_unit))
        vals.append(stat(frame.iloc[np.concatenate([idx_by_unit[i] for i in pick])]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def model_metrics(oof: pd.DataFrame, model: str, seed: int) -> dict[str, Any]:
    y, p = oof["label"].to_numpy(), oof[f"pred_{model}"].to_numpy()
    prec, rec, f1, support = precision_recall_fscore_support(y, p, labels=list(TAXONOMY), zero_division=0)
    return {
        "macro_f1": macro_f1(y, p),
        "accuracy": float((y == p).mean()),
        "macro_f1_ci95": _unit_bootstrap(oof, lambda f: macro_f1(f["label"].to_numpy(), f[f"pred_{model}"].to_numpy()),
                                         seed=seed),
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                          "support": int(support[i])} for i, c in enumerate(TAXONOMY)},
        "confusion_matrix": confusion_matrix(y, p, labels=list(TAXONOMY)).tolist(),
        "by_cutoff_day": {str(d): macro_f1(g["label"].to_numpy(), g[f"pred_{model}"].to_numpy())
                          for d, g in oof.groupby("cutoff_day")},
    }


def paired_diff(oof: pd.DataFrame, a: str, b: str, seed: int) -> dict[str, Any]:
    def diff(f: pd.DataFrame) -> float:
        y = f["label"].to_numpy()
        return macro_f1(y, f[f"pred_{a}"].to_numpy()) - macro_f1(y, f[f"pred_{b}"].to_numpy())

    return {"delta_macro_f1": diff(oof), "ci95": _unit_bootstrap(oof, diff, seed=seed)}


def error_analysis(oof: pd.DataFrame, model: str) -> dict[str, Any]:
    wrong = oof[oof["label"] != oof[f"pred_{model}"]]
    pairs = wrong.groupby(["label", f"pred_{model}"]).size().sort_values(ascending=False).head(8)
    confusions = []
    for (true, pred), n in pairs.items():
        ex = wrong[(wrong["label"] == true) & (wrong[f"pred_{model}"] == pred)]
        confusions.append({
            "true": true, "predicted": pred, "count": int(n),
            "by_source": {k: int(v) for k, v in ex["source"].value_counts().items()},
            "by_style": {k: int(v) for k, v in ex["style"].value_counts().items()},
            "examples": ex["text"].head(3).tolist(),
        })

    def breakdown(col: str) -> dict[str, dict[str, float]]:
        return {str(k): {"n": len(g), "accuracy": float((g["label"] == g[f"pred_{model}"]).mean())}
                for k, g in oof.groupby(col)}

    typo_bucket = oof["n_typos"].clip(upper=2).map({0: "0 typos", 1: "1 typo", 2: "2+ typos"})
    return {
        "model": model,
        "errors": len(wrong),
        "tickets": len(oof),
        "top_confusions": confusions,
        "accuracy_by_style": breakdown("style"),
        "accuracy_by_source": breakdown("source"),
        "accuracy_by_typos": {str(k): {"n": len(g), "accuracy": float((g["label"] == g[f"pred_{model}"]).mean())}
                              for k, g in oof.groupby(typo_bucket)},
        "accuracy_by_doc_type": breakdown("doc_type"),
    }


def lexical_shift_table(oof: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, sub in (("seen_phrases", oof[~oof["unseen_phrase"]]), ("unseen_phrases", oof[oof["unseen_phrase"]])):
        out[name] = {"tickets": len(sub),
                     **{m: {"accuracy": float((sub["label"] == sub[f"pred_{m}"]).mean()),
                            "macro_f1": macro_f1(sub["label"].to_numpy(), sub[f"pred_{m}"].to_numpy())}
                        for m in MODELS}}
    return out


def confusion_figure(results: dict[str, Any], left: str, right: str) -> Path:
    plotstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    labels = [SHORT[c] for c in TAXONOMY]
    for ax, model in zip(axes, (left, right), strict=True):
        cm = np.array(results["models"][model]["confusion_matrix"], dtype=float)
        row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        ax.imshow(row, cmap=plotstyle.SEQUENTIAL, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                            color="#ffffff" if row[i, j] > 0.55 else plotstyle.TEXT)
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted category")
        ax.grid(False)
        ax.set_title(f"{model}: macro-F1 {results['models'][model]['macro_f1']:.3f}")
    axes[0].set_ylabel("True category")
    fig.suptitle("Ticket classification on unseen robots and later dates (counts; shade = share of the true class)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    path = figure_path("text_confusion.png")
    plotstyle.save(fig, path)
    return path


def per_class_figure(results: dict[str, Any]) -> Path:
    plotstyle.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    kinds = list(KINDS)
    width = 0.26
    x = np.arange(len(TAXONOMY))
    for k, kind in enumerate(kinds):
        vals = [results["models"][kind]["per_class"][c]["f1"] for c in TAXONOMY]
        ax.bar(x + (k - 1) * width, vals, width * 0.92, color=plotstyle.SERIES[k], label=kind)
    for i, c in enumerate(TAXONOMY):
        n = results["models"][kinds[0]]["per_class"][c]["support"]
        ax.text(i, 1.02, f"n={n}", ha="center", fontsize=8, color=plotstyle.TEXT_SECONDARY)
    ax.set_xticks(x, [SHORT[c] for c in TAXONOMY])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 on pooled test folds")
    ax.set_title("Per-class F1: the fault classes and sensor are the hard ones")
    ax.legend(ncol=3, loc="lower left")
    ax.grid(axis="x", visible=False)
    path = figure_path("text_per_class_f1.png")
    plotstyle.save(fig, path)
    return path


def cutoff_figure(results: dict[str, Any], fold_info: list[dict[str, Any]]) -> Path:
    plotstyle.apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    train = pd.DataFrame(fold_info).groupby("cutoff_day")["train_tickets"].mean()
    for k, kind in enumerate(KINDS):
        by = results["models"][kind]["by_cutoff_day"]
        days = [int(d) for d in by]
        ax.plot(days, list(by.values()), marker="o", color=plotstyle.SERIES[k], label=kind)
    ax.axhline(results["models"]["majority"]["macro_f1"], color=plotstyle.NEUTRAL, lw=1, ls="--")
    ax.text(CUTOFF_DAYS[-1], results["models"]["majority"]["macro_f1"] + 0.02, "majority class", ha="right",
            fontsize=8, color=plotstyle.TEXT_SECONDARY)
    ax.set_xticks(CUTOFF_DAYS, [f"day {d}\n~{train.loc[d]:.0f} train tickets" for d in CUTOFF_DAYS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 (test window after cut-off)")
    ax.set_title("Macro-F1 by walk-forward cut-off")
    ax.legend(loc="lower right")
    path = figure_path("text_macro_f1_by_cutoff.png")
    plotstyle.save(fig, path)
    return path


def run(data_dir: Path, seed: int = 0, reuse: bool = False) -> dict[str, Any]:
    oof, fold_info = out_of_fold(data_dir, "main", seed, reuse)
    shift_oof, shift_info = out_of_fold(data_dir, "lexical_shift", seed, reuse)
    results: dict[str, Any] = {
        "protocol": {"cutoff_days": CUTOFF_DAYS, "test_span_days": 10, "embargo_hours": 72,
                     "unit_groups": 5, "folds": len(fold_info),
                     "train_rule": "units not in the test group and label_available_utc <= cutoff - 72 h",
                     "test_rule": "units in the test group and cutoff <= created_at_utc < cutoff + 10 d"},
        "embedding_model": EMBEDDING_MODEL,
        "c_grid": list(C_GRID),
        "test_tickets": len(oof),
        "test_label_counts": {c: int((oof["label"] == c).sum()) for c in TAXONOMY},
        "models": {m: model_metrics(oof, m, seed) for m in MODELS},
        "paired": {
            "embed_lr_minus_tfidf_lr": paired_diff(oof, "embed_lr", "tfidf_lr", seed),
            "tfidf_embed_lr_minus_tfidf_lr": paired_diff(oof, "tfidf_embed_lr", "tfidf_lr", seed),
            "tfidf_lr_minus_majority": paired_diff(oof, "tfidf_lr", "majority", seed),
        },
        "fold_info": fold_info,
        "selected_kind_counts": {k: int(v) for k, v in pd.Series([f["selected_kind"] for f in fold_info]).value_counts().items()},
        "lexical_shift": {
            "held_out_rule": "training tickets built from held-out core phrases are removed",
            "table": lexical_shift_table(shift_oof),
            "paired_unseen_embed_minus_tfidf": paired_diff(shift_oof[shift_oof["unseen_phrase"]], "embed_lr",
                                                           "tfidf_lr", seed),
            "fold_train_tickets_mean": float(np.mean([f["train_tickets"] for f in shift_info])),
        },
    }
    best = max(KINDS, key=lambda k: results["models"][k]["macro_f1"])
    results["best_single_model_on_test"] = best
    results["error_analysis"] = {m: error_analysis(oof, m) for m in ("tfidf_lr", "embed_lr", best)}

    figs = [confusion_figure(results, "tfidf_lr", "embed_lr"), per_class_figure(results),
            cutoff_figure(results, fold_info)]

    with tracked_run("w03_text_classifier", "ticket_classifiers", {"track": "text", "data_dir": str(data_dir)}) as run_:
        import mlflow

        mlflow.log_params({"embedding_model": EMBEDDING_MODEL, "c_grid": str(list(C_GRID)), "folds": len(fold_info),
                           "embargo_hours": 72, "test_tickets": len(oof), "seed": seed})
        for m in MODELS:
            log_flat_metrics({"macro_f1": results["models"][m]["macro_f1"],
                              "accuracy": results["models"][m]["accuracy"]}, prefix=f"{m}.")
            log_flat_metrics({c: v["f1"] for c, v in results["models"][m]["per_class"].items()}, prefix=f"{m}.f1.")
        for name, table in results["lexical_shift"]["table"].items():
            for m in MODELS:
                log_flat_metrics(table[m], prefix=f"lexical_shift.{name}.{m}.")
        for fig in figs:
            mlflow.log_artifact(str(fig))
        log_dict_json(results, "text_classifier_results.json")
        results["mlflow_run_id"] = run_.info.run_id
    write_json(results, "text_classifier_results.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse-oof", action="store_true",
                        help="reuse saved out-of-fold predictions instead of refitting")
    args = parser.parse_args()
    res = run(args.data_dir, args.seed, args.reuse_oof)
    for m in MODELS:
        v = res["models"][m]
        print(f"{m:16s} macro-F1 {v['macro_f1']:.3f} {v['macro_f1_ci95']}  accuracy {v['accuracy']:.3f}")
    for k, v in res["paired"].items():
        print(k, round(v["delta_macro_f1"], 3), v["ci95"])
    print("lexical shift:", res["lexical_shift"]["table"])
    print("MLflow run:", res["mlflow_run_id"])


if __name__ == "__main__":
    main()
