"""Local-LLM paraphrase robustness test for the ticket classifiers (Week 3).

    python -m ingen_fleet_health.experiments.paraphrase_run --data-dir data/w02 [--per-class 30]

Plan v3.0 asks for "templated generation plus LLM-assisted paraphrase". The
corpus itself is not LLM-paraphrased, so it stays regenerable from a seed
without a 3 GB model. Instead a stratified, seeded sample of *test* tickets is
rewritten by a local open-weights model (``Qwen/Qwen2.5-1.5B-Instruct``,
Apache-2.0, run on this machine; no API, nothing leaves the machine) and the
fold classifiers that scored the original ticket score the rewrite too. The
drop in macro-F1 measures how much the classifiers depend on template wording.

The labels of the rewrites are the labels of the originals. A rewrite can
change the meaning; ``manual_check`` in the result is filled in by reading the
pairs (``docs/W03_Log_Corpus.md``), not by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.experiments.common import load_w02
from ingen_fleet_health.experiments.text_classifier_run import EMBARGO, _unit_bootstrap
from ingen_fleet_health.experiments.w03_common import (
    CUTOFF_DAYS,
    load_text,
    text_train_mask,
    write_json,
)
from ingen_fleet_health.explain.alert_summaries import LOCAL_LLM
from ingen_fleet_health.models.text_classifier import (
    KINDS,
    TicketClassifier,
    TicketInputs,
    embed_texts,
    macro_f1,
)
from ingen_fleet_health.text import TAXONOMY
from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

SYSTEM = (
    "You rewrite short maintenance notes written by robot operators and service technicians. Rewrite the note in "
    "different words, the way a different person might write it. Keep exactly the same meaning and level of detail. "
    "Do not add causes, parts, numbers or advice that are not in the note. Output only the rewritten note."
)


def sample_tickets(oof: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for cls in TAXONOMY:
        g = oof[oof["label"] == cls]
        parts.append(g.iloc[np.sort(rng.choice(len(g), min(per_class, len(g)), replace=False))])
    return pd.concat(parts).reset_index(drop=True)


def paraphrase(texts: list[str], cache: Path, model_name: str = LOCAL_LLM) -> list[str]:
    """Greedy rewrites; cached on disk (keyed by the original text)."""
    done: dict[str, str] = {}
    if cache.exists():
        done = dict(pd.read_csv(cache, keep_default_na=False)[["original", "paraphrase"]].itertuples(index=False))
    todo = [t for t in dict.fromkeys(texts) if t not in done]
    if todo:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok: Any = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
        for i, text in enumerate(todo):
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False, repetition_penalty=1.05,
                                     pad_token_id=tok.eos_token_id)
            rewrite = " ".join(str(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)).split())
            done[text] = rewrite.strip().strip('"')
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                cache.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"original": list(done), "paraphrase": list(done.values())}).to_csv(cache, index=False)
                print(f"paraphrased {i + 1}/{len(todo)}", flush=True)
    return [done[t] for t in texts]


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def run(data_dir: Path, per_class_max: int = 30, seed: int = 0) -> dict[str, Any]:
    data = load_w02(data_dir)
    text = load_text(data_dir, data.groups)
    oof = pd.read_parquet(data_dir / "predictions" / "text_classifier_oof_main.parquet")
    folds = {f["fold_id"]: f for f in json.loads(
        (data_dir / "predictions" / "text_classifier_folds_main.json").read_text(encoding="utf-8"))}
    sample = sample_tickets(oof, per_class_max, seed)
    sample["paraphrase"] = paraphrase(sample["text"].astype(str).tolist(), data_dir / "text" / "paraphrase_qwen.csv")
    para_emb = embed_texts(sample["paraphrase"].tolist(), cache_path=data_dir / "text" / "embeddings_paraphrase.npy")
    pos = pd.Series(np.arange(len(text.corpus)), index=text.corpus["ticket_id"].astype(str))
    cutoffs = dict(zip(CUTOFF_DAYS, data.cutoffs(CUTOFF_DAYS), strict=True))

    reproduced = 0
    for fold_id, rows in sample.groupby("fold_id").groups.items():
        info = folds[str(fold_id)]
        train = text_train_mask(text, int(info["group"]), cutoffs[int(info["cutoff_day"])] - EMBARGO)
        idx = np.flatnonzero(train)
        orig = text.inputs.take(pos.loc[sample.loc[rows, "ticket_id"].astype(str)].to_numpy())
        para = TicketInputs(sample.loc[rows, "paraphrase"].to_numpy(dtype=object), para_emb[np.asarray(rows)])
        for kind in KINDS:
            model = TicketClassifier(kind, float(info["chosen_c"][kind]), seed).fit(text.inputs.take(idx), text.labels[idx])
            sample.loc[rows, f"orig_{kind}"] = model.predict(orig)
            sample.loc[rows, f"para_{kind}"] = model.predict(para)
        sel = info["selected_kind"]
        sample.loc[rows, "orig_selected"] = sample.loc[rows, f"orig_{sel}"]
        sample.loc[rows, "para_selected"] = sample.loc[rows, f"para_{sel}"]
        reproduced += int((sample.loc[rows, f"orig_{sel}"] == sample.loc[rows, "pred_selected"]).sum())

    y = sample["label"].to_numpy()
    models: dict[str, Any] = {}
    for kind in (*KINDS, "selected"):
        def delta(f: pd.DataFrame, k: str = kind) -> float:
            yy = f["label"].to_numpy()
            return macro_f1(yy, f[f"para_{k}"].to_numpy()) - macro_f1(yy, f[f"orig_{k}"].to_numpy())

        models[kind] = {
            "macro_f1_original": macro_f1(y, sample[f"orig_{kind}"].to_numpy()),
            "macro_f1_paraphrase": macro_f1(y, sample[f"para_{kind}"].to_numpy()),
            "delta": delta(sample),
            "delta_ci95": _unit_bootstrap(sample, delta, seed=seed),
            "prediction_unchanged_share": float((sample[f"orig_{kind}"] == sample[f"para_{kind}"]).mean()),
        }
    per_class = {c: {"n": int((y == c).sum()),
                     "recall_original": float((sample.loc[y == c, "orig_selected"] == c).mean()),
                     "recall_paraphrase": float((sample.loc[y == c, "para_selected"] == c).mean())} for c in TAXONOMY}
    jac = [token_jaccard(a, b) for a, b in zip(sample["text"], sample["paraphrase"], strict=True)]
    flips = sample[(sample["orig_selected"] == sample["label"]) & (sample["para_selected"] != sample["label"])]
    results: dict[str, Any] = {
        "synthetic_notice": "Originals are synthetic template tickets; rewrites come from a local open-weights model. "
                            "No hosted LLM API was called.",
        "llm": LOCAL_LLM, "decoding": "greedy, max 64 new tokens, repetition_penalty 1.05", "system_prompt": SYSTEM,
        "sample": {"tickets": len(sample), "per_class_max": per_class_max, "seed": seed, "folds": int(sample["fold_id"].nunique())},
        "reproduces_oof_selected_predictions": reproduced / len(sample),
        "token_jaccard_original_vs_paraphrase": {"median": float(np.median(jac)), "mean": float(np.mean(jac))},
        "identical_rewrites": int((sample["text"].str.strip().str.lower() == sample["paraphrase"].str.strip().str.lower()).sum()),
        "models": models,
        "per_class_selected": per_class,
        "correct_then_wrong_examples": flips[["ticket_id", "label", "text", "paraphrase", "para_selected"]].head(12)
        .to_dict(orient="records"),
        "pairs_for_manual_check": sample.sample(min(30, len(sample)), random_state=seed)[["ticket_id", "label", "text", "paraphrase"]]
        .to_dict(orient="records"),
    }
    with tracked_run("w03_text_classifier", "paraphrase_robustness", {"track": "text", "data_dir": str(data_dir)}) as run_:
        import mlflow

        mlflow.log_params({"llm": LOCAL_LLM, "per_class": per_class_max, "seed": seed, "tickets": len(sample)})
        for kind, m in models.items():
            log_flat_metrics({k: v for k, v in m.items() if not isinstance(v, list)}, prefix=f"{kind}.")
        log_dict_json(results, "paraphrase_results.json")
        results["mlflow_run_id"] = run_.info.run_id
    write_json(results, "paraphrase_results.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = run(args.data_dir, args.per_class, args.seed)
    for kind, m in res["models"].items():
        print(f"{kind:15s} macro-F1 original {m['macro_f1_original']:.3f} -> paraphrase {m['macro_f1_paraphrase']:.3f} "
              f"(delta {m['delta']:+.3f} {np.round(m['delta_ci95'], 3)}), unchanged {m['prediction_unchanged_share']:.2f}")
    print("reproduces OOF:", res["reproduces_oof_selected_predictions"], "| jaccard:", res["token_jaccard_original_vs_paraphrase"])
    print("MLflow run:", res["mlflow_run_id"])


if __name__ == "__main__":
    main()
