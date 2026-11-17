"""Shared paths, loading and text-fold construction for Week 3 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.config import REPO_ROOT
from ingen_fleet_health.experiments.common import _json_default
from ingen_fleet_health.models.text_classifier import TicketInputs, embed_texts
from ingen_fleet_health.text.corpus import load_corpus

RESULTS_DIR = REPO_ROOT / "reports" / "w03"
FIGURES_DIR = REPO_ROOT / "docs" / "figures" / "w03"
CUTOFF_DAYS = [20, 30, 40, 50]
TEST_SPAN = pd.Timedelta(days=10)
VAL_SPAN = pd.Timedelta(days=6)


def write_json(payload: dict[str, Any], name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def read_json(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
    return data


def figure_path(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


@dataclass
class TextData:
    corpus: pd.DataFrame          # observable columns only
    inputs: TicketInputs
    group: np.ndarray             # unit group of each ticket (same groups as Week 2)

    @property
    def labels(self) -> np.ndarray:
        return self.corpus["label"].to_numpy()


def load_text(data_dir: Path, groups: dict[str, int], subdir: str = "text") -> TextData:
    corpus = load_corpus(data_dir, subdir).sort_values("ticket_id").reset_index(drop=True)
    texts = corpus["text"].astype(str).tolist()
    emb = embed_texts(texts, cache_path=data_dir / subdir / "embeddings_minilm.npy")
    group = corpus["unit_id"].map(groups)
    if group.isna().any():
        raise ValueError("Every ticket's unit must belong to a unit group")
    return TextData(corpus, TicketInputs(np.array(texts, dtype=object), emb), group.to_numpy(dtype=int))


def text_train_mask(text: TextData, test_group: int, train_end: pd.Timestamp) -> np.ndarray:
    """Tickets whose label may be used for a fold: not a test unit, and the
    label was confirmed by ``train_end`` (= cut-off minus the embargo)."""
    return np.asarray((text.group != test_group) & (text.corpus["label_available_utc"] <= train_end).to_numpy())


def text_test_mask(text: TextData, test_group: int, cutoff: pd.Timestamp, span: pd.Timedelta) -> np.ndarray:
    created = text.corpus["created_at_utc"]
    return np.asarray((text.group == test_group) & (created >= cutoff).to_numpy() & (created < cutoff + span).to_numpy())


def check_text_fold(text: TextData, train_mask: np.ndarray, test_mask: np.ndarray, embargo: pd.Timedelta) -> None:
    """Same audit as ``evaluation.splits.check_fold``, for tickets."""
    corpus = text.corpus
    shared = set(corpus.loc[train_mask, "unit_id"]) & set(corpus.loc[test_mask, "unit_id"])
    if shared:
        raise AssertionError(f"units in both text train and test: {sorted(shared)}")
    if train_mask.any() and test_mask.any():
        gap = corpus.loc[test_mask, "created_at_utc"].min() - corpus.loc[train_mask, "label_available_utc"].max()
        if gap < embargo:
            raise AssertionError(f"text embargo {gap} shorter than {embargo}")
