"""Ticket classifiers for the maintenance-log corpus (Week 3).

* ``majority``        -- no-skill reference: always the most frequent training class
* ``tfidf_lr``        -- baseline: word (1-2) + character (2-5) TF-IDF, logistic regression
* ``embed_lr``        -- frozen sentence-transformer embeddings, logistic regression
* ``tfidf_embed_lr``  -- both feature sets side by side, logistic regression

Leakage notes. The sentence encoder is a frozen pretrained model: embedding a
ticket uses nothing but that ticket's own text, so embeddings may be computed
once for the whole corpus. Everything that is *fitted* (TF-IDF vocabulary and
IDF weights, the logistic regression, the choice of C) sees training tickets
only; ``crossfit_fold`` additionally scores every training-unit ticket with a
model that never saw that unit's group.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ingen_fleet_health.text import TAXONOMY

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KINDS = ("tfidf_lr", "embed_lr", "tfidf_embed_lr")
C_GRID = (1.0, 10.0, 100.0, 1000.0)


@dataclass
class TicketInputs:
    """Aligned per-ticket inputs: raw text and its frozen embedding."""

    texts: np.ndarray      # (n,) object
    emb: np.ndarray        # (n, d) float32, L2-normalised

    def take(self, idx: np.ndarray) -> TicketInputs:
        return TicketInputs(self.texts[idx], self.emb[idx])

    def __len__(self) -> int:
        return len(self.texts)


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL, cache_path: Path | None = None) -> np.ndarray:
    """L2-normalised sentence embeddings; cached on disk next to the corpus."""
    import hashlib

    digest = hashlib.sha256(("\n".join(texts) + model_name).encode("utf-8")).hexdigest()[:16]
    if cache_path is not None:
        cache_file = cache_path.with_name(f"{cache_path.stem}_{digest}.npy")
        if cache_file.exists():
            return np.asarray(np.load(cache_file))
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    emb = np.asarray(model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False),
                     dtype=np.float32)
    if cache_path is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, emb)
    return emb


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean F1 over the full taxonomy (absent classes count as 0)."""
    return float(f1_score(y_true, y_pred, labels=list(TAXONOMY), average="macro", zero_division=0))


class TicketClassifier:
    """One of ``KINDS`` with a common fit / predict_proba over ``TicketInputs``.
    ``predict_proba`` always returns columns in ``TAXONOMY`` order."""

    def __init__(self, kind: str, C: float = 10.0, seed: int = 0) -> None:
        if kind not in (*KINDS, "majority"):
            raise ValueError(f"Unknown classifier kind: {kind}")
        self.kind = kind
        self.C = C
        self.seed = seed
        self.word: TfidfVectorizer | None = None
        self.char: TfidfVectorizer | None = None
        self.model: LogisticRegression | None = None
        self.classes_: np.ndarray = np.array([])
        self.prior_: np.ndarray = np.zeros(len(TAXONOMY))

    def _design(self, inputs: TicketInputs, fit: bool) -> Any:
        blocks = []
        if self.kind in ("tfidf_lr", "tfidf_embed_lr"):
            if fit:
                self.word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
                self.char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5),
                                            sublinear_tf=True, min_df=2)
                blocks += [self.word.fit_transform(inputs.texts), self.char.fit_transform(inputs.texts)]
            else:
                assert self.word is not None and self.char is not None
                blocks += [self.word.transform(inputs.texts), self.char.transform(inputs.texts)]
        if self.kind in ("embed_lr", "tfidf_embed_lr"):
            blocks.append(sparse.csr_matrix(inputs.emb.astype(np.float64)))
        return sparse.hstack(blocks, format="csr")

    def fit(self, inputs: TicketInputs, y: np.ndarray) -> TicketClassifier:
        y = np.asarray(y)
        counts = np.array([(y == c).sum() for c in TAXONOMY], dtype=float)
        self.prior_ = counts / max(counts.sum(), 1.0)
        self.classes_ = np.array([c for c, n in zip(TAXONOMY, counts, strict=True) if n > 0])
        if self.kind == "majority" or len(self.classes_) < 2:
            self.model = None
            return self
        self.model = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, random_state=self.seed)
        self.model.fit(self._design(inputs, fit=True), y)
        return self

    def predict_proba(self, inputs: TicketInputs) -> np.ndarray:
        out = np.zeros((len(inputs), len(TAXONOMY)))
        if len(inputs) == 0:
            return out
        if self.model is None:
            if self.kind == "majority" or len(self.classes_) < 2:
                out[:, int(np.argmax(self.prior_))] = 1.0
            return out
        proba = self.model.predict_proba(self._design(inputs, fit=False))
        for j, cls in enumerate(self.model.classes_):
            out[:, TAXONOMY.index(str(cls))] = proba[:, j]
        return out

    def predict(self, inputs: TicketInputs) -> np.ndarray:
        return np.array(TAXONOMY)[np.argmax(self.predict_proba(inputs), axis=1)]


@dataclass
class FoldTextScores:
    """Per-ticket class probabilities for one evaluation fold.

    ``proba[kind]`` has one row per corpus ticket: tickets of training units
    are scored by a model that excluded their unit group (cross-fit), tickets
    of test units by the model fitted on all training tickets."""

    proba: dict[str, np.ndarray]
    chosen_c: dict[str, float]
    cv_macro_f1: dict[str, float]
    selected_kind: str
    n_train: int


def crossfit_fold(
    inputs: TicketInputs,
    labels: np.ndarray,
    group: np.ndarray,
    train_mask: np.ndarray,
    test_group: int,
    kinds: tuple[str, ...] = KINDS,
    c_grid: tuple[float, ...] = C_GRID,
    seed: int = 0,
) -> FoldTextScores:
    """Fit the ticket classifiers for one fold without touching test units.

    ``train_mask`` marks tickets whose label may be used (training units,
    label confirmed before the training cut-off). For each kind, C is chosen
    by leave-one-training-group-out macro-F1 on those tickets; the kind with
    the best such score is ``selected_kind`` (used for the fusion text signal).
    """
    labels = np.asarray(labels)
    train_groups = sorted({int(g) for g in group[train_mask]} - {test_group})
    if (group[train_mask] == test_group).any():
        raise AssertionError("training tickets contain the test unit group")
    n = len(inputs)
    proba: dict[str, np.ndarray] = {}
    chosen: dict[str, float] = {}
    cv: dict[str, float] = {}
    train_idx = np.flatnonzero(train_mask)
    for kind in kinds:
        best: tuple[float, float, np.ndarray] | None = None
        for c in c_grid:
            p = np.zeros((n, len(TAXONOMY)))
            for held in train_groups:
                fit_idx = np.flatnonzero(train_mask & (group != held))
                score_idx = np.flatnonzero(group == held)
                model = TicketClassifier(kind, c, seed).fit(inputs.take(fit_idx), labels[fit_idx])
                p[score_idx] = model.predict_proba(inputs.take(score_idx))
            pred = np.array(TAXONOMY)[np.argmax(p[train_idx], axis=1)]
            score = macro_f1(labels[train_idx], pred)
            if best is None or score > best[0]:
                best = (score, c, p)
        assert best is not None
        cv[kind], chosen[kind], p = best
        full = TicketClassifier(kind, chosen[kind], seed).fit(inputs.take(train_idx), labels[train_idx])
        test_idx = np.flatnonzero(group == test_group)
        p[test_idx] = full.predict_proba(inputs.take(test_idx))
        proba[kind] = p
    majority = TicketClassifier("majority").fit(inputs.take(train_idx), labels[train_idx])
    proba["majority"] = majority.predict_proba(inputs)
    selected = max(kinds, key=lambda k: cv[k])
    return FoldTextScores(proba, chosen, cv, selected, len(train_idx))
