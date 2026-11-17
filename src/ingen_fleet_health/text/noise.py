"""Deliberate, seeded text noise: paraphrase, abbreviations, typos, casing.

Every function takes an explicit ``numpy`` generator so a ticket's noise is
reproducible from the corpus seed.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ingen_fleet_health.text.templates import (
    ABBREVIATIONS,
    PHRASE_ABBREVIATIONS,
    SYNONYMS,
)

_WORD = re.compile(r"[A-Za-z]+")
_KEY_NEIGHBOURS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh", "h": "gj", "i": "uo",
    "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip", "p": "o", "q": "w", "r": "et",
    "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}


def paraphrase(text: str, rng: np.random.Generator) -> str:
    """Rule-based synonym swap (the offline stand-in for LLM paraphrase)."""
    for key, options in SYNONYMS.items():
        pattern = re.compile(rf"\b{re.escape(key)}\b")
        if pattern.search(text) and rng.random() < 0.5:
            text = pattern.sub(str(rng.choice(options)), text, count=1)
    return text


def abbreviate(text: str, probability: float, rng: np.random.Generator) -> tuple[str, int]:
    count = 0
    for phrase, short in PHRASE_ABBREVIATIONS.items():
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(text) and rng.random() < probability:
            text = pattern.sub(short, text, count=1)
            count += 1

    def swap(match: re.Match[str]) -> str:
        nonlocal count
        word = match.group(0)
        short = ABBREVIATIONS.get(word.lower())
        if short is None or rng.random() >= probability:
            return word
        count += 1
        return short

    return _WORD.sub(swap, text), count


def _typo(word: str, rng: np.random.Generator) -> str:
    i = int(rng.integers(1, len(word) - 1))
    kind = int(rng.integers(0, 4))
    if kind == 0:                                   # swap neighbours
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == 1:                                   # drop a letter
        return word[:i] + word[i + 1:]
    if kind == 2:                                   # double a letter
        return word[:i] + word[i] + word[i:]
    near = _KEY_NEIGHBOURS.get(word[i].lower())     # adjacent key
    return word[:i] + str(rng.choice(list(near))) + word[i + 1:] if near else word


def add_typos(text: str, probability: float, rng: np.random.Generator) -> tuple[str, int]:
    count = 0

    def maybe(match: re.Match[str]) -> str:
        nonlocal count
        word = match.group(0)
        if len(word) < 4 or word.isupper() or rng.random() >= probability:
            return word
        count += 1
        return _typo(word, rng)

    return _WORD.sub(maybe, text), count


def apply_noise(text: str, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[str, dict[str, int]]:
    """Paraphrase -> abbreviations -> typos -> punctuation -> casing."""
    text = paraphrase(text, rng)
    text, n_abbrev = abbreviate(text, float(cfg["abbreviation_probability"]), rng)
    text, n_typos = add_typos(text, float(cfg["typo_probability"]), rng)
    if rng.random() < float(cfg["drop_punctuation_fraction"]):
        text = re.sub(r"[.,;:]", "", text)
    u = rng.random()
    if u < float(cfg["lowercase_fraction"]):
        text = text.lower()
    elif u < float(cfg["lowercase_fraction"]) + float(cfg["uppercase_fraction"]):
        text = text.upper()
    else:
        text = text[:1].upper() + text[1:]
    return re.sub(r"\s+", " ", text).strip(), {"n_abbrev": n_abbrev, "n_typos": n_typos}
