"""FinBERT sentiment scoring for financial headlines.

Uses ProsusAI/finbert, a BERT model fine-tuned on financial text, instead of
a generic sentiment lexicon. Finance headlines are full of words that generic
tools misread: "bull", "short", "beat expectations", "crash diet of cost
cuts". FinBERT was trained on financial phrasebank data and handles this
domain language.

Scoring is deterministic (no sampling), so per-headline scores are cached to
CSV and reused. Each headline gets p_positive / p_negative / p_neutral
(softmax probabilities) and a single signed score = p_positive - p_negative
in [-1, 1].
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import PROCESSED_DIR

MODEL_NAME = "ProsusAI/finbert"
CACHE_PATH = PROCESSED_DIR / "headline_sentiment.csv"


def score_headlines(
    headlines: pd.DataFrame,
    batch_size: int = 64,
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Score every unique headline text with FinBERT.

    Returns the input frame with p_positive, p_negative, p_neutral and
    sent_score columns added. Results are cached; pass force=True to rescore.
    """
    if cache_path is None:
        cache_path = CACHE_PATH
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if len(cached) == len(headlines):
            return cached

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # Score each unique text once; the dataset repeats some headlines
    # across dates.
    unique_texts = headlines["headline"].drop_duplicates().tolist()
    probs = []
    with torch.no_grad():
        for start in range(0, len(unique_texts), batch_size):
            batch = unique_texts[start : start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=1))
    probs = torch.cat(probs).numpy()

    # ProsusAI/finbert label order: positive, negative, neutral.
    lookup = pd.DataFrame(
        probs, columns=["p_positive", "p_negative", "p_neutral"]
    ).assign(headline=unique_texts)

    scored = headlines.merge(lookup, on="headline", how="left")
    scored["sent_score"] = scored["p_positive"] - scored["p_negative"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(cache_path, index=False)
    return scored


def label_headlines(scored: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """Hard label per headline from the signed score."""
    return pd.cut(
        scored["sent_score"],
        bins=[-1.01, -abs(threshold) - 1e-12, abs(threshold), 1.01],
        labels=["negative", "neutral", "positive"],
    )
