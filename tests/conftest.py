"""Shared synthetic fixtures. Tests run without the Kaggle download or
FinBERT: they exercise the pipeline's logic on constructed data where the
right answer is known."""

import numpy as np
import pandas as pd
import pytest

from src.data import NYSE_BDAY


@pytest.fixture
def scored_frame() -> pd.DataFrame:
    """A synthetic 'scored headlines' frame: 40 genuine NYSE sessions
    (bdate_range would include holidays like MLK Day, which the pipeline's
    session calendar rightly rejects), 3 headlines per day."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-04", periods=40, freq=NYSE_BDAY)
    rows = []
    close = 100.0
    for i, d in enumerate(dates):
        close *= 1 + rng.normal(0, 0.01)
        for j in range(3):
            p_pos = rng.uniform(0, 1)
            p_neg = rng.uniform(0, 1 - p_pos)
            p_neu = 1 - p_pos - p_neg
            rows.append(
                {
                    "date": d,
                    "headline": f"headline {i}-{j}",
                    "close": round(close, 2),
                    "p_positive": p_pos,
                    "p_negative": p_neg,
                    "p_neutral": p_neu,
                    "sent_score": p_pos - p_neg,
                }
            )
    return pd.DataFrame(rows)
