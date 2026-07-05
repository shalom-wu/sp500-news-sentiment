"""Daily sentiment aggregation and leakage-safe feature construction.

The leakage rule for this project: the feature row indexed at date t may
only use headlines dated <= t and prices dated <= t. It is used to predict
the move from close(t) to close(t+1). Every feature below is either
computed from day-t headlines, or from strictly earlier days via shift()
and rolling windows over past rows. Nothing indexed at t touches t+1 data.

tests/test_leakage.py enforces this mechanically by corrupting future rows
and asserting features at t do not change.
"""

from __future__ import annotations

import pandas as pd

from .data import build_price_frame


def aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-headline FinBERT scores to one row per date.

    Aggregates use only that date's headlines: mean signed score, share of
    positive / negative headlines, mean class probabilities, and count.
    """
    daily_sent = (
        scored.assign(
            is_pos=scored["sent_score"] > 0.05,
            is_neg=scored["sent_score"] < -0.05,
        )
        .groupby("date")
        .agg(
            sent_mean=("sent_score", "mean"),
            sent_median=("sent_score", "median"),
            sent_std=("sent_score", "std"),
            share_pos=("is_pos", "mean"),
            share_neg=("is_neg", "mean"),
            p_pos_mean=("p_positive", "mean"),
            p_neg_mean=("p_negative", "mean"),
            n_headlines=("headline", "size"),
        )
        .reset_index()
    )
    daily_sent["sent_std"] = daily_sent["sent_std"].fillna(0.0)
    return daily_sent


def build_model_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Join daily sentiment with prices and add lag/rolling features.

    Feature columns (information available at end of day t):
      sent_mean, share_pos, share_neg, sent_std   day-t headlines
      sent_mean_lag1                              day t-1 sentiment
      sent_mean_roll3                             mean over days t-2..t
      sent_change                                 day t minus day t-1
      log_n_headlines                             news volume on day t
      ret                                         day t close-to-close return
      ret_lag1                                    day t-1 return
      vol5                                        std of returns over t-4..t

    Target columns (information from day t+1 — used ONLY as labels):
      next_ret, next_up, gap_to_next, next_day_ok
    """
    import numpy as np

    daily_sent = aggregate_daily(scored)
    prices = build_price_frame(scored)
    df = prices.merge(daily_sent, on="date", how="left", validate="1:1")
    # build_price_frame counts raw headlines; keep the sentiment-side count.
    df = df.drop(columns=["n_headlines_x"]).rename(
        columns={"n_headlines_y": "n_headlines"}
    )

    # Past-only derived features. shift(1) moves yesterday's value onto
    # today's row; rolling(...) over the already-shifted-or-current series
    # never sees rows after t because rolling windows look backward.
    df["sent_mean_lag1"] = df["sent_mean"].shift(1)
    df["sent_mean_roll3"] = df["sent_mean"].rolling(3, min_periods=1).mean()
    df["sent_change"] = df["sent_mean"] - df["sent_mean_lag1"]
    df["log_n_headlines"] = np.log1p(df["n_headlines"])
    df["ret_lag1"] = df["ret"].shift(1)
    df["vol5"] = df["ret"].rolling(5, min_periods=2).std()

    df["next_up"] = (df["next_ret"] > 0).astype("float")
    df.loc[df["next_ret"].isna(), "next_up"] = float("nan")
    return df


FEATURE_COLS = [
    "sent_mean",
    "share_pos",
    "share_neg",
    "sent_std",
    "sent_mean_lag1",
    "sent_mean_roll3",
    "sent_change",
    "log_n_headlines",
    "ret",
    "ret_lag1",
    "vol5",
]

TARGET_COL = "next_up"


def modeling_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for next-day modeling: a genuine next-trading-day pair
    exists (next_day_ok) and all features/labels are present."""
    cols = FEATURE_COLS + [TARGET_COL, "date", "next_ret", "gap_to_next"]
    sub = df.loc[df["next_day_ok"] == True, cols]  # noqa: E712
    return sub.dropna().reset_index(drop=True)
