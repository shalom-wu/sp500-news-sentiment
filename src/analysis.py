"""Correlation and significance testing for sentiment vs. market movement.

Two associations are deliberately kept separate throughout the project:

  same-day   sentiment(t) vs return(t-1 -> t). Many headlines are recaps
             written AFTER the move ("Stocks Plunge As..."), so this
             association is contemporaneous and partly reverse-causal.
             It validates the sentiment scorer; it is NOT predictive skill.

  next-day   sentiment(t) vs return(t -> t+1). Day-t headlines exist before
             the close(t+1) that defines the move, so this is the honest
             predictive question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def corr_table(df: pd.DataFrame, sent_col: str = "sent_mean") -> pd.DataFrame:
    """Pearson and Spearman correlations of daily sentiment with same-day
    and next-day returns, with p-values and sample sizes."""
    rows = []
    for label, ret_col, mask in [
        ("same-day", "ret", df["ret"].notna()),
        ("next-day", "next_ret", df["next_ret"].notna() & df["next_day_ok"]),
    ]:
        sub = df.loc[mask & df[sent_col].notna(), [sent_col, ret_col]]
        pr, pp = stats.pearsonr(sub[sent_col], sub[ret_col])
        sr, sp = stats.spearmanr(sub[sent_col], sub[ret_col])
        rows.append(
            {
                "relationship": label,
                "n": len(sub),
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
            }
        )
    return pd.DataFrame(rows)


def hac_regression(
    df: pd.DataFrame, ret_col: str, sent_col: str = "sent_mean", lags: int = 5
):
    """OLS of returns on sentiment with Newey-West (HAC) standard errors,
    which are robust to the autocorrelation and heteroskedasticity that
    daily financial returns always have."""
    import statsmodels.api as sm

    mask = df[ret_col].notna() & df[sent_col].notna()
    if ret_col == "next_ret":
        mask &= df["next_day_ok"].fillna(False)
    sub = df.loc[mask]
    X = sm.add_constant(sub[[sent_col]])
    return sm.OLS(sub[ret_col], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def direction_table(df: pd.DataFrame, sent_col: str = "sent_mean") -> pd.DataFrame:
    """Share of up next-days conditional on day-t sentiment tercile, with a
    chi-squared test of independence."""
    sub = df.loc[
        df["next_day_ok"].fillna(False) & df["next_up"].notna() & df[sent_col].notna()
    ].copy()
    sub["tercile"] = pd.qcut(sub[sent_col], 3, labels=["low", "mid", "high"])
    table = sub.groupby("tercile", observed=True).agg(
        n=("next_up", "size"),
        share_up=("next_up", "mean"),
        mean_next_ret=("next_ret", "mean"),
    )
    contingency = pd.crosstab(sub["tercile"], sub["next_up"])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    table.attrs["chi2"] = chi2
    table.attrs["chi2_p"] = p
    return table


def subgroup_correlations(df: pd.DataFrame, sent_col: str = "sent_mean") -> pd.DataFrame:
    """Next-day correlation split by news volume and by volatility regime,
    to ask WHERE any signal concentrates."""
    base = df.loc[
        df["next_day_ok"].fillna(False)
        & df["next_ret"].notna()
        & df[sent_col].notna()
        & df["vol5"].notna()
    ].copy()
    med_n = base["n_headlines"].median()
    med_v = base["vol5"].median()
    groups = {
        "high news volume": base["n_headlines"] > med_n,
        "low news volume": base["n_headlines"] <= med_n,
        "high volatility": base["vol5"] > med_v,
        "low volatility": base["vol5"] <= med_v,
    }
    rows = []
    for name, mask in groups.items():
        sub = base.loc[mask]
        r, p = stats.pearsonr(sub[sent_col], sub["next_ret"])
        rows.append({"subgroup": name, "n": len(sub), "pearson_r": r, "p": p})
    return pd.DataFrame(rows)


def crisis_windows() -> dict[str, tuple[str, str]]:
    """Known stress periods used as sanity checks on the sentiment scorer."""
    return {
        "2008 financial crisis": ("2008-09-01", "2009-03-31"),
        "2011 US downgrade": ("2011-07-15", "2011-10-15"),
        "2020 COVID crash": ("2020-02-15", "2020-04-15"),
        "2022 bear market": ("2022-01-01", "2022-10-31"),
    }


def crisis_sentiment(df: pd.DataFrame, sent_col: str = "sent_mean") -> pd.DataFrame:
    """Mean sentiment inside each crisis window vs. the full-sample mean.
    If the scorer works, crisis windows should be clearly more negative."""
    overall = df[sent_col].mean()
    rows = [{"period": "full sample", "n_days": df[sent_col].notna().sum(), "mean_sent": overall}]
    for name, (start, end) in crisis_windows().items():
        sub = df.loc[(df["date"] >= start) & (df["date"] <= end), sent_col]
        rows.append({"period": name, "n_days": sub.notna().sum(), "mean_sent": sub.mean()})
    out = pd.DataFrame(rows)
    out["vs_full_sample"] = out["mean_sent"] - overall
    return out
