"""Presentation-quality figures. All figures save to reports/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def sentiment_and_price(df: pd.DataFrame, crisis: dict[str, tuple[str, str]]) -> Path:
    """Monthly mean sentiment vs. S&P 500 close, crisis windows shaded."""
    m = (
        df.set_index("date")[["sent_mean", "close"]]
        .resample("ME")
        .mean()
        .dropna()
    )
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(m.index, m["sent_mean"], color="#1f77b4", lw=1.8, label="Monthly mean FinBERT sentiment")
    ax1.axhline(0, color="grey", lw=0.8, ls="--")
    ax1.set_ylabel("Mean sentiment score", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(m.index, m["close"], color="#444444", lw=1.4, alpha=0.7, label="S&P 500 close")
    ax2.set_ylabel("S&P 500 close", color="#444444")
    ax2.grid(False)
    for name, (s, e) in crisis.items():
        ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="red", alpha=0.08)
    ax1.set_title("Headline sentiment tracks the market's worst periods (shaded = known stress windows)")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    return _save(fig, "01_sentiment_vs_price.png")


def news_volume(df: pd.DataFrame) -> Path:
    yearly = df.groupby(df["date"].dt.year)["n_headlines"].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    yearly.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title("Headline volume by year (coverage is thin before 2011)")
    ax.set_xlabel("")
    ax.set_ylabel("Headlines")
    return _save(fig, "02_news_volume_by_year.png")


def sentiment_distribution(scored: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(scored["sent_score"], bins=60, ax=ax, color="#1f77b4")
    ax.set_title("Per-headline FinBERT score (positive minus negative probability)")
    ax.set_xlabel("Signed sentiment score")
    return _save(fig, "03_sentiment_distribution.png")


def scatter_with_fit(df: pd.DataFrame, ret_col: str, title: str, name: str) -> Path:
    mask = df[ret_col].notna() & df["sent_mean"].notna()
    if ret_col == "next_ret":
        mask &= df["next_day_ok"].fillna(False)
    sub = df.loc[mask]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.regplot(
        data=sub, x="sent_mean", y=ret_col, ax=ax,
        scatter_kws={"alpha": 0.25, "s": 14}, line_kws={"color": "crimson"},
    )
    ax.set_title(title)
    ax.set_xlabel("Daily mean FinBERT sentiment")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    return _save(fig, name)


def direction_by_tercile(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    table["share_up"].plot(kind="bar", ax=ax, color=["#d62728", "#7f7f7f", "#2ca02c"])
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_title("Share of UP next-days by day-t sentiment tercile")
    ax.set_xlabel("Sentiment tercile on day t")
    ax.set_ylabel("Share of next days up")
    ax.set_ylim(0.4, 0.65)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for i, v in enumerate(table["share_up"]):
        ax.annotate(f"{v:.1%}", (i, v), ha="center", va="bottom", fontsize=13)
    return _save(fig, "05_direction_by_tercile.png")


def model_vs_baselines(results: list) -> Path:
    names = [r.name for r in results]
    accs = [r.accuracy for r in results]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(names, accs, color=["#1f77b4" if "walk" in n else "#9e9e9e" for n in names])
    ax.set_xlim(0.4, max(accs) + 0.06)
    ax.set_title("Out-of-sample accuracy, model vs. baselines (same test days)")
    ax.set_xlabel("Accuracy")
    for b, a in zip(bars, accs):
        ax.annotate(f"{a:.1%}", (a, b.get_y() + b.get_height() / 2), va="center", ha="left", fontsize=12)
    return _save(fig, "06_model_vs_baselines.png")


def backtest_curves(bt: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(bt["date"], bt["bh_curve"], label="Buy & hold (same days)", color="#444444")
    ax.plot(bt["date"], bt["strat_curve"], label="Sentiment signal (no costs)", color="#1f77b4")
    ax.set_title("Illustrative signal-following curve — methodology exhibit, not a strategy")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    return _save(fig, "07_backtest_illustration.png")


def crisis_bars(crisis_df: pd.DataFrame) -> Path:
    sub = crisis_df[crisis_df["period"] != "full sample"].set_index("period")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sub["mean_sent"].plot(kind="barh", ax=ax, color="#d62728")
    ax.axvline(crisis_df.loc[crisis_df["period"] == "full sample", "mean_sent"].iloc[0],
               color="black", ls="--", lw=1, label="Full-sample mean")
    ax.set_title("Mean sentiment in known stress windows (sanity check)")
    ax.set_xlabel("Mean FinBERT sentiment")
    ax.legend()
    return _save(fig, "04_crisis_sanity_check.png")
