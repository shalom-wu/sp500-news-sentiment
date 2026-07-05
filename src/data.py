"""Load, clean, and align the S&P 500 headline/price dataset.

Dataset: "S&P 500 with Financial News Headlines (2008-2024)" (Kaggle,
dyutidasmahaptra). Each row is one headline with the date it was published
and the S&P 500 closing price ("CP") on that date. Prices only exist on
dates that have at least one headline, so the trading calendar here is
"dates present in the dataset", not the full NYSE calendar. Alignment
functions below track the calendar-day gap to the next observation so that
downstream analysis can restrict itself to genuine next-trading-day pairs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

DATASET_SLUG = "dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024"
CSV_NAME = "sp500_headlines_2008_2024.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class NYSEHolidays(AbstractHolidayCalendar):
    """NYSE full-closure holidays (not the federal calendar: markets close
    on Good Friday but stay open on Columbus Day and Veterans Day)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        # Unscheduled full closures within the sample period.
        Holiday("Hurricane Sandy day 1", year=2012, month=10, day=29),
        Holiday("Hurricane Sandy day 2", year=2012, month=10, day=30),
        Holiday("G.H.W. Bush mourning", year=2018, month=12, day=5),
        Holiday("Gerald Ford mourning", year=2007, month=1, day=2),
    ]


NYSE_BDAY = CustomBusinessDay(calendar=NYSEHolidays())


def next_session(dates: pd.Series) -> pd.Series:
    """The next expected NYSE trading session after each date (vectorized:
    build the session calendar once, then look up the following session)."""
    sessions = pd.date_range(
        dates.min(), dates.max() + pd.Timedelta(days=10), freq=NYSE_BDAY
    )
    idx = sessions.searchsorted(dates, side="right")
    return pd.Series(sessions[idx], index=dates.index)


def download_raw(force: bool = False) -> Path:
    """Download the dataset from Kaggle (anonymous) into data/raw/."""
    dest = RAW_DIR / CSV_NAME
    if dest.exists() and not force:
        return dest
    import kagglehub

    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    src = cache_path / CSV_NAME
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return dest


def load_headlines(path: Path | None = None) -> pd.DataFrame:
    """Load and clean the raw CSV.

    Returns one row per (date, headline): duplicates dropped, dates parsed,
    sorted by date. Columns: date, headline, close.
    """
    if path is None:
        path = download_raw()
    df = pd.read_csv(path)
    df = df.rename(columns={"Title": "headline", "Date": "date", "CP": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df["headline"] = df["headline"].str.strip()
    df = df.dropna(subset=["headline", "date", "close"])
    df = df.drop_duplicates(subset=["date", "headline"])
    return df.sort_values("date").reset_index(drop=True)


def build_price_frame(headlines: pd.DataFrame) -> pd.DataFrame:
    """One row per observed trading date with close, returns, and gap info.

    Columns:
      date, close, n_headlines
      ret            close(t-1) -> close(t) simple return
      gap_from_prev  calendar days since previous observed date
      next_ret       close(t) -> close(t_next) return (the prediction target)
      gap_to_next    calendar days until next observed date
      next_day_ok    True when the next observed date IS the next expected
                     NYSE session after t — i.e. next_ret is a genuine
                     next-trading-day move, not a multi-session jump across
                     dates the dataset is missing
    """
    daily = (
        headlines.groupby("date")
        .agg(close=("close", "first"), n_headlines=("headline", "size"))
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["ret"] = daily["close"].pct_change()
    daily["gap_from_prev"] = daily["date"].diff().dt.days
    daily["next_ret"] = daily["close"].shift(-1) / daily["close"] - 1
    daily["gap_to_next"] = (daily["date"].shift(-1) - daily["date"]).dt.days
    daily["next_day_ok"] = daily["date"].shift(-1) == next_session(daily["date"])
    return daily


def coverage_report(headlines: pd.DataFrame) -> pd.DataFrame:
    """Per-year coverage: headline count, observed days, and an approximate
    share of the ~252 trading days per year that appear in the dataset."""
    per_year = headlines.groupby(headlines["date"].dt.year).agg(
        headlines=("headline", "size"), days=("date", "nunique")
    )
    per_year["approx_coverage"] = (per_year["days"] / 252).round(2)
    return per_year
