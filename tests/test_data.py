import pandas as pd
import pytest

from src.data import build_price_frame, next_session


def make_headlines(dates_closes):
    rows = []
    for d, c in dates_closes:
        rows.append({"date": pd.Timestamp(d), "headline": f"h {d}", "close": c})
    return pd.DataFrame(rows)


class TestNextSession:
    def test_weekday_to_next_weekday(self):
        out = next_session(pd.Series([pd.Timestamp("2021-03-02")]))  # Tue
        assert out.iloc[0] == pd.Timestamp("2021-03-03")

    def test_friday_skips_weekend(self):
        out = next_session(pd.Series([pd.Timestamp("2021-03-05")]))  # Fri
        assert out.iloc[0] == pd.Timestamp("2021-03-08")  # Mon

    def test_good_friday_is_closed(self):
        # Thu 2021-04-01; Good Friday 2021-04-02 -> next session Mon 04-05.
        out = next_session(pd.Series([pd.Timestamp("2021-04-01")]))
        assert out.iloc[0] == pd.Timestamp("2021-04-05")

    def test_hurricane_sandy_closure(self):
        # Fri 2012-10-26; markets closed Mon 10-29 and Tue 10-30.
        out = next_session(pd.Series([pd.Timestamp("2012-10-26")]))
        assert out.iloc[0] == pd.Timestamp("2012-10-31")


class TestBuildPriceFrame:
    def test_one_row_per_date_and_returns(self):
        h = make_headlines([("2021-03-01", 100.0), ("2021-03-01", 100.0), ("2021-03-02", 102.0)])
        d = build_price_frame(h)
        assert len(d) == 2
        assert d.loc[1, "ret"] == pytest.approx(0.02)
        assert d.loc[0, "next_ret"] == pytest.approx(0.02)

    def test_next_day_ok_true_for_consecutive_sessions(self):
        h = make_headlines([("2021-03-01", 100.0), ("2021-03-02", 101.0)])
        d = build_price_frame(h)
        assert bool(d.loc[0, "next_day_ok"]) is True

    def test_next_day_ok_true_across_weekend(self):
        h = make_headlines([("2021-03-05", 100.0), ("2021-03-08", 101.0)])  # Fri -> Mon
        d = build_price_frame(h)
        assert bool(d.loc[0, "next_day_ok"]) is True

    def test_next_day_ok_false_when_session_missing(self):
        # Tue -> Thu with Wednesday a normal session the dataset lacks:
        # this pair spans two sessions and must NOT count as next-day.
        h = make_headlines([("2021-03-02", 100.0), ("2021-03-04", 101.0)])
        d = build_price_frame(h)
        assert bool(d.loc[0, "next_day_ok"]) is False

    def test_last_row_has_no_next(self):
        h = make_headlines([("2021-03-01", 100.0), ("2021-03-02", 101.0)])
        d = build_price_frame(h)
        assert pd.isna(d.loc[1, "next_ret"])
        assert not bool(d.loc[1, "next_day_ok"])
