import numpy as np
import pandas as pd
import pytest

from src.features import aggregate_daily, build_model_frame


class TestAggregateDaily:
    def test_one_row_per_date(self, scored_frame):
        daily = aggregate_daily(scored_frame)
        assert len(daily) == scored_frame["date"].nunique()

    def test_mean_matches_manual(self, scored_frame):
        daily = aggregate_daily(scored_frame).set_index("date")
        d0 = scored_frame["date"].iloc[0]
        manual = scored_frame.loc[scored_frame["date"] == d0, "sent_score"].mean()
        assert daily.loc[d0, "sent_mean"] == pytest.approx(manual)

    def test_shares_bounded(self, scored_frame):
        daily = aggregate_daily(scored_frame)
        assert daily["share_pos"].between(0, 1).all()
        assert daily["share_neg"].between(0, 1).all()


class TestBuildModelFrame:
    def test_lag_is_previous_day(self, scored_frame):
        df = build_model_frame(scored_frame)
        assert df["sent_mean_lag1"].iloc[5] == pytest.approx(df["sent_mean"].iloc[4])

    def test_roll3_is_backward_looking(self, scored_frame):
        df = build_model_frame(scored_frame)
        expected = df["sent_mean"].iloc[3:6].mean()
        assert df["sent_mean_roll3"].iloc[5] == pytest.approx(expected)

    def test_vol5_uses_past_returns_only(self, scored_frame):
        df = build_model_frame(scored_frame)
        expected = df["ret"].iloc[6:11].std()
        assert df["vol5"].iloc[10] == pytest.approx(expected)

    def test_next_up_matches_next_ret_sign(self, scored_frame):
        df = build_model_frame(scored_frame).dropna(subset=["next_ret"])
        assert ((df["next_ret"] > 0).astype(float) == df["next_up"]).all()
