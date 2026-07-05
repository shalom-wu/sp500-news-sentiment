"""The leakage audit.

The contract: every feature at date t is computed only from headlines and
prices dated <= t. These tests enforce that mechanically instead of by
code review — corrupt everything AFTER a cutoff date and assert that no
feature value at or before the cutoff moves. If someone later adds a
feature that peeks forward (a centered rolling window, a shift(-1), a
full-sample normalization), these tests fail.
"""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, TARGET_COL, build_model_frame, modeling_subset


def corrupt_after(scored: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Garble all rows dated strictly after the cutoff: flip and rescale
    sentiment, replace closes with noise. If any feature at t <= cutoff
    depends on post-cutoff data, its value must change."""
    out = scored.copy()
    after = out["date"] > cutoff
    rng = np.random.default_rng(7)
    out.loc[after, ["p_positive", "p_negative", "p_neutral"]] = rng.dirichlet(
        [1, 1, 1], after.sum()
    )
    out.loc[after, "sent_score"] = -out.loc[after, "sent_score"] * 0.31 + 0.17
    out.loc[after, "close"] = rng.uniform(50, 500, after.sum()).round(2)
    return out


class TestNoFeatureUsesTheFuture:
    def test_features_invariant_to_future_corruption(self, scored_frame):
        cutoff = scored_frame["date"].sort_values().unique()[24]
        base = build_model_frame(scored_frame)
        corrupted = build_model_frame(corrupt_after(scored_frame, cutoff))

        past = base["date"] <= cutoff
        for col in FEATURE_COLS:
            pd.testing.assert_series_equal(
                base.loc[past, col],
                corrupted.loc[past, col],
                obj=f"feature '{col}' changed when only post-cutoff data changed",
            )

    def test_label_does_depend_on_the_future(self, scored_frame):
        """Counter-check that the test itself has teeth: the LABEL at the
        cutoff date must change under future corruption, because it is
        defined by the next day's close."""
        dates = scored_frame["date"].sort_values().unique()
        cutoff = dates[24]
        base = build_model_frame(scored_frame).set_index("date")
        corrupted = build_model_frame(corrupt_after(scored_frame, cutoff)).set_index("date")
        assert (
            base.loc[cutoff, "next_ret"] != corrupted.loc[cutoff, "next_ret"]
        ), "corruption harness is broken: next_ret at cutoff did not change"


class TestModelingSubset:
    def test_excludes_non_consecutive_pairs(self, scored_frame):
        # Delete one full day so the prior day's 'next' spans two sessions.
        dates = scored_frame["date"].sort_values().unique()
        removed = dates[10]
        gapped = scored_frame[scored_frame["date"] != removed]
        sub = modeling_subset(build_model_frame(gapped))
        assert dates[9] not in set(sub["date"])

    def test_no_missing_values(self, scored_frame):
        sub = modeling_subset(build_model_frame(scored_frame))
        assert not sub[FEATURE_COLS + [TARGET_COL]].isna().any().any()

    def test_label_is_next_day_direction(self, scored_frame):
        df = build_model_frame(scored_frame)
        sub = modeling_subset(df)
        closes = df.set_index("date")["close"]
        for _, row in sub.head(10).iterrows():
            t = row["date"]
            later = closes.index[closes.index > t]
            t1 = later.min()
            assert row[TARGET_COL] == float(closes[t1] > closes[t])
