import numpy as np
import pandas as pd
import pytest

from src.model import (
    WalkForwardResult,
    _folds,
    accuracy_vs_baseline_test,
    baseline_results,
    walk_forward,
)


class TestFolds:
    def test_train_strictly_before_test(self):
        for tr, te in _folds(1000, n_folds=5, min_train=300):
            assert tr.max() < te.min()

    def test_folds_cover_tail_without_overlap(self):
        seen = []
        for _, te in _folds(1000, n_folds=5, min_train=300):
            seen.extend(te.tolist())
        assert seen == list(range(300, 1000))


def synthetic_modeling_frame(n=800, signal=False, seed=0):
    """Frame shaped like modeling_subset output. With signal=True the
    sentiment feature perfectly encodes tomorrow's direction."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2015-01-01", periods=n),
            "next_up": y,
            "next_ret": np.where(y == 1, 0.005, -0.005),
            "gap_to_next": 1.0,
        }
    )
    from src.features import FEATURE_COLS

    for col in FEATURE_COLS:
        df[col] = rng.normal(0, 1, n)
    if signal:
        df["sent_mean"] = y + rng.normal(0, 0.05, n)
    df["ret"] = rng.normal(0, 0.01, n)
    return df


class TestWalkForward:
    def test_learns_planted_signal(self):
        res = walk_forward(synthetic_modeling_frame(signal=True), min_train=200)
        assert res.accuracy > 0.95

    def test_no_signal_means_coin_flip(self):
        res = walk_forward(synthetic_modeling_frame(signal=False), min_train=200)
        assert 0.40 < res.accuracy < 0.60

    def test_predictions_only_on_out_of_sample_days(self):
        df = synthetic_modeling_frame()
        res = walk_forward(df, min_train=200)
        assert len(res.predictions) == len(df) - 200
        assert res.predictions["date"].min() == df["date"].iloc[200]


class TestBaselines:
    def test_always_up_accuracy_is_up_share(self):
        df = synthetic_modeling_frame()
        oos = df["date"].iloc[200:]
        base = baseline_results(df, oos)
        always_up = next(r for r in base if r.name == "always up")
        assert always_up.accuracy == pytest.approx(
            df["next_up"].iloc[200:].mean()
        )

    def test_mcnemar_identical_predictions_p1(self):
        df = synthetic_modeling_frame()
        oos = df["date"].iloc[200:]
        base = baseline_results(df, oos)
        out = accuracy_vs_baseline_test(base[0], base[0])
        assert out["p_value"] == 1.0

    def test_mcnemar_detects_dominant_model(self):
        preds = pd.DataFrame(
            {
                "date": pd.bdate_range("2020-01-01", periods=100),
                "y_true": 1.0,
                "y_pred": [1.0] * 90 + [0.0] * 10,
                "y_prob": 0.5,
                "next_ret": 0.001,
                "fold": 0,
            }
        )
        good = WalkForwardResult("good", preds)
        bad_preds = preds.copy()
        bad_preds["y_pred"] = [0.0] * 60 + [1.0] * 40
        bad = WalkForwardResult("bad", bad_preds)
        out = accuracy_vs_baseline_test(good, bad)
        assert out["p_value"] < 0.01
