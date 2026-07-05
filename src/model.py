"""Walk-forward classification of next-day direction, with honest baselines.

Validation is walk-forward (expanding window): the model is fit on all days
strictly before each test block and evaluated on the block. Test blocks are
contiguous in time and never precede training data, so no fold ever trains
on the future. Features are scaled inside each fold using training rows
only (Pipeline), so scaling cannot leak test-set statistics either.

Baselines matter here: US equity indices drift upward, so "always predict
up" already gets ~53-55% of days right. A sentiment model is only
interesting relative to that, not relative to 50%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS, TARGET_COL


@dataclass
class WalkForwardResult:
    name: str
    predictions: pd.DataFrame  # date, y_true, y_pred, y_prob, fold
    accuracy: float = field(init=False)
    balanced_accuracy: float = field(init=False)
    auc: float = field(init=False)

    def __post_init__(self):
        p = self.predictions
        self.accuracy = float((p["y_true"] == p["y_pred"]).mean())
        self.balanced_accuracy = float(
            balanced_accuracy_score(p["y_true"], p["y_pred"])
        )
        try:
            self.auc = float(roc_auc_score(p["y_true"], p["y_prob"]))
        except ValueError:
            self.auc = float("nan")


def _folds(n: int, n_folds: int, min_train: int):
    """Expanding-window folds over row indices 0..n-1 (rows must already be
    sorted by date). Yields (train_idx, test_idx) with train strictly
    before test."""
    test_size = (n - min_train) // n_folds
    for k in range(n_folds):
        start = min_train + k * test_size
        end = n if k == n_folds - 1 else start + test_size
        yield np.arange(0, start), np.arange(start, end)


def make_estimator(kind: str = "logistic"):
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)
        )
    if kind == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0
        )
    raise ValueError(kind)


def walk_forward(
    df: pd.DataFrame,
    kind: str = "logistic",
    n_folds: int = 5,
    min_train: int = 500,
    feature_cols: list[str] | None = None,
) -> WalkForwardResult:
    """Walk-forward evaluation. df must be the modeling_subset frame,
    sorted by date, one row per day t with features(<=t) and next_up."""
    feats = feature_cols or FEATURE_COLS
    df = df.sort_values("date").reset_index(drop=True)
    X, y = df[feats].to_numpy(), df[TARGET_COL].to_numpy()
    preds = []
    for fold, (tr, te) in enumerate(_folds(len(df), n_folds, min_train)):
        est = make_estimator(kind)
        est.fit(X[tr], y[tr])
        prob = est.predict_proba(X[te])[:, 1]
        preds.append(
            pd.DataFrame(
                {
                    "date": df.loc[te, "date"].values,
                    "y_true": y[te],
                    "y_pred": (prob >= 0.5).astype(float),
                    "y_prob": prob,
                    "next_ret": df.loc[te, "next_ret"].values,
                    "fold": fold,
                }
            )
        )
    return WalkForwardResult(name=f"{kind} (walk-forward)", predictions=pd.concat(preds, ignore_index=True))


def baseline_results(df: pd.DataFrame, oos_dates: pd.Series) -> list[WalkForwardResult]:
    """Baselines evaluated on exactly the same out-of-sample days as the
    model: always-up, yesterday's-direction persistence, and coin flip."""
    sub = df.loc[df["date"].isin(oos_dates)].sort_values("date").reset_index(drop=True)
    y = sub[TARGET_COL].to_numpy()
    rng = np.random.default_rng(0)
    frames = {
        "always up": np.ones_like(y),
        "persistence (today's direction)": (sub["ret"] > 0).astype(float).to_numpy(),
        "coin flip": rng.integers(0, 2, len(y)).astype(float),
    }
    out = []
    for name, pred in frames.items():
        out.append(
            WalkForwardResult(
                name=name,
                predictions=pd.DataFrame(
                    {
                        "date": sub["date"],
                        "y_true": y,
                        "y_pred": pred,
                        "y_prob": pred,
                        "next_ret": sub["next_ret"],
                        "fold": -1,
                    }
                ),
            )
        )
    return out


def accuracy_vs_baseline_test(model: WalkForwardResult, baseline: WalkForwardResult):
    """Is the model's out-of-sample accuracy better than the baseline's on
    the same days? Exact McNemar test on their disagreement days — the
    correct paired test for two classifiers on identical samples."""
    m = model.predictions.set_index("date")["y_pred"] == model.predictions.set_index("date")["y_true"]
    b = baseline.predictions.set_index("date")["y_pred"] == baseline.predictions.set_index("date")["y_true"]
    b = b.reindex(m.index)
    model_only = int((m & ~b).sum())
    base_only = int((~m & b).sum())
    n_disagree = model_only + base_only
    if n_disagree == 0:
        return {"model_only_correct": 0, "baseline_only_correct": 0, "p_value": 1.0}
    p = stats.binomtest(model_only, n_disagree, 0.5).pvalue
    return {
        "model_only_correct": model_only,
        "baseline_only_correct": base_only,
        "p_value": float(p),
    }


def naive_backtest(result: WalkForwardResult) -> pd.DataFrame:
    """Illustrative only — NOT a trading strategy or advice.

    "Hold the index on days the model says up, cash otherwise", ignoring
    costs, slippage, taxes and execution, vs. buy-and-hold on the same
    days. Purpose: translate accuracy numbers into 'what would following
    this signal have looked like', which is the honest way to show how
    little a small edge means after realism sets in.
    """
    p = result.predictions.sort_values("date").copy()
    p["strategy_ret"] = np.where(p["y_pred"] == 1.0, p["next_ret"], 0.0)
    p["bh_curve"] = (1 + p["next_ret"]).cumprod()
    p["strat_curve"] = (1 + p["strategy_ret"]).cumprod()
    return p[["date", "next_ret", "strategy_ret", "bh_curve", "strat_curve"]]
