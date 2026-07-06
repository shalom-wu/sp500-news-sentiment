# Financial News Sentiment and Next-Day S&P 500 Movement

This repository tests whether daily financial-news sentiment is associated with next-trading-day S&P 500 returns. Headlines are scored with FinBERT, aligned to market closes, and evaluated with leakage checks, walk-forward validation, and simple baselines.

This is a research analysis, not a trading system or investment recommendation.

## Key findings
1. **The sentiment scorer works.** Mean FinBERT headline sentiment drops to
   **−0.34** during the 2008 financial crisis and **−0.20** during the 2020
   COVID crash, against a full-sample mean of −0.06. The monthly sentiment
   series tracks every major stress regime in the sample.

2. **Same-day correlation is strong (r = 0.18, p < 0.0001) — and mostly
   meaningless as prediction.** Many headlines are recaps written *after*
   the close ("Stocks Plunge As…"). This number validates the scorer and
   illustrates reverse causality; treating it as skill is the first trap
   this project is built to avoid.

3. **The next-day relationship is much weaker:
   next-day r = 0.031, Newey–West p = 0.052.** Directionally consistent —
   after the most-positive-sentiment days the next session is up **56.8%**
   of the time vs **51.8%** after the most-negative (χ² p = 0.050) — but
   borderline, and concentrated in high-volatility periods.

4. **A leakage-audited model cannot exploit it.** Walk-forward logistic
   regression reaches **53.5%** out-of-sample accuracy vs **53.9%** for
   simply always predicting "up" (McNemar p = 0.45), with AUC ≈ 0.50.
   The market's upward drift is the whole ballgame; sentiment adds nothing
   a model can use at daily index level.

5. **The naive backtest "wins" anyway ($3.60 vs $2.94 per $1, no costs) —
   which is the best exhibit in the repo.** A skill-free classifier
   produced a market-beating equity curve over 16 years by luckily sitting
   out a few bad days. That is exactly how spurious trading systems get
   sold, and why backtest curves are not evidence.

![Sentiment vs price](reports/figures/01_sentiment_vs_price.png)

## Methodology controls

Anyone can correlate sentiment with returns and find *something*. The value
of this project is in the controls that make the answer trustworthy:

- **Leakage auditing, enforced by tests.** Every feature at day *t* is built
  only from headlines and prices dated ≤ *t*. `tests/test_leakage.py` proves
  it mechanically: it corrupts all data after a cutoff date and asserts that
  no feature before the cutoff changes by a single bit. Add a feature that
  peeks forward and the suite fails.
- **A real trading calendar.** Prices exist only on dates with headlines, so
  "the next row" is sometimes several sessions away. An NYSE session calendar
  (holidays, plus unscheduled closures like Hurricane Sandy 2012) restricts
  "next-day" claims to genuine consecutive sessions — 91.5% of dates qualify,
  and the rest are excluded rather than silently mislabeled.
- **Domain-appropriate sentiment.** FinBERT (ProsusAI), fine-tuned on
  financial text, rather than a generic lexicon that misreads "bull",
  "short", or "beat expectations".
- **The right baseline.** The S&P 500 rises on ~54% of days, so "always up"
  — not 50% — is the bar. Baselines are evaluated on the identical
  out-of-sample days as the model, compared with McNemar's paired test.
- **Return regressions.** Newey–West (HAC) errors on all return regressions;
  walk-forward (expanding-window) validation with per-fold scaling; the
  same-day/next-day distinction maintained everywhere.

## Repository structure

```
├── data-sources.md          # dataset provenance, known issues, handling
├── notebooks/               # executed, in reading order
│   ├── 01_data_cleaning_and_alignment.ipynb
│   ├── 02_sentiment_scoring.ipynb
│   ├── 03_correlation_analysis.ipynb
│   └── 04_modeling_and_backtest.ipynb
├── src/
│   ├── data.py              # download, dedupe, NYSE-calendar alignment
│   ├── sentiment.py         # FinBERT scoring (cached, deterministic)
│   ├── features.py          # leakage-safe daily features and labels
│   ├── analysis.py          # correlations, HAC regressions, subgroups
│   ├── model.py             # walk-forward models, baselines, McNemar
│   └── plots.py             # all figures
├── reports/
│   ├── deck.md              # 8-slide findings deck
│   └── figures/
├── tests/                   # 29 tests, incl. the mechanical leakage audit
└── data/processed/daily_sentiment.csv   # date-level aggregates (committed)
```

## Reproducing the analysis

```bash
git clone <this-repo> && cd sp500-news-sentiment
pip install -r requirements.txt
pytest                                  # 29 tests, ~5s, no downloads needed
jupyter lab notebooks/                  # run 01 → 04 in order
```

Notebook 01 downloads the raw dataset from Kaggle automatically (anonymous,
no API key). Notebook 02 scores headlines with FinBERT — ~5 minutes on CPU,
cached thereafter; the committed `data/processed/daily_sentiment.csv` lets
notebooks 03–04 run without rescoring. Python 3.11+.

## SQL and Power BI layer

The [sql/](sql) folder adds a DuckDB validation and KPI layer over the included
headline sentiment outputs. It checks daily grain, duplicate date coverage,
missing forward returns, next-trading-day validity, and sentiment score ranges.

```bash
python scripts/run_sql.py
```

Exports land in `data/powerbi/`: daily market sentiment fact data, year-level
sentiment, sentiment-tercile forward returns, monthly news volume, and a
one-row KPI summary. The [power-bi/](power-bi) folder contains dashboard specs,
DAX, refresh steps, manual build instructions, and mockups. No `.pbix` is
included yet; I did not create a placeholder.

The raw dataset is now included in `data/raw/`, so the Kaggle download is
optional rather than required for review.

## Limitations

- **Timestamps are date-only.** Pre-close vs post-close publication cannot be
  distinguished, so same-day results are association, not causation, and the
  next-day design is the only clean predictive test available.
- **Close-to-close only** — no intraday prices, opens, or volume.
- **Index-level aggregation** — no stock-level cross-section, where text
  signals are typically stronger and better studied.
- **Uneven coverage**: 2008–2010 include only ~36–52% of trading days
  (results are re-verified on the dense 2011+ subsample); headline volume
  grows 30× across the sample as the source mix shifts.
- **Multiple comparisons**: with several subsamples inspected, borderline
  p ≈ 0.05 findings are suggestive, not established.
- The relationship measured is historical (2008–2024) and index-specific.
  None of it is investment advice.

## Data & attribution

Dataset: [S&P 500 with Financial News Headlines, 2008–2024](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024)
by Dyuti Dasmahapatra (Kaggle) — real headlines, real closing prices
(spot-verified against official S&P 500 closes). Raw and processed data are
included for review; see [data-sources.md](data-sources.md) and
`data/data_manifest.md`.
Sentiment model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert).

## License

MIT — see [LICENSE](LICENSE). © 2026 Shalom Wu
