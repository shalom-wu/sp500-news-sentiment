# S&P 500 News Sentiment Research

This repository tests whether daily financial-news sentiment is associated with next-trading-day S&P 500 returns. Headlines are scored with FinBERT, aligned to market closes, and evaluated with leakage checks, walk-forward validation, and simple baselines.

This is a research analysis, not a trading system or investment recommendation.

## Project Summary

| Area | Details |
|---|---|
| Business question | Does financial-news sentiment add usable next-day signal beyond the market's normal upward drift? |
| Data | S&P 500 prices and financial news headlines from a public Kaggle dataset covering 2008-2024. |
| Methods | FinBERT sentiment scoring, trading-calendar alignment, leakage-audited features, HAC regressions, walk-forward validation, McNemar baseline comparison. |
| Main outputs | Research deck, daily sentiment table, model comparison, backtest illustration, Power BI-ready exports. |
| Tools | Python, pytest, DuckDB SQL, FinBERT, Power BI build documentation. |

## Key Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | The sentiment scorer captures stress periods. | Mean headline sentiment drops during the 2008 financial crisis and the 2020 COVID crash. |
| 2 | Same-day correlation is visible but not predictive evidence. | Many same-day headlines are post-close recaps, so same-day movement is vulnerable to reverse causality. |
| 3 | Next-day signal is weak. | Next-day correlation is about 0.031 with borderline statistical significance. |
| 4 | A leakage-audited model does not beat the baseline. | Walk-forward logistic regression reaches about 53.5% accuracy versus 53.9% for always predicting "up." |
| 5 | The naive backtest is a warning sign, not proof. | A lucky equity curve can outperform without reliable predictive skill. |

![Sentiment vs price](reports/figures/01_sentiment_vs_price.png)

## Data

The project uses [S&P 500 with Financial News Headlines, 2008-2024](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024) by Dyuti Dasmahapatra on Kaggle. The sentiment model is [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert).

The raw dataset is included in `data/raw/`, and the processed daily sentiment file is included in `data/processed/daily_sentiment.csv`. Source notes are in [data-sources.md](data-sources.md) and [data/data_manifest.md](data/data_manifest.md).

## Methodology

1. Deduplicate and align headline dates to valid S&P 500 trading sessions.
2. Score headlines with FinBERT and aggregate them to daily sentiment features.
3. Build features using only information available at or before day `t`.
4. Compare same-day association with next-trading-day prediction.
5. Evaluate walk-forward models against the correct baseline: the S&P 500's historical tendency to rise more often than fall.

## Repository Contents

| Path | Purpose |
|---|---|
| [notebooks/](notebooks) | Four executed notebooks covering cleaning, sentiment scoring, correlation analysis, and modeling. |
| [src/](src) | Data loading, sentiment scoring, leakage-safe features, modeling, analysis, and plotting code. |
| [tests/](tests) | Leakage tests, feature tests, model tests, and data checks. |
| [reports/](reports) | Research deck and generated figures. |
| [sql/](sql) | DuckDB validation and KPI views. |
| [power-bi/](power-bi) | Dashboard brief, DAX, model notes, refresh steps, and mockups. |
| [data/](data) | Raw, processed, and Power BI-ready data files. |

## Reproduce

Requires Python 3.11+.

```bash
git clone https://github.com/shalom-wu/sp500-news-sentiment-research.git
cd sp500-news-sentiment-research
pip install -r requirements.txt
pytest
jupyter lab notebooks/
python scripts/run_sql.py
```

Notebook 02 scores headlines with FinBERT and can take several minutes on CPU. The committed processed file lets the later notebooks run without rescoring.

## Reporting Layer

The [sql/](sql) folder validates daily grain, duplicate date coverage, missing forward returns, next-trading-day validity, and sentiment score ranges. The SQL runner exports dashboard-ready tables to `data/powerbi/`.

The [power-bi/](power-bi) folder contains dashboard specifications, DAX, refresh steps, manual build instructions, and mockups. No placeholder `.pbix` file is included.

## Limitations

- Headlines have dates but not reliable publication times, so pre-close and post-close news cannot be separated.
- The analysis is close-to-close only; it does not use intraday prices, opens, or volume.
- Index-level aggregation can dilute text signals that may be stronger at the stock level.
- Coverage is uneven across years, and headline volume rises materially across the sample.
- The results are historical and index-specific. They are not investment advice.

## License And Credit

MIT License. Copyright (c) 2026 Shalom Wu.

Data credit: Dyuti Dasmahapatra's Kaggle dataset, "S&P 500 with Financial News Headlines, 2008-2024." Sentiment model credit: ProsusAI/finbert. See [data-sources.md](data-sources.md) and [data/data_manifest.md](data/data_manifest.md).
