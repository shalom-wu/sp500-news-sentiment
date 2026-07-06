# Data Manifest

This repo includes the raw headline file, processed sentiment outputs, and Power BI-ready tables. A reviewer does not need to download data from Kaggle to inspect the finished project.

| File | Type | Shape / size | Used by | Notes |
|---|---|---:|---|---|
| `raw/sp500_headlines_2008_2024.csv` | Real raw public dataset | 19,127 x 3, 1.5 MB | Python, notebooks, SQL | Kaggle dataset by Dyuti Dasmahapatra; metadata checked 2026-07-06; license `CC-BY-SA-4.0`. |
| `processed/headline_sentiment.csv` | Derived | 18,153 x 7, 2.2 MB | Python review, Power BI context | Exact date/headline duplicates removed; FinBERT probabilities and sentiment score added. |
| `processed/daily_sentiment.csv` | Derived aggregate | 3,507 x 14, 579.8 KB | Notebooks, SQL, Power BI | Daily sentiment, headline count, close-to-close returns, and next-trading-day flag. |
| `powerbi/fact_daily_market_sentiment.csv` | Derived | 3,507 x 14, 576.4 KB | Power BI | Dashboard fact table exported by `python scripts/run_sql.py`. |
| `powerbi/kpi_sentiment_summary.csv` | Derived aggregate | 1 x 8, <1 KB | Power BI | Correlation and coverage summary. |
| `powerbi/kpi_sentiment_by_year.csv` | Derived aggregate | 17 x 6, <1 KB | Power BI | Year-level sentiment and next-day return cuts. |
| `powerbi/kpi_sentiment_tercile_forward_returns.csv` | Derived aggregate | 3 x 5, <1 KB | Power BI | Next-day returns by sentiment bucket. |
| `powerbi/kpi_news_volume_by_month.csv` | Derived aggregate | 195 x 5, 6.0 KB | Power BI | Monthly news volume and sentiment trend. |

The source contains real headlines and index closes, but it has date-only headlines and no intraday timestamps. The project therefore treats same-day movement as contemporaneous and uses next-day movement only when the next observed date is a true next trading session.
