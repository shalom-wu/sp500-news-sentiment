# Data Sources

## Primary dataset

**S&P 500 with Financial News Headlines (2008–2024)**
Kaggle, published by Dyuti Dasmahapatra:
<https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024>

- 19,127 rows (18,153 after removing exact date+headline duplicates)
- Columns: `Title` (headline text), `Date`, `CP` (S&P 500 closing price on that date)
- Span: 2008-01-02 to 2024-03-04, 3,507 distinct trading dates
- Real headlines and real index prices. Spot checks of `CP` against official
  S&P 500 closes (e.g. 2020-03-23 = 2237.40, 2008-09-29 = 1106.42,
  2013-12-31 = 1848.36, 2022-01-03 = 4796.56) match exactly.

The raw CSV is included at `data/raw/sp500_headlines_2008_2024.csv` so a
reviewer can inspect the project without a Kaggle download. The processed
files in `data/processed/` are derived from that raw file by the notebook and
`src/` pipeline. The Kaggle metadata check on 2026-07-06 returned license
`CC-BY-SA-4.0`.

## Known issues and how they are handled

| Issue | Detail | Handling |
| --- | --- | --- |
| Duplicate rows | 974 exact (date, headline) duplicates | Dropped in `load_headlines` |
| Sparse early coverage | 2008–2010 have headlines on only ~36–52% of trading days; 2011+ is ~86–100% | Coverage documented per year; "next-day" analysis restricted to genuine consecutive sessions |
| Prices only on headline days | The dataset has no price for a trading day with no headlines, so a naive `shift(-1)` return can silently span several sessions | An NYSE session calendar (weekends + market holidays + unscheduled closures like Hurricane Sandy 2012) marks each pair as `next_day_ok` only when the next observed date is the *immediately following* trading session; 91.5% of days qualify |
| Date-only timestamps | No time of day, so a headline dated *t* may have been published before or after that day's close | Same-day sentiment↔return association is treated as contemporaneous (and partly reverse-causal, since many headlines are market recaps); only day-*t*-headlines → day-*t+1*-move is treated as a predictive relationship |
| Close-only prices | No open/high/low/volume | Movement is measured close-to-close; volatility as a 5-day rolling std of returns |

## Sentiment model

**ProsusAI/finbert** (Hugging Face): BERT fine-tuned for financial sentiment
on the Financial PhraseBank. Downloaded at runtime by `src/sentiment.py`.
Chosen over generic lexicons because financial language inverts everyday
polarity ("bull", "short covering", "beat expectations", "cut guidance").

## Reference closes used in spot checks

Official S&P 500 daily closing values as widely published (e.g. S&P Dow Jones
Indices / financial press archives). Used only to validate the dataset's `CP`
column, not as an input to the analysis.
