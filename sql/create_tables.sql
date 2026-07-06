-- DuckDB local views over the included project-ready data.

CREATE OR REPLACE VIEW daily_sentiment AS
SELECT
    CAST(date AS DATE) AS date,
    close::DOUBLE AS close,
    ret::DOUBLE AS same_day_return,
    next_ret::DOUBLE AS next_day_return,
    gap_to_next::DOUBLE AS gap_to_next,
    next_day_ok::BOOLEAN AS next_day_ok,
    sent_mean::DOUBLE AS sentiment_mean,
    sent_median::DOUBLE AS sentiment_median,
    sent_std::DOUBLE AS sentiment_std,
    share_pos::DOUBLE AS share_positive,
    share_neg::DOUBLE AS share_negative,
    p_pos_mean::DOUBLE AS avg_positive_probability,
    p_neg_mean::DOUBLE AS avg_negative_probability,
    n_headlines::INTEGER AS headline_count
FROM read_csv_auto('data/processed/daily_sentiment.csv', HEADER = TRUE);

CREATE OR REPLACE VIEW headline_sentiment AS
SELECT
    headline,
    CAST(date AS DATE) AS date,
    close::DOUBLE AS close,
    p_positive::DOUBLE AS p_positive,
    p_negative::DOUBLE AS p_negative,
    p_neutral::DOUBLE AS p_neutral,
    sent_score::DOUBLE AS sentiment_score
FROM read_csv_auto('data/processed/headline_sentiment.csv', HEADER = TRUE);

CREATE OR REPLACE VIEW v_daily_market_sentiment AS
SELECT *
FROM daily_sentiment;
