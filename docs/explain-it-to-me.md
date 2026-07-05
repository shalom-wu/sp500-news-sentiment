# Explain It To Me: News Sentiment and the Stock Market

*A zero-background guide to what this project is, what it found, and how the
code works — written so you can explain it out loud with confidence.*

---

## Section 1: What is this project, in plain English?

### The question

Every day, financial news outlets publish headlines — "Stocks Rally on Fed
Optimism," "Recession Fears Grip Markets." Some sound optimistic, some sound
scared. The question this project asks is simple to state:

> **If today's headlines sound unusually positive or negative, is the stock
> market more likely to go up or down *tomorrow*?**

That's it. Not "can I predict stock prices" — nobody reliably can, and this
project doesn't try. The question is whether there's *any measurable
statistical relationship* between the tone of the news and the very next
day's market move, and if so, how big it is.

### Why this is a real question and not a get-rich scheme

This is a genuinely studied question in academic finance, with a paper trail
going back decades (the landmark study looked at a Wall Street Journal
column and found pessimistic language predicted slightly lower returns).
It sits at the center of a real tension:

- **Markets are supposed to absorb information fast.** If news moved prices
  predictably, traders would exploit it until it stopped working. So theory
  says the relationship should be almost zero.
- **But "almost" is doing a lot of work.** Around crises and panics, does
  the tone of the news carry a little extra information? That's an
  empirical question you can only answer with data and discipline.

The honest expected answer — which is also roughly what I found — is "there's
a faint trace, but nothing you could use." The point of the project is
*demonstrating the discipline needed to reach that answer without fooling
yourself*, because the ways to accidentally fool yourself are numerous and
subtle, and they're the same mistakes that make fake "AI trading systems"
look good in sales decks.

### What was actually built

Four things, in sequence:

1. **A cleaned, aligned dataset.** 18,153 real headlines from 2008–2024
   matched to the S&P 500's real closing price on each date, with careful
   handling of missing days, weekends, and holidays.
2. **A sentiment score for every headline**, produced by FinBERT — a
   language model specifically trained on financial text, so it knows
   "beat expectations" is good news and "short squeeze" isn't about height.
3. **A statistical analysis** asking whether daily average sentiment lines
   up with the market's same-day move (it does, but for a boring reason)
   and next-day move (barely, right at the edge of what statisticians
   would call "real").
4. **A prediction model with a referee.** A machine-learning model tries to
   predict tomorrow's direction from today's sentiment — under strict rules
   that prevent it from cheating by peeking at the future — and gets
   compared against the dumbest possible strategy ("always predict up").
   The dumb strategy wins. That result is reported plainly, because it's
   the true result.

---

## Section 2: The elevator versions

### 30 seconds

"I tested whether the tone of financial news headlines predicts the next
day's S&P 500 move, using 16 years of real headlines scored with a
finance-tuned language model. Finding: headline sentiment clearly *reflects*
what the market did that same day, but its ability to *predict* tomorrow is
tiny — right at the edge of statistical significance — and a properly
validated model can't turn it into predictions that beat just assuming the
market goes up. The real work in the project is the rigor: preventing data
leakage, using honest baselines, and not overselling a weak signal."

### 2 minutes

Add these four beats:

1. **Why a finance-specific sentiment model matters:** generic sentiment
   tools misread finance. "Bull market," "crushed estimates," "shorts
   getting squeezed" — ordinary sentiment dictionaries get these wrong.
   I used FinBERT, which is trained on financial text. Sanity check: its
   average sentiment plunges exactly where it should — the 2008 crisis
   (−0.34 vs −0.06 normally) and the COVID crash (−0.20).

2. **The same-day trap:** sentiment correlates strongly with the *same
   day's* return (r = 0.18, wildly significant). Sounds impressive, means
   little — headlines like "Stocks Plunge As…" are written *after* the drop.
   The news is describing the move, not causing it. Separating that from
   genuine prediction is the core intellectual move in the project.

3. **The honest number:** for *next-day* returns, correlation drops to
   0.03 with a p-value of 0.052 — the faintest trace of signal, mostly in
   high-volatility periods, exactly what the academic literature would
   predict.

4. **The model result:** out of sample, my best model gets 53.5% of
   directions right. Always predicting "up" gets 53.9%. Statistically
   indistinguishable. And here's my favorite part: the naive backtest
   *still* beat buy-and-hold — a skill-free model got lucky dodging a few
   bad days across 16 years. That's a live demonstration of why backtest
   curves sell fake trading systems.

### 5 minutes

Walk the 2-minute version, then add:

- **The alignment problem nobody notices:** the dataset only has prices on
  days that have headlines. So "the next row in the spreadsheet" is
  sometimes three sessions later, and a lazy analysis would call a
  three-day move a "next-day" move. I built an NYSE trading calendar —
  weekends, holidays, even the two days the market closed for Hurricane
  Sandy — so that "next day" always means the literal next trading session.
  About 8.5% of days fail that test and are excluded from next-day claims.

- **What leakage means and how I policed it:** data leakage is letting
  information from the future sneak into what a model knows at prediction
  time. My rule: everything the model sees on day *t* must come from day
  *t* or earlier. It's enforced by an automated test that deliberately
  corrupts all data after a cutoff date and verifies that no feature
  before the cutoff changes. If anyone adds a leaky feature, the test
  fails. Validation is walk-forward: the model is always trained on the
  past and tested on a later period, the way real forecasting works.

- **Why "always up" is the right baseline:** the S&P rose on about 54% of
  days in this sample. A model boasting "53% accuracy!" is literally worse
  than a rock with "UP" painted on it. Any honest evaluation of a
  direction predictor must clear the majority-class baseline, and mine
  doesn't — which I report as the finding, not as a failure to hide.

- **What I'd defend as actually useful:** not trading. But sharp negative
  sentiment spikes reliably coincide with genuinely eventful days, so this
  pipeline could serve as an anomaly flag for human review — "the news got
  unusually grim today, someone should look" — and as a reusable template
  for stress-testing any proposed text-based signal before money touches it.

---

## Section 3: How the code actually works

### The map — what each piece does, in reading order

| Read | File | What it does |
| --- | --- | --- |
| 1st | `data-sources.md` | Where the data comes from and every known wart |
| 2nd | `src/data.py` | Downloads the Kaggle CSV, removes 974 duplicate rows, parses dates, builds the daily price table, and — the interesting part — builds an NYSE session calendar to decide which day-pairs are genuine "next trading day" pairs |
| 3rd | `src/sentiment.py` | Runs every headline through FinBERT; caches results so the 5-minute scoring job runs once |
| 4th | `src/features.py` | Collapses headline scores into one row per day and builds the model's inputs under the no-future-information rule |
| 5th | `src/analysis.py` | The statistics: correlations with p-values, regressions with autocorrelation-robust errors, the sentiment-tercile table, subgroup splits |
| 6th | `src/model.py` | Walk-forward model training, the baselines, the paired significance test, the illustrative backtest |
| 7th | `notebooks/01–04` | The narrative versions of all of the above, executed with outputs visible |
| 8th | `tests/` | 29 tests, the crown jewel being `test_leakage.py` |

`src/plots.py` makes the figures; `reports/deck.md` is the findings deck.

### The key functions, in plain terms

- **`data.load_headlines()`** — reads the CSV, strips whitespace, drops
  exact duplicate (date, headline) pairs, sorts by date. Boring on purpose;
  every downstream step trusts it.

- **`data.next_session()` / `build_price_frame()`** — the alignment brain.
  For each date, computes what the *next NYSE session* should be (skipping
  weekends, holidays, Hurricane Sandy). If the next date actually present in
  the data isn't that session, the pair is flagged `next_day_ok = False` and
  excluded from every "next-day" claim. This is what makes "next-day return"
  mean what it says.

- **`sentiment.score_headlines()`** — batches headlines through FinBERT,
  getting P(positive), P(negative), P(neutral) for each; the working score
  is P(positive) − P(negative), between −1 and +1. Deterministic, cached.

- **`features.build_model_frame()`** — produces one row per trading day:
  today's average sentiment, share of positive/negative headlines,
  yesterday's sentiment, a 3-day sentiment average, news volume, today's
  return, recent volatility — plus the *label*, tomorrow's direction.
  Features look backward; only the label looks forward. That separation is
  the entire leakage design, and it's what the tests attack.

- **`model.walk_forward()`** — trains on days 1…N, predicts the next block,
  extends the training window, repeats. No shuffling, ever — shuffling
  time-series data is how you accidentally train on Thursday to predict
  Tuesday. Feature scaling is re-fit inside each fold on training rows only.

- **`model.baseline_results()` / `accuracy_vs_baseline_test()`** — evaluates
  "always up," "same as today," and a coin flip on the *identical* test days
  as the model, then runs McNemar's test — the statistically correct way to
  ask "is classifier A actually better than classifier B on the same data,
  or is the gap luck?"

### What "leakage-audited" means — the concept to nail

**Leakage** = any path by which information from the future contaminates
what the model knows when it makes a prediction. It's the number-one way
financial ML results end up fake. Three classic leaks and this project's
countermeasures:

1. **Feature leaks** (a rolling average centered on today quietly includes
   tomorrow). → Every feature is built with backward-looking operations
   (`shift`, trailing windows).
2. **Split leaks** (random train/test splits put Wednesday in training and
   Tuesday in test). → Walk-forward validation; training data always
   strictly precedes test data.
3. **Preprocessing leaks** (normalizing with the full dataset's mean, which
   includes the test period). → Scalers are fit per-fold on training rows
   only, inside a pipeline.

The audit part: **`tests/test_leakage.py` doesn't trust the code review —
it attacks the pipeline.** It picks a cutoff date, garbles every price and
sentiment value *after* the cutoff (flips signs, injects noise), rebuilds
all features, and asserts that every feature value *before* the cutoff is
byte-for-byte identical. If any feature secretly touched the future, the
corruption would change it and the test would fail. A second test confirms
the *label* does change under corruption — proving the attack itself works.

### How to run it end to end

```bash
pip install -r requirements.txt
pytest                      # verify the pipeline logic — 29 tests, ~5 seconds
jupyter lab notebooks/      # run 01 → 04 in order
```

Notebook 01 pulls the data from Kaggle automatically. Notebook 02 needs
~5 minutes of CPU the first time (FinBERT scoring, cached afterward).
Notebooks 03–04 also run standalone off the committed daily aggregates.

### If someone technical asks to see the code

Point them at **`tests/test_leakage.py` first** — it communicates the
project's standards in thirty lines, and it's the part most portfolio
projects don't have. Then `src/data.py` (`build_price_frame`) for the
NYSE-calendar alignment, then `src/model.py` (`walk_forward`) for the
validation design. The notebooks are the narrative; the tests are the proof.

---

## Section 4: Questions you'll get, and answers you can say out loud

**"So are you predicting stock prices?"**

> "No — and that's deliberate. Predicting prices at daily level is close to
> impossible, and anyone claiming otherwise should worry you. I answered a
> different question: is there *any* measurable statistical relationship
> between news tone and the next day's move? The answer is 'a faint trace,
> too weak to use' — and the project's value is that I can show that
> conclusion is airtight: no data leakage, honest baselines, proper
> significance testing. It's a study of how to evaluate a signal, using a
> famous question as the subject."

**"Why FinBERT instead of a normal sentiment tool?"**

> "Financial language breaks generic sentiment tools. 'Bullish' reads as
> aggressive, 'crushed estimates' reads as violent, 'short' reads as
> negative height. FinBERT is a language model fine-tuned on financial
> text, so it scores headlines the way a finance reader would. I verified
> it behaves sensibly before trusting it: its average score drops exactly
> where history says it should — the 2008 crisis, the COVID crash, the
> 2022 bear market."

**"What's data leakage, and why does it matter here?"**

> "Leakage is accidentally letting a model see the future during training —
> like normalizing with statistics that include the test period, or letting
> a rolling average peek one day ahead. In financial ML it's endemic,
> because even a whiff of tomorrow's data produces beautiful fake accuracy.
> If I hadn't controlled for it, I could honestly-looking-ly report a model
> that 'predicts the market' — and it would be worthless. I went a step
> further than being careful: there's an automated test that corrupts all
> post-cutoff data and verifies no feature before the cutoff changes. The
> pipeline provably can't see the future."

**"How confident are you in the findings?"**

> "Very confident in the negative findings — no exploitable daily signal;
> the model doesn't beat 'always up.' Those are robust across model types
> and time periods, and they match what the efficient-markets literature
> predicts. Appropriately *unconfident* in the positive whisper: the
> next-day correlation sits at p ≈ 0.05, and I looked at several subgroups,
> which inflates the chance a borderline result is noise. I'd call it
> 'suggestive, consistent with published research, not established by this
> dataset.' Being calibrated about that difference is the skill the project
> demonstrates."

**"What would you do differently with more data or resources?"**

> "Four upgrades, in order: timestamps — knowing whether a headline ran
> before or after the close would let me measure reaction speed instead of
> assuming it; stock-level data — sentiment effects are better documented
> in the cross-section than at index level; richer sources — full articles,
> earnings calls, more outlets, with source metadata to control for the
> changing mix; and intraday prices, to separate the overnight reaction
> from the next day's drift. The evaluation framework wouldn't change —
> that's the reusable part."

**"Walk me through this function."** *(pick `walk_forward` — it's the one
they'll likely point at)*

> "It simulates honest forecasting. Sort all days chronologically; train on
> the first 500, predict the next block; grow the training window, predict
> the next block; repeat five times. Every prediction is made by a model
> that has never seen that day or anything after it. Scaling is re-fit
> inside each fold on training rows only, so even the normalization can't
> leak test-set information. Then the baselines get evaluated on exactly
> the same test days, and McNemar's test tells me whether the model's
> accuracy difference is real or luck. It came back luck — p = 0.45."

**"Your backtest beat buy-and-hold. Isn't that a trading edge?"**

> "No — and that's my favorite exhibit. The classifier behind that curve
> has an AUC of 0.50, meaning zero ability to rank days. It was in the
> market 90% of the time and got lucky sitting out a few bad days across
> 16 years. Before costs, before slippage, before taxes. That's precisely
> how spurious trading systems are marketed — a pretty equity curve on top
> of statistical noise — and having built one by accident, under honest
> conditions, is the best demonstration of why I don't trust backtests
> that aren't backed by a significant, mechanism-backed signal."

---

## Glossary

- **Sentiment scoring** — assigning a number to text for how
  positive/negative it reads. Here: FinBERT's P(positive) − P(negative),
  from −1 to +1 per headline.
- **FinBERT** — a BERT language model fine-tuned on financial text
  (ProsusAI). Understands finance-specific tone.
- **Leakage (lookahead bias)** — future information contaminating a model's
  inputs at training/prediction time, inflating measured performance.
- **Walk-forward validation** — testing chronologically: train on the past,
  predict a later block, expand, repeat. The time-series alternative to
  random train/test splits.
- **Baseline** — the simple strategy a model must beat to matter. Here,
  "always predict up" (53.9% accurate), because the index drifts upward.
- **Majority class** — the more common outcome (here, "up" days). Predicting
  it always is often embarrassingly hard to beat.
- **Backtesting** — replaying a decision rule over history to see how it
  would have done. Illustrative at best; without costs and significance
  testing it's marketing, not evidence.
- **Significance testing / p-value** — the probability of seeing a result
  at least this extreme by luck alone if there is no real effect. p = 0.052
  means "right on the conventional edge; do not build a shrine to it."
- **Newey–West (HAC) errors** — regression error bars corrected for the
  autocorrelation and volatility clustering that daily returns always have;
  without them, p-values are overconfident.
- **McNemar's test** — paired significance test for whether two classifiers
  evaluated on the same days genuinely differ, based only on the days where
  they disagree.
- **AUC** — probability the model ranks a random "up" day above a random
  "down" day. 0.5 = coin flip, 1.0 = perfect. Ours: 0.50.
- **Tercile** — one-third of the sample after sorting. "Top sentiment
  tercile" = the third of days with the most positive headlines.
- **Close-to-close return** — percent change from one day's closing price
  to the next day's closing price. The only return type available here.
- **Reverse causality** — B causing A when you hoped A caused B. Here:
  the day's market move causing the day's headline tone.
