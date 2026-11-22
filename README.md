# MAC 312 Project 4: Classification with Unequal Error Costs and Rejection

Spam filter with three actions: **deliver** (0), **spam folder** (1), **hold for human inspection** (2).
Real email sent to spam costs `c_fp = 10`, delivered spam costs `c_fn = 1`, inspection costs `c_I`.
The team plan is in [`Project4_Team_Work_Plan.pdf`](Project4_Team_Work_Plan.pdf).

## Reproduce everything

```bash
python -m pip install -r requirements.txt
python -m pytest                 # unit tests (downloads nothing; Spambase test skips if not cached)
python -m project4.run_all       # downloads Spambase once, regenerates every figure and table
```

Figures land in `project4/figures/` (PDF + PNG) and tables in `project4/results/` (CSV + LaTeX).
The report in `report/` includes them directly. Build it with
`cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main`,
or import the repo into Overleaf with its GitHub sync.

The 15-slide presentation is [`slides/slides.pdf`](slides/slides.pdf). Its source `slides/slides.tex`
uses the same figures; build it with `cd slides && pdflatex slides && pdflatex slides`.



## Layout and owners

```
project4/
  data.py          C1 (M6)  load_spambase, make_gaussian, resample_to_prior, split, preprocess
  decision.py      C1 (M6)  LossMatrix, thresholds, bayes_action  (actions 0/1/2)
  metrics.py       C1 (M6)  confusion_with_reject, realized_loss, accuracy, reject_rate, summarize
  config.py        C1 (M6)  shared constants: costs, seeds, output folders
  plotting.py      C1 (M6)  shared figure style and colour order
  analytic.py      T6 (M6)  closed-form 1-D Gaussian model: accuracy, cost, rejection, Bayes risk
  models/gaussian.py    C2 (M3)  GaussianClassifier(shared_cov, reg).fit(X, y).predict_proba(X)
  models/logistic.py    C3 (M2)  LogisticRegression(l2).fit(X, y).predict_proba(X)
  calibration.py        C4 (M5)  log_loss, brier, reliability_bins, ece, temperature_scale, platt_fit
  prior_shift.py        T5 (M5)  adjust_posterior, shifted_thresholds, bayes_action_shifted
  experiments/
    t5_prior_correction.py T5 (M5)  figure of the prior-shift correction
    t6_analytic.py        T6 (M6)  analytic figures and the accuracy-vs-cost table
    task2_cost_sweep.py   C3 (M2)
    task3_calibration.py  C4 (M5)
    task4_prior_shift.py  C5 (M1)
    task5_inspection.py   C6 (M4)
  run_all.py       C1 (M6)  runs every experiment's main(); missing ones are skipped
tests/             unit tests (pytest)
report/            LaTeX report: main.tex, notation.tex, one file per section
```

## Interface rules (frozen in week 1)

- `predict_proba(X)` always returns `P(y=1 | x)` as a 1-D array.
- Actions are integers: `0` deliver, `1` spam, `2` inspect. Use `decision.DELIVER/SPAM/INSPECT`.
- `bayes_action(p, L)` delivers if `p <= theta0`, sends to spam if `p >= theta1`, and inspects in between.
  With no inspect band (`c_I >= c_fp*c_fn/(c_fp+c_fn)`) it is the single threshold `c_fp/(c_fp+c_fn)`.
- `accuracy(y, a)` is accuracy on non-inspected emails by default; pass `on_reject="correct"` for
  system accuracy (the inspector is always right).
- Splits: `split(X, y, seed)` for seeds `config.SEEDS` (0-19), stratified 60/20/20. Fit preprocessing
  with `preprocess(split)`, which uses training statistics only. Never choose anything on the test part.
- Every experiment script keeps its configuration at the top, defines `main()`, and saves its figures
  and CSVs through `config.FIGURES_DIR` / `config.RESULTS_DIR`. Report means with standard errors over
  the 20 splits (`metrics.mean_and_se`).

## Workflow

Each member works on their own branch and merges through a pull request approved by their named reviewer:

| Code block | Author | Reviewer |
|---|---|---|
| C1 infrastructure | M6 | M1 |
| C2 Gaussian classifiers | M3 | M2 |
| C3 logistic + cost sweep | M2 | M3 |
| C4 calibration metrics | M5 | M4 |
| C5 prior shift | M1 | M5 |
| C6 inspection curves | M4 | M6 |

Figures and tables come only from scripts, never pasted by hand. Use the macros in
`report/notation.tex` for every symbol.

## Data

UCI Spambase (Hopkins et al., 1999; https://doi.org/10.24432/C53G6X): 4,601 emails, 57 features, 1,813 spam.
`load_spambase()` downloads it into `data/` (git-ignored) and checks its SHA-256.
