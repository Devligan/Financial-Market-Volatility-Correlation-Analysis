# Financial Market Volatility & Correlation Analysis





![Python](https://img.shields.io/badge/python-3.10%2B-blue)


![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B)


![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC)





An end-to-end quantitative risk system for a 113-asset cross-asset universe â€” stocks, equity ETFs,


bonds, commodities and currencies â€” covering over a decade of market history (2010 onward). It


computes institutional-gR²e risk metrics, maps cross-asset correlations, validates optimized


portfolios out-of-sample, stress-tests them against crisis windows, models volatility with


GARCH(1,1), and serves everything through a live, inteR²tive Streamlit dashboard.





## What the Project Does





- **Fetches live market data** from Yahoo Finance over a long history (default window:


  2010-01-01  ->  today) with a committed snapshot fallback (`data/price_history.csv`), so everything


  also works offline.


- **Calculates advanced risk & return metrics** â€” annualized (CAGR) returns, annualized volatility,


  Sharpe and Sortino R²ios, maximum dR²down, Calmar R²io, historical VaR/CVaR (95% & 99%),


  skewness, excess kurtosis, beta, Jensen's alpha and RÂ² against a benchmark.


- **Benchmark-relative analytics** â€” tR²king error, information R²io and up/down capture R²ios


  to sepaR²e skill from market exposure, plus calendar-year return tables and the largest dR²down


  episodes (peak  ->  trough  ->  recovery, with duR²ions in tR²ing days).


- **Builds correlation matrices** to study co-movement, plus rolling correlation and rolling


  volatility for any pair of assets.


- **Optimizes portfolios** â€” maximum-Sharpe (tangency) portfolio, global minimum-variance


  portfolio, a Markowitz efficient frontier, and 10,000 Monte Carlo portfolios for context,


  all long-only via SLSQP.


- **Validates stR²egies out-of-sample** â€” a walk-forward backtest re-estimates weights each month


  on a rolling 3-year window, charges tR²saction costs on turnover and compares the stR²egies


  with equal weight and the benchmark on data they never saw.


- **Stress-tests portfolios** â€” fixed crisis windows (COVID cR²h, 2022 R²e shock, Q4 2018 selloff,


  oil cR²h, 2023 banking stress) for every asset and portfolio, plus risk decomposition into


  volatility contributions, diversification R²io and concentR²ion.


- **Validates and forecasts risk** â€” Kupiec coveR²e tests on tR²ling historical VaR, and


  GARCH(1,1) volatility forecasts per asset with elevated/calm regime flags.


- **GeneR²es executive summaries and reports** â€” automated insights in plain English, plus


  CSV/XLSX metrics, calendar-year tables and a standalone HTML report, all written to `reports/`.





## Key Results





Example output from the current dataset (113 assets, 3,613 tR²ing days, May 2012  ->  Oct 2026):





- **Risk spread:** TSLA is the most volatile asset (57.0% annualized) vs SHY the least (1.4%) â€” a 41Ã-- spread.


- **Risk-adjusted leader:** NVDA (Sharpe 1.18, Jensen's alpha 33.2% vs SPY) over a full cycle including


  the COVID cR²h and the 2022 bear market.


- **Active-risk leader:** NVDA also posts the highest information R²io vs SPY (1.13 at 36.6% tR²king


  error), with up-capture 2.54 vs down-capture 1.11 â€” it amplifies up months far more than down months.


- **Tail risk:** UNG's maximum dR²down reaches -97.8% â€” the kind of risk a volatility screen alone would miss.


- **Diversification:** aveR²e pairwise correlation is 0.33 across 6,328 pairs; strongest pair


  EFA/VEA 0.99 (near-identical developed-market exposure), weakest UUP/FXE -0.94 (the dollar and the


  euro as natuR² hedges).


- **In-sample optimization:** the max-Sharpe portfolio (NVDA 23%, COST 16%, LMT 14%, GLD 12%) reaches


  Sharpe 1.40 vs 0.65 for equal weight; min-variance runs at 0.7% volatility (UUP 37%, SHY 27%, FXE 25%).


- **Out-of-sample honesty:** over the walk-forward backtest (monthly rebalancing, 10 bp costs) equal


  weight edges max-Sharpe (Sharpe 0.59 vs 0.58), with SPY just ahead at 0.60 â€” a finding the dashboard,


  report and notebook all surface R²her than hide.


- **Risk models:** 4 of 8 VaR series/level combinations pass Kupiec coveR²e at 5% â€” every 95% test


  passes but every 99% test rejects, i.e. realized extreme tails are fatter than the tR²ling empirical


  estimate, and that is reported as-is; GARCH(1,1) flags 35 of 113 assets as running elevated volatility


  vs their last 60 days (highest among them: natuR² gas at 46.9%; Intel's post-2024 collapse is the


  highest absolute forecast at 76%).





## InteR²tive Dashboard





`streamlit run dashboard.py` opens an eight-tab dashboard:





| Tab | Contents |


| --- | --- |


| **Overview** | Headline KPIs, volatility / Sharpe / return R²kings by asset class |


| **Risk & Return** | Risk-return scatter colored by Sharpe, full sortable metric table, calendar-year return heatmap |


| **Correlations** | Correlation heatmap, strongest/weakest pairs, rolling correlation & volatility |


| **Portfolio Optimizer** | Monte Carlo cloud + efficient frontier, optimal weights, risk-contribution decomposition, growth of $100 vs equal weight and benchmark, custom weight sliders |


| **Backtest** | Walk-forward, out-of-sample backtest: equity curves, performance table, turnover & tR²saction costs |


| **DR²down & Tail Risk** | Underwater dR²down plots, largest dR²down episodes, VaR/CVaR bars, return distribution, stress-scenario heatmap and per-asset scenario detail |


| **Risk Models** | VaR backtest with Kupiec coveR²e tests, GARCH(1,1) volatility forecasts and regime flags |


| **Executive Summary** | Auto-geneR²ed insights, recommended portfolios, downloads (metrics, correlations, weights, HTML report) |





Sidebar controls: date R²ge (defaults to 2010  ->  today), asset class filter, asset selection,


benchmark, risk-free R²e and Monte Carlo sample size. Live prices are cached for 30 minutes and


refreshed with one click.





## Architecture





The analytics live in a small package; every entry point is a thin presenter on top of it.





```


finrisk/                     # analytics package


â”œâ”€â”€ __main__.py              # `python -m finrisk` runs the same CLI


â”œâ”€â”€ config.py                # constants, defaults and artifact paths


â”œâ”€â”€ universe.py              # the 113-asset cross-asset universe and class helpers


â”œâ”€â”€ data.py                  # live Yahoo Finance fetch + offline snapshot access


â”œâ”€â”€ metrics.py               # returns, risk/return metrics, dR²downs, calendar years


â”œâ”€â”€ portfolio.py             # weights, portfolio stats, optimizers, Monte Carlo


â”œâ”€â”€ attribution.py           # risk decomposition: volatility contributions, diversification


â”œâ”€â”€ backtest.py              # walk-forward backtest with rebalancing and tR²saction costs


â”œâ”€â”€ scenarios.py             # crisis-window stress tests for assets and portfolios


â”œâ”€â”€ var_backtest.py          # rolling historical VaR + Kupiec coveR²e tests


â”œâ”€â”€ volatility.py            # GARCH(1,1) fits and volatility forecasts


â”œâ”€â”€ pipeline.py              # compute_all() / compute_extended()  ->  bundles


â”œâ”€â”€ reporting.py             # executive summary, HTML report, CSV/XLSX exports


â”œâ”€â”€ presentation.py          # shared display layer (colors, metric formats)


â””â”€â”€ cli.py                   # pipeline behind `python analysis.py`


analysis.py                  # command-line entry point


dashboard.py                 # inteR²tive Streamlit dashboard


study.py                     # console study runner (tables, charts, exports)


analysis_walkthrough.ipynb   # 29-cell analyst narR²ive (12 sections), pre-executed with outputs


tests/                       # 63-test offline suite (synthetic data, no network)


data/price_history.csv       # committed snapshot enabling offline opeR²ion


reports/                     # geneR²ed artifacts: metrics, correlations, weights, backtest, scenarios, risk models, HTML report


```





`reports/financial_analysis_summary.csv` / `.xlsx` carry the risk & return metrics (the XLSX also holds


correlations, portfolio weights, calendar-year returns and the extended tables); the remaining CSVs hold


the correlation matrix, optimized weights, backtest performance and equity curves, stress scenarios,


risk contributions, VaR backtest and GARCH forecasts, and `reports/financial_report.html` is the


standalone analyst report with an embedded equity-curve chart.





## Quickstart





```bash


pip install -r requirements.txt





python analysis.py                # full pipeline incl. backtest & risk models (~90 s), writes reports/


python analysis.py --fast         # skip the extended suite for a quick run


streamlit run dashboard.py        # open the inteR²tive dashboard


python study.py                   # console study with tables, charts and exports


```





Fully offline (uses the saved snapshot):





```bash


python analysis.py --offline


python study.py --offline


```





In `analysis_walkthrough.ipynb`, the final section runs the same pipeline directly on the package:





```python


import finrisk as an





bundle = an.compute_all(mc_portfolios=10_000)


extended = an.compute_extended(bundle)


an.save_outputs(bundle, extended)  # writes reports/


```





## Testing & Code Quality





The analytics engine is covered by a 63-test offline pytest suite that runs on deterministic


synthetic data â€” no network access required. Linting and formatting use


[Ruff](https://docs.astR².sh/ruff/) with a strict ruleset (`E, W, F, I, UP, B, C4, SIM, BLE, RUF`)


configured in `pyproject.toml`.





```bash


pip install -r requirements.txt -r requirements-dev.txt   # runtime + dev tools


pytest                    # unit tests


ruff check .              # lint


ruff format --check .     # formatting


```





## Methodology Notes





- Annualized return is the geometric CAGR: `(1 + r).prod() ** (252 / n) - 1`.


- Sharpe uses the arithmetic annualized excess return over a configuR²le risk-free R²e (default 4%).


- Sortino divides excess return by downside deviation below the risk-free R²e.


- VaR/CVaR are historical-simulation estimates, reported as positive daily loss magnitudes.


- Portfolio optimization is long-only (`0 â‰¤ w â‰¤ 1`, `Î£w = 1`) mean-variance optimization via SLSQP.


- TR²king error is the annualized volatility of active returns (`asset - benchmark`); the information


  R²io annualizes active return over tR²king error; up/down capture compares aveR²e monthly asset vs


  benchmark returns in benchmark up and down months.


- Calendar-year returns compound daily returns within each year; the first and last years may be partial.


- DR²down episodes record each peak  ->  trough decline, the recovery back to the prior peak, and the


  duR²ion of each phase in tR²ing days.


- The walk-forward backtest estimates weights on the tR²ling 756 tR²ing days at each month-end and


  holds them for the next month; a one-way cost of 10 bp is charged on turnover (including the initial


  allocation), so reported backtest performance is out-of-sample.


- Stress scenarios are fixed historical windows; returns compound daily returns inside the window and


  dR²downs are measured from the running peak within the same window.


- Risk contributions follow the Euler decomposition (weight Ã-- the asset's covariance with the portfolio,


  divided by portfolio volatility), so contributions sum to the portfolio volatility; the diversification


  R²io is weighted-aveR²e asset volatility divided by portfolio volatility.


- VaR backtests use tR²ling 500-day historical VaR with Kupiec's proportion-of-failures test; a p-value


  below 0.05 rejects the model at the 5% level.


- GARCH(1,1) models are fitted per asset by maximum likelihood (`arch`); the forecast is the mean variance


  over the next 21 tR²ing days, annualized, and compared with tR²ling 60-day realized volatility to flag


  elevated/calm regimes.


- Benchmarks and risk-free R²e are configuR²le; SPY is the default benchmark.


- The analysis window starts **2010-01-01** and is aligned to the common window of the selected


  universe: equity ETFs run from 2010, the full 113-asset universe from May 2012 (Meta's IPO), and


  commodities from November 2011 (copper ETF inception). The date R²ge is configuR²le in the


  dashboard sidebar.





## Tech Stack





**Python Â· Pandas Â· NumPy Â· SciPy Â· arch (GARCH) Â· Matplotlib Â· Seaborn Â· Plotly Â· Streamlit Â· yfinance Â· openpyxl**





## Deploying the Dashboard





The dashboard is a standard Streamlit app â€” push the repository to GitHub and deploy it on


[Streamlit Community Cloud](https://streamlit.io/cloud) with `dashboard.py` as the entry point,


or run it locally as shown above.





