"""Interactive Streamlit dashboard for financial risk, correlation and portfolio analysis.



Run with:

    streamlit run dashboard.py



Data flow: sidebar controls -> live Yahoo Finance fetch (30-minute cache) with a

local snapshot fallback -> shared analytics engine (the ``finrisk`` package) -> eight analysis tabs.

"""



from __future__ import annotations



import datetime as dt

from dataclasses import dataclass



import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st



import finrisk as an

from finrisk import presentation as pres



st.set_page_config(page_title="Financial Volatility & Correlation Analysis", page_icon="📈", layout="wide")



DEFAULT_START = dt.date.fromisoformat(an.DEFAULT_START)  # long-horizon default (2010)



CLASS_LABELS = {

    "All": None,

    "Stocks": "STOCK",

    "ETFs": "ETF",

    "Bonds": "BOND",

    "Commodities": "COMMODITY",

    "Currencies": "FX",

}

PORTFOLIO_CHOICES = ["Max Sharpe", "Min Variance", "Equal Weight"]





@dataclass(frozen=True)

class Controls:

    """Sidebar selections for one analysis run."""



    start_date: dt.date

    end_date: dt.date

    tickers: list[str]

    benchmark: str | None

    risk_free: float

    mc_portfolios: int





# ---------------------------------------------------------------------------

# Data access: live fetch with snapshot fallback (cached)

# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner="Downloading market data from Yahoo Finance…")

def load_prices(start: dt.date, end: dt.date, tickers: tuple[str, ...]):

    """Try a live Yahoo Finance fetch for the selected universe; fall back to the snapshot."""

    selected = list(tickers)

    try:

        prices = an.fetch_prices(

            selected,

            start=str(start),

            end=str(end + dt.timedelta(days=1)),

        )

        if set(selected) == set(an.default_tickers()):

            an.save_prices(prices)  # only refresh the snapshot with the complete universe

        return prices, "live", None

    except Exception as exc:  # noqa: BLE001 - network failures must degrade gracefully

        try:

            prices = an.load_prices(tickers=selected)

            prices = prices.loc[str(start) : str(end)]

            if prices.empty:

                raise RuntimeError(f"snapshot has no prices between {start} and {end}")

            return prices, "snapshot", str(exc)

        except Exception as snapshot_exc:  # noqa: BLE001

            return None, "none", f"{exc} | {snapshot_exc}"





@st.cache_data(show_spinner="Running risk analytics…")

def analyze(prices: pd.DataFrame, rf: float, benchmark: str | None, mc_portfolios: int) -> an.AnalysisBundle:

    """Run the shared analytics engine on the selected prices (cached per selection)."""

    return an.compute_all(

        tickers=list(prices.columns),

        prices=prices,

        rf=rf,

        benchmark=benchmark,

        mc_portfolios=mc_portfolios,

    )





@st.cache_data(show_spinner=False)

def cached_backtest(

    tickers: tuple[str, ...],

    start: dt.date,

    end: dt.date,

    rf: float,

    cost_bps: float,

    benchmark: str | None,

) -> an.BacktestResult:

    """Walk-forward backtest for the current selection (cached; the first run can take up to a minute)."""

    prices, _source, _error = load_prices(start, end, tickers)

    returns = an.compute_returns(prices)

    return an.walk_forward_backtest(returns, benchmark=benchmark, lookback=756, cost_bps=cost_bps, rf=rf)





@st.cache_data(show_spinner=False)

def cached_garch(tickers: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:

    """GARCH(1,1) forecasts for the current selection (cached)."""

    prices, _source, _error = load_prices(start, end, tickers)

    returns = an.compute_returns(prices)

    return an.forecast_universe(returns)





# ---------------------------------------------------------------------------

# Formatting & presentation helpers

# ---------------------------------------------------------------------------

def class_color(ticker: str) -> str:

    return pres.CLASS_COLORS.get(an.asset_class(ticker), pres.CLASS_COLORS["OTHER"])





def clean_label(ticker: str) -> str:

    return f"{ticker} · {an.asset_name(ticker)}" if ticker in an.ASSET_UNIVERSE else ticker





def portfolio_weight_map(bundle: an.AnalysisBundle, custom: pd.Series | None = None) -> dict[str, pd.Series]:

    """All portfolio choices (optimized, equal weight, benchmark, optional custom) keyed by label."""

    weights = dict(bundle.portfolio_choices())

    if custom is not None:

        weights["Custom"] = custom

    return weights





def portfolio_comparison(bundle: an.AnalysisBundle, weights: dict[str, pd.Series]) -> pd.DataFrame:

    rows = {}

    for label, series in weights.items():

        stats, _ = an.portfolio_stats(bundle.returns, series, rf=bundle.risk_free)

        rows[label] = {key: stats[key] for key in pres.PORTFOLIO_METRICS}

    return pd.DataFrame(rows).T





def growth_curve(returns: pd.DataFrame, weights) -> pd.Series:

    series = an.portfolio_returns(returns, weights)

    return 100 * (1 + series).cumprod()





# ---------------------------------------------------------------------------

# Sidebar

# ---------------------------------------------------------------------------

def render_sidebar(today: dt.date) -> Controls:

    with st.sidebar:

        st.header("Market Data")

        date_range = st.date_input(

            "Date range",

            value=(DEFAULT_START, today),

            min_value=dt.date(2000, 1, 1),

            max_value=today,

        )

        if isinstance(date_range, tuple | list) and len(date_range) == 2:

            start_date, end_date = date_range

        elif isinstance(date_range, dt.date):

            start_date = end_date = date_range

        else:

            start_date, end_date = DEFAULT_START, today



        if st.button("Refresh market data", use_container_width=True):

            # Clear every cache layer so a refresh cannot serve stale analytics.

            load_prices.clear()

            analyze.clear()

            cached_backtest.clear()

            cached_garch.clear()

            st.rerun()



        st.divider()

        st.header("Analysis")

        class_choice = st.radio("Asset class", list(CLASS_LABELS), index=0)

        options = an.tickers_of_class(CLASS_LABELS[class_choice])

        tickers = st.multiselect(

            "Assets",

            options=options,

            default=options,

            format_func=clean_label,

            key=f"assets_{class_choice}",

        )



        if tickers:

            default_benchmark = "SPY" if "SPY" in tickers else tickers[0]

            benchmark = st.selectbox(

                "Benchmark",

                options=tickers,

                index=tickers.index(default_benchmark),

                format_func=clean_label,

                key=f"benchmark_{class_choice}",

            )

        else:

            benchmark = None



        rf_pct = st.slider("Risk-free rate (%)", 0.0, 10.0, 4.0, 0.25)

        mc_portfolios = st.slider("Monte Carlo portfolios", 2000, 20000, 8000, 1000)



    return Controls(start_date, end_date, list(tickers), benchmark, rf_pct / 100.0, mc_portfolios)





# ---------------------------------------------------------------------------

# Header, status and headline metric cards

# ---------------------------------------------------------------------------

def render_data_status(

    bundle: an.AnalysisBundle,

    available: list[str],

    prices: pd.DataFrame,

    source: str,

    fetch_error: str | None,

) -> None:

    with st.sidebar:

        if source == "live":

            st.success(f"Live data · cached 30 min · fetched {bundle.fetched_at:%H:%M}")

        else:

            st.warning("Offline snapshot in use (live fetch failed).")

            if fetch_error:

                st.caption(f"Fetch error: {fetch_error[:160]}")

        st.caption(

            f"{len(available)} assets · {len(bundle.returns)} trading days · "

            f"{prices.index[0].date()} → {prices.index[-1].date()}"

        )





def render_metric_cards(bundle: an.AnalysisBundle, metrics: pd.DataFrame, n_assets: int) -> None:

    col1, col2, col3, col4 = st.columns(4)

    most_volatile = metrics["Annualized_Volatility"].idxmax()

    col1.metric("Most Volatile", most_volatile, f"{metrics.loc[most_volatile, 'Annualized_Volatility']:.1%}")

    best_sharpe = metrics["Sharpe_Ratio"].idxmax()

    col2.metric("Best Sharpe Ratio", best_sharpe, f"{metrics.loc[best_sharpe, 'Sharpe_Ratio']:.2f}")

    best_sortino = metrics["Sortino_Ratio"].idxmax()

    col3.metric("Best Sortino Ratio", best_sortino, f"{metrics.loc[best_sortino, 'Sortino_Ratio']:.2f}")

    avg_corr = an.average_correlation(bundle.correlation)

    col4.metric(

        "Avg Correlation",

        "All Assets" if n_assets > 1 else "—",

        f"{avg_corr:.3f}" if not np.isnan(avg_corr) else "n/a",

        help="Mean pairwise correlation across the selected universe. Lower values mean better diversification.",

    )





# ---------------------------------------------------------------------------

# Tab 1 — Overview

# ---------------------------------------------------------------------------

def render_overview(metrics: pd.DataFrame) -> None:

    st.subheader("Annualized Volatility")

    vol_data = metrics["Annualized_Volatility"].sort_values()

    fig = go.Figure(

        go.Bar(

            x=vol_data.values,

            y=vol_data.index,

            orientation="h",

            marker_color=[class_color(t) for t in vol_data.index],

            text=[f"{value:.1%}" for value in vol_data.values],

            textposition="outside",

            customdata=[an.asset_class(t) for t in vol_data.index],

            hovertemplate="%{y} · %{customdata}<br>Annualized volatility: %{x:.2%}<extra></extra>",

        )

    )

    fig.update_layout(height=max(380, 22 * len(vol_data)), margin={"r": 80})

    fig.update_xaxes(tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)



    left, right = st.columns(2)

    with left:

        st.subheader("Sharpe Ratio")

        sharpe_data = metrics["Sharpe_Ratio"].sort_values()

        fig = go.Figure(

            go.Bar(

                x=sharpe_data.values,

                y=sharpe_data.index,

                orientation="h",

                marker_color=[class_color(t) for t in sharpe_data.index],

                text=[f"{value:.2f}" for value in sharpe_data.values],

                textposition="outside",

            )

        )

        fig.add_vline(x=1, line_dash="dash", line_color="green", annotation_text="Good performance (1.0)")

        fig.update_layout(height=max(360, 20 * len(sharpe_data)), margin={"r": 60})

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Annualized Return (CAGR)")

        ret_data = metrics["Annualized_Return"].sort_values()

        fig = go.Figure(

            go.Bar(

                x=ret_data.values,

                y=ret_data.index,

                orientation="h",

                marker_color=[class_color(t) for t in ret_data.index],

                text=[f"{value:.1%}" for value in ret_data.values],

                textposition="outside",

            )

        )

        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        fig.update_layout(height=max(360, 20 * len(ret_data)), margin={"r": 60})

        fig.update_xaxes(tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)



    st.caption("Colors — blue: stocks · purple: ETFs · green: bonds · orange: commodities · teal: currencies")

    with st.expander("Average risk profile by asset class"):

        st.dataframe(pres.metric_style(an.class_summary(metrics)), use_container_width=True)





# ---------------------------------------------------------------------------

# Tab 2 — Risk & Return

# ---------------------------------------------------------------------------

def render_risk_and_return(bundle: an.AnalysisBundle) -> None:

    metrics = bundle.metrics

    st.subheader("Risk vs Return")

    scatter_df = metrics.reset_index(names="Ticker")

    scatter_df["Class"] = [an.asset_class(t) for t in scatter_df["Ticker"]]

    fig = px.scatter(

        scatter_df,

        x="Annualized_Volatility",

        y="Annualized_Return",

        color="Sharpe_Ratio",

        symbol="Class",

        color_continuous_scale="Viridis",

        hover_data=["Sortino_Ratio", "Max_Drawdown", "VaR_95", "Calmar_Ratio"],

        text="Ticker",

        height=560,

    )

    fig.update_traces(textposition="top center", marker={"size": 13, "line": {"width": 1, "color": "white"}})

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_xaxes(tickformat=".0%")

    fig.update_yaxes(tickformat=".0%")

    fig.update_layout(coloraxis_colorbar_title="Sharpe")

    st.plotly_chart(fig, use_container_width=True)

    st.info(

        "**Top-left**: low risk, positive return (ideal) · **bottom-left**: low risk, negative return · "

        "**right**: high risk. Marker color encodes the Sharpe ratio."

    )



    st.subheader("Full Risk & Return Table")

    display = metrics.sort_values("Sharpe_Ratio", ascending=False)

    st.dataframe(pres.metric_style(display), use_container_width=True, height=420)

    st.download_button(

        "Download metrics CSV",

        metrics.round(4).to_csv().encode("utf-8"),

        f"risk_return_metrics_{dt.date.today()}.csv",

        "text/csv",

    )



    st.subheader("Calendar-Year Returns")

    yearly = an.calendar_year_returns(bundle.returns)

    annotate = len(yearly.columns) <= 12

    fig = go.Figure(

        go.Heatmap(

            z=yearly.to_numpy(),

            x=list(yearly.columns),

            y=[str(year) for year in yearly.index],

            colorscale="RdYlGn",

            zmid=0,

            text=[[f"{value:.0%}" for value in row] for row in yearly.to_numpy()] if annotate else None,

            texttemplate="%{text}" if annotate else None,

            textfont={"size": 9},

            hovertemplate="%{y} · %{x}: %{z:.2%}<extra></extra>",

        )

    )

    fig.update_layout(height=max(320, 28 * len(yearly.index)))

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Total return per calendar year (daily compounding). The first and last years may be partial.")





# ---------------------------------------------------------------------------

# Tab 3 — Correlations

# ---------------------------------------------------------------------------

def render_correlations(bundle: an.AnalysisBundle, available: list[str]) -> None:

    if len(available) < 2:

        st.info("Select at least two assets to analyze correlations.")

        return



    st.subheader("Correlation Matrix")

    corr = bundle.correlation

    labels = list(corr.columns)

    annotate = len(labels) <= 50  # keep large-universe matrices readable

    fig = go.Figure(

        go.Heatmap(

            z=corr.values,

            x=labels,

            y=labels,

            colorscale="RdBu_r",

            zmid=0,

            zmin=-1,

            zmax=1,

            text=corr.round(2).values if annotate else None,

            texttemplate="%{text}" if annotate else None,

            textfont={"size": 9},

        )

    )

    fig.update_layout(height=max(480, 32 * len(labels)))

    st.plotly_chart(fig, use_container_width=True)



    (hi_a, hi_b, hi_val), (lo_a, lo_b, lo_val) = an.correlation_extremes(corr)

    col1, col2 = st.columns(2)

    col1.success(f"Highest: **{hi_a} ↔ {hi_b}** ({hi_val:.2f})")

    col2.info(f"Lowest: **{lo_a} ↔ {lo_b}** ({lo_val:.2f})")



    st.subheader("Rolling Analysis")

    control1, control2, control3 = st.columns(3)

    with control1:

        asset_a = st.selectbox("Asset A", available, index=0, key="roll_a")

    with control2:

        asset_b = st.selectbox("Asset B", available, index=min(1, len(available) - 1), key="roll_b")

    with control3:

        window = st.select_slider("Rolling window (days)", options=[21, 30, 60, 90, 120, 180, 252], value=90)



    roll_corr = an.rolling_correlation(bundle.returns, asset_a, asset_b, window=window)

    fig = go.Figure(

        go.Scatter(

            x=roll_corr.index,

            y=roll_corr.values,

            mode="lines",

            line={"color": pres.ACCENT},

            name=f"{asset_a} vs {asset_b}",

        )

    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    if not roll_corr.empty:

        fig.add_hline(

            y=roll_corr.mean(),

            line_dash="dash",

            line_color="green",

            annotation_text=f"mean {roll_corr.mean():.2f}",

        )

    fig.update_layout(

        title=f"{window}-day rolling correlation — {asset_a} vs {asset_b}",

        yaxis_range=[-1.05, 1.05],

        height=380,

    )

    st.plotly_chart(fig, use_container_width=True)



    st.subheader(f"{window}-day Rolling Volatility")

    fig = go.Figure()

    for ticker in available:

        rolling = an.rolling_volatility(bundle.returns[[ticker]], window=window)[ticker] * 100

        fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines", name=ticker, opacity=0.8))

    fig.update_layout(height=420, yaxis_title="Annualized volatility (%)")

    st.plotly_chart(fig, use_container_width=True)





# ---------------------------------------------------------------------------

# Tab 4 — Portfolio Optimizer

# ---------------------------------------------------------------------------

def render_portfolio_optimizer(bundle: an.AnalysisBundle, available: list[str]) -> None:

    st.subheader("Mean-Variance Optimization")

    st.caption(

        "A cloud of random long-only portfolios (Monte Carlo) with the Markowitz efficient frontier, "

        "the maximum-Sharpe and minimum-variance portfolios, and a growth comparison against equal weight."

    )



    mc = bundle.monte_carlo

    plot_mc = mc.sample(min(4000, len(mc)), random_state=0)

    fig = px.scatter(

        plot_mc,

        x="Annualized_Volatility",

        y="Annualized_Return",

        color="Sharpe_Ratio",

        color_continuous_scale="Viridis",

        opacity=0.35,

        height=560,

    )

    fig.update_traces(marker={"size": 6}, hovertemplate="vol %{x:.2%} · return %{y:.2%}<extra></extra>")

    if not bundle.frontier.empty:

        fig.add_trace(

            go.Scatter(

                x=bundle.frontier["Annualized_Volatility"],

                y=bundle.frontier["Annualized_Return"],

                mode="lines",

                name="Efficient frontier",

                line={"color": "crimson", "width": 3},

            )

        )

    marker_specs = [

        ("Max Sharpe", bundle.max_sharpe, "star", "red"),

        ("Min Variance", bundle.min_variance, "diamond", "black"),

        ("Equal Weight", bundle.equal_weight, "square", pres.ACCENT),

    ]

    for label, weights, symbol, color in marker_specs:

        stats, _ = an.portfolio_stats(bundle.returns, weights, rf=bundle.risk_free)

        fig.add_trace(

            go.Scatter(

                x=[stats["Annualized_Volatility"]],

                y=[stats["Annualized_Return"]],

                mode="markers+text",

                name=label,

                text=[label],

                textposition="bottom center",

                marker={"size": 15, "symbol": symbol, "color": color, "line": {"width": 1, "color": "white"}},

            )

        )

    fig.update_xaxes(tickformat=".0%")

    fig.update_yaxes(tickformat=".0%")

    fig.update_layout(coloraxis_colorbar_title="Sharpe")

    st.plotly_chart(fig, use_container_width=True)



    st.subheader("Portfolio Weights")

    portfolio_choice = st.radio("Portfolio", [*PORTFOLIO_CHOICES, "Custom"], horizontal=True)

    custom_weights = None

    if portfolio_choice == "Custom":

        if len(available) > 12:

            st.caption("Tip: deselect assets in the sidebar to reduce the number of sliders.")

        weight_columns = st.columns(3)

        raw = {}

        for index, ticker in enumerate(available):

            with weight_columns[index % 3]:

                raw[ticker] = st.slider(ticker, 0, 100, round(100 / len(available)), 1, key=f"custom_{ticker}")

        try:

            custom_weights = pd.Series(an.normalize_weights(raw.values()), index=available, name="Custom")

        except ValueError:

            st.warning("Custom portfolio needs at least one non-zero weight.")



    weight_map = portfolio_weight_map(bundle, custom_weights)

    chosen_weights = weight_map.get(portfolio_choice)



    if chosen_weights is not None:

        stats, _ = an.portfolio_stats(bundle.returns, chosen_weights, rf=bundle.risk_free)

        w1, w2, w3, w4 = st.columns(4)

        w1.metric("Expected CAGR", f"{stats['Annualized_Return']:.1%}")

        w2.metric("Volatility", f"{stats['Annualized_Volatility']:.1%}")

        w3.metric("Sharpe", f"{stats['Sharpe_Ratio']:.2f}")

        w4.metric("Max Drawdown", f"{stats['Max_Drawdown']:.1%}")



        weight_series = chosen_weights[chosen_weights > 0.001].sort_values()

        fig = go.Figure(

            go.Bar(

                x=weight_series.values,

                y=weight_series.index,

                orientation="h",

                marker_color=[class_color(t) for t in weight_series.index],

                text=[f"{value:.1%}" for value in weight_series.values],

                textposition="outside",

            )

        )

        fig.update_layout(height=max(320, 24 * len(weight_series)), xaxis_title="Weight", margin={"r": 60})

        fig.update_xaxes(tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)



        st.download_button(

            "Download weights CSV",

            chosen_weights.to_frame("Weight").round(4).to_csv().encode("utf-8"),

            f"portfolio_weights_{portfolio_choice.lower().replace(' ', '_')}.csv",

            "text/csv",

        )



        st.markdown("**Risk contribution**")

        profile = an.portfolio_risk_profile(bundle.returns, chosen_weights)

        r1, r2, r3 = st.columns(3)

        r1.metric("Diversification ratio", f"{profile['Diversification_Ratio']:.2f}")

        r2.metric("Concentration (HHI)", f"{profile['Concentration_HHI']:.2f}")

        r3.metric("Top risk asset", f"{profile['Top_Risk_Asset']} · {profile['Top_Risk_Percent']:.0%}")

        risk_frame = an.risk_contributions(bundle.returns, chosen_weights)

        top_risk = risk_frame.sort_values("Risk_Percent", ascending=False).head(12).sort_values("Risk_Percent")

        fig = go.Figure(

            go.Bar(

                x=top_risk["Risk_Percent"],

                y=top_risk.index,

                orientation="h",

                marker_color=[class_color(t) for t in top_risk.index],

                text=[f"{value:.1%}" for value in top_risk["Risk_Percent"]],

                textposition="outside",

            )

        )

        fig.update_layout(

            height=max(320, 24 * len(top_risk)),

            xaxis_title="Share of portfolio volatility",

            margin={"r": 70},

        )

        fig.update_xaxes(tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)

        st.caption("Euler decomposition: each asset's share of total portfolio volatility.")



    st.subheader("Portfolio Comparison")

    comparison = portfolio_comparison(bundle, weight_map)

    st.dataframe(pres.metric_style(comparison), use_container_width=True)



    st.subheader("Growth of $100")

    fig = go.Figure()

    colors = {"Max Sharpe": "red", "Min Variance": "black", "Equal Weight": pres.ACCENT, "Custom": pres.CUSTOM}

    if bundle.benchmark:

        colors[f"Benchmark ({bundle.benchmark})"] = "gray"

    for label in comparison.index:

        growth = growth_curve(bundle.returns, weight_map[label])

        fig.add_trace(

            go.Scatter(x=growth.index, y=growth.values, mode="lines", name=label, line={"color": colors.get(label)})

        )

    fig.update_layout(height=420, yaxis_title="Value of $100 invested")

    st.plotly_chart(fig, use_container_width=True)





# ---------------------------------------------------------------------------

# Tab 5 — Drawdown & Tail Risk

# ---------------------------------------------------------------------------

def render_drawdown_and_tail_risk(bundle: an.AnalysisBundle, metrics: pd.DataFrame, available: list[str]) -> None:

    st.subheader("Drawdown Analysis")

    portfolio_subjects = [f"{label} portfolio" for label in PORTFOLIO_CHOICES]

    subject = st.selectbox("Analyze drawdown for", [*portfolio_subjects, *available])



    if subject in portfolio_subjects:

        weights = bundle.portfolio_choices()[subject.removesuffix(" portfolio")]

        subject_series = an.portfolio_returns(bundle.returns, weights)

    else:

        subject_series = bundle.returns[subject]



    drawdowns = an.drawdown_series(subject_series.to_frame("Subject"))["Subject"]

    max_dd_value = drawdowns.min()

    trough_date = drawdowns.idxmin()

    col1, col2, col3 = st.columns(3)

    col1.metric("Max Drawdown", f"{max_dd_value:.1%}")

    col2.metric("Trough Date", f"{trough_date.date()}")

    col3.metric("Daily VaR 95%", f"{-subject_series.quantile(0.05):.2%}")



    fig = go.Figure(

        go.Scatter(

            x=drawdowns.index,

            y=drawdowns.values,

            mode="lines",

            fill="tozeroy",

            line={"color": pres.NEGATIVE},

        )

    )

    fig.update_layout(title=f"Underwater plot — {subject}", yaxis_title="Drawdown", height=380)

    fig.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)



    st.subheader("Largest Drawdown Episodes")

    episodes = an.drawdown_episodes(subject_series, top=5)

    for column in ("Peak", "Trough", "Recovery"):

        episodes[column] = episodes[column].dt.strftime("%Y-%m-%d").fillna("—")

    episodes["Depth"] = episodes["Depth"].map(lambda value: f"{value:.1%}")

    episodes["Trough_Days"] = episodes["Trough_Days"].map("{:.0f}".format)

    episodes["Recovery_Days"] = episodes["Recovery_Days"].map(lambda value: "—" if pd.isna(value) else f"{value:.0f}")

    st.dataframe(episodes, use_container_width=True, hide_index=True)



    st.subheader("Tail Risk (Historical Simulation)")

    left, right = st.columns(2)

    with left:

        var_data = metrics[["VaR_95", "CVaR_95"]].sort_values("VaR_95")

        fig = go.Figure()

        fig.add_trace(

            go.Bar(

                x=var_data.index,

                y=var_data["VaR_95"],

                name="VaR 95%",

                marker_color=[class_color(t) for t in var_data.index],

            )

        )

        fig.add_trace(

            go.Scatter(

                x=var_data.index,

                y=var_data["CVaR_95"],

                name="CVaR 95%",

                mode="markers",

                marker={"symbol": "line-ew", "size": 14, "color": "black"},

            )

        )

        fig.update_layout(title="Daily VaR / CVaR by asset", yaxis_title="Daily loss", height=420)

        fig.update_yaxes(tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig = go.Figure(go.Histogram(x=subject_series.values, nbinsx=80, marker_color=pres.ACCENT))

        var95 = -subject_series.quantile(0.05)

        cvar95 = -subject_series[subject_series <= subject_series.quantile(0.05)].mean()

        fig.add_vline(x=-var95, line_dash="dash", line_color="red", annotation_text=f"VaR 95% · {var95:.2%}")

        fig.add_vline(x=-cvar95, line_dash="dot", line_color="darkred", annotation_text=f"CVaR 95% · {cvar95:.2%}")

        fig.update_layout(title=f"Daily return distribution — {subject}", xaxis_title="Daily return", height=420)

        fig.update_xaxes(tickformat=".1%")

        st.plotly_chart(fig, use_container_width=True)



    st.caption(

        "VaR 95% = the daily loss exceeded on ~5% of trading days. CVaR 95% = the average loss in that worst 5% tail. "

        "Empirical (historical) estimates, no normality assumption."

    )



    st.subheader("Stress Scenarios")

    st.caption(

        "Total return of each portfolio inside historical crisis windows; blank cells fall outside the selected sample."

    )

    weights_map = bundle.portfolio_choices()

    scenario_returns = an.scenario_return_table(bundle.returns, weights_map)

    scenario_drawdowns = an.scenario_drawdown_table(bundle.returns, weights_map)

    scenario_text = [

        [f"{value:.0%}" if pd.notna(value) else "—" for value in scenario_returns.loc[name]]

        for name in scenario_returns.index

    ]

    fig = go.Figure(

        go.Heatmap(

            z=scenario_returns.to_numpy() * 100,

            x=list(scenario_returns.columns),

            y=list(scenario_returns.index),

            text=scenario_text,

            texttemplate="%{text}",

            colorscale="RdYlGn",

            zmid=0,

            colorbar={"title": "Return (%)"},

        )

    )

    fig.update_layout(height=110 + 55 * len(scenario_returns), yaxis_autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Maximum drawdown inside each window"):

        st.dataframe(

            scenario_drawdowns.map(lambda value: f"{value:.1%}" if pd.notna(value) else "—"),

            use_container_width=True,

        )

    scenario_name = st.selectbox("Asset detail", list(an.SCENARIOS), key="scenario_assets")

    asset_table = an.scenario_asset_table(bundle.returns, scenario_name)

    st.dataframe(pres.metric_style(asset_table), use_container_width=True)





# ---------------------------------------------------------------------------

# Tab 6 — Executive Summary

# ---------------------------------------------------------------------------

def render_executive_summary(bundle: an.AnalysisBundle, metrics: pd.DataFrame, risk_free: float) -> None:

    st.subheader("Key Insights")

    for bullet in an.executive_summary(bundle):

        st.markdown(f"- {bullet}")



    st.subheader("Portfolio Recommendation")

    sharpe_stats, _ = an.portfolio_stats(bundle.returns, bundle.max_sharpe, rf=bundle.risk_free)

    minvar_stats, _ = an.portfolio_stats(bundle.returns, bundle.min_variance, rf=bundle.risk_free)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Maximum Sharpe portfolio**")

        st.dataframe(

            bundle.max_sharpe[bundle.max_sharpe > 0.005]

            .sort_values(ascending=False)

            .rename("Weight")

            .to_frame()

            .style.format("{:.2%}"),

        )

        st.caption(

            f"CAGR {sharpe_stats['Annualized_Return']:.1%} · vol {sharpe_stats['Annualized_Volatility']:.1%} · "

            f"Sharpe {sharpe_stats['Sharpe_Ratio']:.2f} · max drawdown {sharpe_stats['Max_Drawdown']:.1%}"

        )

    with col2:

        st.markdown("**Minimum variance portfolio**")

        st.dataframe(

            bundle.min_variance[bundle.min_variance > 0.005]

            .sort_values(ascending=False)

            .rename("Weight")

            .to_frame()

            .style.format("{:.2%}"),

        )

        st.caption(

            f"CAGR {minvar_stats['Annualized_Return']:.1%} · vol {minvar_stats['Annualized_Volatility']:.1%} · "

            f"Sharpe {minvar_stats['Sharpe_Ratio']:.2f} · max drawdown {minvar_stats['Max_Drawdown']:.1%}"

        )



    st.subheader("Downloads")

    down1, down2, down3, down4 = st.columns(4)

    with down1:

        st.download_button(

            "Metrics CSV",

            metrics.round(4).to_csv().encode("utf-8"),

            "financial_analysis_summary.csv",

            "text/csv",

        )

    with down2:

        st.download_button(

            "Correlations CSV",

            bundle.correlation.round(6).to_csv().encode("utf-8"),

            "asset_correlations.csv",

            "text/csv",

        )

    with down3:

        weights_export = pd.DataFrame(

            {

                "Max_Sharpe_Weight": bundle.max_sharpe,

                "Min_Variance_Weight": bundle.min_variance,

                "Equal_Weight": bundle.equal_weight,

            }

        ).round(4)

        st.download_button(

            "Portfolio weights CSV",

            weights_export.to_csv().encode("utf-8"),

            "portfolio_optimization.csv",

            "text/csv",

        )

    with down4:

        st.download_button(

            "Executive report (HTML)",

            an.report_html(bundle).encode("utf-8"),

            "financial_report.html",

            "text/html",

        )



    with st.expander("Methodology & metric definitions"):

        st.markdown(f"""

**Returns** — daily simple returns from adjusted close prices; annualized over {an.TRADING_DAYS} trading days.



**Annualized return** — geometric CAGR: `(1+r).prod() ** (252/n) - 1`.



**Sharpe ratio** — `(annualized arithmetic return − risk-free rate) / annualized volatility`

with a risk-free rate of **{risk_free:.1%}**.



**Sortino ratio** — excess return divided by downside deviation (returns below the risk-free rate).



**Max drawdown / Calmar** — deepest peak-to-trough decline of the cumulative wealth curve;

Calmar = CAGR / |max drawdown|.



**VaR / CVaR (95%, 99%)** — historical simulation: the daily loss at the 5%/1% quantile, and the

average loss beyond it. Reported as positive loss magnitudes.



**Beta / Alpha** — CAPM regression against **{bundle.benchmark or "the selected benchmark"}**;

alpha is annualized Jensen's alpha.



**Tracking error / information ratio** — annualized volatility of active returns (asset − benchmark),

and annualized active return divided by tracking error.



**Up / down capture** — average monthly return in the months the benchmark rose (fell),

relative to the benchmark's own average over those months.



**Portfolio optimization** — long-only mean-variance optimization (SLSQP): maximum Sharpe

(tangency portfolio), global minimum variance, and a Markowitz efficient frontier with

Monte Carlo sampling of the weight simplex.

        """)





# ---------------------------------------------------------------------------

# Tab 5 — Backtest

# ---------------------------------------------------------------------------

def render_backtest(controls: Controls, benchmark: str | None) -> None:

    st.subheader("Walk-Forward Backtest")

    st.caption(

        "Weights are re-estimated every month on the trailing 3-year window and held out-of-sample; "

        "10 bp one-way costs are charged on turnover. The benchmark is buy-and-hold."

    )

    if not st.toggle("Run walk-forward backtest", key="run_backtest"):

        st.info(

            "Enable the toggle to run the out-of-sample backtest "

            "(the first run can take up to a minute with a large universe; cached afterwards)."

        )

        return

    try:

        with st.spinner("Running walk-forward backtest…"):

            result = cached_backtest(

                tuple(controls.tickers),

                controls.start_date,

                controls.end_date,

                controls.risk_free,

                10.0,

                benchmark,

            )

    except ValueError as exc:

        st.warning(str(exc))

        return



    curves = result.equity_curves()

    fig = go.Figure()

    for column in curves.columns:

        fig.add_trace(go.Scatter(x=curves.index, y=curves[column], mode="lines", name=column))

    fig.update_layout(height=460, yaxis={"title": "Growth of $100 (log scale)", "type": "log"})

    st.plotly_chart(fig, use_container_width=True)



    display_columns = [

        "Annualized_Return",

        "Annualized_Volatility",

        "Sharpe_Ratio",

        "Sortino_Ratio",

        "Max_Drawdown",

        "Calmar_Ratio",

        "Avg_Turnover",

        "Annual_Cost",

        "Total_Return",

    ]

    st.dataframe(pres.metric_style(result.stats[display_columns]), use_container_width=True)

    st.caption(

        f"Estimation window: {result.lookback} trading days · {result.cost_bps:.0f} bp one-way cost per rebalance · "

        "turnover and cost columns are averages over the backtest."

    )





# ---------------------------------------------------------------------------

# Tab 7 — Risk Models

# ---------------------------------------------------------------------------

def render_risk_models(bundle: an.AnalysisBundle, controls: Controls) -> None:

    st.subheader("VaR Backtest")

    st.caption("Trailing 500-day historical VaR per portfolio with Kupiec coverage tests (p < 0.05 rejects the model).")

    series_map = {

        label: an.portfolio_returns(bundle.returns, weights) for label, weights in bundle.portfolio_choices().items()

    }

    var_table = an.var_backtest_summary(series_map, window=500)

    var_display = var_table.copy()

    for column in ("Breach_Rate", "Expected_Rate", "Avg_Breach_Loss"):

        var_display[column] = var_display[column].map(lambda value: f"{value:.1%}" if pd.notna(value) else "—")

    for column in ("LR_Statistic", "P_Value"):

        var_display[column] = var_display[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "—")

    st.dataframe(var_display, use_container_width=True, hide_index=True)



    subject = st.selectbox("VaR detail", list(series_map), key="var_subject")

    series = series_map[subject]

    var = an.rolling_var(series, window=500, level=0.05)

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=series.index,

            y=series.values,

            mode="lines",

            name="Daily return",

            line={"color": pres.ACCENT, "width": 1},

            opacity=0.7,

        )

    )

    fig.add_trace(

        go.Scatter(x=var.index, y=-var.values, mode="lines", name="VaR 95%", line={"color": pres.NEGATIVE, "width": 2})

    )

    breaches = series[series < -var]

    fig.add_trace(

        go.Scatter(

            x=breaches.index, y=breaches.values, mode="markers", name="Breaches", marker={"color": "black", "size": 5}

        )

    )

    fig.update_layout(height=420, title=f"Returns vs VaR — {subject}", yaxis_title="Daily return")

    st.plotly_chart(fig, use_container_width=True)



    st.subheader("GARCH(1,1) Volatility Forecasts")

    st.caption(

        "Maximum-likelihood GARCH(1,1) per asset: conditional volatility and a 21-day forecast vs the last 60 days "

        "of realized volatility."

    )

    if not st.toggle("Fit GARCH(1,1) models", key="run_garch"):

        st.info(

            "Enable the toggle to fit one GARCH model per selected asset "

            "(up to a minute with a large universe; cached afterwards)."

        )

        return

    with st.spinner("Fitting GARCH(1,1) models…"):

        forecast = cached_garch(tuple(controls.tickers), controls.start_date, controls.end_date)

    forecast_display = forecast.copy()

    for column in ("Conditional_Vol", "Forecast_Vol", "Realized_Vol_60d"):

        forecast_display[column] = forecast_display[column].map(

            lambda value: f"{value:.1%}" if pd.notna(value) else "—"

        )

    forecast_display["Forecast_vs_Realized"] = forecast["Forecast_vs_Realized"].map(

        lambda value: f"{value:.2f}" if pd.notna(value) else "—"

    )

    st.dataframe(

        forecast_display[["Name", "Forecast_Vol", "Realized_Vol_60d", "Forecast_vs_Realized", "Regime"]],

        use_container_width=True,

    )

    top = forecast.head(15).iloc[::-1]

    fig = go.Figure(

        go.Bar(

            x=top["Forecast_Vol"],

            y=top.index,

            orientation="h",

            marker_color=[class_color(t) for t in top.index],

            text=[f"{value:.1%}" for value in top["Forecast_Vol"]],

            textposition="outside",

        )

    )

    fig.update_layout(height=480, xaxis_title="GARCH 21-day forecast (annualized)", margin={"r": 80})

    fig.update_xaxes(tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)

    st.caption(

        "Elevated = forecast at least 15% above the last 60 days of realized volatility; Calm = at least 15% below."

    )





# ---------------------------------------------------------------------------

# Entry point

# ---------------------------------------------------------------------------

def main() -> None:

    today = dt.date.today()

    st.title("Financial Volatility & Correlation Analysis")

    st.caption("Live market data · advanced risk metrics · mean-variance portfolio optimization")



    controls = render_sidebar(today)

    if controls.start_date >= controls.end_date:

        st.error("The start date must be before the end date.")

        st.stop()

    if not controls.tickers:

        st.error("Select at least one asset in the sidebar to run the analysis.")

        st.stop()



    prices_all, source, fetch_error = load_prices(controls.start_date, controls.end_date, tuple(controls.tickers))

    if prices_all is None:

        st.error(

            "Could not load market data — no live connection and no local snapshot found.\n\n"

            "Run `python analysis.py` while online to create `data/price_history.csv`, then reload."

        )

        st.stop()



    missing = [t for t in controls.tickers if t not in prices_all.columns]

    available = [t for t in controls.tickers if t in prices_all.columns]

    if missing:

        st.warning(f"No price data for: {', '.join(missing)}")

    if not available:

        st.error("Select at least one asset with available data.")

        st.stop()

    benchmark = controls.benchmark if controls.benchmark in available else None



    prices = prices_all[available]

    bundle = analyze(prices, controls.risk_free, benchmark, controls.mc_portfolios)

    bundle.source, bundle.error = source, fetch_error

    metrics = bundle.metrics



    render_data_status(bundle, available, prices, source, fetch_error)

    render_metric_cards(bundle, metrics, len(available))



    tabs = st.tabs(

        [

            "Overview",

            "Risk & Return",

            "Correlations",

            "Portfolio Optimizer",

            "Backtest",

            "Drawdown & Tail Risk",

            "Risk Models",

            "Executive Summary",

        ]

    )

    with tabs[0]:

        render_overview(metrics)

    with tabs[1]:

        render_risk_and_return(bundle)

    with tabs[2]:

        render_correlations(bundle, available)

    with tabs[3]:

        render_portfolio_optimizer(bundle, available)

    with tabs[4]:

        render_backtest(controls, benchmark)

    with tabs[5]:

        render_drawdown_and_tail_risk(bundle, metrics, available)

    with tabs[6]:

        render_risk_models(bundle, controls)

    with tabs[7]:

        render_executive_summary(bundle, metrics, controls.risk_free)



    st.caption(

        f"Data: Yahoo Finance via yfinance · source: {source} · analytics recomputed per selection · "

        f"benchmark: {bundle.benchmark or 'n/a'}"

    )





main()
