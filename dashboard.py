import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Financial Market Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Market Volatility & Correlation Analysis")

# Load data
try:
    summary_export = pd.read_csv("financial_analysis_summary.csv", index_col=0)
    correlation_matrix = pd.read_csv("asset_correlations.csv", index_col=0)
except FileNotFoundError:
    st.error("⚠️ Data files not found. Please run the analysis script first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
asset_filter = st.sidebar.selectbox("Asset Type:", ['All', 'Stocks Only', 'Commodities Only'])

# Filter data
if asset_filter == 'Stocks Only':
    filtered_data = summary_export[summary_export.index.str.contains('STOCK')]
elif asset_filter == 'Commodities Only':
    filtered_data = summary_export[summary_export.index.str.contains('COMMODITY')]
else:
    filtered_data = summary_export

# Key metrics at top
col1, col2, col3, col4 = st.columns(4)

most_volatile = summary_export['Annualized_Volatility'].idxmax()
most_volatile_val = summary_export['Annualized_Volatility'].max()
col1.metric("Most Volatile", most_volatile.replace('_STOCK', '').replace('_COMMODITY', ''), f"{most_volatile_val:.1%}")

best_sharpe = summary_export['Sharpe_Ratio'].idxmax()
best_sharpe_val = summary_export['Sharpe_Ratio'].max()
col2.metric("Best Sharpe Ratio", best_sharpe.replace('_STOCK', '').replace('_COMMODITY', ''), f"{best_sharpe_val:.2f}")

highest_return = summary_export['Annualized_Return'].idxmax()
highest_return_val = summary_export['Annualized_Return'].max()
col3.metric("Highest Return", highest_return.replace('_STOCK', '').replace('_COMMODITY', ''),
            f"{highest_return_val:.1%}")

avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
col4.metric("Avg Correlation", "All Assets", f"{avg_correlation:.3f}")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk-Return", "Correlations", "Data"])

with tab1:
    st.subheader("Volatility Comparison")

    # Volatility bar chart
    vol_data = filtered_data['Annualized_Volatility'].sort_values()
    colors = ['blue' if 'STOCK' in asset else 'orange' for asset in vol_data.index]
    clean_names = [asset.replace('_STOCK', '').replace('_COMMODITY', '') for asset in vol_data.index]

    fig1 = go.Figure(go.Bar(
        x=vol_data.values,
        y=clean_names,
        orientation='h',
        marker_color=colors,
        text=[f'{vol:.1%}' for vol in vol_data.values],
        textposition='outside'
    ))
    fig1.update_layout(title='Annualized Volatility', xaxis_title='Volatility', height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Sharpe Ratio Comparison")

    # Sharpe ratio chart
    sharpe_data = filtered_data['Sharpe_Ratio'].sort_values()
    colors = ['blue' if 'STOCK' in asset else 'orange' for asset in sharpe_data.index]
    clean_names = [asset.replace('_STOCK', '').replace('_COMMODITY', '') for asset in sharpe_data.index]

    fig2 = go.Figure(go.Bar(
        x=sharpe_data.values,
        y=clean_names,
        orientation='h',
        marker_color=colors,
        text=[f'{ratio:.2f}' for ratio in sharpe_data.values],
        textposition='outside'
    ))
    fig2.update_layout(title='Sharpe Ratios', xaxis_title='Sharpe Ratio', height=400)
    fig2.add_vline(x=1, line_dash="dash", line_color="green", annotation_text="Good Performance")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Risk vs Return Analysis")

    # Risk-return scatter
    fig3 = go.Figure()

    # Add stocks
    stocks = [asset for asset in filtered_data.index if 'STOCK' in asset]
    if stocks:
        stock_data = filtered_data.loc[stocks]
        fig3.add_trace(go.Scatter(
            x=stock_data['Annualized_Volatility'],
            y=stock_data['Annualized_Return'],
            mode='markers+text',
            name='Stocks',
            text=[asset.replace('_STOCK', '') for asset in stocks],
            textposition="top center",
            marker=dict(size=12, color='blue')
        ))

    # Add commodities
    commodities = [asset for asset in filtered_data.index if 'COMMODITY' in asset]
    if commodities:
        commodity_data = filtered_data.loc[commodities]
        fig3.add_trace(go.Scatter(
            x=commodity_data['Annualized_Volatility'],
            y=commodity_data['Annualized_Return'],
            mode='markers+text',
            name='Commodities',
            text=[asset.replace('_COMMODITY', '') for asset in commodities],
            textposition="top center",
            marker=dict(size=12, color='orange', symbol='diamond')
        ))

    fig3.update_layout(
        title='Risk-Return Profile',
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return',
        height=500
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig3, use_container_width=True)

    st.info("**Top-left**: Low risk, positive returns (ideal) | **Top-right**: High risk, positive returns")

with tab3:
    st.subheader("Correlation Matrix")

    # Interactive heatmap
    clean_labels = [asset.replace('_STOCK', '').replace('_COMMODITY', '') for asset in correlation_matrix.columns]

    fig4 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=clean_labels,
        y=clean_labels,
        colorscale='RdBu_r',
        zmid=0,
        text=correlation_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig4.update_layout(title='Asset Correlations', height=600)
    st.plotly_chart(fig4, use_container_width=True)

    # Show highest/lowest correlations
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    max_corr_idx = upper_tri.stack().idxmax()
    max_corr_value = upper_tri.stack().max()
    min_corr_idx = upper_tri.stack().idxmin()
    min_corr_value = upper_tri.stack().min()

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            f"Highest: {max_corr_idx[0].split('_')[0]} vs {max_corr_idx[1].split('_')[0]} ({max_corr_value:.3f})")
    with col2:
        st.info(f"Lowest: {min_corr_idx[0].split('_')[0]} vs {min_corr_idx[1].split('_')[0]} ({min_corr_value:.3f})")

with tab4:
    st.subheader("Complete Data")

    # Search box
    search = st.text_input("Search assets:")

    display_data = summary_export.copy()
    if search:
        mask = display_data.index.str.contains(search.upper(), case=False)
        display_data = display_data[mask]

    # Clean up names for display
    display_data.index = [idx.replace('_STOCK', ' (Stock)').replace('_COMMODITY', ' (Commodity)')
                          for idx in display_data.index]

    st.dataframe(display_data, height=400)

    # Download button
    csv = filtered_data.to_csv()
    st.download_button("Download CSV", csv, "financial_data.csv", "text/csv")

    # Quick stats
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Volatility Stats**")
        vol_stats = summary_export['Annualized_Volatility'].describe()
        for stat, value in vol_stats.items():
            if stat == 'count':
                st.write(f"{stat}: {int(value)}")
            else:
                st.write(f"{stat}: {value:.2%}")

    with col2:
        st.write("**Return Stats**")
        return_stats = summary_export['Annualized_Return'].describe()
        for stat, value in return_stats.items():
            if stat == 'count':
                st.write(f"{stat}: {int(value)}")
            else:
                st.write(f"{stat}: {value:.2%}")

# Key insights at bottom
st.subheader("Key Insights")

stock_assets = [asset for asset in summary_export.index if 'STOCK' in asset]
commodity_assets = [asset for asset in summary_export.index if 'COMMODITY' in asset]

if stock_assets and commodity_assets:
    avg_stock_vol = summary_export.loc[stock_assets, 'Annualized_Volatility'].mean()
    avg_commodity_vol = summary_export.loc[commodity_assets, 'Annualized_Volatility'].mean()

    st.write(f"• **Assets analyzed**: {len(stock_assets)} stocks, {len(commodity_assets)} commodities")
    st.write(f"• **Average volatility**: Stocks {avg_stock_vol:.1%}, Commodities {avg_commodity_vol:.1%}")
    st.write(f"• **Best performer**: {best_sharpe.split('_')[0]} (Sharpe: {best_sharpe_val:.2f})")
    st.write(
        f"• **Volatility range**: {summary_export['Annualized_Volatility'].min():.1%} to {summary_export['Annualized_Volatility'].max():.1%}")