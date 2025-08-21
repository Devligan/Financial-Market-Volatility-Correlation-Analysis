import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-saved results
summary_export = pd.read_csv("financial_analysis_summary.csv", index_col=0)
correlation_matrix = pd.read_csv("asset_correlations.csv", index_col=0)

st.title("📊 Financial Market Volatility & Correlation Study")

st.header("Summary Statistics")
st.dataframe(summary_export)

st.header("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r", center=0, ax=ax)
st.pyplot(fig)

st.header("Key Insights")
st.markdown(f"""
- Most volatile asset: **{summary_export['Annualized_Volatility'].idxmax()}**
- Best Sharpe ratio: **{summary_export['Sharpe_Ratio'].idxmax()}**
""")
