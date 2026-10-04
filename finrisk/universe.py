"""Asset universe: the cross-asset panel (stocks, ETFs, bonds, commodities, FX)."""

from __future__ import annotations

ASSET_UNIVERSE: dict[str, dict[str, str]] = {
    # --- Individual stocks ---
    "AAPL": {"name": "Apple Inc.", "class": "STOCK"},
    "MSFT": {"name": "Microsoft Corp.", "class": "STOCK"},
    "GOOGL": {"name": "Alphabet Inc.", "class": "STOCK"},
    "AMZN": {"name": "Amazon.com Inc.", "class": "STOCK"},
    "NVDA": {"name": "NVIDIA Corp.", "class": "STOCK"},
    "META": {"name": "Meta Platforms Inc.", "class": "STOCK"},
    "TSLA": {"name": "Tesla Inc.", "class": "STOCK"},
    "JPM": {"name": "JPMorgan Chase & Co.", "class": "STOCK"},
    "XOM": {"name": "Exxon Mobil Corp.", "class": "STOCK"},
    "V": {"name": "Visa Inc.", "class": "STOCK"},
    "UNH": {"name": "UnitedHealth Group Inc.", "class": "STOCK"},
    "PG": {"name": "Procter & Gamble Co.", "class": "STOCK"},
    "KO": {"name": "Coca-Cola Co.", "class": "STOCK"},
    "JNJ": {"name": "Johnson & Johnson", "class": "STOCK"},
    "WMT": {"name": "Walmart Inc.", "class": "STOCK"},
    "DIS": {"name": "Walt Disney Co.", "class": "STOCK"},
    "BAC": {"name": "Bank of America Corp.", "class": "STOCK"},
    "CVX": {"name": "Chevron Corp.", "class": "STOCK"},
    "PFE": {"name": "Pfizer Inc.", "class": "STOCK"},
    "CSCO": {"name": "Cisco Systems Inc.", "class": "STOCK"},
    "INTC": {"name": "Intel Corp.", "class": "STOCK"},
    "ORCL": {"name": "Oracle Corp.", "class": "STOCK"},
    "PEP": {"name": "PepsiCo Inc.", "class": "STOCK"},
    "MCD": {"name": "McDonald's Corp.", "class": "STOCK"},
    "NKE": {"name": "NIKE Inc.", "class": "STOCK"},
    "HD": {"name": "Home Depot Inc.", "class": "STOCK"},
    "TMO": {"name": "Thermo Fisher Scientific Inc.", "class": "STOCK"},
    "ABT": {"name": "Abbott Laboratories", "class": "STOCK"},
    "LMT": {"name": "Lockheed Martin Corp.", "class": "STOCK"},
    "CAT": {"name": "Caterpillar Inc.", "class": "STOCK"},
    "GS": {"name": "Goldman Sachs Group Inc.", "class": "STOCK"},
    "MS": {"name": "Morgan Stanley", "class": "STOCK"},
    "C": {"name": "Citigroup Inc.", "class": "STOCK"},
    "IBM": {"name": "IBM Corp.", "class": "STOCK"},
    "MMM": {"name": "3M Co.", "class": "STOCK"},
    "WFC": {"name": "Wells Fargo & Co.", "class": "STOCK"},
    "TXN": {"name": "Texas Instruments Inc.", "class": "STOCK"},
    "QCOM": {"name": "Qualcomm Inc.", "class": "STOCK"},
    "COST": {"name": "Costco Wholesale Corp.", "class": "STOCK"},
    "VZ": {"name": "Verizon Communications Inc.", "class": "STOCK"},
    "MRK": {"name": "Merck & Co. Inc.", "class": "STOCK"},
    # --- Broad-market, sector & style ETFs ---
    "SPY": {"name": "S&P 500 ETF", "class": "ETF"},
    "QQQ": {"name": "Nasdaq-100 ETF", "class": "ETF"},
    "IWM": {"name": "Russell 2000 ETF", "class": "ETF"},
    "XLE": {"name": "Energy Sector ETF", "class": "ETF"},
    "XLF": {"name": "Financials Sector ETF", "class": "ETF"},
    "XLK": {"name": "Technology Sector ETF", "class": "ETF"},
    "XLV": {"name": "Health Care Sector ETF", "class": "ETF"},
    "XLP": {"name": "Consumer Staples Sector ETF", "class": "ETF"},
    "XLY": {"name": "Consumer Discretionary Sector ETF", "class": "ETF"},
    "XLI": {"name": "Industrials Sector ETF", "class": "ETF"},
    "XLB": {"name": "Materials Sector ETF", "class": "ETF"},
    "XLU": {"name": "Utilities Sector ETF", "class": "ETF"},
    "VTV": {"name": "Vanguard Value ETF", "class": "ETF"},
    "VUG": {"name": "Vanguard Growth ETF", "class": "ETF"},
    "VYM": {"name": "Vanguard High Dividend Yield ETF", "class": "ETF"},
    "VIG": {"name": "Vanguard Dividend Appreciation ETF", "class": "ETF"},
    "IWD": {"name": "Russell 1000 Value ETF", "class": "ETF"},
    "IWF": {"name": "Russell 1000 Growth ETF", "class": "ETF"},
    "USMV": {"name": "MSCI USA Min Volatility ETF", "class": "ETF"},
    "SPLV": {"name": "S&P 500 Low Volatility ETF", "class": "ETF"},
    "DVY": {"name": "Select Dividend ETF", "class": "ETF"},
    "SMH": {"name": "Semiconductor ETF", "class": "ETF"},
    "XBI": {"name": "Biotech ETF", "class": "ETF"},
    "XRT": {"name": "Retail ETF", "class": "ETF"},
    "IYR": {"name": "US Real Estate ETF", "class": "ETF"},
    "EFA": {"name": "MSCI EAFE International ETF", "class": "ETF"},
    "EEM": {"name": "MSCI Emerging Markets ETF", "class": "ETF"},
    "EWJ": {"name": "Japan ETF", "class": "ETF"},
    "EWZ": {"name": "Brazil ETF", "class": "ETF"},
    "FXI": {"name": "China Large-Cap ETF", "class": "ETF"},
    "EWH": {"name": "Hong Kong ETF", "class": "ETF"},
    "EWA": {"name": "Australia ETF", "class": "ETF"},
    "EWC": {"name": "Canada ETF", "class": "ETF"},
    "EWU": {"name": "United Kingdom ETF", "class": "ETF"},
    "EWG": {"name": "Germany ETF", "class": "ETF"},
    "EWY": {"name": "South Korea ETF", "class": "ETF"},
    "EWQ": {"name": "France ETF", "class": "ETF"},
    "EWI": {"name": "Italy ETF", "class": "ETF"},
    "EWT": {"name": "Taiwan ETF", "class": "ETF"},
    "VEA": {"name": "Developed Markets ETF", "class": "ETF"},
    "VWO": {"name": "Emerging Markets ETF", "class": "ETF"},
    # --- Fixed income & credit ---
    "TLT": {"name": "20+ Year Treasury Bond ETF", "class": "BOND"},
    "IEF": {"name": "7-10 Year Treasury Bond ETF", "class": "BOND"},
    "SHY": {"name": "1-3 Year Treasury Bond ETF", "class": "BOND"},
    "TIP": {"name": "Inflation-Protected Bond ETF", "class": "BOND"},
    "LQD": {"name": "Investment-Grade Corporate Bond ETF", "class": "BOND"},
    "BIV": {"name": "Intermediate-Term Bond ETF", "class": "BOND"},
    "BND": {"name": "Total Bond Market ETF", "class": "BOND"},
    "HYG": {"name": "High-Yield Corporate Bond ETF", "class": "BOND"},
    "JNK": {"name": "High-Yield Bond ETF", "class": "BOND"},
    "EMB": {"name": "Emerging Markets Bond ETF", "class": "BOND"},
    "MUB": {"name": "Municipal Bond ETF", "class": "BOND"},
    "FLOT": {"name": "Floating Rate Bond ETF", "class": "BOND"},
    # --- Currencies ---
    "UUP": {"name": "US Dollar Index ETF", "class": "FX"},
    "FXE": {"name": "Euro Currency Trust", "class": "FX"},
    "FXY": {"name": "Japanese Yen Trust", "class": "FX"},
    "FXA": {"name": "Australian Dollar Trust", "class": "FX"},
    "FXB": {"name": "British Pound Trust", "class": "FX"},
    # --- Commodities ---
    "GLD": {"name": "Gold ETF", "class": "COMMODITY"},
    "SLV": {"name": "Silver ETF", "class": "COMMODITY"},
    "PPLT": {"name": "Platinum ETF", "class": "COMMODITY"},
    "PALL": {"name": "Palladium ETF", "class": "COMMODITY"},
    "CPER": {"name": "Copper ETF", "class": "COMMODITY"},
    "DBB": {"name": "Base Metals ETF", "class": "COMMODITY"},
    "USO": {"name": "Crude Oil ETF", "class": "COMMODITY"},
    "UNG": {"name": "Natural Gas ETF", "class": "COMMODITY"},
    "UGA": {"name": "Gasoline ETF", "class": "COMMODITY"},
    "DBA": {"name": "Agriculture ETF", "class": "COMMODITY"},
    "CORN": {"name": "Corn ETF", "class": "COMMODITY"},
    "WEAT": {"name": "Wheat ETF", "class": "COMMODITY"},
    "SOYB": {"name": "Soybean ETF", "class": "COMMODITY"},
    "DBC": {"name": "Diversified Commodity ETF", "class": "COMMODITY"},
}


def default_tickers() -> list[str]:
    """Tickers analyzed by default (the full expanded universe)."""
    return list(ASSET_UNIVERSE)


def asset_name(ticker: str) -> str:
    return ASSET_UNIVERSE.get(ticker, {}).get("name", ticker)


def asset_class(ticker: str) -> str:
    return ASSET_UNIVERSE.get(ticker, {}).get("class", "OTHER")


def tickers_of_class(klass: str) -> list[str]:
    if klass in ("All", "ALL", None):
        return default_tickers()
    return [t for t, meta in ASSET_UNIVERSE.items() if meta["class"] == klass]
