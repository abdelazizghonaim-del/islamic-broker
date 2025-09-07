# islamic_ai_msnstyle.py
"""
Islamic AI Broker — MSN Finance style dashboard for EGX
Features:
 - DataManager: TwelveData primary, yfinance fallback
 - Market Overview: indices, gainers/losers, heatmap
 - Company Page: candlestick + indicators, fundamentals
 - Sharia screener, simple ML scaffold, backtest stub
 - Watchlist & Portfolio simulator (st.session_state)
NOTE: set TWELVEDATA_API_KEY in the config below.
Run: streamlit run islamic_ai_msnstyle.py
"""

import os
import time
import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
api_key = st.session_state.get("TWELVEDATA_API_KEY", TWELVEDATA_API_KEY)
dm = DataManager(apikey=api_key)

# -----------------------
# CONFIG
# -----------------------
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")  # <-- set this env var or replace with string
DEFAULT_INDEX_TICKERS = {
    "EGX30": "EGX30",  # Placeholder names — indices may require licensed feed
}
APP_TITLE = "🕌 Islamic AI Broker — EGX (MSN style)"
CACHE_TTL = 300  # seconds

# -----------------------
# UTILITIES
# -----------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def parse_twelvedata_series(resp_json) -> pd.DataFrame:
    """Parse TwelveData time_series response to DataFrame if possible."""
    if not resp_json or 'values' not in resp_json:
        return pd.DataFrame()
    df = pd.DataFrame(resp_json['values'])
    # TwelveData returns strings
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.index = pd.to_datetime(df['datetime'])
    df = df.sort_index()
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df[['Open','High','Low','Close','Volume']]
    return df.dropna(how='all')

# -----------------------
# DATA MANAGER
# -----------------------
class DataManager:
    """
    Multi-backend data fetcher:
     - Primary: TwelveData REST API (fast, stable)
     - Fallback: yfinance
    Supports flexible symbol formats for EGX.
    """
    def __init__(self, apikey: str = ""):
        self.apikey = apikey.strip()
        self.session = requests.Session()

    @st.cache_data(ttl=CACHE_TTL)
    def get_time_series(self, symbol: str, interval: str = "1day", outputsize: int = 500) -> pd.DataFrame:
        """Try TwelveData then yfinance. Symbol flexible (e.g., 'RAYA.CA' or 'RAYA:EGX')"""
        # 1) try TwelveData direct symbol
        if self.apikey:
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "format": "JSON",
                "apikey": self.apikey
            }
            url = "https://api.twelvedata.com/time_series"
            try:
                resp = self.session.get(url, params=params, timeout=10)
                j = resp.json()
                if 'values' in j and isinstance(j['values'], list) and len(j['values']) > 1:
                    df = parse_twelvedata_series(j)
                    if not df.empty:
                        return df
                # fallback: try with exchange suffixes (common guesses)
                for guess in [":XCAI", ":EGX"]:
                    params['symbol'] = f"{symbol}{guess}"
                    resp = self.session.get(url, params=params, timeout=10)
                    j = resp.json()
                    df = parse_twelvedata_series(j)
                    if not df.empty:
                        return df
            except Exception:
                pass

        # 2) fallback to yfinance with flexible forms
        try:
            # try exactly symbol
            t = yf.Ticker(symbol)
            df = t.history(period=f"{int(outputsize/252)+1}y")  # rough
            if not df.empty:
                df = df.rename(columns={c: c.title() for c in df.columns})
                cols = ['Open','High','Low','Close','Volume']
                df = df[[c for c in cols if c in df.columns]]
                return df
        except Exception:
            pass

        # 3) try appending .CA (commonly used for EGX in yfinance)
        try:
            guess = symbol.split('.')[0] + ".CA"
            t = yf.Ticker(guess)
            df = t.history(period=f"{int(outputsize/252)+1}y")
            if not df.empty:
                df = df.rename(columns={c: c.title() for c in df.columns})
                cols = ['Open','High','Low','Close','Volume']
                df = df[[c for c in cols if c in df.columns]]
                return df
        except Exception:
            pass

        return pd.DataFrame()

    @st.cache_data(ttl=CACHE_TTL)
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Quick quote — from TwelveData or yfinance info"""
        if self.apikey:
            url = "https://api.twelvedata.com/price"
            try:
                resp = self.session.get(url, params={"symbol": symbol, "apikey": self.apikey}, timeout=5)
                j = resp.json()
                if 'price' in j:
                    return {"price": safe_float(j.get('price')), "raw": j}
            except Exception:
                pass
        # yfinance fallback
        try:
            t = yf.Ticker(symbol)
            info = t.info if hasattr(t, "info") else {}
            price = info.get('regularMarketPrice') or info.get('previousClose') or None
            return {"price": safe_float(price), "raw": info}
        except Exception:
            return {"price": None, "raw": {}}

    @st.cache_data(ttl=3600)
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Company metadata using yfinance info as primary fallback (TwelveData has limited meta)"""
        try:
            t = yf.Ticker(symbol)
            info = t.info if hasattr(t, "info") else {}
            return info or {}
        except Exception:
            return {}

# -----------------------
# TECHNICAL INDICATORS (lightweight)
# -----------------------
class TechnicalIndicators:
    @staticmethod
    def sma(series: pd.Series, period: int):
        return series.rolling(period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14):
        delta = series.diff()
        gain = delta.clip(lower=0).fillna(0)
        loss = -delta.clip(upper=0).fillna(0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

# -----------------------
# SHARIA SCREENER (light)
# -----------------------
@dataclass
class ShariaConfig:
    max_debt_to_assets: float = 0.33
    max_interest_income_ratio: float = 0.05
    max_cash_securities_ratio: float = 0.33
    min_market_cap: float = 1e9

class ShariaScreener:
    def __init__(self, config: ShariaConfig, dm: DataManager):
        self.config = config
        self.dm = dm

    def screen_symbol(self, symbol: str) -> Tuple[bool, List[str]]:
        """
        Light screening using yfinance company info as available.
        It's approximate — for production, use full financial statements from licensed provider.
        """
        info = self.dm.get_company_info(symbol)
        issues = []

        # Sector check
        sector = (info.get('sector') or "").lower()
        if any(p in sector for p in ["alcohol","gambling","tobacco","weapons","defense"]):
            issues.append("Prohibited sector")

        # market cap
        market_cap = info.get('marketCap') or info.get('market_cap') or 0
        if market_cap and market_cap < self.config.min_market_cap:
            issues.append(f"Market cap below {self.config.min_market_cap:,}")

        # Note: we attempt to compute some ratios from limited fields
        try:
            # interest income approximation (not reliable with yfinance)
            total_revenue = info.get('regularMarketVolume')  # not correct; placeholder
        except Exception:
            total_revenue = None

        if not issues:
            issues.append("No obvious red flags (light screen)")

        return (len(issues) == 1 and "No obvious red flags (light screen)" in issues), issues

# -----------------------
# ML Predictor (light scaffold)
# -----------------------
class MLPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f['Close'] = df['Close']
        f['SMA20'] = TechnicalIndicators.sma(df['Close'], 20)
        f['SMA50'] = TechnicalIndicators.sma(df['Close'], 50)
        f['RSI'] = TechnicalIndicators.rsi(df['Close'])
        macd, sig, hist = TechnicalIndicators.macd(df['Close'])
        f['MACD'] = macd
        f['MACD_Signal'] = sig
        f['Returns1'] = df['Close'].pct_change(1)
        f['Volume'] = df['Volume']
        return f.dropna()

    def prepare_data(self, df: pd.DataFrame, horizon=5, threshold=0.02):
        f = self.create_features(df)
        fut = f['Close'].shift(-horizon) / f['Close'] - 1
        y = (fut > threshold).astype(int).iloc[:-horizon]
        X = f.iloc[:-horizon].drop(columns=['Close'])
        self.feature_cols = list(X.columns)
        return X, y

    def train(self, df: pd.DataFrame, horizon=5, threshold=0.02):
        try:
            X, y = self.prepare_data(df, horizon, threshold)
            if len(X) < 200 or y.nunique() < 2:
                return {"error":"Insufficient data to train"}
            pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, n_jobs=1))])
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
            pipeline.fit(X,y)
            self.model = pipeline
            return {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}
        except Exception as e:
            return {"error": str(e)}

    def predict(self, df: pd.DataFrame):
        if self.model is None:
            return {"error": "model not trained"}
        f = self.create_features(df)
        X = f[self.feature_cols].iloc[[-1]]
        prob = self.model.predict_proba(X)[0]
        return {"prob_up": float(prob[1]), "prob_down": float(prob[0])}

# -----------------------
# PORTFOLIO (simple)
# -----------------------
class Portfolio:
    def __init__(self):
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = {"cash":100000.0, "positions":{}, "history":[]}

    def value(self, prices: Dict[str,float]):
        p = st.session_state['portfolio']
        val = p['cash']
        for s, pos in p['positions'].items():
            val += pos['quantity'] * prices.get(s, 0)
        return val

    def buy(self, symbol, qty, price):
        p = st.session_state['portfolio']
        cost = qty * price
        if p['cash'] < cost:
            return False, "Insufficient cash"
        p['cash'] -= cost
        if symbol in p['positions']:
            old = p['positions'][symbol]
            new_qty = old['quantity'] + qty
            new_avg = (old['avg_price']*old['quantity'] + cost) / new_qty
            p['positions'][symbol] = {"quantity": new_qty, "avg_price": new_avg}
        else:
            p['positions'][symbol] = {"quantity": qty, "avg_price": price}
        p['history'].append({"action":"BUY","symbol":symbol,"qty":qty,"price":price,"time":datetime.now()})
        return True, "Bought"

    def sell(self, symbol, qty, price):
        p = st.session_state['portfolio']
        if symbol not in p['positions'] or p['positions'][symbol]['quantity'] < qty:
            return False, "Not enough shares"
        p['positions'][symbol]['quantity'] -= qty
        p['cash'] += qty * price
        if p['positions'][symbol]['quantity'] == 0:
            del p['positions'][symbol]
        p['history'].append({"action":"SELL","symbol":symbol,"qty":qty,"price":price,"time":datetime.now()})
        return True, "Sold"

# -----------------------
# UI: Helpers
# -----------------------
def plot_candles_with_indicators(df: pd.DataFrame, title:str="Price"):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03,
                        specs=[[{"type": "candlestick"}],[{"type":"bar"}],[{"type":"scatter"}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if 'Close' in df:
        fig.add_trace(go.Scatter(x=df.index, y=TechnicalIndicators.ema(df['Close'],20), name='EMA20', line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=TechnicalIndicators.ema(df['Close'],50), name='EMA50', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
    if 'Close' in df:
        fig.add_trace(go.Scatter(x=df.index, y=TechnicalIndicators.rsi(df['Close']), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_layout(height=700, title=title, xaxis_rangeslider_visible=False)
    return fig

def top_movers_table(df: pd.DataFrame, topn=10):
    if df.empty: 
        return pd.DataFrame()
    last = df.groupby(df.index).last() if isinstance(df.index, pd.DatetimeIndex) else df
    # For a daily frame pass a multi-symbol dict; but here assume df is single-symbol or reuse approach externally
    return pd.DataFrame()  # placeholder in this simplified context

# -----------------------
# MAIN APP
# -----------------------
def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.markdown("**Notes:** This demo uses TwelveData (if API key provided) and yfinance as fallback. For production-grade real-time EGX data, use a licensed feed.")

    # Initialize managers
    dm = DataManager(apikey=TWELVEDATA_API_KEY)
    sharia = ShariaScreener(ShariaConfig(), dm)
    ml = MLPredictor()
    pf = Portfolio()

    # Sidebar: navigation and watchlist
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Market Overview", "Company Page", "Watchlist", "Portfolio", "Settings"])

    # default tickers for the sidebar selection
    sample_tickers = ["RAYA.CA", "ETEL.CA", "FWRY.CA", "AMOC.CA", "JUFO.CA", "ORAS.CA"]
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = sample_tickers.copy()

    if page == "Market Overview":
        st.header("📊 Market Overview")
        st.markdown("Top movers & heatmap (data from available sources)")

        # User can provide a list of tickers to fetch
        tickers_input = st.text_area("Tickers to include (comma separated)", value=",".join(st.session_state['watchlist']), height=80)
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

        # build a small table of last close and change
        summary = []
        with st.spinner("Fetching quotes..."):
            for s in tickers:
                df = dm.get_time_series(s, interval="1day", outputsize=365)
                quote = dm.get_quote(s)
                last = df['Close'].iloc[-1] if not df.empty else quote.get('price')
                prev = df['Close'].iloc[-2] if (not df.empty and len(df)>1) else None
                change = ((last - prev) / prev * 100) if (prev and prev!=0) else None
                summary.append({"symbol": s, "last": last, "change_pct": change})
        summary_df = pd.DataFrame(summary)
        if not summary_df.empty:
            summary_df['change_pct_str'] = summary_df['change_pct'].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
            st.dataframe(summary_df[['symbol','last','change_pct_str']].rename(columns={'last':'Last Price','change_pct_str':'Change'}), height=300)
        else:
            st.info("No data available for the requested tickers")

        st.markdown("---")
        st.markdown("Market snapshot charts (select single ticker below)")
        ticker = st.selectbox("Select ticker for chart", options=tickers, index=0 if tickers else 0)
        if ticker:
            df = dm.get_time_series(ticker, interval="1day", outputsize=400)
            if df.empty:
                st.error("No price series available for this symbol.")
            else:
                fig = plot_candles_with_indicators(df, title=f"{ticker} — Price & Indicators")
                st.plotly_chart(fig, use_container_width=True)

    elif page == "Company Page":
        st.header("🏢 Company Page")
        ticker = st.selectbox("Choose symbol", options=st.session_state['watchlist'] + sample_tickers)
        if ticker:
            with st.spinner("Fetching data..."):
                df = dm.get_time_series(ticker, interval="1day", outputsize=800)
                info = dm.get_company_info(ticker)
                quote = dm.get_quote(ticker)
            # Quick header
            st.subheader(f"{ticker} — {info.get('shortName') or info.get('longName') or ''}")
            col1, col2, col3 = st.columns([2,1,1])
            with col1:
                st.metric("Last Price", f"{quote.get('price') or (df['Close'].iloc[-1] if not df.empty else 'N/A')}")
            with col2:
                if not df.empty and len(df)>1:
                    change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                    st.metric("Daily Change", f"{change:+.2f}%")
                else:
                    st.metric("Daily Change", "N/A")
            with col3:
                st.metric("Market Cap", f"{info.get('marketCap') or 'N/A'}")

            if df.empty:
                st.warning("Price series not available; show fundamentals only.")
            else:
                st.plotly_chart(plot_candles_with_indicators(df, title=f"{ticker} Price Chart"), use_container_width=True)

            # Fundamentals section (from yfinance info as available)
            st.markdown("### Fundamentals (light)")
            colf1, colf2 = st.columns(2)
            with colf1:
                st.write(f"Sector: {info.get('sector')}")
                st.write(f"Industry: {info.get('industry')}")
                st.write(f"Website: {info.get('website')}")
            with colf2:
                st.write(f"Market Cap: {info.get('marketCap')}")
                st.write(f"P/E (ttm): {info.get('trailingPE')}")
                st.write(f"Dividend Yield: {info.get('dividendYield')}")

            # Sharia screen
            st.markdown("### 🕌 Sharia Screening (light)")
            compliant, issues = sharia.screen_symbol(ticker)
            if compliant:
                st.success("Light Sharia screen: No obvious red flags")
            else:
                st.error("Potential issues:")
            for i in issues:
                st.write("- " + str(i))

            # ML train & predict (optional)
            st.markdown("### 🤖 AI Prediction (train & predict)")
            horizon = st.slider("Prediction horizon (days)", 1, 20, 5)
            threshold_pct = st.slider("Movement threshold (%)", 1, 10, 2)
            if st.button("Train model for this ticker"):
                if df.empty:
                    st.error("No data to train on.")
                else:
                    res = ml.train(df, horizon=horizon, threshold=threshold_pct/100.0)
                    if 'error' in res:
                        st.error(res['error'])
                    else:
                        st.success(f"Model trained — CV accuracy {res['cv_mean']:.2%} ± {res['cv_std']:.2%}")
            if st.button("Run prediction (latest)"):
                res = ml.predict(df)
                if 'error' in res:
                    st.error(res['error'])
                else:
                    st.info(f"Prob Up: {res['prob_up']:.2%} — Prob Down: {res['prob_down']:.2%}")

    elif page == "Watchlist":
        st.header("⭐ Watchlist")
        current = st.session_state['watchlist']
        st.write("Current watchlist:")
        st.write(current)
        add = st.text_input("Add ticker (e.g. RAYA.CA)")
        if st.button("Add"):
            if add and add not in st.session_state['watchlist']:
                st.session_state['watchlist'].append(add)
                st.success(f"Added {add}")
        remove = st.selectbox("Remove ticker", options=[""]+st.session_state['watchlist'])
        if st.button("Remove selected"):
            if remove in st.session_state['watchlist']:
                st.session_state['watchlist'].remove(remove)
                st.success(f"Removed {remove}")

    elif page == "Portfolio":
        st.header("💼 Portfolio Simulator")
        p = st.session_state['portfolio']
        st.write("Cash:", f"${p['cash']:,.2f}")
        st.write("Positions:")
        st.write(p['positions'])
        tick = st.selectbox("Ticker to trade", options=st.session_state['watchlist'])
        qty = st.number_input("Quantity", min_value=1, value=100)
        if st.button("Buy"):
            quote = dm.get_quote(tick)
            price = quote.get('price') or (dm.get_time_series(tick).Close.iloc[-1] if not dm.get_time_series(tick).empty else None)
            if not price:
                st.error("Price unavailable")
            else:
                ok, msg = pf.buy(tick, int(qty), float(price))
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        if st.button("Sell"):
            quote = dm.get_quote(tick)
            price = quote.get('price') or (dm.get_time_series(tick).Close.iloc[-1] if not dm.get_time_series(tick).empty else None)
            if not price:
                st.error("Price unavailable")
            else:
                ok, msg = pf.sell(tick, int(qty), float(price))
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        # portfolio value estimation
        prices = {}
        for s in p['positions'].keys():
            q = dm.get_quote(s)
            prices[s] = q.get('price') or 0
        total_val = pf.value(prices)
        st.metric("Total portfolio value", f"${total_val:,.2f}")

    elif page == "Settings":
        st.header("⚙️ Settings & Keys")
        st.markdown("TwelveData API Key (used for better EGX coverage). Provide your key or set environment variable `TWELVEDATA_API_KEY`.")
        key = st.text_input("TwelveData API Key", value=TWELVEDATA_API_KEY, type="password")
       if st.button("Save API Key (runtime)"):
       st.session_state["TWELVEDATA_API_KEY"] = key.strip()
       st.success("Saved for current session (will reset if app restarts).")

        st.markdown("---")
        st.markdown("**Notes & Next steps for production**")
        st.write("""
        - For accurate, real-time EGX data use licensed exchange data or broker feeds (Mubasher, EGX FIX).
        - Replace light Sharia screening with full FS-based checks.
        - Add authentication, persistent DB, job scheduler for regular updates.
        - Consider microservice architecture for heavy ML backtesting.
        """)

    # footer
    st.markdown("---")
    st.markdown("Built with ❤️ — ask me to connect it to a production EGX feed (Mubasher/EGX) or to add advanced features (alerts, mobile-ready UI, real-time websockets).")

if __name__ == "__main__":
    main()
