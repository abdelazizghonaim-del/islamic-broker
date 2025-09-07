# islamic_ai_broker_egx.py
"""
Islamic AI Broker — EGX (MSN-style light)
- TwelveData primary, yfinance fallback
- Market Overview, Company Page, Watchlist, Portfolio, Settings
- Uses st.session_state for runtime API key storage (no global assignment)
Run: streamlit run islamic_ai_broker_egx.py
"""

import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================
# CONFIG
# ==========================
DEFAULT_APP_TITLE = "🕌 Islamic AI Broker — EGX (MSN style)"
CACHE_TTL = 300  # caching TTL for data fetches (seconds)
DEFAULT_WATCHLIST = ["RAYA.CA", "ETEL.CA", "FWRY.CA", "AMOC.CA", "JUFO.CA", "ORAS.CA"]

# default TwelveData key from env (optional)
DEFAULT_TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()

# ==========================
# UTILITIES
# ==========================
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def ensure_watchlist():
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = DEFAULT_WATCHLIST.copy()

def ensure_portfolio():
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = {"cash": 100000.0, "positions": {}, "history": []}

# ==========================
# DATA MANAGER
# ==========================
def parse_twelvedata_series(resp_json) -> pd.DataFrame:
    if not resp_json or "values" not in resp_json:
        return pd.DataFrame()
    vals = resp_json["values"]
    df = pd.DataFrame(vals)
    # TwelveData uses 'datetime' and string numbers
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "datetime" in df.columns:
        df.index = pd.to_datetime(df["datetime"])
    df = df.sort_index()
    # standardize column names
    mapping = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    rename_cols = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_cols)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]
    return df.dropna(how="all")

class DataManager:
    """Handles data fetching with TwelveData primary and yfinance fallback."""
    def __init__(self, apikey: str = ""):
        self.apikey = (apikey or "").strip()
        self.session = requests.Session()

    @st.cache_data(ttl=CACHE_TTL)
    def get_time_series(self, symbol: str, interval: str = "1day", outputsize: int = 500) -> pd.DataFrame:
        """Return DataFrame with Open, High, Low, Close, Volume (or empty DataFrame)."""
        # Try TwelveData if API key provided
        if self.apikey:
            url = "https://api.twelvedata.com/time_series"
            params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": self.apikey}
            try:
                resp = self.session.get(url, params=params, timeout=8)
                j = resp.json()
                df = parse_twelvedata_series(j)
                if not df.empty:
                    return df
                # Try some guesses: symbol + ":EGX" or symbol + ".CA" if common
                guesses = [f"{symbol}:EGX", f"{symbol}.CA", f"{symbol}:XCAI"]
                for g in guesses:
                    params["symbol"] = g
                    resp = self.session.get(url, params=params, timeout=8)
                    j = resp.json()
                    df = parse_twelvedata_series(j)
                    if not df.empty:
                        return df
            except Exception:
                # silent fallback to yfinance
                pass

        # Fallback: yfinance
        try:
            t = yf.Ticker(symbol)
            # approximate period based on outputsize
            years = max(1, int(outputsize / 252))
            df = t.history(period=f"{years}y", interval=interval)
            if not df.empty:
                df = df.rename(columns={c: c.title() for c in df.columns})
                cols = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in cols if c in df.columns]]
                return df
        except Exception:
            pass

        # try adding .CA
        try:
            guess = symbol.split(".")[0] + ".CA"
            t = yf.Ticker(guess)
            df = t.history(period="3y", interval=interval)
            if not df.empty:
                df = df.rename(columns={c: c.title() for c in df.columns})
                cols = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in cols if c in df.columns]]
                return df
        except Exception:
            pass

        return pd.DataFrame()

    @st.cache_data(ttl=CACHE_TTL)
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Quick price quote from TwelveData or yfinance."""
        if self.apikey:
            url = "https://api.twelvedata.com/price"
            try:
                resp = self.session.get(url, params={"symbol": symbol, "apikey": self.apikey}, timeout=5)
                j = resp.json()
                if "price" in j:
                    return {"price": safe_float(j.get("price")), "raw": j}
            except Exception:
                pass
        # yfinance fallback
        try:
            t = yf.Ticker(symbol)
            info = t.info if hasattr(t, "info") else {}
            price = info.get("regularMarketPrice") or info.get("previousClose") or None
            return {"price": safe_float(price), "raw": info}
        except Exception:
            return {"price": None, "raw": {}}

    @st.cache_data(ttl=3600)
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        try:
            t = yf.Ticker(symbol)
            info = t.info if hasattr(t, "info") else {}
            return info or {}
        except Exception:
            return {}

# ==========================
# TECHNICAL INDICATORS (light)
# ==========================
class TechnicalIndicators:
    @staticmethod
    def sma(series: pd.Series, period: int):
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
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

# ==========================
# SHARIA SCREENER (light)
# ==========================
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

    def screen(self, symbol: str) -> Tuple[bool, List[str]]:
        info = self.dm.get_company_info(symbol)
        issues = []
        # Very light heuristics using info fields available in yfinance
        sector = (info.get("sector") or "").lower()
        if any(x in sector for x in ["alcohol", "gambling", "tobacco", "weapons", "defense"]):
            issues.append(f"Prohibited sector: {sector}")

        market_cap = info.get("marketCap") or info.get("market_cap") or 0
        if market_cap and market_cap < self.config.min_market_cap:
            issues.append(f"Market cap too small: {market_cap}")

        # If we couldn't find any flag, return a benign result
        if not issues:
            issues.append("No obvious red flags (light screen)")
            return True, issues
        return False, issues

# ==========================
# ML Predictor (VERY LIGHT SCAFFOLD)
# ==========================
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

class MLPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["Close"] = df["Close"]
        f["SMA20"] = TechnicalIndicators.sma(df["Close"], 20)
        f["SMA50"] = TechnicalIndicators.sma(df["Close"], 50)
        f["RSI"] = TechnicalIndicators.rsi(df["Close"])
        macd, macdsig, macdhist = TechnicalIndicators.macd(df["Close"])
        f["MACD"] = macd
        f["MACD_Signal"] = macdsig
        f["Returns1"] = df["Close"].pct_change(1)
        f["Volume"] = df.get("Volume", pd.Series(np.nan, index=df.index))
        return f.dropna()

    def prepare(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02):
        f = self.create_features(df)
        fut = f["Close"].shift(-horizon) / f["Close"] - 1
        y = (fut > threshold).astype(int).iloc[:-horizon]
        X = f.iloc[:-horizon].drop(columns=["Close"])
        self.feature_cols = list(X.columns)
        return X, y

    def train(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02):
        try:
            X, y = self.prepare(df, horizon, threshold)
            if len(X) < 150 or y.nunique() < 2:
                return {"error": "Insufficient data to train (need >= ~150 rows and at least 2 classes)"}
            pipeline = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=100, n_jobs=1))])
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="accuracy")
            pipeline.fit(X, y)
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

# ==========================
# PORTFOLIO
# ==========================
class Portfolio:
    def __init__(self):
        ensure_portfolio()

    def buy(self, symbol: str, qty: int, price: float) -> Tuple[bool, str]:
        p = st.session_state["portfolio"]
        cost = qty * price
        if p["cash"] < cost:
            return False, "Insufficient cash"
        p["cash"] -= cost
        if symbol in p["positions"]:
            pos = p["positions"][symbol]
            new_qty = pos["quantity"] + qty
            new_avg = (pos["avg_price"] * pos["quantity"] + cost) / new_qty
            p["positions"][symbol] = {"quantity": new_qty, "avg_price": new_avg}
        else:
            p["positions"][symbol] = {"quantity": qty, "avg_price": price}
        p["history"].append({"action": "BUY", "symbol": symbol, "qty": qty, "price": price, "time": datetime.now()})
        return True, "Bought"

    def sell(self, symbol: str, qty: int, price: float) -> Tuple[bool, str]:
        p = st.session_state["portfolio"]
        if symbol not in p["positions"] or p["positions"][symbol]["quantity"] < qty:
            return False, "Not enough shares"
        p["positions"][symbol]["quantity"] -= qty
        p["cash"] += qty * price
        if p["positions"][symbol]["quantity"] == 0:
            del p["positions"][symbol]
        p["history"].append({"action": "SELL", "symbol": symbol, "qty": qty, "price": price, "time": datetime.now()})
        return True, "Sold"

    def total_value(self, price_map: Dict[str, float]) -> float:
        p = st.session_state["portfolio"]
        total = p["cash"]
        for s, pos in p["positions"].items():
            total += pos["quantity"] * price_map.get(s, 0.0)
        return total

# ==========================
# PLOTTING HELPERS
# ==========================
def plot_candles_with_indicators(df: pd.DataFrame, title: str = "Price & Indicators"):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    # EMA lines
    if "Close" in df.columns and len(df) >= 10:
        fig.add_trace(go.Scatter(x=df.index, y=TechnicalIndicators.ema(df["Close"], 20), name="EMA20", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=TechnicalIndicators.ema(df["Close"], 50), name="EMA50", line=dict(width=1)), row=1, col=1)
    # Volume
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    # RSI
    if "Close" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=TechnicalIndicators.rsi(df["Close"]), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=750)
    return fig

# ==========================
# MAIN APP
# ==========================
def main():
    st.set_page_config(page_title=DEFAULT_APP_TITLE, layout="wide")
    st.title(DEFAULT_APP_TITLE)

    # ensure defaults
    ensure_watchlist()
    ensure_portfolio()

    # runtime API key stored in session_state (if provided in UI)
    runtime_key = st.session_state.get("TWELVEDATA_API_KEY", DEFAULT_TWELVEDATA_API_KEY)
    data_manager = DataManager(apikey=runtime_key)
    sharia = ShariaScreener(ShariaConfig(), data_manager)
    ml = MLPredictor()
    pf = Portfolio()

    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Market Overview", "Company Page", "Watchlist", "Portfolio", "Settings"])

    # ---------- Market Overview ----------
    if page == "Market Overview":
        st.header("📊 Market Overview")
        st.markdown("Provide a list of tickers to fetch (EGX tickers often use .CA suffix on yfinance).")

        tickers_text = st.text_area("Tickers (comma separated)", value=",".join(st.session_state["watchlist"]), height=120)
        tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

        if not tickers:
            st.info("Add tickers to the text box to begin.")
        else:
            # fetch quotes & last closes
            results = []
            with st.spinner("Fetching data..."):
                for s in tickers:
                    df = data_manager.get_time_series(s, interval="1day", outputsize=365)
                    q = data_manager.get_quote(s)
                    last = None
                    prev = None
                    if not df.empty:
                        last = df["Close"].iloc[-1]
                        prev = df["Close"].iloc[-2] if len(df) > 1 else None
                    else:
                        last = q.get("price")
                    change_pct = None
                    if prev and prev != 0:
                        change_pct = (last - prev) / prev * 100
                    results.append({"symbol": s, "last": last, "change_pct": change_pct})

            summary_df = pd.DataFrame(results)
            if not summary_df.empty:
                summary_df["Change"] = summary_df["change_pct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                summary_df["Last Price"] = summary_df["last"].apply(lambda x: f"{x:.4g}" if pd.notna(x) else "N/A")
                st.dataframe(summary_df[["symbol", "Last Price", "Change"]].rename(columns={"symbol": "Ticker"}), height=350)
            else:
                st.info("No data available for these tickers.")

            # Select one to view chart
            st.markdown("---")
            st.markdown("### View chart for a ticker")
            ticker = st.selectbox("Chart ticker", options=tickers, index=0)
            if ticker:
                df = data_manager.get_time_series(ticker, interval="1day", outputsize=1200)
                if df.empty:
                    st.error("No historical price series available for this symbol.")
                else:
                    fig = plot_candles_with_indicators(df, title=f"{ticker} — Price & Indicators")
                    st.plotly_chart(fig, use_container_width=True)

    # ---------- Company Page ----------
    elif page == "Company Page":
        st.header("🏢 Company Page")
        # selection from watchlist or custom
        choices = st.session_state["watchlist"] + DEFAULT_WATCHLIST
        ticker = st.selectbox("Select ticker", options=list(dict.fromkeys(choices)))
        if ticker:
            with st.spinner("Fetching series & info..."):
                df = data_manager.get_time_series(ticker, interval="1day", outputsize=1200)
                info = data_manager.get_company_info(ticker)
                quote = data_manager.get_quote(ticker)

            # Header metrics
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader(f"{ticker} — {info.get('shortName') or info.get('longName') or ''}")
            with col2:
                price_display = quote.get("price") or (df["Close"].iloc[-1] if not df.empty else "N/A")
                st.metric("Last Price", f"{price_display}")
            with col3:
                market_cap = info.get("marketCap") or "N/A"
                st.metric("Market Cap", f"{market_cap}")

            # Chart
            if not df.empty:
                st.plotly_chart(plot_candles_with_indicators(df, title=f"{ticker} Price Chart"), use_container_width=True)
            else:
                st.warning("No price series available — showing fundamentals if available.")

            # Fundamentals (light)
            st.markdown("### Fundamentals (light)")
            colf1, colf2 = st.columns(2)
            with colf1:
                st.write("Sector:", info.get("sector"))
                st.write("Industry:", info.get("industry"))
                st.write("Website:", info.get("website"))
            with colf2:
                st.write("Trailing P/E:", info.get("trailingPE"))
                st.write("Forward P/E:", info.get("forwardPE"))
                st.write("Dividend yield:", info.get("dividendYield"))

            # Sharia screening (light)
            st.markdown("### 🕌 Sharia Screening (light)")
            compliant, issues = sharia.screen(ticker)
            if compliant:
                st.success("Light Sharia screen: No obvious red flags")
            else:
                st.error("Potential issues found:")
            for it in issues:
                st.write("- " + str(it))

            # ML: train & predict (optional)
            st.markdown("### 🤖 AI Prediction (optional)")
            horizon = st.slider("Prediction horizon (days)", 1, 20, 5)
            threshold_pct = st.slider("Movement threshold (%)", 1, 10, 2)
            train_col, pred_col = st.columns(2)
            with train_col:
                if st.button("Train model for this ticker"):
                    if df.empty:
                        st.error("Not enough data to train.")
                    else:
                        res = ml.train(df, horizon=horizon, threshold=threshold_pct / 100.0)
                        if "error" in res:
                            st.error(res["error"])
                        else:
                            st.success(f"Trained. CV Accuracy: {res['cv_mean']:.2%} ± {res['cv_std']:.2%}")
            with pred_col:
                if st.button("Run prediction (latest)"):
                    res = ml.predict(df)
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        st.info(f"Prob Up: {res['prob_up']:.2%} — Prob Down: {res['prob_down']:.2%}")

    # ---------- Watchlist ----------
    elif page == "Watchlist":
        st.header("⭐ Watchlist")
        st.write("Current watchlist:")
        st.write(st.session_state["watchlist"])

        add_col, rem_col = st.columns(2)
        with add_col:
            new_t = st.text_input("Add ticker (e.g. RAYA.CA)", "")
            if st.button("Add ticker"):
                if new_t and new_t not in st.session_state["watchlist"]:
                    st.session_state["watchlist"].append(new_t.strip().upper())
                    st.success(f"Added {new_t.strip().upper()}")
        with rem_col:
            rem = st.selectbox("Remove ticker", options=[""] + st.session_state["watchlist"])
            if st.button("Remove selected ticker"):
                if rem and rem in st.session_state["watchlist"]:
                    st.session_state["watchlist"].remove(rem)
                    st.success(f"Removed {rem}")

    # ---------- Portfolio ----------
    elif page == "Portfolio":
        st.header("💼 Portfolio Simulator")
        ensure_portfolio()
        p = st.session_state["portfolio"]
        st.metric("Cash balance", f"${p['cash']:,.2f}")
        st.write("Positions:")
        st.write(p["positions"])

        col1, col2 = st.columns(2)
        with col1:
            tick = st.selectbox("Ticker", options=st.session_state["watchlist"])
            qty = st.number_input("Quantity", min_value=1, value=100, step=1)
        with col2:
            if st.button("Buy"):
                q = data_manager.get_quote(tick)
                price = q.get("price") or None
                if price is None:
                    df_tmp = data_manager.get_time_series(tick, interval="1day", outputsize=10)
                    if df_tmp.empty:
                        st.error("Price unavailable; cannot execute trade.")
                        price = None
                    else:
                        price = df_tmp["Close"].iloc[-1]
                if price:
                    ok, msg = pf.buy(tick, int(qty), float(price))
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            if st.button("Sell"):
                q = data_manager.get_quote(tick)
                price = q.get("price") or None
                if price is None:
                    df_tmp = data_manager.get_time_series(tick, interval="1day", outputsize=10)
                    if df_tmp.empty:
                        st.error("Price unavailable; cannot execute trade.")
                        price = None
                    else:
                        price = df_tmp["Close"].iloc[-1]
                if price:
                    ok, msg = pf.sell(tick, int(qty), float(price))
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        # Portfolio value
        # build price map for positions
        price_map = {}
        for s in p["positions"].keys():
            q = data_manager.get_quote(s)
            price_map[s] = q.get("price") or 0.0
        total_val = pf.total_value(price_map)
        st.metric("Total portfolio value", f"${total_val:,.2f}")

        # history
        st.markdown("Transaction history (latest 20)")
        hist_df = pd.DataFrame(p["history"][-20:]) if p["history"] else pd.DataFrame()
        if not hist_df.empty:
            st.dataframe(hist_df)
        else:
            st.info("No transactions yet.")

    # ---------- Settings ----------
    elif page == "Settings":
        st.header("⚙️ Settings & Keys")
        st.markdown("Provide a TwelveData API key to improve EGX coverage (optional). Store your key in session for runtime or use environment variable or Streamlit Secrets for persistence.")

        current_key = st.session_state.get("TWELVEDATA_API_KEY", DEFAULT_TWELVEDATA_API_KEY)
        key_input = st.text_input("TwelveData API Key", value=current_key, type="password")

        if st.button("Save API Key (runtime)"):
            st.session_state["TWELVEDATA_API_KEY"] = key_input.strip()
            st.success("Saved for current session (will reset if app restarts). Please reload page to use the new key for cached functions.")

        if st.button("Clear runtime API Key"):
            if "TWELVEDATA_API_KEY" in st.session_state:
                del st.session_state["TWELVEDATA_API_KEY"]
            st.success("Cleared runtime API key.")

        st.markdown("---")
        st.markdown("**Notes & next steps for production**")
        st.write("""
        - For production-grade real-time EGX data, subscribe to licensed feeds (EGX FIX, Mubasher, broker feed).
        - Replace the light Sharia checks with full financial-statement based screening (use licensed fundamentals).
        - Persist watchlists & portfolios to a DB instead of session_state for multi-user setups.
        - Add authentication, job scheduler (e.g. APScheduler, Celery) for recurring data refreshes, and a microservice backend if scaling.
        """)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ — tell me which production integration you'd like next (Mubasher feed, EGX FIX, or broker API).")

if __name__ == "__main__":
    main()
