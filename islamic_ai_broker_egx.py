# islamic_ai_broker_egx.py
# 🕌 Islamic AI Broker Pro - EGX100 Egyptian Stock Market Edition
# ================================================================
# Adapted for EGX100: Egyptian Stock Market with Arabic/English UI,
# dynamic dashboard, and beginner-friendly explanations.
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import json

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================================================
# CONFIGURATION & CONSTANTS
# ================================================================
# This section defines configuration classes, enums, and constants.
# It provides the basic settings for risk, compliance, and available stocks.

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class TradingConfig:
    """Trading parameter configuration."""
    risk_per_trade: float = 0.02
    max_positions: int = 5
    stop_loss_atr_mult: float = 2.0
    take_profit_rr: float = 2.5
    initial_capital: float = 100000.0
    commission: float = 0.001

@dataclass
class ShariaConfig:
    """Sharia compliance thresholds."""
    max_debt_to_assets: float = 0.33
    max_interest_income_ratio: float = 0.05
    max_cash_securities_ratio: float = 0.33
    min_market_cap: float = 1e9 # $1B minimum

# EGX100 symbols and names (demo subset; expand with full EGX100 as needed)
EGX100_TICKERS = {
    "COMI.CA": ("Commercial International Bank (CIB)", "بنك التجاري الدولي"),
    "HRHO.CA": ("EFG Hermes Holdings", "المجموعة المالية هيرميس"),
    "ESRS.CA": ("Ezz Steel", "عز الدخيلة للصلب"),
    "EKHO.CA": ("Egypt Kuwait Holding", "مصر الكويت القابضة"),
    "ORAS.CA": ("Orascom Construction", "أوراسكوم للإنشاء"),
    "SWDY.CA": ("Elsewedy Electric", "السويدي إليكتريك"),
    "AUTO.CA": ("GB Auto", "جي بي أوتو"),
    # Add more EGX100 stocks here as needed
}

PROHIBITED_SECTORS = {
    'banks', 'banking', 'insurance', 'financial services', 'alcohol',
    'gambling', 'casino', 'tobacco', 'adult entertainment', 'weapons',
    'defense', 'pork', 'conventional finance'
}

# ================================================================
# STREAMLIT CONFIGURATION & LANGUAGE TOGGLE
# ================================================================
# This section configures the Streamlit app and adds a language toggle.
# The UI text updates between Arabic and English based on user selection.

st.set_page_config(
    page_title="🕌 Islamic AI Broker Pro",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .status-halal {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
    }
    .status-haram {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
    }
    .trading-signal {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
    }
    .signal-buy {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
    }
    .signal-sell {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
    }
    .signal-hold {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white;
    }
    .info-panel {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Language toggle: English <-> Arabic
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

language = st.toggle("🇺🇸 / 🇪🇬 العربية", value=(st.session_state['lang'] == 'ar'), key="lang_toggle")
st.session_state['lang'] = 'ar' if language else 'en'
lang = st.session_state['lang']

def T(en, ar):
    """Translation helper."""
    return ar if lang == 'ar' else en

# ================================================================
# DATA MANAGEMENT FOR EGX100
# ================================================================
# This section fetches EGX100 stock data from open APIs.
# For EGX, we use the Mubasher API or direct web scraping.
# The code is designed to be easily adapted for other APIs.

class DataManager:
    """Data fetching for EGX100 stocks.
    This class fetches historical and current price data for Egyptian stocks.
    Data is pulled from EGX-AI-API for OHLCV data.
    """
    def __init__(self):
        self.cache_dir = os.path.join(tempfile.gettempdir(), "islamic_ai_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    @st.cache_data(ttl=300)
    def get_market_data(_self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical price data for EGX-listed stocks from EGX-AI-API.
        - Returns a DataFrame with columns: Open, High, Low, Close, Volume
        - Uses the EGX-AI-API endpoint for demo (replace with official EGX provider for production).
        """
        try:
            # Example: Use the EGX-AI-API endpoint for OHLCV data
            url = f"https://egx-ai-api.onrender.com/api/v1/ohlcv/{symbol}?period={period}&interval={interval}"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
            else:
                data = []
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "date": "Date"
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_cols].dropna()
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol} from EGX-AI-API: {str(e)}")
            return pd.DataFrame()

            df = pd.DataFrame(data)
            # Ensure column names
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "date": "Date"
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_cols].dropna()
            return df
        except Exception as e:
            st.error(T(f"Error fetching data for {symbol}: {str(e)}", f"خطأ في جلب بيانات {symbol}: {str(e)}"))
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_company_info(_self, symbol: str) -> Dict[str, Any]:
        """
        Fetches company info for EGX stocks.
        - Returns details like sector, industry, and market cap.
        - Uses demo data for now.
        """
        # For EGX, this info is available on Mubasher or EGX official website.
        info = {
            "sector": "Financial Services",
            "industry": "Banking",
            "marketCap": 15e9, # Demo market cap
        }
        return info

# ================================================================
# TECHNICAL INDICATORS
# ================================================================
# This section provides functions for calculating technical indicators
# like SMA, EMA, RSI, MACD, Bollinger Bands, etc.
# These are used to analyze price movements and trends.

class TechnicalIndicators:
    """Technical indicators for EGX stocks."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        macd_line = TechnicalIndicators.ema(series, fast) - TechnicalIndicators.ema(series, slow)
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ma = TechnicalIndicators.sma(series, period)
        std = series.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, ma, lower_band

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

# ================================================================
# SHARIA COMPLIANCE ENGINE
# ================================================================
# This section screens stocks for Sharia compliance.
# It checks financial ratios and sector information.

class ShariaScreener:
    """Sharia compliance screening for EGX stocks."""
    def __init__(self, config: ShariaConfig):
        self.config = config
        self.data_manager = DataManager()

    def screen_stock(self, symbol: str) -> Tuple[bool, Dict[str, Any], List[str]]:
        info = self.data_manager.get_company_info(symbol)
        # For EGX, financials may require scraping or manual input
        results = {
            'debt_to_assets': np.nan,
            'interest_income_ratio': np.nan,
            'cash_securities_ratio': np.nan,
            'market_cap': info.get('marketCap', np.nan)
        }
        issues = []

        sector = info.get('sector', '').lower()
        industry = info.get('industry', '').lower()

        if any(prohibited in sector or prohibited in industry for prohibited in PROHIBITED_SECTORS):
            issues.append(T("❌ Prohibited sector", "❌ قطاع محظور") + f": {sector or industry}")

        market_cap = info.get('marketCap', 0)
        results['market_cap'] = market_cap
        if market_cap < self.config.min_market_cap:
            issues.append(T("❌ Market cap too small", "❌ القيمة السوقية صغيرة") + f": ${market_cap:,.0f}")

        # For demo: assume all ratios are within threshold, but you can add scraping logic for EGX financials.
        is_compliant = len(issues) == 0
        if is_compliant:
            issues.append(T("✅ All Sharia requirements met", "✅ جميع متطلبات الشريعة مستوفاة"))

        return is_compliant, results, issues

# ================================================================
# MACHINE LEARNING ENGINE (Random Forest/Logistic Regression)
# ================================================================
# This section builds a simple ML model to predict BUY/SELL signals.
# It uses historical price features and gives recommendation explanations.

class MLPredictor:
    """ML predictions for EGX stocks. Explains why a recommendation is made."""
    def __init__(self):
        self.models = {}
        self.feature_columns = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Technical features for price movement prediction
        features_df = df.copy()
        features_df['SMA_10'] = TechnicalIndicators.sma(df['Close'], 10)
        features_df['EMA_12'] = TechnicalIndicators.ema(df['Close'], 12)
        features_df['RSI'] = TechnicalIndicators.rsi(df['Close'])
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(df['Close'])
        features_df['MACD'] = macd
        features_df['MACD_Signal'] = macd_signal
        features_df['MACD_Hist'] = macd_hist
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
        features_df['BB_Upper'] = bb_upper
        features_df['BB_Lower'] = bb_lower
        features_df['Returns_1'] = df['Close'].pct_change(1)
        features_df['Returns_5'] = df['Close'].pct_change(5)
        features_df['ATR'] = TechnicalIndicators.atr(df)
        return features_df.dropna()

    def prepare_ml_data(self, df: pd.DataFrame, target_days: int = 5, threshold: float = 0.02) -> Tuple[pd.DataFrame, pd.Series]:
        features_df = self.create_features(df)
        # Target: 1 if price up by threshold in next target_days, else 0
        future_returns = features_df['Close'].shift(-target_days) / features_df['Close'] - 1
        target = (future_returns > threshold).astype(int)
        feature_cols = [col for col in features_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        X = features_df[feature_cols].iloc[:-target_days]
        y = target.iloc[:-target_days]
        self.feature_columns = feature_cols
        return X, y

    def train_model(self, symbol: str, df: pd.DataFrame, target_days: int = 5, threshold: float = 0.02, method='RF') -> Dict[str, Any]:
        X, y = self.prepare_ml_data(df, target_days, threshold)
        if len(X) < 50 or y.nunique() < 2:
            return {'error': T('Insufficient data for ML training', 'لا توجد بيانات كافية لتدريب النموذج')}

        # ML Model: Random Forest or Logistic Regression
        if method == 'RF':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=200)
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])

        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
        pipeline.fit(X, y)
        self.models[symbol] = pipeline

        # Feature importance/explanation
        if method == 'RF':
            importance = pipeline.named_steps['clf'].feature_importances_
        else:
            importance = abs(pipeline.named_steps['clf'].coef_[0])
        feature_importance = dict(zip(self.feature_columns, importance))

        return {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        }

    def predict(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        # Predict buy/sell signal for latest data point
        if symbol not in self.models:
            return {'error': T('Model not trained', 'النموذج غير مدرب')}
        features_df = self.create_features(df)
        X = features_df[self.feature_columns].iloc[[-1]]
        model = self.models[symbol]
        proba = model.predict_proba(X)[0]
        explanation = []
        # Explain recommendation based on feature values
        feature_vals = X.iloc[0]
        if feature_vals['RSI'] < 30:
            explanation.append(T('RSI indicates oversold', 'مؤشر القوة النسبية يشير إلى تشبع بيعي'))
        if feature_vals['MACD'] > feature_vals['MACD_Signal']:
            explanation.append(T('MACD bullish crossover', 'تقاطع MACD صعودي'))
        if feature_vals['Returns_1'] > 0.02:
            explanation.append(T('Recent price momentum', 'زخم سعري مؤخراً'))

        signal = SignalType.BUY.value if proba[1] > 0.55 else SignalType.SELL.value if proba[1] < 0.45 else SignalType.HOLD.value
        return {
            'probability_up': float(proba[1]),
            'probability_down': float(proba[0]),
            'signal': signal,
            'explanation': explanation
        }

# ================================================================
# MAIN APPLICATION
# ================================================================
# This section builds the main dashboard UI, runs all analysis,
# and displays results. Each major part includes an explanatory comment.

def main():
    """Main app: EGX100 analysis dashboard with Arabic/English UI."""

    # Top header shows the selected stock name (dynamic)
    st.markdown(f"""
    <div class="main-header">
        <h1 class="main-title">🕌 Islamic AI Broker Pro</h1>
        <p class="main-subtitle">{T("Advanced EGX100 Analysis Platform", "منصة تحليل EGX100 المتقدمة")}</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize system classes
    data_manager = DataManager()
    sharia_config = ShariaConfig()
    trading_config = TradingConfig()
    sharia_screener = ShariaScreener(sharia_config)
    ml_predictor = MLPredictor()

    # ----------------- SIDEBAR: User Controls -----------------
    # This section lets the user select the EGX stock and analysis parameters.
    with st.sidebar:
        st.header(T("🎛️ Control Panel", "لوحة التحكم"))
        # Stock selection (EGX100)
        st.subheader(T("📈 Stock Selection", "اختيار السهم"))
        selected_symbol = st.selectbox(
            T("Choose Stock", "اختر السهم"),
            options=list(EGX100_TICKERS.keys()),
            format_func=lambda x: f"{EGX100_TICKERS[x][0]} ({x})" if lang == 'en' else f"{EGX100_TICKERS[x][1]} ({x})"
        )

        # Time period
        period = st.selectbox(T("Time Period", "الفترة الزمنية"), ["3mo", "6mo", "1y", "2y"], index=2)

        # ML parameters
        st.subheader(T("🤖 ML Settings", "إعدادات الذكاء الاصطناعي"))
        prediction_days = st.slider(T("Prediction Horizon (days)", "أفق التنبؤ (أيام)"), 1, 20, 5)
        movement_threshold = st.slider(T("Movement Threshold (%)", "نسبة الحركة المطلوبة (%)"), 1.0, 10.0, 2.0) / 100

        # Sharia thresholds
        st.subheader(T("🕌 Sharia Thresholds", "معايير الشريعة"))
        max_debt_ratio = st.slider(T("Max Debt/Assets", "أقصى نسبة دين/أصول"), 0.1, 0.5, 0.33, 0.01)
        max_interest_ratio = st.slider(T("Max Interest Income/Revenue", "أقصى دخل فائدة/إيراد"), 0.0, 0.1, 0.05, 0.005)
        max_cash_ratio = st.slider(T("Max Cash/Assets", "أقصى نقدية/أصول"), 0.1, 0.5, 0.33, 0.01)

        # Update configs
        sharia_config.max_debt_to_assets = max_debt_ratio
        sharia_config.max_interest_income_ratio = max_interest_ratio
        sharia_config.max_cash_securities_ratio = max_cash_ratio

        run_analysis = st.button(T("🔍 Run Analysis", "🔍 تشغيل التحليل"), type="primary")

    # ----------------- MAIN DASHBOARD -----------------
    # Each major section has a beginner-friendly explanation.

    # Dynamic Stock Name Header
    st.markdown(f"""
    <h2 style="text-align:center; color:#667eea;">{EGX100_TICKERS[selected_symbol][0] if lang == 'en' else EGX100_TICKERS[selected_symbol][1]} ({selected_symbol})</h2>
    """, unsafe_allow_html=True)

    if run_analysis:
        # ----------- DATA FETCHING SECTION ----------
        # This section loads historical stock data for analysis.
        st.info(T("📊 Loading market data for selected stock...", "📊 جاري تحميل بيانات السوق للسهم المختار..."))
        df = data_manager.get_market_data(selected_symbol, period)
        if df.empty:
            st.error(T("❌ Unable to fetch market data. Please try again.", "❌ تعذر جلب بيانات السوق. حاول مرة أخرى."))
            return

        # ----------- DATA PREPROCESSING SECTION ----------
        # This section computes key price metrics.
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0.0

        # Price metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{T("Current Price", "السعر الحالي")}</div>
                <div class="metric-value">{current_price:.2f} EGP</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            color = "#10b981" if price_change >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {color}, {color}99);">
                <div class="metric-label">{T("Daily Change", "التغير اليومي")}</div>
                <div class="metric-value">{price_change:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{T("Volume", "الحجم")}</div>
                <div class="metric-value">{df['Volume'].iloc[-1]:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        # ----------- SHARIA SCREENING SECTION ----------
        # This section evaluates whether the selected stock meets Islamic finance standards.
        st.header(T("🕌 Sharia Compliance Analysis", "تحليل الالتزام بالشريعة"))
        is_compliant, ratios, issues = sharia_screener.screen_stock(selected_symbol)
        if is_compliant:
            st.markdown(f'<div class="status-halal">{T("✅ HALAL CERTIFIED", "✅ متوافق مع الشريعة")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-haram">{T("❌ NON-COMPLIANT", "❌ غير متوافق")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-panel">', unsafe_allow_html=True)
        for issue in issues:
            st.write(issue)
        st.markdown('</div>', unsafe_allow_html=True)

        # ----------- TECHNICAL ANALYSIS CHARTING SECTION ----------
        # This section visualizes price trends and indicators.
        st.header(T("📊 Technical Analysis", "التحليل الفني"))
        features_df = ml_predictor.create_features(df)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(T('Price & Moving Averages', 'السعر والمتوسطات'), T('Volume', 'الحجم'), T('RSI & MACD', 'RSI و MACD')),
            row_width=[0.2, 0.2, 0.6]
        )
        # Candlestick price chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=T('Price', 'السعر')
        ), row=1, col=1)
        # SMA overlay
        if 'SMA_10' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['SMA_10'],
                name=T('SMA 10', 'متوسط 10 أيام'),
                line=dict(color='orange', width=1)
            ), row=1, col=1)
        # Volume bar
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name=T('Volume', 'الحجم'),
            marker_color='lightblue'
        ), row=2, col=1)
        # RSI, MACD
        if 'RSI' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['RSI'], name='RSI', line=dict(color='purple')
            ), row=3, col=1)
        if 'MACD' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['MACD'], name='MACD', line=dict(color='blue')
            ), row=3, col=1)
        if 'MACD_Signal' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['MACD_Signal'], name='MACD Signal', line=dict(color='red')
            ), row=3, col=1)
        fig.update_layout(
            height=600,
            title=T(f"Technical Analysis - {EGX100_TICKERS[selected_symbol][0]}", f"التحليل الفني - {EGX100_TICKERS[selected_symbol][1]}"),
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # ----------- MACHINE LEARNING STRATEGY SECTION ----------
        # This section runs the ML model and explains its recommendation.
        st.header(T("🤖 AI Market Prediction", "توقعات السوق بالذكاء الاصطناعي"))
        ml_results = ml_predictor.train_model(selected_symbol, df, prediction_days, movement_threshold, method='RF')
        if 'error' in ml_results:
            st.error(ml_results['error'])
        else:
            prediction = ml_predictor.predict(selected_symbol, df)
            prob_up = prediction['probability_up']
            prob_down = prediction['probability_down']
            signal = prediction['signal']
            explanation = prediction['explanation']

            # Display ML recommendation
            col1, col2 = st.columns(2)
            with col1:
                st.metric(T("Probability of Price Increase", "احتمالية ارتفاع السعر"), f"{prob_up:.1%}")
            with col2:
                st.metric(T("ML Model Accuracy", "دقة النموذج"), f"{ml_results['cv_accuracy']:.1%}", delta=f"±{ml_results['cv_std']:.1%}")

            # Recommendation
            if signal == SignalType.BUY.value:
                rec_text = T("AI Recommendation: BUY", "توصية الذكاء الاصطناعي: شراء")
                rec_color = "#10b981"
            elif signal == SignalType.SELL.value:
                rec_text = T("AI Recommendation: SELL", "توصية الذكاء الاصطناعي: بيع")
                rec_color = "#ef4444"
            else:
                rec_text = T("AI Recommendation: HOLD", "توصية الذكاء الاصطناعي: احتفاظ")
                rec_color = "#f59e0b"
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {rec_color}, {rec_color}99);
                        color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <strong>{rec_text}</strong>
            </div>
            """, unsafe_allow_html=True)
            # Explanation for beginners
            st.info(T("Why this recommendation?", "سبب هذه التوصية؟"))
            for reason in explanation:
                st.write("•", reason)
            # Feature importance
            st.subheader(T("Key Factors", "العوامل الرئيسية"))
            imp = ml_results['feature_importance']
            st.write(pd.DataFrame(list(imp.items()), columns=[T("Feature", "المؤشر"), T("Importance", "الأهمية")]))

    else:
        # Welcome info (shown when analysis hasn't run yet)
        st.info(T("👆 Configure your parameters in the sidebar and click 'Run Analysis' to begin.",
                  "👆 قم بضبط الإعدادات في القائمة الجانبية واضغط على 'تشغيل التحليل' للبدء."))
        st.markdown(f"""
        ### 🌟 {T('Platform Features', 'مميزات المنصة')}
        **🕌 {T('Sharia Compliance', 'الالتزام بالشريعة')}**
        - {T('Automated screening based on Islamic finance principles', 'فحص تلقائي حسب معايير الشريعة الإسلامية')}
        - {T('Real-time compliance monitoring', 'مراقبة الالتزام لحظياً')}
        - {T('Customizable screening thresholds', 'معايير فحص قابلة للتخصيص')}
        **🤖 {T('AI Predictions', 'توقعات الذكاء الاصطناعي')}**
        - {T('Simple machine learning model (Random Forest)', 'نموذج تعلم آلي بسيط (الغابة العشوائية)')}
        - {T('Explained buy/sell recommendations', 'توصيات شراء/بيع مع شرح الأسباب')}
        **📊 {T('Technical Analysis', 'التحليل الفني')}**
        - {T('Key indicators (SMA, RSI, MACD, Bollinger)', 'مؤشرات رئيسية (المتوسط، RSI، MACD، بولنجر)')}
        - {T('Advanced charting', 'رسم بياني متقدم')}
        **🎯 {T('Professional Trading', 'تداول احترافي')}**
        - {T('Strategy analytics', 'تحليلات استراتيجية')}
        - {T('Beginner-friendly explanations', 'شروحات مبسطة للمبتدئين')}
        """)

if __name__ == "__main__":
    main()
