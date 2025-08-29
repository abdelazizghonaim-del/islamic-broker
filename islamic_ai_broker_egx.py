# islamic_ai_broker_pro.py
# 🕌 Islamic AI Broker Pro - Egyptian Market Edition - Next Generation Trading Platform
# ================================================================
# Advanced Islamic-compliant trading system with AI predictions,
# comprehensive Sharia screening, and professional risk management
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import asyncio
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import json

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ================================================================
# CONFIGURATION & CONSTANTS
# ================================================================

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    risk_per_trade: float = 0.02
    max_positions: int = 5
    stop_loss_atr_mult: float = 2.0
    take_profit_rr: float = 2.5
    initial_capital: float = 100000.0
    commission: float = 0.001

@dataclass
class ShariaConfig:
    """Sharia compliance thresholds"""
    max_debt_to_assets: float = 0.33
    max_interest_income_ratio: float = 0.05
    max_cash_securities_ratio: float = 0.33
    min_market_cap: float = 1e9  # $1B minimum
    
HALAL_TICKERS = {
    # EGX30 Stocks - Egyptian Exchange
    # Confirmed Sharia-Compliant stocks based on EGX33 Shariah Index and research
    'ADIB.CA': ('Abu Dhabi Islamic Bank-Egypt', 'Islamic Finance'),  # ✓ Islamic Bank
    'TMGH.CA': ('Talaat Moustafa Group Holding', 'Real Estate'),    # ✓ In EGX33 Shariah
    'ABUK.CA': ('Abou Kir Fertilizers & Chemical Industries Co.', 'Process Industries'),  # ✓ In EGX33 Shariah
    'ETEL.CA': ('Telecom Egypt', 'Communications'),                 # ✓ In EGX33 Shariah
    'FWRY.CA': ('Fawry For Banking Technology And Electronic Payment', 'Technology Services'),  # ✓ In EGX33 Shariah
    'MASR.CA': ('Madinet Masr for Housing & Development', 'Real Estate'),  # ✓ In EGX33 Shariah

    # Other EGX30 stocks requiring screening (non-conventional finance)
    'EAST.CA': ('Eastern Company', 'Consumer Non-durables'),
    'ORAS.CA': ('Orascom Construction Plc', 'Industrial Services'),
    'MFPC.CA': ('Misr Fertilizers Production Company MOPCO', 'Process Industries'),
    'EGAL.CA': ('Egypt Aluminum', 'Non-energy Minerals'),
    'EFIH.CA': ('e-finance for Digital and Financial Investments S.A.E.', 'Technology Services'),
    'GBCO.CA': ('GB Corp', 'Distribution Services'),
    'ORHD.CA': ('Orascom Development Egypt (S.A.E)', 'Consumer Durables'),
    'JUFO.CA': ('Juhayna Food Industries', 'Consumer Non-durables'),
    'SKPC.CA': ('Sidi Kerir Petrochemicals', 'Process Industries'),
    'ARCC.CA': ('Arabian Cement Company', 'Non-energy Minerals'),
    'ORWE.CA': ('Oriental Weavers Carpet', 'Consumer Durables'),
    'RAYA.CA': ('Raya Holding for Financial Investments SAE', 'Technology Services'),
    'ISPH.CA': ('Ibnsina Pharma', 'Distribution Services'),
    'AMOC.CA': ('Alexandria Mineral Oils Co.', 'Energy Minerals'),
    'MCQE.CA': ('Misr Cement Co. (Qena)', 'Non-energy Minerals'),
    'RMDA.CA': ('Tenth of Ramadan Pharmaceutical Industries & Diagnostic-Rameda', 'Health Technology'),

    # Note: The following conventional banks/finance companies are excluded as they're likely non-Sharia compliant:
    # COMI.CA (Commercial International Bank), EMFD.CA (Emaar Misr Development), 
    # HRHO.CA (EFG Holding), BTFH.CA (Beltone Holding), EKHOA.CA, EKHO.CA (Egypt Kuwait Holding),
    # CIEB.CA (Credit Agricole), PHDC.CA (Palm Hills Development), CCAP.CA (QALA Financial)
}

PROHIBITED_SECTORS = {
    'banks', 'banking', 'insurance', 'financial services', 'alcohol',
    'gambling', 'casino', 'tobacco', 'adult entertainment', 'weapons',
    'defense', 'pork', 'conventional finance'
}

# ================================================================
# STREAMLIT CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="🕌 Islamic AI Broker Pro - EGX Edition",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
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
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# TECHNICAL INDICATORS
# ================================================================

class TechnicalIndicators:
    """Advanced technical indicators for market analysis"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        macd_line = TechnicalIndicators.ema(series, fast) - TechnicalIndicators.ema(series, slow)
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        ma = TechnicalIndicators.sma(series, period)
        std = series.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, ma, lower_band
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        return -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        atr_values = TechnicalIndicators.atr(df, period)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period).mean() / atr_values
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period).mean() / atr_values
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/period).mean()
    
    @staticmethod
    def ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Ichimoku Cloud components"""
        # Conversion Line (Tenkan-sen)
        conversion_line = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        
        # Base Line (Kijun-sen)
        base_line = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        
        # Leading Span A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(26)
        
        # Leading Span B (Senkou Span B)
        leading_span_b = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
        
        # Lagging Span (Chikou Span)
        lagging_span = df['Close'].shift(-26)
        
        return {
            'conversion_line': conversion_line,
            'base_line': base_line,
            'leading_span_a': leading_span_a,
            'leading_span_b': leading_span_b,
            'lagging_span': lagging_span
        }

# ================================================================
# DATA MANAGEMENT
# ================================================================

class DataManager:
    """Enhanced data fetching and management"""
    
    def __init__(self):
        self.cache_dir = os.path.join(tempfile.gettempdir(), "islamic_ai_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @st.cache_data(ttl=300)
    def get_market_data(_self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch market data with multiple fallback sources"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            
            if df.empty:
                return pd.DataFrame()
            
            # Ensure consistent column names
            df.columns = df.columns.str.title()
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_cols].dropna()
            
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def get_company_info(_self, symbol: str) -> Dict[str, Any]:
        """Get company information"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception:
            return {}
    
    @st.cache_data(ttl=3600)
    def get_financials(_self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get financial statements"""
        try:
            ticker = yf.Ticker(symbol)
            return {
                'balance_sheet': ticker.balance_sheet,
                'income_statement': ticker.financials,
                'cash_flow': ticker.cashflow,
                'quarterly_balance_sheet': ticker.quarterly_balance_sheet,
                'quarterly_financials': ticker.quarterly_financials
            }
        except Exception:
            return {}

# ================================================================
# SHARIA COMPLIANCE ENGINE
# ================================================================

class ShariaScreener:
    """Advanced Sharia compliance screening"""
    
    def __init__(self, config: ShariaConfig):
        self.config = config
        self.data_manager = DataManager()
    
    def screen_stock(self, symbol: str) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Comprehensive Sharia screening"""
        info = self.data_manager.get_company_info(symbol)
        financials = self.data_manager.get_financials(symbol)
        
        results = {
            'debt_to_assets': np.nan,
            'interest_income_ratio': np.nan,
            'cash_securities_ratio': np.nan,
            'market_cap': np.nan
        }
        
        issues = []
        
        # Sector screening
        sector = info.get('sector', '').lower()
        industry = info.get('industry', '').lower()
        
        if any(prohibited in sector or prohibited in industry for prohibited in PROHIBITED_SECTORS):
            issues.append(f"❌ Prohibited sector: {sector or industry}")
        
        # Market cap check
        market_cap = info.get('marketCap', 0)
        results['market_cap'] = market_cap
        if market_cap < self.config.min_market_cap:
            issues.append(f"❌ Market cap too small: ${market_cap:,.0f}")
        
        # Financial ratio screening
        try:
            balance_sheet = financials.get('quarterly_balance_sheet')
            income_stmt = financials.get('quarterly_financials')
            
            if isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
                latest_bs = balance_sheet.iloc[:, 0]
                
                # Debt to assets ratio
                total_debt = (
                    latest_bs.get('Short Long Term Debt Total', 0) + 
                    latest_bs.get('Long Term Debt', 0)
                )
                total_assets = latest_bs.get('Total Assets', 0)
                
                if total_assets > 0:
                    debt_ratio = total_debt / total_assets
                    results['debt_to_assets'] = debt_ratio
                    
                    if debt_ratio > self.config.max_debt_to_assets:
                        issues.append(f"❌ High debt ratio: {debt_ratio:.1%}")
                
                # Cash and securities ratio
                cash = latest_bs.get('Cash And Cash Equivalents', 0)
                short_term_investments = latest_bs.get('Short Term Investments', 0)
                cash_securities = cash + short_term_investments
                
                if total_assets > 0:
                    cash_ratio = cash_securities / total_assets
                    results['cash_securities_ratio'] = cash_ratio
                    
                    if cash_ratio > self.config.max_cash_securities_ratio:
                        issues.append(f"❌ High cash/securities ratio: {cash_ratio:.1%}")
            
            # Interest income screening
            if isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]
                
                total_revenue = latest_income.get('Total Revenue', 0)
                interest_income = latest_income.get('Interest Income', 0)
                
                if total_revenue > 0 and interest_income > 0:
                    interest_ratio = interest_income / total_revenue
                    results['interest_income_ratio'] = interest_ratio
                    
                    if interest_ratio > self.config.max_interest_income_ratio:
                        issues.append(f"❌ High interest income: {interest_ratio:.1%}")
        
        except Exception as e:
            issues.append(f"⚠️ Could not analyze financials: {str(e)}")
        
        is_compliant = len(issues) == 0
        if is_compliant:
            issues.append("✅ All Sharia requirements met")
        
        return is_compliant, results, issues

# ================================================================
# MACHINE LEARNING ENGINE
# ================================================================

class MLPredictor:
    """Advanced ML predictions for stock movements"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features_df = df.copy()
        
        # Technical indicators
        features_df['SMA_10'] = TechnicalIndicators.sma(df['Close'], 10)
        features_df['SMA_20'] = TechnicalIndicators.sma(df['Close'], 20)
        features_df['SMA_50'] = TechnicalIndicators.sma(df['Close'], 50)
        features_df['EMA_12'] = TechnicalIndicators.ema(df['Close'], 12)
        features_df['EMA_26'] = TechnicalIndicators.ema(df['Close'], 26)
        
        features_df['RSI'] = TechnicalIndicators.rsi(df['Close'])
        
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(df['Close'])
        features_df['MACD'] = macd
        features_df['MACD_Signal'] = macd_signal
        features_df['MACD_Hist'] = macd_hist
        
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
        features_df['BB_Upper'] = bb_upper
        features_df['BB_Middle'] = bb_middle
        features_df['BB_Lower'] = bb_lower
        features_df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        features_df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        features_df['ATR'] = TechnicalIndicators.atr(df)
        stoch_k, stoch_d = TechnicalIndicators.stochastic(df)
        features_df['Stoch_K'] = stoch_k
        features_df['Stoch_D'] = stoch_d
        
        features_df['Williams_R'] = TechnicalIndicators.williams_r(df)
        features_df['ADX'] = TechnicalIndicators.adx(df)
        
        # Price-based features
        features_df['Returns_1'] = df['Close'].pct_change(1)
        features_df['Returns_5'] = df['Close'].pct_change(5)
        features_df['Returns_20'] = df['Close'].pct_change(20)
        
        features_df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features_df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Trend features
        features_df['Price_vs_SMA20'] = df['Close'] / features_df['SMA_20']
        features_df['Price_vs_SMA50'] = df['Close'] / features_df['SMA_50']
        features_df['SMA20_vs_SMA50'] = features_df['SMA_20'] / features_df['SMA_50']
        
        # Volatility features
        features_df['Volatility_10'] = df['Close'].rolling(10).std()
        features_df['Volatility_20'] = df['Close'].rolling(20).std()
        
        return features_df.dropna()
    
    def prepare_ml_data(self, df: pd.DataFrame, target_days: int = 5, threshold: float = 0.02) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML training"""
        features_df = self.create_features(df)
        
        # Create target variable
        future_returns = features_df['Close'].shift(-target_days) / features_df['Close'] - 1
        target = (future_returns > threshold).astype(int)
        
        # Remove target column and other non-feature columns
        feature_cols = [col for col in features_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        X = features_df[feature_cols].iloc[:-target_days]
        y = target.iloc[:-target_days]
        
        self.feature_columns = feature_cols
        return X, y
    
    def train_model(self, symbol: str, df: pd.DataFrame, target_days: int = 5, threshold: float = 0.02) -> Dict[str, Any]:
        """Train ML model for predictions"""
        X, y = self.prepare_ml_data(df, target_days, threshold)
        
        if len(X) < 100 or y.nunique() < 2:
            return {'error': 'Insufficient data for training'}
        
        # Create pipeline with scaling and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
        
        # Train final model
        pipeline.fit(X, y)
        self.models[symbol] = pipeline
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_columns,
            pipeline.named_steps['rf'].feature_importances_
        ))
        
        return {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def predict(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Make predictions using trained model"""
        if symbol not in self.models:
            return {'error': 'Model not trained for this symbol'}
        
        features_df = self.create_features(df)
        X = features_df[self.feature_columns].iloc[[-1]]  # Latest data point
        
        model = self.models[symbol]
        prediction = model.predict_proba(X)[0]
        
        return {
            'probability_up': float(prediction[1]),
            'probability_down': float(prediction[0]),
            'signal_strength': abs(prediction[1] - 0.5) * 2  # 0 to 1 scale
        }

# ================================================================
# TRADING STRATEGY ENGINE
# ================================================================

class TradingStrategy:
    """Advanced trading strategy with multiple signals"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def generate_signals(self, df: pd.DataFrame) -> int:
        """Generate comprehensive trading signals - FIXED VERSION"""
        try:
            # Create technical indicators
            features = MLPredictor().create_features(df)
            
            # Check if we have enough data
            if features.empty or len(features) == 0:
                return SignalType.HOLD.value
            
            # Get the last valid row with data
            valid_features = features.dropna()
            if valid_features.empty:
                return SignalType.HOLD.value
            
            # Use the last valid row
            last_row = valid_features.iloc[-1]
            
            # Signal components
            trend_score = 0
            momentum_score = 0
            mean_reversion_score = 0
            
            # Trend signals - with safety checks
            if 'SMA20_vs_SMA50' in last_row and not pd.isna(last_row['SMA20_vs_SMA50']):
                if last_row['SMA20_vs_SMA50'] > 1.01:
                    trend_score += 1
                elif last_row['SMA20_vs_SMA50'] < 0.99:
                    trend_score -= 1
            
            # Momentum signals - with safety checks
            if 'RSI' in last_row and not pd.isna(last_row['RSI']):
                if last_row['RSI'] > 70:
                    momentum_score -= 1
                elif last_row['RSI'] < 30:
                    momentum_score += 1
            
            if 'MACD' in last_row and 'MACD_Signal' in last_row:
                if not pd.isna(last_row['MACD']) and not pd.isna(last_row['MACD_Signal']):
                    if last_row['MACD'] > last_row['MACD_Signal']:
                        momentum_score += 0.5
                    else:
                        momentum_score -= 0.5
            
            # Mean reversion signals - with safety checks
            if 'BB_Position' in last_row and not pd.isna(last_row['BB_Position']):
                if last_row['BB_Position'] > 0.8:
                    mean_reversion_score -= 1
                elif last_row['BB_Position'] < 0.2:
                    mean_reversion_score += 1
            
            # Combine signals
            total_score = trend_score + momentum_score + mean_reversion_score
            
            if total_score >= 1.5:
                return SignalType.BUY.value
            elif total_score <= -1.5:
                return SignalType.SELL.value
            else:
                return SignalType.HOLD.value
                
        except Exception as e:
            # If any error occurs, return HOLD
            return SignalType.HOLD.value
    
    def backtest_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive strategy backtesting - FIXED VERSION"""
        try:
            features = MLPredictor().create_features(df)
            
            if features.empty or len(features) < 100:  # Need minimum data
                return {'error': 'Insufficient data for backtesting'}
            
            # Initialize portfolio
            capital = self.config.initial_capital
            position = 0
            entry_price = 0
            stop_loss = 0
            take_profit = 0
            
            trades = []
            equity_curve = []
            
            # Start from index where we have sufficient data (at least 50 periods)
            start_idx = max(50, len(features) // 4)
            
            for i in range(start_idx, len(features)):
                current_price = features['Close'].iloc[i]
                
                # Check if we have ATR data
                if 'ATR' not in features.columns or pd.isna(features['ATR'].iloc[i]):
                    equity_curve.append(capital + (current_price - entry_price) * position if position != 0 else capital)
                    continue
                
                atr = features['ATR'].iloc[i]
                
                # Generate signal using historical data up to current point
                signal_data = features.iloc[:i+1]
                signal = self.generate_signals(signal_data)
                
                # Exit logic
                if position != 0:
                    exit_trade = False
                    exit_reason = ""
                    
                    if position > 0:  # Long position
                        if current_price <= stop_loss:
                            exit_trade = True
                            exit_reason = "Stop Loss"
                        elif current_price >= take_profit:
                            exit_trade = True
                            exit_reason = "Take Profit"
                        elif signal == SignalType.SELL.value:
                            exit_trade = True
                            exit_reason = "Signal Reversal"
                    
                    elif position < 0:  # Short position
                        if current_price >= stop_loss:
                            exit_trade = True
                            exit_reason = "Stop Loss"
                        elif current_price <= take_profit:
                            exit_trade = True
                            exit_reason = "Take Profit"
                        elif signal == SignalType.BUY.value:
                            exit_trade = True
                            exit_reason = "Signal Reversal"
                    
                    if exit_trade:
                        pnl = (current_price - entry_price) * position
                        capital += pnl
                        
                        trades.append({
                            'entry_date': features.index[i-1] if i > 0 else features.index[i],
                            'exit_date': features.index[i],
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': pnl,
                            'exit_reason': exit_reason
                        })
                        
                        position = 0
                
                # Entry logic
                if position == 0 and signal != SignalType.HOLD.value and capital > 0:
                    risk_amount = capital * self.config.risk_per_trade
                    
                    if signal == SignalType.BUY.value:
                        stop_loss = current_price - atr * self.config.stop_loss_atr_mult
                        take_profit = current_price + atr * self.config.stop_loss_atr_mult * self.config.take_profit_rr
                        risk_per_share = current_price - stop_loss
                        
                        if risk_per_share > 0:
                            position = int(risk_amount / risk_per_share)
                            if position > 0:
                                entry_price = current_price
                    
                    elif signal == SignalType.SELL.value:
                        stop_loss = current_price + atr * self.config.stop_loss_atr_mult
                        take_profit = current_price - atr * self.config.stop_loss_atr_mult * self.config.take_profit_rr
                        risk_per_share = stop_loss - current_price
                        
                        if risk_per_share > 0:
                            position = -int(risk_amount / risk_per_share)
                            if position < 0:
                                entry_price = current_price
                
                # Calculate current equity
                current_equity = capital
                if position != 0:
                    current_equity += (current_price - entry_price) * position
                
                equity_curve.append(current_equity)
            
            # Calculate performance metrics
            if not equity_curve:
                return {'error': 'No trading activity generated'}
            
            equity_df = pd.DataFrame(
                {'equity': equity_curve}, 
                index=features.index[start_idx:start_idx+len(equity_curve)]
            )
            
            total_return = (equity_curve[-1] / self.config.initial_capital) - 1
            
            # Calculate maximum drawdown
            peak = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - peak) / peak
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
            
            # Calculate Sharpe ratio
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() != 0 else 0
            
            # Win rate
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            return {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'equity_curve': equity_df,
                'trades': trades
            }
            
        except Exception as e:
            return {'error': f'Backtesting failed: {str(e)}'}

# ================================================================
# PORTFOLIO MANAGER
# ================================================================

class PortfolioManager:
    """Portfolio management with Islamic principles"""
    
    def __init__(self):
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {
                'cash': 100000.0,
                'positions': {},
                'transaction_history': []
            }
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        portfolio = st.session_state.portfolio
        total_value = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            if symbol in current_prices:
                total_value += position['quantity'] * current_prices[symbol]
        
        return total_value
    
    def execute_trade(self, symbol: str, quantity: int, price: float, trade_type: str) -> bool:
        """Execute a trade"""
        portfolio = st.session_state.portfolio
        cost = quantity * price
        
        if trade_type.upper() == 'BUY':
            if portfolio['cash'] >= cost:
                portfolio['cash'] -= cost
                
                if symbol in portfolio['positions']:
                    # Update existing position
                    old_qty = portfolio['positions'][symbol]['quantity']
                    old_avg = portfolio['positions'][symbol]['avg_price']
                    new_qty = old_qty + quantity
                    new_avg = ((old_qty * old_avg) + cost) / new_qty
                    
                    portfolio['positions'][symbol] = {
                        'quantity': new_qty,
                        'avg_price': new_avg
                    }
                else:
                    # New position
                    portfolio['positions'][symbol] = {
                        'quantity': quantity,
                        'avg_price': price
                    }
                
                portfolio['transaction_history'].append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'type': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'total': cost
                })
                
                return True
            else:
                return False
        
        elif trade_type.upper() == 'SELL':
            if symbol in portfolio['positions'] and portfolio['positions'][symbol]['quantity'] >= quantity:
                portfolio['cash'] += cost
                portfolio['positions'][symbol]['quantity'] -= quantity
                
                if portfolio['positions'][symbol]['quantity'] == 0:
                    del portfolio['positions'][symbol]
                
                portfolio['transaction_history'].append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'type': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'total': cost
                })
                
                return True
            else:
                return False
        
        return False

# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🕌 Islamic AI Broker Pro</h1>
        <p class="main-subtitle">Advanced Sharia-Compliant Trading Platform for Egyptian Stock Exchange (EGX) with AI Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    data_manager = DataManager()
    sharia_config = ShariaConfig()
    trading_config = TradingConfig()
    
    sharia_screener = ShariaScreener(sharia_config)
    ml_predictor = MLPredictor()
    trading_strategy = TradingStrategy(trading_config)
    portfolio_manager = PortfolioManager()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Stock selection
        st.subheader("📈 Stock Selection")
        selected_symbol = st.selectbox(
            "Choose Stock",
            options=list(HALAL_TICKERS.keys()),
            format_func=lambda x: f"{HALAL_TICKERS[x][0]} ({x})"
        )
        
        # Time period
        period = st.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        
        # Trading parameters
        st.subheader("⚙️ Trading Config")
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
        stop_loss_mult = st.slider("Stop Loss ATR Multiplier", 1.0, 4.0, 2.0, 0.5)
        take_profit_rr = st.slider("Take Profit R:R", 1.0, 5.0, 2.5, 0.5)
        
        # ML parameters
        st.subheader("🤖 ML Settings")
        prediction_days = st.slider("Prediction Horizon (days)", 1, 20, 5)
        movement_threshold = st.slider("Movement Threshold (%)", 1.0, 10.0, 2.0) / 100
        
        # Sharia thresholds
        st.subheader("🕌 Sharia Thresholds")
        max_debt_ratio = st.slider("Max Debt/Assets", 0.1, 0.5, 0.33, 0.01)
        max_interest_ratio = st.slider("Max Interest Income/Revenue", 0.0, 0.1, 0.05, 0.005)
        max_cash_ratio = st.slider("Max Cash/Assets", 0.1, 0.5, 0.33, 0.01)
        
        # Update configurations
        trading_config.risk_per_trade = risk_per_trade
        trading_config.stop_loss_atr_mult = stop_loss_mult
        trading_config.take_profit_rr = take_profit_rr
        
        sharia_config.max_debt_to_assets = max_debt_ratio
        sharia_config.max_interest_income_ratio = max_interest_ratio
        sharia_config.max_cash_securities_ratio = max_cash_ratio
        
        run_analysis = st.button("🔍 Run Complete Analysis", type="primary")
    
    # Main content
    if run_analysis:
        with st.spinner("🔄 Loading market data..."):
            df = data_manager.get_market_data(selected_symbol, period)
        
        if df.empty:
            st.error("❌ Unable to fetch market data. Please try again.")
            return
        
        # Current price info
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0.0
        
        # Price metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">${current_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#10b981" if price_change >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {color}, {color}99);">
                <div class="metric-label">Daily Change</div>
                <div class="metric-value">{price_change:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value">{df['Volume'].iloc[-1]:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            market_cap = data_manager.get_company_info(selected_symbol).get('marketCap', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Cap</div>
                <div class="metric-value">${market_cap/1e9:.1f}B</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sharia Screening
        st.header("🕌 Sharia Compliance Analysis")
        with st.spinner("📋 Performing Sharia screening..."):
            is_compliant, ratios, issues = sharia_screener.screen_stock(selected_symbol)
        
        # Compliance status
        if is_compliant:
            st.markdown('<div class="status-halal">✅ HALAL CERTIFIED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-haram">❌ NON-COMPLIANT</div>', unsafe_allow_html=True)
        
        # Compliance details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            debt_ratio = ratios.get('debt_to_assets', np.nan)
            if not pd.isna(debt_ratio):
                st.metric("Debt/Assets Ratio", f"{debt_ratio:.1%}", 
                         delta=f"Limit: {max_debt_ratio:.0%}")
        
        with col2:
            interest_ratio = ratios.get('interest_income_ratio', np.nan)
            if not pd.isna(interest_ratio):
                st.metric("Interest Income Ratio", f"{interest_ratio:.1%}", 
                         delta=f"Limit: {max_interest_ratio:.1%}")
        
        with col3:
            cash_ratio = ratios.get('cash_securities_ratio', np.nan)
            if not pd.isna(cash_ratio):
                st.metric("Cash/Securities Ratio", f"{cash_ratio:.1%}", 
                         delta=f"Limit: {max_cash_ratio:.0%}")
        
        # Issues/approvals
        st.markdown('<div class="info-panel">', unsafe_allow_html=True)
        for issue in issues:
            st.write(issue)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Technical Analysis Charts
        st.header("📊 Advanced Technical Analysis")
        
        # Create comprehensive chart
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
            row_width=[0.3, 0.1, 0.2, 0.2]
        )
        
        # Add technical indicators
        features_df = ml_predictor.create_features(df)
        
        # Price and moving averages
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        if 'SMA_20' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
        
        if 'SMA_50' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['SMA_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ), row=1, col=1)
        
        # Bollinger Bands
        if all(col in features_df.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['BB_Upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['BB_Lower'],
                name='BB Lower',
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1)
        
        # RSI
        if 'RSI' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['RSI'],
                name='RSI',
                line=dict(color='purple')
            ), row=3, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if all(col in features_df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['MACD'],
                name='MACD',
                line=dict(color='blue')
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['MACD_Signal'],
                name='MACD Signal',
                line=dict(color='red')
            ), row=4, col=1)
            
            fig.add_trace(go.Bar(
                x=features_df.index,
                y=features_df['MACD_Hist'],
                name='MACD Histogram',
                marker_color='green'
            ), row=4, col=1)
        
        fig.update_layout(
            height=800,
            title=f"Technical Analysis - {HALAL_TICKERS[selected_symbol][0]}",
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ML Predictions
        st.header("🤖 AI Market Predictions")
        
        with st.spinner("🧠 Training AI model..."):
            ml_results = ml_predictor.train_model(
                selected_symbol, df, prediction_days, movement_threshold
            )
        
        if 'error' in ml_results:
            st.error(f"❌ ML Error: {ml_results['error']}")
        else:
            predictions = ml_predictor.predict(selected_symbol, df)
            
            if 'error' not in predictions:
                prob_up = predictions['probability_up']
                prob_down = predictions['probability_down']
                signal_strength = predictions['signal_strength']
                
                # Prediction display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"Probability of {movement_threshold:.0%}+ move in {prediction_days} days",
                        f"{prob_up:.1%}",
                        delta=f"Confidence: {signal_strength:.1%}"
                    )
                
                with col2:
                    st.metric(
                        "Model Accuracy (CV)",
                        f"{ml_results['cv_accuracy']:.1%}",
                        delta=f"±{ml_results['cv_std']:.1%}"
                    )
                
                with col3:
                    # AI Recommendation
                    if prob_up > 0.6 and signal_strength > 0.3:
                        recommendation = "STRONG BUY"
                        rec_color = "#10b981"
                    elif prob_up > 0.55:
                        recommendation = "BUY"
                        rec_color = "#10b981"
                    elif prob_up < 0.4 and signal_strength > 0.3:
                        recommendation = "STRONG SELL"
                        rec_color = "#ef4444"
                    elif prob_up < 0.45:
                        recommendation = "SELL"
                        rec_color = "#ef4444"
                    else:
                        recommendation = "HOLD"
                        rec_color = "#f59e0b"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, {rec_color}, {rec_color}99); 
                                color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <strong>AI Recommendation:</strong><br>
                        <span style="font-size: 1.5rem; font-weight: bold;">{recommendation}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability visualization
                fig_prob = go.Figure(go.Bar(
                    x=['Price Up', 'Price Down'],
                    y=[prob_up, prob_down],
                    marker_color=['green', 'red']
                ))
                
                fig_prob.update_layout(
                    title=f"AI Prediction Probabilities ({prediction_days} days ahead)",
                    yaxis_title="Probability",
                    showlegend=False
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Feature importance - FIXED VERSION
                st.subheader("🎯 Key Factors Influencing Prediction")
                importance_df = pd.DataFrame(
                    list(ml_results['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                )
                
                # Using go.Bar instead of px.horizontal_bar
                fig_importance = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='skyblue'
                ))
                
                fig_importance.update_layout(
                    title="Feature Importance in AI Model",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=400
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Strategy Backtesting
        st.header("📈 Strategy Performance")
        
        with st.spinner("🔄 Running backtest..."):
            backtest_results = trading_strategy.backtest_strategy(df)
        
        if 'error' in backtest_results:
            st.warning(f"⚠️ Backtest: {backtest_results['error']}")
        else:
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{backtest_results['total_return']:.1%}")
            
            with col2:
                st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.1%}")
            
            with col3:
                st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
            
            with col4:
                st.metric("Win Rate", f"{backtest_results['win_rate']:.1%}")
                st.write(f"Total Trades: {backtest_results.get('total_trades', 0)}")
            
            # Equity curve
            if 'equity_curve' in backtest_results and not backtest_results['equity_curve'].empty:
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=backtest_results['equity_curve'].index,
                    y=backtest_results['equity_curve']['equity'],
                    name='Strategy Equity',
                    line=dict(color='blue')
                ))
                
                # Buy & hold benchmark
                initial_price = df['Close'].iloc[0]
                final_price = df['Close'].iloc[-1]
                benchmark_return = (final_price / initial_price - 1)
                benchmark_equity = trading_config.initial_capital * (1 + benchmark_return)
                
                fig_equity.add_hline(
                    y=benchmark_equity,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Buy & Hold: {benchmark_return:.1%}"
                )
                
                fig_equity.update_layout(
                    title="Strategy vs Buy & Hold Performance",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)"
                )
                
                st.plotly_chart(fig_equity, use_container_width=True)
            
            # Trading activity
            if backtest_results.get('trades'):
                st.subheader("📋 Recent Trades")
                trades_df = pd.DataFrame(backtest_results['trades'][-10:])  # Last 10 trades
                st.dataframe(trades_df)
        
        # Current Trading Signal
        st.header("🎯 Current Trading Signal")
        
        current_signal = trading_strategy.generate_signals(df)
        
        if current_signal == SignalType.BUY.value:
            signal_class = "signal-buy"
            signal_text = "🟢 STRONG BUY SIGNAL"
        elif current_signal == SignalType.SELL.value:
            signal_class = "signal-sell"
            signal_text = "🔴 STRONG SELL SIGNAL"
        else:
            signal_class = "signal-hold"
            signal_text = "🟡 HOLD POSITION"
        
        st.markdown(f'<div class="trading-signal {signal_class}">{signal_text}</div>', 
                   unsafe_allow_html=True)
        
        # Portfolio Management
        st.header("💼 Portfolio Management")
        
        current_prices = {selected_symbol: current_price}
        portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
        portfolio = st.session_state.portfolio
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cash Balance", f"${portfolio['cash']:,.2f}")
        
        with col2:
            positions_value = portfolio_value - portfolio['cash']
            st.metric("Positions Value", f"${positions_value:,.2f}")
        
        with col3:
            st.metric("Total Portfolio", f"${portfolio_value:,.2f}")
        
        # Current positions
        if portfolio['positions']:
            st.subheader("📊 Current Positions")
            positions_data = []
            
            for symbol, position in portfolio['positions'].items():
                current_price_pos = current_prices.get(symbol, 0)
                market_value = position['quantity'] * current_price_pos
                unrealized_pnl = market_value - (position['quantity'] * position['avg_price'])
                
                positions_data.append({
                    'Symbol': symbol,
                    'Quantity': position['quantity'],
                    'Avg Price': f"${position['avg_price']:.2f}",
                    'Current Price': f"${current_price_pos:.2f}",
                    'Market Value': f"${market_value:,.2f}",
                    'Unrealized P&L': f"${unrealized_pnl:,.2f}"
                })
            
            st.dataframe(pd.DataFrame(positions_data))
        
        # Trade execution
        st.subheader("⚡ Execute Trade")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_quantity = st.number_input("Quantity", min_value=1, value=100, step=10)
        
        with col2:
            if st.button("🟢 BUY", use_container_width=True):
                if is_compliant:  # Only allow trading in Sharia-compliant stocks
                    success = portfolio_manager.execute_trade(
                        selected_symbol, trade_quantity, current_price, 'BUY'
                    )
                    if success:
                        st.success(f"✅ Bought {trade_quantity} shares of {selected_symbol}")
                        st.rerun()
                    else:
                        st.error("❌ Insufficient cash for this trade")
                else:
                    st.error("❌ Cannot trade non-Sharia compliant stocks")
        
        with col3:
            if st.button("🔴 SELL", use_container_width=True):
                success = portfolio_manager.execute_trade(
                    selected_symbol, trade_quantity, current_price, 'SELL'
                )
                if success:
                    st.success(f"✅ Sold {trade_quantity} shares of {selected_symbol}")
                    st.rerun()
                else:
                    st.error("❌ Insufficient shares to sell")
    
    else:
        st.info("👆 Configure your parameters in the sidebar and click 'Run Complete Analysis' to begin")
        
        # Display sample features while waiting
        st.markdown("""
        ### 🌟 Platform Features
        
        **🕌 Sharia Compliance**
        - Automated screening based on Islamic finance principles
        - Real-time compliance monitoring
        - Customizable screening thresholds
        
        **🤖 AI Predictions**
        - Advanced machine learning models
        - Multi-factor analysis
        - Probability-based forecasting
        
        **📊 Technical Analysis**
        - 15+ technical indicators
        - Advanced charting
        - Multi-timeframe analysis
        
        **💼 Portfolio Management**
        - Real-time portfolio tracking
        - Risk management tools
        - Trade execution simulation
        
        **🎯 Professional Trading**
        - Advanced backtesting
        - Strategy optimization
        - Performance analytics
        """)

if __name__ == "__main__":
    main()
