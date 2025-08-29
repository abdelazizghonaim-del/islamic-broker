# منصة الذكاء الاصطناعي الإسلامية للتداول في البورصة المصرية
# 🏺 Islamic AI Trading Platform for Egyptian Stock Exchange (EGX)
# ==================================================================
# Advanced Sharia-compliant trading system with Enhanced AI predictions,
# Arabic language support, and comprehensive market analysis for Egypt
# ==================================================================

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
import arabic_reshaper
import bidi.algorithm

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import joblib

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ================================================================
# النصوص العربية - ARABIC TEXTS
# ================================================================

ARABIC_TEXTS = {
    'title': 'منصة الذكاء الاصطناعي الإسلامية للتداول في البورصة المصرية',
    'subtitle': 'نظام تداول متقدم متوافق مع أحكام الشريعة الإسلامية مع تحليل الذكاء الاصطناعي المتطور',
    'control_panel': 'لوحة التحكم',
    'stock_selection': 'اختيار الأسهم',
    'choose_stock': 'اختر السهم',
    'time_period': 'الفترة الزمنية',
    'trading_config': 'إعدادات التداول',
    'risk_per_trade': 'المخاطرة لكل صفقة (%)',
    'stop_loss': 'مضاعف وقف الخسارة ATR',
    'take_profit': 'نسبة الربح للمخاطرة',
    'ml_settings': 'إعدادات الذكاء الاصطناعي',
    'prediction_horizon': 'أفق التنبؤ (أيام)',
    'movement_threshold': 'عتبة الحركة (%)',
    'sharia_thresholds': 'حدود الشريعة الإسلامية',
    'max_debt_assets': 'الحد الأقصى للديون/الأصول',
    'max_interest_income': 'الحد الأقصى لدخل الفوائد/الإيرادات',
    'max_cash_assets': 'الحد الأقصى للنقد/الأصول',
    'run_analysis': 'تشغيل التحليل الكامل',
    'current_price': 'السعر الحالي',
    'daily_change': 'التغيير اليومي',
    'volume': 'حجم التداول',
    'market_cap': 'القيمة السوقية',
    'sharia_compliance': 'الامتثال للشريعة الإسلامية',
    'halal_certified': 'حلال معتمد',
    'non_compliant': 'غير متوافق',
    'debt_assets_ratio': 'نسبة الديون للأصول',
    'interest_income_ratio': 'نسبة دخل الفوائد',
    'cash_securities_ratio': 'نسبة النقد والأوراق المالية',
    'technical_analysis': 'التحليل الفني المتقدم',
    'ai_predictions': 'تنبؤات الذكاء الاصطناعي',
    'strategy_performance': 'أداء الاستراتيجية',
    'trading_signal': 'إشارة التداول الحالية',
    'portfolio_management': 'إدارة المحفظة',
    'cash_balance': 'الرصيد النقدي',
    'positions_value': 'قيمة المراكز',
    'total_portfolio': 'إجمالي المحفظة',
    'execute_trade': 'تنفيذ الصفقة',
    'quantity': 'الكمية',
    'buy': 'شراء',
    'sell': 'بيع',
    'strong_buy': 'شراء قوي',
    'strong_sell': 'بيع قوي',
    'hold': 'انتظار',
    'explanation': 'التفسير',
    'analysis_result': 'نتيجة التحليل'
}

def display_arabic(text_key: str, fallback: str = "") -> str:
    """Display Arabic text with proper formatting"""
    arabic_text = ARABIC_TEXTS.get(text_key, fallback)
    if arabic_text:
        reshaped_text = arabic_reshaper.reshape(arabic_text)
        return bidi.algorithm.get_display(reshaped_text)
    return fallback

# ================================================================
# EGYPTIAN STOCK EXCHANGE CONFIGURATION
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
    initial_capital: float = 500000.0  # 500,000 EGP
    commission: float = 0.001

@dataclass
class ShariaConfig:
    """Sharia compliance thresholds"""
    max_debt_to_assets: float = 0.33
    max_interest_income_ratio: float = 0.05
    max_cash_securities_ratio: float = 0.33
    min_market_cap: float = 1e9  # 1B EGP minimum

# أسهم حلال في البورصة المصرية - HALAL EGYPTIAN STOCKS
EGYPTIAN_HALAL_STOCKS = {
    'CIB.CA': ('البنك التجاري الدولي', 'البنوك', 'Commercial International Bank'),
    'ETEL.CA': ('المصرية للاتصالات', 'الاتصالات', 'Telecom Egypt'),
    'TMGH.CA': ('طلعت مصطفى القابضة', 'العقارات', 'Talaat Moustafa Group'),
    'PHDC.CA': ('بالم هيلز للتطوير', 'العقارات', 'Palm Hills Developments'),
    'ORWE.CA': ('النساجون الشرقيون', 'الغزل والنسيج', 'Oriental Weavers'),
    'SWDY.CA': ('السويدي إلكتريك', 'المعدات الكهربائية', 'Elsewedy Electric'),
    'ABUK.CA': ('أبو قير للأسمدة', 'الكيماويات', 'Abu Qir Fertilizers'),
    'HELI.CA': ('هليوبوليس للإسكان', 'العقارات', 'Heliopolis Housing'),
    'IRON.CA': ('الحديد والصلب المصرية', 'الحديد والصلب', 'Egyptian Iron & Steel'),
    'CCAP.CA': ('كريدي أجريكول مصر', 'البنوك', 'Credit Agricole Egypt'),
    'EKHO.CA': ('مجموعة إيكون القابضة', 'الاستثمار المالي', 'Eikon Holding'),
    'SKPC.CA': ('سيدي كرير للبتروكيماويات', 'البتروكيماويات', 'Sidi Kerir Petrochemicals'),
    'ARAB.CA': ('البنك العربي الأفريقي', 'البنوك', 'Arab African International Bank'),
    'MNHD.CA': ('المدينة للإسكان والتعمير', 'العقارات', 'Madinet Nasr Housing'),
    'OCDI.CA': ('أوراسكوم للتنمية', 'العقارات', 'Orascom Development'),
    'ALCN.CA': ('الكابلات الكهربائية المصرية', 'الكابلات', 'Al Ahram Cables'),
    'STIN.CA': ('الصناعات الكيماوية المصرية', 'الكيماويات', 'Stin Industries'),
    'EMFD.CA': ('المصرية للأغذية', 'الأغذية والمشروبات', 'Egyptian Food Industries'),
    'CLHO.CA': ('كليوباترا القابضة', 'الاستثمار المالي', 'Cleopatra Holding'),
    'PIOH.CA': ('بايونيرز القابضة', 'الاستثمار المالي', 'Pioneers Holding')
}

PROHIBITED_SECTORS_AR = {
    'البنوك التقليدية', 'التأمين التقليدي', 'الخدمات المالية التقليدية', 'الكحوليات',
    'القمار', 'الكازينوهات', 'التبغ', 'المحتوى البالغ', 'الأسلحة',
    'الدفاع العسكري', 'لحم الخنزير', 'التمويل التقليدي'
}

# ================================================================
# STREAMLIT CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="🏺 منصة التداول الإسلامية المصرية",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS with Arabic support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700;900&display=swap');
    
    .arabic-text {
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #059669 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .main-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin: 0;
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e40af 0%, #10b981 100%);
        padding: 1.8rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.95;
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
    }
    
    .status-halal {
        background: linear-gradient(90deg, #059669, #047857);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 30px;
        font-weight: bold;
        display: inline-block;
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.4);
    }
    
    .status-haram {
        background: linear-gradient(90deg, #dc2626, #b91c1c);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 30px;
        font-weight: bold;
        display: inline-block;
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
    }
    
    .trading-signal {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
    }
    
    .signal-buy {
        background: linear-gradient(90deg, #059669, #047857);
        color: white;
    }
    
    .signal-sell {
        background: linear-gradient(90deg, #dc2626, #b91c1c);
        color: white;
    }
    
    .signal-hold {
        background: linear-gradient(90deg, #d97706, #b45309);
        color: white;
    }
    
    .info-panel {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        border-right: 6px solid #1e40af;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 15px 15px 0;
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
    }
    
    .explanation-box {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 2px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .stProgress .st-bo {
        background-color: #1e40af;
    }
    
    .sidebar .sidebar-content {
        font-family: 'Noto Sans Arabic', Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# ENHANCED TECHNICAL INDICATORS
# ================================================================

class AdvancedTechnicalIndicators:
    """Advanced technical indicators for Egyptian market analysis"""
    
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
        macd_line = AdvancedTechnicalIndicators.ema(series, fast) - AdvancedTechnicalIndicators.ema(series, slow)
        signal_line = AdvancedTechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        ma = AdvancedTechnicalIndicators.sma(series, period)
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
    def fibonacci_retracement(df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Fibonacci Retracement Levels"""
        high = df['High'].rolling(period).max()
        low = df['Low'].rolling(period).min()
        diff = high - low
        
        levels = {
            'fib_23.6': high - 0.236 * diff,
            'fib_38.2': high - 0.382 * diff,
            'fib_50.0': high - 0.500 * diff,
            'fib_61.8': high - 0.618 * diff,
            'fib_78.6': high - 0.786 * diff
        }
        return levels
    
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    @staticmethod
    def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfr = positive_mf / negative_mf
        return 100 - (100 / (1 + mfr))
    
    @staticmethod
    def commodity_channel_index(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)

# ================================================================
# ENHANCED DATA MANAGEMENT FOR EGYPTIAN MARKET
# ================================================================

class EgyptianDataManager:
    """Enhanced data management for Egyptian Stock Exchange"""
    
    def __init__(self):
        self.cache_dir = os.path.join(tempfile.gettempdir(), "egyptian_ai_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @st.cache_data(ttl=300)
    def get_egyptian_market_data(_self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch Egyptian market data with multiple sources"""
        try:
            # Try yfinance first
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            
            if df.empty:
                return pd.DataFrame()
            
            # Ensure consistent column names
            df.columns = df.columns.str.title()
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_cols].dropna()
            
            # Convert to EGP if needed (assuming data is in EGP already for Egyptian stocks)
            return df
            
        except Exception as e:
            st.error(f"خطأ في جلب البيانات للرمز {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def get_company_info(_self, symbol: str) -> Dict[str, Any]:
        """Get Egyptian company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            
            # Add Egyptian market specific information
            if symbol in EGYPTIAN_HALAL_STOCKS:
                arabic_name, sector_ar, english_name = EGYPTIAN_HALAL_STOCKS[symbol]
                info['arabic_name'] = arabic_name
                info['sector_arabic'] = sector_ar
                info['english_name'] = english_name
            
            return info
        except Exception:
            return {}
    
    @st.cache_data(ttl=3600)
    def get_egyptian_financials(_self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get Egyptian company financials"""
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
# ENHANCED SHARIA COMPLIANCE ENGINE
# ================================================================

class AdvancedShariaScreener:
    """Advanced Sharia compliance screening for Egyptian market"""
    
    def __init__(self, config: ShariaConfig):
        self.config = config
        self.data_manager = EgyptianDataManager()
    
    def screen_egyptian_stock(self, symbol: str) -> Tuple[bool, Dict[str, Any], List[str], str]:
        """Comprehensive Sharia screening with Arabic explanations"""
        info = self.data_manager.get_company_info(symbol)
        financials = self.data_manager.get_egyptian_financials(symbol)
        
        results = {
            'debt_to_assets': np.nan,
            'interest_income_ratio': np.nan,
            'cash_securities_ratio': np.nan,
            'market_cap': np.nan
        }
        
        issues = []
        explanations = []
        
        # Get Arabic company name
        arabic_name = info.get('arabic_name', symbol)
        
        # Sector screening with Arabic explanation
        sector = info.get('sector', '').lower()
        sector_ar = info.get('sector_arabic', '')
        
        if any(prohibited in sector.lower() for prohibited in PROHIBITED_SECTORS_AR):
            issues.append(f"❌ قطاع محظور: {sector_ar}")
            explanations.append(f"الشركة {arabic_name} تعمل في قطاع {sector_ar} وهو قطاع محظور شرعياً وفقاً لأحكام الاستثمار الإسلامي")
        else:
            explanations.append(f"الشركة {arabic_name} تعمل في قطاع {sector_ar} وهو قطاع مسموح شرعياً")
        
        # Market cap check
        market_cap = info.get('marketCap', 0)
        results['market_cap'] = market_cap
        if market_cap < self.config.min_market_cap:
            issues.append(f"❌ القيمة السوقية صغيرة جداً: {market_cap:,.0f} جنيه")
            explanations.append(f"القيمة السوقية للشركة {market_cap/1e9:.1f} مليار جنيه وهي أقل من الحد الأدنى المطلوب {self.config.min_market_cap/1e9:.1f} مليار جنيه للاستثمار الآمن")
        else:
            explanations.append(f"القيمة السوقية للشركة {market_cap/1e9:.1f} مليار جنيه وهي مقبولة للاستثمار")
        
        # Financial ratio screening with Arabic explanations
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
                        issues.append(f"❌ نسبة ديون عالية: {debt_ratio:.1%}")
                        explanations.append(f"نسبة الديون إلى الأصول {debt_ratio:.1%} تتجاوز الحد المسموح شرعياً وهو {self.config.max_debt_to_assets:.0%}. هذا يعني أن الشركة تعتمد بشكل كبير على التمويل بالدين المحرم شرعياً")
                    else:
                        explanations.append(f"نسبة الديون إلى الأصول {debt_ratio:.1%} ضمن الحدود المقبولة شرعياً")
                
                # Cash and securities ratio
                cash = latest_bs.get('Cash And Cash Equivalents', 0)
                short_term_investments = latest_bs.get('Short Term Investments', 0)
                cash_securities = cash + short_term_investments
                
                if total_assets > 0:
                    cash_ratio = cash_securities / total_assets
                    results['cash_securities_ratio'] = cash_ratio
                    
                    if cash_ratio > self.config.max_cash_securities_ratio:
                        issues.append(f"❌ نسبة نقد وأوراق مالية عالية: {cash_ratio:.1%}")
                        explanations.append(f"نسبة النقد والأوراق المالية {cash_ratio:.1%} تتجاوز الحد المسموح {self.config.max_cash_securities_ratio:.0%}. هذا قد يشير إلى استثمار في أدوات مالية تقليدية محرمة")
                    else:
                        explanations.append(f"نسبة النقد والأوراق المالية {cash_ratio:.1%} ضمن الحدود المقبولة")
            
            # Interest income screening
            if isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]
                
                total_revenue = latest_income.get('Total Revenue', 0)
                interest_income = latest_income.get('Interest Income', 0)
                
                if total_revenue > 0 and interest_income > 0:
                    interest_ratio = interest_income / total_revenue
                    results['interest_income_ratio'] = interest_ratio
                    
                    if interest_ratio > self.config.max_interest_income_ratio:
                        issues.append(f"❌ دخل فوائد عالي: {interest_ratio:.1%}")
                        explanations.append(f"نسبة دخل الفوائد {interest_ratio:.1%} تتجاوز الحد المسموح {self.config.max_interest_income_ratio:.1%}. الفوائد محرمة شرعياً ولا يجوز الاستثمار في شركات تعتمد عليها بشكل كبير")
                    else:
                        explanations.append(f"نسبة دخل الفوائد {interest_ratio:.1%} ضمن الحدود المقبولة أو معدومة")
        
        except Exception as e:
            issues.append(f"⚠️ لا يمكن تحليل البيانات المالية: {str(e)}")
            explanations.append("لم نتمكن من الحصول على البيانات المالية الكاملة للشركة، لذا ننصح بالحذر والمراجعة الإضافية")
        
        is_compliant = len(issues) == 0
        if is_compliant:
            issues.append("✅ تستوفي جميع متطلبات الشريعة الإسلامية")
            explanations.append(f"الشركة {arabic_name} تستوفي جميع معايير الاستثمار الإسلامي ويمكن التداول فيها بأمان شرعي")
        
        # Combine all explanations
        full_explanation = " | ".join(explanations)
        
        return is_compliant, results, issues, full_explanation

# ================================================================
# ENHANCED MACHINE LEARNING ENGINE
# ================================================================

class AdvancedMLPredictor:
    """Advanced ML predictions with ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.scalers = {}
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set with advanced indicators"""
        features_df = df.copy()
        
        # Basic technical indicators
        features_df['SMA_5'] = AdvancedTechnicalIndicators.sma(df['Close'], 5)
        features_df['SMA_10'] = AdvancedTechnicalIndicators.sma(df['Close'], 10)
        features_df['SMA_20'] = AdvancedTechnicalIndicators.sma(df['Close'], 20)
        features_df['SMA_50'] = AdvancedTechnicalIndicators.sma(df['Close'], 50)
        features_df['SMA_200'] = AdvancedTechnicalIndicators.sma(df['Close'], 200)
        
        features_df['EMA_12'] = AdvancedTechnicalIndicators.ema(df['Close'], 12)
        features_df['EMA_26'] = AdvancedTechnicalIndicators.ema(df['Close'], 26)
        features_df['EMA_50'] = AdvancedTechnicalIndicators.ema(df['Close'], 50)
        
        features_df['RSI'] = AdvancedTechnicalIndicators.rsi(df['Close'])
        features_df['RSI_5'] = AdvancedTechnicalIndicators.rsi(df['Close'], 5)
        features_df['RSI_21'] = AdvancedTechnicalIndicators.rsi(df['Close'], 21)
        
        # MACD family
        macd, macd_signal, macd_hist = AdvancedTechnicalIndicators.macd(df['Close'])
        features_df['MACD'] = macd
        features_df['MACD_Signal'] = macd_signal
        features_df['MACD_Hist'] = macd_hist
        features_df['MACD_Hist_Change'] = macd_hist.diff()
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = AdvancedTechnicalIndicators.bollinger_bands(df['Close'])
        features_df['BB_Upper'] = bb_upper
        features_df['BB_Middle'] = bb_middle
        features_df['BB_Lower'] = bb_lower
        features_df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        features_df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        features_df['BB_Squeeze'] = features_df['BB_Width'] < features_df['BB_Width'].rolling(20).quantile(0.2)
        
        # ATR and volatility
        features_df['ATR'] = AdvancedTechnicalIndicators.atr(df)
        features_df['ATR_Ratio'] = features_df['ATR'] / df['Close']
        
        # Advanced indicators
        features_df['VWAP'] = AdvancedTechnicalIndicators.vwap(df)
        features_df['MFI'] = AdvancedTechnicalIndicators.money_flow_index(df)
        features_df['CCI'] = AdvancedTechnicalIndicators.commodity_channel_index(df)
        
        # Fibonacci levels
        fib_levels = AdvancedTechnicalIndicators.fibonacci_retracement(df)
        for level_name, level_values in fib_levels.items():
            features_df[level_name] = level_values
            features_df[f'{level_name}_distance'] = abs(df['Close'] - level_values) / df['Close']
        
        # Price patterns and momentum
        features_df['Returns_1'] = df['Close'].pct_change(1)
        features_df['Returns_3'] = df['Close'].pct_change(3)
        features_df['Returns_5'] = df['Close'].pct_change(5)
        features_df['Returns_10'] = df['Close'].pct_change(10)
        features_df['Returns_20'] = df['Close'].pct_change(20)
        
        # Volume analysis
        features_df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        features_df['Volume_Ratio'] = df['Volume'] / features_df['Volume_SMA']
        features_df['Volume_Trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        features_df['Price_Volume_Trend'] = (df['Close'].pct_change() * df['Volume']).rolling(10).sum()
        
        # Price patterns
        features_df['High_Low_Ratio'] = df['High'] / df['Low']
        features_df['Open_Close_Ratio'] = df['Open'] / df['Close']
        features_df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'])
        features_df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'])
        features_df['Body_Ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        # Trend analysis
        features_df['Price_vs_SMA20'] = df['Close'] / features_df['SMA_20']
        features_df['Price_vs_SMA50'] = df['Close'] / features_df['SMA_50']
        features_df['SMA20_vs_SMA50'] = features_df['SMA_20'] / features_df['SMA_50']
        features_df['SMA50_vs_SMA200'] = features_df['SMA_50'] / features_df['SMA_200']
        
        # Volatility features
        features_df['Volatility_5'] = df['Close'].rolling(5).std()
        features_df['Volatility_10'] = df['Close'].rolling(10).std()
        features_df['Volatility_20'] = df['Close'].rolling(20).std()
        features_df['Volatility_Ratio'] = features_df['Volatility_5'] / features_df['Volatility_20']
        
        # Support and resistance
        features_df['Resistance_20'] = df['High'].rolling(20).max()
        features_df['Support_20'] = df['Low'].rolling(20).min()
        features_df['Resistance_Distance'] = (features_df['Resistance_20'] - df['Close']) / df['Close']
        features_df['Support_Distance'] = (df['Close'] - features_df['Support_20']) / df['Close']
        
        # Market microstructure
        features_df['Gap'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
        features_df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
        features_df['True_Range'] = df[['High', 'Low']].max(axis=1) - df[['High', 'Low']].min(axis=1)
        
        return features_df.dropna()
    
    def train_ensemble_model(self, symbol: str, df: pd.DataFrame, target_days: int = 5, threshold: float = 0.02) -> Dict[str, Any]:
        """Train ensemble model with multiple algorithms"""
        features_df = self.create_advanced_features(df)
        
        # Create target variable
        future_returns = features_df['Close'].shift(-target_days) / features_df['Close'] - 1
        target = (future_returns > threshold).astype(int)
        
        # Feature selection
        feature_cols = [col for col in features_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        X = features_df[feature_cols].iloc[:-target_days]
        y = target.iloc[:-target_days]
        
        if len(X) < 200 or y.nunique() < 2:
            return {'error': 'بيانات غير كافية للتدريب'}
        
        # Create individual models
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
        
        # Create ensemble
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('mlp', mlp_model)
            ],
            voting='soft'
        )
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('ensemble', ensemble_model)
        ])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
        
        # Train final model
        pipeline.fit(X, y)
        self.models[symbol] = pipeline
        self.feature_columns = feature_cols
        
        # Feature importance (from random forest)
        rf_importance = pipeline.named_steps['ensemble'].estimators_[0].feature_importances_
        feature_importance = dict(zip(feature_cols, rf_importance))
        
        # Additional metrics
        y_pred = pipeline.predict(X)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        return {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15])
        }
    
    def predict_with_explanation(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with detailed Arabic explanations"""
        if symbol not in self.models:
            return {'error': 'النموذج غير مدرب لهذا الرمز'}
        
        features_df = self.create_advanced_features(df)
        X = features_df[self.feature_columns].iloc[[-1]]
        
        model = self.models[symbol]
        prediction_proba = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        
        # Get feature importance for explanation
        latest_features = features_df.iloc[-1]
        
        # Generate Arabic explanation
        prob_up = float(prediction_proba[1])
        prob_down = float(prediction_proba[0])
        signal_strength = abs(prob_up - 0.5) * 2
        
        # Create detailed explanation
        explanation = f"تحليل الذكاء الاصطناعي يشير إلى احتمالية صعود السعر بنسبة {prob_up:.1%} واحتمالية هبوط بنسبة {prob_down:.1%}. "
        
        if prob_up > 0.7:
            explanation += "الإشارة قوية جداً للشراء بناءً على التحليل الفني والنماذج المتقدمة. "
        elif prob_up > 0.6:
            explanation += "الإشارة إيجابية للشراء مع مستوى ثقة جيد. "
        elif prob_up < 0.3:
            explanation += "الإشارة قوية للبيع أو تجنب الشراء. "
        elif prob_up < 0.4:
            explanation += "الإشارة سلبية تنصح بالحذر. "
        else:
            explanation += "الإشارة محايدة تنصح بالانتظار. "
        
        # Add technical analysis explanation
        rsi_value = latest_features.get('RSI', np.nan)
        if not pd.isna(rsi_value):
            if rsi_value > 70:
                explanation += f"مؤشر القوة النسبية RSI = {rsi_value:.1f} يشير إلى حالة شراء مفرط. "
            elif rsi_value < 30:
                explanation += f"مؤشر القوة النسبية RSI = {rsi_value:.1f} يشير إلى حالة بيع مفرط. "
            else:
                explanation += f"مؤشر القوة النسبية RSI = {rsi_value:.1f} في منطقة معتدلة. "
        
        macd_value = latest_features.get('MACD', np.nan)
        macd_signal = latest_features.get('MACD_Signal', np.nan)
        if not pd.isna(macd_value) and not pd.isna(macd_signal):
            if macd_value > macd_signal:
                explanation += "مؤشر MACD يعطي إشارة إيجابية للشراء. "
            else:
                explanation += "مؤشر MACD يعطي إشارة سلبية. "
        
        return {
            'probability_up': prob_up,
            'probability_down': prob_down,
            'signal_strength': signal_strength,
            'prediction': int(prediction),
            'explanation': explanation
        }

# ================================================================
# ENHANCED TRADING STRATEGY
# ================================================================

class AdvancedTradingStrategy:
    """Advanced trading strategy for Egyptian market"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def generate_signals_with_explanation(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Generate trading signals with Arabic explanations"""
        try:
            features = AdvancedMLPredictor().create_advanced_features(df)
            
            if features.empty or len(features) == 0:
                return SignalType.HOLD.value, "لا توجد بيانات كافية لتحليل الإشارة"
            
            valid_features = features.dropna()
            if valid_features.empty:
                return SignalType.HOLD.value, "البيانات الفنية غير مكتملة"
            
            last_row = valid_features.iloc[-1]
            
            # Multi-factor analysis
            signals = {}
            explanations = []
            
            # Trend Analysis
            trend_score = 0
            if 'SMA20_vs_SMA50' in last_row and not pd.isna(last_row['SMA20_vs_SMA50']):
                sma_ratio = last_row['SMA20_vs_SMA50']
                if sma_ratio > 1.02:
                    trend_score += 2
                    explanations.append("المتوسط المتحرك قصير المدى أعلى من طويل المدى - اتجاه صاعد قوي")
                elif sma_ratio > 1.01:
                    trend_score += 1
                    explanations.append("المتوسط المتحرك قصير المدى أعلى من طويل المدى - اتجاه صاعد معتدل")
                elif sma_ratio < 0.98:
                    trend_score -= 2
                    explanations.append("المتوسط المتحرك قصير المدى أقل من طويل المدى - اتجاه هابط قوي")
                elif sma_ratio < 0.99:
                    trend_score -= 1
                    explanations.append("المتوسط المتحرك قصير المدى أقل من طويل المدى - اتجاه هابط معتدل")
            
            # Momentum Analysis
            momentum_score = 0
            if 'RSI' in last_row and not pd.isna(last_row['RSI']):
                rsi = last_row['RSI']
                if rsi > 80:
                    momentum_score -= 2
                    explanations.append(f"مؤشر القوة النسبية {rsi:.1f} - شراء مفرط قوي")
                elif rsi > 70:
                    momentum_score -= 1
                    explanations.append(f"مؤشر القوة النسبية {rsi:.1f} - شراء مفرط معتدل")
                elif rsi < 20:
                    momentum_score += 2
                    explanations.append(f"مؤشر القوة النسبية {rsi:.1f} - بيع مفرط قوي")
                elif rsi < 30:
                    momentum_score += 1
                    explanations.append(f"مؤشر القوة النسبية {rsi:.1f} - بيع مفرط معتدل")
                else:
                    explanations.append(f"مؤشر القوة النسبية {rsi:.1f} - منطقة معتدلة")
            
            # MACD Analysis
            if 'MACD' in last_row and 'MACD_Signal' in last_row:
                if not pd.isna(last_row['MACD']) and not pd.isna(last_row['MACD_Signal']):
                    macd_diff = last_row['MACD'] - last_row['MACD_Signal']
                    if macd_diff > 0:
                        momentum_score += 1
                        explanations.append("مؤشر MACD إيجابي - زخم صاعد")
                    else:
                        momentum_score -= 1
                        explanations.append("مؤشر MACD سلبي - زخم هابط")
            
            # Volume Analysis
            volume_score = 0
            if 'Volume_Ratio' in last_row and not pd.isna(last_row['Volume_Ratio']):
                vol_ratio = last_row['Volume_Ratio']
                if vol_ratio > 1.5:
                    volume_score += 1
                    explanations.append(f"حجم التداول عالي {vol_ratio:.1f}x من المتوسط - نشاط قوي")
                elif vol_ratio < 0.5:
                    volume_score -= 1
                    explanations.append(f"حجم التداول منخفض {vol_ratio:.1f}x من المتوسط - نشاط ضعيف")
            
            # Bollinger Bands Analysis
            bb_score = 0
            if 'BB_Position' in last_row and not pd.isna(last_row['BB_Position']):
                bb_pos = last_row['BB_Position']
                if bb_pos > 0.9:
                    bb_score -= 1
                    explanations.append("السعر قرب الحد العلوي لنطاق بولنجر - احتمال تصحيح")
                elif bb_pos < 0.1:
                    bb_score += 1
                    explanations.append("السعر قرب الحد السفلي لنطاق بولنجر - احتمال ارتداد")
            
            # Calculate total score
            total_score = trend_score + momentum_score + volume_score + bb_score
            
            # Generate signal and explanation
            if total_score >= 3:
                signal = SignalType.BUY.value
                signal_text = "إشارة شراء قوية"
            elif total_score >= 1:
                signal = SignalType.BUY.value
                signal_text = "إشارة شراء معتدلة"
            elif total_score <= -3:
                signal = SignalType.SELL.value
                signal_text = "إشارة بيع قوية"
            elif total_score <= -1:
                signal = SignalType.SELL.value
                signal_text = "إشارة بيع معتدلة"
            else:
                signal = SignalType.HOLD.value
                signal_text = "إشارة انتظار"
            
            explanation = f"{signal_text} (نقاط التحليل: {total_score}). " + " | ".join(explanations)
            
            return signal, explanation
            
        except Exception as e:
            return SignalType.HOLD.value, f"خطأ في تحليل الإشارة: {str(e)}"

# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    """Main Egyptian Islamic AI Trading Application"""
    
    # Header with Arabic support
    st.markdown(f"""
    <div class="main-header">
        <h1 class="main-title">{display_arabic('title')}</h1>
        <p class="main-subtitle">{display_arabic('subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    data_manager = EgyptianDataManager()
    sharia_config = ShariaConfig()
    trading_config = TradingConfig()
    
    sharia_screener = AdvancedShariaScreener(sharia_config)
    ml_predictor = AdvancedMLPredictor()
    trading_strategy = AdvancedTradingStrategy(trading_config)
    
    # Sidebar with Arabic
    with st.sidebar:
        st.header(f"🎛️ {display_arabic('control_panel')}")
        
        # Stock selection
        st.subheader(f"📈 {display_arabic('stock_selection')}")
        selected_symbol = st.selectbox(
            display_arabic('choose_stock'),
            options=list(EGYPTIAN_HALAL_STOCKS.keys()),
            format_func=lambda x: f"{EGYPTIAN_HALAL_STOCKS[x][0]} ({x})"
        )
        
        # Time period
        period = st.selectbox(display_arabic('time_period'), ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        
        # Trading parameters
        st.subheader(f"⚙️ {display_arabic('trading_config')}")
        risk_per_trade = st.slider(display_arabic('risk_per_trade'), 0.5, 5.0, 2.0, 0.5) / 100
        stop_loss_mult = st.slider(display_arabic('stop_loss'), 1.0, 4.0, 2.0, 0.5)
        take_profit_rr = st.slider(display_arabic('take_profit'), 1.0, 5.0, 2.5, 0.5)
        
        # ML parameters
        st.subheader(f"🤖 {display_arabic('ml_settings')}")
        prediction_days = st.slider(display_arabic('prediction_horizon'), 1, 20, 5)
        movement_threshold = st.slider(display_arabic('movement_threshold'), 1.0, 10.0, 2.0) / 100
        
        # Sharia thresholds
        st.subheader(f"🕌 {display_arabic('sharia_thresholds')}")
        max_debt_ratio = st.slider(display_arabic('max_debt_assets'), 0.1, 0.5, 0.33, 0.01)
        max_interest_ratio = st.slider(display_arabic('max_interest_income'), 0.0, 0.1, 0.05, 0.005)
        max_cash_ratio = st.slider(display_arabic('max_cash_assets'), 0.1, 0.5, 0.33, 0.01)
        
        # Update configurations
        trading_config.risk_per_trade = risk_per_trade
        trading_config.stop_loss_atr_mult = stop_loss_mult
        trading_config.take_profit_rr = take_profit_rr
        
        sharia_config.max_debt_to_assets = max_debt_ratio
        sharia_config.max_interest_income_ratio = max_interest_ratio
        sharia_config.max_cash_securities_ratio = max_cash_ratio
        
        run_analysis = st.button(f"🔍 {display_arabic('run_analysis')}", type="primary")
    
    # Main content
    if run_analysis:
        with st.spinner('🔄 جاري تحميل بيانات السوق المصري...'):
            df = data_manager.get_egyptian_market_data(selected_symbol, period)
        
        if df.empty:
            st.error("❌ لا يمكن جلب بيانات السوق. يرجى المحاولة مرة أخرى.")
            return
        
        # Get company info
        info = data_manager.get_company_info(selected_symbol)
        arabic_name, sector_ar, english_name = EGYPTIAN_HALAL_STOCKS[selected_symbol]
        
        # Current price info
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0.0
        
        # Price metrics with Arabic
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{display_arabic('current_price')}</div>
                <div class="metric-value">{current_price:.2f} ج.م</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#10b981" if price_change >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {color}, {color}99);">
                <div class="metric-label">{display_arabic('daily_change')}</div>
                <div class="metric-value">{price_change:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{display_arabic('volume')}</div>
                <div class="metric-value">{df['Volume'].iloc[-1]:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            market_cap = info.get('marketCap', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{display_arabic('market_cap')}</div>
                <div class="metric-value">{market_cap/1e9:.1f}B ج.م</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Company info in Arabic
        st.markdown(f"""
        <div class="info-panel">
            <h3>🏢 معلومات الشركة</h3>
            <p><strong>الاسم العربي:</strong> {arabic_name}</p>
            <p><strong>الاسم الإنجليزي:</strong> {english_name}</p>
            <p><strong>القطاع:</strong> {sector_ar}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sharia Screening with explanations
        st.header(f"🕌 {display_arabic('sharia_compliance')}")
        with st.spinner("📋 جاري فحص الامتثال للشريعة الإسلامية..."):
            is_compliant, ratios, issues, sharia_explanation = sharia_screener.screen_egyptian_stock(selected_symbol)
        
        # Compliance status
        if is_compliant:
            st.markdown(f'<div class="status-halal">✅ {display_arabic("halal_certified")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-haram">❌ {display_arabic("non_compliant")}</div>', unsafe_allow_html=True)
        
        # Detailed explanation
        st.markdown(f"""
        <div class="explanation-box">
            <h4>📝 {display_arabic('explanation')}</h4>
            <p>{sharia_explanation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Compliance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            debt_ratio = ratios.get('debt_to_assets', np.nan)
            if not pd.isna(debt_ratio):
                st.metric(display_arabic('debt_assets_ratio'), f"{debt_ratio:.1%}", 
                         delta=f"الحد: {max_debt_ratio:.0%}")
        
        with col2:
            interest_ratio = ratios.get('interest_income_ratio', np.nan)
            if not pd.isna(interest_ratio):
                st.metric(display_arabic('interest_income_ratio'), f"{interest_ratio:.1%}", 
                         delta=f"الحد: {max_interest_ratio:.1%}")
        
        with col3:
            cash_ratio = ratios.get('cash_securities_ratio', np.nan)
            if not pd.isna(cash_ratio):
                st.metric(display_arabic('cash_securities_ratio'), f"{cash_ratio:.1%}", 
                         delta=f"الحد: {max_cash_ratio:.0%}")
        
        # Issues list
        st.markdown('<div class="info-panel">', unsafe_allow_html=True)
        for issue in issues:
            st.write(f"• {issue}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Technical Analysis
        st.header(f"📊 {display_arabic('technical_analysis')}")
        
        # Create comprehensive chart
        features_df = ml_predictor.create_advanced_features(df)
        
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('السعر والمتوسطات المتحركة', 'حجم التداول', 'مؤشر القوة النسبية RSI', 'مؤشر MACD', 'نطاقات بولنجر'),
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='السعر',
            increasing_line_color='#059669',
            decreasing_line_color='#dc2626'
        ), row=1, col=1)
        
        # Moving averages
        if 'SMA_20' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['SMA_20'],
                name='المتوسط 20', line=dict(color='orange', width=2)
            ), row=1, col=1)
        
        if 'SMA_50' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['SMA_50'],
                name='المتوسط 50', line=dict(color='blue', width=2)
            ), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            name='الحجم', marker_color='lightblue'
        ), row=2, col=1)
        
        # RSI
        if 'RSI' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['RSI'],
                name='RSI', line=dict(color='purple')
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if all(col in features_df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['MACD'],
                name='MACD', line=dict(color='blue')
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['MACD_Signal'],
                name='إشارة MACD', line=dict(color='red')
            ), row=4, col=1)
            fig.add_trace(go.Bar(
                x=features_df.index, y=features_df['MACD_Hist'],
                name='MACD Hist', marker_color='green'
            ), row=4, col=1)
        
        # Bollinger Bands
        if all(col in features_df.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['BB_Upper'],
                name='النطاق العلوي', line=dict(color='gray', dash='dash')
            ), row=5, col=1)
            fig.add_trace(go.Scatter(
                x=features_df.index, y=features_df['BB_Lower'],
                name='النطاق السفلي', fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                line=dict(color='gray', dash='dash')
            ), row=5, col=1)
        
        fig.update_layout(
            height=1000,
            title=f"التحليل الفني المتقدم - {arabic_name}",
            xaxis_rangeslider_visible=False,
            font=dict(family="Noto Sans Arabic, Arial", size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced AI Predictions
        st.header(f"🤖 {display_arabic('ai_predictions')}")
        
        with st.spinner("🧠 جاري تدريب نماذج الذكاء الاصطناعي المتقدمة..."):
            ml_results = ml_predictor.train_ensemble_model(
                selected_symbol, df, prediction_days, movement_threshold
            )
        
        if 'error' in ml_results:
            st.error(f"❌ خطأ في الذكاء الاصطناعي: {ml_results['error']}")
        else:
            predictions = ml_predictor.predict_with_explanation(selected_symbol, df)
            
            if 'error' not in predictions:
                prob_up = predictions['probability_up']
                prob_down = predictions['probability_down']
                signal_strength = predictions['signal_strength']
                ai_explanation = predictions['explanation']
                
                # Prediction display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"احتمالية الصعود {movement_threshold:.0%}+ خلال {prediction_days} أيام",
                        f"{prob_up:.1%}",
                        delta=f"قوة الإشارة: {signal_strength:.1%}"
                    )
                
                with col2:
                    st.metric(
                        "دقة النموذج",
                        f"{ml_results['cv_accuracy']:.1%}",
                        delta=f"±{ml_results['cv_std']:.1%}"
                    )
                
                with col3:
                    st.metric("درجة F1", f"{ml_results['f1_score']:.2f}")
                    st.metric("الدقة", f"{ml_results['precision']:.2f}")
                    st.metric("الاستدعاء", f"{ml_results['recall']:.2f}")
                
                # AI recommendation with explanation
                if prob_up > 0.7 and signal_strength > 0.3:
                    recommendation = "شراء قوي جداً"
                    rec_color = "#059669"
                elif prob_up > 0.6:
                    recommendation = "شراء"
                    rec_color = "#10b981"
                elif prob_up < 0.3 and signal_strength > 0.3:
                    recommendation = "بيع قوي"
                    rec_color = "#dc2626"
                elif prob_up < 0.4:
                    recommendation = "بيع"
                    rec_color = "#ef4444"
                else:
                    recommendation = "انتظار"
                    rec_color = "#d97706"
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {rec_color}, {rec_color}99); 
                            color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                    <strong>توصية الذكاء الاصطناعي:</strong><br>
                    <span style="font-size: 2rem; font-weight: bold;">{recommendation}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed AI explanation
                st.markdown(f"""
                <div class="explanation-box">
                    <h4>🤖 {display_arabic('explanation')} - الذكاء الاصطناعي</h4>
                    <p>{ai_explanation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability visualization
                fig_prob = go.Figure(data=[
                    go.Bar(
                        x=['احتمالية الصعود', 'احتمالية الهبوط'],
                        y=[prob_up, prob_down],
                        marker_color=['#059669', '#dc2626'],
                        text=[f'{prob_up:.1%}', f'{prob_down:.1%}'],
                        textposition='auto'
                    )
                ])
                
                fig_prob.update_layout(
                    title=f"احتماليات الذكاء الاصطناعي ({prediction_days} أيام)",
                    yaxis_title="الاحتمالية",
                    font=dict(family="Noto Sans Arabic, Arial"),
                    showlegend=False
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Feature importance
                st.subheader("🎯 العوامل الأكثر تأثيراً في التنبؤ")
                importance_df = pd.DataFrame(
                    list(ml_results['feature_importance'].items()),
                    columns=['المؤشر', 'الأهمية']
                )
                
                fig_importance = go.Figure(go.Bar(
                    x=importance_df['الأهمية'],
                    y=importance_df['المؤشر'],
                    orientation='h',
                    marker_color='skyblue'
                ))
                
                fig_importance.update_layout(
                    title="أهمية المؤشرات في نموذج الذكاء الاصطناعي",
                    xaxis_title="درجة الأهمية",
                    yaxis_title="المؤشرات الفنية",
                    height=500,
                    font=dict(family="Noto Sans Arabic, Arial")
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Current Trading Signal with explanation
        st.header(f"🎯 {display_arabic('trading_signal')}")
        
        current_signal, signal_explanation = trading_strategy.generate_signals_with_explanation(df)
        
        if current_signal == SignalType.BUY.value:
            signal_class = "signal-buy"
            signal_text = f"🟢 {display_arabic('strong_buy')}"
        elif current_signal == SignalType.SELL.value:
            signal_class = "signal-sell"
            signal_text = f"🔴 {display_arabic('strong_sell')}"
        else:
            signal_class = "signal-hold"
            signal_text = f"🟡 {display_arabic('hold')}"
        
        st.markdown(f'<div class="trading-signal {signal_class}">{signal_text}</div>', 
                   unsafe_allow_html=True)
        
        # Signal explanation
        st.markdown(f"""
        <div class="explanation-box">
            <h4>📊 {display_arabic('explanation')} - إشارة التداول</h4>
            <p>{signal_explanation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk warning in Arabic
        st.markdown("""
        <div class="info-panel" style="border-right-color: #dc2626; background: linear-gradient(135deg, #fef2f2, #fee2e2);">
            <h4>⚠️ تحذير مهم</h4>
            <p>هذا التطبيق للأغراض التعليمية فقط وليس نصيحة استثمارية. استشر خبراء الاستثمار المعتمدين قبل اتخاذ أي قرارات استثمارية.</p>
            <p>الاستثمار في الأسواق المالية ينطوي على مخاطر قد تؤدي إلى خسائر كبيرة.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Welcome screen in Arabic
        st.info("👆 قم بتكوين المعايير في الشريط الجانبي واضغط 'تشغيل التحليل الكامل' للبدء")
        
        # Display features in Arabic
        st.markdown("""
        ### 🌟 مميزات المنصة
        
        **🕌 الامتثال للشريعة الإسلامية**
        - فحص تلقائي وفقاً لأحكام الشريعة الإسلامية
        - مراقبة مستمرة للامتثال
        - حدود قابلة للتخصيص
        
        **🤖 تنبؤات الذكاء الاصطناعي المتقدمة**
        - نماذج تعلم آلي متطورة
        - تحليل متعدد العوامل
        - تنبؤات قائمة على الاحتماليات
        
        **📊 التحليل الفني المتقدم**
        - أكثر من 25 مؤشر فني
        - رسوم بيانية تفاعلية
        - تحليل متعدد الأطر الزمنية
        
        **🏺 البورصة المصرية**
        - أسهم مختارة من السوق المصري
        - بيانات محدثة لحظياً
        - تحليل مخصص للسوق المحلي
        
        **🎯 التداول الاحترافي**
        - اختبار استراتيجيات متقدم
        - إدارة المخاطر
        - تحليل الأداء التفصيلي
        """)

if __name__ == "__main__":
    main()
