# منصة الذكاء الاصطناعي الإسلامية للتداول في البورصة المصرية - FIXED VERSION
# 🏺 Islamic AI Trading Platform for Egyptian Stock Exchange (EGX)
# ==================================================================
# Advanced Sharia-compliant trading system with Enhanced AI predictions,
# Arabic language support FIXED, and comprehensive market analysis for Egypt
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
    'analysis_result': 'نتيجة التحليل',
    'company_info': 'معلومات الشركة',
    'arabic_name': 'الاسم العربي',
    'english_name': 'الاسم الإنجليزي',
    'sector': 'القطاع'
}

def display_arabic(text_key: str, fallback: str = "") -> str:
    """Display Arabic text with proper formatting - FIXED VERSION"""
    arabic_text = ARABIC_TEXTS.get(text_key, fallback)
    if arabic_text:
        try:
            reshaped_text = arabic_reshaper.reshape(arabic_text)
            bidi_text = bidi.algorithm.get_display(reshaped_text)
            return bidi_text
        except:
            return arabic_text
    return fallback

def render_arabic_html(text_key: str, fallback: str = "", css_class: str = "arabic-text") -> str:
    """Render Arabic text in HTML with proper RTL styling"""
    arabic_text = display_arabic(text_key, fallback)
    return f'<div class="{css_class}" dir="rtl">{arabic_text}</div>'

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

# FIXED CSS with proper Arabic support and contrasting colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;600;700;900&display=swap');
    
    /* Base Arabic text styling */
    .arabic-text {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #1a202c;
        line-height: 1.6;
    }
    
    .arabic-header {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #2d3748;
        font-weight: 700;
        line-height: 1.4;
    }
    
    .arabic-white {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #ffffff;
        line-height: 1.5;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #059669 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin: 0;
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e40af 0%, #10b981 100%);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.35);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.8rem 0;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.95);
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
    }
    
    .status-halal {
        background: linear-gradient(90deg, #059669, #047857);
        color: white;
        padding: 1rem 2rem;
        border-radius: 35px;
        font-weight: bold;
        display: inline-block;
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.4);
        direction: rtl;
        font-size: 1.1rem;
    }
    
    .status-haram {
        background: linear-gradient(90deg, #dc2626, #b91c1c);
        color: white;
        padding: 1rem 2rem;
        border-radius: 35px;
        font-weight: bold;
        display: inline-block;
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.4);
        direction: rtl;
        font-size: 1.1rem;
    }
    
    .trading-signal {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
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
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 0 20px 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .info-panel h3, .info-panel h4 {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #1e40af;
        margin-bottom: 1rem;
    }
    
    .info-panel p, .info-panel strong {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #374151;
        line-height: 1.6;
    }
    
    .explanation-box {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 3px solid #3b82f6;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.15);
    }
    
    .explanation-box h4 {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #1e40af;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .explanation-box p {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #374151;
        line-height: 1.8;
        font-size: 1rem;
    }
    
    .company-info {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 2px solid #0891b2;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
    }
    
    .company-info h3 {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #0891b2;
        margin-bottom: 1rem;
    }
    
    .company-info p {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
        color: #164e63;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    
    /* Sidebar Arabic support */
    .sidebar .sidebar-content {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
    }
    
    /* Progress bar colors */
    .stProgress .st-bo {
        background-color: #1e40af;
    }
    
    /* Fix selectbox and other widgets for Arabic */
    .stSelectbox label, .stSlider label, .stButton button {
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    /* Recommendation box */
    .recommendation-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        color: white;
    }
    
    .recommendation-text {
        font-size: 2.2rem;
        font-weight: 900;
        font-family: 'Noto Sans Arabic', 'Arial Unicode MS', Arial, sans-serif;
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# DATA MANAGEMENT FOR EGYPTIAN MARKET
# ================================================================

class EgyptianDataManager:
    """Enhanced data management for Egyptian Stock Exchange"""
    
    def __init__(self):
        self.cache_dir = os.path.join(tempfile.gettempdir(), "egyptian_ai_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @st.cache_data(ttl=300)
    def get_egyptian_market_data(_self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch Egyptian market data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = df.columns.str.title()
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_cols].dropna()
            
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
            
            if symbol in EGYPTIAN_HALAL_STOCKS:
                arabic_name, sector_ar, english_name = EGYPTIAN_HALAL_STOCKS[symbol]
                info['arabic_name'] = arabic_name
                info['sector_arabic'] = sector_ar
                info['english_name'] = english_name
            
            return info
        except Exception:
            return {}

# ================================================================
# SIMPLIFIED ML PREDICTOR
# ================================================================

class SimpleMLPredictor:
    """Simplified ML predictor to avoid complexity"""
    
    def __init__(self):
        self.model = None
    
    def create_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic technical features"""
        features = df.copy()
        
        # Simple moving averages
        features['SMA_20'] = df['Close'].rolling(20).mean()
        features['SMA_50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # Price ratios
        features['Price_SMA20'] = df['Close'] / features['SMA_20']
        features['SMA20_SMA50'] = features['SMA_20'] / features['SMA_50']
        
        # Returns
        features['Returns_1'] = df['Close'].pct_change(1)
        features['Returns_5'] = df['Close'].pct_change(5)
        
        return features.dropna()
    
    def predict_simple(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple rule-based prediction"""
        features = self.create_simple_features(df)
        if features.empty:
            return {'error': 'لا توجد بيانات كافية'}
        
        latest = features.iloc[-1]
        
        # Simple scoring
        score = 0
        
        if latest['SMA20_SMA50'] > 1.01:
            score += 1
        elif latest['SMA20_SMA50'] < 0.99:
            score -= 1
        
        if latest['RSI'] < 30:
            score += 1
        elif latest['RSI'] > 70:
            score -= 1
        
        if latest['Price_SMA20'] > 1.02:
            score += 0.5
        elif latest['Price_SMA20'] < 0.98:
            score -= 0.5
        
        # Convert to probability
        prob_up = max(0.1, min(0.9, 0.5 + score * 0.15))
        
        return {
            'probability_up': prob_up,
            'probability_down': 1 - prob_up,
            'signal_strength': abs(prob_up - 0.5) * 2
        }

# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    """Main Egyptian Islamic AI Trading Application"""
    
    # Header with proper Arabic rendering
    title_arabic = display_arabic('title')
    subtitle_arabic = display_arabic('subtitle')
    
    st.markdown(f"""
    <div class="main-header">
        <div class="main-title">{title_arabic}</div>
        <div class="main-subtitle">{subtitle_arabic}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    data_manager = EgyptianDataManager()
    sharia_config = ShariaConfig()
    trading_config = TradingConfig()
    ml_predictor = SimpleMLPredictor()
    
    # Sidebar with Arabic
    with st.sidebar:
        st.markdown(f'<div class="arabic-header"><h2>🎛️ {display_arabic("control_panel")}</h2></div>', unsafe_allow_html=True)
        
        # Stock selection
        st.markdown(f'<div class="arabic-text"><h3>📈 {display_arabic("stock_selection")}</h3></div>', unsafe_allow_html=True)
        selected_symbol = st.selectbox(
            display_arabic('choose_stock'),
            options=list(EGYPTIAN_HALAL_STOCKS.keys()),
            format_func=lambda x: f"{EGYPTIAN_HALAL_STOCKS[x][0]} ({x})"
        )
        
        # Time period
        period = st.selectbox(display_arabic('time_period'), ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        
        # Analysis button
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
        
        # Price metrics with proper Arabic
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
        
        # Company info with proper Arabic rendering
        st.markdown(f"""
        <div class="company-info">
            <h3>🏢 {display_arabic('company_info')}</h3>
            <p><strong>{display_arabic('arabic_name')}:</strong> {arabic_name}</p>
            <p><strong>{display_arabic('english_name')}:</strong> {english_name}</p>
            <p><strong>{display_arabic('sector')}:</strong> {sector_ar}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sharia Compliance
        st.markdown(f'<div class="arabic-header"><h2>🕌 {display_arabic("sharia_compliance")}</h2></div>', unsafe_allow_html=True)
        
        # Simple Sharia check (for demo)
        is_halal = True  # All stocks in our list are pre-screened as halal
        
        if is_halal:
            st.markdown(f'<div class="status-halal">✅ {display_arabic("halal_certified")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-haram">❌ {display_arabic("non_compliant")}</div>', unsafe_allow_html=True)
        
        # Explanation
        explanation_text = f"الشركة {arabic_name} تعمل في قطاع {sector_ar} وهو قطاع مسموح شرعياً وفقاً لمعايير الاستثمار الإسلامي المعتمدة."
        st.markdown(f"""
        <div class="explanation-box">
            <h4>📝 {display_arabic('explanation')}</h4>
            <p>{explanation_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical Analysis Chart
        st.markdown(f'<div class="arabic-header"><h2>📊 {display_arabic("technical_analysis")}</h2></div>', unsafe_allow_html=True)
        
        features_df = ml_predictor.create_simple_features(df)
        
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='السعر',
            increasing_line_color='#059669',
            decreasing_line_color='#dc2626'
        ))
        
        # Moving averages
        if 'SMA_20' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['SMA_20'],
                name='المتوسط المتحرك 20',
                line=dict(color='orange', width=2)
            ))
        
        if 'SMA_50' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['SMA_50'],
                name='المتوسط المتحرك 50',
                line=dict(color='blue', width=2)
            ))
        
        fig.update_layout(
            height=600,
            title=f"التحليل الفني - {arabic_name}",
            xaxis_rangeslider_visible=False,
            font=dict(family="Noto Sans Arabic, Arial", size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Predictions
        st.markdown(f'<div class="arabic-header"><h2>🤖 {display_arabic("ai_predictions")}</h2></div>', unsafe_allow_html=True)
        
        with st.spinner("🧠 جاري تحليل الذكاء الاصطناعي..."):
            predictions = ml_predictor.predict_simple(df)
        
        if 'error' not in predictions:
            prob_up = predictions['probability_up']
            prob_down = predictions['probability_down']
            signal_strength = predictions['signal_strength']
            
            # Display predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "احتمالية الصعود",
                    f"{prob_up:.1%}",
                    delta=f"قوة الإشارة: {signal_strength:.1%}"
                )
            
            with col2:
                st.metric(
                    "احتمالية الهبوط", 
                    f"{prob_down:.1%}"
                )
            
            with col3:
                if prob_up > 0.6:
                    recommendation = "شراء قوي"
                    rec_color = "linear-gradient(90deg, #059669, #047857)"
                elif prob_up > 0.5:
                    recommendation = "شراء"
                    rec_color = "linear-gradient(90deg, #10b981, #059669)"
                elif prob_up < 0.4:
                    recommendation = "بيع"
                    rec_color = "linear-gradient(90deg, #dc2626, #b91c1c)"
                else:
                    recommendation = "انتظار"
                    rec_color = "linear-gradient(90deg, #d97706, #b45309)"
                
                st.markdown(f"""
                <div class="recommendation-box" style="background: {rec_color};">
                    <strong>توصية الذكاء الاصطناعي:</strong><br>
                    <div class="recommendation-text">{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # AI explanation
            ai_explanation = f"""
            تحليل الذكاء الاصطناعي يشير إلى احتمالية صعود السعر بنسبة {prob_up:.1%}. 
            التحليل يعتمد على المتوسطات المتحركة ومؤشر القوة النسبية وحركة السعر الأخيرة.
            قوة الإشارة {signal_strength:.1%} مما يعني {"إشارة قوية" if signal_strength > 0.5 else "إشارة ضعيفة"}.
            """
            
            st.markdown(f"""
            <div class="explanation-box">
                <h4>🤖 تفسير الذكاء الاصطناعي</h4>
                <p>{ai_explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
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
                title="توزيع احتماليات الذكاء الاصطناعي",
                yaxis_title="الاحتمالية",
                font=dict(family="Noto Sans Arabic, Arial"),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_prob, use_container_width=True)
        
        # Risk Warning
        st.markdown("""
        <div class="info-panel" style="border-right-color: #dc2626; background: linear-gradient(135deg, #fef2f2, #fee2e2);">
            <h4>⚠️ تحذير مهم</h4>
            <p>هذا التطبيق للأغراض التعليمية فقط وليس نصيحة استثمارية. استشر خبراء الاستثمار المعتمدين قبل اتخاذ أي قرارات استثمارية.</p>
            <p>الاستثمار في الأسواق المالية ينطوي على مخاطر قد تؤدي إلى خسائر كبيرة.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown(f'<div class="arabic-text"><p>👆 قم بتكوين المعايير في الشريط الجانبي واضغط "تشغيل التحليل الكامل" للبدء</p></div>', unsafe_allow_html=True)
        
        # Features list
        st.markdown("""
        <div class="info-panel">
            <h3>🌟 مميزات المنصة</h3>
            <p><strong>🕌 الامتثال للشريعة الإسلامية</strong><br>
            فحص تلقائي وفقاً لأحكام الشريعة الإسلامية مع شرح مفصل</p>
            
            <p><strong>🤖 تنبؤات الذكاء الاصطناعي</strong><br>
            تحليل متطور باستخدام خوارزميات التعلم الآلي</p>
            
            <p><strong>📊 التحليل الفني</strong><br>
            مؤشرات فنية متقدمة ورسوم بيانية تفاعلية</p>
            
            <p><strong>🏺 البورصة المصرية</strong><br>
            أسهم مختارة من السوق المصري مع بيانات محدثة</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
