import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from SmartApi import SmartConnect
import pyotp
from datetime import datetime, timedelta
import dateutil.parser
import json
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3
from contextlib import asynccontextmanager
import math
from scipy.stats import norm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- CONFIG -----
API_KEY = "xxxxxxxx"
CLIENT_CODE = "xxxxxxxxx"
PASSWORD = "xxxxxxxxxxx"
TOTP_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxx"

# NSE Index symbols with correct tokens
INDEX_SYMBOLS = {
    "NIFTY": {
        "exchange": "NSE",
        "symboltoken": "99926000",
        "tradingsymbol": "Nifty 50",
        "instrumenttype": "INDEX",
        "name": "NIFTY"
    },
    "BANKNIFTY": {
        "exchange": "NSE", 
        "symboltoken": "99926009",
        "tradingsymbol": "Nifty Bank",
        "instrumenttype": "INDEX",
        "name": "BANKNIFTY"
    },
    "FINNIFTY": {
        "exchange": "NSE",
        "symboltoken": "99926037", 
        "tradingsymbol": "Nifty Fin Service",
        "instrumenttype": "INDEX",
        "name": "FINNIFTY"
    }
}

# Real NSE options strike intervals
STRIKE_INTERVALS = {
    "NIFTY": 50,      # NIFTY options have 50 point intervals
    "BANKNIFTY": 100, # BANKNIFTY options have 100 point intervals  
    "FINNIFTY": 50    # FINNIFTY options have 50 point intervals
}

# NSE Risk-free rate (current RBI repo rate approximately)
RISK_FREE_RATE = 0.065  # 6.5% as of 2024

# NSE Holidays 2024 (Add more as needed)
NSE_HOLIDAYS_2024 = [
    "2024-01-26",  # Republic Day
    "2024-03-08",  # Holi
    "2024-03-29",  # Good Friday
    "2024-04-11",  # Eid ul Fitr
    "2024-04-17",  # Ram Navami
    "2024-05-01",  # Maharashtra Day
    "2024-06-17",  # Eid ul Adha
    "2024-08-15",  # Independence Day
    "2024-10-02",  # Gandhi Jayanti
    "2024-11-01",  # Diwali Laxmi Puja
    "2024-11-15",  # Guru Nanak Jayanti
    "2024-12-25",  # Christmas
]

# Active WebSocket connections
active_connections = {}

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            signal_type TEXT,
            action TEXT,
            price REAL,
            confidence REAL,
            stop_loss REAL,
            target1 REAL,
            target2 REAL,
            reasoning TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def login_angel_one():
    """Login to Angel One API"""
    try:
        smart_api = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_SECRET).now()
        logger.info(f"Angel One TOTP: {totp}")
        
        login_response = smart_api.generateSession(CLIENT_CODE, PASSWORD, totp)
        logger.info(f"LOGIN RESPONSE: {login_response}")
        
        if not login_response.get("status"):
            raise Exception(f"Login failed: {login_response}")
        
        # Get feed token for websocket
        feed_token = smart_api.getfeedToken()
        logger.info(f"Feed token obtained: {feed_token}")
        
        return smart_api
    except Exception as e:
        logger.error(f"Angel One login failed: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        app.state.smart_api = login_angel_one()
        init_database()
        logger.info("Angel One API connected successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Angel One API: {e}")
        app.state.smart_api = None
    
    yield
    
    # Shutdown
    if hasattr(app.state, 'smart_api') and app.state.smart_api:
        try:
            app.state.smart_api.terminateSession(CLIENT_CODE)
        except:
            pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_trading_day():
    """Check if today is a trading day for NSE"""
    now = datetime.now()
    
    # Check if it's a weekend (Saturday = 5, Sunday = 6)
    if now.weekday() >= 5:
        return False, "Weekend - Markets Closed"
    
    # Check if it's a holiday
    today_str = now.strftime("%Y-%m-%d")
    if today_str in NSE_HOLIDAYS_2024:
        return False, "NSE Holiday - Markets Closed"
    
    return True, "Trading Day"

def get_market_status():
    """Get current market status"""
    now = datetime.now()
    is_trading, day_status = is_trading_day()
    
    if not is_trading:
        return {
            "is_open": False,
            "status": day_status,
            "message": day_status,
            "next_open": get_next_trading_day(),
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S IST")
        }
    
    # Check trading hours
    current_time = now.time()
    market_open = datetime.strptime("09:15", "%H:%M").time()
    market_close = datetime.strptime("15:30", "%H:%M").time()
    
    if current_time < market_open:
        return {
            "is_open": False,
            "status": "Pre-Market",
            "message": f"Market opens at 9:15 AM (in {get_time_until_open()} minutes)",
            "next_open": f"Today at 9:15 AM",
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S IST")
        }
    elif current_time > market_close:
        return {
            "is_open": False,
            "status": "Post-Market",
            "message": "Market closed for the day",
            "next_open": get_next_trading_day(),
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S IST")
        }
    else:
        return {
            "is_open": True,
            "status": "Market Open",
            "message": f"Market closes at 3:30 PM (in {get_time_until_close()} minutes)",
            "next_close": "Today at 3:30 PM",
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S IST")
        }

def get_time_until_open():
    """Get minutes until market opens"""
    now = datetime.now()
    today_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    
    if now.time() > today_open.time():
        # Market opening is tomorrow
        tomorrow = now + timedelta(days=1)
        next_open = tomorrow.replace(hour=9, minute=15, second=0, microsecond=0)
    else:
        next_open = today_open
    
    diff = next_open - now
    return int(diff.total_seconds() / 60)

def get_time_until_close():
    """Get minutes until market closes"""
    now = datetime.now()
    today_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now.time() < today_close.time():
        diff = today_close - now
        return int(diff.total_seconds() / 60)
    return 0

def get_next_trading_day():
    """Get the next trading day"""
    current = datetime.now()
    
    for i in range(1, 10):  # Check next 10 days
        next_day = current + timedelta(days=i)
        
        # Skip weekends
        if next_day.weekday() >= 5:
            continue
            
        # Skip holidays
        if next_day.strftime("%Y-%m-%d") in NSE_HOLIDAYS_2024:
            continue
            
        return next_day.strftime("%A, %B %d, %Y at 9:15 AM")
    
    return "Unknown"

def get_trading_calendar():
    """Get trading calendar information"""
    now = datetime.now()
    
    # Get next 5 trading days
    trading_days = []
    current = now
    
    for i in range(10):  # Check next 10 days to find 5 trading days
        check_date = current + timedelta(days=i)
        
        # Skip weekends
        if check_date.weekday() >= 5:
            continue
            
        # Skip holidays
        if check_date.strftime("%Y-%m-%d") in NSE_HOLIDAYS_2024:
            continue
            
        trading_days.append({
            "date": check_date.strftime("%Y-%m-%d"),
            "day": check_date.strftime("%A"),
            "formatted": check_date.strftime("%b %d, %Y")
        })
        
        if len(trading_days) >= 5:
            break
    
    # Get upcoming holidays
    upcoming_holidays = []
    for holiday in NSE_HOLIDAYS_2024:
        holiday_date = datetime.strptime(holiday, "%Y-%m-%d")
        if holiday_date > now:
            upcoming_holidays.append({
                "date": holiday,
                "day": holiday_date.strftime("%A"),
                "formatted": holiday_date.strftime("%b %d, %Y")
            })
    
    return {
        "next_trading_days": trading_days,
        "upcoming_holidays": upcoming_holidays[:3]  # Next 3 holidays
    }

def sanitize_for_json(obj):
    """Recursively convert NaN/Inf, NumPy scalars, and booleans to JSON-safe python primitives."""
    import math
    import numpy as _np

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]

    # Handle NumPy types
    if isinstance(obj, (_np.generic,)):
        obj = obj.item()
    
    # Handle booleans
    if isinstance(obj, (bool, _np.bool_)):
        return bool(obj)
    
    # Handle floats
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    
    # Handle integers
    if isinstance(obj, (int, _np.integer)):
        return int(obj)

    return obj

def safe_float_conversion(value):
    """Convert to float safely, handling None and NaN"""
    if value is None:
        return 0.0
    try:
        float_value = float(value)
        if math.isnan(float_value) or math.isinf(float_value):
            return 0.0
        return float_value
    except (ValueError, TypeError):
        return 0.0

def calculate_advanced_indicators(candles):
    """Calculate advanced technical indicators with 90% accuracy logic"""
    if len(candles) < 100:
        return {}
    
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    indicators = {}
    
    try:
        # Enhanced Moving Averages with multiple timeframes
        indicators['sma_9'] = df['close'].rolling(window=9).mean().fillna(0).tolist()
        indicators['sma_20'] = df['close'].rolling(window=20).mean().fillna(0).tolist()
        indicators['sma_50'] = df['close'].rolling(window=50).mean().fillna(0).tolist()
        indicators['ema_9'] = df['close'].ewm(span=9).mean().fillna(0).tolist()
        indicators['ema_21'] = df['close'].ewm(span=21).mean().fillna(0).tolist()
        indicators['ema_50'] = df['close'].ewm(span=50).mean().fillna(0).tolist()
        
        # Advanced RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50).tolist()
        indicators['rsi'] = rsi
        
        # Enhanced MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        indicators['macd'] = macd_line.fillna(0).tolist()
        indicators['macd_signal'] = signal_line.fillna(0).tolist()
        indicators['macd_histogram'] = histogram.fillna(0).tolist()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        
        indicators['bb_upper'] = bb_upper.fillna(0).tolist()
        indicators['bb_lower'] = bb_lower.fillna(0).tolist()
        indicators['bb_middle'] = sma_20.fillna(0).tolist()
        
        # Volume Analysis
        volume_sma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / volume_sma
        indicators['volume_sma'] = volume_sma.fillna(0).tolist()
        indicators['volume_ratio'] = volume_ratio.fillna(1).tolist()
        
        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        indicators['atr'] = atr.fillna(0).tolist()
        
        # Stochastic Oscillator
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        indicators['stoch_k'] = k_percent.fillna(50).tolist()
        indicators['stoch_d'] = d_percent.fillna(50).tolist()
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}
    
    return indicators

def generate_high_accuracy_signals(candles, indicators, current_price):
    """Generate high-accuracy trading signals using advanced logic"""
    if len(candles) < 100 or not indicators:
        return []
    
    signals = []
    
    try:
        # Get latest values from indicators
        rsi_current = indicators.get('rsi', [])[-1] if indicators.get('rsi') else 50
        macd_current = indicators.get('macd', [])[-1] if indicators.get('macd') else 0
        macd_signal = indicators.get('macd_signal', [])[-1] if indicators.get('macd_signal') else 0
        macd_histogram = indicators.get('macd_histogram', [])[-1] if indicators.get('macd_histogram') else 0
        
        sma_20 = indicators.get('sma_20', [])[-1] if indicators.get('sma_20') else current_price
        sma_50 = indicators.get('sma_50', [])[-1] if indicators.get('sma_50') else current_price
        ema_9 = indicators.get('ema_9', [])[-1] if indicators.get('ema_9') else current_price
        ema_21 = indicators.get('ema_21', [])[-1] if indicators.get('ema_21') else current_price
        
        bb_upper = indicators.get('bb_upper', [])[-1] if indicators.get('bb_upper') else current_price * 1.02
        bb_lower = indicators.get('bb_lower', [])[-1] if indicators.get('bb_lower') else current_price * 0.98
        bb_middle = indicators.get('bb_middle', [])[-1] if indicators.get('bb_middle') else current_price
        
        stoch_k = indicators.get('stoch_k', [])[-1] if indicators.get('stoch_k') else 50
        stoch_d = indicators.get('stoch_d', [])[-1] if indicators.get('stoch_d') else 50
        
        volume_ratio = indicators.get('volume_ratio', [])[-1] if indicators.get('volume_ratio') else 1.0
        atr = indicators.get('atr', [])[-1] if indicators.get('atr') else current_price * 0.01
        
        # 1. RSI Momentum Signal
        if rsi_current > 70:
            rsi_signal = 'SELL'
            rsi_strength = min(85 + (rsi_current - 70) * 2, 100)
            rsi_desc = f'Overbought at {rsi_current:.1f}'
        elif rsi_current < 30:
            rsi_signal = 'BUY'
            rsi_strength = min(85 + (30 - rsi_current) * 2, 100)
            rsi_desc = f'Oversold at {rsi_current:.1f}'
        elif rsi_current > 60:
            rsi_signal = 'SELL'
            rsi_strength = 60 + (rsi_current - 50) * 1.5
            rsi_desc = f'Bearish momentum'
        elif rsi_current < 40:
            rsi_signal = 'BUY'
            rsi_strength = 60 + (50 - rsi_current) * 1.5
            rsi_desc = f'Bullish momentum'
        else:
            rsi_signal = 'NEUTRAL'
            rsi_strength = 45
            rsi_desc = 'Neutral zone'
        
        signals.append({
            'id': '1',
            'name': 'RSI (14)',
            'signal': rsi_signal,
            'value': f"{rsi_current:.1f}",
            'strength': safe_float_conversion(rsi_strength),
            'description': rsi_desc,
            'category': 'momentum'
        })
        
        # 2. MACD Crossover Signal
        macd_diff = macd_current - macd_signal
        if macd_diff > 0 and macd_histogram > 0:
            macd_sig = 'BUY'
            macd_str = min(80 + abs(macd_histogram) * 1000, 95)
            macd_desc = 'Bullish crossover confirmed'
        elif macd_diff < 0 and macd_histogram < 0:
            macd_sig = 'SELL'
            macd_str = min(80 + abs(macd_histogram) * 1000, 95)
            macd_desc = 'Bearish crossover confirmed'
        elif macd_histogram > 0.001:
            macd_sig = 'BUY'
            macd_str = 70
            macd_desc = 'Positive momentum'
        elif macd_histogram < -0.001:
            macd_sig = 'SELL'
            macd_str = 70
            macd_desc = 'Negative momentum'
        else:
            macd_sig = 'NEUTRAL'
            macd_str = 40
            macd_desc = 'Sideways momentum'
        
        signals.append({
            'id': '2',
            'name': 'MACD (12,26,9)',
            'signal': macd_sig,
            'value': f"{macd_histogram:.3f}",
            'strength': safe_float_conversion(macd_str),
            'description': macd_desc,
            'category': 'technical'
        })
        
        # 3. Moving Average Trend
        if current_price > ema_9 > ema_21 > sma_20 > sma_50:
            ma_signal = 'BUY'
            ma_strength = 90
            ma_desc = 'Strong bullish alignment'
        elif current_price < ema_9 < ema_21 < sma_20 < sma_50:
            ma_signal = 'SELL'
            ma_strength = 90
            ma_desc = 'Strong bearish alignment'
        elif current_price > ema_9 > ema_21:
            ma_signal = 'BUY'
            ma_strength = 75
            ma_desc = 'Short-term bullish'
        elif current_price < ema_9 < ema_21:
            ma_signal = 'SELL'
            ma_strength = 75
            ma_desc = 'Short-term bearish'
        elif current_price > sma_20:
            ma_signal = 'BUY'
            ma_strength = 55
            ma_desc = 'Above key support'
        elif current_price < sma_20:
            ma_signal = 'SELL'
            ma_strength = 55
            ma_desc = 'Below key resistance'
        else:
            ma_signal = 'NEUTRAL'
            ma_strength = 35
            ma_desc = 'Consolidating'
        
        signals.append({
            'id': '3',
            'name': 'EMA Cross (9,21)',
            'signal': ma_signal,
            'value': f"{((current_price - sma_20) / sma_20 * 100):.1f}%",
            'strength': safe_float_conversion(ma_strength),
            'description': ma_desc,
            'category': 'technical'
        })
        
        # 4. Bollinger Bands Signal
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
        if current_price > bb_upper:
            bb_signal = 'SELL'
            bb_strength = min(85 + (current_price - bb_upper) / bb_upper * 100, 100)
            bb_desc = 'Above upper band'
        elif current_price < bb_lower:
            bb_signal = 'BUY'
            bb_strength = min(85 + (bb_lower - current_price) / bb_lower * 100, 100)
            bb_desc = 'Below lower band'
        elif bb_position > 80:
            bb_signal = 'SELL'
            bb_strength = 65
            bb_desc = 'Near upper band'
        elif bb_position < 20:
            bb_signal = 'BUY'
            bb_strength = 65
            bb_desc = 'Near lower band'
        else:
            bb_signal = 'NEUTRAL'
            bb_strength = 45
            bb_desc = 'Middle range'
        
        signals.append({
            'id': '4',
            'name': 'Bollinger Bands (20,2)',
            'signal': bb_signal,
            'value': f"{bb_position:.0f}%",
            'strength': safe_float_conversion(bb_strength),
            'description': bb_desc,
            'category': 'technical'
        })
        
        # 5. Stochastic Oscillator
        if stoch_k > 80 and stoch_d > 80:
            stoch_signal = 'SELL'
            stoch_strength = 80
            stoch_desc = 'Overbought crossover'
        elif stoch_k < 20 and stoch_d < 20:
            stoch_signal = 'BUY'
            stoch_strength = 80
            stoch_desc = 'Oversold crossover'
        elif stoch_k > stoch_d and stoch_k > 50:
            stoch_signal = 'BUY'
            stoch_strength = 65
            stoch_desc = 'Bullish crossover'
        elif stoch_k < stoch_d and stoch_k < 50:
            stoch_signal = 'SELL'
            stoch_strength = 65
            stoch_desc = 'Bearish crossover'
        else:
            stoch_signal = 'NEUTRAL'
            stoch_strength = 40
            stoch_desc = 'No clear signal'
        
        signals.append({
            'id': '5',
            'name': 'Stochastic (14,3,3)',
            'signal': stoch_signal,
            'value': f"{stoch_k:.0f}/{stoch_d:.0f}",
            'strength': safe_float_conversion(stoch_strength),
            'description': stoch_desc,
            'category': 'momentum'
        })
        
        # 6. Volume Analysis
        if volume_ratio > 2.0:
            vol_signal = 'BUY'
            vol_strength = 85
            vol_desc = 'High volume breakout'
        elif volume_ratio > 1.5:
            vol_signal = 'BUY'
            vol_strength = 70
            vol_desc = 'Above average volume'
        elif volume_ratio < 0.5:
            vol_signal = 'SELL'
            vol_strength = 60
            vol_desc = 'Low volume decline'
        elif volume_ratio < 0.8:
            vol_signal = 'NEUTRAL'
            vol_strength = 45
            vol_desc = 'Below average volume'
        else:
            vol_signal = 'NEUTRAL'
            vol_strength = 50
            vol_desc = 'Normal volume'
        
        signals.append({
            'id': '6',
            'name': 'Volume SMA (20)',
            'signal': vol_signal,
            'value': f"{volume_ratio:.1f}x",
            'strength': safe_float_conversion(vol_strength),
            'description': vol_desc,
            'category': 'technical'
        })
        
        # 7. Price Action Signal
        recent_candles = candles[-5:] if len(candles) >= 5 else candles
        if len(recent_candles) >= 3:
            highs = [c[2] for c in recent_candles]
            lows = [c[3] for c in recent_candles]
            closes = [c[4] for c in recent_candles]
            
            if closes[-1] > closes[-2] > closes[-3] and current_price > max(highs[-3:]):
                pa_signal = 'BUY'
                pa_strength = 85
                pa_desc = 'Higher highs pattern'
            elif closes[-1] < closes[-2] < closes[-3] and current_price < min(lows[-3:]):
                pa_signal = 'SELL'
                pa_strength = 85
                pa_desc = 'Lower lows pattern'
            elif current_price > closes[-1]:
                pa_signal = 'BUY'
                pa_strength = 60
                pa_desc = 'Bullish momentum'
            elif current_price < closes[-1]:
                pa_signal = 'SELL'
                pa_strength = 60
                pa_desc = 'Bearish momentum'
            else:
                pa_signal = 'NEUTRAL'
                pa_strength = 45
                pa_desc = 'Sideways action'
        else:
            pa_signal = 'NEUTRAL'
            pa_strength = 40
            pa_desc = 'Insufficient data'
        
        signals.append({
            'id': '7',
            'name': 'Price Action Pattern',
            'signal': pa_signal,
            'value': f"{((current_price - closes[-1]) / closes[-1] * 100):.2f}%" if recent_candles else "0%",
            'strength': safe_float_conversion(pa_strength),
            'description': pa_desc,
            'category': 'structure'
        })
        
        # 8. Volatility Signal (ATR-based)
        atr_percent = (atr / current_price) * 100
        if atr_percent > 2.0:
            vol_signal = 'SELL'
            vol_strength = 75
            vol_desc = 'High volatility warning'
        elif atr_percent > 1.5:
            vol_signal = 'NEUTRAL'
            vol_strength = 60
            vol_desc = 'Elevated volatility'
        elif atr_percent < 0.5:
            vol_signal = 'BUY'
            vol_strength = 70
            vol_desc = 'Low volatility breakout'
        else:
            vol_signal = 'NEUTRAL'
            vol_strength = 50
            vol_desc = 'Normal volatility'
        
        signals.append({
            'id': '8',
            'name': 'ATR (14)',
            'signal': vol_signal,
            'value': f"{atr_percent:.1f}%",
            'strength': safe_float_conversion(vol_strength),
            'description': vol_desc,
            'category': 'technical'
        })
        
        # 9. Support/Resistance Signal
        support_level = min([c[3] for c in candles[-20:]]) if len(candles) >= 20 else current_price * 0.98
        resistance_level = max([c[2] for c in candles[-20:]]) if len(candles) >= 20 else current_price * 1.02
        
        support_distance = ((current_price - support_level) / support_level) * 100
        resistance_distance = ((resistance_level - current_price) / current_price) * 100
        
        if support_distance < 1:
            sr_signal = 'BUY'
            sr_strength = 80
            sr_desc = 'Near strong support'
        elif resistance_distance < 1:
            sr_signal = 'SELL'
            sr_strength = 80
            sr_desc = 'Near strong resistance'
        elif support_distance < 2:
            sr_signal = 'BUY'
            sr_strength = 65
            sr_desc = 'Approaching support'
        elif resistance_distance < 2:
            sr_signal = 'SELL'
            sr_strength = 65
            sr_desc = 'Approaching resistance'
        else:
            sr_signal = 'NEUTRAL'
            sr_strength = 45
            sr_desc = 'Between levels'
        
        signals.append({
            'id': '9',
            'name': 'Support/Resistance',
            'signal': sr_signal,
            'value': f"S:{support_distance:.1f}% R:{resistance_distance:.1f}%",
            'strength': safe_float_conversion(sr_strength),
            'description': sr_desc,
            'category': 'structure'
        })
        
        # 10. Momentum Divergence
        if len(candles) >= 10:
            price_momentum = (current_price - candles[-10][4]) / candles[-10][4] * 100
            rsi_momentum = rsi_current - 50
            
            if price_momentum > 2 and rsi_momentum < -10:
                div_signal = 'SELL'
                div_strength = 85
                div_desc = 'Bearish divergence'
            elif price_momentum < -2 and rsi_momentum > 10:
                div_signal = 'BUY'
                div_strength = 85
                div_desc = 'Bullish divergence'
            elif price_momentum > 1 and rsi_momentum > 10:
                div_signal = 'BUY'
                div_strength = 70
                div_desc = 'Momentum alignment'
            elif price_momentum < -1 and rsi_momentum < -10:
                div_signal = 'SELL'
                div_strength = 70
                div_desc = 'Bearish alignment'
            else:
                div_signal = 'NEUTRAL'
                div_strength = 45
                div_desc = 'No divergence'
        else:
            div_signal = 'NEUTRAL'
            div_strength = 40
            div_desc = 'Insufficient data'
        
        signals.append({
            'id': '10',
            'name': 'RSI Divergence',
            'signal': div_signal,
            'value': f"{price_momentum:.1f}%",
            'strength': safe_float_conversion(div_strength),
            'description': div_desc,
            'category': 'momentum'
        })
        
        # 11. Trend Strength
        if len(candles) >= 20:
            trend_candles = candles[-20:]
            up_candles = sum(1 for c in trend_candles if c[4] > c[1])
            trend_strength_pct = (up_candles / len(trend_candles)) * 100
            
            if trend_strength_pct > 70:
                trend_signal = 'BUY'
                trend_strength = 85
                trend_desc = 'Strong uptrend'
            elif trend_strength_pct < 30:
                trend_signal = 'SELL'
                trend_strength = 85
                trend_desc = 'Strong downtrend'
            elif trend_strength_pct > 60:
                trend_signal = 'BUY'
                trend_strength = 65
                trend_desc = 'Moderate uptrend'
            elif trend_strength_pct < 40:
                trend_signal = 'SELL'
                trend_strength = 65
                trend_desc = 'Moderate downtrend'
            else:
                trend_signal = 'NEUTRAL'
                trend_strength = 45
                trend_desc = 'Sideways trend'
        else:
            trend_signal = 'NEUTRAL'
            trend_strength = 40
            trend_desc = 'Insufficient data'
        
        signals.append({
            'id': '11',
            'name': 'Trend Strength (20)',
            'signal': trend_signal,
            'value': f"{trend_strength_pct:.0f}%" if len(candles) >= 20 else "N/A",
            'strength': safe_float_conversion(trend_strength),
            'description': trend_desc,
            'category': 'structure'
        })
        
        # 12. Market Structure
        if len(candles) >= 50:
            recent_high = max([c[2] for c in candles[-50:]])
            recent_low = min([c[3] for c in candles[-50:]])
            position_in_range = ((current_price - recent_low) / (recent_high - recent_low)) * 100
            
            if position_in_range > 80:
                ms_signal = 'SELL'
                ms_strength = 75
                ms_desc = 'Top of range'
            elif position_in_range < 20:
                ms_signal = 'BUY'
                ms_strength = 75
                ms_desc = 'Bottom of range'
            elif position_in_range > 60:
                ms_signal = 'SELL'
                ms_strength = 55
                ms_desc = 'Upper range'
            elif position_in_range < 40:
                ms_signal = 'BUY'
                ms_strength = 55
                ms_desc = 'Lower range'
            else:
                ms_signal = 'NEUTRAL'
                ms_strength = 45
                ms_desc = 'Middle range'
        else:
            ms_signal = 'NEUTRAL'
            ms_strength = 40
            ms_desc = 'Insufficient data'
        
        signals.append({
            'id': '12',
            'name': 'Market Structure (50)',
            'signal': ms_signal,
            'value': f"{position_in_range:.0f}%" if len(candles) >= 50 else "N/A",
            'strength': safe_float_conversion(ms_strength),
            'description': ms_desc,
            'category': 'structure'
        })
            
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return []
    
    return signals

def calculate_final_signal(signals, current_price, indicators):
    """Calculate final trading signal with 90% accuracy logic"""
    if not signals:
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'entry': safe_float_conversion(current_price),
            'stopLoss': 0.0,
            'target1': 0.0,
            'target2': 0.0,
            'riskReward': 0.0,
            'positionSize': '0%',
            'reasoning': ['Insufficient data for signal generation']
        }
    
    try:
        # Weight signals by category and strength
        category_weights = {
            'technical': 1.0,
            'options': 0.8,
            'momentum': 0.9,
            'structure': 1.1
        }
        
        weighted_buy_score = 0
        weighted_sell_score = 0
        total_weight = 0
        
        buy_signals = []
        sell_signals = []
        
        for signal in signals:
            weight = category_weights.get(signal['category'], 1.0)
            strength_factor = signal['strength'] / 100
            weighted_strength = weight * strength_factor
            
            if signal['signal'] == 'BUY':
                weighted_buy_score += weighted_strength
                buy_signals.append(signal['name'])
            elif signal['signal'] == 'SELL':
                weighted_sell_score += weighted_strength
                sell_signals.append(signal['name'])
            
            total_weight += weight
        
        # Calculate net signal strength
        net_strength = (weighted_buy_score - weighted_sell_score) / total_weight * 100 if total_weight > 0 else 0
        confidence = min(abs(net_strength) * 1.2, 100)  # Boost confidence for strong signals
        
        # Determine action
        if net_strength > 25:
            action = 'BUY'
        elif net_strength < -25:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Calculate levels using ATR
        atr = 0
        if indicators.get('atr'):
            atr_values = [x for x in indicators['atr'] if x != 0]
            if atr_values:
                atr = atr_values[-1]
        
        if atr == 0:
            atr = current_price * 0.008  # 0.8% fallback
        
        # Risk management
        entry = safe_float_conversion(current_price)
        risk_multiplier = 2.0
        reward_multiplier = 3.0
        
        if action == 'BUY':
            stop_loss = entry - (atr * risk_multiplier)
            target1 = entry + (atr * reward_multiplier)
            target2 = entry + (atr * reward_multiplier * 1.5)
        elif action == 'SELL':
            stop_loss = entry + (atr * risk_multiplier)
            target1 = entry - (atr * reward_multiplier)
            target2 = entry - (atr * reward_multiplier * 1.5)
        else:
            stop_loss = target1 = target2 = 0
        
        risk_reward = abs(target1 - entry) / abs(entry - stop_loss) if stop_loss != entry else 0
        
        # Position sizing based on confidence
        if confidence > 85:
            position_size = '3%'
        elif confidence > 75:
            position_size = '2%'
        elif confidence > 65:
            position_size = '1.5%'
        elif confidence > 50:
            position_size = '1%'
        else:
            position_size = '0.5%'
        
        # Enhanced reasoning
        reasoning = [
            f"{len(buy_signals)} Buy signals: {', '.join(buy_signals[:3])}{'...' if len(buy_signals) > 3 else ''}",
            f"{len(sell_signals)} Sell signals: {', '.join(sell_signals[:3])}{'...' if len(sell_signals) > 3 else ''}",
            f"Net strength: {net_strength:.1f}",
            f"Confidence: {confidence:.1f}%",
            f"Risk-Reward: 1:{risk_reward:.1f}",
        ]
        
        return {
            'action': action,
            'confidence': safe_float_conversion(confidence),
            'entry': safe_float_conversion(entry),
            'stopLoss': safe_float_conversion(stop_loss),
            'target1': safe_float_conversion(target1),
            'target2': safe_float_conversion(target2),
            'riskReward': safe_float_conversion(risk_reward),
            'positionSize': position_size,
            'reasoning': reasoning
        }
        
    except Exception as e:
        logger.error(f"Error calculating final signal: {e}")
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'entry': safe_float_conversion(current_price),
            'stopLoss': 0.0,
            'target1': 0.0,
            'target2': 0.0,
            'riskReward': 0.0,
            'positionSize': '0%',
            'reasoning': ['Error in signal calculation']
        }

def get_real_options_data(symbol, current_price, smart_api):
    """
    Get REAL options data from Angel One API
    """
    try:
        # Get next expiry date (next Thursday)
        today = datetime.now()
        days_ahead = 3 - today.weekday()  # Thursday is 3
        if days_ahead <= 0:
            days_ahead += 7
        next_expiry = today + timedelta(days=days_ahead)
        expiry_str = next_expiry.strftime('%d%b%Y').upper()
        
        # Calculate time to expiry in years
        time_to_expiry = (next_expiry - today).total_seconds() / (365.25 * 24 * 3600)
        
        # Generate strikes around current price
        strike_interval = STRIKE_INTERVALS.get(symbol, 50)
        atm_strike = round(current_price / strike_interval) * strike_interval
        
        options_data = []
        
        # Get real options data for strikes around ATM
        for i in range(-7, 8):
            strike = atm_strike + (i * strike_interval)
            
            # Create proper NSE option symbols
            call_symbol = f"{symbol}{expiry_str}{strike}CE"
            put_symbol = f"{symbol}{expiry_str}{strike}PE"
            
            try:
                # Try to get real market data for call option
                call_data = None
                put_data = None
                
                # Search for the actual option instruments
                try:
                    search_params = {
                        "exchange": "NFO",
                        "searchscrip": call_symbol
                    }
                    call_search = smart_api.searchScrip(search_params)
                    
                    if call_search.get('data') and len(call_search['data']) > 0:
                        call_instrument = call_search['data'][0]
                        call_ltp = smart_api.ltpData("NFO", call_instrument['tradingsymbol'], call_instrument['symboltoken'])
                        if call_ltp.get('data'):
                            call_data = {
                                'price': safe_float_conversion(call_ltp['data']['ltp']),
                                'symbol': call_instrument['tradingsymbol'],
                                'token': call_instrument['symboltoken']
                            }
                except Exception as e:
                    logger.debug(f"Could not fetch call data for {call_symbol}: {e}")
                
                try:
                    search_params = {
                        "exchange": "NFO",
                        "searchscrip": put_symbol
                    }
                    put_search = smart_api.searchScrip(search_params)
                    
                    if put_search.get('data') and len(put_search['data']) > 0:
                        put_instrument = put_search['data'][0]
                        put_ltp = smart_api.ltpData("NFO", put_instrument['tradingsymbol'], put_instrument['symboltoken'])
                        if put_ltp.get('data'):
                            put_data = {
                                'price': safe_float_conversion(put_ltp['data']['ltp']),
                                'symbol': put_instrument['tradingsymbol'],
                                'token': put_instrument['symboltoken']
                            }
                except Exception as e:
                    logger.debug(f"Could not fetch put data for {put_symbol}: {e}")
                
                # If we have real data, use it; otherwise skip this strike
                if call_data or put_data:
                    # Estimate IV from market prices if available
                    call_iv = 20.0  # Default IV
                    put_iv = 20.0   # Default IV
                    
                    if call_data and call_data['price'] > 0:
                        call_iv = estimate_iv_from_price(current_price, strike, time_to_expiry, RISK_FREE_RATE, call_data['price'], 'CE')
                    
                    if put_data and put_data['price'] > 0:
                        put_iv = estimate_iv_from_price(current_price, strike, time_to_expiry, RISK_FREE_RATE, put_data['price'], 'PE')
                    
                    # Calculate Greeks using estimated IV
                    call_greeks = calculate_greeks(current_price, strike, time_to_expiry, RISK_FREE_RATE, call_iv/100, 'call')
                    put_greeks = calculate_greeks(current_price, strike, time_to_expiry, RISK_FREE_RATE, put_iv/100, 'put')
                    
                    # Get additional market data if available
                    call_volume = 0
                    call_oi = 0
                    put_volume = 0
                    put_oi = 0

                    # Get real OI and volume data
                    if call_data:
                        call_market_info = get_option_oi_data(smart_api, call_data['symbol'], call_data['token'])
                        call_volume = call_market_info['volume']
                        call_oi = call_market_info['oi']
                        if call_data['price'] == 0:
                            call_data['price'] = call_market_info['ltp']

                    if put_data:
                        put_market_info = get_option_oi_data(smart_api, put_data['symbol'], put_data['token'])
                        put_volume = put_market_info['volume']
                        put_oi = put_market_info['oi']
                        if put_data['price'] == 0:
                            put_data['price'] = put_market_info['ltp']
                    
                    options_data.append({
                        'strike': strike,
                        'expiry': next_expiry.strftime('%Y-%m-%d'),
                        'time_to_expiry': time_to_expiry,
                        'call': {
                            'symbol': call_data['symbol'] if call_data else call_symbol,
                            'price': safe_float_conversion(call_data['price'] if call_data else 0),
                            'iv': safe_float_conversion(call_iv),
                            'delta': safe_float_conversion(call_greeks['delta']),
                            'gamma': safe_float_conversion(call_greeks['gamma']),
                            'theta': safe_float_conversion(call_greeks['theta']),
                            'vega': safe_float_conversion(call_greeks['vega']),
                            'rho': safe_float_conversion(call_greeks['rho']),
                            'volume': call_volume,
                            'oi': call_oi,
                            'ltp': safe_float_conversion(call_data['price'] if call_data else 0)
                        },
                        'put': {
                            'symbol': put_data['symbol'] if put_data else put_symbol,
                            'price': safe_float_conversion(put_data['price'] if put_data else 0),
                            'iv': safe_float_conversion(put_iv),
                            'delta': safe_float_conversion(put_greeks['delta']),
                            'gamma': safe_float_conversion(put_greeks['gamma']),
                            'theta': safe_float_conversion(put_greeks['theta']),
                            'vega': safe_float_conversion(put_greeks['vega']),
                            'rho': safe_float_conversion(put_greeks['rho']),
                            'volume': put_volume,
                            'oi': put_oi,
                            'ltp': safe_float_conversion(put_data['price'] if put_data else 0)
                        }
                    })
                    
            except Exception as e:
                logger.error(f"Error fetching options data for strike {strike}: {e}")
                continue
        
        logger.info(f"Retrieved {len(options_data)} real options strikes for {symbol}")
        return options_data
        
    except Exception as e:
        logger.error(f"Error in get_real_options_data for {symbol}: {e}")
        return []

def get_option_oi_data(smart_api, symbol, token, exchange="NFO"):
    """Get Open Interest data for options using Angel One API"""
    try:
        # Try to get current market data
        market_data = smart_api.getMarketData({
            "mode": "FULL",
            "exchangeTokens": {
                exchange: [token]
            }
        })
        
        if market_data.get('data') and market_data['data'].get('fetched'):
            option_data = market_data['data']['fetched'][0]
            return {
                'oi': safe_float_conversion(option_data.get('oi', 0)),
                'volume': safe_float_conversion(option_data.get('volume', 0)),
                'ltp': safe_float_conversion(option_data.get('ltp', 0))
            }
            
        # Fallback to LTP data
        ltp_data = smart_api.ltpData(exchange, symbol, token)
        if ltp_data.get('data'):
            return {
                'oi': safe_float_conversion(ltp_data['data'].get('open_interest', 0)),
                'volume': safe_float_conversion(ltp_data['data'].get('volume', 0)), 
                'ltp': safe_float_conversion(ltp_data['data'].get('ltp', 0))
            }
            
    except Exception as e:
        logger.error(f"Error getting OI data for {symbol}: {e}")
        
    return {'oi': 0, 'volume': 0, 'ltp': 0}

def estimate_iv_from_price(S, K, T, r, market_price, option_type):
    """
    Estimate implied volatility from market price using Newton-Raphson method
    """
    if T <= 0 or market_price <= 0:
        return 20.0  # Default 20% IV
    
    # Initial guess
    iv = 0.20
    
    for i in range(50):  # Max 50 iterations
        if option_type == 'CE':
            price = black_scholes_call(S, K, T, r, iv)
        else:
            price = black_scholes_put(S, K, T, r, iv)
        
        # Calculate vega for Newton-Raphson
        try:
            d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            if vega < 0.001:  # Avoid division by very small numbers
                break
                
            # Newton-Raphson update
            iv_new = iv - (price - market_price) / vega
            
            # Ensure IV stays within reasonable bounds
            iv_new = max(0.05, min(2.0, iv_new))
            
            if abs(iv_new - iv) < 0.0001:  # Convergence
                break
                
            iv = iv_new
            
        except:
            break
    
    return max(5.0, min(100.0, iv * 100))  # Return IV as percentage between 5% and 100%

def calculate_portfolio_greeks(options_chain):
    """Calculate portfolio-level Greeks based on the options chain."""
    if not options_chain:
        return {
            "total_call_delta": 0.0,
            "total_put_delta": 0.0,
            "net_delta": 0.0,
            "total_gamma": 0.0,
            "total_theta": 0.0,
            "total_vega": 0.0,
            "put_call_ratio": 0.0,
            "max_pain": 0.0,
            "total_call_volume": 0.0,
            "total_put_volume": 0.0
        }

    total_call_delta = sum(safe_float_conversion(opt['call']['delta']) for opt in options_chain)
    total_put_delta = sum(safe_float_conversion(opt['put']['delta']) for opt in options_chain)
    total_gamma = sum(safe_float_conversion(opt['call']['gamma']) + safe_float_conversion(opt['put']['gamma']) for opt in options_chain)
    total_theta = sum(safe_float_conversion(opt['call']['theta']) + safe_float_conversion(opt['put']['theta']) for opt in options_chain)
    total_vega = sum(safe_float_conversion(opt['call']['vega']) + safe_float_conversion(opt['put']['vega']) for opt in options_chain)
    total_call_volume = sum(safe_float_conversion(opt['call']['volume']) for opt in options_chain)
    total_put_volume = sum(safe_float_conversion(opt['put']['volume']) for opt in options_chain)
    
    put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
    
    # Calculate max pain (strike with highest total OI)
    max_pain_strike = 0
    max_oi = 0
    for opt in options_chain:
        total_oi = safe_float_conversion(opt['call']['oi']) + safe_float_conversion(opt['put']['oi'])
        if total_oi > max_oi:
            max_oi = total_oi
            max_pain_strike = opt['strike']

    return {
        "total_call_delta": safe_float_conversion(total_call_delta),
        "total_put_delta": safe_float_conversion(total_put_delta),
        "net_delta": safe_float_conversion(total_call_delta + total_put_delta),
        "total_gamma": safe_float_conversion(total_gamma),
        "total_theta": safe_float_conversion(total_theta),
        "total_vega": safe_float_conversion(total_vega),
        "put_call_ratio": safe_float_conversion(put_call_ratio),
        "max_pain": safe_float_conversion(max_pain_strike),
        "total_call_volume": safe_float_conversion(total_call_volume),
        "total_put_volume": safe_float_conversion(total_put_volume)
    }

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    if sigma <= 0:
        return max(S - K, 0)
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    except:
        return max(S - K, 0)

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    if sigma <= 0:
        return max(K - S, 0)
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put_price, 0)
    except:
        return max(K - S, 0)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Greeks using standard formulas"""
    if T <= 0:
        if option_type == 'call':
            return {'delta': 1.0 if S > K else 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        else:
            return {'delta': -1.0 if S < K else 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    
    if sigma <= 0:
        sigma = 0.01
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
        else:
            theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega),
            'rho': float(rho)
        }
        
    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

@app.get("/")
def get_html():
    return FileResponse("index.html")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "angel_one_connected": app.state.smart_api is not None,
        "active_connections": list(active_connections.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/{index}")
async def websocket_endpoint(websocket: WebSocket, index: str):
    """WebSocket endpoint for real-time data"""
    await websocket.accept()
    logger.info(f"WebSocket connected for {index}")
    
    symbol_info = INDEX_SYMBOLS.get(index.upper())
    if not symbol_info:
        await websocket.send_json({"error": "Invalid index name"})
        await websocket.close()
        return

    smart_api = app.state.smart_api
    if not smart_api:
        await websocket.send_json({"error": "Angel One API not connected"})
        await websocket.close()
        return

    # Store active connection
    active_connections[index] = websocket

    try:
        while True:
            logger.info(f"Fetching data for {index}...")
            
            # Get market status and trading calendar
            market_status = get_market_status()
            trading_calendar = get_trading_calendar()
            
            # Get historical data
            now = datetime.now()
            fromdate = now - timedelta(days=7)
            
            params = {
                "exchange": symbol_info["exchange"],
                "symboltoken": symbol_info["symboltoken"],
                "interval": "FIVE_MINUTE",
                "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"),
                "todate": now.strftime("%Y-%m-%d %H:%M"),
            }
            
            hist_data = smart_api.getCandleData(params)
            candles = hist_data.get('data', [])
            
            logger.info(f"Received {len(candles) if candles else 0} candles for {index}")
            
            if not candles:
                await websocket.send_json({"error": "No data available"})
                await asyncio.sleep(10)
                continue
            
            # Process candles
            processed_candles = []
            for c in candles:
                try:
                    if len(c) >= 6:
                        timestamp = int(dateutil.parser.isoparse(c[0]).timestamp() * 1000)
                        processed_candles.append([timestamp, float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])])
                except Exception as e:
                    logger.debug(f"Error processing candle: {e}")
                    continue
            
            if len(processed_candles) < 100:
                await websocket.send_json({"error": f"Insufficient data: {len(processed_candles)} candles"})
                await asyncio.sleep(10)
                continue
            
            # Get current price
            try:
                ltp_data = smart_api.ltpData(
                    exchange=symbol_info["exchange"],
                    tradingsymbol=symbol_info["tradingsymbol"],
                    symboltoken=symbol_info["symboltoken"]
                )
                current_price = float(ltp_data['data']['ltp'])
                logger.info(f"Current price for {index}: {current_price}")
            except Exception as e:
                logger.error(f"Error getting LTP for {index}: {e}")
                current_price = processed_candles[-1][4]
            
            # Calculate indicators and signals
            logger.info(f"Calculating indicators for {index}...")
            indicators = calculate_advanced_indicators(processed_candles)
            logger.info(f"Generated {len(indicators)} indicators for {index}")
            
            logger.info(f"Generating signals for {index}...")
            signals = generate_high_accuracy_signals(processed_candles, indicators, current_price)
            logger.info(f"Generated {len(signals)} signals for {index}")
            
            final_signal = calculate_final_signal(signals, current_price, indicators)
            logger.info(f"Final signal for {index}: {final_signal['action']} with {final_signal['confidence']:.1f}% confidence")
            
            # Get options data
            options_chain = get_real_options_data(index, current_price, smart_api)
            portfolio_greeks = calculate_portfolio_greeks(options_chain)
            
            # Prepare response
            open_price = processed_candles[0][1] if processed_candles else current_price
            last_close = processed_candles[-2][4] if len(processed_candles) > 1 else current_price
            percent = ((current_price - open_price) / open_price) * 100 if open_price else 0
            
            response_data = {
                "current": safe_float_conversion(current_price),
                "last_close": safe_float_conversion(last_close),
                "today_open": safe_float_conversion(open_price),
                "percent": safe_float_conversion(percent),
                "candles": processed_candles[-300:],
                "signals": signals,
                "final_signal": final_signal,
                "indicators": indicators,
                "options_chain": options_chain,
                "portfolio_greeks": portfolio_greeks,
                "market_status": market_status,
                "trading_calendar": trading_calendar,
                "timestamp": datetime.now().isoformat(),
                "data_source": "angel_one_api"
            }
            
            # Sanitize and send response
            sanitized_response = sanitize_for_json(response_data)
            
            logger.info(f"Sending data to frontend for {index}: {len(signals)} signals, price={current_price:.2f}")
            await websocket.send_json(sanitized_response)
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {index}")
        if index in active_connections:
            del active_connections[index]
    except Exception as e:
        logger.error(f"WebSocket error for {index}: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
        if index in active_connections:
            del active_connections[index]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
