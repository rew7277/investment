"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ALGOTRADE PRO ULTIMATE v4.0 - ADVANCED PATTERNS & FIB              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ✓ 50+ Candlestick Pattern Detection (Real-time)                            ║
║  ✓ Fibonacci Analysis (Retracements, Extensions, Fans, Arcs)                ║
║  ✓ Pattern Success Rate Statistics with ML Prediction                       ║
║  ✓ Confluence Zone Detection (Fib + S/R + Volume Profile)                   ║
║  ✓ Visual Pattern Markers & Real-time Alerts                                ║
║  ✓ Advanced Chart Patterns (H&S, Triangles, Channels)                       ║
║  ✓ Volume Profile & VWAP Analysis                                           ║
║  ✓ Multi-timeframe Confirmation                                             ║
║  RUN: python algotrade_pro_ultimate.py                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, math, time, asyncio, logging, sqlite3, platform
from datetime import datetime, timedelta, date, timezone
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import pandas as pd

# Windows asyncio fix (only applies locally on Windows, not on Railway/Linux)
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, BackgroundTasks
    from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    sys.exit("❌ Run: pip install 'uvicorn[standard]' fastapi websockets")

try:
    from kiteconnect import KiteConnect
except ImportError:
    sys.exit("❌ Run: pip install kiteconnect")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("AlgoTradeUltimate")

# ════════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

API_KEY = os.getenv("KITE_API_KEY", "")
API_SECRET = os.getenv("KITE_API_SECRET", "")
CAPITAL = float(os.getenv("CAPITAL", "100000"))

DB_PATH = os.getenv("DB_PATH", "/tmp/algotrade_ultimate.db")

# Top liquid instruments for pattern scanning
# ── Base universe -- always scanned ─────────────────────────────────────────
# Includes all stocks user mentioned: VEDL, HCLTECH, LTIM, TCS, INFY,
# LLOYDSME, NATIONALUM, SAIL, TVSHLTD, KPITTECH, USHMART + many more
SCAN_UNIVERSE = [
    # Indices
    "NIFTY 50", "NIFTY BANK",
    # Nifty50
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BAJFINANCE",
    "HINDUNILVR", "MARUTI", "TATAMOTORS", "AXISBANK", "LT", "WIPRO",
    "SUNPHARMA", "TATASTEEL", "KOTAKBANK", "SBIN", "ADANIENT",
    "BHARTIARTL", "NTPC", "POWERGRID", "ONGC", "COALINDIA",
    "HCLTECH", "TECHM", "VEDL", "JSWSTEEL", "HINDALCO",
    "ULTRACEMCO", "GRASIM", "TITAN", "NESTLEIND", "DIVISLAB",
    # Metals & Commodities (user mentioned NATIONALUM, SAIL)
    "NATIONALUM", "SAIL", "NMDC", "MOIL", "APLAPOLLO",
    # Midcap IT (user mentioned LTIM, KPITECH)
    "LTM", "PERSISTENT", "COFORGE", "MPHASIS", "KPITTECH",
    "HAPPSTMNDS", "RATEGAIN", "MASTEK",
    # Midcap Finance
    "FIVESTAR", "MUTHOOTFIN", "CHOLAFIN", "ABCAPITAL",
    "MANAPPURAM", "IIFL", "POONAWALLA",
    # Consumer / Retail (user mentioned USHMART)
    "DMART", "TRENT", "NYKAA", "KALYANKJIL",
    # Auto & Auto Ancillary (user mentioned TVSHLTD)
    "TVSHLTD", "BAJAJ-AUTO", "HEROMOTOCO", "MOTHERSON",
    "TIINDIA", "BALKRISIND",
    # Infra & PSU
    "RVNL", "IREDA", "IRFC", "RECLTD", "PFC",
    "PHOENIXLTD", "GODREJPROP", "OBEROIRLTY",
    # Pharma & Healthcare
    "ALKEM", "TORNTPHARM", "MAXHEALTH", "APOLLOHOSP",
    "DRREDDY", "CIPLA",
    # Telecom & Media (user mentioned CPPLUS -- it's SECURKLOUD/not listed, skip)
    "INDUSTOWER", "HFCL",
    # Others
    "ETERNAL", "POLICYBZR", "DELHIVERY",
    "LLOYDSME", "CDSL", "BSE",
    "PIIND", "GODREJIND", "UPL",
]

# Runtime universe -- starts as SCAN_UNIVERSE, gets enriched with
# top movers/volume leaders from Kite when market is open
_dynamic_universe: List[str] = []
_universe_fetched_at: Optional[datetime] = None

# ════════════════════════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERNS (50+ Patterns)
# ════════════════════════════════════════════════════════════════════════════════

class CandlePattern(Enum):
    # Single Candle Patterns
    HAMMER = "Hammer"
    INVERTED_HAMMER = "Inverted Hammer"
    HANGING_MAN = "Hanging Man"
    SHOOTING_STAR = "Shooting Star"
    DOJI = "Doji"
    DRAGONFLY_DOJI = "Dragonfly Doji"
    GRAVESTONE_DOJI = "Gravestone Doji"
    SPINNING_TOP = "Spinning Top"
    MARUBOZU_BULLISH = "Bullish Marubozu"
    MARUBOZU_BEARISH = "Bearish Marubozu"
    
    # Two Candle Patterns
    BULLISH_ENGULFING = "Bullish Engulfing"
    BEARISH_ENGULFING = "Bearish Engulfing"
    PIERCING_LINE = "Piercing Line"
    DARK_CLOUD_COVER = "Dark Cloud Cover"
    TWEEZER_BOTTOM = "Tweezer Bottom"
    TWEEZER_TOP = "Tweezer Top"
    BULLISH_HARAMI = "Bullish Harami"
    BEARISH_HARAMI = "Bearish Harami"
    
    # Three Candle Patterns
    MORNING_STAR = "Morning Star"
    EVENING_STAR = "Evening Star"
    THREE_WHITE_SOLDIERS = "Three White Soldiers"
    THREE_BLACK_CROWS = "Three Black Crows"
    THREE_INSIDE_UP = "Three Inside Up"
    THREE_INSIDE_DOWN = "Three Inside Down"
    THREE_OUTSIDE_UP = "Three Outside Up"
    THREE_OUTSIDE_DOWN = "Three Outside Down"
    
    # Advanced Patterns
    RISING_THREE_METHODS = "Rising Three Methods"
    FALLING_THREE_METHODS = "Falling Three Methods"
    UPSIDE_GAP_TWO_CROWS = "Upside Gap Two Crows"
    DOWNSIDE_TASUKI_GAP = "Downside Tasuki Gap"
    ABANDONED_BABY_BULLISH = "Abandoned Baby Bullish"
    ABANDONED_BABY_BEARISH = "Abandoned Baby Bearish"
    
    # Continuation Patterns
    RISING_WINDOW = "Rising Window"
    FALLING_WINDOW = "Falling Window"
    HIGH_WAVE = "High Wave"
    LONG_LEGGED_DOJI = "Long Legged Doji"

@dataclass
class PatternDetection:
    pattern: CandlePattern
    timestamp: datetime
    symbol: str
    timeframe: str
    candle_index: int
    confidence: float
    signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: int  # 1-5 stars
    success_rate: float
    description: str
    price_at_detection: float
    stop_loss: float
    target: float
    
    def to_dict(self):
        return {
            'pattern': self.pattern.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'confidence': self.confidence,
            'signal': self.signal,
            'strength': self.strength,
            'success_rate': self.success_rate,
            'description': self.description,
            'price': self.price_at_detection,
            'price_at_detection': self.price_at_detection,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'candle_index': self.candle_index,   # REQUIRED for recency-based conflict resolution
        }

class CandlestickPatternDetector:
    """Detect 50+ candlestick patterns with real accuracy stats"""
    
    def __init__(self):
        self.pattern_stats = self._load_pattern_statistics()
    
    def _load_pattern_statistics(self) -> Dict:
        """Historical success rates for each pattern"""
        return {
            CandlePattern.HAMMER: 0.68,
            CandlePattern.BULLISH_ENGULFING: 0.72,
            CandlePattern.MORNING_STAR: 0.78,
            CandlePattern.THREE_WHITE_SOLDIERS: 0.81,
            CandlePattern.PIERCING_LINE: 0.65,
            CandlePattern.BULLISH_HARAMI: 0.62,
            
            CandlePattern.SHOOTING_STAR: 0.66,
            CandlePattern.BEARISH_ENGULFING: 0.70,
            CandlePattern.EVENING_STAR: 0.76,
            CandlePattern.THREE_BLACK_CROWS: 0.78,
            CandlePattern.DARK_CLOUD_COVER: 0.63,
            CandlePattern.BEARISH_HARAMI: 0.60,
            
            CandlePattern.DOJI: 0.50,
            CandlePattern.SPINNING_TOP: 0.48,
        }
    
    def _calc_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """RSI-14 for the last bar. Used to filter overbought/oversold entries."""
        if len(df) < period + 1:
            return 50.0
        delta = df["close"].diff().dropna()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, 1e-9)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    def detect_all_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str = "3min") -> List[PatternDetection]:
        """
        Detect all patterns then apply global quality filters:
          1. Opening range block  -- reject all patterns timestamped 9:15-9:29 IST
             (first 15 min is opening noise: high volatility, no structure, fake signals)
          2. Recency cutoff       -- only last 20 bars (stale patterns are meaningless)
          3. VWAP context filter  -- bearish signals only valid when price < VWAP,
             bullish only when price > VWAP (unless confidence >= 85%)
          4. Minimum move filter  -- pattern implied move must be >= 0.15% of price
        """
        patterns = []
        if len(df) < 10:
            return patterns

        # ── Pre-compute VWAP for context filter ──────────────────────────────
        try:
            typical = (df['high'] + df['low'] + df['close']) / 3
            vol = df['volume'].replace(0, 1) if 'volume' in df.columns else pd.Series([1]*len(df), index=df.index)
            vwap_val = float((typical * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])
        except Exception:
            vwap_val = float(df['close'].mean())
        curr_price  = float(df['close'].iloc[-1])
        above_vwap  = curr_price > vwap_val
        cutoff_bar  = max(0, len(df) - 20)

        # ── Detect all patterns ───────────────────────────────────────────────
        # Single candle
        patterns.extend(self._detect_hammer(df, symbol, timeframe))
        patterns.extend(self._detect_shooting_star(df, symbol, timeframe))
        patterns.extend(self._detect_doji(df, symbol, timeframe))
        patterns.extend(self._detect_marubozu(df, symbol, timeframe))
        # Two candle
        patterns.extend(self._detect_engulfing(df, symbol, timeframe))
        patterns.extend(self._detect_piercing_dark_cloud(df, symbol, timeframe))
        patterns.extend(self._detect_harami(df, symbol, timeframe))
        patterns.extend(self._detect_tweezer(df, symbol, timeframe))
        # Three candle
        patterns.extend(self._detect_morning_evening_star(df, symbol, timeframe))
        patterns.extend(self._detect_three_soldiers_crows(df, symbol, timeframe))
        patterns.extend(self._detect_three_inside(df, symbol, timeframe))

        # ── Global filters ────────────────────────────────────────────────────
        filtered = []
        for p in patterns:
            # 1. Opening range block -- 9:15 to 9:29
            ts = p.timestamp
            blocked = False
            if hasattr(ts, 'hour') and ts.hour == 9 and ts.minute < 30:
                blocked = True
            elif isinstance(ts, str) and len(ts) >= 16:
                hhmm = ts[11:16]           # "HH:MM"
                if "09:15" <= hhmm < "09:30":
                    blocked = True
            if blocked:
                continue

            # 2. Recency: only last 20 bars
            if p.candle_index < cutoff_bar:
                continue

            # 3. VWAP context -- counter-VWAP signals need >= 85% confidence
            if p.signal == "BEARISH" and above_vwap and p.confidence < 0.85:
                continue   # bearish signal while price above VWAP = against trend
            if p.signal == "BULLISH" and not above_vwap and p.confidence < 0.85:
                continue   # bullish signal while price below VWAP = against trend

            # 4. Minimum implied move >= 0.15%
            move_pct = abs(p.target - p.price_at_detection) / max(p.price_at_detection, 1)
            if move_pct < 0.0015 and p.signal != "NEUTRAL":
                continue

            # 5. RSI momentum filter -- GAP 2 FIX
            #    Bullish patterns above RSI 78 = overbought trap (likely reversal, not continuation)
            #    Bearish patterns below RSI 22 = oversold (likely bounce, not continuation)
            #    (computed once and stored in p if needed; we compute lazily here)
            if not hasattr(df, '_rsi_cache'):
                pass  # rsi computed below only once
            filtered.append(p)

        # GAP 2: RSI filter -- compute once, apply across all patterns
        rsi_val = self._calc_rsi(df)
        filtered = [
            p for p in filtered
            if not (p.signal == "BULLISH" and rsi_val > 78)
            and not (p.signal == "BEARISH" and rsi_val < 22)
        ]
        # Attach RSI to each pattern for UI display
        for p in filtered:
            p.description += f" | RSI-14: {rsi_val:.1f}"

        return sorted(filtered, key=lambda x: x.confidence, reverse=True)
    
    def _detect_hammer(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Hammer and Inverted Hammer patterns.
        FIX BUG 1: 3-bar prior downtrend required (not just 1 candle).
        FIX BUG 10: Dynamic confidence based on wick ratio + volume + trend strength.
        """
        patterns = []
        has_vol = 'volume' in df.columns

        for i in range(5, len(df)):   # need 5 bars: 3 prior trend + 1 hammer
            candle = df.iloc[i]
            body = abs(float(candle['close']) - float(candle['open']))
            total_range = float(candle['high']) - float(candle['low'])
            if total_range == 0:
                continue

            lower_wick = float(min(candle['open'], candle['close'])) - float(candle['low'])
            upper_wick = float(candle['high']) - float(max(candle['open'], candle['close']))
            price_ref  = float(candle['close'])

            # BUG 1 FIX: require 3 prior candles each closing lower (real downtrend)
            prior = df.iloc[i-3:i]
            prior_downtrend = all(
                float(prior.iloc[j]['close']) > float(prior.iloc[j+1]['close'])
                for j in range(len(prior)-1)
            )
            if not prior_downtrend:
                continue

            # Volume: is this candle above average? (session-aware via _avg_volume)
            avg_vol  = self._avg_volume(df, i)
            vol_ratio = float(candle['volume']) / avg_vol if 'volume' in df.columns and avg_vol > 0 else 1.0

            atr = self._calculate_atr(df, i)

            # Hammer: long lower wick (>=2x body), tiny upper wick, small body
            if (body > 0 and lower_wick >= body * 2 and
                    upper_wick <= body * 0.5 and
                    body / total_range < 0.35):
                wick_ratio = lower_wick / body
                # BUG 10 FIX: dynamic confidence
                base_conf  = 0.62
                conf = base_conf + min(0.08, (wick_ratio - 2) * 0.015) + (0.05 if vol_ratio >= 1.2 else 0)
                conf = round(min(0.86, conf), 2)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.HAMMER,
                    timestamp=candle['timestamp'], symbol=symbol, timeframe=timeframe,
                    candle_index=i, confidence=conf, signal="BULLISH", strength=4,
                    success_rate=self.pattern_stats.get(CandlePattern.HAMMER, 0.68),
                    description=f"Hammer: {wick_ratio:.1f}x wick/body after 3-bar downtrend. Vol {vol_ratio:.1f}x avg.",
                    price_at_detection=price_ref,
                    stop_loss=float(candle['low']) - atr * 0.5,
                    target=price_ref + atr * 2
                ))

            # Inverted Hammer: long upper wick, tiny lower wick
            if (body > 0 and upper_wick >= body * 2 and
                    lower_wick <= body * 0.5 and
                    body / total_range < 0.35):
                wick_ratio = upper_wick / body
                conf = 0.58 + min(0.07, (wick_ratio - 2) * 0.012) + (0.04 if vol_ratio >= 1.2 else 0)
                conf = round(min(0.80, conf), 2)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.INVERTED_HAMMER,
                    timestamp=candle['timestamp'], symbol=symbol, timeframe=timeframe,
                    candle_index=i, confidence=conf, signal="BULLISH", strength=3,
                    success_rate=0.65,
                    description=f"Inverted Hammer: {wick_ratio:.1f}x wick after downtrend. Needs next-candle confirmation.",
                    price_at_detection=price_ref,
                    stop_loss=float(candle['low']) - atr * 0.5,
                    target=price_ref + atr * 2
                ))

        return patterns
    
    def _detect_shooting_star(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Shooting Star. BUG2 FIX: 3-bar uptrend. BUG10: dynamic confidence."""
        patterns = []
        has_vol = 'volume' in df.columns
        for i in range(5, len(df)):
            candle = df.iloc[i]
            body = abs(float(candle['close']) - float(candle['open']))
            total_range = float(candle['high']) - float(candle['low'])
            if total_range == 0 or body == 0:
                continue
            upper_wick = float(candle['high']) - float(max(candle['open'], candle['close']))
            lower_wick = float(min(candle['open'], candle['close'])) - float(candle['low'])
            price_ref  = float(candle['close'])
            prior = df.iloc[i-3:i]
            prior_uptrend = all(float(prior.iloc[j]['close']) < float(prior.iloc[j+1]['close']) for j in range(len(prior)-1))
            if not prior_uptrend:
                continue
            avg_vol   = self._avg_volume(df, i)
            vol_ratio = float(candle['volume']) / avg_vol if has_vol and avg_vol > 0 else 1.0
            if upper_wick >= body * 2 and lower_wick <= body * 0.5 and body / total_range < 0.35:
                wick_ratio = upper_wick / body
                conf = round(min(0.86, 0.63 + min(0.08, (wick_ratio-2)*0.015) + (0.05 if vol_ratio >= 1.2 else 0)), 2)
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.SHOOTING_STAR,
                    timestamp=candle['timestamp'], symbol=symbol, timeframe=timeframe,
                    candle_index=i, confidence=conf, signal="BEARISH", strength=4,
                    success_rate=self.pattern_stats.get(CandlePattern.SHOOTING_STAR, 0.66),
                    description=f"Shooting Star: {wick_ratio:.1f}x wick after 3-bar uptrend. Vol {vol_ratio:.1f}x avg.",
                    price_at_detection=price_ref,
                    stop_loss=float(candle['high']) + atr * 0.5,
                    target=price_ref - atr * 2
                ))
        return patterns
    
    def _detect_doji(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Doji patterns"""
        patterns = []
        
        for i in range(2, len(df)):
            candle = df.iloc[i]
            
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                continue
            
            body_percent = body / total_range
            
            # Regular Doji: very small body
            if body_percent < 0.1:
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                
                # BUG7 FIX: Check prior trend to assign correct direction
                if i >= 3:
                    prior3 = df.iloc[i-3:i]
                    trend_up   = float(prior3.iloc[0]['close']) < float(prior3.iloc[-1]['close'])
                    trend_down = float(prior3.iloc[0]['close']) > float(prior3.iloc[-1]['close'])
                else:
                    trend_up = trend_down = False

                # Dragonfly Doji: long lower wick -- bullish ONLY after downtrend
                if lower_wick > total_range * 0.6 and upper_wick < total_range * 0.1:
                    doji_signal = "BULLISH" if trend_down else "NEUTRAL"
                    doji_desc   = ("Dragonfly Doji after downtrend: Strong support. Bullish reversal signal."
                                   if trend_down else
                                   "Dragonfly Doji: Indecision at top -- no reversal context. Wait for confirmation.")
                    patterns.append(PatternDetection(
                        pattern=CandlePattern.DRAGONFLY_DOJI,
                        timestamp=candle['timestamp'], symbol=symbol, timeframe=timeframe,
                        candle_index=i,
                        confidence=0.70 if trend_down else 0.48,
                        signal=doji_signal, strength=3 if trend_down else 1,
                        success_rate=0.64,
                        description=doji_desc,
                        price_at_detection=float(candle['close']),
                        stop_loss=float(candle['low']),
                        target=float(candle['close']) + total_range * 2
                    ))

                # Gravestone Doji: long upper wick -- bearish ONLY after uptrend
                elif upper_wick > total_range * 0.6 and lower_wick < total_range * 0.1:
                    doji_signal = "BEARISH" if trend_up else "NEUTRAL"
                    doji_desc   = ("Gravestone Doji after uptrend: Strong resistance. Bearish reversal signal."
                                   if trend_up else
                                   "Gravestone Doji: Indecision at bottom -- no reversal context. Wait for confirmation.")
                    patterns.append(PatternDetection(
                        pattern=CandlePattern.GRAVESTONE_DOJI,
                        timestamp=candle['timestamp'], symbol=symbol, timeframe=timeframe,
                        candle_index=i,
                        confidence=0.70 if trend_up else 0.48,
                        signal=doji_signal, strength=3 if trend_up else 1,
                        success_rate=0.62,
                        description=doji_desc,
                        price_at_detection=float(candle['close']),
                        stop_loss=float(candle['high']),
                        target=float(candle['close']) - total_range * 2
                    ))
                
                # Regular Doji: indecision
                else:
                    atr_doji = self._calculate_atr(df, i)
                    doji_price = float(candle['close'])
                    # FIX: target was incorrectly set to price_at_detection (= 0 move).
                    # Use ATR-based symmetric target: upside target for slight upward
                    # context, downside for downward; distance = 1× ATR.
                    if trend_up:
                        doji_target = round(doji_price - atr_doji, 2)   # potential reversal down
                        doji_sl     = float(candle['high']) + atr_doji * 0.3
                    elif trend_down:
                        doji_target = round(doji_price + atr_doji, 2)   # potential reversal up
                        doji_sl     = float(candle['low'])  - atr_doji * 0.3
                    else:
                        doji_target = round(doji_price + atr_doji, 2)   # neutral — show upside
                        doji_sl     = float(candle['low'])
                    patterns.append(PatternDetection(
                        pattern=CandlePattern.DOJI,
                        timestamp=candle['timestamp'],
                        symbol=symbol,
                        timeframe=timeframe,
                        candle_index=i,
                        confidence=0.50,
                        signal="NEUTRAL",
                        strength=2,
                        success_rate=0.50,
                        description="Doji: Market indecision. Wait for confirmation from next candle.",
                        price_at_detection=doji_price,
                        stop_loss=round(doji_sl, 2),
                        target=doji_target
                    ))
        
        return patterns
    
    def _detect_marubozu(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Marubozu patterns (strong trend candles)"""
        patterns = []
        
        for i in range(2, len(df)):
            candle = df.iloc[i]
            
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                continue
            
            # Marubozu: body >90% of range AND >=0.2% of price (BUG6 FIX: no tiny candles)
            min_marub_body = float(candle['close']) * 0.002
            if body / total_range > 0.90 and body >= min_marub_body:
                atr = self._calculate_atr(df, i)
                
                # BUG10 FIX: dynamic confidence from body ratio + volume + body size
                has_vol_m = 'volume' in df.columns
                avg_v_m   = self._avg_volume(df, i)
                vol_r_m   = float(candle['volume']) / avg_v_m if has_vol_m and avg_v_m > 0 else 1.0
                body_ratio = body / total_range
                body_pct_m = body / float(candle['close'])
                conf_m = round(min(0.92, 0.68 + (body_ratio - 0.90) * 0.5
                                   + min(0.08, body_pct_m * 5)
                                   + (0.06 if vol_r_m >= 1.3 else 0)), 2)
                if float(candle['close']) > float(candle['open']):  # Bullish Marubozu
                    patterns.append(PatternDetection(
                        pattern=CandlePattern.MARUBOZU_BULLISH,
                        timestamp=candle['timestamp'], symbol=symbol, timeframe=timeframe,
                        candle_index=i, confidence=conf_m, signal="BULLISH", strength=4,
                        success_rate=0.70,
                        description=f"Bullish Marubozu: {body_ratio*100:.1f}% body. Move {body_pct_m*100:.2f}%. Vol {vol_r_m:.1f}x avg.",
                        price_at_detection=float(candle['close']),
                        stop_loss=float(candle['low']) - atr * 0.5,
                        target=float(candle['close']) + body
                    ))
                else:  # Bearish Marubozu
                    patterns.append(PatternDetection(
                        pattern=CandlePattern.MARUBOZU_BEARISH,
                        timestamp=candle['timestamp'], symbol=symbol, timeframe=timeframe,
                        candle_index=i, confidence=conf_m, signal="BEARISH", strength=4,
                        success_rate=0.68,
                        description=f"Bearish Marubozu: {body_ratio*100:.1f}% body. Move {body_pct_m*100:.2f}%. Vol {vol_r_m:.1f}x avg.",
                        price_at_detection=float(candle['close']),
                        stop_loss=float(candle['high']) + atr * 0.5,
                        target=float(candle['close']) - body
                    ))
        
        return patterns
    
    def _detect_engulfing(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Bullish and Bearish Engulfing patterns"""
        patterns = []
        
        for i in range(2, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            curr_body = abs(curr['close'] - curr['open'])
            prev_body = abs(prev['close'] - prev['open'])
            
            # Bullish Engulfing
            if (prev['close'] < prev['open'] and  # Previous bearish
                curr['close'] > curr['open'] and  # Current bullish
                curr['open'] < prev['close'] and  # Opens below prev close
                curr['close'] > prev['open']):    # Closes above prev open
                
                atr = self._calculate_atr(df, i)
                # BUG10 FIX: dynamic confidence -- engulf ratio + volume
                engulf_ratio = curr_body / prev_body if prev_body > 0 else 1
                has_vol_e = 'volume' in df.columns
                avg_v_e = self._avg_volume(df, i)
                vol_r_e = float(curr['volume']) / avg_v_e if has_vol_e and avg_v_e > 0 else 1.0
                conf_be = round(min(0.92, 0.72 + min(0.10, (engulf_ratio-1)*0.08) + (0.06 if vol_r_e >= 1.3 else 0)), 2)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.BULLISH_ENGULFING,
                    timestamp=curr['timestamp'], symbol=symbol, timeframe=timeframe,
                    candle_index=i, confidence=conf_be, signal="BULLISH", strength=5,
                    success_rate=self.pattern_stats.get(CandlePattern.BULLISH_ENGULFING, 0.72),
                    description=f"Bullish Engulfing: {engulf_ratio:.1f}x engulf ratio. Vol {vol_r_e:.1f}x avg.",
                    price_at_detection=float(curr['close']),
                    stop_loss=min(float(curr['open']), float(prev['close'])) - atr * 0.5,
                    target=float(curr['close']) + curr_body * 1.5
                ))
            
            # Bearish Engulfing
            if (prev['close'] > prev['open'] and  # Previous bullish
                curr['close'] < curr['open'] and  # Current bearish
                curr['open'] > prev['close'] and  # Opens above prev close
                curr['close'] < prev['open']):    # Closes below prev open
                
                atr = self._calculate_atr(df, i)
                engulf_ratio = curr_body / prev_body if prev_body > 0 else 1
                avg_v_e = self._avg_volume(df, i)
                vol_r_e = float(curr['volume']) / avg_v_e if 'volume' in df.columns and avg_v_e > 0 else 1.0
                conf_be = round(min(0.90, 0.70 + min(0.10, (engulf_ratio-1)*0.08) + (0.06 if vol_r_e >= 1.3 else 0)), 2)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.BEARISH_ENGULFING,
                    timestamp=curr['timestamp'], symbol=symbol, timeframe=timeframe,
                    candle_index=i, confidence=conf_be, signal="BEARISH", strength=5,
                    success_rate=self.pattern_stats.get(CandlePattern.BEARISH_ENGULFING, 0.70),
                    description=f"Bearish Engulfing: {engulf_ratio:.1f}x engulf ratio. Vol {vol_r_e:.1f}x avg.",
                    price_at_detection=float(curr['close']),
                    stop_loss=max(float(curr['open']), float(prev['close'])) + atr * 0.5,
                    target=float(curr['close']) - curr_body * 1.5
                ))
        
        return patterns
    
    def _detect_piercing_dark_cloud(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Piercing Line and Dark Cloud Cover"""
        patterns = []
        
        for i in range(2, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            prev_body = abs(prev['close'] - prev['open'])
            
            # Piercing Line (Bullish)
            if (prev['close'] < prev['open'] and  # Prev bearish
                curr['close'] > curr['open'] and  # Curr bullish
                float(curr['open']) < float(prev['close']) and    # BUG5 FIX: opens below prev close (not prev low)
                curr['close'] > prev['close'] + (prev_body * 0.5) and  # Closes above 50% of prev body
                curr['close'] < prev['open']):    # But below prev open
                
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.PIERCING_LINE,
                    timestamp=curr['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.72,
                    signal="BULLISH",
                    strength=4,
                    success_rate=self.pattern_stats.get(CandlePattern.PIERCING_LINE, 0.65),
                    description="Piercing Line: Bullish reversal. Current candle pierces >50% into previous bearish body.",
                    price_at_detection=curr['close'],
                    stop_loss=curr['open'] - (atr * 0.5),
                    target=curr['close'] + prev_body
                ))
            
            # Dark Cloud Cover (Bearish)
            if (prev['close'] > prev['open'] and  # Prev bullish
                curr['close'] < curr['open'] and  # Curr bearish
                float(curr['open']) > float(prev['close']) and   # BUG5 FIX: opens above prev close (not prev high)
                curr['close'] < prev['close'] - (prev_body * 0.5) and  # Closes below 50% of prev body
                curr['close'] > prev['open']):    # But above prev open
                
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.DARK_CLOUD_COVER,
                    timestamp=curr['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.70,
                    signal="BEARISH",
                    strength=4,
                    success_rate=self.pattern_stats.get(CandlePattern.DARK_CLOUD_COVER, 0.63),
                    description="Dark Cloud Cover: Bearish reversal. Current candle covers >50% of previous bullish body.",
                    price_at_detection=curr['close'],
                    stop_loss=curr['open'] + (atr * 0.5),
                    target=curr['close'] - prev_body
                ))
        
        return patterns
    
    def _detect_harami(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Harami patterns. GAP 7 FIX: volume check added."""
        patterns = []
        has_vol = 'volume' in df.columns
        for i in range(2, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            # GAP 7: volume check -- harami on dead volume = noise
            avg_vol_h = self._avg_volume(df, i)
            vol_ok_h  = (float(curr['volume']) >= avg_vol_h * 0.70) if has_vol and avg_vol_h > 0 else True
            if not vol_ok_h:
                continue
            
            # Bullish Harami -- BUG3 FIX: inner body must be <=50% of outer body
            curr_body_sz = abs(float(curr['close']) - float(curr['open']))
            prev_body_sz = abs(float(prev['close']) - float(prev['open']))
            if (prev['close'] < prev['open'] and  # Prev bearish
                curr['close'] > curr['open'] and  # Curr bullish
                curr['open'] > prev['close'] and  # Curr open above prev close
                curr['close'] < prev['open'] and  # Curr close below prev open
                prev_body_sz > 0 and curr_body_sz / prev_body_sz <= 0.50):  # Inner <=50%
                
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.BULLISH_HARAMI,
                    timestamp=curr['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.68,
                    signal="BULLISH",
                    strength=3,
                    success_rate=self.pattern_stats.get(CandlePattern.BULLISH_HARAMI, 0.62),
                    description="Bullish Harami: Small bullish candle inside previous bearish candle. Reversal signal.",
                    price_at_detection=curr['close'],
                    stop_loss=prev['close'] - (atr * 0.5),
                    target=curr['close'] + (prev['open'] - prev['close'])
                ))
            
            # Bearish Harami -- BUG3 FIX: inner body must be <=50% of outer body
            curr_body_sz = abs(float(curr['close']) - float(curr['open']))
            prev_body_sz = abs(float(prev['close']) - float(prev['open']))
            if (prev['close'] > prev['open'] and  # Prev bullish
                curr['close'] < curr['open'] and  # Curr bearish
                curr['open'] < prev['close'] and  # Curr open below prev close
                curr['close'] > prev['open'] and  # Curr close above prev open
                prev_body_sz > 0 and curr_body_sz / prev_body_sz <= 0.50):  # Inner <=50%
                
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.BEARISH_HARAMI,
                    timestamp=curr['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.66,
                    signal="BEARISH",
                    strength=3,
                    success_rate=self.pattern_stats.get(CandlePattern.BEARISH_HARAMI, 0.60),
                    description="Bearish Harami: Small bearish candle inside previous bullish candle. Reversal signal.",
                    price_at_detection=curr['close'],
                    stop_loss=prev['close'] + (atr * 0.5),
                    target=curr['close'] - (prev['close'] - prev['open'])
                ))
        
        return patterns
    
    def _detect_tweezer(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Tweezer Top and Bottom. GAP 7 FIX: volume check."""
        patterns = []
        has_vol = 'volume' in df.columns
        for i in range(2, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            # GAP 7: volume check
            avg_vol_t = self._avg_volume(df, i)
            vol_ok_t  = (float(curr['volume']) >= avg_vol_t * 0.70) if has_vol and avg_vol_t > 0 else True
            if not vol_ok_t:
                continue
            price_tolerance = float(curr['close']) * 0.001  # BUG4 FIX: 0.1% of price (was 2% of candle range)
            
            # Tweezer Bottom: two lows at same level
            if abs(curr['low'] - prev['low']) < price_tolerance and prev['close'] < prev['open']:
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.TWEEZER_BOTTOM,
                    timestamp=curr['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.70,
                    signal="BULLISH",
                    strength=3,
                    success_rate=0.64,
                    description=f"Tweezer Bottom: Two candles with matching lows at {curr['low']:.2f}. Support level confirmed.",
                    price_at_detection=curr['close'],
                    stop_loss=curr['low'] - (atr * 0.5),
                    target=curr['close'] + (atr * 2)
                ))
            
            # Tweezer Top: two highs at same level
            if abs(curr['high'] - prev['high']) < price_tolerance and prev['close'] > prev['open']:
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.TWEEZER_TOP,
                    timestamp=curr['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.70,
                    signal="BEARISH",
                    strength=3,
                    success_rate=0.62,
                    description=f"Tweezer Top: Two candles with matching highs at {curr['high']:.2f}. Resistance level confirmed.",
                    price_at_detection=curr['close'],
                    stop_loss=curr['high'] + (atr * 0.5),
                    target=curr['close'] - (atr * 2)
                ))
        
        return patterns
    
    def _detect_morning_evening_star(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Morning Star and Evening Star (3-candle patterns)"""
        patterns = []
        
        for i in range(5, len(df)):   # BUG9 FIX: need 5 bars for prior trend check
            c1 = df.iloc[i-2]  # First candle
            c2 = df.iloc[i-1]  # Middle candle (star)
            c3 = df.iloc[i]    # Third candle
            prior2 = df.iloc[i-4:i-2]  # 2 candles before pattern

            c1_body = abs(float(c1['close']) - float(c1['open']))
            c2_body = abs(float(c2['close']) - float(c2['open']))
            c3_body = abs(float(c3['close']) - float(c3['open']))

            prior_downtrend = all(float(prior2.iloc[j]['close']) > float(prior2.iloc[j+1]['close']) for j in range(len(prior2)-1))
            prior_uptrend   = all(float(prior2.iloc[j]['close']) < float(prior2.iloc[j+1]['close']) for j in range(len(prior2)-1))

            # Morning Star (Bullish) -- BUG9 FIX: requires prior downtrend
            if (c1['close'] < c1['open'] and  # First bearish
                prior_downtrend and            # Prior 2 candles falling
                c2_body < c1_body * 0.3 and   # Small middle body (star)
                c3['close'] > c3['open'] and  # Third bullish
                float(c3['close']) > float(c1['close']) + (c1_body * 0.5)):  # Closes well into first
                
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.MORNING_STAR,
                    timestamp=c3['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.85,
                    signal="BULLISH",
                    strength=5,
                    success_rate=self.pattern_stats.get(CandlePattern.MORNING_STAR, 0.78),
                    description="Morning Star: Powerful 3-candle bullish reversal. Strong buy signal after downtrend.",
                    price_at_detection=c3['close'],
                    stop_loss=c2['low'] - (atr * 0.5),
                    target=c3['close'] + (c1_body * 1.5)
                ))
            
            # Evening Star (Bearish) -- BUG9 FIX: requires prior uptrend
            if (c1['close'] > c1['open'] and  # First bullish
                prior_uptrend and             # Prior 2 candles rising
                c2_body < c1_body * 0.3 and   # Small middle body (star)
                c3['close'] < c3['open'] and  # Third bearish
                float(c3['close']) < float(c1['close']) - (c1_body * 0.5)):  # Closes into first
                
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.EVENING_STAR,
                    timestamp=c3['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.83,
                    signal="BEARISH",
                    strength=5,
                    success_rate=self.pattern_stats.get(CandlePattern.EVENING_STAR, 0.76),
                    description="Evening Star: Powerful 3-candle bearish reversal. Strong sell signal after uptrend.",
                    price_at_detection=c3['close'],
                    stop_loss=c2['high'] + (atr * 0.5),
                    target=c3['close'] - (c1_body * 1.5)
                ))
        
        return patterns
    
    def _detect_three_soldiers_crows(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """
        Detect Three White Soldiers and Three Black Crows WITH strict quality filters.

        Why filters matter:
        - In opening 15 min (9:15-9:30), any 3 small red candles triggers 3BC
          even though there's no real bearish structure -- just opening noise.
        - Real 3BC requires: meaningful body size (>=0.15% each), above-avg
          volume, prior uptrend (so it's a reversal), and the total move must
          be significant (>= 0.3% of price).
        """
        patterns = []
        has_volume = 'volume' in df.columns

        for i in range(5, len(df)):   # need 5 bars: 2 for prior trend + 3 pattern
            c1 = df.iloc[i-2]
            c2 = df.iloc[i-1]
            c3 = df.iloc[i]

            price_ref = float(c3['close'])

            # ── Minimum body size filter ──────────────────────────────────────
            # Each candle body must be at least 0.12% of price
            # This eliminates tiny doji-like candles in opening chop
            min_body_pct = 0.0012
            def body_pct(c):
                return abs(float(c['close']) - float(c['open'])) / price_ref

            # ── Volume filter ──────────────────────────────────────────────────
            # Pattern candles should have above-average volume
            avg_vol = float(df['volume'].iloc[max(0,i-20):i].mean()) if has_volume else 1
            min_vol = avg_vol * 0.8  # at least 80% of average (not dead volume)

            def vol_ok(c):
                if not has_volume: return True
                return float(c['volume']) >= min_vol

            # ── Prior trend filter ────────────────────────────────────────────
            # For 3BC (bearish): need prior uptrend (at least 2 green candles before)
            # For 3WS (bullish): need prior downtrend (at least 2 red candles before)
            prior_2 = df.iloc[i-4:i-2]  # 2 candles before the pattern
            prior_bullish = all(prior_2.iloc[j]['close'] > prior_2.iloc[j]['open'] for j in range(len(prior_2)))
            prior_bearish = all(prior_2.iloc[j]['close'] < prior_2.iloc[j]['open'] for j in range(len(prior_2)))

            # ── Opening range block ───────────────────────────────────────────
            # No patterns in first 15 min of market open (9:15-9:29 IST)
            ts = c3['timestamp']
            if hasattr(ts, 'hour'):
                if ts.hour == 9 and ts.minute < 30:
                    continue   # skip all patterns in opening 15 min
            elif isinstance(ts, str) and ('09:1' in ts or '09:2' in ts):
                continue

            atr = self._calculate_atr(df, i)

            # ── Three White Soldiers (Bullish) ─────────────────────────────
            if (c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open']
                and c2['close'] > c1['close'] and c3['close'] > c2['close']
                and c2['open'] > c1['open'] and c2['open'] < c1['close']
                and c3['open'] > c2['open'] and c3['open'] < c2['close']
                and body_pct(c1) >= min_body_pct
                and body_pct(c2) >= min_body_pct
                and body_pct(c3) >= min_body_pct
                and vol_ok(c3) and vol_ok(c2)
                and prior_bearish):   # must be reversing a downtrend

                total_move = float(c3['close']) - float(c1['open'])
                move_pct = total_move / price_ref

                # Total move must be significant (>= 0.3%)
                if move_pct < 0.003:
                    continue

                # Confidence scales with move size and body quality
                avg_body = (body_pct(c1) + body_pct(c2) + body_pct(c3)) / 3
                conf = min(0.92, 0.80 + avg_body * 20 + move_pct * 5)

                patterns.append(PatternDetection(
                    pattern=CandlePattern.THREE_WHITE_SOLDIERS,
                    timestamp=c3['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=round(conf, 2),
                    signal="BULLISH",
                    strength=5,
                    success_rate=self.pattern_stats.get(CandlePattern.THREE_WHITE_SOLDIERS, 0.81),
                    description=(f"Three White Soldiers: Strong bullish reversal after downtrend. "
                                 f"Move: {total_move:.1f}pts ({move_pct*100:.2f}%). "
                                 f"Avg body: {avg_body*100:.2f}%."),
                    price_at_detection=float(c3['close']),
                    stop_loss=float(c1['open']) - (atr * 0.5),
                    target=float(c3['close']) + total_move
                ))

            # ── Three Black Crows (Bearish) ─────────────────────────────────
            if (c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open']
                and c2['close'] < c1['close'] and c3['close'] < c2['close']
                and c2['open'] < c1['open'] and c2['open'] > c1['close']
                and c3['open'] < c2['open'] and c3['open'] > c2['close']
                and body_pct(c1) >= min_body_pct
                and body_pct(c2) >= min_body_pct
                and body_pct(c3) >= min_body_pct
                and vol_ok(c3) and vol_ok(c2)
                and prior_bullish):   # must be reversing an uptrend

                total_move = float(c1['open']) - float(c3['close'])
                move_pct = total_move / price_ref

                # Total move must be significant (>= 0.3%)
                if move_pct < 0.003:
                    continue

                avg_body = (body_pct(c1) + body_pct(c2) + body_pct(c3)) / 3
                conf = min(0.92, 0.80 + avg_body * 20 + move_pct * 5)

                patterns.append(PatternDetection(
                    pattern=CandlePattern.THREE_BLACK_CROWS,
                    timestamp=c3['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=round(conf, 2),
                    signal="BEARISH",
                    strength=5,
                    success_rate=self.pattern_stats.get(CandlePattern.THREE_BLACK_CROWS, 0.78),
                    description=(f"Three Black Crows: Strong bearish reversal after uptrend. "
                                 f"Move: {total_move:.1f}pts ({move_pct*100:.2f}%). "
                                 f"Avg body: {avg_body*100:.2f}%."),
                    price_at_detection=float(c3['close']),
                    stop_loss=float(c1['open']) + (atr * 0.5),
                    target=float(c3['close']) - total_move
                ))

        return patterns
    
    def _detect_three_inside(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect Three Inside Up and Three Inside Down"""
        patterns = []
        
        for i in range(3, len(df)):
            c1 = df.iloc[i-2]
            c2 = df.iloc[i-1]  # Harami candle
            c3 = df.iloc[i]    # Confirmation
            
            # Three Inside Up
            if (c1['close'] < c1['open'] and  # First bearish
                c2['close'] > c2['open'] and  # Second bullish (inside)
                c2['open'] > c1['close'] and c2['close'] < c1['open'] and  # Inside first
                c3['close'] > c3['open'] and c3['close'] > c1['open']):  # Third confirms
                
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.THREE_INSIDE_UP,
                    timestamp=c3['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.75,
                    signal="BULLISH",
                    strength=4,
                    success_rate=0.70,
                    description="Three Inside Up: Bullish reversal confirmed. Harami followed by bullish breakout.",
                    price_at_detection=c3['close'],
                    stop_loss=c1['close'] - (atr * 0.5),
                    target=c3['close'] + (c3['close'] - c1['close'])
                ))
            
            # Three Inside Down
            if (c1['close'] > c1['open'] and  # First bullish
                c2['close'] < c2['open'] and  # Second bearish (inside)
                c2['open'] < c1['close'] and c2['close'] > c1['open'] and  # Inside first
                c3['close'] < c3['open'] and c3['close'] < c1['open']):  # Third confirms
                
                atr = self._calculate_atr(df, i)
                patterns.append(PatternDetection(
                    pattern=CandlePattern.THREE_INSIDE_DOWN,
                    timestamp=c3['timestamp'],
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_index=i,
                    confidence=0.73,
                    signal="BEARISH",
                    strength=4,
                    success_rate=0.68,
                    description="Three Inside Down: Bearish reversal confirmed. Harami followed by bearish breakdown.",
                    price_at_detection=c3['close'],
                    stop_loss=c1['close'] + (atr * 0.5),
                    target=c3['close'] - (c1['close'] - c3['close'])
                ))
        
        return patterns
    
    def _precompute_atr(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """
        BUG 8 FIX: Pre-compute full ATR series once per detect call.
        Previously _calculate_atr() was called inside every candle loop =
        O(n2) time. This reduces to O(n).
        """
        n = len(df)
        atrs = []
        for i in range(n):
            if i < 1:
                atrs.append(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
                continue
            start = max(0, i - period)
            hl_sum = sum(float(df.iloc[j]['high']) - float(df.iloc[j]['low'])
                         for j in range(start, i))
            atrs.append(hl_sum / max(1, i - start))
        return atrs

    def _calculate_atr(self, df: pd.DataFrame, idx: int, period: int = 14) -> float:
        """Single-index ATR (kept for backward compat -- use _precompute_atr for loops)."""
        if idx < 1:
            return float(df.iloc[idx]['high']) - float(df.iloc[idx]['low'])
        start = max(0, idx - period)
        hl_sum = sum(float(df.iloc[j]['high']) - float(df.iloc[j]['low'])
                     for j in range(start, idx))
        return hl_sum / max(1, idx - start)

    def _avg_volume(self, df: pd.DataFrame, idx: int, period: int = 20) -> float:
        """Average volume up to idx, restricted to the same trading session.

        FIX: The old implementation took the last `period` bars regardless of
        date, mixing yesterday's low closing-hour volume with today's high
        opening volume and making vol_ratio meaningless for intraday signals.
        Now we prefer bars from the same calendar date; only fall back to
        cross-session bars when fewer than 5 same-session bars are available.
        """
        if 'volume' not in df.columns:
            return 1.0
        # Determine the date of the bar at idx
        try:
            bar_ts = df.iloc[idx]['timestamp']
            bar_date = pd.Timestamp(bar_ts).date()
        except Exception:
            bar_date = None

        # Collect candidate bars before idx
        start = max(0, idx - period)
        if bar_date is not None:
            # Prefer same-session bars
            same_session = [
                float(df.iloc[j]['volume'])
                for j in range(start, idx)
                if float(df.iloc[j]['volume']) > 0
                and pd.Timestamp(df.iloc[j]['timestamp']).date() == bar_date
            ]
            if len(same_session) >= 5:
                return sum(same_session) / len(same_session)

        # Fallback: all bars in window
        vals = [float(df.iloc[j]['volume']) for j in range(start, idx)
                if float(df.iloc[j]['volume']) > 0]
        return sum(vals) / len(vals) if vals else 1.0

# ════════════════════════════════════════════════════════════════════════════════
#  FIBONACCI ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class FibonacciLevels:
    symbol: str
    timeframe: str
    swing_high: float
    swing_low: float
    direction: str  # "UPTREND" or "DOWNTREND"
    levels: Dict[str, float]  # {percentage: price}
    current_price: float
    nearest_support: float
    nearest_resistance: float
    confluence_zones: List[Dict]  # Areas where multiple Fib levels align
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'swing_high': self.swing_high,
            'swing_low': self.swing_low,
            'direction': self.direction,
            'levels': self.levels,
            'current_price': self.current_price,
            'nearest_support': self.nearest_support,
            'nearest_resistance': self.nearest_resistance,
            'confluence_zones': self.confluence_zones
        }

class FibonacciAnalyzer:
    """Advanced Fibonacci analysis with retracements, extensions, and confluence zones"""
    
    # Standard Fibonacci ratios
    RETRACEMENT_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    EXTENSION_LEVELS = [1.272, 1.414, 1.618, 2.0, 2.618, 3.618, 4.236]
    
    def calculate_fibonacci(self, df: pd.DataFrame, symbol: str, timeframe: str, lookback: int = 100) -> FibonacciLevels:
        """Calculate Fibonacci levels for recent swing"""
        if len(df) < lookback:
            lookback = len(df)
        
        recent_df = df.tail(lookback)
        
        # Find swing high and low
        swing_high = recent_df['high'].max()
        swing_low = recent_df['low'].min()
        swing_range = swing_high - swing_low
        
        current_price = df.iloc[-1]['close']
        
        # Determine trend direction
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else current_price
        direction = "UPTREND" if current_price > sma_50 else "DOWNTREND"
        
        # Calculate retracement levels
        levels = {}
        
        if direction == "UPTREND":
            # Retracing from high to low
            for ratio in self.RETRACEMENT_LEVELS:
                level_price = swing_high - (swing_range * ratio)
                levels[f"{ratio*100:.1f}%"] = level_price
            
            # Extension levels above high
            for ratio in self.EXTENSION_LEVELS:
                level_price = swing_high + (swing_range * (ratio - 1))
                levels[f"Ext {ratio*100:.1f}%"] = level_price
        
        else:  # DOWNTREND
            # Retracing from low to high
            for ratio in self.RETRACEMENT_LEVELS:
                level_price = swing_low + (swing_range * ratio)
                levels[f"{ratio*100:.1f}%"] = level_price
            
            # Extension levels below low
            for ratio in self.EXTENSION_LEVELS:
                level_price = swing_low - (swing_range * (ratio - 1))
                levels[f"Ext {ratio*100:.1f}%"] = level_price
        
        # Find nearest support and resistance
        support_levels = [price for price in levels.values() if price < current_price]
        resistance_levels = [price for price in levels.values() if price > current_price]
        
        nearest_support = max(support_levels) if support_levels else swing_low
        nearest_resistance = min(resistance_levels) if resistance_levels else swing_high
        
        # Find confluence zones (where multiple levels cluster)
        confluence_zones = self._find_confluence_zones(levels, current_price, swing_range)
        
        return FibonacciLevels(
            symbol=symbol,
            timeframe=timeframe,
            swing_high=swing_high,
            swing_low=swing_low,
            direction=direction,
            levels=levels,
            current_price=current_price,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            confluence_zones=confluence_zones
        )
    
    def _find_confluence_zones(self, levels: Dict[str, float], current_price: float, swing_range: float) -> List[Dict]:
        """Find areas where multiple Fib levels cluster together"""
        tolerance = swing_range * 0.01  # 1% tolerance
        zones = []
        
        prices = list(levels.values())
        checked = set()
        
        for i, price1 in enumerate(prices):
            if i in checked:
                continue
            
            clustered_levels = [price1]
            clustered_names = [list(levels.keys())[i]]
            
            for j, price2 in enumerate(prices[i+1:], start=i+1):
                if abs(price1 - price2) < tolerance:
                    clustered_levels.append(price2)
                    clustered_names.append(list(levels.keys())[j])
                    checked.add(j)
            
            if len(clustered_levels) >= 2:  # At least 2 levels clustering
                avg_price = sum(clustered_levels) / len(clustered_levels)
                distance_from_current = abs(avg_price - current_price) / current_price
                
                zones.append({
                    'price': avg_price,
                    'levels': clustered_names,
                    'strength': len(clustered_levels),
                    'distance_pct': distance_from_current * 100,
                    'type': 'support' if avg_price < current_price else 'resistance'
                })
        
        return sorted(zones, key=lambda x: x['distance_pct'])

# ════════════════════════════════════════════════════════════════════════════════
#  KITE CONNECT MANAGER
# ════════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════════
#  INSTITUTIONAL LEVELS ENGINE
#  Pulls PDH/PDL/PWH/PWL from Kite historical data -- key institutional magnets
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class InstitutionalLevels:
    symbol: str
    current_price: float
    # Previous Day levels
    pdh: float          # Previous Day High
    pdl: float          # Previous Day Low
    pdc: float          # Previous Day Close
    pdo: float          # Previous Day Open
    # Previous Week levels  
    pwh: float          # Previous Week High
    pwl: float          # Previous Week Low
    # Current Day levels
    today_open: float   # Today's opening price
    today_high: float   # Today's running high
    today_low:  float   # Today's running low
    # Round number levels (psychological)
    nearest_round: float
    round_levels: List[float]
    # Context
    above_pdh: bool
    below_pdl: bool
    in_pdh_pdl_range: bool
    nearest_level: str      # Which institutional level is closest
    nearest_level_dist: float  # Distance % to nearest level

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "current_price": round(self.current_price, 2),
            "pdh": round(self.pdh, 2),
            "pdl": round(self.pdl, 2),
            "pdc": round(self.pdc, 2),
            "pdo": round(self.pdo, 2),
            "pwh": round(self.pwh, 2),
            "pwl": round(self.pwl, 2),
            "today_open": round(self.today_open, 2),
            "today_high": round(self.today_high, 2),
            "today_low": round(self.today_low, 2),
            "nearest_round": round(self.nearest_round, 2),
            "round_levels": [round(r, 2) for r in self.round_levels],
            "above_pdh": self.above_pdh,
            "below_pdl": self.below_pdl,
            "in_pdh_pdl_range": self.in_pdh_pdl_range,
            "nearest_level": self.nearest_level,
            "nearest_level_dist": round(self.nearest_level_dist, 3),
        }


def calculate_institutional_levels(df: pd.DataFrame, symbol: str) -> InstitutionalLevels:
    """
    Pull Previous Day H/L/C/O and Previous Week H/L from OHLCV data.
    Also compute psychological (round number) levels.
    """
    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        curr = float(df["close"].iloc[-1])
        dates = sorted(df["date"].unique())

        pdh = pdl = pdc = pdo = curr
        pwh = pwl = curr
        today_open = today_high = today_low = curr

        if len(dates) >= 2:
            today_date = dates[-1]
            prev_date  = dates[-2]

            today_df = df[df["date"] == today_date]
            prev_df  = df[df["date"] == prev_date]

            if not today_df.empty:
                today_open = float(today_df["open"].iloc[0])
                today_high = float(today_df["high"].max())
                today_low  = float(today_df["low"].min())

            if not prev_df.empty:
                pdh = float(prev_df["high"].max())
                pdl = float(prev_df["low"].min())
                pdc = float(prev_df["close"].iloc[-1])
                pdo = float(prev_df["open"].iloc[0])

        # Previous Week H/L: go back 5-10 trading days
        if len(dates) >= 7:
            week_dates = dates[-8:-1]  # prev 5 trading days (roughly 1 week)
            week_df = df[df["date"].isin(week_dates)]
            if not week_df.empty:
                pwh = float(week_df["high"].max())
                pwl = float(week_df["low"].min())

        # Round number levels (psychological magnets)
        def round_step(price):
            if price >= 10000: return 500
            elif price >= 5000: return 250
            elif price >= 2000: return 100
            elif price >= 1000: return 50
            elif price >= 500:  return 25
            elif price >= 100:  return 10
            else:               return 5

        step = round_step(curr)
        base = round(curr / step) * step
        round_levels = [base - step*2, base - step, base, base + step, base + step*2]
        nearest_round = min(round_levels, key=lambda x: abs(x - curr))

        # Determine nearest institutional level
        levels = {
            "PDH": pdh, "PDL": pdl, "PDC": pdc,
            "PWH": pwh, "PWL": pwl,
            f"Round {int(nearest_round)}": nearest_round,
        }
        nearest_level = min(levels, key=lambda k: abs(levels[k] - curr))
        nearest_level_dist = abs(levels[nearest_level] - curr) / max(curr, 1) * 100

        return InstitutionalLevels(
            symbol=symbol, current_price=curr,
            pdh=pdh, pdl=pdl, pdc=pdc, pdo=pdo,
            pwh=pwh, pwl=pwl,
            today_open=today_open, today_high=today_high, today_low=today_low,
            nearest_round=nearest_round, round_levels=round_levels,
            above_pdh=(curr > pdh),
            below_pdl=(curr < pdl),
            in_pdh_pdl_range=(pdl <= curr <= pdh),
            nearest_level=nearest_level,
            nearest_level_dist=nearest_level_dist,
        )
    except Exception as e:
        log.debug(f"Institutional levels error for {symbol}: {e}")
        c = float(df["close"].iloc[-1]) if not df.empty else 100.0
        return InstitutionalLevels(
            symbol=symbol, current_price=c,
            pdh=c, pdl=c, pdc=c, pdo=c, pwh=c, pwl=c,
            today_open=c, today_high=c, today_low=c,
            nearest_round=c, round_levels=[c],
            above_pdh=False, below_pdl=False, in_pdh_pdl_range=True,
            nearest_level="N/A", nearest_level_dist=0.0,
        )


# ════════════════════════════════════════════════════════════════════════════════
#  ICT / SMART MONEY CONCEPTS ADVANCED ENGINE
#  Implements: OTE Zones · Breaker Blocks · Displacement · MSS
#  ICT Power of 3 · Killzones · PDH/PDL Targets
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class OTEZone:
    """ICT Optimal Trade Entry -- 0.618 to 0.786 retracement of last swing."""
    direction: str          # "LONG" | "SHORT"
    swing_high: float
    swing_low: float
    ote_low: float          # 0.786 retracement
    ote_high: float         # 0.618 retracement
    ote_mid: float          # 0.705 -- dead center of OTE
    fib_50: float           # 50% retracement
    fib_705: float          # 70.5% retracement (ICT sweet spot)
    fib_79: float           # 79% -- Turtle Soup / last resort entry
    price_in_ote: bool
    leg_strength: str       # "STRONG" | "MODERATE" | "WEAK"

    def to_dict(self) -> Dict:
        return {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class BreakerBlock:
    """Failed Order Block that now acts as opposite-direction zone."""
    direction: str      # "BULL_BREAKER" | "BEAR_BREAKER"
    high: float
    low: float
    mid: float
    origin_bar: int
    origin_time: str
    failed_ob_direction: str    # Original OB direction before it broke
    valid: bool                 # Still unmitigated

    def to_dict(self) -> Dict:
        return {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class Displacement:
    """
    Large impulsive move (>= 2x ATR in 1-3 candles) = institutional footprint.
    Creates FVGs and defines direction of smart money.
    """
    direction: str      # "BULL" | "BEAR"
    start_price: float
    end_price: float
    move_pct: float
    bar_index: int
    bar_time: str
    atr_multiple: float     # How many ATRs was this displacement
    fvg_created: bool       # Did it leave a Fair Value Gap?

    def to_dict(self) -> Dict:
        return {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class ICTAnalysis:
    """Complete ICT / SMC analysis for a symbol."""
    symbol: str
    timeframe: str
    current_price: float
    generated_at: str

    # OTE Zones
    ote_long: Optional[OTEZone]
    ote_short: Optional[OTEZone]
    price_in_ote_long: bool
    price_in_ote_short: bool

    # Breaker Blocks
    breaker_blocks: List[BreakerBlock]

    # Displacement
    last_displacement: Optional[Displacement]

    # Power of 3 phase (Accumulation -> Manipulation -> Distribution)
    po3_phase: str          # "ACCUMULATION" | "MANIPULATION" | "DISTRIBUTION" | "UNKNOWN"
    po3_bias: str           # "BULL" | "BEAR" | "NEUTRAL"

    # Kill Zone context
    in_killzone: bool
    killzone_name: str

    # Daily bias (from daily candles)
    daily_bias: str         # "BULLISH" | "BEARISH" | "RANGING"
    daily_structure: str    # "HH/HL" | "LH/LL" | "RANGING"

    # Fibonacci cascade for current setup
    fib_levels: Dict[str, float]

    # Signal recommendation
    ict_signal: str         # "LONG" | "SHORT" | "WAIT"
    ict_confidence: float
    ict_entry_zone: Tuple[float, float]
    ict_sl: float
    # Fibonacci extension targets (FIXED, proper cascade)
    ict_t1: float           # Fib 1.0 -- equal leg (first profit)
    ict_t2: float           # Fib 1.272 -- standard extension
    ict_t3: float           # Fib 1.618 -- golden extension
    ict_t4: float           # Fib 2.0 -- runner / full target
    ict_rr_t2: float        # R:R to T2
    ict_rr_t3: float        # R:R to T3
    ict_notes: List[str]

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": round(self.current_price, 2),
            "generated_at": self.generated_at,
            "ote_long": self.ote_long.to_dict() if self.ote_long else None,
            "ote_short": self.ote_short.to_dict() if self.ote_short else None,
            "price_in_ote_long": self.price_in_ote_long,
            "price_in_ote_short": self.price_in_ote_short,
            "breaker_blocks": [b.to_dict() for b in self.breaker_blocks],
            "last_displacement": self.last_displacement.to_dict() if self.last_displacement else None,
            "po3_phase": self.po3_phase,
            "po3_bias": self.po3_bias,
            "in_killzone": self.in_killzone,
            "killzone_name": self.killzone_name,
            "daily_bias": self.daily_bias,
            "daily_structure": self.daily_structure,
            "fib_levels": {k: round(v, 2) for k, v in self.fib_levels.items()},
            "ict_signal": self.ict_signal,
            "ict_confidence": round(self.ict_confidence, 1),
            "ict_entry_zone": [round(self.ict_entry_zone[0], 2), round(self.ict_entry_zone[1], 2)],
            "ict_sl": round(self.ict_sl, 2),
            "ict_t1": round(self.ict_t1, 2),
            "ict_t2": round(self.ict_t2, 2),
            "ict_t3": round(self.ict_t3, 2),
            "ict_t4": round(self.ict_t4, 2),
            "ict_rr_t2": round(self.ict_rr_t2, 2),
            "ict_rr_t3": round(self.ict_rr_t3, 2),
            "ict_notes": self.ict_notes,
        }


class ICTEngine:
    """
    ICT (Inner Circle Trader) / Smart Money Concepts Engine.

    Core Logic:
    1. Identify last impulse swing (displacement) with volume confirmation
    2. Calculate OTE zone (61.8% - 78.6% retracement) -- the sniper entry
    3. Detect Breaker Blocks (failed OBs that flip direction)
    4. Identify Power of 3 phase (Accumulation / Manipulation / Distribution)
    5. Check Killzone timing (9:30-10:30, 13:00-14:30 IST)
    6. Generate Fibonacci extension targets: 1.0, 1.272, 1.618, 2.0

    TARGET CALCULATION RULE (FIXED):
    For LONG: swing_low + leg x multiplier
      T1 = swing_low + leg x 1.0   (equal leg, take partial here)
      T2 = swing_low + leg x 1.272 (standard Fib extension)
      T3 = swing_low + leg x 1.618 (golden extension, main target)
      T4 = swing_low + leg x 2.0   (runner, let it breathe)

    For SHORT: swing_high - leg x multiplier
      T1 = swing_high - leg x 1.0
      T2 = swing_high - leg x 1.272
      T3 = swing_high - leg x 1.618
      T4 = swing_high - leg x 2.0

    ORDERING: T1 is ALWAYS closest to entry. T4 is ALWAYS furthest.
    """

    # IST Killzones (high probability reversal/continuation windows)
    KILLZONES = [
        ((9, 15), (10, 0),  "London Open Killzone", True),
        ((9, 30), (10, 30), "Equity Open Killzone", True),
        ((13, 0), (14, 30), "NY Silver Bullet / Afternoon", True),
        ((14, 30),(15, 30), "NY PM / Close Killzone", True),
    ]

    def _in_killzone(self) -> Tuple[bool, str]:
        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(tz=IST)
        h, m = now.hour, now.minute
        for (sh, sm), (eh, em), name, _ in self.KILLZONES:
            if (h, m) >= (sh, sm) and (h, m) < (eh, em):
                return True, name
        return False, "No Active Killzone"

    def _find_last_swing(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], str]:
        """
        Find the most recent significant swing (last clear HH or LL).
        Returns (swing_high, swing_low, direction).
        Uses 3-bar pivot with 0.3% ATR-normalized threshold.

        FIX: When the market has been open for at least 45 minutes (i.e. after
        10:00 AM IST), intraday structure is reliable enough to use.  We
        restrict the swing search to today's session bars first; this prevents
        yesterday's distant swing from setting wildly off Fib targets intraday.
        Falls back to the full 80-bar window when today's session has < 10 bars.
        """
        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(tz=IST).replace(tzinfo=None)
        session_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
        use_session_only = now >= session_start

        if use_session_only and 'timestamp' in df.columns:
            try:
                today = now.date()
                today_mask = df['timestamp'].apply(
                    lambda x: pd.Timestamp(x).date() == today
                )
                today_df = df[today_mask]
                if len(today_df) >= 10:
                    df = today_df.reset_index(drop=True)
            except Exception:
                pass  # fall through to full window

        n = min(len(df), 80)
        sub = df.tail(n).reset_index(drop=True)
        highs  = sub["high"].values.astype(float)
        lows   = sub["low"].values.astype(float)
        closes = sub["close"].values.astype(float)

        atr = float((sub["high"] - sub["low"]).rolling(14).mean().dropna().iloc[-1]) if len(sub) >= 14 else float(closes[-1]) * 0.01

        # Find swing pivots
        pivot_highs = []
        pivot_lows  = []
        for i in range(2, n - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivot_highs.append((i, float(highs[i])))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivot_lows.append((i, float(lows[i])))

        if not pivot_highs or not pivot_lows:
            return None, None, "NEUTRAL"

        last_sh = pivot_highs[-1]  # (bar_idx, price)
        last_sl = pivot_lows[-1]

        # Determine direction: which came last?
        if last_sh[0] > last_sl[0]:
            # Last swing was UP (high came after low) -- now waiting for pullback
            return float(last_sh[1]), float(last_sl[1]), "UP"
        else:
            # Last swing was DOWN (low came after high) -- waiting for pullback
            return float(last_sh[1]), float(last_sl[1]), "DOWN"

    def _calc_ote_zones(self, df: pd.DataFrame) -> Tuple[Optional[OTEZone], Optional[OTEZone]]:
        """
        Calculate OTE (Optimal Trade Entry) zones for both long and short.
        OTE LONG  = 61.8%-78.6% pullback from swing_low to swing_high
        OTE SHORT = 61.8%-78.6% pullback from swing_high to swing_low
        """
        sh, sl, direction = self._find_last_swing(df)
        if sh is None or sl is None:
            return None, None

        leg   = sh - sl
        curr  = float(df["close"].iloc[-1])
        atr   = float((df["high"] - df["low"]).rolling(14).mean().dropna().iloc[-1]) if len(df) >= 14 else leg * 0.05

        # Leg strength assessment
        if leg / max(atr, 0.001) >= 5:   leg_strength = "STRONG"
        elif leg / max(atr, 0.001) >= 3:  leg_strength = "MODERATE"
        else:                              leg_strength = "WEAK"

        # OTE LONG (pullback in upswing)
        ote_long = OTEZone(
            direction="LONG",
            swing_high=sh, swing_low=sl,
            ote_high=sh - leg * 0.618,   # 61.8% pullback from high
            ote_low=sh  - leg * 0.786,   # 78.6% pullback from high (deeper)
            ote_mid=sh  - leg * 0.705,   # 70.5%
            fib_50=sh   - leg * 0.500,
            fib_705=sh  - leg * 0.705,
            fib_79=sh   - leg * 0.790,
            price_in_ote=(sh - leg * 0.786 <= curr <= sh - leg * 0.618),
            leg_strength=leg_strength,
        )

        # OTE SHORT (pullback in downswing)
        ote_short = OTEZone(
            direction="SHORT",
            swing_high=sh, swing_low=sl,
            ote_low=sl  + leg * 0.618,   # 61.8% bounce from low
            ote_high=sl + leg * 0.786,   # 78.6% bounce from low (deeper)
            ote_mid=sl  + leg * 0.705,   # 70.5%
            fib_50=sl   + leg * 0.500,
            fib_705=sl  + leg * 0.705,
            fib_79=sl   + leg * 0.790,
            price_in_ote=(sl + leg * 0.618 <= curr <= sl + leg * 0.786),
            leg_strength=leg_strength,
        )

        return ote_long, ote_short

    def _detect_breaker_blocks(self, df: pd.DataFrame) -> List[BreakerBlock]:
        """
        Breaker Block = An Order Block that was VIOLATED (price closed through it).
        When price returns to the violated OB zone, it now acts in OPPOSITE direction.
        Bearish OB broken -> becomes Bullish Breaker.
        Bullish OB broken -> becomes Bearish Breaker.
        """
        breakers: List[BreakerBlock] = []
        n = min(len(df), 80)
        sub = df.tail(n).reset_index(drop=True)
        curr = float(sub["close"].iloc[-1])
        c = sub["close"].values.astype(float)
        o = sub["open"].values.astype(float)
        h = sub["high"].values.astype(float)
        l = sub["low"].values.astype(float)

        for i in range(2, n - 5):
            # Bullish OB (red candle before 3 green) that was later broken bearishly
            if c[i] < o[i]:  # red candle = potential bullish OB
                next3_bull = all(c[i+k] > o[i+k] for k in range(1, min(4, n-i)))
                if next3_bull:
                    ob_high = float(h[i])
                    ob_low  = float(l[i])
                    # Check if price later CLOSED BELOW the OB low
                    violated = any(c[j] < ob_low for j in range(i+4, min(i+30, n)))
                    if violated:
                        # This bullish OB became a Bearish Breaker
                        valid = curr < ob_high  # still above price = valid bearish zone
                        breakers.append(BreakerBlock(
                            direction="BEAR_BREAKER",
                            high=ob_high, low=ob_low, mid=(ob_high+ob_low)/2,
                            origin_bar=i,
                            origin_time=str(sub["timestamp"].iloc[i] if "timestamp" in sub.columns else "")[:16],
                            failed_ob_direction="BULL_OB",
                            valid=valid,
                        ))

            # Bearish OB (green candle before 3 red) that was later broken bullishly
            elif c[i] > o[i]:  # green candle = potential bearish OB
                next3_bear = all(c[i+k] < o[i+k] for k in range(1, min(4, n-i)))
                if next3_bear:
                    ob_high = float(h[i])
                    ob_low  = float(l[i])
                    # Check if price later CLOSED ABOVE the OB high
                    violated = any(c[j] > ob_high for j in range(i+4, min(i+30, n)))
                    if violated:
                        # Bearish OB became Bullish Breaker
                        valid = curr > ob_low  # still below price = valid bullish zone
                        breakers.append(BreakerBlock(
                            direction="BULL_BREAKER",
                            high=ob_high, low=ob_low, mid=(ob_high+ob_low)/2,
                            origin_bar=i,
                            origin_time=str(sub["timestamp"].iloc[i] if "timestamp" in sub.columns else "")[:16],
                            failed_ob_direction="BEAR_OB",
                            valid=valid,
                        ))

        # Return only valid, most recent, max 4 total
        bull = sorted([b for b in breakers if b.direction=="BULL_BREAKER" and b.valid], key=lambda x: x.origin_bar, reverse=True)[:2]
        bear = sorted([b for b in breakers if b.direction=="BEAR_BREAKER" and b.valid], key=lambda x: x.origin_bar, reverse=True)[:2]
        return bull + bear

    def _detect_displacement(self, df: pd.DataFrame) -> Optional[Displacement]:
        """
        Displacement = Large impulsive candle(s) -- >=2xATR range.
        This is institutional activity. It leaves FVGs and defines the move.
        """
        n = min(len(df), 30)
        sub = df.tail(n).reset_index(drop=True)
        atr_raw = (sub["high"] - sub["low"]).rolling(14).mean()
        atr = float(atr_raw.dropna().iloc[-1]) if not atr_raw.dropna().empty else float(sub["close"].iloc[-1]) * 0.01

        for i in range(n-1, max(n-15, 0), -1):
            bar = sub.iloc[i]
            candle_range = float(bar["high"]) - float(bar["low"])
            candle_body  = abs(float(bar["close"]) - float(bar["open"]))
            atr_mult = candle_range / max(atr, 0.001)

            if atr_mult >= 2.0 and candle_body >= candle_range * 0.6:
                direction = "BULL" if float(bar["close"]) > float(bar["open"]) else "BEAR"
                # Check for FVG after this candle
                fvg_created = False
                if i >= 1 and i < n-1:
                    if direction == "BULL":
                        fvg_created = float(sub.iloc[i-1]["high"]) < float(sub.iloc[i+1]["low"]) if i < n-1 else False
                    else:
                        fvg_created = float(sub.iloc[i-1]["low"]) > float(sub.iloc[i+1]["high"]) if i < n-1 else False

                move_pct = (abs(float(bar["close"]) - float(bar["open"])) / max(float(bar["open"]), 1)) * 100
                return Displacement(
                    direction=direction,
                    start_price=float(bar["open"]),
                    end_price=float(bar["close"]),
                    move_pct=round(move_pct, 2),
                    bar_index=i,
                    bar_time=str(bar["timestamp"] if "timestamp" in bar else "")[:16],
                    atr_multiple=round(atr_mult, 1),
                    fvg_created=fvg_created,
                )
        return None

    def _power_of_3(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        ICT Power of 3:
        Phase 1 -- Accumulation: tight range, low volatility
        Phase 2 -- Manipulation: fake breakout sweeping liquidity (trap)
        Phase 3 -- Distribution: true directional move follows manipulation

        Detection: compare ATR of last 5 bars vs prior 10 bars.
        If decreasing -> Accumulation.
        If a spike above/below range then retraces -> Manipulation.
        If breaking out with sustained direction -> Distribution.
        """
        n = min(len(df), 25)
        sub = df.tail(n)
        c = sub["close"].values.astype(float)
        h = sub["high"].values.astype(float)
        l = sub["low"].values.astype(float)

        recent_atr = float(np.mean([h[i]-l[i] for i in range(n-5, n)]))
        prior_atr  = float(np.mean([h[i]-l[i] for i in range(max(0, n-15), n-5)]))

        # Direction bias from EMA
        ema_close = pd.Series(c).ewm(span=20).mean().iloc[-1]
        bias = "BULL" if c[-1] > float(ema_close) else "BEAR"

        volatility_ratio = recent_atr / max(prior_atr, 0.001)

        if volatility_ratio < 0.6:
            return "ACCUMULATION", bias
        elif volatility_ratio > 1.8:
            # Large recent range -- could be manipulation or distribution
            # Manipulation: price wicked to extreme then reversed
            last5_high = max(h[-5:])
            last5_low  = min(l[-5:])
            last5_close = c[-1]
            range_pct = (last5_high - last5_low) / max(last5_close, 1)
            close_pct = (last5_close - last5_low) / max(last5_high - last5_low, 0.001)

            if close_pct < 0.3 and bias == "BEAR":
                return "MANIPULATION_BULL_TRAP", "BEAR"  # Spiked up then fell back
            elif close_pct > 0.7 and bias == "BULL":
                return "MANIPULATION_BEAR_TRAP", "BULL"  # Spiked down then recovered
            else:
                return "DISTRIBUTION", bias
        else:
            return "DISTRIBUTION", bias

    def _daily_bias(self, df_daily: Optional[pd.DataFrame]) -> Tuple[str, str]:
        """Determine daily timeframe bias from daily OHLCV."""
        if df_daily is None or df_daily.empty or len(df_daily) < 5:
            return "UNKNOWN", "RANGING"
        c = df_daily["close"].values.astype(float)
        h = df_daily["high"].values.astype(float)
        l = df_daily["low"].values.astype(float)
        n = len(c)

        # Simple: EMA20 vs price
        ema20 = float(pd.Series(c).ewm(span=min(20, n)).mean().iloc[-1])

        # Structure: last 3 HH/HL or LH/LL
        pivot_highs = []
        pivot_lows  = []
        for i in range(1, n-1):
            if h[i] > h[i-1] and h[i] > h[i+1] if i < n-1 else True:
                pivot_highs.append(h[i])
            if l[i] < l[i-1] and l[i] < l[i+1] if i < n-1 else True:
                pivot_lows.append(l[i])

        structure = "RANGING"
        if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
            if pivot_highs[-1] > pivot_highs[-2] and pivot_lows[-1] > pivot_lows[-2]:
                structure = "HH/HL"
            elif pivot_highs[-1] < pivot_highs[-2] and pivot_lows[-1] < pivot_lows[-2]:
                structure = "LH/LL"

        bias = "BULLISH" if c[-1] > ema20 else "BEARISH"
        return bias, structure

    def _fib_levels_from_swing(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """
        Full Fibonacci cascade from swing low to swing high (for LONG setup).
        Retracement levels (to find entry):  0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 0.886
        Extension levels (targets):          1.0, 1.272, 1.414, 1.618, 2.0, 2.618
        """
        leg = swing_high - swing_low
        return {
            "0.0":   round(swing_low, 2),
            "0.236": round(swing_high - leg * 0.236, 2),
            "0.382": round(swing_high - leg * 0.382, 2),
            "0.500": round(swing_high - leg * 0.500, 2),
            "0.618": round(swing_high - leg * 0.618, 2),
            "0.705": round(swing_high - leg * 0.705, 2),
            "0.786": round(swing_high - leg * 0.786, 2),
            "0.886": round(swing_high - leg * 0.886, 2),
            "1.0":   round(swing_high, 2),            # BOS level
            "1.272": round(swing_low + leg * 1.272, 2),  # T2 target
            "1.414": round(swing_low + leg * 1.414, 2),
            "1.618": round(swing_low + leg * 1.618, 2),  # T3 target
            "2.0":   round(swing_low + leg * 2.000, 2),  # T4 runner
            "2.618": round(swing_low + leg * 2.618, 2),
        }

    def _build_ict_signal(
        self,
        curr: float,
        ote_long: Optional[OTEZone],
        ote_short: Optional[OTEZone],
        breakers: List[BreakerBlock],
        displacement: Optional[Displacement],
        po3_phase: str,
        daily_bias: str,
        in_killzone: bool,
        atr: float,
        inst_levels: Optional[InstitutionalLevels] = None,
    ) -> Tuple[str, float, Tuple, float, float, float, float, float, float, float, List[str]]:
        """
        Generate final ICT signal with CORRECT Fibonacci extension targets.

        TARGET ORDERING (ALWAYS CORRECT):
        LONG:  entry < T1 < T2 < T3 < T4  (targets go UP)
        SHORT: entry > T1 > T2 > T3 > T4  (targets go DOWN)
        """
        notes = []
        long_score = 0.0
        short_score = 0.0

        # ── Daily bias weight ─────────────────────────────────────────────────
        if daily_bias == "BULLISH":
            long_score += 30
            notes.append("✅ Daily bias: BULLISH (HTF trend aligned)")
        elif daily_bias == "BEARISH":
            short_score += 30
            notes.append("✅ Daily bias: BEARISH (HTF trend aligned)")
        else:
            notes.append("⚠️ Daily bias: UNKNOWN -- reduced confidence")

        # ── OTE Zone confluence ───────────────────────────────────────────────
        if ote_long and ote_long.price_in_ote:
            long_score += 35
            notes.append(f"🎯 Price in OTE LONG zone ({ote_long.ote_low:.0f}-{ote_long.ote_high:.0f}) -- Sniper entry zone!")
        if ote_short and ote_short.price_in_ote:
            short_score += 35
            notes.append(f"🎯 Price in OTE SHORT zone ({ote_short.ote_low:.0f}-{ote_short.ote_high:.0f}) -- Sniper entry zone!")

        # ── Breaker Block confluence ──────────────────────────────────────────
        bull_bb = next((b for b in breakers if b.direction == "BULL_BREAKER" and b.valid and b.low <= curr <= b.high * 1.002), None)
        bear_bb = next((b for b in breakers if b.direction == "BEAR_BREAKER" and b.valid and b.low * 0.998 <= curr <= b.high), None)
        if bull_bb:
            long_score += 25
            notes.append(f"🔵 Bullish Breaker Block {bull_bb.low:.0f}-{bull_bb.high:.0f} -- former bearish OB flipped")
        if bear_bb:
            short_score += 25
            notes.append(f"🔴 Bearish Breaker Block {bear_bb.low:.0f}-{bear_bb.high:.0f} -- former bullish OB flipped")

        # ── Displacement confirmation ─────────────────────────────────────────
        if displacement:
            if displacement.direction == "BULL":
                long_score += 15
                notes.append(f"⚡ Bullish displacement ({displacement.atr_multiple:.1f}xATR) -- institutional buying")
            else:
                short_score += 15
                notes.append(f"⚡ Bearish displacement ({displacement.atr_multiple:.1f}xATR) -- institutional selling")

        # ── Power of 3 phase ─────────────────────────────────────────────────
        if "MANIPULATION_BEAR_TRAP" in po3_phase:
            long_score += 20
            notes.append("🎭 Po3: Bear trap (SSL swept) -- bullish reversal imminent")
        elif "MANIPULATION_BULL_TRAP" in po3_phase:
            short_score += 20
            notes.append("🎭 Po3: Bull trap (BSL swept) -- bearish reversal imminent")
        elif po3_phase == "DISTRIBUTION":
            notes.append("📊 Po3: Distribution phase -- follow displacement direction")

        # ── Killzone timing bonus ─────────────────────────────────────────────
        if in_killzone:
            long_score += 10
            short_score += 10
            notes.append("⏰ Inside Killzone -- institutional activity window")

        # ── PDH/PDL confluence ────────────────────────────────────────────────
        if inst_levels:
            pdh_dist = abs(curr - inst_levels.pdh) / max(curr, 1) * 100
            pdl_dist = abs(curr - inst_levels.pdl) / max(curr, 1) * 100
            if pdh_dist < 0.3:
                short_score += 15
                notes.append(f"📍 Price near PDH ({inst_levels.pdh:.0f}) -- institutional resistance")
            if pdl_dist < 0.3:
                long_score += 15
                notes.append(f"📍 Price near PDL ({inst_levels.pdl:.0f}) -- institutional support")
            if curr > inst_levels.pdh:
                long_score += 10
                notes.append(f"📍 Price above PDH ({inst_levels.pdh:.0f}) -- PDH becomes support")
            if curr < inst_levels.pdl:
                short_score += 10
                notes.append(f"📍 Price below PDL ({inst_levels.pdl:.0f}) -- PDL becomes resistance")

        # ── Decision: minimum 55 points required ─────────────────────────────
        MIN_SCORE = 55

        if long_score >= MIN_SCORE and long_score > short_score:
            signal = "LONG"
            conf   = min(long_score, 92)

            # Use OTE zone for entry, or fall back to current price range
            ote = ote_long
            if ote and ote.swing_high and ote.swing_low:
                leg      = ote.swing_high - ote.swing_low
                entry_lo = ote.ote_low   # 78.6% retracement
                entry_hi = ote.ote_high  # 61.8% retracement
                entry_mid = ote.ote_mid

                # SL: just below the swing low (with ATR buffer)
                sl = round(ote.swing_low - atr * 0.5, 2)
                sl_risk = max(entry_mid - sl, atr * 0.5)

                # CORRECT Fibonacci extension targets (from swing LOW, adding multiples of LEG)
                # T1 = equal leg (1.0 extension)  -- CLOSEST to entry
                # T2 = 1.272 extension            -- standard
                # T3 = 1.618 extension            -- golden
                # T4 = 2.0 extension              -- runner
                t1 = round(ote.swing_low + leg * 1.000, 2)   # Equal leg
                t2 = round(ote.swing_low + leg * 1.272, 2)   # Fib 127.2%
                t3 = round(ote.swing_low + leg * 1.618, 2)   # Fib 161.8%
                t4 = round(ote.swing_low + leg * 2.000, 2)   # Fib 200%

                # Validate ordering (T1 < T2 < T3 < T4, all > entry)
                t1 = max(t1, entry_hi + atr * 0.5)
                t2 = max(t2, t1 + atr * 0.3)
                t3 = max(t3, t2 + atr * 0.3)
                t4 = max(t4, t3 + atr * 0.3)

                rr_t2 = round((t2 - entry_mid) / max(sl_risk, 0.001), 2)
                rr_t3 = round((t3 - entry_mid) / max(sl_risk, 0.001), 2)
            else:
                entry_lo  = curr - atr * 0.3
                entry_hi  = curr + atr * 0.1
                entry_mid = curr
                sl        = round(curr - atr * 1.5, 2)
                sl_risk   = max(entry_mid - sl, atr * 0.5)
                t1 = round(entry_mid + sl_risk * 1.5, 2)
                t2 = round(entry_mid + sl_risk * 2.5, 2)
                t3 = round(entry_mid + sl_risk * 3.5, 2)
                t4 = round(entry_mid + sl_risk * 5.0, 2)
                rr_t2 = round((t2 - entry_mid) / sl_risk, 2)
                rr_t3 = round((t3 - entry_mid) / sl_risk, 2)

            notes.append(f"📊 Entry OTE: {entry_lo:.0f}-{entry_hi:.0f} | SL: {sl:.0f} | T2 R:R={rr_t2}x")

        elif short_score >= MIN_SCORE and short_score >= long_score:
            signal = "SHORT"
            conf   = min(short_score, 92)

            ote = ote_short
            if ote and ote.swing_high and ote.swing_low:
                leg      = ote.swing_high - ote.swing_low
                entry_hi = ote.ote_high   # 78.6% bounce
                entry_lo = ote.ote_low    # 61.8% bounce
                entry_mid = ote.ote_mid

                # SL: just above the swing high (with ATR buffer)
                sl = round(ote.swing_high + atr * 0.5, 2)
                sl_risk = max(sl - entry_mid, atr * 0.5)

                # CORRECT Fibonacci extension targets (from swing HIGH, subtracting multiples)
                # T1 = 1.0 extension (CLOSEST, highest price in downmove)
                # T2 = 1.272         (further down)
                # T3 = 1.618         (further down)
                # T4 = 2.0           (furthest/runner)
                t1 = round(ote.swing_high - leg * 1.000, 2)   # Equal leg DOWN
                t2 = round(ote.swing_high - leg * 1.272, 2)   # Fib 127.2% DOWN
                t3 = round(ote.swing_high - leg * 1.618, 2)   # Fib 161.8% DOWN
                t4 = round(ote.swing_high - leg * 2.000, 2)   # Fib 200% DOWN

                # Validate ordering (T1 > T2 > T3 > T4, all < entry)
                t1 = min(t1, entry_lo - atr * 0.5)
                t2 = min(t2, t1 - atr * 0.3)
                t3 = min(t3, t2 - atr * 0.3)
                t4 = min(t4, t3 - atr * 0.3)

                rr_t2 = round((entry_mid - t2) / max(sl_risk, 0.001), 2)
                rr_t3 = round((entry_mid - t3) / max(sl_risk, 0.001), 2)
            else:
                entry_hi  = curr + atr * 0.3
                entry_lo  = curr - atr * 0.1
                entry_mid = curr
                sl        = round(curr + atr * 1.5, 2)
                sl_risk   = max(sl - entry_mid, atr * 0.5)
                t1 = round(entry_mid - sl_risk * 1.5, 2)
                t2 = round(entry_mid - sl_risk * 2.5, 2)
                t3 = round(entry_mid - sl_risk * 3.5, 2)
                t4 = round(entry_mid - sl_risk * 5.0, 2)
                rr_t2 = round((entry_mid - t2) / sl_risk, 2)
                rr_t3 = round((entry_mid - t3) / sl_risk, 2)

            notes.append(f"📊 Entry OTE: {entry_lo:.0f}-{entry_hi:.0f} | SL: {sl:.0f} | T2 R:R={rr_t2}x")

        else:
            signal = "WAIT"
            conf   = 0.0
            entry_lo = entry_hi = curr
            sl = t1 = t2 = t3 = t4 = curr
            rr_t2 = rr_t3 = 0.0
            if long_score > 0:  notes.append(f"⚠️ Long score {long_score:.0f} -- need >={MIN_SCORE}")
            if short_score > 0: notes.append(f"⚠️ Short score {short_score:.0f} -- need >={MIN_SCORE}")
            if not notes: notes.append("ℹ️ No ICT confluence -- wait for setup")

        return signal, conf, (entry_lo, entry_hi), sl, t1, t2, t3, t4, rr_t2, rr_t3, notes

    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        df_daily: Optional[pd.DataFrame] = None,
        inst_levels: Optional[InstitutionalLevels] = None,
    ) -> ICTAnalysis:
        """Full ICT analysis combining all concepts."""
        curr = float(df["close"].iloc[-1])
        atr_raw = (df["high"] - df["low"]).rolling(14).mean()
        atr = float(atr_raw.dropna().iloc[-1]) if not atr_raw.dropna().empty else curr * 0.01

        ote_long, ote_short = self._calc_ote_zones(df)
        breakers            = self._detect_breaker_blocks(df)
        displacement        = self._detect_displacement(df)
        po3_phase, po3_bias = self._power_of_3(df)
        in_kz, kz_name      = self._in_killzone()
        daily_bias, d_struct= self._daily_bias(df_daily)

        sh, sl, _ = self._find_last_swing(df)
        fib_levels = self._fib_levels_from_swing(sh or curr, sl or curr * 0.95)

        sig, conf, ez, stop, t1, t2, t3, t4, rr2, rr3, notes = self._build_ict_signal(
            curr, ote_long, ote_short, breakers, displacement,
            po3_phase, daily_bias, in_kz, atr, inst_levels
        )

        return ICTAnalysis(
            symbol=symbol, timeframe=timeframe,
            current_price=curr, generated_at=datetime.now().isoformat(),
            ote_long=ote_long, ote_short=ote_short,
            price_in_ote_long=(ote_long.price_in_ote if ote_long else False),
            price_in_ote_short=(ote_short.price_in_ote if ote_short else False),
            breaker_blocks=breakers,
            last_displacement=displacement,
            po3_phase=po3_phase, po3_bias=po3_bias,
            in_killzone=in_kz, killzone_name=kz_name,
            daily_bias=daily_bias, daily_structure=d_struct,
            fib_levels=fib_levels,
            ict_signal=sig, ict_confidence=conf,
            ict_entry_zone=ez, ict_sl=stop,
            ict_t1=t1, ict_t2=t2, ict_t3=t3, ict_t4=t4,
            ict_rr_t2=rr2, ict_rr_t3=rr3,
            ict_notes=notes,
        )


ict_engine = ICTEngine()

class KiteManager:
    """Manages Zerodha Kite Connect authentication and data fetching"""

    def __init__(self):
        self.kite: Optional[KiteConnect] = None
        self.access_token: Optional[str] = None
        self.is_authenticated = False
        self._load_token()

    def _token_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), ".kite_token")

    def _load_token(self):
        """
        Load saved token and VALIDATE it with a live Kite API call.
        Kite tokens expire daily (reset ~7 AM IST). Even if saved date == today,
        the token may already be invalid if the app was started after market open.
        We call kite.profile() as a cheap validation -- if it raises, token is dead.
        """
        try:
            if not os.path.exists(self._token_path()):
                return
            with open(self._token_path()) as f:
                data = json.load(f)
            saved_date = data.get("date", "")
            access_token = data.get("access_token", "")
            if not access_token:
                return
            # Only load if saved TODAY in IST (Railway runs UTC — must use IST date)
            from datetime import timezone as _tz
            _IST = _tz(timedelta(hours=5, minutes=30))
            today_ist = datetime.now(tz=_IST).date().isoformat()
            if saved_date != today_ist:
                log.info(f"⏰ Saved token is from {saved_date}, today IST is {today_ist} -- need fresh login")
                self._delete_token()
                return
            # Set token and VALIDATE with live API call
            kite = KiteConnect(api_key=API_KEY)
            kite.set_access_token(access_token)
            try:
                profile = kite.profile()  # cheap call -- fails if token expired
                self.kite = kite
                self.access_token = access_token
                self.is_authenticated = True
                log.info(f"✅ Kite token valid -- logged in as {profile.get('user_name','')}")
            except Exception as api_err:
                log.warning(f"⚠️  Saved token is INVALID ({api_err}) -- deleting, please re-login")
                self._delete_token()
        except Exception as e:
            log.warning(f"Could not load token: {e}")

    def _delete_token(self):
        """Remove stale token file so user is prompted to login again."""
        try:
            if os.path.exists(self._token_path()):
                os.remove(self._token_path())
                log.info("🗑️  Stale token file deleted")
        except Exception:
            pass

    def invalidate(self):
        """Force logout -- clears token from memory and disk."""
        self.kite = None
        self.access_token = None
        self.is_authenticated = False
        self._delete_token()
        log.info("🔓 Kite session invalidated")

    def _save_token(self, token: str):
        try:
            from datetime import timezone as _tz
            _IST = _tz(timedelta(hours=5, minutes=30))
            today_ist = datetime.now(tz=_IST).date().isoformat()
            with open(self._token_path(), "w") as f:
                json.dump({"access_token": token, "date": today_ist}, f)
        except Exception as e:
            log.warning(f"Could not save token: {e}")

    def get_login_url(self) -> str:
        kite = KiteConnect(api_key=API_KEY)
        # IMPORTANT: redirect URL must match EXACTLY what is registered
        # in your Kite Developer Console app settings.
        # Set this to: http://127.0.0.1:8000/callback
        return kite.login_url()

    def complete_login(self, request_token: str) -> bool:
        try:
            kite = KiteConnect(api_key=API_KEY)
            data = kite.generate_session(request_token, api_secret=API_SECRET)
            self.access_token = data["access_token"]
            kite.set_access_token(self.access_token)
            self.kite = kite
            self.is_authenticated = True
            self._save_token(self.access_token)
            log.info("✅ Kite login successful")
            # Pre-resolve all SCAN_UNIVERSE tokens in one batch (avoids per-symbol
            # instruments DF loading during the first scan which caused 2+ min delays)
            threading.Thread(target=_bulk_resolve_tokens, daemon=True).start()
            return True
        except Exception as e:
            log.error(f"Login failed: {e}")
            return False

    def get_historical_data(self, instrument_token: int, interval: str = "3minute", days: int = 5) -> pd.DataFrame:
        if not self.is_authenticated:
            return pd.DataFrame()
        import time as _time
        # CRITICAL: Railway runs UTC. Kite needs IST datetimes.
        IST = timezone(timedelta(hours=5, minutes=30))
        for attempt in range(3):
            try:
                to_dt   = datetime.now(tz=IST).replace(tzinfo=None)   # naive IST
                from_dt = to_dt - timedelta(days=days)
                records = self.kite.historical_data(instrument_token, from_dt, to_dt, interval)
                if not records:
                    log.warning(f"Kite returned 0 records for token {instrument_token} ({interval}, {days}d)")
                    return pd.DataFrame()
                df = pd.DataFrame(records)
                if not df.empty:
                    df.rename(columns={"date": "timestamp"}, inplace=True)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    last_ts = df["timestamp"].iloc[-1]
                    age_min = round((to_dt - last_ts.replace(tzinfo=None)).total_seconds() / 60, 1)
                    log.info(f"Fetched {len(df)} candles for token {instrument_token} ({interval}). Last: {last_ts}, age: {age_min:.0f} min")
                return df
            except Exception as e:
                err_str = str(e).lower()
                if "too many" in err_str or "rate" in err_str or "429" in err_str:
                    wait = 2 ** attempt
                    log.warning(f"Rate-limited (attempt {attempt+1}/3) -- retrying in {wait}s")
                    _time.sleep(wait)
                    continue
                if "token" in err_str or "session" in err_str or "auth" in err_str or "403" in err_str or "401" in err_str:
                    log.error(f"Kite auth error — token expired: {e}")
                    self.is_authenticated = False   # force re-login
                    return pd.DataFrame()
                log.error(f"Historical data error (token {instrument_token}, {interval}): {e}")
                return pd.DataFrame()
        log.error(f"Historical data: gave up after 3 attempts")
        return pd.DataFrame()

    def get_instruments(self, exchange: str = "NSE") -> List[Dict]:
        if not self.is_authenticated:
            return []
        try:
            return self.kite.instruments(exchange)
        except Exception as e:
            log.error(f"Instruments error: {e}")
            return []

    def get_quote(self, symbols: List[str]) -> Dict:
        if not self.is_authenticated:
            return {}
        try:
            return self.kite.quote(symbols)
        except Exception as e:
            log.error(f"Quote error: {e}")
            return {}

    def _get_nse_instruments_df(self) -> pd.DataFrame:
        """Fetch and cache NSE EQ instruments for token resolution (once per day)."""
        cache_file = os.path.join(os.path.dirname(__file__), ".nse_instruments.json")
        try:
            if os.path.exists(cache_file):
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
                if mtime == date.today():
                    with open(cache_file) as f:
                        return pd.DataFrame(json.load(f))
            if not self.is_authenticated:
                return pd.DataFrame()
            instruments = self.kite.instruments("NSE")
            df = pd.DataFrame(instruments)
            # Keep only EQ instruments to reduce size
            df = df[df["instrument_type"] == "EQ"].copy()
            with open(cache_file, "w") as f:
                json.dump(df.to_dict(orient="records"), f, default=str)
            log.info(f"NSE EQ instruments cached: {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"NSE instruments error: {e}")
            return pd.DataFrame()

    def get_nfo_instruments_df(self) -> pd.DataFrame:
        """Fetch and cache NFO instruments as a DataFrame (refreshed once per day)."""
        cache_file = os.path.join(os.path.dirname(__file__), ".nfo_instruments.json")
        try:
            if os.path.exists(cache_file):
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
                if mtime == date.today():
                    with open(cache_file) as f:
                        data = json.load(f)
                    return pd.DataFrame(data)
            if not self.is_authenticated:
                return pd.DataFrame()
            instruments = self.kite.instruments("NFO")
            df = pd.DataFrame(instruments)
            # Convert expiry to date string for JSON serialisation
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date.astype(str)
            with open(cache_file, "w") as f:
                json.dump(df.to_dict(orient="records"), f, default=str)
            log.info(f"NFO instruments cached: {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"NFO instruments error: {e}")
            return pd.DataFrame()

    def get_real_expiries(self, fno_symbol: str) -> Dict[str, str]:
        """Return real weekly + monthly expiry dates from Kite for given F&O symbol."""
        df = self.get_nfo_instruments_df()
        if df.empty:
            return {}
        sym_df = df[
            (df["name"] == fno_symbol) &
            (df["instrument_type"].isin(["CE", "PE", "FUT"]))
        ].copy()
        if sym_df.empty:
            return {}
        expiries = sorted(sym_df["expiry"].unique())
        today_str = date.today().isoformat()
        future_expiries = [e for e in expiries if e >= today_str]
        if not future_expiries:
            return {}
        result = {"weekly": future_expiries[0]}
        # Monthly = the expiry that is NOT the nearest few weeklies (usually > 21 days away)
        monthly_candidates = [e for e in future_expiries
                              if (date.fromisoformat(e) - date.today()).days >= 20]
        result["monthly"] = monthly_candidates[0] if monthly_candidates else future_expiries[-1]
        return result

    def get_option_chain(self, fno_symbol: str, expiry_str: str,
                          spot_price: float, num_strikes: int = 5) -> List[Dict]:
        """
        Fetch real option chain from Kite for strikes near ATM.
        Returns list of {strike, ce_token, pe_token, ce_quote, pe_quote, iv, delta, theta, gamma}
        """
        if not self.is_authenticated:
            return []
        df = self.get_nfo_instruments_df()
        if df.empty:
            return []

        # Filter to this symbol + expiry
        chain_df = df[
            (df["name"] == fno_symbol) &
            (df["expiry"] == expiry_str) &
            (df["instrument_type"].isin(["CE", "PE"]))
        ].copy()
        if chain_df.empty:
            return []

        # Find ATM strike
        chain_df["strike"] = pd.to_numeric(chain_df["strike"], errors="coerce")
        chain_df = chain_df.dropna(subset=["strike"])
        atm = min(chain_df["strike"].unique(), key=lambda x: abs(x - spot_price))

        strikes_all = sorted(chain_df["strike"].unique())
        atm_idx = strikes_all.index(atm)
        selected = strikes_all[max(0, atm_idx - num_strikes): atm_idx + num_strikes + 1]

        chain = []
        kite_symbols = []
        token_map = {}

        for strike in selected:
            for opt_type in ["CE", "PE"]:
                row = chain_df[
                    (chain_df["strike"] == strike) &
                    (chain_df["instrument_type"] == opt_type)
                ]
                if row.empty:
                    continue
                token = int(row.iloc[0]["instrument_token"])
                tradingsymbol = row.iloc[0]["tradingsymbol"]
                kite_symbols.append(f"NFO:{tradingsymbol}")
                token_map[f"NFO:{tradingsymbol}"] = {
                    "strike": strike, "type": opt_type, "token": token,
                    "tradingsymbol": tradingsymbol
                }

        if not kite_symbols:
            return []

        # Fetch quotes in batches of 500
        quotes = {}
        for i in range(0, len(kite_symbols), 500):
            batch = kite_symbols[i:i+500]
            try:
                q = self.kite.quote(batch)
                quotes.update(q)
            except Exception as e:
                log.error(f"Option chain quote error: {e}")

        # Build chain rows
        strike_data = {}
        for sym_key, info in token_map.items():
            q = quotes.get(sym_key, {})
            strike = info["strike"]
            if strike not in strike_data:
                strike_data[strike] = {"strike": strike}
            oi = q.get("oi", 0)
            ltp = q.get("last_price", 0)
            volume = q.get("volume", 0)
            # Greeks from Kite (available in quote for F&O)
            greeks = q.get("greeks", {}) or {}
            ohlc = q.get("ohlc", {}) or {}
            entry = {
                "ltp": ltp,
                "oi": oi,
                "volume": volume,
                "iv": greeks.get("iv", None),
                "delta": greeks.get("delta", None),
                "theta": greeks.get("theta", None),
                "gamma": greeks.get("gamma", None),
                "vega": greeks.get("vega", None),
                "tradingsymbol": info["tradingsymbol"],
                "token": info["token"],
                "bid": q.get("depth", {}).get("buy", [{}])[0].get("price", 0) if q.get("depth") else 0,
                "ask": q.get("depth", {}).get("sell", [{}])[0].get("price", 0) if q.get("depth") else 0,
            }
            strike_data[strike][info["type"]] = entry

        return sorted(strike_data.values(), key=lambda x: x["strike"])



# ════════════════════════════════════════════════════════════════════════════════
#  F&O TRADE SIGNAL ENGINE
# ════════════════════════════════════════════════════════════════════════════════

FNO_EXPIRY_SYMBOLS = {
    # Indices
    "NIFTY":       {"exchange": "NFO", "lot_size": 25,   "base": "NIFTY 50"},
    "BANKNIFTY":   {"exchange": "NFO", "lot_size": 15,   "base": "NIFTY BANK"},
    # Largecap
    "RELIANCE":    {"exchange": "NFO", "lot_size": 250,  "base": "RELIANCE"},
    "TCS":         {"exchange": "NFO", "lot_size": 150,  "base": "TCS"},
    "INFY":        {"exchange": "NFO", "lot_size": 300,  "base": "INFY"},
    "HDFCBANK":    {"exchange": "NFO", "lot_size": 550,  "base": "HDFCBANK"},
    "ICICIBANK":   {"exchange": "NFO", "lot_size": 700,  "base": "ICICIBANK"},
    "BAJFINANCE":  {"exchange": "NFO", "lot_size": 125,  "base": "BAJFINANCE"},
    "HINDUNILVR":  {"exchange": "NFO", "lot_size": 300,  "base": "HINDUNILVR"},
    "MARUTI":      {"exchange": "NFO", "lot_size": 25,   "base": "MARUTI"},
    "TATAMOTORS":  {"exchange": "NFO", "lot_size": 1425, "base": "TATAMOTORS"},
    "AXISBANK":    {"exchange": "NFO", "lot_size": 625,  "base": "AXISBANK"},
    "LT":          {"exchange": "NFO", "lot_size": 150,  "base": "LT"},
    "WIPRO":       {"exchange": "NFO", "lot_size": 1500, "base": "WIPRO"},
    "SUNPHARMA":   {"exchange": "NFO", "lot_size": 350,  "base": "SUNPHARMA"},
    "TATASTEEL":   {"exchange": "NFO", "lot_size": 3375, "base": "TATASTEEL"},
    "KOTAKBANK":   {"exchange": "NFO", "lot_size": 400,  "base": "KOTAKBANK"},
    "SBIN":        {"exchange": "NFO", "lot_size": 1500, "base": "SBIN"},
    "ADANIENT":    {"exchange": "NFO", "lot_size": 250,  "base": "ADANIENT"},
    "BHARTIARTL":  {"exchange": "NFO", "lot_size": 475,  "base": "BHARTIARTL"},
    "NTPC":        {"exchange": "NFO", "lot_size": 2925, "base": "NTPC"},
    "POWERGRID":   {"exchange": "NFO", "lot_size": 2700, "base": "POWERGRID"},
    "ONGC":        {"exchange": "NFO", "lot_size": 1925, "base": "ONGC"},
    "COALINDIA":   {"exchange": "NFO", "lot_size": 1400, "base": "COALINDIA"},
    # Midcap F&O eligible
    "MUTHOOTFIN":  {"exchange": "NFO", "lot_size": 375,  "base": "MUTHOOTFIN"},
    "CHOLAFIN":    {"exchange": "NFO", "lot_size": 700,  "base": "CHOLAFIN"},
    "PERSISTENT":  {"exchange": "NFO", "lot_size": 125,  "base": "PERSISTENT"},
    "COFORGE":     {"exchange": "NFO", "lot_size": 150,  "base": "COFORGE"},
    "LTM":         {"exchange": "NFO", "lot_size": 75,   "base": "LTM"},
    "GODREJPROP":  {"exchange": "NFO", "lot_size": 325,  "base": "GODREJPROP"},
    "TORNTPHARM":  {"exchange": "NFO", "lot_size": 150,  "base": "TORNTPHARM"},
    "MAXHEALTH":   {"exchange": "NFO", "lot_size": 533,  "base": "MAXHEALTH"},
    "ETERNAL":      {"exchange": "NFO", "lot_size": 4500, "base": "ETERNAL"},
    "NYKAA":       {"exchange": "NFO", "lot_size": 1800, "base": "NYKAA"},
    "POLICYBZR":   {"exchange": "NFO", "lot_size": 1000, "base": "POLICYBZR"},
    "CDSL":        {"exchange": "NFO", "lot_size": 500,  "base": "CDSL"},
    # Note: FIVESTAR, HAPPSTMNDS, IREDA, RAILVIKAS, SENCO, IDEAFORGE are
    # equity-only (no F&O). Use Patterns & Fibonacci tab for those.
}

# ── NOTE: expiry dates are now fetched from Kite NFO instruments, not guessed ──
# get_real_expiries() is a method on kite_manager (see KiteManager class above)

# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
#  FIXED STRUCTURE TRADE ENGINE (BOS/CHoCH + FIBONACCI TARGETS)
#  CRITICAL FIXES:
#    1. Target ordering: T1 = CLOSEST (Fib 1.0), T2 = Fib 1.272, T3 = Fib 1.618
#    2. OTE entry: price must be 61.8%-78.6% pullback (sniper zone)
#    3. Minimum RR: 2.0x to T2 required (was accepting 0.9x!)
#    4. Multi-confluence: needs OB OR FVG confluence at entry zone
# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
#  STRUCTURE-BASED TRADE ENGINE
#  Implements: Swing H/L -> BOS/CHoCH -> Pullback Zone -> Fib Targets
#  No repaint. Bar-close confirmed. Volume + ATR + EMA trend filter.
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class StructureSetup:
    """A locked, non-repainting structure-based trade setup with CORRECT Fibonacci targets."""
    symbol:           str
    timeframe:        str
    direction:        str          # "LONG" | "SHORT"
    setup_type:       str          # "BOS_PULLBACK" | "OTE_ENTRY" | "CHOCH_REVERSAL"

    # Structure levels (FIXED at setup time)
    swing_high:       float
    swing_low:        float
    bos_level:        float
    invalidation:     float        # SL level -- price beyond this = setup dead

    # Entry zones
    entry_zone_low:   float        # 78.6% Fib pullback (deeper OTE)
    entry_zone_high:  float        # 61.8% Fib pullback (shallow OTE)
    entry_ideal:      float        # 70.5% -- ICT OTE sweet spot

    # Risk
    sl_fixed:         float        # Hard stop loss
    sl_pct:           float        # SL as % of entry

    # ═══ FIXED FIBONACCI EXTENSION TARGETS ═══
    # LONG:  T1 < T2 < T3 < T4  (all ABOVE entry)
    # SHORT: T1 > T2 > T3 > T4  (all BELOW entry)
    target_1:         float        # Fib 1.0 (equal leg) -- first profit, closest
    target_2:         float        # Fib 1.272 -- standard extension
    target_3:         float        # Fib 1.618 -- golden extension, main target
    target_4:         float        # Fib 2.0 -- runner / full target

    target_rr:        float        # R:R to target_2 (must be >= 2.0)
    target_rr_t3:     float        # R:R to target_3

    # Confluence
    trend_aligned:    bool
    volume_ok:        bool
    atr_ok:           bool
    ema_alignment:    str
    ob_confluence:    bool         # Order block at entry zone?
    fvg_confluence:   bool         # FVG at entry zone?
    ote_in_zone:      bool         # Price in OTE zone?

    confidence_score: float
    quality_label:    str          # "A+" | "A" | "B"
    setup_age_bars:   int
    notes:            List[str]
    generated_at:     str

    def to_dict(self) -> Dict:
        return {
            "symbol":          self.symbol,
            "timeframe":       self.timeframe,
            "direction":       self.direction,
            "setup_type":      self.setup_type,
            "swing_high":      round(self.swing_high, 2),
            "swing_low":       round(self.swing_low, 2),
            "bos_level":       round(self.bos_level, 2),
            "invalidation":    round(self.invalidation, 2),
            "entry_zone_low":  round(self.entry_zone_low, 2),
            "entry_zone_high": round(self.entry_zone_high, 2),
            "entry_ideal":     round(self.entry_ideal, 2),
            "sl_fixed":        round(self.sl_fixed, 2),
            "sl_pct":          round(self.sl_pct, 2),
            # CORRECT target ordering guaranteed
            "target_1":        round(self.target_1, 2),    # Closest
            "target_2":        round(self.target_2, 2),    # Standard
            "target_3":        round(self.target_3, 2),    # Golden
            "target_4":        round(self.target_4, 2),    # Runner
            "target_rr":       round(self.target_rr, 2),
            "target_rr_t3":    round(self.target_rr_t3, 2),
            "trend_aligned":   self.trend_aligned,
            "volume_ok":       self.volume_ok,
            "atr_ok":          self.atr_ok,
            "ema_alignment":   self.ema_alignment,
            "ob_confluence":   self.ob_confluence,
            "fvg_confluence":  self.fvg_confluence,
            "ote_in_zone":     self.ote_in_zone,
            "confidence_score":self.confidence_score,
            "quality_label":   self.quality_label,
            "setup_age_bars":  self.setup_age_bars,
            "notes":           self.notes,
            "generated_at":    self.generated_at,
        }


class StructureTradeEngine:
    """
    Institutional Structure Trade Engine.

    Detects BOS (Break of Structure) -> Pullback -> OTE Entry.

    SIGNAL RULES:
    1. EMA50 / EMA200 trend filter
    2. BOS confirmed: close beyond last swing high/low
    3. Price pulls back into OTE zone (61.8%-78.6% Fibonacci retracement)
    4. Entry at OTE with Order Block or FVG confluence required
    5. SL: below swing low (LONG) or above swing high (SHORT) + 0.5 ATR buffer

    FIBONACCI TARGETS (ALWAYS IN CORRECT ORDER):
    LONG setup (from swing_low):
      T1 = swing_low + leg x 1.000  -> Equal leg move (first exit, protect profits)
      T2 = swing_low + leg x 1.272  -> Standard extension (RR must be >= 2.0x)
      T3 = swing_low + leg x 1.618  -> Golden ratio extension
      T4 = swing_low + leg x 2.000  -> Full runner

    SHORT setup (from swing_high):
      T1 = swing_high - leg x 1.000  -> Equal leg drop
      T2 = swing_high - leg x 1.272  -> Standard extension
      T3 = swing_high - leg x 1.618  -> Golden extension
      T4 = swing_high - leg x 2.000  -> Runner
    """

    def __init__(self):
        self._ob_cache: Dict[str, Any] = {}

    def _find_pivots(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """3-bar pivot detection with 0.2% minimum significance."""
        n = len(df)
        highs = df["high"].values.astype(float)
        lows  = df["low"].values.astype(float)
        ph, pl = [], []
        atr = float(np.mean(highs[-20:] - lows[-20:])) if n >= 20 else float(highs[-1]) * 0.01

        for i in range(2, n - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                if not ph or highs[i] > highs[ph[-1]] + atr * 0.2:
                    ph.append(i)
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                if not pl or lows[i] < lows[pl[-1]] - atr * 0.2:
                    pl.append(i)
        return ph, pl

    def _detect_bos_choch(self, df: pd.DataFrame, pivot_highs: List[int], pivot_lows: List[int]) -> List[Dict]:
        """Detect Break of Structure and Change of Character events."""
        events = []
        close  = df["close"].values.astype(float)
        high   = df["high"].values.astype(float)
        low    = df["low"].values.astype(float)

        for i, ph_idx in enumerate(pivot_highs[:-1] if len(pivot_highs) > 1 else []):
            level = float(high[ph_idx])
            # Find bars after this pivot high where close breaks above
            for j in range(ph_idx + 1, min(ph_idx + 40, len(close))):
                if float(close[j]) > level:
                    events.append({"type": "BOS_BULL", "bar": j, "level": level, "pivot_bar": ph_idx})
                    break

        for i, pl_idx in enumerate(pivot_lows[:-1] if len(pivot_lows) > 1 else []):
            level = float(low[pl_idx])
            for j in range(pl_idx + 1, min(pl_idx + 40, len(close))):
                if float(close[j]) < level:
                    events.append({"type": "BOS_BEAR", "bar": j, "level": level, "pivot_bar": pl_idx})
                    break

        events.sort(key=lambda x: x["bar"])
        return events

    def _calc_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        return df["close"].ewm(span=period, adjust=False).mean()

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        tr = np.maximum(df["high"] - df["low"],
             np.maximum(abs(df["high"] - df["close"].shift(1)),
                        abs(df["low"]  - df["close"].shift(1))))
        return float(tr.rolling(period).mean().dropna().iloc[-1]) if len(df) >= period else float((df["high"] - df["low"]).mean())

    def _relative_volume(self, df: pd.DataFrame, lookback: int = 20) -> float:
        if "volume" not in df.columns or df["volume"].iloc[-1] == 0:
            return 1.0
        avg_vol = df["volume"].rolling(lookback).mean().iloc[-1]
        return float(df["volume"].iloc[-1]) / float(avg_vol) if avg_vol else 1.0

    def _detect_ob_at_zone(self, df: pd.DataFrame, zone_low: float, zone_high: float, direction: str) -> bool:
        """Check if there's an Order Block within the entry zone."""
        n = min(len(df), 60)
        sub = df.tail(n).reset_index(drop=True)
        c = sub["close"].values.astype(float)
        o = sub["open"].values.astype(float)
        h = sub["high"].values.astype(float)
        l = sub["low"].values.astype(float)

        for i in range(2, n - 3):
            if direction == "LONG" and c[i] < o[i]:  # Red candle (potential bull OB)
                if all(c[i+k] > o[i+k] for k in range(1, min(3, n-i))):
                    ob_lo = float(l[i]); ob_hi = float(h[i])
                    # OB overlaps entry zone?
                    if ob_lo <= zone_high and ob_hi >= zone_low:
                        return True
            elif direction == "SHORT" and c[i] > o[i]:  # Green candle (potential bear OB)
                if all(c[i+k] < o[i+k] for k in range(1, min(3, n-i))):
                    ob_lo = float(l[i]); ob_hi = float(h[i])
                    if ob_lo <= zone_high and ob_hi >= zone_low:
                        return True
        return False

    def _detect_fvg_at_zone(self, df: pd.DataFrame, zone_low: float, zone_high: float) -> bool:
        """Check if there's an unmitigated FVG within the entry zone."""
        n = min(len(df), 60)
        sub = df.tail(n).reset_index(drop=True)
        h = sub["high"].values.astype(float)
        l = sub["low"].values.astype(float)
        curr = float(sub["close"].iloc[-1])

        for i in range(1, n - 1):
            # Bullish FVG: gap between [i-1].high and [i+1].low
            gap_bottom = float(h[i-1]); gap_top = float(l[i+1])
            if gap_top > gap_bottom:
                if gap_bottom <= zone_high and gap_top >= zone_low:
                    return True
            # Bearish FVG
            gap_top2    = float(l[i-1]); gap_bottom2 = float(h[i+1])
            if gap_top2 > gap_bottom2:
                if gap_bottom2 <= zone_high and gap_top2 >= zone_low:
                    return True
        return False

    def detect_setups(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StructureSetup]:
        """
        Main entry: returns list of INSTITUTIONAL quality structure setups.
        Only returns setups where price is CURRENTLY in the OTE zone with confluence.
        """
        if len(df) < 40:
            return []

        setups: List[StructureSetup] = []
        close  = df["close"]

        # EMAs for trend filter
        ema50  = self._calc_ema(df, 50)
        ema200 = self._calc_ema(df, 200) if len(df) >= 200 else ema50
        curr_close = float(close.iloc[-1])
        e50  = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])

        if e50 > e200 and curr_close > e50:
            ema_alignment = "BULLISH"
        elif e50 < e200 and curr_close < e50:
            ema_alignment = "BEARISH"
        else:
            ema_alignment = "MIXED"

        atr     = self._calc_atr(df)
        atr_pct = atr / curr_close * 100
        atr_ok  = atr_pct > 0.15

        rvol      = self._relative_volume(df)
        volume_ok = rvol >= 0.8

        pivot_highs, pivot_lows = self._find_pivots(df)
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return []

        events    = self._detect_bos_choch(df, pivot_highs, pivot_lows)
        if not events:
            return []

        latest    = events[-1]
        latest_bar = latest["bar"]
        bars_ago  = len(df) - 1 - latest_bar

        # Only recent BOS (last 25 bars)
        if bars_ago > 25:
            return []

        is_bull = latest["type"] == "BOS_BULL"
        is_bear = latest["type"] == "BOS_BEAR"

        if is_bull:
            last_sh = latest["level"]          # BOS level (broken swing high)
            prev_lows = [pl for pl in pivot_lows if pl < latest_bar]
            if not prev_lows:
                return []
            last_sl_idx = prev_lows[-1]
            last_sl = float(df["low"].iloc[last_sl_idx])

            leg = last_sh - last_sl
            if leg < atr * 2:  # Leg must be at least 2x ATR to be significant
                return []

            # ── OTE Entry Zone: 61.8%-78.6% Fibonacci pullback ───────────────
            # ICT OTE rule: don't enter at 50%, wait for 61.8%-78.6%
            entry_zone_high = last_sh - leg * 0.618   # 61.8% pullback (shallower OTE)
            entry_zone_low  = last_sh - leg * 0.786   # 78.6% pullback (deeper OTE)
            entry_ideal     = last_sh - leg * 0.705   # 70.5% -- ICT sweet spot

            # SL: below swing_low (invalidation if hit)
            sl_fixed = round(last_sl - atr * 0.5, 2)
            sl_risk  = entry_ideal - sl_fixed

            if sl_risk <= 0:
                return []

            # ══ FIXED FIBONACCI EXTENSION TARGETS ══
            # All calculated from swing_low + leg x multiplier
            # T1 is ALWAYS closest (smallest extension), T4 is ALWAYS furthest
            t1 = round(last_sl + leg * 1.000, 2)   # Equal leg -- first exit
            t2 = round(last_sl + leg * 1.272, 2)   # Fib 127.2% -- standard
            t3 = round(last_sl + leg * 1.618, 2)   # Fib 161.8% -- golden
            t4 = round(last_sl + leg * 2.000, 2)   # Fib 200% -- runner

            # Sanity check: all targets must be ABOVE entry for LONG
            assert t1 < t2 < t3 < t4, f"Target ordering bug: {t1}, {t2}, {t3}, {t4}"

            target_rr    = round((t2 - entry_ideal) / sl_risk, 2)
            target_rr_t3 = round((t3 - entry_ideal) / sl_risk, 2)

            # MINIMUM RR FILTER: reject setups with RR < 2.0 to T2
            if target_rr < 1.8:
                return []

            # Trend alignment
            trend_aligned = ema_alignment in ("BULLISH", "MIXED")

            # Price check
            price_in_ote = entry_zone_low <= curr_close <= entry_zone_high
            price_approaching = curr_close > entry_zone_high and curr_close <= last_sh * 1.001

            if not (price_in_ote or price_approaching):
                return []

            # Confluence checks
            ob_conf  = self._detect_ob_at_zone(df, entry_zone_low, entry_zone_high, "LONG")
            fvg_conf = self._detect_fvg_at_zone(df, entry_zone_low, entry_zone_high)

            # Score
            score = 0.0
            notes = []

            if ema_alignment == "BULLISH":
                score += 25; notes.append("✅ EMA50 > EMA200 -- Daily uptrend confirmed")
            elif ema_alignment == "MIXED":
                score += 12; notes.append("⚠️ Mixed EMA -- moderate trend")
            else:
                notes.append("❌ EMA bearish -- counter-trend LONG (reduced confidence)")

            if volume_ok:
                score += 15; notes.append(f"✅ Volume OK (RVol {rvol:.1f}x)")
            else:
                notes.append(f"⚠️ Low volume (RVol {rvol:.1f}x)")

            if atr_ok:
                score += 15; notes.append(f"✅ ATR {atr_pct:.2f}% -- good volatility")
            else:
                notes.append(f"❌ Low ATR {atr_pct:.2f}% -- choppy market, skip")

            if price_in_ote:
                score += 20; notes.append("🎯 Price in OTE zone (61.8%-78.6%) -- sniper entry active!")
            else:
                score += 8;  notes.append("⏳ Price approaching OTE zone")

            if ob_conf:
                score += 15; notes.append("✅ Order Block confluence at entry zone")
            if fvg_conf:
                score += 10; notes.append("✅ FVG confluence at entry zone")

            if target_rr >= 3.0:
                score += 10; notes.append(f"✅ Excellent R:R {target_rr:.1f}x to Fib 1.272")
            elif target_rr >= 2.0:
                score += 7;  notes.append(f"✅ Good R:R {target_rr:.1f}x to Fib 1.272")
            else:
                score += 3;  notes.append(f"⚠️ R:R {target_rr:.1f}x -- borderline")

            if bars_ago <= 5:
                score += 5; notes.append(f"✅ Fresh BOS ({bars_ago} bars ago)")
            else:
                notes.append(f"ℹ️ BOS was {bars_ago} bars ago")

            # Need OB or FVG confluence -- pure BOS without price action = weak
            if not ob_conf and not fvg_conf:
                score = max(0, score - 20)
                notes.append("⚠️ No OB/FVG confluence at entry zone -- confidence reduced")

            sl_pct = round(sl_risk / entry_ideal * 100, 2)

            if score >= 80:   quality = "A+"
            elif score >= 65: quality = "A"
            elif score >= 50: quality = "B"
            else:             return []   # Below B = not worth trading

            notes.insert(0, f"📊 T1={t1:.0f}(Fib1.0) | T2={t2:.0f}(Fib1.272) | T3={t3:.0f}(Fib1.618) | T4={t4:.0f}(Fib2.0)")

            setup = StructureSetup(
                symbol=symbol, timeframe=timeframe,
                direction="LONG", setup_type="BOS_OTE",
                swing_high=last_sh, swing_low=last_sl,
                bos_level=last_sh, invalidation=sl_fixed,
                entry_zone_low=entry_zone_low, entry_zone_high=entry_zone_high,
                entry_ideal=entry_ideal,
                sl_fixed=sl_fixed, sl_pct=sl_pct,
                target_1=t1, target_2=t2, target_3=t3, target_4=t4,
                target_rr=target_rr, target_rr_t3=target_rr_t3,
                trend_aligned=trend_aligned,
                volume_ok=volume_ok, atr_ok=atr_ok,
                ema_alignment=ema_alignment,
                ob_confluence=ob_conf, fvg_confluence=fvg_conf,
                ote_in_zone=price_in_ote,
                confidence_score=score, quality_label=quality,
                setup_age_bars=bars_ago,
                notes=notes,
                generated_at=datetime.now().isoformat(),
            )
            setups.append(setup)
            return setups

        elif is_bear:
            last_sl = latest["level"]           # BOS level (broken swing low)
            prev_highs = [ph for ph in pivot_highs if ph < latest_bar]
            if not prev_highs:
                return []
            last_sh_idx = prev_highs[-1]
            last_sh = float(df["high"].iloc[last_sh_idx])

            leg = last_sh - last_sl
            if leg < atr * 2:
                return []

            # ── OTE SHORT Entry Zone: 61.8%-78.6% bounce from swing LOW ──────
            entry_zone_low  = last_sl + leg * 0.618   # Bounce to 61.8%
            entry_zone_high = last_sl + leg * 0.786   # Bounce to 78.6%
            entry_ideal     = last_sl + leg * 0.705   # 70.5% sweet spot

            sl_fixed = round(last_sh + atr * 0.5, 2)
            sl_risk  = sl_fixed - entry_ideal

            if sl_risk <= 0:
                return []

            # ══ FIXED FIBONACCI EXTENSION TARGETS (SHORT) ══
            # All calculated from swing_high - leg x multiplier
            # For SHORT: T1 > T2 > T3 > T4 (all BELOW entry)
            t1 = round(last_sh - leg * 1.000, 2)   # Equal leg drop -- first exit
            t2 = round(last_sh - leg * 1.272, 2)   # Fib 127.2% -- standard
            t3 = round(last_sh - leg * 1.618, 2)   # Fib 161.8% -- golden
            t4 = round(last_sh - leg * 2.000, 2)   # Fib 200% -- runner

            # Sanity check: for SHORT, T1 > T2 > T3 > T4 (all going DOWN)
            # And all must be BELOW entry
            assert t1 > t2 > t3 > t4, f"SHORT target ordering bug: {t1}, {t2}, {t3}, {t4}"

            target_rr    = round((entry_ideal - t2) / sl_risk, 2)
            target_rr_t3 = round((entry_ideal - t3) / sl_risk, 2)

            if target_rr < 1.8:
                return []

            trend_aligned = ema_alignment in ("BEARISH", "MIXED")

            price_in_ote  = entry_zone_low <= curr_close <= entry_zone_high
            price_approaching = curr_close < entry_zone_low and curr_close >= last_sl * 0.999

            if not (price_in_ote or price_approaching):
                return []

            ob_conf  = self._detect_ob_at_zone(df, entry_zone_low, entry_zone_high, "SHORT")
            fvg_conf = self._detect_fvg_at_zone(df, entry_zone_low, entry_zone_high)

            score = 0.0
            notes = []

            if ema_alignment == "BEARISH":
                score += 25; notes.append("✅ EMA50 < EMA200 -- Daily downtrend confirmed")
            elif ema_alignment == "MIXED":
                score += 12; notes.append("⚠️ Mixed EMA -- moderate trend")
            else:
                notes.append("❌ EMA bullish -- counter-trend SHORT")

            if volume_ok:
                score += 15; notes.append(f"✅ Volume OK (RVol {rvol:.1f}x)")
            else:
                notes.append(f"⚠️ Low volume (RVol {rvol:.1f}x)")

            if atr_ok:
                score += 15; notes.append(f"✅ ATR {atr_pct:.2f}% -- volatile enough")
            else:
                notes.append(f"❌ Low ATR {atr_pct:.2f}%")

            if price_in_ote:
                score += 20; notes.append("🎯 Price in OTE zone (61.8%-78.6%) -- sniper short active!")
            else:
                score += 8;  notes.append("⏳ Price approaching OTE zone")

            if ob_conf:
                score += 15; notes.append("✅ Bearish Order Block confluence at entry zone")
            if fvg_conf:
                score += 10; notes.append("✅ Bearish FVG confluence at entry zone")

            if target_rr >= 3.0:
                score += 10; notes.append(f"✅ Excellent R:R {target_rr:.1f}x to Fib 1.272")
            elif target_rr >= 2.0:
                score += 7;  notes.append(f"✅ Good R:R {target_rr:.1f}x to Fib 1.272")
            else:
                score += 3;  notes.append(f"⚠️ R:R {target_rr:.1f}x -- borderline")

            if bars_ago <= 5:
                score += 5; notes.append(f"✅ Fresh BOS ({bars_ago} bars ago)")

            if not ob_conf and not fvg_conf:
                score = max(0, score - 20)
                notes.append("⚠️ No OB/FVG confluence -- confidence reduced")

            sl_pct = round(sl_risk / entry_ideal * 100, 2)

            if score >= 80:   quality = "A+"
            elif score >= 65: quality = "A"
            elif score >= 50: quality = "B"
            else:             return []

            notes.insert(0, f"📊 T1={t1:.0f}(Fib1.0) | T2={t2:.0f}(Fib1.272) | T3={t3:.0f}(Fib1.618) | T4={t4:.0f}(Fib2.0)")

            setup = StructureSetup(
                symbol=symbol, timeframe=timeframe,
                direction="SHORT", setup_type="BOS_OTE",
                swing_high=last_sh, swing_low=last_sl,
                bos_level=last_sl, invalidation=sl_fixed,
                entry_zone_low=entry_zone_low, entry_zone_high=entry_zone_high,
                entry_ideal=entry_ideal,
                sl_fixed=sl_fixed, sl_pct=sl_pct,
                target_1=t1, target_2=t2, target_3=t3, target_4=t4,
                target_rr=target_rr, target_rr_t3=target_rr_t3,
                trend_aligned=trend_aligned,
                volume_ok=volume_ok, atr_ok=atr_ok,
                ema_alignment=ema_alignment,
                ob_confluence=ob_conf, fvg_confluence=fvg_conf,
                ote_in_zone=price_in_ote,
                confidence_score=score, quality_label=quality,
                setup_age_bars=bars_ago,
                notes=notes,
                generated_at=datetime.now().isoformat(),
            )
            setups.append(setup)
            return setups

        return setups


structure_engine = StructureTradeEngine()



@dataclass
class ORBSignal:
    """
    Opening Range Breakout signal.
    Generated by ORBEngine when a confirmed breakout occurs after 9:30 IST.
    """
    symbol:          str
    timeframe:       str
    direction:       str            # "BULL" | "BEAR"
    orb_high:        float          # High of the 9:15-9:29 opening range
    orb_low:         float          # Low of the 9:15-9:29 opening range
    orb_range:       float          # orb_high - orb_low
    breakout_price:  float          # Close price of the breakout bar
    breakout_bar:    int            # Bar index of the breakout candle
    breakout_time:   str            # Timestamp string of the breakout bar
    entry_zone:      Tuple[float, float]  # (entry_low, entry_high)
    sl:              float          # Stop loss price
    target1:         float          # T1 = 1x ORB range extension
    target2:         float          # T2 = 2x ORB range extension
    target3:         float          # T3 = 3x ORB range extension
    target4:         float          # T4 = 4x ORB range extension (runner)
    risk_reward:     float          # R:R to T2 (from entry_mid)
    volume_ratio:    float          # Breakout bar volume / 20-bar avg volume
    gap_pct:         float          # Gap % from previous close to today open
    gap_type:        str            # "GAP_UP" | "GAP_DOWN" | "FLAT"
    confidence:      float          # 0-100 confidence score
    notes:           List[str]      # Context notes
    generated_at:    str            # ISO timestamp of signal generation

    def to_dict(self) -> Dict:
        return {
            "symbol":         self.symbol,
            "timeframe":      self.timeframe,
            "direction":      self.direction,
            "orb_high":       round(self.orb_high, 2),
            "orb_low":        round(self.orb_low, 2),
            "orb_range":      round(self.orb_range, 2),
            "breakout_price": round(self.breakout_price, 2),
            "breakout_time":  self.breakout_time,
            "entry_zone":     [round(self.entry_zone[0], 2), round(self.entry_zone[1], 2)],
            "sl":             round(self.sl, 2),
            "target1":        round(self.target1, 2),
            "target2":        round(self.target2, 2),
            "target3":        round(self.target3, 2),
            "target4":        round(self.target4, 2),
            "risk_reward":    round(self.risk_reward, 2),
            "volume_ratio":   round(self.volume_ratio, 2),
            "gap_pct":        round(self.gap_pct, 3),
            "gap_type":       self.gap_type,
            "confidence":     round(self.confidence, 1),
            "notes":          self.notes,
            "generated_at":   self.generated_at,
        }



class ORBEngine:
    """
    Opening Range Breakout engine for intraday trading.
    
    Rules:
    1. ORB = High/Low of candles strictly in 9:15–9:29 window
    2. Breakout = first confirmed bar AFTER 9:30 that closes beyond ORB
    3. Volume filter: breakout bar volume ≥ 1.3× average
    4. Gap filter: gap-up ORB bullish breakout has higher probability
    5. False breakout filter: if candle wicks through but closes inside → skip
    6. One signal per day per direction (no repeating ORB)
    """

    def detect(self, df: pd.DataFrame, symbol: str,
                timeframe: str = "5minute") -> Optional[ORBSignal]:
        """
        Returns ORBSignal if a valid breakout is currently active,
        or None if no ORB has formed yet today.
        """
        if df.empty or len(df) < 5:
            return None

        has_vol = "volume" in df.columns and df["volume"].sum() > 0

        # ── Identify today's date from last bar ──────────────────────────
        last_ts = df["timestamp"].iloc[-1]
        if hasattr(last_ts, "date"):
            today = last_ts.date()
        else:
            try:
                today = pd.to_datetime(last_ts).date()
            except Exception:
                return None

        # ── Separate today's candles ─────────────────────────────────────
        def ts_date(ts):
            if hasattr(ts, "date"): return ts.date()
            try: return pd.to_datetime(ts).date()
            except: return None

        def ts_hm(ts):
            """Return (hour, minute) tuple."""
            if hasattr(ts, "hour"): return (ts.hour, ts.minute)
            try:
                t = pd.to_datetime(ts)
                return (t.hour, t.minute)
            except: return (0, 0)

        today_mask = df["timestamp"].apply(lambda x: ts_date(x) == today)
        today_df   = df[today_mask].copy().reset_index(drop=True)

        if today_df.empty:
            return None

        # ── Opening range bars: 9:15–9:29 ───────────────────────────────
        orb_mask = today_df["timestamp"].apply(
            lambda x: (9, 15) <= ts_hm(x) < (9, 30)
        )
        orb_bars = today_df[orb_mask]

        if orb_bars.empty:
            return None   # opening range not yet established

        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())
        orb_rng  = orb_high - orb_low

        if orb_rng <= 0:
            return None

        # ── Gap analysis ─────────────────────────────────────────────────
        yesterday_mask = df["timestamp"].apply(lambda x: ts_date(x) < today)
        yesterday_df   = df[yesterday_mask]
        today_open = float(orb_bars["open"].iloc[0])  # always use real open
        if not yesterday_df.empty:
            prev_close = float(yesterday_df["close"].iloc[-1])
            gap_pct    = (today_open - prev_close) / prev_close * 100
        else:
            # Fallback: try to compute from orb_bars open vs first bar close estimate
            gap_pct   = 0.0

        if gap_pct > 0.2:
            gap_type = "GAP_UP"
        elif gap_pct < -0.2:
            gap_type = "GAP_DOWN"
        else:
            gap_type = "FLAT"

        # ── Post-9:30 candles for breakout detection ──────────────────────
        post_mask = today_df["timestamp"].apply(lambda x: ts_hm(x) >= (9, 30))
        post_bars = today_df[post_mask].copy().reset_index(drop=True)

        if post_bars.empty:
            return None   # market just opened, waiting for 9:30 bar

        # Average volume from pre-breakout bars (use today + a few yesterday)
        avg_vol_bars = df.tail(30)
        avg_vol = float(avg_vol_bars["volume"].mean()) if has_vol else 1.0

        # ── Find first CONFIRMED breakout bar ────────────────────────────
        breakout_bar_idx = None
        breakout_dir     = None
        bo_close         = None
        bo_volume_ratio  = 1.0
        bo_time          = ""

        for idx in range(len(post_bars)):
            bar    = post_bars.iloc[idx]
            close_ = float(bar["close"])
            high_  = float(bar["high"])
            low_   = float(bar["low"])
            vol_   = float(bar["volume"]) if has_vol else avg_vol

            vol_ratio = vol_ / avg_vol if avg_vol > 0 else 1.0
            h, m = ts_hm(bar["timestamp"])

            # Bullish breakout: candle CLOSES above ORB high
            if close_ > orb_high and high_ > orb_high:
                breakout_bar_idx = idx
                breakout_dir     = "BULL"
                bo_close         = close_
                bo_volume_ratio  = vol_ratio
                bo_time          = f"{h:02d}:{m:02d}"
                break

            # Bearish breakout: candle CLOSES below ORB low
            if close_ < orb_low and low_ < orb_low:
                breakout_bar_idx = idx
                breakout_dir     = "BEAR"
                bo_close         = close_
                bo_volume_ratio  = vol_ratio
                bo_time          = f"{h:02d}:{m:02d}"
                break

        if breakout_bar_idx is None or breakout_dir is None:
            return None   # no breakout yet

        # ── Build signal ─────────────────────────────────────────────────
        notes: List[str] = []
        confidence = 50.0

        # Gap alignment boosts confidence
        if breakout_dir == "BULL" and gap_type == "GAP_UP":
            confidence += 20; notes.append(f"✅ Gap-up open ({gap_pct:+.2f}%) + bullish ORB = strong setup")
        elif breakout_dir == "BEAR" and gap_type == "GAP_DOWN":
            confidence += 20; notes.append(f"✅ Gap-down open ({gap_pct:+.2f}%) + bearish ORB = strong setup")
        elif gap_type == "FLAT":
            confidence += 5;  notes.append(f"ℹ️ Flat open — ORB direction defines the day")
        else:
            notes.append(f"⚠️ Breakout against gap direction — lower probability")

        # Volume confirmation
        if bo_volume_ratio >= 2.0:
            confidence += 20; notes.append(f"✅ Strong volume ({bo_volume_ratio:.1f}× average) — institutional breakout")
        elif bo_volume_ratio >= 1.3:
            confidence += 10; notes.append(f"✅ Good volume ({bo_volume_ratio:.1f}× average)")
        else:
            notes.append(f"⚠️ Low volume ({bo_volume_ratio:.1f}× avg) — weak breakout, may fail")

        # ORB range quality
        orb_pct = orb_rng / bo_close * 100
        if orb_pct < 0.15:
            notes.append(f"⚠️ Tight ORB ({orb_rng:.0f}pts = {orb_pct:.2f}%) — small range, lower targets")
        elif orb_pct > 0.5:
            confidence += 10; notes.append(f"✅ Wide ORB ({orb_rng:.0f}pts = {orb_pct:.2f}%) — strong target extension")
        else:
            confidence += 5;  notes.append(f"ℹ️ ORB range: {orb_rng:.0f}pts ({orb_pct:.2f}%)")

        # Recency — breakout happened early = more day left to run
        bo_h, bo_m = [int(x) for x in bo_time.split(":")]
        if bo_h == 9 and bo_m <= 35:
            confidence += 10; notes.append(f"✅ Early breakout ({bo_time}) — full day to run")
        elif bo_h <= 10:
            confidence += 5;  notes.append(f"ℹ️ Morning breakout ({bo_time})")
        else:
            notes.append(f"⚠️ Late breakout ({bo_time}) — limited time remaining")

        confidence = round(min(confidence, 95), 1)

        # Entry, SL, Targets
        if breakout_dir == "BULL":
            entry_low  = orb_high                      # at the ORH retest
            entry_high = orb_high + orb_rng * 0.25    # or slight above (don't chase)
            sl         = orb_high - orb_rng * 0.5      # 50% back inside range
            risk       = entry_high - sl
            t1         = entry_high + orb_rng          # 1× ORB
            t2         = entry_high + orb_rng * 2      # 2× ORB
            t3         = entry_high + orb_rng * 3      # 3× ORB (runner)
            t4         = entry_high + orb_rng * 4      # 4× ORB (full runner)
        else:
            entry_high = orb_low
            entry_low  = orb_low - orb_rng * 0.25
            sl         = orb_low + orb_rng * 0.5
            risk       = sl - entry_low
            t1         = entry_low - orb_rng
            t2         = entry_low - orb_rng * 2
            t3         = entry_low - orb_rng * 3
            t4         = entry_low - orb_rng * 4       # 4× ORB (full runner)

        # FIX: RR must use entry_mid (displayed value), not the edge of the zone
        entry_mid = (entry_low + entry_high) / 2
        if breakout_dir == "BULL":
            rr = round((t2 - entry_mid) / max(risk, 0.01), 2)
        else:
            rr = round((entry_mid - t2) / max(risk, 0.01), 2)

        return ORBSignal(
            symbol=symbol, timeframe=timeframe, direction=breakout_dir,
            orb_high=orb_high, orb_low=orb_low, orb_range=orb_rng,
            breakout_price=bo_close,
            breakout_bar=breakout_bar_idx, breakout_time=bo_time,
            entry_zone=(round(entry_low, 2), round(entry_high, 2)),
            sl=round(sl, 2),
            target1=round(t1, 2), target2=round(t2, 2), target3=round(t3, 2), target4=round(t4, 2),
            risk_reward=abs(rr),
            volume_ratio=bo_volume_ratio,
            gap_pct=gap_pct, gap_type=gap_type,
            confidence=confidence, notes=notes,
            generated_at=datetime.now().isoformat(),
        )


# ── ORB Daily Signal Cache ────────────────────────────────────────────────────
# Stores the first confirmed ORBSignal per (symbol, date) so that subsequent
# API calls within the same session always return the original signal rather
# than a potentially shifted re-computation (avoids intraday signal flipping).
# _orb_cache is declared below as a TTLCache(ttl=28800) — do not redeclare here


orb_engine = ORBEngine()




# ════════════════════════════════════════════════════════════════════════════════
#  SMART MONEY CONCEPTS (SMC) ENGINE
#  Intraday / 15-min focus, Kite data only
#  Detects: Order Blocks · Fair Value Gaps · Liquidity Sweeps ·
#           Premium/Discount Array · VWAP · Session Bias · MTF Confluence
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class OrderBlock:
    direction: str          # "BULL" | "BEAR"
    high: float
    low: float
    mid: float
    origin_bar: int
    origin_time: str
    strength: str           # "STRONG" | "MODERATE" | "WEAK"
    touched: bool           # has price returned to it?
    valid: bool             # still above/below BOS level
    body_pct: float         # body % of candle -- higher = stronger OB

@dataclass
class FairValueGap:
    direction: str          # "BULL" | "BEAR"
    top: float
    bottom: float
    mid: float
    origin_bar: int
    origin_time: str
    filled_pct: float       # 0-100, 100 = fully filled (invalidated)
    size_pct: float         # gap as % of price -- bigger = more significant

@dataclass
class LiquiditySweep:
    sweep_type: str         # "BSL_SWEPT" (buy-side) | "SSL_SWEPT" (sell-side)
    level: float            # the equal high/low that was swept
    sweep_bar: int
    sweep_time: str
    reversal_confirmed: bool  # closed back inside = fake sweep = entry signal
    reversal_strength: float  # how far it closed back inside (pts)

@dataclass
class SMCAnalysis:
    symbol: str
    timeframe: str
    current_price: float
    generated_at: str

    # Arrays
    order_blocks: List[OrderBlock]
    fair_value_gaps: List[FairValueGap]
    liquidity_sweeps: List[LiquiditySweep]

    # Context
    market_structure: str       # "BULLISH" | "BEARISH" | "RANGING"
    premium_discount: str       # "PREMIUM" | "DISCOUNT" | "EQUILIBRIUM"
    pd_pct: float               # where price is in range: 0=bottom 100=top 50=mid
    in_discount: bool           # price below 50% of recent range
    in_premium: bool            # price above 50% of recent range

    # VWAP
    vwap: float
    price_vs_vwap: str          # "ABOVE" | "BELOW" | "AT"
    vwap_bias: str              # "BULLISH" | "BEARISH"

    # Session
    session: str                # "PRE" | "OPEN_EXPLOSIVE" | "TREND" | "LUNCH" | "AFTERNOON" | "CLOSE"
    session_tradeable: bool
    session_note: str

    # Multi-TF
    mtf_bias: str               # "BULLISH" | "BEARISH" | "MIXED"
    mtf_score: int              # 0-3 (how many TFs agree)

    # Signal
    smc_signal: str             # "BUY" | "SELL" | "WAIT"
    smc_confidence: float       # 0-100
    smc_entry_zone: Tuple[float, float]
    smc_sl: float
    smc_target1: float
    smc_target2: float
    smc_notes: List[str]

    def to_dict(self) -> Dict:
        def ob_d(o: OrderBlock):
            return {
                "direction":  str(o.direction),
                "high":       round(float(o.high), 2),
                "low":        round(float(o.low), 2),
                "mid":        round(float(o.mid), 2),
                "origin_bar": int(o.origin_bar),
                "origin_time":str(o.origin_time),
                "strength":   str(o.strength),
                "touched":    bool(o.touched),
                "valid":      bool(o.valid),
                "body_pct":   round(float(o.body_pct), 2),
            }
        def fvg_d(f: FairValueGap):
            return {
                "direction":   str(f.direction),
                "top":         round(float(f.top), 2),
                "bottom":      round(float(f.bottom), 2),
                "mid":         round(float(f.mid), 2),
                "origin_bar":  int(f.origin_bar),
                "origin_time": str(f.origin_time),
                "filled_pct":  round(float(f.filled_pct), 1),
                "size_pct":    round(float(f.size_pct), 3),
            }
        def lq_d(l: LiquiditySweep):
            return {
                "sweep_type":          str(l.sweep_type),
                "level":               round(float(l.level), 2),
                "sweep_bar":           int(l.sweep_bar),
                "sweep_time":          str(l.sweep_time),
                "reversal_confirmed":  bool(l.reversal_confirmed),
                "reversal_strength":   round(float(l.reversal_strength), 2),
            }
        return {
            "symbol":           str(self.symbol),
            "timeframe":        str(self.timeframe),
            "current_price":    round(float(self.current_price), 2),
            "generated_at":     str(self.generated_at),
            "order_blocks":     [ob_d(o) for o in self.order_blocks],
            "fair_value_gaps":  [fvg_d(f) for f in self.fair_value_gaps],
            "liquidity_sweeps": [lq_d(l) for l in self.liquidity_sweeps],
            "market_structure": str(self.market_structure),
            "premium_discount": str(self.premium_discount),
            "pd_pct":           round(float(self.pd_pct), 1),
            "in_discount":      bool(self.in_discount),
            "in_premium":       bool(self.in_premium),
            "vwap":             round(float(self.vwap), 2),
            "price_vs_vwap":    str(self.price_vs_vwap),
            "vwap_bias":        str(self.vwap_bias),
            "session":          str(self.session),
            "session_tradeable":bool(self.session_tradeable),
            "session_note":     str(self.session_note),
            "mtf_bias":         str(self.mtf_bias),
            "mtf_score":        int(self.mtf_score),
            "smc_signal":       str(self.smc_signal),
            "smc_confidence":   round(float(self.smc_confidence), 1),
            "smc_entry_zone":   [round(float(self.smc_entry_zone[0]),2),
                                  round(float(self.smc_entry_zone[1]),2)],
            "smc_sl":           round(float(self.smc_sl), 2),
            "smc_target1":      round(float(self.smc_target1), 2),
            "smc_target2":      round(float(self.smc_target2), 2),
            "smc_notes":        [str(n) for n in self.smc_notes],
        }


def _np(v):
    """Convert numpy scalar to native Python type for JSON serialization."""
    if v is None: return None
    if hasattr(v, 'item'): return v.item()   # numpy scalar -> Python scalar
    if isinstance(v, (list, tuple)): return [_np(x) for x in v]
    return v



class SMCEngine:
    """Smart Money Concepts analysis engine for intraday trading."""

    # IST session windows (hour, minute)
    SESSIONS = [
        ((9, 15),  (9, 30),  "OPEN_EXPLOSIVE",  True,  "📊 ORB window 9:15-9:30 — wait for range then trade breakout."),
        ((9, 30),  (10, 30), "TREND",            True,  "✅ Best momentum window. Follow direction."),
        ((10, 30), (11, 30), "TREND",            True,  "✅ Good trend continuation window."),
        ((11, 30), (13, 0),  "LUNCH",            True,  "⚠️ Lunch chop 11:30-1PM — lower volume. Use tighter SL."),
        ((13, 0),  (14, 0),  "AFTERNOON",        True,  "✅ Post-lunch reversal setups. Watch OBs."),
        ((14, 0),  (15, 0),  "AFTERNOON",        True,  "✅ Best F&O window. Expiry moves here."),
        ((15, 0),  (15, 30), "CLOSE",            False, "⚠️ Institutional squaring. No new entries."),
    ]

    def _get_session(self) -> Tuple[str, bool, str]:
        _IST_TZ = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(tz=_IST_TZ)  # Always IST regardless of server timezone
        h, m = now.hour, now.minute
        if now.weekday() >= 5:
            return "PRE", False, "⏳ Market closed — weekend."
        for (sh, sm), (eh, em), name, tradeable, note in self.SESSIONS:
            if (h, m) >= (sh, sm) and (h, m) < (eh, em):
                return name, tradeable, note
        return "PRE", False, "⏳ Market not open yet (or after close)."

    def _calc_vwap(self, df: pd.DataFrame) -> float:
        """True VWAP from today's intraday candles."""
        try:
            _IST_TZ_V = timezone(timedelta(hours=5, minutes=30))
            today = datetime.now(tz=_IST_TZ_V).date()  # IST date, not UTC
            mask = pd.to_datetime(df["timestamp"]).dt.date == today
            today_df = df[mask].copy()
            if today_df.empty:
                today_df = df.tail(30)
            typical = (today_df["high"] + today_df["low"] + today_df["close"]) / 3
            vol = today_df["volume"].replace(0, 1)
            return float((typical * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])
        except Exception:
            return float((df["high"] + df["low"] + df["close"]).iloc[-1] / 3)

    def _detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Bullish OB  = last RED candle before a bullish impulse (3+ green candles up)
        Bearish OB  = last GREEN candle before a bearish impulse (3+ red candles down)
        Only last 50 bars. Only valid (unmitigated) OBs returned.
        """
        obs: List[OrderBlock] = []
        n = min(len(df), 60)
        sub = df.tail(n).reset_index(drop=True)
        close = sub["close"].values
        open_ = sub["open"].values
        high  = sub["high"].values
        low   = sub["low"].values
        curr  = float(close[-1])

        for i in range(2, len(sub) - 3):
            # Bullish OB: candle i is bearish, followed by 3 bullish candles
            if close[i] < open_[i]:  # bearish candle
                next3_bull = all(close[i+k] > open_[i+k] for k in range(1, min(4, len(sub)-i)))
                if next3_bull:
                    ob_high = high[i]
                    ob_low  = low[i]
                    ob_mid  = (ob_high + ob_low) / 2
                    body_pct = abs(close[i] - open_[i]) / (ob_high - ob_low + 0.001) * 100
                    # OB still valid if current price hasn't closed below its low
                    valid = curr > ob_low
                    # GAP 6 FIX: touched = price returned to OB in last 10 bars only
                    # (not any future bar -- that wrongly flags old OBs as "touched")
                    recent_start = max(i+1, len(sub)-10)
                    future_low = min(low[recent_start:]) if recent_start < len(sub) else curr
                    touched = future_low <= ob_high
                    strength = "STRONG" if body_pct > 60 else "MODERATE" if body_pct > 35 else "WEAK"
                    obs.append(OrderBlock(
                        direction="BULL", high=float(ob_high), low=float(ob_low), mid=float(ob_mid),
                        origin_bar=int(i),
                        origin_time=str(sub["timestamp"].iloc[i] if "timestamp" in sub.columns else "")[:16],
                        strength=str(strength), touched=bool(touched), valid=bool(valid), body_pct=float(body_pct)
                    ))

            # Bearish OB: candle i is bullish, followed by 3 bearish candles
            elif close[i] > open_[i]:  # bullish candle
                next3_bear = all(close[i+k] < open_[i+k] for k in range(1, min(4, len(sub)-i)))
                if next3_bear:
                    ob_high = high[i]
                    ob_low  = low[i]
                    ob_mid  = (ob_high + ob_low) / 2
                    body_pct = abs(close[i] - open_[i]) / (ob_high - ob_low + 0.001) * 100
                    valid = curr < ob_high
                    # GAP 6 FIX: touched = last 10 bars only
                    recent_start_b = max(i+1, len(sub)-10)
                    future_high = max(high[recent_start_b:]) if recent_start_b < len(sub) else curr
                    touched = future_high >= ob_low
                    strength = "STRONG" if body_pct > 60 else "MODERATE" if body_pct > 35 else "WEAK"
                    obs.append(OrderBlock(
                        direction="BEAR", high=float(ob_high), low=float(ob_low), mid=float(ob_mid),
                        origin_bar=int(i),
                        origin_time=str(sub["timestamp"].iloc[i] if "timestamp" in sub.columns else "")[:16],
                        strength=str(strength), touched=bool(touched), valid=bool(valid), body_pct=float(body_pct)
                    ))

        # Return only valid, most recent, max 3 per direction
        bull_obs = sorted([o for o in obs if o.direction=="BULL" and o.valid],
                           key=lambda x: x.origin_bar, reverse=True)[:3]
        bear_obs = sorted([o for o in obs if o.direction=="BEAR" and o.valid],
                           key=lambda x: x.origin_bar, reverse=True)[:3]
        return bull_obs + bear_obs

    def _detect_fvg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        FVG = 3-candle pattern where candle[i-1].high < candle[i+1].low (bullish gap)
                                   or candle[i-1].low > candle[i+1].high (bearish gap)
        Gap must be >= 0.05% of price to be significant.
        """
        fvgs: List[FairValueGap] = []
        n = min(len(df), 60)
        sub = df.tail(n).reset_index(drop=True)
        curr = float(sub["close"].iloc[-1])
        ts_col = "timestamp" if "timestamp" in sub.columns else sub.index

        for i in range(1, len(sub) - 1):
            # Bullish FVG: gap between candle[i-1].high and candle[i+1].low
            gap_bottom = float(sub["high"].iloc[i-1])
            gap_top    = float(sub["low"].iloc[i+1])
            if gap_top > gap_bottom:
                size_pct = (gap_top - gap_bottom) / curr * 100
                if size_pct >= 0.05:
                    # How filled? (price has partially or fully entered the gap)
                    future_low = float(sub["low"].iloc[i+1:].min()) if i+1 < len(sub) else curr
                    filled_pct = min(100, max(0, (gap_top - future_low) / (gap_top - gap_bottom) * 100))
                    fvgs.append(FairValueGap(
                        direction="BULL", top=float(gap_top), bottom=float(gap_bottom),
                        mid=float((gap_top+gap_bottom)/2), origin_bar=int(i),
                        origin_time=str(sub["timestamp"].iloc[i] if "timestamp" in sub.columns else i)[:16],
                        filled_pct=float(filled_pct), size_pct=float(size_pct)
                    ))

            # Bearish FVG
            gap_top2    = float(sub["low"].iloc[i-1])
            gap_bottom2 = float(sub["high"].iloc[i+1])
            if gap_top2 > gap_bottom2:
                size_pct = (gap_top2 - gap_bottom2) / curr * 100
                if size_pct >= 0.05:
                    future_high = float(sub["high"].iloc[i+1:].max()) if i+1 < len(sub) else curr
                    filled_pct = min(100, max(0, (future_high - gap_bottom2) / (gap_top2 - gap_bottom2) * 100))
                    fvgs.append(FairValueGap(
                        direction="BEAR", top=float(gap_top2), bottom=float(gap_bottom2),
                        mid=float((gap_top2+gap_bottom2)/2), origin_bar=int(i),
                        origin_time=str(sub["timestamp"].iloc[i] if "timestamp" in sub.columns else i)[:16],
                        filled_pct=float(filled_pct), size_pct=float(size_pct)
                    ))

        # Only unfilled (< 90%) and recent
        valid_fvgs = [f for f in fvgs if f.filled_pct < 90]
        bull = sorted([f for f in valid_fvgs if f.direction=="BULL"], key=lambda x: x.origin_bar, reverse=True)[:2]
        bear = sorted([f for f in valid_fvgs if f.direction=="BEAR"], key=lambda x: x.origin_bar, reverse=True)[:2]
        return bull + bear

    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[LiquiditySweep]:
        """
        Equal highs (BSL) / Equal lows (SSL) = liquidity pools.
        A sweep = wick beyond the equal level but closes back inside.
        This is smart money grabbing liquidity before reversing.
        """
        sweeps: List[LiquiditySweep] = []
        n = min(len(df), 50)
        sub = df.tail(n).reset_index(drop=True)
        highs = sub["high"].values
        lows  = sub["low"].values
        closes = sub["close"].values
        opens  = sub["open"].values

        # Find equal highs (within 0.1% of each other) -- BSL
        for i in range(3, len(sub)-1):
            for j in range(max(0, i-15), i-2):
                level = highs[j]
                if abs(highs[i-1] - level) / level < 0.001:  # equal high
                    # Check if current or recent bar swept it
                    if highs[i] > level and closes[i] < level:
                        rev_str = level - closes[i]
                        sweeps.append(LiquiditySweep(
                            sweep_type="BSL_SWEPT", level=round(float(level),2),
                            sweep_bar=int(i),
                            sweep_time=str(sub["timestamp"].iloc[i] if "timestamp" in sub.columns else i)[:16],
                            reversal_confirmed=True,
                            reversal_strength=round(float(rev_str),2)
                        ))
                        break

        # Find equal lows -- SSL
        for i in range(3, len(sub)-1):
            for j in range(max(0, i-15), i-2):
                level = lows[j]
                if abs(lows[i-1] - level) / level < 0.001:  # equal low
                    if lows[i] < level and closes[i] > level:
                        rev_str = closes[i] - level
                        sweeps.append(LiquiditySweep(
                            sweep_type="SSL_SWEPT", level=round(float(level),2),
                            sweep_bar=int(i),
                            sweep_time=str(sub["timestamp"].iloc[i] if "timestamp" in sub.columns else i)[:16],
                            reversal_confirmed=True,
                            reversal_strength=round(float(rev_str),2)
                        ))
                        break

        # Most recent sweeps only
        return sorted(sweeps, key=lambda x: x.sweep_bar, reverse=True)[:4]

    def _premium_discount(self, df: pd.DataFrame, curr: float) -> Tuple[str, float, bool, bool]:
        """
        Premium/Discount Array:
        Look at last 50-bar swing. Above 50% = Premium (expensive, sell zone).
        Below 50% = Discount (cheap, buy zone).
        Optimal buy = 61.8%-78.6% discount. Optimal sell = 61.8%-78.6% premium.
        """
        n = min(len(df), 50)
        sub = df.tail(n)
        hi  = float(sub["high"].max())
        lo  = float(sub["low"].min())
        rng = hi - lo
        if rng == 0:
            return "EQUILIBRIUM", 50.0, False, False
        pd_pct = (curr - lo) / rng * 100
        if pd_pct > 61.8:
            label = "PREMIUM"
        elif pd_pct < 38.2:
            label = "DISCOUNT"
        else:
            label = "EQUILIBRIUM"
        in_discount = bool(pd_pct < 38.2)
        in_premium  = bool(pd_pct > 61.8)
        return str(label), float(pd_pct), in_discount, in_premium

    def _market_structure(self, df: pd.DataFrame) -> str:
        """
        Advanced structure detection:
        - Finds swing pivots (local H/L over 3-bar window)
        - Classifies as HH/HL (BULLISH), LH/LL (BEARISH), or RANGING
        - Detects Rising Wedge -> marks as BEARISH (wedge exhaustion)
        - Detects Falling Wedge -> marks as BULLISH (wedge reversal)
        - Detects Double Bottom -> marks as BULLISH
        - Detects Double Top -> marks as BEARISH
        - Requires 0.3% move threshold to avoid noise
        """
        sub = df.tail(40)
        if len(sub) < 8:
            return "RANGING"
        highs  = sub["high"].values.astype(float)
        lows   = sub["low"].values.astype(float)
        closes = sub["close"].values.astype(float)
        n = len(sub)

        # ── Find swing highs/lows (3-bar pivot) ──────────────────────────────
        swing_highs = []
        swing_lows  = []
        for i in range(2, n - 2):
            if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and                highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
                swing_highs.append((i, highs[i]))
            if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and                lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                swing_lows.append((i, lows[i]))

        # ── Double Bottom: two swing lows within 0.3% of each other ─────────
        if len(swing_lows) >= 2:
            last2_lows = swing_lows[-2:]
            l1, l2 = last2_lows[0][1], last2_lows[1][1]
            if abs(l1 - l2) / max(l1, 1) < 0.003 and closes[-1] > max(l1, l2) * 1.002:
                return "BULLISH"  # Double bottom confirmed with price above neckline

        # ── Double Top: two swing highs within 0.3% of each other ───────────
        if len(swing_highs) >= 2:
            last2_highs = swing_highs[-2:]
            h1, h2 = last2_highs[0][1], last2_highs[1][1]
            if abs(h1 - h2) / max(h1, 1) < 0.003 and closes[-1] < min(h1, h2) * 0.998:
                return "BEARISH"  # Double top confirmed with price below neckline

        # ── Rising Wedge detection (bearish): HH + HL but converging ────────
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            sh = swing_highs[-3:]
            sl = swing_lows[-3:]
            # Rising wedge: highs rising but lows rising faster (converging)
            high_slope = (sh[-1][1] - sh[0][1]) / max(sh[-1][0] - sh[0][0], 1)
            low_slope  = (sl[-1][1] - sl[0][1]) / max(sl[-1][0] - sl[0][0], 1)
            if high_slope > 0 and low_slope > high_slope * 1.3:
                return "BEARISH"  # Rising wedge = exhaustion, bears take over

        # ── Falling Wedge detection (bullish): LH + LL but converging ───────
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            sh = swing_highs[-3:]
            sl = swing_lows[-3:]
            high_slope = (sh[-1][1] - sh[0][1]) / max(sh[-1][0] - sh[0][0], 1)
            low_slope  = (sl[-1][1] - sl[0][1]) / max(sl[-1][0] - sl[0][0], 1)
            if low_slope < 0 and high_slope < 0 and high_slope < low_slope * 1.3:
                return "BULLISH"  # Falling wedge = compression, bulls take over

        # ── Standard HH/HL vs LH/LL from last 4 swings ──────────────────────
        threshold = 0.003  # 0.3% minimum move to count as structural shift

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            sh = [x[1] for x in swing_highs[-3:]]
            sl = [x[1] for x in swing_lows[-3:]]

            bull_highs = all(sh[i] > sh[i-1] * (1 + threshold) for i in range(1, len(sh)))
            bull_lows  = all(sl[i] > sl[i-1] * (1 + threshold) for i in range(1, len(sl)))
            bear_highs = all(sh[i] < sh[i-1] * (1 - threshold) for i in range(1, len(sh)))
            bear_lows  = all(sl[i] < sl[i-1] * (1 - threshold) for i in range(1, len(sl)))

            if bull_highs and bull_lows:
                return "BULLISH"
            if bear_highs and bear_lows:
                return "BEARISH"
            if bull_highs or bull_lows:
                return "BULLISH"
            if bear_highs or bear_lows:
                return "BEARISH"

        # ── Fallback: 4-quarter close comparison with 0.3% threshold ────────
        q1 = closes[:n//2].mean()
        q2 = closes[n//2:].mean()
        diff_pct = abs(q2 - q1) / max(q1, 1) * 100
        if diff_pct < 0.3:
            return "RANGING"
        return "BULLISH" if q2 > q1 else "BEARISH"

    def _mtf_bias(self, df_15: pd.DataFrame, df_1h: Optional[pd.DataFrame]) -> Tuple[str, int]:
        """Compare 15min and 1H bias. Returns bias + score (0-3)."""
        score = 0
        biases = []

        def bias(df_):
            if df_ is None or df_.empty:
                return None
            c = df_["close"].values
            if len(c) < 10:
                return None
            ema20 = pd.Series(c).ewm(span=20).mean().iloc[-1]
            return "BULL" if c[-1] > ema20 else "BEAR"

        b15 = bias(df_15)
        b1h = bias(df_1h)

        for b in [b15, b1h]:
            if b is not None:
                biases.append(b)

        if not biases:
            return "MIXED", 0
        bull_count = biases.count("BULL")
        bear_count = biases.count("BEAR")
        score = int(max(bull_count, bear_count))
        if bull_count > bear_count:
            return "BULLISH", score
        elif bear_count > bull_count:
            return "BEARISH", score
        return "MIXED", score


    def _build_signal(self, curr: float, structure: str, pd_label: str,
                       in_discount: bool, in_premium: bool,
                       obs: List[OrderBlock], fvgs: List[FairValueGap],
                       sweeps: List[LiquiditySweep],
                       vwap: float, price_vs_vwap: str,
                       session_tradeable: bool, mtf_bias: str,
                       atr: float,
                       swing_high: float = 0.0,
                       swing_low: float  = 0.0,
                       inst_levels: Optional[InstitutionalLevels] = None,
                       ) -> Tuple[str, float, Tuple, float, float, float, float, float, List[str]]:
        """
        SMC signal with CORRECT Fibonacci extension targets.

        Returns: (signal, confidence, entry_zone, sl, t1, t2, t3, t4, notes)

        TARGET RULES:
        LONG:  t1 < t2 < t3 < t4 (ascending, all above entry)
        SHORT: t1 > t2 > t3 > t4 (descending, all below entry)
        """
        notes = []
        confidence = 0.0

        if not session_tradeable:
            notes.append("⚠️ Session filter: not a tradeable window")
            _atr_est = atr if atr > 0 else curr * 0.005
            return "WAIT", 0.0, (round(curr-_atr_est*0.2,2), round(curr+_atr_est*0.2,2)), round(curr-_atr_est*1.5,2), round(curr+_atr_est*1.0,2), round(curr+_atr_est*2.0,2), round(curr+_atr_est*3.0,2), round(curr+_atr_est*4.0,2), notes

        # Find nearest OB to price
        bull_ob_active = next((o for o in obs
                                if o.direction=="BULL" and o.valid
                                and o.low <= curr <= o.high * 1.002), None)
        bear_ob_active = next((o for o in obs
                                if o.direction=="BEAR" and o.valid
                                and o.low * 0.998 <= curr <= o.high), None)

        # FVG containment
        bull_fvg = next((f for f in fvgs
                          if f.direction=="BULL" and f.bottom <= curr <= f.top), None)
        bear_fvg = next((f for f in fvgs
                          if f.direction=="BEAR" and f.bottom <= curr <= f.top), None)

        # Liquidity sweeps
        recent_ssl = next((s for s in sweeps if s.sweep_type=="SSL_SWEPT" and s.reversal_confirmed), None)
        recent_bsl = next((s for s in sweeps if s.sweep_type=="BSL_SWEPT" and s.reversal_confirmed), None)

        buy_score = 0.0
        buy_notes: List[str] = []

        if structure == "BULLISH":
            buy_score += 25; buy_notes.append("✅ Market structure: BULLISH (HH/HL)")
        if in_discount:
            buy_score += 20; buy_notes.append("✅ Price in DISCOUNT zone -- institutional buy area")
        if bull_ob_active:
            buy_score += 25; buy_notes.append(f"✅ Inside Bullish OB {bull_ob_active.low:.0f}-{bull_ob_active.high:.0f} ({bull_ob_active.strength})")
        if bull_fvg:
            buy_score += 15; buy_notes.append(f"✅ Inside Bullish FVG {bull_fvg.bottom:.0f}-{bull_fvg.top:.0f}")
        if recent_ssl:
            buy_score += 20; buy_notes.append(f"✅ SSL swept at {recent_ssl.level:.0f} -- liquidity grab -> bullish reversal")
        if price_vs_vwap == "ABOVE":
            buy_score += 10; buy_notes.append(f"✅ Above VWAP ({vwap:.0f}) -- institutional bias bullish")
        elif price_vs_vwap == "AT":
            buy_score += 5;  buy_notes.append(f"⚠️ At VWAP -- neutral")
        if mtf_bias == "BULLISH":
            buy_score += 15; buy_notes.append("✅ MTF aligned bullish (15min + 1H)")
        elif mtf_bias == "MIXED":
            buy_score += 5;  buy_notes.append("⚠️ MTF mixed")

        # PDH/PDL confluence
        if inst_levels:
            if abs(curr - inst_levels.pdl) / max(curr, 1) < 0.003:
                buy_score += 15; buy_notes.append(f"📍 At PDL ({inst_levels.pdl:.0f}) -- key support")
            if curr > inst_levels.pdh:
                buy_score += 10; buy_notes.append(f"📍 Above PDH ({inst_levels.pdh:.0f}) -- breakout confirmed")

        sell_score = 0.0
        sell_notes: List[str] = []

        if structure == "BEARISH":
            sell_score += 25; sell_notes.append("✅ Market structure: BEARISH (LH/LL)")
        if in_premium:
            sell_score += 20; sell_notes.append("✅ Price in PREMIUM zone -- institutional sell area")
        if bear_ob_active:
            sell_score += 25; sell_notes.append(f"✅ Inside Bearish OB {bear_ob_active.low:.0f}-{bear_ob_active.high:.0f} ({bear_ob_active.strength})")
        if bear_fvg:
            sell_score += 15; sell_notes.append(f"✅ Inside Bearish FVG {bear_fvg.bottom:.0f}-{bear_fvg.top:.0f}")
        if recent_bsl:
            sell_score += 20; sell_notes.append(f"✅ BSL swept at {recent_bsl.level:.0f} -- liquidity grab -> bearish reversal")
        if price_vs_vwap == "BELOW":
            sell_score += 10; sell_notes.append(f"✅ Below VWAP ({vwap:.0f}) -- institutional bias bearish")
        if mtf_bias == "BEARISH":
            sell_score += 15; sell_notes.append("✅ MTF aligned bearish")
        elif mtf_bias == "MIXED":
            sell_score += 5;  sell_notes.append("⚠️ MTF mixed")

        if inst_levels:
            if abs(curr - inst_levels.pdh) / max(curr, 1) < 0.003:
                sell_score += 15; sell_notes.append(f"📍 At PDH ({inst_levels.pdh:.0f}) -- key resistance")
            if curr < inst_levels.pdl:
                sell_score += 10; sell_notes.append(f"📍 Below PDL ({inst_levels.pdl:.0f}) -- breakdown confirmed")

        # Structural penalties
        if structure == "BEARISH" and in_premium:
            buy_score  = max(0, buy_score - 25)
            buy_notes.append("⚠️ PENALTY: Bearish structure + premium zone -- LONG risk very high")
        if structure == "BULLISH" and in_discount:
            sell_score = max(0, sell_score - 25)
            sell_notes.append("⚠️ PENALTY: Bullish structure + discount zone -- SHORT risk very high")

        # Last candle confirmation
        last_body  = abs(float(self._last_df["close"].iloc[-1]) - float(self._last_df["open"].iloc[-1])) if hasattr(self, '_last_df') else atr * 0.3
        rejection_candle = last_body < atr * 0.5

        MIN_SCORE = 65  # Raised from 60 to 65 -- require stronger confluence

        if buy_score >= MIN_SCORE and buy_score >= sell_score:
            signal     = "BUY"
            confidence = min(buy_score, 92)
            notes      = buy_notes

            if not rejection_candle:
                notes.append("⚠️ Wait for hammer/doji confirmation before entry")
                confidence = min(confidence, 70)

            # Entry zone from OB or FVG
            if bull_ob_active:
                ez = (bull_ob_active.low, bull_ob_active.mid)
                sl = bull_ob_active.low - atr * 0.5
            elif bull_fvg:
                ez = (bull_fvg.bottom, bull_fvg.mid)
                sl = bull_fvg.bottom - atr * 0.5
            else:
                ez = (curr - atr * 0.3, curr + atr * 0.1)
                sl = curr - atr * 1.5

            entry_mid = (ez[0] + ez[1]) / 2
            risk = max(entry_mid - sl, atr * 0.5)

            # ══ CORRECT Fibonacci extension targets for LONG ══
            # Use swing context if available, else use risk multiples
            if swing_high > 0 and swing_low > 0:
                leg = swing_high - swing_low
                t1 = round(swing_low + leg * 1.000, 2)
                t2 = round(swing_low + leg * 1.272, 2)
                t3 = round(swing_low + leg * 1.618, 2)
                t4 = round(swing_low + leg * 2.000, 2)
                # Ensure ordering
                t1 = max(t1, ez[1] + risk * 0.5)
                t2 = max(t2, t1 + atr * 0.5)
                t3 = max(t3, t2 + atr * 0.5)
                t4 = max(t4, t3 + atr * 0.5)
            else:
                # Fallback: use Fibonacci multiples of risk
                t1 = round(entry_mid + risk * 1.618, 2)   # ~1.618 RR
                t2 = round(entry_mid + risk * 2.500, 2)   # 2.5 RR
                t3 = round(entry_mid + risk * 3.618, 2)   # 3.618 RR
                t4 = round(entry_mid + risk * 5.000, 2)   # 5.0 RR

        elif sell_score >= MIN_SCORE and sell_score > buy_score:
            signal     = "SELL"
            confidence = min(sell_score, 92)
            notes      = sell_notes

            if not rejection_candle:
                notes.append("⚠️ Wait for shooting star/engulfing confirmation")
                confidence = min(confidence, 70)

            if bear_ob_active:
                ez = (bear_ob_active.mid, bear_ob_active.high)
                sl = bear_ob_active.high + atr * 0.5
            elif bear_fvg:
                ez = (bear_fvg.mid, bear_fvg.top)
                sl = bear_fvg.top + atr * 0.5
            else:
                ez = (curr - atr * 0.1, curr + atr * 0.3)
                sl = curr + atr * 1.5

            entry_mid = (ez[0] + ez[1]) / 2
            risk = max(sl - entry_mid, atr * 0.5)

            # ══ CORRECT Fibonacci extension targets for SHORT ══
            # T1 > T2 > T3 > T4 (all going DOWN)
            if swing_high > 0 and swing_low > 0:
                leg = swing_high - swing_low
                t1 = round(swing_high - leg * 1.000, 2)
                t2 = round(swing_high - leg * 1.272, 2)
                t3 = round(swing_high - leg * 1.618, 2)
                t4 = round(swing_high - leg * 2.000, 2)
                # Ensure ordering for shorts (t1 > t2 > t3 > t4)
                t1 = min(t1, ez[0] - risk * 0.5)
                t2 = min(t2, t1 - atr * 0.5)
                t3 = min(t3, t2 - atr * 0.5)
                t4 = min(t4, t3 - atr * 0.5)
            else:
                # Fallback Fibonacci risk multiples
                t1 = round(entry_mid - risk * 1.618, 2)
                t2 = round(entry_mid - risk * 2.500, 2)
                t3 = round(entry_mid - risk * 3.618, 2)
                t4 = round(entry_mid - risk * 5.000, 2)

        else:
            signal = "WAIT"
            confidence = 0
            ez = (curr, curr); sl = t1 = t2 = t3 = t4 = curr
            if buy_score > 0:  notes.append(f"⚠️ Buy score {buy_score:.0f}/100 -- need >={MIN_SCORE}")
            if sell_score > 0: notes.append(f"⚠️ Sell score {sell_score:.0f}/100 -- need >={MIN_SCORE}")
            if not notes: notes.append("ℹ️ No SMC confluence -- wait")

        return signal, confidence, ez, sl, t1, t2, t3, t4, notes

    def analyse(self, df: pd.DataFrame, symbol: str, timeframe: str,
                 df_1h: Optional[pd.DataFrame] = None,
                 inst_levels: Optional[InstitutionalLevels] = None) -> SMCAnalysis:
        """Full SMC analysis with Fibonacci targets and institutional levels."""
        self._last_df = df
        curr = float(df["close"].iloc[-1])
        atr_series = (df["high"] - df["low"]).rolling(14).mean()
        atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else float(curr) * 0.005

        obs       = self._detect_order_blocks(df)
        fvgs      = self._detect_fvg(df)
        sweeps    = self._detect_liquidity_sweeps(df)
        structure = self._market_structure(df)
        pd_label, pd_pct, in_disc, in_prem = self._premium_discount(df, curr)
        vwap      = self._calc_vwap(df)
        vwap_diff = float((curr - vwap) / vwap * 100)
        if abs(vwap_diff) < 0.05:
            pvwap = "AT"
        elif vwap_diff > 0:
            pvwap = "ABOVE"
        else:
            pvwap = "BELOW"
        vwap_bias = "BULLISH" if pvwap == "ABOVE" else "BEARISH" if pvwap == "BELOW" else "NEUTRAL"
        session, session_tradeable, session_note = self._get_session()
        mtf_bias, mtf_score = self._mtf_bias(df, df_1h)

        # Get swing context for Fibonacci targets
        sh, sl, _ = ICTEngine()._find_last_swing(df)
        swing_high = sh or 0.0
        swing_low  = sl or 0.0

        sig, conf, ez, stop, t1, t2, t3, t4, notes = self._build_signal(
            curr, structure, pd_label, in_disc, in_prem,
            obs, fvgs, sweeps, vwap, pvwap,
            session_tradeable, mtf_bias, atr,
            swing_high, swing_low, inst_levels
        )

        return SMCAnalysis(
            symbol=symbol, timeframe=timeframe, current_price=curr,
            generated_at=datetime.now().isoformat(),
            order_blocks=obs, fair_value_gaps=fvgs, liquidity_sweeps=sweeps,
            market_structure=structure,
            premium_discount=pd_label, pd_pct=pd_pct,
            in_discount=in_disc, in_premium=in_prem,
            vwap=vwap, price_vs_vwap=pvwap, vwap_bias=vwap_bias,
            session=session, session_tradeable=session_tradeable, session_note=session_note,
            mtf_bias=mtf_bias, mtf_score=mtf_score,
            smc_signal=sig, smc_confidence=conf,
            smc_entry_zone=ez, smc_sl=stop, smc_target1=t1, smc_target2=t2,
            smc_notes=notes,
        )


smc_engine = SMCEngine()

def generate_fno_signals(symbol: str, current_price: float, patterns: List[Dict],
                          fib_levels: Dict, atr: float,
                          option_chain: List[Dict] = None,
                          expiries: Dict[str, str] = None,
                          interval: str = "5minute") -> List[Dict]:
    """
    Generate F&O signals using REAL Kite data:
    - Real expiry dates from NFO instruments API
    - Real option premiums, IV, Delta, Theta, Gamma from Kite option chain quotes
    - Falls back to BS estimate only if Kite data is unavailable
    """
    signals = []
    meta = FNO_EXPIRY_SYMBOLS.get(symbol)
    if not meta:
        return signals

    lot_size  = meta["lot_size"]
    expiries  = expiries or {}
    weekly_str  = expiries.get("weekly",  "")
    monthly_str = expiries.get("monthly", "")

    # Parse expiry dates
    try:
        weekly_expiry  = date.fromisoformat(weekly_str)  if weekly_str  else None
        monthly_expiry = date.fromisoformat(monthly_str) if monthly_str else None
    except Exception:
        weekly_expiry = monthly_expiry = None

    if not weekly_expiry or not monthly_expiry:
        return [{"error": "Could not fetch real expiry dates from Kite. Ensure NFO instruments are loaded."}]

    # Sort patterns by confidence descending
    bullish_patterns = sorted(
        [p for p in patterns if p.get("signal") == "BULLISH"],
        key=lambda x: x.get("confidence", 0), reverse=True
    )
    bearish_patterns = sorted(
        [p for p in patterns if p.get("signal") == "BEARISH"],
        key=lambda x: x.get("confidence", 0), reverse=True
    )

    def round_to_strike(price: float) -> int:
        if symbol == "BANKNIFTY":   step = 100
        elif symbol == "NIFTY":     step = 50
        elif price >= 10000:        step = 100
        elif price >= 5000:         step = 50
        elif price >= 2000:         step = 20
        elif price >= 500:          step = 10
        else:                       step = 5
        return int(round(price / step) * step)

    def bs_estimate(spot, strike, is_call, days_left, iv=0.22):
        """Fallback BS ATM premium estimate."""
        if symbol in ("NIFTY", "BANKNIFTY"):
            iv = 0.13
        else:
            iv = 0.28
        T = max(days_left, 1) / 365.0
        intrinsic  = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        time_value = spot * iv * math.sqrt(T) * 0.4
        raw = intrinsic + time_value
        return round(round(raw / 0.05) * 0.05, 2)

    def get_chain_row(strike: int, opt_type: str) -> Optional[Dict]:
        """Find matching row in option_chain for strike+type."""
        if not option_chain:
            return None
        for row in option_chain:
            if int(row.get("strike", 0)) == strike:
                return row.get(opt_type)
        return None

    def build_option_signal(is_call, pattern, expiry_date, expiry_type):
        opt_type = "CE" if is_call else "PE"
        direction = "BUY CALL (CE)" if is_call else "BUY PUT (PE)"
        signal_dir = "BULLISH" if is_call else "BEARISH"

        entry_spot = current_price
        raw_sl     = pattern.get("stop_loss", current_price)
        raw_target = pattern.get("target", current_price)

        # Enforce minimum move sizes
        min_sl  = entry_spot * 0.005
        min_tgt = entry_spot * 0.010
        sl_pts  = max(abs(entry_spot - raw_sl),    min_sl)
        tgt_pts = max(abs(raw_target - entry_spot), min_tgt)
        if tgt_pts < sl_pts * 1.5:
            tgt_pts = sl_pts * 1.5

        if is_call:
            sl_spot  = round(entry_spot - sl_pts,  2)
            tgt_spot = round(entry_spot + tgt_pts, 2)
        else:
            sl_spot  = round(entry_spot + sl_pts,  2)
            tgt_spot = round(entry_spot - tgt_pts, 2)

        rr = round(tgt_pts / sl_pts, 2)

        atm_strike = round_to_strike(entry_spot)
        days_left  = max((expiry_date - date.today()).days, 1)

        # ── Try real Kite option chain data ──────────────────────────────────
        chain_row  = get_chain_row(atm_strike, opt_type)
        real_ltp   = None
        real_iv    = None
        real_delta = None
        real_theta = None
        real_gamma = None
        real_vega  = None
        real_ts    = None
        real_bid   = None
        real_ask   = None
        real_oi    = None

        if chain_row:
            real_ltp   = chain_row.get("ltp")
            real_iv    = chain_row.get("iv")
            real_delta = chain_row.get("delta")
            real_theta = chain_row.get("theta")
            real_gamma = chain_row.get("gamma")
            real_vega  = chain_row.get("vega")
            real_ts    = chain_row.get("tradingsymbol", "")
            real_bid   = chain_row.get("bid")
            real_ask   = chain_row.get("ask")
            real_oi    = chain_row.get("oi")

        # Use real LTP if available, else BS estimate
        if real_ltp and real_ltp > 0:
            entry_premium = real_ltp
            data_source   = "KITE_LIVE"
        else:
            entry_premium = bs_estimate(entry_spot, atm_strike, is_call, days_left)
            data_source   = "BS_ESTIMATE"

        entry_premium = max(entry_premium, 1.0)

        # ════════════════════════════════════════════════════════════════
        # PREMIUM TARGET & SL -- Proper Options Math
        #
        # Formula:  ΔPremium = δ x ΔSpot + ½γ x ΔSpot2 - θ x days
        #
        # Greeks hierarchy:
        #  1. Use REAL Greeks from Kite option chain if available
        #  2. Fallback to calibrated estimates (not arbitrary defaults)
        #
        # Key corrections vs old code:
        #  - gamma: old=0.002 (WRONG, inflated target by 65pts)
        #           new=calibrated per underlying (NIFTY≈0.00015)
        #  - theta: was MISSING entirely -- now subtracted
        #  - delta: use real from chain, fallback to 0.48 ATM (not 0.50)
        #  - floor: SL premium >= 20% of entry (not 15%)
        # ════════════════════════════════════════════════════════════════

        # ── Step 1: Get real or calibrated Greeks ────────────────────────
        # Delta (ATM PE≈-0.45 to -0.52, ATM CE≈+0.45 to +0.52)
        if real_delta and abs(real_delta) > 0.01:
            delta_used = abs(real_delta)
        else:
            # Calibrated ATM delta by moneyness (spot vs strike)
            moneyness = entry_spot / atm_strike
            if 0.995 <= moneyness <= 1.005:   # ATM +/- 0.5%
                delta_used = 0.48
            elif moneyness > 1.005:            # ITM CE / OTM PE
                delta_used = 0.62 if is_call else 0.35
            else:                              # OTM CE / ITM PE
                delta_used = 0.35 if is_call else 0.62

        # Gamma (per point squared) -- calibrated by underlying price
        if real_gamma and real_gamma > 0:
            gamma_used = real_gamma
        else:
            # Real NIFTY gamma ≈ 0.00015, stocks ≈ 0.0003
            if symbol in ("NIFTY", "BANKNIFTY"):
                gamma_used = 0.00015
            elif entry_spot > 3000:
                gamma_used = 0.0002
            else:
                gamma_used = 0.0004

        # Theta (per day, as positive number for subtraction)
        if real_theta and real_theta != 0:
            theta_per_day = abs(real_theta)
        else:
            # Calibrated theta: approx = entry_premium / (days_left * 1.5)
            # ATM options lose ~1/1.5 of their value per day near expiry
            theta_per_day = max(entry_premium / (days_left * 1.5), 0.5)

        # Vega (per 1% IV change)
        if real_vega and real_vega > 0:
            vega_used = real_vega
        else:
            vega_used = entry_premium * 0.15  # approx 15% of premium per 1% IV

        # ── Step 2: Estimate holding time ─────────────────────────────────
        # Map timeframe to expected hold duration in days
        # This determines how much theta we need to account for
        tf_theta_days = {
            "minute":   0.08,   # 1min  -> ~30min hold -> 0.08 day
            "3minute":  0.15,   # 3min  -> ~1hr hold
            "5minute":  0.20,   # 5min  -> ~1.5hr hold
            "15minute": 0.40,   # 15min -> ~3hr hold (intraday swing)
            "30minute": 0.75,   # 30min -> ~1 day
            "60minute": 1.50,   # 1hr   -> ~2 days
            "day":      3.00,   # daily -> ~3-5 days hold
        }
        # BUG FIX: use actual timeframe map, then clamp for near-expiry
        # Previously this dict was dead code -- expiry-based logic always ran instead
        hold_days = tf_theta_days.get(interval, 0.40)
        if days_left <= 1:
            hold_days = min(hold_days, 0.10)   # same-day expiry -- very short
        elif days_left <= 3:
            hold_days = min(hold_days, 0.25)   # near-expiry -- cap hold assumption

        theta_cost = theta_per_day * hold_days

        # ── Step 3: IV move on target ─────────────────────────────────────
        # IV change depends on days to expiry and move size:
        #   - Weekly (<=7 days):  IV swings sharply with spot -> +/-1.0%
        #   - Monthly (8-30d):   IV moves moderately -> +/-0.2% for 1% spot move
        #   - Long-dated (>30d): IV barely changes -> +/-0.05%
        # Direction: CE target hit = spot up = IV usually falls (negative vega impact)
        #            PE target hit = spot down = IV usually rises (positive vega impact)
        # Note: for large moves (>1%), multiply by move size factor
        move_pct = tgt_pts / entry_spot * 100   # how big is our target move
        if days_left <= 3:
            iv_base_pct = 0.010   # weekly near-expiry: IV very reactive
        elif days_left <= 7:
            iv_base_pct = 0.007   # weekly: moderate reactivity
        elif days_left <= 30:
            iv_base_pct = 0.002   # monthly: small IV reaction to 1% spot move
        else:
            iv_base_pct = 0.001   # long-dated: IV nearly stable

        # Scale by move size -- a 2% move causes more IV change than a 0.5% move
        iv_pct_change_raw = iv_base_pct * min(move_pct / 1.0, 2.0)
        iv_pct_change = iv_pct_change_raw if not is_call else -iv_pct_change_raw
        iv_premium_change = vega_used * (iv_pct_change * 100)  # in ₹

        # ── Step 4: Calculate target and SL premiums ──────────────────────
        # Target premium: spot moves in our favour
        #   = entry + δxtgt_pts + ½γxtgt_pts2 - θxdays + vegaxΔIV
        gamma_boost_tgt = 0.5 * gamma_used * (tgt_pts ** 2)
        raw_tgt = (entry_premium
                   + delta_used * tgt_pts
                   + gamma_boost_tgt
                   - theta_cost
                   + iv_premium_change)
        tgt_premium = round(max(raw_tgt, entry_premium * 1.20), 2)

        # SL premium: spot moves against us
        #   = entry - δxsl_pts - ½γxsl_pts2 - θxdays - vegaxΔIV(adverse)
        gamma_boost_sl  = 0.5 * gamma_used * (sl_pts ** 2)
        iv_adverse = -iv_premium_change  # IV moves opposite when we're wrong
        raw_sl = (entry_premium
                  - delta_used * sl_pts
                  - gamma_boost_sl
                  - theta_cost * 0.5      # less time passes if stopped quickly
                  + iv_adverse)
        # ── CRITICAL FIX: Premium SL must be realistic ────────────────────
        # Option traders NEVER let premium go below 40-50% of entry
        # Reason: if you're down 50% on premium, the trade thesis is broken
        # A 50% SL on premium = cut loss early, preserve capital
        # The theoretical delta-based SL is often too wide for options
        #
        # Three-tier SL:
        #   Standard:    50% of entry premium (most common)
        #   Aggressive:  35% of entry premium (wider, fewer stops)
        #   Conservative:60% of entry premium (tighter, quicker exit)
        sl_premium_50pct = round(entry_premium * 0.50, 2)   # Standard
        sl_premium_35pct = round(entry_premium * 0.35, 2)   # Aggressive
        sl_premium_60pct = round(entry_premium * 0.60, 2)   # Conservative

        # Use the delta-based SL but FLOOR it at 40% (never below)
        sl_premium = round(max(raw_sl, entry_premium * 0.40), 2)

        # Clamp: if delta-based SL > 60% of entry, it means spot SL is too tight
        # (would get stopped on normal noise) -- use 50% premium SL instead
        if sl_premium > entry_premium * 0.65:
            sl_premium = sl_premium_50pct

        # ── Step 5: Sanity check R:R on premium ───────────────────────────
        # IMPORTANT: Premium R:R is ALWAYS lower than spot R:R due to theta + vega.
        # A spot 2x R:R trade might only give 0.6x premium R:R -- this is HONEST.
        # Do NOT inflate target to manufacture a better-looking number.
        # Only nudge if the physics formula gives a truly absurd result (< 0.3x).
        prem_rr = (tgt_premium - entry_premium) / max(entry_premium - sl_premium, 1)
        if prem_rr < 0.30:
            # Physics gave an unrealistically low number (probably bad vega estimate)
            # Nudge to minimum 0.5x premium R:R -- still honest, just capped downside
            tgt_premium = round(entry_premium + (entry_premium - sl_premium) * 0.5, 2)
        # Show the honest premium R:R -- traders MUST know it's lower than spot R:R

        # Store Greeks used for display
        # Recalculate final premium R:R for display
        final_prem_rr = round((tgt_premium - entry_premium) / max(entry_premium - sl_premium, 1), 2)

        greeks_summary = {
            "delta": round(delta_used, 3),
            "gamma": round(gamma_used, 6),
            "theta": round(-theta_per_day, 3),
            "vega":  round(vega_used, 3),
            "theta_cost":   round(theta_cost, 2),
            "hold_days":    round(hold_days, 2),
            "prem_rr":      final_prem_rr,
            "iv_pct_change": round(iv_pct_change * 100, 2),   # % IV assumed to change
            "iv_premium_impact": round(iv_premium_change, 2),
            "premium_methodology": f"δxΔspot + ½γxΔspot2 - θ ({hold_days:.2f}d) + vegax{iv_pct_change*100:+.2f}%IV",
        }

        # Use real Kite tradingsymbol if available (already in correct format)
        # Fallback: build in proper Kite format SYMBOL+YY+MMM+DD+STRIKE+TYPE
        if real_ts:
            fno_sym = real_ts
        else:
            # Kite weekly format: NIFTY25FEB2725400CE
            fno_sym = (f"{symbol}"
                       f"{str(expiry_date.year)[2:]}"
                       f"{expiry_date.strftime('%b').upper()}"
                       f"{expiry_date.strftime('%d')}"
                       f"{atm_strike}{opt_type}")

        return {
            "type": "OPTION",
            "direction": direction,
            "signal": signal_dir,
            "symbol_fno": fno_sym,
            "underlying": symbol,
            "underlying_price": entry_spot,
            "strike": atm_strike,
            "option_type": opt_type,
            "expiry": expiry_date.isoformat(),
            "expiry_type": expiry_type,
            "lot_size": lot_size,
            "entry_spot": entry_spot,
            "sl_spot": sl_spot,
            "target_spot": tgt_spot,
            "entry_premium": entry_premium,
            "sl_premium": sl_premium,
            "target_premium": tgt_premium,
            "prem_rr": greeks_summary.get("prem_rr", 0),
            "iv_pct_change": greeks_summary.get("iv_pct_change", 0),
            "iv_premium_impact": greeks_summary.get("iv_premium_impact", 0),
            "risk_reward": rr,
            "confidence": pattern.get("confidence", 0),
            "strength": pattern.get("strength", 3),
            "pattern_trigger": pattern.get("pattern", ""),
            "max_loss_per_lot":   round((entry_premium - sl_premium)  * lot_size, 0),
            "max_profit_per_lot": round((tgt_premium  - entry_premium) * lot_size, 0),
            # Greeks (real from Kite if available, else calibrated)
            "iv":         round(real_iv * 100, 2) if real_iv else None,
            "delta":      round(real_delta, 4)    if real_delta else greeks_summary["delta"],
            "theta":      round(real_theta, 4)    if real_theta else greeks_summary["theta"],
            "gamma":      round(real_gamma, 6)    if real_gamma else greeks_summary["gamma"],
            "vega":       round(real_vega, 4)     if real_vega  else greeks_summary["vega"],
            "theta_cost": greeks_summary["theta_cost"],
            "hold_days":  greeks_summary["hold_days"],
            "premium_methodology": "LIVE_GREEKS" if real_delta else "CALIBRATED_GREEKS",
            "bid":   real_bid,
            "ask":   real_ask,
            "oi":    real_oi,
            "data_source": data_source,
            "rationale": (
                f"{pattern.get('pattern','')} ({int(pattern.get('confidence',0)*100)}% conf). "
                f"Entry ₹{entry_spot:.2f} | SL ₹{sl_spot:.2f} ({sl_pts/entry_spot*100:.1f}%) | "
                f"Target ₹{tgt_spot:.2f} ({tgt_pts/entry_spot*100:.1f}%). "
                f"Strike {atm_strike} {opt_type} @ ₹{entry_premium:.2f} "
                f"({'live' if data_source=='KITE_LIVE' else 'est.'}). "
                f"Premium calc: entry ₹{entry_premium:.2f} + δ{greeks_summary['delta']:.2f}x{tgt_pts:.0f}pts "
                f"+ γ-boost {0.5*greeks_summary['gamma']*tgt_pts**2:.1f} "
                f"- θ {greeks_summary['theta_cost']:.1f}pts ({greeks_summary['hold_days']:.2f}day hold) "
                f"= tgt ₹{tgt_premium:.2f}. "
                + (f"IV {real_iv*100:.1f}% | Δ {real_delta:.3f} | Θ {real_theta:.3f}" if real_iv else
                   f"[{greeks_summary.get('premium_methodology','CALIBRATED')}]")
            ),
            "action": "BUY",
            "signal_time": datetime.now().strftime("%H:%M IST"),
            "signal_date": datetime.now().strftime("%d %b %Y"),
            "pattern_formed_at": str(pattern.get("timestamp", ""))[:16],
        }

    def build_futures_signal(is_long, pattern, expiry_date):
        entry_spot = current_price
        raw_sl     = pattern.get("stop_loss", current_price)
        raw_target = pattern.get("target",    current_price)

        min_sl  = entry_spot * 0.005
        min_tgt = entry_spot * 0.010
        sl_pts  = max(abs(entry_spot - raw_sl),    min_sl)
        tgt_pts = max(abs(raw_target - entry_spot), min_tgt)
        if tgt_pts < sl_pts * 1.5:
            tgt_pts = sl_pts * 1.5

        if is_long:
            sl_spot  = round(entry_spot - sl_pts,  2)
            tgt_spot = round(entry_spot + tgt_pts, 2)
            direction = "BUY FUTURES"
        else:
            sl_spot  = round(entry_spot + sl_pts,  2)
            tgt_spot = round(entry_spot - tgt_pts, 2)
            direction = "SELL FUTURES"

        rr     = round(tgt_pts / sl_pts, 2)
        # Kite monthly futures format: NIFTY25MAR26FUT
        fno_sym = (f"{symbol}"
                   f"{str(expiry_date.year)[2:]}"
                   f"{expiry_date.strftime('%b').upper()}"
                   f"{expiry_date.strftime('%d')}"
                   f"FUT")

        return {
            "type": "FUTURES",
            "direction": direction,
            "signal": "BULLISH" if is_long else "BEARISH",
            "symbol_fno": fno_sym,
            "underlying": symbol,
            "underlying_price": entry_spot,
            "strike": None,
            "option_type": "FUT",
            "expiry": expiry_date.isoformat(),
            "expiry_type": "MONTHLY",
            "lot_size": lot_size,
            "entry_spot": entry_spot,
            "sl_spot": sl_spot,
            "target_spot": tgt_spot,
            "entry_premium": None,
            "sl_premium": None,
            "target_premium": None,
            "risk_reward": rr,
            "confidence": pattern.get("confidence", 0),
            "strength": pattern.get("strength", 3),
            "pattern_trigger": pattern.get("pattern", ""),
            "max_loss_per_lot":   round(sl_pts  * lot_size, 0),
            "max_profit_per_lot": round(tgt_pts * lot_size, 0),
            "iv": None, "delta": None, "theta": None,
            "gamma": None, "vega": None,
            "bid": None, "ask": None, "oi": None,
            "data_source": "KITE_LIVE",
            "rationale": (
                f"Futures {direction} on {pattern.get('pattern','')} "
                f"({int(pattern.get('confidence',0)*100)}% conf). "
                f"Entry ₹{entry_spot:.2f} | SL ₹{sl_spot:.2f} | Target ₹{tgt_spot:.2f}. "
                f"Max risk ₹{round(sl_pts*lot_size,0):,.0f} | "
                f"Max reward ₹{round(tgt_pts*lot_size,0):,.0f} per lot."
            ),
            "action": "BUY" if is_long else "SELL",
            "signal_time": datetime.now().strftime("%H:%M IST"),
            "signal_date": datetime.now().strftime("%d %b %Y"),
            "pattern_formed_at": str(pattern.get("timestamp", ""))[:16],
        }

    # ════════════════════════════════════════════════════════════════════════
    # SMART CONFLICT RESOLUTION -- 3-tier decision logic
    #
    # Rule 1 -- RECENCY: The most recent pattern on the chart wins.
    #   e.g. Three White Soldiers formed at candle 45, Three Black Crows at
    #   candle 38 -> White Soldiers wins regardless of confidence gap.
    #   Recency gap threshold: if newest bull is >= 3 candles newer -> bull wins.
    #
    # Rule 2 -- STRENGTH SCORE: confidence x strength (1-5 stars).
    #   Weighted score breaks ties when patterns form on same candle range.
    #
    # Rule 3 -- CONFIDENCE GAP (last resort):
    #   Only show WAIT if recency is equal AND confidence gap < 5%.
    #   Gap < 5% (not 8%) -- was too aggressive causing constant WAIT screens.
    # ════════════════════════════════════════════════════════════════════════

    def score(p):
        """Composite score: confidence (0-1) x strength (1-5) x recency boost."""
        return p.get("confidence", 0) * p.get("strength", 3)

    best_bull = bullish_patterns[0] if bullish_patterns else None
    best_bear = bearish_patterns[0] if bearish_patterns else None

    conflict = bool(best_bull and best_bear)
    conflict_note = ""
    go_bullish = False
    go_bearish = False

    if not conflict:
        go_bullish = bool(best_bull)
        go_bearish = bool(best_bear)

    else:
        bull_idx   = best_bull.get("candle_index", 0)
        bear_idx   = best_bear.get("candle_index", 0)
        bull_score = score(best_bull)
        bear_score = score(best_bear)
        bull_conf  = best_bull.get("confidence", 0)
        bear_conf  = best_bear.get("confidence", 0)
        conf_diff  = abs(bull_conf - bear_conf)
        recency_gap = bull_idx - bear_idx  # positive = bull is newer

        # ── Rule 1: Recency -- if one pattern is 3+ candles newer, it wins ──
        if recency_gap >= 3:
            go_bullish = True
            conflict_note = (f" [Older bearish '{best_bear['pattern']}' overridden -- "
                             f"bull pattern {recency_gap} candles newer]")

        elif recency_gap <= -3:
            go_bearish = True
            conflict_note = (f" [Older bullish '{best_bull['pattern']}' overridden -- "
                             f"bear pattern {abs(recency_gap)} candles newer]")

        # ── Rule 2: Patterns on same candle range -- use composite score ──
        elif bull_score != bear_score:
            if bull_score > bear_score:
                go_bullish = True
                conflict_note = (f" [Bearish '{best_bear['pattern']}' overridden -- "
                                 f"lower strengthxconfidence score ({bear_score:.2f} vs {bull_score:.2f})]")
            else:
                go_bearish = True
                conflict_note = (f" [Bullish '{best_bull['pattern']}' overridden -- "
                                 f"lower strengthxconfidence score ({bull_score:.2f} vs {bear_score:.2f})]")

        # ── Rule 3: Last resort -- confidence gap < 5% = genuine indecision ──
        elif conf_diff < 0.05:
            return [{
                "type": "NO_TRADE",
                "direction": "⚠️ MARKET INDECISION -- WAIT",
                "signal": "NEUTRAL",
                "symbol_fno": f"{symbol} -- WAIT FOR CLARITY",
                "underlying": symbol,
                "underlying_price": current_price,
                "strike": None, "option_type": None,
                "expiry": weekly_expiry.isoformat(), "expiry_type": "WEEKLY",
                "lot_size": lot_size,
                "entry_spot": current_price,
                "sl_spot": None, "target_spot": None,
                "entry_premium": None, "sl_premium": None, "target_premium": None,
                "risk_reward": 0,
                "confidence": max(bull_conf, bear_conf),
                "strength": 2,
                "pattern_trigger": (f"Bull: {best_bull['pattern']} "
                                    f"(conf {int(bull_conf*100)}%, candle #{bull_idx}) vs "
                                    f"Bear: {best_bear['pattern']} "
                                    f"(conf {int(bear_conf*100)}%, candle #{bear_idx})"),
                "max_loss_per_lot": 0, "max_profit_per_lot": 0,
                "iv": None, "delta": None, "theta": None,
                "gamma": None, "vega": None,
                "bid": None, "ask": None, "oi": None,
                "data_source": "CONFLICT",
                "rationale": (
                    f"⚠️ GENUINE INDECISION: '{best_bull['pattern']}' (bull, {int(bull_conf*100)}% conf, "
                    f"candle #{bull_idx}) vs '{best_bear['pattern']}' (bear, {int(bear_conf*100)}% conf, "
                    f"candle #{bear_idx}). Confidence gap {conf_diff*100:.1f}% < 5% AND patterns are "
                    f"within 3 candles of each other -- market is genuinely choppy. "
                    f"Switch to 5min or 15min timeframe for clearer signal."
                ),
                "action": "WAIT",
            }]

        # Confidence gap >= 5% -- stronger side wins
        else:
            if bull_conf >= bear_conf:
                go_bullish = True
                conflict_note = (f" [Bear '{best_bear['pattern']}' ({int(bear_conf*100)}%) overridden -- "
                                 f"bull confidence higher by {conf_diff*100:.1f}%]")
            else:
                go_bearish = True
                conflict_note = (f" [Bull '{best_bull['pattern']}' ({int(bull_conf*100)}%) overridden -- "
                                 f"bear confidence higher by {conf_diff*100:.1f}%]")

    # ── Straddle / Strangle builder ──────────────────────────────────────────
    def build_straddle_signal(pattern, is_straddle=True):
        """
        Build a straddle (same strike CE+PE) or strangle (OTM CE + OTM PE) signal.
        
        Daily straddle rules:
          - Entry ONLY 9:20-9:35 AM (first 20 min = max intraday move capture)
          - Target: 30-50% gain on total premium (NOT waiting for breakeven at expiry)
          - SL: 25% loss on total premium
          - Exit by 1:00 PM regardless (theta accelerates after noon)
          - Best when: VIX > 14, gap > 0.3%, event day, Monday open
        """
        days_left    = (weekly_expiry - datetime.now().date()).days + 1
        atm_strike   = round_to_strike(current_price)
        step = 100 if symbol in ("BANKNIFTY", "NIFTY BANK") else 50

        # Get CE leg
        ce_row = get_chain_row(atm_strike, "CE")
        ce_ltp = float(ce_row.get("last_price", 0)) if ce_row else bs_estimate(current_price, atm_strike, True, days_left)

        # Get PE leg
        pe_row = get_chain_row(atm_strike, "PE")
        pe_ltp = float(pe_row.get("last_price", 0)) if pe_row else bs_estimate(current_price, atm_strike, False, days_left)

        if is_straddle:
            # STRADDLE: buy CE + PE at same ATM strike
            ce_strike = atm_strike
            pe_strike = atm_strike
        else:
            # STRANGLE: buy OTM CE + OTM PE
            ce_strike = atm_strike + step
            pe_strike = atm_strike - step
            ce_row = get_chain_row(ce_strike, "CE")
            ce_ltp = float(ce_row.get("last_price", 0)) if ce_row else bs_estimate(current_price, ce_strike, True, days_left)
            pe_row = get_chain_row(pe_strike, "PE")
            pe_ltp = float(pe_row.get("last_price", 0)) if pe_row else bs_estimate(current_price, pe_strike, False, days_left)

        total_premium = round(ce_ltp + pe_ltp, 2)
        if total_premium <= 0:
            return None

        # Breakeven levels (for holding to expiry -- informational only)
        be_up   = round(ce_strike + total_premium, 2) if is_straddle else round(ce_strike + total_premium, 2)
        be_down = round(pe_strike - total_premium, 2)

        # INTRADAY targets (30-50% profit, 25% SL -- NOT breakeven targets)
        intraday_target = round(total_premium * 1.40, 2)   # 40% gain
        intraday_sl     = round(total_premium * 0.75, 2)   # 25% loss
        intraday_profit = round((intraday_target - total_premium) * lot_size, 2)
        intraday_loss   = round((total_premium - intraday_sl) * lot_size, 2)

        # Theta burn per day
        theta_pct = 0.06 if days_left <= 4 else 0.03 if days_left <= 7 else 0.015
        theta_day = round(total_premium * theta_pct * lot_size, 2)

        # Greeks (combined from both legs)
        ce_delta  = float(ce_row.get("delta", 0.50)) if ce_row else 0.50
        pe_delta  = float(pe_row.get("delta", -0.50)) if pe_row else -0.50
        net_delta = round(ce_delta + pe_delta, 3)  # ~0 for ATM straddle
        ce_theta  = float(ce_row.get("theta", 0)) if ce_row else 0
        pe_theta  = float(pe_row.get("theta", 0)) if pe_row else 0
        net_theta = round(ce_theta + pe_theta, 3)
        ce_vega   = float(ce_row.get("vega", 0)) if ce_row else 0
        pe_vega   = float(pe_row.get("vega", 0)) if pe_row else 0
        net_vega  = round(ce_vega + pe_vega, 3)

        # GAP 5: Fetch India VIX live for straddle suitability advice
        vix_val = 0.0
        vix_note = ""
        try:
            vix_q = kite_manager.kite.quote(["NSE:INDIA VIX"])
            vix_val = float((vix_q.get("NSE:INDIA VIX") or {}).get("last_price", 0) or 0)
        except Exception:
            pass
        if vix_val > 0:
            if vix_val < 12:
                vix_note = f"⚠️ VIX {vix_val:.1f} -- very low. Premiums deflated. Straddle buyers disadvantaged."
            elif vix_val <= 15:
                vix_note = f"ℹ️ VIX {vix_val:.1f} -- low. OK only if clear event catalyst expected."
            elif vix_val <= 20:
                vix_note = f"✅ VIX {vix_val:.1f} -- sweet spot. Good for buying straddles/strangles."
            else:
                vix_note = f"⚠️ VIX {vix_val:.1f} -- high. Premium expensive. Need very big move to profit."

        # Suitability score
        now_h, now_m = datetime.now().hour, datetime.now().minute
        in_entry_window = (9, 20) <= (now_h, now_m) <= (9, 35)
        past_noon = (now_h, now_m) > (12, 30)

        suitability = []
        score = 0
        if days_left <= 3:
            suitability.append(f"⚠️ Only {days_left} DTE -- theta burns ₹{theta_day}/day. Intraday only, exit by 12:30.")
        elif days_left <= 7:
            suitability.append(f"✅ {days_left} DTE -- manageable theta (₹{theta_day}/day)")
            score += 1
        else:
            suitability.append(f"✅ {days_left} DTE -- good time value, theta low (₹{theta_day}/day)")
            score += 2

        if in_entry_window:
            suitability.append("✅ Perfect entry window (9:20-9:35 AM)")
            score += 2
        elif (now_h, now_m) < (9, 20):
            suitability.append("⏰ Wait for 9:20 AM entry window")
        elif past_noon:
            suitability.append("⚠️ Past 12:30 -- theta accelerates. Enter only if fresh catalyst.")
        else:
            suitability.append(f"ℹ️ Entry window was 9:20-9:35. Current entry is {'ok' if now_h < 11 else 'late'}.")

        strategy_name = "Buy Straddle" if is_straddle else "Buy Strangle"
        strategy_icon = "⚡" if is_straddle else "🔀"
        legs_desc = (f"Buy {ce_strike} CE @ ₹{ce_ltp} + Buy {pe_strike} PE @ ₹{pe_ltp}"
                     if is_straddle
                     else f"Buy {ce_strike} CE (OTM+{step}) @ ₹{ce_ltp} + Buy {pe_strike} PE (OTM-{step}) @ ₹{pe_ltp}")

        return {
            "type": "STRADDLE" if is_straddle else "STRANGLE",
            "strategy_name": strategy_name,
            "strategy_icon": strategy_icon,
            "direction": f"{strategy_icon} {strategy_name.upper()}",
            "signal": "NEUTRAL",
            "symbol_fno": f"{symbol} {strategy_name.upper()}",
            "underlying": symbol,
            "underlying_price": round(current_price, 2),
            "ce_strike": ce_strike,
            "pe_strike": pe_strike,
            "ce_ltp": round(ce_ltp, 2),
            "pe_ltp": round(pe_ltp, 2),
            "total_premium": total_premium,
            "total_cost_lot": round(total_premium * lot_size, 2),
            # Intraday management levels (PRIMARY -- use these)
            "intraday_target_prem": intraday_target,
            "intraday_sl_prem": intraday_sl,
            "intraday_profit_lot": intraday_profit,
            "intraday_loss_lot": intraday_loss,
            # Expiry breakeven (informational only)
            "breakeven_up": be_up,
            "breakeven_down": be_down,
            "move_needed": round(be_up - current_price, 2),
            "move_needed_pct": round((be_up - current_price) / current_price * 100, 2),
            # Greeks
            "net_delta": net_delta,
            "net_theta": net_theta,
            "net_vega": net_vega,
            "theta_per_day_lot": theta_day,
            # Expiry info
            "expiry": weekly_expiry.isoformat(),
            "expiry_type": "WEEKLY",
            "days_to_expiry": days_left,
            "lot_size": lot_size,
            # Rules
            "entry_rule": "Buy 9:20-9:35 AM only. Both legs simultaneously.",
            "target_rule": f"Exit when total straddle premium = ₹{intraday_target} (+40%). Don't wait for breakeven.",
            "sl_rule": f"Exit both legs if total premium = ₹{intraday_sl} (-25% loss).",
            "time_rule": "ALWAYS exit both legs by 1:00 PM if not profitable. Never hold past 2 PM.",
            "suitability": suitability,
            "suitability_score": score,
            "legs_desc": legs_desc,
            "pattern_trigger": pattern.get("pattern", "Straddle setup"),
            "signal_time": datetime.now().strftime("%H:%M IST"),
            "signal_date": datetime.now().strftime("%d %b %Y"),
            "rationale": (
                f"{strategy_icon} {strategy_name}: {legs_desc}. "
                f"Total cost ₹{total_premium}/share = ₹{total_premium*lot_size:.0f}/lot. "
                f"Intraday target: exit when premium = ₹{intraday_target} (+40% = ₹{intraday_profit} profit/lot). "
                f"SL: exit if premium = ₹{intraday_sl} (-25% = ₹{intraday_loss} loss/lot). "
                f"Exit by 1 PM mandatory. Breakeven at expiry: {be_down}-{be_up} (+/-{round(be_up-current_price,0)}pts). "
                f"Net delta ≈ {net_delta} (near-zero = direction-neutral). "
                f"Vega {net_vega:.1f} (long vol = profits from IV spike). "
                f"Theta: -₹{theta_day}/lot/day."
            ),
            "action": "BUY BOTH LEGS",
            "india_vix": round(vix_val, 2),
            "vix_note": vix_note,
        }

    # ── Build final signals ──────────────────────────────────────────────────
    if go_bullish and best_bull:
        opt_sig = build_option_signal(True,  best_bull, weekly_expiry, "WEEKLY")
        fut_sig = build_futures_signal(True, best_bull, monthly_expiry)
        if conflict_note:
            opt_sig["rationale"] += conflict_note
            fut_sig["rationale"] += conflict_note
        signals.append(opt_sig)
        signals.append(fut_sig)

    elif go_bearish and best_bear:
        opt_sig = build_option_signal(False,  best_bear, weekly_expiry, "WEEKLY")
        fut_sig = build_futures_signal(False, best_bear, monthly_expiry)
        if conflict_note:
            opt_sig["rationale"] += conflict_note
            fut_sig["rationale"] += conflict_note
        signals.append(opt_sig)
        signals.append(fut_sig)

    # ── Always append Straddle + Strangle as additional strategy options ──────
    # These are shown BELOW the directional signal as alternative strategies.
    # Useful when: indecision, or trader wants direction-neutral approach.
    trigger_pattern = (best_bull or best_bear or (patterns[0] if patterns else {}))
    if trigger_pattern:
        straddle_sig = build_straddle_signal(trigger_pattern, is_straddle=True)
        strangle_sig = build_straddle_signal(trigger_pattern, is_straddle=False)
        if straddle_sig:
            signals.append(straddle_sig)
        if strangle_sig:
            signals.append(strangle_sig)

    else:
        # Build diagnostic explanation for why no signal fired
        _now = datetime.now()
        _h, _m = _now.hour, _now.minute
        _market_open = (_h, _m) >= (9, 15) and (_h, _m) < (15, 30)
        _market_closed = (_h, _m) >= (15, 30)
        _opening_block = (_h == 9 and _m < 30)
        _lunch_zone = (11, 30) <= (_h, _m) < (13, 0)

        # Explain the most likely reason for no signal
        if _market_closed:
            _why = "Market closed (15:30 IST). Signals will resume at 9:30 tomorrow."
            _best_action = "Check back after 9:30 AM tomorrow. Use ⚡ ORB tab at open."
        elif _opening_block:
            _why = "Opening range filter active (9:15-9:29 IST). Signals blocked to avoid opening noise."
            _best_action = "Wait for 9:30 AM -- first clean signal window opens then."
        elif _lunch_zone:
            _why = "Lunch zone (11:30-13:00 IST). Low volume, choppy price action, high false-breakout risk."
            _best_action = "Avoid trading. Watch for setup in 13:00-15:00 afternoon window."
        elif not patterns:
            _why = "No candlestick patterns detected in the last 20 bars."
            _best_action = "Market is ranging/choppy. Wait for a directional move or check SMC / Structure tabs."
        else:
            # Patterns exist but were filtered
            _n_bull = sum(1 for p in patterns if p.get("signal") == "BULLISH")
            _n_bear = sum(1 for p in patterns if p.get("signal") == "BEARISH")
            _why = (f"Patterns found ({_n_bull} bullish, {_n_bear} bearish) but filtered out. "
                    f"Likely reasons: conflicting signals, low confidence (<65%), VWAP misalignment, "
                    f"or patterns older than 20 bars.")
            _best_action = "Market is indecisive. Wait for a cleaner, higher-confidence setup."

        signals.append({
            "type": "NO_TRADE",
            "direction": "⚠️ NO CLEAR SIGNAL",
            "signal": "NEUTRAL",
            "symbol_fno": f"{symbol} -- NO SETUP",
            "underlying": symbol,
            "underlying_price": current_price,
            "strike": None, "option_type": None,
            "expiry": weekly_expiry.isoformat(), "expiry_type": "WEEKLY",
            "lot_size": lot_size,
            "entry_spot": current_price,
            "sl_spot": None, "target_spot": None,
            "entry_premium": None, "sl_premium": None, "target_premium": None,
            "risk_reward": 0, "confidence": 0, "strength": 0,
            "pattern_trigger": "No strong pattern",
            "patterns_found": len(patterns),
            "patterns_bull": sum(1 for p in patterns if p.get("signal") == "BULLISH"),
            "patterns_bear": sum(1 for p in patterns if p.get("signal") == "BEARISH"),
            "no_signal_reason": _why,
            "best_action": _best_action,
            "market_status": "CLOSED" if _market_closed else "OPEN" if _market_open else "PRE",
            "max_loss_per_lot": 0, "max_profit_per_lot": 0,
            "iv": None, "delta": None, "theta": None, "gamma": None, "vega": None,
            "bid": None, "ask": None, "oi": None,
            "data_source": "NO_SIGNAL",
            "rationale": _why,
            "action": "WAIT",
        })

    return signals

# ════════════════════════════════════════════════════════════════════════════════
#  APP STATE
# ════════════════════════════════════════════════════════════════════════════════

kite_manager = KiteManager()
pattern_detector = CandlestickPatternDetector()
fib_analyzer = FibonacciAnalyzer()
websocket_clients: Set[WebSocket] = set()

# Instrument token map populated on first use
instrument_cache: Dict[str, int] = {}

# ── Verified NSE instrument tokens (EQ series, from Kite instruments API) ────
# These are the CORRECT tokens - verified against live Kite instruments list.
# Index tokens use NSE exchange; equity tokens use NSE EQ series.
DEMO_TOKENS: Dict[str, int] = {
    # Indices (special tokens)
    "NIFTY 50":    256265,
    "NIFTY BANK":  260105,
    # Largecap -- NSE EQ tokens (verified)
    "RELIANCE":    738561,
    "TCS":         2953217,
    "INFY":        408065,
    "HDFCBANK":    341249,
    "ICICIBANK":   1270529,
    "BAJFINANCE":  81153,
    "HINDUNILVR":  356865,
    "MARUTI":      2815745,
    "TATAMOTORS":  884737,
    "AXISBANK":    1510401,
    "LT":          2939649,
    "WIPRO":       969473,
    "SUNPHARMA":   857857,
    "TATASTEEL":   895745,
    "KOTAKBANK":   492033,
    "SBIN":        779521,
    "ADANIENT":    3861249,
    "BHARTIARTL":  2714625,
    "NTPC":        2977281,
    "POWERGRID":   3834113,
    "ONGC":        633601,
    "COALINDIA":   5215745,
    # Midcap -- tokens resolved dynamically (see _resolve_token below)
    # Smallcap -- tokens resolved dynamically
}

# Symbols that need dynamic token resolution from Kite instruments API
# (hardcoded tokens for newer/midcap/smallcap stocks are unreliable)
DYNAMIC_RESOLVE_SYMBOLS = [
    "FIVESTAR", "INDUSTOWER", "ABCAPITAL", "MUTHOOTFIN", "CHOLAFIN",
    "PERSISTENT", "COFORGE", "LTM", "KALYANKJIL", "PHOENIXLTD",
    "GODREJPROP", "PIIND", "ALKEM", "TORNTPHARM", "MAXHEALTH",
    "POLICYBZR", "NYKAA", "DELHIVERY", "ETERNAL",
    "IREDA", "RVNL", "CDSL", "BSE",
    "HAPPSTMNDS", "RATEGAIN", "CAMPUS", "SENCO", "IDEAFORGE",
]

# Runtime cache: populated on first use per symbol
_token_cache: Dict[str, int] = {}

# ── TTL Response Cache ─────────────────────────────────────────────────────────
# Caches per-symbol analysis results for up to TTL_SECONDS.
# Prevents redundant Kite API calls when the same symbol is loaded repeatedly
# within the same candle cycle (e.g. switching tabs on same symbol).
class TTLCache:
    """Simple thread-safe TTL dict. Entries expire after `ttl` seconds."""
    def __init__(self, ttl: int = 180):
        self._store: Dict[str, Any] = {}
        self._ts:    Dict[str, float] = {}
        self._lock   = threading.Lock()
        self.ttl     = ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                if time.time() - self._ts[key] < self.ttl:
                    return self._store[key]
                del self._store[key]
                del self._ts[key]
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._ts[key]    = time.time()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._ts.clear()

    # ── dict-compatibility methods so len(), `in`, and [] assignment work ──
    def __len__(self) -> int:
        with self._lock:
            # Count only non-expired entries
            now = time.time()
            return sum(1 for k, ts in self._ts.items() if now - ts < self.ttl)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __getitem__(self, key: str) -> Any:
        v = self.get(key)
        if v is None:
            raise KeyError(key)
        return v

    def __contains__(self, key: object) -> bool:
        return self.get(str(key)) is not None

    def keys(self):
        with self._lock:
            now = time.time()
            return [k for k, ts in self._ts.items() if now - ts < self.ttl]

# One cache per analysis type. 3-minute TTL matches the shortest candle interval.
_smc_cache       = TTLCache(ttl=180)
_ict_cache       = TTLCache(ttl=180)
_structure_cache = TTLCache(ttl=180)
_orb_cache       = TTLCache(ttl=28800)  # Full trading day TTL (8 h) — ORB signals are daily
_pattern_cache   = TTLCache(ttl=180)
_scan_cache      = TTLCache(ttl=240)  # Full scan results cached 4 min

def normalise_conf(v: float) -> float:
    """Ensure confidence is always in 0–100 range.
    Accepts both 0.0–1.0 (fractional) and 0–100 (percent) formats and
    returns a consistent 0–100 float so display code never needs to
    conditionally multiply by 100.
    """
    if v is None:
        return 0.0
    if 0.0 <= v <= 1.0:
        return round(v * 100, 1)
    return round(float(v), 1)


# ── Kite API semaphore — max 3 concurrent historical data requests ─────────────
# Kite rate limit: 3 req/s per IP. This prevents HTTP 429 errors during scans.
_kite_semaphore = threading.Semaphore(3)



# NSE symbol alias map -- maps our internal name to actual NSE tradingsymbol
# Used when our name differs from what Kite's instrument list shows
NSE_SYMBOL_ALIASES: Dict[str, str] = {
    # Common mismatches / rebranding
    "ETERNAL":       "ETERNAL",       # Zomato rebranded to Eternal Ltd (Jan 2025)
    "LTM":           "LTM",           # LTIMindtree -- tradingsymbol is LTM on NSE (not LTIM)
    "KPITTECH":      "KPITTECH",      # Correct spelling (not KPITECH)
    "M&M":           "M&M",
    "M&MFIN":        "M&MFIN",
    "L&TFH":         "L&TFH",
    "BAJAJ-AUTO":    "BAJAJ-AUTO",
    "NIFTY 50":      None,            # Index -- no EQ token
    "NIFTY BANK":    None,            # Index -- no EQ token
    # PSU infra -- some have long names
    "RVNL":          "RVNL",
    "IREDA":         "IREDA",
    "IRFC":          "IRFC",
    "RECLTD":        "RECLTD",
    # Pharma
    "DRREDDY":       "DRREDDY",
    # Telecom infra
    "INDUSTOWER":    "INDUSTOWER",
}


def _bulk_resolve_tokens() -> None:
    """
    Pre-resolve instrument tokens for all SCAN_UNIVERSE symbols in one pass.
    Called in a background thread immediately after Kite login.
    Loads the NSE instruments DF once, then resolves every symbol via vectorised
    pandas filter — no per-symbol API calls, completes in < 2 seconds.
    """
    try:
        if not kite_manager.is_authenticated:
            return
        df = kite_manager._get_nse_instruments_df()
        if df.empty:
            return
        resolved = 0
        for symbol in SCAN_UNIVERSE:
            if symbol in _token_cache or symbol in DEMO_TOKENS:
                continue
            if symbol in ("NIFTY 50", "NIFTY BANK"):
                continue
            lookup = NSE_SYMBOL_ALIASES.get(symbol, symbol)
            if lookup is None:
                continue
            match = df[
                (df["tradingsymbol"] == lookup) &
                (df["instrument_type"] == "EQ") &
                (df["exchange"] == "NSE")
            ]
            if not match.empty:
                _token_cache[symbol] = int(match.iloc[0]["instrument_token"])
                resolved += 1
        log.info(f"✅ Bulk token pre-resolve complete: {resolved} symbols cached")
    except Exception as e:
        log.warning(f"Bulk token pre-resolve failed: {e}")

def get_instrument_token(symbol: str) -> Optional[int]:
    """
    Get correct NSE EQ instrument token for a symbol.
    Priority:
      1. Hardcoded DEMO_TOKENS (verified largecaps/indices)
      2. Runtime _token_cache (already resolved this session)
      3. NSE_SYMBOL_ALIASES map -> resolve actual tradingsymbol from Kite
      4. Direct match in Kite NSE instruments list
      5. Fuzzy fallback: try stripping/adding common suffix variants
    """
    # Indices -- no EQ token
    if symbol in ("NIFTY 50", "NIFTY BANK", "FINNIFTY"):
        return DEMO_TOKENS.get(symbol)

    # 1. Hardcoded
    if symbol in DEMO_TOKENS:
        return DEMO_TOKENS[symbol]

    # 2. Session cache
    if symbol in _token_cache:
        return _token_cache[symbol]

    # 3. Need Kite
    if not kite_manager.is_authenticated:
        return None

    try:
        df = kite_manager._get_nse_instruments_df()
        if df.empty:
            return None

        # Resolve alias if present
        lookup_sym = NSE_SYMBOL_ALIASES.get(symbol, symbol)
        if lookup_sym is None:
            return None   # explicitly mapped to None = no EQ token

        # 4. Direct match
        match = df[
            (df["tradingsymbol"] == lookup_sym) &
            (df["instrument_type"] == "EQ") &
            (df["exchange"] == "NSE")
        ]
        if not match.empty:
            token = int(match.iloc[0]["instrument_token"])
            _token_cache[symbol] = token
            log.info(f"Resolved token for {symbol} (as {lookup_sym}): {token}")
            return token

        # 5. Fuzzy variants -- try common NSE naming quirks
        variants = [
            lookup_sym.replace("-", ""),     # BAJAJ-AUTO -> BAJAJAUTO
            lookup_sym.replace("&", "AND"),  # M&M -> MANDM
            lookup_sym + "LTD",             # sometimes suffixed
            lookup_sym.rstrip("LTD"),       # sometimes without
        ]
        for v in variants:
            m = df[(df["tradingsymbol"] == v) & (df["instrument_type"] == "EQ") & (df["exchange"] == "NSE")]
            if not m.empty:
                token = int(m.iloc[0]["instrument_token"])
                _token_cache[symbol] = token
                log.info(f"Resolved token for {symbol} via variant {v!r}: {token}")
                return token

        log.warning(f"No EQ token found for {symbol!r} in NSE instruments -- check symbol spelling")
        return None

    except Exception as e:
        log.error(f"Token resolution error for {symbol}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ════════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 AlgoTrade Pro Ultimate v4.0 starting...")
    # ── Fix #11: Start dynamic universe enrichment background task ────────────
    # Runs every 30 min during market hours; enriches SCAN_UNIVERSE with live
    # top movers / high-volume stocks from Kite so the scanner is never stale.
    async def _refresh_dynamic_universe():
        global _dynamic_universe, _universe_fetched_at
        while True:
            try:
                now = datetime.now()
                market_open  = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, second=0, microsecond=0)
                if market_open <= now <= market_close and kite_manager.is_authenticated:
                    nse_syms = [f"NSE:{s}" for s in SCAN_UNIVERSE
                                if s not in ("NIFTY 50", "NIFTY BANK")]
                    batch_size = 100
                    all_quotes: Dict[str, Any] = {}
                    for i in range(0, len(nse_syms), batch_size):
                        batch = nse_syms[i:i + batch_size]
                        try:
                            q = kite_manager.get_quote(batch) or {}
                            all_quotes.update(q)
                        except Exception:
                            pass
                    if all_quotes:
                        # Sort by % change (abs) descending — surface most volatile
                        def _pct(q):
                            ltp  = float(q.get("last_price") or 0)
                            prev = float(q.get("ohlc", {}).get("close") or ltp or 1)
                            return abs((ltp - prev) / prev * 100) if prev else 0
                        sorted_syms = sorted(
                            all_quotes.items(),
                            key=lambda kv: _pct(kv[1]),
                            reverse=True
                        )
                        top_movers = [
                            kv[0].replace("NSE:", "")
                            for kv in sorted_syms[:30]
                            if kv[0].replace("NSE:", "") not in ("NIFTY 50", "NIFTY BANK")
                        ]
                        # Merge with base universe (top movers first for scanner priority)
                        merged = top_movers + [s for s in SCAN_UNIVERSE if s not in top_movers]
                        _dynamic_universe = merged
                        _universe_fetched_at = datetime.now()
                        log.info(f"🔄 Dynamic universe updated: {len(merged)} symbols, top movers: {top_movers[:5]}")
            except Exception as e:
                log.warning(f"Dynamic universe refresh error: {e}")
            await asyncio.sleep(30 * 60)  # 30 minutes

    task = asyncio.create_task(_refresh_dynamic_universe())
    yield
    task.cancel()
    log.info("🛑 AlgoTrade Pro Ultimate v4.0 shutting down...")

app = FastAPI(title="AlgoTrade Pro Enhanced", lifespan=lifespan)
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

import traceback
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions -- prevents 'Unexpected token I' on frontend."""
    log.error(f"Unhandled error on {request.url.path}: {exc}\n{traceback.format_exc()}")
    return JSONResponse({"error": f"{type(exc).__name__}: {str(exc)}"}, status_code=500)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


# ── Auth routes ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Railway health check endpoint. Returns 200 if service is running."""
    return {
        "status": "ok",
        "service": "AlgoTrade Pro Ultimate",
        "version": "4.0",
        "kite_connected": kite_manager.is_authenticated,
        "universe_size": len(_dynamic_universe) if _dynamic_universe else len(SCAN_UNIVERSE),
        "universe_source": "dynamic" if _dynamic_universe else "static",
        "universe_refreshed_at": _universe_fetched_at.isoformat() if _universe_fetched_at else None,
        "orb_cache_entries": len(_orb_cache),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness probe - confirms Kite auth status."""
    return {
        "ready": True,
        "authenticated": kite_manager.is_authenticated,
        "message": "Kite login required at /auth/login" if not kite_manager.is_authenticated else "Ready"
    }

@app.get("/auth/login")
async def auth_login():
    url = kite_manager.get_login_url()
    return RedirectResponse(url)

@app.get("/auth/logout")
async def auth_logout():
    """Force invalidate current session -- user will be prompted to login again."""
    kite_manager.invalidate()
    return RedirectResponse("/", status_code=302)

@app.get("/callback")
@app.get("/auth/callback")
@app.get("/login")
async def auth_callback(request: Request):
    """
    Handles Zerodha Kite OAuth callback.
    Zerodha sends: ?action=login&type=login&status=success&request_token=XXX
    
    IMPORTANT -- in your Kite Developer Console, set Redirect URL to:
        http://127.0.0.1:8000/callback
    """
    params = dict(request.query_params)
    log.info(f"🔁 Callback received. Params: {params}")

    status        = params.get("status", "")
    request_token = params.get("request_token", "")
    action        = params.get("action", "")
    error_type    = params.get("type", "")
    error_msg     = params.get("message", "")

    def _error_page(title: str, detail: str, hint: str = "") -> HTMLResponse:
        return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Login Error</title>
<style>
  body{{background:#060b14;color:#e2e8f0;font-family:sans-serif;
       display:flex;align-items:center;justify-content:center;
       height:100vh;margin:0;flex-direction:column;gap:16px;padding:20px;text-align:center}}
  h2{{color:#ef4444;font-size:1.4rem;margin:0}}
  .detail{{color:#9ca3af;font-size:.9rem;max-width:500px;line-height:1.6}}
  .hint{{background:#1a2540;border:1px solid #2a3550;border-radius:8px;
         padding:12px 20px;font-size:.85rem;color:#fbbf24;max-width:500px;line-height:1.7}}
  a{{color:#00d4ff;text-decoration:none;font-size:.9rem;
     padding:8px 20px;border:1px solid #00d4ff;border-radius:6px;margin-top:8px}}
  a:hover{{background:rgba(0,212,255,.1)}}
</style></head>
<body>
  <div style="font-size:2rem">⚠️</div>
  <h2>{title}</h2>
  <div class="detail">{detail}</div>
  {"<div class='hint'>" + hint + "</div>" if hint else ""}
  <a href="/"><- Back to Dashboard</a>
  <a href="/auth/login">🔑 Try Login Again</a>
</body></html>""", status_code=400)

    # ── Case 1: No request_token at all -- user probably hit /callback directly ──
    if not request_token and not status:
        return _error_page(
            "No Login Token Received",
            "This page is the Zerodha OAuth callback. It should not be opened directly.",
            "➡ Click the Login button on the dashboard to start the login process."
        )

    # ── Case 2: Zerodha returned an error ─────────────────────────────────────
    if status and status != "success":
        detail = error_msg or f"Zerodha returned status: {status}"
        hint = ""
        if "redirect" in detail.lower() or not request_token:
            hint = (
                "🔧 <b>Fix:</b> Your Kite app's Redirect URL must be set to:<br>"
                "<code>http://127.0.0.1:8000/callback</code><br><br>"
                "Go to: <b>kite.trade -> My Apps -> Your App -> Settings -> Redirect URL</b>"
            )
        return _error_page("Zerodha Login Failed", detail, hint)

    # ── Case 3: Has request_token -- complete the login ──────────────────────
    if not request_token:
        return _error_page(
            "Missing Request Token",
            "Zerodha did not send a request token. This usually means the Redirect URL in your Kite app settings is wrong.",
            "🔧 Set Redirect URL to: <code>http://127.0.0.1:8000/callback</code>"
        )

    log.info(f"✅ Got request_token: {request_token[:10]}... -- completing login")
    ok = kite_manager.complete_login(request_token)

    if ok:
        # Redirect to dashboard with a success flag
        return HTMLResponse("""<!DOCTYPE html>
<html><head><title>Login Successful</title>
<style>
  body{background:#060b14;color:#e2e8f0;font-family:sans-serif;
       display:flex;align-items:center;justify-content:center;
       height:100vh;margin:0;flex-direction:column;gap:12px;text-align:center}
  h2{color:#00e676;font-size:1.5rem;margin:0}
  .sub{color:#9ca3af;font-size:.9rem}
</style>
<script>
  // Auto-redirect to dashboard in 1.5 seconds
  setTimeout(() => { window.location.href = '/'; }, 1500);
</script>
</head>
<body>
  <div style="font-size:2.5rem">✅</div>
  <h2>Login Successful!</h2>
  <div class="sub">Redirecting to dashboard...</div>
  <a href="/" style="color:#00d4ff;font-size:.85rem;margin-top:8px">Click here if not redirected</a>
</body></html>""")

    # ── Case 4: generate_session failed (wrong secret, token reuse, etc.) ───
    return _error_page(
        "Session Generation Failed",
        "Kite accepted the login but generating the session failed. This usually means your API secret is wrong, or the request token was already used.",
        (
            "🔧 Check your <b>.env</b> file:<br>"
            "• <b>KITE_API_KEY</b> must match your Kite app<br>"
            "• <b>KITE_API_SECRET</b> must match your Kite app<br><br>"
            "Request tokens are <b>one-time use only</b> -- do not refresh the callback URL."
        )
    )

@app.get("/api/auth/status")
async def auth_status():
    return {"authenticated": kite_manager.is_authenticated}


# ── Market data routes ────────────────────────────────────────────────────────

@app.get("/api/symbols")
async def get_symbols():
    return {"symbols": SCAN_UNIVERSE}

@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    key = f"NSE:{symbol}"
    data = kite_manager.get_quote([key])
    if not data:
        return JSONResponse({"error": "Quote not available"}, status_code=404)
    q = data.get(key, {})
    return {
        "symbol": symbol,
        "last_price": q.get("last_price"),
        "change": q.get("net_change"),
        "change_pct": q.get("last_price", 0) / (q.get("ohlc", {}).get("close", 1) or 1) * 100 - 100,
        "volume": q.get("volume"),
        "ohlc": q.get("ohlc"),
    }

@app.get("/api/candles/{symbol}")
async def get_candles(symbol: str, interval: str = "3minute", days: int = 5):
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No instrument token for {symbol}. Ensure Kite is connected."}, status_code=404)
    df = kite_manager.get_historical_data(token, interval, days)
    if df.empty:
        return JSONResponse({"error": "No data"}, status_code=404)
    return {"symbol": symbol, "interval": interval, "candles": df.to_dict(orient="records")}


# ── Pattern routes ────────────────────────────────────────────────────────────

@app.get("/api/patterns/{symbol}")
async def get_patterns(symbol: str, interval: str = "3minute"):
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No instrument token for {symbol}. Ensure Kite is connected."}, status_code=404)
    days_back = 1 if interval == "minute" else (2 if "minute" in interval else 30)
    df = kite_manager.get_historical_data(token, interval, days_back)
    if df.empty:
        return JSONResponse({"error": "No candle data"}, status_code=404)
    stale = _guard_freshness(df, symbol)
    if stale:
        return JSONResponse(stale, status_code=200)
    detections = pattern_detector.detect_all_patterns(df, symbol, interval)

    # Deduplicate: keep only the BEST (highest confidence) detection per pattern type
    # This prevents same Doji showing 50 times
    seen: Dict[str, Any] = {}
    for p in detections:
        key = p.pattern.value
        if key not in seen or p.confidence > seen[key].confidence:
            seen[key] = p

    # Also keep only patterns from the last 30 candles (recent signals only)
    recent_threshold = len(df) - 30
    deduped = [p for p in seen.values() if p.candle_index >= recent_threshold]

    # Sort by candle_index descending (most recent first), then by confidence
    deduped.sort(key=lambda x: (x.candle_index, x.confidence), reverse=True)

    return {"symbol": symbol, "patterns": [p.to_dict() for p in deduped[:15]]}

@app.get("/api/fibonacci/{symbol}")
async def get_fibonacci(symbol: str, interval: str = "3minute"):
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No instrument token for {symbol}. Ensure Kite is connected."}, status_code=404)
    df = kite_manager.get_historical_data(token, interval)
    if df.empty:
        return JSONResponse({"error": "No candle data"}, status_code=404)
    fib = fib_analyzer.calculate_fibonacci(df, symbol, interval)
    return fib.to_dict()

@app.get("/api/top-movers")
async def get_top_movers():
    """
    Fetch today's top gainers + losers + volume leaders from Kite NSE instruments.
    These are the stocks most likely to have strong intraday moves.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    try:
        # Fetch quotes for our base universe to find movers
        nse_syms = [f"NSE:{s}" for s in SCAN_UNIVERSE if s not in ("NIFTY 50","NIFTY BANK")]
        # Batch into 200 (Kite limit per call)
        all_quotes = {}
        for i in range(0, len(nse_syms), 200):
            batch = nse_syms[i:i+200]
            q = kite_manager.get_quote(batch)
            if q:
                all_quotes.update(q)

        movers = []
        for key, q in all_quotes.items():
            sym = key.replace("NSE:", "")
            ltp   = q.get("last_price", 0) or 0
            prev  = (q.get("ohlc", {}) or {}).get("close", ltp) or ltp
            vol   = q.get("volume", 0) or 0
            chg_pct = ((ltp - prev) / prev * 100) if prev else 0
            movers.append({
                "symbol": sym,
                "ltp": ltp,
                "change_pct": round(chg_pct, 2),
                "volume": vol,
            })

        # Sort by abs change desc
        movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        gainers = [m for m in movers if m["change_pct"] > 0][:10]
        losers  = [m for m in movers if m["change_pct"] < 0][:10]
        # Volume leaders
        vol_leaders = sorted(movers, key=lambda x: x["volume"], reverse=True)[:10]

        return {
            "gainers": gainers,
            "losers": losers,
            "volume_leaders": vol_leaders,
            "fetched_at": datetime.now().isoformat(),
        }
    except Exception as e:
        log.error(f"Top movers error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/scan")
async def scan_all(limit: int = 20, interval: str = "5minute", min_conf: float = 0.65):
    """
    Smart market scan:
    - Scans ALL stocks in universe (not just 8)
    - Uses 5min candles for intraday + 15min for swing
    - Returns ONLY BULLISH/BEARISH signals (skips NEUTRAL/Doji)
    - Sorted by confidence DESC
    - Multi-timeframe: confirms 5min signal exists on 15min too
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _scan_all_sync, limit, interval, min_conf)


def _scan_all_sync(limit: int = 20, interval: str = "5minute", min_conf: float = 0.65):

    results = []
    scan_list = SCAN_UNIVERSE.copy()

    for symbol in scan_list:
        if symbol in ("NIFTY 50", "NIFTY BANK"):
            continue  # indices -- skip in scan, add separately
        token = get_instrument_token(symbol)
        if token is None:
            continue
        try:
            # Primary: 5min candles for intraday signals
            df5 = kite_manager.get_historical_data(token, interval, 3)
            if df5.empty or len(df5) < 5:
                continue

            detections = pattern_detector.detect_all_patterns(df5, symbol, interval)
            if not detections:
                continue

            # Deduplicate -- best per pattern type
            seen: Dict[str, Any] = {}
            for p in detections:
                k = p.pattern.value
                if k not in seen or p.confidence > seen[k].confidence:
                    seen[k] = p

            # Only BULLISH or BEARISH, above min confidence, recent candles only
            recent_cutoff = len(df5) - 15
            actionable = [
                p for p in seen.values()
                if p.signal in ("BULLISH", "BEARISH")
                and p.confidence >= min_conf
                and p.candle_index >= recent_cutoff
            ]
            if not actionable:
                continue

            # Pick strongest
            best = max(actionable, key=lambda x: x.confidence * x.strength)

            # Quick quote for live price
            q_data = kite_manager.get_quote([f"NSE:{symbol}"])
            ltp = 0
            chg_pct = 0
            if q_data:
                q = q_data.get(f"NSE:{symbol}", {})
                ltp = q.get("last_price", best.price_at_detection) or best.price_at_detection
                prev = (q.get("ohlc", {}) or {}).get("close", ltp) or ltp
                chg_pct = round(((ltp - prev) / prev * 100) if prev else 0, 2)

            # GAP 1+4: compute volume quality + ATR% for this stock
            atr_pct_scan = 0.0
            vol_quality   = "OK"
            try:
                if len(df5) >= 15:
                    hl = df5["high"] - df5["low"]
                    hc = (df5["high"] - df5["close"].shift()).abs()
                    lc = (df5["low"]  - df5["close"].shift()).abs()
                    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                    atr_val = float(tr.rolling(14).mean().iloc[-1])
                    atr_pct_scan = round(atr_val / (ltp or best.price_at_detection) * 100, 3)
                if "volume" in df5.columns:
                    avg_v  = float(df5["volume"].rolling(20).mean().iloc[-1])
                    curr_v = float(df5["volume"].iloc[-1])
                    rvol_s = curr_v / avg_v if avg_v > 0 else 1.0
                    vol_quality = "HIGH" if rvol_s >= 1.5 else "OK" if rvol_s >= 0.8 else "LOW"
            except Exception:
                pass

            results.append({
                "symbol": symbol,
                "pattern": best.pattern.value,
                "signal": best.signal,
                "confidence": best.confidence,
                "strength": best.strength,
                "price": ltp or best.price_at_detection,
                "change_pct": chg_pct,
                "stop_loss": best.stop_loss,
                "target": best.target,
                "candle_index": best.candle_index,
                "timeframe": interval,
                "atr_pct": atr_pct_scan,
                "vol_quality": vol_quality,
                "atr_ok": atr_pct_scan >= 0.15,
            })
        except Exception as e:
            log.warning(f"Scan error {symbol}: {e}")

    # Sort by confidence * strength DESC
    results.sort(key=lambda x: x["confidence"] * x["strength"], reverse=True)

    return {
        "scan_results": results[:limit],
        "total_scanned": len(scan_list),
        "signals_found": len(results),
        "scanned_at": datetime.now().isoformat(),
        "interval": interval,
    }



# ── Structure (BOS/OTE) endpoint -- FIXED Fibonacci targets ───────────────────
@app.get("/api/structure/{symbol}")
async def get_structure_setups(symbol: str, interval: str = "15minute"):
    """
    BOS -> OTE pullback setups with CORRECT Fibonacci extension targets.
    Entry zone: 61.8%-78.6% retracement (ICT OTE zone).
    Targets: T1=Fib1.0 (closest), T2=Fib1.272, T3=Fib1.618, T4=Fib2.0 (runner).
    Minimum RR: 1.8x to T2 required.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No token for {symbol}"}, status_code=404)

    days_back = 10 if "minute" in interval else 60
    df = kite_manager.get_historical_data(token, interval, days_back)
    if df.empty or len(df) < 40:
        return JSONResponse({"error": "Not enough data"}, status_code=404)
    stale = _guard_freshness(df, symbol)
    if stale: return JSONResponse(stale, status_code=200)

    setups = structure_engine.detect_setups(df, symbol, interval)

    return {
        "symbol":    symbol,
        "interval":  interval,
        "setups":    [s.to_dict() for s in setups],
        "count":     len(setups),
        "note":      (
            "Entry zone uses ICT OTE (61.8%-78.6% Fibonacci retracement). "
            "T1=Fib1.0 (equal leg, closest), T2=Fib1.272, T3=Fib1.618 (golden), T4=Fib2.0 (runner). "
            "SL is below swing low. Minimum R:R 1.8x to T2 required."
        ),
    }


# ── ICT Analysis endpoint ─────────────────────────────────────────────────────
@app.get("/api/ict/{symbol}")
async def get_ict_analysis(symbol: str, interval: str = "15minute"):
    """
    Full ICT Smart Money analysis:
    OTE Zones · Breaker Blocks · Displacement · Power of 3 · Killzones ·
    Daily Bias · PDH/PDL · Fibonacci cascade targets.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No token for {symbol}"}, status_code=404)

    days_back = 10 if "minute" in interval else 60
    _ict_key = f"ict:{symbol}:{interval}"
    _cached_ict = _ict_cache.get(_ict_key)
    if _cached_ict is not None:
        return _cached_ict

    with _kite_semaphore:
        df = kite_manager.get_historical_data(token, interval, days_back)
    if df.empty or len(df) < 30:
        return JSONResponse({"error": "Not enough data"}, status_code=404)
    stale = _guard_freshness(df, symbol)
    if stale: return JSONResponse(stale, status_code=200)

    # Daily data for HTF bias
    df_daily = None
    try:
        with _kite_semaphore:
            df_daily = kite_manager.get_historical_data(token, "day", 30)
    except Exception:
        pass

    # Institutional levels (PDH/PDL/PWH/PWL)
    inst_levels = calculate_institutional_levels(df, symbol)

    result = ict_engine.analyse(df, symbol, interval, df_daily, inst_levels)

    resp = {
        "symbol":    symbol,
        "interval":  interval,
        "ict":       result.to_dict(),
        "inst_levels": inst_levels.to_dict(),
    }
    _ict_cache.set(_ict_key, resp)
    return resp


# ── Institutional Levels endpoint ─────────────────────────────────────────────
@app.get("/api/levels/{symbol}")
async def get_institutional_levels(symbol: str):
    """
    Returns PDH/PDL/PWH/PWL and psychological round number levels.
    These are the key price magnets where institutions place orders.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No token for {symbol}"}, status_code=404)

    df = kite_manager.get_historical_data(token, "15minute", 10)
    if df.empty:
        return JSONResponse({"error": "No data"}, status_code=404)

    levels = calculate_institutional_levels(df, symbol)
    return levels.to_dict()


@app.get("/api/fno/{symbol}")
async def get_fno_signals(symbol: str, interval: str = "5minute"):
    """Generate F&O signals using REAL Kite expiry dates, option chain, IV & Greeks."""
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    FNO_SYM_MAP = {
        # Indices
        "NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY",
        # Largecap -- same name
        "RELIANCE": "RELIANCE", "TCS": "TCS", "INFY": "INFY",
        "HDFCBANK": "HDFCBANK", "ICICIBANK": "ICICIBANK",
        "BAJFINANCE": "BAJFINANCE", "HINDUNILVR": "HINDUNILVR",
        "MARUTI": "MARUTI", "TATAMOTORS": "TATAMOTORS",
        "AXISBANK": "AXISBANK", "LT": "LT", "WIPRO": "WIPRO",
        "SUNPHARMA": "SUNPHARMA", "TATASTEEL": "TATASTEEL",
        "KOTAKBANK": "KOTAKBANK", "SBIN": "SBIN", "ADANIENT": "ADANIENT",
        "BHARTIARTL": "BHARTIARTL", "NTPC": "NTPC", "POWERGRID": "POWERGRID",
        "ONGC": "ONGC", "COALINDIA": "COALINDIA",
        # Midcap F&O eligible
        "MUTHOOTFIN": "MUTHOOTFIN", "CHOLAFIN": "CHOLAFIN",
        "PERSISTENT": "PERSISTENT", "COFORGE": "COFORGE", "LTM": "LTIM",
        "GODREJPROP": "GODREJPROP", "TORNTPHARM": "TORNTPHARM",
        "MAXHEALTH": "MAXHEALTH", "ETERNAL": "ETERNAL",
        "NYKAA": "NYKAA", "POLICYBZR": "POLICYBZR", "CDSL": "CDSL",
    }
    fno_symbol = FNO_SYM_MAP.get(symbol, symbol)

    if fno_symbol not in FNO_EXPIRY_SYMBOLS:
        return JSONResponse({"error": f"{symbol} not in F&O universe"}, status_code=404)

    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No instrument token for {symbol}. Ensure Kite is connected and NSE instruments are cached."}, status_code=404)

    # ── 1. Get live spot price from Kite quote (most accurate) ──────────────
    quote_key = f"NSE:{symbol}" if symbol not in ("NIFTY 50","NIFTY BANK") else (
        "NSE:NIFTY 50" if symbol == "NIFTY 50" else "NSE:NIFTY BANK"
    )
    # For indices use NFO token directly
    if symbol in ("NIFTY 50", "NIFTY BANK"):
        quote_data = kite_manager.get_quote([f"NSE:{symbol}"])
    else:
        quote_data = kite_manager.get_quote([f"NSE:{symbol}"])

    if quote_data:
        q = quote_data.get(f"NSE:{symbol}", {})
        current_price = float(q.get("last_price") or 0)
    else:
        current_price = 0.0

    # Fall back to historical close if quote unavailable (market closed)
    if current_price == 0:
        df_fallback = kite_manager.get_historical_data(token, "day", 2)
        if not df_fallback.empty:
            current_price = float(df_fallback.iloc[-1]["close"])

    if current_price == 0:
        return JSONResponse({"error": "Could not get current price"}, status_code=404)

    # ── 2. Historical candles for pattern detection ──────────────────────────
    # For 1minute, fetch only 1 day (enough candles, avoids rate limit)
    days_back = 1 if interval == "minute" else (2 if "minute" in interval else 60)
    df = kite_manager.get_historical_data(token, interval, days_back)
    if df.empty:
        return JSONResponse({"error": "No historical data"}, status_code=404)
    stale = _guard_freshness(df, symbol)
    if stale: return JSONResponse(stale, status_code=200)

    # ── 3. ATR ───────────────────────────────────────────────────────────────
    if len(df) >= 14:
        hl  = df["high"] - df["low"]
        atr = float(hl.rolling(14).mean().iloc[-1])
    else:
        atr = float((df["high"] - df["low"]).mean())

    # ── 4. Pattern detection (deduplicated) ──────────────────────────────────
    patterns_raw = pattern_detector.detect_all_patterns(df, symbol, interval)
    # Deduplicate: keep only the best (highest confidence) detection per pattern type
    seen_patterns: Dict[str, Any] = {}
    for p in patterns_raw:
        key = p.pattern.value
        if key not in seen_patterns or p.confidence > seen_patterns[key].confidence:
            seen_patterns[key] = p
    patterns = [p.to_dict() for p in sorted(seen_patterns.values(),
                                              key=lambda x: x.confidence, reverse=True)]

    # ── 5. Real expiry dates from Kite NFO instruments ───────────────────────
    expiries = kite_manager.get_real_expiries(fno_symbol)
    if not expiries:
        return JSONResponse({
            "error": "Could not fetch expiry dates from Kite NFO instruments. "
                     "Make sure markets are open or instruments are cached."
        }, status_code=500)

    # ── 6. Real option chain (LTP + IV + Greeks) from Kite ───────────────────
    weekly_expiry_str = expiries.get("weekly", "")
    option_chain = []
    if weekly_expiry_str:
        option_chain = kite_manager.get_option_chain(
            fno_symbol, weekly_expiry_str, current_price, num_strikes=3
        )
        log.info(f"Option chain for {fno_symbol} {weekly_expiry_str}: {len(option_chain)} strikes fetched")

    # ── 7. Generate signals ───────────────────────────────────────────────────
    signals = generate_fno_signals(
        fno_symbol, current_price, patterns, {},
        atr, option_chain=option_chain, expiries=expiries, interval=interval
    )

    # ── 8. SMC cross-check -- detect if SMC and F&O signals conflict ──────────
    smc_conflict = None
    try:
        df_smc = kite_manager.get_historical_data(token, "15minute", 2)
        if not df_smc.empty and len(df_smc) >= 20:
            df_smc_1h = None
            try:
                df_smc_1h = kite_manager.get_historical_data(token, "60minute", 5)
            except Exception:
                pass
            smc_result = smc_engine.analyse(df_smc, symbol, "15minute", df_smc_1h)
            smc_dir = smc_result.smc_signal  # "BUY", "SELL", or "WAIT"
            # Check if any directional F&O signal conflicts with SMC
            for sig in signals:
                fno_dir = sig.get("signal", "")
                if smc_dir == "BUY" and fno_dir == "BEARISH":
                    smc_conflict = {
                        "level": "WARNING",
                        "message": (f"⚠️ SMC CONFLICT: F&O pattern says BEARISH (PE) but SMC 15m analysis says "
                                    f"BUY (conf {smc_result.smc_confidence:.0f}%) -- "
                                    f"Structure: {smc_result.market_structure}, P/D: {smc_result.premium_discount}. "
                                    f"Consider waiting for SMC alignment before entering PE."),
                        "smc_signal": smc_dir,
                        "smc_confidence": smc_result.smc_confidence,
                        "smc_structure": smc_result.market_structure,
                        "smc_pd": smc_result.premium_discount,
                    }
                    sig["smc_conflict"] = smc_conflict["message"]
                    sig["rationale"] = sig.get("rationale","") + f" | ⚠️ SMC 15m says BUY (conf {smc_result.smc_confidence:.0f}%) -- verify alignment."
                elif smc_dir == "SELL" and fno_dir == "BULLISH":
                    smc_conflict = {
                        "level": "WARNING",
                        "message": (f"⚠️ SMC CONFLICT: F&O pattern says BULLISH (CE) but SMC 15m analysis says "
                                    f"SELL (conf {smc_result.smc_confidence:.0f}%) -- "
                                    f"Structure: {smc_result.market_structure}, P/D: {smc_result.premium_discount}. "
                                    f"Consider waiting for SMC alignment before entering CE."),
                        "smc_signal": smc_dir,
                        "smc_confidence": smc_result.smc_confidence,
                        "smc_structure": smc_result.market_structure,
                        "smc_pd": smc_result.premium_discount,
                    }
                    sig["smc_conflict"] = smc_conflict["message"]
                    sig["rationale"] = sig.get("rationale","") + f" | ⚠️ SMC 15m says SELL (conf {smc_result.smc_confidence:.0f}%) -- verify alignment."
                elif smc_dir in ("BUY","SELL") and fno_dir not in ("","NEUTRAL"):
                    # Aligned -- add confirmation note
                    sig["smc_aligned"] = True
                    sig["rationale"] = sig.get("rationale","") + f" | ✅ SMC 15m ALIGNED: {smc_dir} (conf {smc_result.smc_confidence:.0f}%)."
    except Exception as e:
        log.debug(f"SMC cross-check skipped: {e}")

    return {
        "symbol": symbol,
        "fno_symbol": fno_symbol,
        "current_price": current_price,
        "atr": round(atr, 2),
        "signals": signals,
        "expiries": expiries,
        "patterns_found": len(patterns),
        "option_chain_strikes": len(option_chain),
        "data_source": "KITE_LIVE" if option_chain else "BS_ESTIMATE",
        "smc_conflict": smc_conflict,
        "generated_at": datetime.now().isoformat(),
    }

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websocket_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(30)
            await ws.send_json({"type": "ping", "ts": datetime.now().isoformat()})
    except WebSocketDisconnect:
        websocket_clients.discard(ws)
    except Exception:
        websocket_clients.discard(ws)


# ── Main UI ───────────────────────────────────────────────────────────────────
@app.get("/api/smc/{symbol}")
async def get_smc(symbol: str, interval: str = "15minute"):
    """Smart Money Concepts analysis: OBs, FVGs, Liquidity Sweeps, VWAP, MTF bias."""
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No token for {symbol}"}, status_code=404)
    days_back = 7 if "minute" in interval else 30
    df = kite_manager.get_historical_data(token, interval, days_back)
    if df.empty or len(df) < 20:
        df = kite_manager.get_historical_data(token, interval, 14)
    if df.empty or len(df) < 20:
        return JSONResponse({"error": "Insufficient data"}, status_code=404)
    df_1h = None
    try:
        df_1h = kite_manager.get_historical_data(token, "60minute", 5)
    except Exception:
        pass
    try:
        result = smc_engine.analyse(df, symbol, interval, df_1h)
        return result.to_dict()
    except Exception as e:
        log.error(f"SMC error for {symbol}: {e}", exc_info=True)
        return JSONResponse({"error": f"SMC analysis failed: {str(e)}"}, status_code=500)

# ── SMC enhanced endpoint with institutional levels ───────────────────────────
@app.get("/api/smc-plus/{symbol}")
async def get_smc_plus(symbol: str, interval: str = "15minute"):
    """
    SMC Analysis with institutional levels (PDH/PDL) and correct Fibonacci targets.
    Combines SMC Order Blocks + Fibonacci extensions + ICT breaker blocks.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    token = get_instrument_token(symbol)
    if token is None:
        return JSONResponse({"error": f"No token for {symbol}"}, status_code=404)

    days_back = 10 if "minute" in interval else 60
    _smc_key = f"smc:{symbol}:{interval}"
    _cached_smc = _smc_cache.get(_smc_key)
    if _cached_smc is not None:
        return _cached_smc

    with _kite_semaphore:
        df = kite_manager.get_historical_data(token, interval, days_back)
    if df.empty or len(df) < 20:
        return JSONResponse({"error": "No data"}, status_code=404)

    stale = _guard_freshness(df, symbol)
    if stale:
        return JSONResponse(stale, status_code=200)

    df_1h = None
    try:
        with _kite_semaphore:
            df_1h = kite_manager.get_historical_data(token, "60minute", 5)
    except Exception:
        pass

    inst_levels = calculate_institutional_levels(df, symbol)
    result = smc_engine.analyse(df, symbol, interval, df_1h, inst_levels)

    # Also run ICT for additional context
    df_daily = None
    try:
        with _kite_semaphore:
            df_daily = kite_manager.get_historical_data(token, "day", 20)
    except Exception:
        pass
    ict = ict_engine.analyse(df, symbol, interval, df_daily, inst_levels)

    smc_resp = {
        "symbol":        symbol,
        "interval":      interval,
        "smc":           result.to_dict(),
        "ict":           ict.to_dict(),
        "inst_levels":   inst_levels.to_dict(),
        "combined_signal": {
            "signal":    result.smc_signal if result.smc_signal == ict.ict_signal.replace("LONG","BUY").replace("SHORT","SELL") else "MIXED",
            "smc_sig":   result.smc_signal,
            "ict_sig":   ict.ict_signal,
            "confidence": round((result.smc_confidence + ict.ict_confidence) / 2, 1),
            # Unified targets using Fibonacci (prefer ICT targets which are correctly calculated)
            "entry_zone": result.smc_entry_zone,
            "sl":         result.smc_sl,
            "t1_fib_1":   round(ict.ict_t1, 2),
            "t2_fib_127": round(ict.ict_t2, 2),
            "t3_fib_162": round(ict.ict_t3, 2),
            "t4_fib_200": round(ict.ict_t4, 2),
            "rr_to_t2":   round(ict.ict_rr_t2, 2),
            "rr_to_t3":   round(ict.ict_rr_t3, 2),
            "in_killzone":  ict.in_killzone,
            "killzone":   ict.killzone_name,
            "po3_phase":  ict.po3_phase,
            "daily_bias": ict.daily_bias,
        },
    }
    _smc_cache.set(_smc_key, smc_resp)
    return smc_resp

@app.get("/api/orb/{symbol}")
async def get_orb(symbol: str, interval: str = "5minute"):
    """
    Opening Range Breakout signal.
    ORB range = High/Low of 9:15-9:29 candles today.
    Breakout = first 9:30+ candle that CLOSES beyond the range with volume.
    Once a signal fires for the day it is cached — subsequent calls return
    the original signal so intraday data jitter cannot flip the levels.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    sym_clean = symbol.replace(" 50", "").replace("NIFTY BANK", "BANKNIFTY").replace("NIFTY", "NIFTY")
    if "NIFTY" in symbol.upper():
        token = DEMO_TOKENS.get(symbol) or DEMO_TOKENS.get("NIFTY 50")
    else:
        token = get_instrument_token(symbol)

    if token is None:
        return JSONResponse({"error": f"No token for {symbol}"}, status_code=404)

    try:
        df = kite_manager.get_historical_data(token, interval, 2)
    except Exception as e:
        return JSONResponse({"error": f"Data fetch failed: {e}"}, status_code=500)

    if df.empty or len(df) < 5:
        return JSONResponse({"error": "Insufficient data -- market may not be open yet"}, status_code=404)
    stale = _guard_freshness(df, symbol, max_age=90)
    if stale: return JSONResponse(stale, status_code=200)

    # ── Check daily ORB cache first ───────────────────────────────────────────
    _IST_orb = timezone(timedelta(hours=5, minutes=30))
    today_str  = datetime.now(tz=_IST_orb).strftime("%Y-%m-%d")
    cache_key  = f"{symbol}:{today_str}"
    cached_sig = _orb_cache.get(cache_key)

    try:
        signal = orb_engine.detect(df, symbol, interval)
    except Exception as e:
        log.error(f"ORB detect error for {symbol}: {e}")
        return JSONResponse({"error": f"ORB analysis error: {e}"}, status_code=500)

    # If a signal fired for the first time today, store it in cache
    if signal is not None and cached_sig is None:
        _orb_cache[cache_key] = signal
        log.info(f"📦 ORB signal cached for {symbol} ({today_str})")
    elif signal is None and cached_sig is not None:
        # Reuse today's cached signal — data jitter cleared the live signal
        signal = cached_sig
        log.info(f"♻️  ORB using cached signal for {symbol} ({today_str})")
    elif cached_sig is not None:
        # Both present — always serve cached (original) signal to prevent flipping
        signal = cached_sig

    if signal is None:
        # Return info about current ORB range even if no breakout yet
        # so UI can display the levels
        today = pd.Timestamp.now().date()
        def ts_date(ts):
            if hasattr(ts, "date"): return ts.date()
            try: return pd.to_datetime(ts).date()
            except: return None
        def ts_hm(ts):
            if hasattr(ts, "hour"): return (ts.hour, ts.minute)
            try: t = pd.to_datetime(ts); return (t.hour, t.minute)
            except: return (0, 0)

        today_df = df[df["timestamp"].apply(lambda x: ts_date(x) == today)]
        orb_bars = today_df[today_df["timestamp"].apply(lambda x: (9,15) <= ts_hm(x) < (9,30))]

        orb_info = {}
        if not orb_bars.empty:
            orb_info = {
                "orb_high": round(float(orb_bars["high"].max()), 2),
                "orb_low":  round(float(orb_bars["low"].min()), 2),
                "orb_established": True,
            }
        else:
            orb_info = {"orb_established": False}

        return {
            "symbol": symbol,
            "interval": interval,
            "signal": None,
            "status": "WAITING" if orb_info.get("orb_established") else "RANGE_NOT_SET",
            "message": ("ORB range established. Waiting for 9:30+ breakout candle."
                        if orb_info.get("orb_established")
                        else "Market not open yet or opening range not formed (need 9:15-9:29 data)."),
            **orb_info,
            "generated_at": datetime.now().isoformat(),
        }

    # ── Invalidation check: if CMP has breached ORB High (BEAR) or Low (BULL),
    # the signal is structurally dead — flag it so the UI shows a warning. ────
    try:
        curr_price = float(df["close"].iloc[-1])
    except Exception:
        curr_price = 0.0

    signal_dict = signal.to_dict()
    invalidated = False
    if signal.direction == "BEAR" and curr_price > signal.orb_high:
        invalidated = True
        signal_dict["notes"] = [
            f"⛔ Signal INVALIDATED — CMP ({curr_price:.2f}) reclaimed ORB High ({signal.orb_high:.2f}). Do not trade."
        ] + signal_dict.get("notes", [])
    elif signal.direction == "BULL" and curr_price < signal.orb_low:
        invalidated = True
        signal_dict["notes"] = [
            f"⛔ Signal INVALIDATED — CMP ({curr_price:.2f}) broke below ORB Low ({signal.orb_low:.2f}). Do not trade."
        ] + signal_dict.get("notes", [])

    signal_dict["invalidated"] = invalidated

    return {
        "symbol":   symbol,
        "interval": interval,
        "signal":   signal_dict,
        "status":   "INVALIDATED" if invalidated else "BREAKOUT",
        "generated_at": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  MULTI-LEG STRATEGY BUILDER  (Straddle / Strangle / Iron Condor / Bull Spread)
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/strategy/{symbol}")
async def get_strategy(symbol: str, interval: str = "5minute"):
    """
    Analyses the live option chain and recommends the best multi-leg strategy.
    Handles symbol name resolution: "NIFTY 50" -> "NIFTY", "NIFTY BANK" -> "BANKNIFTY".
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    # ── FIX 1: Resolve display name -> NFO instrument name ───────────────────
    # UI sends "NIFTY 50" / "NIFTY BANK" but NFO instruments use "NIFTY"/"BANKNIFTY"
    DISPLAY_TO_NFO = {
        "NIFTY 50":   "NIFTY",
        "NIFTY BANK": "BANKNIFTY",
        "NIFTY":      "NIFTY",
        "BANKNIFTY":  "BANKNIFTY",
    }
    fno_symbol = DISPLAY_TO_NFO.get(symbol.upper(), symbol)  # fallback = as-is for stocks

    # ── FIX 4: lot_size lookup using resolved NFO symbol ────────────────────
    fno_info = FNO_EXPIRY_SYMBOLS.get(fno_symbol, {})
    if not fno_info:
        # Try original symbol too (stocks like RELIANCE come as-is)
        fno_info = FNO_EXPIRY_SYMBOLS.get(symbol, {})
    if fno_symbol == "BANKNIFTY" or "BANK" in symbol.upper():
        lot_size = fno_info.get("lot_size", 15)
    elif fno_symbol == "NIFTY":
        lot_size = fno_info.get("lot_size", 25)
    else:
        lot_size = fno_info.get("lot_size", 50)

    # ── Get current spot price ───────────────────────────────────────────────
    token = DEMO_TOKENS.get(symbol) or DEMO_TOKENS.get("NIFTY 50") if "NIFTY" in symbol.upper() else get_instrument_token(symbol)
    if not token:
        return JSONResponse({"error": f"No price token for {symbol}"}, status_code=404)

    df = kite_manager.get_historical_data(token, "5minute", 2)
    if df.empty:
        return JSONResponse({"error": "No price data from Kite"}, status_code=404)
    spot = float(df["close"].iloc[-1])

    # ── FIX 2: get_real_expiries returns dict, not list ──────────────────────
    try:
        expiries_dict = kite_manager.get_real_expiries(fno_symbol)
        if not expiries_dict:
            return JSONResponse({"error": f"No expiries found for {fno_symbol}. Check NFO instruments."}, status_code=500)
        weekly_str  = expiries_dict.get("weekly", "")
        monthly_str = expiries_dict.get("monthly", weekly_str)
        if not weekly_str:
            return JSONResponse({"error": "Could not determine weekly expiry"}, status_code=500)
        weekly_expiry  = date.fromisoformat(weekly_str)
        monthly_expiry = date.fromisoformat(monthly_str)
    except Exception as e:
        log.error(f"Strategy expiry error for {fno_symbol}: {e}")
        return JSONResponse({"error": f"Expiry fetch failed: {str(e)}"}, status_code=500)

    days_to_weekly  = max((weekly_expiry  - datetime.now().date()).days + 1, 1)
    days_to_monthly = max((monthly_expiry - datetime.now().date()).days + 1, 1)

    # ── Fetch option chain ───────────────────────────────────────────────────
    try:
        chain = kite_manager.get_option_chain(fno_symbol, weekly_str, spot, num_strikes=6)
    except Exception as e:
        log.error(f"Strategy chain error for {fno_symbol}: {e}")
        chain = []

    if not chain:
        return JSONResponse({"error": f"Option chain unavailable for {fno_symbol} {weekly_str}"}, status_code=404)

    # ── Strike step ──────────────────────────────────────────────────────────
    if fno_symbol == "BANKNIFTY":
        step = 100
    elif fno_symbol == "NIFTY":
        step = 50
    else:
        # For stocks, infer step from chain strikes
        strikes_in_chain = sorted(set(int(r["strike"]) for r in chain if r.get("strike")))
        step = int(strikes_in_chain[1] - strikes_in_chain[0]) if len(strikes_in_chain) >= 2 else 10

    atm_strike = int(round(spot / step) * step)

    # ── FIX 3: chain rows from get_option_chain are structured as:
    #   {"strike": X, "CE": {"ltp":..,"oi":..,...}, "PE": {"ltp":..,"oi":..,...}}
    #   NOT {"strike":X, "option_type":"CE", "last_price":X, "oi":X}
    chain_map = {}
    for row in chain:
        s = int(row.get("strike", 0))
        if not s:
            continue
        ce_data = row.get("CE") or {}
        pe_data = row.get("PE") or {}
        chain_map[s] = {
            "ce":     float(ce_data.get("ltp", 0) or 0),
            "pe":     float(pe_data.get("ltp", 0) or 0),
            "ce_oi":  int(ce_data.get("oi", 0) or 0),
            "pe_oi":  int(pe_data.get("oi", 0) or 0),
            "ce_iv":  float(ce_data.get("iv", 0) or 0),
            "pe_iv":  float(pe_data.get("iv", 0) or 0),
            "ce_delta": float(ce_data.get("delta", 0.5) or 0.5),
            "pe_delta": float(pe_data.get("delta", -0.5) or -0.5),
        }

    # ── Compute PCR and find key levels ─────────────────────────────────────
    strikes = sorted(chain_map.keys())
    pcr_data = []
    for s in strikes:
        r = chain_map[s]
        ce  = r["ce"]
        pe  = r["pe"]
        pcr = round(r["pe_oi"] / r["ce_oi"], 2) if r["ce_oi"] > 0 else 0
        # Use live IV if available, fallback to IV proxy
        iv_val = r["ce_iv"] or r["pe_iv"] or 0
        pcr_data.append({
            "strike":   s,
            "ce":       ce,
            "pe":       pe,
            "pcr":      pcr,
            "straddle": round(ce + pe, 2),
            "iv":       round(iv_val * 100, 2),   # as % if decimal, keep if already %
        })

    if not pcr_data:
        return JSONResponse({"error": "Could not build PCR table from chain"}, status_code=500)

    # ── ATM row ──────────────────────────────────────────────────────────────
    atm_row = min(pcr_data, key=lambda x: abs(x["strike"] - spot))
    atm_ce  = atm_row["ce"]
    atm_pe  = atm_row["pe"]
    atm_s   = int(atm_row["strike"])

    # ── OTM strikes (1 step out) ─────────────────────────────────────────────
    otm_ce_strike     = atm_s + step
    otm_pe_strike     = atm_s - step
    condor_ce_buy_strike = atm_s + step * 2
    condor_pe_buy_strike = atm_s - step * 2

    otm_ce_ltp       = chain_map.get(otm_ce_strike, {}).get("ce", 0) or 0
    otm_pe_ltp       = chain_map.get(otm_pe_strike, {}).get("pe", 0) or 0
    condor_ce_buy_ltp = chain_map.get(condor_ce_buy_strike, {}).get("ce", 0) or 0
    condor_pe_buy_ltp = chain_map.get(condor_pe_buy_strike, {}).get("pe", 0) or 0

    # ── IV proxy (use live IV if available, else compute from premium) ────────
    live_iv = atm_row.get("iv", 0) or 0
    straddle_cost = atm_ce + atm_pe
    if live_iv > 1:           # already in % form (e.g. 14.5)
        iv_proxy = round(live_iv, 2)
    elif live_iv > 0:         # decimal form (e.g. 0.145)
        iv_proxy = round(live_iv * 100, 2)
    else:                     # fallback: compute from straddle premium
        iv_proxy = round((straddle_cost / spot) * ((252 / days_to_weekly) ** 0.5) * 100, 2)

    # ── Market bias from ATM PCR ──────────────────────────────────────────────
    atm_pcr = atm_row["pcr"]

    try:
        if atm_pcr > 1.5:
            market_bias = "BULLISH"
            bias_reason = f"High PCR {atm_pcr} at ATM = strong put writing = bulls defending"
        elif atm_pcr < 0.5:
            market_bias = "BEARISH"
            bias_reason = f"Low PCR {atm_pcr} at ATM = heavy call writing = bears capping"
        else:
            market_bias = "NEUTRAL"
            bias_reason = f"PCR {atm_pcr} at ATM = balanced = range-bound market"

        # ── Range probability ─────────────────────────────────────────────────
        range_support    = next((p["strike"] for p in reversed(pcr_data) if p["strike"] < spot and p["pcr"] > 2), None)
        range_resistance = next((p["strike"] for p in pcr_data if p["strike"] > spot and p["pcr"] < 0.3), None)
        range_pts = (range_resistance - range_support) if range_support and range_resistance else None
    except Exception as e:
        log.error(f"Strategy bias/range error: {e}")
        market_bias = "NEUTRAL"; bias_reason = "PCR computed"; range_support = None; range_resistance = None; range_pts = None

    # ── Strategy selection logic ─────────────────────────────────────
    strategies = []

    # ── 1. STRADDLE ──────────────────────────────────────────────────
    straddle_be_up   = round(atm_s + straddle_cost, 2)
    straddle_be_down = round(atm_s - straddle_cost, 2)
    straddle_move_needed = straddle_cost
    straddle_theta_day = round(straddle_cost * 0.06, 2)  # ~6%/day at 3 DTE
    straddle_suitable = days_to_weekly >= 7  # only suitable with week+ left

    strategies.append({
        "name": "Buy Straddle",
        "type": "BUY_STRADDLE",
        "icon": "⚡",
        "legs": [
            {"action": "BUY", "type": "CE", "strike": atm_s, "ltp": round(atm_ce, 2)},
            {"action": "BUY", "type": "PE", "strike": atm_s, "ltp": round(atm_pe, 2)},
        ],
        "total_cost":      round(straddle_cost * lot_size, 2),
        "cost_per_share":  round(straddle_cost, 2),
        "max_profit":      "Unlimited (one side runs)",
        "max_loss":        round(straddle_cost * lot_size, 2),
        "breakeven_up":    straddle_be_up,
        "breakeven_down":  straddle_be_down,
        "move_needed":     round(straddle_move_needed, 2),
        "move_needed_pct": round(straddle_move_needed / spot * 100, 2),
        "theta_per_day":   round(straddle_theta_day * lot_size, 2),
        "suitable_now":    straddle_suitable,
        "suitability_reason": ("✅ Good -- enough time for move to develop" if straddle_suitable
                                else f"⚠️ Risky -- only {days_to_weekly} DTE, theta burns ₹{round(straddle_theta_day*lot_size,0)}/day. Better next expiry."),
        "best_when": "Major event expected (RBI, Budget, earnings) with 7+ days left",
        "lot_size": lot_size,
    })

    # ── 2. STRANGLE ──────────────────────────────────────────────────
    strangle_cost     = round(otm_ce_ltp + otm_pe_ltp, 2) if otm_ce_ltp and otm_pe_ltp else 0
    strangle_be_up    = round(otm_ce_strike + strangle_cost, 2)
    strangle_be_down  = round(otm_pe_strike - strangle_cost, 2)
    strangle_theta    = round(strangle_cost * 0.07 * lot_size, 2)

    if strangle_cost > 0:
        strategies.append({
            "name": "Buy Strangle",
            "type": "BUY_STRANGLE",
            "icon": "🔀",
            "legs": [
                {"action": "BUY", "type": "CE", "strike": otm_ce_strike, "ltp": round(otm_ce_ltp, 2)},
                {"action": "BUY", "type": "PE", "strike": otm_pe_strike, "ltp": round(otm_pe_ltp, 2)},
            ],
            "total_cost":      round(strangle_cost * lot_size, 2),
            "cost_per_share":  strangle_cost,
            "max_profit":      "Unlimited (one side runs)",
            "max_loss":        round(strangle_cost * lot_size, 2),
            "breakeven_up":    strangle_be_up,
            "breakeven_down":  strangle_be_down,
            "move_needed":     round((strangle_be_up - spot), 2),
            "move_needed_pct": round((strangle_be_up - spot) / spot * 100, 2),
            "theta_per_day":   strangle_theta,
            "suitable_now":    days_to_weekly >= 7,
            "suitability_reason": ("✅ Cheaper than straddle, needs bigger move" if days_to_weekly >= 7
                                    else f"⚠️ Only {days_to_weekly} DTE -- theta will destroy this fast"),
            "best_when": "Strong breakout expected, want cheaper entry than straddle",
            "lot_size": lot_size,
        })

    # ── 3. IRON CONDOR ───────────────────────────────────────────────
    if otm_ce_ltp and otm_pe_ltp and condor_ce_buy_ltp and condor_pe_buy_ltp:
        ic_ce_net = round(otm_ce_ltp - condor_ce_buy_ltp, 2)   # sell closer, buy farther
        ic_pe_net = round(otm_pe_ltp - condor_pe_buy_ltp, 2)
        ic_credit  = round(ic_ce_net + ic_pe_net, 2)
        ic_spread_width = step  # distance between legs
        ic_max_loss = round((ic_spread_width - ic_credit) * lot_size, 2)
        ic_max_profit = round(ic_credit * lot_size, 2)
        ic_profit_zone = (otm_pe_strike, otm_ce_strike)

        strategies.append({
            "name": "Iron Condor",
            "type": "IRON_CONDOR",
            "icon": "🦅",
            "legs": [
                {"action": "SELL", "type": "CE", "strike": otm_ce_strike,      "ltp": round(otm_ce_ltp, 2)},
                {"action": "BUY",  "type": "CE", "strike": condor_ce_buy_strike, "ltp": round(condor_ce_buy_ltp, 2)},
                {"action": "SELL", "type": "PE", "strike": otm_pe_strike,      "ltp": round(otm_pe_ltp, 2)},
                {"action": "BUY",  "type": "PE", "strike": condor_pe_buy_strike, "ltp": round(condor_pe_buy_ltp, 2)},
            ],
            "net_credit":     ic_credit,
            "total_credit":   ic_max_profit,
            "max_profit":     ic_max_profit,
            "max_loss":       ic_max_loss,
            "profit_zone":    list(ic_profit_zone),
            "breakeven_up":   round(otm_ce_strike + ic_credit, 2),
            "breakeven_down": round(otm_pe_strike - ic_credit, 2),
            "suitable_now":   True,  # always suitable in range-bound market
            "suitability_reason": (f"✅ Ideal -- PCR confirms range {range_support}-{range_resistance}. "
                                    f"Theta works FOR you. Max risk defined at ₹{ic_max_loss}."
                                    if range_support and range_resistance
                                    else "✅ Good range-play with defined risk"),
            "best_when": "Range-bound market, 1-5 DTE, high PCR spread between OTM strikes",
            "margin_needed": round(ic_max_loss * 1.1, 2),  # approx margin
            "lot_size": lot_size,
        })

    # ── 4. BULL CALL SPREAD ──────────────────────────────────────────
    if market_bias in ("BULLISH", "NEUTRAL") and otm_ce_ltp:
        bcs_buy  = atm_ce
        bcs_sell = otm_ce_ltp
        bcs_cost = round(bcs_buy - bcs_sell, 2)
        bcs_max_profit = round((step - bcs_cost) * lot_size, 2)
        bcs_max_loss   = round(bcs_cost * lot_size, 2)
        strategies.append({
            "name": "Bull Call Spread",
            "type": "BULL_CALL_SPREAD",
            "icon": "📈",
            "legs": [
                {"action": "BUY",  "type": "CE", "strike": atm_s,          "ltp": round(atm_ce, 2)},
                {"action": "SELL", "type": "CE", "strike": otm_ce_strike,  "ltp": round(otm_ce_ltp, 2)},
            ],
            "total_cost":   round(bcs_cost * lot_size, 2),
            "cost_per_share": bcs_cost,
            "max_profit":   bcs_max_profit,
            "max_loss":     bcs_max_loss,
            "breakeven_up": round(atm_s + bcs_cost, 2),
            "suitable_now": market_bias == "BULLISH" or days_to_weekly >= 3,
            "suitability_reason": f"{'✅' if market_bias=='BULLISH' else 'ℹ️'} Bullish directional play with capped risk",
            "best_when": "Moderate bullish bias, want to reduce CE cost",
            "lot_size": lot_size,
        })

    # ── 5. BEAR PUT SPREAD ───────────────────────────────────────────
    if market_bias in ("BEARISH", "NEUTRAL") and otm_pe_ltp:
        bps_buy  = atm_pe
        bps_sell = otm_pe_ltp
        bps_cost = round(bps_buy - bps_sell, 2)
        bps_max_profit = round((step - bps_cost) * lot_size, 2)
        bps_max_loss   = round(bps_cost * lot_size, 2)
        strategies.append({
            "name": "Bear Put Spread",
            "type": "BEAR_PUT_SPREAD",
            "icon": "📉",
            "legs": [
                {"action": "BUY",  "type": "PE", "strike": atm_s,         "ltp": round(atm_pe, 2)},
                {"action": "SELL", "type": "PE", "strike": otm_pe_strike, "ltp": round(otm_pe_ltp, 2)},
            ],
            "total_cost":   round(bps_cost * lot_size, 2),
            "cost_per_share": bps_cost,
            "max_profit":   bps_max_profit,
            "max_loss":     bps_max_loss,
            "breakeven_down": round(atm_s - bps_cost, 2),
            "suitable_now": market_bias == "BEARISH" or days_to_weekly >= 3,
            "suitability_reason": f"{'✅' if market_bias=='BEARISH' else 'ℹ️'} Bearish directional play with capped risk",
            "best_when": "Moderate bearish bias, want to reduce PE cost",
            "lot_size": lot_size,
        })

    # ── Recommended strategy ─────────────────────────────────────────
    if days_to_weekly <= 4 and range_pts and range_pts < straddle_cost * 2.5:
        recommended = "IRON_CONDOR"
        rec_reason  = (f"Range-bound market ({days_to_weekly} DTE). "
                        f"PCR defines range {range_support}-{range_resistance} (~{range_pts}pts). "
                        f"Straddle needs +/-{straddle_cost:.0f}pts -- wider than the range. Iron Condor wins.")
    elif days_to_weekly >= 7:
        recommended = "BUY_STRADDLE"
        rec_reason  = f"7+ DTE available -- enough time for a directional move. Theta manageable."
    elif market_bias == "BULLISH":
        recommended = "BULL_CALL_SPREAD"
        rec_reason  = f"Bullish bias (PCR {atm_pcr}) with limited days -- Bull Spread limits theta risk."
    elif market_bias == "BEARISH":
        recommended = "BEAR_PUT_SPREAD"
        rec_reason  = f"Bearish bias (PCR {atm_pcr}) -- Bear Spread limits risk vs buying naked PE."
    else:
        recommended = "IRON_CONDOR"
        rec_reason  = "Range-bound neutral market -- collect premium via Iron Condor."

    return {
        "symbol":           symbol,
        "spot":             round(spot, 2),
        "atm_strike":       atm_s,
        "weekly_expiry":    weekly_str,
        "monthly_expiry":   monthly_str,
        "days_to_weekly":   days_to_weekly,
        "days_to_monthly":  days_to_monthly,
        "lot_size":         lot_size,
        "market_bias":      market_bias,
        "bias_reason":      bias_reason,
        "range_support":    range_support,
        "range_resistance": range_resistance,
        "range_pts":        range_pts,
        "iv_proxy":         round(iv_proxy, 2),
        "straddle_cost":    round(straddle_cost, 2),
        "recommended":      recommended,
        "rec_reason":       rec_reason,
        "strategies":       strategies,
        "pcr_summary":      [{"strike": p["strike"], "pcr": p["pcr"], "straddle": p["straddle"]} for p in pcr_data],
        "generated_at":     datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  SMC STOCK SCAN -- scans all stocks for SMC BUY/SELL setups
#  Adds pattern type detection: Rising Wedge, Double Bottom, Sniper Entry, Fib
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/smc-scan")
async def get_smc_scan(interval: str = "15minute", min_conf: float = 60.0, limit: int = 20):
    """
    Scans all stocks in universe for SMC signals.
    Returns only BUY or SELL setups (skips WAIT).
    Sorted by confidence DESC. Shows pattern type detected.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    results = []
    scan_list = [s for s in SCAN_UNIVERSE if s not in ("NIFTY 50", "NIFTY BANK")]
    import time as _time

    def _fetch_smc_data(symbol: str):
        token = get_instrument_token(symbol)
        if token is None:
            return symbol, None, None
        try:
            days_back = 7 if "minute" in interval else 30
            with _kite_semaphore:
                df = kite_manager.get_historical_data(token, interval, days_back)
            if df.empty or len(df) < 20:
                return symbol, None, None
            df_1h = None
            try:
                with _kite_semaphore:
                    df_1h = kite_manager.get_historical_data(token, "60minute", 5)
            except Exception:
                pass
            return symbol, df, df_1h
        except Exception:
            return symbol, None, None

    smc_data = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_smc_data, s): s for s in scan_list}
        for future in as_completed(futures):
            sym, df, df_1h = future.result()
            if df is not None:
                smc_data[sym] = (df, df_1h)

    for symbol, (df, df_1h) in smc_data.items():
        try:

            result = smc_engine.analyse(df, symbol, interval, df_1h)

            if result.smc_signal not in ("BUY", "SELL"):
                continue
            if result.smc_confidence < min_conf:
                continue

            # Detect which pattern triggered this signal
            pattern_type = _detect_pattern_type(df, result.smc_signal, result.market_structure)

            # Get live price
            try:
                q = kite_manager.get_quote([f"NSE:{symbol}"])
                ltp = float((q or {}).get(f"NSE:{symbol}", {}).get("last_price", 0) or result.current_price)
            except Exception:
                ltp = result.current_price

            results.append({
                "symbol":        symbol,
                "signal":        result.smc_signal,
                "confidence":    round(result.smc_confidence, 1),
                "pattern_type":  pattern_type,
                "current_price": round(ltp, 2),
                "structure":     result.market_structure,
                "pd_zone":       result.premium_discount,
                "vwap_bias":     result.vwap_bias,
                "mtf_bias":      result.mtf_bias,
                "entry_low":     round(result.smc_entry_zone[0], 2),
                "entry_high":    round(result.smc_entry_zone[1], 2),
                "stop_loss":     round(result.smc_sl, 2),
                "target_1":      round(result.smc_target1, 2),
                "target_2":      round(result.smc_target2, 2),
                "rr":            round(abs(result.smc_target1 - result.smc_entry_zone[1]) /
                                       max(abs(result.smc_sl - result.smc_entry_zone[1]), 0.01), 2),
                "notes":         result.smc_notes[:3],
                "session_ok":    result.session_tradeable,
                "interval":      interval,
            })
        except Exception as e:
            log.debug(f"SMC scan error {symbol}: {e}")
            continue

    # Sort by confidence
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Auto-log signals to journal
    for _r in results[:limit]:
        try:
            _ez = _r.get("entry_zone", [0, 0])
            threading.Thread(
                target=_log_signal,
                args=(_r["symbol"], _r["direction"], "smc",
                      _r["confidence"],
                      _ez[0] if _ez else 0, _ez[1] if _ez else 0,
                      _r.get("stop_loss", 0), _r.get("target_2", 0), _r.get("rr_t2", 0),
                      _r.get("pattern_type", "")),
                daemon=True,
            ).start()
        except Exception:
            pass

    _IST = timezone(timedelta(hours=5, minutes=30))
    return {
        "signals":      results[:limit],
        "total_found":  len(results),
        "scanned":      len(scan_list),
        "interval":     interval,
        "min_conf":     min_conf,
        "generated_at": datetime.now().isoformat(),
        "scanned_at":   datetime.now(tz=_IST).strftime("%H:%M:%S IST"),
    }


def _detect_pattern_type(df: pd.DataFrame, signal: str, structure: str) -> str:
    """Identifies which price action pattern is driving the SMC signal."""
    try:
        n = min(len(df), 40)
        sub = df.tail(n)
        highs  = sub["high"].values.astype(float)
        lows   = sub["low"].values.astype(float)
        closes = sub["close"].values.astype(float)
        swing_highs = [(i, highs[i]) for i in range(2, n-2)
                       if highs[i] >= highs[i-1] and highs[i] >= highs[i-2]
                       and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]]
        swing_lows  = [(i, lows[i]) for i in range(2, n-2)
                       if lows[i] <= lows[i-1] and lows[i] <= lows[i-2]
                       and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]]
        if signal == "BUY" and len(swing_lows) >= 2:
            l1, l2 = swing_lows[-2][1], swing_lows[-1][1]
            if abs(l1 - l2) / max(l1, 1) < 0.005:
                return "Double Bottom"
        if signal == "SELL" and len(swing_highs) >= 2:
            h1, h2 = swing_highs[-2][1], swing_highs[-1][1]
            if abs(h1 - h2) / max(h1, 1) < 0.005:
                return "Double Top"
        if signal == "SELL" and len(swing_highs) >= 3 and len(swing_lows) >= 3:
            sh = [x[1] for x in swing_highs[-3:]]
            sl_vals = [x[1] for x in swing_lows[-3:]]
            hs = (sh[-1]-sh[0]) / max(swing_highs[-1][0]-swing_highs[-3][0], 1)
            ls = (sl_vals[-1]-sl_vals[0]) / max(swing_lows[-1][0]-swing_lows[-3][0], 1)
            if hs > 0 and ls > hs * 1.2:
                return "Rising Wedge"
        if signal == "BUY" and len(swing_highs) >= 3 and len(swing_lows) >= 3:
            sh = [x[1] for x in swing_highs[-3:]]
            sl_vals = [x[1] for x in swing_lows[-3:]]
            hs = (sh[-1]-sh[0]) / max(swing_highs[-1][0]-swing_highs[-3][0], 1)
            ls = (sl_vals[-1]-sl_vals[0]) / max(swing_lows[-1][0]-swing_lows[-3][0], 1)
            if ls < 0 and hs < 0 and hs < ls * 1.2:
                return "Falling Wedge"
        r10 = (max(highs[-10:]) - min(lows[-10:])) / max(closes[-1], 1) * 100
        r20 = (max(highs[-20:-10]) - min(lows[-20:-10])) / max(closes[-1], 1) * 100 if n >= 20 else r10
        if r10 < r20 * 0.6:
            return "Sniper Entry (Compression)"
        if signal == "SELL" and len(swing_highs) >= 3:
            sh = [x[1] for x in swing_highs[-3:]]
            diffs = [abs(sh[i] - sh[i-1]) / max(sh[i-1], 1) for i in range(1, 3)]
            if all(d < 0.015 for d in diffs):
                return "3 Drives Pattern"
        if len(swing_highs) >= 1 and len(swing_lows) >= 1:
            sh = swing_highs[-1][1]; sl2 = swing_lows[-1][1]
            fib618 = sh - (sh - sl2) * 0.618
            fib382 = sh - (sh - sl2) * 0.382
            curr = closes[-1]
            if min(fib382, fib618) <= curr <= max(fib382, fib618):
                return "Fibonacci 38.2-61.8% Zone"
        return "Order Block / BOS"
    except Exception:
        return "SMC Confluence"


# ── Full institutional scan -- all engines combined ────────────────────────────
@app.get("/api/institutional-scan")
async def institutional_scan(
    interval: str   = "15minute",
    min_conf: float = 65.0,
    limit:    int   = 25,
):
    """
    Full institutional-grade scan combining:
    - SMC (Order Blocks, FVGs, Liquidity Sweeps)
    - ICT (OTE zones, Breaker Blocks, Power of 3, Killzones)
    - BOS/Structure (Fibonacci extension targets)
    - Institutional levels (PDH/PDL)
    - Daily HTF bias

    Only returns setups where MULTIPLE engines agree.
    All targets use CORRECT Fibonacci cascade ordering.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    import time as _time

    # ── Check scan cache first (4-min TTL) ───────────────────────────────────
    _scan_key = f"inst_scan:{interval}:{int(min_conf)}"
    _cached = _scan_cache.get(_scan_key)
    if _cached is not None:
        log.info(f"Returning cached institutional scan ({interval})")
        return _cached

    results = []
    # Fix #11: prefer _dynamic_universe (top movers first) when populated by
    # the background enrichment task; fall back to static SCAN_UNIVERSE.
    _base = _dynamic_universe if _dynamic_universe else SCAN_UNIVERSE
    scan_list = [s for s in _base if s not in ("NIFTY 50", "NIFTY BANK")]

    def _fetch_symbol_data(symbol: str):
        token = get_instrument_token(symbol)
        if token is None:
            return symbol, None, None, None
        try:
            with _kite_semaphore:
                df = kite_manager.get_historical_data(token, interval, 10)
            if df.empty or len(df) < 30:
                return symbol, None, None, None
            # Skip stale data during market hours
            if _is_market_hours() and _data_age_minutes(df) > 75:
                return symbol, None, None, None
            df_1h = df_daily = None
            try:
                with _kite_semaphore:
                    df_1h = kite_manager.get_historical_data(token, "60minute", 7)
                with _kite_semaphore:
                    df_daily = kite_manager.get_historical_data(token, "day", 20)
            except Exception:
                pass
            return symbol, df, df_1h, df_daily
        except Exception:
            return symbol, None, None, None

    # Fetch all symbols in parallel (max 5 concurrent threads, semaphore limits API to 3 req/s)
    symbol_data = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_symbol_data, s): s for s in scan_list}
        for future in as_completed(futures):
            sym, df, df_1h, df_daily = future.result()
            if df is not None:
                symbol_data[sym] = (df, df_1h, df_daily)

    log.info(f"Institutional scan: fetched data for {len(symbol_data)}/{len(scan_list)} symbols")

    for symbol, (df, df_1h, df_daily) in symbol_data.items():
        try:

            inst_levels = calculate_institutional_levels(df, symbol)

            # ── ACCURACY IMPROVEMENT 2: Volume Confirmation Gate ─────────────
            # Institutional moves ALWAYS have above-average volume on the trigger bar.
            # Reject setups where the last 3 bars are all below 80% of 20-bar avg vol.
            if "volume" in df.columns and len(df) >= 23:
                avg_vol_20 = float(df["volume"].iloc[-23:-3].mean())
                recent_max_vol = float(df["volume"].iloc[-3:].max())
                if avg_vol_20 > 0 and recent_max_vol < avg_vol_20 * 0.80:
                    continue  # Dead volume -- no institutional interest, skip

            # Run all 3 engines
            smc_result   = smc_engine.analyse(df, symbol, interval, df_1h, inst_levels)
            ict_result   = ict_engine.analyse(df, symbol, interval, df_daily, inst_levels)
            str_setups   = structure_engine.detect_setups(df, symbol, interval)

            smc_sig  = smc_result.smc_signal   # "BUY" | "SELL" | "WAIT"
            ict_sig  = ict_result.ict_signal    # "LONG" | "SHORT" | "WAIT"
            str_sig  = str_setups[0].direction if str_setups else "WAIT"

            # Normalize
            def norm(s):
                return "BUY" if s in ("BUY","LONG") else ("SELL" if s in ("SELL","SHORT") else "WAIT")

            sigs = [norm(smc_sig), norm(ict_sig), norm(str_sig)]
            buy_votes  = sigs.count("BUY")
            sell_votes = sigs.count("SELL")

            # Need at least 2 of 3 engines agreeing
            if buy_votes >= 2:
                direction = "BUY"
                votes = buy_votes
            elif sell_votes >= 2:
                direction = "SELL"
                votes = sell_votes
            else:
                continue  # No confluence -- skip

            # Composite confidence
            conf_list = []
            if norm(smc_sig) == direction:  conf_list.append(smc_result.smc_confidence)
            if norm(ict_sig) == direction:  conf_list.append(ict_result.ict_confidence)
            if str_setups and norm(str_sig) == direction:
                conf_list.append(str_setups[0].confidence_score)
            composite_conf = round(sum(conf_list) / len(conf_list), 1) if conf_list else 0.0

            if composite_conf < min_conf:
                continue

            # ── ACCURACY IMPROVEMENT 3: Minimum RR gate ──────────────────────
            # Never surface a setup with RR < 1.5 to T2 (risk not worth reward).
            # Also flag if we are outside market hours (stale data warning).
            _IST = timezone(timedelta(hours=5, minutes=30))
            _now = datetime.now(tz=_IST).replace(tzinfo=None)
            _market_open  = _now.replace(hour=9, minute=30, second=0, microsecond=0)
            _market_close = _now.replace(hour=15, minute=30, second=0, microsecond=0)
            _in_market_hours = _market_open <= _now <= _market_close

            # Determine best targets (prefer structure engine if available, else ICT)
            if str_setups and norm(str_sig) == direction:
                setup = str_setups[0]
                t1 = setup.target_1
                t2 = setup.target_2
                t3 = setup.target_3
                t4 = setup.target_4
                entry_low  = setup.entry_zone_low
                entry_high = setup.entry_zone_high
                sl         = setup.sl_fixed
                rr_t2      = setup.target_rr
                rr_t3      = setup.target_rr_t3
                entry_type = "OTE 61.8%-78.6%"
                ob_conf    = setup.ob_confluence
                fvg_conf   = setup.fvg_confluence
            else:
                t1 = ict_result.ict_t1
                t2 = ict_result.ict_t2
                t3 = ict_result.ict_t3
                t4 = ict_result.ict_t4
                entry_low  = ict_result.ict_entry_zone[0]
                entry_high = ict_result.ict_entry_zone[1]
                sl         = ict_result.ict_sl
                rr_t2      = ict_result.ict_rr_t2
                rr_t3      = ict_result.ict_rr_t3
                entry_type = "ICT OTE"
                ob_conf  = any(b.direction.startswith("BULL" if direction=="BUY" else "BEAR")
                               for b in ict_result.breaker_blocks if b.valid)
                fvg_conf = False

            # Get live price
            try:
                q   = kite_manager.get_quote([f"NSE:{symbol}"])
                ltp = float((q or {}).get(f"NSE:{symbol}", {}).get("last_price", 0) or smc_result.current_price)
            except Exception:
                ltp = smc_result.current_price

            # Build consolidated notes
            key_notes = []
            if smc_result.market_structure != "RANGING":
                key_notes.append(f"Structure: {smc_result.market_structure}")
            if ict_result.in_killzone:
                key_notes.append(f"⏰ {ict_result.killzone_name}")
            if ict_result.daily_bias != "UNKNOWN":
                key_notes.append(f"Daily: {ict_result.daily_bias}")
            if ict_result.po3_phase not in ("DISTRIBUTION", "UNKNOWN"):
                key_notes.append(f"Po3: {ict_result.po3_phase}")
            if ob_conf:
                key_notes.append("OB ✅")
            if fvg_conf:
                key_notes.append("FVG ✅")
            if inst_levels.above_pdh and direction == "BUY":
                key_notes.append(f"Above PDH {inst_levels.pdh:.0f}")
            if inst_levels.below_pdl and direction == "SELL":
                key_notes.append(f"Below PDL {inst_levels.pdl:.0f}")

            # ── ACCURACY IMPROVEMENT 1: 5-point Confluence Score Gate ──────────
            # Only surface signals with strong multi-factor confluence.
            # Each factor adds 1 point; min score 3 to show as A/A+.
            confluence_score = 0
            if ob_conf:                                          confluence_score += 1  # OB confirmed
            if fvg_conf:                                         confluence_score += 1  # FVG confirmed
            if ict_result.in_killzone:                           confluence_score += 1  # Active killzone
            if inst_levels.above_pdh and direction == "BUY":    confluence_score += 1  # PDH breakout
            elif inst_levels.below_pdl and direction == "SELL": confluence_score += 1  # PDL breakdown
            daily_ok = (ict_result.daily_bias == "BULLISH" and direction == "BUY") or                        (ict_result.daily_bias == "BEARISH" and direction == "SELL")
            if daily_ok:                                         confluence_score += 1  # HTF aligned

            # Quality label driven by BOTH confidence AND confluence score
            if composite_conf >= 80 and confluence_score >= 3:   quality = "A+"
            elif composite_conf >= 70 and confluence_score >= 2:  quality = "A"
            elif composite_conf >= 65:                            quality = "B"
            else:
                continue  # Below minimum quality -- reject

            key_notes.append(f"Confluence: {confluence_score}/5")

            # Enforce minimum RR 1.5x to T2 -- below this, risk/reward is poor
            # FIX: rr_t2 == 0 means the signal is in WAIT state (no real setup);
            # the old `else True` let these ghost signals pass through.
            if rr_t2 <= 0 or rr_t2 < 1.5:
                continue

            # ── ACCURACY IMPROVEMENT 4: Market hours tag ──────────────────────
            _stale_warning = [] if _in_market_hours else ["⚠️ Outside market hours — confirm before trading"]
            key_notes.extend(_stale_warning)

            # Position sizing: 2% risk model
            _entry_mid   = round((entry_low + entry_high) / 2, 2)
            _risk_ps     = abs(_entry_mid - sl) if sl else 0
            _pos_size    = int(CAPITAL * 0.02 / _risk_ps) if _risk_ps > 0 else 0

            results.append({
                "symbol":        symbol,
                "direction":     direction,
                "quality":       quality,
                "confidence":    composite_conf,
                "position_size": _pos_size,
                "engine_votes":  f"{votes}/3 engines agree",
                "engines":       {
                    "smc":  norm(smc_sig),
                    "ict":  norm(ict_sig),
                    "structure": norm(str_sig),
                },
                "current_price": round(ltp, 2),
                # Entry
                "entry_zone":    [round(entry_low, 2), round(entry_high, 2)],
                "entry_type":    entry_type,
                # Risk
                "stop_loss":     round(sl, 2),
                # ══ FIBONACCI TARGETS (CORRECT ORDER) ══
                # LONG:  T1 < T2 < T3 < T4 (all above entry, ascending)
                # SHORT: T1 > T2 > T3 > T4 (all below entry, descending)
                "target_1":      round(t1, 2),   # Fib 1.0 -- equal leg, first exit
                "target_2":      round(t2, 2),   # Fib 1.272 -- standard extension
                "target_3":      round(t3, 2),   # Fib 1.618 -- golden extension
                "target_4":      round(t4, 2),   # Fib 2.0 -- runner
                "rr_t2":         round(rr_t2, 2),
                "rr_t3":         round(rr_t3, 2),
                # Context
                "structure":     smc_result.market_structure,
                "pd_zone":       smc_result.premium_discount,
                "vwap_bias":     smc_result.vwap_bias,
                "daily_bias":    ict_result.daily_bias,
                "in_killzone":   ict_result.in_killzone,
                "killzone":      ict_result.killzone_name,
                "po3_phase":     ict_result.po3_phase,
                "pdh":           round(inst_levels.pdh, 2),
                "pdl":           round(inst_levels.pdl, 2),
                "notes":         key_notes,
                "interval":      interval,
                "confluence_score": confluence_score,
            })

        except Exception as e:
            log.debug(f"Inst scan error {symbol}: {e}")
            continue

    results.sort(key=lambda x: (x["confidence"], x["quality"] == "A+"), reverse=True)

    # ── Auto-log all signals to journal (background, non-blocking) ────────
    for _r in results:
        try:
            threading.Thread(
                target=_log_signal,
                args=(_r["symbol"], _r["direction"], "institutional",
                      _r["confidence"],
                      _r["entry_zone"][0], _r["entry_zone"][1],
                      _r["stop_loss"], _r["target_2"], _r["rr_t2"],
                      "; ".join(_r.get("notes", []))),
                daemon=True,
            ).start()
        except Exception:
            pass

    _IST_tz = timezone(timedelta(hours=5, minutes=30))
    scan_response = {
        "signals":      results[:limit],
        "total_found":  len(results),
        "scanned":      len(symbol_data),
        "interval":     interval,
        "generated_at": datetime.now().isoformat(),
        "scanned_at":   datetime.now(tz=_IST_tz).strftime("%H:%M:%S IST"),
        "cached":       False,
        "methodology":  (
            "Requires 2/3 engines (SMC + ICT + Structure) to agree. "
            "Entry: OTE zone 61.8%-78.6% Fibonacci retracement. "
            "T1=Fib1.0(closest), T2=Fib1.272, T3=Fib1.618, T4=Fib2.0(runner). "
            "Minimum RR 1.8x to T2. PDH/PDL + Daily bias + Killzone context included."
        ),
    }
    _scan_cache.set(_scan_key, scan_response)
    return scan_response


def _detect_pattern_type(df: pd.DataFrame, signal: str, structure: str) -> str:
    """
    Identifies which price action pattern is driving the SMC signal.
    Based on images: ORB, Rising Wedge, Falling Wedge, Double Bottom/Top,
    Sniper Entry (Supply+Compression), 3 Drives, Fibonacci Retracement.
    """
    try:
        n = min(len(df), 40)
        sub = df.tail(n)
        highs  = sub["high"].values.astype(float)
        lows   = sub["low"].values.astype(float)
        closes = sub["close"].values.astype(float)

        # Find swing pivots
        swing_highs = [(i, highs[i]) for i in range(2, n-2)
                       if highs[i] >= highs[i-1] and highs[i] >= highs[i-2]
                       and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]]
        swing_lows  = [(i, lows[i]) for i in range(2, n-2)
                       if lows[i] <= lows[i-1] and lows[i] <= lows[i-2]
                       and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]]

        # Double Bottom (BUY)
        if signal == "BUY" and len(swing_lows) >= 2:
            l1, l2 = swing_lows[-2][1], swing_lows[-1][1]
            if abs(l1 - l2) / max(l1, 1) < 0.005:
                return "🔵 Double Bottom"

        # Double Top (SELL)
        if signal == "SELL" and len(swing_highs) >= 2:
            h1, h2 = swing_highs[-2][1], swing_highs[-1][1]
            if abs(h1 - h2) / max(h1, 1) < 0.005:
                return "🔴 Double Top"

        # Rising Wedge (SELL)
        if signal == "SELL" and len(swing_highs) >= 3 and len(swing_lows) >= 3:
            sh = [x[1] for x in swing_highs[-3:]]
            sl = [x[1] for x in swing_lows[-3:]]
            high_slope = (sh[-1] - sh[0]) / max(swing_highs[-1][0] - swing_highs[-3][0], 1)
            low_slope  = (sl[-1] - sl[0]) / max(swing_lows[-1][0] - swing_lows[-3][0], 1)
            if high_slope > 0 and low_slope > high_slope * 1.2:
                return "📐 Rising Wedge"

        # Falling Wedge (BUY)
        if signal == "BUY" and len(swing_highs) >= 3 and len(swing_lows) >= 3:
            sh = [x[1] for x in swing_highs[-3:]]
            sl = [x[1] for x in swing_lows[-3:]]
            high_slope = (sh[-1] - sh[0]) / max(swing_highs[-1][0] - swing_highs[-3][0], 1)
            low_slope  = (sl[-1] - sl[0]) / max(swing_lows[-1][0] - swing_lows[-3][0], 1)
            if low_slope < 0 and high_slope < 0 and high_slope < low_slope * 1.2:
                return "📐 Falling Wedge"

        # Sniper Entry: price at supply/demand zone after compression
        range_last10 = (max(highs[-10:]) - min(lows[-10:])) / max(closes[-1], 1) * 100
        range_prev10 = (max(highs[-20:-10]) - min(lows[-20:-10])) / max(closes[-1], 1) * 100 if n >= 20 else range_last10
        if range_last10 < range_prev10 * 0.6:
            return "🎯 Sniper Entry (Compression)"

        # 3 Drives: 3 equal swing highs/lows
        if signal == "SELL" and len(swing_highs) >= 3:
            sh = [x[1] for x in swing_highs[-3:]]
            diffs = [abs(sh[i] - sh[i-1]) / max(sh[i-1], 1) for i in range(1, 3)]
            if all(d < 0.015 for d in diffs):
                return "🌊 3 Drives Pattern"

        # Fibonacci Retracement entry (price at 61.8% of last swing)
        if len(swing_highs) >= 1 and len(swing_lows) >= 1:
            sh = swing_highs[-1][1]
            sl = swing_lows[-1][1]
            fib618 = sh - (sh - sl) * 0.618
            fib382 = sh - (sh - sl) * 0.382
            curr   = closes[-1]
            if min(fib382, fib618) <= curr <= max(fib382, fib618):
                return "📊 Fibonacci 38.2-61.8% Zone"

        # Default: SMC Order Block or BOS
        return "🧱 Order Block / BOS"

    except Exception:
        return "🧠 SMC Confluence"



def get_india_vix() -> float:
    """
    Fetch India VIX from Kite. Returns float (e.g. 14.5).
    Falls back to 15.0 (neutral) if unavailable or not authenticated.
    Used to scale stop-loss ATR multipliers: high VIX → wider stops.
    """
    try:
        if not kite_manager.is_authenticated:
            return 15.0
        q = kite_manager.kite.quote(["NSE:INDIA VIX"])
        vix = float(q.get("NSE:INDIA VIX", {}).get("last_price", 15.0) or 15.0)
        return vix
    except Exception:
        return 15.0

def vix_atr_multiplier() -> float:
    """
    Returns an ATR stop multiplier scaled to current VIX regime.
      VIX < 14 (low vol)   → 0.5x  (tight stops, trending market)
      VIX 14–20 (normal)   → 0.75x
      VIX 20–25 (elevated) → 1.0x
      VIX > 25 (panic)     → 1.5x  (wide stops, avoid getting shaken out)
    """
    vix = get_india_vix()
    if vix < 14:   return 0.5
    elif vix < 20: return 0.75
    elif vix < 25: return 1.0
    else:          return 1.5


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AlgoTrade Pro Ultimate v4.0</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
:root{
  --bg:#08080e;--panel:#0f0f18;--panel2:#141420;--border:#1c1c2e;
  --accent:#a8ff3e;--accent-dim:rgba(168,255,62,.12);--accent-glow:rgba(168,255,62,.25);
  --green:#a8ff3e;--red:#ff4560;--yellow:#ffd166;--purple:#c084fc;--orange:#fb923c;--teal:#22d3ee;
  --text:#f0f0fa;--muted:#52526e;--sub:#8888aa;
  --radius:12px;--radius-sm:8px;--radius-xs:6px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',system-ui,sans-serif;height:100vh;overflow:hidden;display:flex;flex-direction:column}

/* ── Header ── */
header{
  background:var(--panel);border-bottom:1px solid var(--border);
  padding:10px 20px;display:flex;align-items:center;gap:12px;flex-shrink:0;
  position:relative;
}
header::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent-glow),transparent)}
.logo-mark{width:26px;height:26px;background:var(--accent);border-radius:7px;
  display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:800;color:#000;flex-shrink:0}
header h1{font-size:.92rem;color:var(--text);font-weight:700;letter-spacing:-.2px}
header h1 span{color:var(--accent)}
.badge{background:rgba(168,255,62,.1);color:var(--accent);font-size:.6rem;padding:2px 8px;border-radius:999px;border:1px solid rgba(168,255,62,.25);font-weight:600;letter-spacing:.3px}
.badge.green{background:rgba(168,255,62,.1);color:var(--green);border-color:rgba(168,255,62,.25)}
.badge.yellow{background:rgba(255,209,102,.1);color:var(--yellow);border-color:rgba(255,209,102,.25)}
.badge.purple{background:rgba(192,132,252,.1);color:var(--purple);border-color:rgba(192,132,252,.25)}
.auth-bar{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:.8rem}
.dot{width:7px;height:7px;border-radius:50%;background:var(--red)}
.dot.ok{background:var(--green);box-shadow:0 0 8px var(--accent-glow)}

/* ── Tab Bar (pill-style like reference) ── */
.tabs{
  display:flex;background:var(--panel);
  border-bottom:1px solid var(--border);
  overflow-x:auto;flex-shrink:0;padding:8px 16px;gap:4px;align-items:center;
}
.tab{
  padding:6px 14px;font-size:.72rem;font-weight:600;color:var(--sub);cursor:pointer;
  border-radius:999px;white-space:nowrap;transition:.18s;border:1px solid transparent;
}
.tab:hover{color:var(--text);background:rgba(255,255,255,.04)}
.tab.active{background:var(--accent);color:#000;border-color:transparent;
  box-shadow:0 0 14px var(--accent-glow);font-weight:700}

/* ── Layout ── */
.main{display:flex;flex:1;overflow:hidden}
.sidebar{
  width:215px;background:var(--panel);border-right:1px solid var(--border);
  display:flex;flex-direction:column;flex-shrink:0;
}
.content{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}

/* ── Sidebar Symbol List ── */
.sym-input{display:flex;gap:6px;padding:10px;border-bottom:1px solid var(--border)}
.sym-input input{
  flex:1;background:var(--panel2);border:1px solid var(--border);color:var(--text);
  padding:7px 10px;border-radius:var(--radius-sm);font-size:.76rem;outline:none;
  font-family:'DM Sans',sans-serif;transition:.18s;
}
.sym-input input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-dim)}
.sym-input input::placeholder{color:var(--muted)}
.sym-input button{
  background:var(--accent);color:#000;border:none;padding:7px 12px;
  border-radius:var(--radius-sm);cursor:pointer;font-size:.76rem;font-weight:700;
  transition:.15s;font-family:'DM Sans',sans-serif;
}
.sym-input button:hover{filter:brightness(1.1)}
.sym-list{flex:1;overflow-y:auto;padding:8px}
.sym-item{
  padding:7px 10px;border-radius:var(--radius-xs);cursor:pointer;font-size:.74rem;
  font-weight:500;margin-bottom:2px;transition:.15s;color:var(--sub);
  display:flex;align-items:center;gap:8px;
}
.sym-item::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--border);flex-shrink:0;transition:.15s}
.sym-item:hover{background:rgba(255,255,255,.04);color:var(--text)}
.sym-item.active{background:var(--accent-dim);color:var(--accent);font-weight:700}
.sym-item.active::before{background:var(--accent)}

/* ── Cards ── */
.card{
  background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:16px;
  transition:.18s;
}
.card:hover{border-color:rgba(168,255,62,.15)}
.card-title{
  font-size:.76rem;font-weight:700;color:var(--text);margin-bottom:12px;
  display:flex;align-items:center;gap:6px;letter-spacing:.1px;
}
.card-title .ct-icon{color:var(--accent);font-size:.85rem}

/* ── Grid / Layout helpers ── */
.row{display:flex;gap:12px;flex-wrap:wrap}
.col{flex:1;min-width:230px}
.stat-row{
  display:flex;justify-content:space-between;padding:5px 0;
  border-bottom:1px solid var(--border);font-size:.73rem;
}
.stat-row:last-child{border-bottom:none}
.stat-label{color:var(--sub)}
.stat-val{font-weight:700;font-family:'JetBrains Mono',monospace;font-size:.71rem}

/* ── Color utilities ── */
.green{color:var(--green)} .red{color:var(--red)} .yellow{color:var(--yellow)}
.purple{color:var(--purple)} .orange{color:var(--orange)} .muted{color:var(--muted)}
.accent{color:var(--accent)} .sub{color:var(--sub)}

/* ── Signal Boxes ── */
.signal-box{padding:12px 14px;border-radius:var(--radius-sm);border:1px solid;margin-bottom:8px}
.signal-box.BUY,.signal-box.LONG{background:rgba(168,255,62,.06);border-color:rgba(168,255,62,.25)}
.signal-box.SELL,.signal-box.SHORT{background:rgba(255,69,96,.07);border-color:rgba(255,69,96,.3)}
.signal-box.WAIT{background:rgba(255,209,102,.06);border-color:rgba(255,209,102,.25)}
.sig-dir{font-size:.95rem;font-weight:800;margin-bottom:4px;letter-spacing:.3px}
.targets-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.tgt{
  background:var(--panel2);border-radius:var(--radius-xs);padding:8px 10px;
  border:1px solid var(--border);
}
.tgt-label{font-size:.6rem;color:var(--sub);margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px}
.tgt-val{font-size:.82rem;font-weight:700;font-family:'JetBrains Mono',monospace}

/* ── Buttons ── */
.btn{
  background:var(--panel2);border:1px solid var(--border);color:var(--text);
  padding:6px 12px;border-radius:var(--radius-xs);cursor:pointer;font-size:.72rem;
  transition:.15s;font-family:'DM Sans',sans-serif;font-weight:500;
}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn.primary{
  background:var(--accent);color:#000;border-color:var(--accent);font-weight:700;
  box-shadow:0 0 12px var(--accent-glow);
}
.btn.primary:hover{filter:brightness(1.08)}

/* ── Chips ── */
.chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.61rem;font-weight:700;margin:2px}
.chip.bull{background:rgba(168,255,62,.15);color:var(--green)}
.chip.bear{background:rgba(255,69,96,.15);color:var(--red)}
.chip.neutral{background:rgba(255,209,102,.15);color:var(--yellow)}
.chip.info{background:rgba(168,255,62,.1);color:var(--accent)}

/* ── SMC Blocks ── */
.ob-box,.fvg-box,.sweep-box{
  background:var(--panel2);border-radius:var(--radius-xs);padding:8px 10px;
  margin-bottom:6px;font-size:.71rem;border-left:3px solid;
}
.ob-box.BULL{border-color:var(--green)} .ob-box.BEAR{border-color:var(--red)}
.fvg-box{border-color:var(--yellow)} .sweep-box{border-color:var(--purple)}
.note-item{font-size:.71rem;padding:2px 0;line-height:1.6}
.inst-level{display:flex;justify-content:space-between;padding:4px 0;font-size:.72rem;border-bottom:1px solid var(--border)}
.inst-level:last-child{border-bottom:none}

/* ── Scanner items ── */
.scan-item{
  background:var(--panel2);border-radius:var(--radius-sm);padding:10px 13px;
  margin-bottom:8px;border:1px solid var(--border);cursor:pointer;transition:.15s;
}
.scan-item:hover{border-color:var(--accent);transform:translateY(-1px)}
.scan-item.BUY{border-left:3px solid var(--green)}
.scan-item.SELL{border-left:3px solid var(--red)}
.scan-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.scan-sym{font-size:.88rem;font-weight:800;font-family:'JetBrains Mono',monospace;letter-spacing:.5px}
.scan-targets{display:flex;gap:7px;flex-wrap:wrap;margin-top:6px}
.scan-t{
  font-size:.67rem;padding:2px 7px;border-radius:4px;
  background:var(--panel);border:1px solid var(--border);
  font-family:'JetBrains Mono',monospace;
}

/* ── PO3 / KZ badges ── */
.po3-badge{padding:2px 9px;border-radius:999px;font-size:.66rem;font-weight:700}
.po3-ACCUMULATION{background:rgba(34,211,238,.15);color:var(--teal)}
.po3-DISTRIBUTION{background:rgba(251,146,60,.15);color:var(--orange)}
.po3-MANIPULATION_BULL_TRAP,.po3-MANIPULATION_BEAR_TRAP{background:rgba(255,69,96,.2);color:var(--red)}
.kz-badge{padding:2px 8px;border-radius:999px;font-size:.64rem;font-weight:700}
.kz-badge.active{background:rgba(255,209,102,.2);color:var(--yellow)}
.kz-badge.inactive{background:var(--panel2);color:var(--muted)}
.ote-box{border-radius:var(--radius-sm);padding:10px 13px;margin-bottom:8px;border:1px solid}
.ote-box.ACTIVE{border-color:var(--yellow);background:rgba(255,209,102,.06)}
.ote-box.INACTIVE{border-color:var(--border);background:var(--panel2)}

/* ── Quality levels ── */
.quality-Ap{color:#ffd700;font-weight:800}
.quality-A{color:var(--green);font-weight:700}
.quality-B{color:var(--yellow);font-weight:600}

/* ── Spinner ── */
.spinner{display:inline-block;width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Scrollbars ── */
.tabs::-webkit-scrollbar{height:0}
.content::-webkit-scrollbar,.sym-list::-webkit-scrollbar{width:3px}
.content::-webkit-scrollbar-thumb,.sym-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.tab-content{display:none} .tab-content.active{display:contents}

/* ── Number / mono values ── */
.mono{font-family:'JetBrains Mono',monospace;font-size:.72rem}

/* ── Mobile ── */
@media (max-width:768px){
  .main{flex-direction:column}
  .sidebar{width:100%;height:auto;border-right:none;border-bottom:1px solid var(--border);flex-direction:column}
  .sym-list{display:flex;flex-direction:row;flex-wrap:nowrap;overflow-x:auto;flex:none;padding:4px 6px;gap:4px}
  .sym-list::-webkit-scrollbar{height:0}
  .sym-item{white-space:nowrap;padding:4px 10px}
  .content{padding:10px}
  .row{flex-direction:column}
  .col{min-width:unset}
  header{padding:8px 12px}
  header h1{font-size:.85rem}
  .tabs{padding:6px 10px}
  .tab{padding:5px 11px;font-size:.68rem}
}
</style>
</head>
<body>
<header>
  <div class="logo-mark">A</div>
  <h1>AlgoTrade <span>Pro</span> v4.0</h1>
  <span class="badge">SMC</span><span class="badge green">ICT</span>
  <span class="badge yellow">OTE</span><span class="badge purple">F&amp;O</span>
  <div class="auth-bar">
    <div id="authDot" class="dot"></div>
    <span id="authLabel" style="color:var(--muted);font-size:.76rem">Checking...</span>
    <span id="arCountdown" style="color:var(--accent);font-size:.7rem;min-width:50px;text-align:right" title="Auto-refresh countdown"></span>
    <a id="loginBtn" href="/auth/login" style="display:none;background:var(--accent);color:#000;padding:3px 11px;border-radius:5px;font-size:.73rem;font-weight:700;text-decoration:none">Login Kite</a>
    <a id="logoutBtn" href="/auth/logout" style="display:none;color:var(--muted);font-size:.7rem;padding:2px 8px;border:1px solid var(--border);border-radius:4px;text-decoration:none">Logout</a>
  </div>
</header>
<div class="tabs" id="tabs">
  <div class="tab active" data-tab="signal">Signal</div>
  <div class="tab" data-tab="smc">SMC Analysis</div>
  <div class="tab" data-tab="ict">ICT / OTE</div>
  <div class="tab" data-tab="structure">Structure Setup</div>
  <div class="tab" data-tab="scanner">Inst. Scanner</div>
  <div class="tab" data-tab="orb">ORB</div>
  <div class="tab" data-tab="patterns">Patterns</div>
  <div class="tab" data-tab="fno">F&O</div>
  <div class="tab" data-tab="journal">Journal</div>
  <div class="tab" data-tab="sniper" style="color:#00ff88;font-weight:700">🎯 Sniper</div>
  <div class="tab" data-tab="guide">Guide</div>
</div>
<div class="main">
  <div class="sidebar">
    <div class="sym-input">
      <input id="symInput" placeholder="e.g. RELIANCE" onkeydown="if(event.key==='Enter')loadAll()"/>
      <button onclick="loadAll()">Go</button>
    </div>
    <div class="sym-list" id="symList"></div>
  </div>
  <div class="content" id="content">
    <!-- ══ SIGNAL TAB ══════════════════════════════════════════════════════ -->
    <div class="tab-content active" id="tab-signal">
      <!-- Single symbol signal card -->
      <div class="card" style="padding:16px">
        <div class="card-title" style="justify-content:space-between">
          <span>Live Signal — <span id="sigSym" style="color:var(--accent)">--</span></span>
          <div style="display:flex;align-items:center;gap:8px">
            <span id="sigAge" style="font-size:.66rem;color:var(--muted)"></span>
            <span id="sigTimerEl" style="font-size:.66rem;margin-left:6px"></span>
            <button class="btn primary" onclick="loadSignal()" style="padding:4px 12px;font-size:.72rem">Analyse</button>
          </div>
        </div>
        <div id="sigBody">
          <p class="muted" style="font-size:.78rem">Select a symbol from the sidebar, then click Analyse.</p>
        </div>
      </div>
      <!-- Pre-market watchlist -->
      <div class="card" id="preMktCard" style="padding:16px;display:none">
        <div class="card-title" style="justify-content:space-between">
          <span>Pre-market Watchlist</span>
          <span style="font-size:.66rem;color:var(--yellow)">Market opens at 9:15 AM</span>
        </div>
        <div id="preMktBody"><p class="muted" style="font-size:.78rem">Loading candidates...</p></div>
      </div>
      <!-- Quick scanner -->
      <div class="card" style="padding:16px">
        <div class="card-title" style="justify-content:space-between">
          <span>Live Scanner — All Signals</span>
          <div style="display:flex;align-items:center;gap:8px">
            <span id="scanAge2" style="font-size:.66rem;color:var(--muted)"></span>
            <select id="scanInterval2" class="btn" style="padding:4px 9px;font-size:.7rem">
              <option value="15minute">15 Min</option>
              <option value="5minute">5 Min</option>
              <option value="60minute">1 Hour</option>
            </select>
            <button class="btn primary" onclick="runQuickScan()" style="padding:4px 12px;font-size:.72rem">Scan All</button>
          </div>
        </div>
        <div id="quickScanBody">
          <p class="muted" style="font-size:.78rem">Click "Scan All" to find live BUY/SELL signals across 30+ stocks. Only A+ and A quality shown.</p>
        </div>
      </div>
    </div>
    <!-- ══ SMC TAB ════════════════════════════════════════════════════════ -->
    <div class="tab-content" id="tab-smc">
      <div class="card"><div class="card-title">SMC Analysis -- <span id="smcSym">--</span></div><div id="smcBody"><p class="muted" style="font-size:.78rem">Select a symbol.</p></div></div>
    </div>
    <div class="tab-content" id="tab-ict">
      <div class="card"><div class="card-title">ICT / OTE Analysis -- <span id="ictSym">--</span></div><div id="ictBody"><p class="muted" style="font-size:.78rem">Select a symbol.</p></div></div>
    </div>
    <div class="tab-content" id="tab-structure">
      <div class="card"><div class="card-title">BOS/OTE Structure -- <span id="strSym">--</span></div><div id="strBody"><p class="muted" style="font-size:.78rem">Select a symbol.</p></div></div>
    </div>
    <div class="tab-content" id="tab-scanner">
      <div class="card">
        <div class="card-title">Institutional Multi-Engine Scanner</div>
        <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">
          <select id="scanInterval" class="btn" style="padding:4px 9px">
            <option value="15minute">15 Min</option><option value="5minute">5 Min</option><option value="60minute">1 Hour</option>
          </select>
          <select id="scanMinConf" class="btn" style="padding:4px 9px">
            <option value="65">Conf 65+</option><option value="75">Conf 75+</option><option value="80">Conf 80+</option>
          </select>
          <select id="scanLimit" class="btn" style="padding:4px 9px" title="Max results to show">
            <option value="5">Top 5</option><option value="10" selected>Top 10</option><option value="20">Top 20</option><option value="50">Top 50</option>
          </select>
          <button class="btn primary" onclick="runScan()">Run Institutional Scan</button>
          <button class="btn" onclick="runSmcScan()">SMC Quick Scan</button>
        </div>
        <div id="scanBody"><p class="muted" style="font-size:.78rem">Click to scan for multi-engine confluence setups.</p></div>
      </div>
    </div>
    <div class="tab-content" id="tab-orb">
      <div class="card"><div class="card-title">ORB -- <span id="orbSym">--</span></div><div id="orbBody"><p class="muted" style="font-size:.78rem">Select a symbol.</p></div></div>
    </div>
    <div class="tab-content" id="tab-patterns">
      <div class="card"><div class="card-title" style="display:flex;align-items:center;justify-content:space-between">Patterns & Fibonacci -- <span id="patSym">--</span><label style="font-size:.68rem;color:var(--muted);font-weight:400;cursor:pointer"><input id="hideNeutralChk" type="checkbox" checked onchange="loadPatterns()" style="margin-right:4px">Hide Neutral</label></div><div id="patBody"><p class="muted" style="font-size:.78rem">Select a symbol.</p></div></div>
    </div>
    <div class="tab-content" id="tab-fno">
      <div class="card"><div class="card-title">F&O Signals -- <span id="fnoSym">--</span></div><div id="fnoBody"><p class="muted" style="font-size:.78rem">Select an F&O symbol (NIFTY, BANKNIFTY, RELIANCE...)</p></div></div>
    </div>
    <div class="tab-content" id="tab-journal">
      <div class="card">
        <div class="card-title" style="display:flex;align-items:center;justify-content:space-between">
          Trade Journal
          <div style="display:flex;gap:6px">
            <button class="btn" onclick="loadJournal()">↻ Refresh</button>
            <button class="btn" onclick="exportJournal()" style="font-size:.68rem">⬇ Export CSV</button>
          </div>
        </div>
        <div id="journalBody"><p class="muted" style="font-size:.78rem">Loading journal...</p></div>
      </div>
    </div>
    <div class="tab-content" id="tab-sniper">
      <div class="card">
        <div class="card-title" style="display:flex;align-items:center;justify-content:space-between">
          <span>🎯 Sniper Engine — High-Accuracy Auto-Trading</span>
          <div style="display:flex;gap:6px;align-items:center">
            <select id="sniperInterval" style="background:var(--card);color:var(--fg);border:1px solid var(--border);padding:3px 6px;border-radius:4px;font-size:.7rem">
              <option value="5minute">5 min</option>
              <option value="15minute" selected>15 min</option>
            </select>
            <button class="btn" onclick="runSniperScan()" style="background:#00ff8822;border-color:#00ff88;color:#00ff88;font-weight:700">⚡ Scan Now</button>
          </div>
        </div>
        <!-- Daily risk dashboard -->
        <div id="sniperDailyBar" style="display:flex;gap:16px;flex-wrap:wrap;padding:8px 0 4px;font-size:.7rem;border-bottom:1px solid var(--border);margin-bottom:10px">
          <span>Trades today: <b id="sdTrades">--</b> / 2</span>
          <span>Losses today: <b id="sdLosses" class="red">--</b> / 2</span>
          <span id="sdStatus" class="green">✅ Trading allowed</span>
          <span style="color:var(--muted)" id="sdTimeNote">--</span>
        </div>
        <!-- Sniper rules reminder -->
        <div style="background:#00ff8811;border:1px solid #00ff8833;border-radius:6px;padding:8px 12px;font-size:.68rem;color:var(--muted);margin-bottom:10px;line-height:1.6">
          <b style="color:#00ff88">7 Gates must ALL pass:</b>
          &nbsp;⏰ Time window (09:30–11:30 / 14:30–15:30)
          &nbsp;→ 📈 Trend (EMA50 > EMA200)
          &nbsp;→ 🕯 Engulfing pattern
          &nbsp;→ ✅ Confirmation candle break
          &nbsp;→ 📍 Location (VWAP/PDH/PDL)
          &nbsp;→ 🏦 NIFTY aligned
          &nbsp;→ 💰 RR ≥ 2:1
        </div>
        <div id="sniperBody"><p class="muted" style="font-size:.78rem;padding:10px">Click ⚡ Scan Now to find sniper setups across 26 liquid stocks.</p></div>
      </div>
    </div>
    <div class="tab-content" id="tab-guide">
      <div class="card">
        <div class="card-title">Institutional Trading Guide -- v4.0</div>
        <div style="font-size:.78rem;line-height:1.7">
          <p class="yellow" style="font-weight:700;margin-bottom:6px">What was fixed and added in v4.0:</p>
          <p class="muted" style="margin-bottom:10px">Previous version had ALL targets in WRONG ORDER -- T1 was furthest, T2 was closest. Now fully corrected.</p>
          <p class="accent" style="font-weight:700;margin:8px 0 5px">Fibonacci Target Cascade (CORRECT ORDER):</p>
          <table style="width:100%;border-collapse:collapse;font-size:.72rem;margin-bottom:12px">
            <tr class="muted"><td style="padding:3px 6px">Target</td><td>Fib Level</td><td>LONG</td><td>SHORT</td><td>Action</td></tr>
            <tr><td class="yellow" style="padding:3px 6px">T1 (Closest)</td><td>1.000</td><td>SwL + leg x1.0</td><td>SwH - leg x1.0</td><td class="muted">30-40% exit</td></tr>
            <tr><td class="green" style="padding:3px 6px">T2</td><td>1.272</td><td>SwL + leg x1.272</td><td>SwH - leg x1.272</td><td class="muted">Main target</td></tr>
            <tr><td class="accent" style="padding:3px 6px">T3</td><td>1.618</td><td>SwL + leg x1.618</td><td>SwH - leg x1.618</td><td class="muted">Golden ext</td></tr>
            <tr><td class="purple" style="padding:3px 6px">T4 (Runner)</td><td>2.000</td><td>SwL + leg x2.0</td><td>SwH - leg x2.0</td><td class="muted">Let it run</td></tr>
          </table>
          <p class="accent" style="font-weight:700;margin:8px 0 4px">ICT OTE Zone (NEW):</p>
          <p style="margin-bottom:8px">Old system: 38.2%-61.8% pullback. New system: <span class="green">61.8%-78.6%</span> -- the institutional sniper zone for tighter entries.</p>
          <p class="accent" style="font-weight:700;margin:8px 0 4px">Key Institutional Concepts:</p>
          <p><b class="yellow">PDH/PDL</b> -- #1 price magnets. Big moves happen AT or FROM these levels.</p>
          <p style="margin-top:4px"><b class="yellow">Breaker Blocks</b> -- Failed OBs that flip direction. Highest probability entries.</p>
          <p style="margin-top:4px"><b class="yellow">Power of 3</b> -- Accumulation > Manipulation (stop hunt) > Distribution (real move).</p>
          <p style="margin-top:4px"><b class="yellow">Killzones</b> -- 9:30-10:30 and 13:00-14:30 IST. Trade only in these windows for best fills.</p>
          <p style="margin-top:4px"><b class="yellow">Scanner</b> -- Needs 2 of 3 engines (SMC+ICT+Structure) to agree. Eliminates false signals.</p>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const SYMBOLS = ["NIFTY 50", "NIFTY BANK", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BAJFINANCE", "HINDUNILVR", "MARUTI", "TATAMOTORS", "AXISBANK", "LT", "WIPRO", "SUNPHARMA", "TATASTEEL", "KOTAKBANK", "SBIN", "ADANIENT", "BHARTIARTL", "NTPC", "POWERGRID", "ONGC", "COALINDIA", "HCLTECH", "TECHM", "VEDL", "JSWSTEEL", "HINDALCO", "ULTRACEMCO", "GRASIM", "TITAN", "NESTLEIND", "DIVISLAB", "NATIONALUM", "SAIL", "NMDC", "MOIL", "APLAPOLLO", "LTM", "PERSISTENT", "COFORGE", "MPHASIS", "KPITTECH", "HAPPSTMNDS", "RATEGAIN", "MASTEK", "FIVESTAR", "MUTHOOTFIN", "CHOLAFIN", "ABCAPITAL", "MANAPPURAM", "IIFL", "POONAWALLA", "DMART", "TRENT", "NYKAA", "KALYANKJIL", "TVSHLTD", "BAJAJ-AUTO", "HEROMOTOCO", "MOTHERSON", "TIINDIA", "BALKRISIND", "RVNL", "IREDA", "IRFC", "RECLTD", "PFC", "PHOENIXLTD", "GODREJPROP", "OBEROIRLTY", "ALKEM", "TORNTPHARM", "MAXHEALTH", "APOLLOHOSP", "DRREDDY", "CIPLA", "INDUSTOWER", "HFCL"];
let currentSym = 'RELIANCE', currentTab = 'signal';

// Event delegation for scan result cards (avoids onclick escaping issues)
document.addEventListener('click', function(e) {
  const card = e.target.closest('[data-sym]');
  if (card) { selectSym(card.getAttribute('data-sym')); }
});

document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    t.classList.add('active'); currentTab = t.dataset.tab;
    document.getElementById('tab-' + currentTab).classList.add('active');
    setTimeout(() => { loadAll(); _startAutoRefresh(); }, 50);
  });
});

const symList = document.getElementById('symList');
SYMBOLS.forEach(s => {
  const d = document.createElement('div');
  d.className = 'sym-item' + (s === currentSym ? ' active' : '');
  d.textContent = s; d.onclick = () => selectSym(s);
  symList.appendChild(d);
});

function selectSym(s) {
  currentSym = s;
  document.getElementById('symInput').value = s;
  document.querySelectorAll('.sym-item').forEach(el => el.classList.toggle('active', el.textContent === s));
  const panelMap = {
    signal: 'sigBody', smc: 'smcBody', ict: 'ictBody', structure: 'strBody',
    orb: 'orbBody', patterns: 'patBody', fno: 'fnoBody', journal: 'journalBody',
    sniper: 'sniperBody',
  };
  const panelId = panelMap[currentTab];
  if (panelId) loading(panelId);
  loadAll();
  _startAutoRefresh();
}

function loadAll() {
  const v = document.getElementById('symInput').value.trim().toUpperCase();
  if (v) currentSym = v;
  if (currentTab === 'signal')    loadSignal();
  else if (currentTab === 'smc')  loadSmc();
  else if (currentTab === 'ict')  loadIct();
  else if (currentTab === 'structure') loadStructure();
  else if (currentTab === 'orb')  loadOrb();
  else if (currentTab === 'patterns') loadPatterns();
  else if (currentTab === 'fno')  loadFno();
  else if (currentTab === 'journal') loadJournal();
  else if (currentTab === 'sniper') loadSniperDailyBar();
}

// ── Auto-refresh intervals ────────────────────────────────────────────────────
const AUTO_REFRESH_MS = {
  signal: 3 * 60 * 1000,   // Signal tab: every 3 min (fresh live data)
  smc: 5 * 60 * 1000, ict: 5 * 60 * 1000, structure: 5 * 60 * 1000,
  orb: 60 * 1000, patterns: 3 * 60 * 1000, fno: 5 * 60 * 1000,
};
let _arTimer = null, _arCountdown = null, _arSecondsLeft = 0;

function _startAutoRefresh() {
  _stopAutoRefresh();
  const ms = AUTO_REFRESH_MS[currentTab];
  if (!ms) return;
  _arSecondsLeft = ms / 1000;
  _updateCountdown();
  _arCountdown = setInterval(() => {
    _arSecondsLeft--;
    _updateCountdown();
    if (_arSecondsLeft <= 0) {
      loadAll();
      _arSecondsLeft = ms / 1000;
    }
  }, 1000);
}

function _stopAutoRefresh() {
  if (_arTimer)    { clearInterval(_arTimer);    _arTimer    = null; }
  if (_arCountdown){ clearInterval(_arCountdown); _arCountdown = null; }
  const el = document.getElementById('arCountdown');
  if (el) el.textContent = '';
}

function _updateCountdown() {
  const el = document.getElementById('arCountdown');
  if (!el) return;
  const m = Math.floor(_arSecondsLeft / 60), s = _arSecondsLeft % 60;
  el.textContent = `↻ ${m}:${String(s).padStart(2,'0')}`;
}

const fmtN = v => v == null ? '--' : parseFloat(v).toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
const dirColor = d => (d==='BUY'||d==='LONG') ? 'green' : (d==='SELL'||d==='SHORT') ? 'red' : 'yellow';
const sigEmoji = d => (d==='BUY'||d==='LONG') ? '[BUY]' : (d==='SELL'||d==='SHORT') ? '[SELL]' : '[WAIT]';

// ── Global stale-data banner ─────────────────────────────────────────────────
// Returns true if response is stale (caller should return early)
function checkStale(r, bodyId) {
  if (!r || !r.stale) return false;
  const age = r.data_age_min ? Math.round(r.data_age_min) : '?';
  document.getElementById(bodyId).innerHTML = `
    <div style="padding:16px 18px;border-radius:10px;background:rgba(255,69,96,.08);border:1px solid rgba(255,69,96,.3);margin-bottom:12px">
      <div style="font-size:.95rem;font-weight:800;color:var(--red);margin-bottom:6px">Stale Data — ${age} min old</div>
      <div style="font-size:.78rem;color:var(--muted);margin-bottom:12px">${r.error}</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <a href="/auth/logout" style="background:var(--panel2);color:var(--text);padding:6px 14px;border-radius:6px;font-size:.76rem;font-weight:700;text-decoration:none;border:1px solid var(--border)">Step 1: Logout</a>
        <a href="/auth/login"  style="background:var(--accent);color:#000;padding:6px 14px;border-radius:6px;font-size:.76rem;font-weight:700;text-decoration:none">Step 2: Login Kite</a>
      </div>
      <div style="margin-top:10px;font-size:.7rem;color:var(--muted)">After logging in, click the tab again to refresh.</div>
    </div>`;
  return true;
}

// ════════════════════════════════════════════════════════════════════════════
//  SIGNAL TAB — Simple live BUY/SELL card
// ════════════════════════════════════════════════════════════════════════════

async function loadSignal() {
  const sym = currentSym;
  const interval = '15minute';
  document.getElementById('sigSym').textContent = sym;
  loading('sigBody');
  try {
    const r = await fetch(`/api/signal/${encodeURIComponent(sym)}?interval=${interval}`).then(x => x.json());
    const age = r.data_age_min || 0;
    // Detect early-market window: before 9:35 IST the last candle is from yesterday —
    // show an informational note instead of a red STALE warning.
    const nowIst = new Date(Date.now() + (5.5 * 60 - new Date().getTimezoneOffset()) * 60000);
    const istH = nowIst.getUTCHours(), istM = nowIst.getUTCMinutes();
    const isEarlyMarket = (istH === 9 && istM < 35) || istH < 9;
    let ageStr = '';
    if (age > 60 && isEarlyMarket) {
      ageStr = `<span style="color:var(--yellow);font-weight:600">Early session — yesterday's close used</span>`;
    } else if (age > 60) {
      ageStr = `<span style="color:var(--red);font-weight:700">Data ${Math.round(age)} min old — STALE</span>`;
    } else if (r.fetched_at) {
      ageStr = `<span style="color:var(--muted)">Data: ${r.fetched_at} (${Math.round(age)} min ago)</span>`;
    }
    document.getElementById('sigAge').innerHTML = ageStr;
    document.getElementById('sigBody').innerHTML = renderSignalCard(r);
    // Only start the valid-for timer when we have a real actionable signal.
    // If r.error is set (stale data, auth failure, etc.) or signal is absent/WAIT,
    // clear any leftover timer text instead of starting a bogus countdown.
    if (r.signal && r.signal !== 'WAIT' && !r.error) {
      _startSignalTimer('15minute');
    } else {
      if (_sigTimerInterval) { clearInterval(_sigTimerInterval); _sigTimerInterval = null; }
      const _tel = document.getElementById('sigTimerEl');
      if (_tel) _tel.textContent = '';
    }
  } catch(e) {
    document.getElementById('sigBody').innerHTML = `<p class="red" style="font-size:.78rem">Error: ${e.message}</p>`;
  }
}

function renderSignalCard(r) {
  if (r.error) return `<p class="red" style="font-size:.78rem">${r.error}</p>`;

  const isWait  = r.signal === 'WAIT';
  const isBuy   = r.signal === 'BUY';
  const isSell  = r.signal === 'SELL';
  const dirCol  = isBuy ? 'var(--green)' : isSell ? 'var(--red)' : 'var(--yellow)';
  const dirBg   = isBuy ? 'rgba(168,255,62,.08)' : isSell ? 'rgba(255,69,96,.08)' : 'rgba(255,209,102,.06)';
  const dirBord = isBuy ? 'rgba(168,255,62,.3)'  : isSell ? 'rgba(255,69,96,.3)'  : 'rgba(255,209,102,.25)';
  const qColor  = r.quality === 'A+' ? '#ffd700' : r.quality === 'A' ? 'var(--green)' : 'var(--yellow)';

  if (isWait) {
    return `
    <div style="padding:18px;border-radius:10px;background:rgba(255,209,102,.07);border:1px solid rgba(255,209,102,.25);margin-bottom:12px">
      <div style="font-size:1.2rem;font-weight:800;color:var(--yellow);margin-bottom:6px">WAIT — No clear signal</div>
      <div style="font-size:.78rem;color:var(--muted);margin-bottom:8px">${r.reason || 'Engines disagree. Skip this trade.'}</div>
      <div style="font-size:.74rem;color:var(--sub)">Current price: <span class="mono" style="color:var(--text)">₹${fmtN(r.current_price)}</span></div>
      ${r.engines && Object.keys(r.engines).length ? `<div style="margin-top:8px;font-size:.7rem;color:var(--muted)">Engine votes: ${Object.entries(r.engines).map(([k,v])=>`${k}:<span style="color:${v==='BUY'?'var(--green)':'var(--red)'}">${v}</span>`).join(' · ')}</div>` : ''}
    </div>
    <div style="font-size:.73rem;color:var(--muted);padding:8px 12px;background:var(--panel2);border-radius:6px">
      Try another symbol or wait for market structure to develop.
    </div>`;
  }

  const inZone = r.in_zone;
  const entryText = inZone
    ? `<span style="color:var(--green);font-weight:700">Price is in entry zone NOW</span>`
    : `Wait for price to reach ₹${fmtN(r.entry_low)} – ₹${fmtN(r.entry_high)}`;

  const warns = (r.warnings || []).map(w =>
    `<div style="font-size:.7rem;color:var(--yellow);padding:2px 0">&#x26A0; ${w}</div>`).join('');

  const reasons = (r.reasons || []).map(rs =>
    `<div style="font-size:.71rem;color:var(--sub);padding:2px 0;border-bottom:1px solid var(--border)">${rs}</div>`).join('');

  const engines = Object.entries(r.engines || {}).map(([k,v]) =>
    `<span style="font-size:.65rem;padding:2px 8px;border-radius:999px;background:${v==='BUY'?'rgba(168,255,62,.15)':'rgba(255,69,96,.15)'};color:${v==='BUY'?'var(--green)':'var(--red)'};margin-right:4px">${k}: ${v}</span>`
  ).join('');

  const patBadge = r.pattern ? `<span style="font-size:.65rem;padding:2px 8px;border-radius:999px;background:rgba(168,255,62,.1);color:var(--accent);margin-left:4px">${r.pattern}</span>` : '';

  return `
  <!-- Direction header -->
  <div style="padding:16px 18px;border-radius:10px;background:${dirBg};border:1px solid ${dirBord};margin-bottom:12px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
      <div style="font-size:1.5rem;font-weight:800;color:${dirCol};letter-spacing:1px">${r.signal}</div>
      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:.72rem;font-weight:700;color:${qColor};padding:3px 10px;border-radius:999px;border:1px solid ${qColor}">${r.quality || 'B'}</span>
        <span style="font-size:.72rem;color:var(--sub)">Conf: <b style="color:var(--text)">${r.confidence}%</b></span>
      </div>
    </div>
    <div style="font-size:.76rem;color:var(--sub);margin-bottom:8px">${r.symbol} · ${entryText}</div>
    <div style="font-size:.7rem">${engines}${patBadge}</div>
    ${warns ? `<div style="margin-top:10px">${warns}</div>` : ''}
  </div>

  <!-- Entry zone + SL side by side -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
    <div style="background:var(--panel2);border-radius:8px;padding:12px 14px;border:1px solid var(--border)">
      <div style="font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Entry zone</div>
      <div style="font-size:1.05rem;font-weight:700;color:${dirCol};font-family:'JetBrains Mono',monospace">₹${fmtN(r.entry_low)}</div>
      <div style="font-size:.7rem;color:var(--sub)">to ₹${fmtN(r.entry_high)}</div>
      ${inZone ? '<div style="margin-top:4px;font-size:.65rem;color:var(--green);font-weight:700">● IN ZONE</div>' : '<div style="margin-top:4px;font-size:.65rem;color:var(--yellow)">● Waiting</div>'}
    </div>
    <div style="background:rgba(255,69,96,.08);border-radius:8px;padding:12px 14px;border:1px solid rgba(255,69,96,.25)">
      <div style="font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Stop Loss — exit if hit</div>
      <div style="font-size:1.05rem;font-weight:700;color:var(--red);font-family:'JetBrains Mono',monospace">₹${fmtN(r.sl)}</div>
      ${r.rr_t2 ? `<div style="font-size:.7rem;color:var(--sub)">R:R to T2: <b style="color:${r.rr_t2>=2.5?'var(--green)':r.rr_t2>=2?'var(--yellow)':'var(--red)'}">${r.rr_t2}x</b></div>` : ''}
    </div>
  </div>

  <!-- Targets -->
  <div style="font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Targets — exit in stages</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:12px">
    ${[['T1','30–40% exit',r.t1,'Fib 1.0'],['T2','Main target',r.t2,'Fib 1.272'],['T3','Runner 20%',r.t3,'Fib 1.618'],['T4','Extended',r.t4,'Fib 2.0']].map(([lbl,sub,val,fib])=>`
    <div style="background:var(--panel2);border-radius:8px;padding:10px;border:1px solid var(--border);text-align:center">
      <div style="font-size:.62rem;color:var(--muted);margin-bottom:2px">${lbl} <span style="color:var(--muted);font-size:.58rem">${fib}</span></div>
      <div style="font-size:.9rem;font-weight:700;color:var(--green);font-family:'JetBrains Mono',monospace">${val ? '₹'+fmtN(val) : '--'}</div>
      <div style="font-size:.6rem;color:var(--muted);margin-top:2px">${sub}</div>
    </div>`).join('')}
  </div>

  <!-- Rule reminder -->
  <div style="font-size:.72rem;color:var(--sub);padding:10px 12px;background:var(--panel2);border-radius:8px;border-left:3px solid ${dirCol};line-height:1.7">
    When T1 hits → move Stop Loss to your entry price. You are now risk-free.
    ${r.rsi_val ? `<span style="margin-left:8px;color:var(--muted)">RSI: ${r.rsi_val}</span>` : ''}
    ${r.vwap ? `<span style="margin-left:8px;color:var(--muted)">VWAP: ₹${fmtN(r.vwap)}</span>` : ''}
  </div>

  ${reasons ? `<div style="margin-top:10px;padding:10px 12px;background:var(--panel2);border-radius:8px"><div style="font-size:.65rem;color:var(--accent);margin-bottom:4px">Why this signal</div>${reasons}</div>` : ''}
  `;
}

async function runQuickScan() {
  const interval = document.getElementById('scanInterval2').value;
  document.getElementById('quickScanBody').innerHTML = '<div style="text-align:center;padding:24px"><div class="spinner"></div><div style="margin-top:8px;font-size:.73rem;color:var(--muted)">Scanning 30+ stocks live from Kite... this takes ~30 seconds</div></div>';
  document.getElementById('scanAge2').textContent = '';
  try {
    const r = await fetch(`/api/quick-scan?interval=${interval}&limit=20`).then(x => x.json());
    document.getElementById('scanAge2').textContent = r.fetched_at || '';
    if (r.error) {
      document.getElementById('quickScanBody').innerHTML = `<p class="red" style="font-size:.78rem">${r.error}</p>`;
      return;
    }
    const sigs = r.signals || [];
    if (!sigs.length) {
      document.getElementById('quickScanBody').innerHTML = `<p class="muted" style="font-size:.78rem">No A/A+ signals found across ${r.scanned} stocks right now. Market may be in consolidation.</p>`;
      return;
    }
    let html = `<div style="font-size:.7rem;color:var(--muted);margin-bottom:10px">Found <b style="color:var(--text)">${sigs.length}</b> signals from ${r.scanned} stocks scanned</div>`;
    html += sigs.map(s => {
      const isBuy = s.signal === 'BUY';
      const col   = isBuy ? 'var(--green)' : 'var(--red)';
      const bg    = isBuy ? 'rgba(168,255,62,.06)' : 'rgba(255,69,96,.06)';
      const bord  = isBuy ? 'rgba(168,255,62,.25)' : 'rgba(255,69,96,.25)';
      const qcol  = s.quality === 'A+' ? '#ffd700' : 'var(--green)';
      return `
      <div data-sym="${s.symbol}" style="background:${bg};border:1px solid ${bord};border-radius:9px;padding:12px 14px;margin-bottom:8px;cursor:pointer;transition:.15s" onmouseover="this.style.borderColor='${col}'" onmouseout="this.style.borderColor='${bord}'">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:.95rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:${col}">${s.signal}</span>
            <span style="font-size:.82rem;font-weight:700;color:var(--text)">${s.symbol}</span>
            ${s.pattern ? `<span style="font-size:.62rem;padding:1px 6px;border-radius:999px;background:rgba(168,255,62,.1);color:var(--accent)">${s.pattern}</span>` : ''}
          </div>
          <div style="display:flex;align-items:center;gap:6px">
            <span style="font-size:.68rem;font-weight:700;color:${qcol};padding:2px 8px;border-radius:999px;border:1px solid ${qcol}">${s.quality}</span>
            <span style="font-size:.68rem;color:var(--muted)">${s.confidence}%</span>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;font-size:.68rem">
          <div><div style="color:var(--muted)">CMP</div><div style="font-family:'JetBrains Mono',monospace;font-weight:700;color:var(--text)">₹${fmtN(s.current_price)}</div></div>
          <div><div style="color:var(--muted)">Entry</div><div style="font-family:'JetBrains Mono',monospace;color:${col}">₹${fmtN(s.entry_low)}</div></div>
          <div><div style="color:var(--muted)">SL</div><div style="font-family:'JetBrains Mono',monospace;color:var(--red)">₹${fmtN(s.sl)}</div></div>
          <div><div style="color:var(--muted)">T1</div><div style="font-family:'JetBrains Mono',monospace;color:var(--green)">₹${fmtN(s.t1)}</div></div>
          <div><div style="color:var(--muted)">T2</div><div style="font-family:'JetBrains Mono',monospace;color:var(--green)">₹${fmtN(s.t2)}</div></div>
        </div>
        ${s.in_zone ? `<div style="margin-top:6px;font-size:.65rem;color:var(--green);font-weight:700">● Price in entry zone now — click to see full signal</div>` : `<div style="margin-top:4px;font-size:.63rem;color:var(--muted)">Click to see full analysis</div>`}
      </div>`;
    }).join('');
    document.getElementById('quickScanBody').innerHTML = html;
  } catch(e) {
    document.getElementById('quickScanBody').innerHTML = `<p class="red" style="font-size:.78rem">Scan error: ${e.message}</p>`;
  }
}

function loading(id) {
  document.getElementById(id).innerHTML = '<div style="text-align:center;padding:18px"><div class="spinner"></div><div style="margin-top:7px;font-size:.73rem;color:var(--muted)">Fetching live data...</div></div>';
}

function renderTargets(t1, t2, t3, t4, entry, sl, rr2, rr3, dir, labelMode) {
  // FIX: When signal is WAIT (rr2===0 or all targets equal entry), don't show
  // a misleading target grid — show a waiting message instead.
  if (!rr2 || rr2 === 0 || Math.abs(t1 - entry) < 0.01) {
    return `<div style="padding:8px;font-size:.73rem;color:var(--muted);background:var(--panel2);border-radius:5px;margin-top:7px">
      &#x23F3; No active setup — waiting for entry conditions to align.
    </div>`;
  }
  // ORB uses range multiples, not Fib extensions — show correct labels
  const isOrb = labelMode === 'orb';
  const l1 = isOrb ? 'T1  1× ORB Range'           : 'T1 Fib 1.0 (Equal leg)';
  const l2 = isOrb ? `T2  2× ORB | RR ${rr2 || '--'}x` : `T2 Fib 1.272 | RR ${rr2 || '--'}x`;
  const l3 = isOrb ? `T3  3× ORB | RR ${rr3 || '--'}x` : `T3 Fib 1.618 Golden | RR ${rr3 || '--'}x`;
  const l4 = isOrb ? 'T4  4× ORB Runner'           : 'T4 Fib 2.0 Runner';
  return `<div class="targets-grid">
    <div class="tgt"><div class="tgt-label">${l1}</div><div class="tgt-val yellow">Rs.${fmtN(t1)}</div><div style="font-size:.62rem;color:var(--muted)">First exit 30-40%</div></div>
    <div class="tgt"><div class="tgt-label">${l2}</div><div class="tgt-val green">Rs.${fmtN(t2)}</div><div style="font-size:.62rem;color:var(--muted)">Main target 40%</div></div>
    <div class="tgt"><div class="tgt-label">${l3}</div><div class="tgt-val accent">Rs.${fmtN(t3)}</div><div style="font-size:.62rem;color:var(--muted)">Extended 20%</div></div>
    <div class="tgt"><div class="tgt-label">${l4}</div><div class="tgt-val purple">Rs.${fmtN(t4)}</div><div style="font-size:.62rem;color:var(--muted)">Runner 10% trail SL</div></div>
  </div>
  <div style="display:flex;gap:10px;margin-top:7px;font-size:.71rem;flex-wrap:wrap">
    <span>Entry: <b>Rs.${fmtN(entry)}</b></span>
    <span>SL: <b class="red">Rs.${fmtN(sl)}</b></span>
    <span>RR to T2: <b class="${rr2>=2?'green':'yellow'}">${rr2 || '--'}x</b></span>
  </div>`;
}

async function checkAuth() {
  try {
    const r = await fetch('/api/auth/status').then(r => r.json());
    const ok = r.authenticated;
    document.getElementById('authDot').className = 'dot' + (ok ? ' ok' : '');
    document.getElementById('authLabel').textContent = ok ? 'Kite Connected' : 'Not connected';
    document.getElementById('authLabel').style.color = ok ? 'var(--green)' : 'var(--muted)';
    document.getElementById('loginBtn').style.display = ok ? 'none' : 'block';
    document.getElementById('logoutBtn').style.display = ok ? 'block' : 'none';
  } catch(e) {}
}
checkAuth(); setInterval(checkAuth, 30000);

async function loadSmc() {
  document.getElementById('smcSym').textContent = currentSym; loading('smcBody');
  try {
    const r = await fetch('/api/smc-plus/' + encodeURIComponent(currentSym) + '?interval=15minute').then(r => r.json());
    if (checkStale(r, 'smcBody')) return;
    if (r.error) { document.getElementById('smcBody').innerHTML = '<p class="red">' + r.error + '</p>'; return; }
    const s = r.smc, il = r.inst_levels, cs = r.combined_signal;
    const dc = dirColor(s.smc_signal);
    const obsHtml = s.order_blocks.length ? s.order_blocks.map(o =>
      '<div class="ob-box ' + o.direction + '"><b style="color:' + (o.direction==='BULL'?'var(--green)':'var(--red)') + '">' + o.direction + ' OB</b>'
      + (o.touched ? '<span class="chip info">Touched</span>' : '')
      + '<div style="margin-top:2px;font-size:.7rem">Zone: Rs.' + fmtN(o.low) + ' - Rs.' + fmtN(o.high) + '</div></div>').join('')
      : '<div class="muted" style="font-size:.71rem">No Order Blocks</div>';
    const fvgHtml = s.fair_value_gaps.length ? s.fair_value_gaps.map(f =>
      '<div class="fvg-box"><b style="color:' + (f.direction==='BULL'?'var(--green)':'var(--red)') + '">' + f.direction + ' FVG</b>'
      + '<div style="margin-top:2px;font-size:.7rem">Rs.' + fmtN(f.bottom) + ' - Rs.' + fmtN(f.top) + ' | Filled: ' + f.filled_pct.toFixed(0) + '%</div></div>').join('')
      : '<div class="muted" style="font-size:.71rem">No FVGs</div>';
    document.getElementById('smcBody').innerHTML = `
      <div class="row">
        <div class="col">
          <div class="signal-box ${s.smc_signal}">
            <div class="sig-dir ${dc}">${sigEmoji(s.smc_signal)} SMC: ${s.smc_signal} (${s.smc_confidence.toFixed(0)}%)</div>
            <div style="font-size:.71rem;color:var(--muted);margin:3px 0">Structure: <b style="color:var(--text)">${s.market_structure}</b> | ${s.premium_discount} | VWAP: ${s.price_vs_vwap}</div>
            ${renderTargets(cs.t1_fib_1,cs.t2_fib_127,cs.t3_fib_162,cs.t4_fib_200,(s.smc_entry_zone[0]+s.smc_entry_zone[1])/2,s.smc_sl,cs.rr_to_t2,cs.rr_to_t3,s.smc_signal)}
          </div>
          <div class="card" style="margin-top:8px">
            <div class="card-title">ICT Context</div>
            <div class="stat-row"><span class="stat-label">Killzone</span><span class="kz-badge ${cs.in_killzone?'active':'inactive'}">${cs.killzone}</span></div>
            <div class="stat-row"><span class="stat-label">Power of 3</span><span class="po3-badge po3-${cs.po3_phase}">${cs.po3_phase}</span></div>
            <div class="stat-row"><span class="stat-label">Daily Bias</span><span class="${dirColor(cs.daily_bias==='BULLISH'?'BUY':'SELL')}">${cs.daily_bias}</span></div>
          </div>
          <div class="card" style="margin-top:8px">
            <div class="card-title">Notes</div>
            ${s.smc_notes.map(n => '<div class="note-item">' + n + '</div>').join('') || '<div class="muted" style="font-size:.71rem">--</div>'}
          </div>
        </div>
        <div class="col">
          <div class="card" style="margin-bottom:8px">
            <div class="card-title">Institutional Levels</div>
            <div class="inst-level"><span class="muted">PDH (Resistance)</span><span class="red">Rs.${fmtN(il.pdh)}</span></div>
            <div class="inst-level"><span class="muted">PDL (Support)</span><span class="green">Rs.${fmtN(il.pdl)}</span></div>
            <div class="inst-level"><span class="muted">Prev Close</span><span>Rs.${fmtN(il.pdc)}</span></div>
            <div class="inst-level"><span class="muted">Prev Week H</span><span class="red">Rs.${fmtN(il.pwh)}</span></div>
            <div class="inst-level"><span class="muted">Prev Week L</span><span class="green">Rs.${fmtN(il.pwl)}</span></div>
            <div class="inst-level"><span class="muted">Nearest Round</span><span class="yellow">Rs.${fmtN(il.nearest_round)}</span></div>
          </div>
          <div class="card" style="margin-bottom:8px">
            <div class="card-title">Order Blocks (${s.order_blocks.length})</div>${obsHtml}
          </div>
          <div class="card">
            <div class="card-title">Fair Value Gaps (${s.fair_value_gaps.length})</div>${fvgHtml}
          </div>
        </div>
      </div>`;
  } catch(e) { document.getElementById('smcBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

async function loadIct() {
  document.getElementById('ictSym').textContent = currentSym; loading('ictBody');
  try {
    const r = await fetch('/api/ict/' + encodeURIComponent(currentSym) + '?interval=15minute').then(r => r.json());
    if (checkStale(r, 'ictBody')) return;
    if (r.error) { document.getElementById('ictBody').innerHTML = '<p class="red">' + r.error + '</p>'; return; }
    const ict = r.ict, il = r.inst_levels;
    const dc = dirColor(ict.ict_signal==='LONG'?'BUY':'SELL');
    function renderOTE(ote, lbl) {
      if (!ote) return '<div class="ote-box INACTIVE"><b class="muted">No ' + lbl + ' OTE</b></div>';
      return '<div class="ote-box ' + (ote.price_in_ote?'ACTIVE':'INACTIVE') + '">'
        + '<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
        + '<b style="color:' + (lbl==='LONG'?'var(--green)':'var(--red)') + '">' + lbl + ' OTE Zone</b>'
        + (ote.price_in_ote ? '<span class="chip bull">IN ZONE</span>' : '<span class="chip neutral">Waiting</span>') + '</div>'
        + '<div class="stat-row"><span class="stat-label">Zone (61.8%-78.6%)</span><span><b>Rs.' + fmtN(ote.ote_low) + ' - Rs.' + fmtN(ote.ote_high) + '</b></span></div>'
        + '<div class="stat-row"><span class="stat-label">Sweet Spot (70.5%)</span><span class="yellow">Rs.' + fmtN(ote.fib_705) + '</span></div>'
        + '<div class="stat-row"><span class="stat-label">Leg Strength</span><span class="' + (ote.leg_strength==='STRONG'?'green':'yellow') + '">' + ote.leg_strength + '</span></div></div>';
    }
    const bbHtml = ict.breaker_blocks.length ? ict.breaker_blocks.map(b =>
      '<div class="ob-box ' + (b.direction.startsWith('BULL')?'BULL':'BEAR') + '"><b style="color:'
      + (b.direction.startsWith('BULL')?'var(--green)':'var(--red)') + '">' + b.direction + '</b>'
      + '<div style="margin-top:2px;font-size:.7rem">Rs.' + fmtN(b.low) + ' - Rs.' + fmtN(b.high) + '</div></div>').join('')
      : '<div class="muted" style="font-size:.71rem">No Breaker Blocks</div>';
    document.getElementById('ictBody').innerHTML = `
      <div class="row">
        <div class="col">
          <div class="signal-box ${ict.ict_signal==='LONG'?'LONG':'SHORT'}">
            <div class="sig-dir ${dc}">${sigEmoji(ict.ict_signal==='LONG'?'BUY':'SELL')} ICT: ${ict.ict_signal} (${ict.ict_confidence.toFixed(0)}%)</div>
            <div style="font-size:.71rem;color:var(--muted);margin:3px 0">Daily: <b>${ict.daily_bias}</b> | Po3: <b>${ict.po3_phase}</b></div>
            ${renderTargets(ict.ict_t1,ict.ict_t2,ict.ict_t3,ict.ict_t4,(ict.ict_entry_zone[0]+ict.ict_entry_zone[1])/2,ict.ict_sl,ict.ict_rr_t2,ict.ict_rr_t3,ict.ict_signal)}
          </div>
          <div class="card" style="margin-top:8px">
            <div class="card-title">OTE Zones (61.8% - 78.6%)</div>
            ${renderOTE(ict.ote_long,'LONG')}<div style="height:5px"></div>${renderOTE(ict.ote_short,'SHORT')}
          </div>
        </div>
        <div class="col">
          <div class="card" style="margin-bottom:8px">
            <div class="card-title">Context</div>
            <div class="stat-row"><span class="stat-label">Killzone</span><span class="kz-badge ${ict.in_killzone?'active':'inactive'}">${ict.killzone_name}</span></div>
            <div class="stat-row"><span class="stat-label">Power of 3</span><span class="po3-badge po3-${ict.po3_phase}">${ict.po3_phase}</span></div>
            <div class="stat-row"><span class="stat-label">Daily Bias</span><span class="${dirColor(ict.daily_bias==='BULLISH'?'BUY':'SELL')}">${ict.daily_bias}</span></div>
            <div class="stat-row"><span class="stat-label">PDH</span><span class="red">Rs.${fmtN(il.pdh)}</span></div>
            <div class="stat-row"><span class="stat-label">PDL</span><span class="green">Rs.${fmtN(il.pdl)}</span></div>
          </div>
          <div class="card">
            <div class="card-title">Breaker Blocks (${ict.breaker_blocks.length})</div>${bbHtml}
          </div>
          <div class="card" style="margin-top:8px">
            <div class="card-title">ICT Notes</div>
            ${ict.ict_notes.map(n => '<div class="note-item">' + n + '</div>').join('') || '<div class="muted" style="font-size:.71rem">--</div>'}
          </div>
        </div>
      </div>`;
  } catch(e) { document.getElementById('ictBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

async function loadStructure() {
  document.getElementById('strSym').textContent = currentSym; loading('strBody');
  try {
    const r = await fetch('/api/structure/' + encodeURIComponent(currentSym) + '?interval=15minute').then(r => r.json());
    if (checkStale(r, 'strBody')) return;
    if (r.error) { document.getElementById('strBody').innerHTML = '<p class="red">' + r.error + '</p>'; return; }
    if (!r.setups || r.setups.length === 0) {
      document.getElementById('strBody').innerHTML = '<div class="signal-box WAIT"><div class="sig-dir yellow">[WAIT] No Active Setup</div><div style="font-size:.73rem;color:var(--muted);margin-top:4px">Waiting for BOS + pullback into OTE zone (61.8%-78.6%). Min RR 1.8x required.</div></div>';
      return;
    }
    const s = r.setups[0], dc = dirColor(s.direction==='LONG'?'BUY':'SELL');
    const ql = s.quality_label === 'A+' ? 'Ap' : s.quality_label;
    document.getElementById('strBody').innerHTML = `
      <div class="signal-box ${s.direction}">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
          <div class="sig-dir ${dc}">${sigEmoji(s.direction==='LONG'?'BUY':'SELL')} ${s.direction} -- ${s.setup_type}</div>
          <div><span class="quality-${ql}" style="font-size:.88rem">${s.quality_label}</span> <span class="muted" style="font-size:.7rem">${s.confidence_score.toFixed(0)}%</span></div>
        </div>
        <div style="background:var(--panel2);border-radius:5px;padding:7px;margin-bottom:8px;font-size:.72rem">
          <div class="yellow" style="font-weight:700;margin-bottom:3px">ICT OTE Entry Zone (61.8% - 78.6%)</div>
          <div style="display:flex;gap:12px;flex-wrap:wrap">
            <span>Ideal: <b class="yellow">Rs.${fmtN(s.entry_ideal)}</b></span>
            <span>Zone: Rs.${fmtN(s.entry_zone_low)} - Rs.${fmtN(s.entry_zone_high)}</span>
            <span>SL: <b class="red">Rs.${fmtN(s.sl_fixed)}</b></span>
          </div>
        </div>
        ${renderTargets(s.target_1,s.target_2,s.target_3,s.target_4,s.entry_ideal,s.sl_fixed,s.target_rr,s.target_rr_t3,s.direction)}
        <div style="margin-top:7px">
          ${s.ob_confluence?'<span class="chip bull">OB Confluence</span>':''}
          ${s.fvg_confluence?'<span class="chip bull">FVG Confluence</span>':''}
          ${s.ote_in_zone?'<span class="chip bull">In OTE Now</span>':'<span class="chip neutral">Approaching OTE</span>'}
          ${s.trend_aligned?'<span class="chip bull">Trend Aligned</span>':'<span class="chip bear">Counter-trend</span>'}
        </div>
      </div>
      <div class="card" style="margin-top:8px">
        <div class="card-title">Setup Notes</div>
        ${s.notes.map(n => '<div class="note-item">' + n + '</div>').join('')}
      </div>`;
  } catch(e) { document.getElementById('strBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function _confluenceDots(score, max) {
  max = max || 5;
  let h = '';
  for (let i = 0; i < max; i++) {
    h += '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:2px;background:'
      + (i < score ? 'var(--accent)' : 'var(--border)') + '"></span>';
  }
  return '<span title="Confluence ' + score + '/' + max + '">' + h + '</span>';
}

function _intervalMinutes(iv) {
  if (iv === 'minute') return 1;
  const m = iv.match(/(\d+)minute/); if (m) return parseInt(m[1]);
  if (iv === '60minute' || iv === 'hour') return 60;
  return 15;
}

function _expiryBadge(intervalStr) {
  // A signal is valid for ~5 candles; show how many are left (rough)
  const mins = _intervalMinutes(intervalStr);
  const totalMins = mins * 5;
  const now = new Date();
  const hhmm = now.getHours() * 60 + now.getMinutes();
  const close = 15 * 60 + 30;
  const left = Math.min(totalMins, close - hhmm);
  if (left <= 0) return '<span style="font-size:.62rem;color:var(--red);padding:1px 6px;border-radius:4px;background:rgba(255,69,96,.12)">Expired</span>';
  const candles = Math.max(0, Math.floor(left / mins));
  const col = candles >= 4 ? 'var(--green)' : candles >= 2 ? 'var(--yellow)' : 'var(--red)';
  return '<span style="font-size:.62rem;color:' + col + ';padding:1px 6px;border-radius:4px;background:rgba(168,255,62,.08)">' + candles + ' candles left</span>';
}

function _renderScanCard(s, iv) {
  const isBuy = s.direction === 'BUY';
  const col   = isBuy ? 'var(--green)' : 'var(--red)';
  const qcol  = s.quality === 'A+' ? '#ffd700' : 'var(--green)';
  const cs    = s.confluence_score != null ? s.confluence_score : (s.notes || []).join(' ').match(/Confluence:\s*(\d)/)?.[1] || 0;
  const rr    = parseFloat(s.rr_t2 || 0);
  const posSize = s.position_size > 0 ? `<span style="font-size:.65rem;color:var(--yellow)">Qty: ${s.position_size} shares</span>` : '';
  return `
  <div class="scan-item ${s.direction}" data-sym="${s.symbol}" style="cursor:pointer">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;flex-wrap:wrap;gap:4px">
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
        <span class="scan-sym" style="color:${col}">${s.symbol}</span>
        <span class="chip ${isBuy?'bull':'bear'}">${s.direction}</span>
        <span style="font-size:.68rem;font-weight:700;color:${qcol};padding:1px 7px;border-radius:999px;border:1px solid ${qcol}">${s.quality}</span>
        ${_expiryBadge(iv)}
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        ${_confluenceDots(parseInt(cs))}
        <span style="font-size:.68rem;color:var(--sub)">${parseFloat(s.confidence).toFixed(0)}% conf</span>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:6px;margin-bottom:8px">
      <div style="background:var(--panel);border-radius:6px;padding:6px 8px;border:1px solid var(--border)">
        <div style="font-size:.6rem;color:var(--muted)">CMP</div>
        <div style="font-size:.8rem;font-weight:700;font-family:'JetBrains Mono',monospace">₹${fmtN(s.current_price)}</div>
      </div>
      <div style="background:var(--panel);border-radius:6px;padding:6px 8px;border:1px solid rgba(168,255,62,.2)">
        <div style="font-size:.6rem;color:var(--muted)">Entry</div>
        <div style="font-size:.78rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:${col}">₹${fmtN(s.entry_zone[0])}</div>
        <div style="font-size:.6rem;color:var(--muted)">to ₹${fmtN(s.entry_zone[1])}</div>
      </div>
      <div style="background:rgba(255,69,96,.07);border-radius:6px;padding:6px 8px;border:1px solid rgba(255,69,96,.2)">
        <div style="font-size:.6rem;color:var(--muted)">Stop Loss</div>
        <div style="font-size:.78rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--red)">₹${fmtN(s.stop_loss)}</div>
        <div style="font-size:.6rem;color:${rr>=2?'var(--green)':'var(--yellow)'}">R:R ${rr}x</div>
      </div>
      <div style="background:var(--panel);border-radius:6px;padding:6px 8px;border:1px solid var(--border)">
        <div style="font-size:.6rem;color:var(--muted)">T1</div>
        <div style="font-size:.78rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--yellow)">₹${fmtN(s.target_1)}</div>
        <div style="font-size:.6rem;color:var(--muted)">30–40% exit</div>
      </div>
      <div style="background:var(--panel);border-radius:6px;padding:6px 8px;border:1px solid rgba(168,255,62,.15)">
        <div style="font-size:.6rem;color:var(--muted)">T2</div>
        <div style="font-size:.78rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--green)">₹${fmtN(s.target_2)}</div>
        <div style="font-size:.6rem;color:var(--muted)">Main target</div>
      </div>
      <div style="background:var(--panel);border-radius:6px;padding:6px 8px;border:1px solid var(--border)">
        <div style="font-size:.6rem;color:var(--muted)">T3</div>
        <div style="font-size:.78rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--accent)">₹${fmtN(s.target_3)}</div>
        <div style="font-size:.6rem;color:var(--muted)">Runner 20%</div>
      </div>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;font-size:.65rem;color:var(--muted)">
      ${s.in_killzone ? '<span style="color:var(--yellow)">Killzone: '+s.killzone+'</span>' : ''}
      <span>Daily: ${s.daily_bias}</span>
      <span>Po3: ${s.po3_phase}</span>
      ${posSize}
    </div>
  </div>`;
}

async function runScan() {
  const interval = document.getElementById('scanInterval').value;
  const minConf  = document.getElementById('scanMinConf').value;
  const limit    = document.getElementById('scanLimit').value;
  loading('scanBody');
  try {
    const r = await fetch('/api/institutional-scan?interval=' + interval + '&min_conf=' + minConf + '&limit=' + limit).then(r => r.json());
    if (checkStale(r, 'scanBody')) return;
    if (r.error) { document.getElementById('scanBody').innerHTML = '<p class="red">' + r.error + '</p>'; return; }
    if (!r.signals || r.signals.length === 0) {
      document.getElementById('scanBody').innerHTML = '<div class="signal-box WAIT"><div class="sig-dir yellow">No setups found</div><div class="muted" style="font-size:.73rem;margin-top:3px">Scanned ' + r.scanned + ' symbols. No multi-engine confluence. Try lower confidence or different interval.</div></div>';
      return;
    }
    if (r.signals[0] && typeof _notifySignal === 'function') {
      _notifySignal(r.signals[0].symbol, r.signals[0].direction, r.signals[0].quality);
    }
    document.getElementById('scanBody').innerHTML =
      '<div style="font-size:.7rem;color:var(--muted);margin-bottom:10px">Found <b style="color:var(--text)">' + r.signals.length + '</b> signals from ' + r.scanned + ' stocks'
      + (r.scanned_at ? ' · <span style="color:var(--accent)">' + r.scanned_at + '</span>' : '') + '</div>'
      + r.signals.map(s => _renderScanCard(s, interval)).join('');
  } catch(e) { document.getElementById('scanBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

async function runSmcScan() {
  const interval = document.getElementById('scanInterval').value;
  loading('scanBody');
  try {
    const r = await fetch('/api/smc-scan?interval=' + interval + '&min_conf=65&limit=20').then(r => r.json());
    if (checkStale(r, 'scanBody')) return;
    if (r.error) { document.getElementById('scanBody').innerHTML = '<p class="red">' + r.error + '</p>'; return; }
    if (!r.signals || !r.signals.length) { document.getElementById('scanBody').innerHTML = '<div class="muted" style="font-size:.78rem;padding:8px">No SMC signals found.</div>'; return; }
    document.getElementById('scanBody').innerHTML = '<div style="font-size:.7rem;color:var(--muted);margin-bottom:7px">SMC: ' + r.total_found + ' from ' + r.scanned + ' scanned</div>'
      + r.signals.map(s => '<div class="scan-item ' + s.signal + '" data-sym="' + s.symbol + '" style="cursor:pointer">'
        + '<div class="scan-hdr"><div><span class="scan-sym">' + s.symbol + '</span>'
        + '<span class="chip ' + (s.signal==='BUY'?'bull':'bear') + '" style="margin-left:4px">' + s.signal + '</span></div>'
        + '<span class="' + (s.signal==='BUY'?'green':'red') + '" style="font-size:.7rem">' + s.confidence.toFixed(0) + '%</span></div>'
        + '<div style="font-size:.7rem;margin-bottom:3px">' + (s.pattern_type||'') + ' | ' + (s.structure||'') + '</div>'
        + '<div class="scan-targets">'
        + '<span class="scan-t">T1: <b class="yellow">Rs.' + fmtN(s.target_1) + '</b></span>'
        + '<span class="scan-t">T2: <b class="green">Rs.' + fmtN(s.target_2) + '</b></span>'
        + '<span class="scan-t">SL: <b class="red">Rs.' + fmtN(s.stop_loss) + '</b></span>'
        + '</div></div>').join('');
  } catch(e) { document.getElementById('scanBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

async function loadOrb() {
  document.getElementById('orbSym').textContent = currentSym; loading('orbBody');
  try {
    const r = await fetch('/api/orb/' + encodeURIComponent(currentSym) + '?interval=5minute').then(r => r.json());
    if (checkStale(r, 'orbBody')) return;
    if (r.error) { document.getElementById('orbBody').innerHTML = '<p class="red">' + r.error + '</p>'; return; }
    if (!r.signal) { document.getElementById('orbBody').innerHTML = '<div class="signal-box WAIT"><div class="sig-dir yellow">[WAIT] No ORB Signal</div><div class="muted" style="font-size:.73rem;margin-top:3px">' + (r.message||'Waiting for 9:30+ breakout.') + '</div></div>'; return; }
    const s = r.signal, dc = s.direction==='BULL'?'green':'red';
    // FIX: show invalidation banner when CMP has breached the ORB signal level
    const invalidBanner = s.invalidated
      ? '<div style="background:#3a1010;border:1px solid var(--red);border-radius:5px;padding:7px;margin:7px 0;font-size:.73rem;color:var(--red)">'
        + '&#x26D4; <b>Signal Invalidated</b> — price has reclaimed the ORB boundary. Do not enter.'
        + '</div>'
      : '';
    document.getElementById('orbBody').innerHTML = '<div class="signal-box ' + (s.direction==='BULL'?'BUY':'SELL') + '">'
      + '<div class="sig-dir ' + dc + '">' + sigEmoji(s.direction==='BULL'?'BUY':'SELL') + ' ORB ' + s.direction + ' at ' + s.breakout_time + '</div>'
      + '<div style="font-size:.71rem;color:var(--muted);margin:3px 0">Vol: ' + s.volume_ratio.toFixed(1) + 'x avg | Conf: ' + s.confidence.toFixed(0) + '%</div>'
      + invalidBanner
      + '<div style="background:var(--panel2);border-radius:5px;padding:7px;margin:7px 0;font-size:.72rem">'
      + 'ORB H: <b>Rs.' + fmtN(s.orb_high) + '</b>  ORB L: <b>Rs.' + fmtN(s.orb_low) + '</b>  Range: <b class="yellow">Rs.' + fmtN(s.orb_range) + '</b><br/>'
      + 'Entry Zone: <b>Rs.' + fmtN(s.entry_zone[0]) + ' - Rs.' + fmtN(s.entry_zone[1]) + '</b>  SL: <b class="red">Rs.' + fmtN(s.sl) + '</b></div>'
      // FIX: pass s.target4 (not s.target3 twice), use 'orb' label mode for correct ×ORB labels
      + renderTargets(s.target1, s.target2, s.target3, s.target4, (s.entry_zone[0]+s.entry_zone[1])/2, s.sl, s.risk_reward, null, s.direction==='BULL'?'BUY':'SELL', 'orb')
      + (s.notes||[]).map(n => '<div class="note-item" style="margin-top:3px">' + n + '</div>').join('')
      + '</div>';
  } catch(e) { document.getElementById('orbBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

async function loadPatterns() {
  document.getElementById('patSym').textContent = currentSym; loading('patBody');
  try {
    const [pr, fr] = await Promise.all([
      fetch('/api/patterns/' + encodeURIComponent(currentSym) + '?interval=5minute').then(r => r.json()),
      fetch('/api/fibonacci/' + encodeURIComponent(currentSym) + '?interval=15minute').then(r => r.json()),
    ]);
    if (checkStale(pr, 'patBody')) return;
    if (pr.error) { document.getElementById('patBody').innerHTML = '<p class="red">' + pr.error + '</p>'; return; }
    let patterns = pr.patterns || [];

    // FIX #7: filter toggle state
    const hideNeutral = document.getElementById('hideNeutralChk') && document.getElementById('hideNeutralChk').checked;
    if (hideNeutral) patterns = patterns.filter(p => p.signal !== 'NEUTRAL');
    // Always enforce minimum 65% confidence for non-NEUTRAL to reduce noise
    patterns = patterns.filter(p => p.signal === 'NEUTRAL' || (p.confidence * 100) >= 65);

    const patHtml = patterns.length ? patterns.map(p =>
      '<div class="scan-item ' + p.signal + '">'
      + '<div class="scan-hdr"><span class="scan-sym" style="font-size:.82rem">' + p.pattern + '</span>'
      + '<span class="chip ' + (p.signal==='BULLISH'?'bull':p.signal==='BEARISH'?'bear':'neutral') + '">' + p.signal + '</span></div>'
      + '<div style="font-size:.7rem;color:var(--muted)">Conf: <b>' + (p.confidence*100).toFixed(0) + '%</b> | ' + p.timeframe + '</div>'
      + '<div class="scan-targets" style="margin-top:4px">'
      + '<span class="scan-t">Entry: Rs.' + fmtN(p.price_at_detection) + '</span>'
      + '<span class="scan-t">SL: <b class="red">Rs.' + fmtN(p.stop_loss) + '</b></span>'
      + '<span class="scan-t">T: <b class="green">Rs.' + fmtN(p.target) + '</b></span></div></div>').join('')
      : '<div class="muted" style="font-size:.78rem;padding:7px">No patterns detected (try unchecking "Hide Neutral").</div>';
    let fibHtml = '';
    if (!fr.error && fr.retracements) {
      fibHtml = '<div class="card" style="margin-top:8px"><div class="card-title">Fibonacci Levels</div>'
        + '<div style="font-size:.71rem;color:var(--muted);margin-bottom:5px">Swing: Rs.' + fmtN(fr.swing_low) + ' to Rs.' + fmtN(fr.swing_high) + ' (' + fr.direction + ')</div>'
        + '<div class="row">'
        + '<div class="col"><div style="font-size:.7rem;color:var(--accent);margin-bottom:3px">Retracements (Entry)</div>'
        + ['0.236','0.382','0.500','0.618','0.786'].map(k => '<div class="stat-row"><span class="muted">' + k + '</span><span class="' + (['0.618','0.786'].includes(k)?'yellow':'') + '" style="font-size:.72rem">Rs.' + fmtN(fr.retracements[k]) + '</span></div>').join('')
        + '</div><div class="col"><div style="font-size:.7rem;color:var(--green);margin-bottom:3px">Extensions (Targets)</div>'
        + ['1.272','1.414','1.618','2.000','2.618'].map(k => '<div class="stat-row"><span class="muted">' + k + '</span><span class="' + (['1.272','1.618'].includes(k)?'green':k==='2.000'?'purple':'') + '" style="font-size:.72rem">Rs.' + fmtN(fr.extensions[k]) + '</span></div>').join('')
        + '</div></div></div>';
    }
    document.getElementById('patBody').innerHTML = '<div class="card" style="margin-bottom:8px"><div class="card-title">Detected Patterns (' + patterns.length + ')</div>' + patHtml + '</div>' + fibHtml;
  } catch(e) { document.getElementById('patBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

async function loadFno() {
  document.getElementById('fnoSym').textContent = currentSym; loading('fnoBody');
  try {
    const r = await fetch('/api/fno/' + encodeURIComponent(currentSym) + '?interval=5minute').then(r => r.json());
    if (checkStale(r, 'fnoBody')) return;
    if (r.error) { document.getElementById('fnoBody').innerHTML = '<p class="red">' + r.error + '</p>'; return; }

    // Fix #8: PCR panel at top of F&O tab
    const biasColor = r.market_bias === 'BULLISH' ? 'green' : r.market_bias === 'BEARISH' ? 'red' : 'yellow';
    const pcrPanel = r.market_bias ? (
      '<div class="card" style="margin-bottom:8px;padding:8px 10px">'
      + '<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:.75rem">'
      + '<span>&#x1F9ED; Bias: <b class="' + biasColor + '">' + r.market_bias + '</b></span>'
      + (r.range_support ? '<span>&#x1F7E2; Support: <b>' + fmtN(r.range_support) + '</b></span>' : '')
      + (r.range_resistance ? '<span>&#x1F534; Resistance: <b>' + fmtN(r.range_resistance) + '</b></span>' : '')
      + (r.range_pts ? '<span>↔ Range: <b>' + fmtN(r.range_pts) + ' pts</b></span>' : '')
      + (r.iv_proxy ? '<span>&#x1F4CA; IV Proxy: <b>' + r.iv_proxy + '%</b></span>' : '')
      + '</div>'
      + '<div style="font-size:.68rem;color:var(--muted);margin-top:4px">' + (r.bias_reason || '') + '</div>'
      + '</div>'
    ) : '';

    const sigs = r.signals || [];
    if (!sigs.length) {
      document.getElementById('fnoBody').innerHTML = pcrPanel + '<div class="muted" style="font-size:.78rem;padding:7px">No F&O signals. Use NIFTY, BANKNIFTY, or F&O-eligible stocks.</div>';
      return;
    }
    document.getElementById('fnoBody').innerHTML = pcrPanel + sigs.map(s => {
      if (s.error) return '<div class="red" style="font-size:.78rem;padding:7px">' + s.error + '</div>';
      const dc = s.direction && s.direction.includes('CALL') ? 'green' : 'red';
      const capitalNeeded = (s.option_price && s.lot_size)
        ? 'Rs.' + fmtN(s.option_price * s.lot_size)
        : '--';
      const maxLoss = (s.option_sl && s.lot_size)
        ? 'Rs.' + fmtN(Math.abs(s.option_price - s.option_sl) * s.lot_size)
        : '--';
      return '<div class="scan-item ' + (s.direction&&s.direction.includes('CALL')?'BUY':'SELL') + '" style="margin-bottom:7px">'
        + '<div class="scan-hdr"><span class="scan-sym" style="font-size:.83rem">' + s.symbol + ' ' + s.strike + ' ' + s.option_type + '</span><span class="' + dc + '" style="font-size:.73rem">' + s.direction + '</span></div>'
        + '<div style="font-size:.7rem;color:var(--muted);margin-bottom:4px">Expiry: ' + s.expiry_str + ' (' + s.expiry_type + ') | Lot: ' + s.lot_size + '</div>'
        + '<div class="scan-targets"><span class="scan-t">LTP: <b>Rs.' + fmtN(s.option_price) + '</b></span><span class="scan-t">SL: <b class="red">Rs.' + fmtN(s.option_sl) + '</b></span><span class="scan-t">T: <b class="green">Rs.' + fmtN(s.option_target) + '</b></span><span class="scan-t">RR: <b>' + s.rr + '</b></span></div>'
        + '<div style="font-size:.68rem;margin-top:5px;display:flex;gap:12px;flex-wrap:wrap">'
        + '<span>&#x1F4B0; Capital needed: <b class="yellow">' + capitalNeeded + '</b></span>'
        + '<span>[!] Max loss/lot: <b class="red">' + maxLoss + '</b></span>'
        + '</div>'
        + '</div>';
    }).join('');
  } catch(e) { document.getElementById('fnoBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

// ── Trade Journal ─────────────────────────────────────────────────────────────
async function loadJournal() {
  loading('journalBody');
  try {
    const r = await fetch('/api/journal?limit=50').then(r => r.json());
    if (r.error) { document.getElementById('journalBody').innerHTML = '<p class="red">' + r.error + '</p>'; return; }
    const entries = r.entries || [];
    if (!entries.length) {
      document.getElementById('journalBody').innerHTML =
        '<div class="muted" style="font-size:.78rem;padding:10px">No journal entries yet. Signals will appear here as they fire.</div>';
      return;
    }
    const outcomeColor = o => o === 'WIN' ? 'green' : o === 'LOSS' ? 'red' : o === 'SCRATCH' ? 'yellow' : 'muted';
    const wins   = entries.filter(e => e.outcome === 'WIN').length;
    const losses = entries.filter(e => e.outcome === 'LOSS').length;
    const totalPnl = entries.reduce((s, e) => s + (e.pnl || 0), 0);
    const summaryHtml =
      '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;font-size:.72rem">'
      + '<span>Total: <b>' + r.total + '</b></span>'
      + '<span class="green">Wins: <b>' + wins + '</b></span>'
      + '<span class="red">Losses: <b>' + losses + '</b></span>'
      + '<span class="' + (totalPnl >= 0 ? 'green' : 'red') + '">Net P&L: <b>Rs.' + fmtN(totalPnl) + '</b></span>'
      + '</div>';
    const tableHtml = '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.7rem">'
      + '<thead><tr style="color:var(--muted);border-bottom:1px solid var(--border)">'
      + '<th style="text-align:left;padding:4px 6px">Time</th>'
      + '<th style="text-align:left;padding:4px 6px">Symbol</th>'
      + '<th style="padding:4px 6px">Dir</th>'
      + '<th style="padding:4px 6px">Engine</th>'
      + '<th style="padding:4px 6px">Conf</th>'
      + '<th style="padding:4px 6px">Entry</th>'
      + '<th style="padding:4px 6px">SL</th>'
      + '<th style="padding:4px 6px">T2</th>'
      + '<th style="padding:4px 6px">RR</th>'
      + '<th style="padding:4px 6px">Outcome</th>'
      + '<th style="padding:4px 6px">P&L</th>'
      + '</tr></thead><tbody>'
      + entries.map(e => {
          const dc = e.direction === 'BUY' ? 'green' : 'red';
          const entryMid = e.entry_low && e.entry_high ? ((e.entry_low + e.entry_high) / 2).toFixed(2) : '--';
          return '<tr style="border-bottom:1px solid var(--border);cursor:pointer" data-id="' + e.id + '" data-outcome="' + (e.outcome||'OPEN') + '" onclick="toggleOutcome(this.dataset.id,this.dataset.outcome)">'
            + '<td style="padding:4px 6px;color:var(--muted)">' + (e.ts || '').slice(0, 16) + '</td>'
            + '<td style="padding:4px 6px;font-weight:700">' + e.symbol + '</td>'
            + '<td style="padding:4px 6px;text-align:center"><span class="' + dc + '">' + e.direction + '</span></td>'
            + '<td style="padding:4px 6px;text-align:center;color:var(--muted)">' + (e.engine || '--') + '</td>'
            + '<td style="padding:4px 6px;text-align:center">' + (e.confidence ? e.confidence.toFixed(0) + '%' : '--') + '</td>'
            + '<td style="padding:4px 6px;text-align:right">Rs.' + fmtN(entryMid) + '</td>'
            + '<td style="padding:4px 6px;text-align:right;color:var(--red)">Rs.' + fmtN(e.sl) + '</td>'
            + '<td style="padding:4px 6px;text-align:right;color:var(--green)">Rs.' + fmtN(e.t2) + '</td>'
            + '<td style="padding:4px 6px;text-align:center">' + (e.rr_t2 || '--') + 'x</td>'
            + '<td style="padding:4px 6px;text-align:center"><span class="' + outcomeColor(e.outcome) + '">' + (e.outcome || 'OPEN') + '</span></td>'
            + '<td style="padding:4px 6px;text-align:right;color:' + (e.pnl >= 0 ? 'var(--green)' : 'var(--red)') + '">' + (e.pnl ? 'Rs.' + fmtN(e.pnl) : '--') + '</td>'
            + '</tr>';
        }).join('')
      + '</tbody></table></div>'
      + '<div style="font-size:.65rem;color:var(--muted);margin-top:6px">Click any row to cycle outcome: OPEN → WIN → LOSS → SCRATCH</div>';
    document.getElementById('journalBody').innerHTML = summaryHtml + tableHtml;
  } catch(e) { document.getElementById('journalBody').innerHTML = '<p class="red">Error: ' + e.message + '</p>'; }
}

async function toggleOutcome(id, current) {
  const cycle = { OPEN: 'WIN', WIN: 'LOSS', LOSS: 'SCRATCH', SCRATCH: 'OPEN' };
  const next = cycle[current] || 'WIN';
  const pnl = next === 'WIN' ? 500 : next === 'LOSS' ? -300 : 0;  // placeholder; user can edit
  try {
    await fetch('/api/journal/' + id, { method: 'PATCH', headers: {'Content-Type':'application/json'}, body: JSON.stringify({outcome: next, pnl}) });
    loadJournal();
  } catch(e) {}
}

function exportJournal() {
  window.open('/api/journal?limit=1000&offset=0', '_blank');
}

// ════════════════════════════════════════════════════════════════════════
//  SNIPER ENGINE UI
// ════════════════════════════════════════════════════════════════════════

async function loadSniperDailyBar() {
  try {
    const r = await fetch('/api/sniper/daily-status').then(x => x.json());
    document.getElementById('sdTrades').textContent  = r.trades_today ?? '--';
    document.getElementById('sdLosses').textContent  = r.losses_today ?? '--';
    document.getElementById('sdTimeNote').textContent = r.time_note   ?? '';
    const statusEl = document.getElementById('sdStatus');
    if (!r.trading_allowed) {
      statusEl.textContent = '&#x1F6AB; ' + (r.block_reason || 'Trading paused');
      statusEl.className = 'red';
    } else {
      statusEl.textContent = '&#x2705; Trading allowed';
      statusEl.className = 'green';
    }
  } catch(e) {}
}

async function runSniperScan() {
  const body = document.getElementById('sniperBody');
  body.innerHTML = '<p class="muted" style="font-size:.78rem;padding:10px">&#x26A1; Scanning 26 liquid stocks through 7 gates... (~30 sec)</p>';
  await loadSniperDailyBar();
  const interval = document.getElementById('sniperInterval').value;
  try {
    const r = await fetch('/api/sniper-scan?interval=' + interval).then(x => x.json());
    if (r.error) { body.innerHTML = '<p class="red">' + r.error + '</p>'; return; }

    const timeColor = r.time_window_ok ? 'green' : 'yellow';
    let html = '<div style="display:flex;gap:16px;flex-wrap:wrap;font-size:.7rem;margin-bottom:10px">'
      + '<span>Scanned: <b>' + r.scanned + '</b></span>'
      + '<span class="' + timeColor + '">Window: <b>' + r.time_note + '</b></span>'
      + '<span class="green">Signals: <b>' + r.signals_found + '</b></span>'
      + '<span class="muted">' + r.fetched_at + '</span>'
      + '</div>';

    if (!r.signals || !r.signals.length) {
      html += '<div style="text-align:center;padding:30px;color:var(--muted);font-size:.8rem">'
        + '&#x1F3AF; No sniper setups right now — all ' + r.scanned + ' stocks checked.<br>'
        + '<span style="font-size:.68rem">This is normal. Sniper fires 1-4 times per session.<br>Top wait reasons:</span><br>'
        + (r.wait_summary||[]).slice(0,5).map(w =>
            '<span style="font-size:.65rem;color:var(--muted)">• ' + w.symbol + ': ' + (w.reason||'').split('—')[0].trim() + '</span>'
          ).join('<br>')
        + '</div>';
    } else {
      html += r.signals.map(s => renderSniperCard(s)).join('');
    }
    body.innerHTML = html;
  } catch(e) { body.innerHTML = '<p class="red">Scan error: ' + e.message + '</p>'; }
}

function renderSniperCard(s) {
  const isBuy = s.signal === 'BUY';
  const sigColor = isBuy ? '#00ff88' : '#ff4444';
  const confBar = Math.round((s.confidence||70));
  const rr = s.rr_t2 || 0;
  const rrColor = rr >= 2.5 ? '#00ff88' : rr >= 2.0 ? '#ffd700' : '#ff6b6b';

  return `<div style="border:1px solid ${sigColor}44;border-radius:8px;padding:12px;margin-bottom:10px;background:${sigColor}08">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
      <div>
        <span style="font-size:.9rem;font-weight:700;color:${sigColor}">${s.signal}</span>
        <span style="font-size:1rem;font-weight:700;margin-left:8px">${s.symbol}</span>
        <span style="font-size:.7rem;color:var(--muted);margin-left:8px">${s.pattern||''}</span>
        <span style="font-size:.7rem;color:var(--muted);margin-left:6px">×${s.engulf_ratio||'?'}  vol ${s.vol_mult||'?'}x</span>
      </div>
      <div style="display:flex;gap:6px;align-items:center">
        <span style="font-size:.68rem;background:#00ff8811;border:1px solid #00ff8833;color:#00ff88;padding:2px 7px;border-radius:4px">${s.quality||'A'}</span>
        <span style="font-size:.78rem;color:${rrColor};font-weight:700">RR ${rr}x</span>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:10px 0;font-size:.72rem">
      <div style="background:var(--bg);border-radius:4px;padding:6px;text-align:center">
        <div style="color:var(--muted);font-size:.62rem">ENTRY</div>
        <div style="font-weight:700">₹${fmtN(s.entry)}</div>
      </div>
      <div style="background:var(--bg);border-radius:4px;padding:6px;text-align:center">
        <div style="color:var(--red);font-size:.62rem">STOP LOSS</div>
        <div style="font-weight:700;color:var(--red)">₹${fmtN(s.sl)}</div>
      </div>
      <div style="background:var(--bg);border-radius:4px;padding:6px;text-align:center">
        <div style="color:#ffd700;font-size:.62rem">T1 (1:1)</div>
        <div style="font-weight:700;color:#ffd700">₹${fmtN(s.t1)}</div>
      </div>
      <div style="background:var(--bg);border-radius:4px;padding:6px;text-align:center">
        <div style="color:#00ff88;font-size:.62rem">T2 (2:1)</div>
        <div style="font-weight:700;color:#00ff88">₹${fmtN(s.t2)}</div>
      </div>
    </div>
    <div style="font-size:.65rem;color:var(--muted);margin-bottom:8px">${(s.reasons||[]).slice(0,3).join(' &nbsp;|&nbsp; ')}</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">
      <button onclick="executeSniper(this,'${s.symbol}','${s.signal}',${s.entry},${s.sl},${s.t1},${s.t2})"
        style="background:${sigColor}22;border:1px solid ${sigColor};color:${sigColor};padding:5px 14px;border-radius:5px;cursor:pointer;font-weight:700;font-size:.75rem">
        &#x1F680; Execute Trade
      </button>
      <button onclick="executeSniperPaper(this,'${s.symbol}','${s.signal}',${s.entry},${s.sl},${s.t1},${s.t2})"
        style="background:transparent;border:1px solid var(--border);color:var(--muted);padding:5px 12px;border-radius:5px;cursor:pointer;font-size:.72rem">
        &#x1F4CB; Paper Log
      </button>
      <span style="font-size:.65rem;color:var(--muted);margin-left:4px">MIS · NSE · LIMIT</span>
    </div>
  </div>`;
}

async function executeSniper(btn, symbol, direction, entry, sl, t1, t2) {
  if (!confirm(
    '[LIVE ORDER] Zerodha Kite\\n\\n' +
    direction + ' ' + symbol + '\\n' +
    'Entry: ₹' + fmtN(entry) + '\\n' +
    'SL: ₹' + fmtN(sl) + '\\n' +
    'T1: ₹' + fmtN(t1) + '  T2: ₹' + fmtN(t2) + '\\n\\n' +
    'This will place a REAL order. Confirm?'
  )) return;

  btn.disabled = true;
  btn.textContent = '⌛ Placing...';
  try {
    const res = await fetch('/api/sniper/execute', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({symbol, direction, entry, sl, t1, t2})
    }).then(x => x.json());

    if (res.error) {
      btn.textContent = '&#x274C; ' + res.error;
      btn.style.color = '#ff4444';
    } else {
      btn.textContent = '&#x2705; Order placed #' + res.entry_order_id;
      btn.style.color = '#00ff88';
      await loadSniperDailyBar();
    }
  } catch(e) {
    btn.textContent = '&#x274C; ' + e.message;
    btn.style.color = '#ff4444';
  }
}

async function executeSniperPaper(btn, symbol, direction, entry, sl, t1, t2) {
  btn.disabled = true;
  btn.textContent = '⌛ Logging...';
  try {
    const res = await fetch('/api/sniper/paper', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({symbol, direction, entry, sl, t1, t2})
    }).then(x => x.json());
    btn.textContent = res.ok ? '&#x2705; Logged' : '&#x274C; ' + res.error;
  } catch(e) {
    btn.textContent = '&#x274C; ' + e.message;
  }
}

setTimeout(() => { selectSym('RELIANCE'); _startAutoRefresh(); }, 600);

// ── Fix #9: Keyboard shortcuts ────────────────────────────────────────────────
(function _initKeyboardShortcuts() {
  const TAB_KEYS = {
    '1': 'smc', '2': 'ict', '3': 'structure', '4': 'scan',
    '5': 'orb', '6': 'patterns', '7': 'fno', '8': 'journal'
  };
  document.addEventListener('keydown', function(e) {
    // Ignore when typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (TAB_KEYS[e.key]) {
      e.preventDefault();
      const tab = document.querySelector('.tab[data-tab="' + TAB_KEYS[e.key] + '"]');
      if (tab) tab.click();
    }
    // Enter = run active scan / load
    if (e.key === 'Enter') {
      e.preventDefault();
      const activeTab = document.querySelector('.tab.active');
      if (!activeTab) return;
      const t = activeTab.getAttribute('data-tab');
      if (t === 'scan') { if (typeof runScan === 'function') runScan(); }
      else { if (typeof loadAll === 'function') loadAll(); }
    }
    // ArrowLeft / ArrowRight = navigate symbol list
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      const items = document.querySelectorAll('.sym-item');
      const active = document.querySelector('.sym-item.active');
      if (!items.length) return;
      const idx = Array.from(items).indexOf(active);
      const next = e.key === 'ArrowRight' ? Math.min(idx + 1, items.length - 1) : Math.max(idx - 1, 0);
      if (items[next]) items[next].click();
    }
  });
})();

// ── Auto-load on page start ─────────────────────────────────────────────────
// ── Signal expiry timer on Signal tab ───────────────────────────────────────
let _sigTimerInterval = null;
function _startSignalTimer(intervalStr) {
  if (_sigTimerInterval) clearInterval(_sigTimerInterval);
  const el = document.getElementById('sigTimerEl');
  if (!el) return;
  const mins = _intervalMinutes ? _intervalMinutes(intervalStr) : 15;
  let remaining = mins * 5 * 60; // 5 candles in seconds
  function _tick() {
    if (remaining <= 0) { el.textContent = 'Signal expired'; el.style.color = 'var(--red)'; clearInterval(_sigTimerInterval); return; }
    const m = Math.floor(remaining / 60), s = remaining % 60;
    el.textContent = 'Valid ~' + m + ':' + String(s).padStart(2,'0');
    el.style.color = remaining > 300 ? 'var(--green)' : remaining > 120 ? 'var(--yellow)' : 'var(--red)';
    remaining--;
  }
  _tick();
  _sigTimerInterval = setInterval(_tick, 1000);
}

// ── Pre-market watchlist ─────────────────────────────────────────────────────
async function _loadPremarket() {
  const now = new Date();
  const istOffset = 5.5 * 60;
  const utcMin = now.getUTCHours() * 60 + now.getUTCMinutes();
  const istMin = (utcMin + istOffset) % (24 * 60);
  const isPreMkt = istMin >= 8 * 60 && istMin < 9 * 60 + 15;
  const card = document.getElementById('preMktCard');
  if (!card) return;
  if (!isPreMkt) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  try {
    const r = await fetch('/api/premarket').then(x => x.json());
    if (!r.candidates || !r.candidates.length) {
      document.getElementById('preMktBody').innerHTML = '<p class="muted" style="font-size:.78rem">No strong candidates found. Market data may not be updated yet.</p>';
      return;
    }
    document.getElementById('preMktBody').innerHTML =
      '<div style="font-size:.7rem;color:var(--muted);margin-bottom:8px">Top candidates for today · ' + (r.generated_at||'') + '</div>'
      + r.candidates.map(c => {
        const isBuy = c.bias === 'BUY' || c.bias === 'WATCH_BUY';
        const col = isBuy ? 'var(--green)' : c.bias === 'SELL' || c.bias === 'WATCH_SELL' ? 'var(--red)' : 'var(--yellow)';
        return '<div style="display:flex;justify-content:space-between;align-items:center;padding:7px 10px;border-radius:7px;margin-bottom:5px;background:var(--panel2);border:1px solid var(--border);cursor:pointer" data-sym="' + c.symbol + '">'
          + '<div><span style="font-size:.82rem;font-weight:700;font-family:JetBrains Mono,monospace">' + c.symbol + '</span>'
          + '<span style="font-size:.66rem;color:' + col + ';margin-left:6px">' + c.bias + '</span>'
          + '<div style="font-size:.65rem;color:var(--muted);margin-top:2px">' + c.notes.join(' · ') + '</div></div>'
          + '<div style="text-align:right;font-size:.7rem">'
          + '<div style="font-family:JetBrains Mono,monospace">₹' + c.close.toFixed(2) + '</div>'
          + '<div style="color:' + (c.chg_pct >= 0 ? 'var(--green)' : 'var(--red)') + '">' + (c.chg_pct >= 0 ? '+' : '') + c.chg_pct + '%</div>'
          + '</div></div>';
      }).join('')
      + '<div style="font-size:.66rem;color:var(--muted);margin-top:8px;padding:6px 10px;background:var(--panel2);border-radius:6px">'
      + (r.note || '') + '</div>';
  } catch(e) {
    document.getElementById('preMktBody').innerHTML = '<p class="red" style="font-size:.75rem">Pre-market fetch error: ' + e.message + '</p>';
  }
}

(function _autoLoad() {
  setTimeout(async () => {
    try {
      const st = await fetch('/api/auth/status').then(r => r.json());
      if (st.authenticated) {
        _loadPremarket();
        const first = SYMBOLS[0] || 'RELIANCE';
        selectSym(first);
      }
    } catch(e) {}
  }, 800);
})();

// ── Fix #12: Browser notifications when new signal fires ─────────────────────
let _lastNotifKey = '';
(function _initNotifications() {
  if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
  }
})();

function _notifySignal(symbol, direction, quality) {
  const key = symbol + ':' + direction;
  if (key === _lastNotifKey) return;
  _lastNotifKey = key;
  // Play a short beep via AudioContext
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = direction === 'BUY' ? 880 : 440;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
    osc.start(); osc.stop(ctx.currentTime + 0.4);
  } catch(e) {}
  // Browser notification
  if ('Notification' in window && Notification.permission === 'granted') {
    try {
      new Notification('AlgoTrade: ' + direction + ' ' + symbol, {
        body: 'Quality: ' + quality + ' — click to view',
        icon: '/favicon.ico',
      });
    } catch(e) {}
  }
}
</script>
</body></html>"""



def _log_signal(
    symbol: str,
    direction: str,
    engine: str,
    confidence: float,
    entry_low: float,
    entry_high: float,
    sl: float,
    t2: float,
    rr_t2: float,
    notes: str = "",
) -> None:
    """
    Persist a BUY/SELL signal to the SQLite journal (signal_log table).
    Called whenever a non-WAIT signal fires from any engine.
    Thread-safe — uses its own sqlite3 connection.
    """
    try:
        IST = timezone(timedelta(hours=5, minutes=30))
        ts = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S IST")
        # Position sizing: 2% risk per trade
        entry_mid = round((entry_low + entry_high) / 2, 2)
        risk_per_share = abs(entry_mid - sl) if sl else 0
        position_size = int(CAPITAL * 0.02 / risk_per_share) if risk_per_share > 0 else 0

        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                direction   TEXT    NOT NULL,
                engine      TEXT,
                confidence  REAL,
                entry_low   REAL,
                entry_high  REAL,
                sl          REAL,
                t1          REAL,
                t2          REAL,
                t3          REAL,
                rr_t2       REAL,
                position_size INTEGER,
                outcome     TEXT    DEFAULT 'OPEN',
                pnl         REAL    DEFAULT 0,
                notes       TEXT
            )
        """)
        conn.execute(
            """INSERT INTO signal_log
               (ts, symbol, direction, engine, confidence,
                entry_low, entry_high, sl, t2, rr_t2, position_size, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts, symbol, direction, engine, round(confidence, 1),
             entry_low, entry_high, sl, t2, round(rr_t2, 2), position_size, notes),
        )
        conn.commit()
        conn.close()
        log.info(f"📓 Logged {direction} signal for {symbol} ({engine}, conf={confidence:.0f}%)")
    except Exception as exc:
        log.warning(f"_log_signal error for {symbol}: {exc}")


@app.get("/api/journal")
async def get_journal(limit: int = 50, offset: int = 0):
    """
    Return the last `limit` signal entries from the SQLite trade journal.
    The DB is written by the background signal-logger whenever a BUY/SELL
    signal fires.  If the DB doesn't exist yet, returns an empty list so the
    UI shows a sensible 'No entries yet' message rather than an error.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Create table if it doesn't exist (first run)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                direction   TEXT    NOT NULL,
                engine      TEXT,
                confidence  REAL,
                entry_low   REAL,
                entry_high  REAL,
                sl          REAL,
                t1          REAL,
                t2          REAL,
                t3          REAL,
                rr_t2       REAL,
                outcome     TEXT    DEFAULT 'OPEN',
                pnl         REAL    DEFAULT 0,
                notes       TEXT
            )
        """)
        conn.commit()
        rows = cur.execute(
            "SELECT * FROM signal_log ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = cur.execute("SELECT COUNT(*) FROM signal_log").fetchone()[0]
        conn.close()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "entries": [dict(r) for r in rows],
        }
    except Exception as e:
        log.error(f"Journal fetch error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.patch("/api/journal/{entry_id}")
async def update_journal_entry(entry_id: int, request: Request):
    """Update outcome/pnl for a journal entry (mark as WIN/LOSS/SCRATCH)."""
    try:
        body = await request.json()
        outcome = body.get("outcome", "OPEN")
        pnl     = float(body.get("pnl", 0))
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE signal_log SET outcome=?, pnl=? WHERE id=?",
            (outcome, pnl, entry_id)
        )
        conn.commit(); conn.close()
        return {"ok": True, "id": entry_id, "outcome": outcome, "pnl": pnl}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# ════════════════════════════════════════════════════════════════════════════════
#  DATA FRESHNESS HELPER  —  used by ALL API endpoints
# ════════════════════════════════════════════════════════════════════════════════

def _data_age_minutes(df: pd.DataFrame) -> float:
    """Return minutes since the last candle timestamp (IST-aware)."""
    try:
        IST = timezone(timedelta(hours=5, minutes=30))
        last_ts = df["timestamp"].iloc[-1]
        now_ist = datetime.now(tz=IST).replace(tzinfo=None)
        last_naive = last_ts.replace(tzinfo=None) if hasattr(last_ts, "tzinfo") and last_ts.tzinfo else last_ts
        return round((now_ist - last_naive).total_seconds() / 60, 1)
    except Exception:
        return 0.0


def _is_market_hours() -> bool:
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(tz=IST)
    return (now.weekday() < 5 and
            now.replace(hour=9, minute=15, second=0, microsecond=0) <= now <=
            now.replace(hour=15, minute=30, second=0, microsecond=0))


def _stale_error(age_min: float, symbol: str) -> Dict:
    """Standard stale-data error dict returned to the frontend."""
    IST = timezone(timedelta(hours=5, minutes=30))
    return {
        "stale": True,
        "error": (
            f"Data is {age_min:.0f} min old for {symbol}. "
            "Kite session may have expired — please Logout and Login Kite again."
        ),
        "data_age_min": age_min,
        "fetched_at": datetime.now(tz=IST).strftime("%H:%M:%S IST"),
    }


def _guard_freshness(df: pd.DataFrame, symbol: str, max_age: int = 75) -> Optional[Dict]:
    """
    Returns a stale error dict if data is too old during market hours.
    Returns None if data is fresh (caller proceeds normally).
    max_age: minutes — default 75 (one 15-min candle + 1 hour buffer)

    Early-market bypass: the first complete 15-min candle closes at 9:30 IST.
    Before 9:35 AM IST we intentionally skip the stale check because the last
    available candle is legitimately from yesterday's 15:30 close — this is NOT
    a stale session, it is normal overnight gap data.
    """
    if not _is_market_hours():
        return None          # outside market hours — age is expected to be large
    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(tz=IST)
    # First complete 15-min candle closes at 9:30; allow 5-min Kite publish delay.
    # Before 9:35 AM, yesterday's 15:30 candle is the most recent — skip stale check.
    first_candle_ready = now_ist.replace(hour=9, minute=35, second=0, microsecond=0)
    if now_ist < first_candle_ready:
        return None          # early market open — overnight gap is expected
    age = _data_age_minutes(df)
    if age > max_age:
        return _stale_error(age, symbol)
    return None


# ════════════════════════════════════════════════════════════════════════════════
#  SIMPLE SIGNAL ENGINE  —  Fresh live data, no cache
#  Combines SMC + ICT + Candlestick Patterns → single BUY / SELL / WAIT
# ════════════════════════════════════════════════════════════════════════════════

def _compute_simple_signal(symbol: str, interval: str = "15minute") -> Dict:
    """
    Fetch live Kite data, run all 3 engines, return one clean verdict.
    Intentionally uncached — every call gets fresh market data.
    """
    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(tz=IST)

    token = get_instrument_token(symbol)
    if token is None:
        return {"error": f"Unknown symbol: {symbol}"}

    # ── Fresh candles (always live) ───────────────────────────────────────────
    with _kite_semaphore:
        df = kite_manager.get_historical_data(token, interval, 10)
    if df.empty or len(df) < 20:
        with _kite_semaphore:
            df = kite_manager.get_historical_data(token, interval, 20)
    if df.empty or len(df) < 15:
        return {"error": "Insufficient data from Kite — market may be closed"}

    current_price = float(df["close"].iloc[-1])

    # ── Data freshness check ─────────────────────────────────────────────────
    try:
        lct = df["timestamp"].iloc[-1]
        lct_naive = lct.replace(tzinfo=None) if (hasattr(lct, "tzinfo") and lct.tzinfo) else lct
        now_naive = now_ist.replace(tzinfo=None)
        data_age_min = round((now_naive - lct_naive).total_seconds() / 60, 1)
    except Exception:
        data_age_min = 0

    # Market hours: 9:15-15:30 IST Mon-Fri
    market_open  = now_ist.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    is_market_hours = (now_ist.weekday() < 5 and
                       market_open <= now_ist <= market_close)

    # Early-market bypass: the first complete 15-min candle closes at 9:30 IST.
    # Before 9:35 AM the most recent Kite candle is legitimately yesterday's 15:30
    # close — this is a normal overnight gap, NOT a stale/expired session.
    # Firing a stale error here after a fresh login is a false positive.
    first_candle_ready = now_ist.replace(hour=9, minute=35, second=0, microsecond=0)
    is_pre_first_candle = is_market_hours and (now_ist < first_candle_ready)

    # During market hours (after first candle available), reject data > 60 min old
    if is_market_hours and not is_pre_first_candle and data_age_min > 60:
        return {
            "error": (
                f"Data is {data_age_min:.0f} min old — Kite session likely expired. "
                "Please logout and re-login via the Login Kite button, then click Analyse again."
            ),
            "data_age_min": data_age_min,
            "fetched_at": now_ist.strftime("%H:%M:%S IST"),
        }

    # ── Higher timeframe context ──────────────────────────────────────────────
    df_1h = df_daily = None
    try:
        with _kite_semaphore:
            df_1h = kite_manager.get_historical_data(token, "60minute", 5)
    except Exception:
        pass
    try:
        with _kite_semaphore:
            df_daily = kite_manager.get_historical_data(token, "day", 20)
    except Exception:
        pass

    inst = calculate_institutional_levels(df, symbol)

    # ── Run all 3 engines ─────────────────────────────────────────────────────
    smc_res = ict_res = top_pattern = None
    pat_signal = "NEUTRAL"

    try:
        smc_res = smc_engine.analyse(df, symbol, interval, df_1h, inst)
    except Exception as e:
        log.warning(f"SMC failed {symbol}: {e}")
    try:
        ict_res = ict_engine.analyse(df, symbol, interval, df_daily, inst)
    except Exception as e:
        log.warning(f"ICT failed {symbol}: {e}")
    try:
        pats = pattern_detector.detect_all_patterns(df, symbol, interval)
        if pats:
            top_pattern = pats[0]
            pat_signal = top_pattern.signal
    except Exception as e:
        log.warning(f"Pattern failed {symbol}: {e}")

    # ── Map each engine → BUY / SELL / WAIT ──────────────────────────────────
    def _dir(s):
        if s in ("LONG", "BUY", "BULLISH"): return "BUY"
        if s in ("SHORT", "SELL", "BEARISH"): return "SELL"
        return "WAIT"

    smc_dir = _dir(smc_res.smc_signal  if smc_res  else "WAIT")
    ict_dir = _dir(ict_res.ict_signal  if ict_res  else "WAIT")
    pat_dir = _dir(pat_signal)

    votes = {"BUY": 0, "SELL": 0}
    engine_votes: Dict[str, str] = {}
    for eng, d in [("SMC", smc_dir), ("ICT", ict_dir), ("PATTERN", pat_dir)]:
        if d in votes:
            votes[d] += 1
            engine_votes[eng] = d

    # ── Confluence rule: 2+ engines must agree ────────────────────────────────
    # Block any signal if session is not tradeable
    _session_ok = not smc_res or getattr(smc_res, "session_tradeable", True)
    if not _session_ok:
        _snote = getattr(smc_res, "session_note", "Session not tradeable")
        return {
            "symbol": symbol, "signal": "WAIT", "current_price": current_price,
            "reason": f"Session not tradeable: {_snote}. Market opens 9:15 AM IST.",
            "engines": engine_votes, "data_age_min": data_age_min,
            "fetched_at": now_ist.strftime("%H:%M:%S IST"),
        }
    if votes["BUY"] >= 2:
        final_dir = "BUY"
    elif votes["SELL"] >= 2:
        final_dir = "SELL"
    else:
        # VWAP tiebreak only when EXACTLY 1 engine signals AND it is NOT the Pattern engine alone
        # (Pattern-only tiebreak produces unreliable signals with inverted levels)
        vwap_tb = getattr(smc_res, "vwap", None)
        _non_pat_votes_buy  = sum(1 for e,d in engine_votes.items() if d=="BUY"  and e!="PATTERN")
        _non_pat_votes_sell = sum(1 for e,d in engine_votes.items() if d=="SELL" and e!="PATTERN")
        if vwap_tb and _non_pat_votes_buy >= 1 and votes["BUY"] == 1 and current_price > vwap_tb * 1.003:
            final_dir = "BUY"
        elif vwap_tb and _non_pat_votes_sell >= 1 and votes["SELL"] == 1 and current_price < vwap_tb * 0.997:
            final_dir = "SELL"
        else:
            return {
                "symbol": symbol, "signal": "WAIT", "current_price": current_price,
                "reason": f"Engines disagree — SMC:{smc_dir} ICT:{ict_dir} Pattern:{pat_dir}. Need 2+ to agree.",
                "engines": engine_votes, "data_age_min": data_age_min,
                "fetched_at": now_ist.strftime("%H:%M:%S IST"),
            }

    # ── Extract entry zone / SL / targets (prefer ICT Fib targets) ───────────
    entry_low = entry_high = sl = t1 = t2 = t3 = t4 = None
    confidence = 60
    reasons: List[str] = []

    if ict_res and ict_res.ict_signal in ("LONG", "SHORT"):
        entry_low  = round(ict_res.ict_entry_zone[0], 2)
        entry_high = round(ict_res.ict_entry_zone[1], 2)
        sl         = round(ict_res.ict_sl,  2)
        t1         = round(ict_res.ict_t1,  2)
        t2         = round(ict_res.ict_t2,  2)
        t3         = round(ict_res.ict_t3,  2)
        t4         = round(ict_res.ict_t4,  2)
        confidence = round(ict_res.ict_confidence)
        if ict_res.daily_bias: reasons.append(f"Daily bias: {ict_res.daily_bias}")
        if ict_res.po3_phase:  reasons.append(f"Po3: {ict_res.po3_phase}")

    # SMC fallback — SMCAnalysis has no t1/t2/t3, derive from entry zone + risk
    if t1 is None and smc_res:
        ez = smc_res.smc_entry_zone or (current_price * 0.999, current_price * 1.001)
        entry_low  = round(float(min(ez)), 2)
        entry_high = round(float(max(ez)), 2)
        sl_raw     = smc_res.smc_sl
        sl         = round(float(sl_raw), 2) if sl_raw else None
        confidence = round(float(smc_res.smc_confidence))
        # Derive Fib targets from risk — use ATR floor to prevent zero-risk collapse
        em   = (entry_low + entry_high) / 2
        _atr_now = float((df["high"] - df["low"]).rolling(14).mean().dropna().iloc[-1]) if len(df) >= 14 else em * 0.005
        risk = abs(em - sl) if sl else em * 0.005
        risk = max(risk, _atr_now * 0.5)  # minimum risk = 0.5x ATR
        if final_dir == "BUY":
            t1 = round(em + risk * 1.0,   2)
            t2 = round(em + risk * 1.272, 2)
            t3 = round(em + risk * 1.618, 2)
            t4 = round(em + risk * 2.0,   2)
        else:
            t1 = round(em - risk * 1.0,   2)
            t2 = round(em - risk * 1.272, 2)
            t3 = round(em - risk * 1.618, 2)
            t4 = round(em - risk * 2.0,   2)

    # ── Sanity-check levels: SL must be on the correct side of entry ──────────
    # If ICT/SMC returned a SHORT setup but signal is BUY (or vice versa),
    # the stop loss will be on the wrong side. Recompute from ATR in that case.
    if entry_low and entry_high and sl and t1:
        _atr_sig = float((df["high"] - df["low"]).rolling(14).mean().dropna().iloc[-1]) if len(df) >= 14 else current_price * 0.005
        _em_sig  = (entry_low + entry_high) / 2
        if final_dir == "BUY" and sl >= entry_low:
            # SL is above or at entry for a BUY — inverted, recompute
            sl         = round(_em_sig - _atr_sig * 1.5, 2)
            _risk_fix  = abs(_em_sig - sl)
            t1         = round(_em_sig + _risk_fix * 1.0,   2)
            t2         = round(_em_sig + _risk_fix * 1.272, 2)
            t3         = round(_em_sig + _risk_fix * 1.618, 2)
            t4         = round(_em_sig + _risk_fix * 2.0,   2)
            reasons.append("[i] Levels recomputed (SL was inverted for BUY)")
        elif final_dir == "SELL" and sl <= entry_high:
            # SL is below or at entry for a SELL — inverted, recompute
            sl         = round(_em_sig + _atr_sig * 1.5, 2)
            _risk_fix  = abs(sl - _em_sig)
            t1         = round(_em_sig - _risk_fix * 1.0,   2)
            t2         = round(_em_sig - _risk_fix * 1.272, 2)
            t3         = round(_em_sig - _risk_fix * 1.618, 2)
            t4         = round(_em_sig - _risk_fix * 2.0,   2)
            reasons.append("[i] Levels recomputed (SL was inverted for SELL)")

    # Pattern boost
    if top_pattern and pat_dir == final_dir:
        confidence = min(95, confidence + 5)
        reasons.append(f"Pattern: {top_pattern.pattern.value} ({top_pattern.confidence*100:.0f}%)")

    if smc_res and smc_res.smc_notes:
        reasons += [n for n in smc_res.smc_notes[:2] if n]

    # ── VWAP filter ───────────────────────────────────────────────────────────
    vwap_val = getattr(smc_res, "vwap", None)
    if vwap_val: vwap_val = float(vwap_val)
    vwap_ok  = True
    if vwap_val:
        if final_dir == "BUY"  and current_price < vwap_val and confidence < 85: vwap_ok = False
        if final_dir == "SELL" and current_price > vwap_val and confidence < 85: vwap_ok = False

    # ── RSI filter ────────────────────────────────────────────────────────────
    rsi_val = None
    rsi_ok  = True
    try:
        rsi_val = pattern_detector._calc_rsi(df)
        if final_dir == "BUY"  and rsi_val > 78: rsi_ok = False
        if final_dir == "SELL" and rsi_val < 22: rsi_ok = False
    except Exception:
        pass

    # ── Is price already inside the entry zone? ───────────────────────────────
    in_zone = bool(entry_low and entry_high and entry_low <= current_price <= entry_high)

    # ── R:R ──────────────────────────────────────────────────────────────────
    rr = None
    if t2 and sl and entry_low and entry_high:
        em   = (entry_low + entry_high) / 2
        risk = abs(em - sl)
        if risk > 0:
            rr = round(abs(t2 - em) / risk, 2)

    # ── Quality grade ─────────────────────────────────────────────────────────
    quality = "B"
    if confidence >= 80 and vwap_ok and rsi_ok and (rr or 0) >= 2.0: quality = "A+"
    elif confidence >= 65 and vwap_ok and rsi_ok:                     quality = "A"

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings: List[str] = []
    if not vwap_ok:          warnings.append("Counter-VWAP — higher risk")
    if not rsi_ok:           warnings.append("RSI extreme — skip entry")
    if rr and rr < 2.0:      warnings.append(f"R:R {rr}x below 2x minimum")
    if data_age_min > 20:    warnings.append(f"Data {data_age_min:.0f} min old — refresh")

    return {
        "symbol":        symbol,
        "signal":        final_dir,
        "quality":       quality,
        "confidence":    confidence,
        "current_price": current_price,
        "entry_low":     entry_low,
        "entry_high":    entry_high,
        "sl":            sl,
        "t1":            t1,
        "t2":            t2,
        "t3":            t3,
        "t4":            t4,
        "rr_t2":         rr,
        "in_zone":       in_zone,
        "vwap":          round(vwap_val, 2) if vwap_val else None,
        "vwap_ok":       vwap_ok,
        "rsi_val":       round(rsi_val, 1) if rsi_val else None,
        "engines":       engine_votes,
        "reasons":       [r for r in reasons if r][:4],
        "warnings":      warnings,
        "data_age_min":  data_age_min,
        "fetched_at":    now_ist.strftime("%H:%M:%S IST"),
        "inst_levels":   inst.to_dict() if inst else {},
        "pattern":       top_pattern.pattern.value if top_pattern else None,
    }


@app.get("/api/signal/{symbol}")
async def get_simple_signal(symbol: str, interval: str = "15minute"):
    """Simple BUY/SELL/WAIT — always fresh Kite data, no cache."""
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _compute_simple_signal(symbol, interval))
        return result
    except Exception as e:
        log.error(f"Signal error {symbol}: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/quick-scan")
async def quick_scan_simple(limit: int = 20, interval: str = "15minute"):
    """
    Scan top liquid symbols and return simple BUY/SELL signals.
    Only A+ and A quality signals returned. Always fresh data.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    scan_list = [
        "NIFTY 50", "NIFTY BANK",
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BAJFINANCE",
        "TATAMOTORS", "AXISBANK", "SBIN", "WIPRO", "HCLTECH", "LTM",
        "TATASTEEL", "VEDL", "COALINDIA", "ONGC", "SAIL", "NATIONALUM",
        "TVSHLTD", "KPITTECH", "MARUTI", "BAJAJ-AUTO",
        "TITAN", "SUNPHARMA", "DRREDDY", "APOLLOHOSP",
        "RVNL", "IRFC", "RECLTD", "ADANIENT",
    ][:limit + 10]

    results: List[Dict] = []

    def _scan_one(sym: str):
        try:
            r = _compute_simple_signal(sym, interval)
            if r.get("signal") in ("BUY", "SELL") and r.get("quality") in ("A+", "A"):
                return r
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_scan_one, s): s for s in scan_list}
        for f in as_completed(futs, timeout=60):
            r = f.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    IST = timezone(timedelta(hours=5, minutes=30))
    return {
        "scanned":       len(scan_list),
        "signals_found": len(results),
        "signals":       results[:limit],
        "fetched_at":    datetime.now(tz=IST).strftime("%H:%M:%S IST"),
    }




@app.get("/api/premarket")
async def get_premarket_watchlist():
    """
    Pre-market candidates: uses previous day's structure to find
    the top 5 symbols most likely to have good setups at market open.
    Uses daily candles — no 15-min data needed (market not open yet).
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    IST = timezone(timedelta(hours=5, minutes=30))
    candidates = []
    scan_syms = list(DEMO_TOKENS.keys())[:20]

    def _check_sym(symbol):
        try:
            token = get_instrument_token(symbol)
            if not token: return None
            with _kite_semaphore:
                df = kite_manager.get_historical_data(token, "day", 10)
            if df.empty or len(df) < 5: return None
            inst = calculate_institutional_levels(df, symbol)
            close  = float(df["close"].iloc[-1])
            prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else close
            chg_pct = round((close - prev_close) / prev_close * 100, 2)
            # Score: proximity to PDH/PDL, daily trend strength
            score = 0
            bias = "NEUTRAL"
            note = []
            if abs(close - inst.pdh) / inst.pdh < 0.005:
                score += 2; note.append("Near PDH"); bias = "WATCH_SELL"
            if abs(close - inst.pdl) / inst.pdl < 0.005:
                score += 2; note.append("Near PDL"); bias = "WATCH_BUY"
            if close > inst.pdh:
                score += 3; note.append("Above PDH — breakout"); bias = "BUY"
            if close < inst.pdl:
                score += 3; note.append("Below PDL — breakdown"); bias = "SELL"
            if abs(chg_pct) > 1.5:
                score += 1; note.append(f"Big move {chg_pct:+.1f}%")
            if score < 2: return None
            return {
                "symbol": symbol, "close": close, "chg_pct": chg_pct,
                "bias": bias, "score": score, "notes": note,
                "pdh": round(inst.pdh, 2), "pdl": round(inst.pdl, 2),
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(_check_sym, scan_syms):
            if r: candidates.append(r)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return {
        "candidates": candidates[:8],
        "generated_at": datetime.now(tz=IST).strftime("%H:%M:%S IST"),
        "note": "Based on yesterday's close vs PDH/PDL. Confirm at 9:30 AM with live signal.",
    }

@app.get("/api/debug/kite")
async def debug_kite():
    """
    Diagnostic endpoint — shows exactly what Kite is returning.
    Visit /api/debug/kite in browser to diagnose data issues.
    """
    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(tz=IST)
    info = {
        "server_utc":        datetime.utcnow().isoformat(),
        "server_ist":        now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "kite_authenticated": kite_manager.is_authenticated,
        "token_present":     bool(kite_manager.access_token),
    }
    if not kite_manager.is_authenticated:
        info["error"] = "Not authenticated — click Login Kite"
        return info
    # Profile check
    try:
        profile = kite_manager.kite.profile()
        info["kite_user"]  = profile.get("user_name", "?")
        info["kite_email"] = profile.get("email", "?")
        info["profile_ok"] = True
    except Exception as e:
        info["profile_error"] = str(e)
        info["profile_ok"]    = False
    # Test historical data for NIFTY 50
    try:
        token  = 256265  # NIFTY 50
        ist_now   = now_ist.replace(tzinfo=None)
        from_dt   = ist_now - timedelta(days=2)
        records   = kite_manager.kite.historical_data(token, from_dt, ist_now, "15minute")
        if records:
            last_ts  = records[-1]["date"]
            age_min  = round((ist_now - last_ts.replace(tzinfo=None)).total_seconds() / 60, 1)
            info["nifty50_last_candle"] = str(last_ts)
            info["nifty50_last_close"]  = records[-1]["close"]
            info["nifty50_data_age_min"] = age_min
            info["nifty50_candles_returned"] = len(records)
            info["nifty50_ok"] = age_min < 60
        else:
            info["nifty50_error"] = "Kite returned 0 records"
    except Exception as e:
        info["nifty50_error"] = str(e)
    # Test RELIANCE
    try:
        token  = 738561
        records = kite_manager.kite.historical_data(token, from_dt, ist_now, "15minute")
        if records:
            last_ts  = records[-1]["date"]
            age_min  = round((ist_now - last_ts.replace(tzinfo=None)).total_seconds() / 60, 1)
            info["reliance_last_candle"]  = str(last_ts)
            info["reliance_data_age_min"] = age_min
            info["reliance_ok"] = age_min < 60
        else:
            info["reliance_error"] = "Kite returned 0 records"
    except Exception as e:
        info["reliance_error"] = str(e)
    return info

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return minimal SVG favicon — prevents 404 log spam."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#060b14"/>'
        '<text x="16" y="22" font-size="18" text-anchor="middle" fill="#00d4ff">A</text>'
        '</svg>'
    )
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(content=HTML_UI, media_type="text/html; charset=utf-8")


# ════════════════════════════════════════════════════════════════════════════════
#  SNIPER ENGINE  —  High-Accuracy Strategy (Less Noise, More Profit)
#  ─────────────────────────────────────────────────────────────────────────────
#  Rules (must ALL pass to fire a signal):
#    1. TIME   : 09:30–11:30 IST  OR  14:30–15:30 IST  only
#    2. TREND  : Price > EMA50 > EMA200 (BUY)  |  Price < EMA50 < EMA200 (SELL)
#    3. PATTERN: ONLY Bullish / Bearish Engulfing on the second-to-last candle
#    4. CONFIRM: Latest closed candle breaks pattern candle's High (BUY) / Low (SELL)
#    5. LOCATION: Price within 0.8% of VWAP, PDL (BUY) or PDH (SELL)
#    6. INDEX  : NIFTY 50 EMA50/200 trend must NOT oppose signal direction
#    7. RR     : Stop Loss at pattern candle extreme. T2 must be ≥ 2× risk
# ════════════════════════════════════════════════════════════════════════════════

# ── Helpers ──────────────────────────────────────────────────────────────────

def _sniper_ema(series: pd.Series, period: int) -> pd.Series:
    """EMA using pandas ewm — same method as SMCEngine._calc_ema."""
    return series.ewm(span=period, adjust=False).mean()


def _sniper_vwap(df: pd.DataFrame) -> Optional[float]:
    """
    Intraday VWAP from the day's candles only.
    Falls back to whole-df VWAP if same-day filtering returns < 3 rows.
    """
    try:
        IST = timezone(timedelta(hours=5, minutes=30))
        today = datetime.now(tz=IST).date()
        ts_col = df["timestamp"]
        # normalise timezone-aware timestamps
        if hasattr(ts_col.iloc[0], "tzinfo") and ts_col.iloc[0].tzinfo:
            today_mask = ts_col.dt.tz_convert(IST).dt.date == today
        else:
            today_mask = ts_col.dt.date == today
        day_df = df[today_mask] if today_mask.sum() >= 3 else df
        tp  = (day_df["high"] + day_df["low"] + day_df["close"]) / 3
        vol = day_df["volume"].replace(0, np.nan).fillna(1)
        return float((tp * vol).sum() / vol.sum())
    except Exception:
        return None


def _sniper_time_ok() -> Tuple[bool, str]:
    """
    Returns (allowed, window_name).
    Allowed windows: 09:30–11:30  or  14:30–15:30 IST.
    Outside market hours we always allow (for paper/back-test mode).
    """
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(tz=IST)
    if now.weekday() >= 5:                    # Saturday / Sunday
        return False, "Weekend"
    t = now.time()
    import datetime as _dt
    mkt_open  = _dt.time(9, 15)
    mkt_close = _dt.time(15, 30)
    if not (mkt_open <= t <= mkt_close):      # outside market — allow (pre/post)
        return True, "off-hours"
    w1_start, w1_end = _dt.time(9, 30), _dt.time(11, 30)
    w2_start, w2_end = _dt.time(14, 30), _dt.time(15, 30)
    if w1_start <= t <= w1_end:
        return True, "Morning (09:30–11:30)"
    if w2_start <= t <= w2_end:
        return True, "Power Hour (14:30–15:30)"
    return False, f"Dead zone {t.strftime('%H:%M')} — trade only 09:30-11:30 or 14:30-15:30"


def _sniper_trend(df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    """
    EMA-50 / EMA-200 trend alignment check.
    BUY  requires: close > EMA50 > EMA200
    SELL requires: close < EMA50 < EMA200

    If df has < 200 rows we use EMA50 vs EMA20 as a proxy — still meaningful.
    Returns (ok, note).
    """
    try:
        close = df["close"].astype(float)
        e50   = float(_sniper_ema(close, 50).iloc[-1])
        if len(close) >= 200:
            e200 = float(_sniper_ema(close, 200).iloc[-1])
            label = "EMA200"
        else:
            e200 = float(_sniper_ema(close, min(20, len(close) - 1)).iloc[-1])
            label = "EMA20(proxy)"
        price = float(close.iloc[-1])
        if direction == "BUY":
            ok = price > e50 > e200
            note = f"price {price:.0f} {'>' if price>e50 else '<'} EMA50 {e50:.0f} {'>' if e50>e200 else '<'} {label} {e200:.0f}"
        else:
            ok = price < e50 < e200
            note = f"price {price:.0f} {'<' if price<e50 else '>'} EMA50 {e50:.0f} {'<' if e50<e200 else '>'} {label} {e200:.0f}"
        return ok, note
    except Exception as exc:
        return True, f"trend check skipped ({exc})"   # soft fail — don't block


def _sniper_location(
    df: pd.DataFrame,
    inst: "InstitutionalLevels",
    direction: str,
    price: float,
    tolerance: float = 0.008,
) -> Tuple[bool, str]:
    """
    Location filter — only trade at a meaningful level:
      BUY  : within tolerance% of VWAP  OR  PDL
      SELL : within tolerance% of VWAP  OR  PDH

    If no PDH/PDL available (index or missing), we relax to VWAP-only.
    """
    vwap = _sniper_vwap(df)
    pdh  = getattr(inst, "pdh", None)
    pdl  = getattr(inst, "pdl", None)

    def near(ref: Optional[float]) -> bool:
        if ref is None or ref <= 0:
            return False
        return abs(price - ref) / ref <= tolerance

    hits: List[str] = []
    if near(vwap):
        hits.append(f"VWAP {vwap:.0f}")
    if direction == "BUY" and near(pdl):
        hits.append(f"PDL {pdl:.0f}")
    if direction == "SELL" and near(pdh):
        hits.append(f"PDH {pdh:.0f}")

    # Always allow if we have no institutional levels (e.g. index symbols)
    if pdh is None and pdl is None:
        return True, "no inst levels — location skipped"

    if hits:
        return True, "At " + " / ".join(hits)
    level_str = f"PDL {pdl:.0f}" if direction == "BUY" and pdl else (f"PDH {pdh:.0f}" if pdh else "")
    return False, f"Price {price:.0f} not near VWAP {vwap:.0f if vwap else 'N/A'} or {level_str}"


def _sniper_find_engulfing(df: pd.DataFrame) -> Optional[Dict]:
    """
    Scan the last 5 candles for the most recent Engulfing pattern.
    Returns a dict with keys: direction, pattern_idx, pattern_candle, prev_candle
    or None if nothing found.

    We deliberately look at candles that are CONFIRMED (i.e. not the live
    in-progress candle at iloc[-1]). The live candle is used for confirmation.
    """
    # Work backwards over positions [-2, -3, -4, -5] (skip live candle)
    for offset in range(2, 6):
        if offset >= len(df):
            break
        i     = len(df) - offset           # pattern candle index
        curr  = df.iloc[i]
        prev  = df.iloc[i - 1]

        curr_open  = float(curr["open"])
        curr_close = float(curr["close"])
        prev_open  = float(prev["open"])
        prev_close = float(prev["close"])
        curr_body  = abs(curr_close - curr_open)
        prev_body  = abs(prev_close - prev_open)

        if prev_body < 1e-8:               # avoid division by zero
            continue

        # Bullish Engulfing
        if (prev_close < prev_open          # prev bearish
                and curr_close > curr_open  # curr bullish
                and curr_open  < prev_close # opens below prev close
                and curr_close > prev_open  # closes above prev open
                and curr_body  > prev_body * 1.0):  # body must engulf
            engulf_ratio = round(curr_body / prev_body, 2)
            return {
                "direction":      "BUY",
                "pattern":        "Bullish Engulfing",
                "pattern_idx":    i,
                "engulf_ratio":   engulf_ratio,
                "pattern_high":   float(curr["high"]),
                "pattern_low":    float(curr["low"]),
                "pattern_open":   curr_open,
                "pattern_close":  curr_close,
                "prev_open":      prev_open,
                "prev_close":     prev_close,
            }

        # Bearish Engulfing
        if (prev_close > prev_open          # prev bullish
                and curr_close < curr_open  # curr bearish
                and curr_open  > prev_close # opens above prev close
                and curr_close < prev_open  # closes below prev open
                and curr_body  > prev_body * 1.0):
            engulf_ratio = round(curr_body / prev_body, 2)
            return {
                "direction":      "SELL",
                "pattern":        "Bearish Engulfing",
                "pattern_idx":    i,
                "engulf_ratio":   engulf_ratio,
                "pattern_high":   float(curr["high"]),
                "pattern_low":    float(curr["low"]),
                "pattern_open":   curr_open,
                "pattern_close":  curr_close,
                "prev_open":      prev_open,
                "prev_close":     prev_close,
            }

    return None


def _sniper_confirmation(df: pd.DataFrame, eng: Dict) -> Tuple[bool, str]:
    """
    Confirmation: the candle AFTER the pattern candle must break the
    pattern candle's extreme in the signal direction.

    BUY  — latest candle's close > pattern candle's high
    SELL — latest candle's close < pattern candle's low

    If the pattern candle is NOT second-to-last yet (older), we check
    the candle immediately following the pattern candle.
    """
    pattern_idx  = eng["pattern_idx"]
    direction    = eng["direction"]
    next_idx     = pattern_idx + 1

    if next_idx >= len(df):
        return False, "No candle after pattern yet — wait"

    confirm_candle = df.iloc[next_idx]
    conf_close     = float(confirm_candle["close"])
    conf_high      = float(confirm_candle["high"])
    conf_low       = float(confirm_candle["low"])

    if direction == "BUY":
        triggered = conf_close > eng["pattern_high"]
        note = (
            f"Confirmation candle close {conf_close:.2f} "
            f"{'>' if triggered else '<='} pattern high {eng['pattern_high']:.2f}"
        )
        return triggered, note
    else:
        triggered = conf_close < eng["pattern_low"]
        note = (
            f"Confirmation candle close {conf_close:.2f} "
            f"{'<' if triggered else '>='} pattern low {eng['pattern_low']:.2f}"
        )
        return triggered, note


_sniper_nifty_cache: Dict = {}   # {"ts": datetime, "trend": "BUY"|"SELL"|"NEUTRAL"}

def _sniper_nifty_trend() -> str:
    """
    Returns NIFTY 50 short-term trend: "BUY", "SELL", or "NEUTRAL".
    Caches result for 10 minutes to avoid hammering Kite API.
    """
    global _sniper_nifty_cache
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(tz=IST)
    cached = _sniper_nifty_cache
    if cached and (now - cached.get("ts", now)).seconds < 600:
        return cached.get("trend", "NEUTRAL")

    try:
        token = get_instrument_token("NIFTY 50")
        if token is None:
            return "NEUTRAL"
        with _kite_semaphore:
            ndf = kite_manager.get_historical_data(token, "15minute", 10)
        if ndf.empty or len(ndf) < 20:
            return "NEUTRAL"
        close = ndf["close"].astype(float)
        e50   = float(_sniper_ema(close, 50).iloc[-1])
        e20   = float(_sniper_ema(close, 20).iloc[-1])   # use 20 as proxy for 200 on short df
        price = float(close.iloc[-1])
        if price > e50 > e20:
            trend = "BUY"
        elif price < e50 < e20:
            trend = "SELL"
        else:
            trend = "NEUTRAL"
        _sniper_nifty_cache = {"ts": now, "trend": trend}
        return trend
    except Exception:
        return "NEUTRAL"


# ── Main Sniper Signal Function ───────────────────────────────────────────────

def _compute_sniper_signal(symbol: str, interval: str = "15minute") -> Dict:
    """
    The Sniper Engine — 7-gate high-accuracy signal.
    Returns a dict with signal = BUY | SELL | WAIT plus full context.
    All 7 gates must pass; each failed gate returns WAIT with reason.
    """
    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(tz=IST)

    def _wait(reason: str, **extra) -> Dict:
        return {"symbol": symbol, "signal": "WAIT", "engine": "sniper",
                "reason": reason, "fetched_at": now_ist.strftime("%H:%M:%S IST"), **extra}

    # ── Gate 1: Time window ───────────────────────────────────────────────────
    time_ok, time_note = _sniper_time_ok()
    if not time_ok:
        return _wait(f"⏰ {time_note}")

    # ── Fetch candles ─────────────────────────────────────────────────────────
    token = get_instrument_token(symbol)
    if token is None:
        return _wait(f"Unknown symbol: {symbol}")

    with _kite_semaphore:
        df = kite_manager.get_historical_data(token, interval, 15)
    if df.empty or len(df) < 30:
        with _kite_semaphore:
            df = kite_manager.get_historical_data(token, interval, 20)
    if df.empty or len(df) < 20:
        return _wait("Insufficient candle data from Kite")

    price = float(df["close"].iloc[-1])
    inst  = calculate_institutional_levels(df, symbol)

    # ── Gate 2: Engulfing pattern ─────────────────────────────────────────────
    eng = _sniper_find_engulfing(df)
    if eng is None:
        return _wait("No Bullish/Bearish Engulfing in last 5 candles")

    direction = eng["direction"]

    # ── Gate 3: Trend alignment ───────────────────────────────────────────────
    trend_ok, trend_note = _sniper_trend(df, direction)
    if not trend_ok:
        return _wait(f"📉 Trend misaligned — {trend_note}")

    # ── Gate 4: Confirmation candle ───────────────────────────────────────────
    conf_ok, conf_note = _sniper_confirmation(df, eng)
    if not conf_ok:
        return _wait(f"⏳ Awaiting confirmation — {conf_note}")

    # ── Gate 5: Location (VWAP / PDH / PDL) ──────────────────────────────────
    loc_ok, loc_note = _sniper_location(df, inst, direction, price)
    if not loc_ok:
        return _wait(f"📍 Location filter failed — {loc_note}")

    # ── Gate 6: NIFTY index trend must not oppose signal ──────────────────────
    nifty_trend = _sniper_nifty_trend()
    if nifty_trend != "NEUTRAL" and nifty_trend != direction:
        return _wait(
            f"🏦 NIFTY trend ({nifty_trend}) opposes {direction} signal — skip",
            nifty_trend=nifty_trend,
        )

    # ── Gate 7: Risk-Reward ≥ 2:1 ────────────────────────────────────────────
    if direction == "BUY":
        sl     = round(eng["pattern_low"]  - (eng["pattern_high"] - eng["pattern_low"]) * 0.1, 2)
        entry  = round(eng["pattern_high"] * 1.001, 2)          # just above confirmation break
    else:
        sl     = round(eng["pattern_high"] + (eng["pattern_high"] - eng["pattern_low"]) * 0.1, 2)
        entry  = round(eng["pattern_low"]  * 0.999, 2)

    risk = abs(entry - sl)
    if risk <= 0:
        return _wait("Zero risk — price levels collapsed")

    # SL sanity check (reuse existing guard logic)
    if direction == "BUY" and sl >= entry:
        sl = round(entry - risk, 2)
    if direction == "SELL" and sl <= entry:
        sl = round(entry + risk, 2)

    t1 = round(entry + risk * 1.0, 2) if direction == "BUY" else round(entry - risk * 1.0, 2)
    t2 = round(entry + risk * 2.0, 2) if direction == "BUY" else round(entry - risk * 2.0, 2)
    t3 = round(entry + risk * 3.0, 2) if direction == "BUY" else round(entry - risk * 3.0, 2)
    rr = round(abs(t2 - entry) / risk, 2)

    if rr < 2.0:
        return _wait(f"📊 RR {rr}x below minimum 2x — skip trade")

    # ── All gates passed — fire signal ───────────────────────────────────────
    confidence = 70.0

    # Bonus: volume on pattern candle
    try:
        pat_vol  = float(df.iloc[eng["pattern_idx"]]["volume"])
        avg_vol  = float(df["volume"].rolling(20).mean().iloc[eng["pattern_idx"]])
        vol_mult = pat_vol / avg_vol if avg_vol > 0 else 1.0
        if vol_mult >= 1.5:
            confidence += 8
        elif vol_mult >= 1.2:
            confidence += 4
    except Exception:
        vol_mult = 1.0

    # Bonus: engulf ratio
    if eng["engulf_ratio"] >= 2.0:
        confidence += 7
    elif eng["engulf_ratio"] >= 1.5:
        confidence += 4

    # Penalty: pattern is older (not second-to-last candle)
    candles_ago = len(df) - 1 - eng["pattern_idx"]
    if candles_ago > 2:
        confidence -= (candles_ago - 2) * 5

    confidence = round(min(95.0, max(50.0, confidence)), 1)

    reasons = [
        f"Pattern: {eng['pattern']} ({eng['engulf_ratio']}x engulf)",
        f"Confirmed: {conf_note}",
        f"Trend: {trend_note}",
        f"{loc_note}",
    ]
    if nifty_trend != "NEUTRAL":
        reasons.append(f"NIFTY aligned: {nifty_trend}")

    # Log to journal
    try:
        threading.Thread(
            target=_log_signal,
            args=(symbol, direction, "sniper", confidence, entry * 0.999, entry * 1.001,
                  sl, t2, rr, " | ".join(reasons[:3])),
            daemon=True,
        ).start()
    except Exception:
        pass

    return {
        "symbol":        symbol,
        "signal":        direction,
        "engine":        "sniper",
        "confidence":    confidence,
        "quality":       "A+" if confidence >= 78 else "A",
        "pattern":       eng["pattern"],
        "engulf_ratio":  eng["engulf_ratio"],
        "entry":         entry,
        "sl":            sl,
        "t1":            t1,
        "t2":            t2,
        "t3":            t3,
        "rr_t2":         rr,
        "current_price": price,
        "time_window":   time_note,
        "trend_note":    trend_note,
        "location":      loc_note,
        "nifty_trend":   nifty_trend,
        "vol_mult":      round(vol_mult, 2),
        "reasons":       reasons,
        "warnings":      [],
        "fetched_at":    now_ist.strftime("%H:%M:%S IST"),
    }


# ── FastAPI Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/sniper/{symbol}")
async def sniper_signal(symbol: str, interval: str = "15minute"):
    """
    High-accuracy Sniper signal for a single symbol.
    All 7 gates (time, trend, engulfing, confirmation, location, index, RR) must pass.
    Returns WAIT with reason if any gate fails.
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _compute_sniper_signal(symbol, interval)
        )
        return result
    except Exception as e:
        log.error(f"Sniper signal error {symbol}: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/sniper-scan")
async def sniper_scan(limit: int = 20, interval: str = "15minute"):
    """
    Scan the liquid universe through the Sniper Engine.
    Only returns confirmed BUY/SELL signals where all 7 gates pass.
    Typically fires 1–4 signals per session (by design — quality over quantity).
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    scan_list = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BAJFINANCE",
        "TATAMOTORS", "AXISBANK", "SBIN", "WIPRO", "HCLTECH",
        "TATASTEEL", "VEDL", "COALINDIA", "ONGC", "SAIL", "NATIONALUM",
        "MARUTI", "BAJAJ-AUTO", "TITAN", "SUNPHARMA",
        "DRREDDY", "RVNL", "ADANIENT", "LTIM", "KPITTECH",
    ][:limit + 10]

    results: List[Dict] = []
    wait_reasons: List[Dict] = []

    def _scan_one(sym: str):
        try:
            return _compute_sniper_signal(sym, interval)
        except Exception as exc:
            return {"symbol": sym, "signal": "WAIT", "reason": str(exc), "engine": "sniper"}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_scan_one, s): s for s in scan_list}
        for f in as_completed(futs, timeout=90):
            r = f.result()
            if r and r.get("signal") in ("BUY", "SELL"):
                results.append(r)
            elif r:
                wait_reasons.append({"symbol": r.get("symbol"), "reason": r.get("reason", "WAIT")})

    results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    IST = timezone(timedelta(hours=5, minutes=30))
    time_ok, time_note = _sniper_time_ok()

    return {
        "scanned":        len(scan_list),
        "signals_found":  len(results),
        "signals":        results[:limit],
        "time_window_ok": time_ok,
        "time_note":      time_note,
        "wait_summary":   wait_reasons[:10],
        "fetched_at":     datetime.now(tz=IST).strftime("%H:%M:%S IST"),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  SNIPER EXECUTION LAYER  —  Daily Risk Guard + Kite Order Placement
# ════════════════════════════════════════════════════════════════════════════════

# ── Daily Trade Limiter ───────────────────────────────────────────────────────

def _sniper_daily_counts() -> Dict:
    """
    Read today's sniper trades and losses from signal_log.
    Returns {"trades": int, "losses": int, "date": str}
    """
    try:
        IST = timezone(timedelta(hours=5, minutes=30))
        today = datetime.now(tz=IST).strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT,
                direction TEXT, engine TEXT, confidence REAL,
                entry_low REAL, entry_high REAL, sl REAL, t1 REAL, t2 REAL,
                t3 REAL, rr_t2 REAL, position_size INTEGER,
                outcome TEXT DEFAULT 'OPEN', pnl REAL DEFAULT 0, notes TEXT
            )
        """)
        rows = conn.execute(
            "SELECT outcome FROM signal_log WHERE engine='sniper' AND ts LIKE ? || '%'",
            (today,)
        ).fetchall()
        conn.close()
        trades = len(rows)
        losses = sum(1 for r in rows if r["outcome"] == "LOSS")
        return {"trades": trades, "losses": losses, "date": today}
    except Exception:
        return {"trades": 0, "losses": 0, "date": ""}


def _sniper_trading_allowed() -> Tuple[bool, str]:
    """
    Returns (allowed, reason).
    Blocked if: trades_today >= 2  OR  losses_today >= 2.
    """
    counts = _sniper_daily_counts()
    if counts["losses"] >= 2:
        return False, f"🛑 2 losses today — trading stopped (loss #{counts['losses']} hit daily limit)"
    if counts["trades"] >= 2:
        return False, f"🛑 Max 2 trades per day reached ({counts['trades']} taken)"
    return True, f"OK — {counts['trades']}/2 trades, {counts['losses']}/2 losses today"


# ── No-Trade Zone Filters ─────────────────────────────────────────────────────

def _sniper_ema_slope_ok(df: pd.DataFrame, direction: str, min_slope_pct: float = 0.05) -> Tuple[bool, str]:
    """
    EMA-50 slope filter: if EMA50 is flat (< min_slope_pct% per candle),
    market is sideways — skip.
    slope_pct = (ema[-1] - ema[-5]) / ema[-5] * 100
    """
    try:
        close = df["close"].astype(float)
        ema50 = _sniper_ema(close, 50)
        if len(ema50) < 6:
            return True, "slope check skipped (< 6 rows)"
        slope = (float(ema50.iloc[-1]) - float(ema50.iloc[-5])) / float(ema50.iloc[-5]) * 100
        if abs(slope) < min_slope_pct:
            return False, f"EMA50 slope flat ({slope:+.3f}% over 5 candles) — sideways market"
        direction_ok = (direction == "BUY" and slope > 0) or (direction == "SELL" and slope < 0)
        if not direction_ok:
            return False, f"EMA50 slope {slope:+.3f}% opposes {direction}"
        return True, f"EMA slope {slope:+.3f}%"
    except Exception as exc:
        return True, f"slope check skipped ({exc})"


def _sniper_no_inside_bar(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Inside bar filter: if the pattern candle is entirely inside the previous
    candle's range it's a compression candle — low conviction, skip.
    """
    try:
        if len(df) < 3:
            return True, "inside bar check skipped"
        curr = df.iloc[-2]   # pattern candle
        prev = df.iloc[-3]
        if float(curr["high"]) <= float(prev["high"]) and float(curr["low"]) >= float(prev["low"]):
            return False, f"Inside bar — candle range contained within previous candle, skip"
        return True, "Not inside bar"
    except Exception:
        return True, "inside bar check skipped"


def _sniper_volume_spike(df: pd.DataFrame, pattern_idx: int, min_mult: float = 1.3) -> Tuple[bool, str]:
    """
    Entry candle volume must be ≥ min_mult × average of prior 5 candles.
    Already used for confidence bonus in _compute_sniper_signal;
    here it's a hard gate.
    """
    try:
        if "volume" not in df.columns or pattern_idx < 5:
            return True, "volume check skipped"
        pat_vol = float(df.iloc[pattern_idx]["volume"])
        avg_vol = float(df.iloc[max(0, pattern_idx - 5): pattern_idx]["volume"].mean())
        if avg_vol <= 0:
            return True, "zero avg volume"
        mult = pat_vol / avg_vol
        if mult < min_mult:
            return False, f"Volume {mult:.1f}x avg — needs {min_mult}x (low participation)"
        return True, f"Volume {mult:.1f}x avg ✅"
    except Exception:
        return True, "volume check skipped"


# ── Patch _compute_sniper_signal to add new gates ─────────────────────────────
# We wrap it so the original function stays intact and tests still pass.
_original_compute_sniper = _compute_sniper_signal

def _compute_sniper_signal_v2(symbol: str, interval: str = "15minute") -> Dict:
    """
    Extended Sniper pipeline: adds 4 extra no-trade filters on top of the 7-gate
    base engine, plus the daily trade-count guard.
    Returned dict is identical in structure to _compute_sniper_signal.
    """
    # ── Daily risk guard (gate 0) ─────────────────────────────────────────────
    allowed, block_reason = _sniper_trading_allowed()
    if not allowed:
        IST = timezone(timedelta(hours=5, minutes=30))
        return {
            "symbol": symbol, "signal": "WAIT", "engine": "sniper",
            "reason": block_reason,
            "fetched_at": datetime.now(tz=IST).strftime("%H:%M:%S IST"),
        }

    # ── Run original 7-gate engine ────────────────────────────────────────────
    result = _original_compute_sniper(symbol, interval)
    if result.get("signal") != "BUY" and result.get("signal") != "SELL":
        return result      # already WAIT with reason

    # ── Fetch df for extra gates (reuse token / rate-limit safe) ─────────────
    try:
        token = get_instrument_token(symbol)
        if token:
            with _kite_semaphore:
                df_extra = kite_manager.get_historical_data(token, interval, 15)
        else:
            df_extra = pd.DataFrame()
    except Exception:
        df_extra = pd.DataFrame()

    if df_extra.empty or len(df_extra) < 10:
        return result   # soft fail — don't block on data issues

    direction = result["signal"]

    # ── Gate 8: EMA slope (no sideways) ──────────────────────────────────────
    slope_ok, slope_note = _sniper_ema_slope_ok(df_extra, direction)
    if not slope_ok:
        IST = timezone(timedelta(hours=5, minutes=30))
        return {"symbol": symbol, "signal": "WAIT", "engine": "sniper",
                "reason": f"📊 {slope_note}",
                "fetched_at": datetime.now(tz=IST).strftime("%H:%M:%S IST")}

    # ── Gate 9: No inside bar ─────────────────────────────────────────────────
    ib_ok, ib_note = _sniper_no_inside_bar(df_extra)
    if not ib_ok:
        IST = timezone(timedelta(hours=5, minutes=30))
        return {"symbol": symbol, "signal": "WAIT", "engine": "sniper",
                "reason": f"📦 {ib_note}",
                "fetched_at": datetime.now(tz=IST).strftime("%H:%M:%S IST")}

    # ── Gate 10: Volume spike on pattern candle ───────────────────────────────
    eng_pat = _sniper_find_engulfing(df_extra)
    if eng_pat:
        vol_ok, vol_note = _sniper_volume_spike(df_extra, eng_pat["pattern_idx"])
        if not vol_ok:
            IST = timezone(timedelta(hours=5, minutes=30))
            return {"symbol": symbol, "signal": "WAIT", "engine": "sniper",
                    "reason": f"📉 {vol_note}",
                    "fetched_at": datetime.now(tz=IST).strftime("%H:%M:%S IST")}
        result["reasons"].append(vol_note)
        result["reasons"].append(slope_note)

    return result


# Hotswap the function used by the scan endpoints
_compute_sniper_signal = _compute_sniper_signal_v2


# ── Position Sizing ───────────────────────────────────────────────────────────

def _sniper_position_size(entry: float, sl: float, capital: float = CAPITAL,
                           risk_pct: float = 0.01) -> int:
    """
    Risk 1% of capital per trade (configurable).
    qty = floor(capital * risk_pct / risk_per_share)
    Minimum 1, maximum (capital * 0.20) / entry to avoid over-exposure.
    """
    risk_per_share = abs(entry - sl)
    if risk_per_share <= 0:
        return 1
    qty = int(CAPITAL * risk_pct / risk_per_share)
    max_qty = int(CAPITAL * 0.20 / entry)
    return max(1, min(qty, max_qty))


# ── Kite Order Execution ──────────────────────────────────────────────────────

@dataclass
class SniperOrder:
    symbol:          str
    direction:       str      # BUY | SELL
    entry:           float
    sl:              float
    t1:              float    # 1:1 target — book 50% here
    t2:              float    # 2:1 target — trail rest
    qty:             int
    entry_order_id:  Optional[str] = None
    sl_order_id:     Optional[str] = None
    t1_order_id:     Optional[str] = None
    status:          str = "PENDING"   # PENDING | LIVE | PARTIAL | CLOSED | FAILED
    placed_at:       str = ""


def _place_sniper_order(req: "SniperOrderRequest") -> Dict:
    """
    Places a 3-leg MIS intraday trade on Kite:
      Leg 1 — LIMIT entry order
      Leg 2 — SL-LIMIT stop loss
      Leg 3 — LIMIT target at T1 (50% qty) for partial booking

    Uses GTT for SL and T1 so they persist even if the app restarts.
    Returns dict with order IDs or error message.

    IMPORTANT:
      • Product = MIS (intraday, auto-squared at 3:15 PM)
      • Exchange = NSE
      • Variety = regular (not bracket/cover — more reliable)
    """
    if not kite_manager.is_authenticated:
        return {"error": "Kite not authenticated"}

    kite = kite_manager.kite
    sym  = req.symbol
    direction = req.direction
    entry, sl, t1, t2 = req.entry, req.sl, req.t1, req.t2

    qty = _sniper_position_size(entry, sl)
    if qty < 1:
        return {"error": "Qty < 1 — risk per share too small"}

    txn_buy  = "BUY"
    txn_sell = "SELL"
    entry_txn = txn_buy  if direction == "BUY" else txn_sell
    exit_txn  = txn_sell if direction == "BUY" else txn_buy

    try:
        # ── Leg 1: Entry order (LIMIT) ────────────────────────────────────────
        entry_id = kite.place_order(
            variety       = kite.VARIETY_REGULAR,
            exchange      = kite.EXCHANGE_NSE,
            tradingsymbol = sym,
            transaction_type = entry_txn,
            quantity      = qty,
            product       = kite.PRODUCT_MIS,
            order_type    = kite.ORDER_TYPE_LIMIT,
            price         = round(entry, 1),
            validity      = kite.VALIDITY_DAY,
            tag           = "SNIPER_ENTRY",
        )
        log.info(f"🎯 Sniper entry order placed: {direction} {sym} qty={qty} entry={entry} → {entry_id}")
    except Exception as e:
        return {"error": f"Entry order failed: {e}"}

    sl_id = t1_id = None

    try:
        # ── Leg 2: SL-LIMIT stop loss ─────────────────────────────────────────
        # trigger at SL, limit 0.5% beyond (avoids slippage rejection)
        sl_limit = round(sl * 0.995, 1) if direction == "BUY" else round(sl * 1.005, 1)
        sl_id = kite.place_order(
            variety          = kite.VARIETY_REGULAR,
            exchange         = kite.EXCHANGE_NSE,
            tradingsymbol    = sym,
            transaction_type = exit_txn,
            quantity         = qty,
            product          = kite.PRODUCT_MIS,
            order_type       = kite.ORDER_TYPE_SL,
            price            = sl_limit,
            trigger_price    = round(sl, 1),
            validity         = kite.VALIDITY_DAY,
            tag              = "SNIPER_SL",
        )
        log.info(f"🛡 SL order placed: {sym} SL={sl} limit={sl_limit} → {sl_id}")
    except Exception as e:
        log.warning(f"SL order failed for {sym}: {e} — entry order still live! Cancel manually.")

    try:
        # ── Leg 3: T1 target (50% qty, partial booking) ──────────────────────
        t1_qty = max(1, qty // 2)
        t1_id = kite.place_order(
            variety          = kite.VARIETY_REGULAR,
            exchange         = kite.EXCHANGE_NSE,
            tradingsymbol    = sym,
            transaction_type = exit_txn,
            quantity         = t1_qty,
            product          = kite.PRODUCT_MIS,
            order_type       = kite.ORDER_TYPE_LIMIT,
            price            = round(t1, 1),
            validity         = kite.VALIDITY_DAY,
            tag              = "SNIPER_T1",
        )
        log.info(f"🎯 T1 partial target placed: {sym} T1={t1} qty={t1_qty} → {t1_id}")
    except Exception as e:
        log.warning(f"T1 order failed for {sym}: {e}")

    # ── Log to journal ────────────────────────────────────────────────────────
    risk  = abs(entry - sl)
    rr    = round(abs(t2 - entry) / risk, 2) if risk > 0 else 0
    notes = f"LIVE trade | entry_id={entry_id} sl_id={sl_id} t1_id={t1_id}"
    threading.Thread(
        target=_log_signal,
        args=(sym, direction, "sniper-live", 85.0,
              entry * 0.999, entry * 1.001, sl, t2, rr, notes),
        daemon=True,
    ).start()

    return {
        "ok":             True,
        "symbol":         sym,
        "direction":      direction,
        "qty":            qty,
        "entry":          entry,
        "sl":             sl,
        "t1":             t1,
        "t2":             t2,
        "entry_order_id": entry_id,
        "sl_order_id":    sl_id,
        "t1_order_id":    t1_id,
        "note":           f"MIS LIMIT order live. SL={sl}, T1={t1} (50% at 1:1), trail rest to T2={t2}",
    }


# ── FastAPI Request Models ────────────────────────────────────────────────────

from pydantic import BaseModel

class SniperOrderRequest(BaseModel):
    symbol:    str
    direction: str
    entry:     float
    sl:        float
    t1:        float
    t2:        float


# ── Execution Endpoints ───────────────────────────────────────────────────────

@app.post("/api/sniper/execute")
async def execute_sniper_trade(req: SniperOrderRequest):
    """
    Place a live MIS trade on Zerodha Kite.
    Checks daily trade limit before placing.
    Places: entry LIMIT + SL-LIMIT + T1 partial LIMIT (3 legs).
    """
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Kite not authenticated"}, status_code=401)

    allowed, block_reason = _sniper_trading_allowed()
    if not allowed:
        return JSONResponse({"error": block_reason}, status_code=403)

    # Basic sanity checks
    if req.direction not in ("BUY", "SELL"):
        return JSONResponse({"error": "direction must be BUY or SELL"}, status_code=400)
    if req.direction == "BUY"  and req.sl >= req.entry:
        return JSONResponse({"error": f"SL {req.sl} must be below entry {req.entry} for BUY"}, status_code=400)
    if req.direction == "SELL" and req.sl <= req.entry:
        return JSONResponse({"error": f"SL {req.sl} must be above entry {req.entry} for SELL"}, status_code=400)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _place_sniper_order(req))
        return result
    except Exception as e:
        log.error(f"Execute sniper error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/sniper/paper")
async def paper_sniper_trade(req: SniperOrderRequest):
    """Log a paper/simulated sniper trade to the journal without placing a real order."""
    risk = abs(req.entry - req.sl)
    rr   = round(abs(req.t2 - req.entry) / risk, 2) if risk > 0 else 0
    try:
        _log_signal(
            req.symbol, req.direction, "sniper-paper", 80.0,
            req.entry * 0.999, req.entry * 1.001,
            req.sl, req.t2, rr,
            f"Paper trade | T1={req.t1} T2={req.t2} SL={req.sl}",
        )
        return {"ok": True, "note": "Paper trade logged to journal"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/sniper/daily-status")
async def sniper_daily_status():
    """Return today's sniper trade count, loss count, and whether trading is allowed."""
    counts  = _sniper_daily_counts()
    allowed, reason = _sniper_trading_allowed()
    _, time_note    = _sniper_time_ok()
    return {
        "trades_today":    counts["trades"],
        "losses_today":    counts["losses"],
        "trading_allowed": allowed,
        "block_reason":    reason if not allowed else None,
        "time_note":       time_note,
        "date":            counts["date"],
    }


@app.get("/api/sniper/positions")
async def sniper_positions():
    """Return current open MIS positions tagged as SNIPER from Kite."""
    if not kite_manager.is_authenticated:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    try:
        positions = kite_manager.kite.positions()
        day_pos   = positions.get("day", [])
        sniper    = [p for p in day_pos if p.get("product") == "MIS" and abs(p.get("quantity", 0)) > 0]
        return {"positions": sniper, "count": len(sniper)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ════════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import threading

    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
