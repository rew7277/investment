"""
═══════════════════════════════════════════════════════════════════
  INSTITUTIONAL TRADER PRO — Complete Engine
  Fixed version with all required components
═══════════════════════════════════════════════════════════════════
"""

import os, logging, json, time
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import requests

log = logging.getLogger("ENGINE")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CFG = {
    "KITE_API_KEY":    os.environ.get("KITE_API_KEY", ""),
    "KITE_API_SECRET": os.environ.get("KITE_API_SECRET", ""),
    "ACCESS_TOKEN":    os.environ.get("KITE_ACCESS_TOKEN", ""),
    "PAPER_TRADE":     os.environ.get("PAPER_TRADE", "true").lower() == "true",
    
    # Scan timing
    "SCAN_TIME": os.environ.get("SCAN_TIME", "9:30"),
    
    # Risk parameters
    "RISK_PER_TRADE": 0.01,
    "MAX_POSITION_SIZE": 0.20,
    "ATR_PERIOD": 14,
    "ATR_SL_MULTIPLIER": 2.0,
    "PROFIT_TARGET_RR": 2.5,
    "CAPITAL": 1000000,  # Default capital for paper trading
    
    # Technical parameters
    "EMA_PERIOD": 50,
    "RSI_PERIOD": 14,
    "RSI_MIN": 50,
    "RSI_MAX": 70,
    "VOL_PERIOD": 20,
    "VOL_MIN_SURGE": 1.5,
    "BREAKOUT_LOOKBACK": 20,
    "BREAKOUT_TOLERANCE": 0.02,
    "VIX_MAX": 20,
    "MIN_SCORE": 13,
    "GAP_UP_MIN_PCT": 2.0,  # Minimum gap-up percentage to alert
    
    # Index instrument tokens (NSE)
    "NIFTY_TOKEN": 256265,      # NIFTY 50
    "BANKNIFTY_TOKEN": 260105,  # NIFTY BANK
    "FINNIFTY_TOKEN": 257801,   # NIFTY FIN SERVICE
    "MIDCAP_TOKEN": 288009,     # NIFTY MIDCAP 100
}


# ═════════════════════════════════════════════════════════════
# TOKEN PERSISTENCE
# ═════════════════════════════════════════════════════════════
TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".kite_token")

def save_token(token: str):
    """Save access token to disk."""
    try:
        with open(TOKEN_FILE, 'w') as f:
            json.dump({"token": token, "timestamp": datetime.now().isoformat()}, f)
        log.info("[Token] Saved to disk")
    except Exception as e:
        log.error(f"[Token] Save failed: {e}")

def load_token() -> Optional[str]:
    """Load access token from disk."""
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE) as f:
                data = json.load(f)
                return data.get("token")
    except Exception as e:
        log.error(f"[Token] Load failed: {e}")
    return None

def is_token_fresh(max_age_hours: int = 24) -> bool:
    """Check if saved token is still fresh."""
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE) as f:
                data = json.load(f)
                timestamp = datetime.fromisoformat(data["timestamp"])
                age = (datetime.now() - timestamp).total_seconds() / 3600
                return age < max_age_hours
    except Exception:
        pass
    return False


# ═════════════════════════════════════════════════════════════
# KITE LAYER
# ═════════════════════════════════════════════════════════════
class KiteLayer:
    """Wrapper for Kite API calls with logging."""
    
    def __init__(self, kite_instance):
        self._k = kite_instance
        log.info("[Kite] Initialized")
    
    def get_historical_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol."""
        try:
            instruments = pd.DataFrame(self._k.instruments("NSE"))
            row = instruments[instruments["tradingsymbol"] == symbol]
            if row.empty:
                return pd.DataFrame()
            
            token = int(row.iloc[0]["instrument_token"])
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self._k.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            return df
            
        except Exception as e:
            log.error(f"[Kite] Historical data error for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> dict:
        """Get live quote for a symbol."""
        try:
            key = f"NSE:{symbol}"
            data = self._k.quote([key])
            if key in data:
                return data[key]
            return {}
        except Exception as e:
            log.error(f"[Kite] Quote error for {symbol}: {e}")
            return {}
    
    def get_quotes(self, symbols: List[str]) -> dict:
        """Get quotes for multiple symbols."""
        try:
            keys = [f"NSE:{s}" for s in symbols]
            return self._k.quote(keys)
        except Exception as e:
            log.error(f"[Kite] Quotes error: {e}")
            return {}
    
    def get_vix(self) -> float:
        """Get current India VIX value."""
        try:
            data = self._k.ltp(["NSE:INDIA VIX"])
            if "NSE:INDIA VIX" in data:
                return float(data["NSE:INDIA VIX"]["last_price"])
            return 15.0
        except Exception:
            return 15.0
    
    def get_all_stocks(self) -> List[str]:
        """Get all NSE stocks from instruments."""
        try:
            instruments = pd.DataFrame(self._k.instruments("NSE"))
            # Filter for equity stocks only
            equity = instruments[instruments["segment"] == "NSE"]
            stocks = equity["tradingsymbol"].unique().tolist()
            log.info(f"[Kite] Loaded {len(stocks)} NSE stocks")
            return stocks
        except Exception as e:
            log.error(f"[Kite] Failed to load stocks: {e}")
            # Return curated list as fallback
            return NSEClient().get_all_stocks()
    
    def get_batch_quotes(self, symbols: List[str], batch_size: int = 500) -> dict:
        """Get quotes for a list of symbols in batches."""
        all_quotes = {}
        
        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                keys = [f"NSE:{s}" for s in batch]
                quotes = self._k.quote(keys)
                all_quotes.update(quotes)
            except Exception as e:
                log.error(f"[Kite] Batch quotes error (batch {i//batch_size + 1}): {e}")
        
        return all_quotes
    
    def get_available_capital(self) -> float:
        """Get available capital from margins."""
        try:
            margins = self._k.margins("equity")
            return float(margins.get("available", {}).get("live_balance", 0))
        except Exception as e:
            log.error(f"[Kite] Margins error: {e}")
            return 0.0
    
    def get_ohlcv(self, token: int, days: int = 400) -> pd.DataFrame:
        """Fetch OHLCV data for an instrument token."""
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self._k.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            return df
            
        except Exception as e:
            log.error(f"[Kite] OHLCV error for token {token}: {e}")
            return pd.DataFrame()


# ═════════════════════════════════════════════════════════════
# NSE CLIENT
# ═════════════════════════════════════════════════════════════
class NSEClient:
    """Fetches NSE stock universe and FII/DII data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })
    
    def get_all_stocks(self) -> List[str]:
        """Return a curated list of liquid NSE stocks."""
        # Nifty 50 + Nifty Next 50 + select midcaps
        return [
            # NIFTY 50
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
            "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
            "ASIANPAINT", "MARUTI", "TITAN", "BAJFINANCE", "SUNPHARMA",
            "ULTRACEMCO", "NESTLEIND", "WIPRO", "HCLTECH", "TECHM",
            "POWERGRID", "NTPC", "ONGC", "TATAMOTORS", "INDUSINDBK",
            "BAJAJFINSV", "JSWSTEEL", "GRASIM", "COALINDIA", "TATASTEEL",
            "M&M", "ADANIPORTS", "DRREDDY", "APOLLOHOSP", "DIVISLAB",
            "BRITANNIA", "EICHERMOT", "HEROMOTOCO", "CIPLA", "HINDALCO",
            "BAJAJ-AUTO", "SHREECEM", "BPCL", "UPL", "TATACONSUM",
            "SBILIFE", "HAVELLS", "PIDILITIND", "DABUR",
            
            # NIFTY NEXT 50
            "ADANIGREEN", "ADANIPOWER", "ASHOKLEY", "AUBANK", "BANDHANBNK",
            "BERGEPAINT", "COLPAL", "DLF", "GODREJCP", "INDIGO",
            "LICHSGFIN", "LUPIN", "MUTHOOTFIN", "NMDC", "PEL",
            "PFC", "RECLTD", "SIEMENS", "TORNTPHARM", "VEDL",
        ]
    
    def get_fii_dii_data(self) -> dict:
        """Fetch FII/DII trading data."""
        # Mock data - in production, fetch from NSE
        return {
            "fii_net": 1500.0,
            "dii_net": -800.0,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
    
    def get_fii_dii(self) -> dict:
        """Alias for get_fii_dii_data."""
        return self.get_fii_dii_data()
    
    def get_india_vix(self) -> Optional[float]:
        """Fetch India VIX value."""
        # Mock data - in production, fetch from NSE
        # Returns None to trigger fallback to Kite
        return None
    
    def get_option_chain_pcr(self, symbol: str) -> dict:
        """Fetch option chain PCR (Put-Call Ratio) data."""
        # Mock data - in production, fetch from NSE option chain API
        return {
            "pcr": 1.2,
            "calls_oi": 15000000,
            "puts_oi": 18000000,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
    
    def prefetch_scan_data(self):
        """Prefetch any necessary data before scan."""
        # Placeholder for any pre-scan data fetching
        pass


# ═════════════════════════════════════════════════════════════
# UNIVERSE MANAGER
# ═════════════════════════════════════════════════════════════
class UniverseManager:
    """Manages the stock universe."""
    
    def __init__(self, nse_client: NSEClient, kite_layer=None):
        self.nse = nse_client
        self.kite = kite_layer
        self.universe = []
        self.instruments_df = None
    
    def load_universe(self) -> List[str]:
        """Load the trading universe."""
        self.universe = self.nse.get_all_stocks()
        log.info(f"[Universe] Loaded {len(self.universe)} stocks")
        return self.universe
    
    def get_universe(self) -> pd.DataFrame:
        """Get current universe as DataFrame with instrument details."""
        if self.instruments_df is None and self.kite:
            # Get instruments from Kite
            try:
                instruments = pd.DataFrame(self.kite._k.instruments("NSE"))
                # Filter for equity stocks only
                equity = instruments[instruments["segment"] == "NSE"]
                equity = equity[equity["instrument_type"] == "EQ"]
                self.instruments_df = equity[["tradingsymbol", "instrument_token", "name"]].copy()
                self.instruments_df.rename(columns={"tradingsymbol": "symbol"}, inplace=True)
            except Exception as e:
                log.error(f"[Universe] Failed to load instruments: {e}")
                # Fallback to simple list
                if not self.universe:
                    self.load_universe()
                self.instruments_df = pd.DataFrame({"symbol": self.universe})
        
        if self.instruments_df is None:
            if not self.universe:
                self.load_universe()
            self.instruments_df = pd.DataFrame({"symbol": self.universe})
        
        return self.instruments_df
    
    def filter_for_deep_scan(self, quote_data: dict) -> List[dict]:
        """Filter stocks for deep technical analysis based on initial screening."""
        candidates = []
        
        for key, quote in quote_data.items():
            symbol = key.replace("NSE:", "")
            
            # Extract quote data
            ltp = quote.get("last_price", 0)
            if ltp <= 0:
                continue
            
            volume = quote.get("volume", 0)
            avg_volume = quote.get("average_price", 0)  # Using as proxy
            open_price = quote.get("ohlc", {}).get("open", 0)
            prev_close = quote.get("ohlc", {}).get("close", 0)
            
            # Calculate gap %
            gap_pct = 0
            if prev_close > 0:
                gap_pct = ((open_price - prev_close) / prev_close) * 100
            
            # Calculate volume surge
            vol_surge = False
            if avg_volume > 0 and volume > 0:
                vol_surge = volume > (avg_volume * 1.5)
            
            # Include if gap-up OR high volume OR significant price change
            chg_pct = quote.get("net_change", 0)
            
            if gap_pct >= 2.0 or vol_surge or abs(chg_pct) >= 3.0 or volume > 100000:
                candidates.append({
                    "symbol": symbol,
                    "ltp": ltp,
                    "volume": volume,
                    "gap_pct": gap_pct,
                    "chg_pct": chg_pct,
                    "vol_surge": vol_surge
                })
        
        # Sort by gap % and volume
        candidates.sort(key=lambda x: (x.get("gap_pct", 0), x.get("volume", 0)), reverse=True)
        
        # Return top 200 candidates for deep analysis
        return candidates[:200]


# ═════════════════════════════════════════════════════════════
# TECHNICAL ENGINE
# ═════════════════════════════════════════════════════════════
class TechnicalEngine:
    """Calculate technical indicators and scores."""
    
    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """Alias for calculate_indicators - computes all technical indicators."""
        return TechnicalEngine.calculate_indicators(df)
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        if df.empty or len(df) < 60:
            return df
        
        # EMA 50
        df['ema50'] = df['close'].ewm(span=CFG['EMA_PERIOD'], adjust=False).mean()
        
        # RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CFG['RSI_PERIOD']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CFG['RSI_PERIOD']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume MA
        df['vol_ma'] = df['volume'].rolling(window=CFG['VOL_PERIOD']).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=CFG['ATR_PERIOD']).mean()
        
        # VWAP (approximation using daily data)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # 20-day high
        df['high_20d'] = df['high'].rolling(window=CFG['BREAKOUT_LOOKBACK']).max()
        
        return df
    
    @staticmethod
    def analyze(df: pd.DataFrame, quote: dict) -> dict:
        """Analyze technical setup."""
        if df.empty or len(df) < 60:
            return {"score": 0, "signals": []}
        
        latest = df.iloc[-1]
        ltp = quote.get('last_price', latest['close'])
        
        signals = []
        score = 0
        
        # Trend (EMA)
        if ltp > latest['ema50']:
            signals.append("Above EMA50")
            score += 1
        
        # RSI
        rsi = latest['rsi']
        if CFG['RSI_MIN'] <= rsi <= CFG['RSI_MAX']:
            signals.append(f"RSI healthy ({rsi:.0f})")
            score += 1
        
        # Volume
        if latest['volume'] > latest['vol_ma'] * CFG['VOL_MIN_SURGE']:
            signals.append("Volume surge")
            score += 1
        
        # Breakout
        if ltp >= latest['high_20d'] * (1 - CFG['BREAKOUT_TOLERANCE']):
            signals.append("Near 20D high")
            score += 1
        
        return {
            "score": score,
            "signals": signals,
            "rsi": rsi,
            "ema50": latest['ema50'],
            "atr": latest['atr'],
            "vwap": latest.get('vwap', ltp),
        }


# ═════════════════════════════════════════════════════════════
# FUNDAMENTAL ENGINE (Placeholder)
# ═════════════════════════════════════════════════════════════
class FundamentalEngine:
    @staticmethod
    def analyze(symbol: str) -> dict:
        return {"score": 0, "signals": []}


# ═════════════════════════════════════════════════════════════
# BREAKOUT SCANNER
# ═════════════════════════════════════════════════════════════
class BreakoutScanner:
    @staticmethod
    def detect_breakout(df: pd.DataFrame, ltp: float) -> dict:
        if df.empty:
            return {"breakout": False, "score": 0}
        
        latest = df.iloc[-1]
        high_20 = latest.get('high_20d', 0)
        
        if ltp >= high_20 * 0.98:
            return {"breakout": True, "score": 2, "level": high_20}
        return {"breakout": False, "score": 0}


# ═════════════════════════════════════════════════════════════
# INSTITUTIONAL ENGINE (Placeholder)
# ═════════════════════════════════════════════════════════════
class InstitutionalEngine:
    @staticmethod
    def analyze() -> dict:
        return {"score": 0, "signals": []}


# ═════════════════════════════════════════════════════════════
# MARKET REGIME
# ═════════════════════════════════════════════════════════════
class MarketRegime:
    @staticmethod
    def analyze(vix: float, fii_dii: dict) -> dict:
        regime_score = 0
        regime = "Neutral"
        
        if vix < CFG['VIX_MAX']:
            regime_score += 1
            regime = "Bullish" if vix < 15 else "Moderate"
        else:
            regime = "Volatile"
        
        return {
            "regime": regime,
            "score": regime_score,
            "vix": vix,
            "fii_net": fii_dii.get("fii_net", 0),
            "dii_net": fii_dii.get("dii_net", 0),
        }
    
    @staticmethod
    def check(nifty_df: pd.DataFrame, vix: float, 
              banknifty_df: pd.DataFrame = None,
              finnifty_df: pd.DataFrame = None,
              midcap_df: pd.DataFrame = None,
              pcr: float = None) -> tuple:
        """
        Check market regime across multiple indices.
        Returns: (score, flags, regime_ok, label, confidence)
        """
        score = 0
        flags = []
        
        # VIX check (0-2 points)
        if vix is not None:
            if vix < 15:
                score += 2
                flags.append("LOW_VIX")
            elif vix < 20:
                score += 1
                flags.append("MODERATE_VIX")
            else:
                flags.append("HIGH_VIX")
        
        # Nifty trend check (0-2 points)
        if not nifty_df.empty and len(nifty_df) > 50:
            last_close = nifty_df.iloc[-1]['close']
            ema50 = nifty_df.iloc[-1].get('ema50', last_close)
            
            if last_close > ema50:
                score += 2
                flags.append("NIFTY_BULLISH")
            elif last_close > ema50 * 0.98:
                score += 1
                flags.append("NIFTY_NEUTRAL")
            else:
                flags.append("NIFTY_BEARISH")
        
        # Bank Nifty alignment (0-1 point)
        if banknifty_df is not None and not banknifty_df.empty and len(banknifty_df) > 50:
            last_close = banknifty_df.iloc[-1]['close']
            ema50 = banknifty_df.iloc[-1].get('ema50', last_close)
            if last_close > ema50:
                score += 1
                flags.append("BANKNIFTY_ALIGNED")
        
        # PCR check (0-1 point)
        if pcr is not None:
            if 0.8 < pcr < 1.3:
                score += 1
                flags.append("PCR_BALANCED")
            elif pcr > 1.3:
                flags.append("PCR_BULLISH")
            else:
                flags.append("PCR_BEARISH")
        
        # Mid cap participation (0-1 point)
        if midcap_df is not None and not midcap_df.empty and len(midcap_df) > 20:
            last_close = midcap_df.iloc[-1]['close']
            ma20 = midcap_df['close'].rolling(20).mean().iloc[-1]
            if last_close > ma20:
                score += 1
                flags.append("MIDCAP_STRONG")
        
        # Determine regime label and confidence
        if score >= 6:
            label = "BULLISH"
            confidence = 80 + (score - 6) * 5
        elif score >= 4:
            label = "SIDEWAYS BULLISH"
            confidence = 60 + (score - 4) * 10
        elif score >= 2:
            label = "SIDEWAYS"
            confidence = 40 + (score - 2) * 10
        else:
            label = "BEARISH"
            confidence = max(20, score * 10)
        
        regime_ok = score >= 3
        
        return (score, flags, regime_ok, label, min(confidence, 95))


# ═════════════════════════════════════════════════════════════
# RISK MANAGER
# ═════════════════════════════════════════════════════════════
class RiskManager:
    @staticmethod
    def calculate_position(ltp: float, atr: float, capital: float = 1000000) -> dict:
        if ltp <= 0 or atr <= 0:
            return {}
        
        sl_price = ltp - (CFG['ATR_SL_MULTIPLIER'] * atr)
        risk_per_share = ltp - sl_price
        
        if risk_per_share <= 0:
            return {}
        
        risk_amount = capital * CFG['RISK_PER_TRADE']
        shares = int(risk_amount / risk_per_share)
        max_shares = int((capital * CFG['MAX_POSITION_SIZE']) / ltp)
        shares = min(shares, max_shares)
        
        if shares <= 0:
            return {}
        
        target_price = ltp + (CFG['PROFIT_TARGET_RR'] * risk_per_share)
        
        return {
            "shares": shares,
            "entry": round(ltp, 2),
            "sl": round(sl_price, 2),
            "tp": round(target_price, 2),
            "risk_amount": round(shares * risk_per_share, 2),
        }


# ═════════════════════════════════════════════════════════════
# PLACEHOLDER ENGINES
# ═════════════════════════════════════════════════════════════
class SMCEngine:
    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        return {"score": 0, "bos": False}

class FibonacciEngine:
    @staticmethod
    def find_levels(df: pd.DataFrame) -> dict:
        return {"score": 0, "levels": []}

class PivotEngine:
    @staticmethod
    def calculate_pivots(data: dict) -> dict:
        """Calculate pivot points from OHLC data."""
        high = data.get("high", 0)
        low = data.get("low", 0)
        close = data.get("close", 0)
        
        if not all([high, low, close]):
            return {"r1": 0, "s1": 0, "pp": 0, "r2": 0, "s2": 0}
        
        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        s1 = (2 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        
        return {
            "pp": round(pp, 2),
            "r1": round(r1, 2),
            "s1": round(s1, 2),
            "r2": round(r2, 2),
            "s2": round(s2, 2),
        }
    
    @staticmethod
    def compute_all(nifty_df=None, banknifty_df=None, finnifty_df=None, midcap_df=None) -> dict:
        """Compute pivot levels for all indices."""
        result = {}
        
        def _compute_index(df, key):
            if df is None or df.empty:
                return {
                    "ltp": 0, "pp": 0, "r1": 0, "s1": 0, "r2": 0, "s2": 0,
                    "change_pct": 0, "high": 0, "low": 0, "close": 0
                }
            
            last = df.iloc[-1]
            prev_close = df.iloc[-2]['close'] if len(df) > 1 else last['close']
            
            pivots = PivotEngine.calculate_pivots({
                "high": last['high'],
                "low": last['low'],
                "close": prev_close
            })
            
            change_pct = 0
            if prev_close > 0:
                change_pct = ((last['close'] - prev_close) / prev_close) * 100
            
            return {
                "ltp": round(last['close'], 2),
                "change_pct": round(change_pct, 2),
                "high": round(last['high'], 2),
                "low": round(last['low'], 2),
                "close": round(last['close'], 2),
                **pivots
            }
        
        result["NIFTY"] = _compute_index(nifty_df, "NIFTY")
        result["BANKNIFTY"] = _compute_index(banknifty_df, "BANKNIFTY")
        result["FINNIFTY"] = _compute_index(finnifty_df, "FINNIFTY")
        result["MIDCAP"] = _compute_index(midcap_df, "MIDCAP")
        
        return result

class OptionEngine:
    @staticmethod
    def analyze() -> List[dict]:
        return []
    
    @staticmethod
    def build_signal(kl, underlying: str, ltp: float, direction: str,
                    underlying_sl: float, underlying_tp: float, vix: float,
                    oc_data: dict, equity_score: int = 0, equity_signal: str = "",
                    pivot_data: dict = None) -> dict:
        """Build option signal with strike selection and Greeks estimation."""
        
        # Mock implementation - returns a basic signal structure
        # In production, this would:
        # 1. Calculate optimal strike price (ATM/OTM)
        # 2. Fetch real option chain data
        # 3. Calculate Greeks (delta, theta, vega)
        # 4. Determine lot size and risk
        
        # Simple ATM strike calculation
        if direction == "CALL":
            strike = int(ltp / 100) * 100  # Round to nearest 100
            strike_label = f"{underlying} {strike} CE"
        else:
            strike = int(ltp / 100) * 100
            strike_label = f"{underlying} {strike} PE"
        
        # Mock option premium
        option_premium = ltp * 0.02  # ~2% of underlying
        
        return {
            "underlying": underlying,
            "underlying_ltp": ltp,
            "direction": direction,
            "strike": strike,
            "strike_label": strike_label,
            "premium": round(option_premium, 2),
            "underlying_sl": underlying_sl,
            "underlying_tp": underlying_tp,
            "signal": direction,
            "vix": vix,
            "equity_score": equity_score,
            "equity_signal": equity_signal,
        }

class NiftyOptionsEngine:
    @staticmethod
    def scan_live() -> List[dict]:
        return []
    
    @staticmethod
    def get_trend(idx_data: dict) -> str:
        """Determine trend from index data."""
        if not idx_data:
            return "UNKNOWN"
        
        ltp = idx_data.get("ltp", 0)
        r1 = idx_data.get("r1", 0)
        s1 = idx_data.get("s1", 0)
        pp = idx_data.get("pp", 0)
        
        if not all([ltp, pp]):
            return "UNKNOWN"
        
        if ltp > pp and ltp > r1:
            return "BULLISH"
        elif ltp < pp and ltp < s1:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def generate_signal(kl, idx_key: str, idx_data: dict, oc_data: dict, vix: float) -> Optional[dict]:
        """Generate F&O signal for an index."""
        
        if not idx_data:
            return None
        
        ltp = idx_data.get("ltp", 0)
        if ltp <= 0:
            return None
        
        # Get trend
        trend = NiftyOptionsEngine.get_trend(idx_data)
        
        # Only generate signals for clear trends
        if trend == "SIDEWAYS" or trend == "UNKNOWN":
            return None
        
        # Get pivot levels
        r1 = idx_data.get("r1", 0)
        s1 = idx_data.get("s1", 0)
        pp = idx_data.get("pp", 0)
        
        # Determine direction
        direction = "CALL" if trend == "BULLISH" else "PUT"
        
        # Set targets
        if direction == "CALL":
            sl = s1 if s1 > 0 else ltp * 0.98
            tp = r1 if r1 > 0 else ltp * 1.02
        else:
            sl = r1 if r1 > 0 else ltp * 1.02
            tp = s1 if s1 > 0 else ltp * 0.98
        
        # Calculate strike
        strike = int(ltp / 100) * 100
        strike_label = f"{idx_key} {strike} {'CE' if direction == 'CALL' else 'PE'}"
        
        # Mock premium
        premium = ltp * 0.015
        
        # Calculate confidence based on trend strength
        change_pct = idx_data.get("change_pct", 0)
        confidence = min(95, 60 + abs(change_pct) * 5)
        
        return {
            "symbol": idx_key,
            "type": "INDEX",
            "underlying_ltp": ltp,
            "direction": direction,
            "trend": trend,
            "strike": strike,
            "strike_label": strike_label,
            "premium": round(premium, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "vix": vix,
            "confidence": round(confidence, 1),
            "pp": pp,
            "r1": r1,
            "s1": s1,
            "pcr": oc_data.get("pcr", 1.0),
        }

class TopMoversEngine:
    @staticmethod
    def analyze(quotes: dict) -> dict:
        return {"gainers": [], "losers": [], "momentum": [], "volume": [], "reversal": []}
    
    @staticmethod
    def get_high_probability(movers: dict, signals: List[dict]) -> List[dict]:
        """Extract high probability setups from movers and signals."""
        if not movers or not signals:
            return []
        
        # Create signal lookup
        sig_map = {s.get("symbol"): s for s in signals}
        
        results = []
        seen = set()
        
        # Check gainers and momentum for high-scoring signals
        for category in ["gainers", "momentum"]:
            for mover in movers.get(category, []):
                symbol = mover.get("symbol")
                if not symbol or symbol in seen:
                    continue
                
                sig = sig_map.get(symbol)
                if not sig:
                    continue
                
                score = sig.get("score", 0)
                if score >= 13:  # Minimum tradeable score
                    seen.add(symbol)
                    results.append({
                        **mover,
                        "score": score,
                        "signal": sig.get("signal", "BUY"),
                        "sl": sig.get("sl", 0),
                        "tp": sig.get("tp", 0),
                        "vol_surge": sig.get("vol_surge", False),
                        "above_vwap": sig.get("above_vwap", False),
                        "smc_bos": sig.get("smc_bos", False),
                        "breakdown": sig.get("breakdown", {}),
                        "category": "⚡ High Probability" if score >= 20 else "🎯 BUY Candidate"
                    })
                    
                    if len(results) >= 10:
                        break
            
            if len(results) >= 10:
                break
        
        # Sort by score descending
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

class LiveIndexRefresher:
    @staticmethod
    def refresh_indices(kite: KiteLayer) -> dict:
        return {}
    
    @staticmethod
    def refresh(kite: KiteLayer) -> dict:
        """Refresh live index data."""
        try:
            indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCAP"]
            result = {}
            
            for idx in indices:
                try:
                    symbol_map = {
                        "NIFTY": "NIFTY 50",
                        "BANKNIFTY": "NIFTY BANK",
                        "FINNIFTY": "NIFTY FIN SERVICE",
                        "MIDCAP": "NIFTY MIDCAP 100"
                    }
                    
                    quote = kite.get_quote(symbol_map.get(idx, idx))
                    if quote:
                        result[idx] = {
                            "ltp": quote.get("last_price", 0),
                            "change_pct": quote.get("net_change", 0),
                            "volume": quote.get("volume", 0),
                        }
                    else:
                        result[idx] = {"error": "No data"}
                except Exception as e:
                    result[idx] = {"error": str(e)}
            
            return result
        except Exception as e:
            log.error(f"[LiveRefresh] Error: {e}")
            return {}


# ═════════════════════════════════════════════════════════════
# SCORING FUNCTION
# ═════════════════════════════════════════════════════════════
def build_score(symbol: str, df: pd.DataFrame, quote: dict, vix: float) -> dict:
    """Build complete score for a stock."""
    
    if df.empty or len(df) < 60:
        return {"symbol": symbol, "score": 0, "signal": "NO DATA"}
    
    # Calculate indicators
    df = TechnicalEngine.calculate_indicators(df)
    
    # Analyze
    tech = TechnicalEngine.analyze(df, quote)
    breakout = BreakoutScanner.detect_breakout(df, quote.get('last_price', 0))
    smc = SMCEngine.analyze(df)
    fib = FibonacciEngine.find_levels(df)
    
    # Regime check
    regime_score = 1 if vix < CFG['VIX_MAX'] else 0
    
    # Total score (out of 30)
    total_score = tech['score'] + breakout['score'] + smc['score'] + fib['score'] + regime_score
    
    # Signal
    if total_score >= 20:
        signal = "STRONG BUY"
    elif total_score >= CFG['MIN_SCORE']:
        signal = "BUY"
    else:
        signal = "WATCH"
    
    ltp = quote.get('last_price', df.iloc[-1]['close'])
    
    # Position sizing
    position = RiskManager.calculate_position(ltp, tech.get('atr', 10))
    
    return {
        "symbol": symbol,
        "score": total_score,
        "signal": signal,
        "ltp": ltp,
        "chg_pct": quote.get('net_change', 0),
        "sl": position.get('sl', 0),
        "tp": position.get('tp', 0),
        "category": signal,
        "breakdown": {
            "technical": {"score": tech['score']},
            "breakout": {"score": breakout['score']},
            "smc": {"score": smc['score']},
            "fibonacci": {"score": fib['score']},
        },
        "above_vwap": ltp > tech.get('vwap', ltp),
        "vol_surge": "Volume surge" in tech.get('signals', []),
        "smc_bos": smc.get('bos', False),
    }
