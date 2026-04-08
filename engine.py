"""
═══════════════════════════════════════════════════════════════════
  INSTITUTIONAL TRADER PRO — Full NSE Universe Engine
  Scans ALL ~1900 NSE EQ stocks, not just 15.

  KITE API CALLS IN THIS FILE:
  ┌─────────────────────────────────────────────────────────────┐
  │  kite.instruments("NSE")          → Full NSE stock list     │
  │  kite.quote([tokens])             → Batch LTP + OHLC data   │
  │  kite.ltp([tokens])               → Fast price scan         │
  │  kite.historical_data(token, ...) → OHLCV candle history    │
  │  kite.place_order(...)            → Buy / Sell orders       │
  │  kite.orders()                    → Order book              │
  │  kite.positions()                 → Open positions          │
  │  kite.margins()                   → Available capital       │
  └─────────────────────────────────────────────────────────────┘

  YAHOO FINANCE: REMOVED. All scoring is now 100% Kite + NSE API.
  No rate-limit issues. Scans run in ~3 minutes flat.

  FIBONACCI ENGINE: ADDED (v2).
  Computes retracement/extension levels from Kite OHLCV data.
  Integrated into SMC confluence detection.

  TOTAL SCORE: 30 pts (was 27)
    5 Technical + 5 Breakout + 5 Fundamental(Kite proxy)
    + 5 Institutional + 2 Regime + 5 SMC + 3 Fibonacci
═══════════════════════════════════════════════════════════════════
"""

import os, time, logging, requests, json, math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np

log = logging.getLogger("ENGINE")

# ─────────────────────────────────────────────────────────────
# TOKEN STORE
# ─────────────────────────────────────────────────────────────
TOKEN_FILE = "/tmp/kite_token.json"

def save_token(access_token: str):
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump({"access_token": access_token,
                       "saved_at": datetime.now().isoformat()}, f)
        CFG["ACCESS_TOKEN"] = access_token
        log.info(f"  [Auth] ✅ Token saved to {TOKEN_FILE}")
    except Exception as e:
        log.warning(f"  [Auth] Could not save token: {e}")
        CFG["ACCESS_TOKEN"] = access_token

def load_token() -> str:
    env_token = os.environ.get("KITE_ACCESS_TOKEN", "")
    if env_token:
        return env_token
    try:
        with open(TOKEN_FILE) as f:
            data = json.load(f)
        token = data.get("access_token", "")
        if token:
            log.info(f"  [Auth] Token loaded from file (saved: {data.get('saved_at','?')})")
            return token
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"  [Auth] Token file read error: {e}")
    return ""


def is_token_fresh(max_age_hours: int = 8) -> bool:
    """
    Returns True if the saved token was generated within max_age_hours.
    Kite tokens expire at midnight IST; 8h is a safe intraday window.
    Checks env-var tokens always as 'fresh' (Railway secrets are permanent).
    """
    if os.environ.get("KITE_ACCESS_TOKEN", ""):
        return True          # env-var token managed externally
    try:
        with open(TOKEN_FILE) as f:
            data = json.load(f)
        saved_at = datetime.fromisoformat(data.get("saved_at", ""))
        age = (datetime.now() - saved_at).total_seconds() / 3600
        fresh = age <= max_age_hours
        if not fresh:
            log.warning(f"  [Auth] Token age {age:.1f}h > {max_age_hours}h — may be stale")
        return fresh
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CFG = {
    # ── Kite credentials ──────────────────────────────────────
    "KITE_API_KEY":    os.environ.get("KITE_API_KEY",    ""),
    "KITE_API_SECRET": os.environ.get("KITE_API_SECRET", ""),
    "ACCESS_TOKEN":    load_token(),

    # ── Universe filters ─────────────────────────────────────
    "MIN_PRICE":       50,
    "MAX_PRICE":       99999,
    "UNIVERSE_CAP":    500,
    "GAP_SCAN_ALL":    True,
    "PRELOAD_TOKENS":  True,

    # ── Index tokens (Kite) ───────────────────────────────────
    "NIFTY_TOKEN":     256265,
    "BANKNIFTY_TOKEN": 260105,

    # ── Technical params ──────────────────────────────────────
    "EMA_FAST":        20,
    "EMA_SLOW":        50,
    "EMA_TREND":       200,
    "RSI_PERIOD":      14,
    "RSI_BUY_MIN":     50,
    "RSI_BUY_MAX":     70,
    "RSI_EXIT":        40,
    "ATR_PERIOD":      14,
    "ATR_SL":          1.5,
    "ATR_TP":          3.5,
    "ADX_PERIOD":      14,
    "ADX_MIN":         25,
    "ST_ATR_PERIOD":   10,
    "ST_MULTIPLIER":   3.0,
    "VOL_MA_PERIOD":   20,
    "VOL_SURGE_X":     1.5,

    # ── Breakout params ───────────────────────────────────────
    "GAP_UP_MIN_PCT":  2.0,
    "RS_PERIOD":       20,
    "CONSOL_DAYS":     10,

    # ── Fundamental thresholds (Kite proxy — no Yahoo) ───────
    "VOL_ACCUM_X":     1.3,    # 5d vol > 1.3x 20d vol = accumulation
    "HIGH_52W_PCT":    90,     # price within 90% of 52w high
    "ST_BULL_DAYS":    8,      # supertrend bullish for ≥8 of last 10 days
    "RS_FUND_PCT":     5,      # outperform NIFTY by ≥5% over 20d
    "MACD_POS_DAYS":   4,      # MACD histogram +ve for ≥4 of last 5 days

    # ── Fibonacci params ──────────────────────────────────────
    "FIB_LOOKBACK":    50,     # candles for swing high/low detection
    "FIB_TOLERANCE":   0.02,   # 2% tolerance for level confluence
    "FIB_GOLDEN_LOW":  0.618,  # Golden Zone lower bound (61.8% retrace)
    "FIB_GOLDEN_HIGH": 0.382,  # Golden Zone upper bound (38.2% retrace)

    # ── Institutional thresholds (NSE API) ───────────────────
    "MIN_DELIVERY_PCT": 45,
    "PCR_BULLISH_MIN":  0.85,
    "VIX_MAX":          20,

    # ── Scoring gates (max = 30) ──────────────────────────────
    # 5 tech + 5 breakout + 5 fundamental + 5 institutional
    # + 2 regime + 5 SMC + 3 Fibonacci = 30
    "SCORE_STRONG_BUY": 20,   # ≥20/30 → STRONG BUY  (~67%)
    "SCORE_BUY":        13,   # ≥13/30 → BUY          (~43%)
    "SCORE_WATCHLIST":   9,   # ≥ 9/30 → WATCHLIST    (~30%)

    # ── Trade mode ───────────────────────────────────────────
    # "swing"      → Daily candles, tight SL (default)
    # "positional" → Weekly candles, wider SL — stocks held weeks/months
    # "intraday"   → 15min candles, very tight SL
    "TRADE_MODE":      os.environ.get("TRADE_MODE", "swing"),

    # ── Risk ─────────────────────────────────────────────────
    "CAPITAL":          1_000_000,
    "RISK_PCT":         0.015,
    "MAX_OPEN_TRADES":  6,
    "BROKERAGE":        20,
    "SLIPPAGE_PCT":     0.0005,

    # ── Live settings ─────────────────────────────────────────
    "PAPER_TRADE":      os.environ.get("PAPER_TRADE", "true").lower() == "true",
    "SCAN_TIME":        os.environ.get("SCAN_TIME", "09:20"),
    "DATA_INTERVAL":    "day",
    "PRODUCT":          "CNC",
}


# ═════════════════════════════════════════════════════════════
# ███  KITE API LAYER  ████████████████████████████████████████
# ═════════════════════════════════════════════════════════════
class KiteLayer:
    """
    Single point of contact for ALL Kite API calls.
    Every method documents exactly which Kite endpoint it uses.
    """

    def __init__(self, kite):
        self._k = kite

    # ── 1. INSTRUMENTS ────────────────────────────────────────
    def get_all_nse_instruments(self) -> pd.DataFrame:
        """KITE API: kite.instruments("NSE")"""
        log.info("  [Kite] instruments('NSE') → fetching full NSE universe...")
        instruments = self._k.instruments("NSE")
        df = pd.DataFrame(instruments)
        df = df[df["segment"] == "NSE"]
        df = df[df["instrument_type"] == "EQ"]
        df = df[["instrument_token", "tradingsymbol", "name", "tick_size", "lot_size"]]
        df.columns = ["token", "symbol", "name", "tick", "lot"]
        df = df.reset_index(drop=True)
        log.info(f"  [Kite] → {len(df)} EQ instruments loaded")
        return df

    # ── 2. BATCH LTP ──────────────────────────────────────────
    def get_batch_ltp(self, tokens: List[int]) -> Dict[int, float]:
        """KITE API: kite.ltp([...]) in chunks of 500"""
        result = {}
        BATCH = 500
        chunks = [tokens[i:i+BATCH] for i in range(0, len(tokens), BATCH)]
        for chunk in chunks:
            keys = [str(t) for t in chunk]
            try:
                data = self._k.ltp(keys)
                for key, val in data.items():
                    tok = int(key.split(":")[-1]) if ":" in key else int(key)
                    result[tok] = val["last_price"]
            except Exception as e:
                log.warning(f"  [Kite] ltp batch error: {e}")
            time.sleep(0.1)
        return result

    # ── 3. BATCH QUOTES ───────────────────────────────────────
    def get_batch_quotes(self, symbols: List[str], exchange: str = "NSE") -> dict:
        """KITE API: kite.quote([...]) — OHLC + volume"""
        result = {}
        keys = [f"{exchange}:{s}" for s in symbols]
        BATCH = 500
        chunks = [keys[i:i+BATCH] for i in range(0, len(keys), BATCH)]
        for chunk in chunks:
            try:
                data = self._k.quote(chunk)
                for key, val in data.items():
                    sym = key.split(":")[-1]
                    result[sym] = {
                        "ltp":        val["last_price"],
                        "open":       val["ohlc"]["open"],
                        "high":       val["ohlc"]["high"],
                        "low":        val["ohlc"]["low"],
                        "prev_close": val["ohlc"]["close"],
                        "volume":     val["volume"],
                        "avg_price":  val.get("average_price", 0),
                        "buy_qty":    val.get("buy_quantity", 0),
                        "sell_qty":   val.get("sell_quantity", 0),
                        "oi":         val.get("oi", 0),
                    }
            except Exception as e:
                log.warning(f"  [Kite] quote batch error: {e}")
            time.sleep(0.15)
        return result

    # ── 4. HISTORICAL OHLCV ───────────────────────────────────
    def get_ohlcv(self, token: int, days: int = 300,
                  interval: str = "day") -> pd.DataFrame:
        """
        KITE API: kite.historical_data(instrument_token, from_date, to_date, interval)
        Interval options: minute, 3minute, 5minute, 10minute, 15minute,
                          30minute, 60minute, day, week
        For positional/stock mode, use interval="week".
        """
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=days)
        records = self._k.historical_data(
            instrument_token = token,
            from_date        = from_dt,
            to_date          = to_dt,
            interval         = interval,
            continuous       = False,
            oi               = False,
        )
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    # ── 5. PLACE ORDER ────────────────────────────────────────
    def place_order(self, symbol: str, exchange: str, txn_type: str,
                    qty: int, order_type: str = "MARKET",
                    price: float = 0.0, tag: str = "ITP") -> Optional[str]:
        """KITE API: kite.place_order(...)"""
        try:
            order_id = self._k.place_order(
                variety          = self._k.VARIETY_REGULAR,
                tradingsymbol    = symbol,
                exchange         = exchange,
                transaction_type = txn_type,
                quantity         = qty,
                order_type       = self._k.ORDER_TYPE_MARKET if order_type == "MARKET"
                                   else self._k.ORDER_TYPE_LIMIT,
                product          = self._k.PRODUCT_CNC if CFG["PRODUCT"] == "CNC"
                                   else self._k.PRODUCT_MIS,
                price            = price if order_type == "LIMIT" else 0,
                tag              = tag,
            )
            log.info(f"  [Kite] place_order → {txn_type} {qty} {symbol} | ID: {order_id}")
            return str(order_id)
        except Exception as e:
            log.error(f"  [Kite] place_order FAILED {symbol}: {e}")
            return None

    # ── 6. ORDER BOOK ─────────────────────────────────────────
    def get_orders(self) -> List[dict]:
        """KITE API: kite.orders()"""
        try:
            return self._k.orders()
        except Exception as e:
            log.warning(f"  [Kite] orders(): {e}")
            return []

    # ── 7. POSITIONS ──────────────────────────────────────────
    def get_positions(self) -> dict:
        """KITE API: kite.positions()"""
        try:
            return self._k.positions()
        except Exception as e:
            log.warning(f"  [Kite] positions(): {e}")
            return {"day": [], "net": []}

    # ── 8. MARGINS / CAPITAL ──────────────────────────────────
    def get_available_capital(self) -> float:
        """KITE API: kite.margins(segment='equity')"""
        try:
            margins = self._k.margins(segment="equity")
            bal = float(margins["available"]["live_balance"])
            if bal > 0:
                return bal
            bal = float(margins["available"].get("cash", 0) or
                        margins["available"].get("opening_balance", 0) or 0)
            return bal if bal > 0 else CFG["CAPITAL"]
        except Exception as e:
            log.warning(f"  [Kite] margins(): {e}")
            return CFG["CAPITAL"]

    # ── 9. GTT ────────────────────────────────────────────────
    def place_gtt(self, symbol: str, exchange: str, qty: int,
                  entry: float, sl: float, tp: float) -> Optional[str]:
        """KITE API: kite.place_gtt(...) — OCO SL + TP"""
        try:
            gtt_id = self._k.place_gtt(
                trigger_type   = self._k.GTT_TYPE_OCO,
                tradingsymbol  = symbol,
                exchange       = exchange,
                trigger_values = [sl, tp],
                last_price     = entry,
                orders         = [
                    {
                        "exchange":         exchange,
                        "tradingsymbol":    symbol,
                        "transaction_type": self._k.TRANSACTION_TYPE_SELL,
                        "quantity":         qty,
                        "order_type":       self._k.ORDER_TYPE_MARKET,
                        "product":          self._k.PRODUCT_CNC,
                        "price":            sl * 0.99,
                    },
                    {
                        "exchange":         exchange,
                        "tradingsymbol":    symbol,
                        "transaction_type": self._k.TRANSACTION_TYPE_SELL,
                        "quantity":         qty,
                        "order_type":       self._k.ORDER_TYPE_LIMIT,
                        "product":          self._k.PRODUCT_CNC,
                        "price":            tp,
                    },
                ],
            )
            log.info(f"  [Kite] place_gtt → {symbol} SL:{sl} TP:{tp} | GTT:{gtt_id}")
            return str(gtt_id)
        except Exception as e:
            log.error(f"  [Kite] place_gtt FAILED {symbol}: {e}")
            return None

    # ── 10. PROFILE ───────────────────────────────────────────
    def get_profile(self) -> dict:
        """KITE API: kite.profile()"""
        try:
            return self._k.profile()
        except Exception as e:
            log.warning(f"  [Kite] profile(): {e}")
            return {}


# ═════════════════════════════════════════════════════════════
# ███  UNIVERSE MANAGER  ██████████████████████████████████████
# ═════════════════════════════════════════════════════════════
class UniverseManager:

    def __init__(self, kl: KiteLayer):
        self._kl    = kl
        self._cache = None

    def get_universe(self) -> pd.DataFrame:
        if self._cache is None or self._cache.empty:
            self._cache = self._kl.get_all_nse_instruments()
        return self._cache

    def get_gap_scan_universe(self) -> List[dict]:
        df = self.get_universe()
        return df[["symbol", "token"]].to_dict("records")

    def filter_for_deep_scan(self, quote_data: dict) -> List[dict]:
        # Suffixes that indicate non-equity instruments — skip them
        EXCLUDE_SUFFIXES = (
            "-BE", "-BL", "-BZ", "-IL", "-SM", "-GS",
            "NIFTY", "SENSEX", "BANKNIFTY", "FINNIFTY",
        )
        df = self.get_universe()
        token_map = dict(zip(df["symbol"], df["token"]))
        candidates = []
        for sym, q in quote_data.items():
            # Skip index derivatives, bond ETFs, illiquid instruments
            if any(sym.upper().endswith(sfx) or sym.upper().startswith(sfx)
                   for sfx in EXCLUDE_SUFFIXES):
                continue
            ltp        = q.get("ltp", 0)
            prev_close = q.get("prev_close", 0)
            volume     = q.get("volume", 0)
            open_p     = q.get("open", 0)
            if not prev_close or ltp < CFG["MIN_PRICE"]:
                continue
            gap_pct    = (open_p - prev_close) / prev_close * 100 if prev_close else 0
            change_pct = (ltp - prev_close) / prev_close * 100 if prev_close else 0
            gap_score  = max(gap_pct, 0)
            vol_score  = volume / 1_000_000
            candidates.append({
                "symbol":     sym,
                "token":      token_map.get(sym, 0),
                "ltp":        ltp,
                "prev_close": prev_close,
                "open":       open_p,
                "gap_pct":    round(gap_pct, 2),
                "change_pct": round(change_pct, 2),
                "volume":     volume,
                "rank_score": gap_score * 2 + vol_score,
            })
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)
        gap_ups = [c for c in candidates if c["gap_pct"] >= CFG["GAP_UP_MIN_PCT"]]
        rest    = [c for c in candidates if c["gap_pct"] < CFG["GAP_UP_MIN_PCT"]]
        final   = gap_ups + rest[:max(0, CFG["UNIVERSE_CAP"] - len(gap_ups))]
        log.info(f"  [Universe] {len(candidates)} stocks → {len(gap_ups)} gap-ups + "
                 f"{len(final)-len(gap_ups)} top volume = {len(final)} for deep scan")
        return final


# ═════════════════════════════════════════════════════════════
# ███  NSE PUBLIC API CLIENT  █████████████████████████████████
# ═════════════════════════════════════════════════════════════
class NSEClient:
    BASE    = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/json, text/plain, */*",
        "Referer":    "https://www.nseindia.com/",
    }

    def __init__(self):
        self.sess    = requests.Session()
        self.sess.headers.update(self.HEADERS)
        self._cache  = {}          # prefetch cache: key → data
        self._warm_up()

    def prefetch_scan_data(self):
        """
        Batch-warm the NSE session and cache VIX, FII/DII, block-deal data
        before the per-stock loop begins.  Delivery% is still fetched per-stock
        but with a short timeout since the session is already warm.
        Call once between STEP 4 and STEP 5.
        """
        log.info("  [NSE] Prefetching VIX / FII / block-deals …")
        self._cache["vix"]         = self._get("allIndices")
        self._cache["fii"]         = self._get("fiidiiTradeReact")
        self._cache["block_deals"] = self._get("block-deal")
        log.info("  [NSE] Prefetch complete — delivery% will use warmed session")

    def _warm_up(self):
        try:
            self.sess.get(self.BASE, timeout=10)
            self.sess.get(f"{self.BASE}/market-data/live-equity-market", timeout=10)
        except Exception:
            pass

    def _get(self, path: str, params: dict = None) -> dict:
        try:
            r = self.sess.get(f"{self.BASE}/api/{path}", params=params, timeout=15)
            if r.status_code == 401:
                self._warm_up()
                r = self.sess.get(f"{self.BASE}/api/{path}", params=params, timeout=15)
            return r.json() if r.ok else {}
        except Exception as e:
            log.warning(f"  [NSE] {path}: {e}")
            return {}

    def get_fii_dii(self) -> dict:
        data = self._get("fiidiiTradeReact")
        res  = {"fii_net": 0.0, "dii_net": 0.0, "fii_bullish": False, "dii_bullish": False}
        try:
            rows = data if isinstance(data, list) else (data.get("data") or [])
            if rows:
                row      = rows[0]
                fii_buy  = float(row.get("buyValue",      row.get("fiiBuyValue",  0)) or 0)
                fii_sell = float(row.get("sellValue",     row.get("fiiSellValue", 0)) or 0)
                dii_buy  = float(row.get("dii_buyValue",  row.get("diiBuyValue",  0)) or 0)
                dii_sell = float(row.get("dii_sellValue", row.get("diiSellValue", 0)) or 0)
                res["fii_net"]     = fii_buy - fii_sell
                res["dii_net"]     = dii_buy - dii_sell
                res["fii_bullish"] = bool(res["fii_net"] > 0)
                res["dii_bullish"] = bool(res["dii_net"] > 0)
        except Exception as e:
            log.warning(f"  [NSE] FII parse: {e}")
        return res

    def get_delivery_pct(self, symbol: str) -> dict:
        res = {"delivery_pct": 0.0, "institutional": False}
        try:
            r = self.sess.get(
                f"{self.BASE}/api/deliveryToTrading",
                params={"series": "EQ", "symbol": symbol},
                timeout=3,      # fast — session already warm from prefetch
            )
            if r.status_code == 401:
                self._warm_up()
                r = self.sess.get(
                    f"{self.BASE}/api/deliveryToTrading",
                    params={"series": "EQ", "symbol": symbol},
                    timeout=3,
                )
            data = r.json() if r.ok else {}
            if data and "data" in data and data["data"]:
                pct = float(data["data"][-1].get("deliveryToTradedQty", 0))
                res["delivery_pct"]  = pct
                res["institutional"] = pct >= CFG["MIN_DELIVERY_PCT"]
        except Exception:
            pass
        return res

    def get_option_chain_pcr(self, symbol: str = "NIFTY") -> dict:
        endpoint = "option-chain-indices" if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"] \
                   else "option-chain-equities"
        data = self._get(endpoint, {"symbol": symbol})
        res  = {"pcr": 1.0, "bullish_oi": False, "total_ce_oi": 0, "total_pe_oi": 0}
        try:
            if data and "filtered" in data:
                ce_oi = sum(r["CE"].get("openInterest", 0) for r in data["filtered"]["data"] if "CE" in r)
                pe_oi = sum(r["PE"].get("openInterest", 0) for r in data["filtered"]["data"] if "PE" in r)
                if ce_oi:
                    res.update({
                        "pcr":         round(pe_oi / ce_oi, 2),
                        "bullish_oi":  (pe_oi / ce_oi) >= CFG["PCR_BULLISH_MIN"],
                        "total_ce_oi": ce_oi,
                        "total_pe_oi": pe_oi,
                    })
        except Exception as e:
            log.warning(f"  [NSE] PCR parse: {e}")
        return res

    def get_india_vix(self) -> float:
        data = self._get("allIndices")
        try:
            if data and "data" in data:
                for idx in data["data"]:
                    name = (idx.get("index") or idx.get("indexSymbol") or "").upper()
                    if "VIX" in name:
                        return float(idx.get("last") or idx.get("lastPrice") or 15.0)
        except Exception:
            pass
        return 15.0

    def get_block_deals(self, symbol: str) -> dict:
        # Use prefetched cache if available — avoids one NSE call per stock
        data = self._cache.get("block_deals") or self._get("block-deal")
        res  = {"block_buy": False, "deals": []}
        try:
            if data and "data" in data:
                buys = [d for d in data["data"]
                        if d.get("symbol","").upper() == symbol.upper()
                        and d.get("buySell","").upper() == "BUY"]
                res["block_buy"] = len(buys) > 0
                res["deals"]     = buys[:3]
        except Exception:
            pass
        return res


# ═════════════════════════════════════════════════════════════
# ███  TECHNICAL ENGINE  ██████████████████████████████████████
# ═════════════════════════════════════════════════════════════
class TechnicalEngine:

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 30:
            return pd.DataFrame()
        df = df.copy()
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

        df["ema_fast"]  = close.ewm(span=CFG["EMA_FAST"],  adjust=False).mean()
        df["ema_slow"]  = close.ewm(span=CFG["EMA_SLOW"],  adjust=False).mean()
        df["ema_trend"] = close.ewm(span=CFG["EMA_TREND"], adjust=False).mean()

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(CFG["RSI_PERIOD"]).mean()
        loss  = (-delta.clip(upper=0)).rolling(CFG["RSI_PERIOD"]).mean()
        df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low  - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(CFG["ATR_PERIOD"]).mean()

        up  = high - high.shift()
        dn  = low.shift() - low
        pdm = up.where((up > dn) & (up > 0), 0.0)
        ndm = dn.where((dn > up) & (dn > 0), 0.0)
        atr_s = tr.rolling(CFG["ADX_PERIOD"]).mean()
        dip   = 100 * pdm.rolling(CFG["ADX_PERIOD"]).mean() / atr_s
        dim   = 100 * ndm.rolling(CFG["ADX_PERIOD"]).mean() / atr_s
        dx    = (dip - dim).abs() / (dip + dim).replace(0, np.nan) * 100
        df["adx"]      = dx.rolling(CFG["ADX_PERIOD"]).mean()
        df["di_plus"]  = dip
        df["di_minus"] = dim

        hl2   = (high + low) / 2
        st_at = tr.rolling(CFG["ST_ATR_PERIOD"]).mean()
        ub    = hl2 + CFG["ST_MULTIPLIER"] * st_at
        lb    = hl2 - CFG["ST_MULTIPLIER"] * st_at
        st    = [np.nan] * len(df)
        dr    = [1] * len(df)
        for i in range(1, len(df)):
            c = close.iloc[i]
            dr[i] = 1 if c > ub.iloc[i-1] else (-1 if c < lb.iloc[i-1] else dr[i-1])
            st[i] = lb.iloc[i] if dr[i] == 1 else ub.iloc[i]
        df["supertrend"] = st
        df["st_bull"]    = [d == 1 for d in dr]

        df["vol_ma"]    = vol.rolling(CFG["VOL_MA_PERIOD"]).mean()
        df["vol_surge"] = vol > CFG["VOL_SURGE_X"] * df["vol_ma"]

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"]      = ema12 - ema26
        df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_bull"] = df["macd"] > df["macd_sig"]

        _52w_window       = min(252, len(df))
        df["high_52w"]    = high.rolling(_52w_window).max()
        df["at_52w_high"] = close >= df["high_52w"].shift(1).fillna(high)

        op = df["open"]
        body_sz  = (close - op).abs()
        rng      = high - low
        lwr_wick = pd.concat([op, close], axis=1).min(axis=1) - low
        upr_wick = high - pd.concat([op, close], axis=1).max(axis=1)
        bull_eng = (close > op) & (close.shift() < op.shift()) & (close > op.shift()) & (op < close.shift())
        hammer   = (lwr_wick > 2 * body_sz) & (upr_wick < 0.3 * rng) & (close > op)
        df["bull_candle"] = bull_eng | hammer

        df["cross_up"]   = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift() <= df["ema_slow"].shift())
        df["cross_down"] = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift() >= df["ema_slow"].shift())

        _core = ["ema_fast", "ema_slow", "ema_trend", "rsi", "atr",
                 "adx", "macd", "macd_sig", "supertrend", "vol_ma"]
        return df.dropna(subset=_core).reset_index(drop=True)

    @staticmethod
    def score(row: pd.Series) -> Tuple[int, List[str]]:
        score = 0; flags = []

        ema_bull = float(row["ema_fast"]) > float(row["ema_slow"]) \
                   and float(row["close"]) > float(row["ema_fast"])
        if row.get("cross_up", False):
            score += 1; flags.append("✅ EMA Golden Cross (fresh)")
        elif ema_bull:
            score += 1; flags.append("✅ Price above both EMAs")
        else:
            flags.append("❌ EMA bearish structure")

        rsi = float(row.get("rsi", 0))
        if CFG["RSI_BUY_MIN"] < rsi < CFG["RSI_BUY_MAX"]:
            score += 1; flags.append(f"✅ RSI {rsi:.1f} (50–70 zone)")
        elif rsi >= CFG["RSI_BUY_MAX"]:
            flags.append(f"⚠️ RSI {rsi:.1f} overbought")
        else:
            flags.append(f"❌ RSI {rsi:.1f} weak")

        adx = float(row.get("adx", 0))
        if adx >= CFG["ADX_MIN"] and float(row.get("di_plus",0)) > float(row.get("di_minus",0)):
            score += 1; flags.append(f"✅ ADX {adx:.1f} strong trend")
        else:
            flags.append(f"❌ ADX {adx:.1f} no trend")

        if row.get("st_bull", False):
            score += 1; flags.append("✅ Supertrend bullish")
        else:
            flags.append("❌ Supertrend bearish")

        if row.get("vol_surge", False) and row.get("bull_candle", False):
            score += 1; flags.append("✅ Volume surge + bullish candle")
        elif row.get("vol_surge", False):
            flags.append("⚠️ Volume surge, no candle confirm")
        else:
            flags.append("❌ No volume confirmation")

        return score, flags


# ═════════════════════════════════════════════════════════════
# ███  BREAKOUT SCANNER  ██████████████████████████████████████
# ═════════════════════════════════════════════════════════════
class BreakoutScanner:

    @staticmethod
    def score(df: pd.DataFrame, row: pd.Series,
              nifty_df: pd.DataFrame, quote: dict) -> Tuple[int, List[str]]:
        score = 0; flags = []

        if row.get("at_52w_high", False):
            score += 1; flags.append("✅ 52-Week High Breakout")
        else:
            flags.append("❌ Not at 52w high")

        gap_pct = quote.get("gap_pct", 0)
        if gap_pct >= CFG["GAP_UP_MIN_PCT"]:
            score += 1; flags.append(f"✅ Gap-Up {gap_pct:.1f}% (event-driven)")
        elif gap_pct > 0:
            flags.append(f"⚠️ Small gap {gap_pct:.1f}%")
        else:
            flags.append(f"❌ No gap (or gap-down {gap_pct:.1f}%)")

        try:
            stk_ret   = df["close"].pct_change(CFG["RS_PERIOD"]).iloc[-1]
            nifty_ret = nifty_df["close"].pct_change(CFG["RS_PERIOD"]).iloc[-1]
            rs = stk_ret - nifty_ret
            if rs > 0.03:
                score += 1; flags.append(f"✅ RS vs NIFTY +{rs*100:.1f}%")
            elif rs > 0:
                flags.append(f"⚠️ Slight RS +{rs*100:.1f}%")
            else:
                flags.append(f"❌ Underperforming NIFTY {rs*100:.1f}%")
        except Exception:
            flags.append("⚠️ RS data N/A")

        try:
            recent_rng = df["high"].tail(CFG["CONSOL_DAYS"]).max() \
                       - df["low"].tail(CFG["CONSOL_DAYS"]).min()
            avg_rng    = (df["high"] - df["low"]).mean()
            tight      = recent_rng < 0.6 * avg_rng * CFG["CONSOL_DAYS"]
            if tight and row.get("vol_surge", False):
                score += 1; flags.append("✅ Tight consolidation breakout")
            else:
                flags.append("❌ No consolidation pattern")
        except Exception:
            flags.append("⚠️ Pattern N/A")

        if row.get("macd_bull", False) and float(row.get("macd", -1)) > 0:
            score += 1; flags.append("✅ MACD bullish above zero")
        elif row.get("macd_bull", False):
            flags.append("⚠️ MACD bullish below zero")
        else:
            flags.append("❌ MACD bearish")

        return score, flags


# ═════════════════════════════════════════════════════════════
# ███  FUNDAMENTAL ENGINE (Kite-Native Proxy) ████████████████
# ─────────────────────────────────────────────────────────────
# Yahoo Finance REMOVED. All 5 fundamental points now computed
# entirely from Kite OHLCV data — zero external API calls.
# Zero rate-limit risk. Scan stays at ~3 minutes.
#
# Proxy mapping:
#   Old Yahoo metric     →  New Kite-native proxy
#   ─────────────────       ─────────────────────────────────
#   Revenue Growth       →  Volume accumulation (5d vs 20d avg)
#   PAT Growth           →  Price near 52-week high
#   D/E Ratio            →  Supertrend sustained 10+ days
#   ROE                  →  Relative strength vs NIFTY > 5%
#   Promoter Holding     →  MACD histogram sustained positive
# ═════════════════════════════════════════════════════════════
class FundamentalEngine:

    @staticmethod
    def compute(df: pd.DataFrame, row: pd.Series,
                nifty_df: pd.DataFrame) -> dict:
        """
        Score 5 fundamental-proxy points using only Kite OHLCV data.
        No Yahoo Finance. No rate limits. Instant results.

        Parameters
        ----------
        df       : OHLCV dataframe with computed technical indicators
        row      : last row of df (df.iloc[-1])
        nifty_df : NIFTY 50 OHLCV with computed indicators
        """
        score = 0
        flags = []

        if df is None or len(df) < 50:
            return {
                "score": 2, "max": 5,
                "flags": ["⚠️ Insufficient history — neutral fundamental (2/5)"],
                "available": False,
                "pe": None, "roe": None, "revenue_growth": None,
                "pat_growth": None, "de_ratio": None, "promoter_pct": None,
                "mcap_cr": None,
            }

        # ── F1: Volume accumulation trend ─────────────────────
        # Rising institutional demand = proxy for revenue growth.
        # 5-day avg volume > 1.3x 20-day avg volume.
        try:
            vol_5d  = float(df["volume"].tail(5).mean())
            vol_20d = float(df["volume"].tail(20).mean())
            ratio   = vol_5d / vol_20d if vol_20d > 0 else 0
            if ratio >= CFG["VOL_ACCUM_X"]:
                score += 1
                flags.append(f"✅ Volume accumulation {ratio:.1f}x (institutional demand)")
            else:
                flags.append(f"❌ Volume flat {ratio:.1f}x 20d avg (no accumulation)")
        except Exception:
            score += 1; flags.append("⚠️ Volume trend N/A — neutral")

        # ── F2: Price near 52-week high ───────────────────────
        # Strong earnings companies make 52w highs. Proxy for PAT growth.
        try:
            w = min(252, len(df))
            high_52w  = float(df["high"].tail(w).max())
            ltp       = float(row["close"])
            proximity = ltp / high_52w if high_52w > 0 else 0
            threshold = CFG["HIGH_52W_PCT"] / 100.0
            if proximity >= threshold:
                score += 1
                flags.append(f"✅ Price at {proximity*100:.1f}% of 52w high (PAT momentum)")
            else:
                flags.append(f"❌ Price only {proximity*100:.1f}% of 52w high (weak)")
        except Exception:
            score += 1; flags.append("⚠️ 52w high N/A — neutral")

        # ── F3: Supertrend sustained bullish ─────────────────
        # Low-debt companies trend cleanly without volatile reversals.
        # Proxy for healthy D/E ratio.
        try:
            if "st_bull" in df.columns:
                bull_days = int(df["st_bull"].tail(10).sum())
                if bull_days >= CFG["ST_BULL_DAYS"]:
                    score += 1
                    flags.append(f"✅ Supertrend bullish {bull_days}/10 days (D/E health proxy)")
                else:
                    flags.append(f"❌ Supertrend bullish only {bull_days}/10 days (choppy)")
            else:
                score += 1; flags.append("⚠️ Supertrend N/A — neutral")
        except Exception:
            score += 1; flags.append("⚠️ Supertrend history N/A — neutral")

        # ── F4: Relative strength vs NIFTY (20-day) ──────────
        # High-ROE companies consistently outperform the index.
        # Proxy for return on equity.
        try:
            stk_ret   = float(df["close"].pct_change(20).iloc[-1])
            nifty_ret = float(nifty_df["close"].pct_change(20).iloc[-1])
            rs_pct    = (stk_ret - nifty_ret) * 100
            threshold = CFG["RS_FUND_PCT"]
            if rs_pct >= threshold:
                score += 1
                flags.append(f"✅ RS vs NIFTY +{rs_pct:.1f}% over 20d (ROE proxy)")
            else:
                flags.append(f"❌ RS vs NIFTY {rs_pct:.1f}% — underperforming (low ROE proxy)")
        except Exception:
            score += 1; flags.append("⚠️ RS data N/A — neutral")

        # ── F5: MACD histogram sustained positive ────────────
        # Strong fundamental companies show sustained, expanding MACD.
        # Proxy for promoter conviction / insider confidence.
        try:
            if "macd" in df.columns and "macd_sig" in df.columns:
                hist      = df["macd"] - df["macd_sig"]
                pos_days  = int((hist.tail(5) > 0).sum())
                expanding = (len(hist) >= 3 and
                             float(hist.iloc[-1]) > float(hist.iloc[-2]) > float(hist.iloc[-3]))
                if pos_days >= CFG["MACD_POS_DAYS"] and expanding:
                    score += 1
                    flags.append(f"✅ MACD histogram +ve {pos_days}/5 days + expanding (conviction)")
                elif pos_days >= 3:
                    flags.append(f"⚠️ MACD positive {pos_days}/5 days (moderate conviction)")
                else:
                    flags.append(f"❌ MACD negative/mixed ({pos_days}/5 positive)")
            else:
                score += 1; flags.append("⚠️ MACD data N/A — neutral")
        except Exception:
            score += 1; flags.append("⚠️ MACD history N/A — neutral")

        return {
            "score": score, "max": 5, "flags": flags,
            "available": True,
            # Yahoo fields set to None — no longer fetched
            "pe": None, "roe": None, "revenue_growth": None,
            "pat_growth": None, "de_ratio": None,
            "promoter_pct": None, "mcap_cr": None,
        }


# ═════════════════════════════════════════════════════════════
# ███  FIBONACCI ENGINE  ██████████████████████████████████████
# ─────────────────────────────────────────────────────────────
# Computes Fibonacci retracement and extension levels from
# Kite OHLCV data. Scores 3 points based on:
#
#   Fib-1: Price in Golden Zone (38.2%–61.8% retracement)
#           The highest-probability institutional entry zone.
#
#   Fib-2: Price resting AT a key Fib level (within 2%)
#           Indicates price is pausing at a known support level.
#
#   Fib-3: 1.618 extension target available (profitable trade setup)
#           Provides a market-structure-based take-profit target.
#
# WORKS FOR STOCKS AND TRADES EQUALLY:
#   Swing trading   → Daily candles (current default)
#   Stock investing → Weekly candles (set TRADE_MODE=positional)
#   Intraday        → 15min candles
#   The logic is identical — only the candle interval changes.
#
# SMC CONFLUENCE BOOST:
#   When Fib Golden Zone aligns with an Order Block or FVG,
#   it creates a "double confluence" — the highest-probability
#   setup in Smart Money trading. SMCEngine flags this explicitly.
# ═════════════════════════════════════════════════════════════
class FibonacciEngine:

    # Standard retracement ratios (from swing high downward)
    FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    # Extension ratios (from swing low upward — take-profit targets)
    FIB_EXTENSIONS = [1.272, 1.618, 2.0, 2.618]

    @staticmethod
    def compute(df: pd.DataFrame) -> dict:
        """
        Compute Fibonacci levels and score entry quality.

        Returns
        -------
        dict with keys:
            score              : int (0–3)
            max                : int (3)
            flags              : list[str]
            swing_high         : float
            swing_low          : float
            golden_zone_high   : float  (38.2% retracement)
            golden_zone_low    : float  (61.8% retracement)
            fib_50             : float  (50% retracement)
            fib_tp_1618        : float  (1.618 extension — best TP target)
            price_in_golden    : bool
            fib_at_key_level   : bool
            levels             : dict  (all computed levels)
        """
        _empty = {
            "score": 0, "max": 3,
            "flags": ["⚠️ Fibonacci: insufficient data"],
            "swing_high": None, "swing_low": None,
            "golden_zone_high": None, "golden_zone_low": None,
            "fib_50": None, "fib_tp_1618": None,
            "price_in_golden": False, "fib_at_key_level": False,
            "levels": {}
        }

        if df is None or len(df) < 20:
            return _empty

        score = 0
        flags = []

        # ── Swing high and low from last N candles ────────────
        lookback   = min(CFG["FIB_LOOKBACK"], len(df))
        recent     = df.tail(lookback)
        swing_high = float(recent["high"].max())
        swing_low  = float(recent["low"].min())
        rng        = swing_high - swing_low

        # Reject flat price action — Fib is meaningless in sideways chop
        if rng < 0.005 * swing_high:
            return {**_empty, "flags": ["⚠️ Fibonacci: price range too flat (<0.5%) — no valid levels"]}

        ltp = float(df["close"].iloc[-1])

        # ── Compute all Fibonacci levels ───────────────────────
        levels = {}
        for ratio in FibonacciEngine.FIB_RATIOS:
            price = swing_high - ratio * rng
            key   = f"{ratio*100:.1f}%"
            levels[key] = round(price, 2)

        for ratio in FibonacciEngine.FIB_EXTENSIONS:
            price = swing_low + ratio * rng
            key   = f"{ratio*100:.1f}% ext"
            levels[key] = round(price, 2)

        golden_zone_high = levels["38.2%"]   # upper bound of golden zone
        golden_zone_low  = levels["61.8%"]   # lower bound of golden zone
        fib_50           = levels["50.0%"]
        fib_tp_1618      = levels["161.8% ext"]

        # ── Fib-1: Price in Golden Zone ───────────────────────
        # Tolerance = 1% of total range on each side
        tol = rng * 0.01
        in_golden = bool((golden_zone_low - tol) <= ltp <= (golden_zone_high + tol))

        if in_golden:
            score += 1
            flags.append(
                f"✅ Fib: Price ₹{ltp:.1f} in Golden Zone "
                f"(₹{golden_zone_low:.1f}–₹{golden_zone_high:.1f}) — prime entry"
            )
        elif ltp < golden_zone_low - tol:
            pct_below = (golden_zone_low - ltp) / rng * 100
            flags.append(
                f"⚠️ Fib: Price below Golden Zone by {pct_below:.1f}% range "
                f"— deeper retracement (may find 78.6% support at ₹{levels['78.6%']:.1f})"
            )
        else:
            pct_above = (ltp - golden_zone_high) / rng * 100
            flags.append(
                f"❌ Fib: Price above Golden Zone by {pct_above:.1f}% range "
                f"— no pullback entry (wait for retrace to ₹{golden_zone_high:.1f})"
            )

        # ── Fib-2: Price resting AT a key Fib level ──────────
        # Check if price is within FIB_TOLERANCE (2%) of any key level
        key_levels = {
            "23.6%": levels["23.6%"],
            "38.2%": levels["38.2%"],
            "50.0%": levels["50.0%"],
            "61.8%": levels["61.8%"],
            "78.6%": levels["78.6%"],
        }
        at_key   = False
        key_name = None
        key_px   = None
        for name, px in key_levels.items():
            if px > 0 and abs(ltp - px) / px <= CFG["FIB_TOLERANCE"]:
                at_key   = True
                key_name = name
                key_px   = px
                break

        if at_key:
            score += 1
            dist_pct = abs(ltp - key_px) / key_px * 100
            flags.append(
                f"✅ Fib: Price at {key_name} level ₹{key_px:.1f} "
                f"(within {dist_pct:.1f}%) — institutional support zone"
            )
        else:
            # Show nearest level
            nearest_name, nearest_px = min(
                key_levels.items(), key=lambda x: abs(x[1] - ltp)
            )
            dist = abs(ltp - nearest_px) / nearest_px * 100
            flags.append(
                f"❌ Fib: Not at key level — nearest is {nearest_name} "
                f"₹{nearest_px:.1f} ({dist:.1f}% away)"
            )

        # ── Fib-3: 1.618 extension as take-profit ────────────
        # Valid only when current price is below the extension target.
        # This becomes the recommended TP instead of ATR-based TP.
        if ltp < fib_tp_1618:
            upside_pct = (fib_tp_1618 - ltp) / ltp * 100
            score += 1
            flags.append(
                f"✅ Fib: 1.618 ext TP target ₹{fib_tp_1618:.1f} "
                f"({upside_pct:.1f}% upside from current price)"
            )
        else:
            flags.append(
                f"⚠️ Fib: Price already above 1.618 extension ₹{fib_tp_1618:.1f} "
                f"— extension exhausted, consider 2.0 ext ₹{levels.get('200.0% ext', 0):.1f}"
            )

        return {
            "score":            score,
            "max":              3,
            "flags":            flags,
            "swing_high":       round(swing_high,       2),
            "swing_low":        round(swing_low,        2),
            "golden_zone_high": round(golden_zone_high, 2),
            "golden_zone_low":  round(golden_zone_low,  2),
            "fib_50":           round(fib_50,           2),
            "fib_tp_1618":      round(fib_tp_1618,      2),
            "price_in_golden":  in_golden,
            "fib_at_key_level": at_key,
            "levels":           levels,
        }


# ═════════════════════════════════════════════════════════════
# ███  INSTITUTIONAL INTELLIGENCE  ████████████████████████████
# ═════════════════════════════════════════════════════════════
class InstitutionalEngine:

    @staticmethod
    def score(nse: NSEClient, symbol: str,
              fii_data: dict, oc_data: dict, vix: float) -> Tuple[int, List[str]]:
        score = 0; flags = []

        fn = fii_data.get("fii_net", 0)
        if fii_data.get("fii_bullish"):
            score += 1; flags.append(f"✅ FII Net Buy ₹{fn/1e7:.0f}Cr")
        else:
            flags.append(f"❌ FII Net Sell ₹{abs(fn)/1e7:.0f}Cr")

        d    = nse.get_delivery_pct(symbol)
        dpct = d["delivery_pct"]
        if d["institutional"]:
            score += 1; flags.append(f"✅ Delivery {dpct:.0f}% (institutional)")
        else:
            flags.append(f"❌ Delivery {dpct:.0f}% (low)")

        pcr = oc_data.get("pcr", 1.0)
        if oc_data.get("bullish_oi"):
            score += 1; flags.append(f"✅ PCR {pcr} bullish OI")
        else:
            flags.append(f"❌ PCR {pcr} bearish")

        if vix < CFG["VIX_MAX"]:
            score += 1; flags.append(f"✅ VIX {vix:.1f} low fear")
        elif vix < 25:
            flags.append(f"⚠️ VIX {vix:.1f} elevated")
        else:
            flags.append(f"❌ VIX {vix:.1f} high fear")

        bd = nse.get_block_deals(symbol)
        if bd["block_buy"]:
            score += 1; flags.append("✅ Block deal buying detected")
        else:
            flags.append("❌ No block deals")

        return score, flags


# ═════════════════════════════════════════════════════════════
# ███  MARKET REGIME  █████████████████████████████████████████
# ═════════════════════════════════════════════════════════════
class MarketRegime:

    @staticmethod
    def check(nifty_df: pd.DataFrame, vix: float) -> Tuple[int, List[str], bool, str]:
        score = 0; flags = []
        try:
            if nifty_df is None or len(nifty_df) < 3:
                log.warning("  [Regime] NIFTY data insufficient — using neutral regime")
                return 1, ["⚠️ NIFTY data unavailable (market may be closed)"], True, "NEUTRAL ⚪"

            ema200  = nifty_df["close"].ewm(span=min(200, len(nifty_df)), adjust=False).mean()
            last    = float(nifty_df["close"].iloc[-1])
            ema_val = float(ema200.iloc[-1])

            if last > ema_val:
                score += 1; flags.append(f"✅ NIFTY {last:.0f} > 200EMA {ema_val:.0f}")
            else:
                flags.append(f"❌ NIFTY {last:.0f} < 200EMA {ema_val:.0f} (Bear)")

            if vix < CFG["VIX_MAX"]:
                score += 1; flags.append(f"✅ VIX {vix:.1f} calm market")
            else:
                flags.append(f"❌ VIX {vix:.1f} high volatility")

            bullish = (score >= 1)
            label   = "BULL 🟢" if last > ema_val else "BEAR 🔴"
        except Exception as e:
            log.warning(f"  [Regime] Error: {e}")
            bullish = True; label = "NEUTRAL ⚪"
            flags.append(f"⚠️ Regime check error: {str(e)[:50]}")
        return score, flags, bullish, label


# ═════════════════════════════════════════════════════════════
# ███  RISK MANAGER  ██████████████████████████████████████████
# ═════════════════════════════════════════════════════════════
class RiskManager:

    @staticmethod
    def position_size(entry: float, sl: float,
                      available_capital: float) -> int:
        risk_amt    = available_capital * CFG["RISK_PCT"]
        risk_per_sh = entry - sl
        if risk_per_sh <= 0:
            return 0
        qty = math.floor(risk_amt / risk_per_sh)
        max_by_cap = math.floor(available_capital * 0.20 / entry)
        return max(min(qty, max_by_cap), 1)

    @staticmethod
    def sl_tp(entry: float, atr: float,
              fib_d: dict = None) -> Tuple[float, float]:
        """
        Compute stop-loss and take-profit.

        Stop-loss: Always ATR-based (objective, volatility-adjusted).
        Take-profit: Fibonacci 1.618 extension when available and valid,
                     otherwise ATR × 3.5 (fallback).

        The Fib TP is preferred because it aligns with a market-structure
        target — where institutional sellers are likely to emerge —
        rather than an arbitrary volatility multiple.
        """
        sl     = round(entry - CFG["ATR_SL"] * atr, 2)
        atr_tp = round(entry + CFG["ATR_TP"] * atr, 2)

        # Use Fib 1.618 extension as TP when:
        # 1. Fibonacci scored at least 2/3 (meaningful setup)
        # 2. Fib TP is above entry by at least 2%
        # 3. Fib TP is below 3x the ATR TP distance (not astronomically far)
        if fib_d and fib_d.get("score", 0) >= 2:
            fib_tp = fib_d.get("fib_tp_1618")
            if (fib_tp and fib_tp > entry * 1.02
                    and fib_tp <= entry + 3 * (atr_tp - entry)):
                return sl, round(fib_tp, 2)

        return sl, atr_tp

    @staticmethod
    def trailing_stop(pos: dict, ltp: float, atr: float) -> dict:
        entry  = pos["entry"]
        sl     = pos["sl"]
        profit = ltp - entry
        new_sl = sl

        if profit >= 2.0 * atr:   new_sl = ltp - 1.0 * atr
        elif profit >= 1.5 * atr: new_sl = entry + 0.5 * atr
        elif profit >= 1.0 * atr: new_sl = entry

        if new_sl > sl:
            return {**pos, "sl": round(new_sl, 2), "trailing": True}
        return pos

    @staticmethod
    def check_exit(ltp: float, pos: dict, row: pd.Series) -> Optional[str]:
        if ltp <= pos["sl"]:                              return "STOP_LOSS"
        if ltp >= pos["tp"]:                              return "TARGET_HIT"
        if row.get("cross_down", False):                  return "EMA_DEATH_CROSS"
        if float(row.get("rsi", 100)) < CFG["RSI_EXIT"]: return "RSI_WEAKNESS"
        if not row.get("st_bull", True):                  return "SUPERTREND_FLIP"
        return None


# ═════════════════════════════════════════════════════════════
# ███  SMC ENGINE — Smart Money Concepts  █████████████████████
# ─────────────────────────────────────────────────────────────
# Identifies institutional footprints:
#   S1. Bullish Order Block (OB) — last bearish candle before
#       a strong upward impulse. Price returning to OB = entry.
#       Now enhanced: checks for Fib + OB confluence.
#
#   S2. Break of Structure (BOS) — price breaks above prior
#       swing high, confirming bullish market structure shift.
#
#   S3. Fair Value Gap (FVG) / Imbalance — 3-candle pattern
#       where candle[i-2].high < candle[i].low (bullish gap).
#       Now enhanced: checks for Fib + FVG confluence.
#
#   S4. Liquidity Sweep — price briefly dips below a swing low
#       then reverses up strongly (stop-hunt by smart money).
#
#   S5. Bullish CHoCH — Change of Character: first higher high
#       after a series of lower highs (structural reversal).
#
# FIBONACCI CONFLUENCE (new in v2):
#   When an OB or FVG aligns with a Fibonacci level (within 2%),
#   the flag is upgraded to "HIGH PROBABILITY CONFLUENCE".
#   This is the most powerful setup in SMC trading.
#   The confluence is annotated in flags but does not change the
#   SMC score — Fib scoring is handled by FibonacciEngine separately.
# ═════════════════════════════════════════════════════════════
class SMCEngine:

    @staticmethod
    def _get_fib_levels(df: pd.DataFrame) -> dict:
        """Quick Fib computation for confluence checking inside SMC."""
        try:
            lookback   = min(CFG["FIB_LOOKBACK"], len(df))
            recent     = df.tail(lookback)
            swing_high = float(recent["high"].max())
            swing_low  = float(recent["low"].min())
            rng        = swing_high - swing_low
            if rng < 0.005 * swing_high:
                return {}
            return {
                "23.6": round(swing_high - 0.236 * rng, 2),
                "38.2": round(swing_high - 0.382 * rng, 2),
                "50.0": round(swing_high - 0.500 * rng, 2),
                "61.8": round(swing_high - 0.618 * rng, 2),
                "78.6": round(swing_high - 0.786 * rng, 2),
            }
        except Exception:
            return {}

    @staticmethod
    def _fib_confluence(price: float, fib_levels: dict) -> Optional[str]:
        """Return Fib level label if price is within 2% of a key level."""
        tol = CFG["FIB_TOLERANCE"]
        for label, px in fib_levels.items():
            if px > 0 and abs(price - px) / px <= tol:
                return f"{label}%"
        return None

    @staticmethod
    def compute(df: pd.DataFrame) -> dict:
        """
        Run all SMC detections on OHLCV dataframe.
        Returns dict with scores, flags, raw data, and Fib confluence notes.
        """
        if df is None or len(df) < 20:
            return {
                "score": 0, "max": 5,
                "flags": ["⚠️ Insufficient data for SMC"],
                "ob": None, "fvg": None, "bos": False,
                "choch": False, "sweep": False,
            }

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        op    = df["open"].values
        n     = len(df)

        score   = 0
        flags   = []
        results = {"ob": None, "fvg": None, "bos": False, "choch": False, "sweep": False}

        # Pre-compute Fib levels for confluence detection
        fib_levels = SMCEngine._get_fib_levels(df)

        # ── S1: Bullish Order Block + Fib Confluence ──────────
        ob_found = False
        for i in range(n - 4, max(n - 30, 2), -1):
            bearish = close[i] < op[i]
            if not bearish:
                continue
            future_high  = max(high[i+1:i+4]) if i + 4 <= n else max(high[i+1:])
            impulse_pct  = (future_high - close[i]) / close[i] * 100
            if impulse_pct >= 1.0:
                ltp     = close[-1]
                ob_high = op[i]
                ob_low  = close[i]
                near_ob = ob_low * 0.97 <= ltp <= ob_high * 1.05
                ob_found = True
                results["ob"] = {
                    "ob_high": round(float(ob_high), 2),
                    "ob_low":  round(float(ob_low),  2),
                    "near":    near_ob,
                    "impulse": round(impulse_pct, 1),
                }
                # Check Fib confluence — is the OB at a Fib level?
                ob_mid     = (ob_high + ob_low) / 2
                fib_label  = SMCEngine._fib_confluence(ob_mid, fib_levels)
                conf_note  = f" + Fib {fib_label} confluence 🔥" if fib_label else ""

                if near_ob:
                    score += 1
                    flags.append(
                        f"✅ SMC: Price at Bullish OB (₹{ob_low:.1f}–₹{ob_high:.1f}){conf_note}"
                    )
                else:
                    flags.append(
                        f"⚠️ SMC: OB at ₹{ob_low:.1f}–₹{ob_high:.1f} "
                        f"(price not retesting yet{conf_note})"
                    )
                break
        if not ob_found:
            flags.append("❌ SMC: No bullish Order Block found")

        # ── S2: Break of Structure (BOS) ──────────────────────
        try:
            lookback   = min(20, n - 1)
            window     = high[-(lookback+1):-1]
            swing_high = float(np.max(window))
            current    = float(close[-1])
            if current > swing_high:
                score += 1
                results["bos"] = True
                flags.append(f"✅ SMC: BOS — price ₹{current:.1f} above swing high ₹{swing_high:.1f}")
            else:
                flags.append(f"❌ SMC: No BOS — price ₹{current:.1f} below swing ₹{swing_high:.1f}")
        except Exception:
            flags.append("⚠️ SMC: BOS check N/A")

        # ── S3: Fair Value Gap + Fib Confluence ───────────────
        fvg_found = False
        for i in range(n - 1, max(n - 15, 2), -1):
            fvg_high = float(low[i])
            fvg_low  = float(high[i - 2])
            if fvg_low < fvg_high:
                ltp    = float(close[-1])
                filled = ltp <= fvg_high
                fvg_found = True
                results["fvg"] = {
                    "fvg_low":  round(fvg_low,  2),
                    "fvg_high": round(fvg_high, 2),
                    "filled":   filled,
                }
                # Fib confluence on FVG midpoint
                fvg_mid   = (fvg_high + fvg_low) / 2
                fib_label = SMCEngine._fib_confluence(fvg_mid, fib_levels)
                conf_note = f" + Fib {fib_label} confluence 🔥" if fib_label else ""

                if filled:
                    score += 1
                    flags.append(
                        f"✅ SMC: Price in Bullish FVG "
                        f"(₹{fvg_low:.1f}–₹{fvg_high:.1f}){conf_note}"
                    )
                else:
                    flags.append(
                        f"⚠️ SMC: FVG at ₹{fvg_low:.1f}–₹{fvg_high:.1f} "
                        f"(not yet filled{conf_note})"
                    )
                break
        if not fvg_found:
            flags.append("❌ SMC: No recent Fair Value Gap")

        # ── S4: Liquidity Sweep ───────────────────────────────
        try:
            lkb      = min(15, n - 3)
            lows_win = low[-(lkb+2):-2]
            swing_lo = float(np.min(lows_win))
            swept    = float(low[-2]) < swing_lo and float(close[-1]) > swing_lo
            swept_1  = float(low[-1]) < swing_lo and float(close[-1]) > swing_lo
            if swept or swept_1:
                score += 1
                results["sweep"] = True
                flags.append(f"✅ SMC: Liquidity sweep below ₹{swing_lo:.1f} — reversal confirmed")
            else:
                flags.append(f"❌ SMC: No liquidity sweep detected")
        except Exception:
            flags.append("⚠️ SMC: Sweep check N/A")

        # ── S5: Change of Character (CHoCH) ───────────────────
        try:
            swing_highs = []
            for i in range(2, min(20, n - 1)):
                if high[-(i+1)] > high[-i] and high[-(i+1)] > high[-(i+2)]:
                    swing_highs.append(float(high[-(i+1)]))
            if len(swing_highs) >= 2:
                lower_highs = all(swing_highs[j] > swing_highs[j+1]
                                  for j in range(len(swing_highs)-1))
                if lower_highs and float(high[-1]) > swing_highs[0]:
                    score += 1
                    results["choch"] = True
                    flags.append("✅ SMC: CHoCH — first higher high after lower highs (reversal)")
                else:
                    flags.append("❌ SMC: No CHoCH — structure not reversed")
            else:
                flags.append("⚠️ SMC: Insufficient swing data for CHoCH")
        except Exception:
            flags.append("⚠️ SMC: CHoCH check N/A")

        return {"score": score, "max": 5, "flags": flags, **results}


# ═════════════════════════════════════════════════════════════
# ███  SCORE AGGREGATOR  ██████████████████════════════════════
# ─────────────────────────────────────────────────────────────
# Max score = 30 (was 27):
#   5 Technical + 5 Breakout + 5 Fundamental (Kite proxy)
#   + 5 Institutional + 2 Regime + 5 SMC + 3 Fibonacci
# ═════════════════════════════════════════════════════════════
def build_score(tech_s, tech_f, brk_s, brk_f,
                fund_d, inst_s, inst_f, reg_s, reg_f,
                smc_d=None, fib_d=None) -> dict:
    fund_s = fund_d.get("score", 0)
    fund_f = fund_d.get("flags", [])
    smc_s  = (smc_d or {}).get("score", 0)
    smc_f  = (smc_d or {}).get("flags", [])
    fib_s  = (fib_d or {}).get("score", 0)
    fib_f  = (fib_d or {}).get("flags", [])

    total = tech_s + brk_s + fund_s + inst_s + reg_s + smc_s + fib_s
    MAX   = 30

    if   total >= CFG.get("SCORE_STRONG_BUY", 20): sig, cls = "STRONG BUY 🔥", "strong-buy"
    elif total >= CFG.get("SCORE_BUY",        13): sig, cls = "BUY ✅",         "buy"
    elif total >= CFG.get("SCORE_WATCHLIST",   9): sig, cls = "WATCHLIST 👀",   "watch"
    else:                                           sig, cls = "AVOID ❌",        "avoid"

    return {
        "total": total, "max": MAX, "signal": sig, "signal_class": cls,
        "breakdown": {
            "technical":     {"score": tech_s, "max": 5, "flags": tech_f},
            "breakout":      {"score": brk_s,  "max": 5, "flags": brk_f},
            "fundamental":   {"score": fund_s, "max": 5, "flags": fund_f},
            "institutional": {"score": inst_s, "max": 5, "flags": inst_f},
            "regime":        {"score": reg_s,  "max": 2, "flags": reg_f},
            "smc":           {"score": smc_s,  "max": 5, "flags": smc_f},
            "fibonacci":     {"score": fib_s,  "max": 3, "flags": fib_f},
        }
    }
