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

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CFG = {
    # ── Kite credentials ──────────────────────────────────────
    "KITE_API_KEY":    os.environ.get("KITE_API_KEY",    ""),
    "KITE_API_SECRET": os.environ.get("KITE_API_SECRET", ""),
    "ACCESS_TOKEN":    load_token(),   # env var → saved file → empty

    # ── Universe filters ─────────────────────────────────────
    # We pull ALL NSE EQ stocks from Kite and filter down
    "MIN_PRICE":       50,       # skip sub-₹50 penny stocks
    "MAX_PRICE":       99999,    # no upper limit
    "UNIVERSE_CAP":    500,      # max stocks to score per scan (top by volume)
    "GAP_SCAN_ALL":    True,     # gap scanner runs on ALL ~1900 stocks
    "PRELOAD_TOKENS":  True,     # cache instrument list on startup

    # ── Index tokens (Kite) ───────────────────────────────────
    "NIFTY_TOKEN":     256265,   # NIFTY 50 index
    "BANKNIFTY_TOKEN": 260105,   # BANK NIFTY index

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
    "GAP_UP_MIN_PCT":  2.0,      # gap > 2% = event-driven move
    "RS_PERIOD":       20,       # relative strength lookback days
    "CONSOL_DAYS":     10,

    # ── Fundamental thresholds (yfinance) ────────────────────
    "MIN_REV_GROWTH":  10,
    "MIN_PAT_GROWTH":  5,
    "MAX_DE_RATIO":    2.0,
    "MIN_ROE":         12,
    "MIN_PROMOTER":    35,

    # ── Institutional thresholds (NSE API) ───────────────────
    "MIN_DELIVERY_PCT": 45,
    "PCR_BULLISH_MIN":  0.85,
    "VIX_MAX":          20,

    # ── Scoring gates (max = 27: 5 tech + 5 breakout + 5 fund + 5 inst + 2 regime + 5 SMC) ─
    "SCORE_STRONG_BUY": 18,   # ≥18/27 → STRONG BUY  (~67%)
    "SCORE_BUY":        12,   # ≥12/27 → BUY          (~44%)
    "SCORE_WATCHLIST":   8,   # ≥ 8/27 → WATCHLIST    (~30%)

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
# All Kite Connect calls are centralised here.
# ═════════════════════════════════════════════════════════════
class KiteLayer:
    """
    Single point of contact for ALL Kite API calls.
    Every method documents exactly which Kite endpoint it uses.
    """

    def __init__(self, kite):
        self._k = kite   # KiteConnect instance

    # ── 1. INSTRUMENTS ────────────────────────────────────────
    def get_all_nse_instruments(self) -> pd.DataFrame:
        """
        KITE API: kite.instruments("NSE")
        Returns ALL ~1900 NSE equity instruments with tokens,
        symbols, lot sizes, tick sizes.
        Called ONCE at startup and cached.
        """
        log.info("  [Kite] instruments('NSE') → fetching full NSE universe...")
        instruments = self._k.instruments("NSE")
        df = pd.DataFrame(instruments)

        # Keep only regular equity (EQ segment), not BE/SM/etc.
        df = df[df["segment"] == "NSE"]
        df = df[df["instrument_type"] == "EQ"]
        df = df[["instrument_token", "tradingsymbol", "name", "tick_size", "lot_size"]]
        df.columns = ["token", "symbol", "name", "tick", "lot"]
        # NOTE: last_price in instrument master is always 0 — do NOT filter by price here.
        # Price filtering happens in filter_for_deep_scan using live quote LTP.
        df = df.reset_index(drop=True)

        log.info(f"  [Kite] → {len(df)} EQ instruments loaded")
        return df

    # ── 2. BATCH LTP (fast gap scanner) ──────────────────────
    def get_batch_ltp(self, tokens: List[int]) -> Dict[int, float]:
        """
        KITE API: kite.ltp(["NSE:SYMBOL", ...])
        Fastest way to get current price for many stocks.
        Kite allows up to 500 tokens per call.
        We chunk into batches of 500 automatically.
        """
        result = {}
        BATCH = 500
        chunks = [tokens[i:i+BATCH] for i in range(0, len(tokens), BATCH)]

        for chunk in chunks:
            # Kite needs "NSE:TOKEN" format for ltp by token
            keys = [str(t) for t in chunk]
            try:
                data = self._k.ltp(keys)
                for key, val in data.items():
                    # key is like "NSE:738561"
                    tok = int(key.split(":")[-1]) if ":" in key else int(key)
                    result[tok] = val["last_price"]
            except Exception as e:
                log.warning(f"  [Kite] ltp batch error: {e}")
            time.sleep(0.1)   # Kite rate limit respect

        return result

    # ── 3. BATCH QUOTES (richer data: OHLC + volume) ─────────
    def get_batch_quotes(self, symbols: List[str], exchange: str = "NSE") -> dict:
        """
        KITE API: kite.quote(["NSE:RELIANCE", "NSE:TCS", ...])
        Returns OHLC, volume, OI, last_price for each symbol.
        Max 500 per call.
        Used for gap-up detection (needs open + prev close).
        """
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
                        "prev_close": val["ohlc"]["close"],   # PREVIOUS day close
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
        KITE API: kite.historical_data(instrument_token, from_date,
                                        to_date, interval)
        Fetches candle data. Interval options:
          minute, 3minute, 5minute, 10minute, 15minute,
          30minute, 60minute, day
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
        """
        KITE API: kite.place_order(variety, tradingsymbol, exchange,
                                    transaction_type, quantity,
                                    order_type, product, price, tag)
        txn_type: "BUY" or "SELL"
        Returns order_id string or None on failure.
        """
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
        """
        KITE API: kite.orders()
        Returns today's complete order book.
        """
        try:
            return self._k.orders()
        except Exception as e:
            log.warning(f"  [Kite] orders(): {e}")
            return []

    # ── 7. POSITIONS ──────────────────────────────────────────
    def get_positions(self) -> dict:
        """
        KITE API: kite.positions()
        Returns {"day": [...], "net": [...]}
        day  = intraday positions
        net  = net positions across days
        """
        try:
            return self._k.positions()
        except Exception as e:
            log.warning(f"  [Kite] positions(): {e}")
            return {"day": [], "net": []}

    # ── 8. MARGINS / CAPITAL ──────────────────────────────────
    def get_available_capital(self) -> float:
        """
        KITE API: kite.margins(segment="equity")
        Returns available cash for equity trades.
        Falls back to CFG["CAPITAL"] when paper-trading (margins API returns 0).
        """
        try:
            margins = self._k.margins(segment="equity")
            bal = float(margins["available"]["live_balance"])
            if bal > 0:
                return bal
            # Paper trading: live_balance is 0; try cash/opening_balance instead
            bal = float(margins["available"].get("cash", 0) or
                        margins["available"].get("opening_balance", 0) or 0)
            return bal if bal > 0 else CFG["CAPITAL"]
        except Exception as e:
            log.warning(f"  [Kite] margins(): {e}")
            return CFG["CAPITAL"]

    # ── 9. GTT (Good Till Triggered orders for SL/TP) ────────
    def place_gtt(self, symbol: str, exchange: str, qty: int,
                  entry: float, sl: float, tp: float) -> Optional[str]:
        """
        KITE API: kite.place_gtt(trigger_type, tradingsymbol, exchange,
                                   trigger_values, last_price, orders)
        Places a two-legged GTT — auto SL + TP without keeping app running.
        This is the institutional approach: set-and-forget.
        """
        try:
            gtt_id = self._k.place_gtt(
                trigger_type   = self._k.GTT_TYPE_OCO,   # One Cancels Other
                tradingsymbol  = symbol,
                exchange       = exchange,
                trigger_values = [sl, tp],
                last_price     = entry,
                orders         = [
                    {   # SL leg
                        "exchange":         exchange,
                        "tradingsymbol":    symbol,
                        "transaction_type": self._k.TRANSACTION_TYPE_SELL,
                        "quantity":         qty,
                        "order_type":       self._k.ORDER_TYPE_MARKET,
                        "product":          self._k.PRODUCT_CNC,
                        "price":            sl * 0.99,
                    },
                    {   # TP leg
                        "exchange":         exchange,
                        "tradingsymbol":    symbol,
                        "transaction_type": self._k.TRANSACTION_TYPE_SELL,
                        "quantity":         qty,
                        "order_type":       self._k.ORDER_TYPE_LIMIT,
                        "product":          self._k.PRODUCT_CNC,
                        "price":            tp,
                    }
                ]
            )
            log.info(f"  [Kite] GTT placed {symbol} | SL:{sl} TP:{tp} | GTT_ID:{gtt_id}")
            return str(gtt_id)
        except Exception as e:
            log.error(f"  [Kite] GTT FAILED {symbol}: {e}")
            return None


# ═════════════════════════════════════════════════════════════
# ███  UNIVERSE MANAGER  ██████████████████████████████████████
# Pulls full NSE list, applies filters, returns scan universe
# ═════════════════════════════════════════════════════════════
class UniverseManager:
    """
    Manages the stock universe dynamically from Kite.
    NO hardcoded list. Every scan day we pull fresh from Kite.
    """

    def __init__(self, kite_layer: KiteLayer):
        self.kl      = kite_layer
        self._cache  = None
        self._cached_at = None

    def get_universe(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Returns filtered universe of stocks to scan.
        Cache refreshes every 24 hours (instrument list rarely changes).
        """
        now = datetime.now()
        cache_stale = (self._cached_at is None or
                       (now - self._cached_at).seconds > 86400)

        if force_refresh or cache_stale or self._cache is None:
            log.info("  [Universe] Refreshing instrument list from Kite...")
            self._cache = self.kl.get_all_nse_instruments()
            self._cached_at = now

        return self._cache

    def get_gap_scan_universe(self) -> List[dict]:
        """
        Returns ALL NSE EQ stocks for gap-up scan.
        We need symbols + tokens only — fast ltp scan.
        """
        df = self.get_universe()
        return df[["symbol", "token"]].to_dict("records")

    def filter_for_deep_scan(self, quote_data: dict) -> List[dict]:
        """
        After batch quote scan, select top candidates for full analysis.
        Criteria:
          1. Gap-up > 2% OR price above yesterday's close > 1%
          2. Volume significant (can't filter without avg, so we sort by volume)
          3. Price ≥ MIN_PRICE
          4. Top UNIVERSE_CAP stocks by gap % + volume score
        Returns list of dicts with symbol, token, quote data.
        """
        df = self.get_universe()
        token_map = dict(zip(df["symbol"], df["token"]))

        candidates = []
        for sym, q in quote_data.items():
            ltp        = q.get("ltp", 0)
            prev_close = q.get("prev_close", 0)
            volume     = q.get("volume", 0)
            open_p     = q.get("open", 0)

            # Price filter uses LIVE LTP from quotes (instrument master has last_price=0)
            if not prev_close or ltp < CFG["MIN_PRICE"]:
                continue

            gap_pct     = (open_p - prev_close) / prev_close * 100 if prev_close else 0
            change_pct  = (ltp - prev_close) / prev_close * 100 if prev_close else 0

            # Composite score to rank candidates
            gap_score = max(gap_pct, 0)
            vol_score = volume / 1_000_000   # normalised

            candidates.append({
                "symbol":      sym,
                "token":       token_map.get(sym, 0),
                "ltp":         ltp,
                "prev_close":  prev_close,
                "open":        open_p,
                "gap_pct":     round(gap_pct, 2),
                "change_pct":  round(change_pct, 2),
                "volume":      volume,
                "rank_score":  gap_score * 2 + vol_score,
            })

        # Sort by rank_score (gap-ups + volume leaders first)
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)

        # Always include significant gap-ups regardless of cap
        gap_ups = [c for c in candidates if c["gap_pct"] >= CFG["GAP_UP_MIN_PCT"]]
        rest    = [c for c in candidates if c["gap_pct"] < CFG["GAP_UP_MIN_PCT"]]

        final = gap_ups + rest[:max(0, CFG["UNIVERSE_CAP"] - len(gap_ups))]
        log.info(f"  [Universe] {len(candidates)} stocks → {len(gap_ups)} gap-ups + "
                 f"{len(final)-len(gap_ups)} top volume = {len(final)} for deep scan")
        return final


# ═════════════════════════════════════════════════════════════
# ███  NSE PUBLIC API CLIENT  █████████████████████████████████
# FII/DII, Delivery %, Option Chain, VIX, Block Deals
# No auth required — public NSE endpoints
# ═════════════════════════════════════════════════════════════
class NSEClient:
    BASE    = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/json, text/plain, */*",
        "Referer":    "https://www.nseindia.com/",
    }

    def __init__(self):
        self.sess = requests.Session()
        self.sess.headers.update(self.HEADERS)
        self._warm_up()

    def _warm_up(self):
        """NSE requires a browser-like session (cookies) before API calls."""
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
        """NSE API: /api/fiidiiTradeReact — FII and DII net buying/selling."""
        data = self._get("fiidiiTradeReact")
        res  = {"fii_net": 0.0, "dii_net": 0.0, "fii_bullish": False, "dii_bullish": False}
        try:
            # API sometimes returns bare list, sometimes {"data": [...]}
            rows = data if isinstance(data, list) else (data.get("data") or [])
            if rows:
                row = rows[0]
                fii_buy  = float(row.get("buyValue",      row.get("fiiBuyValue",  0)) or 0)
                fii_sell = float(row.get("sellValue",     row.get("fiiSellValue", 0)) or 0)
                dii_buy  = float(row.get("dii_buyValue",  row.get("diiBuyValue",  0)) or 0)
                dii_sell = float(row.get("dii_sellValue", row.get("diiSellValue", 0)) or 0)
                res["fii_net"]     = fii_buy - fii_sell
                res["dii_net"]     = dii_buy - dii_sell
                res["fii_bullish"] = res["fii_net"] > 0
                res["dii_bullish"] = res["dii_net"] > 0
        except Exception as e:
            log.warning(f"  [NSE] FII parse: {e}")
        return res

    def get_delivery_pct(self, symbol: str) -> dict:
        """NSE API: /api/deliveryToTrading — Delivery % signals institutional buying."""
        data = self._get("deliveryToTrading", {"series": "EQ", "symbol": symbol})
        res  = {"delivery_pct": 0.0, "institutional": False}
        try:
            if data and "data" in data and data["data"]:
                pct = float(data["data"][-1].get("deliveryToTradedQty", 0))
                res["delivery_pct"]  = pct
                res["institutional"] = pct >= CFG["MIN_DELIVERY_PCT"]
        except Exception:
            pass
        return res

    def get_option_chain_pcr(self, symbol: str = "NIFTY") -> dict:
        """NSE API: /api/option-chain-indices — PCR from total OI."""
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
        """NSE API: /api/allIndices — India VIX fear gauge."""
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
        """NSE API: /api/block-deal — Detect institutional block/bulk buying."""
        data = self._get("block-deal")
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

        up   = high - high.shift()
        dn   = low.shift() - low
        pdm  = up.where((up > dn) & (up > 0), 0.0)
        ndm  = dn.where((dn > up) & (dn > 0), 0.0)
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

        # Use dynamic window so we never produce all-NaN when data < 252 trading days
        _52w_window       = min(252, len(df))
        df["high_52w"]    = high.rolling(_52w_window).max()
        df["at_52w_high"] = close >= df["high_52w"].shift(1).fillna(high)

        op = df["open"]
        body_sz   = (close - op).abs()
        rng       = high - low
        lwr_wick  = pd.concat([op, close], axis=1).min(axis=1) - low
        upr_wick  = high - pd.concat([op, close], axis=1).max(axis=1)
        bull_eng  = (close > op) & (close.shift() < op.shift()) & (close > op.shift()) & (op < close.shift())
        hammer    = (lwr_wick > 2 * body_sz) & (upr_wick < 0.3 * rng) & (close > op)
        df["bull_candle"] = bull_eng | hammer

        df["cross_up"]   = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift() <= df["ema_slow"].shift())
        df["cross_down"] = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift() >= df["ema_slow"].shift())

        # Only drop rows where CORE indicators are NaN; optional cols (high_52w, at_52w_high)
        # use dynamic windows so they won't NaN-out everything, but just in case:
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

        # B1: 52-week high breakout
        if row.get("at_52w_high", False):
            score += 1; flags.append("✅ 52-Week High Breakout")
        else:
            flags.append("❌ Not at 52w high")

        # B2: Gap-up (from batch quote data — Kite open vs prev_close)
        gap_pct = quote.get("gap_pct", 0)
        if gap_pct >= CFG["GAP_UP_MIN_PCT"]:
            score += 1; flags.append(f"✅ Gap-Up {gap_pct:.1f}% (event-driven)")
        elif gap_pct > 0:
            flags.append(f"⚠️ Small gap {gap_pct:.1f}%")
        else:
            flags.append(f"❌ No gap (or gap-down {gap_pct:.1f}%)")

        # B3: Relative Strength vs NIFTY
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

        # B4: Consolidation breakout
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

        # B5: MACD
        if row.get("macd_bull", False) and float(row.get("macd", -1)) > 0:
            score += 1; flags.append("✅ MACD bullish above zero")
        elif row.get("macd_bull", False):
            flags.append("⚠️ MACD bullish below zero")
        else:
            flags.append("❌ MACD bearish")

        return score, flags


# ═════════════════════════════════════════════════════════════
# ███  FUNDAMENTAL ENGINE  ████████████████████████████████████
# 100% Kite + NSE API — no Yahoo Finance, no 429 errors.
#
#  Old (Yahoo)          →  New (Kite/NSE data we already have)
#  ─────────────────────────────────────────────────────────────
#  Revenue Growth       →  F1: Volume trend (institutional accumulation)
#  PAT Growth           →  F2: Distance from 52-week high (earnings momentum)
#  D/E Ratio            →  F3: Supertrend direction (clean-trend = low-debt proxy)
#  ROE                  →  F4: Relative Strength vs NIFTY (quality outperformance)
#  Promoter Holding %   →  F5: Delivery % from NSE API (conviction buying)
# ═════════════════════════════════════════════════════════════
class FundamentalEngine:

    _cache: Dict[str, dict] = {}

    @classmethod
    def get(cls, symbol: str, df: pd.DataFrame, row: pd.Series,
            nifty_df: pd.DataFrame, nse: "NSEClient") -> dict:
        """
        Score 5 fundamental-proxy points using only data
        already fetched from Kite + NSE — zero external HTTP calls.
        """
        if symbol in cls._cache:
            return cls._cache[symbol]

        score = 0
        flags = []

        # ── F1: Volume Trend (Revenue Growth proxy) ───────────
        # Rising 20-day avg volume vs prior 20-day avg =
        # institutions quietly accumulating → sales growing.
        try:
            vol         = df["volume"]
            vol_recent  = float(vol.tail(20).mean())
            vol_prior   = float(vol.iloc[-40:-20].mean()) if len(vol) >= 40 \
                          else vol_recent
            vol_chg_pct = (vol_recent - vol_prior) / vol_prior * 100 \
                          if vol_prior else 0
            if vol_recent > vol_prior * 1.20:
                score += 1
                flags.append(f"✅ Vol trend +{vol_chg_pct:.0f}% (accumulation / rev-growth proxy)")
            else:
                flags.append(f"❌ Vol flat/declining {vol_chg_pct:+.0f}% vs prior 20d")
        except Exception:
            flags.append("⚠️ Volume trend N/A")

        # ── F2: Price vs 52-Week High (PAT Growth proxy) ──────
        # Stocks within 5% of 52w high have earnings momentum;
        # businesses with declining profits rarely break highs.
        try:
            w           = min(252, len(df))
            high_52w    = float(df["high"].tail(w).max())
            ltp         = float(row["close"])
            from_high   = (high_52w - ltp) / high_52w * 100
            if from_high <= 5:
                score += 1
                flags.append(f"✅ Within 5% of 52w High ({from_high:.1f}% away) — earnings momentum")
            elif from_high <= 15:
                flags.append(f"⚠️ Near 52w High ({from_high:.1f}% away)")
            else:
                flags.append(f"❌ {from_high:.1f}% below 52w High — weak momentum")
        except Exception:
            flags.append("⚠️ 52w High check N/A")

        # ── F3: Supertrend Direction (D/E Ratio proxy) ────────
        # Low-debt companies sustain clean price trends.
        # Supertrend flipping bearish often coincides with
        # over-leveraged balance sheets under rate stress.
        try:
            if row.get("st_bull", False):
                score += 1
                flags.append("✅ Supertrend bullish — low-debt trending structure")
            else:
                flags.append("❌ Supertrend bearish — possible debt / margin stress")
        except Exception:
            flags.append("⚠️ Supertrend N/A")

        # ── F4: Relative Strength vs NIFTY (ROE proxy) ────────
        # High-ROE businesses outperform the index consistently.
        # 20-day RS > +3% = institutional preference = ROE proxy.
        try:
            if nifty_df is not None and len(nifty_df) >= 20 and len(df) >= 20:
                stk_ret   = (float(df["close"].iloc[-1])     - float(df["close"].iloc[-20]))     \
                            / float(df["close"].iloc[-20])
                nifty_ret = (float(nifty_df["close"].iloc[-1]) - float(nifty_df["close"].iloc[-20])) \
                            / float(nifty_df["close"].iloc[-20])
                rs = (stk_ret - nifty_ret) * 100
                if rs >= 3.0:
                    score += 1
                    flags.append(f"✅ RS vs NIFTY +{rs:.1f}% (ROE-proxy — quality outperformer)")
                elif rs >= 0:
                    flags.append(f"⚠️ Slight RS +{rs:.1f}% vs NIFTY")
                else:
                    flags.append(f"❌ Underperforming NIFTY RS {rs:.1f}%")
            else:
                flags.append("⚠️ RS data N/A")
        except Exception:
            flags.append("⚠️ RS check N/A")

        # ── F5: Delivery % (Promoter Holding proxy) ───────────
        # High delivery % = conviction buying, not intraday noise.
        # Promoter-heavy stocks always show high institutional delivery.
        try:
            d    = nse.get_delivery_pct(symbol)
            dpct = d.get("delivery_pct", 0)
            if dpct >= CFG["MIN_DELIVERY_PCT"]:
                score += 1
                flags.append(f"✅ Delivery {dpct:.0f}% — institutional conviction (promoter proxy)")
            else:
                flags.append(f"❌ Delivery {dpct:.0f}% — speculative, low conviction")
        except Exception:
            flags.append("⚠️ Delivery data N/A")

        res = {
            "score":     score,
            "flags":     flags,
            # pe / roe / mcap_cr kept as None for UI backwards-compat
            "pe":        None,
            "roe":       None,
            "mcap_cr":   None,
            "available": True,
        }
        cls._cache[symbol] = res
        return res


# ═════════════════════════════════════════════════════════════
# ███  INSTITUTIONAL INTELLIGENCE  ████████████████████████████
# ═════════════════════════════════════════════════════════════
class InstitutionalEngine:

    @staticmethod
    def score(nse: NSEClient, symbol: str,
              fii_data: dict, oc_data: dict, vix: float,
              block_deals_cache: dict = None) -> Tuple[int, List[str]]:
        score = 0; flags = []

        # I1: FII flow
        fn = fii_data.get("fii_net", 0)
        if fii_data.get("fii_bullish"):
            score += 1; flags.append(f"✅ FII Net Buy ₹{fn/1e7:.0f}Cr")
        else:
            flags.append(f"❌ FII Net Sell ₹{abs(fn)/1e7:.0f}Cr")

        # I2: Delivery %
        d = nse.get_delivery_pct(symbol)
        dpct = d["delivery_pct"]
        if d["institutional"]:
            score += 1; flags.append(f"✅ Delivery {dpct:.0f}% (institutional)")
        else:
            flags.append(f"❌ Delivery {dpct:.0f}% (low)")

        # I3: PCR
        pcr = oc_data.get("pcr", 1.0)
        if oc_data.get("bullish_oi"):
            score += 1; flags.append(f"✅ PCR {pcr} bullish OI")
        else:
            flags.append(f"❌ PCR {pcr} bearish")

        # I4: VIX
        if vix < CFG["VIX_MAX"]:
            score += 1; flags.append(f"✅ VIX {vix:.1f} low fear")
        elif vix < 25:
            flags.append(f"⚠️ VIX {vix:.1f} elevated")
        else:
            flags.append(f"❌ VIX {vix:.1f} high fear")

        # I5: Block deal — use pre-fetched cache if provided (avoids 500 HTTP calls per scan)
        bd_raw = block_deals_cache if block_deals_cache is not None else nse._get("block-deal")
        buys   = [d for d in bd_raw.get("data", [])
                  if d.get("symbol", "").upper() == symbol.upper()
                  and d.get("buySell", "").upper() == "BUY"]
        bd = {"block_buy": len(buys) > 0, "deals": buys[:3]}
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

            ema200   = nifty_df["close"].ewm(span=min(200, len(nifty_df)), adjust=False).mean()
            last     = float(nifty_df["close"].iloc[-1])
            ema_val  = float(ema200.iloc[-1])

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
        """Risk CFG.RISK_PCT of available capital per trade."""
        risk_amt    = available_capital * CFG["RISK_PCT"]
        risk_per_sh = entry - sl
        if risk_per_sh <= 0:
            return 0
        qty = math.floor(risk_amt / risk_per_sh)
        # Also cap by available capital
        max_by_cap = math.floor(available_capital * 0.20 / entry)  # max 20% in one stock
        return max(min(qty, max_by_cap), 1)

    @staticmethod
    def sl_tp(entry: float, atr: float) -> Tuple[float, float]:
        sl = entry - CFG["ATR_SL"] * atr
        tp = entry + CFG["ATR_TP"] * atr
        return round(sl, 2), round(tp, 2)

    @staticmethod
    def trailing_stop(pos: dict, ltp: float, atr: float) -> dict:
        entry  = pos["entry"]
        sl     = pos["sl"]
        profit = ltp - entry
        new_sl = sl

        if profit >= 2.0 * atr:   new_sl = ltp - 1.0 * atr     # lock profit
        elif profit >= 1.5 * atr: new_sl = entry + 0.5 * atr
        elif profit >= 1.0 * atr: new_sl = entry                 # breakeven

        if new_sl > sl:
            return {**pos, "sl": round(new_sl, 2), "trailing": True}
        return pos

    @staticmethod
    def check_exit(ltp: float, pos: dict, row: pd.Series) -> Optional[str]:
        if ltp <= pos["sl"]:               return "STOP_LOSS"
        if ltp >= pos["tp"]:               return "TARGET_HIT"
        if row.get("cross_down", False):   return "EMA_DEATH_CROSS"
        if float(row.get("rsi", 100)) < CFG["RSI_EXIT"]: return "RSI_WEAKNESS"
        if not row.get("st_bull", True):   return "SUPERTREND_FLIP"
        return None


# ═════════════════════════════════════════════════════════════
# ███  SMC ENGINE — Smart Money Concepts  █████████████████████
# ─────────────────────────────────────────────────────────────
# Identifies institutional footprints:
#   S1. Bullish Order Block (OB) — last bearish candle before
#       a strong upward impulse. Price returning to OB = entry.
#   S2. Break of Structure (BOS) — price breaks above prior
#       swing high, confirming bullish market structure shift.
#   S3. Fair Value Gap (FVG) / Imbalance — 3-candle pattern
#       where candle[i-2].high < candle[i].low (bullish gap).
#       Price tends to return and fill the gap.
#   S4. Liquidity Sweep — price briefly dips below a swing low
#       then reverses up strongly (stop-hunt by smart money).
#   S5. Bullish CHoCH — Change of Character: first higher high
#       after a series of lower highs (structural reversal).
# ═════════════════════════════════════════════════════════════
class SMCEngine:

    @staticmethod
    def compute(df: pd.DataFrame) -> dict:
        """
        Run all SMC detections on OHLCV dataframe.
        Returns a dict with scores + flags + raw data for display.
        """
        if df is None or len(df) < 20:
            return {"score": 0, "max": 5, "flags": ["⚠️ Insufficient data for SMC"],
                    "ob": None, "fvg": None, "bos": False, "choch": False, "sweep": False}

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        op    = df["open"].values
        n     = len(df)

        score = 0
        flags = []
        results = {"ob": None, "fvg": None, "bos": False, "choch": False, "sweep": False}

        # ── S1: Bullish Order Block ────────────────────────────
        # Find last bearish candle (close < open) followed by a
        # strong up-move of ≥1% within 3 candles.
        ob_found = False
        for i in range(n - 4, max(n - 30, 2), -1):
            bearish = close[i] < op[i]
            if not bearish:
                continue
            # Check if next 1-3 candles have a strong bullish impulse
            future_high = max(high[i+1:i+4]) if i + 4 <= n else max(high[i+1:])
            impulse_pct = (future_high - close[i]) / close[i] * 100
            if impulse_pct >= 1.0:
                # Check if current price is near/inside the OB (within 3%)
                ltp = close[-1]
                ob_high = op[i]    # top of bearish candle body
                ob_low  = close[i] # bottom
                near_ob = ob_low * 0.97 <= ltp <= ob_high * 1.05
                ob_found = True
                results["ob"] = {
                    "ob_high": round(float(ob_high), 2),
                    "ob_low":  round(float(ob_low),  2),
                    "near":    near_ob,
                    "impulse": round(impulse_pct, 1),
                }
                if near_ob:
                    score += 1
                    flags.append(f"✅ SMC: Price at Bullish OB ({ob_low:.1f}–{ob_high:.1f})")
                else:
                    flags.append(f"⚠️ SMC: OB exists ({ob_low:.1f}–{ob_high:.1f}), price not yet retesting")
                break
        if not ob_found:
            flags.append("❌ SMC: No bullish Order Block found")

        # ── S2: Break of Structure (BOS) ──────────────────────
        # Last 20 candles: find prior swing high (local max),
        # then check if price has broken above it.
        try:
            lookback = min(20, n - 1)
            window   = high[-(lookback+1):-1]
            swing_high = float(np.max(window))
            current    = float(close[-1])
            if current > swing_high:
                score += 1
                results["bos"] = True
                flags.append(f"✅ SMC: BOS — price {current:.1f} above swing high {swing_high:.1f}")
            else:
                flags.append(f"❌ SMC: No BOS — price {current:.1f} below swing {swing_high:.1f}")
        except Exception:
            flags.append("⚠️ SMC: BOS check N/A")

        # ── S3: Fair Value Gap (FVG) ──────────────────────────
        # Bullish FVG: candle[i-2].high < candle[i].low
        # Scan last 15 candles for an unfilled bullish FVG.
        fvg_found = False
        for i in range(n - 1, max(n - 15, 2), -1):
            fvg_high = float(low[i])        # bottom of current candle
            fvg_low  = float(high[i - 2])   # top of candle 2 back
            if fvg_low < fvg_high:          # gap exists
                ltp    = float(close[-1])
                filled = ltp <= fvg_high    # price has entered FVG
                fvg_found = True
                results["fvg"] = {
                    "fvg_low":  round(fvg_low,  2),
                    "fvg_high": round(fvg_high, 2),
                    "filled":   filled,
                }
                if filled:
                    score += 1
                    flags.append(f"✅ SMC: Price in Bullish FVG ({fvg_low:.1f}–{fvg_high:.1f})")
                else:
                    flags.append(f"⚠️ SMC: FVG at {fvg_low:.1f}–{fvg_high:.1f} (not yet filled)")
                break
        if not fvg_found:
            flags.append("❌ SMC: No recent Fair Value Gap")

        # ── S4: Liquidity Sweep (stop hunt) ───────────────────
        # Price dips below a swing low then closes back above it
        # within 1-3 candles → smart money swept retail stops.
        try:
            lkb      = min(15, n - 3)
            lows_win = low[-(lkb+2):-2]
            swing_lo = float(np.min(lows_win))
            # Last 2 candles: did price wick below swing_lo then close above?
            swept   = float(low[-2]) < swing_lo and float(close[-1]) > swing_lo
            swept_1 = float(low[-1]) < swing_lo and float(close[-1]) > swing_lo
            if swept or swept_1:
                score += 1
                results["sweep"] = True
                flags.append(f"✅ SMC: Liquidity sweep below {swing_lo:.1f} — reversal signal")
            else:
                flags.append(f"❌ SMC: No liquidity sweep detected")
        except Exception:
            flags.append("⚠️ SMC: Sweep check N/A")

        # ── S5: Change of Character (CHoCH) ───────────────────
        # After ≥2 lower highs, price makes a higher high →
        # first sign of bullish structural reversal.
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
                    flags.append("✅ SMC: CHoCH — first higher high after lower highs")
                else:
                    flags.append("❌ SMC: No CHoCH — structure not reversed")
            else:
                flags.append("⚠️ SMC: Insufficient swing data for CHoCH")
        except Exception:
            flags.append("⚠️ SMC: CHoCH check N/A")

        return {"score": score, "max": 5, "flags": flags, **results}



def build_score(tech_s, tech_f, brk_s, brk_f,
                fund_d, inst_s, inst_f, reg_s, reg_f,
                smc_d=None) -> dict:
    fund_s = fund_d.get("score", 0)
    fund_f = fund_d.get("flags", [])
    smc_s  = (smc_d or {}).get("score", 0)
    smc_f  = (smc_d or {}).get("flags", [])
    total  = tech_s + brk_s + fund_s + inst_s + reg_s + smc_s
    MAX    = 27   # 5 tech + 5 breakout + 5 fundamental + 5 institutional + 2 regime + 5 SMC

    if   total >= CFG.get("SCORE_STRONG_BUY", 16): sig, cls = "STRONG BUY 🔥", "strong-buy"
    elif total >= CFG.get("SCORE_BUY",        11): sig, cls = "BUY ✅",         "buy"
    elif total >= CFG.get("SCORE_WATCHLIST",    7): sig, cls = "WATCHLIST 👀",   "watch"
    else:                                            sig, cls = "AVOID ❌",        "avoid"

    return {
        "total": total, "max": MAX, "signal": sig, "signal_class": cls,
        "breakdown": {
            "technical":     {"score": tech_s, "max": 5, "flags": tech_f},
            "breakout":      {"score": brk_s,  "max": 5, "flags": brk_f},
            "fundamental":   {"score": fund_s, "max": 5, "flags": fund_f},
            "institutional": {"score": inst_s, "max": 5, "flags": inst_f},
            "regime":        {"score": reg_s,  "max": 2, "flags": reg_f},
            "smc":           {"score": smc_s,  "max": 5, "flags": smc_f},
        }
    }
