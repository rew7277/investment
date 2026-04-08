"""
═══════════════════════════════════════════════════════════════
  INSTITUTIONAL TRADER PRO — app.py
  Scans FULL NSE universe. No hardcoded stock list.
  Every Kite API call is labelled.
═══════════════════════════════════════════════════════════════
"""

import os, json, logging, threading, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string, redirect, request
import numpy as np
import pandas as pd


def _np_sanitize(obj):
    """Recursively convert numpy scalars/arrays to native Python types.

    Flask's jsonify cannot serialise numpy.bool_, numpy.int64, numpy.float64,
    etc.  This helper walks the entire state dict before it hits json.dumps.
    """
    if isinstance(obj, dict):
        return {k: _np_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_sanitize(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

from apscheduler.schedulers.background import BackgroundScheduler

from engine import (
    CFG, KiteLayer, UniverseManager, NSEClient,
    TechnicalEngine, FundamentalEngine, BreakoutScanner,
    InstitutionalEngine, MarketRegime, RiskManager,
    SMCEngine, FibonacciEngine, PivotEngine, OptionEngine,
    build_score, save_token, load_token, is_token_fresh
)

try:
    from kiteconnect import KiteConnect
    KITE_OK = True
except ImportError:
    KITE_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trader.log")]
)
log = logging.getLogger("APP")

app   = Flask(__name__)
nse   = NSEClient()
kl: KiteLayer           = None   # initialised on first scan
universe: UniverseManager = None

# Thread lock — all writes to STATE go through this to prevent race conditions
_STATE_LOCK = threading.Lock()

# Paper portfolio persistence — survives server restarts
PAPER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_portfolio.json")

def load_paper_portfolio() -> dict:
    default = {"capital": 10000.0, "cash": 10000.0, "holdings": {}, "pnl_realised": 0.0}
    try:
        with open(PAPER_FILE) as f:
            data = json.load(f)
        log.info(f"  [Paper] Portfolio loaded from {PAPER_FILE}")
        return {**default, **data}
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"  [Paper] Could not load portfolio: {e}")
    return default

def save_paper_portfolio():
    try:
        with open(PAPER_FILE, "w") as f:
            json.dump(STATE["paper_portfolio"], f, indent=2)
    except Exception as e:
        log.warning(f"  [Paper] Could not save portfolio: {e}")

STATE = {
    "signals":       [],
    "positions":     {},
    "trade_log":     [],
    "regime":        {},
    "fii_dii":       {},
    "vix":           15.0,
    "gap_alerts":    [],
    "universe_size": 0,
    "last_scan":     None,
    "scanning":      False,
    "indices":       {},          # PivotEngine snapshot for 4 indices
    "scan_progress": {          # real-time progress tracking
        "step":       0,        # current step 0-7
        "step_label": "",       # e.g. "Loading NSE universe..."
        "stocks_done": 0,       # stocks analysed so far
        "stocks_total": 0,      # total candidates
        "pct":        0,        # 0-100
    },
    "paper_portfolio": load_paper_portfolio(),   # loaded from disk — persists restarts
    "kite_calls":    [],
    "errors":        [],
    "auth": {
        "status":      "unknown",
        "user_name":   None,
        "user_id":     None,
        "token_saved": None,
        "login_url":   None,
    }
}


# ═════════════════════════════════════════════════════════════
# KITE SESSION SETUP
# ═════════════════════════════════════════════════════════════
def build_login_url() -> str:
    try:
        from kiteconnect import KiteConnect
        return KiteConnect(api_key=CFG["KITE_API_KEY"]).login_url()
    except Exception:
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={CFG['KITE_API_KEY']}"


def init_kite(access_token: str = None) -> KiteLayer:
    if not KITE_OK:
        raise RuntimeError("kiteconnect not installed")
    token = access_token or CFG["ACCESS_TOKEN"]
    if not token:
        STATE["auth"]["status"] = "missing"
        raise RuntimeError("No access token. Visit /kite-login to authenticate.")
    kite = KiteConnect(api_key=CFG["KITE_API_KEY"])
    kite.set_access_token(token)
    try:
        profile = kite.profile()
        STATE["auth"].update({
            "status":    "ok",
            "user_name": profile["user_name"],
            "user_id":   profile["user_id"],
        })
        log.info(f"  [Kite] ✅ Authenticated as: {profile['user_name']} ({profile['user_id']})")
    except Exception as e:
        STATE["auth"]["status"] = "invalid"
        raise RuntimeError(f"Kite auth failed: {e}")
    return KiteLayer(kite)


def set_progress(step: int, label: str, done: int = 0, total: int = 0):
    """Update real-time scan progress — polled by /api/progress every 1s."""
    pct = int(done / total * 100) if total > 0 else int(step / 7 * 100)
    STATE["scan_progress"] = {
        "step": step, "step_label": label,
        "stocks_done": done, "stocks_total": total, "pct": pct,
    }
    log.info(f"  [Progress] Step {step}/7 · {pct}% · {label}")


def log_kite_call(method: str, detail: str = ""):
    STATE["kite_calls"].append({
        "time":   datetime.now().strftime("%H:%M:%S"),
        "method": method,
        "detail": detail,
    })
    if len(STATE["kite_calls"]) > 50:
        STATE["kite_calls"] = STATE["kite_calls"][-50:]


# ═════════════════════════════════════════════════════════════
# FULL NSE SCAN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════
def run_full_scan():
    global kl, universe

    if STATE["scanning"]:
        log.info("Scan already running.")
        return

    STATE["scanning"] = True
    STATE["scan_progress"] = {"step":0,"step_label":"Initialising...","stocks_done":0,"stocks_total":0,"pct":0}
    log.info(f"\n{'═'*65}")
    log.info(f"  FULL NSE INSTITUTIONAL SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"{'═'*65}")

    # ── Token freshness gate ───────────────────────────────────
    if not is_token_fresh(max_age_hours=8):
        warn_msg = ("⚠️  Kite token may be stale (>8h old). "
                    "Visit /kite-login to refresh before market open.")
        log.warning(f"  [Auth] {warn_msg}")
        STATE["errors"].append({"time": str(datetime.now()), "msg": warn_msg})
        # Do NOT abort — the token might still work; let Kite decide

    try:
        # ── INIT KITE ─────────────────────────────────────────
        if kl is None:
            kl = init_kite()
        if universe is None:
            universe = UniverseManager(kl)

        # ─────────────────────────────────────────────────────
        # STEP 1: Get Full NSE Universe
        # KITE API: kite.instruments("NSE")
        # ─────────────────────────────────────────────────────
        log.info("  STEP 1 — Loading full NSE universe...")
        set_progress(1, "Loading NSE universe...")
        instr_df = universe.get_universe()
        all_symbols = instr_df["symbol"].tolist()
        log_kite_call("instruments('NSE')", f"{len(all_symbols)} EQ stocks loaded")
        log.info(f"  → {len(all_symbols)} NSE EQ instruments available")

        # ─────────────────────────────────────────────────────
        # STEP 2: Batch Quote Scan — ALL NSE stocks
        # KITE API: kite.quote([...]) in chunks of 500
        # This one call (or a few batches) scans ALL ~1900 stocks
        # for gap-ups, volume surges, and price moves
        # ─────────────────────────────────────────────────────
        log.info(f"  STEP 2 — Batch quote scan on {len(all_symbols)} stocks...")
        set_progress(2, f"Batch quoting {len(all_symbols)} NSE stocks...")
        log.info(f"           (Kite kite.quote() — {math.ceil(len(all_symbols)/500)} batches of 500)")
        quote_data = kl.get_batch_quotes(all_symbols)
        log_kite_call("quote(all_nse)", f"{len(quote_data)} quotes fetched in {math.ceil(len(all_symbols)/500)} batches")
        log.info(f"  → {len(quote_data)} live quotes received")

        # ─────────────────────────────────────────────────────
        # STEP 3: Filter to Deep-Scan Candidates
        # Gap-ups always included, then top-volume stocks
        # ─────────────────────────────────────────────────────
        log.info("  STEP 3 — Filtering to deep-scan candidates...")
        set_progress(3, "Filtering deep-scan candidates...")
        candidates = universe.filter_for_deep_scan(quote_data)
        STATE["universe_size"] = len(quote_data)

        # Capture all gap-up alerts (even if not in deep scan)
        # Capture all gap-up alerts (use candidates which have live gap_pct)
        gap_alerts = sorted(
            [{"symbol": c["symbol"], "gap_pct": c["gap_pct"],
              "ltp": c["ltp"], "volume": c["volume"]}
             for c in candidates if c.get("gap_pct", 0) >= CFG["GAP_UP_MIN_PCT"]],
            key=lambda x: x["gap_pct"], reverse=True
        )
        STATE["gap_alerts"] = gap_alerts[:30]
        log.info(f"  → {len(gap_alerts)} gap-up stocks found today (≥{CFG['GAP_UP_MIN_PCT']}%)")
        log.info(f"  → {len(candidates)} candidates selected for full analysis")

        # ─────────────────────────────────────────────────────
        # STEP 4: Market Regime + Macro Data  (MULTI-INDEX)
        # KITE API: kite.historical_data(...) × 4 indexes
        # NSE API:  India VIX, FII/DII flows
        # Indexes: NIFTY 50, BANKNIFTY, FINNIFTY, MIDCAP 150
        # ─────────────────────────────────────────────────────
        log.info("  STEP 4 — Multi-index regime check (NIFTY / BANKNIFTY / FINNIFTY / MIDCAP)...")
        set_progress(4, "Fetching 4 indexes for regime check...")

        def _fetch_index(token: int, name: str, days: int = 400) -> pd.DataFrame:
            try:
                raw = kl.get_ohlcv(token, days=days)
                if raw.empty:
                    log.warning(f"  [Regime] {name} — no data returned")
                    return pd.DataFrame()
                df = TechnicalEngine.compute(raw)
                if df.empty and not raw.empty:
                    return raw
                log.info(f"  [Regime] {name} — {len(df)} candles loaded")
                return df
            except Exception as e:
                log.warning(f"  [Regime] {name} fetch failed: {e}")
                return pd.DataFrame()

        nifty_df     = _fetch_index(CFG["NIFTY_TOKEN"],     "NIFTY 50")
        banknifty_df = _fetch_index(CFG["BANKNIFTY_TOKEN"], "BANKNIFTY")
        finnifty_df  = _fetch_index(CFG["FINNIFTY_TOKEN"],  "FINNIFTY")
        midcap_df    = _fetch_index(CFG["MIDCAP_TOKEN"],    "MIDCAP 150")

        log_kite_call(
            "historical_data",
            f"4 indexes — NIFTY({len(nifty_df)}) BANKNIFTY({len(banknifty_df)}) "
            f"FINNIFTY({len(finnifty_df)}) MIDCAP({len(midcap_df)}) candles"
        )

        vix = nse.get_india_vix()
        if vix is None:
            # Fallback: Kite quote for India VIX index token
            try:
                vix_quote = kl.get_batch_quotes(["INDIA VIX"])
                vix_data  = vix_quote.get("INDIA VIX", {})
                if vix_data.get("ltp"):
                    vix = round(float(vix_data["ltp"]), 2)
                    log.info(f"  [Kite] India VIX = {vix:.2f} (from Kite quote fallback)")
            except Exception as _ve:
                log.warning(f"  [Kite] VIX quote fallback failed: {_ve}")
        if vix is None:
            log.warning("  [Regime] VIX completely unavailable — regime check will skip fear gate")
        fii = nse.get_fii_dii()
        oc  = nse.get_option_chain_pcr("NIFTY")
        STATE["vix"]     = vix if vix is not None else 0.0
        STATE["fii_dii"] = {**fii, "pcr": oc.get("pcr", "—")}

        reg_s, reg_f, regime_ok, reg_label, reg_conf = MarketRegime.check(
            nifty_df, vix,
            banknifty_df = banknifty_df if not banknifty_df.empty else None,
            finnifty_df  = finnifty_df  if not finnifty_df.empty  else None,
            midcap_df    = midcap_df    if not midcap_df.empty    else None,
            pcr          = oc.get("pcr"),
        )
        # reg_s_stock: normalized 0-2 pts for the 30-pt stock scoring system
        # reg_s (0-7) drives regime label/confidence; only 2 pts flow into build_score
        reg_s_stock = round(min(reg_s, 7.0) / 7.0 * 2.0, 1)
        STATE["regime"] = {
            "score":      reg_s,
            "label":      reg_label,
            "flags":      reg_f,
            "vix":        vix,
            "confidence": reg_conf,
        }

        # ── Compute pivot levels for all 4 indices ────────────
        STATE["indices"] = PivotEngine.compute_all(
            nifty_df     if not nifty_df.empty     else None,
            banknifty_df if not banknifty_df.empty else None,
            finnifty_df  if not finnifty_df.empty  else None,
            midcap_df    if not midcap_df.empty    else None,
        )

        log.info(f"  → Regime: {reg_label} ({reg_conf:.0f}%) | VIX: {vix:.1f} | FII: ₹{fii.get('fii_net',0)/1e7:.0f}Cr | PCR: {oc.get('pcr',1)}")

        bear_market   = (reg_label.startswith("BEAR"))
        sideways_mkt  = (reg_label.startswith("SIDEWAYS"))
        extreme_fear  = (vix > 30)
        regime_ok     = not extreme_fear
        bear_premium  = 4 if bear_market else 0

        # ── ACCURACY FILTER 1: Time gate (10:00–14:00 IST) ───
        # Avoids opening volatility and closing manipulation.
        # Only active in intraday mode; swing/positional trade any time.
        now_hour = datetime.now().hour
        time_ok  = True
        if CFG.get("TRADE_MODE") == "intraday":
            time_ok = (10 <= now_hour <= 14)
            if not time_ok:
                log.info(f"  [Accuracy] Time gate: hour={now_hour} — outside 10-14 window, no new intraday entries")

        # ── ACCURACY FILTER 2: Daily trade limit (max 3) ─────
        today_str   = datetime.now().strftime("%Y-%m-%d")
        trades_today = sum(
            1 for t in STATE["trade_log"]
            if t.get("date","").startswith(today_str)
        )
        MAX_DAILY_TRADES = 3

        # ─────────────────────────────────────────────────────
        # STEP 4b: NSE Data Prefetch
        # ─────────────────────────────────────────────────────
        log.info("  STEP 4b — NSE prefetch (block-deals cache)…")
        set_progress(4, "Prefetching NSE block deals & FII data...")
        try:
            nse.prefetch_scan_data()
        except Exception as _pe:
            log.warning(f"  [NSE] Prefetch failed (non-fatal): {_pe}")

        # ─────────────────────────────────────────────────────
        # STEP 5: Get available capital from Kite — HARD STOP if invalid
        # ─────────────────────────────────────────────────────
        available_cap = kl.get_available_capital()
        if available_cap < 1000:
            if not CFG["PAPER_TRADE"]:
                raise RuntimeError(
                    f"Capital check returned ₹{available_cap:.0f} — aborting scan to prevent "
                    "unintended orders. Check Kite margins or set PAPER_TRADE=true."
                )
            log.warning(f"  [Capital] Kite returned ₹{available_cap:.2f} — using CFG fallback ₹{CFG['CAPITAL']:,} (paper mode)")
            available_cap = CFG["CAPITAL"]
        log_kite_call("margins('equity')", f"₹{available_cap:,.0f} available")
        log.info(f"  STEP 5 — Available capital: ₹{available_cap:,.0f}")
        set_progress(5, f"Capital confirmed ₹{available_cap:,.0f} — starting deep analysis...")

        # BEAR market hard gate — block all new buys, just score and track
        if bear_market:
            log.warning(f"  [Regime] ⚠️  BEAR MARKET detected — all new BUY orders are BLOCKED this scan")

        # ─────────────────────────────────────────────────────
        # STEP 6: Deep Analysis — PARALLEL via ThreadPoolExecutor
        # KITE API: kite.historical_data(token, ...) per stock
        # Using 6 workers → ~5× faster than serial scan
        # ─────────────────────────────────────────────────────
        log.info(f"  STEP 6 — Parallel deep analysis on {len(candidates)} stocks (6 workers)...")

        # Take a snapshot of current positions — workers read this, main thread writes
        positions_snapshot = dict(STATE["positions"])
        open_count_snap    = len(positions_snapshot)

        signals      = []
        skipped      = 0
        _done_lock   = threading.Lock()
        done_counter = [0]   # mutable list for closure

        set_progress(6, f"Parallel deep-analysing {len(candidates)} candidates...", 0, len(candidates))

        def _analyse(cand: dict):
            """Analyse one stock — runs in a worker thread. Returns sig dict or None."""
            sym   = cand["symbol"]
            token = cand["token"]
            if not token:
                return None
            try:
                df = kl.get_ohlcv(token, days=300)
                if df.empty or len(df) < 60:
                    return None
                df = TechnicalEngine.compute(df)
                if df.empty:
                    return None

                row     = df.iloc[-1]
                tech_s,  tech_f  = TechnicalEngine.score(row)
                brk_s,   brk_f   = BreakoutScanner.score(df, row, nifty_df, cand)
                fund_d           = FundamentalEngine.compute(df, row, nifty_df)
                inst_s,  inst_f  = InstitutionalEngine.score(nse, sym, fii, oc, vix)
                smc_d            = SMCEngine.compute(df)
                fib_d            = FibonacciEngine.compute(df)
                full             = build_score(tech_s, tech_f, brk_s, brk_f,
                                               fund_d, inst_s, inst_f,
                                               reg_s_stock, reg_f,   # ← normalized 0-2
                                               smc_d, fib_d)

                entry  = cand["ltp"] * (1 + CFG["SLIPPAGE_PCT"])
                sl, tp = RiskManager.sl_tp(entry, float(row["atr"]), fib_d)
                qty    = RiskManager.position_size(entry, sl, available_cap)

                tp_source = (
                    "FIB 1.618"
                    if fib_d.get("score", 0) >= 2
                       and fib_d.get("fib_tp_1618")
                       and fib_d["fib_tp_1618"] > entry * 1.02
                    else "ATR×3.5"
                )

                return {
                    "symbol":       sym,
                    "ltp":          round(float(cand["ltp"]), 2),
                    "gap_pct":      float(cand.get("gap_pct", 0)),
                    "change_pct":   float(cand.get("change_pct", 0)),
                    "volume":       int(cand.get("volume", 0)),
                    "vol_surge":    bool(row.get("vol_surge", False)),   # ← F3: volume spike
                    "vol_ma":       round(float(row.get("vol_ma", 0)), 0),
                    "vwap":         round(float(row.get("vwap", 0)), 2),
                    "above_vwap":   bool(row.get("above_vwap", False)),
                    "rsi":          round(float(row.get("rsi", 0)), 1),
                    "adx":          round(float(row.get("adx", 0)), 1),
                    "atr":          round(float(row.get("atr", 0)), 2),
                    "sl":           float(sl),
                    "tp":           float(tp),
                    "tp_source":    tp_source,
                    "qty":          int(qty),
                    "score":        int(full["total"]),
                    "max_score":    int(full["max"]),
                    "signal":       full["signal"],
                    "signal_class": full["signal_class"],
                    "breakdown":    full["breakdown"],
                    "action":       "HOLD",
                    "time":         str(datetime.now()),
                    "token":        token,
                    "entry":        float(entry),
                    "pe":           fund_d.get("pe"),
                    "roe":          fund_d.get("roe"),
                    "mcap_cr":      fund_d.get("mcap_cr"),
                    "smc_ob":       smc_d.get("ob"),
                    "smc_fvg":      smc_d.get("fvg"),
                    "smc_bos":      bool(smc_d.get("bos", False)),
                    "smc_choch":    bool(smc_d.get("choch", False)),
                    "smc_sweep":    bool(smc_d.get("sweep", False)),
                    "fib_score":        int(fib_d.get("score", 0)),
                    "fib_golden_high":  fib_d.get("golden_zone_high"),
                    "fib_golden_low":   fib_d.get("golden_zone_low"),
                    "fib_tp_1618":      fib_d.get("fib_tp_1618"),
                    "fib_swing_high":   fib_d.get("swing_high"),
                    "fib_swing_low":    fib_d.get("swing_low"),
                    "fib_in_golden":    bool(fib_d.get("price_in_golden", False)),
                    "fib_at_key":       bool(fib_d.get("fib_at_key_level", False)),
                }
            except Exception as e:
                with _STATE_LOCK:
                    STATE["errors"].append({"time": str(datetime.now()), "msg": f"{sym}: {str(e)[:80]}"})
                    if len(STATE["errors"]) > 30:
                        STATE["errors"] = STATE["errors"][-30:]
                return None

        # Run all stock analyses in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_map = {executor.submit(_analyse, cand): cand for cand in candidates}
            for future in as_completed(future_map):
                sig = future.result()
                with _done_lock:
                    done_counter[0] += 1
                    n = done_counter[0]
                if sig is None:
                    skipped += 1
                else:
                    log_kite_call("historical_data", f"{sig['symbol']} — scored {sig['score']}/30")
                    signals.append(sig)
                if n % 10 == 0 or n == len(candidates):
                    cand_sym = future_map[future]["symbol"]
                    set_progress(6, f"Analysed {n}/{len(candidates)} · skipped {skipped} · {cand_sym}",
                                 n, len(candidates))

        log.info(f"  → Parallel analysis complete: {len(signals)} scored, {skipped} skipped")

        # ── Entry / Exit decisions (serial — needs consistent state) ──
        open_count = open_count_snap
        for sig in signals:
            sym   = sig["symbol"]
            token = sig["token"]
            entry = sig["entry"]
            sl    = sig["sl"]
            tp    = sig["tp"]
            qty   = sig["qty"]
            score = sig["score"]

            # ── BEAR market hard gate — no new buys ───────────
            # ACCURACY FILTERS applied here:
            #   F1 — Time gate (intraday: 10-14h only)
            #   F2 — Skip SIDEWAYS regime
            #   F3 — Volume spike confirmation (soft — only gate BUY, not STRONG BUY)
            #   F4 — Max 3 trades per day
            buy_threshold  = CFG["SCORE_BUY"]
            # F3: volume confirmation — skip regular BUY if no volume surge,
            # but allow STRONG BUY through regardless (high conviction overrides)
            vol_confirmed  = (sig.get("vol_surge", False)
                              or score >= CFG["SCORE_STRONG_BUY"])

            # ── 80% WIN RATE GATES ──────────────────────────────
            # Only STRONG BUY (score>=20) triggers real orders.
            # Also gate: minimum R:R of 2.0
            rr_adequate = True
            if sig.get('tp',0) > 0 and sig.get('entry',0) > 0 and sig.get('sl',0) > 0:
                rr_val = (sig['tp'] - sig['entry']) / max(sig['entry'] - sig['sl'], 0.01)
                rr_adequate = rr_val >= 2.0
            if (score >= CFG["SCORE_STRONG_BUY"]
                    and sym not in STATE["positions"]
                    and open_count < CFG["MAX_OPEN_TRADES"]
                    and regime_ok
                    and not bear_market
                    and not sideways_mkt
                    and time_ok
                    and vol_confirmed
                    and rr_adequate
                    and trades_today < MAX_DAILY_TRADES):

                if CFG["PAPER_TRADE"]:
                    log.info(f"  📝 PAPER BUY {sym} | {sig['signal']} | Score {score}/{sig['max_score']} | Qty {qty}")
                else:
                    order_id = kl.place_order(sym, "NSE", "BUY", qty)
                    log_kite_call("place_order", f"BUY {qty} {sym}")
                    if order_id:
                        gtt_id = kl.place_gtt(sym, "NSE", qty, entry, sl, tp)
                        log_kite_call("place_gtt", f"{sym} SL:{sl} TP:{tp}")

                with _STATE_LOCK:
                    STATE["positions"][sym] = {
                        "entry": entry, "sl": sl, "tp": tp, "qty": qty,
                        "score": score, "date": str(datetime.now()),
                        "token": token, "trailing": False,
                    }
                open_count   += 1
                trades_today += 1
                sig["action"] = "BUY"

            # ── Exit: SELL ─────────────────────────────────────
            elif sym in STATE["positions"]:
                pos    = STATE["positions"][sym]
                # Get latest row atr for trailing stop
                atr_val = sig.get("atr", entry * 0.02)
                pos    = RiskManager.trailing_stop(pos, sig["ltp"], atr_val)
                with _STATE_LOCK:
                    STATE["positions"][sym] = pos
                reason = RiskManager.check_exit(sig["ltp"], pos, None)

                if reason:
                    pnl = (sig["ltp"] - pos["entry"]) * pos["qty"] - CFG["BROKERAGE"] * 2
                    if CFG["PAPER_TRADE"]:
                        log.info(f"  📝 PAPER SELL {sym} | {reason} | PnL ₹{pnl:.0f}")
                    else:
                        kl.place_order(sym, "NSE", "SELL", pos["qty"])
                        log_kite_call("place_order", f"SELL {pos['qty']} {sym} — {reason}")

                    with _STATE_LOCK:
                        STATE["trade_log"].append({
                            "symbol": sym, "entry": pos["entry"],
                            "exit": sig["ltp"], "qty": pos["qty"],
                            "pnl": round(pnl, 2), "reason": reason,
                            "date": str(datetime.now()),
                        })
                        del STATE["positions"][sym]
                    open_count -= 1
                    sig["action"] = f"SELL ({reason})"        # ─────────────────────────────────────────────────────
        # STEP 7: Sync positions with Kite
        # KITE API: kite.positions() + kite.orders()
        # ─────────────────────────────────────────────────────
        if not CFG["PAPER_TRADE"]:
            live_pos = kl.get_positions()
            log_kite_call("positions()", f"{len(live_pos.get('net',[]))} net positions")
            live_orders = kl.get_orders()
            log_kite_call("orders()", f"{len(live_orders)} orders today")

        # Sort by score
        STATE["signals"]   = sorted(signals, key=lambda x: x["score"], reverse=True)
        STATE["last_scan"] = str(datetime.now())
        set_progress(7, "Scan complete ✅", len(candidates), len(candidates))

        buys       = [s for s in signals if s["action"] == "BUY"]
        strong_buy = [s for s in signals if s["score"] >= CFG["SCORE_STRONG_BUY"]]
        buy_grade  = [s for s in signals if s["score"] >= CFG["SCORE_BUY"]]
        watch      = [s for s in signals if s["score"] >= CFG["SCORE_WATCHLIST"]]
        top5       = [(s["symbol"], s["score"]) for s in signals[:5]]

        log.info(f"\n  ✅ SCAN COMPLETE")
        log.info(f"     Stocks scanned:   {len(quote_data)}")
        log.info(f"     Deep analysed:    {len(candidates)}")
        log.info(f"     Skipped (no data):{skipped}")
        log.info(f"     Gap-ups found:    {len(gap_alerts)}")
        log.info(f"     Signals scored:   {len(signals)}")
        log.info(f"     Strong Buy (≥{CFG['SCORE_STRONG_BUY']}): {len(strong_buy)}")
        log.info(f"     Buy grade  (≥{CFG['SCORE_BUY']}): {len(buy_grade)}")
        log.info(f"     Watchlist  (≥{CFG['SCORE_WATCHLIST']}): {len(watch)}")
        log.info(f"     Buys triggered:   {len(buys)}")
        log.info(f"     Open positions:   {list(STATE['positions'].keys()) or 'None'}")
        log.info(f"     Bear premium:     +{bear_premium} pts (VIX={vix:.1f}, regime={reg_label})")
        log.info(f"     Extreme fear:     {'YES — all buys blocked' if extreme_fear else 'No'}")
        log.info(f"     Available cap:    ₹{available_cap:,.0f}")
        if top5:
            log.info(f"     Top 5 scores:     {top5}")

    except Exception as e:
        err = f"Scan failed: {e}"
        log.error(err)
        STATE["errors"].append({"time": str(datetime.now()), "msg": err})
    finally:
        STATE["scanning"] = False


# ═════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═════════════════════════════════════════════════════════════
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/state")
def api_state():
    tl    = STATE["trade_log"]
    pnl   = sum(t["pnl"] for t in tl)
    wins  = [t for t in tl if t["pnl"] > 0]
    wr    = round(len(wins)/len(tl)*100, 1) if tl else 0
    return jsonify(_np_sanitize({
        "signals":       STATE["signals"][:100],    # top 100 by score
        "positions":     STATE["positions"],
        "trade_log":     tl[-30:],
        "gap_alerts":    STATE["gap_alerts"],
        "regime":        STATE["regime"],
        "fii_dii":       STATE["fii_dii"],
        "vix":           STATE["vix"],
        "last_scan":     STATE["last_scan"],
        "scanning":      STATE["scanning"],
        "scan_progress": STATE["scan_progress"],
        "paper_trade":   CFG["PAPER_TRADE"],
        "kite_calls":    STATE["kite_calls"][-20:],
        "errors":        STATE["errors"][-30:],
        "stats": {
            "universe_size":  STATE["universe_size"],
            "signals_count":  len(STATE["signals"]),
            "gap_alerts":     len(STATE["gap_alerts"]),
            "open_positions": len(STATE["positions"]),
            "total_trades":   len(tl),
            "total_pnl":      round(pnl, 2),
            "win_rate":       wr,
        },
        "auth": STATE["auth"],
    }))


@app.route("/api/progress")
def api_progress():
    """Polled every 1s during scan to show real-time progress."""
    return jsonify(_np_sanitize({
        "scanning":      STATE["scanning"],
        "scan_progress": STATE["scan_progress"],
    }))


@app.route("/api/paper/buy", methods=["POST"])
def paper_buy():
    """Manual paper trade buy — deducts from cash, adds to holdings."""
    data   = request.get_json(force=True) or {}
    sym    = data.get("symbol", "").upper().strip()
    qty    = int(data.get("qty", 0))
    price  = float(data.get("price", 0))
    port   = STATE["paper_portfolio"]
    if not sym or qty <= 0 or price <= 0:
        return jsonify({"ok": False, "error": "symbol/qty/price required"}), 400
    cost = round(price * qty, 2)
    if cost > port["cash"]:
        return jsonify({"ok": False, "error": f"Insufficient cash ₹{port['cash']:.0f}"}), 400
    port["cash"] = round(port["cash"] - cost, 2)
    if sym in port["holdings"]:
        h = port["holdings"][sym]
        total_qty = h["qty"] + qty
        h["avg_price"] = round((h["avg_price"] * h["qty"] + price * qty) / total_qty, 2)
        h["qty"]       = total_qty
        h["invested"]  = round(h["avg_price"] * total_qty, 2)
    else:
        port["holdings"][sym] = {"qty": qty, "avg_price": round(price, 2), "invested": cost, "symbol": sym}
    log.info(f"  [Paper] BUY {qty} {sym} @ ₹{price} | Cash left ₹{port['cash']:.0f}")
    save_paper_portfolio()
    return jsonify({"ok": True, "cash": port["cash"], "holdings": port["holdings"]})


@app.route("/api/paper/sell", methods=["POST"])
def paper_sell():
    """Manual paper trade sell — adds proceeds to cash."""
    data   = request.get_json(force=True) or {}
    sym    = data.get("symbol", "").upper().strip()
    qty    = int(data.get("qty", 0))
    price  = float(data.get("price", 0))
    port   = STATE["paper_portfolio"]
    if sym not in port["holdings"]:
        return jsonify({"ok": False, "error": f"{sym} not in holdings"}), 400
    h = port["holdings"][sym]
    if qty > h["qty"]:
        return jsonify({"ok": False, "error": f"Only {h['qty']} held"}), 400
    proceeds = round(price * qty, 2)
    pnl      = round((price - h["avg_price"]) * qty, 2)
    port["cash"]          = round(port["cash"] + proceeds, 2)
    port["pnl_realised"]  = round(port["pnl_realised"] + pnl, 2)
    h["qty"] -= qty
    if h["qty"] <= 0:
        del port["holdings"][sym]
    else:
        h["invested"] = round(h["avg_price"] * h["qty"], 2)
    log.info(f"  [Paper] SELL {qty} {sym} @ ₹{price} | PnL ₹{pnl} | Cash ₹{port['cash']:.0f}")
    save_paper_portfolio()
    return jsonify({"ok": True, "pnl": pnl, "cash": port["cash"], "holdings": port["holdings"]})


@app.route("/api/paper/reset", methods=["POST"])
def paper_reset():
    """Reset paper portfolio to ₹10,000."""
    STATE["paper_portfolio"] = {
        "capital": 10000.0, "cash": 10000.0,
        "holdings": {}, "pnl_realised": 0.0,
    }
    save_paper_portfolio()
    return jsonify({"ok": True})


@app.route("/api/paper/state")
def paper_state():
    port   = STATE["paper_portfolio"]
    # Compute unrealised PnL from live signals
    unrealised = 0.0
    holdings_enriched = {}
    sig_map = {s["symbol"]: s["ltp"] for s in STATE["signals"]}
    for sym, h in port["holdings"].items():
        ltp  = sig_map.get(sym, h["avg_price"])
        unrl = round((ltp - h["avg_price"]) * h["qty"], 2)
        pct  = round((ltp - h["avg_price"]) / h["avg_price"] * 100, 2) if h["avg_price"] else 0
        unrealised += unrl
        holdings_enriched[sym] = {**h, "ltp": ltp, "unrealised": unrl, "pct": pct}
    total_value = round(port["cash"] + sum(h["avg_price"]*h["qty"] for h in port["holdings"].values()), 2)
    return jsonify(_np_sanitize({
        "capital":       port["capital"],
        "cash":          port["cash"],
        "holdings":      holdings_enriched,
        "pnl_realised":  port["pnl_realised"],
        "pnl_unrealised": round(unrealised, 2),
        "total_value":   total_value,
        "total_pnl":     round(port["pnl_realised"] + unrealised, 2),
        "total_pnl_pct": round((total_value - port["capital"]) / port["capital"] * 100, 2),
    }))


@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    if not STATE["scanning"]:
        threading.Thread(target=run_full_scan, daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})


@app.route("/api/gap-alerts")
def gap_alerts():
    return jsonify(STATE["gap_alerts"])


@app.route("/api/indices")
def api_indices():
    """Returns pivot + indicator snapshot for NIFTY, BANKNIFTY, FINNIFTY, MIDCAP."""
    return jsonify(_np_sanitize(STATE.get("indices", {})))


@app.route("/api/kite-calls")
def kite_calls():
    """Shows every Kite API method called — for transparency."""
    return jsonify(STATE["kite_calls"])


@app.route("/api/signals/buy")
def signals_buy():
    """Returns only BUY / STRONG BUY signals — the actionable list."""
    min_score = int(request.args.get("min_score", CFG["SCORE_BUY"]))
    sigs = [s for s in STATE["signals"] if s.get("score", 0) >= min_score]
    return jsonify({
        "count":        len(sigs),
        "min_score":    min_score,
        "bear_market":  STATE["regime"].get("label", ""),
        "signals":      sigs[:50],
    })


@app.route("/api/fno-signals")
def fno_signals():
    """
    Institutional F&O signal panel — PRO version.

    Improvements over v1:
      • Real option LTP fetched from Kite NFO (kite.ltp("NFO:NIFTY2641024000CE"))
      • Correct NFO symbol built from next weekly expiry date
      • Structure-based SL (delta-weighted from underlying pivot SL)
      • T1/T2/T3 anchored to underlying TP structure, min % enforced
      • IV assessment: cheap / fair / expensive relative to VIX
      • Time-of-day window filter (9:45–11:30, 13:45–15:00)
      • OI confirmation score from NSE option chain
      • Setup quality score 0–5 per signal
    """
    try:
        if not STATE["signals"] and not STATE.get("indices"):
            return jsonify({"error": "No scan data yet — run a full scan first."})

        reg_label  = STATE["regime"].get("label", "")
        reg_conf   = float(STATE["regime"].get("confidence", 50.0))
        vix        = float(STATE.get("vix", 15.0) or 15.0)
        bear_blk   = reg_label.startswith("BEAR")
        sw_blk     = reg_label.startswith("SIDEWAYS")
        idx_pivots = STATE.get("indices", {})
        oc_nifty   = nse.get_option_chain_pcr("NIFTY")
        fii        = STATE.get("fii_dii", {})

        # Need kl for real LTP fetch; gracefully fall back if not available
        _kl = kl    # global KiteLayer — may be None before first scan

        KNOWN_LOTS = {
            "NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
            "RELIANCE": 250, "TCS": 150, "INFY": 300,
            "HDFCBANK": 550, "ICICIBANK": 700, "SBIN": 1500,
            "TATAMOTORS": 1425, "WIPRO": 1500, "BAJFINANCE": 125, "LT": 175,
        }
        INDEX_SYMS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCAP"}

        results = []
        seen    = set()

        # ──────────────────────────────────────────────────────
        # BLOCK 1: Index signals (NIFTY, BANKNIFTY, FINNIFTY)
        # ──────────────────────────────────────────────────────
        for idx_key, idx_data in idx_pivots.items():
            if not idx_data or idx_data.get("error"):
                continue
            ltp   = float(idx_data.get("ltp", 0))
            trend = idx_data.get("trend", "SIDEWAYS")
            if trend == "SIDEWAYS" or ltp < 100:
                continue

            direction = "CALL" if trend == "BULL" else "PUT"

            # Hard pivot gate: CALL only above PP, PUT only below PP
            if direction == "CALL" and not idx_data.get("above_pivot", False):
                continue
            if direction == "PUT" and idx_data.get("above_pivot", True):
                continue

            pivot  = float(idx_data.get("pivot", ltp))
            r1     = float(idx_data.get("r1",    ltp * 1.01))
            r2     = float(idx_data.get("r2",    ltp * 1.02))
            s1     = float(idx_data.get("s1",    ltp * 0.99))
            s2     = float(idx_data.get("s2",    ltp * 0.98))

            und_sl = s1 if direction == "CALL" else r1
            und_tp = r1 if direction == "CALL" else s1

            lot = KNOWN_LOTS.get(idx_key, 50)

            # Build full signal (fetches real Kite LTP internally)
            sig = OptionEngine.build_signal(
                kl            = _kl,
                underlying    = idx_key,
                ltp           = ltp,
                direction     = direction,
                underlying_sl = und_sl,
                underlying_tp = und_tp,
                vix           = vix,
                oc_data       = oc_nifty,
                equity_score  = 0,
                equity_signal = trend,
                pivot_data    = idx_data,
            )
            sig.update({
                "symbol":     idx_key,
                "type":       "INDEX",
                "lot_size":   int(lot),
                "confidence": round(reg_conf, 1),
                "bear_blocked":     bear_blk,
                "sideways_blocked": sw_blk,
            })
            results.append(sig)
            seen.add(idx_key)

        # ──────────────────────────────────────────────────────
        # BLOCK 2: Top equity stock F&O signals
        # ──────────────────────────────────────────────────────
        for eq_sig in STATE["signals"]:
            sym   = eq_sig["symbol"]
            score = eq_sig.get("score", 0)
            if score < CFG["SCORE_WATCHLIST"] or sym in seen or len(results) >= 20:
                break
            seen.add(sym)

            ltp = float(eq_sig.get("ltp", 0))
            if ltp < 50:
                continue

            atr = float(eq_sig.get("atr", ltp * 0.02))
            sl  = float(eq_sig.get("sl",  ltp - 1.5 * atr))
            tp  = float(eq_sig.get("tp",  ltp + 3.5 * atr))

            direction = "CALL" if eq_sig.get("signal_class") in ("buy", "strong-buy") else "PUT"

            # Pivot conflict check against NIFTY
            nifty_piv = idx_pivots.get("NIFTY", {})
            above_pp  = nifty_piv.get("above_pivot", True)
            pivot_warn = (direction == "CALL" and not above_pp) or \
                         (direction == "PUT"  and above_pp)
            if pivot_warn and score < CFG["SCORE_STRONG_BUY"]:
                continue

            # Fetch OI for stocks (best-effort from NSE)
            oc_stock = {}
            try:
                oc_stock = nse.get_option_chain_pcr(sym)
            except Exception:
                oc_stock = oc_nifty   # fallback to NIFTY OI context

            lot = KNOWN_LOTS.get(sym, 500)

            sig = OptionEngine.build_signal(
                kl            = _kl,
                underlying    = sym,
                ltp           = ltp,
                direction     = direction,
                underlying_sl = sl,
                underlying_tp = tp,
                vix           = vix,
                oc_data       = oc_stock,
                equity_score  = int(score),
                equity_signal = eq_sig.get("signal", ""),
                pivot_data    = nifty_piv,
            )
            sig.update({
                "symbol":        sym,
                "type":          "STOCK",
                "lot_size":      int(lot),
                "confidence":    round(reg_conf, 1),
                "above_vwap":    eq_sig.get("above_vwap"),
                "vol_surge":     eq_sig.get("vol_surge", False),
                "smc_bos":       bool(eq_sig.get("smc_bos", False)),
                "smc_ob":        eq_sig.get("smc_ob"),
                "bear_blocked":     bear_blk,
                "sideways_blocked": sw_blk,
            })
            results.append(sig)

        return jsonify(_np_sanitize({
            "count":            len(results),
            "signals":          results,
            "regime":           reg_label,
            "vix":              vix,
            "bear_blocked":     bear_blk,
            "sideways_blocked": sw_blk,
            "confidence":       round(reg_conf, 1),
            "note": (
                "Signals with premium_live=true use real Kite NFO LTP. "
                "premium_live=false = IV-model estimate — verify on Zerodha before trading."
            ),
        }))

    except Exception as e:
        log.error(f"  [FnO] /api/fno-signals error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500



@app.route("/api/top-picks")
def top_picks():
    """High-conviction STRONG BUY picks with R:R >= 2.0 and volume confirmation."""
    sigs   = STATE.get("signals", [])
    regime = STATE.get("regime", {})
    vix    = float(STATE.get("vix", 15) or 15)
    picks  = []
    for s in sigs:
        score = s.get("score", 0)
        if score < CFG["SCORE_STRONG_BUY"]: continue
        entry = s.get("entry", s.get("ltp", 0))
        sl    = s.get("sl", 0)
        tp    = s.get("tp", 0)
        if entry <= 0 or sl <= 0 or tp <= 0: continue
        rr = round((tp - entry) / max(entry - sl, 0.01), 2)
        if rr < 2.0: continue
        if not s.get("vol_surge", False): continue
        picks.append({
            **s, "rr": rr,
            "trade_plan": f"Entry Rs{entry:.1f} | SL Rs{sl:.1f} | TP Rs{tp:.1f} | R:R {rr}x",
            "conviction": "VERY HIGH" if score >= 24 else "HIGH",
        })
        if len(picks) >= 10: break
    return jsonify(_np_sanitize({
        "count":  len(picks),
        "picks":  picks,
        "regime": regime.get("label", "--"),
        "vix":    vix,
        "note":   f"STRONG BUY >={CFG['SCORE_STRONG_BUY']}/30 + R:R >=2.0 + volume confirmed",
    }))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": str(datetime.now()),
                    "paper": CFG["PAPER_TRADE"],
                    "universe": STATE["universe_size"],
                    "auth": STATE["auth"]["status"]})


# ═════════════════════════════════════════════════════════════
# KITE AUTH ROUTES — zero-touch daily login
# ═════════════════════════════════════════════════════════════

@app.route("/kite-login")
def kite_login():
    """Visit this every morning to re-authenticate with Zerodha."""
    url = build_login_url()
    log.info(f"  [Auth] Redirecting to Zerodha: {url}")
    return redirect(url)


@app.route("/callback")
def kite_callback():
    """Zerodha redirects here with ?request_token= after login."""
    global kl, universe
    request_token = request.args.get("request_token")
    status        = request.args.get("status", "")
    if status != "success" or not request_token:
        err = request.args.get("message", "Login cancelled or failed.")
        log.error(f"  [Auth] Callback error: {err}")
        return _auth_page("❌ Login Failed", err, False), 400
    try:
        from kiteconnect import KiteConnect
        kite_tmp     = KiteConnect(api_key=CFG["KITE_API_KEY"])
        session      = kite_tmp.generate_session(request_token, api_secret=CFG["KITE_API_SECRET"])
        access_token = session["access_token"]
        save_token(access_token)
        STATE["auth"]["token_saved"] = datetime.now().isoformat()
        kl       = init_kite(access_token)
        universe = None  # force rebuild with new session
        log.info(f"  [Auth] ✅ Session active for {STATE['auth']['user_name']}")
        return _auth_page(
            f"✅ Logged in as {STATE['auth']['user_name']}",
            "Token saved. Scanner is ready — press FULL SCAN on the dashboard.",
            True
        )
    except Exception as e:
        log.error(f"  [Auth] Token exchange failed: {e}")
        return _auth_page("❌ Token Exchange Failed", str(e), False), 500


@app.route("/api/auth-status")
def auth_status():
    return jsonify(STATE["auth"])


@app.route("/api/token-status")
def token_status():
    """Quick endpoint — returns token age and freshness without a Kite call."""
    fresh = is_token_fresh(max_age_hours=8)
    token_saved = STATE["auth"].get("token_saved")
    age_h = None
    try:
        import json as _json
        with open("/tmp/kite_token.json") as _f:
            _d = _json.load(_f)
        from datetime import datetime as _dt
        age_h = round((_dt.now() - _dt.fromisoformat(_d["saved_at"])).total_seconds() / 3600, 1)
    except Exception:
        pass
    return jsonify({
        "fresh":       fresh,
        "age_hours":   age_h,
        "status":      STATE["auth"]["status"],
        "login_url":   STATE["auth"].get("login_url"),
        "user_name":   STATE["auth"].get("user_name"),
    })


def _auth_page(title: str, message: str, success: bool) -> str:
    colour = "#00ff9d" if success else "#ff2d55"
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Kite Auth</title>
<style>
  body{{background:#010408;color:#cdd9e5;font-family:'JetBrains Mono',monospace;
        display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}}
  .card{{background:#070f1a;border:1px solid #0e2040;border-radius:12px;
          padding:2.5rem 3rem;max-width:480px;text-align:center}}
  h2{{color:{colour};font-size:1.3rem;margin-bottom:1rem}}
  p{{color:#8899aa;line-height:1.7;font-size:.85rem}}
  a{{display:inline-block;margin-top:1.5rem;padding:.6rem 1.4rem;
     background:#0e2040;border:1px solid #00e5ff;color:#00e5ff;
     border-radius:6px;text-decoration:none;font-size:.8rem}}
</style></head>
<body><div class="card">
  <h2>{title}</h2><p>{message}</p>
  <a href="/">← Back to Dashboard</a>
</div></body></html>"""


# ═════════════════════════════════════════════════════════════
# DASHBOARD HTML  (Bloomberg Terminal Style)
# ═════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════
# SCHEDULER
# ═════════════════════════════════════════════════════════════
def start_scheduler():
    h, m  = CFG["SCAN_TIME"].split(":")
    sched = BackgroundScheduler(timezone="Asia/Kolkata")
    sched.add_job(run_full_scan, "cron", hour=int(h), minute=int(m),
                  day_of_week="mon-fri", id="morning_scan")
    sched.add_job(run_full_scan, "cron", hour=14, minute=0,
                  day_of_week="mon-fri", id="midday_scan")
    sched.start()
    log.info(f"📅 Scheduler: {CFG['SCAN_TIME']} + 14:00 IST (Mon–Fri)")


def probe_existing_token():
    """
    Validate saved token silently on startup.
    Also warns if the token is older than 8 hours (stale risk).
    Doesn't crash if missing — dashboard shows auth banner.
    """
    global kl
    STATE["auth"]["login_url"] = build_login_url()
    token = CFG["ACCESS_TOKEN"]
    if not token:
        STATE["auth"]["status"] = "missing"
        log.warning("  [Auth] ⚠️  No token — visit /kite-login to authenticate")
        return
    # Freshness warning before we even try Kite
    if not is_token_fresh(max_age_hours=8):
        STATE["auth"]["status"] = "stale"
        log.warning("  [Auth] ⚠️  Token is >8h old — may have expired. Visit /kite-login")
    try:
        kl = init_kite(token)
        if STATE["auth"]["status"] == "stale":
            pass   # Kite confirmed it; keep stale label so dashboard can warn
        log.info("  [Auth] ✅ Token valid — ready to scan")
    except Exception as e:
        log.warning(f"  [Auth] Token invalid ({e}) — visit /kite-login")


# ── Runs under both gunicorn and `python app.py` ─────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>InsTrade · NSE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:   #000; --bg1: #0a0a0a; --bg2: #111; --bg3: #1a1a1a;
    --border: rgba(255,255,255,0.07); --border2: rgba(255,255,255,0.12);
    --text: #f5f5f7; --muted: rgba(255,255,255,0.45);
    --green: #30d158; --green-dim: rgba(48,209,88,0.12);
    --red: #ff453a;   --red-dim:   rgba(255,69,58,0.12);
    --amber: #ffd60a; --amber-dim: rgba(255,214,10,0.1);
    --blue:  #0a84ff; --blue-dim:  rgba(10,132,255,0.1);
    --purple: #bf5af2;
    --r: 14px; --r-sm: 10px;
  }
  html { background: var(--bg); color: var(--text); font-family: 'DM Sans', -apple-system, sans-serif; font-size: 14px; -webkit-font-smoothing: antialiased; }
  body { min-height: 100vh; padding: 0 0 60px; }

  /* TOPBAR */
  .topbar {
    position: sticky; top: 0; z-index: 100;
    background: rgba(0,0,0,0.75); backdrop-filter: blur(28px) saturate(180%);
    -webkit-backdrop-filter: blur(28px) saturate(180%);
    border-bottom: 0.5px solid var(--border2);
    padding: 0 24px; display: flex; align-items: center; height: 52px; gap: 0;
  }
  .logo { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 600; letter-spacing: -0.3px; color: var(--text); text-decoration: none; margin-right: 24px; flex-shrink: 0; }
  .logo-mark { width: 30px; height: 30px; background: var(--blue); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: #fff; }
  .exch-badge { font-size: 10px; font-weight: 500; color: var(--muted); border: 0.5px solid var(--border2); padding: 2px 7px; border-radius: 5px; letter-spacing: 1px; }
  .top-pills { display: flex; align-items: center; gap: 5px; flex: 1; overflow-x: auto; scrollbar-width: none; }
  .top-pills::-webkit-scrollbar { display: none; }
  .pill { display: flex; align-items: center; gap: 6px; padding: 5px 11px; border-radius: 20px; font-size: 11.5px; font-weight: 500; white-space: nowrap; border: 0.5px solid var(--border2); background: var(--bg2); color: var(--muted); cursor: default; transition: background 0.2s; flex-shrink: 0; }
  .pill:hover { background: var(--bg3); }
  .pill .val { color: var(--text); font-weight: 600; font-family: 'DM Mono', monospace; }
  .pill .dot { width: 6px; height: 6px; border-radius: 50%; }
  .pill-bear { border-color: var(--red-dim); background: var(--red-dim); color: #ff6961; }
  .pill-bear .val { color: var(--red); }
  .pill-bull { border-color: var(--green-dim); background: var(--green-dim); color: #5de872; }
  .pill-bull .val { color: var(--green); }
  .top-right { display: flex; align-items: center; gap: 8px; margin-left: 12px; flex-shrink: 0; }
  .btn-ghost { padding: 6px 14px; border-radius: 20px; border: 0.5px solid var(--border2); background: transparent; color: var(--muted); font-family: 'DM Sans', sans-serif; font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.2s; }
  .btn-ghost:hover { background: var(--bg3); color: var(--text); }
  .btn-ghost.active { color: var(--text); background: var(--bg3); border-color: rgba(255,255,255,0.2); }
  .btn-scan { padding: 6px 16px; border-radius: 20px; border: none; background: var(--blue); color: #fff; font-family: 'DM Sans', sans-serif; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .btn-scan:hover { background: #0071e3; }
  .btn-scan:disabled { opacity: 0.5; cursor: not-allowed; }
  .clock { font-family: 'DM Mono', monospace; font-size: 12px; color: var(--muted); margin-left: 12px; }

  /* REGIME ALERT */
  .regime-banner { display: flex; align-items: center; gap: 10px; padding: 11px 18px; border-radius: var(--r-sm); margin-bottom: 20px; font-size: 12.5px; line-height: 1.5; }
  .regime-banner.bear { background: var(--red-dim); border: 0.5px solid rgba(255,69,58,0.25); color: #ff6961; }
  .regime-banner.bull { background: var(--green-dim); border: 0.5px solid rgba(48,209,88,0.25); color: #5de872; }
  .regime-banner.sideways { background: var(--amber-dim); border: 0.5px solid rgba(255,214,10,0.25); color: #ffe566; }
  .regime-banner strong { font-weight: 600; }
  .regime-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .regime-dot.bear { background: var(--red); box-shadow: 0 0 0 3px rgba(255,69,58,0.25); animation: pulse-r 2s infinite; }
  .regime-dot.bull { background: var(--green); box-shadow: 0 0 0 3px rgba(48,209,88,0.25); animation: pulse-g 2s infinite; }
  .regime-dot.sideways { background: var(--amber); }
  @keyframes pulse-r { 0%,100%{box-shadow:0 0 0 3px rgba(255,69,58,0.25)}50%{box-shadow:0 0 0 7px rgba(255,69,58,0.06)} }
  @keyframes pulse-g { 0%,100%{box-shadow:0 0 0 3px rgba(48,209,88,0.25)}50%{box-shadow:0 0 0 7px rgba(48,209,88,0.06)} }

  /* PROGRESS BAR */
  .progress-wrap { background: var(--bg2); border: 0.5px solid var(--border2); border-radius: var(--r-sm); padding: 14px 18px; margin-bottom: 20px; display: none; }
  .progress-wrap.active { display: block; }
  .progress-label { font-size: 12px; color: var(--muted); margin-bottom: 8px; display: flex; justify-content: space-between; }
  .progress-bar-bg { height: 3px; background: var(--bg3); border-radius: 2px; overflow: hidden; }
  .progress-bar-fill { height: 100%; background: var(--blue); border-radius: 2px; transition: width 0.4s ease; width: 0%; }

  /* MAIN */
  .main { max-width: 1340px; margin: 0 auto; padding: 28px 24px 0; }

  /* STATS ROW */
  .stats-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 22px; }
  .stat-card { background: var(--bg1); border: 0.5px solid var(--border); border-radius: var(--r-sm); padding: 14px 16px; }
  .stat-card .lbl { font-size: 10.5px; font-weight: 500; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .stat-card .val { font-size: 21px; font-weight: 600; font-family: 'DM Mono', monospace; letter-spacing: -1px; color: var(--text); }
  .stat-card .sub { font-size: 10.5px; color: var(--muted); margin-top: 3px; }
  .stat-card.c-green .val { color: var(--green); }
  .stat-card.c-amber .val { color: var(--amber); }
  .stat-card.c-blue  .val { color: var(--blue); }
  .stat-card.c-red   .val { color: var(--red); }

  /* SECTION TITLE */
  .sec-title { font-size: 10.5px; font-weight: 600; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 12px; display: flex; align-items: center; gap: 10px; }
  .sec-title::after { content: ''; flex: 1; height: 0.5px; background: var(--border); }

  /* INDEX GRID */
  .index-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
  .index-card { background: var(--bg1); border: 0.5px solid var(--border); border-radius: var(--r); padding: 20px 20px 16px; position: relative; overflow: hidden; transition: border-color 0.25s, transform 0.2s; cursor: default; }
  .index-card:hover { border-color: var(--border2); transform: translateY(-2px); }
  .index-card .idx-lbl { font-size: 10.5px; font-weight: 600; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }
  .tag { font-size: 9.5px; padding: 2px 8px; border-radius: 5px; font-weight: 600; letter-spacing: 0.5px; }
  .tag-bull { background: var(--green-dim); color: var(--green); }
  .tag-bear { background: var(--red-dim); color: var(--red); }
  .tag-side { background: var(--amber-dim); color: var(--amber); }
  .idx-price { font-size: 26px; font-weight: 600; font-family: 'DM Mono', monospace; letter-spacing: -1.5px; margin-bottom: 4px; }
  .idx-chg { font-size: 12.5px; font-weight: 500; font-family: 'DM Mono', monospace; }
  .idx-chg.up { color: var(--green); }
  .idx-chg.dn { color: var(--red); }
  .mini-chart { height: 36px; margin: 12px 0 14px; }
  .mini-chart svg { width: 100%; height: 100%; }
  .pivot-section { border-top: 0.5px solid var(--border); padding-top: 12px; }
  .pivot-row { display: flex; justify-content: space-between; align-items: center; padding: 2px 0; font-size: 10.5px; }
  .pivot-row .pk { color: var(--muted); font-weight: 500; }
  .pivot-row .pv { font-family: 'DM Mono', monospace; font-size: 10.5px; }
  .pivot-row.r  .pv { color: rgba(48,209,88,0.75); }
  .pivot-row.s  .pv { color: rgba(255,69,58,0.75); }
  .pivot-row.pp .pk, .pivot-row.pp .pv { color: var(--amber); }
  .no-data-msg { text-align: center; color: var(--muted); font-size: 11px; padding: 24px 0; }

  /* BOTTOM GRID */
  .bottom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 24px; }
  .panel { background: var(--bg1); border: 0.5px solid var(--border); border-radius: var(--r); padding: 20px; }
  .panel-title { font-size: 11px; font-weight: 600; color: var(--muted); letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 16px; }
  .regime-big { font-size: 40px; font-weight: 700; font-family: 'DM Mono', monospace; letter-spacing: -2px; line-height: 1; }
  .regime-big.bear { color: var(--red); }
  .regime-big.bull { color: var(--green); }
  .regime-big.sideways { color: var(--amber); }
  .regime-sub { font-size: 12.5px; color: var(--muted); margin-top: 5px; margin-bottom: 18px; }
  .crit-list { list-style: none; display: flex; flex-direction: column; gap: 7px; }
  .crit-item { display: flex; align-items: flex-start; gap: 9px; font-size: 11.5px; color: var(--muted); line-height: 1.4; }
  .crit-icon { flex-shrink: 0; margin-top: 1px; width: 14px; height: 14px; }
  .crit-icon.pass { color: var(--green); }
  .crit-icon.fail { color: var(--red); }
  .crit-icon.warn { color: var(--amber); }
  .crit-txt { flex: 1; }
  .crit-pts { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 10.5px; }

  /* SCORE GRID */
  .score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
  .score-item { background: var(--bg2); border: 0.5px solid var(--border); border-radius: 10px; padding: 13px 13px 11px; }
  .score-item .si-lbl { font-size: 9.5px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
  .score-item .si-val { font-size: 19px; font-weight: 700; font-family: 'DM Mono', monospace; color: var(--text); }
  .score-item .si-max { font-size: 10.5px; color: var(--muted); font-family: 'DM Mono', monospace; }
  .score-bar { height: 3px; background: var(--bg3); border-radius: 2px; margin-top: 9px; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 2px; transition: width 0.9s cubic-bezier(.4,0,.2,1); }
  .score-legend { margin-top: 16px; padding: 13px; background: var(--bg2); border-radius: 10px; border: 0.5px solid var(--border); }
  .score-legend .leg-title { font-size: 11px; color: var(--muted); margin-bottom: 8px; }
  .score-legend .leg-row { display: flex; gap: 12px; font-size: 10.5px; font-weight: 600; }

  /* TABS */
  .tab-bar { display: flex; gap: 0; border-bottom: 0.5px solid var(--border); margin-bottom: 22px; }
  .tab { padding: 10px 18px; font-size: 13px; font-weight: 500; color: var(--muted); cursor: pointer; border-bottom: 1.5px solid transparent; margin-bottom: -0.5px; transition: all 0.2s; white-space: nowrap; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--text); border-bottom-color: var(--blue); }

  /* TABLES */
  .tbl-wrap { background: var(--bg1); border: 0.5px solid var(--border); border-radius: var(--r); overflow: hidden; margin-bottom: 24px; }
  table { width: 100%; border-collapse: collapse; }
  thead th { text-align: left; font-size: 10.5px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.7px; padding: 10px 14px; border-bottom: 0.5px solid var(--border); white-space: nowrap; }
  tbody td { padding: 10px 14px; border-bottom: 0.5px solid var(--border); font-size: 12.5px; vertical-align: middle; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: var(--bg2); }
  .no-data { padding: 32px; text-align: center; color: var(--muted); font-size: 12px; }
  .sym { font-weight: 600; color: var(--text); font-size: 13px; }
  .mono { font-family: 'DM Mono', monospace; }
  .g { color: var(--green); } .r { color: var(--red); } .a { color: var(--amber); } .b { color: var(--blue); }
  .badge { font-size: 10px; padding: 2px 8px; border-radius: 5px; font-weight: 600; font-family: 'DM Mono', monospace; }
  .badge-sb  { background: var(--green-dim); color: var(--green); }
  .badge-buy { background: var(--blue-dim);  color: var(--blue);  }
  .badge-wl  { background: var(--amber-dim); color: var(--amber); }
  .badge-av  { background: var(--red-dim);   color: var(--red);   }
  .badge-gap-up { background: var(--green-dim); color: var(--green); }
  .badge-gap-ex { background: rgba(255,214,10,0.15); color: var(--amber); }

  /* PAPER TRADE PANEL */
  .paper-grid { display: grid; grid-template-columns: 360px 1fr; gap: 16px; margin-bottom: 24px; }
  .paper-form { background: var(--bg1); border: 0.5px solid var(--border); border-radius: var(--r); padding: 20px; }
  .form-title { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 16px; }
  .form-row { margin-bottom: 12px; }
  .form-row label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 5px; font-weight: 500; }
  .form-row input { width: 100%; background: var(--bg2); border: 0.5px solid var(--border2); border-radius: 8px; padding: 8px 12px; color: var(--text); font-family: 'DM Mono', monospace; font-size: 13px; outline: none; transition: border-color 0.2s; }
  .form-row input:focus { border-color: var(--blue); }
  .form-btns { display: flex; gap: 8px; margin-top: 14px; }
  .btn-buy  { flex: 1; padding: 9px; border-radius: 8px; border: none; background: var(--green); color: #000; font-weight: 700; font-size: 12px; cursor: pointer; }
  .btn-sell { flex: 1; padding: 9px; border-radius: 8px; border: none; background: var(--red);   color: #fff; font-weight: 700; font-size: 12px; cursor: pointer; }
  .btn-reset { padding: 9px 14px; border-radius: 8px; border: 0.5px solid var(--border2); background: transparent; color: var(--muted); font-size: 12px; cursor: pointer; }
  .paper-stats { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 14px; }
  .ps-card { background: var(--bg2); border: 0.5px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .ps-card .ps-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; }
  .ps-card .ps-val { font-size: 18px; font-weight: 600; font-family: 'DM Mono', monospace; color: var(--text); }

  /* KITE LOG */
  .kite-log { background: var(--bg2); border: 0.5px solid var(--border); border-radius: var(--r); padding: 14px; max-height: 300px; overflow-y: auto; font-family: 'DM Mono', monospace; font-size: 11px; }
  .kl-row { display: flex; gap: 10px; padding: 4px 0; border-bottom: 0.5px solid var(--border); }
  .kl-row:last-child { border-bottom: none; }
  .kl-time { color: var(--muted); flex-shrink: 0; }
  .kl-method { color: var(--blue); flex-shrink: 0; }
  .kl-detail { color: var(--muted); }

  /* TOASTS */
  .toast { position: fixed; bottom: 24px; right: 24px; background: var(--bg2); border: 0.5px solid var(--border2); border-radius: 10px; padding: 12px 18px; font-size: 12.5px; z-index: 9999; display: none; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }

  /* ANIMATIONS */
  @keyframes fadeSlide { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
  .anim { animation: fadeSlide 0.45s ease both; }

  /* RESPONSIVE */
  @media (max-width: 1100px) { .stats-row { grid-template-columns: repeat(3,1fr); } .index-grid { grid-template-columns: repeat(2,1fr); } }
  @media (max-width: 768px)  { .bottom-grid { grid-template-columns: 1fr; } .paper-grid { grid-template-columns: 1fr; } .index-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<!-- TOPBAR -->
<header class="topbar">
  <a class="logo" href="#">
    <div class="logo-mark">IT</div>
    InsTrade <span class="exch-badge">NSE</span>
  </a>
  <div class="top-pills">
    <div class="pill" id="pill-regime"><span class="dot" id="regime-dot" style="background:var(--muted)"></span> Regime <span class="val" id="pill-regime-val">—</span></div>
    <div class="pill">VIX <span class="val" id="pill-vix">—</span></div>
    <div class="pill">PCR <span class="val" id="pill-pcr">—</span></div>
    <div class="pill">FII <span class="val" id="pill-fii">—</span></div>
    <div class="pill">Scanned <span class="val" id="pill-scanned">0</span></div>
    <div class="pill">Signals <span class="val g" id="pill-signals">0</span></div>
    <div class="pill">Gap-Ups <span class="val a" id="pill-gaps">0</span></div>
    <div class="pill">Portfolio <span class="val" id="pill-portfolio">₹10,000</span></div>
    <div class="pill">P&L <span class="val" id="pill-pnl">+₹0</span></div>
  </div>
  <div class="top-right">
    <span class="btn-ghost active" id="badge-mode">PAPER</span>
    <button class="btn-ghost" onclick="location.href='/kite-login'">Zerodha</button>
    <button class="btn-scan" id="scan-btn" onclick="triggerScan()">Full Scan</button>
  </div>
  <div class="clock" id="clock">--:--:--</div>
</header>

<!-- MAIN -->
<div class="main">

  <!-- Regime Banner -->
  <div class="regime-banner bear" id="regime-banner" style="display:none">
    <div class="regime-dot bear" id="regime-dot-banner"></div>
    <p id="regime-banner-text"><strong>Bear Market Active</strong> — New BUY orders blocked. Displaying watchlist ideas only.</p>
  </div>

  <!-- Progress Bar -->
  <div class="progress-wrap" id="progress-wrap">
    <div class="progress-label">
      <span id="progress-label">Scanning…</span>
      <span id="progress-pct">0%</span>
    </div>
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="progress-fill"></div></div>
  </div>

  <!-- Stats Row -->
  <div class="stats-row">
    <div class="stat-card c-blue anim"><div class="lbl">Scanned</div><div class="val" id="st-scanned">0</div><div class="sub">NSE Universe</div></div>
    <div class="stat-card c-amber anim"><div class="lbl">Gap Alerts</div><div class="val" id="st-gaps">0</div><div class="sub">≥2% gap stocks</div></div>
    <div class="stat-card c-green anim"><div class="lbl">Signals Found</div><div class="val" id="st-signals">0</div><div class="sub">Total scored</div></div>
    <div class="stat-card anim"><div class="lbl">Open Positions</div><div class="val" id="st-positions">0</div><div class="sub">of 6 max slots</div></div>
    <div class="stat-card c-green anim"><div class="lbl">Session P&amp;L</div><div class="val" id="st-pnl">+₹0</div><div class="sub" id="st-pnl-sub">Win 0 / 0 trades</div></div>
    <div class="stat-card anim"><div class="lbl">Portfolio</div><div class="val" id="st-portfolio">₹10K</div><div class="sub" id="st-portfolio-sub">Paper mode</div></div>
  </div>

  <!-- Index Cards -->
  <div class="sec-title">Live Indices · Pivot Levels</div>
  <div class="index-grid" id="index-grid">
    <div class="index-card"><p class="no-data-msg">Run a Full Scan to load index data</p></div>
  </div>

  <!-- Regime + Scoring -->
  <div class="bottom-grid">
    <div class="panel">
      <div class="panel-title" id="regime-panel-title">Market Regime Engine</div>
      <div class="regime-big bear" id="regime-big">—</div>
      <div class="regime-sub" id="regime-sub">Run a scan to evaluate market regime</div>
      <ul class="crit-list" id="crit-list"><li class="crit-item" style="justify-content:center;padding:16px 0"><span style="color:var(--muted);font-size:11.5px">No data yet</span></li></ul>
    </div>
    <div class="panel">
      <div class="panel-title">Signal Scoring Engine · Max 30 pts</div>
      <div class="score-grid">
        <div class="score-item"><div class="si-lbl">Technical</div><div><span class="si-val">×5</span> <span class="si-max">/ 5 pts</span></div><div class="score-bar"><div class="score-fill" style="width:100%;background:var(--blue)"></div></div></div>
        <div class="score-item"><div class="si-lbl">Breakout</div><div><span class="si-val">×5</span> <span class="si-max">/ 5 pts</span></div><div class="score-bar"><div class="score-fill" style="width:100%;background:var(--green)"></div></div></div>
        <div class="score-item"><div class="si-lbl">Fundamental</div><div><span class="si-val">×5</span> <span class="si-max">/ 5 pts</span></div><div class="score-bar"><div class="score-fill" style="width:100%;background:var(--amber)"></div></div></div>
        <div class="score-item"><div class="si-lbl">Institutional</div><div><span class="si-val">×5</span> <span class="si-max">/ 5 pts</span></div><div class="score-bar"><div class="score-fill" style="width:100%;background:var(--blue)"></div></div></div>
        <div class="score-item"><div class="si-lbl">Regime Gate</div><div><span class="si-val">×2</span> <span class="si-max">/ 2 pts</span></div><div class="score-bar"><div class="score-fill" style="width:100%;background:var(--red)"></div></div></div>
        <div class="score-item"><div class="si-lbl">SMC</div><div><span class="si-val">×5</span> <span class="si-max">/ 5 pts</span></div><div class="score-bar"><div class="score-fill" style="width:100%;background:var(--green)"></div></div></div>
        <div class="score-item" style="grid-column:span 2"><div class="si-lbl">Fibonacci</div><div><span class="si-val">×3</span> <span class="si-max">/ 3 pts</span></div><div class="score-bar"><div class="score-fill" style="width:100%;background:var(--purple)"></div></div></div>
      </div>
      <div class="score-legend">
        <div class="leg-title">Signal strength thresholds</div>
        <div class="leg-row">
          <span class="g">20+ Strong Buy</span>
          <span class="b">13+ Buy</span>
          <span class="a">9+ Watchlist</span>
          <span class="r">&lt;9 Avoid</span>
        </div>
      </div>
    </div>
  </div>

  <!-- TABS -->
  <div class="tab-bar">
    <div class="tab active" onclick="switchTab('signals',this)">Signals</div>
    <div class="tab" onclick="switchTab('paper',this)">Paper Trade</div>
    <div class="tab" onclick="switchTab('gaps',this)">Gap Alerts</div>
    <div class="tab" onclick="switchTab('positions',this)">Positions</div>
    <div class="tab" onclick="switchTab('trades',this)">Trade Log</div>
    <div class="tab" onclick="switchTab('kite',this)">Kite API Log</div>
  </div>

  <!-- SIGNALS TAB -->
  <div id="tab-signals">
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Symbol</th><th>LTP</th><th>Gap%</th>
          <th style="color:var(--blue)">Tech</th><th style="color:var(--green)">Brk</th>
          <th style="color:var(--amber)">Fund</th><th style="color:var(--blue)">Inst</th>
          <th style="color:var(--red)">SMC</th><th style="color:var(--purple)">Fib</th>
          <th>Score</th><th>Signal</th><th>SL / TP</th>
        </tr></thead>
        <tbody id="sig-body"><tr><td colspan="12" class="no-data">Press Full Scan to load signals</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- PAPER TRADE TAB -->
  <div id="tab-paper" style="display:none">
    <div class="paper-grid">
      <div class="paper-form">
        <div class="form-title">Manual Order</div>
        <div class="form-row"><label>Symbol</label><input id="pf-sym" placeholder="e.g. RELIANCE" style="text-transform:uppercase"></div>
        <div class="form-row"><label>Price (₹)</label><input id="pf-price" type="number" placeholder="0.00"></div>
        <div class="form-row"><label>Quantity</label><input id="pf-qty" type="number" placeholder="1" value="1"></div>
        <div class="form-btns">
          <button class="btn-buy" onclick="paperOrder('buy')">BUY</button>
          <button class="btn-sell" onclick="paperOrder('sell')">SELL</button>
          <button class="btn-reset" onclick="paperReset()">Reset</button>
        </div>
        <div id="paper-msg" style="margin-top:10px;font-size:11.5px;color:var(--muted)"></div>
      </div>
      <div>
        <div class="paper-stats">
          <div class="ps-card"><div class="ps-lbl">Cash</div><div class="ps-val" id="pp-cash">₹10,000</div></div>
          <div class="ps-card"><div class="ps-lbl">Unrealised</div><div class="ps-val" id="pp-unrl">₹0</div></div>
          <div class="ps-card"><div class="ps-lbl">Realised P&L</div><div class="ps-val" id="pp-real">₹0</div></div>
        </div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>Symbol</th><th>Qty</th><th>Avg</th><th>LTP</th><th>Unrealised</th></tr></thead>
            <tbody id="pp-body"><tr><td colspan="5" class="no-data">No holdings</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- GAP ALERTS TAB -->
  <div id="tab-gaps" style="display:none">
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Symbol</th><th>Gap %</th><th>LTP</th><th>Volume</th><th>Category</th></tr></thead>
        <tbody id="gap-body"><tr><td colspan="5" class="no-data">No gap alerts — run a scan first</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- POSITIONS TAB -->
  <div id="tab-positions" style="display:none">
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Symbol</th><th>Entry</th><th>SL</th><th>TP</th><th>Qty</th><th>Score</th><th>Unrealised</th><th>Mode</th></tr></thead>
        <tbody id="pos-body"><tr><td colspan="8" class="no-data">No open positions</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- TRADES TAB -->
  <div id="tab-trades" style="display:none">
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>Reason</th><th>Time</th></tr></thead>
        <tbody id="trade-body"><tr><td colspan="7" class="no-data">No completed trades</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- KITE LOG TAB -->
  <div id="tab-kite" style="display:none">
    <div class="kite-log" id="kite-log"><div style="color:var(--muted);text-align:center;padding:20px">No API calls logged yet</div></div>
  </div>

</div><!-- /main -->

<div class="toast" id="toast">🔍 Scan triggered — this may take 2–3 minutes…</div>

<script>
// ── State ──────────────────────────────────────────────────
let allSigs = [], progressTimer = null;

// ── Clock ──────────────────────────────────────────────────
setInterval(() => {
  const el = document.getElementById('clock');
  if (el) el.textContent = new Date().toLocaleTimeString('en-IN', {hour12: false});
}, 1000);

// ── Tab switching ──────────────────────────────────────────
const TABS = ['signals','paper','gaps','positions','trades','kite'];
function switchTab(name, el) {
  TABS.forEach(t => {
    document.getElementById('tab-'+t).style.display = (t === name ? '' : 'none');
  });
  document.querySelectorAll('.tab-bar .tab').forEach(t => t.classList.remove('active'));
  if (el) el.classList.add('active');
}

// ── Score-bar animation on load ────────────────────────────
setTimeout(() => {
  document.querySelectorAll('.score-fill').forEach(el => {
    const w = el.style.width; el.style.width = '0';
    setTimeout(() => { el.style.width = w; }, 60);
  });
}, 400);

// ── Scan trigger ───────────────────────────────────────────
async function triggerScan() {
  const btn = document.getElementById('scan-btn');
  btn.disabled = true; btn.textContent = '⟳ Scanning…';
  document.getElementById('toast').style.display = 'block';
  setTimeout(() => document.getElementById('toast').style.display = 'none', 3000);
  await fetch('/api/scan', {method: 'POST'});
  startProgressPoll();
}

// ── Progress polling ───────────────────────────────────────
function startProgressPoll() {
  if (progressTimer) clearInterval(progressTimer);
  document.getElementById('progress-wrap').classList.add('active');
  progressTimer = setInterval(async () => {
    try {
      const d = await (await fetch('/api/state')).json();
      const p = d.scan_progress || {};
      document.getElementById('progress-label').textContent = p.step_label || 'Scanning…';
      document.getElementById('progress-pct').textContent = (p.pct || 0) + '%';
      document.getElementById('progress-fill').style.width = (p.pct || 0) + '%';
      if (!d.scanning) {
        clearInterval(progressTimer);
        document.getElementById('progress-wrap').classList.remove('active');
        document.getElementById('scan-btn').disabled = false;
        document.getElementById('scan-btn').textContent = 'Full Scan';
        render(d);
        loadIndexCards();
        loadPaperState();
      }
    } catch(e) {}
  }, 1500);
}

// ── Main state fetch ───────────────────────────────────────
async function fetchState() {
  try {
    const d = await (await fetch('/api/state')).json();
    render(d);
    if (d.scanning) startProgressPoll();
  } catch(e) {}
}

function render(d) {
  const sigs = d.signals || [];
  const regime = d.regime || {};
  const fii = d.fii_dii || {};
  const vix = d.vix || 0;
  allSigs = sigs;

  // Pills
  const rl = (regime.label || '—').split(' ')[0];
  const isBear = rl.startsWith('BEAR'), isBull = rl.startsWith('BULL');
  const pReg = document.getElementById('pill-regime');
  pReg.className = 'pill ' + (isBear ? 'pill-bear' : isBull ? 'pill-bull' : '');
  document.getElementById('pill-regime-val').textContent = rl;
  const dot = document.getElementById('regime-dot');
  dot.style.background = isBear ? 'var(--red)' : isBull ? 'var(--green)' : 'var(--amber)';
  document.getElementById('pill-vix').textContent = vix ? vix.toFixed(1) : '—';
  document.getElementById('pill-vix').className = 'val ' + (vix > 20 ? 'r' : vix > 15 ? 'a' : 'g');
  document.getElementById('pill-pcr').textContent = fii.pcr || '—';
  const fiiNet = fii.fii_net;
  document.getElementById('pill-fii').textContent = fiiNet != null ? (fiiNet >= 0 ? '+' : '') + '₹' + (Math.abs(fiiNet)/1e7).toFixed(0) + 'Cr' : '—';
  document.getElementById('pill-fii').className = 'val ' + (fiiNet > 0 ? 'g' : fiiNet < 0 ? 'r' : '');
  document.getElementById('pill-scanned').textContent = (d.stats?.universe_size || 0).toLocaleString('en-IN');
  document.getElementById('pill-signals').textContent = (d.stats?.signals_count || 0).toLocaleString('en-IN');
  document.getElementById('pill-gaps').textContent = d.stats?.gap_alerts || 0;

  // Stats row
  document.getElementById('st-scanned').textContent = (d.stats?.universe_size || 0).toLocaleString('en-IN');
  document.getElementById('st-gaps').textContent = d.stats?.gap_alerts || 0;
  document.getElementById('st-signals').textContent = d.stats?.signals_count || 0;
  document.getElementById('st-positions').textContent = d.stats?.open_positions || 0;
  const tl = d.trade_log || [];
  const totalPnl = tl.reduce((a,t) => a + t.pnl, 0);
  const wins = tl.filter(t => t.pnl > 0).length;
  const stPnl = document.getElementById('st-pnl');
  stPnl.textContent = (totalPnl >= 0 ? '+' : '') + '₹' + Math.abs(totalPnl).toLocaleString('en-IN');
  stPnl.className = 'val ' + (totalPnl >= 0 ? 'c-green' : 'c-red');
  document.getElementById('st-pnl-sub').textContent = `Win ${wins} / ${tl.length} trades`;

  // Regime banner
  const banner = document.getElementById('regime-banner');
  const bannerText = document.getElementById('regime-banner-text');
  const bannerDot = document.getElementById('regime-dot-banner');
  if (regime.label) {
    banner.style.display = 'flex';
    banner.className = 'regime-banner ' + (isBear ? 'bear' : isBull ? 'bull' : 'sideways');
    bannerDot.className = 'regime-dot ' + (isBear ? 'bear' : isBull ? 'bull' : 'sideways');
    bannerText.innerHTML = isBear
      ? '<strong>Bear Market Active</strong> — New BUY orders blocked. Displaying watchlist ideas only.'
      : isBull
      ? '<strong>Bull Market Confirmed</strong> — Scan active for BUY setups.'
      : '<strong>Sideways Market</strong> — Reduced position sizing recommended.';
  } else { banner.style.display = 'none'; }

  // Regime panel
  const regScore = regime.score || 0; const regConf = regime.confidence || 0;
  document.getElementById('regime-panel-title').textContent = `Market Regime Engine · ${regScore}/7 pts · ${regConf.toFixed(1)}%`;
  const rb = document.getElementById('regime-big');
  rb.textContent = rl || '—';
  rb.className = 'regime-big ' + (isBear ? 'bear' : isBull ? 'bull' : 'sideways');
  document.getElementById('regime-sub').textContent = isBull ? 'Momentum confirmed — regime gate OPEN' : 'Bull requires ≥ 4/7 points';
  const flags = regime.flags || [];
  const cl = document.getElementById('crit-list');
  if (flags.length) {
    cl.innerHTML = flags.map(f => {
      const isPass = f.startsWith('✅'), isFail = f.startsWith('❌'), isWarn = f.startsWith('⚠');
      const cls = isPass ? 'pass' : isFail ? 'fail' : 'warn';
      const ic = isPass
        ? `<svg class="crit-icon pass" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><polyline points="4,7 6.5,9.5 10,5" stroke="currentColor" stroke-width="1.5"/></svg>`
        : isFail
        ? `<svg class="crit-icon fail" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><line x1="4.5" y1="4.5" x2="9.5" y2="9.5" stroke="currentColor" stroke-width="1.5"/><line x1="9.5" y1="4.5" x2="4.5" y2="9.5" stroke="currentColor" stroke-width="1.5"/></svg>`
        : `<svg class="crit-icon warn" viewBox="0 0 14 14" fill="none"><path d="M7 2L12.5 12H1.5L7 2Z" stroke="currentColor" stroke-width="1.5"/><line x1="7" y1="6" x2="7" y2="8.5" stroke="currentColor" stroke-width="1.5"/><circle cx="7" cy="10.2" r="0.6" fill="currentColor"/></svg>`;
      return `<li class="crit-item">${ic}<span class="crit-txt">${f.replace(/^[✅❌⚠️]\s*/,'')}</span></li>`;
    }).join('');
  } else { cl.innerHTML = '<li class="crit-item" style="justify-content:center;padding:12px 0"><span style="color:var(--muted);font-size:11.5px">No regime data yet</span></li>'; }

  // Signals table
  renderSignals(sigs);
  // Gap alerts
  renderGaps(d.gap_alerts || []);
  // Positions
  renderPositions(d.positions || {}, sigs);
  // Trades
  renderTrades(tl);
  // Kite log
  renderKiteLog(d.kite_calls || []);
}

function renderSignals(sigs) {
  const tb = document.getElementById('sig-body');
  if (!sigs.length) { tb.innerHTML = '<tr><td colspan="12" class="no-data">Press Full Scan to load signals</td></tr>'; return; }
  tb.innerHTML = sigs.map(s => {
    const sc = s.score || 0;
    const sigLabel = sc >= 20 ? 'STRONG BUY' : sc >= 13 ? 'BUY' : sc >= 9 ? 'WATCHLIST' : 'AVOID';
    const sigCls   = sc >= 20 ? 'badge-sb' : sc >= 13 ? 'badge-buy' : sc >= 9 ? 'badge-wl' : 'badge-av';
    const bk = s.breakdown || {};
    return `<tr onclick="autofillPaper(${JSON.stringify(s).replace(/"/g,'&quot;')})">
      <td class="sym">${s.symbol}</td>
      <td class="mono">₹${(s.ltp||0).toLocaleString('en-IN')}</td>
      <td class="${s.gap_pct>0?'g':'r'} mono">${s.gap_pct>0?'+':''}${(s.gap_pct||0).toFixed(1)}%</td>
      <td class="mono b">${(bk.technical?.score||0)}/${bk.technical?.max||5}</td>
      <td class="mono g">${(bk.breakout?.score||0)}/${bk.breakout?.max||5}</td>
      <td class="mono a">${(bk.fundamental?.score||0)}/${bk.fundamental?.max||5}</td>
      <td class="mono b">${(bk.institutional?.score||0)}/${bk.institutional?.max||5}</td>
      <td class="mono">${(bk.smc?.score||0)}/${bk.smc?.max||5}</td>
      <td class="mono" style="color:var(--purple)">${(bk.fibonacci?.score||0)}/${bk.fibonacci?.max||3}</td>
      <td class="mono" style="font-weight:700;color:${sc>=20?'var(--green)':sc>=13?'var(--blue)':sc>=9?'var(--amber)':'var(--red)'}">${sc}/30</td>
      <td><span class="badge ${sigCls}">${sigLabel}</span></td>
      <td class="mono" style="font-size:11px"><span class="r">₹${(s.sl||0).toLocaleString('en-IN')}</span> · <span class="g">₹${(s.tp||0).toLocaleString('en-IN')}</span></td>
    </tr>`;
  }).join('');
}

function renderGaps(gaps) {
  const tb = document.getElementById('gap-body');
  if (!gaps.length) { tb.innerHTML = '<tr><td colspan="5" class="no-data">No gap alerts detected</td></tr>'; return; }
  tb.innerHTML = gaps.map(g => {
    const cat = g.gap_pct >= 10 ? '🔥 Explosive' : g.gap_pct >= 5 ? '🚀 Strong' : '📈 Moderate';
    const catCls = g.gap_pct >= 10 ? 'badge badge-gap-ex' : 'badge badge-gap-up';
    return `<tr>
      <td class="sym">${g.symbol}</td>
      <td class="g mono">+${g.gap_pct.toFixed(2)}%</td>
      <td class="mono">₹${(g.ltp||0).toLocaleString('en-IN')}</td>
      <td class="mono">${(g.volume||0).toLocaleString('en-IN')}</td>
      <td><span class="${catCls}">${cat}</span></td>
    </tr>`;
  }).join('');
}

function renderPositions(pos, sigs) {
  const tb = document.getElementById('pos-body');
  const entries = Object.entries(pos || {});
  if (!entries.length) { tb.innerHTML = '<tr><td colspan="8" class="no-data">No open positions</td></tr>'; return; }
  const sigMap = {}; (sigs||[]).forEach(s => { sigMap[s.symbol] = s.ltp; });
  tb.innerHTML = entries.map(([sym,p]) => {
    const ltp  = sigMap[sym] || p.entry;
    const unrl = ((ltp - p.entry) * p.qty).toFixed(2);
    return `<tr>
      <td class="sym">${sym}</td>
      <td class="mono">₹${p.entry.toFixed(2)}</td>
      <td class="mono r">₹${p.sl.toFixed(2)}</td>
      <td class="mono g">₹${p.tp.toFixed(2)}</td>
      <td class="mono">${p.qty}</td>
      <td class="mono b">${p.score}/30</td>
      <td class="mono ${unrl>=0?'g':'r'}">${unrl>=0?'+':''}₹${Math.abs(unrl).toLocaleString('en-IN')}</td>
      <td style="font-size:11px;color:${p.trailing?'var(--amber)':'var(--muted)'}">${p.trailing?'TRAILING':'FIXED'}</td>
    </tr>`;
  }).join('');
}

function renderTrades(tl) {
  const tb = document.getElementById('trade-body');
  const t2 = [...(tl||[])].reverse();
  if (!t2.length) { tb.innerHTML = '<tr><td colspan="7" class="no-data">No completed trades</td></tr>'; return; }
  tb.innerHTML = t2.map(t => `<tr>
    <td class="sym">${t.symbol}</td>
    <td class="mono">₹${t.entry.toFixed(2)}</td>
    <td class="mono">₹${t.exit.toFixed(2)}</td>
    <td class="mono">${t.qty}</td>
    <td class="mono ${t.pnl>=0?'g':'r'}" style="font-weight:600">${t.pnl>=0?'+':''}₹${Math.abs(t.pnl).toLocaleString('en-IN')}</td>
    <td style="font-size:11px;color:var(--muted)">${t.reason}</td>
    <td style="font-size:11px;color:var(--muted)">${new Date(t.date).toLocaleTimeString('en-IN',{hour12:false})}</td>
  </tr>`).join('');
}

function renderKiteLog(calls) {
  const el = document.getElementById('kite-log');
  if (!calls?.length) { el.innerHTML = '<div style="color:var(--muted);text-align:center;padding:20px">No API calls logged yet</div>'; return; }
  el.innerHTML = [...calls].reverse().map(c =>
    `<div class="kl-row"><span class="kl-time">${c.time}</span><span class="kl-method">kite.${c.method}</span><span class="kl-detail">${c.detail}</span></div>`
  ).join('');
}

// ── Index cards ────────────────────────────────────────────
async function loadIndexCards() {
  try {
    const d = await (await fetch('/api/indices')).json();
    const grid = document.getElementById('index-grid');
    const keys = ['NIFTY','BANKNIFTY','FINNIFTY','MIDCAP'];
    const labels = {'NIFTY':'NIFTY 50','BANKNIFTY':'BANK NIFTY','FINNIFTY':'FIN NIFTY','MIDCAP':'MIDCAP 150'};
    let html = '';
    let any = false;
    keys.forEach(k => {
      const idx = d[k]; if (!idx || idx.error) return; any = true;
      const trend = idx.trend || 'SIDEWAYS';
      const isBull = trend === 'BULL', isBear = trend === 'BEAR';
      const ltp = idx.ltp || 0; const chg = idx.change_pct || 0;
      const tagCls = isBull ? 'tag-bull' : isBear ? 'tag-bear' : 'tag-side';
      const chgCls = chg >= 0 ? 'up' : 'dn';
      const lineCol = isBull ? 'rgba(48,209,88,0.55)' : isBear ? 'rgba(255,69,58,0.55)' : 'rgba(255,214,10,0.55)';
      const fillCol = isBull ? 'rgba(48,209,88,0.07)' : isBear ? 'rgba(255,69,58,0.07)' : 'rgba(255,214,10,0.07)';
      const pts = isBull ? '0,36 25,30 50,26 75,20 100,15 125,12 150,9 175,6 200,3'
                         : isBear ? '0,4 25,8 50,12 75,18 100,22 125,26 150,30 175,34 200,36'
                         : '0,20 25,18 50,22 75,19 100,21 125,18 150,22 175,20 200,19';
      html += `<div class="index-card anim">
        <div class="idx-lbl">${labels[k]||k} <span class="tag ${tagCls}">${trend}</span></div>
        <div class="idx-price">₹${ltp.toLocaleString('en-IN',{minimumFractionDigits:0})}</div>
        <div class="idx-chg ${chgCls}">${chg>=0?'+':''}${chg.toFixed(2)}% today</div>
        <div class="mini-chart">
          <svg viewBox="0 0 200 40" preserveAspectRatio="none">
            <polyline fill="none" stroke="${lineCol}" stroke-width="1.5" points="${pts}"/>
            <polyline fill="${fillCol}" stroke="none" points="${pts} 200,40 0,40"/>
          </svg>
        </div>
        <div class="pivot-section">
          ${[['R3',idx.r3],['R2',idx.r2],['R1',idx.r1]].filter(x=>x[1]).map(([l,v])=>`<div class="pivot-row r"><span class="pk">${l}</span><span class="pv">₹${Math.round(v).toLocaleString('en-IN')}</span></div>`).join('')}
          <div class="pivot-row pp"><span class="pk">PP</span><span class="pv">₹${Math.round(idx.pivot||0).toLocaleString('en-IN')}</span></div>
          ${[['S1',idx.s1],['S2',idx.s2],['S3',idx.s3]].filter(x=>x[1]).map(([l,v])=>`<div class="pivot-row s"><span class="pk">${l}</span><span class="pv">₹${Math.round(v).toLocaleString('en-IN')}</span></div>`).join('')}
        </div>
      </div>`;
    });
    grid.innerHTML = any ? html : '<div class="index-card" style="grid-column:1/-1"><p class="no-data-msg">Run a Full Scan to load index data</p></div>';
  } catch(e) {}
}

// ── Paper trade ────────────────────────────────────────────
async function loadPaperState() {
  try {
    const d = await (await fetch('/api/paper/state')).json();
    document.getElementById('pp-cash').textContent = '₹' + (d.cash||0).toLocaleString('en-IN');
    const unrl = document.getElementById('pp-unrl');
    unrl.textContent = (d.pnl_unrealised>=0?'+':'') + '₹' + Math.abs(d.pnl_unrealised||0).toLocaleString('en-IN');
    unrl.className = 'ps-val ' + (d.pnl_unrealised>=0?'g':'r');
    const real = document.getElementById('pp-real');
    real.textContent = (d.pnl_realised>=0?'+':'') + '₹' + Math.abs(d.pnl_realised||0).toLocaleString('en-IN');
    real.className = 'ps-val ' + (d.pnl_realised>=0?'g':'r');
    document.getElementById('pill-portfolio').textContent = '₹' + (d.total_value||10000).toLocaleString('en-IN');
    const pnlEl = document.getElementById('pill-pnl');
    pnlEl.textContent = (d.total_pnl>=0?'+':'') + '₹' + Math.abs(d.total_pnl||0).toLocaleString('en-IN');
    pnlEl.className = 'val ' + (d.total_pnl>=0?'g':'r');
    document.getElementById('st-portfolio').textContent = '₹' + Math.round((d.total_value||10000)/1000) + 'K';
    document.getElementById('st-portfolio-sub').textContent = `${(d.total_pnl_pct||0).toFixed(1)}% total return`;
    const tb = document.getElementById('pp-body');
    const h = d.holdings || {};
    const entries = Object.values(h);
    if (!entries.length) { tb.innerHTML = '<tr><td colspan="5" class="no-data">No holdings</td></tr>'; return; }
    tb.innerHTML = entries.map(h => {
      const unrl = h.unrealised || 0;
      return `<tr>
        <td class="sym">${h.symbol}</td>
        <td class="mono">${h.qty}</td>
        <td class="mono">₹${h.avg_price.toFixed(2)}</td>
        <td class="mono">₹${h.ltp.toFixed(2)}</td>
        <td class="mono ${unrl>=0?'g':'r'}">${unrl>=0?'+':''}₹${Math.abs(unrl).toLocaleString('en-IN')}</td>
      </tr>`;
    }).join('');
  } catch(e) {}
}

function autofillPaper(s) {
  document.getElementById('pf-sym').value   = s.symbol;
  document.getElementById('pf-price').value = s.ltp;
  document.getElementById('pf-qty').value   = 1;
}

async function paperOrder(side) {
  const sym   = document.getElementById('pf-sym').value.trim().toUpperCase();
  const price = parseFloat(document.getElementById('pf-price').value);
  const qty   = parseInt(document.getElementById('pf-qty').value);
  const msg   = document.getElementById('paper-msg');
  if (!sym || !price || !qty) { msg.textContent = '⚠ Fill all fields'; msg.style.color='var(--amber)'; return; }
  try {
    const r = await (await fetch('/api/paper/'+side, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({symbol:sym,price,qty})})).json();
    msg.style.color = r.ok ? 'var(--green)' : 'var(--red)';
    msg.textContent = r.ok ? `✅ ${side.toUpperCase()} ${qty} ${sym} @ ₹${price}` : '❌ ' + r.error;
    if (r.ok) loadPaperState();
  } catch(e) { msg.textContent = '❌ Request failed'; msg.style.color='var(--red)'; }
}

async function paperReset() {
  if (!confirm('Reset paper portfolio to ₹10,000?')) return;
  await fetch('/api/paper/reset', {method:'POST'});
  loadPaperState();
}

// ── Kick off ───────────────────────────────────────────────
fetchState();
loadPaperState();
loadIndexCards();
setInterval(fetchState,    30000);
setInterval(loadPaperState,15000);
setInterval(loadIndexCards,60000);
</script>
</body>
</html>
"""

log.info(f"🚀 Institutional Trader Pro — Full NSE Edition")
log.info(f"   Mode: {'PAPER' if CFG['PAPER_TRADE'] else '⚠ LIVE'} | SMC Engine: ON | Fibonacci Engine: ON | Yahoo Finance: REMOVED")
probe_existing_token()
start_scheduler()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)