"""
═══════════════════════════════════════════════════════════════
  INSTITUTIONAL TRADER PRO — app.py
  Scans FULL NSE universe. No hardcoded stock list.
  Every Kite API call is labelled.
═══════════════════════════════════════════════════════════════
"""

import os, json, logging, threading, math
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
    SMCEngine, FibonacciEngine,
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
    "scan_progress": {          # real-time progress tracking
        "step":       0,        # current step 0-7
        "step_label": "",       # e.g. "Loading NSE universe..."
        "stocks_done": 0,       # stocks analysed so far
        "stocks_total": 0,      # total candidates
        "pct":        0,        # 0-100
    },
    "paper_portfolio": {        # paper trading portfolio state
        "capital":    10000.0,  # starting capital ₹10,000
        "cash":       10000.0,  # available cash
        "holdings":   {},       # symbol → {qty, avg_price, invested}
        "pnl_realised": 0.0,
    },
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

        reg_s, reg_f, regime_ok, reg_label = MarketRegime.check(
            nifty_df, vix,
            banknifty_df = banknifty_df if not banknifty_df.empty else None,
            finnifty_df  = finnifty_df  if not finnifty_df.empty  else None,
            midcap_df    = midcap_df    if not midcap_df.empty    else None,
        )
        STATE["regime"] = {"score": reg_s, "label": reg_label, "flags": reg_f, "vix": vix}
        log.info(f"  → Regime: {reg_label} | VIX: {vix:.1f} | FII: ₹{fii.get('fii_net',0)/1e7:.0f}Cr | PCR: {oc.get('pcr',1)}")

        bear_market   = (reg_label.startswith("BEAR"))
        extreme_fear  = (vix > 30)
        regime_ok     = not extreme_fear
        bear_premium  = 4 if bear_market else 0

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
        # STEP 5: Get available capital from Kite
        # ─────────────────────────────────────────────────────
        available_cap = kl.get_available_capital()
        if available_cap < 1000:
            log.warning(f"  [Capital] Kite returned ₹{available_cap:.2f} — using CFG fallback ₹{CFG['CAPITAL']:,}")
            available_cap = CFG["CAPITAL"]
        log_kite_call("margins('equity')", f"₹{available_cap:,.0f} available")
        log.info(f"  STEP 5 — Available capital: ₹{available_cap:,.0f}")
        set_progress(5, f"Capital confirmed ₹{available_cap:,.0f} — starting deep analysis...")

        # ─────────────────────────────────────────────────────
        # STEP 6: Deep Analysis — Historical data per stock
        # KITE API: kite.historical_data(token, ...) per stock
        # ─────────────────────────────────────────────────────
        log.info(f"  STEP 6 — Deep analysis on {len(candidates)} stocks...")
        signals    = []
        open_count = len(STATE["positions"])
        set_progress(6, f"Deep-analysing {len(candidates)} candidates...", 0, len(candidates))

        for i, cand in enumerate(candidates, 1):
            sym   = cand["symbol"]
            token = cand["token"]
            quote = cand

            if not token:
                continue

            # Update progress every 10 stocks
            if i % 10 == 0 or i == 1:
                set_progress(6, f"Analysing {sym} ({i}/{len(candidates)})...", i, len(candidates))

            try:
                # KITE API: kite.historical_data(token, from_dt, to_dt, interval)
                df = kl.get_ohlcv(token, days=300)
                if df.empty or len(df) < 60:
                    continue
                df = TechnicalEngine.compute(df)
                if df.empty:
                    continue
                log_kite_call("historical_data", f"{sym} — {len(df)} candles")

                row = df.iloc[-1]

                # Score all engines including SMC and Fibonacci
                tech_s,  tech_f  = TechnicalEngine.score(row)
                brk_s,   brk_f   = BreakoutScanner.score(df, row, nifty_df, quote)
                fund_d           = FundamentalEngine.compute(df, row, nifty_df)
                inst_s,  inst_f  = InstitutionalEngine.score(nse, sym, fii, oc, vix)
                smc_d            = SMCEngine.compute(df)
                fib_d            = FibonacciEngine.compute(df)
                full             = build_score(tech_s, tech_f, brk_s, brk_f,
                                               fund_d, inst_s, inst_f, reg_s, reg_f,
                                               smc_d, fib_d)

                entry  = cand["ltp"] * (1 + CFG["SLIPPAGE_PCT"])
                sl, tp = RiskManager.sl_tp(entry, float(row["atr"]), fib_d)
                qty    = RiskManager.position_size(entry, sl, available_cap)

                # Label whether TP is Fib-based or ATR-based for display
                tp_source = (
                    "FIB 1.618"
                    if fib_d.get("score", 0) >= 2
                       and fib_d.get("fib_tp_1618")
                       and fib_d["fib_tp_1618"] > entry * 1.02
                    else "ATR×3.5"
                )

                sig = {
                    "symbol":       sym,
                    "ltp":          round(cand["ltp"], 2),
                    "gap_pct":      cand.get("gap_pct", 0),
                    "change_pct":   cand.get("change_pct", 0),
                    "volume":       cand.get("volume", 0),
                    "rsi":          round(float(row.get("rsi", 0)), 1),
                    "adx":          round(float(row.get("adx", 0)), 1),
                    "atr":          round(float(row.get("atr", 0)), 2),
                    "sl":           sl,
                    "tp":           tp,
                    "tp_source":    tp_source,
                    "qty":          qty,
                    "score":        full["total"],
                    "max_score":    full["max"],   # 30
                    "signal":       full["signal"],
                    "signal_class": full["signal_class"],
                    "breakdown":    full["breakdown"],
                    "action":       "HOLD",
                    "time":         str(datetime.now()),
                    "pe":           fund_d.get("pe"),
                    "roe":          fund_d.get("roe"),
                    "mcap_cr":      fund_d.get("mcap_cr"),
                    # SMC fields for display
                    "smc_ob":       smc_d.get("ob"),
                    "smc_fvg":      smc_d.get("fvg"),
                    "smc_bos":      smc_d.get("bos", False),
                    "smc_choch":    smc_d.get("choch", False),
                    "smc_sweep":    smc_d.get("sweep", False),
                    # Fibonacci fields for display
                    "fib_score":        fib_d.get("score", 0),
                    "fib_golden_high":  fib_d.get("golden_zone_high"),
                    "fib_golden_low":   fib_d.get("golden_zone_low"),
                    "fib_tp_1618":      fib_d.get("fib_tp_1618"),
                    "fib_swing_high":   fib_d.get("swing_high"),
                    "fib_swing_low":    fib_d.get("swing_low"),
                    "fib_in_golden":    fib_d.get("price_in_golden", False),
                    "fib_at_key":       fib_d.get("fib_at_key_level", False),
                }

                # ── Entry: BUY ────────────────────────────────
                buy_threshold = CFG["SCORE_BUY"] + bear_premium
                if (full["total"] >= buy_threshold
                        and sym not in STATE["positions"]
                        and open_count < CFG["MAX_OPEN_TRADES"]
                        and regime_ok):

                    if CFG["PAPER_TRADE"]:
                        log.info(f"  📝 PAPER BUY {sym} | {full['signal']} | Score {full['total']}/{full['max']} | Qty {qty}")
                    else:
                        # KITE API: kite.place_order(...)
                        order_id = kl.place_order(sym, "NSE", "BUY", qty)
                        log_kite_call("place_order", f"BUY {qty} {sym}")
                        if order_id:
                            # KITE API: kite.place_gtt(...) — auto SL+TP
                            gtt_id = kl.place_gtt(sym, "NSE", qty, entry, sl, tp)
                            log_kite_call("place_gtt", f"{sym} SL:{sl} TP:{tp}")

                    STATE["positions"][sym] = {
                        "entry": entry, "sl": sl, "tp": tp, "qty": qty,
                        "score": full["total"], "date": str(datetime.now()),
                        "token": token, "trailing": False,
                    }
                    open_count += 1
                    sig["action"] = "BUY"

                # ── Exit: SELL ────────────────────────────────
                elif sym in STATE["positions"]:
                    pos    = STATE["positions"][sym]
                    pos    = RiskManager.trailing_stop(pos, cand["ltp"], float(row["atr"]))
                    STATE["positions"][sym] = pos
                    reason = RiskManager.check_exit(cand["ltp"], pos, row)

                    if reason:
                        pnl = (cand["ltp"] - pos["entry"]) * pos["qty"] - CFG["BROKERAGE"] * 2
                        if CFG["PAPER_TRADE"]:
                            log.info(f"  📝 PAPER SELL {sym} | {reason} | PnL ₹{pnl:.0f}")
                        else:
                            # KITE API: kite.place_order(SELL)
                            kl.place_order(sym, "NSE", "SELL", pos["qty"])
                            log_kite_call("place_order", f"SELL {pos['qty']} {sym} — {reason}")

                        STATE["trade_log"].append({
                            "symbol": sym, "entry": pos["entry"],
                            "exit": cand["ltp"], "qty": pos["qty"],
                            "pnl": round(pnl, 2), "reason": reason,
                            "date": str(datetime.now()),
                        })
                        del STATE["positions"][sym]
                        open_count -= 1
                        sig["action"] = f"SELL ({reason})"

                signals.append(sig)

                if i % 50 == 0:
                    log.info(f"  → Progress: {i}/{len(candidates)} stocks analysed")

            except Exception as e:
                STATE["errors"].append({"time": str(datetime.now()), "msg": f"{sym}: {str(e)[:80]}"})

        # ─────────────────────────────────────────────────────
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
        "errors":        STATE["errors"][-5:],
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
    return jsonify({"ok": True, "pnl": pnl, "cash": port["cash"], "holdings": port["holdings"]})


@app.route("/api/paper/reset", methods=["POST"])
def paper_reset():
    """Reset paper portfolio to ₹10,000."""
    STATE["paper_portfolio"] = {
        "capital": 10000.0, "cash": 10000.0,
        "holdings": {}, "pnl_realised": 0.0,
    }
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


def trigger_scan():
    if not STATE["scanning"]:
        threading.Thread(target=run_full_scan, daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})


@app.route("/api/gap-alerts")
def gap_alerts():
    return jsonify(STATE["gap_alerts"])


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
    Derives simple F&O option trade ideas from the top equity signals.
    Uses the scored equity signals + ATR to suggest ATM/OTM strikes.
    For proper F&O execution use algotrade_pro_enhanced.py locally.
    """
    if not STATE["signals"]:
        return jsonify({"error": "No scan data yet — run a full scan first."})

    # Load NFO instrument list for lot sizes
    nfo_lots = {}
    try:
        nfo_path = os.path.join(os.path.dirname(__file__), "_nfo_instruments.json")
        if os.path.exists(nfo_path):
            with open(nfo_path) as f:
                nfo_data = json.load(f)
            for item in nfo_data:
                sym = item.get("tradingsymbol","").replace("-EQ","").split("-")[0]
                if item.get("instrument_type") == "FUT":
                    nfo_lots[sym] = item.get("lot_size", 1)
    except Exception:
        pass

    KNOWN_LOTS = {  # fallback lot sizes for major F&O stocks
        "NIFTY":256, "BANKNIFTY":30, "FINNIFTY":40,
        "RELIANCE":250, "TCS":150, "INFY":300, "HDFCBANK":550,
        "ICICIBANK":700, "SBIN":1500, "TATAMOTORS":1425,
        "WIPRO":1500, "BAJFINANCE":125, "LT":175,
    }

    results = []
    seen    = set()
    for sig in STATE["signals"]:
        sym   = sig["symbol"]
        score = sig.get("score", 0)
        if score < CFG["SCORE_WATCHLIST"]:
            break                           # sorted by score; rest are worse
        if sym in seen or len(results) >= 15:
            break
        seen.add(sym)

        ltp = sig.get("ltp", 0)
        atr = sig.get("atr", ltp * 0.02)   # fallback 2% ATR
        sl  = sig.get("sl",  ltp - 1.5 * atr)
        tp  = sig.get("tp",  ltp + 3.5 * atr)

        direction = "CALL" if sig.get("signal_class") in ("buy","strong-buy") else "PUT"

        # ATM strike: round to nearest 50 for most stocks, 100 for > ₹5000
        step = 100 if ltp > 5000 else 50 if ltp > 1000 else 10
        atm  = round(ltp / step) * step

        lot  = nfo_lots.get(sym) or KNOWN_LOTS.get(sym) or 500

        # Premium estimate: ~2×ATR for ATM option (very rough Black-Scholes proxy)
        premium_est = round(2.0 * atr, 1)
        sl_opt      = round(premium_est * 0.40, 1)   # SL at 40% of premium
        tp_opt      = round(premium_est * 2.20, 1)   # TP at 2.2× premium
        rr          = round(tp_opt / sl_opt, 1) if sl_opt else 0

        results.append({
            "symbol":        sym,
            "equity_score":  score,
            "equity_signal": sig.get("signal",""),
            "ltp":           ltp,
            "direction":     direction,
            "strike":        atm,
            "option_type":   direction,
            "premium_est":   premium_est,
            "lot_size":      lot,
            "sl_option":     sl_opt,
            "tp_option":     tp_opt,
            "rr":            rr,
            "equity_sl":     sl,
            "equity_tp":     tp,
            "smc_bos":       sig.get("smc_bos", False),
            "smc_ob":        sig.get("smc_ob"),
            "note":          (
                "⚠️ Use algotrade_pro_enhanced.py for real-time option chain data. "
                "This is an indicative idea based on equity analysis."
            ),
        })

    return jsonify({
        "count":   len(results),
        "signals": results,
        "regime":  STATE["regime"].get("label",""),
        "vix":     STATE["vix"],
    })


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
log.info(f"🚀 Institutional Trader Pro — Full NSE Edition")
log.info(f"   Mode: {'PAPER' if CFG['PAPER_TRADE'] else '⚠ LIVE'} | SMC Engine: ON | Fibonacci Engine: ON | Yahoo Finance: REMOVED")
probe_existing_token()
start_scheduler()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>INSTRADE NSE — Institutional Scanner</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400;500;700&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --void: #010408;
  --deep: #050c14;
  --panel: #070f1a;
  --surface: #0a1628;
  --border: #0e2040;
  --border2: #152d55;
  --cyan: #00e5ff;
  --cyan2: #00b8d4;
  --cyan-glow: rgba(0,229,255,0.15);
  --green: #00ff9d;
  --green2: #00cc7a;
  --green-glow: rgba(0,255,157,0.12);
  --amber: #ffaa00;
  --amber-glow: rgba(255,170,0,0.12);
  --red: #ff2d55;
  --red-glow: rgba(255,45,85,0.12);
  --purple: #b060ff;
  --steel: #4a7a9b;
  --muted: #2a4a6b;
  --dim: #1a3050;
  --text: #c8dff0;
  --text2: #7aaac8;
  --orb: 'Orbitron', monospace;
  --jet: 'JetBrains Mono', monospace;
  --raj: 'Rajdhani', sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--void);color:var(--text);font-family:var(--jet);font-size:11px;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;z-index:0;background-image:linear-gradient(rgba(0,229,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,0.025) 1px,transparent 1px);background-size:40px 40px;pointer-events:none}
body::after{content:'';position:fixed;inset:0;z-index:0;background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(0,100,180,0.08) 0%,transparent 70%);pointer-events:none}

/* SCAN LINE */
.scanline{position:fixed;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);box-shadow:0 0 20px var(--cyan);z-index:999;animation:scandown 6s linear infinite;opacity:0.4}
@keyframes scandown{0%{top:-2px}100%{top:100vh}}

/* TOPBAR */
#topbar{position:sticky;top:0;z-index:100;height:48px;display:flex;align-items:stretch;background:rgba(5,12,20,0.97);border-bottom:1px solid var(--border2);backdrop-filter:blur(12px)}
.logo-zone{display:flex;align-items:center;gap:.6rem;padding:0 1.4rem;border-right:1px solid var(--border2);min-width:180px}
.logo-hex{width:28px;height:28px;background:linear-gradient(135deg,var(--cyan),var(--purple));clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:900;color:#000}
.logo-text{font-family:var(--orb);font-size:13px;font-weight:700;letter-spacing:.15em;color:var(--cyan);text-shadow:0 0 20px var(--cyan)}
.logo-text span{color:var(--text2);font-weight:400}
.pills{display:flex;flex:1;overflow:hidden}
.pill{display:flex;align-items:center;gap:.4rem;padding:0 1rem;border-right:1px solid var(--border);font-size:9px;font-family:var(--raj);letter-spacing:.08em}
.pill-label{color:var(--muted);font-size:8px;font-weight:600}
.pill-val{font-size:12px;font-weight:700;font-family:var(--jet)}
.pill-dot{width:4px;height:4px;border-radius:50%;background:var(--cyan);box-shadow:0 0 6px var(--cyan);animation:pulse-dot 2s ease-in-out infinite}
@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.2}}
.bar-right{display:flex;align-items:center;gap:.8rem;padding:0 1rem;margin-left:auto}
.mode-badge{font-family:var(--orb);font-size:8px;font-weight:700;padding:.2rem .6rem;letter-spacing:.15em;border-radius:2px}
.mode-paper{background:rgba(255,170,0,.1);color:var(--amber);border:1px solid rgba(255,170,0,.3)}
.mode-live{background:rgba(0,255,157,.1);color:var(--green);border:1px solid rgba(0,255,157,.3)}
.scan-btn{font-family:var(--orb);font-size:9px;font-weight:700;letter-spacing:.12em;padding:.35rem 1rem;background:linear-gradient(135deg,var(--cyan),var(--purple));color:#000;border:none;border-radius:2px;cursor:pointer;transition:.2s;position:relative;overflow:hidden}
.scan-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--purple),var(--cyan));opacity:0;transition:.2s}
.scan-btn:hover::before{opacity:1}
.scan-btn:disabled{opacity:.4;cursor:not-allowed}
.scan-btn span{position:relative;z-index:1}
.scan-btn.running{animation:btn-scan 1s ease-in-out infinite alternate}
@keyframes btn-scan{0%{box-shadow:0 0 10px var(--cyan)}100%{box-shadow:0 0 30px var(--cyan),0 0 60px var(--cyan2)}}
#clock{font-family:var(--orb);font-size:11px;color:var(--steel);letter-spacing:.1em}

/* LAYOUT */
main{position:relative;z-index:1;padding:.8rem;display:grid;gap:.8rem;padding-bottom:30px}

/* PANEL */
.panel{background:var(--panel);border:1px solid var(--border);border-radius:3px;position:relative;overflow:hidden}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--border2),transparent)}
.panel-accent-cyan::after{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,var(--cyan),transparent)}
.panel-accent-green::after{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,var(--green),transparent)}
.panel-accent-amber::after{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,var(--amber),transparent)}
.panel-accent-purple::after{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,var(--purple),transparent)}
.panel-accent-red::after{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,var(--red),transparent)}

.ph{display:flex;align-items:center;justify-content:space-between;padding:.45rem .85rem;background:rgba(0,0,0,.3);border-bottom:1px solid var(--border)}
.ph-l{font-family:var(--raj);font-size:9px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--steel)}
.ph-r{font-size:9px;color:var(--muted);font-family:var(--jet)}

/* SCAN PROGRESS BAR */
#scan-progress-wrap{display:none;position:sticky;top:48px;z-index:98;background:rgba(5,12,20,.98);border-bottom:1px solid var(--border2);padding:.55rem 1.2rem}
.prog-inner{display:flex;align-items:center;gap:1rem}
.prog-label{font-family:var(--raj);font-size:10px;font-weight:700;color:var(--cyan);letter-spacing:.08em;min-width:280px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.prog-track{flex:1;height:4px;background:var(--border2);border-radius:2px;overflow:hidden}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--purple));border-radius:2px;transition:width .4s ease;width:0%}
.prog-pct{font-family:var(--orb);font-size:10px;color:var(--cyan);min-width:38px;text-align:right}
.prog-steps{display:flex;gap:.3rem;margin-top:.35rem}
.prog-step{height:2px;flex:1;background:var(--dim);border-radius:1px;transition:background .3s}
.prog-step.done{background:var(--cyan)}
.prog-step.active{background:var(--purple);animation:prog-pulse .8s ease-in-out infinite alternate}
@keyframes prog-pulse{0%{opacity:.5}100%{opacity:1}}

/* STAT CARDS */
.stat-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:.8rem}
.stat-body{padding:.9rem 1rem 1rem 1rem}
.stat-icon{font-size:16px;margin-bottom:.4rem;opacity:.6}
.stat-val{font-family:var(--orb);font-size:20px;font-weight:700;line-height:1;letter-spacing:-.02em}
.stat-label{font-size:8px;color:var(--muted);margin-top:.35rem;letter-spacing:.1em;font-family:var(--raj);text-transform:uppercase}
.stat-sub{font-size:9px;color:var(--steel);margin-top:.2rem}

/* MACRO ROW */
.macro-grid{display:grid;grid-template-columns:1.1fr 1.1fr 1.8fr;gap:.8rem}

/* REGIME */
.regime-body{padding:.8rem 1rem}
.regime-label{font-family:var(--orb);font-size:20px;font-weight:900;letter-spacing:.05em;margin-bottom:.5rem}
.regime-flag{font-size:9px;line-height:2;color:var(--muted);font-family:var(--raj);font-weight:500}
.regime-flag.ok{color:var(--green)}
.regime-flag.fail{color:var(--red)}
.regime-flag.warn{color:var(--amber)}

/* FLOW */
.flow-grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
.flow-row{display:flex;flex-direction:column;padding:.55rem .85rem;border-bottom:1px solid var(--border);border-right:1px solid var(--border)}
.flow-row:nth-child(even){border-right:none}
.flow-row:nth-last-child(-n+2){border-bottom:none}
.flow-key{font-size:8px;font-weight:600;letter-spacing:.1em;color:var(--muted);font-family:var(--raj);margin-bottom:.2rem}
.flow-val{font-size:15px;font-weight:700;font-family:var(--jet)}
.flow-sub{font-size:8px;color:var(--steel);margin-top:.1rem;font-family:var(--raj)}

/* ENGINE CHIPS */
.engine-chips{padding:.7rem .85rem .4rem;display:flex;flex-wrap:wrap;gap:.4rem}
.chip{font-family:var(--raj);font-size:9px;font-weight:700;padding:.25rem .6rem;border-radius:2px;letter-spacing:.1em}
.chip-cyan{background:rgba(0,229,255,.1);color:var(--cyan);border:1px solid rgba(0,229,255,.25)}
.chip-green{background:rgba(0,255,157,.1);color:var(--green);border:1px solid rgba(0,255,157,.25)}
.chip-purple{background:rgba(176,96,255,.1);color:var(--purple);border:1px solid rgba(176,96,255,.25)}
.chip-amber{background:rgba(255,170,0,.1);color:var(--amber);border:1px solid rgba(255,170,0,.25)}
.chip-red{background:rgba(255,45,85,.1);color:var(--red);border:1px solid rgba(255,45,85,.25)}
.score-legend{padding:.3rem .85rem .7rem;display:flex;gap:1.2rem}
.sl-item{font-size:9px;font-family:var(--raj);font-weight:600}

/* TABS */
.tabs{display:flex;border-bottom:1px solid var(--border);background:var(--panel);margin-bottom:.8rem}
.tab{padding:.55rem 1.2rem;font-family:var(--raj);font-size:10px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:.15s;position:relative}
.tab:hover{color:var(--text2)}
.tab.active{color:var(--cyan);border-bottom-color:var(--cyan)}
.tab.active::before{content:'';position:absolute;bottom:-1px;left:0;right:0;height:8px;background:linear-gradient(0deg,rgba(0,229,255,.05),transparent)}
.tab-badge{display:inline-block;background:var(--cyan);color:#000;font-size:7px;font-weight:900;padding:1px 4px;border-radius:2px;margin-left:.3rem;vertical-align:middle}

/* SIGNAL TABLE */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse}
th{font-family:var(--raj);font-size:8px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);padding:.4rem .7rem;border-bottom:1px solid var(--border2);background:rgba(0,0,0,.3);text-align:left;white-space:nowrap}
td{padding:.45rem .7rem;border-bottom:1px solid var(--border);font-size:10px;vertical-align:middle}
tr:last-child td{border-bottom:none}
tr{transition:background .1s}
tr:hover td{background:rgba(0,229,255,.03);cursor:pointer}
tr.selected td{background:rgba(0,229,255,.06)}
.sym-cell strong{font-family:var(--raj);font-size:12px;font-weight:700;color:var(--cyan);letter-spacing:.04em}

/* SCORE BAR */
.score-bar-wrap{display:flex;align-items:center;gap:.5rem}
.score-pips{display:flex;gap:1.5px}
.pip{width:5px;height:10px;border-radius:1px;background:var(--dim);transition:background .2s}
.pip.lit{background:var(--cyan);box-shadow:0 0 4px var(--cyan2)}
.pip.lit.hi{background:var(--green);box-shadow:0 0 4px var(--green2)}
.pip.lit.lo{background:var(--amber)}
.score-num{font-family:var(--orb);font-size:9px;color:var(--steel);min-width:28px}

/* SIGNAL LABELS */
.sig-label{font-family:var(--raj);font-size:9px;font-weight:700;padding:.18rem .45rem;border-radius:2px;letter-spacing:.08em;white-space:nowrap}
.sig-sb{background:rgba(0,255,157,.1);color:var(--green);border:1px solid rgba(0,255,157,.3)}
.sig-b{background:rgba(0,229,255,.1);color:var(--cyan);border:1px solid rgba(0,229,255,.3)}
.sig-w{background:rgba(255,170,0,.1);color:var(--amber);border:1px solid rgba(255,170,0,.3)}
.sig-av{background:rgba(74,122,155,.07);color:var(--muted);border:1px solid var(--border2)}
.act-buy{font-family:var(--raj);font-size:9px;font-weight:700;padding:.18rem .45rem;border-radius:2px;background:rgba(0,255,157,.12);color:var(--green);border:1px solid rgba(0,255,157,.3)}
.act-sell{font-family:var(--raj);font-size:9px;font-weight:700;padding:.18rem .45rem;border-radius:2px;background:rgba(255,45,85,.12);color:var(--red);border:1px solid rgba(255,45,85,.3)}
.act-hold{font-family:var(--raj);font-size:9px;font-weight:700;padding:.18rem .45rem;border-radius:2px;background:transparent;color:var(--muted);border:1px solid var(--border2)}

/* BREAKDOWN PANEL */
.bd-body{padding:.8rem;max-height:480px;overflow-y:auto}
.eng-block{margin-bottom:1rem}
.eng-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem}
.eng-name{font-family:var(--raj);font-size:9px;font-weight:700;letter-spacing:.15em;text-transform:uppercase}
.eng-score{font-family:var(--orb);font-size:10px;font-weight:700}
.eng-track{height:3px;background:var(--dim);border-radius:2px;margin-bottom:.5rem;overflow:hidden}
.eng-fill{height:100%;border-radius:2px;transition:width .5s ease}
.eng-flag{font-size:9px;line-height:1.9;font-family:var(--raj);font-weight:500;color:var(--muted)}
.eng-flag.ok{color:var(--green)}
.eng-flag.fail{color:var(--red)}
.eng-flag.warn{color:var(--amber)}

/* GAP ALERT */
.gap-sym{font-family:var(--raj);font-size:12px;font-weight:700;color:var(--amber)}
.gap-pct{font-family:var(--orb);font-size:13px;font-weight:700;color:var(--green)}
.gap-tag{font-family:var(--raj);font-size:9px;font-weight:700;padding:.15rem .4rem;border-radius:2px}
.gap-exp{background:rgba(255,45,85,.1);color:var(--red);border:1px solid rgba(255,45,85,.3)}
.gap-str{background:rgba(0,255,157,.1);color:var(--green);border:1px solid rgba(0,255,157,.3)}
.gap-mod{background:rgba(255,170,0,.1);color:var(--amber);border:1px solid rgba(255,170,0,.3)}

/* PAPER PORTFOLIO */
.port-header{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;padding:.7rem .85rem;border-bottom:1px solid var(--border)}
.port-metric{display:flex;flex-direction:column;gap:.2rem}
.port-metric-label{font-size:8px;color:var(--muted);font-family:var(--raj);font-weight:600;letter-spacing:.1em;text-transform:uppercase}
.port-metric-val{font-family:var(--orb);font-size:16px;font-weight:700}
.trade-form{display:flex;gap:.5rem;align-items:center;padding:.6rem .85rem;border-bottom:1px solid var(--border);background:rgba(0,0,0,.2);flex-wrap:wrap}
.trade-form input{background:var(--surface);border:1px solid var(--border2);color:var(--text);font-family:var(--jet);font-size:10px;padding:.35rem .6rem;border-radius:2px;width:100px}
.trade-form input:focus{outline:none;border-color:var(--cyan)}
.trade-form input::placeholder{color:var(--muted)}
.btn-buy{background:rgba(0,255,157,.15);color:var(--green);border:1px solid rgba(0,255,157,.4);font-family:var(--raj);font-size:9px;font-weight:700;padding:.35rem .8rem;border-radius:2px;cursor:pointer;letter-spacing:.1em}
.btn-buy:hover{background:rgba(0,255,157,.25)}
.btn-sell{background:rgba(255,45,85,.15);color:var(--red);border:1px solid rgba(255,45,85,.4);font-family:var(--raj);font-size:9px;font-weight:700;padding:.35rem .8rem;border-radius:2px;cursor:pointer;letter-spacing:.1em}
.btn-sell:hover{background:rgba(255,45,85,.25)}
.btn-reset{background:rgba(74,122,155,.1);color:var(--steel);border:1px solid var(--border2);font-family:var(--raj);font-size:9px;font-weight:700;padding:.35rem .8rem;border-radius:2px;cursor:pointer;letter-spacing:.1em}
.port-msg{font-size:9px;font-family:var(--raj);padding:.1rem .4rem;border-radius:2px}

/* FNO */
.fno-card{background:var(--surface);border:1px solid var(--border2);border-radius:3px;padding:.7rem .9rem;margin:.5rem;display:inline-block;min-width:200px;vertical-align:top}
.fno-sym{font-family:var(--raj);font-size:14px;font-weight:700;color:var(--cyan)}
.fno-tag{font-family:var(--raj);font-size:9px;font-weight:700;padding:.15rem .45rem;border-radius:2px;margin-left:.4rem}
.fno-call{background:rgba(0,255,157,.1);color:var(--green);border:1px solid rgba(0,255,157,.3)}
.fno-put{background:rgba(255,45,85,.1);color:var(--red);border:1px solid rgba(255,45,85,.3)}
.fno-row{display:flex;justify-content:space-between;font-size:9px;padding:.2rem 0;border-bottom:1px solid var(--border)}
.fno-row:last-child{border-bottom:none}
.fno-key{color:var(--muted);font-family:var(--raj);font-weight:600}
.fno-val{font-family:var(--jet);color:var(--text)}

/* KITE LOG */
.kl-row{display:flex;gap:.8rem;padding:.35rem .85rem;border-bottom:1px solid var(--border);font-size:9px}
.kl-row:last-child{border-bottom:none}
.kl-time{color:var(--muted);min-width:55px}
.kl-method{color:var(--cyan);font-weight:700;min-width:180px}
.kl-detail{color:var(--steel)}

/* TWO COL */
.two-col{display:grid;grid-template-columns:1.4fr 1fr;gap:.8rem}

/* TICKER */
#ticker{position:fixed;bottom:0;left:0;right:0;height:26px;background:rgba(5,12,20,.97);border-top:1px solid var(--border2);display:flex;align-items:center;overflow:hidden;z-index:100}
.tk-track{display:flex;gap:2.5rem;animation:ticker-scroll 60s linear infinite;white-space:nowrap;padding-left:100%}
@keyframes ticker-scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.tk-sym{font-family:var(--raj);font-size:10px;font-weight:700;color:var(--steel);margin-right:.25rem}
.tk-val{font-size:10px;font-family:var(--jet)}
.tk-sep{color:var(--border2)}

/* TOAST */
#toast{position:fixed;top:52px;right:1rem;z-index:300;background:var(--cyan);color:#000;font-family:var(--orb);font-size:9px;font-weight:700;letter-spacing:.15em;padding:.4rem .9rem;border-radius:2px;display:none;box-shadow:0 4px 20px rgba(0,229,255,.4)}

/* UTILS */
.g{color:var(--green)}.r{color:var(--red)}.c{color:var(--cyan)}.a{color:var(--amber)}.d{color:var(--muted)}
.no-data{padding:2rem;text-align:center;color:var(--muted);font-family:var(--raj);font-size:10px;letter-spacing:.1em}
.pulse{animation:pulse-anim 1s ease-in-out infinite}
@keyframes pulse-anim{0%,100%{opacity:1}50%{opacity:.3}}
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:var(--void)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
</style>
</head>
<body>
<div class="scanline"></div>

<!-- TOPBAR -->
<div id="topbar">
  <div class="logo-zone">
    <div class="logo-hex">IT</div>
    <div>
      <div class="logo-text">INSTRADE <span>| NSE</span></div>
    </div>
  </div>
  <div class="pills">
    <div class="pill"><span class="pill-label">REGIME</span>&nbsp;<span class="pill-val" id="p-regime">—</span></div>
    <div class="pill"><div class="pill-dot"></div><span class="pill-label">VIX</span>&nbsp;<span class="pill-val" id="p-vix">—</span></div>
    <div class="pill"><span class="pill-label">FII</span>&nbsp;<span class="pill-val" id="p-fii">—</span></div>
    <div class="pill"><span class="pill-label">PCR</span>&nbsp;<span class="pill-val" id="p-pcr">—</span></div>
    <div class="pill"><span class="pill-label">SCANNED</span>&nbsp;<span class="pill-val c" id="p-uni">0</span></div>
    <div class="pill"><span class="pill-label">GAP-UPS</span>&nbsp;<span class="pill-val a" id="p-gaps">0</span></div>
    <div class="pill"><span class="pill-label">PORTFOLIO</span>&nbsp;<span class="pill-val" id="p-port">₹10,000</span></div>
    <div class="pill"><span class="pill-label">P&amp;L</span>&nbsp;<span class="pill-val" id="p-pnl">+₹0</span></div>
  </div>
  <div class="bar-right">
    <span id="clock">00:00:00</span>
    <span id="mode-badge" class="mode-badge mode-paper">PAPER</span>
    <a id="auth-btn" href="/kite-login" style="display:none;padding:.3rem .8rem;background:rgba(255,45,85,.15);border:1px solid rgba(255,45,85,.4);border-radius:4px;color:#ff2d55;font-family:var(--raj);font-size:9px;font-weight:700;letter-spacing:.1em;text-decoration:none">⚠ LOGIN</a>
    <span id="auth-ok" style="display:none;padding:.3rem .8rem;background:rgba(0,255,157,.08);border:1px solid rgba(0,255,157,.25);border-radius:4px;color:var(--green);font-family:var(--raj);font-size:9px;font-weight:700;letter-spacing:.06em">● ZERODHA</span>
    <button class="scan-btn" id="scan-btn" onclick="triggerScan()"><span>▶ FULL SCAN</span></button>
  </div>
</div>

<!-- TOKEN STALE BANNER -->
<div id="token-banner" style="display:none;position:sticky;top:48px;z-index:99;background:rgba(255,170,0,.12);border-bottom:1px solid rgba(255,170,0,.35);padding:.45rem 1.2rem;font-family:var(--raj);font-size:10px;font-weight:600;color:var(--amber);letter-spacing:.04em;text-align:center">
  ⚠ &nbsp;Kite token may be stale (&gt;8h old). Scan results could fail.&nbsp;
  <a href="/kite-login" style="color:var(--amber);text-decoration:underline">Re-login now →</a>
</div>

<!-- SCAN PROGRESS BAR -->
<div id="scan-progress-wrap">
  <div class="prog-inner">
    <span class="prog-label pulse" id="prog-label">Initialising scan...</span>
    <div class="prog-track"><div class="prog-fill" id="prog-fill"></div></div>
    <span class="prog-pct" id="prog-pct">0%</span>
  </div>
  <div class="prog-steps" id="prog-steps">
    <div class="prog-step" id="ps-1"></div>
    <div class="prog-step" id="ps-2"></div>
    <div class="prog-step" id="ps-3"></div>
    <div class="prog-step" id="ps-4"></div>
    <div class="prog-step" id="ps-5"></div>
    <div class="prog-step" id="ps-6"></div>
    <div class="prog-step" id="ps-7"></div>
  </div>
</div>

<main>

<!-- STAT CARDS — 6 wide -->
<div class="stat-grid">
  <div class="panel panel-accent-cyan">
    <div class="ph"><span class="ph-l">NSE Universe</span></div>
    <div class="stat-body">
      <div class="stat-val c" id="s-uni">0</div>
      <div class="stat-label">Stocks scanned</div>
    </div>
  </div>
  <div class="panel panel-accent-amber">
    <div class="ph"><span class="ph-l">Gap-Up Alerts</span></div>
    <div class="stat-body">
      <div class="stat-val a" id="s-gaps">0</div>
      <div class="stat-label">≥2% gap stocks</div>
    </div>
  </div>
  <div class="panel panel-accent-green">
    <div class="ph"><span class="ph-l">Signals Found</span></div>
    <div class="stat-body">
      <div class="stat-val g" id="s-sigs">0</div>
      <div class="stat-label">Total scored</div>
    </div>
  </div>
  <div class="panel panel-accent-purple">
    <div class="ph"><span class="ph-l">Open Positions</span></div>
    <div class="stat-body">
      <div class="stat-val" style="color:var(--purple)" id="s-pos">0</div>
      <div class="stat-label">of 6 max slots</div>
    </div>
  </div>
  <div class="panel panel-accent-green">
    <div class="ph"><span class="ph-l">Session P&amp;L</span></div>
    <div class="stat-body">
      <div class="stat-val g" id="s-pnl">+₹0</div>
      <div class="stat-label" id="s-wr">Win 0% · 0 trades</div>
    </div>
  </div>
  <div class="panel panel-accent-cyan">
    <div class="ph"><span class="ph-l">Portfolio Value</span></div>
    <div class="stat-body">
      <div class="stat-val c" id="s-port-val">₹10,000</div>
      <div class="stat-label" id="s-port-pct">+0.00% from ₹10k</div>
    </div>
  </div>
</div>

<!-- MACRO ROW -->
<div class="macro-grid">
  <!-- REGIME -->
  <div class="panel panel-accent-cyan">
    <div class="ph"><span class="ph-l">Market Regime</span><span class="ph-r" id="r-score">—</span></div>
    <div class="regime-body">
      <div class="regime-label" id="r-label">—</div>
      <div id="r-flags"></div>
    </div>
  </div>

  <!-- FLOW — now 2×2 grid -->
  <div class="panel panel-accent-amber">
    <div class="ph"><span class="ph-l">Institutional Flow</span></div>
    <div class="flow-grid">
      <div class="flow-row">
        <span class="flow-key">FII NET</span>
        <span class="flow-val" id="f-fii">—</span>
        <span class="flow-sub" id="f-fii-dir">—</span>
      </div>
      <div class="flow-row">
        <span class="flow-key">DII NET</span>
        <span class="flow-val" id="f-dii">—</span>
        <span class="flow-sub" id="f-dii-dir">—</span>
      </div>
      <div class="flow-row">
        <span class="flow-key">NIFTY PCR</span>
        <span class="flow-val" id="f-pcr">—</span>
        <span class="flow-sub" id="f-pcr-dir">—</span>
      </div>
      <div class="flow-row">
        <span class="flow-key">INDIA VIX</span>
        <span class="flow-val" id="f-vix">—</span>
        <span class="flow-sub" id="f-vix-dir">—</span>
      </div>
    </div>
  </div>

  <!-- SCORING ENGINE -->
  <div class="panel panel-accent-purple">
    <div class="ph"><span class="ph-l">Scoring Engine — Max 30 Points</span></div>
    <div class="engine-chips">
      <span class="chip chip-cyan">TECHNICAL ×5</span>
      <span class="chip chip-green">BREAKOUT ×5</span>
      <span class="chip chip-purple">FUNDAMENTAL ×5</span>
      <span class="chip chip-amber">INSTITUTIONAL ×5</span>
      <span class="chip chip-red">REGIME ×2</span>
      <span class="chip" style="background:rgba(0,229,255,.07);color:#00ffcc;border:1px solid rgba(0,255,204,.25)">SMC ×5</span>
      <span class="chip" style="background:rgba(255,170,0,.07);color:#ffd700;border:1px solid rgba(255,215,0,.3)">FIBONACCI ×3</span>
    </div>
    <div class="score-legend">
      <span class="sl-item g">●&nbsp;20+ STRONG BUY</span>
      <span class="sl-item c">●&nbsp;13+ BUY</span>
      <span class="sl-item a">●&nbsp;9+ WATCHLIST</span>
      <span class="sl-item d">●&nbsp;&lt;9 AVOID</span>
    </div>
  </div>
</div>

<!-- TABS -->
<div>
<div class="tabs">
  <div class="tab active" onclick="showTab('signals')">SIGNALS</div>
  <div class="tab" onclick="showTab('paper')">📊 PAPER TRADE</div>
  <div class="tab" onclick="showTab('fno')">F&amp;O IDEAS</div>
  <div class="tab" onclick="showTab('gaps')">GAP ALERTS</div>
  <div class="tab" onclick="showTab('positions')">POSITIONS</div>
  <div class="tab" onclick="showTab('trades')">TRADE LOG</div>
  <div class="tab" onclick="showTab('kite')">KITE API LOG</div>
</div>

<!-- SIGNALS TAB -->
<div id="tab-signals">
<div class="two-col">
  <div class="panel">
    <div class="ph">
      <span class="ph-l">Signal Scanner</span>
      <span class="ph-r">
        <span style="color:var(--green);margin-right:.6rem" id="sig-sb-ct"></span>
        <span id="sig-ct">0 scored</span>
      </span>
    </div>
    <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>SYMBOL</th><th>LTP</th><th>GAP%</th>
        <th style="color:var(--cyan)">TECH</th>
        <th style="color:var(--green)">BRK</th>
        <th style="color:var(--purple)">FUND</th>
        <th style="color:var(--amber)">INST</th>
        <th style="color:#00ffcc">SMC</th>
        <th style="color:#ffd700">FIB</th>
        <th>SCORE/30</th><th>SIGNAL</th><th>SL / TP</th><th>ACTION</th>
      </tr></thead>
      <tbody id="sig-body"><tr><td colspan="13" class="no-data">▶ Press FULL SCAN to load NSE signals</td></tr></tbody>
    </table>
    </div>
  </div>
  <div class="panel">
    <div class="ph"><span class="ph-l">Analysis Breakdown</span><span class="ph-r c" id="bd-sym">select a row</span></div>
    <div class="bd-body" id="bd-body"><div class="no-data">Click any stock row to see engine breakdown</div></div>
  </div>
</div>
</div>

<!-- PAPER TRADE TAB -->
<div id="tab-paper" style="display:none">
<div class="panel panel-accent-green">
  <div class="ph">
    <span class="ph-l">Paper Portfolio — ₹10,000 Starting Capital</span>
    <span class="ph-r" id="paper-last-update">—</span>
  </div>
  <!-- Portfolio metrics -->
  <div class="port-header">
    <div class="port-metric">
      <span class="port-metric-label">Total Value</span>
      <span class="port-metric-val c" id="pp-value">₹10,000</span>
    </div>
    <div class="port-metric">
      <span class="port-metric-label">Available Cash</span>
      <span class="port-metric-val" id="pp-cash">₹10,000</span>
    </div>
    <div class="port-metric">
      <span class="port-metric-label">Unrealised P&amp;L</span>
      <span class="port-metric-val" id="pp-unrl">₹0</span>
    </div>
    <div class="port-metric">
      <span class="port-metric-label">Realised P&amp;L</span>
      <span class="port-metric-val" id="pp-real">₹0</span>
    </div>
  </div>
  <!-- Trade entry form -->
  <div class="trade-form">
    <input type="text" id="pf-sym" placeholder="SYMBOL e.g. RELIANCE" style="width:150px">
    <input type="number" id="pf-qty" placeholder="QTY" min="1">
    <input type="number" id="pf-price" placeholder="PRICE ₹" step="0.05">
    <button class="btn-buy" onclick="paperBuy()">▲ BUY</button>
    <button class="btn-sell" onclick="paperSell()">▼ SELL</button>
    <button class="btn-reset" onclick="paperReset()">↺ RESET ₹10K</button>
    <span id="pf-msg" class="port-msg"></span>
  </div>
  <!-- Holdings table -->
  <div class="tbl-wrap">
  <table>
    <thead><tr>
      <th>SYMBOL</th><th>QTY</th><th>AVG PRICE</th><th>INVESTED</th>
      <th>CURR LTP</th><th>CURR VALUE</th><th>UNREALISED P&amp;L</th><th>RETURN %</th><th>ACTION</th>
    </tr></thead>
    <tbody id="paper-holdings"><tr><td colspan="9" class="no-data">No holdings — buy stocks from the signals tab</td></tr></tbody>
  </table>
  </div>
</div>
</div>

<!-- F&O TAB — card-based layout -->
<div id="tab-fno" style="display:none">
<div class="panel">
  <div class="ph">
    <span class="ph-l">F&amp;O Option Ideas <span style="font-size:8px;color:var(--muted)">derived from equity signals</span></span>
    <span class="ph-r" id="fno-ct">—</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;padding:.7rem">
    <div style="grid-column:1/-1;background:rgba(255,170,0,.06);border:1px solid rgba(255,170,0,.2);border-radius:3px;padding:.5rem .8rem;font-size:9px;color:var(--amber);font-family:var(--raj)">
      ⚠️ Premium estimates are ATR-based approximations (2×ATR for ATM). Run a scan first for signal data. Always verify strikes on Zerodha option chain before placing any order.
    </div>
    <div id="fno-cards" style="grid-column:1/-1;display:flex;flex-wrap:wrap;gap:.6rem">
      <div class="no-data" style="width:100%">Run a scan to see F&O ideas</div>
    </div>
  </div>
</div>
</div>

<!-- GAP ALERTS TAB -->
<div id="tab-gaps" style="display:none">
<div class="panel">
  <div class="ph"><span class="ph-l">Gap-Up Alerts — Full NSE Scan</span><span class="ph-r" id="gap-ct">—</span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>GAP %</th><th>LTP</th><th>VOLUME</th><th>CLASSIFICATION</th></tr></thead>
    <tbody id="gap-body"><tr><td colspan="5" class="no-data">—</td></tr></tbody>
  </table>
</div>
</div>

<!-- POSITIONS TAB -->
<div id="tab-positions" style="display:none">
<div class="panel">
  <div class="ph"><span class="ph-l">Open Positions</span><span class="ph-r" id="pos-pnl-hdr"></span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>ENTRY</th><th>CURR SL</th><th>TARGET</th><th>QTY</th><th>SCORE</th><th>UNRL P&amp;L</th><th>SL TYPE</th></tr></thead>
    <tbody id="pos-body"><tr><td colspan="8" class="no-data">No open positions</td></tr></tbody>
  </table>
</div>
</div>

<!-- TRADES TAB -->
<div id="tab-trades" style="display:none">
<div class="panel">
  <div class="ph"><span class="ph-l">Trade Log</span><span class="ph-r" id="trade-summary"></span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>ENTRY</th><th>EXIT</th><th>QTY</th><th>P&amp;L</th><th>REASON</th><th>TIME</th></tr></thead>
    <tbody id="trade-body"><tr><td colspan="7" class="no-data">No completed trades</td></tr></tbody>
  </table>
</div>
</div>

<!-- KITE LOG TAB -->
<div id="tab-kite" style="display:none">
<div class="panel">
  <div class="ph"><span class="ph-l">Kite API Call Log</span><span class="ph-r">live · last 20 calls</span></div>
  <div id="kite-log"><div class="no-data">No API calls logged yet</div></div>
</div>
</div>

</div><!-- end tabs wrapper -->
</main>

<div id="ticker"><div class="tk-track" id="tk-track">—</div></div>
<div id="toast">SCAN INITIATED</div>

<script>
let allSigs = []; let selSym = null; let progressPoll = null;
const TABS = ['signals','paper','fno','gaps','positions','trades','kite'];

function showTab(t) {
  TABS.forEach(x => {
    const el = document.getElementById('tab-'+x);
    if(el) el.style.display = x===t?'':'none';
  });
  document.querySelectorAll('.tab').forEach((el,i) => {
    el.classList.toggle('active', TABS[i]===t);
  });
  if(t==='fno') loadFno();
  if(t==='paper') loadPaperState();
}

/* ── PROGRESS BAR ────────────────── */
function startProgressPoll() {
  const wrap = document.getElementById('scan-progress-wrap');
  wrap.style.display = 'block';
  progressPoll = setInterval(async () => {
    try {
      const d = await (await fetch('/api/progress')).json();
      const p = d.scan_progress || {};
      document.getElementById('prog-label').textContent = p.step_label || 'Scanning...';
      document.getElementById('prog-fill').style.width = (p.pct||0)+'%';
      document.getElementById('prog-pct').textContent = (p.pct||0)+'%';
      // colour step indicators
      for(let s=1;s<=7;s++){
        const el = document.getElementById('ps-'+s);
        if(!el) continue;
        if(s < p.step) el.className='prog-step done';
        else if(s === p.step) el.className='prog-step active';
        else el.className='prog-step';
      }
      if(!d.scanning) {
        clearInterval(progressPoll); progressPoll = null;
        setTimeout(() => { wrap.style.display='none'; }, 3000);
      }
    } catch(e) {}
  }, 800);
}

/* ── F&O CARDS ───────────────────── */
async function loadFno() {
  const container = document.getElementById('fno-cards');
  try {
    const r = await (await fetch('/api/fno-signals')).json();
    if(r.error){ container.innerHTML=`<div class="no-data">${r.error}</div>`; return; }
    document.getElementById('fno-ct').textContent = (r.count||0)+' ideas · VIX '+(r.vix||0).toFixed(1);
    const sigs = r.signals||[];
    if(!sigs.length){ container.innerHTML='<div class="no-data">No signals above threshold</div>'; return; }
    container.innerHTML = sigs.map(s => {
      const isBull = s.direction==='CALL';
      const tagCls = isBull?'fno-call':'fno-put';
      const rrColor = s.rr>=2?'var(--green)':s.rr>=1.5?'var(--amber)':'var(--red)';
      return `<div class="fno-card">
        <div style="margin-bottom:.5rem">
          <span class="fno-sym">${s.symbol}</span>
          <span class="fno-tag ${tagCls}">${s.direction}</span>
          <span style="font-family:var(--raj);font-size:8px;color:var(--muted);margin-left:.4rem">${s.equity_signal.split(' ')[0]} ${s.equity_score}/30</span>
        </div>
        <div class="fno-row"><span class="fno-key">LTP</span><span class="fno-val">₹${s.ltp.toLocaleString('en-IN')}</span></div>
        <div class="fno-row"><span class="fno-key">STRIKE</span><span class="fno-val c">₹${s.strike}</span></div>
        <div class="fno-row"><span class="fno-key">PREM ~</span><span class="fno-val a">₹${s.premium_est}</span></div>
        <div class="fno-row"><span class="fno-key">LOT SIZE</span><span class="fno-val">${s.lot_size}</span></div>
        <div class="fno-row"><span class="fno-key">OPT SL</span><span class="fno-val r">₹${s.sl_option}</span></div>
        <div class="fno-row"><span class="fno-key">OPT TP</span><span class="fno-val g">₹${s.tp_option}</span></div>
        <div class="fno-row"><span class="fno-key">R:R</span><span class="fno-val" style="color:${rrColor};font-weight:700">${s.rr}x</span></div>
        <div class="fno-row"><span class="fno-key">EQ SL→TP</span><span class="fno-val" style="font-size:8px"><span class="r">₹${s.equity_sl}</span> → <span class="g">₹${s.equity_tp}</span></span></div>
        ${s.smc_bos?'<div style="font-size:8px;color:#00ffcc;margin-top:.3rem">✅ SMC Break of Structure</div>':''}
      </div>`;
    }).join('');
  } catch(e) { container.innerHTML='<div class="no-data">Error: '+e.message+'</div>'; }
}

/* ── PAPER TRADE ─────────────────── */
async function loadPaperState() {
  try {
    const d = await (await fetch('/api/paper/state')).json();
    document.getElementById('pp-value').textContent = '₹'+d.total_value.toLocaleString('en-IN');
    document.getElementById('pp-value').className = 'port-metric-val '+(d.total_pnl>=0?'g':'r');
    document.getElementById('pp-cash').textContent = '₹'+d.cash.toLocaleString('en-IN');
    const ul = document.getElementById('pp-unrl');
    ul.textContent = (d.pnl_unrealised>=0?'+':'')+'₹'+Math.abs(d.pnl_unrealised).toLocaleString('en-IN');
    ul.className = 'port-metric-val '+(d.pnl_unrealised>=0?'g':'r');
    const rl = document.getElementById('pp-real');
    rl.textContent = (d.pnl_realised>=0?'+':'')+'₹'+Math.abs(d.pnl_realised).toLocaleString('en-IN');
    rl.className = 'port-metric-val '+(d.pnl_realised>=0?'g':'r');
    // topbar portfolio
    document.getElementById('p-port').textContent = '₹'+d.total_value.toLocaleString('en-IN');
    document.getElementById('p-port').className = 'pill-val '+(d.total_pnl>=0?'g':'r');
    // stat card
    document.getElementById('s-port-val').textContent = '₹'+d.total_value.toLocaleString('en-IN');
    document.getElementById('s-port-val').className = 'stat-val '+(d.total_pnl>=0?'g':'r');
    document.getElementById('s-port-pct').textContent = (d.total_pnl_pct>=0?'+':'')+d.total_pnl_pct.toFixed(2)+'% from ₹10k';
    document.getElementById('paper-last-update').textContent = 'refreshed '+new Date().toLocaleTimeString('en-IN',{hour12:false});
    // holdings table
    const tb = document.getElementById('paper-holdings');
    const entries = Object.entries(d.holdings||{});
    if(!entries.length){
      tb.innerHTML='<tr><td colspan="9" class="no-data">No holdings — enter a symbol and qty to buy</td></tr>'; return;
    }
    tb.innerHTML = entries.map(([sym,h])=>{
      const val   = h.ltp * h.qty;
      const pnlCls = h.unrealised>=0?'g':'r';
      return `<tr>
        <td><strong style="font-family:var(--raj);color:var(--cyan);font-size:12px">${sym}</strong></td>
        <td style="font-family:var(--jet)">${h.qty}</td>
        <td style="font-family:var(--jet)">₹${h.avg_price.toLocaleString('en-IN')}</td>
        <td style="font-family:var(--jet)">₹${h.invested.toLocaleString('en-IN')}</td>
        <td style="font-family:var(--jet)">${h.ltp?'₹'+h.ltp.toLocaleString('en-IN'):'—'}</td>
        <td style="font-family:var(--jet)">${h.ltp?'₹'+val.toLocaleString('en-IN'):'—'}</td>
        <td class="${pnlCls}" style="font-family:var(--jet);font-weight:700">${h.unrealised>=0?'+':''}₹${Math.abs(h.unrealised).toLocaleString('en-IN')}</td>
        <td class="${pnlCls}" style="font-family:var(--jet)">${h.pct>=0?'+':''}${h.pct.toFixed(2)}%</td>
        <td>
          <button class="btn-sell" style="font-size:8px;padding:.2rem .5rem" onclick="quickSell('${sym}',${h.qty},${h.ltp||h.avg_price})">SELL ALL</button>
        </td>
      </tr>`;
    }).join('');
  } catch(e) {}
}

async function paperBuy() {
  const sym = document.getElementById('pf-sym').value.trim().toUpperCase();
  const qty = parseInt(document.getElementById('pf-qty').value)||0;
  const price = parseFloat(document.getElementById('pf-price').value)||0;
  if(!sym||!qty||!price){ showPfMsg('Fill all fields','r'); return; }
  try {
    const r = await (await fetch('/api/paper/buy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym,qty,price})})).json();
    if(r.ok){ showPfMsg(`✅ Bought ${qty} ${sym} @ ₹${price}`,'g'); loadPaperState(); }
    else showPfMsg('❌ '+r.error,'r');
  } catch(e){ showPfMsg('Error','r'); }
}

async function paperSell() {
  const sym = document.getElementById('pf-sym').value.trim().toUpperCase();
  const qty = parseInt(document.getElementById('pf-qty').value)||0;
  const price = parseFloat(document.getElementById('pf-price').value)||0;
  if(!sym||!qty||!price){ showPfMsg('Fill all fields','r'); return; }
  try {
    const r = await (await fetch('/api/paper/sell',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym,qty,price})})).json();
    if(r.ok){ showPfMsg(`✅ Sold ${qty} ${sym} @ ₹${price} | PnL ₹${r.pnl}`,'g'); loadPaperState(); }
    else showPfMsg('❌ '+r.error,'r');
  } catch(e){ showPfMsg('Error','r'); }
}

async function quickSell(sym, qty, price) {
  document.getElementById('pf-sym').value = sym;
  document.getElementById('pf-qty').value = qty;
  document.getElementById('pf-price').value = price;
  await paperSell();
}

async function paperReset() {
  if(!confirm('Reset paper portfolio to ₹10,000?')) return;
  await fetch('/api/paper/reset',{method:'POST'});
  showPfMsg('Portfolio reset to ₹10,000','a');
  loadPaperState();
}

function showPfMsg(msg, cls) {
  const el = document.getElementById('pf-msg');
  el.textContent = msg;
  el.className = 'port-msg '+(cls==='g'?'g':cls==='r'?'r':'a');
  setTimeout(()=>{ el.textContent=''; }, 4000);
}

/* ── FETCH + RENDER ──────────────── */
async function fetchState() {
  try { const d = await (await fetch('/api/state')).json(); render(d); } catch(e) {}
}

function fmt(n)   { return (n>=0?'+':'')+'₹'+Math.abs(n||0).toLocaleString('en-IN'); }
function fmtCr(n) { return (n>=0?'+':'')+'₹'+Math.abs((n||0)/1e7).toFixed(1)+'Cr'; }

function render(d) {
  allSigs = d.signals || [];
  const s = d.stats;

  // mode badge
  const mb = document.getElementById('mode-badge');
  mb.textContent = d.paper_trade ? 'PAPER' : 'LIVE';
  mb.className   = 'mode-badge '+(d.paper_trade?'mode-paper':'mode-live');

  // pills
  const reg = d.regime || {};
  const rl  = reg.label || '—';
  const pe = document.getElementById('p-regime');
  pe.textContent = rl; pe.className = 'pill-val '+(rl.includes('BULL')?'g':'r');
  const vix = d.vix || 15;
  const pv = document.getElementById('p-vix');
  pv.textContent = vix.toFixed(1); pv.className = 'pill-val '+(vix<15?'g':vix<20?'a':'r');
  const fn = d.fii_dii?.fii_net || 0;
  const pf = document.getElementById('p-fii');
  pf.textContent = fmtCr(fn); pf.className = 'pill-val '+(fn>=0?'g':'r');
  document.getElementById('p-pcr').textContent = d.fii_dii?.pcr || '—';
  document.getElementById('p-uni').textContent  = s.universe_size.toLocaleString();
  document.getElementById('p-gaps').textContent = s.gap_alerts;
  const pnl = s.total_pnl;
  const pp  = document.getElementById('p-pnl');
  pp.textContent = fmt(pnl); pp.className = 'pill-val '+(pnl>=0?'g':'r');

  // stat cards
  document.getElementById('s-uni').textContent  = s.universe_size.toLocaleString();
  document.getElementById('s-gaps').textContent = s.gap_alerts;
  document.getElementById('s-sigs').textContent = s.signals_count;
  document.getElementById('s-pos').textContent  = s.open_positions;
  const spe = document.getElementById('s-pnl');
  spe.textContent = fmt(pnl); spe.className = 'stat-val '+(pnl>=0?'g':'r');
  document.getElementById('s-wr').textContent = `Win ${s.win_rate}% · ${s.total_trades} trades`;

  // regime
  document.getElementById('r-label').textContent  = rl;
  document.getElementById('r-label').className    = 'regime-label '+(rl.includes('BULL')?'g':'r');
  document.getElementById('r-score').textContent  = `${reg.score||0}/2`;
  document.getElementById('r-flags').innerHTML    = (reg.flags||[]).map(f=>{
    const cls = f.startsWith('✅')?'ok':f.startsWith('❌')?'fail':'warn';
    return `<div class="regime-flag ${cls}">${f}</div>`;
  }).join('');

  // institutional flow — now with direction labels
  const dn = d.fii_dii?.dii_net || 0;
  const pcr = d.fii_dii?.pcr;
  document.getElementById('f-fii').textContent     = fn!==0?fmtCr(fn):'—';
  document.getElementById('f-fii').className       = 'flow-val '+(fn>=0?'g':'r');
  document.getElementById('f-fii-dir').textContent = fn>0?'NET BUYING':fn<0?'NET SELLING':'NEUTRAL';
  document.getElementById('f-dii').textContent     = dn!==0?fmtCr(dn):'—';
  document.getElementById('f-dii').className       = 'flow-val '+(dn>=0?'g':'r');
  document.getElementById('f-dii-dir').textContent = dn>0?'NET BUYING':dn<0?'NET SELLING':'NEUTRAL';
  document.getElementById('f-pcr').textContent     = pcr?pcr:'—';
  document.getElementById('f-pcr').className       = 'flow-val '+(pcr>=0.85?'g':pcr>=0.7?'a':'r');
  document.getElementById('f-pcr-dir').textContent = pcr?((pcr>=0.85?'BULLISH OI':'BEARISH OI')):'—';
  document.getElementById('f-vix').textContent     = vix?vix.toFixed(1):'—';
  document.getElementById('f-vix').className       = 'flow-val '+(vix<15?'g':vix<20?'a':'r');
  document.getElementById('f-vix-dir').textContent = vix<15?'VERY CALM':vix<20?'CALM':vix<25?'ELEVATED':'HIGH FEAR';

  // tab badge for signals
  const sbCount = (d.signals||[]).filter(x=>x.signal_class==='strong-buy').length;
  const sigTabBadge = sbCount>0?`<span class="tab-badge">${sbCount}</span>`:'';

  renderSignals(d.signals);
  renderGaps(d.gap_alerts);
  renderPositions(d.positions, d.signals);
  renderTrades(d.trade_log);
  renderKiteLog(d.kite_calls);
  renderTicker(d.signals, d.gap_alerts);

  // auth badge
  const auth = d.auth || {};
  const authOk     = document.getElementById('auth-ok');
  const authBtn    = document.getElementById('auth-btn');
  const tokenBanner= document.getElementById('token-banner');
  if(auth.status==='ok'){
    authOk.style.display='inline-block'; authOk.title=`${auth.user_name} (${auth.user_id})`;
    authBtn.style.display='none'; tokenBanner.style.display='none';
  } else if(auth.status==='stale'){
    authOk.style.display='inline-block'; authOk.title=`${auth.user_name} — token may be stale`;
    authOk.style.color='var(--amber)'; authBtn.style.display='none'; tokenBanner.style.display='block';
  } else {
    authOk.style.display='none'; authBtn.style.display='inline-block'; tokenBanner.style.display='none';
  }

  if(selSym){ const found=d.signals.find(x=>x.symbol===selSym); if(found) showBreakdown(found); }

  // refresh paper state if on paper tab
  if(document.getElementById('tab-paper').style.display!=='none') loadPaperState();
}

function renderSignals(sigs) {
  const sbCount = (sigs||[]).filter(x=>x.signal_class==='strong-buy').length;
  const buyCount = (sigs||[]).filter(x=>x.signal_class==='buy').length;
  document.getElementById('sig-ct').textContent = (sigs||[]).length+' scored';
  document.getElementById('sig-sb-ct').textContent = sbCount?`🔥${sbCount} STRONG BUY`:'';
  const tb = document.getElementById('sig-body');
  if(!sigs?.length){ tb.innerHTML='<tr><td colspan="13" class="no-data">▶ Press FULL SCAN to load signals</td></tr>'; return; }
  tb.innerHTML = sigs.map(s => {
    const bd = s.breakdown||{};
    const ts=bd.technical?.score||0,bs=bd.breakout?.score||0,fs=bd.fundamental?.score||0,is=bd.institutional?.score||0;
    const ss=bd.smc?.score||0;
    const gp=s.gap_pct||0;
    const cls=s.signal_class||'avoid';
    const sigCls=cls==='strong-buy'?'sig-sb':cls==='buy'?'sig-b':cls==='watch'?'sig-w':'sig-av';
    const actCls=s.action==='BUY'?'act-buy':s.action.startsWith('SELL')?'act-sell':'act-hold';
    const pips=Array.from({length:30},(_,i)=>{
      const on=i<s.score,hi=on&&s.score>=22,lo=on&&s.score<14;
      return `<div class="pip ${on?(hi?'lit hi':lo?'lit lo':'lit'):''}"></div>`;
    }).join('');
    const smcTags=[];
    if(s.smc_bos)   smcTags.push('<span style="color:#00ffcc;font-size:8px">BOS</span>');
    if(s.smc_choch) smcTags.push('<span style="color:#00e5ff;font-size:8px">CHoCH</span>');
    if(s.smc_sweep) smcTags.push('<span style="color:#ffaa00;font-size:8px">SWP</span>');
    if(s.smc_ob)    smcTags.push('<span style="color:#b060ff;font-size:8px">OB</span>');
    if(s.smc_fvg)   smcTags.push('<span style="color:#ff2d55;font-size:8px">FVG</span>');
    const smcCell=smcTags.length?smcTags.join(' '):`<span style="color:var(--muted);font-size:8px">${ss}/5</span>`;
    const fs2=(s.breakdown?.fibonacci?.score??s.fib_score??0);
    const fibColor=fs2>=3?'#ffd700':fs2>=2?'#daa520':'var(--muted)';
    const fibLabel=s.fib_in_golden?'★GZ':s.fib_at_key?'◆KEY':`${fs2}/3`;
    const fibCell=`<span style="color:${fibColor};font-size:9px;font-weight:700">${fibLabel}</span>`;
    const tpSrc=s.tp_source==='FIB 1.618'?`<span style="color:#ffd700;font-size:7px">FIB</span>`:`<span style="color:var(--muted);font-size:7px">ATR</span>`;
    return `<tr class="${s.symbol===selSym?'selected':''}" onclick="selectSym('${s.symbol}')">
      <td class="sym-cell"><strong>${s.symbol}</strong></td>
      <td style="font-family:var(--jet)">₹${s.ltp.toLocaleString('en-IN')}</td>
      <td class="${gp>=2?'a':gp>0?'':'r'}">${gp>=0?'+':''}${gp.toFixed(1)}%</td>
      <td style="color:var(--cyan)">${ts}/5</td>
      <td style="color:var(--green)">${bs}/5</td>
      <td style="color:var(--purple)">${fs}/5</td>
      <td style="color:var(--amber)">${is}/5</td>
      <td>${smcCell}</td>
      <td>${fibCell}</td>
      <td><div class="score-bar-wrap"><div class="score-pips">${pips}</div><span class="score-num">${s.score}/30</span></div></td>
      <td><span class="${sigCls} sig-label">${s.signal}</span></td>
      <td style="font-family:var(--jet);font-size:9px"><span class="r">₹${s.sl}</span> <span class="d">/</span> <span class="g">₹${s.tp}</span>${tpSrc}</td>
      <td><span class="${actCls}">${s.action}</span></td>
    </tr>`;
  }).join('');
}

function selectSym(sym) {
  selSym = sym;
  document.querySelectorAll('#sig-body tr').forEach(r=>{
    r.classList.toggle('selected', r.querySelector('strong')?.textContent===sym);
  });
  const s = allSigs.find(x=>x.symbol===sym);
  if(s){ showBreakdown(s); autofillPaperForm(s); }
}

function autofillPaperForm(s) {
  document.getElementById('pf-sym').value   = s.symbol;
  document.getElementById('pf-price').value = s.ltp;
  document.getElementById('pf-qty').value   = 1;
}

function showBreakdown(s) {
  document.getElementById('bd-sym').textContent = s.symbol;
  const bd = s.breakdown||{};
  const engs = [
    {k:'technical',    l:'Technical',    c:'var(--cyan)',   fc:'#00e5ff'},
    {k:'breakout',     l:'Breakout',     c:'var(--green)',  fc:'#00ff9d'},
    {k:'fundamental',  l:'Fundamental',  c:'var(--purple)', fc:'#b060ff'},
    {k:'institutional',l:'Institutional',c:'var(--amber)',  fc:'#ffaa00'},
    {k:'regime',       l:'Regime',       c:'var(--red)',    fc:'#ff2d55'},
    {k:'smc',          l:'SMC',          c:'#00ffcc',       fc:'#00ffcc'},
    {k:'fibonacci',    l:'Fibonacci',    c:'#ffd700',       fc:'#ffd700'},
  ];
  const meta = s.pe?`<div style="font-size:9px;color:var(--muted);font-family:var(--raj);padding:.2rem 0 .7rem;border-bottom:1px solid var(--border);margin-bottom:.6rem">PE: ${s.pe||'—'} &nbsp;|&nbsp; ROE: ${s.roe||'—'}% &nbsp;|&nbsp; MCap: ₹${((s.mcap_cr||0)).toLocaleString('en-IN')}Cr</div>`:'';
  let fibZone='';
  if(s.fib_golden_high||s.fib_swing_high){
    const inGz=s.fib_in_golden?'⭐ IN GOLDEN ZONE':'—';
    const gz=s.fib_golden_high?`₹${s.fib_golden_low}–₹${s.fib_golden_high}`:'—';
    const tp16=s.fib_tp_1618?`₹${s.fib_tp_1618}`:'—';
    const tpSrc=s.tp_source||'ATR×3.5';
    fibZone=`<div style="background:rgba(255,215,0,.04);border:1px solid rgba(255,215,0,.15);border-radius:3px;padding:.55rem .7rem;margin-bottom:.8rem;font-family:var(--raj);font-size:9px">
      <div style="color:#ffd700;font-weight:700;letter-spacing:.12em;margin-bottom:.4rem">FIBONACCI LEVELS</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.25rem .8rem;color:var(--muted)">
        <span>Swing High</span><span style="color:var(--text);font-family:var(--jet)">₹${s.fib_swing_high||'—'}</span>
        <span>Swing Low</span><span style="color:var(--text);font-family:var(--jet)">₹${s.fib_swing_low||'—'}</span>
        <span>Golden Zone</span><span style="color:#ffd700;font-family:var(--jet)">${gz}</span>
        <span>Zone status</span><span style="color:${s.fib_in_golden?'#ffd700':'var(--muted)'};">${inGz}</span>
        <span>1.618 ext (TP)</span><span style="color:var(--green);font-family:var(--jet)">${tp16}</span>
        <span>TP source</span><span style="color:${tpSrc==='FIB 1.618'?'#ffd700':'var(--steel)'}">${tpSrc}</span>
      </div>
    </div>`;
  }
  let html=meta+fibZone;
  engs.forEach(e=>{
    const en=bd[e.k]||{score:0,max:5,flags:[]};
    const pct=(en.score/en.max)*100;
    html+=`<div class="eng-block">
      <div class="eng-head"><span class="eng-name" style="color:${e.c}">${e.l}</span><span class="eng-score" style="color:${e.c}">${en.score} / ${en.max}</span></div>
      <div class="eng-track"><div class="eng-fill" style="width:${pct}%;background:${e.fc}"></div></div>
      ${(en.flags||[]).map(f=>{const fc=f.startsWith('✅')?'ok':f.startsWith('❌')?'fail':'warn';return `<div class="eng-flag ${fc}">${f}</div>`;}).join('')}
    </div>`;
  });
  document.getElementById('bd-body').innerHTML=html;
}

function renderGaps(gaps) {
  document.getElementById('gap-ct').textContent=(gaps||[]).length+' today';
  const tb=document.getElementById('gap-body');
  if(!gaps?.length){tb.innerHTML='<tr><td colspan="5" class="no-data">No gap-ups detected</td></tr>';return;}
  tb.innerHTML=gaps.map(g=>{
    const tag=g.gap_pct>=10?'<span class="gap-tag gap-exp">🔥 EXPLOSIVE</span>':g.gap_pct>=5?'<span class="gap-tag gap-str">🚀 STRONG</span>':'<span class="gap-tag gap-mod">📈 MODERATE</span>';
    return `<tr><td class="gap-sym">${g.symbol}</td><td class="gap-pct">+${g.gap_pct.toFixed(2)}%</td><td style="font-family:var(--jet)">₹${(g.ltp||0).toLocaleString('en-IN')}</td><td style="font-family:var(--jet)">${(g.volume||0).toLocaleString('en-IN')}</td><td>${tag}</td></tr>`;
  }).join('');
}

function renderPositions(pos, sigs) {
  const tb = document.getElementById('pos-body');
  const e  = Object.entries(pos||{});
  if(!e.length){ tb.innerHTML='<tr><td colspan="8" class="no-data">No open positions</td></tr>'; return; }
  const sigMap = {};
  (sigs||[]).forEach(s=>{ sigMap[s.symbol]=s.ltp; });
  let totalUnrl = 0;
  tb.innerHTML = e.map(([sym,p])=>{
    const ltp   = sigMap[sym] || p.entry;
    const unrl  = (ltp - p.entry) * p.qty;
    totalUnrl  += unrl;
    const uCls  = unrl>=0?'g':'r';
    return `<tr>
      <td><strong style="font-family:var(--raj);color:var(--cyan)">${sym}</strong></td>
      <td style="font-family:var(--jet)">₹${p.entry.toFixed(2)}</td>
      <td class="r" style="font-family:var(--jet)">₹${p.sl.toFixed(2)}</td>
      <td class="g" style="font-family:var(--jet)">₹${p.tp.toFixed(2)}</td>
      <td>${p.qty}</td>
      <td><span style="font-family:var(--orb);font-size:9px;color:var(--cyan)">${p.score}/30</span></td>
      <td class="${uCls}" style="font-family:var(--jet);font-weight:700">${unrl>=0?'+':''}₹${Math.abs(unrl).toLocaleString('en-IN')}</td>
      <td><span style="font-family:var(--raj);font-size:9px;font-weight:700;color:${p.trailing?'var(--amber)':'var(--muted)'}">${p.trailing?'TRAILING':'FIXED'}</span></td>
    </tr>`;
  }).join('');
  document.getElementById('pos-pnl-hdr').textContent = `Unrealised: ${totalUnrl>=0?'+':''}₹${Math.abs(totalUnrl).toLocaleString('en-IN')}`;
  document.getElementById('pos-pnl-hdr').className = 'ph-r '+(totalUnrl>=0?'g':'r');
}

function renderTrades(tl) {
  const tb = document.getElementById('trade-body');
  const t2 = [...(tl||[])].reverse();
  if(!t2.length){ tb.innerHTML='<tr><td colspan="7" class="no-data">No completed trades</td></tr>'; return; }
  const wins = t2.filter(t=>t.pnl>0).length;
  const totalPnl = t2.reduce((a,t)=>a+t.pnl,0);
  document.getElementById('trade-summary').textContent = `${wins}W/${t2.length-wins}L · Total ₹${totalPnl.toFixed(0)}`;
  document.getElementById('trade-summary').className = 'ph-r '+(totalPnl>=0?'g':'r');
  tb.innerHTML = t2.map(t=>`<tr>
    <td><strong style="font-family:var(--raj);color:var(--cyan)">${t.symbol}</strong></td>
    <td style="font-family:var(--jet)">₹${t.entry.toFixed(2)}</td>
    <td style="font-family:var(--jet)">₹${t.exit.toFixed(2)}</td>
    <td>${t.qty}</td>
    <td class="${t.pnl>=0?'g':'r'}" style="font-family:var(--jet);font-weight:700">${t.pnl>=0?'+':''}₹${Math.abs(t.pnl).toLocaleString('en-IN')}</td>
    <td style="font-size:9px;color:var(--muted)">${t.reason}</td>
    <td style="font-size:9px;color:var(--muted)">${new Date(t.date).toLocaleTimeString('en-IN',{hour12:false})}</td>
  </tr>`).join('');
}

function renderKiteLog(calls) {
  const el=document.getElementById('kite-log');
  if(!calls?.length){el.innerHTML='<div class="no-data">No API calls logged yet</div>';return;}
  el.innerHTML=[...calls].reverse().map(c=>`<div class="kl-row"><span class="kl-time">${c.time}</span><span class="kl-method">kite.${c.method}</span><span class="kl-detail">${c.detail}</span></div>`).join('');
}

function renderTicker(sigs, gaps) {
  const items=[];
  (gaps||[]).slice(0,8).forEach(g=>items.push(`<span class="tk-sym">${g.symbol}</span><span class="tk-val a">+${g.gap_pct.toFixed(1)}% GAP</span><span class="tk-sep">·</span>`));
  (sigs||[]).slice(0,15).forEach(s=>{
    const cls=s.signal_class==='strong-buy'?'g':s.signal_class==='buy'?'c':s.signal_class==='watch'?'a':'d';
    items.push(`<span class="tk-sym">${s.symbol}</span><span class="tk-val ${cls}">${s.signal} ${s.score}/30</span><span class="tk-sep">·</span>`);
  });
  if(!items.length) return;
  document.getElementById('tk-track').innerHTML=[...items,...items].join(' ');
}

async function triggerScan() {
  const btn=document.getElementById('scan-btn');
  btn.disabled=true; btn.querySelector('span').textContent='⟳ SCANNING'; btn.classList.add('running');
  document.getElementById('toast').style.display='block';
  setTimeout(()=>document.getElementById('toast').style.display='none',2500);
  await fetch('/api/scan',{method:'POST'});
  startProgressPoll();
  const poll=setInterval(async()=>{
    try {
      const d=await (await fetch('/api/state')).json();
      if(!d.scanning){
        clearInterval(poll); render(d);
        btn.disabled=false; btn.querySelector('span').textContent='▶ FULL SCAN'; btn.classList.remove('running');
      }
    } catch(e){clearInterval(poll);}
  },2500);
}

fetchState();
loadPaperState();
setInterval(fetchState, 30000);
setInterval(loadPaperState, 15000);
setInterval(()=>{ const t=document.getElementById('clock'); if(t) t.textContent=new Date().toLocaleTimeString('en-IN',{hour12:false}); }, 1000);
</script>
</body>
</html>"""
