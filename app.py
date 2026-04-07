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

from apscheduler.schedulers.background import BackgroundScheduler

from engine import (
    CFG, KiteLayer, UniverseManager, NSEClient,
    TechnicalEngine, FundamentalEngine, BreakoutScanner,
    InstitutionalEngine, MarketRegime, RiskManager, build_score,
    save_token, load_token
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
    "kite_calls":    [],
    "errors":        [],
    # ── Auth state (updated on login) ──────────────────────────
    "auth": {
        "status":      "unknown",   # "ok" | "missing" | "invalid"
        "user_name":   None,
        "user_id":     None,
        "token_saved": None,        # ISO timestamp when token was last saved
        "login_url":   None,        # filled at startup
    }
}


# ═════════════════════════════════════════════════════════════
# KITE SESSION SETUP
# ═════════════════════════════════════════════════════════════
def build_login_url() -> str:
    """Generate the Kite login URL pointing back to this app's /callback."""
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=CFG["KITE_API_KEY"])
        return kite.login_url()
    except Exception:
        api_key = CFG["KITE_API_KEY"]
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def init_kite(access_token: str = None) -> KiteLayer:
    """
    Creates KiteConnect instance and wraps in KiteLayer.
    Optionally accepts a fresh access_token (used by /callback).
    """
    if not KITE_OK:
        raise RuntimeError("kiteconnect not installed")

    token = access_token or CFG["ACCESS_TOKEN"]
    if not token:
        STATE["auth"]["status"] = "missing"
        raise RuntimeError(
            "No access token. Visit /kite-login to authenticate with Zerodha."
        )

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
    log.info(f"\n{'═'*65}")
    log.info(f"  FULL NSE INSTITUTIONAL SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"{'═'*65}")

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
        log.info(f"           (Kite kite.quote() — {math.ceil(len(all_symbols)/500)} batches of 500)")
        quote_data = kl.get_batch_quotes(all_symbols)
        log_kite_call("quote(all_nse)", f"{len(quote_data)} quotes fetched in {math.ceil(len(all_symbols)/500)} batches")
        log.info(f"  → {len(quote_data)} live quotes received")

        # ─────────────────────────────────────────────────────
        # STEP 3: Filter to Deep-Scan Candidates
        # Gap-ups always included, then top-volume stocks
        # ─────────────────────────────────────────────────────
        log.info("  STEP 3 — Filtering to deep-scan candidates...")
        candidates = universe.filter_for_deep_scan(quote_data)
        STATE["universe_size"] = len(quote_data)

        # Capture all gap-up alerts (even if not in deep scan)
        gap_alerts = sorted(
            [{"symbol": sym, "gap_pct": q["gap_pct"],
              "ltp": q["ltp"], "volume": q["volume"]}
             for sym, q in quote_data.items() if q.get("gap_pct", 0) >= CFG["GAP_UP_MIN_PCT"]],
            key=lambda x: x["gap_pct"], reverse=True
        )
        STATE["gap_alerts"] = gap_alerts[:30]
        log.info(f"  → {len(gap_alerts)} gap-up stocks found today (≥{CFG['GAP_UP_MIN_PCT']}%)")
        log.info(f"  → {len(candidates)} candidates selected for full analysis")

        # ─────────────────────────────────────────────────────
        # STEP 4: Market Regime + Macro Data
        # KITE API: kite.historical_data(NIFTY_TOKEN, ...)
        # NSE API:  India VIX, FII/DII flows
        # ─────────────────────────────────────────────────────
        log.info("  STEP 4 — Macro regime check...")
        nifty_df = kl.get_ohlcv(CFG["NIFTY_TOKEN"], days=300)
        nifty_df = TechnicalEngine.compute(nifty_df)
        log_kite_call("historical_data", f"NIFTY 50 — {len(nifty_df)} candles")

        vix = nse.get_india_vix()
        fii = nse.get_fii_dii()
        oc  = nse.get_option_chain_pcr("NIFTY")
        STATE["vix"]     = vix
        STATE["fii_dii"] = {**fii, "pcr": oc.get("pcr", "—")}

        reg_s, reg_f, regime_ok, reg_label = MarketRegime.check(nifty_df, vix)
        STATE["regime"] = {"score": reg_s, "label": reg_label, "flags": reg_f, "vix": vix}
        log.info(f"  → Regime: {reg_label} | VIX: {vix:.1f} | FII: ₹{fii.get('fii_net',0)/1e7:.0f}Cr | PCR: {oc.get('pcr',1)}")

        # ─────────────────────────────────────────────────────
        # STEP 5: Get available capital from Kite
        # KITE API: kite.margins(segment="equity")
        # ─────────────────────────────────────────────────────
        available_cap = kl.get_available_capital()
        log_kite_call("margins('equity')", f"₹{available_cap:,.0f} available")
        log.info(f"  STEP 5 — Available capital: ₹{available_cap:,.0f}")

        # ─────────────────────────────────────────────────────
        # STEP 6: Deep Analysis — Historical data per stock
        # KITE API: kite.historical_data(token, ...) per stock
        # ─────────────────────────────────────────────────────
        log.info(f"  STEP 6 — Deep analysis on {len(candidates)} stocks...")
        signals    = []
        open_count = len(STATE["positions"])

        for i, cand in enumerate(candidates, 1):
            sym   = cand["symbol"]
            token = cand["token"]
            quote = cand

            if not token:
                continue

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

                # Score all 5 engines
                tech_s,  tech_f  = TechnicalEngine.score(row)
                brk_s,   brk_f   = BreakoutScanner.score(df, row, nifty_df, quote)
                fund_d           = FundamentalEngine.get(sym)
                inst_s,  inst_f  = InstitutionalEngine.score(nse, sym, fii, oc, vix)
                full             = build_score(tech_s, tech_f, brk_s, brk_f,
                                               fund_d, inst_s, inst_f, reg_s, reg_f)

                entry = cand["ltp"] * (1 + CFG["SLIPPAGE_PCT"])
                sl, tp = RiskManager.sl_tp(entry, float(row["atr"]))
                qty    = RiskManager.position_size(entry, sl, available_cap)

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
                    "qty":          qty,
                    "score":        full["total"],
                    "max_score":    full["max"],
                    "signal":       full["signal"],
                    "signal_class": full["signal_class"],
                    "breakdown":    full["breakdown"],
                    "action":       "HOLD",
                    "time":         str(datetime.now()),
                    "pe":           fund_d.get("pe"),
                    "roe":          fund_d.get("roe"),
                    "mcap_cr":      fund_d.get("mcap_cr"),
                }

                # ── Entry: BUY ────────────────────────────────
                if (full["total"] >= CFG["SCORE_BUY"]
                        and sym not in STATE["positions"]
                        and open_count < CFG["MAX_OPEN_TRADES"]
                        and regime_ok):

                    if CFG["PAPER_TRADE"]:
                        log.info(f"  📝 PAPER BUY {sym} | {full['signal']} | Score {full['total']}/22 | Qty {qty}")
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

        log.info(f"\n  ✅ SCAN COMPLETE")
        log.info(f"     Stocks scanned:   {len(quote_data)}")
        log.info(f"     Deep analysed:    {len(candidates)}")
        log.info(f"     Gap-ups found:    {len(gap_alerts)}")
        log.info(f"     Signals scored:   {len(signals)}")
        log.info(f"     Buys triggered:   {sum(1 for s in signals if s['action']=='BUY')}")
        log.info(f"     Open positions:   {list(STATE['positions'].keys()) or 'None'}")

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
    return jsonify({
        "signals":       STATE["signals"][:100],    # top 100 by score
        "positions":     STATE["positions"],
        "trade_log":     tl[-30:],
        "gap_alerts":    STATE["gap_alerts"],
        "regime":        STATE["regime"],
        "fii_dii":       STATE["fii_dii"],
        "vix":           STATE["vix"],
        "last_scan":     STATE["last_scan"],
        "scanning":      STATE["scanning"],
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
    })


@app.route("/api/scan", methods=["POST"])
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


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": str(datetime.now()),
                    "paper": CFG["PAPER_TRADE"],
                    "universe": STATE["universe_size"],
                    "auth": STATE["auth"]["status"]})


# ═════════════════════════════════════════════════════════════
# KITE AUTH ROUTES  — zero-touch daily login flow
# ═════════════════════════════════════════════════════════════

@app.route("/kite-login")
def kite_login():
    """
    Step 1: Redirect user to Zerodha login.
    After login Zerodha redirects to /callback with ?request_token=...
    Just open  https://your-app.railway.app/kite-login  each morning.
    """
    url = build_login_url()
    log.info(f"  [Auth] Redirecting to Zerodha login: {url}")
    return redirect(url)


@app.route("/callback")
def kite_callback():
    """
    Step 2: Zerodha redirects here with ?request_token=...
    We exchange it for an access_token, save it, and reinitialise Kite.
    """
    global kl, universe

    request_token = request.args.get("request_token")
    status        = request.args.get("status", "")

    if status != "success" or not request_token:
        err = request.args.get("message", "Login cancelled or failed.")
        log.error(f"  [Auth] Callback error: {err}")
        return _auth_page("❌ Login Failed", err, success=False), 400

    try:
        from kiteconnect import KiteConnect
        kite_tmp = KiteConnect(api_key=CFG["KITE_API_KEY"])
        session  = kite_tmp.generate_session(request_token,
                                              api_secret=CFG["KITE_API_SECRET"])
        access_token = session["access_token"]

        # Persist token (survives process restarts within same deploy)
        save_token(access_token)
        STATE["auth"]["token_saved"] = datetime.now().isoformat()

        # Re-initialise the live Kite session
        kl       = init_kite(access_token)
        universe = None   # force universe reload with new session

        log.info(f"  [Auth] ✅ New session active for {STATE['auth']['user_name']}")
        return _auth_page(
            f"✅ Logged in as {STATE['auth']['user_name']}",
            "Token saved. The scanner will use this token automatically. "
            "You can close this tab.",
            success=True
        )

    except Exception as e:
        log.error(f"  [Auth] Token exchange failed: {e}")
        return _auth_page("❌ Token Exchange Failed", str(e), success=False), 500


@app.route("/api/auth-status")
def auth_status():
    return jsonify(STATE["auth"])


def _auth_page(title: str, message: str, success: bool) -> str:
    colour = "#00ff9d" if success else "#ff2d55"
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Kite Auth — Institutional Trader</title>
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
  a:hover{{background:#00e5ff;color:#010408}}
</style></head>
<body><div class="card">
  <h2>{title}</h2>
  <p>{message}</p>
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
    On startup: if a token exists (env or saved file), validate it silently.
    Sets STATE["auth"] so the dashboard shows the correct status immediately.
    Does NOT crash the app if token is missing/invalid — user just needs to
    visit /kite-login.
    """
    global kl
    token = CFG["ACCESS_TOKEN"]
    login_url = build_login_url()
    STATE["auth"]["login_url"] = login_url

    if not token:
        STATE["auth"]["status"] = "missing"
        log.warning("  [Auth] ⚠️  No access token found.")
        log.warning(f"  [Auth]    Open  /kite-login  to authenticate.")
        return

    try:
        kl = init_kite(token)
        log.info("  [Auth] ✅ Existing token is valid — ready to scan.")
    except Exception as e:
        log.warning(f"  [Auth] Existing token invalid: {e}")
        log.warning(f"  [Auth] Open  /kite-login  to get a fresh token.")


# Runs under both `python app.py` and gunicorn
probe_existing_token()
start_scheduler()

if __name__ == "__main__":
    log.info("🚀 Institutional Trader Pro — Full NSE Edition")
    log.info(f"   Mode:    {'PAPER' if CFG['PAPER_TRADE'] else '⚠ LIVE'}")
    log.info(f"   Capital: ₹{CFG['CAPITAL']:,}")
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

* { margin:0; padding:0; box-sizing:border-box; }
html { scroll-behavior: smooth; }

body {
  background: var(--void);
  color: var(--text);
  font-family: var(--jet);
  font-size: 11px;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── GRID BG ─────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}
body::after {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,100,180,0.08) 0%, transparent 70%);
  pointer-events: none;
}

/* ── SCAN LINE ───────────────────── */
.scanline {
  position: fixed; top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  box-shadow: 0 0 20px var(--cyan);
  z-index: 999;
  animation: scandown 6s linear infinite;
  opacity: 0.4;
}
@keyframes scandown { 0%{top:-2px} 100%{top:100vh} }

/* ── TOPBAR ──────────────────────── */
#topbar {
  position: sticky; top: 0; z-index: 100;
  height: 48px;
  display: flex; align-items: stretch;
  background: rgba(5,12,20,0.95);
  border-bottom: 1px solid var(--border2);
  backdrop-filter: blur(12px);
}
.logo-zone {
  display: flex; align-items: center; gap: .6rem;
  padding: 0 1.4rem;
  border-right: 1px solid var(--border2);
  min-width: 180px;
}
.logo-hex {
  width: 28px; height: 28px;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 900; color: #000;
}
.logo-text {
  font-family: var(--orb);
  font-size: 13px; font-weight: 700;
  letter-spacing: .15em; color: var(--cyan);
  text-shadow: 0 0 20px var(--cyan);
}
.logo-text span { color: var(--text2); font-weight: 400; }

.pills { display: flex; flex: 1; overflow: hidden; }
.pill {
  display: flex; align-items: center; gap: .4rem;
  padding: 0 1rem;
  border-right: 1px solid var(--border);
  font-size: 9px; font-family: var(--raj);
  letter-spacing: .08em;
}
.pill-label { color: var(--muted); font-size: 8px; font-weight: 600; }
.pill-val { font-size: 12px; font-weight: 700; font-family: var(--jet); }
.pill-dot {
  width: 4px; height: 4px; border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 6px var(--cyan);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:.2} }

.bar-right {
  display: flex; align-items: center; gap: .8rem;
  padding: 0 1rem; margin-left: auto;
}
.mode-badge {
  font-family: var(--orb); font-size: 8px; font-weight: 700;
  padding: .2rem .6rem; letter-spacing: .15em;
  border-radius: 2px;
}
.mode-paper { background: rgba(255,170,0,.1); color: var(--amber); border: 1px solid rgba(255,170,0,.3); }
.mode-live  { background: rgba(0,255,157,.1); color: var(--green); border: 1px solid rgba(0,255,157,.3); }

.scan-btn {
  font-family: var(--orb); font-size: 9px; font-weight: 700;
  letter-spacing: .12em; padding: .35rem 1rem;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  color: #000; border: none; border-radius: 2px;
  cursor: pointer; transition: .2s; position: relative;
  overflow: hidden;
}
.scan-btn::before {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(135deg, var(--purple), var(--cyan));
  opacity: 0; transition: .2s;
}
.scan-btn:hover::before { opacity: 1; }
.scan-btn:disabled { opacity: .4; cursor: not-allowed; }
.scan-btn span { position: relative; z-index: 1; }
.scan-btn.running {
  animation: btn-scan 1s ease-in-out infinite alternate;
}
@keyframes btn-scan {
  0% { box-shadow: 0 0 10px var(--cyan); }
  100% { box-shadow: 0 0 30px var(--cyan), 0 0 60px var(--cyan2); }
}

#clock { font-family: var(--orb); font-size: 11px; color: var(--steel); letter-spacing: .1em; }

/* ── LAYOUT ──────────────────────── */
main { position: relative; z-index: 1; padding: .8rem; display: grid; gap: .8rem; padding-bottom: 30px; }

/* ── PANEL ───────────────────────── */
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 3px;
  position: relative;
  overflow: hidden;
}
.panel::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border2), transparent);
}
.panel-accent-cyan::after {
  content: ''; position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--cyan), transparent);
}
.panel-accent-green::after {
  content: ''; position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--green), transparent);
}
.panel-accent-amber::after {
  content: ''; position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--amber), transparent);
}
.panel-accent-purple::after {
  content: ''; position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--purple), transparent);
}

.ph {
  display: flex; align-items: center; justify-content: space-between;
  padding: .45rem .85rem;
  background: rgba(0,0,0,.3);
  border-bottom: 1px solid var(--border);
}
.ph-l {
  font-family: var(--raj); font-size: 9px; font-weight: 700;
  letter-spacing: .18em; text-transform: uppercase; color: var(--steel);
}
.ph-r { font-size: 9px; color: var(--muted); font-family: var(--jet); }

/* ── STAT CARDS ──────────────────── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: .8rem; }
.stat-body { padding: .9rem 1rem 1rem 1rem; }
.stat-icon { font-size: 16px; margin-bottom: .4rem; opacity: .6; }
.stat-val {
  font-family: var(--orb); font-size: 24px; font-weight: 700;
  line-height: 1; letter-spacing: -.02em;
}
.stat-label { font-size: 8px; color: var(--muted); margin-top: .35rem; letter-spacing: .1em; font-family: var(--raj); text-transform: uppercase; }
.stat-sub { font-size: 9px; color: var(--steel); margin-top: .2rem; }

/* ── MACRO ROW ───────────────────── */
.macro-grid { display: grid; grid-template-columns: 1.1fr 1fr 1.8fr; gap: .8rem; }

/* ── REGIME ──────────────────────── */
.regime-body { padding: .8rem 1rem; }
.regime-label {
  font-family: var(--orb); font-size: 20px; font-weight: 900;
  letter-spacing: .05em; margin-bottom: .5rem;
}
.regime-flag {
  font-size: 9px; line-height: 2; color: var(--muted);
  font-family: var(--raj); font-weight: 500;
}
.regime-flag.ok { color: var(--green); }
.regime-flag.fail { color: var(--red); }
.regime-flag.warn { color: var(--amber); }

/* ── FLOW ────────────────────────── */
.flow-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: .45rem .85rem;
  border-bottom: 1px solid var(--border);
  font-family: var(--raj);
}
.flow-row:last-child { border-bottom: none; }
.flow-key { font-size: 9px; font-weight: 600; letter-spacing: .1em; color: var(--muted); }
.flow-val { font-size: 13px; font-weight: 700; font-family: var(--jet); }

/* ── ENGINE CHIPS ────────────────── */
.engine-chips { padding: .7rem .85rem .4rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.chip {
  font-family: var(--raj); font-size: 9px; font-weight: 700;
  padding: .25rem .6rem; border-radius: 2px; letter-spacing: .1em;
}
.chip-cyan   { background: rgba(0,229,255,.1);  color: var(--cyan);   border: 1px solid rgba(0,229,255,.25); }
.chip-green  { background: rgba(0,255,157,.1);  color: var(--green);  border: 1px solid rgba(0,255,157,.25); }
.chip-purple { background: rgba(176,96,255,.1); color: var(--purple); border: 1px solid rgba(176,96,255,.25); }
.chip-amber  { background: rgba(255,170,0,.1);  color: var(--amber);  border: 1px solid rgba(255,170,0,.25); }
.chip-red    { background: rgba(255,45,85,.1);  color: var(--red);    border: 1px solid rgba(255,45,85,.25); }

.score-legend { padding: .3rem .85rem .7rem; display: flex; gap: 1.2rem; }
.sl-item { font-size: 9px; font-family: var(--raj); font-weight: 600; }

/* ── TABS ────────────────────────── */
.tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  margin-bottom: .8rem;
}
.tab {
  padding: .55rem 1.2rem;
  font-family: var(--raj); font-size: 10px; font-weight: 700;
  letter-spacing: .15em; text-transform: uppercase;
  color: var(--muted); cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: .15s; position: relative;
}
.tab:hover { color: var(--text2); }
.tab.active { color: var(--cyan); border-bottom-color: var(--cyan); }
.tab.active::before {
  content: '';
  position: absolute; bottom: -1px; left: 0; right: 0; height: 8px;
  background: linear-gradient(0deg, rgba(0,229,255,.05), transparent);
}

/* ── SIGNAL TABLE ────────────────── */
.tbl-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th {
  font-family: var(--raj); font-size: 8px; font-weight: 700;
  letter-spacing: .15em; text-transform: uppercase;
  color: var(--muted); padding: .4rem .7rem;
  border-bottom: 1px solid var(--border2);
  background: rgba(0,0,0,.3); text-align: left; white-space: nowrap;
}
td { padding: .45rem .7rem; border-bottom: 1px solid var(--border); font-size: 10px; vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr { transition: background .1s; }
tr:hover td { background: rgba(0,229,255,.03); cursor: pointer; }
tr.selected td { background: rgba(0,229,255,.06); }

.sym-cell strong {
  font-family: var(--raj); font-size: 12px; font-weight: 700;
  color: var(--cyan); letter-spacing: .04em;
}

/* ── SCORE BAR ───────────────────── */
.score-bar-wrap { display: flex; align-items: center; gap: .5rem; }
.score-pips { display: flex; gap: 1.5px; }
.pip {
  width: 5px; height: 10px; border-radius: 1px;
  background: var(--dim);
  transition: background .2s;
}
.pip.lit { background: var(--cyan); box-shadow: 0 0 4px var(--cyan2); }
.pip.lit.hi { background: var(--green); box-shadow: 0 0 4px var(--green2); }
.pip.lit.lo { background: var(--amber); }
.score-num { font-family: var(--orb); font-size: 9px; color: var(--steel); min-width: 28px; }

/* ── SIGNAL LABELS ───────────────── */
.sig-label {
  font-family: var(--raj); font-size: 9px; font-weight: 700;
  padding: .18rem .45rem; border-radius: 2px; letter-spacing: .08em; white-space: nowrap;
}
.sig-sb  { background: rgba(0,255,157,.1);  color: var(--green);  border: 1px solid rgba(0,255,157,.3); }
.sig-b   { background: rgba(0,229,255,.1);  color: var(--cyan);   border: 1px solid rgba(0,229,255,.3); }
.sig-w   { background: rgba(255,170,0,.1);  color: var(--amber);  border: 1px solid rgba(255,170,0,.3); }
.sig-av  { background: rgba(74,122,155,.07); color: var(--muted); border: 1px solid var(--border2); }

.act-buy  { font-family: var(--raj); font-size: 9px; font-weight: 700; padding: .18rem .45rem; border-radius: 2px; background: rgba(0,255,157,.12); color: var(--green); border: 1px solid rgba(0,255,157,.3); }
.act-sell { font-family: var(--raj); font-size: 9px; font-weight: 700; padding: .18rem .45rem; border-radius: 2px; background: rgba(255,45,85,.12);  color: var(--red);   border: 1px solid rgba(255,45,85,.3); }
.act-hold { font-family: var(--raj); font-size: 9px; font-weight: 700; padding: .18rem .45rem; border-radius: 2px; background: transparent; color: var(--muted); border: 1px solid var(--border2); }

/* ── BREAKDOWN PANEL ─────────────── */
.bd-body { padding: .8rem; max-height: 420px; overflow-y: auto; }
.eng-block { margin-bottom: 1rem; }
.eng-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: .35rem;
}
.eng-name { font-family: var(--raj); font-size: 9px; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
.eng-score { font-family: var(--orb); font-size: 10px; font-weight: 700; }
.eng-track { height: 3px; background: var(--dim); border-radius: 2px; margin-bottom: .5rem; overflow: hidden; }
.eng-fill  { height: 100%; border-radius: 2px; transition: width .5s ease; }
.eng-flag  { font-size: 9px; line-height: 1.9; font-family: var(--raj); font-weight: 500; color: var(--muted); }
.eng-flag.ok   { color: var(--green); }
.eng-flag.fail { color: var(--red); }
.eng-flag.warn { color: var(--amber); }

/* ── GAP ALERT ───────────────────── */
.gap-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: .5rem .85rem; border-bottom: 1px solid var(--border);
  transition: background .1s;
}
.gap-row:hover { background: rgba(255,170,0,.03); }
.gap-row:last-child { border-bottom: none; }
.gap-sym { font-family: var(--raj); font-size: 12px; font-weight: 700; color: var(--amber); }
.gap-pct { font-family: var(--orb); font-size: 13px; font-weight: 700; color: var(--green); }
.gap-tag { font-family: var(--raj); font-size: 9px; font-weight: 700; padding: .15rem .4rem; border-radius: 2px; }
.gap-exp  { background: rgba(255,45,85,.1);  color: var(--red);   border: 1px solid rgba(255,45,85,.3); }
.gap-str  { background: rgba(0,255,157,.1);  color: var(--green); border: 1px solid rgba(0,255,157,.3); }
.gap-mod  { background: rgba(255,170,0,.1);  color: var(--amber); border: 1px solid rgba(255,170,0,.3); }

/* ── KITE LOG ────────────────────── */
.kl-row { display: flex; gap: .8rem; padding: .35rem .85rem; border-bottom: 1px solid var(--border); font-size: 9px; }
.kl-row:last-child { border-bottom: none; }
.kl-time { color: var(--muted); min-width: 55px; }
.kl-method { color: var(--cyan); font-weight: 700; min-width: 180px; }
.kl-detail { color: var(--steel); }

/* ── TWO COL ─────────────────────── */
.two-col { display: grid; grid-template-columns: 1.4fr 1fr; gap: .8rem; }

/* ── TICKER ──────────────────────── */
#ticker {
  position: fixed; bottom: 0; left: 0; right: 0; height: 26px;
  background: rgba(5,12,20,.97);
  border-top: 1px solid var(--border2);
  display: flex; align-items: center; overflow: hidden; z-index: 100;
}
.tk-track { display: flex; gap: 2.5rem; animation: ticker-scroll 60s linear infinite; white-space: nowrap; padding-left: 100%; }
@keyframes ticker-scroll { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
.tk-sym { font-family: var(--raj); font-size: 10px; font-weight: 700; color: var(--steel); margin-right: .25rem; }
.tk-val { font-size: 10px; font-family: var(--jet); }
.tk-sep { color: var(--border2); }

/* ── TOAST ───────────────────────── */
#toast {
  position: fixed; top: 52px; right: 1rem; z-index: 300;
  background: var(--cyan); color: #000;
  font-family: var(--orb); font-size: 9px; font-weight: 700; letter-spacing: .15em;
  padding: .4rem .9rem; border-radius: 2px;
  display: none;
  box-shadow: 0 4px 20px rgba(0,229,255,.4);
}

/* ── UTILS ───────────────────────── */
.g  { color: var(--green); }
.r  { color: var(--red); }
.c  { color: var(--cyan); }
.a  { color: var(--amber); }
.d  { color: var(--muted); }
.no-data { padding: 2rem; text-align: center; color: var(--muted); font-family: var(--raj); font-size: 10px; letter-spacing: .1em; }
.pulse { animation: pulse-anim 1s ease-in-out infinite; }
@keyframes pulse-anim { 0%,100%{opacity:1} 50%{opacity:.3} }

::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
</head>
<body>

<div class="scanline"></div>

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
    <div class="pill"><span class="pill-label">POSITIONS</span>&nbsp;<span class="pill-val g" id="p-pos">0</span></div>
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

<main>

<!-- STAT CARDS -->
<div class="stat-grid">
  <div class="panel panel-accent-cyan">
    <div class="ph"><span class="ph-l">NSE Universe</span></div>
    <div class="stat-body">
      <div class="stat-val c" id="s-uni">0</div>
      <div class="stat-label">Stocks scanned this run</div>
    </div>
  </div>
  <div class="panel panel-accent-amber">
    <div class="ph"><span class="ph-l">Gap-Up Alerts</span></div>
    <div class="stat-body">
      <div class="stat-val a" id="s-gaps">0</div>
      <div class="stat-label">≥2% gap stocks today</div>
    </div>
  </div>
  <div class="panel panel-accent-green">
    <div class="ph"><span class="ph-l">Session P&amp;L</span></div>
    <div class="stat-body">
      <div class="stat-val g" id="s-pnl">+₹0</div>
      <div class="stat-label" id="s-wr">Win 0% · 0 trades</div>
    </div>
  </div>
  <div class="panel panel-accent-purple">
    <div class="ph"><span class="ph-l">Open Positions</span></div>
    <div class="stat-body">
      <div class="stat-val" style="color:var(--purple)" id="s-pos">0</div>
      <div class="stat-label">of 6 max slots</div>
    </div>
  </div>
</div>

<!-- MACRO ROW -->
<div class="macro-grid">
  <div class="panel panel-accent-cyan">
    <div class="ph"><span class="ph-l">Market Regime</span><span class="ph-r" id="r-score">—</span></div>
    <div class="regime-body">
      <div class="regime-label" id="r-label">—</div>
      <div id="r-flags"></div>
    </div>
  </div>

  <div class="panel panel-accent-amber">
    <div class="ph"><span class="ph-l">Institutional Flow</span></div>
    <div class="flow-row"><span class="flow-key">FII NET</span><span class="flow-val" id="f-fii">—</span></div>
    <div class="flow-row"><span class="flow-key">DII NET</span><span class="flow-val" id="f-dii">—</span></div>
    <div class="flow-row"><span class="flow-key">NIFTY PCR</span><span class="flow-val" id="f-pcr">—</span></div>
  </div>

  <div class="panel panel-accent-purple">
    <div class="ph"><span class="ph-l">Scoring Engine — Max 22 Points</span></div>
    <div class="engine-chips">
      <span class="chip chip-cyan">TECHNICAL ×5</span>
      <span class="chip chip-green">BREAKOUT ×5</span>
      <span class="chip chip-purple">FUNDAMENTAL ×5</span>
      <span class="chip chip-amber">INSTITUTIONAL ×5</span>
      <span class="chip chip-red">REGIME ×2</span>
    </div>
    <div class="score-legend">
      <span class="sl-item g">●&nbsp;16+ STRONG BUY</span>
      <span class="sl-item c">●&nbsp;11+ BUY</span>
      <span class="sl-item a">●&nbsp;7+ WATCHLIST</span>
      <span class="sl-item d">●&nbsp;&lt;7 AVOID</span>
    </div>
  </div>
</div>

<!-- TABS -->
<div>
<div class="tabs">
  <div class="tab active" onclick="showTab('signals')">SIGNALS</div>
  <div class="tab" onclick="showTab('gaps')">GAP ALERTS</div>
  <div class="tab" onclick="showTab('positions')">POSITIONS</div>
  <div class="tab" onclick="showTab('trades')">TRADE LOG</div>
  <div class="tab" onclick="showTab('kite')">KITE API LOG</div>
</div>

<!-- SIGNALS -->
<div id="tab-signals">
<div class="two-col">
  <div class="panel">
    <div class="ph"><span class="ph-l">Signal Scanner</span><span class="ph-r" id="sig-ct">0 scored</span></div>
    <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>SYMBOL</th><th>LTP</th><th>GAP%</th>
        <th style="color:var(--cyan)">TECH</th>
        <th style="color:var(--green)">BRK</th>
        <th style="color:var(--purple)">FUND</th>
        <th style="color:var(--amber)">INST</th>
        <th>SCORE / 22</th><th>SIGNAL</th><th>SL / TP</th><th>ACTION</th>
      </tr></thead>
      <tbody id="sig-body"><tr><td colspan="11" class="no-data">▶ Press FULL SCAN to load NSE signals</td></tr></tbody>
    </table>
    </div>
  </div>
  <div class="panel">
    <div class="ph"><span class="ph-l">Analysis Breakdown</span><span class="ph-r c" id="bd-sym">select a row</span></div>
    <div class="bd-body" id="bd-body"><div class="no-data">Click any stock row to see 5-engine breakdown</div></div>
  </div>
</div>
</div>

<!-- GAP ALERTS -->
<div id="tab-gaps" style="display:none">
<div class="panel">
  <div class="ph"><span class="ph-l">Gap-Up Alerts — Full NSE Scan</span><span class="ph-r" id="gap-ct">—</span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>GAP %</th><th>LTP</th><th>VOLUME</th><th>CLASSIFICATION</th></tr></thead>
    <tbody id="gap-body"><tr><td colspan="5" class="no-data">—</td></tr></tbody>
  </table>
</div>
</div>

<!-- POSITIONS -->
<div id="tab-positions" style="display:none">
<div class="panel">
  <div class="ph"><span class="ph-l">Open Positions</span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>ENTRY</th><th>CURR SL</th><th>TARGET</th><th>QTY</th><th>SCORE</th><th>SL TYPE</th></tr></thead>
    <tbody id="pos-body"><tr><td colspan="7" class="no-data">No open positions</td></tr></tbody>
  </table>
</div>
</div>

<!-- TRADES -->
<div id="tab-trades" style="display:none">
<div class="panel">
  <div class="ph"><span class="ph-l">Trade Log</span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>ENTRY</th><th>EXIT</th><th>QTY</th><th>P&amp;L</th><th>REASON</th><th>TIME</th></tr></thead>
    <tbody id="trade-body"><tr><td colspan="7" class="no-data">No completed trades</td></tr></tbody>
  </table>
</div>
</div>

<!-- KITE LOG -->
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
let allSigs = []; let selSym = null;
const TABS = ['signals','gaps','positions','trades','kite'];

function showTab(t) {
  TABS.forEach(x => {
    document.getElementById('tab-'+x).style.display = x===t?'':'none';
  });
  document.querySelectorAll('.tab').forEach((el,i) => {
    if(TABS[i]===t) el.classList.add('active'); else el.classList.remove('active');
  });
}

async function fetchState() {
  try { const d = await (await fetch('/api/state')).json(); render(d); }
  catch(e) {}
}

function fmt(n) { return (n>=0?'+':'') + '₹' + Math.abs(n||0).toLocaleString('en-IN'); }
function fmtCr(n) { return (n>=0?'+':'') + '₹' + Math.abs((n||0)/1e7).toFixed(0) + 'Cr'; }

function render(d) {
  allSigs = d.signals || [];
  const s = d.stats;

  // badge
  const mb = document.getElementById('mode-badge');
  mb.textContent = d.paper_trade ? 'PAPER' : 'LIVE';
  mb.className = 'mode-badge ' + (d.paper_trade ? 'mode-paper' : 'mode-live');

  // pills
  const reg = d.regime || {};
  const rl = reg.label || '—';
  document.getElementById('p-regime').textContent = rl;
  document.getElementById('p-regime').className = 'pill-val ' + (rl.includes('BULL')?'g':'r');
  const vix = d.vix || 15;
  document.getElementById('p-vix').textContent = vix.toFixed(1);
  document.getElementById('p-vix').className = 'pill-val ' + (vix<15?'g':vix<20?'a':'r');
  const fn = d.fii_dii?.fii_net || 0;
  document.getElementById('p-fii').textContent = fmtCr(fn);
  document.getElementById('p-fii').className = 'pill-val ' + (fn>=0?'g':'r');
  document.getElementById('p-pcr').textContent = d.fii_dii?.pcr || '—';
  document.getElementById('p-uni').textContent = s.universe_size.toLocaleString();
  document.getElementById('p-gaps').textContent = s.gap_alerts;
  document.getElementById('p-pos').textContent = s.open_positions;
  const pnl = s.total_pnl;
  document.getElementById('p-pnl').textContent = fmt(pnl);
  document.getElementById('p-pnl').className = 'pill-val ' + (pnl>=0?'g':'r');

  // stats
  document.getElementById('s-uni').textContent = s.universe_size.toLocaleString();
  document.getElementById('s-gaps').textContent = s.gap_alerts;
  const pe = document.getElementById('s-pnl');
  pe.textContent = fmt(pnl);
  pe.className = 'stat-val ' + (pnl>=0?'g':'r');
  document.getElementById('s-wr').textContent = `Win ${s.win_rate}% · ${s.total_trades} trades`;
  document.getElementById('s-pos').textContent = s.open_positions;

  // regime
  document.getElementById('r-label').textContent = rl;
  document.getElementById('r-label').className = 'regime-label ' + (rl.includes('BULL')?'g':'r');
  document.getElementById('r-score').textContent = `${reg.score||0}/2`;
  document.getElementById('r-flags').innerHTML = (reg.flags||[]).map(f => {
    const cls = f.startsWith('✅')?'ok':f.startsWith('❌')?'fail':'warn';
    return `<div class="regime-flag ${cls}">${f}</div>`;
  }).join('');

  // flow
  const dn = d.fii_dii?.dii_net || 0;
  document.getElementById('f-fii').textContent = fmtCr(fn);
  document.getElementById('f-fii').className = 'flow-val ' + (fn>=0?'g':'r');
  document.getElementById('f-dii').textContent = fmtCr(dn);
  document.getElementById('f-dii').className = 'flow-val ' + (dn>=0?'g':'r');
  document.getElementById('f-pcr').textContent = d.fii_dii?.pcr || '—';

  renderSignals(d.signals);
  renderGaps(d.gap_alerts);
  renderPositions(d.positions);
  renderTrades(d.trade_log);
  renderKiteLog(d.kite_calls);
  renderTicker(d.signals, d.gap_alerts);

  // auth status badge
  const auth = d.auth || {};
  const authOk  = document.getElementById('auth-ok');
  const authBtn = document.getElementById('auth-btn');
  if (auth.status === 'ok') {
    authOk.style.display = 'inline-block';
    authOk.title = `${auth.user_name} (${auth.user_id})`;
    authBtn.style.display = 'none';
  } else {
    authOk.style.display = 'none';
    authBtn.style.display = 'inline-block';
    authBtn.title = auth.status === 'missing' ? 'No token — click to login' : 'Token invalid — click to re-login';
  }

  if(selSym) {
    const found = d.signals.find(x=>x.symbol===selSym);
    if(found) showBreakdown(found);
  }
}

function renderSignals(sigs) {
  document.getElementById('sig-ct').textContent = (sigs||[]).length + ' scored';
  const tb = document.getElementById('sig-body');
  if (!sigs?.length) {
    tb.innerHTML = '<tr><td colspan="11" class="no-data">▶ Press FULL SCAN to load signals</td></tr>'; return;
  }
  tb.innerHTML = sigs.map(s => {
    const bd  = s.breakdown || {};
    const ts  = bd.technical?.score||0, bs=bd.breakout?.score||0;
    const fs  = bd.fundamental?.score||0, is=bd.institutional?.score||0;
    const gp  = s.gap_pct||0;
    const cls = s.signal_class||'avoid';
    const sigCls = cls==='strong-buy'?'sig-sb':cls==='buy'?'sig-b':cls==='watch'?'sig-w':'sig-av';
    const actCls = s.action==='BUY'?'act-buy':s.action.startsWith('SELL')?'act-sell':'act-hold';
    const isSel  = s.symbol===selSym;

    const pips = Array.from({length:22},(_,i)=>{
      const on = i < s.score;
      const hi = on && s.score >= 16;
      const lo = on && s.score < 11;
      return `<div class="pip ${on?(hi?'lit hi':lo?'lit lo':'lit'):''}"></div>`;
    }).join('');

    return `<tr class="${isSel?'selected':''}" onclick="selectSym('${s.symbol}')">
      <td class="sym-cell"><strong>${s.symbol}</strong></td>
      <td style="font-family:var(--jet)">₹${s.ltp.toLocaleString('en-IN')}</td>
      <td class="${gp>=2?'a':gp>0?'':'r'}">${gp>=0?'+':''}${gp.toFixed(1)}%</td>
      <td style="color:var(--cyan)">${ts}/5</td>
      <td style="color:var(--green)">${bs}/5</td>
      <td style="color:var(--purple)">${fs}/5</td>
      <td style="color:var(--amber)">${is}/5</td>
      <td>
        <div class="score-bar-wrap">
          <div class="score-pips">${pips}</div>
          <span class="score-num">${s.score}/22</span>
        </div>
      </td>
      <td><span class="${sigCls} sig-label">${s.signal}</span></td>
      <td style="font-family:var(--jet);font-size:9px">
        <span class="r">₹${s.sl}</span> <span class="d">/</span> <span class="g">₹${s.tp}</span>
      </td>
      <td><span class="${actCls}">${s.action}</span></td>
    </tr>`;
  }).join('');
}

function selectSym(sym) {
  selSym = sym;
  document.querySelectorAll('#sig-body tr').forEach(r => {
    r.classList.toggle('selected', r.querySelector('strong')?.textContent===sym);
  });
  const s = allSigs.find(x=>x.symbol===sym);
  if(s) showBreakdown(s);
}

function showBreakdown(s) {
  document.getElementById('bd-sym').textContent = s.symbol;
  const bd = s.breakdown || {};
  const engs = [
    {k:'technical',     l:'Technical',     c:'var(--cyan)',   fc:'#00e5ff'},
    {k:'breakout',      l:'Breakout',       c:'var(--green)',  fc:'#00ff9d'},
    {k:'fundamental',   l:'Fundamental',    c:'var(--purple)', fc:'#b060ff'},
    {k:'institutional', l:'Institutional',  c:'var(--amber)',  fc:'#ffaa00'},
    {k:'regime',        l:'Regime',         c:'var(--red)',    fc:'#ff2d55'},
  ];
  const meta = s.pe ? `<div style="font-size:9px;color:var(--muted);font-family:var(--raj);padding:.2rem 0 .7rem;border-bottom:1px solid var(--border);margin-bottom:.6rem">
    PE: ${s.pe||'—'} &nbsp;|&nbsp; ROE: ${s.roe||'—'}% &nbsp;|&nbsp; MCap: ₹${((s.mcap_cr||0)).toLocaleString('en-IN')}Cr
  </div>` : '';
  let html = meta;
  engs.forEach(e => {
    const en  = bd[e.k]||{score:0,max:5,flags:[]};
    const pct = (en.score/en.max)*100;
    html += `<div class="eng-block">
      <div class="eng-head">
        <span class="eng-name" style="color:${e.c}">${e.l}</span>
        <span class="eng-score" style="color:${e.c}">${en.score} / ${en.max}</span>
      </div>
      <div class="eng-track"><div class="eng-fill" style="width:${pct}%;background:${e.fc}"></div></div>
      ${(en.flags||[]).map(f=>{
        const fc=f.startsWith('✅')?'ok':f.startsWith('❌')?'fail':'warn';
        return `<div class="eng-flag ${fc}">${f}</div>`;
      }).join('')}
    </div>`;
  });
  document.getElementById('bd-body').innerHTML = html;
}

function renderGaps(gaps) {
  document.getElementById('gap-ct').textContent = (gaps||[]).length + ' today';
  const tb = document.getElementById('gap-body');
  if(!gaps?.length){tb.innerHTML='<tr><td colspan="5" class="no-data">No gap-ups detected</td></tr>';return;}
  tb.innerHTML = gaps.map(g=>{
    const tag = g.gap_pct>=10?'<span class="gap-tag gap-exp">🔥 EXPLOSIVE</span>'
               :g.gap_pct>=5 ?'<span class="gap-tag gap-str">🚀 STRONG</span>'
               :'<span class="gap-tag gap-mod">📈 MODERATE</span>';
    return `<tr>
      <td class="gap-sym">${g.symbol}</td>
      <td class="gap-pct">+${g.gap_pct.toFixed(2)}%</td>
      <td style="font-family:var(--jet)">₹${(g.ltp||0).toLocaleString('en-IN')}</td>
      <td style="font-family:var(--jet)">${(g.volume||0).toLocaleString('en-IN')}</td>
      <td>${tag}</td>
    </tr>`;
  }).join('');
}

function renderPositions(pos) {
  const tb = document.getElementById('pos-body');
  const e  = Object.entries(pos||{});
  if(!e.length){tb.innerHTML='<tr><td colspan="7" class="no-data">No open positions</td></tr>';return;}
  tb.innerHTML = e.map(([sym,p])=>`<tr>
    <td><strong style="font-family:var(--raj);color:var(--cyan)">${sym}</strong></td>
    <td style="font-family:var(--jet)">₹${p.entry.toFixed(2)}</td>
    <td class="r" style="font-family:var(--jet)">₹${p.sl.toFixed(2)}</td>
    <td class="g" style="font-family:var(--jet)">₹${p.tp.toFixed(2)}</td>
    <td>${p.qty}</td>
    <td><span style="font-family:var(--orb);font-size:9px;color:var(--cyan)">${p.score}/22</span></td>
    <td><span style="font-family:var(--raj);font-size:9px;font-weight:700;color:${p.trailing?'var(--amber)':'var(--muted)'}">${p.trailing?'TRAILING':'FIXED'}</span></td>
  </tr>`).join('');
}

function renderTrades(tl) {
  const tb = document.getElementById('trade-body');
  const t2 = [...(tl||[])].reverse();
  if(!t2.length){tb.innerHTML='<tr><td colspan="7" class="no-data">No completed trades</td></tr>';return;}
  tb.innerHTML = t2.map(t=>`<tr>
    <td><strong style="font-family:var(--raj);color:var(--cyan)">${t.symbol}</strong></td>
    <td style="font-family:var(--jet)">₹${t.entry.toFixed(2)}</td>
    <td style="font-family:var(--jet)">₹${t.exit.toFixed(2)}</td>
    <td>${t.qty}</td>
    <td class="${t.pnl>=0?'g':'r'}" style="font-family:var(--jet);font-weight:700">
      ${t.pnl>=0?'+':''}₹${Math.abs(t.pnl).toLocaleString('en-IN')}
    </td>
    <td style="font-size:9px;color:var(--muted)">${t.reason}</td>
    <td style="font-size:9px;color:var(--muted)">${new Date(t.date).toLocaleTimeString('en-IN',{hour12:false})}</td>
  </tr>`).join('');
}

function renderKiteLog(calls) {
  const el = document.getElementById('kite-log');
  if(!calls?.length){el.innerHTML='<div class="no-data">No API calls logged yet</div>';return;}
  el.innerHTML = [...calls].reverse().map(c=>`<div class="kl-row">
    <span class="kl-time">${c.time}</span>
    <span class="kl-method">kite.${c.method}</span>
    <span class="kl-detail">${c.detail}</span>
  </div>`).join('');
}

function renderTicker(sigs, gaps) {
  const items = [];
  (gaps||[]).slice(0,8).forEach(g=>
    items.push(`<span class="tk-sym">${g.symbol}</span><span class="tk-val a">+${g.gap_pct.toFixed(1)}% GAP</span><span class="tk-sep">·</span>`)
  );
  (sigs||[]).slice(0,15).forEach(s=>{
    const cls = s.signal_class==='strong-buy'?'g':s.signal_class==='buy'?'c':s.signal_class==='watch'?'a':'d';
    items.push(`<span class="tk-sym">${s.symbol}</span><span class="tk-val ${cls}">${s.signal} ${s.score}/22</span><span class="tk-sep">·</span>`);
  });
  if(!items.length) return;
  const full = [...items,...items].join(' ');
  document.getElementById('tk-track').innerHTML = full;
}

async function triggerScan() {
  const btn = document.getElementById('scan-btn');
  btn.disabled = true;
  btn.querySelector('span').textContent = '⟳ SCANNING';
  btn.classList.add('running');
  document.getElementById('toast').style.display = 'block';
  setTimeout(()=>document.getElementById('toast').style.display='none', 2500);
  await fetch('/api/scan',{method:'POST'});
  const poll = setInterval(async()=>{
    try {
      const d = await (await fetch('/api/state')).json();
      if(!d.scanning) {
        clearInterval(poll); render(d);
        btn.disabled=false;
        btn.querySelector('span').textContent='▶ FULL SCAN';
        btn.classList.remove('running');
      }
    } catch(e){clearInterval(poll);}
  }, 2500);
}

fetchState();
setInterval(fetchState, 30000);
setInterval(()=>{
  const t = document.getElementById('clock');
  if(t) t.textContent = new Date().toLocaleTimeString('en-IN',{hour12:false});
}, 1000);
</script>
</body>
</html>"""
