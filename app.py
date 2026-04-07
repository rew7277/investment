"""
═══════════════════════════════════════════════════════════════
  INSTITUTIONAL TRADER PRO — app.py
  Scans FULL NSE universe. No hardcoded stock list.
  Every Kite API call is labelled.
═══════════════════════════════════════════════════════════════
"""

import os, json, logging, threading, math
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string

from apscheduler.schedulers.background import BackgroundScheduler

from engine import (
    CFG, KiteLayer, UniverseManager, NSEClient,
    TechnicalEngine, FundamentalEngine, BreakoutScanner,
    InstitutionalEngine, MarketRegime, RiskManager, build_score
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
    "gap_alerts":    [],   # NEW: all gap-ups from full NSE scan
    "universe_size": 0,    # how many stocks scanned
    "last_scan":     None,
    "scanning":      False,
    "kite_calls":    [],   # log of every Kite API call made
    "errors":        [],
}


# ═════════════════════════════════════════════════════════════
# KITE SESSION SETUP
# ═════════════════════════════════════════════════════════════
def init_kite() -> KiteLayer:
    """
    Creates KiteConnect instance and wraps in KiteLayer.
    KiteLayer is the ONLY object that calls Kite API methods.
    """
    if not KITE_OK:
        raise RuntimeError("kiteconnect not installed — pip install kiteconnect")
    if not CFG["ACCESS_TOKEN"]:
        raise RuntimeError(
            "KITE_ACCESS_TOKEN env var not set.\n"
            "Run: python3 get_token.py  to generate today's token."
        )
    kite = KiteConnect(api_key=CFG["KITE_API_KEY"])
    kite.set_access_token(CFG["ACCESS_TOKEN"])

    # Validate token
    try:
        profile = kite.profile()   # KITE API: kite.profile()
        log.info(f"  [Kite] ✅ Authenticated as: {profile['user_name']}")
    except Exception as e:
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
        }
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
                    "universe": STATE["universe_size"]})


# ═════════════════════════════════════════════════════════════
# DASHBOARD HTML  (Bloomberg Terminal Style)
# ═════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Institutional Trader Pro — Full NSE</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#040810;--s1:#080e1c;--s2:#0c1525;--s3:#111d30;
  --b1:#1a2d45;--b2:#223555;
  --gr:#00ff88;--gr2:#00cc6a;--rd:#ff3355;--go:#ffcc00;
  --bl:#3399ff;--cy:#00ddff;--pu:#bb88ff;--wh:#e0eeff;
  --dm:#3a5570;--dm2:#1d3050;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--wh);font-family:var(--mono);font-size:11px;min-height:100vh;}

/* scanlines */
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,255,136,.008) 3px,rgba(0,255,136,.008) 4px);}

/* TOP BAR */
#bar{position:sticky;top:0;z-index:100;height:38px;display:flex;align-items:stretch;
  background:var(--s1);border-bottom:1px solid var(--b1);}
.br-logo{display:flex;align-items:center;padding:0 1.2rem;
  font-family:var(--sans);font-size:12px;font-weight:800;letter-spacing:.15em;
  border-right:1px solid var(--b1);white-space:nowrap;}
.br-logo em{color:var(--gr);font-style:normal;}
.br-pills{display:flex;flex:1;overflow:hidden;}
.pill{display:flex;align-items:center;gap:.3rem;padding:0 .9rem;
  border-right:1px solid var(--b1);font-size:9px;white-space:nowrap;color:var(--dm);}
.pill strong{font-size:11px;}
.g{color:var(--gr);} .r{color:var(--rd);} .go{color:var(--go);}
.b{color:var(--bl);} .c{color:var(--cy);}
.br-right{display:flex;align-items:center;gap:.5rem;padding:0 .8rem;margin-left:auto;}
.badge{font-size:8px;font-weight:700;padding:.15rem .35rem;border-radius:1px;letter-spacing:.1em;}
.badge.p{background:#ffcc0012;color:var(--go);border:1px solid #ffcc0035;}
.badge.l{background:#00ff8812;color:var(--gr);border:1px solid #00ff8835;}
.btn{background:var(--gr);color:#000;border:none;padding:.28rem .7rem;
  font-family:var(--mono);font-size:9px;font-weight:700;letter-spacing:.1em;
  cursor:pointer;border-radius:1px;transition:.12s;}
.btn:hover{background:var(--gr2);}
.btn:disabled{opacity:.35;cursor:not-allowed;}

main{padding:.8rem;display:grid;gap:.8rem;position:relative;z-index:1;padding-bottom:28px;}

/* TABS */
.tabs{display:flex;border-bottom:1px solid var(--b1);margin-bottom:.8rem;}
.tab{padding:.4rem 1rem;font-size:9px;font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;color:var(--dm);cursor:pointer;border-bottom:2px solid transparent;}
.tab.active{color:var(--gr);border-bottom-color:var(--gr);}

/* CARDS */
.card{background:var(--s1);border:1px solid var(--b1);border-radius:2px;overflow:hidden;}
.ch{display:flex;align-items:center;justify-content:space-between;
  padding:.35rem .7rem;background:var(--s2);border-bottom:1px solid var(--b1);}
.ch-l{font-size:8px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--dm);}
.ch-r{font-size:9px;color:var(--dm);}

/* STAT */
.stat{padding:.7rem;}
.sv{font-size:20px;font-weight:800;font-family:var(--sans);line-height:1.1;}
.sl{font-size:8px;color:var(--dm);margin-top:.25rem;letter-spacing:.06em;}

/* TABLE */
table{width:100%;border-collapse:collapse;}
th{font-size:8px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dm);padding:.3rem .55rem;border-bottom:1px solid var(--b1);
  background:var(--s2);text-align:left;white-space:nowrap;}
td{padding:.4rem .55rem;border-bottom:1px solid var(--dm2);font-size:10px;vertical-align:middle;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:#0a1830;cursor:pointer;}

/* SCORE */
.sbar{display:flex;gap:1.5px;}
.sd{width:6px;height:6px;border-radius:1px;background:var(--dm2);}
.sd.on{background:var(--gr);}

/* SIGNALS */
.sig-strong-buy{color:var(--gr);font-weight:700;}
.sig-buy{color:var(--cy);}
.sig-watch{color:var(--go);}
.sig-avoid{color:var(--dm);}
.ap{font-size:8px;font-weight:700;padding:.12rem .3rem;border-radius:1px;letter-spacing:.08em;white-space:nowrap;}
.ab{background:#00ff8812;color:var(--gr);border:1px solid #00ff8835;}
.as{background:#ff335512;color:var(--rd);border:1px solid #ff335535;}
.ah{background:transparent;color:var(--dm);border:1px solid var(--b1);}

/* BREAKDOWN */
.bd-wrap{padding:.6rem .7rem;max-height:360px;overflow-y:auto;}
.eng-blk{margin-bottom:.8rem;}
.eng-hd{display:flex;justify-content:space-between;font-size:8px;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:.3rem;
  padding-bottom:.2rem;border-bottom:1px solid var(--b1);}
.eng-bar{height:2px;background:var(--b1);border-radius:1px;margin-bottom:.35rem;}
.eng-fill{height:100%;border-radius:1px;transition:.3s;}
.fl{font-size:9.5px;line-height:1.75;color:var(--dm);}
.fl.ok{color:var(--gr);} .fl.wn{color:var(--go);} .fl.no{color:var(--rd);}

/* GRID */
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
.g2r{display:grid;grid-template-columns:2fr 1fr;gap:.8rem;}
.gm{display:grid;grid-template-columns:1.2fr 1fr 1.6fr;gap:.8rem;}

/* GAP ALERT */
.gap-item{display:flex;justify-content:space-between;align-items:center;
  padding:.4rem .7rem;border-bottom:1px solid var(--dm2);font-size:10px;}
.gap-item:last-child{border-bottom:none;}
.gap-pct{font-weight:700;color:var(--gr);}

/* KITE LOG */
.kl-item{display:flex;gap:.6rem;padding:.3rem .7rem;border-bottom:1px solid var(--dm2);font-size:9px;}
.kl-time{color:var(--dm);min-width:52px;}
.kl-method{color:var(--cy);font-weight:700;min-width:160px;}
.kl-detail{color:var(--dm);}

/* TICKER */
#ticker{position:fixed;bottom:0;left:0;right:0;height:24px;background:var(--s1);
  border-top:1px solid var(--b1);overflow:hidden;display:flex;align-items:center;z-index:100;}
.tk-i{display:flex;gap:2rem;animation:tk 50s linear infinite;white-space:nowrap;padding-left:100%;}
@keyframes tk{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.tk-sym{color:var(--dm);margin-right:.3rem;font-size:9px;}
.tk-val{font-size:9px;}

#toast{position:fixed;top:42px;right:.8rem;z-index:200;background:var(--gr);color:#000;
  padding:.3rem .7rem;font-size:9px;font-weight:700;letter-spacing:.1em;border-radius:1px;display:none;}

.no-d{padding:.8rem;text-align:center;color:var(--dm);font-size:9px;}
.pulse{animation:pu 1.2s ease-in-out infinite;}
@keyframes pu{0%,100%{opacity:1}50%{opacity:.25}}

::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--b2);}
</style>
</head>
<body>

<div id="bar">
  <div class="br-logo">Inst<em>Trade</em><span style="color:var(--dm);margin:0 .3rem">|</span>NSE</div>
  <div class="br-pills">
    <div class="pill"><span>REGIME</span>&nbsp;<strong id="p-regime">—</strong></div>
    <div class="pill"><span>VIX</span>&nbsp;<strong id="p-vix">—</strong></div>
    <div class="pill"><span>FII</span>&nbsp;<strong id="p-fii">—</strong></div>
    <div class="pill"><span>PCR</span>&nbsp;<strong id="p-pcr">—</strong></div>
    <div class="pill"><span>SCANNED</span>&nbsp;<strong id="p-uni" class="b">—</strong></div>
    <div class="pill"><span>GAP-UPS</span>&nbsp;<strong id="p-gaps" class="go">—</strong></div>
    <div class="pill"><span>POSITIONS</span>&nbsp;<strong id="p-pos" class="g">—</strong></div>
    <div class="pill"><span>PNL</span>&nbsp;<strong id="p-pnl">—</strong></div>
  </div>
  <div class="br-right">
    <span id="p-time" style="font-size:8px;color:var(--dm)">—</span>
    <span id="m-badge" class="badge p">PAPER</span>
    <button class="btn" id="scan-btn" onclick="triggerScan()">▶ FULL SCAN</button>
  </div>
</div>

<main>

<!-- STATS -->
<div class="g4">
  <div class="card"><div class="ch"><span class="ch-l">NSE Universe</span></div>
    <div class="stat"><div class="sv b" id="s-uni">—</div><div class="sl">Stocks scanned this run</div></div></div>
  <div class="card"><div class="ch"><span class="ch-l">Gap-Up Alerts</span></div>
    <div class="stat"><div class="sv go" id="s-gaps">—</div><div class="sl">≥2% gap stocks today</div></div></div>
  <div class="card"><div class="ch"><span class="ch-l">Session PnL</span></div>
    <div class="stat"><div class="sv" id="s-pnl">₹0</div><div class="sl" id="s-wr">—</div></div></div>
  <div class="card"><div class="ch"><span class="ch-l">Open Positions</span></div>
    <div class="stat"><div class="sv g" id="s-pos">0</div><div class="sl">of 6 max</div></div></div>
</div>

<!-- MACRO ROW -->
<div class="gm">
  <div class="card">
    <div class="ch"><span class="ch-l">Market Regime</span></div>
    <div style="padding:.7rem">
      <div style="font-size:18px;font-weight:800;font-family:var(--sans)" id="r-label">—</div>
      <div id="r-flags" style="margin-top:.4rem"></div>
    </div>
  </div>
  <div class="card">
    <div class="ch"><span class="ch-l">Institutional Flow</span></div>
    <div style="padding:.5rem">
      <div style="display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid var(--b1)">
        <span style="color:var(--dm);font-size:9px">FII NET</span><strong id="f-fii">—</strong></div>
      <div style="display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid var(--b1)">
        <span style="color:var(--dm);font-size:9px">DII NET</span><strong id="f-dii">—</strong></div>
      <div style="display:flex;justify-content:space-between;padding:.3rem 0">
        <span style="color:var(--dm);font-size:9px">NIFTY PCR</span><strong id="f-pcr">—</strong></div>
    </div>
  </div>
  <div class="card">
    <div class="ch"><span class="ch-l">Scoring Engine — 22 Points Total</span></div>
    <div style="padding:.5rem .7rem;display:flex;gap:.5rem;flex-wrap:wrap;align-items:center">
      <span style="background:var(--bl);color:#000;padding:.15rem .4rem;font-size:8px;font-weight:700;border-radius:1px">TECH 5</span>
      <span style="background:var(--cy);color:#000;padding:.15rem .4rem;font-size:8px;font-weight:700;border-radius:1px">BREAKOUT 5</span>
      <span style="background:var(--pu);color:#000;padding:.15rem .4rem;font-size:8px;font-weight:700;border-radius:1px">FUNDAMENTAL 5</span>
      <span style="background:var(--go);color:#000;padding:.15rem .4rem;font-size:8px;font-weight:700;border-radius:1px">INSTITUTIONAL 5</span>
      <span style="background:var(--gr);color:#000;padding:.15rem .4rem;font-size:8px;font-weight:700;border-radius:1px">REGIME 2</span>
    </div>
    <div style="padding:.2rem .7rem .5rem;display:flex;gap:1rem">
      <span style="font-size:9px;color:var(--gr);font-weight:700">16+ STRONG BUY</span>
      <span style="font-size:9px;color:var(--cy)">11+ BUY</span>
      <span style="font-size:9px;color:var(--go)">7+ WATCH</span>
      <span style="font-size:9px;color:var(--dm)">&lt;7 AVOID</span>
    </div>
  </div>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" onclick="showTab('signals')">SIGNALS</div>
  <div class="tab" onclick="showTab('gaps')">GAP ALERTS</div>
  <div class="tab" onclick="showTab('positions')">POSITIONS</div>
  <div class="tab" onclick="showTab('trades')">TRADE LOG</div>
  <div class="tab" onclick="showTab('kite')">KITE API LOG</div>
</div>

<!-- SIGNALS TAB -->
<div id="tab-signals">
<div class="g2r">
  <div class="card">
    <div class="ch"><span class="ch-l">Signal Scanner</span><span class="ch-r" id="sig-ct">—</span></div>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>SYMBOL</th><th>LTP</th><th>GAP%</th>
        <th>TECH</th><th>BRK</th><th>FUND</th><th>INST</th>
        <th>TOTAL/22</th><th>SIGNAL</th><th>SL / TP</th><th>ACTION</th>
      </tr></thead>
      <tbody id="sig-body"><tr><td colspan="11" class="no-d">Run scan to load all NSE signals</td></tr></tbody>
    </table>
    </div>
  </div>
  <div class="card">
    <div class="ch"><span class="ch-l">Analysis Breakdown</span><span class="ch-r" id="bd-sym">click row</span></div>
    <div class="bd-wrap" id="bd-wrap"><div class="no-d">Click any stock row</div></div>
  </div>
</div>
</div>

<!-- GAP ALERTS TAB -->
<div id="tab-gaps" style="display:none">
<div class="card">
  <div class="ch"><span class="ch-l">Gap-Up Alerts — Full NSE Scan</span><span class="ch-r" id="gap-ct">—</span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>GAP %</th><th>LTP</th><th>VOLUME</th><th>SIGNIFICANCE</th></tr></thead>
    <tbody id="gap-body"><tr><td colspan="5" class="no-d">—</td></tr></tbody>
  </table>
</div>
</div>

<!-- POSITIONS TAB -->
<div id="tab-positions" style="display:none">
<div class="card">
  <div class="ch"><span class="ch-l">Open Positions</span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>ENTRY</th><th>CURR SL</th><th>TARGET</th><th>QTY</th><th>SCORE</th><th>SL TYPE</th></tr></thead>
    <tbody id="pos-body"><tr><td colspan="7" class="no-d">No open positions</td></tr></tbody>
  </table>
</div>
</div>

<!-- TRADE LOG TAB -->
<div id="tab-trades" style="display:none">
<div class="card">
  <div class="ch"><span class="ch-l">Trade Log</span></div>
  <table>
    <thead><tr><th>SYMBOL</th><th>ENTRY</th><th>EXIT</th><th>QTY</th><th>PNL</th><th>REASON</th><th>TIME</th></tr></thead>
    <tbody id="trade-body"><tr><td colspan="7" class="no-d">No trades yet</td></tr></tbody>
  </table>
</div>
</div>

<!-- KITE API LOG TAB -->
<div id="tab-kite" style="display:none">
<div class="card">
  <div class="ch"><span class="ch-l">Kite API Calls — Live Log</span><span class="ch-r">Every method call logged here</span></div>
  <div id="kite-log"></div>
</div>
</div>

</main>

<div id="ticker"><div class="tk-i" id="tk-i">—</div></div>
<div id="toast">SCAN STARTED</div>

<script>
let allSigs = []; let selSym = null;
const TABS = ['signals','gaps','positions','trades','kite'];

function showTab(t) {
  TABS.forEach(x => {
    document.getElementById('tab-'+x).style.display = x===t ? '' : 'none';
    document.querySelectorAll('.tab').forEach((el,i) => {
      if(TABS[i]===t) el.classList.add('active'); else el.classList.remove('active');
    });
  });
}

async function fetchState() {
  try {
    const d = await (await fetch('/api/state')).json();
    render(d);
  } catch(e) { console.error(e); }
}

function render(d) {
  allSigs = d.signals || [];
  const s = d.stats;

  // Badge
  const b = document.getElementById('m-badge');
  b.textContent = d.paper_trade ? 'PAPER' : 'LIVE';
  b.className = 'badge ' + (d.paper_trade ? 'p' : 'l');

  // Topbar pills
  const reg = d.regime || {};
  document.getElementById('p-regime').textContent = reg.label || '—';
  document.getElementById('p-regime').className = (reg.label||'').includes('BULL') ? 'g' : 'r';
  const vix = d.vix || 15;
  document.getElementById('p-vix').textContent = vix.toFixed(1);
  document.getElementById('p-vix').className = vix < 15 ? 'g' : vix < 20 ? 'go' : 'r';
  const fn = d.fii_dii?.fii_net || 0;
  document.getElementById('p-fii').textContent = (fn>=0?'+':'') + (fn/1e7).toFixed(0) + 'Cr';
  document.getElementById('p-fii').className = fn >= 0 ? 'g' : 'r';
  document.getElementById('p-pcr').textContent = d.fii_dii?.pcr || '—';
  document.getElementById('p-uni').textContent = s.universe_size.toLocaleString();
  document.getElementById('p-gaps').textContent = s.gap_alerts;
  document.getElementById('p-pos').textContent = s.open_positions;
  const pnl = s.total_pnl;
  document.getElementById('p-pnl').textContent = (pnl>=0?'+':'') + '₹' + Math.abs(pnl).toLocaleString('en-IN');
  document.getElementById('p-pnl').className = pnl >= 0 ? 'g' : 'r';

  // Stats
  document.getElementById('s-uni').textContent = s.universe_size.toLocaleString();
  document.getElementById('s-gaps').textContent = s.gap_alerts;
  const pe = document.getElementById('s-pnl');
  pe.textContent = (pnl>=0?'+':'') + '₹' + Math.abs(pnl).toLocaleString('en-IN');
  pe.className = 'sv ' + (pnl>=0?'g':'r');
  document.getElementById('s-wr').textContent = `Win ${s.win_rate}% | ${s.total_trades} trades`;
  document.getElementById('s-pos').textContent = s.open_positions;

  // Regime
  document.getElementById('r-label').textContent = reg.label || '—';
  document.getElementById('r-label').className = (reg.label||'').includes('BULL') ? 'g' : 'r';
  document.getElementById('r-flags').innerHTML = (reg.flags||[]).map(f=>
    `<div style="font-size:9px;line-height:1.8;color:${f.startsWith('✅')?'var(--gr)':'var(--dm)'}">${f}</div>`
  ).join('');

  // FII
  const dn = d.fii_dii?.dii_net || 0;
  document.getElementById('f-fii').textContent = (fn>=0?'+':'') + '₹'+(fn/1e7).toFixed(0)+'Cr';
  document.getElementById('f-fii').className = fn>=0?'g':'r';
  document.getElementById('f-dii').textContent = (dn>=0?'+':'') + '₹'+(dn/1e7).toFixed(0)+'Cr';
  document.getElementById('f-dii').className = dn>=0?'g':'r';
  document.getElementById('f-pcr').textContent = d.fii_dii?.pcr || '—';

  // Signals
  renderSignals(d.signals);
  renderGapAlerts(d.gap_alerts);
  renderPositions(d.positions);
  renderTrades(d.trade_log);
  renderKiteLog(d.kite_calls);
  renderTicker(d.signals, d.gap_alerts);

  if (selSym) {
    const s2 = d.signals.find(x=>x.symbol===selSym);
    if (s2) showBreakdown(s2);
  }
}

function renderSignals(sigs) {
  document.getElementById('sig-ct').textContent = sigs.length + ' stocks scored';
  const tb = document.getElementById('sig-body');
  if (!sigs.length) { tb.innerHTML = '<tr><td colspan="11" class="no-d">Run scan to load signals</td></tr>'; return; }
  tb.innerHTML = sigs.map(s => {
    const bd = s.breakdown || {};
    const ts = bd.technical?.score || 0, bs = bd.breakout?.score || 0;
    const fs = bd.fundamental?.score || 0, is = bd.institutional?.score || 0;
    const ac = s.action.startsWith('BUY') ? 'ab' : s.action.startsWith('SELL') ? 'as' : 'ah';
    const sc = `sig-${s.signal_class||'avoid'}`;
    const gp = s.gap_pct || 0;
    const isSel = s.symbol === selSym ? 'background:#0a1c32' : '';
    return `<tr style="${isSel}" onclick="selectSym('${s.symbol}')">
      <td><strong>${s.symbol}</strong></td>
      <td>₹${s.ltp.toLocaleString('en-IN')}</td>
      <td class="${gp>=2?'go go':gp>0?'':'r'}">${gp>=0?'+':''}${gp.toFixed(1)}%</td>
      <td style="color:var(--bl)">${ts}/5</td>
      <td style="color:var(--cy)">${bs}/5</td>
      <td style="color:var(--pu)">${fs}/5</td>
      <td style="color:var(--go)">${is}/5</td>
      <td>
        <div class="sbar">${Array.from({length:22},(_,i)=>`<div class="sd ${i<s.score?'on':''}"></div>`).join('')}</div>
        <span style="font-size:8px;color:var(--dm)">${s.score}/22</span>
      </td>
      <td><span class="${sc}">${s.signal}</span></td>
      <td style="font-size:9px"><span class="r">${s.sl}</span>/<span class="g">${s.tp}</span></td>
      <td><span class="ap ${ac}">${s.action}</span></td>
    </tr>`;
  }).join('');
}

function selectSym(sym) {
  selSym = sym;
  const s = allSigs.find(x=>x.symbol===sym);
  if(s) showBreakdown(s);
}

function showBreakdown(s) {
  document.getElementById('bd-sym').textContent = s.symbol;
  const bd = s.breakdown || {};
  const engs = [
    {k:'technical',     l:'Technical',     c:'var(--bl)'},
    {k:'breakout',      l:'Breakout',       c:'var(--cy)'},
    {k:'fundamental',   l:'Fundamental',    c:'var(--pu)'},
    {k:'institutional', l:'Institutional',  c:'var(--go)'},
    {k:'regime',        l:'Regime',         c:'var(--gr)'},
  ];
  const fi = s.pe ? `PE:${s.pe} ROE:${s.roe}% MCap:₹${(s.mcap_cr||0).toLocaleString('en-IN')}Cr` : '';
  let html = fi ? `<div style="font-size:8px;color:var(--dm);padding:.25rem 0 .5rem;border-bottom:1px solid var(--b1);margin-bottom:.5rem">${fi}</div>` : '';
  engs.forEach(e => {
    const en = bd[e.k] || {score:0,max:5,flags:[]};
    const pct = (en.score/en.max)*100;
    html += `<div class="eng-blk">
      <div class="eng-hd"><span style="color:${e.c}">${e.l}</span><span style="color:${e.c}">${en.score}/${en.max}</span></div>
      <div class="eng-bar"><div class="eng-fill" style="width:${pct}%;background:${e.c}"></div></div>
      ${(en.flags||[]).map(f=>`<div class="fl ${f.startsWith('✅')?'ok':f.startsWith('❌')?'no':'wn'}">${f}</div>`).join('')}
    </div>`;
  });
  document.getElementById('bd-wrap').innerHTML = html;
}

function renderGapAlerts(gaps) {
  document.getElementById('gap-ct').textContent = (gaps||[]).length + ' gap-ups today';
  const tb = document.getElementById('gap-body');
  if (!gaps?.length) { tb.innerHTML = '<tr><td colspan="5" class="no-d">No gap-ups detected yet</td></tr>'; return; }
  tb.innerHTML = gaps.map(g => {
    const sig = g.gap_pct >= 10 ? '🔥 EXPLOSIVE' : g.gap_pct >= 5 ? '🚀 STRONG' : '📈 MODERATE';
    return `<tr>
      <td><strong>${g.symbol}</strong></td>
      <td class="gap-pct">+${g.gap_pct}%</td>
      <td>₹${g.ltp.toLocaleString('en-IN')}</td>
      <td>${(g.volume||0).toLocaleString('en-IN')}</td>
      <td style="font-size:9px">${sig}</td>
    </tr>`;
  }).join('');
}

function renderPositions(pos) {
  const tb = document.getElementById('pos-body');
  const e = Object.entries(pos);
  if (!e.length) { tb.innerHTML = '<tr><td colspan="7" class="no-d">No open positions</td></tr>'; return; }
  tb.innerHTML = e.map(([sym,p]) => `<tr>
    <td><strong>${sym}</strong></td>
    <td>₹${p.entry.toFixed(2)}</td>
    <td class="r">₹${p.sl.toFixed(2)}</td>
    <td class="g">₹${p.tp.toFixed(2)}</td>
    <td>${p.qty}</td>
    <td>${p.score}/22</td>
    <td style="font-size:8px;color:${p.trailing?'var(--go)':'var(--dm)'}">${p.trailing?'TRAILING':'FIXED'}</td>
  </tr>`).join('');
}

function renderTrades(tl) {
  const tb = document.getElementById('trade-body');
  const t2 = [...(tl||[])].reverse();
  if (!t2.length) { tb.innerHTML = '<tr><td colspan="7" class="no-d">No trades</td></tr>'; return; }
  tb.innerHTML = t2.map(t => `<tr>
    <td><strong>${t.symbol}</strong></td>
    <td>₹${t.entry.toFixed(2)}</td>
    <td>₹${t.exit.toFixed(2)}</td>
    <td>${t.qty}</td>
    <td class="${t.pnl>=0?'g':'r'}">₹${Math.abs(t.pnl).toLocaleString('en-IN')}</td>
    <td style="font-size:8px;color:var(--dm)">${t.reason}</td>
    <td style="font-size:8px;color:var(--dm)">${new Date(t.date).toLocaleTimeString('en-IN',{hour12:false})}</td>
  </tr>`).join('');
}

function renderKiteLog(calls) {
  const el = document.getElementById('kite-log');
  if (!calls?.length) { el.innerHTML = '<div class="no-d">No Kite API calls logged yet</div>'; return; }
  el.innerHTML = [...calls].reverse().map(c => `<div class="kl-item">
    <span class="kl-time">${c.time}</span>
    <span class="kl-method">kite.${c.method}</span>
    <span class="kl-detail">${c.detail}</span>
  </div>`).join('');
}

function renderTicker(sigs, gaps) {
  const items = [];
  (gaps||[]).slice(0,10).forEach(g =>
    items.push(`<span class="tk-sym">${g.symbol}</span><span class="tk-val go">+${g.gap_pct}% GAP</span>`)
  );
  (sigs||[]).slice(0,20).forEach(s =>
    items.push(`<span class="tk-sym">${s.symbol}</span><span class="tk-val sig-${s.signal_class}">${s.signal} ${s.score}/22</span>`)
  );
  if (!items.length) return;
  const full = [...items,...items].join('&nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp;');
  document.getElementById('tk-i').innerHTML = full;
}

async function triggerScan() {
  const btn = document.getElementById('scan-btn');
  btn.disabled = true; btn.textContent = '⟳ SCANNING';
  btn.classList.add('pulse');
  document.getElementById('toast').style.display = 'block';
  setTimeout(() => document.getElementById('toast').style.display = 'none', 2500);
  await fetch('/api/scan', {method:'POST'});
  const poll = setInterval(async () => {
    try {
      const d = await (await fetch('/api/state')).json();
      if (!d.scanning) {
        clearInterval(poll); render(d);
        btn.disabled = false; btn.textContent = '▶ FULL SCAN';
        btn.classList.remove('pulse');
      }
    } catch(e) { clearInterval(poll); }
  }, 2500);
}

fetchState();
setInterval(fetchState, 30000);
setInterval(() => {
  const t = document.getElementById('p-time');
  if(t) t.textContent = new Date().toLocaleTimeString('en-IN',{hour12:false});
}, 1000);
</script>
</body>
</html>"""


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


if __name__ == "__main__":
    log.info("🚀 Institutional Trader Pro — Full NSE Edition")
    log.info(f"   Mode:    {'PAPER' if CFG['PAPER_TRADE'] else '⚠ LIVE'}")
    log.info(f"   Capital: ₹{CFG['CAPITAL']:,}")
    log.info(f"   Universe: ALL NSE EQ (via kite.instruments)")
    start_scheduler()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
