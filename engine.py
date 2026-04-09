# ==============================================================================
# CRITICAL FIXES FOR engine.py
# ==============================================================================
# Apply these changes to your engine.py file
# This file shows the FIXED CODE for each critical section
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# FIX #1: Add NFO_NAME_MAP to CFG dictionary
# Location: After line 96 in CFG section
# ──────────────────────────────────────────────────────────────────────────────

# Add this to your CFG dictionary:
CFG_NFO_NAME_MAP = {
    # NFO underlying name mapping
    # Kite NFO uses different names than our index references
    "NFO_NAME_MAP": {
        "NIFTY":     "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY":  "FINNIFTY",
        "MIDCAP":    "MIDCPNIFTY",     # ← CRITICAL: NFO uses MIDCPNIFTY, not MIDCAP
        "SENSEX":    "SENSEX",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# FIX #2: Update MIDCAP_TOKEN
# Location: Line 116 in CFG
# ──────────────────────────────────────────────────────────────────────────────

# BEFORE:
# "MIDCAP_TOKEN": 288009,   # NIFTY MIDCAP 150

# AFTER:
CFG_MIDCAP_TOKEN = {
    "MIDCAP_TOKEN": 288265,   # NIFTY MIDCAP SELECT (the one with F&O trading)
    # Note: NIFTY MIDCAP 150 (token 288009) has NO options - don't use it for F&O
}

# ──────────────────────────────────────────────────────────────────────────────
# FIX #3: Update INDEX_TOKENS for LiveIndexRefresher
# Location: Around line 3688 in LiveIndexRefresher class
# ──────────────────────────────────────────────────────────────────────────────

# BEFORE:
# INDEX_TOKENS = {
#     "NIFTY":     "NSE:NIFTY 50",
#     "BANKNIFTY": "NSE:NIFTY BANK",
#     "FINNIFTY":  "NSE:NIFTY FIN SERVICE",
#     "MIDCAP":    "NSE:NIFTY MIDCAP 150",  # ← WRONG
# }

# AFTER:
INDEX_TOKENS_FIXED = {
    "NIFTY":     "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY":  "NSE:NIFTY FIN SERVICE",
    "MIDCAP":    "NSE:NIFTY MIDCAP SELECT",  # ← FIXED: Use SELECT for F&O
}

# ──────────────────────────────────────────────────────────────────────────────
# FIX #4: Fixed get_nfo_instruments() method
# Location: Around line 432
# ──────────────────────────────────────────────────────────────────────────────

def get_nfo_instruments_FIXED(self, underlying: str):
    """
    KITE API: kite.instruments("NFO")
    Returns DataFrame of NFO options for `underlying` (e.g. "NIFTY").
    CRITICAL FIX: Maps user-friendly names to actual NFO names.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Map our index names to Kite's NFO names
    nfo_name_map = CFG.get("NFO_NAME_MAP", {
        "NIFTY": "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY",
        "MIDCAP": "MIDCPNIFTY",  # Critical mapping
        "SENSEX": "SENSEX",
    })
    nfo_name = nfo_name_map.get(underlying, underlying)
    
    # Check cache
    if KiteLayer._nfo_cache_date != today or underlying not in KiteLayer._nfo_cache:
        try:
            log.info(f"  [Kite] instruments('NFO') → loading for {underlying} (NFO name: {nfo_name})...")
            raw = self._k.instruments("NFO")
            df  = pd.DataFrame(raw)
            
            # ── CRITICAL FIX: Use NFO name, not user-friendly name ──
            df  = df[
                (df["name"] == nfo_name) &  # ← FIXED: Use mapped NFO name
                (df["instrument_type"].isin(["CE", "PE"]))
            ].copy()
            
            if df.empty:
                log.error(f"  [Kite] ❌ NO NFO OPTIONS FOUND for {underlying} (tried NFO name: {nfo_name})")
                log.error(f"        Available NFO names: {sorted(pd.DataFrame(raw)['name'].unique()[:20])}")
                return pd.DataFrame()
            
            # Convert types
            if "strike" in df.columns:
                df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
            
            # Cache it
            KiteLayer._nfo_cache[underlying] = df
            KiteLayer._nfo_cache_date = today
            
            log.info(f"  [Kite] ✅ Loaded {len(df)} {underlying} options (NFO name: {nfo_name})")
            
        except Exception as e:
            log.error(f"  [Kite] NFO instruments failed: {e}")
            return pd.DataFrame()
    
    return KiteLayer._nfo_cache.get(underlying, pd.DataFrame())

# ──────────────────────────────────────────────────────────────────────────────
# FIX #5: Add VIX None-safe formatting
# Location: Around line where VIX is formatted (look for VIX format strings)
# ──────────────────────────────────────────────────────────────────────────────

def format_vix_safely(vix_value):
    """
    Safely format VIX value, handling None case.
    This fixes the "unsupported format string passed to NoneType.__format__" error.
    """
    if vix_value is None:
        return "N/A"
    try:
        return f"{vix_value:.2f}"
    except (TypeError, ValueError):
        return "N/A"

# Example usage in regime check:
# BEFORE:
# log.info(f"VIX: {vix:.2f}")  # ← Crashes if vix is None

# AFTER:
# log.info(f"VIX: {format_vix_safely(vix)}")  # ← Safe

# ──────────────────────────────────────────────────────────────────────────────
# FIX #6: Add validation to build_signal() method
# Location: Start of build_signal() method (around line 2540)
# ──────────────────────────────────────────────────────────────────────────────

def build_signal_FIXED(kl, sig: dict, kite_oc, pivot_data: dict = None, atm_override: int = None) -> dict:
    """
    Generate F&O signal from equity signal.
    FIXED: Added validation for underlying and NFO availability.
    """
    # ── CRITICAL VALIDATION: Check underlying is valid ────
    underlying = sig.get("underlying", "")
    if underlying not in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCAP"]:
        log.error(f"  [Option] Invalid underlying: {underlying} - skipping signal")
        return {"error": f"Invalid underlying: {underlying}"}
    
    # ── Validate that NFO instruments are available ───────
    nfo_test = kl.get_nfo_instruments(underlying)
    if nfo_test.empty:
        log.error(f"  [Option] ❌ NFO instruments NOT available for {underlying}")
        log.error(f"        This means Kite doesn't have options data for this index")
        log.error(f"        Check NFO_NAME_MAP in config - likely mismatch")
        return {"error": f"No NFO instruments for {underlying}"}
    
    direction = sig.get("direction", "").upper()
    if direction not in ["CALL", "PUT"]:
        log.error(f"  [Option] Invalid direction: {direction}")
        return {"error": f"Invalid direction: {direction}"}
    
    # Get LTP
    ltp = float(sig.get("ltp", 0))
    if ltp < 100:
        log.error(f"  [Option] Invalid LTP: {ltp}")
        return {"error": f"Invalid LTP: {ltp}"}
    
    # Calculate ATM strike
    step = OptionEngine._STEPS.get(underlying, 50)
    atm  = round(ltp / step) * step
    
    # ── SANITY CHECK: Strike should be near underlying price ──
    strike_diff_pct = abs(atm - ltp) / ltp * 100
    if strike_diff_pct > 10:  # Strike more than 10% away = something's wrong
        log.error(f"  [Option] ❌ STRIKE SANITY CHECK FAILED")
        log.error(f"        Underlying: {underlying}")
        log.error(f"        LTP: ₹{ltp:,.2f}")
        log.error(f"        Calculated ATM: ₹{atm:,.0f}")
        log.error(f"        Difference: {strike_diff_pct:.1f}%")
        log.error(f"        This indicates wrong underlying data or wrong NFO mapping")
        return {"error": f"Strike price anomaly detected: {strike_diff_pct:.1f}% from underlying"}
    
    # Continue with rest of signal building...
    # (rest of your existing code)

# ──────────────────────────────────────────────────────────────────────────────
# FIX #7: Safe regime check with VIX handling
# Location: Wherever regime check happens (look for VIX usage)
# ──────────────────────────────────────────────────────────────────────────────

def regime_check_FIXED(kl, indexes_data: dict) -> dict:
    """
    Multi-index regime check with safe VIX handling.
    FIXED: Handles None VIX without crashing.
    """
    result = {
        "regime_ok": True,
        "vix": None,
        "vix_ok": True,  # Default to True if VIX unavailable
        "details": {}
    }
    
    # Try to get VIX
    try:
        vix_data = kl.get_vix_data()  # Your method to get VIX
        if vix_data and "ltp" in vix_data:
            result["vix"] = float(vix_data["ltp"])
            result["vix_ok"] = result["vix"] < 20  # Your threshold
        else:
            log.warning("[Regime] VIX data unavailable - skipping VIX check")
            result["vix"] = None
            result["vix_ok"] = True  # Don't fail the regime check
    except Exception as e:
        log.warning(f"[Regime] VIX fetch failed: {e} - skipping VIX check")
        result["vix"] = None
        result["vix_ok"] = True
    
    # Log with safe formatting
    vix_str = format_vix_safely(result["vix"])
    log.info(f"[Regime] VIX: {vix_str}, OK: {result['vix_ok']}")
    
    # Check other indexes...
    for idx_name, idx_data in indexes_data.items():
        # Your existing logic here
        pass
    
    return result

# ──────────────────────────────────────────────────────────────────────────────
# FIX #8: Enhanced logging in fetch_option_data
# Location: Around line 2073 in fetch_option_data method
# ──────────────────────────────────────────────────────────────────────────────

def fetch_option_data_FIXED(kl, underlying: str, strike: int, opt_type: str) -> dict:
    """
    Fetch option data with enhanced debugging.
    """
    # Get NFO name mapping
    nfo_name_map = CFG.get("NFO_NAME_MAP", {
        "NIFTY": "NIFTY",
        "BANKNIFTY": "BANKNIFTY", 
        "FINNIFTY": "FINNIFTY",
        "MIDCAP": "MIDCPNIFTY",
        "SENSEX": "SENSEX",
    })
    nfo_name = nfo_name_map.get(underlying, underlying)
    
    # Get NFO instruments
    rows = kl.get_nfo_instruments(underlying)
    
    # Enhanced debug logging
    log.info(f"  [Option] Fetching {underlying} {strike} {opt_type}")
    log.info(f"         NFO name used: {nfo_name}")
    
    if rows.empty:
        log.error(f"  [Option] No NFO instruments found for {underlying}")
        return {"error": "No NFO instruments available"}
    
    # Show available strikes for debugging
    available_strikes = sorted(rows['strike'].unique())
    log.info(f"         Available strikes: {available_strikes[:10]}...")
    log.info(f"         Total instruments: {len(rows)}")
    
    # Filter for specific strike and type
    filtered = rows[
        (rows['strike'] == strike) & 
        (rows['instrument_type'] == opt_type)
    ]
    
    if filtered.empty:
        log.error(f"  [Option] No instrument found for {underlying} {strike} {opt_type}")
        log.error(f"         Tried strike: {strike}")
        log.error(f"         Available: {available_strikes[:20]}")
        return {"error": "Strike not found"}
    
    # Continue with rest of the logic...
    return {}

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY OF CHANGES NEEDED
# ──────────────────────────────────────────────────────────────────────────────

"""
1. Add NFO_NAME_MAP to CFG dictionary
2. Change MIDCAP_TOKEN from 288009 to 288265
3. Update INDEX_TOKENS to use "NSE:NIFTY MIDCAP SELECT"
4. Modify get_nfo_instruments() to use NFO_NAME_MAP
5. Add format_vix_safely() helper function
6. Add validation at start of build_signal()
7. Add strike sanity check in build_signal()
8. Update regime_check to handle None VIX safely
9. Add enhanced logging in fetch_option_data()

Apply these changes carefully, test with paper trading, and monitor logs closely.
"""
