"""
get_token.py — Run this locally every morning before market open.

Your Zerodha redirect URL is: http://127.0.0.1:8000/callback
This script:
  1. Opens the Zerodha login page in your browser
  2. Starts a local server on port 8000 to auto-capture the callback
  3. Generates the access token
  4. Prints it — paste it into Railway as KITE_ACCESS_TOKEN

Usage:
    python get_token.py

Requires:
    pip install kiteconnect
"""

import os
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect

# ── Your credentials ──────────────────────────────────────────
API_KEY    = os.environ.get("KITE_API_KEY",    "zv1aztywqml5bm57")
API_SECRET = os.environ.get("KITE_API_SECRET", "") or input("Enter your API Secret: ").strip()

kite = KiteConnect(api_key=API_KEY)

# ── Step 1: Open login URL in browser ─────────────────────────
login_url = kite.login_url()
print(f"\n🔗 Opening Zerodha login in your browser...")
print(f"   URL: {login_url}\n")
webbrowser.open(login_url)

# ── Step 2: Wait for callback on port 8000 ────────────────────
request_token_holder = {}

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed   = urlparse(self.path)
        params   = parse_qs(parsed.query)
        token    = params.get("request_token", [None])[0]
        status   = params.get("status",        ["unknown"])[0]

        if token and status == "success":
            request_token_holder["token"] = token
            body = b"<h2>Login successful! You can close this tab.</h2>"
            self.send_response(200)
        else:
            body = f"<h2>Login failed. Status: {status}</h2>".encode()
            self.send_response(400)

        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass   # silence HTTP logs

print("⏳ Waiting for Zerodha login callback on http://127.0.0.1:8000 ...")
server = HTTPServer(("127.0.0.1", 8000), CallbackHandler)

# Serve until we get the token (handles exactly one request)
while "token" not in request_token_holder:
    server.handle_request()

server.server_close()
request_token = request_token_holder["token"]
print(f"✅ request_token captured: {request_token}\n")

# ── Step 3: Generate access token ─────────────────────────────
try:
    session      = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = session["access_token"]
except Exception as e:
    print(f"❌ Failed to generate session: {e}")
    raise SystemExit(1)

# ── Step 4: Print instructions ────────────────────────────────
print("=" * 60)
print(f"✅  ACCESS TOKEN:\n    {access_token}")
print("=" * 60)
print("\n👉  Go to Railway → your project → Variables tab")
print(f"    Set:  KITE_ACCESS_TOKEN = {access_token}")
print("\n⚠️   Token expires at 6:00 AM tomorrow — run this again daily.")
print("=" * 60)
