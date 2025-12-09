#!/usr/bin/env python
"""Simple web server for the chatbot interface.

Usage:
  python serve_web.py
  
Then open: http://localhost:8000
"""
import http.server
import socketserver
from pathlib import Path
import os

PORT = 8000
WEB_DIR = Path(__file__).parent / 'web'

os.chdir(WEB_DIR)

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"[WEB] Server running at http://localhost:{PORT}")
    print(f"[WEB] Serving files from: {WEB_DIR}")
    print(f"[WEB] Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[WEB] Server stopped")
