from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = """<!DOCTYPE html>
<html>
<head>
    <title>FreshRoute - DPSRO Optimizer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; text-align: center; }
        .card { max-width: 650px; margin: 40px auto; background: #1e293b; border-radius: 16px; padding: 32px; border: 1px solid #334155; }
        h1 { color: #10b981; }
        .btn { display: inline-block; padding: 12px 24px; margin: 10px; background: #10b981; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; }
        .btn-subtle { background: #3b82f6; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🌾 FreshRoute - DPSRO</h1>
        <h3>Dynamic Perishable Supply-Chain Resilience Optimizer</h3>
        <p>This computational intelligence prototype uses biophysical spoilage kinetics and HiGHS multi-objective linear programming to dynamically route perishable harvests during disruptions.</p>
        <div style="margin-top: 30px;">
            <a class="btn" href="https://share.streamlit.io/deadpool17880/perishable-supply-optimizer/main/dashboard/app.py" target="_blank">🚀 Launch Streamlit Dashboard</a>
            <a class="btn btn-subtle" href="https://github.com/deadpool17880/perishable-supply-optimizer" target="_blank">📦 GitHub Repository</a>
        </div>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode('utf-8'))
