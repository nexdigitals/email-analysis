# Email Analysis - Website Auditor

This workspace contains a Python backend that analyzes websites for common problems (SEO, lead capture, tracking, chat) and writes CSV results, plus a minimal React frontend scaffold to call the API.

Backend:
- `backend/analyze.py` - analysis heuristics with URL validation, retries, optional JS rendering (Playwright)
- `backend/app.py` - Flask API (`/analyze`, `/analyze_csv`, results download/list) with optional bearer auth
- `backend/requirements.txt` - Python dependencies

Frontend:
- `frontend/` - Vite + React app (`frontend/src/App.jsx` UI)

Run the backend (PowerShell):

```powershell
cd 'C:\Users\User\Desktop\email analysis\backend'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: JS rendering
pip install playwright
python -m playwright install chromium
# Optional: env hardening
$env:API_TOKEN="yourtoken"                   # protects all endpoints
$env:ANALYZER_TIMEOUT_SEC="12"               # requests timeout
$env:ANALYZER_PLAYWRIGHT_TIMEOUT_SEC="20"    # playwright timeout
$env:ANALYZER_MAX_WORKERS="8"                # concurrency for batch
$env:MAX_UPLOAD_BYTES="5000000"              # payload limit (~5MB)
$env:ANALYZER_MAX_CONTENT_BYTES="2000000"    # max fetch size (~2MB)
python app.py
```

Backend endpoints (all honor bearer auth if API_TOKEN is set):
- `POST /analyze` JSON: `{url, company?, fullname?, render_js?}` -> single result JSON + CSV saved under `backend/results/`.
- `POST /analyze_csv` multipart/form-data: `file` (CSV with columns `url,company,fullname`), optional `render_js` -> batch results saved to timestamped CSV, returns count and preview.
- Downloads/listing: `/results.csv` (latest), `/results.json` (latest as JSON), `/results_list` (all timestamped files), `/results/<filename>` (csv), `/results/<filename>.json` (json view of a specific file).

Run the frontend (PowerShell):

```powershell
cd 'C:\Users\User\Desktop\email analysis\frontend'
npm install
npm run dev
```

Open the shown URL (typically http://127.0.0.1:5173). The UI targets the backend at http://127.0.0.1:5000 and supports:
- Single URL analysis with company/fullname metadata and optional “Render JS (slow)”
- CSV batch upload
- Downloading latest or specific results; filtering/sorting results table in the UI
