import datetime
import functools
import io
import logging
import os
import re
import shutil
from typing import Optional

import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Import from your new Gemini analyze script
from analyze import analyze_site, analyze_sites_concurrent, results_to_csv, validate_url

load_dotenv()

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "results_latest.csv")

API_TOKEN = os.getenv("API_TOKEN")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "5000000"))

# CRITICAL UPDATE: Reduced to 2. Vision/Screenshots are heavy!
# 8 workers will crash your laptop. 2 is safe.
MAX_WORKERS = int(os.getenv("ANALYZER_MAX_WORKERS", "2"))

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "analyzer")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "results")
_mongo_client: Optional[MongoClient] = None

def get_mongo_collection():
    """Return Mongo collection if URI is configured, else None."""
    global _mongo_client
    if not MONGODB_URI:
        return None
    try:
        if _mongo_client is None:
            _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = _mongo_client[MONGODB_DB]
        return db[MONGODB_COLLECTION]
    except Exception as exc:
        logger.warning(f"Mongo connection issue: {exc}")
        return None

def save_result_to_mongo(doc: dict):
    col = get_mongo_collection()
    if col is None:
        return
    try:
        # Avoid mutating the original dict that we return to clients
        col.insert_one(dict(doc))
    except PyMongoError as exc:
        logger.warning(f"Mongo insert failed: {exc}")

def save_many_to_mongo(docs: list):
    col = get_mongo_collection()
    if col is None or not docs:
        return
    try:
        col.insert_many([dict(d) for d in docs], ordered=False)
    except PyMongoError as exc:
        logger.warning(f"Mongo bulk insert failed: {exc}")

def require_auth(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if API_TOKEN:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer ") or auth_header.split(" ", 1)[1] != API_TOKEN:
                return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

def _parse_render_js(payload: dict, form=None) -> bool:
    # UPDATED: Defaults to TRUE now. 
    # We want Vision/Screenshots to be the standard behavior.
    val = None
    if payload: val = payload.get("render_js")
    if not val and form: val = form.get("render_js")
    
    # Only return False if user specifically says "false" or "0"
    if str(val).lower() in ("0", "false", "no"):
        return False
    return True

@app.route("/analyze", methods=["POST"])
@require_auth
def analyze_endpoint():
    if request.content_length and request.content_length > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Payload too large"}), 413

    payload = request.get_json(silent=True) or {}
    url = payload.get("url")
    company = payload.get("company")
    fullname = payload.get("fullname")
    email = payload.get("email")
    
    if not url: return jsonify({"error": "`url` is required"}), 400
    try:
        validate_url(url)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Defaults to True (Vision Mode)
    render_js = _parse_render_js(payload)

    try:
        result = analyze_site(url, company, fullname, email, render_js=render_js)
    except Exception as e:
        logger.error(f"Analysis Failed: {e}")
        return jsonify({"error": "Analysis failed internally"}), 500

    # Fixed deprecated UTC timestamp
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    out_path = os.path.join(RESULTS_DIR, f"results_{ts}.csv")
    results_to_csv([result], out_path)
    shutil.copyfile(out_path, RESULTS_PATH)

    # Persist to MongoDB if configured
    save_result_to_mongo(result)

    # Remove Mongo _id if inserted, to keep response JSON-serializable
    result.pop("_id", None)
    
    return jsonify({"message": "Analysis complete", "csv_path": RESULTS_PATH, "result": result})

@app.route("/analyze_csv", methods=["POST"])
@require_auth
def analyze_csv_endpoint():
    if request.content_length and request.content_length > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Upload too large"}), 413

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    try:
        df = pd.read_csv(io.BytesIO(f.read()))
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    # Normalize headers
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    col_map = {norm(c): c for c in df.columns}

    def pick(row, keys):
        for k in keys:
            nk = norm(k)
            if nk in col_map:
                val = row.get(col_map[nk])
                if pd.isna(val):
                    continue
                return str(val)
        return None

    sites = []
    for _, row in df.iterrows():
        raw_url = pick(row, ["url", "website", "website_url"])
        if not raw_url:
            continue
        try:
            validate_url(str(raw_url))
        except:
            continue

        sites.append({
            "url": str(raw_url),
            "company": pick(row, ["company", "company_name", "business", "business_name"]),
            "fullname": pick(row, ["fullname", "full_name", "full name", "name", "contact", "contact_name"]),
            "email": pick(row, ["email", "e-mail", "email_address", "contact_email"]),
        })

    if not sites:
        return jsonify({"error": "No valid URLs found in CSV"}), 400

    render_js = _parse_render_js(payload={}, form=request.form)

    # Incremental save to avoid losing partial results; process in chunks of 5
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RESULTS_DIR, f"results_{ts}.csv")
    results = []
    chunk_size = 5
    for start in range(0, len(sites), chunk_size):
        chunk = sites[start:start+chunk_size]
        chunk_results = analyze_sites_concurrent(chunk, max_workers=MAX_WORKERS, render_js=render_js)
        results.extend(chunk_results)
        results_to_csv(results, out_path)
        shutil.copyfile(out_path, RESULTS_PATH)
        save_many_to_mongo(chunk_results)
    preview = results[:5]
    return jsonify({"message": "Batch complete", "count": len(results), "csv_path": RESULTS_PATH, "results_preview": preview})

@app.route("/results.csv", methods=["GET"])
@require_auth
def get_csv():
    if os.path.exists(RESULTS_PATH):
        return send_file(RESULTS_PATH, as_attachment=True)
    return jsonify({"error": "No results file"}), 404

@app.route("/results_list", methods=["GET"])
@require_auth
def results_list():
    files = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith('results_') and fname.endswith('.csv'):
            files.append(fname)
    files.sort(reverse=True)
    return jsonify(files)

@app.route('/results/<path:filename>', methods=['GET'])
@require_auth
def serve_result_file(filename):
    if not filename.startswith('results_') or not filename.endswith('.csv'):
        return abort(404)
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return abort(404)
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

@app.route('/results/<path:filename>.json', methods=['GET'])
@require_auth
def serve_result_file_json(filename):
    fname = f"{filename}.csv"
    if not fname.startswith('results_'):
        return abort(404)
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return abort(404)
    try:
        df = pd.read_csv(path)
        return jsonify(df.where(pd.notnull(df), None).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 500

@app.route("/results.json", methods=["GET"])
@require_auth
def get_results_json():
    if not os.path.exists(RESULTS_PATH):
        return jsonify({"error": "No results file"}), 404
    try:
        df = pd.read_csv(RESULTS_PATH)
        return jsonify(df.where(pd.notnull(df), None).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": f"Failed to read results: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=5000)
