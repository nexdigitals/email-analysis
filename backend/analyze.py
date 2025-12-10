import concurrent.futures
import datetime
import gc  # memory management
import json
import logging
import os
import re
import tempfile
import threading
import time
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import PIL.Image
from google import genai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
PLAYWRIGHT_TIMEOUT = int(os.getenv("ANALYZER_PLAYWRIGHT_TIMEOUT_SEC", "30"))
USER_AGENT = os.getenv("ANALYZER_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
PLAYWRIGHT_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() not in ("0", "false", "no")
# Force single-page at a time to stay within 512MB free tier limits
PLAYWRIGHT_CONCURRENCY = 1
_PLAYWRIGHT_SEMAPHORE = threading.Semaphore(PLAYWRIGHT_CONCURRENCY)
TEXT_SNIPPET_LIMIT = int(os.getenv("ANALYZER_TEXT_SNIPPET_LIMIT", "800"))

# GOOGLE GEMINI SETUP (new google-genai client)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_env_candidates = [m.strip() for m in os.getenv("GEMINI_MODEL_CANDIDATES", "").split(",") if m.strip()]
GEMINI_MODEL_CANDIDATES = []
for m in (_env_candidates or []) + [
    GEMINI_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]:
    if m and m not in GEMINI_MODEL_CANDIDATES:
        GEMINI_MODEL_CANDIDATES.append(m)
_LAST_GOOD_MODEL: Optional[str] = None

if not GEMINI_API_KEY:
    logger.error("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in .env file; Gemini calls will fail.")

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def _strip_code_fences(payload: str) -> str:
    """Remove leading ```json fences or stray backticks to make JSON parseable."""
    if not payload:
        return payload
    # Trim common ```json ... ``` shapes
    fenced = re.match(r"```(?:json)?\s*(.+?)\s*```", payload, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    # Fallback: drop surrounding backticks only
    if payload.startswith("`") and payload.endswith("`"):
        return payload.strip("`").strip()
    return payload.strip()

def _parse_gemini_payload(raw: str) -> Optional[dict]:
    """Try to coerce Gemini output into JSON even if the model returned code fences."""
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        return None

def _looks_garbled(text: str) -> bool:
    """Heuristic: treat as garbled if a large share is non-ASCII or text is mostly symbols."""
    if not text:
        return False
    trimmed = text.strip()
    if len(trimmed) < 50:
        return False
    non_ascii = sum(1 for c in trimmed if ord(c) > 126 or ord(c) < 9)
    symbol_runs = re.findall(r"[^\w\s]{5,}", trimmed)
    if non_ascii / len(trimmed) > 0.25:
        return True
    if any(len(run) > 8 for run in symbol_runs):
        return True
    return False

# -------------------------------------------------------------------------
# BROWSER LAYER (Takes Screenshot + Code)
# -------------------------------------------------------------------------
def fetch_screenshot_and_text(url: str, render_js: bool = True, timeout_sec: Optional[int] = None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    If render_js is False, fall back to a simple requests fetch (faster, no screenshot).
    Otherwise uses Playwright to render and screenshot with bounded concurrency.
    """
    eff_timeout = timeout_sec or PLAYWRIGHT_TIMEOUT
    if not render_js:
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=eff_timeout)
            resp.raise_for_status()
            html = resp.text
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            return None, text, html, None
        except Exception as exc:
            msg = f"HTTP fetch error: {exc}"
            logger.error(msg)
            return None, None, None, msg

    if not _PLAYWRIGHT_SEMAPHORE.acquire(timeout=eff_timeout):
        logger.error("Playwright concurrency limit reached; skipping render.")
        return None, None, None, "Playwright concurrency limit reached"

    browser = None
    context = None
    page = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=PLAYWRIGHT_HEADLESS,
                args=[
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-extensions",
                ],
            )
            context = browser.new_context(
                user_agent=USER_AGENT,
                viewport={'width': 1200, 'height': 720},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.set_default_timeout(eff_timeout * 1000)

            # Block heavy resources (keep images; drop media/fonts)
            def _route_handler(route):
                rtype = route.request.resource_type
                if rtype in ("media", "font"):
                    route.abort()
                else:
                    route.continue_()

            try:
                page.route("**/*", _route_handler)
            except Exception as route_exc:
                logger.debug(f"Route setup issue (continuing without blocking): {route_exc}")
            
            try:
                page.goto(url, wait_until="domcontentloaded")
                time.sleep(2) # small pause for JS
            except Exception as e:
                logger.warning(f"Navigation warning: {e}")

            # 1. Capture Screenshot
            timestamp = int(time.time() * 1000)
            clean_host = urlparse(url).netloc.replace(":", "_").replace(".", "_")
            screenshot_path = os.path.join(tempfile.gettempdir(), f"temp_{clean_host}_{timestamp}.jpg")
            
            try:
                page.screenshot(path=screenshot_path, full_page=False, quality=60, type='jpeg')
            except Exception:
                screenshot_path = None

            # 2. Get Code
            content = page.content()
            try:
                text = page.inner_text("body")
            except:
                text = ""

            return screenshot_path, text, content, None
            
    except Exception as exc:
        msg = f"Browser Error: {exc}"
        logger.error(msg)
        return None, None, None, msg
    finally:
        try:
            if page:
                page.close()
            if context:
                context.close()
            if browser:
                browser.close()
        except Exception as close_exc:
            logger.debug(f"Playwright close issue: {close_exc}")
        _PLAYWRIGHT_SEMAPHORE.release()
        gc.collect()

# -------------------------------------------------------------------------
# TECH DETECTION
# -------------------------------------------------------------------------
def detect_tech_features(html: str) -> List[str]:
    features = []
    if not html: return features
    html_lower = html.lower()
    soup = BeautifulSoup(html, "html.parser")
    
    tech_map = {
        "Chatbot": ["podium", "birdeye", "intercom", "drift", "tidio", "chat-widget", "crisp", "livechat", "zendesk", "hubspot", "gorgias", "freshdesk"],
        "Modern Framework": ["__next", "react", "vue", "nuxt", "gatsby", "wix", "squarespace", "webflow"],
        "Popup": ["popup", "modal", "overlay", "elementor-popup"],
        "Analytics": ["gtag(", "googletagmanager", "analytics.js", "ga('create'", "meta pixel", "fbevents.js"],
        "Booking": ["calendly", "acuity", "booksy", "booking.js"],
    }

    for category, keywords in tech_map.items():
        for k in keywords:
            if k in html_lower:
                features.append(category)
                break 

    script_srcs = [s.get("src", "").lower() for s in soup.find_all("script") if s.get("src")]
    for src in script_srcs:
        if "gtag/js" in src or "googletagmanager" in src:
            if "Analytics" not in features:
                features.append("Analytics")
        if "calendly" in src and "Booking" not in features:
            features.append("Booking")
        if any(x in src for x in ("chat", "widget", "intercom")) and "Chatbot" not in features:
            features.append("Chatbot")

    return features

# -------------------------------------------------------------------------
# AI VISION ANALYSIS (GEMINI 1.5 FLASH)
# -------------------------------------------------------------------------
def _ai_vision_generate(screenshot_path: Optional[str], text: str, url: str, company_name: str, tech_features: List[str]) -> Optional[dict]:
    global _LAST_GOOD_MODEL
    if not GEMINI_API_KEY:
        return {"problem": "Missing API Key", "offer": "Configure Google API Key"}

    tech_context = ", ".join(tech_features) if tech_features else "None detected"
    text_snippet = (text or "")[:TEXT_SNIPPET_LIMIT]

    # Prepare Image for Gemini
    img = None
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            # FIX FOR WINERROR 32: Open, Load, and Close immediately
            with PIL.Image.open(screenshot_path) as p_img:
                p_img.load() # Force load image data into memory
                img = p_img  # Assign the object (it's safe now)
        except Exception as e:
            logger.warning(f"Failed to load image: {e}")

    # The Prompt
    prompt = (
        "Return ONLY one-line JSON matching this schema with no code fences.\n"
        '{\n'
        '  "visual_observation": "string",\n'
        '  "level": "Level 1/2/3",\n'
        '  "problem": "string",\n'
        '  "offer": "string"\n'
        "}\n"
        f"Site: {company_name} ({url}) | Tech: {tech_context}\n"
        f"Text snippet: {text_snippet}\n"
        "Prefer what you SEE in the screenshot; use text only if unseen. Ignore cookie banners/ads.\n"
        "Checks: chat bubble/popup/button visibility; contact/quote text without form fields = high friction; design Modern/Premium vs Dated/Standard.\n"
        "Honesty: if the experience is already strong, state 'No major issues observed' and offer 'Keep current setup.'\n"
        "Offer rule: the offer must directly solve the stated problem (shorten forms, add chat/concierge, streamline contact), not generic marketing. If forms are long/rigid or manual data entry is required, propose an AI chatbot trained on company data that answers any question and simultaneously captures leads/vital info 24/7."
    )

    try:
        parts = [prompt]
        if img:
            import io

            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            parts = [
                genai.types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                prompt,
            ]

        errors = []
        candidates = []
        if _LAST_GOOD_MODEL and _LAST_GOOD_MODEL not in candidates:
            candidates.append(_LAST_GOOD_MODEL)
        candidates.extend([m for m in GEMINI_MODEL_CANDIDATES if m not in candidates])

        client = genai.Client(api_key=GEMINI_API_KEY)

        for model_name in candidates:
            # basic retry/backoff for 429s
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=parts,
                    )

                    raw = getattr(response, "text", None)
                    if not raw and getattr(response, "candidates", None):
                        for cand in response.candidates:
                            parts_list = getattr(getattr(cand, "content", None), "parts", []) or []
                            texts = [getattr(p, "text", "") for p in parts_list if getattr(p, "text", "")]
                            if texts:
                                raw = "\n".join(texts).strip()
                                break

                    parsed = _parse_gemini_payload(raw or "")
                    if parsed:
                        _LAST_GOOD_MODEL = model_name
                        logger.info(f"Gemini success with model={model_name}")
                        parsed["_model"] = model_name
                        return parsed

                    logger.warning(f"Gemini response unparsable for model {model_name}: {raw!r}")
                    errors.append(f"{model_name}: unparsable response")
                    break
                except Exception as exc:
                    err_text = str(exc)
                    if "RESOURCE_EXHAUSTED" in err_text or "429" in err_text:
                        sleep_for = 2 * (attempt + 1)
                        logger.warning(f"Gemini 429/quota for {model_name}, retry {attempt+1}/3 after {sleep_for}s")
                        time.sleep(sleep_for)
                        continue
                    errors.append(f"{model_name}: {exc}")
                    logger.warning(f"Gemini Vision Error ({model_name}): {exc}")
                    break

        if errors:
            logger.error(f"Gemini failed across models: {'; '.join(errors)}")
        return {
            "problem": "AI analysis unavailable right now.",
            "offer": "Retry with a different Gemini model or later.",
            "_errors": errors,
        }

    except Exception as exc:
        logger.warning(f"Gemini Vision Error (fallback): {exc}")
        return {"problem": "AI analysis encountered an unexpected issue.", "offer": "Retry once the Gemini service is stable.", "_errors": [str(exc)]}

# -------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -------------------------------------------------------------------------
def analyze_site(url: str, company_name: Optional[str] = None, fullname: Optional[str] = None, render_js: bool = True, timeout_sec: Optional[int] = None):
    start_ts = time.time()
    res = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "company_fullname": fullname or "",
        "company_name": company_name or "",
        "website_url": url,
        "problem": "",
        "offer": "",
        "errors": [],
    }

    # 1. FETCH
    screenshot_path, body_text, html, fetch_error = fetch_screenshot_and_text(url, render_js=render_js, timeout_sec=timeout_sec)
    
    if fetch_error:
        res["errors"].append(fetch_error)
    
    if not html:
        res["problem"] = "Digital storefront could not be reached."
        res["offer"] = "Verify hosting stability."
        return res

    if _looks_garbled(body_text):
        msg = "Content returned unreadable (likely bot protection or encoding issue)."
        res["errors"].append(msg)
        res["problem"] = "Website is blocking automated crawlers or returned unreadable content."
        res["offer"] = "Retry with different IP/headers or manual review."
        return res

    # 2. DETECT
    tech_features = detect_tech_features(html)
    
    # 3. RULE-BASED CHECKS
    soup = BeautifulSoup(html, "html.parser")
    problems = []
    if not soup.find("h1"): problems.append("Missing H1")
    if not soup.find("meta", attrs={"name": "description"}): problems.append("Missing Meta Desc")
    res["problems"] = problems # Legacy support

    # 4. ANALYZE (Gemini Vision)
    ai_res = _ai_vision_generate(screenshot_path, body_text, url, company_name or "Company", tech_features)
    
    # FIX FOR WINERROR 32: Cleanup AFTER we are sure the file is closed
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            os.remove(screenshot_path)
        except Exception as e:
            logger.warning(f"Could not delete temp file {screenshot_path}: {e}")

    if ai_res:
        res["problem"] = ai_res.get("problem", "Analysis incomplete.")
        res["offer"] = ai_res.get("offer", "Manual review recommended.")
        if ai_res.get("_errors"):
            res["errors"].extend(ai_res["_errors"])
        if ai_res.get("_model"):
            res["model_used"] = ai_res["_model"]
        logger.info(f"Analyzed {url} | Saw: {ai_res.get('visual_observation')}")
    else:
        # Fallback
        res["problem"] = "Standard forms are passive and often result in lost leads."
        res["offer"] = "Deploy a 24/7 AI Agent to capture leads instantly."

    res["duration_sec"] = round(time.time() - start_ts, 2)
    return res

def results_to_csv(rows, path):
    normalized = []
    for r in rows:
        normalized.append({
            "full_name": r.get("company_fullname", ""),
            "company": r.get("company_name", ""),
            "website_url": r.get("website_url", ""),
            "problem": r.get("problem", ""),
            "offer": r.get("offer", ""),
        })
    df = pd.DataFrame(normalized)
    df.to_csv(path, index=False)

def analyze_sites_concurrent(sites, max_workers: int = 2, render_js: bool = True):
    # Force serial execution to avoid multiple Playwright instances on low-memory plans
    max_workers = 1
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(analyze_site, s.get("url"), s.get("company"), s.get("fullname"), render_js) for s in sites]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                logger.warning(f"Concurrent error: {e}")
    return results

def validate_url(url: str) -> str:
    if not url: raise ValueError("URL is required")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"): raise ValueError("URL must start with http or https")
    return url

if __name__ == "__main__":
    print("Module Loaded. Gemini Vision Engine Active.")
