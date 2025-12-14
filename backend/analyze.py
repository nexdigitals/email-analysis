import concurrent.futures
import datetime
import gc  # memory management
import json
import logging
import os
import random
import re
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import PIL.Image
from google import genai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

# --- NEW: Playwright Stealth Import ---
try:
    from playwright_stealth import stealth_sync
except ImportError:
    stealth_sync = None

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
PLAYWRIGHT_TIMEOUT = int(os.getenv("ANALYZER_PLAYWRIGHT_TIMEOUT_SEC", "30"))
# Note: SLOW_SITE_TIMEOUT is handled dynamically inside the fetch function

USER_AGENT = os.getenv(
    "ANALYZER_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)
PLAYWRIGHT_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() not in ("0", "false", "no")

# Concurrency for Playwright (bounded)
PLAYWRIGHT_CONCURRENCY = int(os.getenv("PLAYWRIGHT_CONCURRENCY", "8"))
_PLAYWRIGHT_SEMAPHORE = threading.Semaphore(PLAYWRIGHT_CONCURRENCY)

# Retry knobs (heavy but bounded)
PLAYWRIGHT_MAX_ATTEMPTS = int(os.getenv("PLAYWRIGHT_MAX_ATTEMPTS", "7"))
PLAYWRIGHT_SOFT_RETRIES = int(os.getenv("PLAYWRIGHT_SOFT_RETRIES", "2"))
PLAYWRIGHT_BACKOFF_BASE_SEC = float(os.getenv("PLAYWRIGHT_BACKOFF_BASE_SEC", "0.9"))
PLAYWRIGHT_BACKOFF_MAX_SEC = float(os.getenv("PLAYWRIGHT_BACKOFF_MAX_SEC", "12"))

# Page “settling”
PLAYWRIGHT_SCROLL_MAX_STEPS = int(os.getenv("PLAYWRIGHT_SCROLL_MAX_STEPS", "14"))
PLAYWRIGHT_SCROLL_PAUSE_SEC = float(os.getenv("PLAYWRIGHT_SCROLL_PAUSE_SEC", "0.35"))

# Screenshot settings
PLAYWRIGHT_SCREENSHOT_QUALITY = int(os.getenv("PLAYWRIGHT_SCREENSHOT_QUALITY", "75"))
PLAYWRIGHT_FULL_PAGE_SCREENSHOT = os.getenv("PLAYWRIGHT_FULL_PAGE_SCREENSHOT", "true").lower() not in ("0", "false", "no")

# Optional proxy
ANALYZER_PROXY_SERVER = os.getenv("ANALYZER_PROXY_SERVER", "").strip()
ANALYZER_PROXY_USERNAME = os.getenv("ANALYZER_PROXY_USERNAME", "").strip()
ANALYZER_PROXY_PASSWORD = os.getenv("ANALYZER_PROXY_PASSWORD", "").strip()

# If your server is slow/JS-heavy, this helps capture more text
TEXT_SNIPPET_LIMIT = int(os.getenv("ANALYZER_TEXT_SNIPPET_LIMIT", "3500"))

# GOOGLE GEMINI SETUP (new google-genai client)
_raw_keys = [k.strip() for k in (os.getenv("GEMINI_API_KEYS") or "").split(",") if k.strip()]
_single_fallback = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not _raw_keys and _single_fallback:
    _raw_keys = [_single_fallback]

GEMINI_KEYS = _raw_keys
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_env_candidates = [m.strip() for m in os.getenv("GEMINI_MODEL_CANDIDATES", "").split(",") if m.strip()]
GEMINI_MODEL_CANDIDATES = []
for m in (_env_candidates or []) + [GEMINI_MODEL, "gemini-2.5-flash", "gemini-2.0-flash"]:
    if m and m not in GEMINI_MODEL_CANDIDATES:
        GEMINI_MODEL_CANDIDATES.append(m)

_LAST_GOOD_MODEL: Optional[str] = None

if not GEMINI_KEYS:
    logger.error("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in .env file; Gemini calls will fail.")

# round-robin API keys with cooldowns
_KEY_STATE = {"idx": 0, "cooldowns": {k: 0 for k in GEMINI_KEYS}}


def _next_key(now: Optional[float] = None) -> Optional[str]:
    if not GEMINI_KEYS:
        return None
    now = now or time.time()
    n = len(GEMINI_KEYS)
    for i in range(n):
        idx = (_KEY_STATE["idx"] + i) % n
        key = GEMINI_KEYS[idx]
        if now >= _KEY_STATE["cooldowns"].get(key, 0):
            _KEY_STATE["idx"] = (idx + 1) % n
            return key
    return None


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def _strip_code_fences(payload: str) -> str:
    if not payload:
        return payload
    fenced = re.match(r"```(?:json)?\s*(.+?)\s*```", payload, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    if payload.startswith("`") and payload.endswith("`"):
        return payload.strip("`").strip()
    return payload.strip()


def _parse_gemini_payload(raw: str) -> Optional[dict]:
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def _looks_garbled(text: str) -> bool:
    if not text:
        return False
    trimmed = text.strip()
    if len(trimmed) < 80:
        return False
    non_ascii = sum(1 for c in trimmed if ord(c) > 126 or ord(c) < 9)
    symbol_runs = re.findall(r"[^\w\s]{5,}", trimmed)
    if non_ascii / max(1, len(trimmed)) > 0.25:
        return True
    if any(len(run) > 10 for run in symbol_runs):
        return True
    return False


def _jittered_backoff(attempt: int) -> float:
    base = PLAYWRIGHT_BACKOFF_BASE_SEC * (2 ** attempt)
    base = min(base, PLAYWRIGHT_BACKOFF_MAX_SEC)
    return base * (0.65 + random.random() * 0.7)


def _is_probably_blocked(text: str, html: str) -> Optional[str]:
    blob = " ".join([(text or ""), (html or "")]).lower()
    patterns = [
        ("captcha_or_verification", r"(captcha|verify you are human|are you human|human verification)"),
        ("access_denied", r"(access denied|forbidden|error\s*403|not authorized)"),
        ("rate_limited", r"(too many requests|error\s*429|rate limit)"),
        ("bot_protection", r"(unusual traffic|automated queries|bot detection|security check|checking your browser)"),
    ]
    for label, pat in patterns:
        if re.search(pat, blob):
            return label
    return None


def _safe_tmp_screenshot_path(url: str) -> str:
    timestamp = int(time.time() * 1000)
    host = urlparse(url).netloc.replace(":", "_").replace(".", "_") or "site"
    return os.path.join(tempfile.gettempdir(), f"temp_{host}_{timestamp}.jpg")


# -------------------------------------------------------------------------
# METRIC-LOCKED VALIDATOR (Anti-Hallucination)
# -------------------------------------------------------------------------
def validate_no_hallucination(ai_output: dict) -> bool:
    text = json.dumps(ai_output).lower()
    forbidden_concepts = ["rankings", "google traffic", "keyword volume", "backlinks", "domain authority", "search volume"]
    for concept in forbidden_concepts:
        if concept in text:
            logger.warning(f"Hallucination Guard: Detected forbidden concept '{concept}'")
            return False
    return True


# -------------------------------------------------------------------------
# STRUCTURED DATA EXTRACTION (Deterministic SEO)
# -------------------------------------------------------------------------
def _extract_structured_signals(html: str, base_url: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {
        "title": "",
        "meta_description": "",
        "h1": [],
        "h2": [],
        "forms": 0,
        "contact_links": [],
        "cta_texts": [],
        "seo_signals": {
            "canonical_present": False,
            "robots_content": "",
            "is_noindex": False,
            "img_missing_alt_count": 0,
            "internal_link_count": 0,
            "meta_desc_len": 0,
        },
    }
    if not html:
        return out

    soup = BeautifulSoup(html, "html.parser")

    if soup.title and soup.title.get_text(strip=True):
        out["title"] = soup.title.get_text(" ", strip=True)[:200]

    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        content = md.get("content") or ""
        out["meta_description"] = content[:300]
        out["seo_signals"]["meta_desc_len"] = len(content)

    out["h1"] = [h.get_text(" ", strip=True)[:150] for h in soup.find_all("h1")[:6]]
    out["h2"] = [h.get_text(" ", strip=True)[:150] for h in soup.find_all("h2")[:8]]
    out["forms"] = len(soup.find_all("form"))

    out["seo_signals"]["canonical_present"] = bool(soup.find("link", rel="canonical"))

    robots = soup.find("meta", attrs={"name": "robots"})
    robots_content = (robots.get("content", "") if robots else "").lower()
    out["seo_signals"]["robots_content"] = robots_content
    out["seo_signals"]["is_noindex"] = "noindex" in robots_content

    out["seo_signals"]["img_missing_alt_count"] = sum(1 for img in soup.find_all("img") if not img.get("alt"))

    try:
        base_domain = urlparse(base_url).netloc
        internal_links = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/") or (urlparse(href).netloc == base_domain):
                internal_links += 1
        out["seo_signals"]["internal_link_count"] = internal_links
    except Exception:
        pass

    contact_candidates = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)
        blob = f"{href} {text}".lower()
        if any(k in blob for k in ["contact", "book", "schedule", "appointment", "demo", "call", "quote"]):
            contact_candidates.append({"text": text[:80], "href": href[:200]})
    out["contact_links"] = contact_candidates[:12]

    ctas = []
    for el in soup.find_all(["button", "a"]):
        t = el.get_text(" ", strip=True)
        if not t:
            continue
        tl = t.lower()
        if any(k in tl for k in ["get started", "book", "schedule", "contact", "request", "demo", "quote", "call"]):
            ctas.append(t[:80])

    seen = set()
    uniq = []
    for c in ctas:
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        uniq.append(c)
    out["cta_texts"] = uniq[:12]

    return out


# -------------------------------------------------------------------------
# BROWSER LAYER (SPA-Safe, Adaptive Timeout)
# -------------------------------------------------------------------------
def fetch_screenshot_and_text(
    url: str,
    render_js: bool = True,
    timeout_sec: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Dict[str, object]]:

    extra_context: Dict[str, object] = {
        "blocked_reason": None,
        "attempts": 0,
        "used_playwright": bool(render_js),
        "final_url": "",
        "http_status": None,
        "structured": {},
        "performance": {},
    }

    # SLOW SITE HANDLING
    SLOW_SITE_TIMEOUT = int(os.getenv("ANALYZER_SLOW_SITE_TIMEOUT_SEC", str(PLAYWRIGHT_TIMEOUT)))
    eff_timeout = timeout_sec or SLOW_SITE_TIMEOUT

    if not render_js:
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=eff_timeout)
            extra_context["http_status"] = getattr(resp, "status_code", None)
            resp.raise_for_status()
            html = resp.text
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            extra_context["structured"] = _extract_structured_signals(html, base_url=url)
            extra_context["final_url"] = resp.url if getattr(resp, "url", None) else url
            extra_context["blocked_reason"] = _is_probably_blocked(text, html)
            return None, text, html, None, extra_context
        except Exception as exc:
            msg = f"HTTP fetch error: {exc}"
            logger.error(msg)
            return None, None, None, msg, extra_context

    if not _PLAYWRIGHT_SEMAPHORE.acquire(timeout=eff_timeout):
        logger.error("Playwright concurrency limit reached; skipping render.")
        return None, None, None, "Playwright concurrency limit reached", extra_context

    browser = None
    context = None
    page = None
    last_err: Optional[str] = None
    screenshot_path: Optional[str] = None

    def _make_proxy():
        if not ANALYZER_PROXY_SERVER:
            return None
        proxy = {"server": ANALYZER_PROXY_SERVER}
        if ANALYZER_PROXY_USERNAME:
            proxy["username"] = ANALYZER_PROXY_USERNAME
        if ANALYZER_PROXY_PASSWORD:
            proxy["password"] = ANALYZER_PROXY_PASSWORD
        return proxy

    try:
        with sync_playwright() as p:
            attempt = 0
            while attempt < PLAYWRIGHT_MAX_ATTEMPTS:
                extra_context["attempts"] = attempt + 1
                adaptive_timeout = int(eff_timeout * (1.0 + 0.35 * attempt))
                adaptive_timeout_ms = adaptive_timeout * 1000

                need_new_browser = (
                    browser is None or context is None or page is None
                    or attempt == 0 or attempt > PLAYWRIGHT_SOFT_RETRIES
                )

                if need_new_browser:
                    try:
                        if page: page.close()
                    except: pass
                    try:
                        if context: context.close()
                    except: pass
                    try:
                        if browser: browser.close()
                    except: pass

                    proxy_cfg = _make_proxy()
                    browser = p.chromium.launch(
                        headless=PLAYWRIGHT_HEADLESS,
                        args=[
                            "--disable-dev-shm-usage", "--no-sandbox", "--disable-setuid-sandbox",
                            "--disable-gpu", "--disable-software-rasterizer", "--disable-extensions",
                        ],
                        proxy=proxy_cfg,
                    )
                    context = browser.new_context(
                        user_agent=USER_AGENT,
                        viewport={"width": 1920, "height": 1080},
                        device_scale_factor=1,
                        locale="en-US",
                        java_script_enabled=True,
                    )
                    page = context.new_page()
                    if stealth_sync:
                        try:
                            stealth_sync(page)
                        except: pass
                    
                    # TIMEOUTS (Navigation + Actions)
                    page.set_default_timeout(adaptive_timeout_ms)
                    page.set_default_navigation_timeout(adaptive_timeout_ms)

                main_status = None
                final_url = url
                
                try:
                    response = None
                    # ADAPTIVE WAIT
                    wait_mode = "load" if adaptive_timeout >= 60 else "domcontentloaded"
                    
                    try:
                        response = page.goto(url, wait_until=wait_mode, timeout=adaptive_timeout_ms)
                    except PlaywrightTimeoutError:
                        try:
                            response = page.goto(url, wait_until="load", timeout=adaptive_timeout_ms)
                        except: pass

                    if response:
                        try: main_status = response.status
                        except: pass
                        try: final_url = response.url or url
                        except: final_url = url

                    extra_context["http_status"] = main_status
                    extra_context["final_url"] = final_url

                    try:
                        page.wait_for_load_state("networkidle", timeout=min(8000, adaptive_timeout_ms))
                    except: pass
                    
                    # Scroll
                    try:
                        for _ in range(PLAYWRIGHT_SCROLL_MAX_STEPS):
                            page.evaluate("() => window.scrollBy(0, Math.max(400, window.innerHeight * 0.85))")
                            time.sleep(PLAYWRIGHT_SCROLL_PAUSE_SEC)
                        if not PLAYWRIGHT_FULL_PAGE_SCREENSHOT:
                            page.evaluate("() => window.scrollTo(0, 0)")
                            time.sleep(0.2)
                    except: pass

                    # Screenshot
                    screenshot_path = _safe_tmp_screenshot_path(final_url)
                    try:
                        page.screenshot(
                            path=screenshot_path,
                            full_page=PLAYWRIGHT_FULL_PAGE_SCREENSHOT,
                            quality=PLAYWRIGHT_SCREENSHOT_QUALITY,
                            type="jpeg",
                        )
                    except: screenshot_path = None

                    # Extract Content
                    try: html = page.content()
                    except: html = ""
                    try: body_text = page.inner_text("body")
                    except: body_text = ""

                    extra_context["structured"] = _extract_structured_signals(html, base_url=final_url)
                    extra_context["blocked_reason"] = _is_probably_blocked(body_text, html)

                    # --- FIX 2 & 3: ACCURATE SPA PERF + DURATION MATH ---
                    try:
                        perf_data = page.evaluate("""() => {
                            const nav = performance.getEntriesByType('navigation')[0] || {};
                            const timing = window.performance.timing;
                            const now = performance.now();
                            const resources = performance.getEntriesByType('resource').length;
                            
                            // Navigation Timing L2 (preferred)
                            const navStart = nav.startTime || 0;
                            let loadTime = nav.loadEventEnd ? (nav.loadEventEnd - navStart) : 0;
                            let domTime = nav.domContentLoadedEventEnd ? (nav.domContentLoadedEventEnd - navStart) : 0;

                            // Fallback to L1 (legacy/some browser contexts)
                            if (loadTime === 0 && timing.loadEventEnd > 0) {
                                loadTime = timing.loadEventEnd - timing.navigationStart;
                            }
                            if (domTime === 0 && timing.domContentLoadedEventEnd > 0) {
                                domTime = timing.domContentLoadedEventEnd - timing.navigationStart;
                            }
                            
                            return {
                                "load_time_ms": loadTime,
                                "dom_content_loaded_ms": domTime,
                                "now_ms": Math.round(now),
                                "resource_count": resources
                            }
                        }""")
                        
                        cleaned_perf = {k: (v if v >= 0 else 0) for k, v in perf_data.items()}
                        extra_context["performance"] = cleaned_perf
                    except Exception as e:
                        logger.warning(f"Performance metric extraction failed: {e}")
                        extra_context["performance"] = {"error": "Metrics unavailable"}

                    if html and len(html) > 500:
                        return screenshot_path, body_text, html, None, extra_context

                    last_err = "Empty/insufficient HTML after rendering."
                    raise RuntimeError(last_err)

                except Exception as exc:
                    err_str = str(exc)
                    last_err = f"Playwright attempt {attempt+1}/{PLAYWRIGHT_MAX_ATTEMPTS} failed: {err_str}"
                    logger.warning(last_err)
                    
                    try: html_best = page.content()
                    except: html_best = ""
                    try: text_best = page.inner_text("body")
                    except: text_best = ""
                    
                    if _is_probably_blocked(text_best, html_best):
                        return screenshot_path, text_best, html_best, "Blocked detected", extra_context

                    if attempt < PLAYWRIGHT_MAX_ATTEMPTS - 1:
                        sleep_for = _jittered_backoff(attempt)
                        time.sleep(sleep_for)
                    attempt += 1
                    continue

            return None, None, None, last_err or "Playwright failed after retries.", extra_context

    except Exception as exc:
        msg = f"Browser Error: {exc}"
        logger.error(msg)
        return None, None, None, msg, extra_context

    finally:
        try:
            if page: page.close()
            if context: context.close()
            if browser: browser.close()
        except: pass
        _PLAYWRIGHT_SEMAPHORE.release()
        gc.collect()


def detect_tech_features(html: str) -> List[str]:
    features: List[str] = []
    if not html: return features
    html_lower = html.lower()

    tech_map = {
        "Chatbot": ["podium", "intercom", "drift", "tidio", "chat-widget", "zendesk", "hubspot"],
        "Modern Framework": ["__next", "react", "vue", "nuxt", "gatsby", "webflow"],
        "Analytics": ["gtag(", "googletagmanager", "analytics.js", "ga('create'", "meta pixel"],
        "Booking": ["calendly", "acuity", "booksy", "booking.js"],
    }
    for category, keywords in tech_map.items():
        for k in keywords:
            if k in html_lower:
                features.append(category)
                break
    return features


# -------------------------------------------------------------------------
# GEMINI CALLER (With Secure Logging & Cooldown)
# -------------------------------------------------------------------------
def _call_gemini(prompt: str, img=None) -> Optional[dict]:
    global _LAST_GOOD_MODEL

    parts = [prompt]
    if img:
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        parts = [genai.types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), prompt]

    candidates = []
    if _LAST_GOOD_MODEL and _LAST_GOOD_MODEL not in candidates:
        candidates.append(_LAST_GOOD_MODEL)
    candidates.extend([m for m in GEMINI_MODEL_CANDIDATES if m not in candidates])

    for model_name in candidates:
        for _attempt in range(3):
            api_key = _next_key()
            if not api_key:
                break
            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content(model=model_name, contents=parts)
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
                    parsed["_model"] = model_name
                    return parsed

            except Exception as e:
                err_str = str(e)
                # --- FIX 1: Secure Logging & Real Cooldown ---
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logger.warning(f"Key rate limited ({model_name}). Cooling down key.")
                    _KEY_STATE["cooldowns"][api_key] = time.time() + 60
                
                time.sleep(1)
                continue

    return None


# -------------------------------------------------------------------------
# AI VISION ANALYSIS (UX/UI Only)
# -------------------------------------------------------------------------
def _ai_vision_generate(
    screenshot_path: Optional[str],
    text: str,
    url: str,
    company_name: str,
    tech_features: List[str],
    extra_context: Dict[str, object],
) -> Optional[dict]:
    if not GEMINI_KEYS:
        return {"problem": "Missing API Key", "offer": "Configure Google API Key"}

    tech_context = ", ".join(tech_features) if tech_features else "None detected"
    text_snippet = (text or "")[:TEXT_SNIPPET_LIMIT]
    structured = extra_context.get("structured") or {}
    blocked_reason = extra_context.get("blocked_reason")

    img = None
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            with PIL.Image.open(screenshot_path) as p_img:
                p_img.load()
                img = p_img
        except Exception:
            pass

    prompt = (
        "Return ONLY one-line JSON matching this schema with no code fences.\n"
        "{\n"
        '  "visual_observation": "string",\n'
        '  "level": "Level 1/2/3",\n'
        '  "problem": "string",\n'
        '  "offer": "string"\n'
        "}\n"
        f"Site: {company_name} ({url}) | Tech: {tech_context}\n"
        f"Blocked_reason(if_any): {blocked_reason}\n"
        f"Structured(signals): {json.dumps(structured, ensure_ascii=False)[:1200]}\n"
        f"Text snippet: {text_snippet}\n"
        "Use ONLY what is clearly visible in the screenshot. Use text snippet only for elements not visible.\n"
        "Ignore cookie banners/ads/popups.\n"
        "DO NOT infer performance, SEO, security, conversion rate, or bugs unless explicitly visible.\n"
        "CRITICAL SAFETY RULE: If you cannot point to a specific visible UI element that causes friction (or a specific missing element), "
        "you MUST NOT invent a problem and MUST use the insufficient-evidence problem instead.\n"
        "If the screenshot does not clearly show a problem, set problem to: "
        "'there isn’t enough visible evidence in this screenshot to call out a real issue'.\n"
        "CONTEXT: Your output completes these exact sentences in an email:\n"
        "1. 'Right now, {problem}.'\n"
        "2. 'We recently built a workflow for a client that {offer}.'\n\n"
        "FIELD RULES:\n"
        "- visual_observation: literal UI description of what is visible (or explicitly noting what is missing). No opinions.\n"
        "- problem: start with lowercase, <=15 words. TRACEABILITY RULE: You must implicitly reference the element identified in 'visual_observation'.\n"
        "- offer: start with a verb phrase to complete the sentence. Then add 1-2 sentences describing how this positively helped that client and how it will specifically help this site's improvement. Total length <= 60 words.\n"
        "- HONESTY: If the site looks excellent OR no friction is visible, use either:\n"
        "  problem='your digital presence is actually ahead of the curve' and offer='simply monitors uptime'\n"
        "  OR the 'not enough evidence' problem above.\n"
        "- CONTACT LOGIC: If you see a standard contact form (delayed reply friction) OR if a contact mechanism is missing/hard to find (access friction), "
        "the offer MUST propose an AI Agent that answers instantly and captures data as the new standard.\n"
        "LEVEL GUIDE:\n"
        "- Level 1: no issue visible / insufficient evidence.\n"
        "- Level 2: minor friction visible (extra clicks, generic CTA, unclear next step).\n"
        "- Level 3: clear conversion friction (long form, 'we will get back', gated contact, dead-end CTA) OR MISSING contact options.\n"
    )

    return _call_gemini(prompt, img)


# -------------------------------------------------------------------------
# AI SEO & PERFORMANCE AUDIT (Metric-Locked)
# -------------------------------------------------------------------------
def _ai_seo_perf_generate(extra_context: Dict[str, object]) -> Optional[dict]:
    seo_signals = extra_context.get("structured", {}).get("seo_signals", {})
    perf_metrics = extra_context.get("performance", {})

    # --- FIX 3: SMARTER SPA LOGIC IN PROMPT ---
    prompt = (
        "Return ONLY one-line JSON matching this schema. No markdown.\n"
        "{\n"
        '  "seo_problem": "string",\n'
        '  "performance_problem": "string",\n'
        '  "audit_priority": "High/Medium/Low"\n'
        "}\n\n"
        f"SEO signals (ground truth): {json.dumps(seo_signals)}\n"
        f"Performance metrics (ground truth ms): {json.dumps(perf_metrics)}\n\n"
        "RULES:\n"
        "- You may ONLY reference fields explicitly present above.\n"
        "- If no issue is provable, use: 'no verifiable SEO/Performance issue found from provided signals'.\n"
        "- DO NOT infer rankings, traffic, backlinks, or competitors.\n"
        "- Every claim MUST cite a metric or tag implicitly (e.g., 'Load time is 4500ms' or 'Missing 3 alt tags').\n"
        "- performance_problem: Flag if load_time_ms > 3000. If load_time_ms is 0 (SPA), use now_ms > 7000 as a fallback.\n"
        "- seo_problem: Flag if img_missing_alt_count > 0 or meta_desc_len < 50.\n"
    )

    result = _call_gemini(prompt, img=None)

    if result and not validate_no_hallucination(result):
        logger.warning("SEO/Perf Output discarded due to hallucination risk.")
        return {
            "seo_problem": "Analysis discarded (safety check)",
            "performance_problem": "Analysis discarded (safety check)",
            "audit_priority": "Low",
        }

    return result


# -------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -------------------------------------------------------------------------
def analyze_site(
    url: str,
    company_name: Optional[str] = None,
    fullname: Optional[str] = None,
    email: Optional[str] = None,
    render_js: bool = True,
    timeout_sec: Optional[int] = None,
):
    start_ts = time.time()
    res: Dict[str, object] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "company_fullname": fullname or "",
        "company_name": company_name or "",
        "email": email or "",
        "website_url": url,
        "problem": "",
        "offer": "",
        "seo_audit": {},
        "perf_audit": {},
        "errors": [],
    }

    screenshot_path, body_text, html, fetch_error, extra = fetch_screenshot_and_text(
        url, render_js=render_js, timeout_sec=timeout_sec
    )
    res["fetch_meta"] = extra

    if fetch_error:
        res["errors"].append(fetch_error)

    if not html:
        res["problem"] = "Digital storefront could not be reached."
        res["offer"] = "Verify hosting stability."
        res["duration_sec"] = round(time.time() - start_ts, 2)
        return res

    if _looks_garbled(body_text):
        msg = "Content returned unreadable."
        res["errors"].append(msg)
        res["problem"] = "Website content looks blocked."
        res["offer"] = "Run a manual review."
        if screenshot_path and os.path.exists(screenshot_path):
            try:
                os.remove(screenshot_path)
            except Exception:
                pass
        return res

    tech_features = detect_tech_features(html)

    # True Parallel AI Calls
    ai_vision = None
    ai_audit = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_vision = executor.submit(
            _ai_vision_generate, 
            screenshot_path, body_text or "", url, company_name or "Company", tech_features, extra
        )
        future_audit = executor.submit(
            _ai_seo_perf_generate, 
            extra
        )
        try:
            ai_vision = future_vision.result()
        except Exception as e:
            logger.error(f"Vision AI failed: {e}")
            
        try:
            ai_audit = future_audit.result()
        except Exception as e:
            logger.error(f"Audit AI failed: {e}")

    # Cleanup temp screenshot
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            os.remove(screenshot_path)
        except Exception:
            pass

    if ai_vision:
        res["problem"] = ai_vision.get("problem", "Analysis incomplete.")
        res["offer"] = ai_vision.get("offer", "Manual review recommended.")
        res["level"] = ai_vision.get("level", "Level 1")
    else:
        res["problem"] = "No visible evidence found."
        res["offer"] = "Manual review recommended."

    if ai_audit:
        res["seo_audit"] = {
            "problem": ai_audit.get("seo_problem"),
            "data": extra.get("structured", {}).get("seo_signals"),
        }
        res["perf_audit"] = {
            "problem": ai_audit.get("performance_problem"),
            "data": extra.get("performance"),
        }
        res["audit_priority"] = ai_audit.get("audit_priority")

    res["duration_sec"] = round(time.time() - start_ts, 2)
    return res


def results_to_csv(rows, path):
    normalized = []
    for r in rows:
        normalized.append(
            {
                "full_name": r.get("company_fullname", ""),
                "company": r.get("company_name", ""),
                "email": r.get("email", ""),
                "website_url": r.get("website_url", ""),
                "problem": r.get("problem", ""),
                "offer": r.get("offer", ""),
                "seo_issue": r.get("seo_audit", {}).get("problem", ""),
                "perf_issue": r.get("perf_audit", {}).get("problem", ""),
            }
        )
    df = pd.DataFrame(normalized)
    df.to_csv(path, index=False)


def analyze_sites_concurrent(sites, max_workers: int = 4, render_js: bool = True):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                analyze_site,
                s.get("url"),
                s.get("company"),
                s.get("fullname"),
                s.get("email"),
                render_js,
            )
            for s in sites
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                logger.warning(f"Concurrent error: {e}")
    return results


def validate_url(url: str) -> str:
    if not url: raise ValueError("URL is required")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http or https")
    return url


if __name__ == "__main__":
    print("Module Loaded. Gemini Vision + SPA-Safe SEO/Perf Engine Active. High-Performance Mode.")